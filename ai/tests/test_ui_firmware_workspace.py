"""UI motion and overlays follow firmware bounds; all hardware is fake."""
import asyncio
from contextlib import suppress

import numpy as np
import pytest

from airhockey import server
from airhockey.dynamics import HardwareDynamics, workspace_in_sim
from airhockey.workspace_probe import PROBE_BOUNDS


@pytest.fixture(params=['standard', 'expanded'])
def hardware(monkeypatch, request):
    from airhockey.dynamics import _geom
    bounds = tuple(PROBE_BOUNDS) if request.param == 'expanded' else (
        _geom.WS_MIN_X, _geom.WS_MAX_X, _geom.WS_MIN_Y, _geom.WS_MAX_Y)
    events = []
    class Client:
        def __init__(self, *args): pass
        def connect(self): events.append('connect')
        def get_workspace(self):
            events.append('workspace')
            return bounds
        def enable(self, *args): events.append('enable')
        def set_startup_tension(self, mm): pass
        def disable(self): events.append('disable')
        def close(self): events.append('close')
    class Hardware(HardwareDynamics):
        def reset(self, x, y): self.x, self.y = x, y
        def update(self, x, y, dt):
            self._hw_x_mm, self._hw_y_mm = self._sim_to_mm(x, y)
            self.x, self.y = self._mm_to_sim(self._hw_x_mm, self._hw_y_mm)
            return self.x, self.y
    monkeypatch.setattr('airhockey.hardware.CDPRClient', Client)
    monkeypatch.setattr(server, 'HardwareDynamics', Hardware)
    monkeypatch.setattr(server.VISION, 'latest_pose', lambda: (1500., 450., 2.35))
    monkeypatch.setattr(server, '_camera_objects', lambda: {})
    return Hardware, bounds, events


def test_firmware_bounds_drive_clamp_and_both_coordinate_displays(hardware):
    Hardware, bounds, events = hardware
    hd = Hardware()
    assert events[:3] == ['connect', 'workspace', 'enable']
    for x in (bounds[0], bounds[1]):
        for y in (bounds[2], bounds[3]):
            np.testing.assert_allclose(hd._sim_to_mm(*hd._mm_to_sim(x, y)), (x, y))
    for x, y in ((-1000., -1000.), (3000., 3000.)):
        got = hd._sim_to_mm(*hd._mm_to_sim(x, y))
        np.testing.assert_allclose(got, (np.clip(x, *bounds[:2]), np.clip(y, *bounds[2:])))
    assert hd.workspace_in_sim() == workspace_in_sim(bounds_mm=bounds)
    assert hd.hw_state()['workspace_mm'] == dict(zip(
        ('min_x', 'max_x', 'min_y', 'max_y'), bounds))


class Socket:
    """In-memory WebSocket transport; exercise the real server handler."""
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    async def accept(self): pass
    async def receive_json(self): return await self.incoming.get()
    async def send_json(self, msg): await self.outgoing.put(msg)


async def receive(ws, kind):
    for _ in range(100):
        msg = await asyncio.wait_for(ws.outgoing.get(), timeout=2)
        if msg['type'] == kind:
            return msg
    pytest.fail(f'No {kind} response')


@pytest.mark.asyncio
async def test_control_socket_reaches_firmware_edge_and_restores_sim_defaults(hardware):
    _, bounds, _ = hardware
    ws = Socket()
    task = asyncio.create_task(server.live_game(ws))
    try:
        config = await receive(ws, 'config')
        await ws.incoming.put({'type': 'toggle_hardware'})
        assert (await receive(ws, 'hardware_mode'))['enabled']
        await ws.incoming.put({'type': 'move', 'x': 0., 'y': 0.})
        expected = workspace_in_sim(config['width'], config['height']/2, bounds_mm=bounds)
        for _ in range(100):
            frame = await receive(ws, 'frame')
            if frame['agent_x'] == expected['min_x'] and frame['agent_y'] == expected['min_y']:
                break
        else:
            pytest.fail('UI still clamps the firmware edge target')
        assert frame['hw_ws'] == expected
        np.testing.assert_allclose((frame['hw_x_mm'], frame['hw_y_mm']), (bounds[1], bounds[2]), atol=.05)
        assert frame['hw']['workspace_mm']['max_x'] == bounds[1]
        await ws.incoming.put({'type': 'toggle_hardware'})
        assert not (await receive(ws, 'hardware_mode'))['enabled']
        frame = await receive(ws, 'frame')
        assert frame['hw_ws'] == workspace_in_sim(config['width'], config['height']/2)
        assert 'hw' not in frame
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
