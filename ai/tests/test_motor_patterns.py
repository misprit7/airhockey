"""Offline trajectory and hardware lifecycle checks; never open hardware."""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from airhockey.motor_patterns import CENTER, LOW, HIGH, NAMES, LoadTelemetryNotReady, check_load, plan_pattern

spec = importlib.util.spec_from_file_location('motor_demo', Path(__file__).parents[1]/'bin/motor_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('accel', [10000, 30000, 60000])
def test_firmware_paths(name, accel):
    plan = plan_pattern(name, accel=accel)
    assert np.all(plan.commands >= LOW) and np.all(plan.commands <= HIGH)
    assert np.all(plan.predicted >= LOW+40) and np.all(plan.predicted <= HIGH-40)
    assert np.linalg.norm(plan.velocity, axis=1).max() <= plan.speed*1.001
    assert np.linalg.norm(plan.acceleration, axis=1).max() <= accel*1.001
    assert plan.summary()['predicted_max_error_mm'] < 5
    assert np.linalg.norm(plan.predicted[-1]-plan.reference[-1]) < .1
    assert np.linalg.norm(plan.velocity[-1]) < 1
    np.testing.assert_allclose(plan.reference[0], plan.reference[-1], atol=1e-8)
    assert 4 < plan.summary()['duration_s'] < 20


@pytest.mark.parametrize('speed,accel', [(float('nan'),30000), (4000,float('inf')), (0,30000), (13000,30000), (4000,61000)])
def test_reject_invalid_caps(speed, accel):
    with pytest.raises(ValueError):
        plan_pattern('slalom', speed, accel)


def load(now=100., value=30.):
    return {'sample': {'context': {'fault': False}, 'motors': [
        {'node': node, 'rms_pct': {'valid': True, 'value': value, 'end': now},
         'rms_slow_pct': {'valid': False, 'error': 'unsupported'}} for node in range(4)]}}


def test_load_guard():
    assert check_load(load(), 100.1) == 30
    for sample in (load(99), load(value=85), {'sample': None}):
        with pytest.raises(RuntimeError):
            check_load(sample, 100.1)
    sample = load()
    sample['sample']['motors'][2]['rms_slow_pct'] = {'valid': True, 'end':100., 'value':86.}
    with pytest.raises(RuntimeError, match='RMS load'):
        check_load(sample, 100.1)
    sample = load()
    sample['sample']['motors'][2]['rms_pct']['valid'] = False
    with pytest.raises(RuntimeError, match='four motors'):
        check_load(sample, 100.1)
    sample = load()
    sample['sample']['context']['fault'] = True
    with pytest.raises(RuntimeError, match='fault'):
        check_load(sample, 100.1)


class FakeClient:
    def __init__(self, fail_enable=False, stale=False, fail_disable=False):
        self.events, self.fail_enable, self.stale = [], fail_enable, stale
        self.fail_disable = fail_disable

    def connect(self): self.events.append('connect')
    def close(self): self.events.append('close')
    def enable(self, *pose):
        self.events.append(('enable', pose))
        if self.fail_enable: raise RuntimeError('partial enable failure')
    def disable(self):
        self.events.append('disable')
        if self.fail_disable: raise RuntimeError('failed disable')
    def get_position_sample(self): return (*CENTER, 0, 0, 2 if self.stale else 0)
    def get_motor_load(self): return load(demo.time.monotonic())
    def command_position(self, *args): self.events.append(('command', args))
    def set_ramp(self, ramp): self.events.append(('ramp', ramp))
    def set_limits(self, *args): self.events.append(('limits', args))


def test_brake_keeps_caps_and_disables_even_without_position():
    client = FakeClient()
    demo.shutdown(client, attempted_enable=True, sleep=lambda _: None)
    assert client.events == [('command', (*CENTER, 0)), 'disable', 'close']
    client = FakeClient(stale=True)
    demo.shutdown(client, attempted_enable=True, sleep=lambda _: None)
    assert client.events == ['disable', 'close']
    client = FakeClient(fail_disable=True)
    with pytest.raises(RuntimeError, match='not acknowledged'):
        demo.shutdown(client, attempted_enable=True, sleep=lambda _: None)
    assert client.events[-1] == 'close'


@pytest.mark.parametrize('failure', ['enable', 'perform', 'interrupt', 'camera'])
def test_live_cleanup_uses_only_injected_fake_hardware(monkeypatch, tmp_path, failure):
    from airhockey import hardware
    client = FakeClient(fail_enable=failure=='enable')
    monkeypatch.setattr(hardware, 'CDPRClient', lambda: client)
    # No launcher, socket, camera or real device exists in this test.
    monkeypatch.setattr(demo, 'start_master', lambda _: (_ for _ in ()).throw(AssertionError('must not start master')))
    monkeypatch.setattr(demo.Monitor, 'rest', lambda *_: None)
    def measure():
        if failure == 'camera': raise RuntimeError('ambiguous camera')
        return (*CENTER, np.pi/2)
    monkeypatch.setitem(sys.modules, 'track_mallet', SimpleNamespace(measure=measure))
    def perform(*_):
        if failure == 'interrupt': raise KeyboardInterrupt
        raise RuntimeError('load or timing fault')
    monkeypatch.setattr(demo, 'perform', perform)
    args = SimpleNamespace(existing_master=True, rms_stop=85., repeat=1)
    plan = SimpleNamespace(name='fake', commands=[CENTER])
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        demo.live(args, [plan], tmp_path)
    assert client.events[-1] == 'close'
    assert ('disable' in client.events) == (failure != 'camera')
    if failure != 'camera':
        enable = next(e for e in client.events if isinstance(e, tuple) and e[0]=='enable')
        assert enable[1][2] == pytest.approx(90.)


def test_default_cli_cannot_activate(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs): raise AssertionError('offline must not access hardware')
    monkeypatch.setattr(demo, 'live', forbidden)
    monkeypatch.setattr(demo.socket, 'socket', forbidden)
    assert demo.main(['all', '--output', str(tmp_path)]) == 0
    assert (tmp_path/'preview.html').is_file()
    assert 'slalom' in (tmp_path/'plan.json').read_text()


def test_deadline_miss_does_not_send_late_command(monkeypatch):
    monkeypatch.setattr(demo, 'move_to', lambda *_: None)
    times = iter([10., 10., 10., 10.1])
    monkeypatch.setattr(demo.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(demo.time, 'sleep', lambda _: None)
    monitor = SimpleNamespace(rest=lambda _: None, ready_after_setup=lambda: None,
                              poll=lambda: (*CENTER,0,0,0))
    plan = SimpleNamespace(reference=[CENTER], commands=[CENTER], speed=4000, accel=30000)
    client = FakeClient()
    with pytest.raises(RuntimeError, match='fell'):
        demo.perform(client, monitor, plan, io.StringIO())
    assert client.events == [('limits', (4000,30000))]


def test_small_timing_hiccup_does_not_burst_overdue_commands(monkeypatch):
    import csv
    now = [0.]
    monkeypatch.setattr(demo.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(demo.time, 'sleep', lambda dt: now.__setitem__(0, now[0]+dt))
    monkeypatch.setattr(demo, 'move_to', lambda *_: None)
    monkeypatch.setattr(demo, 'brake', lambda *_: None)
    polls = [0]
    def poll():
        if polls[0] == 0:
            now[0] += .035
        polls[0] += 1
        return (*CENTER, 0, 0, 0)
    monitor = SimpleNamespace(ready_after_setup=lambda: None, rest=lambda _: None,
                              poll=poll, peak=30.)
    plan = SimpleNamespace(name='test', reference=[CENTER]*3, commands=[CENTER]*3,
                           speed=4000, accel=30000)
    sent = []
    client = FakeClient()
    client.command_position = lambda *_: sent.append(now[0])
    output = io.StringIO()
    demo.perform(client, monitor, plan, csv.writer(output))
    assert sent == pytest.approx([.035, .045, .055])
    rows = list(csv.reader(io.StringIO(output.getvalue())))
    assert [float(v) for v in rows[0][-4:]] == pytest.approx([35., 35., 0., .035])


def test_enable_cache_gap_from_recorded_session(monkeypatch):
    # 20260927-111441: ENABLE held the SDK mutex for 0.525s. The
    # previously published sample was 0.605s old; the next was published
    # 0.106s after RAMP returned. No movement should be issued in this gap.
    now = [906302.327793]
    monkeypatch.setattr(demo.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(demo.time, 'sleep', lambda dt: now.__setitem__(0, now[0]+dt))
    client = FakeClient()
    client.get_motor_load = lambda: (load(906301.723124, 1.) if now[0] < 906302.433229
                                     else load(906302.338755, 1.))
    with pytest.raises(LoadTelemetryNotReady, match='stale'):
        check_load(client.get_motor_load(), now[0])
    monitor = demo.Monitor(client, 85.)
    monitor.ready_after_setup()
    assert 906302.433229 <= now[0] < 906302.5
    assert monitor.peak == 1.
    assert client.events == []


def test_setup_requires_new_acquisition_and_times_out_without_motion(monkeypatch):
    now = [100.]
    monkeypatch.setattr(demo.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(demo.time, 'sleep', lambda dt: now.__setitem__(0, now[0]+dt))
    client = FakeClient()
    client.get_motor_load = lambda: load(99.9)
    # Recent pre-enable data still isn't a post-enable observation.
    assert check_load(client.get_motor_load(), 100.) == 30.
    with pytest.raises(RuntimeError, match='timed out'):
        demo.wait_for_load(client, 85., not_before=100., timeout=.2)
    assert now[0] >= 100.2
    assert client.events == []


@pytest.mark.parametrize('failure', ['overload', 'fault', 'invalid'])
def test_setup_does_not_retry_actual_load_failures(monkeypatch, failure):
    monkeypatch.setattr(demo.time, 'monotonic', lambda: 100.)
    def forbidden_sleep(_): raise AssertionError('faults must not be retried')
    monkeypatch.setattr(demo.time, 'sleep', forbidden_sleep)
    sample = load()
    # A stale first channel must not hide a hot motor later in the snapshot.
    sample['sample']['motors'][0]['rms_pct']['end'] = 99.
    if failure == 'overload': sample['sample']['motors'][3]['rms_pct']['value'] = 86.
    elif failure == 'fault': sample['sample']['context']['fault'] = True
    else: sample['sample']['motors'][3]['rms_pct']['value'] = float('nan')
    client = FakeClient()
    client.get_motor_load = lambda: sample
    with pytest.raises(RuntimeError) as exc:
        demo.wait_for_load(client, 85.)
    assert not isinstance(exc.value, LoadTelemetryNotReady)
    assert client.events == []


def test_running_monitor_still_aborts_immediately_on_stale_load(monkeypatch):
    monkeypatch.setattr(demo.time, 'monotonic', lambda: 100.)
    def forbidden_sleep(_): raise AssertionError('running monitor must not wait')
    monkeypatch.setattr(demo.time, 'sleep', forbidden_sleep)
    client = FakeClient()
    client.get_motor_load = lambda: load(99.)
    with pytest.raises(LoadTelemetryNotReady):
        demo.Monitor(client, 85.).poll()
    assert client.events == []


def test_no_position_command_until_after_limits_and_telemetry(monkeypatch):
    events = []
    client = FakeClient()
    client.set_limits = lambda *_: events.append('limits')
    client.command_position = lambda *_: events.append('move')
    def barrier():
        events.append('telemetry wait')
        raise RuntimeError('telemetry unavailable')
    monitor = SimpleNamespace(poll=lambda: (*CENTER, 0, 0, 0), ready_after_setup=barrier)
    with pytest.raises(RuntimeError, match='unavailable'):
        demo.move_to(client, monitor, CENTER)
    assert events == ['limits', 'telemetry wait']


def test_tension_is_forwarded_to_owned_master_without_starting_hardware(monkeypatch):
    connections = iter([False, True])
    monkeypatch.setattr(demo, 'listening', lambda: next(connections))
    monkeypatch.setattr(demo.subprocess, 'run', lambda *a, **kw: None)
    proc = SimpleNamespace(poll=lambda: None)
    launched = []
    def fake_popen(argv, **kwargs):
        launched.append(argv)
        return proc
    monkeypatch.setattr(demo.subprocess, 'Popen', fake_popen)
    assert demo.start_master(io.StringIO(), 1.5) is proc
    assert launched[0][-2:] == ['--tension', '1.5']


@pytest.mark.parametrize('extra', [
    ['--tension', '-1'], ['--tension', 'nan'], ['--tension', 'inf'],
    ['--tension', '1.5', '--existing-master', '--live'],
])
def test_invalid_tension_options_fail_before_planning_or_hardware(monkeypatch, extra):
    def forbidden(*a, **kw): raise AssertionError('must fail before planning or hardware')
    monkeypatch.setattr(demo, 'plan_pattern', forbidden)
    monkeypatch.setattr(demo, 'live', forbidden)
    with pytest.raises(SystemExit) as exc:
        demo.main(['slalom', *extra])
    assert exc.value.code == 2


def test_fast_caps_and_tension_can_be_previewed_offline(monkeypatch, tmp_path):
    import json
    def forbidden(*a, **kw): raise AssertionError('offline must not access hardware')
    monkeypatch.setattr(demo, 'live', forbidden)
    monkeypatch.setattr(demo.socket, 'socket', forbidden)
    assert demo.main(['slalom', '--speed', '12', '--accel', '60', '--tension', '1.5',
                      '--output', str(tmp_path)]) == 0
    metadata = json.loads((tmp_path/'plan.json').read_text())
    assert metadata['startup_tension_mm'] == 1.5
    assert metadata['patterns'][0]['predicted_peak_speed_m_s'] > 2.8
