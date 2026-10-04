"""Regression checks for the September 20 telemetry findings; no hardware."""
import sys
from pathlib import Path

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.physics import TableConfig
from airhockey.rewards import predict_shot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision" / "bin"))


@pytest.fixture
def tracker(monkeypatch):
    from puck_stream import PuckTracker

    tr = PuckTracker()  # reads calibration files only
    monkeypatch.setattr(tr, "candidates", lambda blobs: (blobs, blobs[:, :2]))
    monkeypatch.setattr(tr, "_to_table", lambda xy, z: xy)
    return tr


def corners(x, y, n=4):
    from puck_markers import MARK_R

    a = np.arange(n) * np.pi / 2 + .3
    return np.column_stack([x + MARK_R * np.cos(a), y + MARK_R * np.sin(a), np.ones(n)])


def test_unreachable_square_and_partial_puck_cannot_create_fresh_measurement(tracker):
    tracker.update(0, corners(800, 450))
    blobs = np.concatenate([corners(1500, 450), corners(810, 450, n=3)])
    fix = tracker.update(.005, blobs)
    assert fix[:2] == pytest.approx((800, 450))  # coast from last full square
    assert tracker.n_markers == 0


def test_teleport_rejected_without_poisoning_history(tracker):
    tracker.update(0, corners(800, 450))
    tracker.update(.005, corners(810, 450))
    tracker.update(.010, corners(1500, 450))
    assert tracker.n_markers == 0
    assert tracker.rejected_jumps == 1
    assert len(tracker._hist) == 2
    fix = tracker.update(.015, corners(830, 450))
    assert fix[2] == pytest.approx(2000)


def test_fast_bounce_is_allowed_and_long_reacquisition_resets_velocity(tracker):
    for t, x in [(0, 800), (.005, 850), (.010, 800)]:
        assert tracker.update(t, corners(x, 450))[:2] == pytest.approx((x, 450))
        assert tracker.n_markers == 4
    fix = tracker.update(.2, corners(1200, 450))
    assert fix[2:] == (0, 0)


def test_robot_uses_the_same_puck_assignment(tracker, monkeypatch):
    import mallet_stream

    own = mallet_stream.MalletTracker(tracker, markers=3)
    blobs = np.concatenate([corners(800, 450), [[1500, 500, 1], [1526, 500, 1], [1500, 526, 1]]])
    tracker.update(0, blobs)
    monkeypatch.setattr(mallet_stream, "find_puck", lambda *a: pytest.fail("independent association"))
    def pose(remaining, world):
        np.testing.assert_array_equal(remaining, blobs[4:])
        return np.array([1500., 500.]), 3
    monkeypatch.setattr(own, '_robot_pose', pose)
    fix = own.update(blobs)
    assert fix[:2] == pytest.approx((1500., 500.))


def test_new_rail_model_matches_heldout_oblique_contact():
    # 46.465 s: not used to fit the coefficient. Velocities from the fits
    # either side of contact, in m/s. Normal e already agrees adequately.
    e = BatchPhysicsEngine(1)
    c = e.config
    e.puck_x[:] = c.width - c.puck_radius / 2
    e.puck_y[:] = 1.1
    e.puck_vx[:] = 1.5896547918
    e.puck_vy[:] = 2.9147575325
    e._collide_walls()
    measured = 2.722586733
    assert abs(e.puck_vy[0] - measured) < .15
    assert abs(2.9147575325 * .66 - measured) > .7


def test_robot_teleport_is_rejected_even_inside_workspace(tracker, monkeypatch):
    from mallet_stream import MalletTracker

    own = MalletTracker(tracker, markers=3)
    monkeypatch.setattr(own, '_robot_pose', lambda blobs, world: (world[0], 3))
    for t, x, expected in [(0, 1500, True), (.005, 1800, False), (.010, 1510, True)]:
        blobs = np.concatenate([corners(800, 450), [[x, 500, 1], [x+26, 500, 1], [x, 526, 1]]])
        tracker.update(t, blobs)
        assert (own.update(blobs) is not None) == expected


def test_rejected_marker_evidence_is_logged(tmp_path, tracker):
    import json
    from types import SimpleNamespace
    from airhockey.replay_log import ReplayLog

    path = tmp_path / 'replay.jsonl'
    log = ReplayLog(path, SimpleNamespace(policy='test', live=False, ramp=3))
    tracker.update(0, corners(800, 450))
    blobs = corners(1500, 450)
    tracker.update(.005, blobs)
    log.tracking_diagnostics(.005, tracker, blobs)
    log.tracking_diagnostics(.010, tracker, blobs)
    log.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    rejected = [r for r in records if r['type'] == 'tracking_rejection']
    assert len(rejected) == 1
    assert rejected[0]['blobs_px'] == blobs.tolist()


def test_shot_predictor_and_physics_share_side_rail_parameters():
    c = TableConfig(puck_friction=0, PUCK_DRAG_B=0)
    e = BatchPhysicsEngine(1, c)
    e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = .2, .4, -1.5, 4
    predicted = float(predict_shot(.2, .4, -1.5, 4)[0])
    # Integrate to the goal line, without goals resetting the puck.
    for _ in range(10000):
        dt = .0001
        if e.puck_y[0] + e.puck_vy[0] * dt >= 2:
            crossing = e.puck_x[0] + e.puck_vx[0] * (2-e.puck_y[0]) / e.puck_vy[0]
            break
        e.puck_x += e.puck_vx * dt
        e.puck_y += e.puck_vy * dt
        e._collide_walls()
    else:
        pytest.fail("no crossing")
    assert crossing == pytest.approx(predicted, abs=.001)


@pytest.mark.parametrize("delay", [.006, .0199])
def test_command_delay_keeps_old_command_then_accepts_new_one(delay, monkeypatch):
    e = BatchAirHockeyEnv(1, command_delay_s=delay, physics_dt=.002,
                          opponent_body="robot", opponent_policy="external",
                          action_mode="profile_a")
    e.reset(seed=1)
    own_x = e._agent_dyn['x'].copy()
    old_cap = e._agent_dyn['max_accel'].copy()
    trace = []
    opp_trace = []
    original = e._update_dynamics

    def observe(dyn, *args, **kwargs):
        deferred = kwargs.get('defer_command', False)
        if dyn is e._agent_dyn:
            trace.append((deferred, dyn['command_x'].copy(), dyn['command_accel'].copy()))
        if dyn is e._opp_dyn:
            opp_trace.append(deferred)
        return original(dyn, *args, **kwargs)

    monkeypatch.setattr(e, '_update_dynamics', observe)
    e.step(np.array([[.7, -.2, -1.0]]))
    n_delayed = int(np.ceil(delay / .002))
    assert all(r[0] for r in trace[:n_delayed])
    assert opp_trace == [r[0] for r in trace]
    for _, x, a in trace[:n_delayed]:
        np.testing.assert_array_equal(x, own_x)
        np.testing.assert_array_equal(a, old_cap)
    new_x = e._agent_dyn['command_x'].copy()
    assert not np.allclose(new_x, own_x)
    assert e._agent_dyn['command_accel'][0] == pytest.approx(old_cap[0] * .05)
    trace.clear()
    e.step(np.array([[-.7, -.2, 1.0]]))
    np.testing.assert_array_equal(trace[0][1], new_x)
    e.reset(seed=2)
    np.testing.assert_array_equal(e._agent_dyn['command_x'], e._agent_dyn['x'])


def test_realistic_sensing_enables_independent_command_delay():
    e = BatchAirHockeyEnv(1, **sensing_kwargs(True))
    assert e.command_delay_s == .012
    assert BatchAirHockeyEnv(1, **sensing_kwargs(False)).command_delay_s == 0


def test_tracking_samples_survive_complete_detection_loss(tmp_path, tracker):
    import json
    from types import SimpleNamespace
    from airhockey.replay_log import ReplayLog
    path=tmp_path/'lost.jsonl'
    log=ReplayLog(path,SimpleNamespace(policy='test',live=False,ramp=3))
    blobs=corners(800,450,2)
    report=SimpleNamespace(puck=None,mallet=None,t_puck=None,t_mallet=None)
    for t in (0,.1,.49,.5,.6):
        tracker.update(t,blobs)
        log.tracking_diagnostics(t,tracker,blobs,report)
    log.close()
    samples=[r for r in map(json.loads,path.read_text().splitlines()) if r['type']=='tracking_sample']
    assert len(samples)==2
    assert samples[0]['blobs_px']==blobs.tolist()
    assert not any(s['puck_fresh'] or s['agent_fresh'] for s in samples)
