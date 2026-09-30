import numpy as np

from airhockey.deploy import mm_velocity_to_sim
from airhockey.dynamics import sim_to_table_mm
from airhockey.heuristics import PuckSample, estimate_velocity
from airhockey.perception import PuckPerception


def test_evaluation_preserves_recorded_sensing_mode_when_checkpoint_is_moved(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("sensing_eval", Path(__file__).parents[1]/"bin/eval_neural_player.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'agent.pt'
    assert module.checkpoint_report_sensing(checkpoint, {'args': {'report_sensing': True}})
    assert not module.checkpoint_report_sensing(checkpoint, {})
    (tmp_path/'run.json').write_text(json.dumps({'args': {'report_sensing': True}}))
    assert module.checkpoint_report_sensing(checkpoint, {})
    assert not module.checkpoint_report_sensing(checkpoint, {'args': {'report_sensing': False}})


def test_batched_report_estimator_matches_live_fit_through_bounces_gaps_and_resets():
    n, dt = 5, .005
    p = PuckPerception(n, 1., 2., dt, glare=False, noise=False)
    p.enable_report_estimator()
    x, y = np.full(n, .4), np.full(n, .6)
    p.reset(x, y)
    samples = [[PuckSample(*sim_to_table_mm(x[i], y[i]), 0)] for i in range(n)]
    latest = np.column_stack((x, y))
    rng = np.random.default_rng(99)
    for frame in range(1, 161):
        t = frame*dt
        # Triangle wave has real sign reversals; small jitter exercises the
        # displacement threshold rather than assuming every reversal is a hit.
        x = .2 + .5*np.abs((np.arange(n)*.3 + t*8) % 2 - 1)
        y = .5 + .2*np.sin(np.arange(n) + t*11)
        x += rng.normal(0, .00035, n)
        hidden = rng.random(n) < .08
        hidden[0] |= 40 <= frame <= 95  # expire an entire history
        hidden[1] |= 30 <= frame <= 39  # keep timestamps across a short gap
        if frame == 115:
            mask = np.arange(n) == 3
            p.reset(x, y, mask)
            samples[3] = [PuckSample(*sim_to_table_mm(x[3], y[3]), (frame-1)*dt)]
            latest[3] = [x[3], y[3]]
        got = np.column_stack(p.update(x, y, hidden))
        expected = []
        for i in range(n):
            if not hidden[i]:
                samples[i].insert(0, PuckSample(*sim_to_table_mm(x[i], y[i]), t))
                latest[i] = [x[i], y[i]]
            samples[i] = [s for s in samples[i] if s.t_s >= t-.2-1e-9]
            fit = estimate_velocity(samples[i], window_s=.030)
            velocity = (0., 0.) if fit is None else mm_velocity_to_sim(fit.vx_mm_s, fit.vy_mm_s)
            expected.append([*latest[i], *velocity])
        np.testing.assert_allclose(got, expected, atol=2e-10)


def test_report_sensing_holds_last_real_fix_during_dropout():
    p = PuckPerception(1, 1., 2., .005, glare=False, noise=False)
    p.enable_report_estimator()
    p.reset(np.array([.2]), np.array([.4]))
    for i in range(1, 9):
        before = p.update(np.array([.2+i*.01]), np.array([.4]))
    for i in range(45):
        got = p.update(np.array([.9]), np.array([.9]), hidden=np.array([True]))
        np.testing.assert_allclose(got[:2], before[:2])
        if i < 20:
            np.testing.assert_allclose(got[2], [2.], atol=1e-10)
    np.testing.assert_array_equal(got[2:], [[0.], [0.]])
