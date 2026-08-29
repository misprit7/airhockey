"""The perception model must actually degrade the signal, and in the right ways.

A model that quietly passes ground truth through would be worse than none: it
would look like the sim had been made realistic while changing nothing.
"""

from __future__ import annotations

import numpy as np

from airhockey.perception import COAST_MAX_S, PuckPerception


def _straight_run(vx=2.0, y=1.0, n=60, dt=1 / 60, **kw):
    p = PuckPerception(1, 1.0, 2.0, dt, rng=np.random.default_rng(0), **kw)
    x = np.array([0.15])
    yy = np.array([y])
    p.reset(x, yy)
    out = []
    for _ in range(n):
        x = x + vx * dt
        out.append((float(x[0]),) + tuple(float(v[0]) for v in p.update(x, yy)))
    return out


def test_velocity_estimate_tracks_truth():
    """The slope must converge, or the model is just noise."""
    rows = _straight_run(vx=2.0, y=0.3, glare=False)   # off-centre: no dropout
    est = np.array([r[3] for r in rows[10:]])
    assert abs(est.mean() - 2.0) < 0.05, est.mean()


def test_estimate_is_not_ground_truth():
    """It must differ from truth, or nothing has been modelled."""
    rows = _straight_run(vx=2.0, y=0.3, glare=False)
    est = np.array([r[3] for r in rows[10:]])
    assert est.std() > 1e-4, "velocity estimate is suspiciously exact"


def test_glare_patch_causes_dropout_at_centre():
    """A puck crossing table centre must be lost for a while."""
    p = PuckPerception(1, 1.0, 2.0, 1 / 60, rng=np.random.default_rng(0))
    assert not p.visible(np.array([0.5]), np.array([1.0]))[0]
    assert p.visible(np.array([0.5]), np.array([0.3]))[0]
    assert p.visible(np.array([0.1]), np.array([1.0]))[0]


def test_coasting_extrapolates_then_gives_up():
    """While hidden the estimate must keep moving, and not for ever."""
    dt = 1 / 200
    p = PuckPerception(1, 1.0, 2.0, dt, rng=np.random.default_rng(0),
                       noise=False)
    x, y = np.array([0.30]), np.array([1.0])
    p.reset(x, y)
    for _ in range(20):                       # establish a velocity
        x = x + 2.0 * dt
        p.update(x, y)
    hidden = np.array([0.5]), np.array([1.0])  # dead centre
    first = p.update(*hidden)[0][0]
    second = p.update(*hidden)[0][0]
    assert second > first, "coasted estimate should keep advancing"
    n = int(COAST_MAX_S / dt) + 5
    for _ in range(n):
        p.update(*hidden)
    assert p._coast_t[0] > COAST_MAX_S


def test_off_centre_puck_is_never_dropped():
    rows = _straight_run(vx=2.0, y=0.3)
    est = np.array([r[3] for r in rows[10:]])
    assert abs(est.mean() - 2.0) < 0.10


def test_partial_reset_with_a_boolean_mask():
    """The auto_reset path: some envs end, only those reset.

    reset() compared a boolean mask to slice(None) with `!=`, which numpy
    takes elementwise -- so the first episode to end mid-training crashed the
    run. Every earlier test reset ALL envs and never touched the branch.
    """
    from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
    e = BatchAirHockeyEnv(n_envs=4, max_episode_steps=3, **sensing_kwargs(True))
    e.reset(seed=0)
    for _ in range(3):
        _, _, term, trunc, _ = e.step(np.zeros((4, 2), dtype=np.float32))
    assert trunc.all(), "episodes should have truncated"
    out = e.auto_reset(term, trunc)          # raised before the fix
    assert out is not None and out.shape == (4, e.OBS_DIM)

    # And a genuinely partial mask, straight at the perception layer.
    p = PuckPerception(4, 1.0, 2.0, 1 / 100, rng=np.random.default_rng(0))
    mask = np.array([True, False, True, False])
    p.reset(np.full(4, 0.5), np.full(4, 1.0), mask)
    assert np.all(p._last_x[mask] == 0.5)
