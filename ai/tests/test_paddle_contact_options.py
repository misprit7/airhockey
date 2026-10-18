"""Opt-in measured paddle contact: defaults unchanged, options take effect."""
import numpy as np
import pytest

from airhockey.batch_physics import BatchPhysicsEngine


def _strike(engine, tangential_slip):
    # Puck at rest, paddle moving diagonally into it: the tangential part of
    # the relative velocity is what friction acts on.
    engine.puck_x[:] = 0.5
    engine.puck_y[:] = 0.5
    engine.puck_vx[:] = 0.0
    engine.puck_vy[:] = 0.0
    r = engine.config.puck_radius + engine.config.paddle_radius
    pad_x = np.full(engine.n_envs, 0.5)
    pad_y = np.full(engine.n_envs, 0.5 - r + 0.001)
    engine._collide_paddle(pad_x, pad_y, np.full(engine.n_envs, tangential_slip), np.full(engine.n_envs, 3.0))
    return engine.puck_vx.copy(), engine.puck_vy.copy()


def test_default_contact_is_frictionless_and_draws_the_historical_range():
    e = BatchPhysicsEngine(256, domain_randomize=True)
    e.reset(np.random.default_rng(3))
    assert e.paddle_tangential_range is None
    assert 0.60 <= e.paddle_restitution.min() and e.paddle_restitution.max() <= 0.95
    vx, vy = _strike(e, 2.0)
    assert np.allclose(vx, 0.0)  # head-on normal: no tangential transfer
    assert np.all(vy > 0)


def test_measured_ranges_lower_restitution_and_add_tangential_drag():
    e = BatchPhysicsEngine(256, domain_randomize=True)
    e.paddle_restitution_range = (0.40, 0.65)
    e.paddle_tangential_range = (0.76, 0.84)
    e.reset(np.random.default_rng(3))
    assert 0.40 <= e.paddle_restitution.min() and e.paddle_restitution.max() <= 0.65
    assert 0.76 <= e.paddle_tangential.min() and e.paddle_tangential.max() <= 0.84
    vx, vy = _strike(e, 2.0)
    # The paddle's sideways motion now drags the puck along with it.
    assert np.all(vx > 0.2) and np.all(vx < 2.0)
    assert np.all(vy > 0)


def test_env_rejects_invalid_ranges():
    from airhockey.neural_training import NeuralTrainingEnv
    with pytest.raises(ValueError):
        NeuralTrainingEnv(4, paddle_restitution_range=(0.7, 0.5))
