"""Calm-accel cost: a needless accel cap is charged; a calm teacher guard."""
import numpy as np
import pytest

from airhockey.neural_training import NeuralTrainingEnv


def _rewards(weight, accel_slot, *, puck_speed=0.0, power=2.0):
    env = NeuralTrainingEnv(8, seed=1, calm_accel_weight=weight, calm_accel_power=power)
    env.reset(seed=1)
    e = env.engine
    # Puck at rest on the far half, the far mallet still: nothing to react to.
    e.puck_x[:], e.puck_y[:] = 0.5, 1.5
    e.puck_vx[:], e.puck_vy[:] = 0.0, puck_speed
    e.paddle_opp_vx[:], e.paddle_opp_vy[:] = 0.0, 0.0
    action = np.zeros((8, 6), np.float32)
    action[:, 1] = -0.9
    action[:, 5] = accel_slot
    _, reward, *_ = env.step(action)
    return np.asarray(reward), env


def test_default_is_off():
    env = NeuralTrainingEnv(2, seed=1)
    assert env.calm_accel_weight == 0.0


def test_high_cap_costs_more_than_low_cap_when_calm():
    high, env = _rewards(40.0, 1.0)
    low, _ = _rewards(40.0, -1.0)
    off, _ = _rewards(0.0, 1.0)
    assert np.all(low - high > 0.5)
    assert env.calm_seconds.sum() > 0 and env.calm_high_cap_seconds.sum() > 0
    assert np.all(np.abs(off - high) > 0.5)


def test_linear_cost_is_zero_at_the_floor():
    floor, _ = _rewards(40.0, -1.0, power=1.0)
    off, _ = _rewards(0.0, -1.0, power=1.0)
    mid, _ = _rewards(40.0, -0.2, power=1.0)
    assert np.allclose(floor, off)
    assert np.all(off - mid > 0.1)


def test_cost_fades_when_the_puck_is_fast():
    _, env = _rewards(40.0, 1.0, puck_speed=-3.0)
    assert env.calm_seconds.sum() == pytest.approx(0.0)


def test_invalid_calm_options_are_rejected():
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, calm_accel_weight=-1)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, calm_speed=0)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, calm_accel_power=3)


def test_teacher_guard_is_calm_at_its_post():
    from airhockey.expert_player import ExpertConfig
    c = ExpertConfig()
    assert c.guard_calm_fraction <= 0.1 < c.guard_fraction
