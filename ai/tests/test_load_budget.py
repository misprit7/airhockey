from types import SimpleNamespace
import numpy as np
import pytest
import torch
from airhockey.load_budget import LoadBudgetPolicy
from airhockey.dynamics import MAX_ACCEL_M_S2


class Stub:
    cfg = SimpleNamespace(obs_shape={"state": [42]}, action_dim=3)
    _prev_mean_batch = None
    canceled = None

    def act(self, obs, **kwargs):
        return torch.ones((len(obs), 3))

    def cancel_pending(self, mask):
        self.canceled = mask.copy()


def test_hot_policy_cancels_arrivals_and_keeps_budget_across_points():
    base = Stub()
    p = LoadBudgetPolicy(base)
    obs = torch.zeros((2, 42))
    obs[:, 14] = 60 / MAX_ACCEL_M_S2
    obs[:, 22] = torch.tensor([0.9, 0.7])
    action = p.act(obs)
    assert p.cooling.tolist() == [True, False]
    assert base.canceled.tolist() == [True, False]
    assert action[0, 0] == 0
    np.testing.assert_allclose(
        (0.05 + 0.95 * ((action[0, 2].item() + 1) / 2) ** 2) * 60, 8, atol=1e-5
    )
    np.testing.assert_array_equal(action[1], np.ones(3))
    obs[0, 22] = 0.85
    p.act(obs, t0=torch.tensor([True, False]))
    assert p.cooling[0]
    obs[0, 22] = 0.81
    np.testing.assert_array_equal(p.act(obs)[0], np.ones(3))
    assert not p.cooling.any()


def test_slow_motor_memory_also_triggers_and_invalid_thresholds_fail():
    with pytest.raises(ValueError):
        LoadBudgetPolicy(Stub(), start=0.8, resume=0.9)
    p = LoadBudgetPolicy(Stub())
    obs = torch.zeros((1, 42))
    obs[:, 14] = 40 / MAX_ACCEL_M_S2
    obs[:, 29] = 0.91
    action = p.act(obs)
    assert p.cooling[0]
    np.testing.assert_allclose(
        (0.05 + 0.95 * ((action[0, 2].item() + 1) / 2) ** 2) * 40, 8, atol=1e-5
    )


def test_cooldown_still_tracks_a_goal_threat_with_low_acceleration():
    p = LoadBudgetPolicy(Stub())
    obs = torch.zeros((1, 42))
    obs[:, :4] = torch.tensor([0.62, 1.2, 0.0, -5.0])
    obs[:, 4:6] = torch.tensor([0.5, 0.25])
    obs[:, 13:15] = torch.tensor([1.0, 60 / MAX_ACCEL_M_S2])
    obs[:, 33] = 1
    obs[:, 40] = 0.4
    obs[:, 22] = 0.9
    action = p.act(obs)
    assert action[0, 0] > 0.1
    np.testing.assert_allclose(
        (0.05 + 0.95 * ((action[0, 2].item() + 1) / 2) ** 2) * 60, 8, atol=1e-5
    )


def test_warm_defense_can_use_reserve_but_stops_at_load_ceiling():
    p = LoadBudgetPolicy(Stub(), defense_caps=(8, 15, 25, 40, 60))
    obs = torch.zeros((1, 42))
    obs[:, :4] = torch.tensor([0.64, 0.85, 0.0, -8.0])
    obs[:, 4:6] = torch.tensor([0.4, 0.25])
    obs[:, 13:15] = torch.tensor([1.0, 60 / MAX_ACCEL_M_S2])
    obs[:, 33] = 1
    obs[:, 40] = 0.4
    obs[:, 22] = 0.9
    action = p.act(obs)
    cap = (0.05 + 0.95 * ((action[0, 2].item() + 1) / 2) ** 2) * 60
    assert 8 < cap <= 60
    obs[:, 22] = 0.91
    action = p.act(obs)
    cap = (0.05 + 0.95 * ((action[0, 2].item() + 1) / 2) ** 2) * 60
    assert cap == pytest.approx(8, abs=1e-5)
    assert p.cooling[0]


def test_fast_cooldown_does_not_latch_on_an_untripped_slow_memory():
    p = LoadBudgetPolicy(
        Stub(), slow_budget=dict(start=0.92, resume=0.89, defense_ceiling=0.94)
    )
    obs = torch.zeros((1, 42))
    obs[:, 14] = 60 / MAX_ACCEL_M_S2
    obs[:, 22] = 0.9
    obs[:, 26] = 0.9
    p.act(obs)
    assert p.fast_cooling[0] and not p.slow_cooling[0]
    obs[:, 22] = 0.81
    np.testing.assert_array_equal(p.act(obs, t0=True)[0], np.ones(3))
    assert not p.cooling[0]
    obs[:, 26] = 0.93
    p.act(obs)
    assert p.slow_cooling[0] and p.cooling[0]
    obs[:, 26] = 0.90
    p.act(obs, t0=True)
    assert p.cooling[0]
    obs[:, 26] = 0.88
    p.act(obs)
    assert not p.cooling[0]
