import torch

from airhockey.on_policy import advantages, exploration_scale


def test_terminal_and_truncation_bootstrap_differ_without_reset_leakage():
    reward = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    value = torch.zeros_like(reward)
    next_value = torch.tensor([[10.0, 10.0], [0.0, 0.0]])
    terminated = torch.tensor([[True, False], [True, True]])
    done = torch.ones_like(terminated)
    adv, ret = advantages(reward, value, next_value, terminated, done, gamma=0.9)
    torch.testing.assert_close(adv, torch.tensor([[1.0, 10.0], [100.0, 100.0]]))
    torch.testing.assert_close(ret, adv)


def test_continuing_trajectory_accumulates_discounted_advantage():
    reward = torch.tensor([[1.0], [2.0]])
    zero = torch.zeros_like(reward)
    terminated = torch.tensor([[False], [True]])
    adv, _ = advantages(reward, zero, zero, terminated, terminated, gamma=0.5, lam=1.0)
    torch.testing.assert_close(adv, torch.tensor([[2.0], [2.0]]))


def test_exploration_preserves_precision_but_covers_setup_and_urgent_bursts():
    obs = torch.zeros(4, 42)
    obs[:, :2] = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.9], [0.5, 1.5]])
    obs[:, 4:6] = torch.tensor([[0.5, 0.34], [0.8, 0.6], [0.5, 0.25], [0.5, 0.25]])
    obs[2, 3] = -5
    base = torch.tensor([0.03, 0.03, 0.12])
    scale = exploration_scale(obs, base, 0.15, 0.6)
    torch.testing.assert_close(
        scale,
        torch.tensor(
            [
                [0.03, 0.03, 0.12],
                [0.15, 0.15, 0.12],
                [0.03, 0.03, 0.6],
                [0.03, 0.03, 0.12],
            ]
        ),
    )


def test_soft_penalty_tuning_does_not_change_load_model_or_overload_cost():
    import numpy as np
    from airhockey.thermal import MotorThermal

    a = MotorThermal(2, randomize=False)
    b = MotorThermal(2, randomize=False, soft_start=0.85)
    for model in (a, b):
        model.h[0] = 0.8**2
        model.h[1] = 1
        model.current_squared[:] = 4
    np.testing.assert_array_equal(a.levels, b.levels)
    assert b.penalty(1)[0] < a.penalty(1)[0]
    assert b.penalty(1)[1] == a.penalty(1)[1]
