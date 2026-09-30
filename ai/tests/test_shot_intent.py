import numpy as np
import pytest
import torch
from types import SimpleNamespace

from airhockey.arrival_play import ArrivalPlayController
from airhockey.dynamics import MAX_SPEED_M_S, MAX_ACCEL_M_S2
from airhockey.shot_intent import IntentValue, OptionAccumulator, ShotIntentPolicy


def observation(n=2):
    x = np.zeros((n, 42), np.float32)
    x[:, :2] = [0.5, 0.5]
    x[:, 4:6] = [0.5, 0.33]
    x[:, 13] = 12 / MAX_SPEED_M_S
    x[:, 14] = 60 / MAX_ACCEL_M_S2
    x[:, 33] = 1
    x[:, 40] = 0.5
    return x


class Fallback:
    cfg = SimpleNamespace()
    _prev_mean_batch = None

    def act(self, observation, **kwargs):
        return torch.zeros((len(observation), 3))


def test_new_selector_preserves_fixed_arrival_behavior_and_latches_choice():
    x = observation()
    baseline = ArrivalPlayController()
    policy = ShotIntentPolicy(Fallback(), IntentValue())
    for i in range(4):
        expected, _ = baseline(x, np.zeros((2, 3)), t0=i == 0)
        actual = policy.act(torch.from_numpy(x), t0=i == 0).numpy()
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(policy.last_decision, [i == 0, i == 0])
    # Selective episode reset discards an old committed shot.
    x[0, :2] = [0.2, 0.4]
    policy.act(torch.from_numpy(x), t0=[True, False])
    assert policy.last_decision[0] and not policy.last_decision[1]


def test_options_keep_discounted_delayed_reward_and_terminal_semantics():
    acc = OptionAccumulator(2, gamma=0.5)
    obs = observation()
    acc.start([True, True], obs, np.array([3, 1]))
    acc.add_reward([2, 4])
    acc.add_reward([6, 8])
    rows = acc.finish([True, False], obs, terminal=False)
    np.testing.assert_allclose(rows[2], [5])
    np.testing.assert_allclose(rows[3], [0.25])
    assert rows[1].tolist() == [3] and rows[5].tolist() == [2]
    rows = acc.finish([True, True], obs, terminal=[False, True])
    np.testing.assert_allclose(rows[2], [8])
    np.testing.assert_array_equal(rows[3], [0])
    assert not acc.active.any()


def test_cushion_request_cancels_committed_shot_and_invalid_caps_fail():
    c = ArrivalPlayController()
    x = observation(1)
    c(x, np.zeros((1, 3)))
    assert c.remaining[0] > 0
    x[:, 32:34] = [1, 0]
    out, used = c(x, np.zeros((1, 3)))
    assert c.remaining[0] == 0 and not used[0]
    np.testing.assert_array_equal(out, np.zeros((1, 3)))
    x[:, 14] = 40 / MAX_ACCEL_M_S2
    with pytest.raises(ValueError):
        c(x, np.zeros((1, 3)))


def test_navigation_avoids_the_puck_and_bank_geometry_remains_finite():
    c = ArrivalPlayController()
    waypoint = c.waypoint(
        np.array([0.5, 0.7]), np.array([0.5, 0.33]), np.array([0.5, 0.5])
    )
    assert abs(waypoint[0] - 0.5) > 0.05
    direction = c.direction(
        np.array([[0.5, 0.5], [1.02, 0.5]]), np.array([0.5, 0.5]), np.array([1, 1])
    )
    assert np.isfinite(direction).all()
    assert (direction[:, 1] > 0).all()
