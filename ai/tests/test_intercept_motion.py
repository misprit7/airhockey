import numpy as np
import pytest
from airhockey.intercept_motion import forecast
from airhockey.motion import CartState, advance
from airhockey.interception import InterceptionController
from airhockey.dynamics import MAX_SPEED_M_S, MAX_ACCEL_M_S2


def test_variable_horizon_forecast_matches_independent_profile_rollouts():
    rng = np.random.default_rng(102)
    n = 12
    state = rng.uniform(
        [-800, -800, -2000, -2000, -20000, -20000],
        [800, 800, 2000, 2000, 20000, 20000],
        (n, 6),
    ).astype(np.float32)
    queued = rng.uniform([-800, -800, 8000], [800, 800, 60000], (n, 3)).astype(
        np.float32
    )
    command = rng.uniform([-800, -800, 8000], [800, 800, 60000], (n, 3)).astype(
        np.float32
    )
    steps = rng.integers(0, 800, n)
    predicted = forecast(state, queued, command, steps)
    expected = np.zeros_like(state)
    for i in range(n):
        c = CartState(1)
        for j, key in enumerate(("x", "y", "vx", "vy", "ax", "ay")):
            getattr(c, key)[:] = state[i, j]
        advance(
            c,
            queued[i : i + 1, 0],
            queued[i : i + 1, 1],
            12000,
            queued[i : i + 1, 2],
            0.003,
            0.001,
            15,
        )
        advance(
            c,
            command[i : i + 1, 0],
            command[i : i + 1, 1],
            12000,
            command[i : i + 1, 2],
            0.003,
            0.001,
            int(steps[i]),
        )
        expected[i] = [getattr(c, key)[0] for key in ("x", "y", "vx", "vy", "ax", "ay")]
    np.testing.assert_allclose(predicted, expected, atol=0.002, rtol=1e-6)


def test_interceptor_changes_only_incoming_game_states():
    obs = np.zeros((3, 42), np.float32)
    obs[:, :4] = [0.55, 1.2, 0, -5]
    obs[:, 4:6] = [0.5, 0.25]
    obs[:, 33] = 1
    obs[:, 13] = 12 / MAX_SPEED_M_S
    obs[:, 14] = 60 / MAX_ACCEL_M_S2
    obs[:, 40] = 0.5
    obs[1, 3] = 5
    obs[2, 32:34] = [1, 0]
    action = np.zeros((3, 3), np.float32)
    result, applied = InterceptionController()(obs, action)
    np.testing.assert_array_equal(applied, [True, False, False])
    np.testing.assert_array_equal(result[1:], action[1:])
    assert np.isfinite(result).all() and np.max(abs(result)) <= 1


def test_forecast_rejects_invalid_horizons():
    with pytest.raises(ValueError):
        forecast(np.zeros((1, 6)), np.ones((1, 3)), np.ones((1, 3)), [-1])


def test_threat_filter_parks_for_clear_misses_without_changing_direct_defense():
    obs = np.zeros((2, 42), np.float32)
    obs[:, :4] = [0.5, 1.2, 0, -5]
    obs[:, 4:6] = [0.5, 0.25]
    obs[:, 33] = 1
    obs[:, 13] = 12 / MAX_SPEED_M_S
    obs[:, 14] = 60 / MAX_ACCEL_M_S2
    obs[:, 40] = 0.5
    obs[1, 0] = 0.08  # A straight shot into the end rail, far outside the mouth.
    original, _ = InterceptionController()(obs, np.zeros((2, 3)))
    guarded, used = InterceptionController(threat_only=True)(obs, np.zeros((2, 3)))
    np.testing.assert_allclose(guarded[0], original[0])
    assert used.all()
    np.testing.assert_allclose(guarded[1, 0], 0, atol=1e-6)
    obs[1, 14] = 40 / MAX_ACCEL_M_S2
    with pytest.raises(ValueError):
        InterceptionController(threat_only=True)(obs, np.zeros((2, 3)))
