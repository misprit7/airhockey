import numpy as np
import pytest

from airhockey.physics import TableConfig
from airhockey.puck_prediction import incoming_crossings
from airhockey.skill_benchmark import TrialPhysics


def test_crossings_match_independent_physics_including_banks_and_slow_pucks():
    rng = np.random.default_rng(6109)
    n = 100
    state = np.column_stack(
        (
            rng.uniform(0.1, 0.9, n),
            rng.uniform(0.5, 1.8, n),
            rng.uniform(-7, 7, n),
            -rng.uniform(0.2, 8, n),
        )
    )
    prediction = incoming_crossings(state, [0.25])[:, 0]
    e = TrialPhysics(n, TableConfig())
    for i, key in enumerate(("puck_x", "puck_y", "puck_vx", "puck_vy")):
        getattr(e, key)[:] = state[:, i]
    e.paddle_agent_x[:] = e.paddle_agent_y[:] = -10
    e.paddle_opp_x[:] = e.paddle_opp_y[:] = -10
    actual = np.full((n, 2), np.inf)
    dt = 0.0005
    for tick in range(2400):
        prev_x, prev_y = e.puck_x.copy(), e.puck_y.copy()
        e.step(dt)
        hit = (prev_y > 0.25) & (e.puck_y <= 0.25) & ~np.isfinite(actual[:, 0])
        fraction = (prev_y[hit] - 0.25) / (prev_y[hit] - e.puck_y[hit])
        actual[hit, 0] = prev_x[hit] + fraction * (e.puck_x[hit] - prev_x[hit])
        actual[hit, 1] = (tick + fraction) * dt
    reached = np.isfinite(prediction[:, 1])
    assert reached.sum() > 70
    np.testing.assert_array_equal(reached, np.isfinite(actual[:, 1]))
    np.testing.assert_allclose(prediction[reached, 0], actual[reached, 0], atol=0.015)
    np.testing.assert_allclose(prediction[reached, 1], actual[reached, 1], atol=0.01)


def test_unreachable_and_invalid_crossings():
    p = incoming_crossings([[0.5, 0.5, 0, 0.1], [0.5, 0.5, 0, -0.001]], [0.2, 0.8])
    assert np.isinf(p[:, :, 1]).all()
    with pytest.raises(ValueError):
        incoming_crossings([[0.5, 0.5, float("nan"), -1]], [0.2])
