import numpy as np

from airhockey.physics import TableConfig
from airhockey.shot_flight import first_goal_crossing, open_goal_outcomes


def test_direct_goal_and_end_rail_miss():
    result = open_goal_outcomes([[0.5, 0.5, 0, 3], [0.15, 0.5, 0, 3]])
    np.testing.assert_array_equal(result, [True, False])
    np.testing.assert_array_equal(
        first_goal_crossing([[0.5, 0.5, 0, 3], [0.15, 0.5, 0, 3]])[1], result
    )


def test_calibrated_single_bank_can_be_on_goal_despite_off_goal_direct_ray():
    cfg = TableConfig()
    edge = cfg.width - cfg.puck_radius
    vx = (
        3
        * ((edge - 0.5) + cfg.wall_tangential * (edge - 0.5) / cfg.wall_restitution)
        / (cfg.height - 0.5)
    )
    direct = 0.5 + vx * (cfg.height - 0.5) / 3
    assert direct > cfg.width
    assert open_goal_outcomes([[0.5, 0.5, vx, 3]], config=cfg)[0]
    crossing, aimed = first_goal_crossing([[0.5, 0.5, vx, 3]], config=cfg)
    assert aimed[0]
    np.testing.assert_allclose(crossing, 0.5, atol=0.005)


def test_predicted_aims_survive_actual_physics_across_randomized_rails():
    rng = np.random.default_rng(752)
    n = 500
    launch = np.column_stack(
        (
            rng.uniform(0.1, 0.9, n),
            rng.uniform(0.1, 0.7, n),
            rng.uniform(-10, 10, n),
            rng.uniform(1, 10, n),
        )
    )
    params = dict(
        puck_friction=rng.uniform(0.0005, 0.003, n),
        drag_b=rng.uniform(0.03, 0.04, n),
        wall_restitution=rng.uniform(0.72, 0.86, n),
        wall_tangential=rng.uniform(0.85, 0.95, n),
        end_wall_tangential=np.full(n, 0.66),
    )
    _, aimed = first_goal_crossing(launch, params)
    scored = open_goal_outcomes(launch, params)
    assert aimed.sum() > 100
    assert not (aimed & ~scored).any()


def test_stopped_and_backward_pucks_are_not_on_target():
    _, aimed = first_goal_crossing([[0.5, 0.5, 0, 0.001], [0.5, 0.5, 0, -3]])
    assert not aimed.any()
