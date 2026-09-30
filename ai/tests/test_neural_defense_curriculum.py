import numpy as np

from airhockey.neural_training import NeuralTrainingEnv
from airhockey.shot_flight import first_goal_crossing
from airhockey.skill_benchmark import Fixtures


def test_receiving_curriculum_honors_defense_fraction_and_varies_attacks_and_starts():
    env = NeuralTrainingEnv(
        400, stage=5, productive_receive_drill=True,
        practice_defense_fraction=.6, defense_min_speed=8,
        random_defense_start_fraction=1, randomize=False, realistic=False,
    )
    env.reset(seed=2471)
    np.testing.assert_array_equal(np.bincount(env.kind, minlength=4), [40, 40, 120, 200])
    defense = env.kind == 2
    e = env.engine
    launches = np.column_stack((e.puck_x[defense], 2-e.puck_y[defense],
                                e.puck_vx[defense], -e.puck_vy[defense]))
    speed = np.linalg.norm(launches[:,2:], axis=1)
    assert np.all((speed >= 8) & (speed <= 12))
    _, _, banks, _ = first_goal_crossing(launches, return_route=True)
    assert 20 < np.count_nonzero(banks) < 100
    assert np.ptp(e.paddle_agent_x[defense]) > .4
    assert np.ptp(e.paddle_agent_y[defense]) > .3
    # Explicit benchmark fixtures remain exact despite training randomization.
    fixture = Fixtures(np.array([2]), np.array([[.5,1.2,0.,-9.]]),
                       np.array([[.5,.25]]), np.array([.5]))
    env.reset(mask=np.arange(400)==399, fixtures=fixture)
    np.testing.assert_allclose([e.paddle_agent_x[399],e.paddle_agent_y[399]], [.5,.25])


def test_defense_clear_requires_contact_and_cannot_reward_a_concede_or_full_game():
    env = NeuralTrainingEnv(4, stage=2, defense_clear_reward=100,
                            randomize=False, realistic=False, setup_weight=0)
    env.reset(fixtures=Fixtures(
        np.array([2,2,2,3]),
        np.array([[.5,1.2,0.,2.], [.5,1.2,0.,2.], [.5,.01,0.,-5.], [.5,1.2,0.,2.]]),
        np.tile([.5,.4],(4,1)), np.full(4,.5),
    ))
    env.touch_count[0] = [1,0,1,1]
    _, reward, terminal, _, info = env.step(np.zeros((4,6)))
    assert terminal[0] and reward[0] > 90
    assert not terminal[1] and reward[1] < 1
    assert terminal[2] and info['conceded'][2] == 1 and reward[2] < 0
    assert not terminal[3] and reward[3] < 1


def test_wide_banks_cover_goal_edges_and_medium_speeds():
    from airhockey.policy_benchmark import bank_defense_launches
    p = bank_defense_launches(91827, 1024, speed_range=(3,12), goal_half_width=.14)
    crossing, aimed, banks, _ = first_goal_crossing(np.column_stack((p[:,0],2-p[:,1],p[:,2],-p[:,3])), return_route=True)
    # Near-edge centerline aims can clip the near goal lip; evaluation excludes
    # attacks that do not actually score with the defender removed.
    assert np.mean(aimed) > .9
    assert np.count_nonzero(crossing < .4) > 90
    assert np.count_nonzero(crossing > .6) > 90
    assert np.count_nonzero(np.linalg.norm(p[:,2:],axis=1) < 6) > 250
    assert (banks == 1).all()


def test_wide_defense_training_preserves_explicit_fixtures_and_actor_layout():
    env = NeuralTrainingEnv(400, stage=5, productive_receive_drill=True,
                           practice_defense_fraction=.6, defense_min_speed=3,
                           wide_defense=True, shot_power_exponent=2,
                           shot_conditioned=True, realistic=False, randomize=False)
    obs = env.reset(seed=9732)
    assert obs.shape == (400,45)
    defense = env.kind == 2
    assert np.count_nonzero(np.hypot(env.engine.puck_vx[defense],env.engine.puck_vy[defense]) < 6) > 15
    fixture = Fixtures(np.array([2]), np.array([[.5,1.2,0.,-9.]]),
                       np.array([[.5,.25]]), np.array([.5]))
    env.reset(mask=np.arange(400)==399, fixtures=fixture)
    np.testing.assert_allclose([env.engine.puck_x[399],env.engine.puck_y[399],env.engine.puck_vx[399],env.engine.puck_vy[399]],fixture.puck[0])
