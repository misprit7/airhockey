import numpy as np

from airhockey.arrival_env import ArrivalEnv
from airhockey.policy_benchmark import LegacyTrials, fixtures, bank_defense_launches
from airhockey.skill_benchmark import make_fixtures


def test_untouched_stationary_skill_never_gets_a_referee_goal_or_new_puck():
    env = LegacyTrials(16, randomize=False, realistic=False)
    f = make_fixtures(20261101, 16).take(np.arange(16))
    env.reset(seed=3, fixtures=f)
    xy = f.puck[:, :2].copy()
    action = np.zeros((16, 6))
    position = f.paddle
    action[:, :2] = (
        2
        * (position - env.base._action_low)
        / (env.base._action_high - env.base._action_low)
        - 1
    )
    for _ in range(100):
        env.step(action)
    np.testing.assert_allclose(env.engine.puck_x, xy[:, 0])
    np.testing.assert_allclose(env.engine.puck_y, xy[:, 1])
    assert (env.engine.score_agent == 0).all()
    assert (env.contacts == 0).all()


def test_game_slots_keep_referee_while_skill_slots_disable_it():
    env = ArrivalEnv(4, game_fraction=0.5, randomize=False, realistic=False)
    env.reset(seed=3)
    env.engine.puck_x[:] = 0.5
    env.engine.puck_y[:] = 0.6
    env.engine.puck_vx[:] = env.engine.puck_vy[:] = 0
    env.base._puck_slow_count[:] = 10000
    _, _, _, _, info = env.step(np.zeros((4, 6)))
    assert (info["penalty"][:2] < 0).all()
    np.testing.assert_array_equal(info["penalty"][2:], 0)
    np.testing.assert_allclose(env.engine.puck_y[:2], 1.0)
    np.testing.assert_allclose(env.engine.puck_y[2:], 0.6)


def test_defense_fixtures_are_on_goal_without_a_rail_bounce():
    f, task = fixtures(55, 100)
    p = f.puck[task == 3]
    crossing = p[:, 0] - p[:, 1] * p[:, 2] / p[:, 3]
    assert np.all((crossing >= 0.43) & (crossing <= 0.57))
    assert np.all(
        (np.linalg.norm(p[:, 2:], axis=1) >= 2)
        & (np.linalg.norm(p[:, 2:], axis=1) <= 8)
    )


def test_banked_defense_launches_really_score_without_a_defender():
    from airhockey.physics import TableConfig
    from airhockey.shot_flight import open_goal_outcomes

    p = bank_defense_launches(35, 64)
    assert (p[::2, 2] > 0).all() and (p[1::2, 2] < 0).all()
    speed = np.linalg.norm(p[:, 2:], axis=1)
    assert ((speed >= 8) & (speed <= 12)).all()
    p[:, 1] = TableConfig().height - p[:, 1]
    p[:, 3] *= -1
    assert open_goal_outcomes(p).all()


def test_defense_trial_does_not_get_a_second_serve_after_block():
    env = ArrivalEnv(1, randomize=False, realistic=False)
    env.reset(seed=3)
    env.task[:] = 3
    env.defense_trials = np.ones(1, bool)
    env.engine.puck_x[:] = 0.8
    env.engine.puck_y[:] = 0.8
    env.engine.puck_vx[:] = env.engine.puck_vy[:] = 0
    env.base._puck_slow_count[:] = 10000
    _, _, _, _, info = env.step(np.zeros((1, 6)))
    np.testing.assert_array_equal(info["penalty"], 0)
    np.testing.assert_allclose(env.engine.puck_x, 0.8)
    np.testing.assert_allclose(env.engine.puck_y, 0.8)


def test_selfplay_game_aim_request_is_symmetric():
    env = ArrivalEnv(8, game_fraction=1, selfplay_fraction=1)
    obs = env.reset(seed=31)
    rival = env.opponent_obs()
    np.testing.assert_array_equal(obs[:, 37], rival[:, 37])
    np.testing.assert_allclose(obs[:, 37], 0.5)


def test_direct_defense_fixture_speed_limit_is_explicit():
    import pytest
    from airhockey.policy_benchmark import fixtures
    from airhockey.physics import TableConfig
    with pytest.raises(ValueError):
        fixtures(17,16,defense_speed_range=(10,16))
    cfg=TableConfig();cfg.max_puck_speed=16
    f,tasks=fixtures(17,64,defense_speed_range=(10,16),config=cfg)
    speeds=np.linalg.norm(f.puck[tasks==3,2:],axis=1)
    assert speeds.min()>=10 and speeds.max()<=16
    assert (speeds>12).any()
    assert TableConfig().max_puck_speed==12
