"""Sparse-score experiment: shaping must never reach the PPO reward."""
import numpy as np
import pytest

from airhockey.neural_training import NeuralTrainingEnv
from airhockey.skill_benchmark import Fixtures


def test_goals_only_ignores_all_shaping_and_reward_scaling():
    env = NeuralTrainingEnv(3, goals_only=True, realistic=False, randomize=False,
        load_weight=999, capture_weight=999, setup_weight=999,
        shot_power_weight=999, turnover_weight=999, game_goal_weight=600,
        readiness_cost_weight=999, reward_scale=123)
    env.reset(fixtures=Fixtures(np.full(3, 3),
        np.array([[.5, 1.99, 0., 10.], [.5, .01, 0., -10.], [.5, 1., .2, .1]]),
        np.array([[.2,.3]]*3), np.full(3, .2)))
    # Full-game reset deliberately overrides drill fixtures with a serve.
    # Place actual goal crossings after that reset, away from both paddles.
    env.engine.puck_x[:] = .5
    env.engine.puck_y[:] = [1.99,.01,1.]
    env.engine.puck_vx[:] = 0
    env.engine.puck_vy[:] = [10.,-10.,.1]
    action = np.full((3,6), .8)
    env.set_opponent_action(-action)
    _, reward, terminal, _, info = env.step(action)
    np.testing.assert_array_equal(info['goals'], [1,0,0])
    np.testing.assert_array_equal(info['conceded'], [0,1,0])
    np.testing.assert_array_equal(reward, [1.,-1.,0.])
    assert not terminal.any()


def test_goals_only_reset_allocates_only_full_games_and_no_time_penalty():
    env = NeuralTrainingEnv(8, goals_only=True, stage=0, game_fraction=.1,
                            realistic=False, randomize=False, game_episode_seconds=.06)
    env.reset(seed=211)
    assert np.all(env.kind == 3)
    assert env.base.SHOT_CLOCK_S == 0
    for _ in range(4):
        env.set_opponent_action(np.zeros((8,6)))
        _, reward, terminal, truncated, info = env.step(np.zeros((8,6)))
        np.testing.assert_array_equal(reward, info['goals']-info['conceded'])
        assert not terminal.any()
    assert truncated.all()


@pytest.mark.parametrize('flag', ['terminate_overload','receive_drill','productive_receive_drill','possession_followthrough'])
def test_goals_only_rejects_auxiliary_terminations(flag):
    with pytest.raises(ValueError, match='full games'):
        NeuralTrainingEnv(1, goals_only=True, **{flag:True})
