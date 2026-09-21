"""Run-local actuator limits and the consequences of off-target strikes."""
import json
from pathlib import Path

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.dynamics import AGENT_DR_ACCEL_M_S2, MAX_ACCEL_M_S2
from airhockey.rewards import BatchRewardShaper, curriculum_shaper_kwargs


def shaper(**kwargs):
    settings = dict(proximity_weight=0, contact_reward=0, directed_hit_weight=0,
                    puck_progress_weight=0, defense_weight=0, shot_placement_weight=0,
                    goal_reward=0, goal_penalty=0, entropy_weight=0, shot_mix_weight=0,
                    off_target_penalty=15, on_target_reward=40)
    settings.update(kwargs)
    return BatchRewardShaper(1, **settings)


def frame(x=.5, y=.4, vx=0, vy=0, pad_x=.5, pad_y=.3):
    return {k: np.array([v]) for k, v in dict(
        puck_x=x, puck_y=y, puck_vx=vx, puck_vy=vy,
        pad_x=pad_x, pad_y=pad_y, score_agent=0, score_opponent=0,
        t_side=0).items()}


OBS = np.zeros((1, BatchAirHockeyEnv.OBS_DIM), dtype=np.float32)


def step(s, **kwargs):
    return float(s.compute(OBS, np.zeros(1), info=frame(**kwargs))[0])


def test_miss_cost_is_not_discounted_by_control_gate_and_resets_per_visit():
    s = shaper(control_gate=True, patience_s=2, patience_floor=.05)
    s.reset(OBS, info=frame())
    assert step(s, vx=6, vy=.5) == pytest.approx(-15)
    step(s)  # a second strike in the same possession cannot pay a second fine
    assert step(s, vx=6, vy=.5) == 0
    step(s, y=1.5, vy=2)
    step(s, y=.9, vy=-.5)
    step(s)
    assert step(s, vx=6, vy=.5) == pytest.approx(-15)
    s.reset(OBS, mask=np.array([True]), info=frame())
    assert step(s, vx=6, vy=.5) == pytest.approx(-15)


@pytest.mark.parametrize('vx,vy', [(1, .1), (0, -4), (0, .2)])
def test_control_touches_and_incoming_puck_are_not_missed_shots(vx, vy):
    s = shaper()
    s.reset(OBS, info=frame())
    assert step(s, vx=vx, vy=vy) == 0
    assert s.stats['off_target'] == 0


def test_on_target_reward_and_miss_penalty_can_work_independently():
    s = shaper()
    s.reset(OBS, info=frame())
    assert step(s, vy=4) == pytest.approx(40)
    assert s.stats['shot_attempts'] == s.stats['aimed_attempts'] == 1
    assert s.stats['off_target'] == 0
    s = shaper(on_target_reward=0)
    s.reset(OBS, info=frame())
    assert step(s, vx=6, vy=.5) == pytest.approx(-15)


def test_miss_can_be_corrected_without_losing_the_on_target_reward():
    s = shaper()
    s.reset(OBS, info=frame())
    assert step(s, vx=6, vy=.5) == pytest.approx(-15)
    step(s)
    assert step(s, vy=4) == pytest.approx(40)


def test_miss_penalty_ramps_with_speed_and_goal_resets_it():
    s = shaper()
    s.reset(OBS, info=frame(x=.7, pad_x=.7))
    # Straight shot well outside the mouth, midway through the speed ramp.
    assert step(s, x=.7, pad_x=.7, vy=2.25) == pytest.approx(-7.5)
    info = frame(x=.7, pad_x=.7)
    info['score_opponent'][:] = 1
    s.compute(OBS, np.zeros(1), info=info)
    assert step(s, x=.7, pad_x=.7, vy=3) == pytest.approx(-15)


def test_a_valid_bank_is_on_target_not_a_miss():
    from airhockey.rewards import predict_shot
    s = shaper()
    # Find an actual bank trajectory using the lossy-rail geometry.
    vxs = np.linspace(-6, 6, 1001)
    cfg = s._cfg
    xg, _, nb, _ = predict_shot(np.full_like(vxs, .5), np.full_like(vxs, .4),
                               vxs, np.full_like(vxs, 4), cfg.width, cfg.height,
                               cfg.puck_radius, cfg.wall_restitution, cfg.wall_tangential)
    valid = (nb == 1) & (np.abs(xg - cfg.width / 2) < cfg.goal_width / 4)
    assert valid.any()
    s.reset(OBS, info=frame())
    assert step(s, vx=vxs[np.flatnonzero(valid)[0]], vy=4) == pytest.approx(40)
    assert s.stats['off_target'] == 0


def test_run_local_acceleration_survives_full_and_partial_resets():
    env = BatchAirHockeyEnv(n_envs=4, domain_randomize=True, opponent_body='robot',
                           dynamics_max_accel=60, agent_accel_range=(60, 60),
                           action_mode='profile_a')
    obs = env.reset(seed=123)
    for _ in range(3):
        np.testing.assert_allclose(env._agent_dyn['max_accel'], 60)
        np.testing.assert_allclose(env._opp_dyn['max_accel'], 60)
        np.testing.assert_allclose(obs[:, 14], 60 / MAX_ACCEL_M_S2)
        obs = env.reset(mask=np.array([True, False, True, False]))
    default = BatchAirHockeyEnv(n_envs=2, domain_randomize=True)
    default.reset(seed=123)
    np.testing.assert_allclose(default._agent_dyn['max_accel'], AGENT_DR_ACCEL_M_S2[0])
    assert AGENT_DR_ACCEL_M_S2 == (40, 40)  # deployment defaults stay unchanged


def test_recipe_rewards_goals_fully_without_changing_default_recipe():
    recipe = json.loads((Path(__file__).parents[1] / 'recipes/accel60-accuracy.json').read_text())
    kwargs = curriculum_shaper_kwargs('selfplay') | recipe
    assert kwargs['patience_on_goals'] is False
    assert curriculum_shaper_kwargs('selfplay')['patience_on_goals'] is True
    s = shaper(goal_reward=100, patience_on_goals=kwargs['patience_on_goals'],
               control_gate=True, patience_s=2, patience_floor=.05)
    s.reset(OBS, info=frame())
    info = frame()
    info['score_agent'][:] = 1
    assert s.compute(OBS, np.zeros(1), info=info)[0] == pytest.approx(100)
