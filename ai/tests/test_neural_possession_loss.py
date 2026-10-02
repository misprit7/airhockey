import numpy as np
import pytest

from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_possession import direct_goal_coverage_cost
from airhockey.skill_benchmark import Fixtures


def test_release_uncertainty_exposes_forward_position_without_prescribing_home():
    env = NeuralTrainingEnv(3, realistic=False, randomize=False)
    puck = np.tile([.5, 1.08], (3, 1))
    paddle = np.array([[.5, .12], [.5, .4], [.5, .65]])
    fixed = direct_goal_coverage_cost(puck, paddle, np.zeros_like(paddle), env.decoder.bounds, env.cfg)
    robust = direct_goal_coverage_cost(puck, paddle, np.zeros_like(paddle), env.decoder.bounds, env.cfg, lateral_uncertainty=.25)
    np.testing.assert_array_equal(fixed, 0)
    assert robust[0] < robust[1] < robust[2]
    reflected = paddle.copy(); reflected[:, 0] = 1-reflected[:, 0]
    np.testing.assert_allclose(robust, direct_goal_coverage_cost(puck, reflected, np.zeros_like(paddle), env.decoder.bounds, env.cfg, lateral_uncertainty=.25))


def test_known_fast_incoming_shot_does_not_use_release_uncertainty():
    a = NeuralTrainingEnv(1, realistic=False, randomize=False, readiness_weight=80)
    b = NeuralTrainingEnv(1, realistic=False, randomize=False, readiness_weight=80, readiness_lateral_uncertainty=.25)
    for env in (a, b):
        env.engine.puck_x[:] = .5; env.engine.puck_y[:] = 1.15
        env.engine.puck_vx[:] = 0; env.engine.puck_vy[:] = -10
        env.engine.paddle_agent_x[:] = .5; env.engine.paddle_agent_y[:] = .65
    np.testing.assert_allclose(a.potential(), b.potential())


def test_slow_loss_counts_after_capture_but_not_useful_shot_goal_or_no_opportunity():
    env = NeuralTrainingEnv(5, realistic=False, randomize=False)
    edge = env.decoder.high[1]+env.cfg.puck_radius+env.cfg.paddle_radius+.001
    env.engine.puck_x[:] = .5; env.engine.puck_y[:] = edge
    env.engine.puck_vx[:] = 0; env.engine.puck_vy[:] = .3
    env.entry_active[0] = True; env.entry_reachable_time[0] = .2
    env.entry_flags[0, 0, 1] = True  # Brief capture must not excuse later loss.
    env.entry_flags[0, 1, 3] = True  # Useful fast shot is different.
    env.entry_reachable_time[0, 3] = 0
    env.engine.puck_vy[4] = -1
    exclude = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(env._slow_possession_loss(0, exclude), [True, False, False, False, False])
    assert not env._slow_possession_loss(0, exclude).any()
    env.engine.puck_y[:] = env.cfg.height-edge; env.engine.puck_vy[:] = -.3
    env.entry_active[1] = True; env.entry_reachable_time[1] = .2
    assert env._slow_possession_loss(1, np.zeros(5, bool)).all()
    env.reset(mask=np.array([True, False, False, False, False]))
    assert not env._slow_loss_paid[:, 0].any()
    assert env._slow_loss_paid[1, 1:].all()


def test_penalty_reaches_actual_step_reward_once():
    rewards=[]
    for penalty in [0, 80]:
        env = NeuralTrainingEnv(1, stage=2, game_fraction=0, realistic=False, randomize=False,
                                slow_exit_penalty=penalty)
        y = env.decoder.high[1]+env.cfg.puck_radius+env.cfg.paddle_radius-.001
        env.reset(seed=13, fixtures=Fixtures(np.ones(1, int), np.array([[.5,y,0,.3]]), np.array([[.5,.2]]), np.array([.5])))
        env.entry_active[0] = True; env.entry_reachable_time[0] = .2; env.entry_flags[0, :, 1] = True
        action=np.zeros((1,6));action[:,4]=1;action[:,5]=-1
        rewards.append([env.step(action)[1][0],env.step(action)[1][0]])
        assert env.slow_possession_losses[0,0] == 1
    np.testing.assert_allclose(np.array(rewards[1])-rewards[0],[-80,0],atol=1e-7)


def test_moving_windup_reaims_from_real_puck_position_and_keeps_speed():
    env=NeuralTrainingEnv(16, stage=5, game_fraction=0, fixed_practice_roles=True,
        practice_defense_fraction=1, defense_windup_fraction=1,
        defense_windup_lateral_speed=.8, realistic=False, randomize=False)
    env.reset(seed=571)
    assert env.defense_windup.all()
    assert (np.abs(env.engine.puck_vx) > .01).any()
    initial=env.engine.puck_x.copy()
    env.step(np.zeros((16,6)))
    assert not np.allclose(initial,env.engine.puck_x)
    previous_speed=np.linalg.norm(env._windup_velocity,axis=1)
    before=np.column_stack((env.engine.puck_x,env.engine.puck_y))
    env._windup_release[:]=0
    env.step(np.zeros((16,6)))
    v=env._windup_velocity
    np.testing.assert_allclose(np.linalg.norm(v,axis=1),previous_speed)
    np.testing.assert_allclose(before[:,0]-before[:,1]*v[:,0]/v[:,1],env._windup_aim)
    assert np.isinf(env._windup_release).all()


@pytest.mark.parametrize('kwargs',[{'slow_exit_penalty':-1},{'readiness_lateral_uncertainty':np.nan},{'defense_windup_lateral_speed':-1}])
def test_invalid_new_settings(kwargs):
    with pytest.raises(ValueError):NeuralTrainingEnv(1,**kwargs)


def test_hidden_bank_windups_cover_both_rails_and_reaim_after_drift():
    env=NeuralTrainingEnv(128, stage=5, game_fraction=0, fixed_practice_roles=True,
        practice_defense_fraction=1, defense_windup_fraction=1,
        defense_windup_bank_fraction=.5, defense_windup_lateral_speed=.8,
        realistic=False, randomize=False)
    env.reset(seed=572)
    sides=env._windup_bank_side.copy()
    assert set(sides)=={-1,0,1}
    speed=np.linalg.norm(env._windup_velocity,axis=1)
    env.engine.puck_x[:]+=0.03
    start=np.column_stack((env.engine.puck_x,env.engine.puck_y))
    env._windup_release[:]=0
    env.step(np.zeros((128,6)))
    v=env._windup_velocity
    np.testing.assert_allclose(np.linalg.norm(v,axis=1),speed)
    for side in (-1,0,1):
        q=sides==side;xy=start[q];velocity=v[q]
        if side:
            wall=env.cfg.puck_radius if side<0 else env.cfg.width-env.cfg.puck_radius
            time=(wall-xy[:,0])/velocity[:,0]
            assert (time>0).all()
            xy=xy+velocity*time[:,None]
            velocity=velocity.copy()
            velocity[:,0]*=-env.engine.wall_restitution[q]
            velocity[:,1]*=env.engine.wall_tangential[q]
        crossing=xy[:,0]-xy[:,1]*velocity[:,0]/velocity[:,1]
        np.testing.assert_allclose(crossing,env._windup_aim[q],atol=1e-7)
    # Partial resets clear old hidden routes even if the next drill isn't a bank.
    env.defense_windup_bank_fraction=0
    mask=np.arange(128)%2 == 0
    env.reset(mask=mask)
    assert not env._windup_bank_side[mask].any()
    np.testing.assert_array_equal(env._windup_bank_side[~mask], sides[~mask])


@pytest.mark.parametrize('fraction',[-.1,1.1,float('nan')])
def test_invalid_bank_windup_fraction(fraction):
    with pytest.raises(ValueError,match='windup bank fraction'):
        NeuralTrainingEnv(1,defense_windup_bank_fraction=fraction)
