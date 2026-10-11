import numpy as np
import pytest
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.skill_benchmark import Fixtures
from airhockey.defense_baselines import DefensiveBaseline
from airhockey.neural_setup import workspace_bounds


def test_slow_cushion_rewards_first_deceleration_not_stationary_or_repeated_taps():
    envs=[NeuralTrainingEnv(4,stage=2,realistic=False,randomize=False,
                            slow_cushion_weight=w) for w in (0,40)]
    event=dict(body='agent',indices=np.arange(4),
        incoming=np.array([[0,-.5],[0,-.5],[0,-.5],[0,0]]),
        outgoing_before_speed_cap=np.array([[0,.05],[0,.5],[0,.9],[0,.05]]),
        normal=np.tile([0,1],(4,1)))
    for e in envs:
        e.reset(seed=123);e._contact(event)
    np.testing.assert_allclose((envs[1].reward_events-envs[0].reward_events)[0],[36,0,-32,0])
    for e in envs:
        e.reward_events[:]=0;e._contact(event)
    np.testing.assert_allclose(envs[0].reward_events,envs[1].reward_events)
    # A later possession earns a new first-touch opportunity even though
    # the episode's cumulative touch counter is greater than one.
    for e in envs:
        e.engine.puck_x[:]=.5;e.engine.puck_y[:]=1.2
        e.engine.puck_vx[:]=0;e.engine.puck_vy[:]=0
        e.step(np.zeros((4,6)))
        e.reward_events[:]=0;e._contact(event)
    np.testing.assert_allclose((envs[1].reward_events-envs[0].reward_events)[0],[36,0,-32,0])


def test_strict_capture_requires_slow_recent_contact_and_interior_position():
    e=NeuralTrainingEnv(4,stage=2,realistic=False,randomize=False,
        capture_max_speed=.15,capture_duration=.2,capture_depth_margin=.04)
    top=e.decoder.high[1]
    paddle=np.array([[.5,.3],[.5,.3],[.5,.3],[.5,top-.10]])
    puck=np.column_stack((paddle+[0,.11],np.zeros((4,2))))
    puck[1,2]=.4
    e.reset(seed=9,fixtures=Fixtures(np.full(4,3),puck,paddle,np.full(4,.5)))
    e.contact_clock[0]=[0,0,-100,0]
    e.control_time[0]=.3
    action=np.zeros((4,6));action[:,:2]=2*(paddle-e.decoder.low)/(e.decoder.high-e.decoder.low)-1
    action[:,4]=1;action[:,5]=-1
    e.step(action)
    np.testing.assert_array_equal(e.capture_count[0],[1,0,0,0])
    # No repeat capture payment while holding a puck already captured.
    paid=e.capture_paid.copy();e.step(action)
    np.testing.assert_array_equal(e.capture_count[0],[1,0,0,0])
    np.testing.assert_array_equal(e.capture_paid,paid)


def test_slow_receiving_curriculum_and_exact_fixtures():
    e=NeuralTrainingEnv(400,stage=5,realistic=False,randomize=False,
        productive_receive_drill=True,slow_receiving_fraction=1,recovery_fraction=0,game_fraction=0)
    e.reset(seed=123)
    mask=e.kind==1
    assert mask.sum()>20
    assert np.all((e.engine.puck_y[mask]>=.78)&(e.engine.puck_y[mask]<1))
    assert np.all((e.engine.puck_vy[mask]>=-1.2)&(e.engine.puck_vy[mask]<=-.15))
    f=Fixtures(np.full(400,1),np.tile([.4,.6,0,-4],(400,1)),np.tile([.5,.3],(400,1)),np.full(400,.5))
    e.reset(seed=123,fixtures=f)
    np.testing.assert_allclose(e.engine.puck_vy,-4)


def test_defender_only_uses_observed_physics_and_ignores_shot_request():
    o=dict(workspace_bounds_mm=workspace_bounds('rail30'))
    b=DefensiveBaseline(o)
    x=np.zeros((2,675));x[:,:6]=[.8,1.1,.5,-1,.5,.4]
    x[0,-3:]=[1,0,0];x[1,-3:]=[0,1,0]
    a=b.act(x)
    np.testing.assert_array_equal(a[0],a[1])
    assert np.isfinite(a).all() and np.max(abs(a))<=1
    position,_,_,fraction=b.decoder.unpack(a)
    np.testing.assert_allclose(position[:,1],.18)
    np.testing.assert_allclose(fraction,1)
    # Bank prediction must differ from treating a future rail as nonexistent.
    direct=DefensiveBaseline(o,'deep-direct').act(x)
    assert not np.isclose(a[0,0],direct[0,0])


@pytest.mark.parametrize('kw',[{'capture_max_speed':0},{'capture_duration':0},
    {'slow_cushion_weight':-1},{'slow_receiving_fraction':2},{'control_speed_scale':0}])
def test_invalid_control_settings(kw):
    with pytest.raises(ValueError):NeuralTrainingEnv(1,**kw)
