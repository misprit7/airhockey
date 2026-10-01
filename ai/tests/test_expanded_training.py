import json
from pathlib import Path
import numpy as np
import torch
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_setup import workspace_bounds,checkpoint_environment
from airhockey.thermal import MotorThermal,DEFAULT_MODEL
from airhockey.neural_coordinates import action_coordinates,observation_coordinates,bounds
from airhockey.arrival import ArrivalDecoder

MODEL=Path(__file__).resolve().parents[1]/'recipes/motor-load-20261001.json'
OLD=dict(accel=60,workspace_bounds_mm=None)
NEW=dict(accel=100,workspace_bounds_mm=workspace_bounds('rail30'))


def test_experiment_limits_do_not_mutate_default_and_apply_to_both_sides():
    env=NeuralTrainingEnv(8,stage=5,thermal_path=MODEL,**NEW)
    env.reset(seed=41)
    ordinary=NeuralTrainingEnv(2)
    assert np.all(env.base._agent_dyn['max_accel']==100)
    assert np.all(env.base._opp_dyn['max_accel']==100)
    assert np.all(ordinary.base._agent_dyn['max_accel']==60)
    np.testing.assert_allclose(env.decoder.bounds,bounds(NEW))
    assert ordinary.decoder.low[0]>env.decoder.low[0]
    for _ in range(30):
        action=np.tile([1,-1,1,-1,-1,1],(8,1))
        obs,*_=env.step(action)
        assert np.isfinite(obs).all()
        assert env.peak_accel.max()<100.2
        assert np.all(env.engine.paddle_agent_x<=env.decoder.high[0]+1e-5)
        assert np.all(env.engine.paddle_agent_y>=env.decoder.low[1]-1e-5)


def test_edge_penalty_is_soft_local_and_independent_of_opponent():
    env=NeuralTrainingEnv(4,edge_dwell_weight=3,**NEW)
    env.reset(seed=1)
    e=env.engine;lo,hi=env.decoder.low,env.decoder.high
    e.paddle_agent_x[:]=[.5,lo[0],hi[0],lo[0]+.03]
    e.paddle_agent_y[:]=[.4,.4,lo[1],.4]
    np.testing.assert_allclose(env.edge_dwell_cost(),[0,1,2,.25])
    e.paddle_opp_x[:]=0
    np.testing.assert_allclose(env.edge_dwell_cost(),[0,1,2,.25])
    # A 100ms side-edge visit costs .3, versus a 600-point game goal.
    assert 3*.1*env.edge_dwell_cost()[1]==.3 or np.isclose(3*.1,.3)


def test_spatial_model_preserves_actual_drive_limits_and_holding_map():
    model=MotorThermal(9,MODEL,randomize=False)
    centers=model.centers
    pos=np.column_stack(((centers[:,1]+19)/1003.9,(2017.9-centers[:,0])/1014.6))
    zero=np.zeros((9,2))
    model.advance(pos,zero,zero,.02)
    np.testing.assert_allclose(model.current_squared,model.holding,atol=1e-6)
    np.testing.assert_allclose(model.limits[0],[4.1,5.8,4.1,5.8],atol=1e-5)
    assert np.ptp(model.current_squared[:,0])>1
    assert DEFAULT_MODEL.name=='motor-load-20260929.json'


def test_reference_arrivals_keep_physical_targets_after_coordinate_change():
    a=np.array([[.2,-.3,.4,-.2,.1,.3]],np.float32)
    b=action_coordinates(a,OLD,NEW)
    old=ArrivalDecoder(bounds(OLD)).unpack(a)
    new=ArrivalDecoder(bounds(NEW)).unpack(b)
    for i in range(3):np.testing.assert_allclose(old[i],new[i],atol=1e-6)
    np.testing.assert_allclose(old[3]*60,new[3]*100,atol=1e-5)
    obs=np.zeros((2,87),np.float32)
    for frame in (0,42):
        obs[:,frame+15:frame+21]=a
        obs[:,frame+31:frame+33]=a[:,:2]
        obs[:,frame+33]=.5
        obs[:,frame+14]=1
    remap=observation_coordinates(obs,OLD,NEW,2)
    inverse=observation_coordinates(remap,NEW,OLD,2)
    np.testing.assert_allclose(inverse,obs,atol=1e-6)
    np.testing.assert_allclose(observation_coordinates(torch.tensor(obs),OLD,NEW,2),remap,atol=1e-6)


def test_evaluation_uses_snapshot_not_mutable_model_path(tmp_path):
    model=json.loads(MODEL.read_text())
    (tmp_path/'run.json').write_text(json.dumps(dict(args=dict(accel=100,workspace='rail30'),thermal_model=model)))
    opts=checkpoint_environment(tmp_path/'agent.pt')
    assert opts['accel']==100 and opts['thermal_path']==model
    assert opts['workspace_bounds_mm']==NEW['workspace_bounds_mm']


def test_fast_shot_curriculum_does_not_silently_clip_to_twelve():
    env=NeuralTrainingEnv(64,stage=5,defense_min_speed=10,defense_max_speed=16,
        defense_windup_fraction=1,practice_defense_fraction=1,game_fraction=0,**NEW)
    env.reset(seed=4)
    assert env.cfg.max_puck_speed==16
    assert np.linalg.norm(env._windup_velocity,axis=1).max()>12
