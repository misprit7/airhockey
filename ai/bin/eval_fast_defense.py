#!/usr/bin/env python3
"""Paired first-attack defense benchmark. Simulation only; no hardware I/O.

Hidden release times/targets/routes, verified on-goal counterfactuals, and
fixed seeds shared across candidates. A save ends the attack when contact
reverses its goalward travel or stops it; subsequent possession earns nothing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from eval_neural_player import load
from airhockey.neural_player import PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_setup import checkpoint_environment
from airhockey.neural_coordinates import CoordinateReference
from airhockey.neural_possession import goal_shot_velocity
from airhockey.skill_benchmark import Fixtures
from airhockey.shot_flight import PARAMETERS, open_goal_outcomes
from airhockey.recorder import Recorder, FrameData

BASE = Path('runs/rail30-100-20261001-v1/agent.pt')


def interval(successes, count):
    if not count:
        return [0., 1.]
    p = successes / count
    den = 1 + 1.96**2 / count
    center = (p + 1.96**2 / (2*count)) / den
    half = 1.96*np.sqrt(p*(1-p)/count + 1.96**2/(4*count**2))/den
    return [float(center-half), float(center+half)]


def evaluate(checkpoint, *, count=768, seed=20261051, environment_from=BASE,
             immediate=False, record_prefix=None, device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    net, state = load(checkpoint)
    options = checkpoint_environment(environment_from)
    if net.environment_options != options:
        net = CoordinateReference(net, net.environment_options, options)
    net.to(device).eval()
    env = NeuralTrainingEnv(count, stage=2, seed=seed, report_sensing=True,
        shot_conditioned=net.shot_conditioned, **options,
        defense_windup_lateral_speed=.4, defense_windup_bank_fraction=2/3)
    env.cfg.max_puck_speed = 18
    rng = np.random.default_rng(seed+31)
    route = np.tile(np.array([0,-1,1]), (count+2)//3)[:count]
    rng.shuffle(route)
    speed = rng.uniform(10,18,count)
    puck = np.zeros((count,4))
    puck[:,:2] = rng.uniform([.15,1.08],[.85,1.45],(count,2))
    puck[:,2] = rng.uniform(-.4,.4,count) if not immediate else 0
    paddle = rng.uniform(env.decoder.low+.005,env.decoder.high-.005,(count,2))
    aim = rng.uniform(.5-env.cfg.goal_width/2+env.cfg.puck_radius+.012,
                      .5+env.cfg.goal_width/2-env.cfg.puck_radius-.012,count)
    delay = np.zeros(count) if immediate else rng.uniform(.35,1.2,count)
    obs = env.reset(seed=seed,fixtures=Fixtures(np.full(count,2),puck,paddle,aim))
    env.defense_windup[:] = True
    env._windup_release[:] = delay
    env._windup_aim[:] = aim
    env._windup_bank_side[:] = route
    env._windup_velocity[:] = goal_shot_velocity(puck[:,:2],aim,speed,route,env.cfg,
                                               env.engine.wall_restitution,env.engine.wall_tangential)
    env.engine.paddle_opp_x[:] = .5
    env.engine.paddle_opp_y[:] = 1.8
    loads = rng.choice([.25,.70],count)
    for model in env.loads:
        model.h[:] = loads[:,None,None]**2
        model.observed[:] = loads[:,None,None]
        model.gain[:] = 1.3
    obs = env._features(env.base._make_obs_direct())
    history = PhysicalHistory(net.history);history.reset(obs)
    done=np.zeros(count,bool);saved=done.copy();conceded=done.copy();released=done.copy()
    launch=np.zeros((count,4));release_paddle=np.zeros((count,2))
    peak=np.zeros(count);overload=np.zeros(count);touched=done.copy()
    traces=[]
    e=env.engine
    for tick in range(170):
        releasing=(env.elapsed>=env._windup_release)&~released
        if releasing.any():
            ids=np.flatnonzero(releasing)
            pos=np.column_stack((e.puck_x[ids],e.puck_y[ids]))
            vel=goal_shot_velocity(pos,aim[ids],speed[ids],route[ids],env.cfg,
                                  e.wall_restitution[ids],e.wall_tangential[ids])
            launch[ids]=np.column_stack((pos[:,0],env.cfg.height-pos[:,1],vel[:,0],-vel[:,1]))
            release_paddle[ids]=np.column_stack((e.paddle_agent_x[ids],e.paddle_agent_y[ids]))
            released[ids]=True
        obs,_,terminal,truncated,info=env.step(net.act(history.for_policy(net)))
        history.append(obs)
        active=~done
        touched |= active & (env.touch_count[0]>0) & released
        failed=active & (e.score_opponent>0)
        success=active & released & touched & ~failed & (
            (e.puck_vy>.05) | (np.hypot(e.puck_vx,e.puck_vy)<.3))
        saved |= success;conceded |= failed
        peak=np.maximum(peak,np.where(active,info['load_peak'],0))
        overload=np.maximum(overload,np.where(active,info['overload_seconds'],0))
        if record_prefix:
            traces.append(np.column_stack((e.puck_x,e.puck_y,e.puck_vx,e.puck_vy,
                e.paddle_agent_x,e.paddle_agent_y,e.paddle_opp_x,e.paddle_opp_y,
                e.score_agent,e.score_opponent)).copy())
        done |= failed|success|terminal|truncated|(released&(env.elapsed>delay+1.5))
        if done.all():break
    valid=released & open_goal_outcomes(launch,{k:getattr(e,k) for k in PARAMETERS},env.cfg)
    rows=[dict(valid=bool(valid[i]),saved=bool(saved[i]),conceded=bool(conceded[i]),
        contacted=bool(touched[i]),route=int(route[i]),speed=float(speed[i]),initial_load=float(loads[i]),
        release_paddle=release_paddle[i].tolist(),peak_load=float(peak[i]),
        overload_seconds=float(overload[i]),launch=launch[i].tolist()) for i in range(count)]
    def summary(mask):
        mask=mask&valid;n=int(mask.sum());s=int(saved[mask].sum())
        return dict(trials=n,saved=s,block_rate=s/max(1,n),ci95=interval(s,n),
            conceded=int(conceded[mask].sum()),unresolved=int((~done&mask).sum()),
            mean_release_depth=float(release_paddle[mask,1].mean()) if n else None,
            peak_load=float(peak[mask].max()) if n else None,
            overload_seconds=float(overload[mask].sum()))
    groups={name:summary(route==code) for name,code in [('straight',0),('left_bank',-1),('right_bank',1)]}
    groups.update({name:summary(mask) for name,mask in [('10-14m_s',speed<14),('14-18m_s',speed>=14),('cold',loads<.5),('warm',loads>=.5)]})
    recordings=[]
    if record_prefix:
        for name,code in [('straight',0),('left_bank',-1),('right_bank',1)]:
            for outcome in (False,True):
                ids=np.flatnonzero(valid&(route==code)&(saved==outcome))
                if not len(ids):continue
                i=int(ids[0]);rec=Recorder()
                for t,frame in enumerate(traces):
                    a=frame[i]
                    rec.record(FrameData(time=(t+1)*.02,puck_x=a[0],puck_y=a[1],puck_vx=a[2],puck_vy=a[3],
                        agent_x=a[4],agent_y=a[5],opponent_x=a[6],opponent_y=a[7],score_agent=int(a[8]),score_opponent=int(a[9]),
                        rally_event='hidden release' if abs((t+1)*.02-delay[i])<.025 else ''))
                    if (t+1)*.02>delay[i]+1.5:break
                dest=Path(str(record_prefix)+f'-{name}-'+('save' if outcome else 'miss')+'.json')
                rec.save(dest,metadata=dict(run_name=Path(checkpoint).parent.name,checkpoint=str(checkpoint),
                    simulation_only=True,match_type='isolated defense',scenario=name,selected_outcome='save' if outcome else 'miss',
                    selection='first valid example of each outcome; not a random sample',seed=seed,trial=i,fps=50,
                    workspace_sim=env.base._ws,accel_cap_m_s2=options['accel']))
                recordings.append('/?replay='+dest.name)
    return dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        seed=seed,immediate=immediate,metric='First on-goal attack blocked; no reward for control or counterattack',
        summary=summary(np.ones(count,bool)),groups=groups,details=rows,replays=recordings)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=768);p.add_argument('--seed',type=int,default=20261051)
    p.add_argument('--immediate',action='store_true');p.add_argument('--record-prefix',type=Path)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();r=evaluate(a.checkpoint,count=a.trials,seed=a.seed,immediate=a.immediate,record_prefix=a.record_prefix,device=a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k!='details'}),flush=True)
