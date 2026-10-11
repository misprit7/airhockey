#!/usr/bin/env python3
"""Paired control benchmark: no reward/actor access to fixture labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from eval_neural_player import load
from eval_fast_defense import BASE, interval
from airhockey.neural_player import PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_setup import checkpoint_environment
from airhockey.neural_coordinates import CoordinateReference
from airhockey.skill_benchmark import Fixtures


def evaluate(checkpoint, *, count=384, seed=20261072, device='cpu'):
    torch.set_num_threads(2)
    net, _ = load(checkpoint)
    options = checkpoint_environment(BASE)
    if net.environment_options != options:
        net = CoordinateReference(net, net.environment_options, options)
    net.to(device).eval()
    env = NeuralTrainingEnv(count, stage=2, shot_conditioned=True, report_sensing=True,
        possession_followthrough=True, capture_max_speed=.15, capture_duration=.20,
        capture_depth_margin=.04, **options)
    rng = np.random.default_rng(seed)
    task = np.arange(count) % 3
    puck = np.zeros((count,4))
    puck[:,0] = rng.uniform(.2,.8,count)
    puck[:,1] = rng.uniform(.35,.7,count)
    puck[task==0,1] = rng.uniform(.86,.98,(task==0).sum())
    puck[:,2] = rng.uniform(-.15,.15,count)
    puck[:,3] = rng.uniform(.15,1.0,count)*np.where(task==2,1,-1)
    paddle = rng.uniform(env.decoder.low+.01,env.decoder.high-.01,(count,2))
    # Avoid overlapping initial states; retain natural varied approach angles.
    overlapping = np.linalg.norm(paddle-puck[:,:2],axis=1) < .13
    paddle[overlapping,1] = np.maximum(env.decoder.low[1]+.01,puck[overlapping,1]-.2)
    obs = env.reset(seed=seed,fixtures=Fixtures(np.ones(count,dtype=int),puck,paddle,np.full(count,.5)))
    env.engine.paddle_opp_x[:] = .5
    env.engine.paddle_opp_y[:] = 1.8
    for model in env.loads:
        model.h[:] = .4**2
        model.observed[:] = .4
        model.gain[:] = 1.3
    obs = env._features(env.base._make_obs_direct())
    history = PhysicalHistory(net.history);history.reset(obs)
    done = np.zeros(count,bool)
    settled = done.copy(); converted = done.copy(); fast = done.copy(); weak_exit = done.copy()
    peak = np.zeros(count); overload = np.zeros(count)
    first_contact_depth = np.full(count, np.nan)
    for tick in range(300):
        obs,_,terminal,truncated,info = env.step(net.act(history.for_policy(net)))
        history.append(obs)
        active = ~done
        first = active & np.isnan(first_contact_depth) & (env.touch_count[0]>0)
        first_contact_depth[first] = env.engine.puck_y[first]
        settled |= active & (env.capture_count[0]>0)
        converted |= active & (env.convert_count[0]>0)
        fast |= active & (env.fast_aimed_count[0]>0)
        weak_exit |= active & (env.engine.puck_y>=1) & ~fast & (env.engine.score_agent==0)
        peak = np.maximum(peak,np.where(active,info['load_peak'],0))
        overload = np.maximum(overload,np.where(active,info['overload_seconds'],0))
        done |= terminal|truncated|(env.engine.puck_y>=1)
        if done.all():break
    def summary(mask):
        n=int(mask.sum())
        touched=mask & np.isfinite(first_contact_depth)
        return dict(trials=n,settled=int(settled[mask].sum()),settle_rate=float(settled[mask].mean()),
            settle_ci95=interval(int(settled[mask].sum()),n),
            converted_rate=float(converted[mask].mean()),fast_aimed_rate=float(fast[mask].mean()),
            weak_exit_rate=float(weak_exit[mask].mean()),
            first_contact_depth=float(first_contact_depth[touched].mean()) if touched.any() else None,
            peak_load=float(peak[mask].max()),overload_seconds=float(overload[mask].sum()))
    return dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),seed=seed,
        metric='Touched puck below 0.15 m/s for 0.20 s, within 7 cm of paddle edge and 4 cm inside forward reach; six-second trial continues through a shot',
        summary=summary(np.ones(count,bool)),groups={name:summary(task==i) for i,name in enumerate(['shallow_incoming','interior_incoming','drifting_away'])},
        details=[dict(task=int(task[i]),settled=bool(settled[i]),converted=bool(converted[i]),fast_aimed=bool(fast[i]),weak_exit=bool(weak_exit[i])) for i in range(count)])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=384)
    p.add_argument('--seed',type=int,default=20261072);p.add_argument('--device',default='cpu')
    a=p.parse_args();result=evaluate(a.checkpoint,count=a.trials,seed=a.seed,device=a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
