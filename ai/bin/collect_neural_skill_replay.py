#!/usr/bin/env python3
"""Collect successful learned shots for training-only skill preservation."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
import torch
from airhockey.neural_player import NeuralPlayer,PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.policy_benchmark import fixtures

p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-task',type=int,default=256);p.add_argument('--seed',type=int,default=20263142);a=p.parse_args()
torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
net=NeuralPlayer(state['width'],history=state.get('history',1),shot_conditioned=state.get('shot_conditioned',False));net.load_weights(state['model']);net.eval()
observations=[];actions=[];route_counts={}
for route in ('left','right','straight'):
 f,tasks=fixtures(a.seed,a.per_task,wide=True,defense_speed_range=(8,12));env=NeuralTrainingEnv(len(tasks),stage=2,seed=a.seed,report_sensing=True,shot_conditioned=True,shot_request=route,random_practice_opponent=True)
 rng=np.random.default_rng(a.seed+23);ids=tasks==2
 f.puck[ids,1]=rng.uniform(1,1.25,a.per_task);f.puck[ids,2]=rng.uniform(-1,1,a.per_task);f.puck[ids,3]=-rng.uniform(3,8,a.per_task)
 f.paddle[ids,1]=rng.uniform(.25,.45,a.per_task);travel=(f.puck[ids,1]-.4)/-f.puck[ids,3]
 f.paddle[ids,0]=np.clip(f.puck[ids,0]+f.puck[ids,2]*travel+rng.uniform(-.05,.05,a.per_task),env.decoder.low[0]+.01,env.decoder.high[0]-.01)
 obs=env.reset(seed=a.seed,fixtures=f);env.kind[:]=np.where(tasks==3,2,tasks)
 history=PhysicalHistory(net.history);history.reset(obs);ring=collections.deque(maxlen=16);done=np.zeros(len(tasks),bool);good=np.zeros(len(tasks),bool)
 original=env.engine.contact_callback
 def contact(event):
  ids=event['indices'];before=env.shot_count[0,ids].copy();aimed=env.aimed_count[0,ids].copy();original(event)
  if event['body']=='agent':
   keep=(before==0)&(env.shot_count[0,ids]>0)&(env.aimed_count[0,ids]>aimed)&(env.last_shot_route[0,ids]==env.last_shot_request[0,ids])&(np.linalg.norm(event['outgoing_before_speed_cap'],axis=1)>=6)&(tasks[ids]!=1)
   good[ids[keep]]=True
 env.engine.contact_callback=contact;count=0
 for tick in range(201):
  x=history.for_policy(net).copy();act=net.act(x);ring.append((x,act.copy()));good[:]=False
  obs,_,term,trunc,_=env.step(act);history.append(obs)
  selected=np.flatnonzero(good&~done)
  count+=len(selected)
  for prior,action in ring:
   keep=selected[prior[selected,1]<.9]
   if len(keep):observations.append(prior[keep].copy());actions.append(action[keep].copy())
  done|=term|trunc
  if done.all():break
 route_counts[route]=count
 print(route,count,flush=True)
x=np.concatenate(observations);y=np.concatenate(actions);assert np.isfinite(x).all() and np.isfinite(y).all()
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,observation=x,action=y)
a.output.with_suffix('.json').write_text(json.dumps({'teacher':str(a.checkpoint),'teacher_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'seed':a.seed,'successful_first_requested_fast_shots':route_counts,'examples':len(x),'selection':'own-half observations in the320ms preceding a first requested on-target shot >=6m/s; stationary/fast incoming/defense fixtures only; no outgoing recovery fixtures'},indent=2))
print('examples',len(x),flush=True)
