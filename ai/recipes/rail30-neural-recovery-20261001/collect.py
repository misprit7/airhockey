"""Select successful noisy NEURAL edge episodes for self-imitation. No scripted actions."""
from pathlib import Path
import json,hashlib,sys,time,argparse
import numpy as np
import torch
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'ai'),str(ROOT/'ai/bin')]
from eval_neural_player import load
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_player import PhysicalHistory,RecoveryExplorationBias,recovery_exploration_scale,thermal_effort_exploration_scale
from airhockey.skill_benchmark import Fixtures
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20273126);parser.add_argument('--output',type=Path,default=Path(__file__).parent);args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True);source=ROOT/'runs/_neural-rail30-accel100-exploration-20261001-f/agent_step_036569088.pt'
torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
net,state=load(source);net.to('cuda').eval();cases=256;n=cases*3
settings=dict(stage=5,game_fraction=0,realistic=True,randomize=True,report_sensing=True,random_practice_opponent=True,continuous_rallies=True,shot_conditioned=True,edge_recovery_weight=80,thermal_gain=1.3,possession_followthrough=True)
env=NeuralTrainingEnv(n,seed=args.seed,**settings,**net.environment_options);low,high=env.decoder.low,env.decoder.high;r=env.cfg.puck_radius+env.cfg.paddle_radius
puck=np.zeros((cases,4));left=rng.random(cases)<.5;corner=rng.random(cases)<.6
inset=rng.uniform(env.cfg.puck_radius+.0002,low[0]+.015,cases)
puck[:,0]=np.where(left,inset,env.cfg.width-inset);puck[:,1]=rng.uniform(low[1]+.15,high[1]-.06,cases)
puck[corner,1]=rng.uniform(env.cfg.puck_radius+.0002,low[1]+.015,corner.sum())
pad=rng.uniform(low+.005,high-.005,(cases,2))
overlap=np.linalg.norm(pad-puck[:,:2],axis=1)<r+.005;pad[overlap]=[.5,.4]
obs=env.reset(seed=args.seed,fixtures=Fixtures(np.zeros(n,int),np.repeat(puck,3,axis=0),np.repeat(pad,3,axis=0),np.full(n,.5)),opponent='idle');env.edge_drill[:]=True;env.base._shot_type[:]=np.tile([1,2,3],cases)
levels=rng.uniform(.05,.4,(n,2,4));profile=(np.arange(n)%4==0);levels[profile,0,2]=rng.uniform(.4,.8,profile.sum());levels[profile,1,2]=rng.uniform(.2,.5,profile.sum())
warm=(np.arange(n)%4==1);levels[warm]=rng.uniform(.75,.95,(warm.sum(),1,1))
for model in env.loads:model.h[:]=levels**2;model.observed[:]=levels
obs=env._features(env.base._make_obs_direct());env._potential=env.potential();history=PhysicalHistory(net.history);history.reset(obs)
bias=RecoveryExplorationBias(n,.8,block_steps=12,device='cuda',load_aware=True,mode='edge',workspace=env.base._ws)
active=np.ones(n,bool);restored=np.zeros(n,bool);fast_after=np.zeros(n,bool);controlled_after=np.zeros(n,bool);clock=np.zeros(n);peak=levels.max(axis=(1,2));conceded=np.zeros(n,bool);length=np.zeros(n,int);restore_length=np.zeros(n,int);frames=[];actions=[];original=env.engine.contact_callback

def contact(event):
 ids=event['indices'];before=env.aimed_count[0,ids].copy();original(event)
 if event['body']!='agent':return
 good=active[ids]&(env.aimed_count[0,ids]>before)&(env.last_shot_route[0,ids]==env.last_shot_request[0,ids])&(np.linalg.norm(event['outgoing_before_speed_cap'],axis=1)>=6)
 fast_after[ids]|=good&restored[ids]
env.engine.contact_callback=contact
start=time.monotonic()
for tick in range(600):
 x=torch.as_tensor(history.for_policy(net),device='cuda');context=torch.as_tensor(env.critic_context(),device='cuda')
 with torch.no_grad():
  features=net.trunk(x);mean=net.actor(features);scale=(net.log_std+net.noise(features)).clamp(-4,.5)
  scale=recovery_exploration_scale(x,scale,.12,load_aware=True,quiet=True,mean=mean,timing=True)
  scale=thermal_effort_exploration_scale(x,scale,.6)
  offset=bias.sample(x,context,torch.zeros(n,dtype=torch.bool,device='cuda'))
  action=(mean+offset+scale.exp()*torch.randn_like(mean)).tanh().cpu().numpy()
 frames.append(obs.copy());actions.append(action.copy());length+=active
 obs,reward,term,trunc,info=env.step(action);history.append(obs)
 new_restore=active&~restored&(env.edge_recovery_count[0]>0)
 restore_length[new_restore]=tick+1
 restored|=new_restore
 clock=np.where(active&restored&(env.control_time[0]>0),clock+.02,0);controlled_after|=clock>=.12-1e-9
 conceded|=active&(info['conceded']>0);peak=np.maximum(peak,np.where(active,env.loads[0].levels.max(axis=(1,2)),0))
 active&=~(term|trunc|(env.engine.puck_y>=1)|(env.loads[0].levels.max(axis=(1,2))>=1))
 if tick%100==0:print(tick,'restored',int(restored.sum()),'completed',int(fast_after.sum()),flush=True)
 if not active.any():break
success=restored&fast_after&~conceded&(peak<.9)
# Also retain safe controlled restorations as partial skill examples, clearly labeled.
partial=restored&~conceded&(peak<.9)&~success
length[partial]=restore_length[partial]
chosen=np.flatnonzero(success|partial)
np.savez_compressed(OUT/'successful-episodes.npz',observation=np.stack(frames)[:,chosen],action=np.stack(actions)[:,chosen],length=length[chosen],physical_case=chosen//3,request=chosen%3+1,completed=success[chosen],restored=restored[chosen],controlled_after=controlled_after[chosen],peak_load=peak[chosen],environment_options=json.dumps(net.environment_options),corner=np.repeat(corner,3)[chosen])
report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=args.seed,physical_cases=cases,request_trials=n,learned_neural_actions_with_training_noise_only=True,scripted_feasibility_actions_used=False,restored=int(restored.sum()),fast_after_restore=int(fast_after.sum()),safe_complete=int(success.sum()),safe_partial_retrieval=int(partial.sum()),corner_successes=int((np.repeat(corner,3)&(success|partial)).sum()),selected_ids=chosen.tolist(),elapsed_s=time.monotonic()-start,holdout='Physical cases0..191 training,192..255 validation; all requests of a physical case stay together. External frozen edge suite uses a different seed.')
(OUT/'collection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
