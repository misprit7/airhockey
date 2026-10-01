"""Imitate successful neural fringe attempts while retaining the parent actor's other skills."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'ai'),str(ROOT/'ai/bin')]
from eval_neural_player import load
from airhockey.neural_coordinates import observation_coordinates
OUT=Path(__file__).resolve().parent;RUN=ROOT/'runs/_neural-rail30-accel100-hard-corners-20261001-j'
source=ROOT/'runs/_neural-rail30-accel100-defense-20261001-b/agent_step_080281600.pt'
paths=[OUT/str(seed)/'successful-episodes.npz' for seed in range(20261210,20261216)]
deadline=time.monotonic()+1200
while not all(p.with_name('collection.json').exists() for p in paths):
 if time.monotonic()>deadline:raise TimeoutError('Neural example collection did not finish')
 time.sleep(10)
torch.set_num_threads(2);torch.manual_seed(20261120)
net,state=load(source);state['args']['project_rail_contacts']=True;net.environment_options['project_rail_contacts']=True;net.to('cuda').train();reference,_=load(source);reference.to('cuda').eval().requires_grad_(False)
tx=[];ty=[];tc=[];tcorner=[];vx=[];vy=[];groups=[]
for batch,path in enumerate(paths):
 with np.load(path) as z:
  raw=z['observation'];actions=z['action'];length=z['length'];physical=z['physical_case'];complete=z['completed'];corner=z['corner']
  for i,count in enumerate(length):
   obs=raw[:count,i];indices=np.maximum(np.arange(count)[:,None]-np.arange(net.history),0)
   x=np.column_stack((obs[indices,:42].reshape(count,-1),obs[:,-3:]));y=actions[:count,i]
   training=int(physical[i])<192
   groups.append(dict(batch=batch,physical_case=int(physical[i]),complete=bool(complete[i]),corner=bool(corner[i]),training=training,frames=int(count)))
   if training:tx.append(x);ty.append(y);tc.append(np.full(count,complete[i],bool));tcorner.append(np.full(count,corner[i],bool))
   else:vx.append(x);vy.append(y)
x=torch.as_tensor(np.concatenate(tx),device='cuda');y=torch.as_tensor(np.concatenate(ty),device='cuda');complete=torch.as_tensor(np.concatenate(tc),device='cuda');corner=torch.as_tensor(np.concatenate(tcorner),device='cuda')
valid_x=torch.as_tensor(np.concatenate(vx),device='cuda');valid_y=torch.as_tensor(np.concatenate(vy),device='cuda')
speed=x[:,2:4].square().sum(1).sqrt()*6
lo,hi=np.array([.080087658,.079243051]),np.array([.919912342,.806130495])
fringe=((x[:,0]<lo[0]+.06)|(x[:,0]>hi[0]-.06)|(x[:,1]<lo[1]+.06))&(x[:,1]<hi[1])&(speed<2)
classes=[torch.nonzero(mask,as_tuple=True)[0] for mask in [corner&fringe,~corner&fringe,complete,~complete&fringe]]
assert all(len(ids)>0 for ids in classes),'Need neural successes in all groups'
with np.load(ROOT/'logs/neural-player/requests/possession92/skill-replay.npz') as z:
 anchor=observation_coordinates(z['observation'],dict(accel=60,workspace_bounds_mm=None),net.environment_options,net.history)
 speed=np.linalg.norm(anchor[:,2:4],axis=1)*6
 keep=((anchor[:,0]>lo[0]+.07)&(anchor[:,0]<hi[0]-.07)&(anchor[:,1]>lo[1]+.07))|(speed>2)
 anchor=torch.as_tensor(anchor[keep],device='cuda')
with torch.no_grad():target=torch.cat([reference.actor(reference.trunk(part)).tanh() for part in anchor.split(2048)])
RUN.mkdir(exist_ok=False)
meta=json.loads(source.with_name('run.json').read_text()) if source.with_name('run.json').exists() else json.loads((source.parent/'run.json').read_text())
meta['args']['project_rail_contacts']=True
meta.update(algorithm='successful_neural_trajectory_imitation_v1',controller=None,simulation_only=True,deployment_ready=False,evaluation_dir=str(OUT),initialization=str(source),initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),training_frames=len(x),validation_frames=len(valid_x),groups=groups,data=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],anchor_weight=64,hot_preservation_weight=0,holdout='Whole physical cases 192..255; no scripted actions, only successful noisy neural trajectories; external fringe suite uses separate fixtures.')
(RUN/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
(RUN/'review.json').write_text(json.dumps(dict(summary='Self-imitation of successful neural corner/rail recoveries, with ordinary skills preserved. All six actions remain neural outputs. Independent evaluation pending.'),indent=2))
(RUN/'source').mkdir();(RUN/'source/fit.py').write_bytes(Path(__file__).read_bytes());(RUN/'source/collect.py').write_bytes((OUT/'collect.py').read_bytes())
optimizer=torch.optim.Adam(net.actor_parameters(),lr=1e-5);start=time.monotonic()
for update in range(1,16001):
 ids=torch.cat([pool[torch.randint(len(pool),(512 if j in (0,2) else 256,),device='cuda')] for j,pool in enumerate(classes)])
 ai=torch.randint(len(anchor),(1024,),device='cuda')
 prediction=net.actor(net.trunk(x[ids])).tanh();regular=net.actor(net.trunk(anchor[ai])).tanh()
 skill=(prediction-y[ids]).square().mean();ordinary=(regular-target[ai]).square().mean();loss=skill+64*ordinary
 if not torch.isfinite(loss):raise FloatingPointError('Invalid imitation loss')
 optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.actor_parameters(),1);optimizer.step()
 if update==1 or update%500==0:
  with torch.no_grad():validation=sum(float((net.actor(net.trunk(xx)).tanh()-yy).square().sum()) for xx,yy in zip(valid_x.split(2048),valid_y.split(2048)))/(len(valid_x)*6)
  status=dict(pid=os.getpid(),update=update,target_updates=16000,elapsed_s=time.monotonic()-start,training_loss=float(skill.detach()),preservation_loss=float(ordinary.detach()),validation_loss=validation,running=update<16000)
  tmp=RUN/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(RUN/'status.json');print(json.dumps(status),flush=True)
 if update in [2000,6000,12000,16000]:
  saved=dict(model=net.state_dict(),step=state['step'],width=net.width,history=net.history,shot_conditioned=True,stage=5,args=state['args'],algorithm=meta['algorithm'],supervised_update=update,source_checkpoint=str(source))
  file=RUN/f'agent_update_{update:06d}.pt';tmp=file.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(file)
print('complete',flush=True)
