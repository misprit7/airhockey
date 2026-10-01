"""Simulation-only distillation to preserve physical arrivals after changing units."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from airhockey.neural_player import NeuralPlayer
from airhockey.neural_setup import workspace_bounds
from airhockey.neural_coordinates import action_coordinates,observation_coordinates
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--replay',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=2500)
args=p.parse_args()
root=Path(__file__).resolve().parents[2];out=args.output;out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);torch.manual_seed(20261001)
state=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
teacher=NeuralPlayer(state['width'],history=state['history'],shot_conditioned=True).cuda();teacher.load_weights(state['model']);teacher.eval()
student=NeuralPlayer(state['width'],history=state['history'],shot_conditioned=True).cuda();student.load_weights(state['model'])
old=dict(accel=60,workspace_bounds_mm=None);new=dict(accel=100,workspace_bounds_mm=workspace_bounds('rail30'))
with np.load(args.replay) as data:
 x=data['observation'];print('examples',x.shape,flush=True)
y=[]
for start in range(0,len(x),2048):
 y.append(action_coordinates(teacher.act(x[start:start+2048]),old,new))
y=torch.tensor(np.concatenate(y),device='cuda');x=torch.tensor(observation_coordinates(x,old,new,state['history']),device='cuda')
order=torch.randperm(len(x),device='cuda');test=order[:len(x)//10];train=order[len(x)//10:]
opt=torch.optim.Adam(student.actor_parameters(),lr=3e-5)
for step in range(args.steps+1):
 ids=train[torch.randint(len(train),(2048,),device='cuda')]
 pred=student.actor(student.trunk(x[ids])).tanh()
 loss=(pred-y[ids]).square().mean()
 opt.zero_grad();loss.backward();opt.step()
 if step%250==0:
  with torch.no_grad():
   err=(student.actor(student.trunk(x[test])).tanh()-y[test]).abs()
   print(step,float(loss),err.mean(0).tolist(),flush=True)
state['model']=student.state_dict();state['args']=dict(state['args'],accel=100,workspace='rail30');state['step']=0
state.pop('optimizer',None);state.pop('opponent_pool',None)
torch.save(state,out/'agent.pt')
(out/'run.json').write_text(json.dumps(dict(args=state['args'],workspace_bounds_mm=new['workspace_bounds_mm'],physical_limits=dict(speed_m_s=12,acceleration_m_s2=100),thermal_model=json.loads((root/'ai/recipes/motor-load-20261001.json').read_text()),initialization=str(args.checkpoint),migration='Supervised unit conversion of the same learned policy; physical arrival targets preserved on recorded observations.'),indent=2))
(out/'migration.json').write_text(json.dumps(dict(heldout_action_mae=err.mean(0).tolist(),examples=len(x)),indent=2))
