"""Offline analysis of session 20261002-114149; run from the repository root.

Inputs and outputs are deliberately pinned for reproducibility. No hardware I/O.
"""
import sys,json
from pathlib import Path
import numpy as np, torch
sys.path[:0]=['ai','ai/bin']
import eval_neural_player as ev
from airhockey.physics import TableConfig
path=Path('runs/rail30-100-20261001-v1/agent.pt');net,state=ev.load(path);net.to('cuda');torch.set_num_threads(1)
old=net.act;caps=[];contexts=[]
def act(obs,**kw):
 a=old(obs,**kw);caps.append((.05+.95*((a[:,5]+1)/2)**2)*100);contexts.append(np.column_stack((obs[:,0:4],obs[:,4:6])))
 return a
net.act=act
result=ev.games(net,seconds=60,n=8,seed=20261002,checkpoint=path,initial_load=.4,report_sensing=ev.checkpoint_report_sensing(path,state),continuous_rallies=ev.checkpoint_continuous_rallies(path,state))
c=np.concatenate(caps);o=np.concatenate(contexts);speed=np.linalg.norm(o[:,2:4]*6,axis=1);incoming=(o[:,3]*6<-1)&(o[:,1]<1.2);setup=(o[:,1]>1)&(speed<1.5)
out={}
for label,mask in [('all',np.ones(len(c),bool)),('incoming',incoming),('opponent_setup',setup)]:
 out[label]=dict(n=int(mask.sum()),mean=float(c[mask].mean()),median=float(np.median(c[mask])),p95=float(np.percentile(c[mask],95)),max=float(c[mask].max()),above_90_fraction=float(np.mean(c[mask]>=90)),depth_median=float(np.median(o[mask,5])*1.0146))
Path('logs/analysis/neural-live-20261002/sim-acceleration.json').write_text(json.dumps(dict(accel=out,match=result),indent=2));print(json.dumps(out))
