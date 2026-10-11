"""Recheck earlier candidates against production on a new development seed.
Simulation only. Run from repository root; results are not held-out qualification.
"""
import json,sys
from pathlib import Path
sys.path[:0]=['ai','ai/bin']
import numpy as np
import torch
from eval_neural_player import load,games
base=Path('runs/rail30-100-20261001-v1/agent.pt')
root=Path('logs/neural-player/control-defense-20261011/followup')
root.mkdir(parents=True,exist_ok=True)
parent,_=load(base);parent.to('cpu');torch.set_num_threads(2)
for name,path in [('depth200-20m',Path('runs/_neural-control-defense-depth200-20261011/agent.pt')),('wide-20m',Path('runs/_neural-control-defense-capacity1536-20261011/agent.pt'))]:
 net,_=load(path);net.to('cpu')
 for swap in (False,True):
  replay=Path('ai/recordings')/f'control-defense-review-{name}-{int(swap)}.json'
  r=games(net,seconds=90,n=12,seed=20261099,checkpoint=path,opponent_net=parent,opponent_checkpoint=base,swap_sides=swap,record=replay,initial_load=.4,report_sensing=True,continuous_rallies=True)
  own=1 if swap else 0
  result=dict(name=name,checkpoint=str(path),swap=swap,goals_for=int(np.array(r['score'])[own].sum()),goals_against=int(np.array(r['score'])[1-own].sum()),own_overload_seconds=float(np.array(r['overload_seconds'])[own].sum()),own_peak=float(np.array(r['peak_load'])[own].max()),replay='/?replay='+replay.name,details=r)
  (root/f'{name}-{int(swap)}.json').write_text(json.dumps(result,indent=2))
  print({k:v for k,v in result.items() if k!='details'},flush=True)
