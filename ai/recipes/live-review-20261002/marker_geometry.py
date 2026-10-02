"""Offline analysis of session 20261002-114149; run from the repository root.

Inputs and outputs are deliberately pinned for reproducibility. No hardware I/O.
"""
import sys,json,itertools
import numpy as np
from pathlib import Path
sys.path[:0]=['ai','vision/bin']
from puck_stream import PuckTracker
from puck_markers import find_puck
import track_mallet as tm
out=Path('logs/analysis/neural-live-20261002');cam=np.load(out/'20261002-114149.npz')['cam'];tracker=PuckTracker();corrections=[]
for line in open('logs/run_policy/20261002-114149.replay.jsonl'):
 try:e=json.loads(line)
 except json.JSONDecodeError:break
 if e['type']!='near_contact_tracking':continue
 kept,world=tracker.candidates(np.asarray(e['blobs_px']))
 puck=find_puck(world)
 if puck is None:continue
 free=[i for i in range(len(kept)) if i not in puck[2]]
 if len(free)>8:continue
 fits=[]
 for ids in itertools.combinations(free,3):
  c=[(float(kept[i,2]),kept[i,:2]) for i in ids]
  pose=tm.solve_pose(c,tracker.K,tracker.dist,tracker.rvec,tracker.tvec)
  if abs(pose['disagree'])<=5 and max(abs(r-26.5) for r in pose['r'])<4:fits.append(pose)
 if len(fits)!=1:continue
 p=fits[0]['centre'];corrections.append([e['t'],*p,*(p-np.asarray(e['agent']))])
arr=np.array(corrections);np.savez_compressed(out/'marker-correction.npz',poses=arr)
print('valid fits',len(arr),'median correction',np.median(arr[:,3:],axis=0),'p95 magnitude',np.percentile(np.linalg.norm(arr[:,3:],axis=1),95))
