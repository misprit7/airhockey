"""Offline post-swap fit; preserves raw logs and never opens hardware.

The final 100/120 m/s² session is held out. Acceleration coefficients remain
explicit priors when sparse current sampling cannot identify short peaks.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls, minimize_scalar
from scipy.signal import lfilter
root=Path(__file__).resolve().parents[2]
files=[p for p in sorted((root/'logs/characterization').glob('20260930-*/samples.jsonl')) if p.stat().st_size]
sessions=[]
for p in files:
 ctl={};mot=[{} for _ in range(4)];rms=[[{},{}] for _ in range(4)];meta=None
 for line in p.open():
  r=json.loads(line);ctl[r['ctl_t']]=r['ctl'];sample=r['load']['sample']
  meta=sample['motors']
  for i,m in enumerate(meta):
   f=m['torque_amps']
   if f['valid']:mot[i][f['end']]=f['value']
   for j,k in enumerate(('rms_pct','rms_slow_pct')):
    f=m[k]
    if f['valid']:rms[i][j][f['end']]=f['value']
 raw_t=np.array(sorted(ctl));raw_state=np.array([ctl[x] for x in raw_t])
 # Host queries repeat a cached 50Hz status with microsecond timestamp
 # reconstruction jitter. Differentiate on a uniform grid, never across
 # those near-duplicate times (which invent enormous accelerations).
 t=np.arange(raw_t[0],raw_t[-1],.01)
 state=np.column_stack([np.interp(t,raw_t,raw_state[:,k]) for k in range(4)])
 a=np.gradient(state[:,2:4],t,axis=0)/1000
 sessions.append(dict(name=p.parent.name,t=t,state=state,a=a,mot=mot,rms=rms,meta=meta))
 print(p.parent.name,len(t),[len(x) for x in mot],flush=True)
out=root/'logs/analysis/motor-model-20261001';out.mkdir(parents=True,exist_ok=True)
# Baseline per-site holding current directly measured; distances in mm.
baseline=root/'logs/characterization/20260930-213839-639063'
res=json.loads((baseline/'results.json').read_text())
centers=np.array([r['trial']['start'] for r in res]);holding=np.array([[m['torque_amps']['mean_abs']**2 for m in r['metrics']['motor_load']] for r in res])
def weights(x):
 # Piecewise bilinear across measured 3x3 grid; outside clamps to nearest
 # surveyed boundary, with a separate explicitly provisional extension cost.
 x=np.atleast_2d(x);weights=np.ones((len(x),len(centers)))
 for axis in (0,1):
  nodes=np.unique(centers[:,axis]);q=np.clip(x[:,axis],nodes[0],nodes[-1])
  weights*=np.maximum(0,1-abs(q[:,None]-centers[:,axis])/np.diff(nodes).mean())
 return weights
co=[];reports=[]
for i in range(4):
 xs=[];ys=[];hs=[];group=[]
 for s in sessions:
  t=s['t'];stamps=np.array(sorted(s['mot'][i]));valid=(stamps>=t[0])&(stamps<=t[-1]);stamps=stamps[valid]
  state=np.column_stack([np.interp(stamps,t,s['state'][:,k]) for k in range(4)])
  acc=np.column_stack([np.interp(stamps,t,s['a'][:,k]) for k in range(2)])/60
  x=np.column_stack(((state[:,2:4]**2).sum(1)/4e6,np.maximum(acc,0)**2,np.maximum(-acc,0)**2))
  xs.extend(x);ys.extend([s['mot'][i][v]**2 for v in stamps]);hs.extend(weights(state[:,:2])@holding[:,i]);group.extend([s['name']]*len(stamps))
 x=np.array(xs);y=np.array(ys);h=np.array(hs);group=np.array(group)
 train=group!='20260930-220711-069679';test=~train
 # Positive regularized dynamic cost, conservative 1 A² prior at 60m/s²
 # per signed axis because sampled current misses millisecond peaks.
 c,_=nnls(np.vstack([x[train],np.eye(5)*2]),np.r_[np.maximum(0,y[train]-h[train]),np.array([1,1,1,1,1])*2])
 c=np.maximum(c,[.1,1,1,1,1]);co.append(c.tolist())
 pred=np.sqrt(h[test]+x[test]@c);actual=np.sqrt(y[test])
 reports.append(dict(node=i,samples=len(y),heldout_session='20260930-220711-069679',current_mae_amps=float(np.mean(abs(pred-actual))),underprediction_p95_amps=float(np.percentile(actual-pred,95)),coefficients=c.tolist()))
# Fit thermal memory from observed currents, compare whole held-out session.
fast=[];slow=[];thermal=[]
for i in range(4):
 for j,(limitkey,taukey,factor) in enumerate((('rms_limit_amps','rms_time_constant_s',1),('rms_slow_limit_amps','rms_slow_time_constant_min',60))):
  limit=sessions[-1]['meta'][i][limitkey]['value'];nominal=sessions[-1]['meta'][i][taukey]['value']*factor
  cohorts=[]
  for s in sessions:
   stamps=np.array(sorted(s['rms'][i][j]));curr=np.array(sorted(s['mot'][i]));lo=max(stamps[0],curr[0]);hi=min(stamps[-1],curr[-1]);grid=np.arange(lo,hi,.1)
   if len(grid)<20:continue
   obs=np.interp(grid,stamps,[s['rms'][i][j][t] for t in stamps])/100
   energy=(np.interp(grid,curr,[s['mot'][i][t] for t in curr])/limit)**2
   cohorts.append((s['name'],energy,obs))
  def errors(tau,heldout=False):
   result=[]
   for name,energy,obs in cohorts:
    if (name=='20260930-220711-069679')!=heldout:continue
    decay=np.exp(-.1/tau)
    h,_=lfilter([1-decay],[1,-decay],energy,zi=[decay*obs[0]**2])
    result.extend(100*(np.sqrt(np.maximum(h,0))-obs))
   return np.array(result)
  fit=minimize_scalar(lambda logtau:np.mean(errors(np.exp(logtau))**2),bounds=(np.log(.2),np.log(5000)),method='bounded')
  tau=float(np.exp(fit.x));(fast if j==0 else slow).append(tau)
  er=errors(tau,True);thermal.append(dict(node=i,channel=j,reported_tau_s=nominal,fitted_effective_tau_s=tau,heldout_mae_pct=float(np.mean(abs(er))),heldout_underprediction_max_pct=float(max(0,-er.min()))))
meta=sessions[-1]['meta']
model=dict(schema='spatial-current-v2',status='provisional_measured_post_swap',hardware_revision='four-2331S-RLNA-20260929',pretension_mm=1.5,holding_centers_mm=centers.tolist(),holding_amps_squared=holding.tolist(),motion_coefficients_amps_squared=co,fast_limit_amps=[m['rms_limit_amps']['value'] for m in meta],slow_limit_amps=[m['rms_slow_limit_amps']['value'] for m in meta],fast_tau_s=fast,slow_tau_s=slow,unmeasured_extension_amps_per_100mm=1.0,notes='Measured post-swap holding map and sampled current fit. Per-drive settings differ despite matching motor parts. Effective thermal time constants fitted to measured RMS trajectories; register time constants are not used directly. Signed acceleration coefficients have a conservative 1 A² floor at 60m/s². Expanded region is unmeasured: nearest surveyed holding current plus explicit extension prior. No physical qualification or overload guarantee.',sources=[str(p.relative_to(root)) for p in files],validation=reports,thermal_validation=thermal)
(root/'ai/recipes/motor-load-20261001.json').write_text(json.dumps(model,indent=2))
(out/'report.json').write_text(json.dumps(model,indent=2))
print(json.dumps(dict(current=reports,thermal=thermal),indent=2))
