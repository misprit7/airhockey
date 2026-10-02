"""Offline analysis of session 20261002-114149; run from the repository root.

Inputs and outputs are deliberately pinned for reproducibility. No hardware I/O.
"""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'ai')
from airhockey.motion import CartState,advance
from airhockey.thermal import MotorThermal
from airhockey.dynamics import table_mm_to_sim
out=Path('logs/analysis/neural-live-20261002');a=np.load(out/'20261002-114149.npz');active=np.load(out/'active-ticks.npz');ctl=a['ctl'];cam=a['cam'];puck=a['puck']
def sample(tr,t,columns=(1,2)):
 return np.column_stack([np.interp(t,tr[:,0],tr[:,i]) for i in columns])
def stats(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return dict(n=len(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),mean=float(np.mean(v)))
commands=[];clocks=[];human=[]
for line in open('logs/run_policy/20261002-114149.replay.jsonl'):
 try:r=json.loads(line)
 except json.JSONDecodeError:break
 if r['type']=='clock':clocks.append(r['monotonic']-r['t'])
 if r['type']=='command':commands.append([r['monotonic'],r['x'],r['y'],r['speed'],r['accel']])
 if r['type']=='frame' and r['human'] is not None and r['t']<850:human.append([r['t'],*r['human']])
origin=min(clocks)-.0077;commands=np.array(commands);commands[:,0]-=origin;np.savez_compressed(out/'commands.npz',commands=commands,human=human)
starts=active['t_cam'][::400];n=len(starts);bounds=(1200,1937.5,61.4,904.5)
results=[]
for offset in (0,.002,.004,.008):
 cart=CartState(n);cart.x[:],cart.y[:]=sample(ctl,starts).T;cart.vx[:],cart.vy[:]=sample(ctl,starts,(3,4)).T
 errors=[];camera_errors=[]
 for j in range(500):
  ts=starts+j*.002;ix=np.searchsorted(commands[:,0]+offset,ts,side='right')-1;cmd=commands[ix]
  advance(cart,cmd[:,1],cmd[:,2],cmd[:,3],cmd[:,4],.003,.0002,10,bounds=bounds)
  if j>50:
   position=np.column_stack((cart.x,cart.y));errors.append(np.linalg.norm(position-sample(ctl,ts+.002),axis=1));camera_errors.append(np.linalg.norm(position-sample(cam,ts+.002),axis=1))
 errors=np.array(errors);camera_errors=np.array(camera_errors)
 results.append(dict(command_offset_ms=offset*1000,controller=stats(errors),camera=stats(camera_errors),train=stats(errors[:,::2]),holdout=stats(errors[:,1::2])))
# Same targets, with only the acceleration cap changed: open-loop sensitivity,
# not a closed-loop counterfactual save claim.
goals=json.load(open(out/'review.json'))['goals'];counter=[]
for goal in goals:
 start=goal['crossing_t'];end=puck[puck[:,0]<=goal['t'],0][-1];cart=CartState(2);xy=sample(ctl,[start])[0];v=sample(ctl,[start],(3,4))[0];cart.x[:],cart.y[:]=xy;cart.vx[:],cart.vy[:]=v
 mindist=np.full(2,np.inf);maxspeed=np.zeros(2)
 for j in range(int((end-start)/.002)):
  ts=start+j*.002;ix=np.searchsorted(commands[:,0],ts,side='right')-1;cmd=commands[ix]
  advance(cart,np.repeat(cmd[1],2),np.repeat(cmd[2],2),12000,np.array([cmd[4],100000]),.003,.0002,10,bounds=bounds)
  p=sample(puck,[ts+.002])[0];mindist=np.minimum(mindist,np.hypot(cart.x-p[0],cart.y-p[1]));maxspeed=np.maximum(maxspeed,np.hypot(cart.vx,cart.vy))
 counter.append(dict(t=goal['t'],minimum_center_distance_mm=mindist.tolist(),contact_threshold_mm=91.1,max_speed_m_s=(maxspeed/1000).tolist()))
load=[]
for line in open('logs/motor_load/1790955708784819-2411119.jsonl'):
 try:r=json.loads(line)
 except json.JSONDecodeError:break
 if r['type']!='motor_load' or not r['context']['motors_enabled']:continue
 t=r['monotonic']-origin
 if t>850:break
 c=r['context']['controller']
 if not c['valid']:continue
 values=[m['torque_amps']['value'] if m['torque_amps']['valid'] else np.nan for m in r['motors']]
 load.append([t,c['x_mm'],c['y_mm'],c['vx_mm_s'],c['vy_mm_s'],*values])
load=np.array(load);stationary=np.linalg.norm(load[:,3:5],axis=1)<20
# Require stationary telemetry neighbors too, avoiding short stop/start aliases.
stationary[1:] &= np.linalg.norm(load[:-1,3:5],axis=1)<20
stationary[:-1] &= np.linalg.norm(load[1:,3:5],axis=1)<20
m=MotorThermal(len(load),path='ai/recipes/motor-load-20261001.json',randomize=False)
x,y=table_mm_to_sim(load[:,1],load[:,2]);zero=np.zeros((len(load),2));m.advance(np.column_stack((x,y)),zero,zero,.02)
actual=np.abs(load[:,5:]);pred=np.sqrt(m.current_squared);comparisons=[]
for k in range(4):
 q=stationary&np.isfinite(actual[:,k]);comparisons.append(dict(motor=k,n=int(q.sum()),actual_amps=stats(actual[q,k]),predicted_amps=stats(pred[q,k]),rmse=float(np.sqrt(np.mean((actual[q,k]-pred[q,k])**2)))))
result=dict(motion_rollouts=results,same_target_full_accel=counter,stationary_load=comparisons)
(out/'motion-load.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
