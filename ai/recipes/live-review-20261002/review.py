"""Offline analysis of session 20261002-114149; run from the repository root.

Inputs and outputs are deliberately pinned for reproducibility. No hardware I/O.
"""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'ai')
from airhockey.dynamics import table_mm_to_sim,_geom as g
from airhockey.physics import TableConfig
out=Path('logs/analysis/neural-live-20261002'); stamp='20261002-114149'
a=np.load(out/(stamp+'.npz'));summary=json.load(open(out/(stamp+'.json')))
rows=[r for r in csv.DictReader(open('logs/run_policy/'+stamp+'.ticks.csv')) if r['neural_obs'] and r['blind']=='0']
keys=['t_cam','raw_accel','cmd_accel','cmd_x','cmd_y','ctl_mallet_x','ctl_mallet_y','puck_age_ms','neural_state_age_ms']
x={k:np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in keys}
obs=np.array([[float(v) for v in r['neural_obs'].split('|')][:42] for r in rows]);actions=np.array([[float(v) for v in r['neural_action'].split('|')] for r in rows]);x['obs']=obs;x['actions']=actions
np.savez_compressed(out/'active-ticks.npz',**x)
p=a['puck'];cam=a['cam'];ctl=a['ctl'];t=p[:,0]; cfg=TableConfig()
def stats(v):
 v=np.array(v);v=v[np.isfinite(v)];return {} if not len(v) else dict(n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),max=float(max(v)))
def interp(tr,ts):return np.column_stack([np.interp(ts,tr[:,0],tr[:,k]) for k in (1,2)])
def fit(ts,xy,at):
 coeff=np.polyfit(ts-at,xy,1);res=np.sqrt(np.mean(np.sum((np.polyval(coeff,ts[:,None]-at)-xy)**2,axis=1)))
 return coeff[0],coeff[1],res
# Raw current input: puck pos [0:2], velocity [2:4]*6, own pos [4:6].
# History is newest first; first 42 entries are current physical features.
puck_speed=np.linalg.norm(obs[:,2:4]*6,axis=1)
contexts={'all':np.ones(len(rows),bool),'opponent_setup':(obs[:,1]>1)&(puck_speed<1.5), 'incoming':(obs[:,3]*6<-1)&(obs[:,1]<1.2),'own_half':obs[:,1]<1,'outgoing_fast':(obs[:,3]*6>2)&(obs[:,1]<1)}
metrics={}
for name,mask in contexts.items():
 metrics[name]=dict(n=int(mask.sum()),raw_accel=stats(x['raw_accel'][mask]/1000),command_accel=stats(x['cmd_accel'][mask]/1000),fraction_actor_above_90=float(np.mean((.05+.95*((actions[mask,5]+1)/2)**2)>=.9)),fraction_raw_above_90=float(np.mean(x['raw_accel'][mask]>=90000)),fraction_command_above_90=float(np.mean(x['cmd_accel'][mask]>=90000)),forward_50mm=float(np.mean(x['ctl_mallet_x'][mask]<1250)),robot_depth_from_end_m=stats((g.RAIL_MAX_X-x['ctl_mallet_x'][mask])/1000))
banks=[];last=-1
for i in range(5,len(t)-5):
 if t[i]-last<.12 or max(np.diff(t[i-5:i+6]))>.009:continue
 side=-1 if p[i,2]<g.RAIL_MIN_Y+cfg.puck_radius*1000+22 else 1 if p[i,2]>g.RAIL_MAX_Y-cfg.puck_radius*1000-22 else 0
 if not side or not (150<p[i,1]<1850):continue
 if (side<0 and p[i,2]!=p[i-2:i+3,2].min()) or (side>0 and p[i,2]!=p[i-2:i+3,2].max()):continue
 before,cb,rb=fit(t[i-5:i-1],p[i-5:i-1,1:],t[i]);after,ca,ra=fit(t[i+2:i+6],p[i+2:i+6,1:],t[i])
 if before[1]*side<500 or after[1]*side>-300 or rb>2 or ra>2:continue
 if abs(before[0])<500 or before[0]*after[0]<=0:continue
 # Exclude nearby robot contact; human samples handled below in separate extraction.
 cp=interp(cam,t[i-5:i+6]);separation=np.linalg.norm(cp-p[i-5:i+6,1:],axis=1).min()
 if separation<140:continue
 en=-after[1]/before[1];et=after[0]/before[0]
 if not (.25<en<1.3 and .25<et<1.3):continue
 q=(x['t_cam']>=t[i]+.003)&(x['t_cam']<t[i]+.035)
 actor_vtrans=obs[q,2]*6 # table y -> sim x
 banks.append(dict(t=float(t[i]),side=int(side),point=p[i,1:].tolist(),incoming=before.tolist(),outgoing=after.tolist(),normal=float(en),tangent=float(et),residual_mm=max(rb,ra),actor_after_vx=actor_vtrans.tolist(),actor_wrong_sign=int((actor_vtrans*after[1]<0).sum()),distance_robot_mm=float(separation)))
 last=t[i]
np.savez_compressed(out/'review-arrays.npz',**{k:a[k] for k in a.files})
goals=[]
for event in summary['goal_watchdog_candidates_not_confirmed_scores']:
 if event['end']!='robot' or event['puck_mm'][0]<1975 or abs(event['puck_mm'][1]-(g.RAIL_MIN_Y+g.RAIL_MAX_Y)/2)>190:continue
 end=event['t'];seg=p[(t>end-2)&(t<=end)]
 # Last centerline crossing into robot half before goal.
 crossing=np.flatnonzero((seg[:-1,1]<g.CENTERLINE_X)&(seg[1:,1]>=g.CENTERLINE_X)&(np.diff(seg[:,0])<.06))
 start=float(seg[crossing[-1]+1,0]) if len(crossing) else end-.35
 q=(x['t_cam']>=start)&(x['t_cam']<=end)
 relevant=[b for b in banks if start-.2<b['t']<end]
 robot=interp(cam,[start])[0]
 goals.append(dict(t=end,crossing_t=start,time_midline_to_goal_ms=1000*(end-start),robot_at_midline_mm=robot.tolist(),distance_ahead_goal_mm=float(g.RAIL_MAX_X-robot[0]),banks=relevant,accel=stats(x['cmd_accel'][q]/1000),raw_accel=stats(x['raw_accel'][q]/1000),puck_age_ms=stats(x['puck_age_ms'][q]),speed_m_s=event['speed_m_s']))
result=dict(acceleration=metrics,rail_fit=dict(n=len(banks),normal=stats([b['normal'] for b in banks]),tangent=stats([b['tangent'] for b in banks]),wrong_sign_ticks=sum(b['actor_wrong_sign'] for b in banks),post_bank_ticks=sum(len(b['actor_after_vx']) for b in banks)),banks=banks,goals=goals)
(out/'review.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='banks'},indent=2))
