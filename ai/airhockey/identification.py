"""Offline experiment design and analysis. Never connects to hardware.

Results describe tested trajectories, not a certified acceleration envelope.
All geometry/recorded motion uses table-frame mm; model motion uses SI units.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from airhockey.motor_patterns import LOW, HIGH, CENTER, BOUNDS
from airhockey.follow_test import travel_time


def predict_pulse(trial, ramp_ms=3):
    from airhockey.motion import CartState, advance
    cart=CartState(1);cart.reset(*trial['start'])
    peak_a=peak_v=0.
    direction=np.array(trial['end'])-trial['start']
    direction=direction/np.linalg.norm(direction)
    along_v=[];along_a=[]
    for _ in range(int(np.ceil(trial['duration']/.001))):
        advance(cart,[trial['end'][0]],[trial['end'][1]],trial['speed']*1000,
                trial['accel']*1000,ramp_ms/1000,.0002,5,bounds=tuple(BOUNDS))
        peak_a=max(peak_a,float(np.hypot(cart.ax[0],cart.ay[0]))/1000)
        peak_v=max(peak_v,float(np.hypot(cart.vx[0],cart.vy[0]))/1000)
        along_v.append(float(np.dot([cart.vx[0],cart.vy[0]],direction))/1000)
        along_a.append(float(np.dot([cart.ax[0],cart.ay[0]],direction))/1000)
    peak_index=int(np.argmax(along_v))
    # Separate the initial launch from braking: a large braking peak must
    # not qualify a launch that never exercised the requested acceleration.
    launch=along_a[:peak_index+1]
    stop_index=next((i for i in range(peak_index+1,len(along_v)) if along_v[i]<=0),len(along_v)-1)
    braking=along_a[peak_index:stop_index+1]
    return dict(predicted_peak_accel_m_s2=peak_a,predicted_peak_speed_m_s=peak_v,
                predicted_launch_accel_m_s2=max(launch),
                predicted_braking_accel_m_s2=max(0.,-min(braking)),
                predicted_launch_ms_above_95pct=sum(a>=.95*trial['accel'] for a in launch))


def design(stage='baseline', grid=1, accelerations=(2, 5, 10, 20, 30, 40, 60),
           speed=1.5, repeats=3, stroke=120., hold=8., margin=40.):
    values = [speed, stroke, hold, margin, *accelerations]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('all parameters must be finite and positive')
    if grid not in (1, 3, 5) or not 1 <= repeats <= 20 or speed > 3:
        raise ValueError('grid must be 1/3/5, repeats 1..20, speed <= 3 m/s')
    if stage not in ('baseline', 'sweep', 'endurance', 'workspace'):
        raise ValueError('unknown stage')
    if not accelerations or max(accelerations) > 120 or any(a >= b for a,b in zip(accelerations, accelerations[1:])):
        raise ValueError('acceleration ladder must increase strictly, up to 120 m/s²')
    inset = margin + (stroke/2 if stage in ('sweep','endurance') else 0)
    if np.any(HIGH-LOW <= 2*inset):
        raise ValueError('stroke and margins do not fit the active workspace')
    points = [CENTER] if grid == 1 else [np.array([x,y]) for x in
        np.linspace(LOW[0]+inset,HIGH[0]-inset,grid) for y in
        np.linspace(LOW[1]+inset,HIGH[1]-inset,grid)]
    points.sort(key=lambda p: float(np.linalg.norm(p-CENTER)))
    trials = []
    def add(kind, p, q, a, v, duration, **tags):
        trials.append(dict(id=len(trials), kind=kind, start=np.asarray(p).tolist(),
                           end=np.asarray(q).tolist(), accel=a, speed=v,
                           duration=duration, **tags))
    # A slow survey is compulsory before every acceleration experiment.
    for i,p in enumerate(points):
        add('hold',p,p,.4,.2,hold,site=i)
    if stage in ('baseline','workspace'):
        return trials
    ladder = accelerations if stage == 'sweep' else accelerations[-1:]
    for a in ladder:
        for i,p in enumerate(points):
            for angle in (0,45,90,135):
                u=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
                lo,hi=p-stroke/2*u,p+stroke/2*u
                for rep in range(repeats):
                    for start,end in ((lo,hi),(hi,lo)):
                        add('pulse', start,end,a,speed,
                            travel_time(stroke,speed*1000,a*1000)+.6,
                            site=i,direction_deg=float(np.rad2deg(np.arctan2(*(end-start)[::-1]))),repeat=rep)
    return trials


def summarize_load(rows):
    """Per-drive, independently timestamped readings; never count cache duplicates."""
    result=[]
    for node in range(4):
        channels={key:{} for key in ('torque_amps','rms_pct','rms_slow_pct','bus_voltage_v')}
        for row in rows:
            for motor in row.get('load',{}).get('sample',{}).get('motors',[]):
                if motor.get('node')!=node:continue
                for key,readings in channels.items():
                    f=motor.get(key,{})
                    if f.get('valid') and np.isfinite([f.get('end',np.nan),f.get('value',np.nan)]).all():
                        if rows[0].get('t',-np.inf)<=f['end']<=rows[-1].get('t',np.inf):
                            readings[f['end']]=f['value']
        report={'node':node}
        for key,readings in channels.items():
            if not readings:continue
            times=sorted(readings);values=np.array([readings[t] for t in times])
            summary=dict(samples=len(values),first=float(values[0]),last=float(values[-1]),
                         min=float(values.min()),max=float(values.max()))
            if key=='torque_amps':
                summary.update(mean_abs=float(np.abs(values).mean()),peak_abs=float(np.abs(values).max()))
            if key.startswith('rms') and times[-1]>times[0]:
                summary['rise_percentage_points_per_s']=float((values[-1]-values[0])/(times[-1]-times[0]))
            report[key]=summary
        result.append(report)
    return result


def score(rows, trial, latency=.015):
    """Use fixed camera latency, never fit away motor lag independently per move."""
    ctl={r['ctl_t']:r for r in rows if 'ctl_t' in r}
    cams={r['cam_t']:r for r in rows if 'cam_t' in r and r.get('camera_valid',True)}
    rejected={r.get('cam_t') for r in rows if not r.get('camera_valid',True)}
    if len(ctl)<5 or len(cams)<8:
        return dict(passed=False, reason='insufficient independent motion/camera samples')
    ct=np.array(sorted(ctl)); cp=np.array([ctl[t]['ctl'][:2] for t in ct])
    vt=np.array(sorted(cams)); vp=np.array([cams[t]['cam'][:2] for t in vt])
    t=vt-latency; keep=(t>=ct[0])&(t<=ct[-1]);t=t[keep];p=vp[keep]
    if len(t)<8:return dict(passed=False,reason='insufficient overlapping camera samples')
    ref=np.column_stack([np.interp(t,ct,cp[:,k]) for k in (0,1)])
    gap=np.linalg.norm(p-ref,axis=1)
    raw=np.column_stack([np.interp(vt[keep],ct,cp[:,k]) for k in (0,1)])
    target=np.asarray(trial['end'])
    rest=t>=t[-1]-.25
    residual=float(np.median(np.linalg.norm(p[rest]-target,axis=1)))
    # Local quadratic fits suppress pixel noise. Report their bandwidth;
    # this is a smoothed measurement, not the unobserved instantaneous peak.
    acc=[]; windows=[]
    for i in range(3,len(t)-3):
        sl=slice(i-3,i+4); dt=t[sl]-t[i]
        if np.max(np.diff(t[sl]))>.04:continue
        coeff=np.polynomial.polynomial.polyfit(dt,p[sl]/1000,2)
        acc.append(float(np.linalg.norm(2*coeff[2])));windows.append(float(dt[-1]-dt[0]))
    camera_gap=float(np.max(np.diff(t)))
    return dict(passed=bool(np.percentile(gap,95)<=20 and residual<=8 and camera_gap<=.05),
        tracking_p95_mm=float(np.percentile(gap,95)), tracking_max_mm=float(gap.max()),
        unshifted_tracking_p95_mm=float(np.percentile(np.linalg.norm(p-raw,axis=1),95)),
        end_error_mm=residual,camera_samples=len(t),max_camera_gap_s=camera_gap,
        camera_rejected_frames=len(rejected),tracking_had_gaps=bool(rejected),
        fixed_camera_latency_s=latency,
        smoothed_measured_accel_peak_m_s2=max(acc,default=None),
        accel_fit_window_s=float(np.median(windows)) if windows else None,
        requested_accel_m_s2=trial['accel'],
        motor_load=summarize_load(rows),
        note=('Passing describes observed tracking only; rejected camera frames are excluded. '
              'Short acceleration peaks and motion during camera gaps may be unresolved.'))


def spatial_features(state, low=LOW, high=HIGH):
    """Nine spatial weights × holding, speed², four signed acceleration energies.

    Unlike the old model, each corner and acceleration direction can load
    a different motor. Fixed pretension is part of the experiment metadata.
    """
    state=np.atleast_2d(state); xy=state[:,:2]
    z=(xy-low)/(high-low)
    centers=np.array([(x,y) for x in (0,.5,1) for y in (0,.5,1)])
    w=np.exp(-np.sum((z[:,None]-centers)**2,axis=2)/(.4**2))
    w/=w.sum(axis=1,keepdims=True)
    v=state[:,2:4]/1000; a=state[:,4:6]/1000/60
    motion=np.column_stack([np.ones(len(state)),np.sum(v*v,axis=1)/4,
                            np.maximum(a,0)**2,np.maximum(-a,0)**2])
    return (w[:,:,None]*motion[:,None,:]).reshape(len(state),-1)


def analyze(directory):
    """Fit a candidate only with enough observations and held-out whole trials.

    Export stays separate from production defaults. Sampled current cannot
    reconstruct peaks shorter than the drive polling interval.
    """
    from scipy.optimize import nnls
    directory=Path(directory)
    meta=json.loads((directory/'plan.json').read_text())
    rows=[]
    for line in (directory/'samples.jsonl').read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    ctl={r['ctl_t']:r for r in rows if 'ctl_t' in r}
    times=np.array(sorted(ctl))
    report={'status':'insufficient_data','samples':len(rows),'per_motor':[],
            'warning':'Sampled-current candidate only; validate RMS trajectories before deployment.'}
    if len(times)<20:
        (directory/'fit-report.json').write_text(json.dumps(report,indent=2));return report
    motion=np.array([ctl[t]['ctl'] for t in times])
    accel=np.gradient(motion[:,2:4],times,axis=0)
    states=np.column_stack([motion,accel])
    coefficients=[]
    for node in range(4):
        data={}; motor_metadata=None
        for row in rows:
            if 'load' not in row:continue
            motors=row['load'].get('sample',{}).get('motors',[])
            m=next((m for m in motors if m['node']==node),{})
            if m: motor_metadata=m
            current=m.get('torque_amps',{});stamp=current.get('end',-1)
            if not current.get('valid') or not times[0]<=stamp<=times[-1]:continue
            ix=np.searchsorted(times,stamp); left=max(0,ix-1); right=min(len(times)-1,ix)
            if times[right]-times[left]>.05:continue
            state=np.array([np.interp(stamp,times,states[:,j]) for j in range(6)])
            data[stamp]=(state,float(current['value'])**2,int(row['trial']))
        items=list(data.values())
        if len(items)<200:
            report['per_motor'].append(dict(node=node,samples=len(items),status='insufficient_data'));continue
        x=spatial_features(np.array([v[0] for v in items])); y=np.array([v[1] for v in items]);groups=np.array([v[2] for v in items])
        unique=np.unique(groups)
        if len(unique)<10: report['per_motor'].append(dict(node=node,status='insufficient_trial_groups'));continue
        test=np.isin(groups,unique[::5]); train=~test
        c,_=nnls(np.vstack([x[train],np.eye(x.shape[1])*.1]),np.r_[y[train],np.zeros(x.shape[1])],maxiter=10000)
        predicted=np.sqrt(np.maximum(x[test]@c,0)); actual=np.sqrt(y[test])
        motor_report=dict(node=node,samples=len(items),held_out_trials=unique[::5].tolist(),
            current_mae_amps=float(np.mean(abs(predicted-actual))),
            current_underprediction_p95_amps=float(np.percentile(actual-predicted,95)),
            feature_rank=int(np.linalg.matrix_rank(x[train])))
        # Compare fast/slow load memory against actual readings, initialized
        # at observed heat, without resetting at trial/goal boundaries.
        # This is in-session validation, not an independent endurance test.
        for channel,limit_key,tau_key,factor in (
            ('rms_pct','rms_limit_amps','rms_time_constant_s',1),
            ('rms_slow_pct','rms_slow_limit_amps','rms_slow_time_constant_min',60)):
            limit=(motor_metadata or {}).get(limit_key,{});tau=(motor_metadata or {}).get(tau_key,{})
            if not limit.get('valid') or not tau.get('valid') or min(limit['value'],tau['value'])<=0:continue
            observed={}
            for row in rows:
                for m in row.get('load',{}).get('sample',{}).get('motors',[]):
                    field=m.get(channel,{})
                    if m['node']==node and field.get('valid') and times[0]<=field['end']<=times[-1]:
                        observed[field['end']]=field['value']
            stamps=np.array(sorted(observed))
            if len(stamps)<10:continue
            h=(observed[stamps[0]]/100)**2; errors=[]
            for t0,t1 in zip(stamps,stamps[1:]):
                # Integrate on the controller timeline to retain short pulses
                # that a 10 Hz RMS sample alone cannot represent.
                knots=np.r_[t0,times[(times>t0)&(times<t1)],t1]
                for left,right in zip(knots,knots[1:]):
                    state=np.array([np.interp((left+right)/2,times,states[:,j]) for j in range(6)])
                    i2=float((spatial_features(state)@c)[0]); decay=np.exp(-(right-left)/(tau['value']*factor))
                    h=decay*h+(1-decay)*i2/limit['value']**2
                errors.append(100*np.sqrt(max(h,0))-observed[t1])
            motor_report[channel+'_mae_percentage_points']=float(np.mean(np.abs(errors)))
            motor_report[channel+'_max_underprediction_points']=float(max(0,-min(errors)))
        report['per_motor'].append(motor_report)
        coefficients.append(c.tolist())
    if len(coefficients)==4:
        report['status']='candidate_not_validated'
        model=dict(schema='spatial-current-v1',status=report['status'],bounds_mm=BOUNDS.tolist(),
            coefficients_amps_squared=coefficients,training_experiment=meta,
            note='No automatic deployment. RMS recurrence/limits must use recorded per-drive parameters.')
        (directory/'current-model-candidate.json').write_text(json.dumps(model,indent=2))
    (directory/'fit-report.json').write_text(json.dumps(report,indent=2))
    return report


def preview(directory, plan):
    """Standalone clickable preview; no hardware or camera access."""
    data=json.dumps(plan).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>Hardware characterization</title>
<style>body{font:16px system-ui;max-width:950px;margin:30px auto;background:#151922;color:#ddd}canvas{background:#242d38}button,input{margin:8px}pre{white-space:pre-wrap}</style>
<h1>Hardware characterization — offline plan</h1><p>Selected moves stay inside current firmware bounds. This preview commands nothing.</p>
<canvas id="c" width="700" height="500"></canvas><br><button id="play">Play/pause</button><input id="slider" type="range" min="0" value="0"><pre id="info"></pre>
<script>const plan=DATA,ts=plan.trials,b=plan.bounds_mm,c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('slider');s.max=ts.length-1;let playing=false;
function p(v){return [30+(v[0]-b[0])/(b[1]-b[0])*640,470-(v[1]-b[2])/(b[3]-b[2])*440]}
function draw(){ctx.clearRect(0,0,700,500);ctx.strokeStyle='#8090a0';ctx.strokeRect(30,30,640,440);for(const t of ts){let a=p(t.start),z=p(t.end);ctx.strokeStyle='#435265';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke()}let t=ts[+s.value],a=p(t.start),z=p(t.end);ctx.strokeStyle='#ffb95c';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...z);ctx.stroke();ctx.lineWidth=1;ctx.fillStyle='#62c9ff';ctx.beginPath();ctx.arc(...z,7,0,7);ctx.fill();document.getElementById('info').textContent=JSON.stringify(t,null,2)}
s.oninput=draw;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){s.value=(+s.value+1)%ts.length;draw()}},500);draw();</script>'''.replace('DATA',data)
    (Path(directory)/'preview.html').write_text(html)
