#!/usr/bin/env python3
"""Audit recorded neural sessions offline. Never imports a hardware client."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from airhockey.dynamics import _geom as geom


def stats(values):
    a = np.asarray(values, float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), max=float(a.max()))


def records(path):
    with Path(path).open() as f:
        for line in f:
            if line.endswith('\n'):
                yield json.loads(line)


def json_ready(value):
    """Represent unavailable measurements as JSON null, never nonstandard NaN."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def audit(stamp, out):
    base = Path('logs/run_policy') / stamp
    with base.with_suffix('.replay.jsonl').open() as f:
        metadata = json.loads(next(f))
    accel_limit = metadata.get('accel_mm_s2') or 60000.0
    speed_limit = metadata.get('speed_mm_s') or 12000.0
    bounds = metadata.get('workspace_bounds_mm')
    if bounds is None:
        from airhockey.neural_setup import checkpoint_environment, workspace_bounds
        checkpoint = metadata.get('checkpoint')
        bounds = (checkpoint_environment(checkpoint)['workspace_bounds_mm']
                  if checkpoint and Path(checkpoint).is_file() else workspace_bounds('legacy'))
    xmin, xmax, ymin, ymax = bounds
    with base.with_suffix('.ticks.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'session': stamp, 'ticks': 0}
    def col(k):
        return np.array([float(r[k]) if r[k] else np.nan for r in rows])
    t = col('t_cam'); active = col('blind') == 0
    dt = np.diff(t, append=t[-1]+.02)
    flags = Counter(r['flags'] for r in rows)
    result = dict(session=stamp, ticks=len(rows), duration_s=float(t[-1]-t[0]),
                  active_s=float(dt[active].sum()), active_ticks=int(active.sum()), flags=dict(flags),
                  effective_hz=float((len(t)-1)/(t[-1]-t[0])), tick_interval_ms=stats(np.diff(t)*1000),
                  active_interval_ms=stats(np.diff(t)[active[:-1]&active[1:]]*1000),
                  active_source=dict(Counter(r['mallet_src'] for r,a in zip(rows,active) if a)))
    for key in ('lag_ms','ctl_age_ms','puck_age_ms','policy_ms','io_ms','cmd_accel','cmd_speed'):
        result[key] = stats(col(key)[active])
    result['caps_violations'] = dict(accel_above_session_cap=int((col('cmd_accel')>accel_limit+.1).sum()),
                                    speed_above_session_cap=int((col('cmd_speed')>speed_limit+.1).sum()),
                                    workspace=int(((col('cmd_x')<xmin-.1)|(col('cmd_x')>xmax+.1)|
                                                   (col('cmd_y')<ymin-.1)|(col('cmd_y')>ymax+.1)).sum()))
    result['session_caps'] = dict(accel_mm_s2=accel_limit, speed_mm_s=speed_limit, workspace_bounds_mm=bounds)
    result['active_accel_cap_fractions'] = {str(f):float(np.mean(col('cmd_accel')[active]>=f*accel_limit)) for f in (.5,.75,.9,.95,.99)}
    result['active_above_40_accel_fraction'] = float(np.mean(col('cmd_accel')[active]>40000))
    result['active_puck_age_over_50ms_fraction'] = float(np.mean(col('puck_age_ms')[active]>50))
    neural_rows = [r for r in rows if r.get('neural_obs')]
    obs = np.array([[float(v) for v in r['neural_obs'].split('|')] for r in neural_rows])
    actions = np.array([[float(v) for v in r['neural_action'].split('|')] for r in neural_rows])
    fresh = np.array([[int(v) for v in r['neural_load_fresh'].split('|')] for r in neural_rows])
    result['neural'] = dict(nonfinite_obs=int((~np.isfinite(obs)).sum()), nonfinite_actions=int((~np.isfinite(actions)).sum()),
                          load_fresh_fraction=fresh.mean(axis=0).tolist(), requests=obs[:,-3:].sum(axis=0).tolist(),
                          max_observed_load=obs[:,21:29].max(axis=0).tolist())
    clock=[]; frames=[]; ctl=[]; watchdog=[]; meta=None; loadpath=None; command_lat=[]; rejects=[]
    for e in records(base.with_suffix('.replay.jsonl')):
        typ=e['type']
        if typ=='meta': meta=e
        elif typ=='clock': clock.append(e['monotonic']-e['t'])
        elif typ=='frame': frames.append(e)
        elif typ=='controller' and e.get('sample_monotonic') is not None:
            ctl.append([e['sample_monotonic'],e['x'],e['y'],e['vx'],e['vy']])
        elif typ=='motor_load_source': loadpath=Path(e['source']['path'])
        elif typ=='puck_watchdog': watchdog.append(e)
        elif typ=='tracking_rejection': rejects.append(e['t'])
        elif typ=='command': command_lat.append(1000*(e['ack_monotonic']-e['sent_monotonic']))
    origin=min(clock)-meta['camera_delay_s']
    result['checkpoint']=meta['checkpoint']; result['command_roundtrip_ms']=stats(command_lat)
    result['watchdog']=watchdog;result['tracking_rejections']=len(rejects)
    result['frame_detection_fraction']={k:float(np.mean([e[k] is not None for e in frames])) for k in ('puck','agent','human')}
    cam=np.array([[e['t'],*e['agent']] for e in frames if e['agent']])
    puck=np.array([[e['t'],*e['puck']] for e in frames if e['puck']])
    ctl=np.array(ctl);ctl[:,0]-=origin
    _, ix=np.unique(ctl[:,0],return_index=True);ctl=ctl[ix]
    def sample(times, array, cols=(1,2)):
        return np.column_stack([np.interp(times,array[:,0],array[:,i]) for i in cols])
    def aligned(lag):
        times=cam[:,0]-lag
        ix=np.searchsorted(ctl[:,0],times).clip(1,len(ctl)-1)
        valid=(times>=ctl[0,0])&(times<=ctl[-1,0])&((ctl[ix,0]-ctl[ix-1,0])<.05)
        pred=sample(times,ctl); speed=np.linalg.norm(sample(times,ctl,(3,4)),axis=1)
        err=np.linalg.norm(cam[:,1:]-pred,axis=1)
        return valid,pred,speed,err
    valid,pred,speed,err=aligned(0)
    result['camera_controller_nominal_mm']=stats(err[valid])
    result['camera_controller_moving_mm']=stats(err[valid&(speed>500)])
    result['camera_controller_stationary_mm']=stats(err[valid&(speed<20)])
    result['camera_controller_stationary_bias_mm']=np.median((cam[:,1:]-pred)[valid&(speed<20)],axis=0).tolist()
    scans=[]
    # Split actual play, not a session tail that may contain long idle time.
    first=cam[:,0] < np.median(t[active])
    for lag in np.arange(-.04,.081,.002):
        ok,pr,sp,er=aligned(lag)
        mask=ok&(sp>500)&first
        scans.append((float(np.median(er[mask])) if mask.any() else np.inf,float(lag)))
    lag=min(scans)[1]
    ok,pr,sp,er=aligned(lag)
    result['lag_fitted_on_first_half_ms']=lag*1000
    result['moving_holdout_before_mm']=stats(err[valid&(speed>500)&~first])
    result['moving_holdout_after_mm']=stats(er[ok&(sp>500)&~first])
    result['controller_speed_m_s']=stats(np.hypot(ctl[:,3],ctl[:,4])/1000)
    # Adjacent distinct controller samples, excluding telemetry gaps. This is
    # a differentiated signal, not firmware's internal peak-accel measurement.
    gaps=np.diff(ctl[:,0]); av=np.linalg.norm(np.diff(ctl[:,3:5],axis=0),axis=1)/gaps/1000
    result['controller_fd_accel_m_s2']=stats(av[(gaps>.005)&(gaps<.04)])
    nearby=[]
    for e in watchdog:
        if not e['paused'] or e['reason']!='goal': continue
        seg=puck[(puck[:,0]>e['t']-.25)&(puck[:,0]<=e['t'])]
        if len(seg)<4: continue
        xy=seg[-1,1:]; edge='robot' if xy[0]>geom.CENTERLINE_X else 'human'
        segment=seg[(seg[:,0]>=seg[-1,0]-.03)]
        vel=np.polyfit(segment[:,0]-segment[-1,0],segment[:,1:],1)[0] if len(segment)>2 else [np.nan]*2
        nearby.append(dict(t=e['t'],end=edge,puck_mm=xy.tolist(),speed_m_s=float(np.linalg.norm(vel)/1000)))
    result['goal_watchdog_candidates_not_confirmed_scores']=nearby
    rms=[]; loadtimes=[]; alarms=[]; motor_age=[]
    for e in records(loadpath):
        if e['type']!='motor_load': continue
        levels=[]
        for field in ('rms_pct','rms_slow_pct'):
            for m in e['motors']:
                f=m[field]
                levels.append(f['value'] if f['valid'] else np.nan)
                if f['valid']: motor_age.append(e['monotonic']-f['end'])
        rms.append(levels);loadtimes.append(e['monotonic']-origin)
        if e['context'].get('fault'):
            alarms.append(e['monotonic']-origin)
    rms=np.array(rms);loadtimes=np.array(loadtimes)
    result['motor_load']=dict(path=str(loadpath),peak_pct=np.nanmax(rms,axis=0).tolist(),
                              samples_at_or_above_100=int(np.any(rms>=100,axis=1).sum()),
                              valid_fraction=np.isfinite(rms).mean(axis=0).tolist(),fault_samples=len(alarms),
                              sample_interval_s=stats(np.diff(loadtimes)),field_age_s=stats(motor_age))
    np.savez_compressed(out/(stamp+'.npz'),cam=cam,puck=puck,ctl=ctl,load_t=loadtimes,rms=rms,
                        tick_t=t,active=active,cmd_accel=col('cmd_accel'),lag_ms=col('lag_ms'),
                        cam_error=err,cam_error_valid=valid)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sessions',nargs='+');ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for session in args.sessions:
        result=json_ready(audit(session,args.output_dir)); results.append(result)
        (args.output_dir/(session+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,allow_nan=False),flush=True)
    (args.output_dir/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
