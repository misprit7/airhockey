#!/usr/bin/env python3
"""Offline by default. --live + typed RUN starts a finite physical experiment."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ai'))
from airhockey.identification import design, score, preview, analyze, predict_pulse
from airhockey.motor_patterns import LOW,HIGH,BOUNDS,check_load
from airhockey.follow_test import travel_time
from motor_demo import start_master,stop_master,shutdown,wait_for_load


def validate_load(snapshot, now, args, enabled=True):
    peak=check_load(snapshot,now,args.rms_stop)
    sample=snapshot.get('sample') or {}
    if not snapshot.get('logging_ok'):
        raise RuntimeError('motor-load recording is not healthy')
    if enabled and not sample.get('context',{}).get('motors_enabled'):
        raise RuntimeError('master reports motors disabled')
    for m in sample.get('motors',[]):
        for key in ('rms_slow_pct','torque_amps','bus_voltage_v','status','alerts','encoder_counts'):
            f=m.get(key,{})
            if not f.get('valid') or not 0 <= now-f.get('end',-1) <= .5:
                raise RuntimeError(f"motor {m['node']}: missing/stale {key}")
            if key=='status' and enabled and f['value']!=[1,0]:
                raise RuntimeError(f"motor {m['node']}: drive status {f['value']}")
            if key=='alerts' and any(f['value']):raise RuntimeError(f"motor {m['node']}: active drive alert")
            if key=='bus_voltage_v' and f['value']<args.min_volts:raise RuntimeError('bus voltage below test cutoff')
            if key=='torque_amps' and abs(f['value'])>=args.current_stop:raise RuntimeError('instantaneous current above test cutoff')
    return peak


class Experiment:
    def __init__(self,client,vision,args,output):
        self.client,self.vision,self.args,self.output=client,vision,args,output
        self.rows=[];self.last_load=0;self.snapshot=None;self.last_poll=None
        self.camera_offset=time.monotonic()-time.time()
        self.trial=-1

    def poll(self):
        now=time.monotonic()
        if self.last_poll and now-self.last_poll>.15:raise RuntimeError('monitor loop stalled >150 ms')
        self.last_poll=now
        if now-self.last_load>=.08 or self.snapshot is None:
            self.snapshot=self.client.get_motor_load();self.last_load=time.monotonic()
        validate_load(self.snapshot,time.monotonic(),self.args)
        ctl=self.client.get_position_sample();now=time.monotonic()
        if not np.isfinite(ctl).all() or not 0<=ctl[4]<=.08:raise RuntimeError('stale controller position')
        with self.vision._lock:
            pose=self.vision._pose; stamp=self.vision._pose_t; note=self.vision._note
        cam_t=stamp+self.camera_offset
        if pose is None or note or not np.isfinite(pose).all() or not 0<=now-cam_t<=.08:
            reason=note or self.vision.error or ('no pose' if pose is None else 'invalid/stale pose')
            raise RuntimeError(f'camera tracking failed: {reason}; pose age {(now-cam_t)*1000:.1f} ms')
        if np.any(np.array(pose[:2])<LOW-5) or np.any(np.array(pose[:2])>HIGH+5):
            raise RuntimeError('camera paddle outside current workspace')
        row=dict(t=now,trial=self.trial,ctl_t=now-ctl[4],ctl=list(ctl[:4]),
                 cam_t=cam_t,cam=list(pose),load=self.snapshot)
        self.rows.append(row)
        self.output.write(json.dumps(row,allow_nan=False)+'\n');self.output.flush()
        # Stop on gross disagreement, allowing a bounded camera pipeline lag.
        target_t=cam_t-self.args.camera_latency_ms/1000
        history=self.rows[-100:]
        if len(history)>2 and history[0]['ctl_t']<=target_t<=history[-1]['ctl_t']:
            tt=[r['ctl_t'] for r in history]
            reference=[np.interp(target_t,tt,[r['ctl'][j] for r in history]) for j in (0,1)]
            if np.linalg.norm(np.array(pose[:2])-reference)>self.args.error_stop:
                raise RuntimeError('camera/controller tracking error exceeded stop threshold')
        return row

    def setup_limits(self,speed,accel):
        self.client.set_limits(speed*1000,accel*1000)
        # LIMITS acknowledges serial commands; STATUS is a separate 50 Hz
        # cache and may still describe the preceding configuration. Only
        # wait at this stationary setup boundary, never during a test move.
        wait_for_load(self.client,self.args.rms_stop,not_before=time.monotonic())
        deadline=time.monotonic()+1.
        while True:
            validate_load(self.client.get_motor_load(),time.monotonic(),self.args)
            status=self.client.get_status()
            actual=(status.get('speed_limit'),status.get('accel_limit'))
            if all(isinstance(v,(int,float)) and math.isfinite(v) for v in actual):
                if abs(actual[0]-speed*1000)<=1 and abs(actual[1]-accel*1000)<=1:
                    break
            if time.monotonic()>=deadline:
                raise RuntimeError(
                    f'firmware limit verification timed out: requested {speed*1000:g} mm/s, '
                    f'{accel*1000:g} mm/s²; reported {actual[0]!r}, {actual[1]!r}')
            time.sleep(.02)
        self.last_poll=None;self.snapshot=None

    def observe(self,duration):
        end=time.monotonic()+duration
        while time.monotonic()<end:
            tick=time.monotonic();self.poll();time.sleep(max(0,.005-(time.monotonic()-tick)))

    def settled(self,target):
        self.observe(.35)
        r=self.rows[-1]
        if np.linalg.norm(np.array(r['ctl'][2:]))>10 or np.linalg.norm(np.array(r['ctl'][:2])-target)>2:
            raise RuntimeError('controller did not settle at target')
        recent={r['cam_t']:r for r in self.rows if r['t']>=self.rows[-1]['t']-.2}
        if len(recent)<3 or max(np.linalg.norm(np.array(r['cam'][:2])-target) for r in recent.values())>8:
            raise RuntimeError('paddle did not physically settle within 8 mm of target')

    def reposition(self,target):
        # Call only from a verified rest. Repositioning is itself monitored.
        if getattr(self.args,'quick',False):
            self.poll()
            current=np.array(self.rows[-1]['ctl'][:2])
            if np.linalg.norm(current-target)<=1 and np.linalg.norm(self.rows[-1]['ctl'][2:])<=1:
                self.settled(target)
                return
        current=np.array(self.rows[-1]['ctl'][:2])
        self.setup_limits(.2,.4)
        self.client.command_position(*target,0)
        self.observe(travel_time(float(np.linalg.norm(current-target)),200,400)+.5)
        self.settled(target)


def live(args,plan,directory):
    from airhockey.hardware import CDPRClient
    from airhockey.vision_service import VisionService
    client=CDPRClient();vision=VisionService();proc=None;attempted=False
    handlers={}
    def interrupted(*_):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
    results=[]; diagnostic=None; outcome={'status':'aborted','reason':'startup incomplete'}
    try:
        with (directory/'master.log').open('w') as master_log,(directory/'samples.jsonl').open('w') as samples:
            proc=start_master(master_log,args.tension)
            client.connect();wait_for_load(client,args.rms_stop)
            snapshot=client.get_motor_load()
            # Routine E-stop has already been cleared by the newly owned master.
            validate_load(snapshot,time.monotonic(),args,enabled=False)
            if check_load(snapshot,time.monotonic(),args.rms_stop)>40:
                raise RuntimeError('start with all RMS channels below 40%')
            vision.set_boost(True);vision.start()
            deadline=time.monotonic()+8
            while vision.latest_pose() is None and time.monotonic()<deadline:
                if vision.error:raise RuntimeError(vision.error)
                time.sleep(.02)
            pose=vision.latest_pose()
            if pose is None or vision.status()['note']:raise RuntimeError('no fresh unambiguous calibration')
            if np.any(np.array(pose[:2])<LOW) or np.any(np.array(pose[:2])>HIGH):raise RuntimeError('calibration outside workspace')
            plan['calibration_pose']=list(pose)
            plan['motor_load_source']=client.get_motor_load(metadata_only=True)
            (directory/'plan.json').write_text(json.dumps(plan,indent=2))
            attempted=True;client.enable(pose[0],pose[1],math.degrees(pose[2]))
            client.set_ramp(args.ramp_ms)
            e=Experiment(client,vision,args,samples)
            e.setup_limits(.2,.4);e.observe(.5);e.settled(pose[:2])
            for t in plan['trials']:
                print(f"Trial {t['id']+1}/{len(plan['trials'])}: {t['kind']} site {t['site']} accel {t['accel']} m/s²",flush=True)
                e.trial=-(t['id']+1);e.reposition(np.asarray(t['start']))
                e.trial=t['id'];e.setup_limits(t['speed'],t['accel'])
                e.observe(.2);begin=len(e.rows)
                if t['kind']=='pulse':client.command_position(*t['end'],0)
                e.observe(t['duration']);e.settled(t['end'])
                result=score(e.rows[begin:],t,args.camera_latency_ms/1000)
                if t['kind']=='pulse' and min(t['prediction']['predicted_launch_accel_m_s2'],
                                             t['prediction']['predicted_braking_accel_m_s2'])<.8*t['accel']:
                    result['passed']=False
                    result['reason']='launch or braking did not exercise at least 80% of requested acceleration'
                results.append(dict(trial=t,metrics=result))
                (directory/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(result),flush=True)
                if not result['passed']:raise RuntimeError('tracking qualification failed; remaining ladder cancelled')
                e.observe(args.rest)
            outcome={'status':'complete','note':'No claim beyond tested trajectories; model fit requires separate validation.'}
    except (Exception,KeyboardInterrupt) as exc:
        outcome={'status':'aborted','reason':str(exc) or 'interrupted'}
        try:diagnostic=vision.diagnostic_snapshot()
        except Exception as diagnostic_error:outcome['camera_diagnostic_error']=str(diagnostic_error)
        print('Experiment stopped:',outcome['reason'],file=sys.stderr)
    finally:
        # Cleanup continues even if camera or logging failed. Do not auto-reenable.
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            try:shutdown(client,attempted_enable=attempted)
            except Exception as exc:outcome['shutdown_error']=str(exc);outcome['status']='aborted'
        finally:
            vision.stop();stop_master(proc)
            if diagnostic is not None:
                try:vision.save_diagnostic(directory,diagnostic)
                except Exception as exc:outcome['camera_diagnostic_error']=str(exc)
            for sig,h in handlers.items():signal.signal(sig,h)
            passed_ids={r['trial']['id'] for r in results if r['metrics']['passed']}
            pulses=[t for t in plan['trials'] if t['kind']=='pulse']
            passed_caps=[cap for cap in sorted({t['accel'] for t in pulses})
                         if all(t['id'] in passed_ids for t in pulses if t['accel']==cap)]
            outcome['completed_accel_caps_m_s2']=passed_caps
            outcome['quick_screen']=getattr(args,'quick',False)
            if pulses:
                print('Acceleration caps passing all planned directions:',passed_caps or 'none')
            (directory/'outcome.json').write_text(json.dumps(outcome,indent=2))
    return outcome['status']=='complete'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['baseline','workspace','sweep','endurance'])
    p.add_argument('--quick',action='store_true',help='rough center acceleration screen: 40/60/80/100/120, one pass, short holds, no extra rest')
    p.add_argument('--grid',type=int,choices=[1,3,5],default=1)
    p.add_argument('--accels',help='m/s²; ascending ladder (overrides preset)')
    p.add_argument('--speed',type=float,default=1.5,help='m/s, at most 3')
    p.add_argument('--stroke',type=float,default=120,help='mm per pulse')
    p.add_argument('--repeats',type=int)
    p.add_argument('--hold',type=float,help='stationary recording seconds/site')
    p.add_argument('--rest',type=float)
    p.add_argument('--ramp-ms',type=float,default=3)
    p.add_argument('--tension',type=float,default=1.5,help='fixed startup pretension in mm')
    p.add_argument('--rms-stop',type=float,default=70)
    p.add_argument('--current-stop',type=float,default=12,help='absolute per-drive A cutoff')
    p.add_argument('--min-volts',type=float,default=60,help='IPC-5 bus-voltage cutoff')
    p.add_argument('--error-stop',type=float,default=40,help='gross camera tracking error in mm')
    p.add_argument('--camera-latency-ms',type=float,default=15,help='fixed frame-transfer correction, not fitted per move')
    p.add_argument('--output',type=Path)
    p.add_argument('--configuration-note',default='',help='record changes to drive settings, routing, tuning, etc.')
    p.add_argument('--analyze',type=Path,help='offline fit/report for an existing experiment')
    p.add_argument('--live',action='store_true')
    a=p.parse_args()
    if a.analyze:
        if a.live:p.error('--analyze cannot be combined with --live')
        print(json.dumps(analyze(a.analyze),indent=2));return
    a.stage=a.stage or ('sweep' if a.quick else 'baseline')
    if a.quick and a.stage!='sweep':p.error('--quick is an acceleration sweep preset')
    if a.accels is None:a.accels='40,60,80,100,120' if a.quick else '2,5,10,20,30,40,60'
    if a.repeats is None:a.repeats=1 if a.quick else 3
    if a.hold is None:a.hold=2 if a.quick else 8
    if a.rest is None:a.rest=0 if a.quick else 2
    vals=[a.rest,a.ramp_ms,a.tension,a.rms_stop,a.current_stop,a.min_volts,a.error_stop,a.camera_latency_ms]
    if not np.isfinite(vals).all() or not (0<=a.rest<=60 and .2<=a.ramp_ms<=50 and 0<=a.tension<=3 and
        40<=a.rms_stop<=85 and 1<=a.current_stop<=16 and 50<=a.min_volts<=75 and
        10<=a.error_stop<=60 and 0<=a.camera_latency_ms<=30):p.error('invalid monitor/setup settings')
    try:trials=design(a.stage,a.grid,tuple(float(x) for x in a.accels.split(',')),a.speed,a.repeats,a.stroke,a.hold)
    except ValueError as exc:p.error(str(exc))
    cache={}
    for t in trials:
        if t['kind']=='pulse':
            key=(tuple(t['start']),tuple(t['end']),t['accel'],t['speed'])
            if key not in cache:cache[key]=predict_pulse(t,a.ramp_ms)
            t['prediction']=cache[key]
    directory=a.output or ROOT/'logs/characterization'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True,exist_ok=False)
    plan=dict(schema=1,stage=a.stage,bounds_mm=BOUNDS.tolist(),trials=trials,
              settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              hardware_revision='four-2331S-RLNA-20260929')
    (directory/'plan.json').write_text(json.dumps(plan,indent=2));preview(directory,plan)
    print(f"{len(trials)} trials. Preview: {(directory/'preview.html').resolve().as_uri()}")
    if a.quick:print('Quick screening: one pass per direction, no added rest. Estimates a local cap; does not qualify sustained duty.')
    insufficient=False
    for cap in sorted({t['accel'] for t in trials if t['kind']=='pulse'}):
        predictions=[t['prediction'] for t in trials if t['kind']=='pulse' and t['accel']==cap]
        launch=min(t['predicted_launch_accel_m_s2'] for t in predictions)
        brake=min(t['predicted_braking_accel_m_s2'] for t in predictions)
        print(f'Cap {cap:g}: predicted launch {launch:.1f}, braking {brake:.1f} m/s²')
        insufficient|=min(launch,brake)<.8*cap
    if insufficient:
        print('This profile does not exercise 80% of its cap in both launch and braking. Adjust speed/stroke/ramp before a live test.')
        if a.live:p.error('under-exercised acceleration profile; no hardware accessed')
    if not a.live:return
    print('Remove the puck. Free the camera and stop existing master/policy/UI hardware sessions.\n'
          'This WILL enable and move the robot. Keep the hardware stop accessible. Type RUN to begin.')
    if input('> ').strip()!='RUN':print('Cancelled; no hardware accessed.');return
    ok=live(a,plan,directory)
    print('Session:',directory)
    if not ok:raise SystemExit(1)


if __name__=='__main__':main()
