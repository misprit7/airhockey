import importlib.util
import json
from pathlib import Path
import sys
import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from airhockey.identification import design,score,spatial_features,predict_pulse,analyze,summarize_load
from airhockey.motor_patterns import LOW,HIGH,CENTER


def test_sweep_is_bounded_ordered_and_exercises_both_directions():
    trials=design('sweep',3,(2,5),repeats=2)
    assert len([t for t in trials if t['kind']=='hold'])==9
    for t in trials:
        assert np.all(np.array([t['start'],t['end']])>=LOW+39)
        assert np.all(np.array([t['start'],t['end']])<=HIGH-39)
    pulses=[t for t in trials if t['kind']=='pulse']
    assert [t['accel'] for t in pulses]==sorted(t['accel'] for t in pulses)
    assert len(set(round(t['direction_deg']) for t in pulses))==8
    with pytest.raises(ValueError):design('sweep',accelerations=(10,5))
    with pytest.raises(ValueError):design('sweep',stroke=2000)
    with pytest.raises(ValueError):design('sweep',speed=float('nan'))


def test_stationary_workspace_samples_nearer_edges_than_pulses():
    points=np.array([t['start'] for t in design('workspace',3)])
    np.testing.assert_allclose(points.min(0),LOW+40)
    np.testing.assert_allclose(points.max(0),HIGH-40)


def synthetic_rows(offset=0,delay=.015):
    rows=[]
    for t in np.arange(0,1,.005):
        def position(t):return CENTER+np.array([50*np.sin(np.pi*min(max(t,0),.5)/.5),0]) if t<.5 else CENTER
        rows.append(dict(ctl_t=float(t),ctl=[*position(t),0,0],cam_t=float(t+delay),cam=[*(position(t)+[offset,0]),0]))
    return rows


def test_score_does_not_fit_away_spatial_error_or_accept_missing_camera():
    trial=dict(end=CENTER.tolist(),accel=5)
    assert score(synthetic_rows(),trial)['passed']
    assert not score(synthetic_rows(30),trial)['passed']
    assert not score(synthetic_rows()[:3],trial)['passed']
    assert score(synthetic_rows(),trial)['accel_fit_window_s']>0


def test_model_distinguishes_position_and_acceleration_sign():
    a=spatial_features([*LOW,0,0,20000,0])
    b=spatial_features([*HIGH,0,0,20000,0])
    c=spatial_features([*LOW,0,0,-20000,0])
    assert a.shape==(1,54)
    assert not np.allclose(a,b)
    assert not np.allclose(a,c)
    assert np.all(a>=0)


def test_load_summary_excludes_old_cache_and_preserves_motor_sign():
    rows=[]
    for t,stamp,current,rms in [(1,.9,-20,9),(2,2,-3,10),(3,2,-3,10),(4,4,-5,14)]:
        fields={k:dict(valid=True,end=stamp,value=v) for k,v in
                [('torque_amps',current),('rms_pct',rms)]}
        rows.append(dict(t=t,load=dict(sample=dict(motors=[dict(node=0,**fields)]))))
    report=summarize_load(rows)[0]
    assert report['torque_amps']['samples']==2
    assert report['torque_amps']['mean_abs']==4
    assert report['torque_amps']['peak_abs']==5
    assert report['rms_pct']['rise_percentage_points_per_s']==2


def test_profile_prediction_uses_actual_ramp_and_speed():
    trial=next(t for t in design('sweep',accelerations=(5,),repeats=1) if t['kind']=='pulse')
    prediction=predict_pulse(trial)
    assert 4<=prediction['predicted_peak_accel_m_s2']<=5.1
    assert prediction['predicted_peak_speed_m_s']<=1.51


def test_long_ramp_does_not_confuse_braking_peak_with_launch():
    trial=next(t for t in design('sweep',accelerations=(80,),repeats=1) if t['kind']=='pulse')
    prediction=predict_pulse(trial,ramp_ms=10)
    assert 60<prediction['predicted_launch_accel_m_s2']<64
    assert 75<prediction['predicted_braking_accel_m_s2']<80
    assert prediction['predicted_launch_ms_above_95pct']==0
    trial['speed']=2.5
    prediction=predict_pulse(trial,ramp_ms=10)
    assert prediction['predicted_launch_accel_m_s2']>=79
    assert prediction['predicted_launch_ms_above_95pct']>=5


def test_no_model_export_with_insufficient_data(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    (tmp_path/'samples.jsonl').write_text('{"partial":')
    assert analyze(tmp_path)['status']=='insufficient_data'
    assert not (tmp_path/'current-model-candidate.json').exists()


def test_candidate_fit_and_rms_report_from_synthetic_recording(tmp_path):
    (tmp_path/'plan.json').write_text('{"settings":{"tension":1.5}}')
    rows=[]; heat=np.zeros((2,4));dt=.02
    for i in range(300):
        t=1+i*dt
        ctl=[CENTER[0]+60*np.sin(t),CENTER[1]+60*np.cos(t),60*np.cos(t),-60*np.sin(t)]
        motors=[]
        for node in range(4):
            current=1+node*.3+.1*np.sin(t)
            fields={k:dict(valid=True,end=t,value=v) for k,v in {
                'torque_amps':current,'rms_limit_amps':5.8,'rms_time_constant_s':3.,
                'rms_slow_limit_amps':4.1,'rms_slow_time_constant_min':1.}.items()}
            for j,(channel,limit,tau) in enumerate([('rms_pct',5.8,3),('rms_slow_pct',4.1,60)]):
                decay=np.exp(-dt/tau);heat[j,node]=decay*heat[j,node]+(1-decay)*(current/limit)**2
                fields[channel]=dict(valid=True,end=t,value=float(100*np.sqrt(heat[j,node])))
            motors.append(dict(node=node,**fields))
        rows.append(dict(ctl_t=t,ctl=ctl,trial=i//20,load=dict(sample=dict(motors=motors))))
    (tmp_path/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    report=analyze(tmp_path)
    assert report['status']=='candidate_not_validated'
    assert len(report['per_motor'])==4
    for m in report['per_motor']:
        assert m['held_out_trials']==[0,5,10]
        assert np.isfinite(m['rms_pct_mae_percentage_points'])
        assert m['current_mae_amps']<.5
    assert json.loads((tmp_path/'current-model-candidate.json').read_text())['schema']=='spatial-current-v1'


def test_monitor_rejects_missing_slow_rms_voltage_and_fault():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    args=SimpleNamespace(rms_stop=70,min_volts=60,current_stop=12)
    def field(v):return dict(valid=True,end=10.,value=v)
    motors=[dict(node=i,rms_pct=field(10),rms_slow_pct=field(10),torque_amps=field(1),
                 bus_voltage_v=field(74),status=field([1,0]),alerts=field([0,0,0]),
                 encoder_counts=field(0)) for i in range(4)]
    snapshot=dict(logging_ok=True,sample=dict(context=dict(motors_enabled=True,fault=False),motors=motors))
    assert runner.validate_load(snapshot,10.1,args)==10
    for key,value in [('bus_voltage_v',40),('rms_slow_pct',75),('torque_amps',13),('status',[0,0])]:
        old=motors[0][key]['value']; motors[0][key]['value']=value
        with pytest.raises(RuntimeError):runner.validate_load(snapshot,10.1,args)
        motors[0][key]['value']=old
    motors[0]['rms_slow_pct']['valid']=False
    with pytest.raises(RuntimeError):runner.validate_load(snapshot,10.1,args)


def test_partial_enable_failure_still_disables_and_stops_owned_master(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    import airhockey.hardware
    import airhockey.vision_service
    events=[]
    class Client:
        def connect(self):events.append('connect')
        def get_motor_load(self,**_):return {}
        def enable(self,*_):events.append('enable');raise RuntimeError('partial enable')
        def disable(self):events.append('disable')
        def close(self):events.append('close')
    class Vision:
        error=None
        def set_boost(self,*_):pass
        def start(self):pass
        def stop(self):events.append('camera_stop')
        def latest_pose(self):return (*CENTER,2.356)
        def status(self):return {'note':None}
    monkeypatch.setattr(airhockey.hardware,'CDPRClient',Client)
    monkeypatch.setattr(airhockey.vision_service,'VisionService',Vision)
    monkeypatch.setattr(runner,'start_master',lambda *_: 'owned')
    monkeypatch.setattr(runner,'stop_master',lambda p:events.append('master_stop'))
    monkeypatch.setattr(runner,'wait_for_load',lambda *_:0)
    monkeypatch.setattr(runner,'validate_load',lambda *_,**kw:0)
    monkeypatch.setattr(runner,'check_load',lambda *_:0)
    args=SimpleNamespace(tension=1.5,rms_stop=70)
    assert not runner.live(args,{'trials':[]},tmp_path)
    assert events==['connect','enable','disable','close','camera_stop','master_stop']
    assert json.loads((tmp_path/'outcome.json').read_text())['status']=='aborted'


@pytest.mark.parametrize('reported', ['delayed', 'wrong', 'missing', 'nan', 'fault'])
def test_limit_verification_waits_for_cache_but_rejects_failure(monkeypatch,reported):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    clock=[0.];commands=[];reads=[]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    monkeypatch.setattr(runner,'wait_for_load',lambda *_,**__:None)
    def validate(*_):
        if reported=='fault':raise RuntimeError('motor fault reported')
    monkeypatch.setattr(runner,'validate_load',validate)
    class Client:
        def set_limits(self,*args):commands.append(args)
        def get_motor_load(self):return {}
        def get_status(self):
            reads.append(clock[0])
            if reported=='missing':return {}
            return dict(speed_limit=200,accel_limit=(float('nan') if reported=='nan' else
                        400 if reported=='delayed' and clock[0]>=.04 else 1000))
        def command_position(self,*_):pytest.fail('limit verification must not command motion')
    experiment=runner.Experiment(Client(),None,SimpleNamespace(rms_stop=70),io.StringIO())
    if reported=='delayed':
        experiment.setup_limits(.2,.4)
        assert len(reads)>=3
    else:
        with pytest.raises(RuntimeError,match='motor fault' if reported=='fault' else 'verification timed out'):
            experiment.setup_limits(.2,.4)
    assert commands==[(200,400)]
    assert clock[0]<=1.03


@pytest.mark.parametrize('overrides,expected_caps,repeats,rest',[
    ([],[40,60,80,100,120],1,0),
    (['--accels','50,70','--repeats','2','--rest','1'],[50,70],2,1),
])
def test_quick_screen_preserves_monitor_limits_and_honors_overrides(tmp_path,monkeypatch,overrides,expected_caps,repeats,rest):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    monkeypatch.setattr(runner,'predict_pulse',lambda t,*_,**__: {
        'predicted_launch_accel_m_s2':t['accel'],'predicted_braking_accel_m_s2':t['accel']})
    monkeypatch.setattr(runner,'live',lambda *_:pytest.fail('offline preview opened hardware'))
    output=tmp_path/'preview'
    monkeypatch.setattr(sys,'argv',['characterize_robot','--quick','--output',str(output),*overrides])
    runner.main()
    plan=json.loads((output/'plan.json').read_text())
    pulses=[t for t in plan['trials'] if t['kind']=='pulse']
    assert sorted({t['accel'] for t in pulses})==expected_caps
    assert len(pulses)==len(expected_caps)*8*repeats
    assert plan['settings']['rest']==rest and plan['settings']['hold']==2
    assert plan['settings']['speed']==1.5 and plan['settings']['rms_stop']==70
    assert plan['settings']['current_stop']==12 and plan['settings']['tension']==1.5


def test_under_exercised_launch_refused_before_live_or_confirmation(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    monkeypatch.setattr(runner,'live',lambda *_:pytest.fail('must not access hardware'))
    monkeypatch.setattr('builtins.input',lambda *_:pytest.fail('must reject before RUN prompt'))
    monkeypatch.setattr(sys,'argv',['characterize_robot','--quick','--accels','80',
        '--ramp-ms','10','--live','--output',str(tmp_path/'preview')])
    with pytest.raises(SystemExit) as error:runner.main()
    assert error.value.code==2


def test_camera_grace_discards_bad_frames_and_keeps_electrical_checks(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    clock=[10.];checks=[]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'time',lambda:clock[0])
    monkeypatch.setattr(runner,'validate_load',lambda *_:checks.append(clock[0]))
    vision=SimpleNamespace(_lock=threading.Lock(),_pose=(*CENTER,2.35),_pose_t=9.99,_note=None,error=None)
    client=SimpleNamespace(get_motor_load=lambda:{},get_position_sample=lambda:(*CENTER,0,0,0))
    experiment=runner.Experiment(client,vision,SimpleNamespace(camera_latency_ms=15,error_stop=40),io.StringIO())
    assert experiment.poll()['camera_valid']
    for now in (10.005,10.02):
        clock[0]=now;vision._pose_t=now-.005;vision._note='arms disagree'
        assert not experiment.poll()['camera_valid']
        assert experiment.last_good_camera==9.99
    clock[0]=10.035;vision._pose_t=10.025;vision._note=None
    assert experiment.poll()['camera_valid']
    assert len(checks)==4
    # Repeated newly timestamped bad poses cannot extend the grace window.
    clock[0]=10.08;vision._pose_t=10.075;vision._note='arms disagree'
    with pytest.raises(RuntimeError,match='50 ms grace'):experiment.poll()
    assert len(checks)==5
    def fault(*_):raise RuntimeError('RMS reached cutoff')
    monkeypatch.setattr(runner,'validate_load',fault)
    with pytest.raises(RuntimeError,match='RMS reached'):experiment.poll()


def test_startup_needs_valid_camera_and_recovery_cannot_hide_long_gap(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
    import characterize_robot as runner
    clock=[10.]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'time',lambda:clock[0])
    monkeypatch.setattr(runner,'validate_load',lambda *_:None)
    vision=SimpleNamespace(_lock=threading.Lock(),_pose=(*CENTER,2.35),_pose_t=9.99,_note='missing marker',error=None)
    client=SimpleNamespace(get_motor_load=lambda:{},get_position_sample=lambda:(*CENTER,0,0,0))
    def make():return runner.Experiment(client,vision,SimpleNamespace(camera_latency_ms=15,error_stop=40),io.StringIO())
    with pytest.raises(RuntimeError):make().poll()
    vision._note=None;e=make();e.poll()
    clock[0]=10.03;vision._pose_t=10.02;vision._note='bad';e.poll()
    clock[0]=10.07;vision._pose_t=10.06;vision._note=None
    with pytest.raises(RuntimeError,match='recovered after a gap'):e.poll()


def test_score_excludes_rejected_measurements_and_reports_gaps():
    rows=synthetic_rows()
    for r in rows[50:54]:
        r['camera_valid']=False;r['cam']=[0,0,0]
    metrics=score(rows,dict(end=CENTER.tolist(),accel=80))
    assert metrics['passed'] and metrics['tracking_had_gaps']
    assert metrics['camera_rejected_frames']==4
    assert metrics['tracking_max_mm']<1
    for r in rows[50:65]:r['camera_valid']=False
    assert not score(rows,dict(end=CENTER.tolist(),accel=80))['passed']
