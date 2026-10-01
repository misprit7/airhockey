"""Expanded-region planning and verification without accessing devices."""
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from airhockey.workspace_probe import design_workspace_probe,PROBE_BOUNDS
from airhockey.motor_patterns import LOW,HIGH,BOUNDS
from airhockey.hardware import CDPRClient
import cdpr_geometry as geom

ROOT=Path(__file__).resolve().parents[2]


def test_envelope_has_30mm_rim_clearance_and_leaves_policy_region_unchanged():
    assert PROBE_BOUNDS[1]+geom.MALLET_RADIUS_MM==pytest.approx(geom.RAIL_MAX_X-30)
    assert PROBE_BOUNDS[2]-geom.MALLET_RADIUS_MM==pytest.approx(geom.RAIL_MIN_Y+30)
    assert PROBE_BOUNDS[3]+geom.MALLET_RADIUS_MM==pytest.approx(geom.RAIL_MAX_Y-30)
    np.testing.assert_allclose(BOUNDS,[1350,1917.5,172.9,793])


@pytest.mark.parametrize('probe',[False,True])
def test_compiled_firmware_bounds_match_selected_python_envelope(tmp_path,probe):
    source=tmp_path/'bounds.cpp';binary=tmp_path/'bounds'
    source.write_text('#include <cstdio>\n#include "cdpr_geometry.h"\nint main(){printf("%f %f %f %f",WS_MIN_X,WS_MAX_X,WS_MIN_Y,WS_MAX_Y);}\n')
    subprocess.run(['g++','-std=c++11','-Ishared',*(['-DAIRHOCKEY_PROBE_WORKSPACE'] if probe else []),
                    str(source),'-o',str(binary)],cwd=ROOT,check=True)
    actual=np.fromstring(subprocess.check_output([str(binary)],text=True),sep=' ')
    np.testing.assert_allclose(actual,PROBE_BOUNDS if probe else BOUNDS,atol=.001,rtol=0)


def test_excursions_progress_outward_and_return_inside_tested_region():
    plan=design_workspace_probe();reaches=[t for t in plan if t['kind']=='reach'];previous={}
    assert len(reaches)==39
    assert len({t['site'] for t in reaches if t['outermost']})==8
    for t in reaches:
        start=np.array(t['start']);tip=np.array(t['tip'])
        assert np.all(start>=LOW+39.9) and np.all(start<=HIGH-39.9)
        assert t['start']==t['end'] and t['hold_s']==0
        assert np.all(tip>=PROBE_BOUNDS[[0,2]]) and np.all(tip<=PROBE_BOUNDS[[1,3]])
        prior=previous.get(t['site'],start)
        assert 0<np.linalg.norm(tip-prior)<=40.001
        previous[t['site']]=tip
    assert all(t.get('hold_s',5)==5 for t in design_workspace_probe(hold_s=5))
    with pytest.raises(ValueError):design_workspace_probe(speed=2)
    with pytest.raises(ValueError):design_workspace_probe(step_mm=100)


def test_workspace_query_is_read_only_and_rejects_old_firmware(monkeypatch):
    c=CDPRClient();calls=[]
    def send(cmd):calls.append(cmd);return 'OK WORKSPACE 1200 1937.5 61.4 904.5'
    monkeypatch.setattr(c,'_send',send)
    np.testing.assert_allclose(c.get_workspace(),PROBE_BOUNDS)
    assert calls==['WORKSPACE']
    monkeypatch.setattr(c,'_send',lambda _: 'ERR unknown command')
    with pytest.raises(RuntimeError,match='Update master and firmware'):c.get_workspace()


def test_wrong_firmware_aborts_before_enabling_or_opening_camera(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'ai/bin'))
    import characterize_robot as runner
    import airhockey.hardware
    import airhockey.vision_service
    events=[]
    class Client:
        def connect(self):events.append('connect')
        def get_workspace(self):return tuple(BOUNDS)
        def close(self):events.append('close')
        def enable(self,*_):pytest.fail('wrong envelope must never enable')
    class Vision:
        def start(self):pytest.fail('wrong envelope must fail before camera startup')
        def stop(self):events.append('camera_stop')
    monkeypatch.setattr(airhockey.hardware,'CDPRClient',Client)
    monkeypatch.setattr(airhockey.vision_service,'VisionService',Vision)
    monkeypatch.setattr(runner,'start_master',lambda *_:'owned')
    monkeypatch.setattr(runner,'stop_master',lambda p:events.append('master_stop'))
    args=SimpleNamespace(tension=1.5,rms_stop=70)
    plan=dict(expanded_workspace=True,bounds_mm=PROBE_BOUNDS.tolist(),trials=design_workspace_probe())
    assert not runner.live(args,plan,tmp_path)
    assert events==['connect','close','camera_stop','master_stop']
    assert 'flash teensy41_probe' in json.loads((tmp_path/'outcome.json').read_text())['reason']


@pytest.mark.parametrize('hold_s',[0,5])
def test_excursion_only_requires_tip_to_settle_for_explicit_hold(monkeypatch,hold_s):
    sys.path.insert(0,str(ROOT/'ai/bin'))
    import characterize_robot as runner
    commands=[];settles=[];observations=[]
    client=SimpleNamespace(command_position=lambda *args:commands.append(args))
    e=runner.Experiment(client,None,SimpleNamespace(),io.StringIO(),bounds=PROBE_BOUNDS)
    t=design_workspace_probe(hold_s=hold_s)[1]
    monkeypatch.setattr(e,'require_camera',lambda:None)
    monkeypatch.setattr(e,'poll',lambda:dict(ctl=[*t['tip'],50,0]))
    monkeypatch.setattr(e,'observe',observations.append)
    monkeypatch.setattr(e,'settled',lambda p:settles.append(list(p)))
    e.excursion(t)
    assert commands==[(*t['tip'],0),(*t['start'],0)]
    assert settles==([t['tip'],t['start']] if hold_s else [t['start']])
    assert len(observations)==(2 if hold_s else 1)
    if hold_s:assert observations[0]==hold_s


def test_outward_timeout_aborts_without_blindly_starting_next_move(monkeypatch):
    sys.path.insert(0,str(ROOT/'ai/bin'))
    import characterize_robot as runner
    clock=[0.];commands=[]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    t=design_workspace_probe()[1]
    client=SimpleNamespace(command_position=lambda *args:commands.append(args))
    e=runner.Experiment(client,None,SimpleNamespace(),io.StringIO(),bounds=PROBE_BOUNDS)
    monkeypatch.setattr(e,'require_camera',lambda:None)
    monkeypatch.setattr(e,'poll',lambda:dict(ctl=[*t['start'],0,0]))
    with pytest.raises(RuntimeError,match='did not reach outward target'):e.excursion(t)
    assert commands==[(*t['tip'],0)]
    assert clock[0]<3
