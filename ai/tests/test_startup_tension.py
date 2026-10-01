"""Pretension configuration and failure paths, with no robot connection."""
import pytest
from airhockey.hardware import CDPRClient
from airhockey.dynamics import HardwareDynamics


def test_client_validates_and_requires_explicit_configuration_ack(monkeypatch):
    client=CDPRClient();commands=[]
    def send(command):
        commands.append(command)
        return 'OK PRETENSION 1.50'
    monkeypatch.setattr(client,'_send',send)
    client.set_startup_tension(1.5)
    assert commands==['PRETENSION 1.50']
    for value in (-1,3.1,float('nan'),float('inf')):
        with pytest.raises(ValueError):client.set_startup_tension(value)
    assert len(commands)==1
    monkeypatch.setattr(client,'_send',lambda _: 'ERR unknown command')
    with pytest.raises(RuntimeError,match='Restart cdpr_master'):
        client.set_startup_tension(1.5)


@pytest.mark.parametrize('failure',[None,'configuration','enable'])
def test_configuration_precedes_enable_and_failure_closes_client(monkeypatch,failure):
    events=[]
    class FakeClient:
        def __init__(self,*_):pass
        def connect(self):events.append('connect')
        def set_startup_tension(self,mm):
            events.append(('pretension',mm))
            if failure=='configuration':raise RuntimeError('configuration rejected')
        def enable(self,*pose):
            events.append(('enable',pose))
            if failure=='enable':raise RuntimeError('enable failed')
        def disable(self):events.append('disable')
        def close(self):events.append('close')
    monkeypatch.setattr('airhockey.hardware.CDPRClient',FakeClient)
    def construct():return HardwareDynamics(cal_pose_mm=(1500,450,135),tension_mm=1.5)
    if failure:
        with pytest.raises(RuntimeError):construct()
    else:construct()
    expected=['connect',('pretension',1.5)]
    if failure!='configuration':expected.append(('enable',(1500,450,135)))
    if failure=='enable':expected.append('disable')
    if failure:expected.append('close')
    assert events==expected
