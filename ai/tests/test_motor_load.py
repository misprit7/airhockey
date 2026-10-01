"""Load logging tested without the SDK, ports, motors or a running master."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from airhockey.hardware import CDPRClient
from airhockey.motor_load import records, summarize, latest_snapshot
from airhockey.replay_log import ReplayLog

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    folder = tmp_path_factory.mktemp("motor-load")
    binary = folder / "test_motor_load"
    subprocess.run(["g++", "-std=c++11", "-pthread", "-Wall", "-Wextra", "-Werror",
                    "-Isw/lib", "sw/test/test_motor_load.cpp", "sw/lib/motor_load.cpp",
                    "-o", str(binary)], cwd=ROOT, check=True, capture_output=True)
    path = folder / "load.jsonl"
    result = subprocess.run([str(binary), str(path)], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return path, json.loads(result.stdout)


def test_recorder_preserves_units_errors_timestamps_and_disabled_samples(recording):
    path, cache = recording
    rows = list(records(path))
    samples = [r for r in rows if r["type"] == "motor_load"]
    assert len(samples) == 2
    assert rows[0]["type"] == "meta" and rows[-1]["type"] == "end"
    a, b = samples
    assert a["motors"][0]["rms_pct"]["value"] == 42
    assert a["motors"][0]["torque_amps"]["value"] == -2.5
    assert a["motors"][0]["bus_voltage_v"]["value"] == 74.5
    # Voltage is dynamic: it must update on the next sample, not once/minute.
    assert b["motors"][0]["bus_voltage_v"]["value"] == 10.0
    assert a["motors"][1]["bus_voltage_v"]["value"] is None
    assert not a["motors"][1]["bus_voltage_v"]["valid"]
    assert b["motors"][0]["rms_pct"]["value"] is None
    assert not b["motors"][0]["rms_pct"]["valid"]
    assert b["motors"][0]["rms_pct"]["error"] == "read timeout"
    assert a["motors"][0]["rms_slow_pct"]["error"] == 'unsupported "slow"\nchannel'
    assert a["motors"][1]["encoder_counts"]["value"] is None
    assert a["motors"][1]["encoder_counts"]["error"] == "nonfinite"
    assert a["motors"][0]["status"]["value"] == [1, 0]
    assert b["motors"][0]["status"]["value"] == [0, 0]
    assert a["motors"][0]["alerts"]["value"] == [0, 0, 4294967295]
    for sample in samples:
        for motor in sample["motors"]:
            for key in ("rms_pct", "torque_amps", "status"):
                field = motor[key]
                assert sample["monotonic_start"] <= field["start"] <= field["end"] <= sample["monotonic"]
            field = motor["bus_voltage_v"]
            if field["valid"]:
                assert sample["monotonic_start"] <= field["start"] <= field["end"] <= sample["monotonic"]
    assert b["context"]["fault"]
    assert cache["sample"] == b and cache["logging_ok"]
    assert b["dropped_command_events"] == 6
    assert len([r for r in rows if r["type"] == "command_received"]) == 1024


def test_file_inspector_ignores_partial_last_line_and_never_turns_missing_into_zero(recording, tmp_path):
    path, _ = recording
    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(path.read_bytes() + b'{"type":"motor_load"')
    data = summarize(partial)
    assert data["samples"] == 2
    assert data["motors"][0]["rms_peak"] == 42
    assert data["motors"][0]["invalid_rms_samples"] == 1
    assert data["motors"][0]["rms_slow_peak"] is None
    assert data["motors"][0]["torque_abs_peak_amps"] == 2.5
    assert data["source"]["rms_units"] == "percent_of_shutdown"


def test_cached_api_and_replay_reference_work_before_camera_sync(recording, monkeypatch, tmp_path):
    _, cache = recording
    client = CDPRClient()
    calls = []
    def send(command):
        calls.append(command)
        return "OK " + json.dumps(cache["source"] if command == "LOADMETA" else cache)
    monkeypatch.setattr(client, "_send", send)
    assert client.get_motor_load() == cache
    source = client.get_motor_load(metadata_only=True)
    assert calls == ["LOAD", "LOADMETA"]
    path = tmp_path / "replay.jsonl"
    replay = ReplayLog(path, SimpleNamespace(policy="test", live=False, ramp=3))
    replay.motor_load_source(source)
    replay.close()
    entry = list(records(path))[1]
    assert entry["type"] == "motor_load_source" and entry["source"] == source
    assert entry["monotonic"] >= source["started_monotonic"]


def test_old_master_motor_load_error_is_explicit(monkeypatch):
    client = CDPRClient()
    monkeypatch.setattr(client, "_send", lambda _: "ERR unknown command")
    with pytest.raises(RuntimeError, match="unavailable"):
        client.get_motor_load()


def test_ui_monitor_tracks_field_freshness_and_preserves_signed_current(tmp_path):
    sample=dict(type='motor_load', unix=1000, monotonic=10, context={'motors_enabled':True},
                motors=[dict(node=0,torque_amps=dict(valid=True,value=-2.5,end=9.95),
                             rms_pct=dict(valid=True,value=81,end=9.2),
                             rms_slow_pct=dict(valid=False,value=0,end=10))])
    p=tmp_path/'1000.jsonl'
    p.write_text(json.dumps(sample)+'\n'+ '{"partial":')
    result=latest_snapshot(tmp_path,now=1000.1)
    assert result['state']=='live'
    m=result['motors'][0]
    assert m['torque_amps']['value']==-2.5 and m['torque_amps']['fresh']
    assert not m['rms_pct']['fresh'] and m['rms_pct']['value']==81
    assert m['rms_slow_pct']['value'] is None
    assert result['motors'][1]['torque_amps']['value'] is None
    assert latest_snapshot(tmp_path,now=1002)['state']=='stale'
    assert latest_snapshot(tmp_path,now=999)['state']=='stale'
    p.write_text(json.dumps(sample)+'\n'+json.dumps({'type':'end'})+'\n')
    assert latest_snapshot(tmp_path,now=1000.1)['state']=='stale'


def test_ui_monitor_bounded_tail_and_new_session(tmp_path):
    assert latest_snapshot(tmp_path)['state']=='unavailable'
    sample=dict(type='motor_load',unix=1000,monotonic=10,motors=[])
    p=tmp_path/'old.jsonl'
    p.write_text('x'*300000+'\n'+json.dumps(sample)+'\n')
    assert latest_snapshot(tmp_path,now=1000.1)['state']=='live'
    newer=tmp_path/'new.jsonl'
    newer.write_text('{"type":"meta"}\n')
    import os
    os.utime(newer,ns=(p.stat().st_mtime_ns+1000000,)*2)
    result=latest_snapshot(tmp_path,now=1000.1)
    assert result['state']=='unavailable' and result['source']=='new.jsonl'
