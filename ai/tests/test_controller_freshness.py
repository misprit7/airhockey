"""Recorded controller stalls and serial framing, tested without hardware."""
from pathlib import Path
import subprocess
import math
import pytest
from airhockey.hardware import CDPRClient

ROOT = Path(__file__).resolve().parents[2]


def test_serial_framer_handles_every_split_and_recovers_oversized_lines(tmp_path):
    binary = tmp_path / 'serial_lines'
    subprocess.run(['g++', '-std=c++11', '-Wall', '-Wextra', '-Werror', '-Isw/lib',
                    'sw/test/test_serial_lines.cpp', '-o', str(binary)], cwd=ROOT, check=True)
    subprocess.run([str(binary)], check=True)


def test_real_master_wait_keeps_status_after_ack_and_partial_lines(tmp_path):
    binary = tmp_path / 'master_serial'
    sdk = ROOT / 'sw/third_party/sFoundation/sFoundation'
    subprocess.run(['g++', '-std=c++11', '-O1', '-Wall', '-Wextra',
                    '-Isw/lib', '-Ishared', '-Isw/third_party/sFoundation/inc/inc-pub',
                    'sw/test/test_master_serial.cpp', 'sw/lib/clearpath.cpp',
                    'sw/lib/motor_load.cpp', '-L'+str(sdk), '-Wl,-rpath,'+str(sdk),
                    '-lsFoundation20', '-lpthread', '-o', str(binary)], cwd=ROOT, check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize('timestamp, expected', [('99.63', .37), ('100', 0), ('', math.inf),
                                                ('nan', math.inf), ('101', math.inf)])
def test_cached_position_preserves_real_age(monkeypatch, timestamp, expected):
    client = CDPRClient()
    monkeypatch.setattr(client, '_send', lambda _: 'OK 1833.87 512.26 -542.35 -207 '+timestamp)
    monkeypatch.setattr('airhockey.hardware.time.monotonic', lambda: 100.)
    assert client.get_position() == (1833.87, 512.26, -542.35, -207)
    sample = client.get_position_sample()
    assert sample[:4] == client.get_position()
    assert sample[4] == pytest.approx(expected)


def test_replay_records_controller_timing_without_making_unknown_age_fresh(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from airhockey.replay_log import ReplayLog
    monkeypatch.setattr('airhockey.replay_log.time.monotonic', lambda: 100.)
    path = tmp_path / 'replay.jsonl'
    replay = ReplayLog(path, SimpleNamespace(policy='test', live=False, ramp=3))
    replay.controller(5, 1800, 500, 20, 30, .35)
    replay.controller(6, 1800, 500, 20, 30, math.inf)
    replay.close()
    a, b = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert a['sample_monotonic'] == 99.65 and a['age_s'] == .35
    assert a['vx'] == 20 and a['vy'] == 30
    assert b['sample_monotonic'] is None and b['age_s'] is None
