"""The comparison must remain open loop, including through camera dropouts."""

import copy
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from airhockey.hardware_replay import simulate, load_session, sample, safe_path
from airhockey.replay_log import ReplayLog
from airhockey import replay_server


def session(duration=1.0):
    times = np.arange(-0.04, duration + 0.006, 0.005)
    return dict(
        duration=duration,
        meta={"ramp_s": 0.003},
        events=[],
        tracks={
            "puck": [[float(t), 0.5, 1.1 - 0.2 * t] for t in times],
            "agent": [[float(t), 0.5, 0.35] for t in times],
            "human": [[float(t), 0.5, 1.8] for t in times],
        },
        commands=[[0, 0.5, 0.35, 12, 40]],
    )


def test_future_puck_and_robot_measurements_cannot_correct_simulation():
    a = session()
    b = copy.deepcopy(a)
    for key in ("puck", "agent"):
        for row in b["tracks"][key]:
            if row[0] > 0.001:
                row[1], row[2] = 0.15, 0.6
    ra, rb = simulate(a, [0], duration=0.4)[0], simulate(b, [0], duration=0.4)[0]
    np.testing.assert_array_equal(ra["frames"], rb["frames"])
    assert rb["errors"]["puck"]["mean_mm"] > ra["errors"]["puck"]["mean_mm"] + 100
    assert (
        ra["frames"][-1][2] < ra["frames"][0][2]
    )  # puck evolves from initial velocity


def test_commands_change_robot_and_replayed_human_can_strike_puck():
    a = session()
    b = copy.deepcopy(a)
    b["commands"] = [[0, 0.7, 0.5, 12, 40]]
    ra, rb = simulate(a, [0], duration=0.4)[0], simulate(b, [0], duration=0.4)[0]
    assert abs(rb["frames"][-1][3] - ra["frames"][-1][3]) > 0.15
    b = copy.deepcopy(a)
    for row in b["tracks"]["human"]:
        row[2] = 1.8 - 3 * max(0, row[0])
    rb = simulate(b, [0], duration=0.4)[0]
    assert abs(rb["frames"][-1][2] - ra["frames"][-1][2]) > 0.1


def test_human_gap_stops_rollout_and_missing_initial_state_is_reported():
    s = session()
    s["tracks"]["human"] = [
        x for x in s["tracks"]["human"] if x[0] <= 0.1 or x[0] >= 0.5
    ]
    r = simulate(s, [0, 0.3], duration=0.8)
    assert r[0]["reason"] == "Human tracking gap"
    assert r[0]["frames"][-1][0] <= 0.102
    assert not r[1]["frames"]
    assert "human" in r[1]["reason"]


def test_recorded_end_and_simulated_goal_do_not_create_a_new_serve():
    s = session()
    s["events"] = [{"type": "end", "t": 0.12}]
    r = simulate(s, [0], duration=0.5)[0]
    assert r["frames"][-1][0] <= 0.12
    s = session()
    for row in s["tracks"]["puck"]:
        row[2] = 1.98 + 2 * row[0]
    s["tracks"]["human"] = [[t, 0.1, 1.8] for t, _, _ in s["tracks"]["human"]]
    r = simulate(s, [0], duration=0.5)[0]
    assert "Simulated goal" in r["reason"]
    assert r["frames"][-1][2] > 2
    assert r["frames"][-1][0] < 0.05


def test_grid_rollouts_match_independent_runs():
    s = session(1.5)
    batch = simulate(s, [0, 0.2, 0.4], duration=0.2)
    for r, t in zip(batch, [0, 0.2, 0.4]):
        single = simulate(s, [t], duration=0.2)[0]
        np.testing.assert_allclose(r["frames"], single["frames"], atol=1e-6)


def test_loader_reads_full_frames_and_clock_aligned_commands(tmp_path):
    records = [
        {"type": "meta", "live": True, "ramp": 3, "camera_delay_s": 0.0077},
        {"type": "clock", "t": 10, "monotonic": 110.0077},
        {
            "type": "frame",
            "t": 10,
            "puck": [1003.3, 480],
            "agent": [1650, 480],
            "human": [300, 480],
        },
        {"type": "frame", "t": 10.005, "puck": None, "agent": None, "human": None},
        {
            "type": "command",
            "t": 10.02,
            "monotonic": 110.012,
            "x": 1600,
            "y": 500,
            "speed": 12000,
            "accel": 32000,
        },
    ]
    p = tmp_path / "example.replay.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in records) + '\n{"partial":')
    s = load_session(p.name, tmp_path)
    assert s["duration"] == pytest.approx(0.005)
    assert len(s["tracks"]["puck"]) == 1
    assert s["commands"][0][0] == pytest.approx(0.012)
    assert s["commands"][0][3:] == [12, 32]


def test_writer_keeps_missing_observations_missing_and_flushes_end(tmp_path):
    p = tmp_path / "test.replay.jsonl"
    writer = ReplayLog(p, SimpleNamespace(policy="test", live=False, ramp=3))
    writer.sync(0, 100)
    report = SimpleNamespace(
        puck=[(1000, 400, 0)],
        t_puck=0,
        t_mallet=0,
        t_opponent=-1,
        mallet=(1600, 400),
        opponent=(400, 400),
    )
    writer.frame(0, report)
    writer.frame(0.005, report)
    writer.end()
    writer.close()
    records = [json.loads(line) for line in p.read_text().splitlines()]
    frames = [r for r in records if r["type"] == "frame"]
    assert frames[0]["puck"] == [1000, 400]
    assert frames[0]["human"] is None
    assert frames[1]["puck"] is frames[1]["agent"] is None
    assert records[-1]["type"] == "end"


def test_api_validation_and_path_containment(monkeypatch):
    with pytest.raises(ValueError):
        safe_path("../secret.ticks.csv")
    monkeypatch.setattr(replay_server, "load_session", lambda name: session())
    req = replay_server.SimulationRequest(starts=[0], duration=0.1)
    assert asyncio.run(replay_server.rollout("test", req))[0]["frames"]
    with pytest.raises(HTTPException) as e:
        asyncio.run(
            replay_server.rollout("test", replay_server.SimulationRequest(starts=[-1]))
        )
    assert e.value.status_code == 400
    with pytest.raises(ValidationError):
        replay_server.SimulationRequest(starts=[0] * 33)


def test_sampling_does_not_bridge_tracking_holes():
    got = sample([[0, 0.1, 0.2], [1, 0.3, 0.4]], [0, 0.5, 1, 1.2])
    assert np.isnan(got[1]).all() and np.isnan(got[3]).all()
    np.testing.assert_allclose(got[[0, 2]], [[0.1, 0.2], [0.3, 0.4]])
