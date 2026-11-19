"""The ring test's move list and its reading, on a synthetic table: a mallet
that rings on its own near the goal (with two drives' currents following) and
only rings down after a step at the centre."""
import math

import numpy as np

from airhockey import ring_test
from airhockey.follow_test import duration_estimate, geom, travel_time


def synthetic(segs, caps, ring_hz=21.5):
    """Rows as FollowTest logs them (100 Hz loop, ~50 Hz camera) and drive
    current samples (~100 Hz per motor)."""
    rng = np.random.default_rng(0)
    rows, t, bounds = [], 0.0, []
    prev = (segs[0].x, segs[0].y)
    for i, s in enumerate(segs):
        dwell = s.hold_s + (travel_time(math.hypot(s.x - prev[0], s.y - prev[1]), *caps) if s.kind == "move" else 0)
        bounds.append((t, t + dwell, s))
        k = 0
        while k * 0.01 < dwell:
            now = t + k * 0.01
            row = dict(t=now, seg=i, name=s.name, kind=s.kind, cmd_x=s.x, cmd_y=s.y, cam_t=None, cam_x=None, cam_y=None)
            if k % 2 == 0:
                x, y = s.x, s.y
                since = now - t
                if s.name.startswith("near goal") and not s.name.endswith(": go"):
                    x += 1.0 * math.sin(2 * math.pi * ring_hz * now)            # sustained
                elif s.name.startswith("centre: step"):
                    x += 1.5 * math.exp(-since / 0.25) * math.sin(2 * math.pi * ring_hz * now)   # struck
                row.update(cam_t=now + rng.normal(0, 0.0005), cam_x=x + rng.normal(0, 0.05), cam_y=y + rng.normal(0, 0.05))
            rows.append(row)
            k += 1
        t += dwell
        prev = (s.x, s.y)
    drives = {}
    for m in range(4):
        dt = np.arange(0, t, 0.0101) + 0.002 * m
        amps = 1.2 * (1 if m % 2 else -1) + rng.normal(0, 0.01, len(dt))
        if m in (1, 3):
            near = np.zeros(len(dt), bool)
            for a, b, s in bounds:
                if s.name.startswith("near goal") and not s.name.endswith(": go"):
                    near |= (dt >= a) & (dt < b)
            amps = amps + near * 0.2 * np.sin(2 * math.pi * ring_hz * dt + 0.7)
        drives[m] = (dt, amps)
    return rows, drives


def test_sequence_stays_in_the_box_and_listens_at_five_places():
    segs = ring_test.ring_sequence()
    assert all(geom.WS_MIN_X <= s.x <= geom.WS_MAX_X and geom.WS_MIN_Y <= s.y <= geom.WS_MAX_Y for s in segs)
    quiet = [s for s in segs if s.name.endswith(": quiet")]
    assert len(quiet) == 5 and all(s.hold_s == ring_test.QUIET_S and s.kind == "hold" for s in quiet)
    # The robot's goal is the high-x end: three of the five are hard against it.
    assert sum(s.x > geom.WS_MAX_X - 100 for s in quiet) == 3
    steps = [s for s in segs if ": step" in s.name]
    assert len(steps) == 20 and all(s.kind == "flip" for s in steps)
    assert 25 < duration_estimate(segs, ring_test.TEST_SPEED_MM_S, ring_test.TEST_ACCEL_MM_S2) < 60


def test_tone_finds_a_ring_through_uneven_sampling_and_noise():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 2.0, 100))                 # 50 samples a second, unevenly
    x = 3.0 + 0.4 * t + 0.8 * np.sin(2 * math.pi * 21.5 * t + 1.0) + rng.normal(0, 0.05, len(t))
    tn = ring_test.tone(t, x, 0.3)
    assert abs(tn["hz"] - 21.5) <= 0.25 and abs(tn["amp"] - 0.8) < 0.1 and tn["clear"]
    flat = ring_test.tone(t, 3.0 + 0.4 * t + rng.normal(0, 0.05, len(t)), 0.3)
    assert not flat["clear"]
    assert ring_test.tone(t[:10], x[:10], 0.3) is None


def test_reading_tells_a_sustained_ring_from_a_struck_one_and_names_the_drives():
    caps = (ring_test.TEST_SPEED_MM_S, ring_test.TEST_ACCEL_MM_S2)
    segs = ring_test.ring_sequence()
    rows, drives = synthetic(segs, caps)
    result = ring_test.analyse(rows, segs, caps, drives)
    by_name = {p["name"]: p for p in result["places"]}
    near = by_name["near goal, middle"]["quiet"]
    assert near["camera_x"]["clear"] and abs(near["camera_x"]["hz"] - 21.5) <= 0.5
    assert abs(near["camera_x"]["amp"] - 1.0) < 0.15 and not near["camera_y"]["clear"]
    assert [bool(tn and tn["clear"]) for tn in near["motors"]] == [False, True, False, True]
    assert not by_name["centre"]["quiet"]["camera_x"]["clear"]
    struck = by_name["centre"]["steps"]["x"]["camera_x"]
    assert struck["clear"] and struck["late"] < 0.5 * struck["early"]
    text = "\n".join(result["reading"])
    assert "rings on its own at: near goal, middle" in text and "near goal, low y" in text
    assert "drive current oscillates with it on motors 1, 3" in text
    assert "at centre" in text and "dying away" in text
    report = ring_test.format_report(result)
    assert "near goal, middle: quiet" in report and "motor 3 A" in report

    # Without the drives' currents the reading says so instead of guessing.
    bare = ring_test.analyse(rows, segs, caps, None)
    assert "Drive currents were not recorded" in "\n".join(bare["reading"])
    # Without a camera there is nothing to read.
    blind = [dict(r, cam_t=None, cam_x=None, cam_y=None) for r in rows]
    assert ring_test.analyse(blind, segs, caps, drives)["reading"][0].startswith("No camera")


def test_drive_samples_file_is_read_on_the_tests_clock(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("start_monotonic,end_monotonic,node,torque_amps,valid\n"
                    "100.000,100.002,0,-1.2000,1\n100.002,100.004,1,1.1000,1\n100.010,100.012,0,-1.3000,0\n")
    drives = ring_test.load_drive_samples(path, monotonic_t0=99.0)
    assert set(drives) == {0, 1} and abs(drives[0][0][0] - 1.001) < 1e-9 and drives[1][1][0] == 1.1
    # An encoder file (an older master) carries no currents.
    path.write_text("start_monotonic,end_monotonic,node,encoder_counts,valid\n100.0,100.002,0,5.0,1\n")
    assert ring_test.load_drive_samples(path, 99.0) == {}


def test_finish_reads_the_drive_file_and_rewrites_the_log(tmp_path, monkeypatch):
    import json
    import types
    caps = (ring_test.TEST_SPEED_MM_S, ring_test.TEST_ACCEL_MM_S2)
    segs = ring_test.ring_sequence()
    rows, drives = synthetic(segs, caps)
    monkeypatch.setattr(ring_test, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    with open(tmp_path / "logs" / "drive.csv", "w") as f:
        f.write("start_monotonic,end_monotonic,node,torque_amps,valid\n")
        for m, (dt, amps) in drives.items():
            for t, a in zip(dt, amps):
                f.write(f"{500.0 + t - 0.001:.6f},{500.0 + t + 0.001:.6f},{m},{a:.4f},1\n")
    log_json = tmp_path / "run.json"
    test = types.SimpleNamespace(rows=rows, segments=segs, caps=caps, result=dict(
        sequence="ring", drive_samples="logs/drive.csv", monotonic_t0=500.0, stopped_early=False,
        log_csv=str(tmp_path / "run.csv"), log_json=str(log_json)))
    result = ring_test.finish(test, sleep=lambda s: None)
    assert result["drive_currents"] and "drive current oscillates with it on motors 1, 3" in result["text"]
    assert "samples:" in result["text"] and json.loads(log_json.read_text())["places"][0]["name"] == "near goal, middle"
    # The master's file never appearing costs the currents, not the test.
    test.result["drive_samples"] = "logs/missing.csv"
    late = ring_test.finish(test, wait_s=0.3, sleep=lambda s: None)
    assert not late["drive_currents"] and "Drive currents were not recorded" in late["text"]
