"""The RING TEST (2026-11-18): does the mallet oscillate, where, and who drives it.

Since 2026-11-10 the mallet shakes at 21-22 Hz, mostly along x near the robot's
goal. The tracking test cannot see it: its holds last one second at the centre
of the workspace. This is a second move list for the same loop
(`follow_test.FollowTest`), built to answer three questions:

  SELF-SUSTAINED OR RUNG?  At five places (near the goal at three y, the
      centre, forward) the mallet first HOLDS STILL for three seconds. A ring
      that is there with nothing commanded is being sustained by a loop.
      Then four 30 mm steps (x out and back, y out and back), each followed by
      1.2 s of listening: a ring that starts large and dies away is an
      underdamped resonance being struck, not sustained.
  WHERE?  The amplitude and frequency per place.
  ARE THE DRIVES IN IT?  The master samples every drive's measured CURRENT at
      ~100 Hz per motor meanwhile (`SAMPLE <s> torque`). The encoders resolve
      only 0.38 mm of cable per count, so a small shaft ring reads flat on
      them; the current shows whether a drive's loop is reacting at the ring's
      frequency. Flat current under a ringing mallet means the motion is
      beyond the motor (cable, springs, mallet); a current that oscillates
      with it means the servo loop is part of the ring and its gains matter.

Amplitudes are of the best-fitting sinusoid (zero to peak) after removing a
slow trend, scanned over frequency by least squares, so uneven sampling is
fine. The camera here is the UI's tracker at ~50 frames a second, which sees
up to ~24 Hz; a ring above that would show in the currents only.

The test applies its own caps (`TEST_SPEED_MM_S`, `TEST_ACCEL_MM_S2`) so runs
compare; the caller restores the previous ones. Logs: logs/ring_test/.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from airhockey.follow_test import MARGIN_MM, SETTLE_S, Segment, geom, travel_time

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "ring_test"

TEST_SPEED_MM_S = 3000.0
TEST_ACCEL_MM_S2 = 60000.0
QUIET_S = 2.0           # holding still, nothing commanded
GOAL_INSET_MM = 60.0    # see ring_sequence
LISTEN_S = 1.0          # after each step
STEP_MM = 30.0
QUIET_SKIP_S = 0.6      # let the arrival die out before calling a hold quiet
STEP_SKIP_S = 0.03      # after the step's own travel time

SCAN_HZ = (8.0, 40.0, 0.25)
CAMERA_RING_MM = 0.3    # a clear peak this large in a quiet hold is a ring
CURRENT_RING_A = 0.05
CLEAR_RATIO = 3.0       # peak over the scan's median


def ring_sequence(box: tuple[float, float, float, float] | None = None,
                  margin: float = MARGIN_MM) -> list[Segment]:
    """Five places, each: go there, hold still, step x out and back, step y
    out and back. The robot's goal is at the high-x end of its workspace."""
    x0, x1, y0, y1 = box or (geom.WS_MIN_X, geom.WS_MAX_X, geom.WS_MIN_Y, geom.WS_MAX_Y)
    x0, x1, y0, y1 = x0 + margin, x1 - margin, y0 + margin, y1 - margin
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    side = 0.3 * (y1 - y0)
    # Holding hard against the goal end costs motor 2 about 5.7 A at 2 mm of
    # pretension, above its 4 A RMS limit: the first run (2026-11-18) dwelt
    # there 25 s and tripped RMSOverloadShutdown at 41 s. So the near-goal
    # places sit GOAL_INSET_MM in, and cooler places come between them.
    x1 = max(cx, x1 - GOAL_INSET_MM)
    places = [("near goal, middle", x1, cy), ("centre", cx, cy), ("near goal, low y", x1, cy - side),
              ("forward, middle", x0, cy), ("near goal, high y", x1, cy + side)]
    segs: list[Segment] = []
    for name, x, y in places:
        # Steps go toward the middle of the box, so they never leave it.
        dx = -STEP_MM if x >= cx else STEP_MM
        dy = -STEP_MM if y > cy else STEP_MM
        segs.append(Segment(f"{name}: go", x, y, SETTLE_S, "move"))
        segs.append(Segment(f"{name}: quiet", x, y, QUIET_S, "hold"))
        segs.append(Segment(f"{name}: step x", x + dx, y, LISTEN_S, "flip"))
        segs.append(Segment(f"{name}: step x back", x, y, LISTEN_S, "flip"))
        segs.append(Segment(f"{name}: step y", x, y + dy, LISTEN_S, "flip"))
        segs.append(Segment(f"{name}: step y back", x, y, LISTEN_S, "flip"))
    segs.append(Segment("centre: go", cx, cy, SETTLE_S, "move"))
    return segs


# ── analysis ────────────────────────────────────────────────────────


def tone_scan(t, x, freqs) -> np.ndarray:
    """Amplitude (zero to peak) of the best-fitting sinusoid at each
    frequency, with a constant, a slope and a curve fitted alongside."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    t = t - t.mean()
    base = np.column_stack([np.ones_like(t), t, t * t])
    out = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        w = 2.0 * math.pi * f * t
        coef, *_ = np.linalg.lstsq(np.column_stack([base, np.sin(w), np.cos(w)]), x, rcond=None)
        out[i] = math.hypot(coef[3], coef[4])
    return out


def tone(t, x, min_amp: float) -> dict | None:
    """The strongest tone in a window: {hz, amp, floor, clear, early, late}.

    `early` / `late` are the amplitude at that frequency in the first and
    second half of the window: a struck resonance decays, a sustained one
    does not. None when the window is too short or too sparse to say.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    keep = np.isfinite(t) & np.isfinite(x)
    t, x = t[keep], x[keep]
    if len(t) < 20 or t[-1] - t[0] < 0.5:
        return None
    rate = (len(t) - 1) / (t[-1] - t[0])
    top = min(SCAN_HZ[1], 0.48 * rate)
    if top <= SCAN_HZ[0] + 2.0:
        return None
    freqs = np.arange(SCAN_HZ[0], top + 1e-9, SCAN_HZ[2])
    amps = tone_scan(t, x, freqs)
    k = int(np.argmax(amps))
    floor = float(np.median(amps))
    hz = float(freqs[k])
    mid = 0.5 * (t[0] + t[-1])
    halves = []
    for part in (t <= mid, t > mid):
        halves.append(float(tone_scan(t[part], x[part], [hz])[0]) if part.sum() >= 10 else None)
    # A struck ring is large at first and gone by the end, so its size is
    # judged on the first half; it must still stand out over the whole window.
    strongest = max(float(amps[k]), halves[0] or 0.0)
    return dict(hz=round(hz, 2), amp=round(float(amps[k]), 4), floor=round(floor, 4),
                clear=bool(strongest >= min_amp and amps[k] >= CLEAR_RATIO * max(floor, 1e-9)),
                early=None if halves[0] is None else round(halves[0], 4),
                late=None if halves[1] is None else round(halves[1], 4),
                sample_hz=round(float(rate), 1), top_hz=round(float(top), 1))


def load_drive_samples(path: Path, monotonic_t0: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{motor: (seconds since the test began, amps)} from the master's CSV."""
    out: dict[int, tuple[list, list]] = {m: ([], []) for m in range(4)}
    with open(path) as f:
        reader = csv.DictReader(f)
        column = "torque_amps" if "torque_amps" in (reader.fieldnames or []) else None
        if column is None:
            return {}
        for row in reader:
            if row["valid"] != "1":
                continue
            m = int(row["node"])
            out[m][0].append(0.5 * (float(row["start_monotonic"]) + float(row["end_monotonic"])) - monotonic_t0)
            out[m][1].append(float(row[column]))
    return {m: (np.asarray(t), np.asarray(v)) for m, (t, v) in out.items() if len(t)}


def analyse(rows: list[dict], segs: list[Segment], caps: tuple[float, float],
            drives: dict[int, tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """Per place: the tone in the quiet hold and after each step, for the
    camera's x and y and each drive's current; then a plain reading."""
    drives = drives or {}
    t = np.array([r["t"] for r in rows], float)
    seg = np.array([r["seg"] for r in rows], int)
    cam = [(r["cam_t"], r["cam_x"], r["cam_y"]) for r in rows if r.get("cam_t") is not None]
    cam_t, cam_x, cam_y = (np.array(c, float) for c in zip(*cam)) if cam else (np.zeros(0),) * 3

    def window(i: int) -> tuple[float, float] | None:
        inside = t[seg == i]
        if not len(inside):
            return None
        s = segs[i]
        skip = QUIET_SKIP_S if s.kind == "hold" else STEP_SKIP_S + travel_time(STEP_MM, *caps)
        return float(inside[0]) + skip, float(inside[-1])

    def listen(i: int) -> dict | None:
        span = window(i)
        if span is None:
            return None
        a, b = span
        pick = (cam_t >= a) & (cam_t <= b)
        out = dict(camera_x=tone(cam_t[pick], cam_x[pick], CAMERA_RING_MM),
                   camera_y=tone(cam_t[pick], cam_y[pick], CAMERA_RING_MM), motors=[])
        for m in range(4):
            if m in drives:
                dt, dv = drives[m]
                keep = (dt >= a) & (dt <= b)
                out["motors"].append(tone(dt[keep], dv[keep], CURRENT_RING_A))
            else:
                out["motors"].append(None)
        return out

    places: dict[str, dict] = {}
    for i, s in enumerate(segs):
        place, _, what = s.name.partition(": ")
        if what == "go" or not what:
            continue
        entry = places.setdefault(place, dict(name=place, x=None, y=None, quiet=None, steps={}))
        if what == "quiet":
            entry.update(x=round(s.x, 1), y=round(s.y, 1), quiet=listen(i))
        else:
            entry["steps"][what.removeprefix("step ")] = listen(i)
    result = dict(sequence="ring", places=list(places.values()),
                  caps=dict(speed_mm_s=caps[0], accel_mm_s2=caps[1]),
                  camera=len(cam_t) > 0, drive_currents=bool(drives))
    result["reading"] = reading(result)
    return result


def _strongest(listened: dict | None) -> tuple[str, dict] | None:
    """The clear camera tone with the larger amplitude, as (axis, tone)."""
    if not listened:
        return None
    clear = [(axis, listened[f"camera_{axis}"]) for axis in ("x", "y")
             if listened.get(f"camera_{axis}") and listened[f"camera_{axis}"]["clear"]]
    return max(clear, key=lambda item: item[1]["amp"]) if clear else None


def _motors_with(listened: dict | None, hz: float) -> list[int]:
    """Motors whose current has a clear tone within 1.5 Hz of `hz`."""
    return [m for m, tn in enumerate((listened or {}).get("motors") or [])
            if tn and tn["clear"] and abs(tn["hz"] - hz) <= 1.5]


def reading(result: dict) -> list[str]:
    """What the numbers say, in sentences."""
    lines: list[str] = []
    if not result["camera"]:
        return ["No camera: turn the camera on (Views) and run again; the mallet's motion was not measured."]
    sustained, struck, quiet_places = [], [], []
    for place in result["places"]:
        ring = _strongest(place["quiet"])
        if ring:
            axis, tn = ring
            sustained.append((place, axis, tn))
            continue
        quiet_places.append(place["name"])
        best = None
        for name, listened in place["steps"].items():
            found = _strongest(listened)
            if found and (best is None or (found[1]["early"] or found[1]["amp"]) > (best[2]["early"] or best[2]["amp"])):
                best = (name, found[0], found[1], listened)
        if best:
            struck.append((place, *best))
    if sustained:
        lines.append("HOLDING STILL, the mallet rings on its own at: " + "; ".join(
            f"{p['name']} ({tn['amp']:.2f} mm along {axis} at {tn['hz']:.1f} Hz)" for p, axis, tn in sustained)
            + ". Nothing was commanded, so a loop is sustaining it.")
        for p, axis, tn in sustained:
            motors = _motors_with(p["quiet"], tn["hz"])
            if result["drive_currents"]:
                lines.append(f"  {p['name']}: " + (
                    "drive current oscillates with it on motor" + ("s " if len(motors) > 1 else " ")
                    + ", ".join(map(str, motors)) + " -- the servo loop is part of the ring; its gains matter."
                    if motors else
                    "no drive's current oscillates at that frequency -- the drives are not reacting; "
                    "the motion is beyond the motors (cable, springs, mallet)."))
    else:
        lines.append("Holding still, the mallet does not ring at any of the five places "
                     f"(no clear tone above {CAMERA_RING_MM} mm).")
    for p, name, axis, tn, listened in struck:
        early, late = tn.get("early"), tn.get("late")
        fate = ""
        if early and late is not None:
            fate = (", dying away" if late < 0.5 * early else ", NOT dying away within the 1.2 s"
                    if late > 0.8 * early else ", fading slowly")
        motors = _motors_with(listened, tn["hz"])
        lines.append(f"After a step ({name}) at {p['name']}: {max(tn['amp'], early or 0):.2f} mm along {axis} at "
                     f"{tn['hz']:.1f} Hz{fate}"
                     + (f"; current follows on motor{'s' if len(motors) > 1 else ''} "
                        + ", ".join(map(str, motors)) if motors else
                        "; no drive current at that frequency" if result["drive_currents"] else "") + ".")
    if not sustained and not struck:
        lines.append("No clear ring after the steps either. If the mallet visibly shook, its frequency "
                     "may be above what this camera view resolves (~24 Hz); check the drive currents below.")
    if not result["drive_currents"]:
        lines.append("Drive currents were not recorded (the motor controller is older than this test; "
                     "shut it down on the Robot tab and run again).")
    return lines


def format_report(result: dict) -> str:
    """The reading, then one row per place and window."""
    def cell(tn, unit):
        if tn is None:
            return "       -      "
        mark = "*" if tn["clear"] else " "
        return f"{tn['amp']:5.2f}{unit}@{tn['hz']:4.1f}{mark}"
    lines = list(result["reading"])
    lines.append("")
    lines.append(f"caps {result['caps']['speed_mm_s'] / 1000:g} m/s, {result['caps']['accel_mm_s2'] / 1000:g} m/s^2. "
                 "Largest tone per window, amplitude @ Hz; * = clear of the noise.")
    lines.append(f"{'place / window':30} {'mallet x':>12} {'mallet y':>12}  " +
                 " ".join(f"{'motor ' + str(m) + ' A':>12}" for m in range(4)))
    for place in result["places"]:
        windows = [("quiet", place["quiet"])] + [(f"step {k}", v) for k, v in place["steps"].items()]
        for name, listened in windows:
            if listened is None:
                continue
            label = f"{place['name']}: {name}" if name == "quiet" else f"   {name}"
            lines.append(f"{label:30} {cell(listened['camera_x'], 'mm')} {cell(listened['camera_y'], 'mm')}  "
                         + " ".join(cell(tn, 'A ') for tn in listened["motors"]))
    return "\n".join(lines)


# ── running it from the UI ──────────────────────────────────────────


def quiet_summary(rows, segs, camera=True) -> dict:
    """FollowTest's summarizer slot: the ring analysis needs the drive
    samples, which the master only writes after the run, so it happens in
    `finish`."""
    return {"sequence": "ring"}


def finish(test, wait_s: float = 6.0, sleep=None) -> dict:
    """Analyse a finished FollowTest run of `ring_sequence`: wait for the
    master's drive-current file, read it, and rewrite the run's JSON log with
    the result. Returns the result with `text` (the report) added."""
    import json
    import time
    sleep = sleep or time.sleep
    summary = dict(test.result or {})
    drives = {}
    path = summary.get("drive_samples")
    if path:
        full = ROOT / path
        for _ in range(int(wait_s / 0.1)):      # sampling outlasts the run by ~2 s
            if full.is_file():
                break
            sleep(0.1)
        try:
            drives = load_drive_samples(full, float(summary.get("monotonic_t0", 0.0)))
        except (OSError, ValueError, KeyError):
            drives = {}
    result = analyse(test.rows, test.segments, test.caps or (TEST_SPEED_MM_S, TEST_ACCEL_MM_S2), drives)
    result.update({k: summary[k] for k in ("log_csv", "log_json", "drive_samples", "stopped_early",
                                           "monotonic_t0") if k in summary})
    result["text"] = format_report(result)
    if summary.get("stopped_early"):
        result["text"] = "STOPPED EARLY: only the windows that ran are below.\n" + result["text"]
    if result.get("log_csv"):
        result["text"] += f"\n\nsamples: {result['log_csv']}" + (
            f"\ndrive currents: {result['drive_samples']}" if result.get("drive_samples") else "")
    if result.get("log_json"):
        try:
            with open(result["log_json"], "w") as f:
                json.dump(result, f, indent=1)
        except OSError:
            pass
    return result
