"""A fixed sequence of moves, and where three witnesses say the paddle went.

THE POINT
    Whether the drives keep up with the step stream has so far been argued
    from play logs, where every number is confounded by what the policy
    happened to ask for. This runs the SAME moves every time, at whatever
    caps are currently applied, and compares three accounts of the paddle's
    position throughout:

        Teensy   step counts through the cable model -- what was commanded
        drives   the ClearPaths' own encoders -- what the motors actually did
        camera   the mallet's marker cluster -- where the paddle actually is

    Teensy vs drives isolates the motor (following error, a wrong step
    scale, a slipped cable). Teensy vs camera is the whole chain, and it is
    the number a policy trained on the sim's body cares about, because the
    sim's paddle IS the Teensy's.

THE SEQUENCE
    Holds, single-axis moves of growing size, the four corners, then two
    passages in the policy's own style: full-height flips every 300 ms and
    +-80 mm twitches every 100 ms. The move dwells scale with the caps so
    a gentle run is not scored on moves that had no time to finish; the
    flip dwells deliberately do not, because the point of them is what the
    machine does when asked for more than it can deliver.

READING THE VERDICT
    close     camera and Teensy agree within CLOSE_MOVING_P90_MM at speed
              (after aligning the camera's latency) and within
              CLOSE_REST_P50_MM at rest.
    lagging   they agree at rest but not at speed: the drives fall behind
              the step stream and catch up when it stops. Lower the accel
              cap, or expect the paddle to be late.
    lost      they disagree even at rest. Either the cable model is wrong
              at these positions or the drives lost position (a slipped
              cable, a step scale mismatch -- the encoder rows say which).

    The camera's stamp is taken when the frame arrives, not mid-exposure,
    so it lags the paddle by the transfer time; `lag_ms` is the shift that
    best aligns the two and absorbs that. The search is BOUNDED at
    LAG_SEARCH_S: more than a frame takes to arrive, less than any drive
    lag that matters. A drive 50 ms late is 150 mm behind at 3 m/s, and a
    fit that could absorb it would score that as close -- the first draft
    did exactly that. Arrival lag per move is reported as a second, direct
    measure that no alignment touches.

NO SIM DEPENDENCY
    The routine takes callables and a clock, so the unit tests drive it
    against the firmware's own profile body in virtual time, and the UI
    drives it against cdpr_master and the vision service in real time.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

from airhockey.hardware import counts_to_cable_mm  # noqa: E402

# ── sequence ────────────────────────────────────────────────────────

MARGIN_MM = 25.0        # every target stays this far inside the box
SETTLE_S = 0.4          # added to each move's estimated travel time
FLIP_HOLD_S = 0.30      # full-height flips, the policy's shot style
TWITCH_HOLD_S = 0.10    # short flips, the policy's idle style
TWITCH_MM = 80.0
RAMP_S = 0.003          # the firmware's jerk ramp, for the travel estimate

# ── sampling ────────────────────────────────────────────────────────

SAMPLE_HZ = 100.0
ENC_EVERY_S = 0.25      # an ENC costs ~39 ms on the rig (four sFoundation reads),
                        # during which nothing else is sampled; 0.10 dragged
                        # the loop to 71 Hz (2026-09-06)

# ── scoring ─────────────────────────────────────────────────────────

LAG_SEARCH_S = 0.03     # camera latency search window -- see the docstring
LAG_STEP_S = 0.005
REST_SPEED_MM_S = 50.0  # below this the controller counts as at rest
ARRIVE_MM = 15.0        # "arrived" = within this of the target
CLOSE_MOVING_P90_MM = 30.0
CLOSE_REST_P50_MM = 10.0
CLOSE_ARRIVE_LAG_MS = 60.0   # camera latency + one camera frame, with room
LOST_REST_P50_MM = 25.0
ENC_FIT_MIN_TRAVEL_MM = 20.0
ENC_GAIN_TOL = 0.05     # |gain| this far from 1 = step scale mismatch
ENC_REST_MM = 5.0       # drive vs steps at rest beyond this = lost position
# The encoders are read in their own ~40 ms round trip AFTER the step
# counts, one node after another, so each motor's reading is some tens of
# ms staler than the steps it is compared with -- 60-120 mm of apparent
# lag at 3 m/s of cable that the drive never had (2026-09-06: it read as
# motors 2 and 3 lagging 100 mm). The fit searches a per-motor time shift
# in this window and scores the residual AFTER it.
ENC_LAG_SEARCH_S = 0.08
ENC_LAG_STEP_S = 0.002
ARRIVE_TAIL = 0.4       # arrival is judged on the last 40% of a move's dwell


@dataclass(frozen=True)
class Segment:
    name: str
    x: float
    y: float
    hold_s: float       # dwell; moves add their travel time on top
    kind: str           # "hold" | "move" | "flip"


def sequence(box: tuple[float, float, float, float] | None = None,
             margin: float = MARGIN_MM) -> list[Segment]:
    """The fixed move list, inside `box` = (min_x, max_x, min_y, max_y) mm.

    Defaults to the machine's workspace. Targets are clamped into the box
    less the margin, so a smaller box shortens the moves rather than
    sending the paddle somewhere it cannot go.
    """
    x0, x1, y0, y1 = box or (geom.WS_MIN_X, geom.WS_MAX_X,
                             geom.WS_MIN_Y, geom.WS_MAX_Y)
    x0, x1, y0, y1 = x0 + margin, x1 - margin, y0 + margin, y1 - margin
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    segs: list[Segment] = []

    def clamp(x, y):
        return min(max(x, x0), x1), min(max(y, y0), y1)

    def hold(name, x, y, s):
        segs.append(Segment(name, *clamp(x, y), s, "hold"))

    def move(name, x, y):
        segs.append(Segment(name, *clamp(x, y), SETTLE_S, "move"))

    def flip(name, x, y, s):
        segs.append(Segment(name, *clamp(x, y), s, "flip"))

    hold("start", cx, cy, 1.0)
    move("x +100", cx + 100, cy)
    move("centre", cx, cy)
    move("y +100", cx, cy + 100)
    move("centre", cx, cy)
    move("x +250", cx + 250, cy)
    move("x -250", cx - 250, cy)
    move("centre", cx, cy)
    move("y +250", cx, cy + 250)
    move("y -250", cx, cy - 250)
    move("centre", cx, cy)
    move("corner --", x0, y0)
    move("corner ++", x1, y1)
    move("corner -+", x0, y1)
    move("corner +-", x1, y0)
    move("centre", cx, cy)
    hold("rest", cx, cy, 1.0)
    for i in range(6):
        flip("flip y", cx, y1 if i % 2 == 0 else y0, FLIP_HOLD_S)
    move("centre", cx, cy)
    hold("rest", cx, cy, 1.0)
    for i in range(10):
        flip("twitch", cx, cy + (TWITCH_MM if i % 2 == 0 else -TWITCH_MM),
             TWITCH_HOLD_S)
    move("centre", cx, cy)
    hold("end", cx, cy, 1.5)
    return segs


def travel_time(dist_mm: float, v_max: float, a_max: float,
                ramp_s: float = RAMP_S) -> float:
    """Time for the profile to cover `dist_mm` from rest to rest."""
    if dist_mm <= 0 or v_max <= 0 or a_max <= 0:
        return 0.0
    if dist_mm < v_max * v_max / a_max:            # never reaches the cap
        return 2.0 * math.sqrt(dist_mm / a_max) + 2.0 * ramp_s
    return dist_mm / v_max + v_max / a_max + 2.0 * ramp_s


def duration_estimate(segs: list[Segment], v_max: float, a_max: float,
                      start: tuple[float, float] | None = None) -> float:
    """How long the run will take at these caps (what the UI shows)."""
    prev = start or (segs[0].x, segs[0].y)
    total = 0.0
    for s in segs:
        total += s.hold_s
        if s.kind == "move":
            total += travel_time(math.hypot(s.x - prev[0], s.y - prev[1]),
                                 v_max, a_max)
        prev = (s.x, s.y)
    return total


# ── the run ─────────────────────────────────────────────────────────

ROW_FIELDS = (
    "t", "seg", "name", "kind", "cmd_x", "cmd_y",
    "ctl_x", "ctl_y", "ctl_vx", "ctl_vy", "c0", "c1", "c2", "c3",
    "cam_t", "cam_x", "cam_y",
    "enc0", "enc1", "enc2", "enc3", "trq0", "trq1", "trq2", "trq3", "enc_ms",
    "enc_t",
)


def _default_log_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "follow_test"


class FollowTest:
    """Run the sequence against a rig, sampling as it goes.

    client   has command_position(x, y, speed), get_status(), get_encoders()
             -- the CDPRClient interface. Owned EXCLUSIVELY while running:
             the socket is one reply per command, so nothing else may talk
             to it until `running` is False.
    camera   callable -> (t, x_mm, y_mm) or None, stamped on `clock`.
             None when there is no camera; the verdict then rests on the
             encoders alone and says so.
    on_status  called with every STATUS dict, so the UI can keep drawing
             the controller's belief without a second socket.
    clock / sleep  injectable for virtual-time tests.
    """

    def __init__(self, client, camera=None, on_status=None,
                 log_dir: Path | str | None = None,
                 clock=time.time, sleep=time.sleep,
                 sample_hz: float = SAMPLE_HZ,
                 box: tuple[float, float, float, float] | None = None,
                 segments: list[Segment] | None = None):
        self.client = client
        self.camera = camera
        self.on_status = on_status
        self.log_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
        self.clock = clock
        self.sleep = sleep
        self.sample_hz = sample_hz
        self.segments = segments if segments is not None else sequence(box)
        self.rows: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.progress: dict = {"segment": "", "i": 0, "n": len(self.segments),
                               "elapsed_s": 0.0, "gap_mm": None,
                               "estimate_s": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.caps: tuple[float, float] | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._guarded, daemon=True)
        self._thread.start()

    def stop(self, join_s: float | None = None) -> None:
        self._stop.set()
        if join_s is not None and self._thread is not None:
            self._thread.join(join_s)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _guarded(self) -> None:
        try:
            self.run()
        except Exception as e:                       # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    # ── the loop ─────────────────────────────────────────────────────

    def run(self) -> dict:
        s = self.client.get_status()
        if self.on_status:
            self.on_status(s)
        v_max = float(s.get("speed_limit") or 0.0)
        a_max = float(s.get("accel_limit") or 0.0)
        if v_max <= 0 or a_max <= 0:
            raise RuntimeError("the controller reports no motion caps; "
                               "apply limits first")
        self.caps = (v_max, a_max)
        segs = self.segments
        prev = (s["x"], s["y"])
        self.progress["estimate_s"] = duration_estimate(segs, v_max, a_max, prev)
        c_zero = [s["c0"], s["c1"], s["c2"], s["c3"]]
        enc_zero = None
        period = 1.0 / self.sample_hz
        t0 = self.clock()
        next_t = t0
        seg_i = -1
        seg_end = t0
        seg: Segment | None = None
        last_cam_t = None
        last_enc_t = -math.inf
        rows = self.rows
        try:
            while not self._stop.is_set():
                t = self.clock()
                if t >= seg_end:
                    seg_i += 1
                    if seg_i >= len(segs):
                        break
                    seg = segs[seg_i]
                    self.client.command_position(seg.x, seg.y, v_max)
                    dwell = seg.hold_s
                    if seg.kind == "move":
                        dwell += travel_time(math.hypot(seg.x - prev[0],
                                                        seg.y - prev[1]),
                                             v_max, a_max)
                    seg_end = t + dwell
                    prev = (seg.x, seg.y)
                    self.progress.update(segment=seg.name, i=seg_i + 1)

                s = self.client.get_status()
                if self.on_status:
                    self.on_status(s)
                row = {
                    "t": t - t0, "seg": seg_i, "name": seg.name,
                    "kind": seg.kind, "cmd_x": seg.x, "cmd_y": seg.y,
                    "ctl_x": s["x"], "ctl_y": s["y"],
                    "ctl_vx": s["vx"], "ctl_vy": s["vy"],
                    "c0": s["c0"] - c_zero[0], "c1": s["c1"] - c_zero[1],
                    "c2": s["c2"] - c_zero[2], "c3": s["c3"] - c_zero[3],
                    "cam_t": None, "cam_x": None, "cam_y": None,
                    "enc_ms": None, "enc_t": None,
                }
                for m in range(4):
                    row[f"enc{m}"] = None
                    row[f"trq{m}"] = None
                cam = self.camera() if self.camera else None
                if cam is not None and cam[0] != last_cam_t:
                    last_cam_t = cam[0]
                    row["cam_t"] = cam[0] - t0
                    row["cam_x"] = cam[1]
                    row["cam_y"] = cam[2]
                    self.progress["gap_mm"] = round(
                        math.hypot(cam[1] - s["x"], cam[2] - s["y"]), 1)
                if t - last_enc_t >= ENC_EVERY_S:
                    last_enc_t = t
                    e0 = self.clock()
                    try:
                        e = self.client.get_encoders()
                    except Exception:                # noqa: BLE001
                        e = None
                    if e is not None:
                        e1 = self.clock()
                        row["enc_ms"] = round(1000.0 * (e1 - e0), 2)
                        row["enc_t"] = 0.5 * (e0 + e1) - t0   # the read's midpoint
                        if enc_zero is None:
                            enc_zero = list(e["posn"])
                        for m in range(4):
                            res = e["res"][m]
                            if res:
                                revs = (e["posn"][m] - enc_zero[m]) / res
                                row[f"enc{m}"] = revs * geom.SPOOL_CIRCUMFERENCE_MM
                            row[f"trq{m}"] = e["trq"][m]
                rows.append(row)
                self.progress["elapsed_s"] = round(t - t0, 1)

                next_t += period
                delay = next_t - self.clock()
                if delay > 0:
                    self.sleep(delay)
                else:
                    next_t = self.clock()    # behind: do not burst to catch up
        finally:
            # Leave the paddle somewhere known, whether we finished or were
            # stopped mid-flip. The last segment is the centre anyway.
            end = segs[-1]
            try:
                self.client.command_position(end.x, end.y, v_max)
            except Exception:                        # noqa: BLE001
                pass

        summary = summarize(rows, segs, camera=self.camera is not None)
        summary["caps"] = {"speed_mm_s": v_max, "accel_mm_s2": a_max}
        summary["stopped_early"] = self._stop.is_set()
        summary["estimate_s"] = round(self.progress["estimate_s"], 1)
        summary.update(self._write_logs(rows, summary))
        self.result = summary
        return summary

    def _write_logs(self, rows, summary) -> dict:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            csv_path = self.log_dir / f"{stamp}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in ROW_FIELDS})
            json_path = self.log_dir / f"{stamp}.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=1)
            return {"log_csv": str(csv_path), "log_json": str(json_path)}
        except OSError as e:
            return {"log_error": str(e)}


# ── scoring ─────────────────────────────────────────────────────────

def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def summarize(rows: list[dict], segs: list[Segment], camera: bool = True) -> dict:
    """Stats and a verdict from the sampled rows. Pure; no hardware."""
    out: dict = {"samples": len(rows), "notes": []}
    if len(rows) < 5:
        out.update(verdict="no data", duration_s=0.0)
        return out
    t = np.array([r["t"] for r in rows], dtype=float)
    cx = np.array([r["ctl_x"] for r in rows], dtype=float)
    cy = np.array([r["ctl_y"] for r in rows], dtype=float)
    spd = np.hypot([r["ctl_vx"] for r in rows], [r["ctl_vy"] for r in rows])
    seg_idx = np.array([r["seg"] for r in rows])
    out["duration_s"] = _r(t[-1] - t[0])
    out["sample_hz"] = _r(len(rows) / max(t[-1] - t[0], 1e-6))
    out["ctl_peak_speed_mm_s"] = _r(spd.max(), 0)
    moving = spd > REST_SPEED_MM_S

    # Settled: at rest AND past the midpoint of its segment's dwell, so a
    # paddle that has only just stopped is not scored as "at rest" while
    # the drives are still catching up.
    seg_start = {}
    seg_end = {}
    for i, r in enumerate(rows):
        seg_start.setdefault(r["seg"], t[i])
        seg_end[r["seg"]] = t[i]
    settled = np.array([
        (not moving[i]) and t[i] >= 0.5 * (seg_start[r["seg"]] + seg_end[r["seg"]])
        for i, r in enumerate(rows)])

    # ── camera vs Teensy ────────────────────────────────────────────
    cam_rows = [r for r in rows if r["cam_t"] is not None]
    out["camera_samples"] = len(cam_rows)
    if camera and len(cam_rows) >= 5:
        ct = np.array([r["cam_t"] for r in cam_rows], dtype=float)
        cxm = np.array([r["cam_x"] for r in cam_rows], dtype=float)
        cym = np.array([r["cam_y"] for r in cam_rows], dtype=float)
        out["camera_hz"] = _r(len(cam_rows) / max(ct[-1] - ct[0], 1e-6))
        cam_moving = np.interp(ct, t, spd) > REST_SPEED_MM_S
        cam_settled = np.interp(ct, t, settled.astype(float)) > 0.5

        def gap_at(lag):
            gx = np.interp(ct - lag, t, cx) - cxm
            gy = np.interp(ct - lag, t, cy) - cym
            return np.hypot(gx, gy)

        raw = gap_at(0.0)
        best_lag, best_rms = 0.0, math.inf
        if cam_moving.sum() >= 3:
            for lag in np.arange(0.0, LAG_SEARCH_S + 1e-9, LAG_STEP_S):
                g = gap_at(lag)[cam_moving]
                rms = float(np.sqrt(np.mean(g * g)))
                if rms < best_rms:
                    best_lag, best_rms = float(lag), rms
        aligned = gap_at(best_lag)
        out["lag_ms"] = _r(1000.0 * best_lag, 0)
        gm = aligned[cam_moving]
        out["gap_moving_p50_mm"] = _r(_pct(gm, 50))
        out["gap_moving_p90_mm"] = _r(_pct(gm, 90))
        out["gap_moving_max_mm"] = _r(gm.max() if len(gm) else None)
        out["gap_raw_moving_p90_mm"] = _r(_pct(raw[cam_moving], 90))
        gr = raw[cam_settled]
        out["gap_rest_p50_mm"] = _r(_pct(gr, 50))
        out["gap_rest_max_mm"] = _r(gr.max() if len(gr) else None)
        # camera speed, from consecutive samples
        dct = np.diff(ct)
        ok = dct > 1e-4
        cam_v = np.hypot(np.diff(cxm), np.diff(cym))[ok] / dct[ok]
        out["cam_peak_speed_mm_s"] = _r(cam_v.max() if len(cam_v) else None, 0)
    else:
        out["camera_hz"] = None
        if camera:
            out["notes"].append("camera gave too few samples to score")

    # ── per segment ─────────────────────────────────────────────────
    per_seg = []
    for si, seg in enumerate(segs):
        m = seg_idx == si
        if not m.any():
            continue
        entry = {"name": seg.name, "kind": seg.kind,
                 "ctl_peak_mm_s": _r(spd[m].max(), 0)}
        ts = t[m]
        d_ctl = np.hypot(cx[m] - seg.x, cy[m] - seg.y)
        hit = np.nonzero(d_ctl < ARRIVE_MM)[0]
        t_ctl = ts[hit[0]] if len(hit) else None
        entry["ctl_arrive_s"] = _r(t_ctl - ts[0], 3) if t_ctl is not None else None
        if camera and len(cam_rows) >= 5:
            cm = (ct >= ts[0]) & (ct <= ts[-1])
            if cm.any():
                entry["gap_max_mm"] = _r(aligned[cm].max())
                d_cam = np.hypot(cxm[cm] - seg.x, cym[cm] - seg.y)
                hitc = np.nonzero(d_cam < ARRIVE_MM)[0]
                t_cam = ct[cm][hitc[0]] if len(hitc) else None
                # Arrival needs the camera to have SEEN the end of the move:
                # a tracker that lost the marker cluster near a rail has
                # no opinion, and must not read as a drive that never got
                # there (2026-09-06: four such moves scored a good run as
                # lagging).
                tail = ct[cm] >= ts[-1] - ARRIVE_TAIL * (ts[-1] - ts[0])
                if t_ctl is not None and t_cam is not None:
                    entry["arrive_lag_ms"] = _r(1000.0 * (t_cam - t_ctl), 0)
                elif t_ctl is not None and seg.kind == "move":
                    entry["arrive_lag_ms"] = None
                    if tail.any():
                        entry["cam_never_arrived"] = True
                    else:
                        entry["cam_lost"] = True
            elif seg.kind == "move":
                entry["cam_lost"] = True
        per_seg.append(entry)
    out["segments"] = per_seg
    lags = [e["arrive_lag_ms"] for e in per_seg
            if e.get("kind") == "move" and e.get("arrive_lag_ms") is not None]
    out["arrive_lag_p50_ms"] = _r(_pct(lags, 50), 0)
    out["arrive_lag_max_ms"] = _r(max(lags), 0) if lags else None
    out["moves_cam_never_arrived"] = sum(
        1 for e in per_seg if e.get("cam_never_arrived"))
    lost_moves = [e["name"] for e in per_seg if e.get("cam_lost")]
    out["moves_cam_lost"] = len(lost_moves)
    if lost_moves:
        out["notes"].append("camera lost the paddle on " + ", ".join(lost_moves)
                            + " -- not scored for arrival")

    # ── drives vs Teensy ────────────────────────────────────────────
    enc = []
    enc_rows = [(i, r) for i, r in enumerate(rows) if r.get("enc0") is not None
                or r.get("enc1") is not None]
    # Step counts as a time series per motor, to be read at the ENCODER's
    # own read time rather than the row's.
    steps_t = t
    for m in range(4):
        pairs = [(i, r[f"enc{m}"]) for i, r in enc_rows if r.get(f"enc{m}") is not None]
        e = {"motor": m, "gain": None, "lag_ms": None, "rest_mm": None,
             "moving_mm": None, "trq_peak_pct": None}
        trq = [r[f"trq{m}"] for _, r in enc_rows if r.get(f"trq{m}") is not None]
        if trq:
            e["trq_peak_pct"] = _r(max(abs(v) for v in trq))
        if len(pairs) >= 5:
            idx = np.array([i for i, _ in pairs])
            enc_mm = np.array([v for _, v in pairs], dtype=float)
            step_series = np.array([counts_to_cable_mm(r[f"c{m}"]) for r in rows])
            # When the read was: logged from 2026-09-06; older logs get the
            # row time plus half the round trip.
            t_enc = np.array([
                rows[i]["enc_t"] if rows[i].get("enc_t") is not None
                else t[i] + 0.002 + 0.5 * (rows[i].get("enc_ms") or 0.0) / 1000.0
                for i in idx], dtype=float)
            if step_series.max() - step_series.min() >= ENC_FIT_MIN_TRAVEL_MM:
                best = None
                for lag in np.arange(0.0, ENC_LAG_SEARCH_S + 1e-9, ENC_LAG_STEP_S):
                    st = np.interp(t_enc - lag, steps_t, step_series)
                    gain, offset = np.polyfit(st, enc_mm, 1)
                    resid = enc_mm - (gain * st + offset)
                    rms = float(np.sqrt(np.mean(resid ** 2)))
                    if best is None or rms < best[0]:
                        best = (rms, lag, gain, np.abs(resid))
                _rms, lag, gain, resid = best
                e["gain"] = _r(gain, 3)
                e["lag_ms"] = _r(1000.0 * lag, 0)
                rest_m = settled[idx]
                e["rest_mm"] = _r(resid[rest_m].max()) if rest_m.any() else None
                mov_m = moving[idx]
                e["moving_mm"] = _r(_pct(resid[mov_m], 90)) if mov_m.any() else None
                if abs(abs(gain) - 1.0) > ENC_GAIN_TOL:
                    out["notes"].append(
                        f"motor {m}: encoder moves {abs(gain):.3f} mm per mm "
                        f"of steps (expected 1.000) -- step scale mismatch")
                if e["rest_mm"] is not None and e["rest_mm"] > ENC_REST_MM:
                    out["notes"].append(
                        f"motor {m}: drive {e['rest_mm']:.1f} mm from its "
                        f"steps at rest -- position lost")
        enc.append(e)
    out["encoders"] = enc
    enc_ms = [r["enc_ms"] for r in rows if r.get("enc_ms") is not None]
    out["enc_read_ms"] = _r(_pct(enc_ms, 50), 2)

    # ── verdict ─────────────────────────────────────────────────────
    if not camera or out.get("gap_rest_p50_mm") is None:
        drives_ok = all(e["rest_mm"] is None or e["rest_mm"] <= ENC_REST_MM
                        for e in enc)
        out["verdict"] = ("no camera: drives held their steps" if drives_ok
                          else "no camera: drives lost position")
    elif out["gap_rest_p50_mm"] > LOST_REST_P50_MM:
        out["verdict"] = "lost"
    elif (out["gap_moving_p90_mm"] or 0.0) > CLOSE_MOVING_P90_MM \
            or out["gap_rest_p50_mm"] > CLOSE_REST_P50_MM \
            or (out["arrive_lag_p50_ms"] or 0.0) > CLOSE_ARRIVE_LAG_MS \
            or out["moves_cam_never_arrived"] > 1:
        out["verdict"] = "lagging"
    else:
        out["verdict"] = "close"
    return out


def format_summary(s: dict) -> str:
    """The summary as a few monospace lines, for the UI and the log."""
    if s.get("verdict") == "no data":
        return "no data"
    # A stat can be None when its window had no samples (a run stopped in
    # the first hold, say); print it as 0 rather than crash the report.
    s = {k: (0.0 if v is None and k.startswith(("gap_", "cam_", "ctl_", "lag_"))
             else v) for k, v in s.items()}
    caps = s.get("caps", {})
    lines = [
        f"verdict: {s.get('verdict', '?').upper()}"
        + ("  (stopped early)" if s.get("stopped_early") else ""),
        f"caps {caps.get('speed_mm_s', 0) / 1000:.1f} m/s, "
        f"{caps.get('accel_mm_s2', 0) / 1000:.1f} m/s2   "
        f"{s.get('duration_s', 0):.1f} s, {s.get('samples', 0)} samples",
    ]
    if s.get("camera_hz"):
        lines += [
            f"camera {s['camera_hz']:.0f} Hz, latency {s.get('lag_ms', 0):.0f} ms",
            f"gap moving  p50 {s['gap_moving_p50_mm']:.0f}  p90 "
            f"{s['gap_moving_p90_mm']:.0f}  max {s['gap_moving_max_mm']:.0f} mm"
            f"  (raw p90 {s['gap_raw_moving_p90_mm']:.0f})",
            f"gap at rest p50 {s['gap_rest_p50_mm']:.0f}  max "
            f"{s['gap_rest_max_mm']:.0f} mm",
            f"peak speed  teensy {s['ctl_peak_speed_mm_s'] / 1000:.2f}  camera "
            f"{(s.get('cam_peak_speed_mm_s') or 0) / 1000:.2f} m/s",
        ]
        if s.get("arrive_lag_p50_ms") is not None:
            lines.append(f"arrival lag p50 {s['arrive_lag_p50_ms']:.0f}  max "
                         f"{s['arrive_lag_max_ms']:.0f} ms"
                         + (f", {s['moves_cam_never_arrived']} moves never arrived"
                            if s.get("moves_cam_never_arrived") else ""))
    else:
        lines.append("camera: none")
    encs = s.get("encoders") or []
    if any(e.get("gain") is not None for e in encs):
        lines.append("drives vs steps  " + "  ".join(
            f"m{e['motor']} x{e['gain']:+.3f} rest {e['rest_mm']:.1f} "
            f"move {e['moving_mm']:.0f} (read {e['lag_ms']:.0f} ms late)"
            if e.get("gain") is not None else f"m{e['motor']} --"
            for e in encs))
    if any(e.get("trq_peak_pct") is not None for e in encs):
        lines.append("torque peak %   " + "  ".join(
            f"m{e['motor']} {e['trq_peak_pct']:.0f}"
            if e.get("trq_peak_pct") is not None else f"m{e['motor']} --"
            for e in encs))
    worst = [e for e in s.get("segments", []) if e.get("gap_max_mm") is not None]
    worst.sort(key=lambda e: -e["gap_max_mm"])
    if worst:
        lines.append("worst segments  " + ", ".join(
            f"{e['name']} {e['gap_max_mm']:.0f} mm" for e in worst[:4]))
    for n in s.get("notes", []):
        lines.append(f"! {n}")
    if s.get("log_csv"):
        lines.append(f"log {s['log_csv']}")
    return "\n".join(lines)
