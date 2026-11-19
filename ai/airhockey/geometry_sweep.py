"""The CALIBRATION SWEEP (2026-11-18): measure the cable geometry from the table.

Away from the centre one or two cables go slack while their opposites pull
hard. The motors are position-controlled, so tension is whatever falls out of
how well the four commanded lengths agree with the real machine; a few
millimetres of disagreement is more than the pretension. Play logs cannot fix
the model: the mallet only stands still at the centre and near the goal, and
they carry no currents, so cable stretch (about 1 mm per amp) is invisible.

THE SWEEP is a third move list for `follow_test.FollowTest`: a slow serpentine
over a grid covering the workspace, pausing at each point. Recorded there:
where the camera sees the mallet, where the controller believes it is (hence
the length it has paid out on each cable, through the current model), and each
drive's holding current (`SAMPLE <s> torque`).

THE FIT finds the four motor positions and one spool radius (all spools the
same size, the user's assumption) for which, at every point and for every
cable that is taut,

    true length(camera position)  =  paid-out length  +  stretch * current

with one stretch coefficient per cable and one unknown starting length per
cable. A cable holding under `SLACK_AMPS` is slack: it constrains nothing and
is left out. The result is judged on points the fit did not see (5 folds) and
by how far the parameters move between folds; `recommend` is true only when
both say the new numbers are real. Nothing is applied by this module.

Logs: logs/geometry_sweep/. The camera's own calibration is taken as given:
the fit puts the motors where they must be for the cables to agree with the
camera, which is also the frame the game is played in.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from airhockey.follow_test import SETTLE_S, Segment, geom
from airhockey.ring_test import load_drive_samples

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))
import cable_model  # noqa: E402

LOG_DIR = ROOT / "logs" / "geometry_sweep"

TEST_SPEED_MM_S = 500.0
TEST_ACCEL_MM_S2 = 4000.0
GRID = (6, 6)           # points along x, along y
EDGE_MM = 45.0          # stay this far inside the workspace
HOLD_S = 0.9            # at each point, after the move has settled
MEASURE_S = 0.6         # the last part of the hold is what is averaged
SLACK_AMPS = 0.25       # below this a cable is not pulling (slack reads 0-0.16 A)
MIN_POINTS = 16
FOLDS = 5


def sequence(box: tuple[float, float, float, float] | None = None) -> list[Segment]:
    """Serpentine over the grid, column by column from the forward edge to the
    goal end, so the hot goal-end column comes last and is left at once."""
    x0, x1, y0, y1 = box or (geom.WS_MIN_X, geom.WS_MAX_X, geom.WS_MIN_Y, geom.WS_MAX_Y)
    xs = np.linspace(x0 + EDGE_MM, x1 - EDGE_MM, GRID[0])
    ys = np.linspace(y0 + EDGE_MM, y1 - EDGE_MM, GRID[1])
    segs: list[Segment] = []
    for i, x in enumerate(xs):
        for y in (ys if i % 2 == 0 else ys[::-1]):
            segs.append(Segment(f"point {len(segs) // 2 + 1}: go", float(x), float(y), SETTLE_S, "move"))
            segs.append(Segment(f"point {len(segs) // 2 + 1}: measure", float(x), float(y), HOLD_S, "hold"))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    segs.append(Segment("centre: go", cx, cy, SETTLE_S, "move"))
    return segs


def quiet_summary(rows, segs, camera=True) -> dict:
    return {"sequence": "sweep"}


def nominal_lengths(positions: np.ndarray) -> np.ndarray:
    """Cable lengths (up to a constant) the CURRENT model pays out for the
    controller's positions, (N, 4)."""
    anchors = np.column_stack([geom.MOTOR_X, geom.MOTOR_Y]).astype(float)
    return cable_model.cable_lengths(anchors, positions, np.full(len(positions), geom.MALLET_THETA_RAD))


def collect(rows: list[dict], segs: list[Segment], drives: dict) -> dict:
    """One row per measured point: camera position, controller position and
    each drive's holding current, medians over the end of the hold."""
    t = np.array([r["t"] for r in rows], float)
    seg = np.array([r["seg"] for r in rows], int)
    cam, ctl, amps, angles = [], [], [], []
    for i, s in enumerate(segs):
        if not s.name.endswith(": measure"):
            continue
        inside = np.flatnonzero(seg == i)
        if not len(inside):
            continue
        end = t[inside[-1]]
        window = [rows[k] for k in inside if t[k] >= end - MEASURE_S]
        seen = [(r["cam_x"], r["cam_y"]) for r in window if r.get("cam_x") is not None]
        if len(seen) < 5:
            continue
        current = []
        for m in range(4):
            if m not in drives:
                break
            dt, dv = drives[m]
            keep = (dt >= end - MEASURE_S) & (dt <= end)
            if keep.sum() < 5:
                break
            current.append(float(np.median(dv[keep])))
        if len(current) < 4:
            continue
        cam.append(np.median(np.array(seen, float), axis=0))
        turned = [r["cam_theta_deg"] for r in window if r.get("cam_theta_deg") is not None]
        angles.append(float(np.median(turned)) if turned else float("nan"))
        ctl.append([np.median([r["ctl_x"] for r in window]), np.median([r["ctl_y"] for r in window])])
        amps.append(current)
    return dict(camera=np.array(cam, float).reshape(-1, 2), controller=np.array(ctl, float).reshape(-1, 2),
                amps=np.array(amps, float).reshape(-1, 4), theta_deg=np.array(angles, float))


def _residuals(p, camera, nominal, tension, taut, anchors0, radius0, ref):
    anchors = anchors0 + p[:8].reshape(4, 2)
    radius = radius0 + p[8]
    true = cable_model.cable_lengths(anchors, camera, np.full(len(camera), geom.MALLET_THETA_RAD),
                                     spool_r=radius, ref=ref)
    r = true - (radius / radius0) * nominal - p[9:13][None, :] - p[13:17][None, :] * tension
    return r[taut]


def _solve(camera, nominal, tension, taut, geometry=True):
    anchors0 = np.column_stack([geom.MOTOR_X, geom.MOTOR_Y]).astype(float)
    ref = cable_model.wrap_reference(anchors0, np.array([geom.HOME_X, geom.HOME_Y]))
    args = (camera, nominal, tension, taut, anchors0, geom.SPOOL_RADIUS_MM, ref)
    if geometry:
        fun = lambda p: _residuals(p, *args)                                    # noqa: E731
        start = np.zeros(17)
    else:                       # today's motors and radius: starting lengths and stretch only
        fun = lambda q: _residuals(np.concatenate([np.zeros(9), q]), *args)      # noqa: E731
        start = np.zeros(8)
    lower = np.full(len(start), -np.inf)
    lower[-4:] = 0.0            # a cable does not get shorter under load
    start[-4:] = 1.0            # mm per amp, the 2026-11 estimate; off the bound
    sol = least_squares(fun, start, bounds=(lower, np.inf), loss="soft_l1", f_scale=0.5)
    return (sol.x if geometry else np.concatenate([np.zeros(9), sol.x])), sol, args


def fit(camera: np.ndarray, nominal: np.ndarray, amps: np.ndarray) -> dict:
    """Motor positions, spool radius, per-cable stretch; and whether to believe them."""
    tension = np.abs(amps)
    taut = tension >= SLACK_AMPS
    n = len(camera)
    if n < MIN_POINTS:
        raise ValueError(f"only {n} usable points; the sweep needs {MIN_POINTS}")
    rms = lambda r: float(np.sqrt(np.mean(np.square(r)))) if len(r) else float("nan")   # noqa: E731
    p_now, sol_now, args = _solve(camera, nominal, tension, taut, geometry=False)
    p_fit, sol_fit, _ = _solve(camera, nominal, tension, taut, geometry=True)
    # Points the fit did not see, and how far the answer moves without them.
    order = np.random.default_rng(0).permutation(n)
    held_now, held_fit, spread = [], [], []
    for k in range(FOLDS):
        test = np.zeros(n, bool)
        test[order[k::FOLDS]] = True
        for geometry, bucket in ((False, held_now), (True, held_fit)):
            p, _, _ = _solve(camera[~test], nominal[~test], tension[~test], taut[~test], geometry)
            bucket.append(_residuals(p, camera[test], nominal[test], tension[test], taut[test], *args[4:]))
            if geometry:
                spread.append(p[:9])
    spread = np.array(spread)
    held = (rms(np.concatenate(held_now)), rms(np.concatenate(held_fit)))
    moves = spread.max(0) - spread.min(0)
    anchors0 = args[4]
    shifts = p_fit[:8].reshape(4, 2)
    # Believe it when unseen points agree clearly better AND the answer is
    # stable: no motor position wandering by more than 3 mm between folds.
    recommend = bool(held[1] < 0.75 * held[0] and moves[:8].max() <= 3.0 and moves[8] <= 0.15)
    return dict(
        points=n, slack=[int(v) for v in (~taut).sum(0)],
        rms_now=round(rms(sol_now.fun), 3), rms_fitted=round(rms(sol_fit.fun), 3),
        held_out_now=round(held[0], 3), held_out_fitted=round(held[1], 3),
        motors=[dict(x=round(float(anchors0[m, 0] + shifts[m, 0]), 1), y=round(float(anchors0[m, 1] + shifts[m, 1]), 1),
                     dx=round(float(shifts[m, 0]), 2), dy=round(float(shifts[m, 1]), 2),
                     wander=round(float(max(moves[2 * m], moves[2 * m + 1])), 2)) for m in range(4)],
        spool_radius=round(float(geom.SPOOL_RADIUS_MM + p_fit[8]), 3), spool_radius_now=geom.SPOOL_RADIUS_MM,
        spool_radius_wander=round(float(moves[8]), 3),
        stretch_mm_per_amp=[round(float(v), 2) for v in p_fit[13:17]],
        stretch_mm_per_amp_now=[round(float(v), 2) for v in p_now[13:17]],
        recommend=recommend)


def format_report(result: dict) -> str:
    lines = []
    if result.get("error"):
        return result["error"]
    f = result["fit"]
    lines.append(("RECOMMENDED: the fitted geometry explains points it never saw clearly better, and is stable."
                  if f["recommend"] else
                  "NOT RECOMMENDED: the fitted geometry is not clearly better on unseen points, or not stable. "
                  "Keep today's values."))
    lines.append(f"{f['points']} points. Cable-length disagreement on points left out of the fit: "
                 f"{f['held_out_now']:.2f} mm with today's geometry, {f['held_out_fitted']:.2f} mm fitted "
                 f"(all points: {f['rms_now']:.2f} -> {f['rms_fitted']:.2f} mm).")
    lines.append("")
    lines.append("motor    today (x, y)        fitted (x, y)       shift (dx, dy)    wander between folds")
    for m, mot in enumerate(f["motors"]):
        lines.append(f"  {m}    ({geom.MOTOR_X[m]:7.1f}, {geom.MOTOR_Y[m]:7.1f})  ({mot['x']:7.1f}, {mot['y']:7.1f})  "
                     f"({mot['dx']:+6.2f}, {mot['dy']:+6.2f})    {mot['wander']:.2f} mm")
    lines.append(f"spool radius: {f['spool_radius_now']:.2f} -> {f['spool_radius']:.3f} mm "
                 f"(wander {f['spool_radius_wander']:.3f})")
    lines.append("cable stretch, mm per amp of holding current (motors 0-3): "
                 + " ".join(f"{v:.2f}" for v in f["stretch_mm_per_amp"]))
    lines.append("points where each cable was slack (under %.2f A): " % SLACK_AMPS + " ".join(map(str, f["slack"])))
    turn = result.get("theta")
    if turn:
        lines.append(f"mallet angle over the grid: {turn['min']:.1f} to {turn['max']:.1f} deg "
                     f"(the controller assumes {turn['assumed']:.1f} everywhere); "
                     f"it changes {turn['per_100mm_y']:+.2f} deg per 100 mm of y and "
                     f"{turn['per_100mm_x']:+.2f} per 100 mm of x.")
    lines.append("Nothing has been changed on the robot.")
    return "\n".join(lines)


def finish(test, wait_s: float = 6.0, sleep=None) -> dict:
    """Read the drive currents, reduce the run to points, fit, rewrite the log."""
    sleep = sleep or time.sleep
    summary = dict(test.result or {})
    result = {k: summary[k] for k in ("log_csv", "log_json", "drive_samples", "stopped_early", "monotonic_t0")
              if k in summary}
    result["sequence"] = "sweep"
    drives = {}
    if summary.get("drive_samples"):
        full = ROOT / summary["drive_samples"]
        for _ in range(int(wait_s / 0.1)):
            if full.is_file():
                break
            sleep(0.1)
        try:
            drives = load_drive_samples(full, float(summary.get("monotonic_t0", 0.0)))
        except (OSError, ValueError, KeyError):
            drives = {}
    try:
        if len(drives) < 4:
            raise ValueError("the drive currents were not recorded (shut the motor controller down on the "
                             "Robot tab so the current version starts, and run again)")
        points = collect(test.rows, test.segments, drives)
        result["points"] = {k: np.round(v, 3).tolist() for k, v in points.items()}
        result["fit"] = fit(points["camera"], nominal_lengths(points["controller"]), points["amps"])
        angle = points["theta_deg"]
        seen = np.isfinite(angle)
        if seen.sum() >= 6:
            # A plane through the measured angle: how it varies across the table.
            design = np.column_stack([np.ones(seen.sum()), points["camera"][seen] - points["camera"][seen].mean(0)])
            plane, *_ = np.linalg.lstsq(design, angle[seen], rcond=None)
            result["theta"] = dict(min=float(angle[seen].min()), max=float(angle[seen].max()),
                                   assumed=float(np.degrees(geom.MALLET_THETA_RAD)),
                                   per_100mm_x=float(100 * plane[1]), per_100mm_y=float(100 * plane[2]))
    except ValueError as e:
        result["error"] = f"Calibration sweep could not be fitted: {e}."
    result["text"] = format_report(result)
    if result.get("log_csv"):
        result["text"] += f"\n\nsamples: {result['log_csv']}"
    if result.get("log_json"):
        try:
            with open(result["log_json"], "w") as f:
                json.dump(result, f, indent=1)
        except OSError:
            pass
    return result
