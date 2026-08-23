#!/usr/bin/env python3
"""Fit puck friction and wall restitution from a record_puck.py recording.

Produces the three numbers the simulator currently guesses:

    puck_friction        deceleration coefficient a = mu*g on the air cushion
    wall_restitution     |v_out| / |v_in| normal to each cushion
    (tangential ratio)   |v_t_out| / |v_t_in|, which is 1.0 for a frictionless
                         cushion and less than 1 if the puck picks up spin

METHOD
    Segments the recording into glides separated by contacts. A contact is a
    frame where the velocity DIRECTION turns sharply -- friction cannot turn
    the puck, only slow it, so a heading change beyond a few degrees per
    frame is a collision. Contacts near a rail are wall hits; contacts away
    from the rails are the mallet (or a hand) and are reported separately
    rather than mixed in, since they obey a different restitution.

    Friction comes from the glides: fit speed against time over each segment.
    Reported as a coefficient so it is dimensionless and comparable to the
    simulator's `puck_friction`, which multiplies g.

    Restitution comes from the frames bracketing each wall contact, using
    velocity a few frames either side rather than adjacent ones -- the
    least-squares slope inside PuckTracker spans 6 frames, so the estimates
    immediately either side of an impact are contaminated by both.

WHAT TO LOOK FOR
    Restitution that falls with impact speed is normal and worth modelling.
    A tangential ratio well below 1 means the cushion imparts spin, which the
    simulator does not model at all -- the puck has no orientation state --
    and that would make every bank shot the policy plans systematically
    wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

G_MM_S2 = 9806.65

# A puck radius from the rail counts as "at the wall". Slightly generous so a
# contact is not missed by tracking noise; contacts land within a millimetre
# or two of this in practice.
WALL_BAND_MM = 45.0

# Heading change per frame that means a collision. At 200 Hz a gliding puck
# turns by essentially zero; friction is anti-parallel to velocity and cannot
# steer. 12 degrees is far above noise and far below a real bounce.
TURN_DEG = 12.0

# Frames to skip either side of a contact before trusting a velocity. The
# tracker's slope spans 6 frames, so anything closer mixes pre- and post-.
SKIP = 4
MIN_GLIDE = 15          # frames; shorter segments do not constrain a slope
MIN_SPEED = 200.0       # mm/s; below this the tracker's noise dominates


def load(path):
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln]
    if not rows:
        sys.exit(f"{path} is empty")
    d = {k: np.array([r[k] for r in rows], dtype=float) for k in
         ("seq", "t", "x", "y", "vx", "vy")}
    return d


def segment(d):
    """Split into contiguous runs, cut at frame gaps and at contacts."""
    speed = np.hypot(d["vx"], d["vy"])
    heading = np.arctan2(d["vy"], d["vx"])
    turn = np.abs(np.diff(heading))
    turn = np.minimum(turn, 2 * np.pi - turn)          # wrap
    gap = np.diff(d["seq"]) != 1
    moving = speed[:-1] > MIN_SPEED
    contact = ((turn > np.radians(TURN_DEG)) & moving) | gap

    cuts = np.flatnonzero(contact)
    bounds, start = [], 0
    for c in cuts:
        if c + 1 - start >= MIN_GLIDE:
            bounds.append((start, c + 1))
        start = c + 1
    if len(d["t"]) - start >= MIN_GLIDE:
        bounds.append((start, len(d["t"])))
    return bounds, cuts, np.flatnonzero(gap)


def fit_friction(d, bounds):
    """Least-squares deceleration per glide, weighted by segment length."""
    rows = []
    for a, b in bounds:
        t = d["t"][a + SKIP:b - SKIP]
        if len(t) < MIN_GLIDE:
            continue
        v = np.hypot(d["vx"][a + SKIP:b - SKIP], d["vy"][a + SKIP:b - SKIP])
        if v.min() < MIN_SPEED:
            continue
        slope = np.polyfit(t - t[0], v, 1)[0]           # mm/s per s
        if slope < 0:                                   # decelerating
            rows.append((-slope, len(t), v.mean()))
    if not rows:
        return None
    dec = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows], dtype=float)
    mean = float((dec * w).sum() / w.sum())
    return {"n": len(rows), "decel_mm_s2": mean,
            "mu": mean / G_MM_S2,
            "spread": (float(np.percentile(dec, 25)),
                       float(np.percentile(dec, 75))),
            "speeds": np.array([r[2] for r in rows])}


def wall_of(x, y):
    """Which cushion is this point against, if any."""
    if y - geom.RAIL_MIN_Y < WALL_BAND_MM:
        return "near(-y)", np.array([0.0, 1.0])
    if geom.RAIL_MAX_Y - y < WALL_BAND_MM:
        return "far(+y)", np.array([0.0, -1.0])
    if x - geom.RAIL_MIN_X < WALL_BAND_MM:
        return "human(-x)", np.array([1.0, 0.0])
    if geom.RAIL_MAX_X - x < WALL_BAND_MM:
        return "robot(+x)", np.array([-1.0, 0.0])
    return None, None


def fit_bounces(d, cuts):
    walls, others = [], 0
    n = len(d["t"])
    for c in cuts:
        i, j = c - SKIP, c + 1 + SKIP
        if i < 0 or j >= n or d["seq"][j] - d["seq"][i] > 2 * SKIP + 4:
            continue
        name, nrm = wall_of(d["x"][c], d["y"][c])
        vin = np.array([d["vx"][i], d["vy"][i]])
        vout = np.array([d["vx"][j], d["vy"][j]])
        if np.hypot(*vin) < MIN_SPEED:
            continue
        if name is None:
            others += 1
            continue
        tan = np.array([-nrm[1], nrm[0]])
        vin_n, vout_n = vin @ nrm, vout @ nrm
        if vin_n >= 0 or vout_n <= 0:        # not actually approaching/leaving
            continue
        walls.append({"wall": name, "speed_in": float(np.hypot(*vin)),
                      "e_normal": float(-vout_n / vin_n),
                      "e_tangential": float((vout @ tan) / (vin @ tan))
                      if abs(vin @ tan) > 50 else np.nan})
    return walls, others


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    args = ap.parse_args()

    d = load(args.recording)
    bounds, cuts, gaps = segment(d)
    span = d["t"][-1] - d["t"][0]
    print(f"{len(d['t'])} samples over {span:.1f} s, {len(gaps)} tracking gaps")
    print(f"{len(bounds)} glide segments, {len(cuts)} contacts\n")

    print("── FRICTION " + "─" * 55)
    fr = fit_friction(d, bounds)
    if fr is None:
        print("  no usable glides. Need longer straight runs above "
              f"{MIN_SPEED:.0f} mm/s without wall contact.")
    else:
        lo, hi = fr["spread"]
        print(f"  {fr['n']} usable glides")
        print(f"  deceleration  {fr['decel_mm_s2']:8.1f} mm/s^2   "
              f"(IQR {lo:.0f}-{hi:.0f})")
        print(f"  puck_friction {fr['mu']:8.4f}   "
              f"<- sim currently uses 0.0100")
        if fr["mu"] > 0 and abs(fr["mu"] - 0.01) / 0.01 > 0.25:
            print(f"     that is {fr['mu'] / 0.01:.1f}x the placeholder")
        if hi > 2.5 * max(lo, 1e-9):
            print("     WIDE spread -- deceleration is not a single constant "
                  "here. Likely speed-dependent (air drag) or the table is "
                  "not level. Check the trend below.")
            s, dv = fr["speeds"], None
            order = np.argsort(s)
            print(f"     slowest glides ~{s[order][:3].mean():.0f} mm/s, "
                  f"fastest ~{s[order][-3:].mean():.0f} mm/s")

    print("\n── WALL RESTITUTION " + "─" * 47)
    walls, others = fit_bounces(d, cuts)
    if not walls:
        print("  no clean wall contacts found. Hit each cushion square-on a "
              "few times, at a few speeds.")
    else:
        print(f"  {len(walls)} wall contacts ({others} contacts away from a "
              f"rail, treated as mallet/hand and excluded)\n")
        print(f"  {'wall':<12}{'n':>4}{'e_normal':>12}{'spread':>16}"
              f"{'e_tangential':>14}")
        for name in ("near(-y)", "far(+y)", "human(-x)", "robot(+x)"):
            g = [w for w in walls if w["wall"] == name]
            if not g:
                print(f"  {name:<12}{0:>4}{'--':>12}")
                continue
            e = np.array([w["e_normal"] for w in g])
            et = np.array([w["e_tangential"] for w in g])
            et = et[~np.isnan(et)]
            print(f"  {name:<12}{len(g):>4}{e.mean():>12.3f}"
                  f"{f'{e.min():.2f}-{e.max():.2f}':>16}"
                  f"{(et.mean() if len(et) else float('nan')):>14.3f}")
        all_e = np.array([w["e_normal"] for w in walls])
        print(f"\n  overall e = {all_e.mean():.3f}   "
              f"<- sim currently uses 0.85")

        sp = np.array([w["speed_in"] for w in walls])
        if len(walls) >= 8 and np.ptp(sp) > 500:
            slope = np.polyfit(sp, all_e, 1)[0]
            print(f"  speed dependence: {slope * 1000:+.3f} per m/s of impact "
                  f"speed over {sp.min():.0f}-{sp.max():.0f} mm/s")
            if abs(slope * 1000) > 0.03:
                print("     worth modelling -- a single constant will be "
                      "wrong at one end of the range")
        else:
            print("  not enough speed variety to test speed dependence; "
                  "hit the cushions both softly and hard")

        et = np.array([w["e_tangential"] for w in walls])
        et = et[~np.isnan(et)]
        if len(et) >= 5 and et.mean() < 0.9:
            print(f"\n  TANGENTIAL ratio {et.mean():.3f} < 1: the cushion is "
                  "taking tangential momentum, i.e. the puck is picking up\n"
                  "  SPIN. The simulator has no puck orientation state at "
                  "all, so every bank shot it plans will be\n"
                  "  systematically off. Worth knowing before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
