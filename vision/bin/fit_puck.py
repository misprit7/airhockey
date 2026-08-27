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

# How far the puck CENTRE sits from a rail at the moment of contact: one puck
# radius, plus slack for tracking error and for the rails being stored as the
# INSCRIBED rectangle (so the true wall is a few mm further out in places).
#
# This was a hardcoded 45.0, set when the puck radius was believed to be 31.5.
# It is 40.7, so the band barely reached the contact point and most wall hits
# were being classified as "away from a rail" -- 147 of 175 contacts in the
# first real recording, with the surviving handful producing impossible
# results like a negative tangential ratio.
WALL_BAND_MM = geom.PUCK_RADIUS_MM + 18.0

# Heading change per frame that means a collision. At 200 Hz a gliding puck
# turns by essentially zero; friction is anti-parallel to velocity and cannot
# steer. 12 degrees is far above noise and far below a real bounce.
TURN_DEG = 12.0

# Frames to skip either side of a contact before trusting a velocity. The
# tracker's slope spans 6 frames, so anything closer mixes pre- and post-.
SKIP = 4
MIN_GLIDE = 15          # frames; shorter segments do not constrain a slope
MIN_SPEED = 200.0       # mm/s; below this the tracker's noise dominates

# A free puck can only SLOW DOWN. Any frame where speed rises by more than
# tracking noise means something pushed it -- a hand, almost always. The
# original detector only cut on heading change, so a straight-line push went
# straight through and its segment was fitted as if it were a glide. That is
# what produced a 3700 mm/s^2 "deceleration" at 1 m/s and dragged the whole
# friction fit with it.
#
# Generous against the tracker's ~2 mm/s velocity noise, so only real pushes
# are cut.
ACCEL_TOL_MM_S = 25.0

# Nothing on an air cushion decelerates faster than this. A backstop for
# segments that survive everything else; a puck at 1 m/s losing 3.7 m/s^2
# would stop dead in a quarter second, which is not friction.
MAX_DECEL_MM_S2 = 2000.0


# A contact counts as puck-on-mallet if the two centres are about this far
# apart. Puck radius + mallet radius is the contact distance; generous
# because both are tracked, so the error is the sum of two tracking errors.
CONTACT_MM = geom.PUCK_RADIUS_MM + geom.MALLET_RADIUS_MM + 25.0


def load(path):
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln]
    if not rows:
        sys.exit(f"{path} is empty")
    d = {k: np.array([r[k] for r in rows], dtype=float) for k in
         ("seq", "t", "x", "y", "vx", "vy")}
    # Mallet columns are optional -- present only when the mallet was
    # resolvable that frame -- so they are read separately and padded with
    # NaN. Omitting them here is what made the paddle fit report "no mallet
    # positions" on recordings that plainly had them.
    if any("mx" in r for r in rows):
        d["mx"] = np.array([r.get("mx", np.nan) for r in rows], dtype=float)
        d["my"] = np.array([r.get("my", np.nan) for r in rows], dtype=float)
    # Corner count and spin, present only since the puck went to four
    # markers. Read the same way so old recordings still load.
    for k in ("n", "th", "w"):
        if any(k in r for r in rows):
            d[k] = np.array([r.get(k, np.nan) for r in rows], dtype=float)
    return d


def segment(d):
    """Split into contiguous runs, cut at frame gaps and at contacts."""
    speed = np.hypot(d["vx"], d["vy"])
    heading = np.arctan2(d["vy"], d["vx"])
    turn = np.abs(np.diff(heading))
    turn = np.minimum(turn, 2 * np.pi - turn)          # wrap
    gap = np.diff(d["seq"]) != 1
    moving = speed[:-1] > MIN_SPEED
    # Speeding up is a contact, whatever direction it happened in.
    pushed = np.diff(speed) > ACCEL_TOL_MM_S
    contact = (((turn > np.radians(TURN_DEG)) & moving) | pushed | gap)

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
    """Deceleration per glide, as a constant AND as a + b*v^2.

    A single coefficient cannot describe this surface. On an air cushion the
    Coulomb term is tiny and AERODYNAMIC DRAG dominates at speed, so
    deceleration rises with v^2 -- which is exactly what an eightfold IQR
    across glides from 0.4 to 8.7 m/s means. Reporting only the mean would
    hand the simulator a number that is far too high for a slow puck and far
    too low for a struck one, and striking is the regime that matters.
    """
    rows = []
    for a, b in bounds:
        t = d["t"][a + SKIP:b - SKIP]
        if len(t) < MIN_GLIDE:
            continue
        v = np.hypot(d["vx"][a + SKIP:b - SKIP], d["vy"][a + SKIP:b - SKIP])
        if v.min() < MIN_SPEED:
            continue
        # Reject anything that sped up inside the window even slightly --
        # segment() cuts at the push, but a residual frame either side can
        # still carry it.
        if np.diff(v).max() > ACCEL_TOL_MM_S:
            continue
        slope = np.polyfit(t - t[0], v, 1)[0]           # mm/s per s
        if slope < 0 and -slope < MAX_DECEL_MM_S2:
            rows.append((-slope, len(t), v.mean()))
    if not rows:
        return None
    dec = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows], dtype=float)
    mean = float((dec * w).sum() / w.sum())
    speeds = np.array([r[2] for r in rows])
    out = {"n": len(rows), "decel_mm_s2": mean,
           "mu": mean / G_MM_S2,
           "spread": (float(np.percentile(dec, 25)),
                      float(np.percentile(dec, 75))),
           "speeds": speeds,
           # The accepted segments themselves, so plotting shows what was
           # FITTED rather than re-deriving it and disagreeing. The plot was
           # drawing points the fitter had rejected, which made a clean fit
           # look outlier-ridden.
           "decels": dec, "weights": np.array([r[1] for r in rows])}
    # decel = a + b*v^2, in mm units. Fit only if the speeds actually span a
    # range; otherwise the quadratic term is unconstrained and will fit noise.
    if len(rows) >= 8 and speeds.max() > 3 * max(speeds.min(), 1e-9):
        A = np.column_stack([np.ones_like(speeds), speeds ** 2])
        # Iteratively reweighted least squares (Huber). Plain least squares
        # squares the residual, so one bad segment at 10x the error carries
        # 100x the weight -- which is precisely how a single point at
        # (1 m/s, 3700 mm/s^2) bent the whole curve.
        w = np.ones_like(dec)
        for _ in range(12):
            coef, *_ = np.linalg.lstsq(A * w[:, None], dec * w, rcond=None)
            r = dec - A @ coef
            sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            w = np.clip(1.345 * sigma / np.maximum(np.abs(r), 1e-9), 0, 1)
        resid = dec - A @ coef
        out["drag"] = {"a": float(coef[0]), "b": float(coef[1]),
                       "rms": float(np.sqrt((resid ** 2).mean())),
                       "rms_const": float(np.sqrt(((dec - mean) ** 2).mean()))}
    return out


def wall_of(x, y):
    """Which cushion is this point against, if any.

    The END rails are not continuous cushion: each has a 380 mm goal mouth
    centred on it. A puck arriving there does not bounce -- it goes in, or
    clips the goal edge -- so those contacts are not restitution measurements
    and mixing them in is what dragged the two end rails down to e ~ 0.45
    with an impossible tangential ratio of -5, while the two continuous side
    rails agreed with each other at 0.756 and 0.777.
    """
    y_mid = (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y) / 2.0
    in_goal = abs(y - y_mid) < geom.GOAL_WIDTH_MM / 2.0

    if y - geom.RAIL_MIN_Y < WALL_BAND_MM:
        return "near(-y)", np.array([0.0, 1.0])
    if geom.RAIL_MAX_Y - y < WALL_BAND_MM:
        return "far(+y)", np.array([0.0, -1.0])
    if x - geom.RAIL_MIN_X < WALL_BAND_MM:
        return (None, None) if in_goal else ("human(-x)", np.array([1.0, 0.0]))
    if geom.RAIL_MAX_X - x < WALL_BAND_MM:
        return (None, None) if in_goal else ("robot(+x)", np.array([-1.0, 0.0]))
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
        # A cushion cannot return more normal speed than it received. A
        # ratio above 1 means something pushed -- a hand still in contact --
        # and those were producing the impossible tangential ratios.
        if -vout_n / vin_n > 1.0:
            continue
        walls.append({"wall": name, "speed_in": float(np.hypot(*vin)),
                      "e_normal": float(-vout_n / vin_n),
                      "e_tangential": float((vout @ tan) / (vin @ tan))
                      if abs(vin @ tan) > 50 else np.nan})
    return walls, others


def fit_paddle(d, cuts):
    """Restitution against the mallet, using RELATIVE normal velocity.

    A wall does not move, so |v_out| / |v_in| is the coefficient. A mallet
    does -- it recoils, and on this rig the interesting question is how much.
    Restitution is therefore defined on the relative normal velocity:

        e = -(v_puck_out - v_mallet_out).n / (v_puck_in - v_mallet_in).n

    with n the line of centres. Ignoring the recoil would fold the mallet's
    effective mass into the coefficient and give a number that is only valid
    for the mass it was measured at.

    The recoil is the other half of the point: if the mallet kicks back like
    a free ~170 g plastic disc, then over the ~1 ms of contact the 2.10 N/mm
    springs (28 ms period) and the servo (10s of ms) are far too slow to
    participate, and the robot's mallet is mechanically the same target as a
    hand-held one. If it barely moves, they are not, and the simulator needs
    two coefficients rather than the one it has.
    """
    if "mx" not in d:
        return None
    hits, n = [], len(d["t"])
    for c in cuts:
        i, j = c - SKIP, c + 1 + SKIP
        if i < 0 or j >= n or d["seq"][j] - d["seq"][i] > 2 * SKIP + 4:
            continue
        if np.isnan(d["mx"][c]):
            continue
        sep = np.hypot(d["x"][c] - d["mx"][c], d["y"][c] - d["my"][c])
        if sep > CONTACT_MM:
            continue
        nrm = np.array([d["x"][c] - d["mx"][c], d["y"][c] - d["my"][c]])
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm /= ln

        vin = np.array([d["vx"][i], d["vy"][i]])
        vout = np.array([d["vx"][j], d["vy"][j]])
        if np.hypot(*vin) < MIN_SPEED:
            continue
        # Mallet velocity by finite difference across the same bracket.
        dt_i = d["t"][c] - d["t"][i]
        dt_o = d["t"][j] - d["t"][c]
        mv_in = np.array([(d["mx"][c] - d["mx"][i]) / dt_i,
                          (d["my"][c] - d["my"][i]) / dt_i]) if dt_i > 0 else np.zeros(2)
        mv_out = np.array([(d["mx"][j] - d["mx"][c]) / dt_o,
                           (d["my"][j] - d["my"][c]) / dt_o]) if dt_o > 0 else np.zeros(2)

        rel_in = (vin - mv_in) @ nrm
        rel_out = (vout - mv_out) @ nrm
        if rel_in >= 0 or rel_out <= 0:
            continue
        hits.append({"e": float(-rel_out / rel_in),
                     "speed_in": float(np.hypot(*vin)),
                     "recoil": float(np.hypot(*(mv_out - mv_in)))})
    return hits


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
              f"<- sim currently uses 0.0100 (constant-model equivalent)")
        if "drag" in fr:
            g = fr["drag"]
            print(f"\n  speed-dependent fit  decel = a + b*v^2")
            print(f"    a (rolling)  {g['a']:8.1f} mm/s^2   "
                  f"= mu {g['a'] / G_MM_S2:.5f}")
            print(f"    b (drag)     {g['b']:.3e} 1/mm")
            print(f"    residual rms {g['rms']:7.1f} vs {g['rms_const']:7.1f} "
                  f"for a single constant")
            if g["rms"] < 0.7 * g["rms_const"]:
                print("    -> the quadratic model is clearly better; the "
                      "spread is DRAG, not noise.")
                for v in (1000, 3000, 6000):
                    print(f"       at {v/1000:.0f} m/s: "
                          f"{g['a'] + g['b'] * v * v:7.1f} mm/s^2")
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
    print("\n── PADDLE (MALLET) RESTITUTION " + "─" * 36)
    pad = fit_paddle(d, cuts)
    if pad is None:
        print("  no mallet positions in this recording -- re-record with a "
              "current record_puck.py to get them")
    elif not pad:
        print("  no puck-mallet contacts found. Shoot the puck at the mallet "
              "a few times, at a few speeds.")
    else:
        e = np.array([h["e"] for h in pad])
        rc = np.array([h["recoil"] for h in pad])
        print(f"  {len(pad)} contacts")
        print(f"  e (relative normal)  {e.mean():.3f}   "
              f"spread {e.min():.2f}-{e.max():.2f}   "
              f"<- sim currently uses 0.90")
        print(f"  mallet recoil        {rc.mean():6.0f} mm/s mean, "
              f"{rc.max():6.0f} max")
        print()
        # A free 170 g mallet struck by a ~30 g puck should pick up roughly
        # (2 m /(m + M)) * v_in ~= 0.3 * v_in. Much less means the cables are
        # resisting on the impact timescale, which the timescale argument says
        # they should not.
        sp = np.array([h["speed_in"] for h in pad])
        frac = (rc / np.maximum(sp, 1e-9)).mean()
        print(f"  recoil / impact speed  {frac:.2f}")
        if frac > 0.15:
            print("  -> the mallet moves roughly like a free mass, so the "
                  "springs and motors are\n     too slow to matter during "
                  "contact and ONE paddle_restitution covers both\n     the "
                  "robot's mallet and a hand-held one.")
        else:
            print("  -> the mallet barely recoils, so the cables ARE resisting "
                  "on the impact\n     timescale. The robot's mallet is a "
                  "different target from a hand-held one\n     and the "
                  "simulator needs two coefficients, not one.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
