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

# Frames to skip either side of a contact before trusting a velocity.
#
# PuckTracker's velocity at frame k is a least-squares slope over frames
# k-5..k (deque maxlen 6). A contact detected at cut index c happens BETWEEN
# c and c+1, so the first outgoing frame is c+1 and the first velocity built
# only from outgoing frames is at c+1+(SLOPE_FRAMES-1) = c+6.
#
# SKIP was 4, giving an outgoing sample at c+5 whose window still reached
# back to c -- one pre-impact frame inside every post-impact velocity. The
# synthetic check put a number on it: e_normal came back 0.749 against a
# truth of 0.800 and e_tangential 0.475 against 0.700, a 32% error that
# looked exactly like a physical result about cushions taking spin.
SLOPE_FRAMES = 6
SKIP = SLOPE_FRAMES - 1
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
    # n == 0 means the tracker COASTED that frame: the position is its own
    # constant-velocity extrapolation, not a measurement. Fitting friction to
    # those recovers "friction = 0" from the estimator rather than from the
    # table. record_puck drops them at capture now; recordings made before
    # that fix still carry them, so drop them here too. Removing rows leaves
    # a seq gap, which segment() already cuts on.
    if "n" in d:
        keep = d["n"] > 0
        if not keep.all():
            print(f"  dropped {int((~keep).sum())} coasted samples "
                  f"({100 * (~keep).mean():.1f}%) -- tracker extrapolating")
            d = {k: v[keep] for k, v in d.items()}
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
            rows.append((-slope, len(t), v.mean(), a, b))
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
           "decels": dec, "weights": np.array([r[1] for r in rows]),
           "bounds": [(int(r[3]), int(r[4])) for r in rows]}
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
        # Standard errors, because the two terms are NOT equally determined.
        # Above about 800 mm/s the v^2 term is an order of magnitude larger
        # than the rolling one, so `a` is an intercept extrapolated from data
        # that mostly sits far from v=0 -- and the synthetic check confirms
        # it: b comes back to 0.5%, while `a` lands anywhere from -1 to +16
        # against a truth of 22. Reporting `a` as a measurement without its
        # error invites reading noise as a friction coefficient.
        Aw = A * w[:, None]
        try:
            cov = sigma ** 2 * np.linalg.pinv(Aw.T @ Aw)
            se = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            se = np.array([np.nan, np.nan])
        out["drag"] = {"a": float(coef[0]), "b": float(coef[1]),
                       "se_a": float(se[0]), "se_b": float(se[1]),
                       "rms": float(np.sqrt((resid ** 2).mean())),
                       "rms_const": float(np.sqrt(((dec - mean) ** 2).mean()))}
    return out


# Window of raw POSITION samples used to reconstruct a velocity either side of
# a contact.
BRACKET = 10


def velocity_at(d, k0, k1, t_at):
    """Velocity at time `t_at` from a quadratic fit to RAW POSITIONS.

    Deliberately not the recording's vx/vy. Those are PuckTracker's own
    6-frame least-squares slope, so within 6 frames of an impact every one of
    them is a blend of before and after -- the estimator has memory and the
    positions do not. Bracketing further out to escape the blend then trades
    one bias for another, because the puck decelerates over the gap.

    A quadratic in position solves both at once: it is fitted on clean frames
    outside the smear and EVALUATED at the contact instant, so the drag
    curvature over the bracket is modelled rather than suffered. Returns
    (velocity, rms residual in mm); the residual is the guard -- a window that
    straddles the impact cannot be fitted by one quadratic and says so.
    """
    if k0 < 0 or k1 >= len(d["t"]) or k1 - k0 < 3:
        return None, np.inf
    # A window may MISS frames without being unusable: the fit is against real
    # timestamps, so uneven sampling costs nothing. What it must not do is
    # span a long blind spell, because the puck may have been hit inside one.
    #
    # This used to demand perfect contiguity, which quietly interacted with
    # dropping coasted samples: removing 2.4% of frames punched a seq gap
    # every few hundred rows and disqualified 29 of ~60 real wall contacts.
    # The recording was fine; the check was.
    missing = (d["seq"][k1] - d["seq"][k0]) - (k1 - k0)
    if missing > MAX_WINDOW_GAP:
        return None, np.inf
    tt = d["t"][k0:k1 + 1] - t_at
    A = np.stack([np.ones_like(tt), tt, 0.5 * tt ** 2], axis=1)
    v, res = np.zeros(2), 0.0
    for a, key in enumerate(("x", "y")):
        c, *_ = np.linalg.lstsq(A, d[key][k0:k1 + 1], rcond=None)
        v[a] = c[1]
        res += float(((A @ c - d[key][k0:k1 + 1]) ** 2).sum())
    return v, float(np.sqrt(res / (2 * len(tt))))


# Shortest bracket still worth fitting a quadratic to.
BRACKET_MIN = 5

# Frames a bracket may be missing and still be trusted. Coasted samples and
# short blind spells leave holes; a hole is not a discontinuity.
MAX_WINDOW_GAP = 6


def noise_floor(d, bounds):
    """Bracket residual on CLEAN glide interiors -- the rig's own noise.

    Calibrated from the recording rather than assumed. Averaging four marker
    corners makes the puck centroid far steadier frame to frame than its
    absolute accuracy suggests: this measures 0.04 mm on real data, against
    the ~1 mm of slowly-varying calibration error in the position itself.
    Guessing a threshold in millimetres instead is how a gate set for
    synthetic 0.35 mm noise silently threw away most real wall contacts.
    """
    r = []
    for a, b in bounds:
        for k in range(a + BRACKET, b - BRACKET, 7):
            _v, res = velocity_at(d, k - BRACKET, k - 1, d["t"][k])
            if np.isfinite(res):
                r.append(res)
    return float(np.percentile(r, 99)) if len(r) >= 20 else 0.15


def clean_velocity(d, anchor, direction, t_at, gate):
    """Velocity from the LONGEST clean window on one side of a contact.

    A fixed bracket is the wrong tool. Impacts come in bursts -- a bank shot
    off two rails, a bounce straight back into the mallet -- so ten clean
    frames often do not exist, and demanding them rejected 30 of 57 real wall
    contacts. Shrinking the window until the quadratic actually fits keeps the
    measurement and costs only precision, which is the right trade: the puck
    is rigid and the fit is over-determined at five frames.
    """
    for span in range(BRACKET, BRACKET_MIN - 1, -1):
        k0, k1 = ((anchor - span + 1, anchor) if direction < 0
                  else (anchor, anchor + span - 1))
        v, res = velocity_at(d, k0, k1, t_at)
        if v is not None and res <= gate:
            return v, res
    return None, np.inf


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


def contact_events(cuts):
    """Group per-frame cuts into ONE event per physical contact.

    A bounce does not produce a single cut. The tracker's velocity is a slope
    over 6 frames, so it swings round over ~6 frames and the per-frame heading
    change clears TURN_DEG for several of them in a row. Treating each as its
    own contact measures the same bounce repeatedly, and every repeat but one
    brackets it wrongly -- the synthetic check showed 1222 "contacts" for
    about 150 real bounces, and e_tangential coming back 0.475 against a truth
    of 0.700.

    Returns (first, last) index pairs; the caller brackets OUTSIDE the whole
    event rather than around one arbitrary frame inside it.
    """
    events = []
    for c in cuts:
        if events and c - events[-1][1] <= SLOPE_FRAMES:
            events[-1][1] = int(c)
        else:
            events.append([int(c), int(c)])
    return [(a, b) for a, b in events]


def wall_events(d, events):
    """Events where the puck centre actually reached a cushion.

    Separate from how many are MEASURABLE, and the gap between the two is
    worth printing rather than hiding. Detection is unambiguous -- the
    closest-approach histogram is bimodal, piling up at the 40.7 mm contact
    distance and then empty from 59 to 120 mm -- whereas measuring a bounce
    needs several frames of clean glide on BOTH sides, which continuous play
    often does not leave. Reporting only the measurable count reads as "you
    barely hit the walls", which is a statement about the fitter dressed up
    as one about the table.
    """
    n, hit = len(d["t"]), 0
    for lo, hi in events:
        sp = np.arange(max(0, lo - SLOPE_FRAMES - 4), min(hi + 3, n))
        near = np.minimum(
            np.minimum(d["x"][sp] - geom.RAIL_MIN_X,
                       geom.RAIL_MAX_X - d["x"][sp]),
            np.minimum(d["y"][sp] - geom.RAIL_MIN_Y,
                       geom.RAIL_MAX_Y - d["y"][sp]))
        if near.min() < WALL_BAND_MM:
            hit += 1
    return hit


def fit_bounces(d, events, gate=None):
    walls, others = [], 0
    n = len(d["t"])
    for lo, hi in events:
        i, j = lo - SKIP, hi + 1 + SKIP
        if i < 0 or j >= n or d["seq"][j] - d["seq"][i] > (j - i) + 4:
            continue
        # Classify on the frame that actually touched the cushion: the one
        # CLOSEST TO A RAIL, not the middle of the event. The middle sits 2-3
        # frames after contact, and at 2.4 m/s outgoing that is already 36 mm
        # off the cushion -- outside WALL_BAND_MM, so every clean bounce was
        # being classified "away from a rail" and counted as a mallet hit.
        span = np.arange(max(0, lo - SLOPE_FRAMES - 4), min(hi + 3, n))
        near = np.minimum(
            np.minimum(d["x"][span] - geom.RAIL_MIN_X,
                       geom.RAIL_MAX_X - d["x"][span]),
            np.minimum(d["y"][span] - geom.RAIL_MIN_Y,
                       geom.RAIL_MAX_Y - d["y"][span]))
        c = int(span[int(np.argmin(near))])
        name, nrm = wall_of(d["x"][c], d["y"][c])
        t_c = 0.5 * (d["t"][lo] + d["t"][min(hi + 1, n - 1)])
        vin, r_in = clean_velocity(d, lo - 1, -1, t_c, gate)
        vout, r_out = clean_velocity(d, hi + 1, +1, t_c, gate)
        if vin is None or vout is None:
            continue
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
        walls.append({"wall": name, "lo": int(lo), "hi": int(hi),
                      "c": int(c), "speed_in": float(np.hypot(*vin)),
                      "e_normal": float(-vout_n / vin_n),
                      "e_tangential": float((vout @ tan) / (vin @ tan))
                      if abs(vin @ tan) > 50 else np.nan})
    return walls, others



def fit_spin(d, events, gate=None):
    """Does the tangential momentum lost at a cushion turn into SPIN?

    Until the puck carried four markers this could only be inferred. It now
    has an orientation, so the question is a measurement.

    For a uniform disc, a tangential impulse J at the rim changes both terms:

        dv_t = J/m           dw = -J*R/I,   I = m*R^2/2

    so dw = -2*dv_t/R. Fitting dw against dv_t and comparing the slope to
    -2/R = -0.0491 rad/s per mm/s separates two very different worlds:

      slope ~ -2/R   the cushion GRIPS. Momentum went into spin, the puck
                     leaves rotating, and a sim with no orientation state
                     will misplace every bank shot.
      slope ~ 0      the cushion RUBS. Momentum went into heat, the puck
                     leaves barely spinning, and a plain tangential
                     coefficient is the whole model.
    """
    if "w" not in d:
        return None
    n, rows = len(d["t"]), []
    for lo, hi in events:
        i, j = lo - BRACKET_MIN, hi + BRACKET_MIN
        if i < 0 or j >= n:
            continue
        span = np.arange(max(0, lo - SLOPE_FRAMES - 4), min(hi + 3, n))
        near = np.minimum(
            np.minimum(d["x"][span] - geom.RAIL_MIN_X,
                       geom.RAIL_MAX_X - d["x"][span]),
            np.minimum(d["y"][span] - geom.RAIL_MIN_Y,
                       geom.RAIL_MAX_Y - d["y"][span]))
        c = int(span[int(np.argmin(near))])
        name, nrm = wall_of(d["x"][c], d["y"][c])
        if name is None:
            continue
        t_c = d["t"][c]
        vin, r_in = clean_velocity(d, lo - 1, -1, t_c, gate)
        vout, r_out = clean_velocity(d, hi + 1, +1, t_c, gate)
        if vin is None or vout is None:
            continue
        if vin @ nrm >= 0 or vout @ nrm <= 0:
            continue
        tan = np.array([-nrm[1], nrm[0]])
        # Spin either side, averaged over the same clean frames. w is already
        # a 6-frame slope of the unwrapped angle, so it smears exactly like
        # the velocity does and needs the same standoff.
        w_in = float(np.nanmean(d["w"][max(0, lo - BRACKET):lo]))
        w_out = float(np.nanmean(d["w"][hi + 1:hi + 1 + BRACKET]))
        if not (np.isfinite(w_in) and np.isfinite(w_out)):
            continue
        rows.append({"lo": int(lo), "hi": int(hi), "c": int(c),
                     "dvt": float((vout - vin) @ tan),
                     "dw": w_out - w_in,
                     "vt_in": float(vin @ tan),
                     "speed_in": float(np.hypot(*vin))})
    if len(rows) < 6:
        return {"n": len(rows), "rows": rows}
    dvt = np.array([r["dvt"] for r in rows])
    dw = np.array([r["dw"] for r in rows])
    slope = float((dvt * dw).sum() / (dvt * dvt).sum())     # through origin
    pred = -2.0 / geom.PUCK_RADIUS_MM
    resid = dw - slope * dvt
    ss = float(((dw - dw.mean()) ** 2).sum())
    return {"n": len(rows), "rows": rows, "slope": slope, "predicted": pred,
            "grip": slope / pred,
            "r2": float(1.0 - (resid ** 2).sum() / ss) if ss > 0 else 0.0}


def fit_paddle(d, events, gate=None):
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
    for lo, hi in events:
        i, j = lo - SKIP, hi + 1 + SKIP
        if i < 0 or j >= n or d["seq"][j] - d["seq"][i] > (j - i) + 4:
            continue
        # The line of centres at CLOSEST APPROACH, not at an arbitrary frame
        # of the event. The two are still closing at the first frame and
        # already separating at the last, so either end tilts the normal and
        # feeds a wrong direction straight into e.
        span = np.arange(lo, hi + 1)
        seps = np.hypot(d["x"][span] - d["mx"][span],
                        d["y"][span] - d["my"][span])
        if np.all(np.isnan(seps)):
            continue
        c = int(span[int(np.nanargmin(seps))])
        sep = float(np.nanmin(seps))
        if sep > CONTACT_MM:
            continue
        nrm = np.array([d["x"][c] - d["mx"][c], d["y"][c] - d["my"][c]])
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm /= ln

        t_c = d["t"][c]
        vin, r_in = clean_velocity(d, lo - 1, -1, t_c, gate)
        vout, r_out = clean_velocity(d, hi + 1, +1, t_c, gate)
        if vin is None or vout is None:
            continue
        if np.hypot(*vin) < MIN_SPEED:
            continue
        # The MALLET either side, the same way and over the same windows. A
        # hand-swung mallet accelerates throughout, so a finite difference
        # across the bracket returns its average speed over 50 ms rather than
        # its speed at contact, and that error goes straight into the
        # relative normal velocity that defines e.
        md = {"t": d["t"], "seq": d["seq"], "x": d["mx"], "y": d["my"]}
        mv_in, _ = clean_velocity(md, lo - 1, -1, t_c, np.inf)
        mv_out, _ = clean_velocity(md, hi + 1, +1, t_c, np.inf)
        if mv_in is None or mv_out is None:
            continue
        if not (np.all(np.isfinite(mv_in)) and np.all(np.isfinite(mv_out))):
            continue

        rel_in = (vin - mv_in) @ nrm
        rel_out = (vout - mv_out) @ nrm
        if rel_in >= 0 or rel_out <= 0:
            continue
        hits.append({"lo": int(lo), "hi": int(hi), "c": int(c),
                     "e": float(-rel_out / rel_in),
                     "speed_in": float(np.hypot(*vin)),
                     "recoil": float(np.hypot(*(mv_out - mv_in)))})
    return hits


# ── Synthetic validation ─────────────────────────────────────────────────
# A fitter that has never been run against a known answer is a number
# generator. This builds recordings whose constants ARE known and checks the
# fits come back with them.
#
# The generator feeds the fitter TRACKER-SHAPED data, not truth: positions
# carry centroid noise and the velocities are the same 6-frame least-squares
# slope PuckTracker uses. That is the whole point. Handing over exact
# velocities would skip the one effect SKIP exists to handle -- the slope
# window smearing across an impact -- and would validate a fitter that does
# not exist.
TRUE_A = 22.0          # mm/s^2 rolling
TRUE_B = 3.5e-5        # 1/mm quadratic drag
TRUE_EN = 0.80         # wall, normal
TRUE_ET = 0.70         # wall, tangential
TRUE_EP = 0.75         # paddle, on RELATIVE normal velocity
MASS_RATIO = 2.0       # mallet / puck
POS_NOISE_MM = 0.35


def _advance(p, v, dt, sub=10):
    """Integrate one frame of drag, substepped."""
    h = dt / sub
    for _ in range(sub):
        s = float(np.hypot(*v))
        if s > 1e-9:
            v = v - (TRUE_A + TRUE_B * s * s) * h * (v / s)
        p = p + v * h
    return p, v


def _bounce_walls(p, v):
    """Reflect off the rails, losing e_n normally and e_t tangentially."""
    r = geom.PUCK_RADIUS_MM
    lo_x, hi_x = geom.RAIL_MIN_X + r, geom.RAIL_MAX_X - r
    lo_y, hi_y = geom.RAIL_MIN_Y + r, geom.RAIL_MAX_Y - r
    if p[0] < lo_x and v[0] < 0:
        p[0] = 2 * lo_x - p[0]; v[0] *= -TRUE_EN; v[1] *= TRUE_ET
    elif p[0] > hi_x and v[0] > 0:
        p[0] = 2 * hi_x - p[0]; v[0] *= -TRUE_EN; v[1] *= TRUE_ET
    if p[1] < lo_y and v[1] < 0:
        p[1] = 2 * lo_y - p[1]; v[1] *= -TRUE_EN; v[0] *= TRUE_ET
    elif p[1] > hi_y and v[1] > 0:
        p[1] = 2 * hi_y - p[1]; v[1] *= -TRUE_EN; v[0] *= TRUE_ET
    return p, v


def _impulse(pp, vp, pm, vm):
    """One elastic-with-loss impulse along the line of centres.

    Applied ONCE per approach, not on every overlapping frame: re-applying it
    while the two still overlap injects energy and turns a 0.75 collision into
    whatever the frame rate says.
    """
    n = pp - pm
    ln = float(np.hypot(*n))
    if ln < 1e-9:
        return vp, vm
    n = n / ln
    rel = float((vp - vm) @ n)
    if rel >= 0:                      # already separating
        return vp, vm
    j = -(1.0 + TRUE_EP) * rel / (1.0 + 1.0 / MASS_RATIO)
    return vp + j * n, vm - (j / MASS_RATIO) * n


def _emit(rows, seq0, xs, ys, mxs, mys, rng, dt):
    """Positions -> a record_puck-shaped block, via the tracker's estimator."""
    from puck_stream import _lsq_slope        # the real slope, not a copy
    xs = np.asarray(xs) + rng.normal(0, POS_NOISE_MM, len(xs))
    ys = np.asarray(ys) + rng.normal(0, POS_NOISE_MM, len(ys))
    ts = np.arange(len(xs)) * dt
    for k in range(len(xs)):
        lo = max(0, k - 5)
        vx = _lsq_slope(ts[lo:k + 1], xs[lo:k + 1])
        vy = _lsq_slope(ts[lo:k + 1], ys[lo:k + 1])
        rows.append({"seq": seq0 + k, "t": float(ts[k]) + seq0 * dt,
                     "x": float(xs[k]), "y": float(ys[k]),
                     "vx": float(vx), "vy": float(vy), "n": 4,
                     "mx": float(mxs[k]), "my": float(mys[k])})
    return seq0 + len(xs) + 50            # a seq gap ends the segment


def _simulate(rng, dt=1 / 200.0):
    """Glides, wall bounces and mallet impacts with known coefficients."""
    rows, seq = [], 0
    far = (geom.RAIL_MIN_X - 500.0, geom.RAIL_MIN_Y - 500.0)   # mallet parked

    # 1. Free glides across open table, no wall, no mallet.
    for _ in range(40):
        p = np.array([rng.uniform(500, 1500), rng.uniform(250, 700)])
        ang = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(250, 6000)
        xs, ys = [], []
        for _k in range(rng.integers(45, 90)):
            xs.append(p[0]); ys.append(p[1])
            p, v = _advance(p, v, dt)
            if not (100 < p[0] < 1900 and 100 < p[1] < 880):
                break
        if len(xs) > MIN_GLIDE + 2 * SKIP:
            seq = _emit(rows, seq, xs, ys, [far[0]] * len(xs),
                        [far[1]] * len(xs), rng, dt)

    # 2. Shots into a rail, entering at a spread of angles and speeds.
    for _ in range(40):
        p = np.array([rng.uniform(700, 1300), rng.uniform(350, 600)])
        ang = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(1500, 7000)
        xs, ys = [], []
        for _k in range(150):
            xs.append(p[0]); ys.append(p[1])
            p, v = _advance(p, v, dt)
            p, v = _bounce_walls(p, v)
        seq = _emit(rows, seq, xs, ys, [far[0]] * len(xs),
                    [far[1]] * len(xs), rng, dt)

    # 3. Mallet impacts. The mallet SWINGS -- a stationary one would let a
    #    fitter that ignores recoil pass, which is the bug being guarded.
    for _ in range(90):
        p = np.array([rng.uniform(600, 1400), rng.uniform(300, 650)])
        ang = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(800, 4000)
        contact = geom.PUCK_RADIUS_MM + geom.MALLET_RADIUS_MM
        # Put the mallet ahead of the puck, closing.
        pm = p + v / np.hypot(*v) * rng.uniform(260.0, 340.0)
        off = rng.uniform(-0.35, 0.35)                 # glancing, not head-on
        c, s = np.cos(off), np.sin(off)
        vm = -np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        vm = vm / np.hypot(*vm) * rng.uniform(200, 2500)
        xs, ys, mxs, mys, hit = [], [], [], [], False
        for _k in range(70):
            xs.append(p[0]); ys.append(p[1])
            mxs.append(pm[0]); mys.append(pm[1])
            p, v = _advance(p, v, dt)
            pm = pm + vm * dt
            if not hit and np.hypot(*(p - pm)) <= contact:
                v, vm = _impulse(p, v, pm, vm)
                hit = True
        if hit:
            seq = _emit(rows, seq, xs, ys, mxs, mys, rng, dt)
    return rows


def selftest() -> int:
    rng = np.random.default_rng(7)
    rows = _simulate(rng)
    d = {k: np.array([r[k] for r in rows], float)
         for k in ("seq", "t", "x", "y", "vx", "vy", "n", "mx", "my")}
    bounds, cuts, _gaps = segment(d)
    print(f"synthetic: {len(rows)} samples, {len(bounds)} segments, "
          f"{len(cuts)} contacts")

    gate = 6.0 * noise_floor(d, bounds)
    fr = fit_friction(d, bounds)
    g = fr.get("drag", {})
    print(f"\nFRICTION   truth a={TRUE_A:.1f} b={TRUE_B:.3e}")
    print(f"           fit   a={g.get('a', float('nan')):.1f} "
          f"b={g.get('b', float('nan')):.3e}   ({len(fr['speeds'])} glides)")
    assert abs(g["b"] - TRUE_B) < 0.15 * TRUE_B, f"drag b {g['b']:.3e}"
    # `a` is only required to be CONSISTENT with truth, not close to it. It is
    # genuinely weakly identified here, and a test that demanded accuracy
    # would either fail forever or be silently loosened until it passed.
    assert abs(g["a"] - TRUE_A) < 2.5 * g["se_a"] + 5.0, (
        f"rolling a {g['a']:.1f} +- {g['se_a']:.1f} vs truth {TRUE_A}")

    walls, _others = fit_bounces(d, contact_events(cuts), gate)
    en = float(np.mean([w["e_normal"] for w in walls]))
    et = float(np.nanmean([w["e_tangential"] for w in walls]))
    print(f"\nWALL       truth e_n={TRUE_EN:.3f} e_t={TRUE_ET:.3f}")
    print(f"           fit   e_n={en:.3f} e_t={et:.3f}   ({len(walls)} contacts)")
    assert abs(en - TRUE_EN) < 0.05, f"e_normal {en:.3f}"
    assert abs(et - TRUE_ET) < 0.08, f"e_tangential {et:.3f}"

    hits = fit_paddle(d, contact_events(cuts), gate)
    ep = float(np.median([h["e"] for h in hits])) if hits else float("nan")
    print(f"\nPADDLE     truth e={TRUE_EP:.3f}  (mallet/puck mass {MASS_RATIO})")
    print(f"           fit   e={ep:.3f}   ({len(hits) if hits else 0} contacts)")
    assert hits and len(hits) >= 20, f"only {len(hits) if hits else 0} contacts"
    assert abs(ep - TRUE_EP) < 0.08, f"paddle e {ep:.3f} vs {TRUE_EP}"

    print("\nselftest PASSED — all three fits recover their truth")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?")
    ap.add_argument("--selftest", action="store_true",
                    help="fit synthetic data with known constants")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.recording:
        ap.error("a recording is required (or --selftest)")

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
            print(f"    a (rolling)  {g['a']:8.1f} +- {g.get('se_a', float('nan')):.1f}"
                  f" mm/s^2   = mu {g['a'] / G_MM_S2:.5f}")
            print(f"    b (drag)     {g['b']:.3e} +- {g.get('se_b', float('nan')):.1e} 1/mm")
            if abs(g["a"]) < 2 * g.get("se_a", np.inf):
                print("    NOTE: `a` is consistent with ZERO at 2 sigma -- it is an")
                print("          intercept extrapolated from glides that are mostly fast,")
                print("          where b*v^2 dwarfs it. Do not quote it as a measured")
                print("          rolling coefficient. To pin it down, record long SLOW")
                print("          glides (200-600 mm/s); that is the only regime where")
                print("          the two terms are comparable.")
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

    # Gate calibrated to THIS recording's own frame-to-frame steadiness, not
    # to a number picked on synthetic data. See noise_floor.
    gate = 6.0 * noise_floor(d, bounds)

    print("\n── WALL RESTITUTION " + "─" * 47)
    walls, others = fit_bounces(d, contact_events(cuts), gate)
    if not walls:
        print("  no clean wall contacts found. Hit each cushion square-on a "
              "few times, at a few speeds.")
    else:
        seen = wall_events(d, contact_events(cuts))
        print(f"  {seen} contacts reached a cushion; {len(walls)} could be "
              f"MEASURED cleanly.")
        if seen > 2 * max(len(walls), 1):
            print(f"  The other {seen - len(walls)} are real bounces with no "
                  f"clean glide either side --")
            print("  another contact within a few frames, so there is nothing "
                  "uncontaminated to")
            print("  bracket. Not a shortage of wall hits; a shortage of "
                  "ISOLATED ones. To pin")
            print("  restitution down, hit the puck at a cushion and then let "
                  "it run untouched")
            print("  for half a second before touching it again.")
        print(f"  ({others} contacts away from any rail -- mallet or hand)\n")
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
            print(f"\n  TANGENTIAL ratio {et.mean():.3f} < 1: the cushion "
                  "takes tangential momentum. WHERE it goes -- into spin, or\n"
                  "  into heat -- this number cannot say, and the two want "
                  "different simulators. See the spin\n"
                  "  section below, which measures it directly now that the "
                  "puck has an orientation.")
    sp = fit_spin(d, contact_events(cuts), gate)
    if sp is not None:
        print("\n── SPIN AT THE CUSHION " + "─" * 44)
        if sp["n"] < 6:
            print(f"  only {sp['n']} clean wall contacts carry spin -- too few "
                  f"to separate grip from rub.")
        else:
            print(f"  {sp['n']} contacts.  dw vs dv_t slope "
                  f"{sp['slope']:+.4f} rad/s per mm/s")
            print(f"  a gripping cushion predicts -2/R = {sp['predicted']:+.4f}"
                  f"  ->  {100 * sp['grip']:.0f}% of it   (r2 {sp['r2']:.2f})")
            if sp["grip"] > 0.5:
                print("  -> the cushion GRIPS: the tangential loss really is going into")
                print("     spin, and the sim needs a puck orientation state.")
            elif sp["grip"] < 0.2:
                print("  -> the cushion RUBS: tangential momentum goes to friction, NOT")
                print("     into spin. A tangential coefficient is the whole model and")
                print("     the sim does NOT need orientation to get bounces right.")
            else:
                print("  -> partial: some of the tangential loss becomes spin.")

    print("\n── PADDLE (MALLET) RESTITUTION " + "─" * 36)
    pad = fit_paddle(d, contact_events(cuts), gate)
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
        if frac > 1.2:
            print("  -> ABOVE 1: the mallet left FASTER than the puck arrived, "
                  "which no free mass\n     struck by a lighter one can do. "
                  "The hand was still driving it THROUGH the\n     contact, so "
                  "this is not a free-body collision and the e above is not a\n"
                  "     material property -- it has the arm's work folded into "
                  "it, which is\n     also why the spread is 0.15-1.04.\n"
                  "     A SWUNG mallet cannot measure paddle restitution. Shoot "
                  "the puck at a\n     mallet held STILL, or better, left "
                  "resting free on the table.")
        elif frac > 0.15:
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
