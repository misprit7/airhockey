#!/usr/bin/env python3
"""Solving the puck's four-corner marker square. Pure geometry, no camera.

The puck carries FOUR retroreflectors in a square, 21.85 mm from its centre;
a hand-held mallet carries ONE dot. That way round because a player's hand
wraps the mallet and hides whatever is stuck to it, while nothing ever touches
the puck. Three things fall out of the change, and the third is why this file
is more than a rename:

  * a dropout no longer loses the puck -- any three corners still fix it,
  * the centre is the centre, not wherever one sticker happened to be placed,
  * four corners give ORIENTATION, so spin is measured rather than inferred.

WHY A SOLVER AND NOT A CENTROID
    Averaging three of the four corners puts the "centre" 21.85/3 = 7.3 mm
    toward the missing one. At 200 Hz a 7.3 mm step between frames reads as
    1460 mm/s of velocity that never happened, and it appears exactly when a
    corner drops out -- i.e. correlated with glare and with speed, which is
    the worst possible shape for a friction fit. Three corners have an EXACT
    answer instead: the widest pair is the diagonal, and a diagonal's
    midpoint is the centre.

Lives apart from puck_stream because both the blob-stream tracker and the
image-based track_mallet need it, and because it is worth being able to test
the geometry with no camera, no calibration and no blobtrack:

    python vision/bin/puck_markers.py --selftest
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

# All three lengths come from the one measured number in cdpr_geometry.
MARK_R = geom.PUCK_MARKER_R_MM            # 21.85 — corner to centre
MARK_SIDE = geom.PUCK_MARKER_SIDE_MM      # 30.90 — adjacent corners
MARK_DIAG = geom.PUCK_MARKER_DIAG_MM      # 43.70 — opposite corners

# How far a measured corner spacing may sit from the model. Centroid noise is
# ~0.35 mm and 300 us of exposure smears all four corners the same way, so the
# budget here is back-projection error, not the blobs. Held at 5 mm rather
# than something looser because this is also the test that keeps the ROBOT
# mallet out: its markers are a centre plus two at 26.5 mm radius, i.e.
# spacings 26.5 / 26.5 / 53.0, and 53.0 is outside 43.7 +/- 5. Loosen this and
# the mallet starts passing as a puck.
SQUARE_TOL_MM = 5.0

# Single-linkage radius for grouping blobs into objects. It has to reach the
# DIAGONAL so a puck showing only its two opposite corners stays one group,
# and it stays well short of the ~69 mm from a puck corner to the mallet's
# centre dot at the moment of contact — the closest the two ever get.
LINK_MM = MARK_DIAG + SQUARE_TOL_MM

# More blobs than this in one group is glare, not a puck. Also bounds the
# 4-subset search in find_puck.
MAX_GROUP = 8


def _pdist(pts):
    """Pairwise distances in index order (0,1), (0,2), (1,2), ..."""
    n = len(pts)
    return np.array([float(np.linalg.norm(pts[i] - pts[j]))
                     for i in range(n) for j in range(i + 1, n)])


def groups(world, link_mm=LINK_MM):
    """Single-linkage grouping of points into objects. List of index arrays."""
    world = np.asarray(world, float)
    n = len(world)
    adj = np.linalg.norm(world[:, None, :] - world[None, :, :],
                         axis=2) <= link_mm
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        comp = np.zeros(n, bool)
        comp[i] = True
        while True:
            grown = comp | adj[comp].any(axis=0)
            if grown.sum() == comp.sum():
                break
            comp = grown
        seen |= comp
        out.append(np.flatnonzero(comp))
    return out


def solve_square(pts, prev=None):
    """Centre of the marker square from 2-4 of its corners, or None.

    NEVER the mean of the corners unless all four are there — see the module
    docstring for the 7.3 mm that costs.
    """
    pts = np.asarray(pts, float)
    n = len(pts)
    if n == 4:
        d = np.sort(_pdist(pts))
        model = np.array([MARK_SIDE] * 4 + [MARK_DIAG] * 2)
        rms = float(np.sqrt(((d - model) ** 2).mean()))
        return (pts.mean(axis=0), rms) if rms <= SQUARE_TOL_MM else None
    if n == 3:
        d = _pdist(pts)
        k = int(np.argmax(d))                       # the diagonal
        rest = [d[m] for m in range(3) if m != k]
        rms = float(np.sqrt(((d[k] - MARK_DIAG) ** 2
                             + (rest[0] - MARK_SIDE) ** 2
                             + (rest[1] - MARK_SIDE) ** 2) / 3.0))
        if rms > SQUARE_TOL_MM:
            return None
        i, j = [(0, 1), (0, 2), (1, 2)][k]
        return 0.5 * (pts[i] + pts[j]), rms
    if n == 2:
        d = float(np.linalg.norm(pts[0] - pts[1]))
        mid = 0.5 * (pts[0] + pts[1])
        if abs(d - MARK_DIAG) <= SQUARE_TOL_MM:
            return mid, abs(d - MARK_DIAG)
        if abs(d - MARK_SIDE) <= SQUARE_TOL_MM and prev is not None:
            # Adjacent corners. The centre is off the segment by
            # sqrt(R^2 - (d/2)^2) — 15.4 mm — on one of two sides, and
            # nothing in THIS frame says which. The last known position does.
            u = (pts[1] - pts[0]) / d
            perp = np.array([-u[1], u[0]])
            h = math.sqrt(max(MARK_R ** 2 - (d / 2.0) ** 2, 0.0))
            a, b = mid + h * perp, mid - h * perp
            prev = np.asarray(prev, float)
            c = a if np.hypot(*(a - prev)) < np.hypot(*(b - prev)) else b
            return c, abs(d - MARK_SIDE)
    return None


def square_angle(pts, centre):
    """Square orientation MODULO 90 degrees, radians, from any corners.

    Averaged in the 4-theta domain because the square is invariant under a
    quarter turn: individual corner angles are only comparable once
    multiplied by four, and averaging them directly straddles the branch cut.
    """
    v = np.asarray(pts, float) - np.asarray(centre, float)
    a = np.arctan2(v[:, 1], v[:, 0])
    return 0.25 * math.atan2(float(np.sin(4 * a).mean()),
                             float(np.cos(4 * a).mean()))


def find_puck(world, prev=None):
    """Locate the puck's marker square. (centre, theta, member_idx, rms)."""
    world = np.asarray(world, float)
    if len(world) < 2:
        return None
    best = None
    for g in groups(world):
        if len(g) < 2 or len(g) > MAX_GROUP:
            continue
        # A stray blob inside the group (a glare speck, a rail glint) would
        # break an all-members fit, so once there are more than four try
        # every 4-subset and let the best one speak.
        subsets = [g] if len(g) <= 4 else [np.array(c)
                                           for c in combinations(g, 4)]
        for s in subsets:
            fit = solve_square(world[s], prev)
            if fit is None:
                continue
            c, rms = fit
            # More corners first, then a tighter fit. A real four-corner
            # square beats a two-corner coincidence even when the pair
            # happens to measure closer to the model.
            key = (-len(s), rms)
            if best is None or key < best[0]:
                best = (key, c, s, rms)
    if best is None:
        return None
    _k, c, s, rms = best
    return c, square_angle(world[s], c), s, rms


def _selftest() -> int:
    """The square solver, on synthetic corners. No camera, no blobtrack."""
    rng = np.random.default_rng(0)

    def corners(cx, cy, th, keep=(0, 1, 2, 3)):
        a = th + np.arange(4) * (math.pi / 2)
        p = np.stack([cx + MARK_R * np.cos(a), cy + MARK_R * np.sin(a)], 1)
        return p[list(keep)]

    truth = np.array([812.0, 431.0])
    th = 0.31

    # 1. All four corners.
    c, rms = solve_square(corners(*truth, th))
    assert np.allclose(c, truth, atol=1e-9), c
    assert rms < 1e-9, rms
    got = square_angle(corners(*truth, th), c)
    assert abs((got - th + math.pi / 4) % (math.pi / 2) - math.pi / 4) < 1e-9

    # 2. THE ONE THAT MATTERS: three corners must still be exact, and the
    #    naive mean must not be — that 7.3 mm is why this module exists.
    three = corners(*truth, th, keep=(0, 1, 2))
    c, _ = solve_square(three)
    assert np.allclose(c, truth, atol=1e-9), c
    naive = np.hypot(*(three.mean(axis=0) - truth))
    assert abs(naive - MARK_R / 3.0) < 1e-9, naive

    # 3. Two OPPOSITE corners: the midpoint is the centre.
    c, _ = solve_square(corners(*truth, th, keep=(0, 2)))
    assert np.allclose(c, truth, atol=1e-9), c

    # 4. Two ADJACENT corners are two-valued; the previous position decides.
    two = corners(*truth, th, keep=(0, 1))
    assert solve_square(two, prev=None) is None, "ambiguous pair accepted"
    c, _ = solve_square(two, prev=truth + [3.0, -2.0])
    assert np.allclose(c, truth, atol=1e-9), c
    mirror = 2 * two.mean(axis=0) - truth          # the wrong solution
    c, _ = solve_square(two, prev=mirror + [1.0, 1.0])
    assert np.allclose(c, mirror, atol=1e-9), "continuity ignored"

    # 5. One corner is not a fix.
    assert solve_square(corners(*truth, th, keep=(0,))) is None

    # 6. The ROBOT mallet — a centre marker plus two at 26.5 mm radius —
    #    must not read as a puck. It fails on the 53 mm diagonal.
    r = geom.ARM_MARKER_R_MM
    mallet = np.array([[900.0, 700.0], [900.0 - r, 700.0], [900.0 + r, 700.0]])
    assert solve_square(mallet) is None, "robot mallet passed as a puck"
    assert find_puck(mallet) is None

    # 7. Puck plus the hand mallet's lone dot: pick the square, drop the dot.
    world = np.vstack([corners(*truth, th), [[400.0, 200.0]]])
    c, _ang, members, _ = find_puck(world)
    assert len(members) == 4 and 4 not in members, members
    assert np.allclose(c, truth, atol=1e-9), c

    # 8. With realistic centroid noise the centre still lands inside a
    #    millimetre, which is the scale the velocity fit cares about.
    worst = 0.0
    for _ in range(400):
        cx, cy = rng.uniform(200, 1800), rng.uniform(100, 900)
        p = corners(cx, cy, rng.uniform(-math.pi, math.pi))
        p += rng.normal(0.0, 0.5, p.shape)
        got = solve_square(p)
        assert got is not None, "noise rejected a real square"
        worst = max(worst, float(np.hypot(*(got[0] - [cx, cy]))))
    assert worst < 1.0, worst

    print(f"selftest PASSED — square r={MARK_R} mm, side {MARK_SIDE:.2f}, "
          f"diagonal {MARK_DIAG:.2f}")
    print(f"  3-corner solve exact; naive mean would be {MARK_R / 3.0:.2f} mm "
          f"out ({MARK_R / 3.0 / 0.005:.0f} mm/s of phantom velocity "
          f"at 200 Hz)")
    print(f"  worst centre error over 400 poses at 0.5 mm noise: "
          f"{worst:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
