#!/usr/bin/env python3
"""Solve motor anchor positions from caliper distances to air holes.

Why distances and not offsets: locating an anchor by its x and y offsets
means measuring along axes that are not marked on anything, 100+ mm outside
the rails, with nothing to square against. Measuring the straight-line
distance between two points needs no squareness, no reference edge, and no
guess about which way y runs — you just span it. Three distances to holes
that are not collinear fix the position outright, and the third one turns a
bare solution into a check: if the measurements disagree, the residuals say
so instead of the error hiding in the answer.

This supersedes the optical route for the anchors (vision/bin/
measure_anchors.py). The camera pins down the RAY each anchor sits on very
well, but where it lies along that ray comes entirely from an assumed plane
height, and the anchors are the one thing measured off the table with no way
to cross-check that height — M1 and M2 also fall outside the radius the
intrinsics were calibrated to. Calipers have neither problem.

The air-hole grid is the reference frame: hole (col, row) is at
(col * 25.4, row * 25.4) mm, and the origin hole is the corner nearest the
human player's right.

Measure to the spool AXIS. With a pulley fitted the shaft is buried, so span
to both edges of the pulley along the same line and halve — the midpoint is
the axis, and the difference cross-checks the pulley diameter for free.

Usage:
    # motor 2: 131 mm from the corner hole, 265 mm from 6 holes along y,
    #          259 mm from 6 holes along x
    python shared/fit_anchors.py 2:77,0,131 2:77,6,265 2:71,0,259
    python shared/fit_anchors.py --write 2:77,0,131 2:77,6,265 2:71,0,259
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cdpr_geometry as geom  # noqa: E402

ANCHORS_JSON = (Path(__file__).resolve().parent.parent
                / "vision" / "calib" / "motor_anchors.json")


def parse(tokens):
    """'M:col,row,dist' -> {motor: [(x_mm, y_mm, dist), ...]}"""
    out: dict[int, list] = {}
    for t in tokens:
        try:
            m, rest = t.split(":")
            col, row, dist = rest.split(",")
            m, col, row, dist = int(m), float(col), float(row), float(dist)
        except ValueError:
            sys.exit(f"cannot parse {t!r} — expected M:col,row,distance_mm")
        if not 0 <= m < 4:
            sys.exit(f"motor {m} out of range in {t!r}")
        out.setdefault(m, []).append(
            (col * geom.GRID_PITCH_MM, row * geom.GRID_PITCH_MM, dist))
    return out


def solve(obs, guess):
    """Least-squares position from distances to known points."""
    pts = np.array([[x, y] for x, y, _ in obs], dtype=float)
    d = np.array([v for _, _, v in obs], dtype=float)

    def resid(p):
        return np.linalg.norm(pts - p, axis=1) - d

    # Two references leave a mirror pair about the line joining them; seed
    # from both sides and keep whichever lands nearer where we already think
    # the anchor is. Three non-collinear references remove the ambiguity, so
    # this only matters when someone measured the minimum.
    best = None
    for flip in (1.0, -1.0):
        seed = np.array(guess, dtype=float)
        if flip < 0:
            mid = pts.mean(axis=0)
            seed = mid + (seed - mid) * -1.0
        s = least_squares(resid, seed)
        near = np.linalg.norm(s.x - np.array(guess))
        if best is None or (s.cost, near) < best[0]:
            best = ((s.cost, near), s.x)
    return best[1], resid(best[1])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurements", nargs="+", metavar="M:col,row,dist")
    ap.add_argument("--write", action="store_true",
                    help="update vision/calib/motor_anchors.json and print "
                         "the constants for the header and the mirror")
    args = ap.parse_args()

    groups = parse(args.measurements)
    anchors = {m: [geom.MOTOR_X[m], geom.MOTOR_Y[m]] for m in range(4)}

    print(f"grid pitch {geom.GRID_PITCH_MM} mm; hole (col,row) at "
          f"(col*pitch, row*pitch)\n")
    ok = True
    for m in sorted(groups):
        obs = groups[m]
        if len(obs) < 2:
            print(f"M{m}: only {len(obs)} measurement — need at least 2 "
                  f"(3 to be unambiguous and checkable)")
            ok = False
            continue
        p, r = solve(obs, anchors[m])
        rms = float(np.sqrt((r ** 2).mean()))
        moved = math.hypot(p[0] - anchors[m][0], p[1] - anchors[m][1])
        print(f"M{m}: {len(obs)} measurements -> ({p[0]:8.2f}, {p[1]:8.2f})"
              f"   moves {moved:5.2f} mm from the current value")
        for (x, y, d), e in zip(obs, r):
            print(f"      hole ({x / geom.GRID_PITCH_MM:4.0f},"
                  f"{y / geom.GRID_PITCH_MM:4.0f})  measured {d:7.1f}  "
                  f"residual {e:+6.2f} mm")
        print(f"      rms {rms:.2f} mm"
              + ("" if len(obs) > 2 else
                 "   (2 points fit exactly — this rms means nothing)"))
        if len(obs) > 2 and rms > 2.0:
            print("      !! residuals are large — one measurement disagrees "
                  "with the others; re-check before trusting this")
            ok = False
        across = (f"{-p[1]:.1f} mm outside row 0" if p[1] < 0
                  else f"{p[1] - geom.GRID_Y_MM:.1f} mm outside row 37")
        print(f"      against the grid: {across}, "
              f"{p[0] - geom.GRID_X_MM:+.1f} mm past col 77\n")
        anchors[m] = [round(float(p[0]), 1), round(float(p[1]), 1)]

    unmeasured = [m for m in range(4) if m not in groups]
    if unmeasured:
        print("not measured, left as they are: "
              + ", ".join(f"M{m}" for m in unmeasured) + "\n")

    hx = (geom.WS_MIN_X + geom.WS_MAX_X) / 2.0
    hy = (geom.WS_MIN_Y + geom.WS_MAX_Y) / 2.0
    wrap = [math.atan2(hy - anchors[m][1], hx - anchors[m][0])
            for m in range(4)]

    if not args.write:
        print("(nothing written — add --write once the residuals look right)")
        return 0 if ok else 1

    doc = {}
    if ANCHORS_JSON.exists():
        doc = json.loads(ANCHORS_JSON.read_text())
    doc.update({
        "method": "caliper distances to air holes, trilaterated "
                  "(shared/fit_anchors.py)",
        "anchors_mm": {str(m): anchors[m] for m in range(4)},
        "measurements": {str(m): [[o[0] / geom.GRID_PITCH_MM,
                                   o[1] / geom.GRID_PITCH_MM, o[2]]
                                  for o in groups[m]] for m in sorted(groups)},
    })
    doc.pop("marker_height_mm", None)
    doc.pop("marker_height_note", None)
    ANCHORS_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {ANCHORS_JSON}\n")

    print("shared/cdpr_geometry.h:\n")
    print("constexpr float MOTOR_X[NUM_MOTORS] = {")
    for m in range(4):
        print(f"    {anchors[m][0]:.1f}f, // {m}")
    print("};\nconstexpr float MOTOR_Y[NUM_MOTORS] = {")
    for m in range(4):
        print(f"    {anchors[m][1]:.1f}f, // {m}")
    print("};\nconstexpr float WRAP_REF_ANGLE[NUM_MOTORS] = {")
    for m in range(4):
        print(f"    {wrap[m]:.6f}f, // {m}")
    print("};\n")
    print("shared/cdpr_geometry.py:\n")
    print(f"MOTOR_X = {[a[0] for a in anchors.values()]}")
    print(f"MOTOR_Y = {[a[1] for a in anchors.values()]}")
    print(f"WRAP_REF_ANGLE = {[round(w, 6) for w in wrap]}")
    print("\nthen: python shared/check_geometry.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
