#!/usr/bin/env python3
"""Measure the four motor anchors from retroreflective markers on the spools.

This supersedes the ellipse fit in measure_motors.py, which had two problems
that this does not:

  * it fitted the spool's TOP-FACE OUTLINE, a dim, partly-occluded disc, and
    called its centre the shaft axis. A marker placed on the axis IS the
    axis, so there is nothing to infer.
  * its height came from a tape measure to the top face and was assumed, not
    checked. Here the marker's own height is calipered, and height is the
    dominant error term: a dh error moves an anchor by 0.42*dh for the
    mid-table pair and 0.84*dh for the robot-end pair.

What it does NOT fix: all four anchors sit OUTSIDE the quadrilateral of the
four markers the extrinsics were fitted on (x 38..1918, y 38..902), so the
distortion model is extrapolating out there. The mallet, measured by this
same pipeline INSIDE that region, agrees with the air-hole grid to 1 mm; the
anchors cannot claim that. Caliper one or two against the hole grid to find
out how much extrapolation error is left — that is the only independent
check available.

The camera can only be held by one process. Turn the tracker view off in the
web UI first, or this will fail to open the device.

Usage:
    python vision/bin/measure_anchors.py --height 33.5
    python vision/bin/measure_anchors.py --height 33.5 --write
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402
from calibrate_extrinsics import find_glare  # noqa: E402
from camera import Stream, backproject_pixels  # noqa: E402
from track_mallet import (FIELD_REJECT_PX, field_marker_pixels,  # noqa: E402
                          load_pose)

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
OUT_JSON = CALIB_DIR / "motor_anchors.json"

# Detection is LOCAL, unlike track_mallet's whole-frame search, because here
# we already know roughly where to look. That matters more than it sounds:
#
#  * the robot-end markers land 34 and 47 px from the frame edge, and
#    track_mallet discards anything within 40 px because off-field
#    reflections hug the border. Knowing which blob we want makes that
#    rejection unnecessary.
#  * they also come back at peak brightness ~130 against ~255 for the
#    mid-table pair, being further from the IR ring and viewed at a steeper
#    angle. A single frame-wide threshold set from the brightest blob
#    (track_mallet uses max/2) erases them. A per-window threshold does not.
MIN_AREA = 4
MAX_AREA = 900


def expected_pixels(K, dist, rvec, tvec, height):
    """Where the CURRENT anchor constants say the markers should appear."""
    obj = np.array([[geom.MOTOR_X[m], geom.MOTOR_Y[m], height]
                    for m in range(4)], dtype=np.float64)
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def find_marker(img, glare, known, centre, radius):
    """Brightest isolated blob in a window around `centre`.

    Returns (centroid_px, area, peak, background) or None. Searching a
    window rather than the frame is what lets the dim, edge-hugging
    robot-end markers be found at all — see the note above.
    """
    h, w = img.shape[:2]
    x0 = int(max(0, centre[0] - radius)); x1 = int(min(w, centre[0] + radius))
    y0 = int(max(0, centre[1] - radius)); y1 = int(min(h, centre[1] + radius))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    win = img[y0:y1, x0:x1].astype(np.int16)

    # Blank out anything we already know is not a spool marker, so it can
    # neither win the peak nor pull the centroid.
    blank = glare[y0:y1, x0:x1] > 0
    for k in known:
        kx, ky = k[0] - x0, k[1] - y0
        if -FIELD_REJECT_PX <= kx < (x1 - x0) + FIELD_REJECT_PX and \
           -FIELD_REJECT_PX <= ky < (y1 - y0) + FIELD_REJECT_PX:
            yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
            blank |= (xx - kx) ** 2 + (yy - ky) ** 2 < FIELD_REJECT_PX ** 2
    win[blank] = 0

    bg = float(np.median(win))
    peak = int(win.max())
    if peak - bg < 25:                     # nothing meaningfully bright here
        return None
    thr = bg + 0.45 * (peak - bg)
    bw = (win > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    py, pxx = np.unravel_index(int(np.argmax(win)), win.shape)
    j = int(lab[py, pxx])
    if j == 0:
        return None
    area = int(stats[j, cv2.CC_STAT_AREA])
    if not (MIN_AREA <= area <= MAX_AREA):
        return None

    # Intensity-weighted centroid over the component, local background
    # removed — the same reason calibrate_extrinsics does it: a binary
    # centroid quantises to whichever pixels happened to clear the
    # threshold, and these markers differ in brightness by 2x.
    m = (lab == j)
    wgt = np.clip(win - bg, 0, None) * m
    tot = wgt.sum()
    yy, xx = np.mgrid[:win.shape[0], :win.shape[1]]
    cx = float((wgt * xx).sum() / tot) + x0
    cy = float((wgt * yy).sum() / tot) + y0
    return np.array([cx, cy]), area, peak, bg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, required=True,
                    help="marker height above the PLAYING SURFACE (mm), "
                         "calipered to the top of the marker")
    ap.add_argument("--frames", type=int, default=25,
                    help="frames to median together per burst (default 25)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent bursts to average (default 5). The "
                         "spread across them is the honest precision of this "
                         "measurement — a single burst repeats to only about "
                         "1.5 mm, which is the same size as the errors being "
                         "chased, so do not trust one.")
    ap.add_argument("--radius", type=float, default=90.0,
                    help="px search radius around the expected position")
    ap.add_argument("--exposure", type=int, default=1000)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--write", action="store_true",
                    help="update vision/calib/motor_anchors.json")
    ap.add_argument("--debug", default=str(CALIB_DIR / "anchors_debug.png"))
    args = ap.parse_args()

    K, dist, rvec, tvec, field = load_pose()
    known = field_marker_pixels(K, dist, rvec, tvec, field)
    want = expected_pixels(K, dist, rvec, tvec, args.height)

    stream = Stream(args.exposure, args.gain)
    bursts = []
    try:
        for _ in range(max(1, args.repeats)):
            frames = []
            deadline = time.time() + 20.0
            while len(frames) < args.frames and time.time() < deadline:
                img = stream.grab()
                if img is not None:
                    frames.append(img)
            if not frames:
                break
            # Median rather than mean: a single frame with a stray reflection
            # then cannot drag a centroid, and the markers are static so
            # there is no motion blur to worry about.
            bursts.append(np.median(np.stack(frames), axis=0).astype(np.uint8))
    finally:
        stream.close()
    if not bursts:
        print("no frames — is the tracker view still holding the camera?")
        return 1
    img = bursts[-1]
    print(f"{len(bursts)} bursts of up to {args.frames} frames each\n")

    # Solve each burst independently and average the RESULTS, so the spread
    # across bursts is a real error bar rather than a claim.
    h, w = img.shape[:2]
    per_burst, last = [], [None] * 4
    for b in bursts:
        glare = find_glare(b)
        px = []
        for m in range(4):
            hit = find_marker(b, glare, known, want[m], args.radius)
            px.append(None if hit is None else hit)
        if any(p is None for p in px):
            continue
        last = px
        per_burst.append(backproject_pixels(
            np.array([p[0] for p in px]), K, dist, rvec, tvec, args.height))

    for m in range(4):
        if last[m] is None:
            print(f"  M{m}: NO MARKER within {args.radius:.0f} px of "
                  f"({want[m][0]:.0f}, {want[m][1]:.0f}). Widen --radius, or "
                  f"the marker is not visible from here.")
        else:
            c, area, peak, bg = last[m]
            edge = min(c[0], c[1], w - c[0], h - c[1])
            warn = "   << within 60 px of the frame edge" if edge < 60 else ""
            print(f"  M{m}: ({c[0]:7.1f}, {c[1]:7.1f}) px   "
                  f"{np.linalg.norm(c - want[m]):5.1f} px from prediction   "
                  f"area {area:3d}  peak {peak:3d} over bg {bg:.0f}{warn}")
    if not per_burst:
        print("\nno burst resolved all four markers")
        return 1
    px = [p[0] for p in last]

    stack = np.stack(per_burst)              # (bursts, 4, 2)
    got = stack.mean(axis=0)
    spread = stack.max(axis=0) - stack.min(axis=0)
    print(f"\n{len(per_burst)}/{len(bursts)} bursts resolved all four; "
          f"spread across them (mm):")
    for m in range(4):
        print(f"   M{m}   x {spread[m][0]:4.1f}   y {spread[m][1]:4.1f}")

    print(f"\nback-projected onto z = {args.height} mm\n")
    print("   m      current (x, y)          measured (x, y)        moved")
    for m in range(4):
        dx = got[m][0] - geom.MOTOR_X[m]
        dy = got[m][1] - geom.MOTOR_Y[m]
        print(f"   {m}   ({geom.MOTOR_X[m]:7.1f}, {geom.MOTOR_Y[m]:8.1f})   "
              f"({got[m][0]:7.1f}, {got[m][1]:8.1f})   "
              f"{math.hypot(dx, dy):5.1f} mm  ({dx:+.1f}, {dy:+.1f})")

    # Against the air-hole grid, which is the frame you can put a caliper on.
    print("\nagainst the hole grid (the independent check):")
    P = geom.GRID_PITCH_MM
    for m in range(4):
        x, y = got[m]
        last_col = round(geom.GRID_X_MM / P)
        last_row = round(geom.GRID_Y_MM / P)
        col = round(x / P)
        along = (f"col {col} {x - col * P:+.1f}" if col <= last_col
                 else f"{x - geom.GRID_X_MM:+.1f} past col {last_col}")
        across = (f"{-y:6.1f} mm outside row 0" if y < 0
                  else f"{y - geom.GRID_Y_MM:6.1f} mm outside row {last_row}")
        print(f"   M{m}   {along:>22}   {across}")

    vis = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=3.0), cv2.COLOR_GRAY2BGR)
    for m in range(4):
        p = tuple(int(round(v)) for v in px[m])
        w = tuple(int(round(v)) for v in want[m])
        cv2.drawMarker(vis, w, (90, 90, 220), cv2.MARKER_TILTED_CROSS, 18, 1)
        cv2.circle(vis, p, 12, (90, 230, 230), 2)
        cv2.putText(vis, f"M{m}", (p[0] + 15, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 230, 230), 2)
    cv2.imwrite(args.debug, vis)
    print(f"\ndebug image: {args.debug}  (yellow = matched, red = predicted)")

    if args.write:
        OUT_JSON.write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "method": "retroreflective marker on the spool axis, "
                      "back-projected onto the calipered marker plane",
            "marker_height_mm": args.height,
            "bursts_averaged": len(per_burst),
            "anchors_mm": {str(m): [round(float(got[m][0]), 1),
                                    round(float(got[m][1]), 1)]
                           for m in range(4)},
            "spread_mm": {str(m): [round(float(spread[m][0]), 2),
                                   round(float(spread[m][1]), 2)]
                          for m in range(4)},
        }, indent=2) + "\n")
        print(f"wrote {OUT_JSON}")
        print("\nPaste into shared/cdpr_geometry.h, then mirror into "
              "shared/cdpr_geometry.py and run shared/check_geometry.py:\n")
        print("constexpr float MOTOR_X[NUM_MOTORS] = {")
        for m in range(4):
            print(f"    {got[m][0]:.1f}f, // {m}")
        print("};\nconstexpr float MOTOR_Y[NUM_MOTORS] = {")
        for m in range(4):
            print(f"    {got[m][1]:.1f}f, // {m}")
        print("};")
        print("\nWRAP_REF_ANGLE is derived from the anchors and HOME — "
              "recompute it in the same edit.")
    else:
        print("\n(nothing written — add --write once the matches look right)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
