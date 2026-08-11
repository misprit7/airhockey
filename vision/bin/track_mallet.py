#!/usr/bin/env python3
"""Where is the mallet?

A retroreflective marker sits on top of the mallet. Under the IR ring at
short exposure it reads as a bright blob alongside the six permanent field
markers; this finds it, rejects the known ones, and back-projects it
through the calibrated camera pose.

HEIGHT IS NOT OPTIONAL. The marker rides 67 mm above the playing surface,
and an elevated point projects AWAY from the camera nadir. Back-projecting
it onto the table plane instead of its own plane would put it up to ~46 mm
off near the table ends — larger than the whole calibration error budget.
That is why this is a separate step from the field-plane solve.

The measured point is the MARKER, which equals the mallet centre only if
the marker is centred on the mallet. Any offset shows up as a constant
lever-arm error that rotates with the mallet.

Primary use: tell the controller where the mallet actually is at startup,
instead of assuming it was placed at a nominal point. Prints the exact
`CAL x y` line to paste into the Teensy monitor.

Usage:
    python bin/track_mallet.py                 # one shot, prints CAL line
    python bin/track_mallet.py --watch         # continuous live readout
    python bin/track_mallet.py --image f.png   # offline, from a file
    python bin/track_mallet.py --selftest
"""

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
from calibrate_extrinsics import (MARKER_Z_MM, MARKERS_FILE,  # noqa: E402
                                  find_glare, load_intrinsics,
                                  weighted_centroid)
from camera import Stream, backproject_pixels  # noqa: E402

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"

# Three retroreflectors on the paddle, on TWO different planes — each is
# back-projected onto its own, because an 18 mm height error skews the
# result rather than merely displacing it.
MALLET_Z_MM = 67.0      # centre marker, on top of the paddle
ARM_Z_MM = 49.0         # the two arm markers, lower down on the cross
SPOOL_MARKER_Z_MM = 33.5  # retroreflectors on the four spool axes
ARM_SPAN_DEG = 90.0     # arms 0 and 3 are adjacent, so 90 deg apart
CLUSTER_PX = 60.0       # paddle markers sit within this of each other
# A blob this close to a known permanent marker IS that marker. Generous
# enough to cover projection error at the anchors, which sit outside the
# region the extrinsics were fitted on, and still far short of the ~35 px
# that separates the paddle's own markers from each other.
FIELD_REJECT_PX = 20.0
MIN_AREA = 6
MAX_AREA = 600
BORDER_PX = 40         # off-field reflections hug the frame edge


def load_pose():
    K, dist = load_intrinsics()
    f = CALIB_DIR / "extrinsics.npz"
    if not f.exists():
        sys.exit("extrinsics.npz missing — run calibrate_extrinsics.py first")
    d = np.load(f)
    if not MARKERS_FILE.exists():
        sys.exit(f"{MARKERS_FILE} missing — run calibrate_extrinsics.py first")
    with open(MARKERS_FILE) as fh:
        field = np.array(json.load(fh)["markers_mm"], dtype=np.float64)
    return K, dist, d["rvec"], d["tvec"], field


def field_marker_pixels(K, dist, rvec, tvec, field):
    """Where the permanent retroreflectors land, so we can ignore them.

    Both sets: the six on the playing surface AND the four on the spool
    axes. The spool markers are there so the anchors can be re-measured
    (vision/bin/measure_anchors.py), and blob-for-blob they are
    indistinguishable from a paddle marker — the mid-table pair in
    particular lands ~110 px inside the frame, well clear of the border
    filter that catches off-field junk. Left in, they get counted as paddle
    candidates and the pose solve returns nonsense.

    What separates them is that they are bolted down and their positions
    are known to well under a millimetre, so projecting them costs nothing
    and removes the ambiguity outright.
    """
    obj = np.vstack([
        np.hstack([np.asarray(field, float),
                   np.full((len(field), 1), MARKER_Z_MM)]),
        np.array([[geom.MOTOR_X[m], geom.MOTOR_Y[m], SPOOL_MARKER_Z_MM]
                  for m in range(len(geom.MOTOR_X))], dtype=float),
    ])
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def find_candidates(img, known_px):
    """Bright blobs that are not glare, not a permanent marker, not an
    off-field edge reflection. Returns [(area, centroid), ...] brightest
    first."""
    mask = find_glare(img)
    work = img.copy()
    work[mask > 0] = 0
    thr = max(64, int(work.max()) // 2)
    _, bw = cv2.threshold(work, thr, 255, cv2.THRESH_BINARY)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw)

    out = []
    for j in range(1, n):
        area = stats[j, cv2.CC_STAT_AREA]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        c = weighted_centroid(work, stats, j)
        c = cents[j] if c is None else c
        if not (BORDER_PX <= c[0] < img.shape[1] - BORDER_PX
                and BORDER_PX <= c[1] < img.shape[0] - BORDER_PX):
            continue
        if len(known_px) and np.min(np.linalg.norm(known_px - c,
                                                   axis=1)) < FIELD_REJECT_PX:
            continue  # one of the permanent field markers
        out.append((int(area), c))
    return sorted(out, key=lambda b: -b[0])


def solve_pose(cands, K, dist, rvec, tvec):
    """Centre, orientation and quality from the three paddle markers.

    Which blob is the centre is not assumed — every assignment is scored on
    two things the cross guarantees: the two arm markers sit at equal radius
    from the centre, and 90 deg apart. The wrong centre gets the angle badly
    wrong (about 50 deg), so this is unambiguous rather than a close call.
    """
    px = np.array([c[1] for c in cands])
    best = None
    for ci in range(3):
        ai, bi = [k for k in range(3) if k != ci]
        c = backproject_pixels(px[ci:ci + 1], K, dist, rvec, tvec,
                               MALLET_Z_MM)[0]
        arms = backproject_pixels(px[[ai, bi]], K, dist, rvec, tvec, ARM_Z_MM)
        r = [float(np.linalg.norm(a - c)) for a in arms]
        bear = [math.degrees(math.atan2(a[1] - c[1], a[0] - c[0])) for a in arms]
        inc = (bear[1] - bear[0] + 540) % 360 - 180
        score = abs(abs(inc) - ARM_SPAN_DEG) + 2 * abs(r[0] - r[1])
        if best is None or score < best["score"]:
            best = {"score": score, "centre": c, "arms": arms, "r": r,
                    "bear": bear, "inc": inc}

    # Arm 0 lies along theta, arm 3 along theta+90 (chirality -1). Try both
    # labellings; the right one has the two estimates agreeing.
    out = None
    for a0, a3 in ((0, 1), (1, 0)):
        t0 = best["bear"][a0]
        t3 = best["bear"][a3] - ARM_SPAN_DEG
        dis = (t0 - t3 + 540) % 360 - 180
        if out is None or abs(dis) < abs(out["disagree"]):
            out = {"theta": math.radians((t0 - dis / 2 + 360) % 360),
                   "disagree": dis, "arm0": a0}
    best.update(out)
    best["arm_r"] = float(np.mean(best["r"]))
    return best


def locate(img, K, dist, rvec, tvec, field):
    """Paddle pose in the grid frame, or None if it cannot be resolved.

    Returns (pose, note); pose is a dict with centre/theta, note flags
    anything the caller should not trust."""
    known = field_marker_pixels(K, dist, rvec, tvec, field)
    cands = find_candidates(img, known)
    if not cands:
        return None, "no paddle markers found — is it in frame and lit?"

    # Keep only the tight cluster: paddle markers are close together, stray
    # reflections are not.
    pts = np.array([c[1] for c in cands])
    if len(cands) > 3:
        centre = np.median(pts, axis=0)
        keep = np.argsort(np.linalg.norm(pts - centre, axis=1))[:3]
        cands = [cands[i] for i in sorted(keep)]
        pts = np.array([c[1] for c in cands])

    if len(cands) < 3:
        if len(cands) == 1:
            xy = backproject_pixels(pts, K, dist, rvec, tvec, MALLET_Z_MM)[0]
            return ({"centre": xy, "theta": None, "disagree": None,
                     "arm_r": None},
                    "only one paddle marker visible — position only, no "
                    "orientation")
        return None, f"found {len(cands)} paddle markers, need 3"

    spread = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    if spread > CLUSTER_PX:
        return None, (f"paddle markers span {spread:.0f}px, wider than "
                      f"{CLUSTER_PX:.0f} — a stray reflector is probably "
                      "being counted")

    pose = solve_pose(cands, K, dist, rvec, tvec)
    note = None
    if abs(pose["disagree"]) > 5.0:
        note = (f"the two arms disagree on orientation by "
                f"{pose['disagree']:.1f} deg — check the marker heights "
                f"({MALLET_Z_MM:.0f}/{ARM_Z_MM:.0f} mm)")
    return pose, note


def report(pose, note, K, dist, rvec, tvec):
    xy = pose["centre"]
    if note:
        print(f"  NOTE: {note}")
    print(f"\npaddle centre ({xy[0]:.1f}, {xy[1]:.1f}) mm, grid frame")
    if pose["theta"] is not None:
        print(f"orientation   {math.degrees(pose['theta']):.2f} deg"
              f"   (arms agree to {abs(pose['disagree']):.2f} deg, "
              f"marker radius {pose['arm_r']:.2f} mm)")
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec.reshape(3, 1)).ravel()
    r = float(np.hypot(xy[0] - C[0], xy[1] - C[1]))
    print(f"  height sensitivity here: {r / (C[2] - MALLET_Z_MM):.2f} mm of "
          f"position per mm of height error")
    print("\nPaste into the Teensy monitor to zero the controller here:")
    if pose["theta"] is None:
        print(f"  CAL {xy[0]:.1f} {xy[1]:.1f}")
    else:
        print(f"  CAL {xy[0]:.1f} {xy[1]:.1f} {math.degrees(pose['theta']):.2f}")


def measure(image=None, n_frames=5):
    """Measure the mallet once. Returns (x_mm, y_mm).

    Importable entry point for callers that need the startup position —
    notably the web server, which must not assume the mallet is parked at
    the centre of the robot half. Raises RuntimeError if it cannot get a
    confident reading, because a wrong starting position offsets every
    subsequent command by that error.

    NOTE: this opens the camera, and only one process can hold the
    Spinnaker device. Call it while nothing else is streaming.
    """
    K, dist, rvec, tvec, field = load_pose()
    if image is not None:
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cannot read {image}")
    else:
        s = Stream(1000, 0)
        try:
            frames = [f.astype(np.float32) for f in
                      (s.grab() for _ in range(n_frames)) if f is not None]
        finally:
            s.close()
        if not frames:
            raise RuntimeError("no frames from the camera")
        img = np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)
    pose, note = locate(img, K, dist, rvec, tvec, field)
    if pose is None:
        raise RuntimeError(note)
    if note:
        raise RuntimeError(
            "ambiguous paddle detection, refusing to guess: " + note)
    return (float(pose["centre"][0]), float(pose["centre"][1]),
            float(pose["theta"]))


def run_once(image, K, dist, rvec, tvec, field):
    if image:
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"cannot read {image}")
    else:
        s = Stream(1000, 0)
        try:
            frames = []
            for _ in range(5):
                f = s.grab()
                if f is not None:
                    frames.append(f.astype(np.float32))
            if not frames:
                sys.exit("no frames from the camera")
            # Average a few: the marker is static during calibration and
            # this cuts centroid noise without costing anything.
            img = np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)
        finally:
            s.close()
    pose, note = locate(img, K, dist, rvec, tvec, field)
    if pose is None:
        sys.exit(f"  {note}")
    report(pose, note, K, dist, rvec, tvec)


def run_watch(K, dist, rvec, tvec, field):
    s = Stream(1000, 0)
    print("live mallet tracking — Ctrl+C to stop\n")
    last = time.time()
    fps = 0.0
    try:
        while True:
            img = s.grab()
            if img is None:
                continue
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-6, now - last)
            last = now
            pose, note = locate(img, K, dist, rvec, tvec, field)
            if pose is None:
                sys.stdout.write(f"\r  {note:<78}")
            else:
                xy = pose["centre"]
                th = ("     ---" if pose["theta"] is None
                      else f"{math.degrees(pose['theta']):7.2f}")
                sys.stdout.write(
                    f"\r  x {xy[0]:8.1f}  y {xy[1]:8.1f} mm   theta {th} deg"
                    f"   {fps:5.1f} fps" + ("  (!)" if note else "     "))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        s.close()


def selftest():
    """Render the three paddle markers at a known pose and recover it."""
    global ARM_Z_MM
    K, dist, rvec, tvec, field = load_pose()
    truth_xy = np.array([1400.0, 380.0])
    truth_th = math.radians(122.0)
    arm_r = 25.0

    img = np.zeros((1080, 1440), np.uint8)
    for p in field_marker_pixels(K, dist, rvec, tvec, field):
        cv2.circle(img, tuple(np.round(p).astype(int)), 4, 200, -1)

    def stamp(xy, z):
        pr, _ = cv2.projectPoints(np.array([[xy[0], xy[1], z]]), rvec, tvec,
                                  K, dist)
        cv2.circle(img, tuple(np.round(pr.reshape(2)).astype(int)), 3, 220, -1)

    stamp(truth_xy, MALLET_Z_MM)
    for lbl, ang in ((0, truth_th), (3, truth_th + math.pi / 2)):
        stamp(truth_xy + arm_r * np.array([math.cos(ang), math.sin(ang)]),
              ARM_Z_MM)

    pose, note = locate(img, K, dist, rvec, tvec, field)
    assert pose is not None, f"paddle not found: {note}"
    e_xy = float(np.linalg.norm(pose["centre"] - truth_xy))
    e_th = abs(math.degrees(pose["theta"] - truth_th))
    e_th = min(e_th, 360 - e_th)
    print(f"selftest: centre error {e_xy:.2f} mm, orientation error "
          f"{e_th:.2f} deg, arm radius {pose['arm_r']:.2f} (truth {arm_r})")
    assert e_xy < 2.0, f"centre error {e_xy:.2f} mm"
    assert e_th < 3.0, f"orientation error {e_th:.2f} deg"
    assert note is None, f"unexpected note: {note}"

    # Orientation turns out to be ROBUST to the arm height — both arms
    # shift together, so the bearing barely rotates. What the height does
    # move is the apparent arm radius, which makes that radius the useful
    # cross-check: caliper the real thing and compare.
    r_at = {}
    keep = ARM_Z_MM
    try:
        for z in (ARM_Z_MM, MALLET_Z_MM):
            ARM_Z_MM = z
            r_at[z] = locate(img, K, dist, rvec, tvec, field)[0]["arm_r"]
    finally:
        ARM_Z_MM = keep
    drift = abs(r_at[MALLET_Z_MM] - r_at[keep])
    print(f"selftest: assuming the arms sit on the centre plane instead "
          f"shifts the measured radius {r_at[keep]:.2f} -> "
          f"{r_at[MALLET_Z_MM]:.2f} mm ({drift:.2f})")
    assert drift > 1.0, "arm height is not affecting the radius — suspect"
    print("selftest PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="continuous readout")
    ap.add_argument("--image", help="read a saved frame instead of the camera")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    K, dist, rvec, tvec, field = load_pose()
    if args.watch:
        run_watch(K, dist, rvec, tvec, field)
    else:
        run_once(args.image, K, dist, rvec, tvec, field)


if __name__ == "__main__":
    main()
