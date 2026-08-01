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
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_extrinsics import (MARKER_Z_MM, MARKERS_FILE,  # noqa: E402
                                  find_glare, load_intrinsics,
                                  weighted_centroid)
from camera import Stream, backproject_pixels  # noqa: E402

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"

MALLET_Z_MM = 67.0     # marker top above the playing surface
FIELD_REJECT_PX = 14.0  # a blob this close to a known marker is that marker
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
    """Where the permanent markers land, so we can ignore them."""
    obj = np.hstack([field, np.full((len(field), 1), MARKER_Z_MM)])
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


def locate(img, K, dist, rvec, tvec, field):
    """Mallet marker position in grid-frame mm, or None if not found.

    Returns (xy, note) where note flags anything the caller should worry
    about (nothing found, or more than one plausible blob)."""
    known = field_marker_pixels(K, dist, rvec, tvec, field)
    cands = find_candidates(img, known)
    if not cands:
        return None, "no mallet marker found — is it in frame and lit?"
    note = None
    if len(cands) > 1:
        extra = ", ".join(f"({c[0]:.0f},{c[1]:.0f}) area {a}"
                          for a, c in cands[1:])
        note = (f"{len(cands)} candidates; taking the largest. "
                f"Others: {extra}")
    xy = backproject_pixels(np.array([cands[0][1]]), K, dist, rvec, tvec,
                              MALLET_Z_MM)[0]
    return xy, note


def report(xy, note, K, dist, rvec, tvec):
    if note:
        print(f"  NOTE: {note}")
    print(f"\nmallet marker at ({xy[0]:.1f}, {xy[1]:.1f}) mm, grid frame "
          f"(z = {MALLET_Z_MM:.0f} mm plane)")
    # How much the height assumption is worth here, so a wrong number is
    # obvious rather than silent.
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec.reshape(3, 1)).ravel()
    r = float(np.hypot(xy[0] - C[0], xy[1] - C[1]))
    sens = r / (C[2] - MALLET_Z_MM)
    print(f"  height sensitivity here: {sens:.2f} mm of position per mm of "
          f"height error\n  (so the 67 mm figure wants to be right to a "
          f"millimetre or two)")
    print(f"\nPaste into the Teensy monitor to zero the controller here:\n"
          f"  CAL {xy[0]:.1f} {xy[1]:.1f}")


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
    xy, note = locate(img, K, dist, rvec, tvec, field)
    if xy is None:
        raise RuntimeError(note)
    if note:
        raise RuntimeError(
            "ambiguous mallet detection, refusing to guess: " + note)
    return float(xy[0]), float(xy[1])


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
    xy, note = locate(img, K, dist, rvec, tvec, field)
    if xy is None:
        sys.exit(f"  {note}")
    report(xy, note, K, dist, rvec, tvec)


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
            xy, note = locate(img, K, dist, rvec, tvec, field)
            if xy is None:
                sys.stdout.write(f"\r  {note:<70}")
            else:
                sys.stdout.write(
                    f"\r  x {xy[0]:8.1f}  y {xy[1]:8.1f} mm   {fps:5.1f} fps"
                    + ("   (ambiguous)" if note else "            "))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        s.close()


def selftest():
    """Synthesise a frame with the six field markers plus a mallet marker at
    a known spot, and check it is recovered at the right height."""
    K, dist, rvec, tvec, field = load_pose()
    truth = np.array([1400.0, 380.0])

    img = np.zeros((1080, 1440), np.uint8)
    known = field_marker_pixels(K, dist, rvec, tvec, field)
    for p in known:
        cv2.circle(img, tuple(np.round(p).astype(int)), 4, 200, -1)
    obj = np.array([[truth[0], truth[1], MALLET_Z_MM]])
    mp, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    cv2.circle(img, tuple(np.round(mp.reshape(2)).astype(int)), 3, 220, -1)

    xy, note = locate(img, K, dist, rvec, tvec, field)
    assert xy is not None, f"mallet not found: {note}"
    err = float(np.linalg.norm(xy - truth))
    print(f"selftest: recovered ({xy[0]:.1f}, {xy[1]:.1f}) vs truth "
          f"({truth[0]:.1f}, {truth[1]:.1f}), error {err:.2f} mm")
    assert err < 2.0, f"recovery error {err:.2f} mm"
    assert note is None, f"unexpected ambiguity: {note}"

    # Ignoring the height must be clearly WORSE — proves the plane matters.
    flat = backproject_pixels(mp.reshape(1, 2), K, dist, rvec, tvec, 0.0)[0]
    flat_err = float(np.linalg.norm(flat - truth))
    print(f"selftest: same blob back-projected to the TABLE plane would read "
          f"{flat_err:.1f} mm off")
    assert flat_err > 10 * max(err, 0.1), "height correction not doing work"
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
