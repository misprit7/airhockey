#!/usr/bin/env python3
"""Solve the extrinsics twice — once per marker set — and compare.

For the mounted-to-sticker changeover, while BOTH sets are physically on the
table. The mounted markers sat in known grid cells (CNC-truth) but stood 3.3
mm proud, so the puck hit them. The stickers lie flat but are referenced to
the walls, which are neither straight nor squarely placed around the grid.
This measures what that trade actually costs.

Read-only: solves, reports, saves nothing. Run calibrate_extrinsics.py once
you are happy with the sticker set.

The two sets are 28-71 mm apart on the table (>=20 px), so each blob is
assigned to whichever set's PREDICTED position it lands nearest, using the
existing pose as the prior. A blob that is ambiguous, or a set that comes up
short, fails loudly rather than quietly calibrating against the wrong points.

Usage:
    python vision/bin/compare_marker_sets.py --capture 8
    python vision/bin/compare_marker_sets.py vision/extr_shots/*.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_extrinsics import (CALIB_DIR, capture, find_blobs,  # noqa: E402
                                  find_glare, load_intrinsics, solve)
from camera import backproject_undistorted  # noqa: E402
from table_grid import (CENTERLINE_X, GRID_X_MM, GRID_Y_MM,  # noqa: E402
                        MOUNTED_MARKER_Z_MM, MOUNTED_MARKERS_XY,
                        STICKER_MARKER_Z_MM, STICKER_MARKERS_XY)

SETS = [
    ("mounted (grid cells, 3.3mm proud)", MOUNTED_MARKERS_XY, MOUNTED_MARKER_Z_MM),
    ("sticker (wall-referenced, flat)", STICKER_MARKERS_XY, STICKER_MARKER_Z_MM),
]


def prior_pose():
    f = CALIB_DIR / "extrinsics.npz"
    if not f.exists():
        sys.exit("no existing extrinsics.npz — need a prior pose to tell the "
                 "two marker sets apart")
    d = np.load(f)
    return d["rvec"], d["tvec"]


def predict(markers, z, K, dist, rvec, tvec):
    obj = np.hstack([np.array(markers, float), np.full((len(markers), 1), z)])
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def assign(blobs, want, radius, label):
    """Nearest predicted position wins, with a hard ambiguity check."""
    pts = np.array([b[1] for b in blobs], float)
    chosen, worst = [], 0.0
    for i, w in enumerate(want):
        d = np.linalg.norm(pts - w, axis=1)
        j = int(np.argmin(d))
        if d[j] > radius:
            sys.exit(f"{label}: no blob within {radius:.0f}px of marker {i} "
                     f"predicted at ({w[0]:.0f},{w[1]:.0f}). Found "
                     f"{len(blobs)} blobs. Is that marker present/lit?")
        # Guard the whole point of this script: if the SECOND-nearest blob is
        # nearly as close, the two sets are not being told apart and every
        # number downstream is meaningless.
        second = np.partition(d, 1)[1] if len(d) > 1 else 1e9
        if second < d[j] * 2.0 and second < radius:
            sys.exit(f"{label}: marker {i} is ambiguous — blobs at "
                     f"{d[j]:.1f}px and {second:.1f}px. Sets not separable.")
        chosen.append(j)
        worst = max(worst, d[j])
    if len(set(chosen)) != len(chosen):
        sys.exit(f"{label}: two markers claimed the same blob")
    return pts[chosen], worst


def heldout_error(best, K, z, n_fitted=4):
    """Stripe markers, excluded from the fit: their x should be CENTERLINE_X."""
    errs, meas = [], []
    for e in best["extras"]:
        if len(e) != 2:
            return None, None
        m = backproject_undistorted(e, K, best["rvec"], best["tvec"], z)
        meas.append(m[np.argsort(m[:, 1])])
    meas = np.mean(meas, axis=0)
    for p in meas:
        errs.append(p[0] - CENTERLINE_X)
    return meas, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--capture", type=int, metavar="N")
    ap.add_argument("--exposure", type=int, default=1000)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=12.0,
                    help="px a blob may sit from its predicted spot. Default "
                         "12: the two stripe markers are only ~18px apart in "
                         "the image, so anything near that lets one set match "
                         "the other's blob")
    ap.add_argument("--threshold", type=int, default=55,
                    help="absolute detection threshold. The default relative "
                         "one (half the brightest blob = 127 here) drops the "
                         "corner stickers entirely: they peak at 90-112, and "
                         "the dimmest clears 64 by too little to keep 10 px")
    ap.add_argument("--shots-dir", default=str(
        Path(__file__).resolve().parent.parent / "cmp_shots"))
    args = ap.parse_args()

    images = args.images
    if args.capture:
        images = capture(args.capture, args.exposure, args.gain, args.shots_dir)
    if not images:
        sys.exit(__doc__)

    K, dist = load_intrinsics()
    rvec0, tvec0 = prior_pose()

    # ── Detect once, assign twice ───────────────────────────────────────────
    per_set_frames = {name: [] for name, _, _ in SETS}
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"cannot read {path}")
        blobs = find_blobs(img, find_glare(img), quiet=True, thr=args.threshold)
        line = f"  {Path(path).name}: {len(blobs)} blobs"
        for name, markers, z in SETS:
            want = predict(markers, z, K, dist, rvec0, tvec0)
            pts, worst = assign(blobs, want, args.radius, name)
            per_set_frames[name].append(pts)
            line += f" | {name.split()[0]} max {worst:.1f}px"
        print(line)

    # ── Solve each set independently ────────────────────────────────────────
    results = {}
    print()
    for name, markers, z in SETS:
        best = solve(per_set_frames[name], markers[:4], K, dist,
                     subset=True, z=z)
        R, _ = cv2.Rodrigues(best["rvec"])
        cam = (-R.T @ best["tvec"]).ravel()
        meas, errs = heldout_error(best, K, z)
        results[name] = dict(best=best, cam=cam, R=R, meas=meas, errs=errs, z=z)
        print(f"{name}")
        print(f"  reprojection RMS {best['rms']:.3f} px "
              f"(4 points, 6-DOF pose — weak proxy, see held-out below)")
        print(f"  camera position  x={cam[0]:7.1f} y={cam[1]:7.1f} "
              f"z={cam[2]:7.1f} mm")
        if errs is not None:
            print(f"  HELD-OUT stripe x error: "
                  + ", ".join(f"{e:+.2f}" for e in errs)
                  + f" mm  (worst {max(abs(e) for e in errs):.2f})")

    # ── Compare ─────────────────────────────────────────────────────────────
    a, b = [results[n] for n, _, _ in SETS]
    na, nb = [n for n, _, _ in SETS]
    print("\n" + "=" * 72)
    print("AGREEMENT BETWEEN THE TWO CALIBRATIONS")
    print("=" * 72)
    dcam = b["cam"] - a["cam"]
    print(f"camera position differs by ({dcam[0]:+.1f}, {dcam[1]:+.1f}, "
          f"{dcam[2]:+.1f}) mm, |d| = {np.linalg.norm(dcam):.1f}")
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(b["R"] @ a["R"].T)[0]))
    print(f"orientation differs by {ang:.4f} deg")

    # What actually matters: for a point on the playing surface, how far apart
    # do the two calibrations put it? Project under A, back-project under B.
    print("\nDisagreement on the PLAYING SURFACE (z=0) — project under "
          f"'{na.split()[0]}', read back under '{nb.split()[0]}':")
    probes, labels = [], []
    for fy, ly in ((0.15, "near"), (0.5, "mid"), (0.85, "far")):
        for fx, lx in ((0.15, "human"), (0.5, "centre"), (0.85, "robot")):
            probes.append([fx * GRID_X_MM, fy * GRID_Y_MM, 0.0])
            labels.append(f"{ly}/{lx}")
    probes = np.array(probes)
    px, _ = cv2.projectPoints(probes, a["best"]["rvec"], a["best"]["tvec"], K, dist)
    und = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    back = backproject_undistorted(und, K, b["best"]["rvec"], b["best"]["tvec"], 0.0)
    d = back - probes[:, :2]
    for lab, p, e in zip(labels, probes, d):
        print(f"  {lab:12s} ({p[0]:6.0f},{p[1]:5.0f}) -> off by "
              f"({e[0]:+6.2f}, {e[1]:+6.2f}) mm   |{np.linalg.norm(e):5.2f}|")
    mag = np.linalg.norm(d, axis=1)
    print(f"  mean {mag.mean():.2f} mm, worst {mag.max():.2f} mm "
          f"({labels[int(np.argmax(mag))]})")

    print("\nInterpretation: the held-out stripe error is the honest accuracy "
          "number for each\nset on its own; the table above is how much "
          "switching sets would move the world.")


if __name__ == "__main__":
    main()
