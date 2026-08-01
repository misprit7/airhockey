#!/usr/bin/env python3
"""Is the lens calibration trustworthy WHERE YOU ACTUALLY MEASURE?

A low ChArUco reprojection RMS only says the model fits where the board
was held. Radial distortion away from that region is an extrapolation, and
the table corners sit far outside the middle of the frame — so a calibration
can look excellent and still be wrong by millimetres at the corners.

Three checks:

  1. COVERAGE — how far out in image radius the board reached, versus how
     far out you need to measure. Distortion beyond the data is a guess.
  2. RESIDUAL vs RADIUS — whether the fitted model strains outward.
  3. MODEL DISAGREEMENT — refit under several distortion parameterisations
     that fit the captured data equally well, and measure how far apart
     they land on the table, after removing the part a re-solved camera
     pose would absorb (a plane homography). What survives is systematic
     error that the RMS cannot see. This is the number that matters.

Usage:
    python bin/check_intrinsics.py
    python bin/check_intrinsics.py --images 'calib_shots/*.png'
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_intrinsics import detect, make_detector  # noqa: E402
from table_grid import GRID_X_MM, GRID_Y_MM  # noqa: E402

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"

# Alternative distortion parameterisations. Any of these is a defensible
# model of the same lens; where they disagree, the data is not deciding.
VARIANTS = {
    "k1k2p1p2k3 (current)": 0,
    "k1k2p1p2": cv2.CALIB_FIX_K3,
    "k1k2 only": (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST),
    "rational k1..k6": cv2.CALIB_RATIONAL_MODEL,
}


def load_views(pattern):
    board, det = make_detector()
    obj, img, size = [], [], None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = size or (g.shape[1], g.shape[0])
        op, ip, _, _ = detect(board, det, g)
        if op is not None:
            obj.append(op)
            img.append(ip)
    if len(obj) < 4:
        sys.exit(f"only {len(obj)} usable views in {pattern}")
    return obj, img, size


def coverage(all_px, size, K):
    W, H = size
    cx, cy = K[0, 2], K[1, 2]
    NX, NY = 12, 9
    hist = np.zeros((NY, NX), int)
    for x, y in all_px:
        hist[min(NY - 1, int(y / H * NY)), min(NX - 1, int(x / W * NX))] += 1
    print("\n1. COVERAGE  (image rows top to bottom; '_' = no data)")
    mx = hist.max()
    for r in range(NY):
        row = "".join(" .:-=+*#%@"[min(9, int(9 * v / mx))] if v else "_"
                      for v in hist[r])
        print(f"     |{row}|")
    print(f"     {(hist == 0).sum()}/{NX * NY} cells never saw the board")

    r_obs = np.hypot(all_px[:, 0] - cx, all_px[:, 1] - cy)
    r_corner = float(np.hypot(max(cx, W - cx), max(cy, H - cy)))
    reach = float(r_obs.max()) / r_corner
    print(f"\n     board reached radius {r_obs.max():.0f} px; image corner "
          f"is at {r_corner:.0f} px  ->  {reach:.0%} of the way out")
    for f in (0.7, 0.8, 0.9):
        print(f"       beyond {f:.0%} ({f * r_corner:.0f} px): "
              f"{(r_obs > f * r_corner).sum()} observations")
    # Coverage is a means, not the goal — check 3 below is the real verdict.
    if reach < 0.7:
        print("     VERDICT: FAIL — distortion is extrapolated over the "
              "outer part of the frame.\n"
              "     Recapture with the board in the image CORNERS (hold it "
              "nearer the camera\n     so it covers more of the frame), still "
              "tilted.")
    elif reach < 0.85:
        print("     VERDICT: marginal — the outermost ring is still "
              "extrapolated. Worth another\n     pass aimed only at the "
              "gaps, but judge it on the mm figure in check 3.")
    else:
        print("     VERDICT: ok — data reaches the periphery")
    return r_obs, r_corner


def residual_vs_radius(obj, img, size, K, dist, rvecs, tvecs):
    print("\n2. RESIDUAL vs RADIUS")
    rr, ee = [], []
    for i, (op, ip) in enumerate(zip(obj, img)):
        proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
        p = ip.reshape(-1, 2)
        ee.append(np.linalg.norm(proj.reshape(-1, 2) - p, axis=1))
        rr.append(np.hypot(p[:, 0] - K[0, 2], p[:, 1] - K[1, 2]))
    rr, ee = np.concatenate(rr), np.concatenate(ee)
    for a, b in zip(*[np.linspace(0, rr.max(), 6)[i:i + 5] for i in (0, 1)]):
        m = (rr >= a) & (rr < b)
        if m.sum():
            print(f"     r {a:4.0f}-{b:4.0f} px  n={m.sum():5d}  "
                  f"rms {np.sqrt((ee[m] ** 2).mean()):.3f} px")


def table_pixels(K, dist):
    """A grid of image points covering the playing field, via the solved
    extrinsics if available; otherwise the whole frame."""
    f = CALIB_DIR / "extrinsics.npz"
    if not f.exists():
        return None, None
    d = np.load(f)
    gx, gy = np.meshgrid(np.linspace(0, GRID_X_MM, 9),
                         np.linspace(0, GRID_Y_MM, 5))
    world = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    px, _ = cv2.projectPoints(world, d["rvec"], d["tvec"], K, dist)
    return px.reshape(-1, 2), world[:, :2]


def disagreement(obj, img, size, K, dist):
    print("\n3. MODEL DISAGREEMENT  (the part a re-solved pose cannot hide)")
    px, world = table_pixels(K, dist)
    if px is None:
        print("     extrinsics.npz missing — run calibrate_extrinsics.py to "
              "get this in mm; skipping")
        return
    W, H = size
    inside = (px[:, 0] > -50) & (px[:, 0] < W + 50) & \
             (px[:, 1] > -50) & (px[:, 1] < H + 50)
    px, world = px[inside], world[inside]

    und = {}
    for name, flag in VARIANTS.items():
        rms, Kv, dv, _, _ = cv2.calibrateCamera(obj, img, size, None, None,
                                                flags=flag)
        und[name] = cv2.undistortPoints(px.reshape(-1, 1, 2), Kv, dv,
                                        P=K).reshape(-1, 2)
        print(f"     {name:22s} fit rms {rms:.3f} px")

    base_name = next(iter(VARIANTS))
    base = und[base_name]
    # mm per undistorted pixel, from the table geometry itself
    span_px = np.hypot(*(base[world[:, 0].argmax()] - base[world[:, 0].argmin()]))
    mm_px = (world[:, 0].max() - world[:, 0].min()) / max(span_px, 1e-9)

    worst = np.zeros(len(base))
    for name, u in und.items():
        if name == base_name:
            continue
        # Remove the component a re-solved camera pose would absorb: on a
        # plane that is exactly a homography.
        Hm, _ = cv2.findHomography(u, base, method=0)
        fit = cv2.perspectiveTransform(u.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        d_mm = np.linalg.norm(fit - base, axis=1) * mm_px
        worst = np.maximum(worst, d_mm)
        print(f"     vs {name:22s} max {d_mm.max():5.2f} mm  "
              f"median {np.median(d_mm):5.2f} mm")
    print("     (the rational model is the most flexible, so it diverges "
          "most where the data\n      runs out — treat it as the pessimistic "
          "end and the k1k2* variants as the floor)")

    print(f"\n     scale {mm_px:.2f} mm per px")
    print("     worst-case disagreement across the field (mm):")
    ux = np.unique(world[:, 0])
    uy = np.unique(world[:, 1])[::-1]
    print("        " + "".join(f"{v:7.0f}" for v in ux) + "   <- x mm")
    for yv in uy:
        row = "".join(f"{worst[(world[:, 1] == yv) & (world[:, 0] == xv)][0]:7.2f}"
                      if ((world[:, 1] == yv) & (world[:, 0] == xv)).any()
                      else "      ." for xv in ux)
        print(f"  {yv:6.0f}{row}")
    print(f"\n     WORST {worst.max():.2f} mm — a floor on systematic error "
          "from the lens model alone.")
    if worst.max() > 0.5:
        print("     Reduce it by covering the frame periphery when "
              "capturing intrinsics.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=str(
        Path(__file__).resolve().parent.parent / "calib_shots" / "*.png"))
    args = ap.parse_args()

    obj, img, size = load_views(args.images)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None,
                                                     None)
    all_px = np.concatenate([p.reshape(-1, 2) for p in img])
    print(f"{len(obj)} views, {len(all_px)} corner observations, "
          f"fit rms {rms:.3f} px")

    coverage(all_px, size, K)
    residual_vs_radius(obj, img, size, K, dist, rvecs, tvecs)
    disagreement(obj, img, size, K, dist)


if __name__ == "__main__":
    main()
