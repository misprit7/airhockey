#!/usr/bin/env python3
"""Measure the CDPR motor anchor positions optically.

Step order: intrinsics (calibrate_intrinsics.py) -> extrinsics
(calibrate_extrinsics.py) -> THIS. The motor coordinates are unknown; this
tool measures them, using only the already-solved camera pose.

The white PLA spool tops are detected directly — no markers needed. Take a
normal auto-exposure frame (./build/snap motor_shots 1 1) and either click
the four spools in the preview window, or pass --seeds with rough pixel
coordinates. Around each seed the bright disk is segmented (threshold near
the white peak, so bright floor beyond the table edge doesn't merge in) and
an ellipse is fitted to its non-clipped boundary; the ellipse center is the
spool axis.

Each center is back-projected through the calibrated pose onto the
spool-top plane z = --height (the spools sit ABOVE the playing surface —
this plane is separate from the field plane by construction, which is why
motor measurement cannot ride along in the field-plane solve).

Motor naming uses topology only — which corner of the spool quad each
center falls in relative to the quad's own centroid (0 = far mid-table,
1 = far robot corner, 2 = near robot corner, 3 = near mid-table). No
assumed coordinates enter the measurement.

Multiple frames may be passed (seeds are reused; spools don't move) and are
averaged, with the spread reported.

Usage:
    python bin/measure_motors.py --height 90.0 motor_shots/shot_000.png
    python bin/measure_motors.py --height 90.0 --seeds "648,105 15,150 25,896 655,940" a.png b.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"


def load_calib():
    fi = CALIB_DIR / "intrinsics.npz"
    fe = CALIB_DIR / "extrinsics.npz"
    if not fi.exists():
        sys.exit("intrinsics.npz missing — run calibrate_intrinsics.py first")
    if not fe.exists():
        sys.exit("extrinsics.npz missing — run calibrate_extrinsics.py first")
    di, de = np.load(fi), np.load(fe)
    return (di["camera_matrix"], di["dist_coeffs"], de["rvec"], de["tvec"])


def detect_spool(img, seed, win_r=55, min_area=200):
    """Fit the bright spool disk near `seed`; returns center/axes/quality."""
    H, W = img.shape
    sx, sy = int(seed[0]), int(seed[1])
    x0, x1 = max(0, sx - win_r), min(W, sx + win_r)
    y0, y1 = max(0, sy - win_r), min(H, sy + win_r)
    win = img[y0:y1, x0:x1]
    otsu, _ = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Hug the spool's OWN white level, sampled at the seed, so bright floor
    # beyond the table edge doesn't merge in. Keying off the window MAXIMUM
    # instead fails whenever something brighter than the spool shares the
    # window (a lamp, a specular highlight): the threshold climbs above the
    # spool and the disk vanishes.
    py, px = sy - y0, sx - x0
    patch = win[max(0, py - 4):py + 5, max(0, px - 4):px + 5]
    level = float(np.percentile(patch, 75)) if patch.size else float(win.max())
    thr = max(otsu, level - 40)
    _, bw = cv2.threshold(win, thr, 255, cv2.THRESH_BINARY)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw)
    best, bd = None, 1e9
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < min_area:
            continue
        d = np.hypot(cents[j][0] - (sx - x0), cents[j][1] - (sy - y0))
        if d < bd:
            bd, best = d, j
    if best is None:
        return None
    mask = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    gx, gy = c[:, 0] + x0, c[:, 1] + y0
    keep = (gx > 2) & (gx < W - 3) & (gy > 2) & (gy < H - 3)  # drop clipped edge
    ck = c[keep]
    if len(ck) < 15:
        return None
    ell = cv2.fitEllipse(ck.reshape(-1, 1, 2).astype(np.float32))
    (ex, ey), (a0, a1), ang = ell
    lo, hi = min(a0, a1), max(a0, a1)
    ok = 30 <= lo and hi <= 85 and hi / max(lo, 1e-6) < 1.4
    return {"center": (ex + x0, ey + y0), "axes": (a0, a1), "angle": ang,
            "kept": float(keep.mean()), "ok": ok}


def click_seeds(img):
    pts = []
    disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def cb(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 8, (0, 0, 255), 2)
    cv2.namedWindow("spools", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("spools", cb)
    print("click the 4 spool centers (any order), then any key")
    while True:
        cv2.imshow("spools", disp)
        k = cv2.waitKey(30) & 0xFF
        if len(pts) >= 4 and k != 255:
            break
        if k in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    if len(pts) != 4:
        sys.exit("need 4 seed clicks")
    return pts


def backproject_to_plane(pts_px, K, dist, rvec, tvec, z):
    """Undistort pixels, then intersect their rays with plane z=const."""
    u = cv2.undistortPoints(np.array(pts_px, dtype=np.float64).reshape(-1, 1, 2),
                            K, dist, P=K).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec.reshape(3, 1)).ravel()
    Kinv = np.linalg.inv(K)
    out = []
    for px, py in u:
        d = R.T @ (Kinv @ np.array([px, py, 1.0]))
        t = (z - C[2]) / d[2]
        out.append((C + t * d)[:2])
    return np.array(out)


def top_face_diameter(det, K, dist, rvec, tvec, z):
    """Spool top-face diameter in mm, from the ellipse major axis.

    The major axis is the tangential direction, which perspective does not
    foreshorten, so it reads the true diameter. Nothing in the position fit
    uses it — it is a free consistency check: all four spools are the same
    printed part, so the four numbers should agree, and they should match
    the CAD. Disagreement points at a scale or pose error.
    """
    (ex, ey), (a0, a1), ang = det["center"], det["axes"], det["angle"]
    maj = max(a0, a1)
    th = np.radians(ang + (90.0 if a1 > a0 else 0.0))
    v = np.array([np.cos(th), np.sin(th)]) * maj / 2.0
    p = backproject_to_plane([(ex - v[0], ey - v[1]), (ex + v[0], ey + v[1])],
                             K, dist, rvec, tvec, z)
    return float(np.linalg.norm(p[1] - p[0]))


def name_motors(pts):
    """Motor indices by corner of the quad relative to its own centroid:
    0 = low-x/high-y, 1 = high-x/high-y, 2 = high-x/low-y, 3 = low-x/low-y."""
    c = pts.mean(axis=0)
    named = {}
    for p in pts:
        hx, hy = p[0] > c[0], p[1] > c[1]
        m = 1 if (hx and hy) else 2 if hx else 0 if hy else 3
        if m in named:
            sys.exit("two spools landed in the same quadrant — check seeds")
        named[m] = p
    return named


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", help="ambient frames of the table")
    ap.add_argument("--height", type=float, required=True,
                    help="spool TOP FACE height above table surface (mm)")
    ap.add_argument("--seeds", help='4 rough pixel coords: "x,y x,y x,y x,y" '
                    "(omit for interactive clicking on the first image)")
    args = ap.parse_args()

    K, dist, rvec, tvec = load_calib()

    img0 = cv2.imread(args.images[0], cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        sys.exit(f"cannot read {args.images[0]}")
    if args.seeds:
        seeds = [tuple(map(float, s.split(","))) for s in args.seeds.split()]
        if len(seeds) != 4:
            sys.exit("--seeds needs exactly 4 x,y pairs")
    else:
        seeds = click_seeds(img0)

    per_motor = {m: [] for m in range(4)}
    per_dia = {m: [] for m in range(4)}
    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"cannot read {path}")
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        centers, dets = [], []
        for i, seed in enumerate(seeds):
            det = detect_spool(img, seed)
            if det is None:
                sys.exit(f"{path}: no spool disk found near seed {i} {seed}")
            flag = "" if det["ok"] else "  SUSPECT-SHAPE"
            print(f"  {path} seed {i}: center ({det['center'][0]:.1f}, "
                  f"{det['center'][1]:.1f}) axes ({det['axes'][0]:.1f}, "
                  f"{det['axes'][1]:.1f}) boundary-used {det['kept']:.0%}{flag}")
            centers.append(det["center"])
            dets.append(det)
            cv2.ellipse(canvas, (det["center"], det["axes"], det["angle"]),
                        (0, 255, 0), 1)
            cv2.circle(canvas, tuple(map(int, det["center"])), 2, (0, 0, 255), -1)
        dbg = str(Path(path).with_suffix("")) + "_spools.png"
        cv2.imwrite(dbg, canvas)
        field = backproject_to_plane(centers, K, dist, rvec, tvec, args.height)
        dias = [top_face_diameter(d, K, dist, rvec, tvec, args.height)
                for d in dets]
        for m, p in name_motors(field).items():
            per_motor[m].append(p)
            per_dia[m].append(dias[int(np.argmin(
                np.linalg.norm(field - p, axis=1)))])
    print("  (annotated fits saved next to each frame as *_spools.png)")

    print("\nMeasured motor anchors (mm, field frame):")
    anchors = {}
    for m in range(4):
        pts = np.array(per_motor[m])
        mean = pts.mean(axis=0)
        spread = float(np.linalg.norm(pts - mean, axis=1).max()) if len(pts) > 1 else 0.0
        anchors[m] = mean
        note = f"   (spread {spread:.1f}mm over {len(pts)} frames)" if len(pts) > 1 else ""
        print(f"  Motor {m}: ({mean[0]:8.1f}, {mean[1]:8.1f}){note}")

    dia = {m: float(np.mean(per_dia[m])) for m in range(4)}
    vals = list(dia.values())
    print("\nTop-face diameter (not used in the fit — a free check; all four "
          "are the same part):")
    print("  " + "   ".join(f"M{m} {dia[m]:.1f}mm" for m in range(4))
          + f"   spread {max(vals) - min(vals):.1f}mm")
    if max(vals) - min(vals) > 5.0:
        print("  WARNING: >5mm spread — suspect the pose, the scale, or a "
              "spool whose rim is clipped/saturated")
    print(f"Compare {np.mean(vals):.1f}mm against CAD; a systematic offset "
          "means a scale error.")

    with open(CALIB_DIR / "motor_anchors.json", "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"),
                   "spool_top_height_mm": args.height,
                   "anchors_mm": {str(m): [round(float(v), 1) for v in anchors[m]]
                                  for m in range(4)},
                   "top_face_diameter_mm": {str(m): round(dia[m], 1)
                                            for m in range(4)}}, f, indent=2)
    print(f"saved {CALIB_DIR}/motor_anchors.json")

    xs = ", ".join(f"{anchors[m][0]:.1f}f" for m in range(4))
    ys = ", ".join(f"{anchors[m][1]:.1f}f" for m in range(4))
    print("\n// Paste into fw/include/cdpr_config.h:")
    print(f"constexpr float MOTOR_X[NUM_MOTORS] = {{{xs}}};")
    print(f"constexpr float MOTOR_Y[NUM_MOTORS] = {{{ys}}};")


if __name__ == "__main__":
    main()
