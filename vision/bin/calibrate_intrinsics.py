#!/usr/bin/env python3
"""Interactive lens (intrinsics) calibration with the ChArUco board.

Live view with detection overlay. Hold the printed board in front of the
camera and capture ~20 varied views: tilt it +/-30-45 deg, vary distance,
and reach into all four frame corners (distortion lives there).

Keys:
  SPACE  capture current view (needs enough detected corners)
  c      run calibration on captured views, save, and show undistorted live
  u      toggle undistorted preview (after calibrating)
  q      quit

Results go to vision/calib/intrinsics.npz (+ .json for readability).
Calibrate at the SAME resolution and focus the runtime tracker will use —
intrinsics do not transfer across focus changes or sensor crop modes.

Alternative: --images 'glob' calibrates from saved stills, no camera needed.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_targets import (CHARUCO_COLS, CHARUCO_ROWS, CHARUCO_DICT,  # noqa: E402
                         MARKER_MM, SQUARE_MM)

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
MIN_CORNERS_PER_VIEW = 12   # of (COLS-1)*(ROWS-1) = 70 interior corners
MIN_VIEWS = 8
GOOD_VIEWS = 15


def make_detector():
    d = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS), SQUARE_MM / 1000.0, MARKER_MM / 1000.0, d)
    return board, cv2.aruco.CharucoDetector(board)


def detect(board, detector, gray):
    """Returns (obj_pts, img_pts, ch_corners, n) or (None, None, None, 0)."""
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < MIN_CORNERS_PER_VIEW:
        n = 0 if ch_ids is None else len(ch_ids)
        return None, None, ch_corners, n
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    return obj_pts, img_pts, ch_corners, len(ch_ids)


def run_calibration(obj_list, img_list, size):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, size, None, None)
    per_view = []
    for i, (op, ip) in enumerate(zip(obj_list, img_list)):
        proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
        per_view.append(float(np.sqrt(np.mean((proj - ip) ** 2))))
    return rms, K, dist, per_view


def save_results(size, rms, K, dist, per_view):
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CALIB_DIR / "intrinsics.npz",
             camera_matrix=K, dist_coeffs=dist,
             image_size=np.array(size), rms=rms)
    with open(CALIB_DIR / "intrinsics.json", "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "image_size": list(size),
            "rms_px": round(float(rms), 4),
            "per_view_rms_px": [round(v, 4) for v in per_view],
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.ravel().tolist(),
            "board": {"cols": CHARUCO_COLS, "rows": CHARUCO_ROWS,
                      "square_mm": SQUARE_MM, "marker_mm": MARKER_MM},
        }, f, indent=2)
    print(f"\nSaved {CALIB_DIR}/intrinsics.npz and .json")


def report(rms, K, per_view):
    print(f"\nRMS reprojection error: {rms:.3f} px "
          f"({'good' if rms < 0.5 else 'HIGH — recheck board flatness/scale'})")
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    worst = max(range(len(per_view)), key=lambda i: per_view[i])
    print(f"Worst view: #{worst + 1} at {per_view[worst]:.3f} px "
          "(recapture excluding it if it's an outlier)")
    print("\nA low RMS only means the model fits WHERE THE BOARD WAS. Run\n"
          "  python bin/check_intrinsics.py\n"
          "to confirm the views reached the frame periphery — distortion is\n"
          "extrapolated beyond them, and the table corners live out there.")


def from_images(pattern):
    board, detector = make_detector()
    obj_list, img_list, size = [], [], None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  skip (unreadable): {f}")
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != size:
            sys.exit(f"mixed image sizes: {f}")
        op, ip, _, n = detect(board, detector, img)
        if op is None:
            print(f"  skip ({n} corners): {f}")
            continue
        obj_list.append(op)
        img_list.append(ip)
        print(f"  ok ({n} corners): {f}")
    if len(obj_list) < MIN_VIEWS:
        sys.exit(f"only {len(obj_list)} usable views, need >= {MIN_VIEWS}")
    rms, K, dist, per_view = run_calibration(obj_list, img_list, size)
    report(rms, K, per_view)
    save_results(size, rms, K, dist, per_view)


def live(args):
    board, detector = make_detector()
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Lock autofocus if the driver allows it — intrinsics are per-focus.
    if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("NOTE: could not disable autofocus via OpenCV; if this camera "
              "autofocuses, lock it with v4l2-ctl before calibrating.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    print(f"streaming {w}x{h} — calibrate at the resolution you will track at")

    obj_list, img_list = [], []
    K = dist = None
    undistort = False
    flash = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        op, ip, ch_corners, n = detect(board, detector, gray)

        disp = frame
        if undistort and K is not None:
            disp = cv2.undistort(frame, K, dist)
        elif ch_corners is not None and n:
            cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners)

        color = (0, 220, 0) if op is not None else (0, 140, 255)
        hud = (f"corners {n}/70   captured {len(obj_list)} "
               f"(aim {GOOD_VIEWS}+)   SPACE=capture  c=calibrate  "
               f"u=undistort  q=quit")
        if undistort and K is not None:
            hud = "UNDISTORTED preview   " + hud
        cv2.putText(disp, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2)
        if flash > 0:
            flash -= 1
            cv2.rectangle(disp, (0, 0), (w - 1, h - 1), (255, 255, 255), 14)
        cv2.imshow("intrinsics", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if op is None:
                print(f"not captured — only {n} corners "
                      f"(need >= {MIN_CORNERS_PER_VIEW})")
            else:
                obj_list.append(op)
                img_list.append(ip)
                flash = 4
                print(f"captured view {len(obj_list)} ({n} corners)")
        elif key == ord("c"):
            if len(obj_list) < MIN_VIEWS:
                print(f"need >= {MIN_VIEWS} views, have {len(obj_list)}")
                continue
            print(f"calibrating on {len(obj_list)} views...")
            rms, K, dist, per_view = run_calibration(obj_list, img_list, size)
            report(rms, K, per_view)
            save_results(size, rms, K, dist, per_view)
            undistort = True
            print("showing undistorted preview — straight edges should look "
                  "straight; press u to compare, q when satisfied")
        elif key == ord("u"):
            undistort = not undistort

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0, help="V4L2 device index")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--images", help="glob of still images instead of camera")
    args = ap.parse_args()
    if args.images:
        from_images(args.images)
    else:
        live(args)


if __name__ == "__main__":
    main()
