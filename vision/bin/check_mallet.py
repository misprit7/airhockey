#!/usr/bin/env python3
"""Measure the tracker's mallet error against known air holes.

One position tells you the error is 14 mm. Several tell you WHAT it is,
because the candidates leave different fingerprints:

  constant everywhere        something fixed about the mallet or the centre
                             dot — an offset that travels with the paddle

  grows with distance from   the assumed marker height. Back-projection
  the camera nadir, pointing  converts height error into position error at
  radially                   (radial offset / camera height), so it is zero
                             under the camera and worst at the edges, always
                             along the radius

  anything else              the camera pose, or the hole the mallet was
                             actually centred on

So it asks for several positions and fits both models. Place the mallet
centred on each hole it names — by eye off the rim against the surrounding
holes is fine, a couple of millimetres does not matter against a centimetre.

Reads the pose from the running web server if there is one, so it does not
have to fight it for the camera; otherwise it opens the camera itself.

Usage:
    python vision/bin/check_mallet.py
    python vision/bin/check_mallet.py --holes 50,10 60,18 68,26 60,26
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

SERVER = "http://localhost:8420"
# Spread across the workspace AND across radius from the nadir (roughly 370
# to 790 mm), because that spread is what separates a height error from a
# constant one. The last two sit either side of the nadir in y, where a
# height error pushes them in OPPOSITE directions.
DEFAULT_HOLES = [(50, 10), (60, 18), (68, 26), (60, 26)]


def pose_from_server():
    try:
        with urllib.request.urlopen(f"{SERVER}/camera/status", timeout=2) as r:
            s = json.load(r)
        return s.get("pose")
    except Exception:                                   # noqa: BLE001
        return None


def read_pose(stream, tm, calib, n=15):
    """Median of n readings, from the server if it has the camera."""
    if stream is None:
        got = []
        for _ in range(n):
            p = pose_from_server()
            if p:
                got.append((p["x"], p["y"], p["theta_deg"]))
        if not got:
            return None
        a = np.array(got)
        return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

    K, dist, rvec, tvec, field = calib
    got = []
    while len(got) < n:
        img = stream.grab()
        if img is None:
            continue
        p, _ = tm.locate(img, K, dist, rvec, tvec, field)
        if p is not None and p.get("theta") is not None:
            got.append((p["centre"][0], p["centre"][1],
                        math.degrees(p["theta"])))
    a = np.array(got)
    return np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holes", nargs="*", default=None,
                    help="col,row pairs to place the mallet on")
    args = ap.parse_args()
    holes = ([tuple(int(v) for v in h.split(",")) for h in args.holes]
             if args.holes else DEFAULT_HOLES)

    import cv2
    d = np.load(Path(__file__).resolve().parents[1] / "calib" / "extrinsics.npz")
    R, _ = cv2.Rodrigues(d["rvec"])
    cam = (-R.T @ d["tvec"]).ravel()
    print(f"camera nadir ({cam[0]:.0f}, {cam[1]:.0f}), height {cam[2]:.0f} mm\n")

    stream, tm, calib = None, None, None
    if pose_from_server() is None:
        import track_mallet as tm          # noqa: F811
        from camera import Stream
        calib = tm.load_pose()
        stream = Stream(1000, 0.0)
        print("opened the camera directly\n")
    else:
        print("reading the pose from the running server\n")

    rows = []
    try:
        for col, row in holes:
            tx, ty = col * geom.GRID_PITCH_MM, row * geom.GRID_PITCH_MM
            input(f"  centre the mallet on hole (col {col}, row {row}) "
                  f"= ({tx:.1f}, {ty:.1f}) then press Enter...")
            got = read_pose(stream, tm, calib)
            if got is None:
                print("    no pose — is the camera running and the mallet "
                      "in view?")
                continue
            gx, gy, gth = got
            rows.append((tx, ty, gx, gy, gth))
            print(f"    vision says ({gx:8.2f}, {gy:8.2f})  theta "
                  f"{gth:6.2f}   error ({gx - tx:+6.2f}, {gy - ty:+6.2f})  "
                  f"|{math.hypot(gx - tx, gy - ty):5.2f}| mm\n")
    finally:
        if stream is not None:
            stream.close()

    if len(rows) < 2:
        print("need at least 2 positions to say anything")
        return 1

    a = np.array(rows)
    err = a[:, 2:4] - a[:, 0:2]
    print("\n" + "=" * 64)
    print(f"mean error ({err[:, 0].mean():+.2f}, {err[:, 1].mean():+.2f}) mm; "
          f"spread about that mean "
          f"({err[:, 0].std():.2f}, {err[:, 1].std():.2f})")

    # Height model: error is radial from the nadir, proportional to radius.
    rad = a[:, 0:2] - cam[:2]
    dist = np.linalg.norm(rad, axis=1)
    unit = rad / dist[:, None]
    proj = (err * unit).sum(axis=1)          # radial component of each error
    print("\nposition        radius   error mm        radial   tangential")
    for i, (tx, ty, gx, gy, _) in enumerate(rows):
        tang = err[i] - proj[i] * unit[i]
        print(f"  ({tx:6.0f},{ty:5.0f})  {dist[i]:6.0f}   "
              f"({err[i][0]:+6.2f},{err[i][1]:+6.2f})   {proj[i]:+6.2f}   "
              f"{np.linalg.norm(tang):6.2f}")

    # If it is height, radial/radius is the same constant everywhere.
    k = proj / dist
    print(f"\nradial error / radius: {np.round(k, 5)}")
    print(f"  spread {k.std():.5f} about {k.mean():+.5f}")
    dz = -k.mean() * cam[2]
    print(f"  a pure height error would give a CONSTANT ratio; this one "
          f"implies {dz:+.0f} mm,")
    print(f"  i.e. the marker at {geom.MALLET_Z_MM + dz:.0f} mm rather than "
          f"{geom.MALLET_Z_MM:.0f}")
    const = np.linalg.norm(err - err.mean(axis=0), axis=1).max()
    print(f"\nworst departure from a CONSTANT offset: {const:.2f} mm")
    print("  small -> it travels with the mallet (dot placement / geometry)")
    print("  large -> it depends on where the mallet is (height or pose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
