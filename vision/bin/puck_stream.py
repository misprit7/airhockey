#!/usr/bin/env python3
"""Puck (and paddle) position at ~200 Hz, from the C++ blob tracker.

`vision/build/blobtrack` does the thresholding and centroiding on the frame
the SDK hands it and streams coordinates; this module decides which of those
blobs is the puck. The split is deliberate: pixels are cheap in C++ and the
calibration lives in Python, so neither side has to know about the other's
problem.

This part is expected to OUTLIVE the hardcoded goalie — an RL policy needs
exactly the same puck stream. Keep it free of anything policy-shaped.

Standalone, for checking the tracker sees what you think it sees:

    python vision/bin/puck_stream.py            # live puck position
    python vision/bin/puck_stream.py --raw      # every surviving blob
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

import cdpr_geometry as geom  # noqa: E402
from calibrate_extrinsics import CALIB_DIR, load_intrinsics  # noqa: E402
from camera import backproject_undistorted  # noqa: E402
from table_grid import GRID_X_MM, GRID_Y_MM  # noqa: E402
from track_mallet import MARKER_Z_MM, SPOOL_MARKER_Z_MM, load_pose  # noqa: E402

BLOBTRACK = Path(__file__).resolve().parent.parent / "build" / "blobtrack"

# 300 us keeps motion blur to ~1.5 mm at 5 m/s, but at 0 dB the puck marker
# peaks at 96 against a threshold of 90 — it was detected in 11% of frames.
# 12 dB of gain saturates it (23 px over threshold) and the background only
# rises from 2 to 5, because the scene is dark by construction. Gain is the
# cheap axis here; exposure is not.

# The marker rides on top of the puck, so it is not on the playing surface.
# The camera is ~1506 mm up and the puck reaches ~1000 mm off axis, so
# back-projecting at z=0 instead of the true height would put it ~5 mm out at
# the edges, and the error would grow with distance from centre — i.e. exactly
# where a goalie needs the prediction to be good.
PUCK_Z_MM = 8.0

# A blob within this of a projected permanent marker IS that marker.
MARKER_REJECT_PX = 16.0
# Paddle retroreflectors sit in a tight cluster; the puck marker is alone.
CLUSTER_PX = 60.0
# Off the playing surface by more than this and it is a rail reflection.
OUTSIDE_MM = 40.0


class BlobStream:
    """Lines of blob coordinates from the C++ tracker."""

    def __init__(self, fps=200.0, exposure=300.0, gain=12.0, threshold=90,
                 min_area=4, max_area=4000):
        if not BLOBTRACK.exists():
            sys.exit(f"{BLOBTRACK} not built — run `make -C vision`")
        self.p = subprocess.Popen(
            [str(BLOBTRACK), "--fps", str(fps), "--exposure", str(exposure),
             "--gain", str(gain), "--threshold", str(threshold),
             "--min-area", str(min_area), "--max-area", str(max_area)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            bufsize=1)
        header = self.p.stdout.readline()
        if not header.startswith("#"):
            raise RuntimeError(f"unexpected header from blobtrack: {header!r}")
        _, w, h = header.split()
        self.width, self.height = int(w), int(h)

    def __iter__(self):
        for line in self.p.stdout:
            if not line.startswith("F"):
                continue
            f = line.split()
            n = int(f[3])
            pts = np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                pts[i] = (float(f[4 + 3 * i]), float(f[5 + 3 * i]),
                          float(f[6 + 3 * i]))
            yield int(f[1]), int(f[2]) * 1e-6, pts

    def close(self):
        try:
            self.p.terminate()
            self.p.wait(timeout=2)
        except Exception:      # noqa: BLE001
            self.p.kill()


class PuckTracker:
    """Turns blobs into a puck position in table millimetres.

    Everything bolted down is rejected by PROJECTING where it should be and
    discarding blobs that land there, rather than by trying to tell them apart
    locally — a puck marker and a corner sticker are the same handful of bright
    pixels, and nothing about the blob itself distinguishes them.
    """

    def __init__(self, glare_pad_px=6.0):
        self.K, self.dist, self.rvec, self.tvec, field = load_pose()
        obj = np.vstack([
            np.hstack([field, np.full((len(field), 1), MARKER_Z_MM)]),
            np.array([[geom.MOTOR_X[m], geom.MOTOR_Y[m], SPOOL_MARKER_Z_MM]
                      for m in range(len(geom.MOTOR_X))], float),
        ])
        px, _ = cv2.projectPoints(obj, self.rvec, self.tvec, self.K, self.dist)
        self.known_px = px.reshape(-1, 2)

        # The IR ring's own reflection is a permanent cluster of bright dots
        # near the table centre — about 92 x 103 mm of blind spot. Rejecting by
        # BOUNDING BOX rather than by the saved per-pixel mask, because a blob
        # centroid can sit a pixel outside the mask while the blob is plainly
        # part of the glare.
        g = cv2.imread(str(CALIB_DIR / "glare_mask.png"), cv2.IMREAD_GRAYSCALE)
        if g is not None and g.any():
            ys, xs = np.nonzero(g)
            self.glare = (xs.min() - glare_pad_px, xs.max() + glare_pad_px,
                          ys.min() - glare_pad_px, ys.max() + glare_pad_px)
        else:
            self.glare = None

        self._hist: deque = deque(maxlen=6)

    # ── blob -> table ────────────────────────────────────────────────────
    def _to_table(self, px_xy, z):
        und = cv2.undistortPoints(px_xy.reshape(-1, 1, 2), self.K, self.dist,
                                  P=self.K).reshape(-1, 2)
        return backproject_undistorted(und, self.K, self.rvec, self.tvec, z)

    def candidates(self, blobs):
        """Blobs that are not glare, not a fixed marker, and on the table."""
        if len(blobs) == 0:
            return np.empty((0, 3)), np.empty((0, 2))
        px = blobs[:, :2]
        keep = np.ones(len(px), bool)
        if self.glare:
            x0, x1, y0, y1 = self.glare
            keep &= ~((px[:, 0] > x0) & (px[:, 0] < x1) &
                      (px[:, 1] > y0) & (px[:, 1] < y1))
        if len(self.known_px):
            d = np.linalg.norm(px[:, None, :] - self.known_px[None, :, :],
                               axis=2).min(axis=1)
            keep &= d > MARKER_REJECT_PX
        if not keep.any():
            return np.empty((0, 3)), np.empty((0, 2))
        px = px[keep]
        world = self._to_table(px, PUCK_Z_MM)
        on = ((world[:, 0] > -OUTSIDE_MM) & (world[:, 0] < GRID_X_MM + OUTSIDE_MM) &
              (world[:, 1] > -OUTSIDE_MM) & (world[:, 1] < GRID_Y_MM + OUTSIDE_MM))
        return blobs[keep][on], world[on]

    def update(self, t, blobs):
        """Return (x, y, vx, vy) in mm and mm/s, or None if the puck is lost.

        The paddle carries THREE retroreflectors in a tight cluster and the
        puck carries one, so 'the blob with no near neighbours' separates them
        without needing either to be a particular brightness — which matters
        because both are the same tape.
        """
        kept, world = self.candidates(blobs)
        if len(world) == 0:
            return self._coast(t)

        lone = []
        for i, w in enumerate(world):
            near = np.linalg.norm(kept[:, :2] - kept[i, :2], axis=1)
            if (near < CLUSTER_PX).sum() == 1:      # only itself
                lone.append(i)
        if not lone:
            return self._coast(t)

        # If several survive, believe the one nearest the last known puck;
        # on the first frame, the largest.
        if self._hist and len(lone) > 1:
            lx, ly = self._hist[-1][1], self._hist[-1][2]
            i = min(lone, key=lambda j: (world[j][0] - lx) ** 2
                    + (world[j][1] - ly) ** 2)
        else:
            i = max(lone, key=lambda j: kept[j][2])

        x, y = float(world[i][0]), float(world[i][1])
        self._hist.append((t, x, y))
        return (x, y) + self._velocity()

    def _coast(self, t):
        """Puck not visible this frame — most often the centre blind spot."""
        if not self._hist:
            return None
        t0, x0, y0 = self._hist[-1]
        if t - t0 > 0.15:            # 30 frames: it is genuinely gone
            self._hist.clear()
            return None
        vx, vy = self._velocity()
        return (x0 + vx * (t - t0), y0 + vy * (t - t0), vx, vy)

    def _velocity(self):
        """Least-squares slope over the recent history.

        Fitting a line over several frames rather than differencing the last
        two: at 200 Hz consecutive samples are 5 ms apart, so differencing
        amplifies the ~1 mm centroid noise into ~200 mm/s of velocity noise,
        and the goalie's whole job is downstream of this number.
        """
        if len(self._hist) < 3:
            return 0.0, 0.0
        a = np.array(self._hist)
        t = a[:, 0] - a[0, 0]
        tm = t.mean()
        den = ((t - tm) ** 2).sum()
        if den <= 0:
            return 0.0, 0.0
        vx = ((t - tm) * (a[:, 1] - a[:, 1].mean())).sum() / den
        vy = ((t - tm) * (a[:, 2] - a[:, 2].mean())).sum() / den
        return float(vx), float(vy)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    ap.add_argument("--raw", action="store_true",
                    help="print every surviving blob, not just the puck")
    args = ap.parse_args()

    tr = PuckTracker()
    st = BlobStream(fps=args.fps, exposure=args.exposure, gain=args.gain,
                    threshold=args.threshold)
    print(f"blobtrack {st.width}x{st.height} — ctrl-C to stop\n")
    n = 0
    t_wall = time.time()
    try:
        for seq, t, blobs in st:
            n += 1
            if args.raw:
                kept, world = tr.candidates(blobs)
                if n % 20 == 0:
                    print(f"[{t:7.3f}] {len(blobs)} blobs, {len(kept)} survive:"
                          + "".join(f"  ({w[0]:6.0f},{w[1]:5.0f})" for w in world))
                continue
            p = tr.update(t, blobs)
            if n % 20 == 0:
                rate = n / max(time.time() - t_wall, 1e-9)
                if p is None:
                    print(f"[{t:7.3f}] {rate:5.1f} Hz   puck: --")
                else:
                    x, y, vx, vy = p
                    print(f"[{t:7.3f}] {rate:5.1f} Hz   puck ({x:7.1f},{y:6.1f}) mm"
                          f"   v ({vx:8.1f},{vy:7.1f}) mm/s"
                          f"   |v| {math.hypot(vx, vy):7.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        st.close()


if __name__ == "__main__":
    main()
