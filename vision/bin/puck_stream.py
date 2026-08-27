#!/usr/bin/env python3
"""Puck (and paddle) position at ~200 Hz, from the C++ blob tracker.

`vision/build/blobtrack` does the thresholding and centroiding on the frame
the SDK hands it and streams coordinates; this module decides which of those
blobs is the puck. The split is deliberate: pixels are cheap in C++ and the
calibration lives in Python, so neither side has to know about the other's
problem.

This part is expected to OUTLIVE the hardcoded goalie — an RL policy needs
exactly the same puck stream. Keep it free of anything policy-shaped.

WHICH BLOB IS THE PUCK
    The puck carries FOUR retroreflectors in a square and the mallet carries
    ONE, so the puck is the group of blobs whose spacings SOLVE that square.
    A model-based test, not a brightness or size heuristic -- every marker on
    this table is the same tape, and the mallet is the same distance from the
    lens as the puck.

    The cluster is on the puck and the lone dot on the mallet, which is the
    inverse of the original scheme: a player's hand wraps the mallet and hides
    whatever is stuck to it, while nothing ever touches the puck.

Standalone, for checking the tracker sees what you think it sees:

    python vision/bin/puck_stream.py            # live puck position
    python vision/bin/puck_stream.py --raw      # every surviving blob
    python vision/bin/puck_stream.py --selftest # synthetic puck, no camera

The square itself is solved in puck_markers.py, which is pure geometry and
carries its own selftest.
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
from puck_markers import find_puck  # noqa: E402
from table_grid import GRID_X_MM, GRID_Y_MM  # noqa: E402
from track_mallet import MARKER_Z_MM, SPOOL_MARKER_Z_MM, load_pose  # noqa: E402

BLOBTRACK = Path(__file__).resolve().parent.parent / "build" / "blobtrack"

# 300 us keeps motion blur to ~1.5 mm at 5 m/s, but at 0 dB the puck marker
# peaks at 96 against a threshold of 90 — it was detected in 11% of frames.
# 12 dB of gain saturates it (23 px over threshold) and the background only
# rises from 2 to 5, because the scene is dark by construction. Gain is the
# cheap axis here; exposure is not.

# A blob within this of a projected permanent marker IS that marker.
MARKER_REJECT_PX = 16.0
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


def _lsq_slope(t, y):
    """Least-squares slope of y against t. Zero if there is nothing to fit."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 3:
        return 0.0
    tm = t.mean()
    den = ((t - tm) ** 2).sum()
    if den <= 0:
        return 0.0
    return float(((t - tm) * (y - y.mean())).sum() / den)


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
        # Orientation is tracked separately from position because it survives
        # a different set of dropouts: two adjacent corners fix the angle
        # exactly but leave the centre ambiguous.
        self._spin: deque = deque(maxlen=6)
        self.theta = float("nan")   # radians, UNWRAPPED (see _track_spin)
        self.omega = 0.0            # rad/s about the puck's own axis
        self.n_markers = 0          # corners used for the last fix

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
        world = self._to_table(px, geom.PUCK_MARKER_Z_MM)
        on = ((world[:, 0] > -OUTSIDE_MM) & (world[:, 0] < GRID_X_MM + OUTSIDE_MM) &
              (world[:, 1] > -OUTSIDE_MM) & (world[:, 1] < GRID_Y_MM + OUTSIDE_MM))
        return blobs[keep][on], world[on]

    def update(self, t, blobs):
        """Return (x, y, vx, vy) in mm and mm/s, or None if the puck is lost.

        The puck is the group of blobs that SOLVES the known marker square;
        see find_puck. Spin and corner count land in self.theta / self.omega /
        self.n_markers rather than in the return value, so callers that only
        want position keep their four-tuple.
        """
        _kept, world = self.candidates(blobs)
        if len(world) == 0:
            return self._coast(t)

        prev = (self._hist[-1][1], self._hist[-1][2]) if self._hist else None
        got = find_puck(world, prev)
        if got is None:
            # One visible corner is not a fix: the centre is 21.85 mm away in
            # an unknown direction, and reporting that as a position is worse
            # than coasting because it still looks like a measurement.
            return self._coast(t)

        c, theta, members, _rms = got
        x, y = float(c[0]), float(c[1])
        self.n_markers = len(members)
        self._track_spin(t, theta)
        self._hist.append((t, x, y))
        return (x, y) + self._velocity()

    def _coast(self, t):
        """Puck not visible this frame — most often the centre blind spot."""
        self.n_markers = 0
        if not self._hist:
            return None
        t0, x0, y0 = self._hist[-1]
        if t - t0 > 0.15:            # 30 frames: it is genuinely gone
            self._hist.clear()
            self._spin.clear()
            return None
        vx, vy = self._velocity()
        return (x0 + vx * (t - t0), y0 + vy * (t - t0), vx, vy)

    def _track_spin(self, t, theta):
        """Unwrap the mod-90-degree orientation and fit a rotation rate.

        The square only reports its angle modulo a quarter turn, so successive
        frames are stitched by taking the smallest consistent step. That is
        unambiguous up to 45 degrees per frame — 25 rev/s at 200 Hz, far above
        anything a struck puck does — and self.theta accumulates rather than
        wrapping so that differencing it is safe.
        """
        q = math.pi / 2
        if self._spin:
            t0, last = self._spin[-1]
            if t - t0 > 0.05:        # too long a gap to stitch across
                self._spin.clear()
            else:
                theta = last + (theta - last + q / 2) % q - q / 2
        self._spin.append((t, theta))
        self.theta = theta
        s = np.array(self._spin)
        self.omega = _lsq_slope(s[:, 0], s[:, 1])

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
        return _lsq_slope(a[:, 0], a[:, 1]), _lsq_slope(a[:, 0], a[:, 2])


def _selftest() -> int:
    """A synthetic puck through the REAL camera pose. No camera, no blobtrack.

    puck_markers already tests the square solver on clean coordinates; this
    tests the part that only exists once the calibration is involved —
    projecting corners to pixels, undistorting, back-projecting at the marker
    height, and getting velocity and spin back out.

    The case worth watching is the dropout one. A centroid of three corners
    would put the centre 7.3 mm out in a direction that changes as the puck
    rotates, so the position error would jump every time a corner came and
    went. It has to stay flat across the three rows.
    """
    tr = PuckTracker()
    r = geom.PUCK_MARKER_R_MM
    v = np.array([2400.0, -900.0])          # mm/s
    omega = math.radians(720.0)             # 2 rev/s
    p0 = np.array([1400.0, 700.0])
    dt = 1.0 / 200.0
    rng = np.random.default_rng(1)

    def blobs_at(c, th, drop):
        a = th + np.arange(4) * (math.pi / 2)
        obj = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a),
                        np.full(4, geom.PUCK_MARKER_Z_MM)], 1)
        obj = np.delete(obj, list(drop), axis=0)
        px, _ = cv2.projectPoints(obj.astype(float), tr.rvec, tr.tvec,
                                  tr.K, tr.dist)
        px = px.reshape(-1, 2) + rng.normal(0, 0.15, (len(obj), 2))
        return np.hstack([px, np.full((len(px), 1), 30.0)])

    cases = [("all four corners", lambda k: ()),
             ("one corner dropped", lambda k: (k % 4,)),
             ("two opposite dropped", lambda k: (k % 2, k % 2 + 2))]
    for label, drops in cases:
        tr._hist.clear()
        tr._spin.clear()
        pe, ve, we = [], [], []
        for k in range(60):
            t = k * dt
            p = p0 + v * t
            out = tr.update(t, blobs_at(p, omega * t, drops(k)))
            if out is None or k < 8:        # let the slope fits fill
                continue
            pe.append(math.hypot(out[0] - p[0], out[1] - p[1]))
            ve.append(math.hypot(out[2] - v[0], out[3] - v[1]))
            we.append(tr.omega)
        assert pe, f"{label}: puck never resolved"
        spin = math.degrees(float(np.mean(we)))
        print(f"  {label:22s} pos err max {max(pe):5.2f} mm   "
              f"|v| err max {max(ve):6.1f} mm/s   spin {spin:7.1f} deg/s")
        assert max(pe) < 2.0, f"{label}: {max(pe):.2f} mm"
        assert max(ve) < 60.0, f"{label}: {max(ve):.1f} mm/s"
        assert abs(spin - 720.0) < 30.0, f"{label}: {spin:.1f} deg/s"

    print("selftest PASSED — dropouts cost nothing, which is the whole point "
          "of solving the square rather than averaging it")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    ap.add_argument("--raw", action="store_true",
                    help="print every surviving blob, not just the puck")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic puck through the real pose, no camera")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

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
                          f"   |v| {math.hypot(vx, vy):7.1f}"
                          f"   {tr.n_markers}/4 dots"
                          f"   spin {math.degrees(tr.omega):8.1f} deg/s")
    except KeyboardInterrupt:
        pass
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
