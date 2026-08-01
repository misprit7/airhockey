#!/usr/bin/env python3
"""Live frames from the FLIR camera.

There is no PySpin on this machine, so the camera can only be driven from
C++ while detection happens in Python. `snap --stream` bridges that: it
serves one frame per request over a pipe, so the consumer never falls
behind and never sees a stale frame.
"""

import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VISION = Path(__file__).resolve().parent.parent
SNAP = VISION / "build" / "snap"


class Stream:
    """Frames from `snap --stream`, one per request so we never lag."""

    def __init__(self, exposure, gain):
        if not SNAP.exists():
            sys.exit(f"{SNAP} not built — run `make` in vision/")
        self.p = subprocess.Popen(
            [str(SNAP), "--stream", "--exposure", str(exposure),
             "--gain", str(gain)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        hdr = self._read(16)
        if hdr[:8] != b"SNAPSTRM":
            sys.exit(f"unexpected stream header {hdr[:8]!r}")
        self.w, self.h = struct.unpack("<II", hdr[8:16])

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                sys.exit("camera stream closed unexpectedly")
            buf += c
        return buf

    def grab(self):
        self.p.stdin.write(b"g")
        self.p.stdin.flush()
        if self._read(1)[0] != 1:
            return None
        return np.frombuffer(self._read(self.w * self.h),
                             np.uint8).reshape(self.h, self.w)

    def close(self):
        try:
            self.p.stdin.write(b"q")
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# ── Camera model helpers ────────────────────────────────────────────────
#
# Two names, deliberately. Back-projection needs UNDISTORTED points, but
# some callers already hold undistorted coordinates (they came out of a PnP
# solve) while others hold raw pixels straight from blob detection. A single
# function taking both a `dist` argument and possibly-undistorted points is
# a silent double-undistort waiting to happen, so the distinction is in the
# name instead.

def backproject_undistorted(pts_und, K, rvec, tvec, z):
    """Intersect UNDISTORTED pixel rays with the plane Z = z (grid frame)."""
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec.reshape(3, 1)).ravel()
    Kinv = np.linalg.inv(K)
    out = []
    for u, v in np.asarray(pts_und, dtype=np.float64).reshape(-1, 2):
        d = R.T @ (Kinv @ np.array([u, v, 1.0]))
        t = (z - C[2]) / d[2]
        out.append((C + t * d)[:2])
    return np.array(out)


def backproject_pixels(pts_px, K, dist, rvec, tvec, z):
    """Same, for RAW (still distorted) pixel coordinates."""
    und = cv2.undistortPoints(
        np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2),
        K, dist, P=K).reshape(-1, 2)
    return backproject_undistorted(und, K, rvec, tvec, z)
