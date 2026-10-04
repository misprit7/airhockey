#!/usr/bin/env python3
"""Live frames from the FLIR camera.

There is no PySpin on this machine, so the camera can only be driven from
C++ while detection happens in Python. `snap --stream` bridges that: it
serves one frame per request over a pipe, so the consumer never falls
behind and never sees a stale frame.
"""

import ctypes
import os
import time
import select
import signal
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

    # Frames to throw away after opening. The first frames off this camera
    # come back progressively brighter as the sensor and the IR ring settle:
    # measured on a corner marker, its peak went 25, 44, 133, 183, 194, 194
    # before levelling. Frame 0 is dark enough that a real marker falls below
    # the detector's threshold and simply is not there.
    #
    # This is not cosmetic. Marker detection thresholds on the frame maximum,
    # so an early frame silently loses the DIMMEST markers — the far corners,
    # the ones a pose solve most depends on. It also explains why repeated
    # anchor measurements agreed to 0.1 mm within one camera session but
    # drifted ~1.5 mm between sessions: each session's first burst was
    # measuring partly-cold frames.
    WARMUP_FRAMES = 8

    def __init__(self, exposure, gain, warmup=None):
        if not SNAP.exists():
            raise RuntimeError(f"{SNAP} not built — run `make` in vision/")
        # Die with the parent. Without this, killing the owning process
        # orphans snap still holding the Spinnaker device, and every later
        # attempt to open the camera blocks forever behind it — which
        # presents as "camera running, 0 fps" rather than as an error.
        def _pdeathsig():
            try:
                ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
            except Exception:
                pass

        self.p = subprocess.Popen(
            [str(SNAP), "--stream", "--exposure", str(exposure),
             "--gain", str(gain)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
            preexec_fn=_pdeathsig)

        # Bounded wait for the header: a busy camera makes snap hang rather
        # than exit, and blocking here forever hides the real cause.
        if not select.select([self.p.stdout], [], [], 12.0)[0]:
            self.p.kill()
            self.p.wait()
            raise RuntimeError("camera delivered no stream header within 12s; "
                               "check for another camera owner or a USB/camera stall")
        try:
            hdr = self._read(16)
        except Exception:
            self.close()
            raise
        if hdr[:8] != b"SNAPSTRM":
            self.close()
            raise RuntimeError(f"unexpected stream header {hdr[:8]!r}")
        self.w, self.h = struct.unpack("<II", hdr[8:16])

        n = self.WARMUP_FRAMES if warmup is None else warmup
        try:
            for _ in range(n):
                self.grab()
        except Exception:
            self.close()
            raise

    def _read(self, n, timeout_s=3.0):
        # Buffered read(n) can wait forever after a partial frame. Read the
        # pipe directly with one deadline so USB/process stalls reach the UI.
        chunks = []
        remaining = n
        deadline = time.monotonic() + timeout_s
        while remaining:
            wait = deadline-time.monotonic()
            if wait <= 0 or not select.select([self.p.stdout], [], [], wait)[0]:
                raise TimeoutError("camera frame transfer stalled for %.1f s" % timeout_s)
            chunk = os.read(self.p.stdout.fileno(), min(remaining, 1024*1024))
            if not chunk:
                raise RuntimeError("camera stream closed unexpectedly")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

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
