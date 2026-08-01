"""Owns the camera; publishes paddle pose and an annotated live view.

Only one process can hold the Spinnaker device, and within a process only
one thing should hold the stream — so this is the single owner. Anything
that wants a paddle position asks this rather than opening its own camera,
which is why `latest_pose()` exists alongside the video.

The view is the LOW-EXPOSURE tracking frame, not a photograph of the table.
At 1000 us with the IR ring on, the scene is essentially black and the
retroreflectors are bright dots — that is what makes detection reliable, and
it is deliberately not a watchable picture. What makes it useful is the
overlay: you are seeing what the tracker sees, with everything it has
identified named. Brightness is boosted for display only; detection always
runs on the raw frame.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "vision" / "bin", _ROOT / "shared"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cdpr_geometry as geom  # noqa: E402

# Output is PORTRAIT to match the sim field beside it: the table's long
# axis (grid x) runs vertically with the robot end at the bottom, and grid y
# increases to the right — the same convention the sim canvas uses. That is
# the sensor frame rotated 180 (origin bottom-left) and then 90 clockwise.
DISPLAY_W = 560          # encoded width; detection always uses full res
JPEG_QUALITY = 72
TARGET_FPS = 12.0

# Overlay colours (BGR).
C_FIELD = (90, 200, 90)
C_PADDLE = (60, 190, 255)
C_ARM = (200, 160, 60)
C_GLARE = (60, 110, 230)
C_TEXT = (240, 240, 240)
C_GRID = (150, 150, 150)      # projected field border
C_STRIPE = (200, 90, 200)     # projected centreline — lands on the painted stripe
C_WS = (110, 110, 110)        # workspace limit
C_MOTOR = (90, 230, 230)      # measured motor anchors


class VisionService:
    """Background camera owner. Start it, read `latest_pose()` / `frame_jpeg()`."""

    def __init__(self, exposure_us: int = 1000, gain_db: float = 0.0):
        self._exposure = exposure_us
        self._gain = gain_db
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._pose: tuple[float, float, float] | None = None
        self._note: str | None = None
        self._fps = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def error(self) -> str | None:
        return self._error

    # ── readers ──────────────────────────────────────────────────────
    def latest_pose(self):
        """(x_mm, y_mm, theta_rad) or None. Whoever needs the paddle asks
        here instead of opening the camera a second time."""
        with self._lock:
            return self._pose

    def frame_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        with self._lock:
            pose, note, fps = self._pose, self._note, self._fps
        return {
            "running": self.running,
            "error": self._error,
            "fps": round(fps, 1),
            "note": note,
            "pose": None if pose is None else {
                "x": round(pose[0], 1),
                "y": round(pose[1], 1),
                "theta_deg": round(math.degrees(pose[2]), 2),
            },
        }

    # ── worker ───────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            import track_mallet as tm
            from camera import Stream

            K, dist, rvec, tvec, field = tm.load_pose()
            known_px = tm.field_marker_pixels(K, dist, rvec, tvec, field)
            stream = Stream(self._exposure, self._gain)
        except Exception as e:                       # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            return

        last = time.time()
        try:
            while not self._stop.is_set():
                img = stream.grab()
                if img is None:
                    continue
                now = time.time()
                dt = now - last
                last = now

                pose, note = tm.locate(img, K, dist, rvec, tvec, field)
                jpeg = self._annotate(img, known_px, pose, note, tm,
                                      K, dist, rvec, tvec)
                with self._lock:
                    self._jpeg = jpeg
                    self._note = note
                    self._fps = 0.85 * self._fps + 0.15 / max(dt, 1e-6)
                    if pose is not None and pose.get("theta") is not None:
                        self._pose = (float(pose["centre"][0]),
                                      float(pose["centre"][1]),
                                      float(pose["theta"]))
                # Pace it: the tracker does not need every frame, and the
                # browser cannot use them.
                slack = 1.0 / TARGET_FPS - (time.time() - now)
                if slack > 0:
                    self._stop.wait(slack)
        except Exception as e:                       # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
        finally:
            try:
                stream.close()
            except Exception:                        # noqa: BLE001
                pass

    # ── drawing ──────────────────────────────────────────────────────
    def _annotate(self, img, known_px, pose, note, tm, K, dist, rvec, tvec):
        """Draw in DISPLAY space, not sensor space.

        The view is rotated 180 so +x reads right and the origin is bottom
        left, matching the calibration report. Annotating first and rotating
        afterwards flips every label upside down, so the rotation happens
        first and each point is mapped through it before being drawn.
        """
        # Boost for display only — detection above ran on the raw frame.
        vis = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=3.0),
                           cv2.COLOR_GRAY2BGR)
        vis = cv2.rotate(vis, cv2.ROTATE_180)
        vis = cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE)
        H, W = img.shape[:2]          # SENSOR dims; display is (H wide, W tall)

        def flip(p):
            # sensor (x, y) -> 180 -> 90cw. Composing the two gives
            # (y, W-1-x) in an image that is H wide and W tall.
            return (int(round(p[1])), int(round(W - 1 - p[0])))

        def project(xy, z):
            px, _ = cv2.projectPoints(np.array([[xy[0], xy[1], z]]),
                                      rvec, tvec, K, dist)
            return flip(px.reshape(2))

        # Model geometry drawn THROUGH the calibration, so the view answers
        # "does the code know where the table is?" and not merely "what did
        # it detect?". The centre stripe is visible in the raw frame, so the
        # magenta line landing on it is a live, continuous check that the
        # pose is still right — no separate calibration run needed.
        def polyline(pts, colour, z=0.0, thick=1):
            prev = None
            for q in pts:
                cur = project(q, z)
                if prev is not None:
                    cv2.line(vis, prev, cur, colour, thick, cv2.LINE_AA)
                prev = cur

        n = 24
        lo = lambda a, b, t: a + (b - a) * t                       # noqa: E731
        polyline([(lo(0, geom.GRID_X_MM, i / n), 0) for i in range(n + 1)],
                 C_GRID)
        polyline([(lo(0, geom.GRID_X_MM, i / n), geom.GRID_Y_MM)
                  for i in range(n + 1)], C_GRID)
        polyline([(0, lo(0, geom.GRID_Y_MM, i / n)) for i in range(n + 1)],
                 C_GRID)
        polyline([(geom.GRID_X_MM, lo(0, geom.GRID_Y_MM, i / n))
                  for i in range(n + 1)], C_GRID)
        polyline([(geom.CENTERLINE_X, lo(0, geom.GRID_Y_MM, i / n))
                  for i in range(n + 1)], C_STRIPE)
        # Workspace the paddle is allowed into.
        polyline([(lo(geom.WS_MIN_X, geom.WS_MAX_X, i / n), geom.WS_MIN_Y)
                  for i in range(n + 1)], C_WS)
        polyline([(lo(geom.WS_MIN_X, geom.WS_MAX_X, i / n), geom.WS_MAX_Y)
                  for i in range(n + 1)], C_WS)
        polyline([(geom.WS_MIN_X, lo(geom.WS_MIN_Y, geom.WS_MAX_Y, i / n))
                  for i in range(n + 1)], C_WS)
        polyline([(geom.WS_MAX_X, lo(geom.WS_MIN_Y, geom.WS_MAX_Y, i / n))
                  for i in range(n + 1)], C_WS)
        # Motor anchors, at the spool-top plane where they were measured.
        for m in range(4):
            a = project((geom.MOTOR_X[m], geom.MOTOR_Y[m]), 36.0)
            cv2.drawMarker(vis, a, C_MOTOR, cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"M{m}", (a[0] + 10, a[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_MOTOR, 1,
                        cv2.LINE_AA)

        mask = tm.find_glare(img)
        if mask.any():
            rot = cv2.rotate(cv2.rotate(mask, cv2.ROTATE_180),
                             cv2.ROTATE_90_CLOCKWISE)
            cnts, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, C_GLARE, 2)

        for i, p in enumerate(known_px):
            c = flip(p)
            cv2.circle(vis, c, 13, C_FIELD, 1)
            cv2.putText(vis, f"F{i}", (c[0] + 16, c[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_FIELD, 1,
                        cv2.LINE_AA)

        if pose is not None:
            cx, cy = float(pose["centre"][0]), float(pose["centre"][1])
            c = project((cx, cy), tm.MALLET_Z_MM)
            cv2.circle(vis, c, 26, C_PADDLE, 2)
            cv2.drawMarker(vis, c, C_PADDLE, cv2.MARKER_CROSS, 16, 1)
            th = pose.get("theta")
            if th is not None:
                r = pose.get("arm_r") or 25.0
                tip = project((cx + r * math.cos(th), cy + r * math.sin(th)),
                              tm.ARM_Z_MM)
                cv2.arrowedLine(vis, c, tip, C_ARM, 2, tipLength=0.35)
                cv2.putText(vis, f"{math.degrees(th):.1f} deg",
                            (c[0] + 32, c[1] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ARM, 1,
                            cv2.LINE_AA)
            cv2.putText(vis, f"({cx:.0f}, {cy:.0f}) mm", (c[0] + 32, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_PADDLE, 1,
                        cv2.LINE_AA)

        msg = note or ("paddle not found" if pose is None else None)
        if msg:
            cv2.putText(vis, msg[:52], (12, vis.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GLARE, 1, cv2.LINE_AA)

        vh, vw = vis.shape[:2]
        vis = cv2.resize(vis, (DISPLAY_W, int(vh * DISPLAY_W / vw)),
                         interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", vis,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None


SERVICE = VisionService()
