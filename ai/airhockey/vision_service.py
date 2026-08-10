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
# Encoded width. The sensor is 1440x1080 and the view is rotated into
# portrait, so 1080 is NATIVE — anything less throws pixels away before the
# browser ever sees them. It used to be 560, which is barely half, and that
# was invisible while the view was displayed at about 500 px wide and
# unquestionable. It stopped being invisible the moment the view could zoom:
# magnifying a half-resolution frame just magnifies the resampling.
#
# Detection has always run on the raw full-resolution frame and is unaffected
# either way; this is purely what gets shipped to the browser. The cost is
# bandwidth and JPEG encode time, both of which are cheap next to being
# unable to see what the tracker is looking at.
DISPLAY_W = 1080
# 75, not higher, and the cliff is sharp: OpenCV switches from 4:2:0 to
# 4:4:4 chroma somewhere in the mid-70s, and this frame costs 57 KB at q70
# against 121 KB at q78 for no visible gain. The source is a grayscale
# sensor image — the only colour in it is the overlay — so full chroma
# resolution is spent almost entirely on nothing.
JPEG_QUALITY = 75
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

# Inch/hole coordinates are quoted from the first hole RIGHT of the centre
# stripe. On the 80-column grid the stripe sits between columns 39 and 40,
# so 40 is the reference and holes land on whole numbers.
X_REF_COL = 40


def unproject_grid(nx: int = 41, ny: int = 41, z: float = 0.0) -> dict:
    """Map the VIEW's normalised coordinates to table millimetres.

    Sampled on a grid and interpolated in the browser rather than solved per
    mouse move: undistortion has no closed form, so doing it live would mean
    a round trip per pointer event. The mapping is smooth, so a 41x41 grid
    interpolates to well under the calibration's own error.

    Coordinates are normalised (0..1 across the view, 0..1 down it) so this
    survives any change to DISPLAY_W.
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "vision" / "bin"))
    from camera import backproject_pixels
    from track_mallet import load_pose

    K, dist, rvec, tvec, _ = load_pose()
    sh, sw = 1080, 1440                      # sensor height, width

    us = np.linspace(0.0, 1.0, nx)
    vs = np.linspace(0.0, 1.0, ny)
    px = []
    for v in vs:
        for u in us:
            # Inverse of the 180 + 90CW rotation _annotate applies:
            # forward was (sx, sy) -> (sy, sw - 1 - sx).
            dx, dy = u * (sh - 1), v * (sw - 1)
            px.append((sw - 1 - dy, dx))
    mm = backproject_pixels(np.array(px, dtype=np.float64), K, dist,
                            rvec, tvec, z)
    return {
        "nx": nx, "ny": ny, "z": z,
        "mm": [[round(float(a), 2), round(float(b), 2)] for a, b in mm],
    }


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
            # In inches from the centre stripe, which is the same as counting
            # air holes — the grid is a 25.4 mm pitch. x is measured from the
            # first hole right of the stripe rather than from the origin
            # corner: the stripe is painted and visible, and counting 40-odd
            # holes from the far end by eye is how an off-by-one turns into
            # an argument about a 25 mm calibration error.
            cv2.putText(vis, f"({cx / 25.4 - X_REF_COL:+.2f}, "
                             f"{cy / 25.4:.2f}) in",
                        (c[0] + 32, c[1] - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_PADDLE, 1,
                        cv2.LINE_AA)

        msg = note or ("paddle not found" if pose is None else None)
        if msg:
            cv2.putText(vis, msg[:52], (12, vis.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GLARE, 1, cv2.LINE_AA)

        vh, vw = vis.shape[:2]
        if DISPLAY_W < vw:      # never upscale — that only invents detail
            vis = cv2.resize(vis, (DISPLAY_W, int(vh * DISPLAY_W / vw)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", vis,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None


SERVICE = VisionService()
