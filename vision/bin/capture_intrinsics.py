#!/usr/bin/env python3
"""Continuous, self-selecting ChArUco capture for intrinsics.

Just move the board around in front of the camera. This streams frames,
detects the board on every one, and keeps a frame only when it actually
adds information — no shutter button, no posing to a grid.

Why it exists: the Spinnaker path has no live preview, so intrinsics
capture is blind, and blind capture bunches every view in the middle of the
frame. That yields a low reprojection RMS which is quietly wrong — radial
distortion outside the captured region is an extrapolation, and the table
corners live out there. Coverage, not view count, is what matters.

A frame is kept only if it passes, in order:
  1. DETECTED   — enough ChArUco corners to be worth anything
  2. STILL      — corners moved < a pixel since the last frame, which
                  rejects motion blur far more reliably than any sharpness
                  score (a blurred board still "detects", it just detects
                  in the wrong place)
  3. NEW        — it fills occupancy cells no kept frame has covered, or it
                  contributes a board tilt that is under-represented.
                  Tilt matters independently of position: head-on views
                  barely constrain focal length no matter where they sit.
Everything else is dropped, so the kept set stays small and diverse.

WHERE TO MOVE THE BOARD: toward wherever the map still shows gaps, and
especially the frame edges and corners — 'o' marks periphery cells with no
data yet. Holding the board HIGHER (closer to the camera) makes the frame
corners easier to reach: at table level they fall outside the side rails,
but around halfway up to the camera they fall inside the table footprint.
The lens is wide, so depth of field runs from ~0.5 m to infinity. Keep the
board tilted and vary the tilt direction.

DO NOT touch focus or aperture during or after this — intrinsics are only
valid for the optical configuration they were captured in.

Usage:
    python bin/capture_intrinsics.py                 # -> calib_shots/
    python bin/capture_intrinsics.py -o my_shots --exposure 20000
Then:
    python bin/calibrate_intrinsics.py --images 'calib_shots/*.png'
    python bin/check_intrinsics.py
"""

import argparse
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_intrinsics import make_detector  # noqa: E402
from gen_targets import CHARUCO_COLS, CHARUCO_ROWS, SQUARE_MM  # noqa: E402

VISION = Path(__file__).resolve().parent.parent
SNAP = VISION / "build" / "snap"

CELL = 60               # px per occupancy cell
MIN_CORNERS = 20        # to KEEP a frame (detection itself allows fewer)
MOTION_PX = 1.0         # median corner motion below which the board is still
MIN_NEW_CELLS = 3       # new coverage required to keep a frame
PERIPHERY_FRAC = 0.75   # "out there" = beyond this fraction of corner radius
TILT_EDGES = [0, 12, 25, 40, 90]   # degrees, board normal vs optical axis
TILT_TARGET = 6         # kept views wanted per tilt bin
COVERAGE_TARGET = 0.55  # fraction of reachable cells
MAX_VIEWS = 60          # calibration gains nothing past this


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


def detect_full(board, det, gray):
    """Board detection that also hands back corner IDs, so stillness can be
    judged per-corner rather than by counting."""
    ch_corners, ch_ids, _, _ = det.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < MIN_CORNERS:
        return None, 0 if ch_ids is None else len(ch_ids)
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    if obj is None or len(obj) < 6:
        return None, len(ch_ids)
    return (obj, img, ch_ids.ravel(), ch_corners.reshape(-1, 2)), len(ch_ids)


def median_motion(prev, ids, pts):
    """Median displacement of corners common to both frames, or None if
    they share too few to judge."""
    prev_by_id = prev
    common = [i for i in ids.tolist() if i in prev_by_id]
    if len(common) < 8:
        return None
    cur = {int(i): p for i, p in zip(ids, pts)}
    d = [np.linalg.norm(cur[i] - prev_by_id[i]) for i in common]
    return float(np.median(d))


def provisional_K(w, h):
    """Intrinsics good enough to estimate board tilt. The previous
    calibration if there is one, else a plain pinhole guess."""
    calib = VISION / "calib" / "intrinsics.npz"
    if calib.exists():
        d = np.load(calib)
        if tuple(d["image_size"]) == (w, h):
            return d["camera_matrix"], d["dist_coeffs"]
    return np.array([[w * 0.7, 0, w / 2], [0, w * 0.7, h / 2],
                     [0, 0, 1.0]]), np.zeros(5)


def tilt_deg(obj, img, K, dist):
    ok, rvec, _ = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # angle between the board normal and the optical axis
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


class Coverage:
    def __init__(self, w, h):
        self.nx, self.ny = int(np.ceil(w / CELL)), int(np.ceil(h / CELL))
        self.occ = np.zeros((self.ny, self.nx), bool)
        gy, gx = np.mgrid[0:self.ny, 0:self.nx]
        cx, cy = (gx + 0.5) * CELL, (gy + 0.5) * CELL
        r = np.hypot(cx - w / 2, cy - h / 2)
        self.periph = r > PERIPHERY_FRAC * np.hypot(w / 2, h / 2)
        # Corner cells beyond the image circle are unreachable in practice;
        # count coverage against what can actually be filled.
        self.reachable = r <= np.hypot(w / 2, h / 2)

    def cells(self, pts):
        cx = np.clip((pts[:, 0] // CELL).astype(int), 0, self.nx - 1)
        cy = np.clip((pts[:, 1] // CELL).astype(int), 0, self.ny - 1)
        return set(zip(cy.tolist(), cx.tolist()))

    def new_count(self, pts):
        return sum(0 if self.occ[c] else 1 for c in self.cells(pts))

    def add(self, pts):
        for c in self.cells(pts):
            self.occ[c] = True

    def fraction(self):
        return float((self.occ & self.reachable).sum()
                     / max(1, self.reachable.sum()))

    def periphery_done(self):
        return (int((self.occ & self.periph).sum()),
                int(self.periph.sum()))

    def render(self):
        out = []
        for r in range(self.ny):
            row = ""
            for c in range(self.nx):
                if self.occ[r, c]:
                    row += "#"
                elif not self.reachable[r, c]:
                    row += " "
                elif self.periph[r, c]:
                    row += "o"
                else:
                    row += "."
            out.append("   |" + row + "|")
        return out


def selftest():
    """Exercise the whole selection core on rendered boards — no camera.

    The board is rendered by actually projecting its plane through a pinhole
    camera, so a commanded tilt is a real tilt. (Warping to a hand-picked
    quad and estimating its pose with distortion coefficients the render
    never had would invent tilt that isn't there.)
    """
    board, det = make_detector()
    W, H = 1440, 1080
    K = np.array([[997.0, 0, W / 2], [0, 997.0, H / 2], [0, 0, 1.0]])
    dist = np.zeros(5)
    # Board image must share the plane's aspect or the render shears it.
    sq = SQUARE_MM / 1000.0
    wm, hm = CHARUCO_COLS * sq, CHARUCO_ROWS * sq
    bw = 550
    bh = int(round(bw * hm / wm))
    bimg = board.generateImage((bw, bh))
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    objc = np.float32([[0, 0, 0], [wm, 0, 0], [wm, hm, 0], [0, hm, 0]])

    def frame_at(cx, cy, tilt=0.0, z=1.2):
        """Board centred near image (cx, cy), rotated `tilt` deg about y."""
        rvec = np.array([0.0, np.radians(tilt), 0.0])
        R, _ = cv2.Rodrigues(rvec)
        # put the board's centre on the ray through (cx, cy)
        ray = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
        centre = ray / ray[2] * z
        tvec = centre - R @ np.array([wm / 2, hm / 2, 0.0])
        px, _ = cv2.projectPoints(objc, rvec, tvec, K, dist)
        out = np.full((H, W), 255, np.uint8)
        M = cv2.getPerspectiveTransform(src, px.reshape(-1, 2).astype(np.float32))
        return cv2.warpPerspective(bimg, M, (W, H), dst=out,
                                   borderMode=cv2.BORDER_TRANSPARENT)

    f1 = frame_at(400, 300)
    got, n = detect_full(board, det, f1)
    assert got is not None, f"board not detected in a clean render ({n})"
    obj, ip, ids, _ = got
    pts = ip.reshape(-1, 2)
    print(f"selftest: detected {n} corners")

    prev = {int(i): p for i, p in zip(ids, pts)}
    got2, _ = detect_full(board, det, f1)
    m_still = median_motion(prev, got2[2], got2[1].reshape(-1, 2))
    got3, _ = detect_full(board, det, frame_at(412, 300))
    m_moved = median_motion(prev, got3[2], got3[1].reshape(-1, 2))
    print(f"selftest: motion still {m_still:.3f} px, shifted "
          f"{m_moved:.1f} px")
    assert m_still < MOTION_PX, "identical frames judged as moving"
    assert m_moved > MOTION_PX, "a 12px shift was judged as still"

    t_flat = tilt_deg(obj, ip, K, dist)
    g4, _ = detect_full(board, det, frame_at(700, 540, tilt=35))
    t_tilt = tilt_deg(g4[0], g4[1], K, dist)
    print(f"selftest: tilt commanded 0/35deg -> measured "
          f"{t_flat:.1f}/{t_tilt:.1f}deg")
    assert t_flat < 8, f"flat board measured as {t_flat:.1f}deg"
    assert abs(t_tilt - 35) < 8, f"35deg board measured as {t_tilt:.1f}deg"

    cov = Coverage(W, H)
    first = cov.new_count(pts)
    cov.add(pts)
    assert first > 0 and cov.new_count(pts) == 0, "coverage not idempotent"
    g5, _ = detect_full(board, det, frame_at(1100, 800))
    assert cov.new_count(g5[1].reshape(-1, 2)) >= MIN_NEW_CELLS, \
        "a board in a fresh region added no new cells"
    print(f"selftest: coverage {first} cells from view 1, "
          f"elsewhere adds {cov.new_count(g5[1].reshape(-1, 2))}")
    print("selftest PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(VISION / "calib_shots"))
    ap.add_argument("--exposure", default="auto")
    ap.add_argument("--gain", default="0")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    outdir = Path(args.out)
    if outdir.exists() and any(outdir.glob("shot_*.png")):
        n = len(list(outdir.glob("shot_*.png")))
        sys.exit(f"{outdir} already holds {n} shots. Move them aside first — "
                 "mixing optical\nconfigurations silently corrupts the "
                 f"calibration:\n  mv {outdir} {outdir}_old")
    outdir.mkdir(parents=True, exist_ok=True)

    print(__doc__.split("Usage:")[0].split("A frame is kept")[0])
    board, det = make_detector()
    stream = Stream(args.exposure, args.gain)
    K, dist = provisional_K(stream.w, stream.h)
    cov = Coverage(stream.w, stream.h)
    tilt_hits = [0] * (len(TILT_EDGES) - 1)

    prev = None          # (ids, corners) of the previous detected frame
    kept = 0
    status = "starting..."
    lines_drawn = 0
    last_draw = 0.0
    fps, t_prev = 0.0, time.time()
    announced = False

    try:
        while kept < MAX_VIEWS:
            if select.select([sys.stdin], [], [], 0)[0]:
                if "q" in sys.stdin.readline().lower():
                    break

            img = stream.grab()
            if img is None:
                continue
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
            t_prev = now

            found, n = detect_full(board, det, img)
            if found is None:
                status = f"searching — {n} corners"
                prev = None
            else:
                obj, ip, ids, ch = found
                pts = ip.reshape(-1, 2)
                motion = None if prev is None else median_motion(prev, ids, pts)
                prev = {int(i): p for i, p in zip(ids, pts)}
                if motion is None:
                    status = f"hold still — {n} corners"
                else:
                    if motion > MOTION_PX:
                        status = f"moving ({motion:.1f} px) — hold still"
                    else:
                        new = cov.new_count(pts)
                        t = tilt_deg(obj, ip, K, dist)
                        tb = (int(np.digitize(t, TILT_EDGES) - 1)
                              if t is not None else 0)
                        tb = min(max(tb, 0), len(tilt_hits) - 1)
                        want_tilt = tilt_hits[tb] < TILT_TARGET and new >= 1
                        if new >= MIN_NEW_CELLS or want_tilt:
                            cv2.imwrite(str(outdir / f"shot_{kept:03d}.png"),
                                        img)
                            cov.add(pts)
                            tilt_hits[tb] += 1
                            kept += 1
                            why = ("new area" if new >= MIN_NEW_CELLS
                                   else f"tilt {t:.0f}deg under-sampled")
                            status = f"KEPT #{kept} ({why}, +{new} cells)"
                            prev = None   # force a fresh stillness check
                        else:
                            status = (f"already covered here — move on "
                                      f"(tilt {t:.0f}deg)")

            if now - last_draw < 0.12:
                continue
            last_draw = now
            done, tot = cov.periphery_done()
            body = ([f"   kept {kept:2d}   {fps:4.1f} fps   {status}",
                     f"   coverage {cov.fraction():4.0%} "
                     f"(target {COVERAGE_TARGET:.0%})   "
                     f"periphery {done}/{tot} cells",
                     "   tilt  " + "  ".join(
                         f"{TILT_EDGES[i]}-{TILT_EDGES[i + 1]}deg:{tilt_hits[i]}"
                         for i in range(len(tilt_hits))),
                     ""] + cov.render()
                    + ["", "   # covered   o periphery gap   . gap",
                       "   q + Enter to finish"])
            if lines_drawn:
                sys.stdout.write(f"\033[{lines_drawn}A")
            sys.stdout.write("\n".join(s.ljust(90) for s in body) + "\n")
            sys.stdout.flush()
            lines_drawn = len(body)

            if (not announced and cov.fraction() >= COVERAGE_TARGET
                    and done == tot and min(tilt_hits) >= TILT_TARGET):
                announced = True
                status = "TARGET MET — press q + Enter (more won't hurt)"
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()

    print(f"\n{kept} views saved to {outdir}")
    if kept < 12:
        print("That is too few to calibrate — aim for 20+ with full coverage.")
    print(f"\nNext:\n  python bin/calibrate_intrinsics.py --images "
          f"'{outdir}/*.png'\n  python bin/check_intrinsics.py "
          f"--images '{outdir}/*.png'")


if __name__ == "__main__":
    main()
