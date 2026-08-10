#!/usr/bin/env python3
"""Camera extrinsics (pose over the table) from the permanent markers.

The pose is solved from FOUR markers only: the corner retroreflectors,
each centered in a known grid cell (table_grid.MARKERS_XY[:4]). Those are
the only markers whose position is grid-truth, so they are the only ones
that get a vote.

The two center-stripe markers are deliberately EXCLUDED from the fit and
used as held-out validation instead. Their x is independently known (the
painted stripe is registered to the hole grid, so stripe x == CENTERLINE_X)
while their y is not, which makes the x error an honest, unfitted accuracy
number in mm. Their y is measured and stored so the report can draw them
and so drift between sessions is visible.

Note what four points buys you: a 6-DOF pose from 8 constraints, only 2
spare. Reprojection RMS is therefore a weak accuracy proxy — a bad marker
placement is largely absorbed by the pose rather than showing up as
residual. Trust the held-out stripe error and the multi-frame spread over
the RMS. Pass several frames (the solve is joint over all of them) so
sensor noise averages down and the spread is measurable.

Capture (IR ring on, short exposure — ~1000 us / gain 0 works well: the
markers read bright while the scene stays black):

    ./build/snap extr_shots 8 1 --exposure 1000 --gain 0

The ring's own specular reflection off the table below the camera shows up
as a cluster of small LED dots; it is found automatically (near-peak
blobs merged morphologically, then an area gate), masked out, and the mask
is saved to vision/calib/glare_mask.png for the runtime tracker to reuse.
(The masked region is a permanent tracking blind spot — accepted.)

Marker <-> position correspondence is resolved automatically (8 global
orientation candidates x optimal assignment; the best joint reprojection
wins; poses below the table are rejected). One pose is solved over all
provided frames.

Writes vision/calib/extrinsics.npz/.json — the camera pose consumed by all
later steps (motor measurement, runtime tracking) — and markers.json.

Usage:
    python bin/calibrate_extrinsics.py extr_shots/*.png
    python bin/calibrate_extrinsics.py --selftest
"""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import backproject_undistorted  # noqa: E402
from table_grid import (CENTERLINE_X, GRID_X_MM, GRID_Y_MM,  # noqa: E402
                        MARKERS_XY)

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
MARKERS_FILE = CALIB_DIR / "markers.json"

# ONLY the corners constrain the pose — they are the only grid-truth
# positions. The rest are measured and held out as validation.
CORNERS_XY = MARKERS_XY[:4]
N_MARKERS = len(MARKERS_XY)
N_HELDOUT = N_MARKERS - len(CORNERS_XY)

MARKER_Z_MM = 3.3  # markers sit on 3D-printed mounts above the surface

# The marker layout (corner rectangle + centerline pair) is symmetric under
# a 180 deg rotation, so reprojection alone cannot tell the true pose from
# its rotated twin. One bit of mounting knowledge breaks the tie: which
# image side the robot end appears on. Flip this if the camera is ever
# remounted rotated (the solver will error out telling you to).
ROBOT_END_IMAGE_LEFT = True


def load_intrinsics():
    f = CALIB_DIR / "intrinsics.npz"
    if not f.exists():
        sys.exit("vision/calib/intrinsics.npz missing — run Stage A first")
    d = np.load(f)
    return d["camera_matrix"], d["dist_coeffs"]


def find_glare(img, min_area=400, grow=25):
    """Mask of the IR ring's specular reflection below the camera.

    At short exposure the reflection is a CLUSTER of small dots (the ring's
    individual LEDs), each no bigger than a marker — so single blobs can't
    be area-gated. Instead: threshold near the image peak, morphologically
    close to merge the cluster, then keep components that grew large.
    Markers are isolated, so they stay small and survive.
    """
    thr = max(100, int(img.max()) - 30)
    _, bw = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    mask = np.zeros_like(img)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= min_area:
            mask[lab == j] = 255
    if mask.any():
        mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
    return mask


def weighted_centroid(img, stats, j, pad=4):
    """Intensity-weighted centre of one blob, local background removed.

    A binary connected-component centroid quantises to whichever pixels
    happen to clear the threshold, which biases with blob brightness — and
    the corner markers are dimmer and more foreshortened than the middle
    ones, so that bias is systematic, not noise. Weighting by (I - local
    background) instead uses the full intensity profile and is stable to a
    fraction of a pixel.
    """
    x0 = max(0, stats[j, cv2.CC_STAT_LEFT] - pad)
    y0 = max(0, stats[j, cv2.CC_STAT_TOP] - pad)
    x1 = min(img.shape[1], x0 + stats[j, cv2.CC_STAT_WIDTH] + 2 * pad)
    y1 = min(img.shape[0], y0 + stats[j, cv2.CC_STAT_HEIGHT] + 2 * pad)
    patch = img[y0:y1, x0:x1].astype(np.float64)
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    w = np.clip(patch - np.median(border), 0, None)
    tot = w.sum()
    if tot <= 0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return np.array([(w * xs).sum() / tot, (w * ys).sum() / tot])


def detect_markers(img, mask, expect=len(MARKERS_XY), min_area=10,
                   max_area=600, border=40):
    """Sub-pixel centroids of the marker blobs, glare region excluded.

    `border` drops blobs hugging the frame edge. Every table marker sits on
    the playing field, which projects well inside the image (the closest is
    ~100 px from any edge), whereas stray reflective bits on the rails and
    floor show up in the outermost few pixels. Area cannot separate them —
    some strays are larger than a real marker — but position can.
    """
    work = img.copy()
    work[mask > 0] = 0
    thr = max(64, int(work.max()) // 2)
    _, bw = cv2.threshold(work, thr, 255, cv2.THRESH_BINARY)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw)
    blobs, edge = [], 0
    for j in range(1, n):
        area = stats[j, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        c = weighted_centroid(work, stats, j)
        c = cents[j] if c is None else c
        if not (border <= c[0] < img.shape[1] - border
                and border <= c[1] < img.shape[0] - border):
            edge += 1
            continue
        blobs.append((area, c))
    if edge:
        print(f"    ignored {edge} blob(s) within {border}px of the frame "
              "edge (off-field reflections)")
    if len(blobs) != expect:
        listing = ", ".join(f"area {int(a)} @({c[0]:.0f},{c[1]:.0f})"
                            for a, c in sorted(blobs, key=lambda b: -b[0]))
        sys.exit(f"found {len(blobs)} marker blobs, expected {expect}: "
                 f"[{listing}] — adjust exposure or remove stray reflectors")
    return np.array([b[1] for b in blobs], dtype=np.float64)


def normalize(pts):
    c = pts.mean(axis=0)
    d = pts - c
    s = np.sqrt((d ** 2).sum(axis=1).mean())
    return d / (s + 1e-12)


def sym_transform(k, mirror):
    """The 8 square symmetries (linear part only)."""
    th = k * np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if mirror:
        R = R @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    return R


def pose_is_sane(rvec, tvec, K):
    """Reject mirror/flipped PnP twins: camera must be ABOVE the table,
    probes in front of it, and the robot end on the configured image side."""
    Rm, _ = cv2.Rodrigues(rvec)
    if (-Rm.T @ tvec.reshape(3, 1)).ravel()[2] <= 0:
        return False  # camera below the playing surface
    probes = np.array([[0.0, GRID_Y_MM / 2, 0.0],
                       [GRID_X_MM, GRID_Y_MM / 2, 0.0]])
    pc = (Rm @ probes.T + tvec.reshape(3, 1)).T
    if pc[:, 2].min() <= 0:
        return False  # behind the camera
    pp, _ = cv2.projectPoints(probes, rvec, tvec, K, None)
    robot_left = pp[1, 0, 0] < pp[0, 0, 0]
    return robot_left == ROBOT_END_IMAGE_LEFT


def solve(img_pts_frames, field, K, dist, subset=False):
    """Joint pose over frames; correspondence (and optionally which blobs
    belong to `field` at all, when subset=True) resolved automatically.

    Returns the best pose dict; ["extras"] holds each frame's leftover
    undistorted image points when subset=True.
    """
    field = np.asarray(field, dtype=np.float64)
    und = [cv2.undistortPoints(p.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
           for p in img_pts_frames]
    n_f = len(field)

    best = None
    for k in range(4):
        for mirror in (False, True):
            T = sym_transform(k, mirror)
            nf = normalize(field @ T.T)
            ip_all, op_all, extras = [], [], []
            for u in und:
                subs = (combinations(range(len(u)), n_f) if subset
                        else [tuple(range(len(u)))])
                bsub = None
                for sub in subs:
                    su = u[list(sub)]
                    ni = normalize(su)
                    cost = ((ni[:, None, :] - nf[None, :, :]) ** 2).sum(axis=2)
                    ri, ci = linear_sum_assignment(cost)
                    c = cost[ri, ci].sum()
                    if bsub is None or c < bsub[0]:
                        bsub = (c, list(sub), ri, ci)
                _, sub, ri, ci = bsub
                su = u[sub]
                ip_all.append(su[ri])
                op_all.append(field[ci])
                extras.append(np.delete(u, sub, axis=0))
            ip = np.concatenate(ip_all).astype(np.float64)
            op = np.concatenate(op_all).astype(np.float64)
            op3 = np.hstack([op, np.full((len(op), 1), MARKER_Z_MM)])
            ok, rvec, tvec = cv2.solvePnP(op3, ip, K, None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or not pose_is_sane(rvec, tvec, K):
                continue
            proj, _ = cv2.projectPoints(op3, rvec, tvec, K, None)
            res = proj.reshape(-1, 2) - ip
            rms = float(np.sqrt((res ** 2).mean()))
            if best is None or rms < best["rms"]:
                best = {"rms": rms, "rvec": rvec, "tvec": tvec,
                        "k": k, "mirror": mirror, "res": res,
                        "extras": extras}
    if best is None:
        sys.exit("no sane pose candidate found — if the camera was "
                 "remounted rotated, flip ROBOT_END_IMAGE_LEFT="
                 f"{ROBOT_END_IMAGE_LEFT} in this file")
    return best


def mm_per_px(K, rvec, tvec, px):
    """Local plane scale at a pixel — for quoting residuals in mm."""
    p = np.array([px, px + np.array([1.0, 0.0])])
    w = backproject_undistorted(p, K, rvec, tvec, MARKER_Z_MM)
    return float(np.linalg.norm(w[1] - w[0]))


def report_spread(frames):
    """Per-marker centroid spread across frames — the measurement noise
    floor, which bounds how much of the residual is random vs systematic."""
    if len(frames) < 2:
        print("  NOTE: one frame only — pass several (e.g. `snap extr_shots "
              "8 1 ...`) to average noise down and expose the noise floor")
        return
    ref = frames[0]
    tracks = [[p] for p in ref]
    for f in frames[1:]:
        for i, p in enumerate(ref):
            d = np.linalg.norm(f - p, axis=1)
            j = int(np.argmin(d))
            if d[j] > 10.0:
                print("  WARNING: markers moved between frames "
                      f"({d[j]:.1f}px) — was the table bumped?")
            tracks[i].append(f[j])
    sp = [np.asarray(t).std(axis=0) for t in tracks]
    worst = max(float(np.hypot(*s)) for s in sp)
    print(f"  centroid spread over {len(frames)} frames: worst marker "
          f"{worst:.3f} px  (this is the random floor; anything above it "
          "in the residuals is systematic)")


def validate_heldout(best, K):
    """Measure the excluded stripe markers and score them against the one
    thing independently known about them: x lies on the grid centerline."""
    meas = []
    for e in best["extras"]:
        if len(e) != N_HELDOUT:
            sys.exit(f"expected {N_HELDOUT} non-corner markers, got {len(e)}")
        m = backproject_undistorted(e, K, best["rvec"], best["tvec"],
                                 MARKER_Z_MM)
        meas.append(m[np.argsort(m[:, 1])])
    meas = np.mean(meas, axis=0)

    print("\nHELD-OUT CHECK — stripe markers, not used in the fit.")
    print("x is known independently (painted stripe is registered to the "
          "hole grid); y is measured.")
    err = []
    for p in meas:
        e = p[0] - CENTERLINE_X
        err.append(abs(e))
        print(f"  measured ({p[0]:7.1f}, {p[1]:7.1f})   "
              f"x error {e:+6.2f} mm vs centerline {CENTERLINE_X:.1f}")
    print(f"  worst held-out error: {max(err):.2f} mm  <- the honest "
          "accuracy number near table centre")
    if max(err) > 3.0:
        print("  WARNING: >3mm. Most likely causes, in order: distortion "
              "model extrapolated past the calibration data at the corner "
              "markers; corner stickers not centred in their cells; grid "
              "count/pitch wrong.")
    return meas


def save_markers(meas):
    """Corners (grid-truth) + stripe pair at centerline x, measured y."""
    markers = [list(map(float, p)) for p in CORNERS_XY] + \
              [[CENTERLINE_X, round(float(p[1]), 1)] for p in meas]
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    with open(MARKERS_FILE, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"),
                   "note": "first 4 = grid-truth corners, the ONLY markers "
                           "in the fit; rest are held-out validation "
                           "(x = grid centerline, y measured)",
                   "n_fitted": len(CORNERS_XY),
                   "markers_mm": markers}, f, indent=2)
    print(f"saved {MARKERS_FILE}")


def report_and_save(best, n_frames, n_markers, K):
    R, _ = cv2.Rodrigues(best["rvec"])
    cam_pos = (-R.T @ best["tvec"]).ravel()
    per_pt = np.linalg.norm(best["res"], axis=1)

    scale = mm_per_px(K, best["rvec"], best["tvec"],
                      np.array([K[0, 2], K[1, 2]]))
    print(f"\norientation hypothesis: rot90 x{best['k']}"
          f"{' + mirror' if best['mirror'] else ''}")
    print(f"reprojection RMS {best['rms']:.3f} px over {n_frames} frame(s) "
          f"of {n_markers} FITTED markers, max {per_pt.max():.3f} px "
          f"(~{best['rms'] * scale:.2f} mm at {scale:.2f} mm/px)")
    print("  (4 points fit a 6-DOF pose, so this is a weak accuracy proxy — "
          "read the held-out check above instead)")
    if best["rms"] > 1.0:
        print("WARNING: high residuals — check marker placement (1.5 pitch "
              "inset), stray blobs, or intrinsics staleness")
    print(f"camera position (grid frame, mm): "
          f"x={cam_pos[0]:.1f} y={cam_pos[1]:.1f} z={cam_pos[2]:.1f} "
          f"(z above the playing surface)")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CALIB_DIR / "extrinsics.npz",
             rvec=best["rvec"], tvec=best["tvec"], camera_pos=cam_pos,
             rms=best["rms"], dot_height_mm=MARKER_Z_MM)
    with open(CALIB_DIR / "extrinsics.json", "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "frame": "grid (origin = corner hole nearest human's right)",
            "camera_pos_mm": [round(float(v), 2) for v in cam_pos],
            "rvec": best["rvec"].ravel().tolist(),
            "tvec": best["tvec"].ravel().tolist(),
            "rms_px": round(best["rms"], 4),
        }, f, indent=2)
    print(f"saved {CALIB_DIR}/extrinsics.npz and .json")


def read_frames(images):
    """Detect marker blobs in each image; save the first glare mask."""
    frames = []
    glare_saved = False
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"cannot read {path}")
        mask = find_glare(img)
        n_mask = int((mask > 0).sum())
        if not glare_saved:
            CALIB_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(CALIB_DIR / "glare_mask.png"), mask)
            glare_saved = True
            print(f"  glare mask: {n_mask} px masked "
                  f"({'saved' if n_mask else 'EMPTY — no glare cluster found'})"
                  f" -> calib/glare_mask.png")
        pts = detect_markers(img, mask)
        frames.append(pts)
        print(f"  {path}: {len(pts)} markers")
    return frames


def run(images):
    K, dist = load_intrinsics()
    frames = read_frames(images)
    report_spread(frames)
    best = solve(frames, CORNERS_XY, K, dist, subset=True)
    meas = validate_heldout(best, K)
    save_markers(meas)
    report_and_save(best, len(frames), len(CORNERS_XY), K)


def selftest():
    rng = np.random.default_rng(5)
    K = np.array([[997.0, 0, 712.7], [0, 995.6, 563.2], [0, 0, 1.0]])
    # True pose consistent with the real mounting: robot end on image LEFT.
    R0, _ = cv2.Rodrigues(np.array([np.pi + 0.06, 0.04, 0.10]))
    Rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.pi]))
    R = Rz @ R0
    rvec_true, _ = cv2.Rodrigues(R)
    cam = np.array([960.0, 440.0, 1810.0])
    tvec_true = (-R @ cam.reshape(3, 1))

    # "True" marker layout: grid corners + stripe pair NOT at nominal spots
    # (offset stripe, half-square inset — like the real table).
    field = np.array(list(CORNERS_XY) + [[988.0, 12.5], [988.0, 924.9]])
    obj = np.hstack([field, np.full((len(field), 1), MARKER_Z_MM)])
    frames = []
    for _ in range(2):
        proj, _ = cv2.projectPoints(obj, rvec_true, tvec_true, K, np.zeros(5))
        pts = proj.reshape(-1, 2) + rng.normal(0, 0.3, (len(field), 2))
        frames.append(rng.permutation(pts))  # scrambled order

    # The real path: corners-only subset solve, stripe pair held out.
    best = solve(frames, CORNERS_XY, K, np.zeros(5), subset=True)
    meas = backproject_undistorted(best["extras"][0], K, best["rvec"],
                                best["tvec"], MARKER_Z_MM)
    meas = meas[np.argsort(meas[:, 1])]
    merr = np.abs(meas - field[4:]).max()
    print(f"selftest held-out: stripe markers measured to {merr:.2f}mm")
    assert merr < 5.0, f"stripe measurement error {merr:.2f}mm"

    Rb, _ = cv2.Rodrigues(best["rvec"])
    cam_est = (-Rb.T @ best["tvec"]).ravel()
    err = np.linalg.norm(cam_est - cam)
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(Rb @ R.T)[0]))
    print(f"selftest: rms {best['rms']:.3f}px, camera err {err:.2f}mm, "
          f"orientation err {ang:.4f}deg, hypothesis k={best['k']} "
          f"mirror={best['mirror']}")
    assert best["rms"] < 0.6, "did not reach noise floor"
    assert err < 8.0, f"camera position error {err:.2f}mm"
    assert ang < 0.15, f"orientation error {ang:.4f}deg"

    # glare masking on a synthetic frame: big saturated blob + dim markers
    img = np.zeros((1080, 1440), np.uint8)
    cv2.circle(img, (720, 540), 40, 255, -1)          # glare
    for x, y in [(100, 100), (1340, 100), (100, 980), (1340, 980),
                 (720, 100), (720, 980)]:
        cv2.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), 220, -1)
    mask = find_glare(img)
    assert mask[540, 720] > 0 and mask[100, 100] == 0, "glare mask wrong"
    pts = detect_markers(img, mask)
    assert len(pts) == len(MARKERS_XY), "marker detection through mask failed"
    print("selftest: glare masking + marker detection PASSED")
    print("selftest PASSED")


def capture(n, exposure, gain, out_dir):
    """Grab n frames straight from the camera, so re-calibrating is one
    command rather than snap-then-glob. The camera can only be held by one
    process: stop the tracker view in the web UI first."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camera import Stream

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()          # frames from before the camera moved are poison
    stream = Stream(exposure, gain)
    paths = []
    try:
        while len(paths) < n:
            img = stream.grab()
            if img is None:
                continue
            p = out / f"shot_{len(paths):03d}.png"
            cv2.imwrite(str(p), img)
            paths.append(str(p))
    finally:
        stream.close()
    print(f"captured {len(paths)} frames at {exposure} us / gain {gain} "
          f"into {out}\n")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", help="low-exposure frames")
    ap.add_argument("--capture", type=int, metavar="N",
                    help="grab N frames from the camera and solve from those. "
                         "TAKE THE MALLET OFF THE TABLE FIRST — its "
                         "retroreflectors are indistinguishable from field "
                         "markers and the solve refuses to guess.")
    ap.add_argument("--exposure", type=int, default=1000)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--shots-dir", default=str(
        Path(__file__).resolve().parent.parent / "extr_shots"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    images = args.images
    if args.capture:
        images = capture(args.capture, args.exposure, args.gain,
                         args.shots_dir)
    if not images:
        sys.exit(__doc__)
    run(images)


if __name__ == "__main__":
    main()
