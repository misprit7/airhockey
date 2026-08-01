#!/usr/bin/env python3
"""Visual verification report for the camera calibration.

Generates a self-contained HTML page with canvas-based pan/zoom viewers
(drag to pan, scroll to zoom, double-click to fit), a toggleable annotation
overlay, and a live cursor readout of grid-frame mm coordinates computed
through the full camera model — hover any hole and it should read a
multiple of 25.4.

  1. The low-exposure calibration frame: detected marker blobs vs the
     stored markers reprojected through the solved pose, per-marker
     residuals, and the glare-mask outline.
  2. An ambient frame (optional but recommended): the entire hole grid,
     field border, centerline, rails, and markers projected through the
     pose. If it lines up with the visible table everywhere, the
     calibration is good.

Both frames must be shot with the camera in its calibrated mounting.
Images are displayed rotated 180 deg so the grid axes read in natural
graph convention (origin bottom-left, +x right, +y up).

Usage:
    python bin/calib_report.py extr_shots/shot_000.png \
        [--ambient shots/ambient.png] [-o calib/report.html]
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_extrinsics import (MARKER_Z_MM, MARKERS_FILE,  # noqa: E402
                                  detect_markers, find_glare,
                                  load_intrinsics)
from table_grid import (CENTERLINE_X, GRID_X_MM, GRID_Y_MM,  # noqa: E402
                        N_COLS, N_ROWS, PITCH_MM, RAIL_MAX_X, RAIL_MAX_Y,
                        RAIL_MIN_X, RAIL_MIN_Y)

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"

# Annotation colors (BGRA on the transparent overlay layer).
GREEN = (80, 220, 80, 255)
MAGENTA = (230, 80, 230, 255)
ORANGE = (60, 160, 255, 255)
CYAN = (255, 210, 90, 255)
YELLOW = (80, 235, 245, 255)
RED = (70, 70, 255, 255)
WHITE = (240, 240, 240, 255)
SHADOW = (20, 24, 26, 255)

MM_MAP_STEP = 20  # px between pixel->mm map samples embedded for the HUD


def load_all():
    K, dist = load_intrinsics()
    fe = CALIB_DIR / "extrinsics.npz"
    if not fe.exists():
        sys.exit("extrinsics.npz missing — run calibrate_extrinsics.py first")
    de = np.load(fe)
    if not MARKERS_FILE.exists():
        sys.exit(f"{MARKERS_FILE} missing — run calibrate_extrinsics.py")
    with open(MARKERS_FILE) as f:
        d = json.load(f)
    markers = np.array(d["markers_mm"], dtype=np.float64)
    return (K, dist, de["rvec"], de["tvec"], de["camera_pos"], markers,
            int(d.get("n_fitted", len(markers))))


def load_motors():
    """Measured motor anchors, if measure_motors.py has been run."""
    f = CALIB_DIR / "motor_anchors.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def project(field_xy, K, dist, rvec, tvec, z=0.0):
    """Grid-frame (x, y) points at height z -> raw (distorted) pixels."""
    obj = np.hstack([np.asarray(field_xy, np.float64),
                     np.full((len(field_xy), 1), z)])
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def in_frame(p, shape, margin=4):
    return (-margin <= p[0] < shape[1] + margin
            and -margin <= p[1] < shape[0] + margin)


# Display frames are rotated 180 deg (origin bottom-left, +x right, +y up).
# Detection/projection run in the ORIGINAL pixel frame; coordinates are
# flipped just before drawing.
def flipper(shape):
    H, W = shape[:2]
    return lambda p: np.array([W - 1 - p[0], H - 1 - p[1]])


def mm_map(shape, K, dist, rvec, tvec, step=MM_MAP_STEP):
    """Sampled DISPLAY-pixel -> grid-frame mm map (z=0 plane) for the HUD."""
    H, W = shape[:2]
    xs = np.arange(0, W + step, step, dtype=np.float64)
    ys = np.arange(0, H + step, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    # display -> original pixels (180 deg rotation)
    pts = np.stack([W - 1 - gx.ravel(), H - 1 - gy.ravel()], axis=1)
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec.reshape(3, 1)).ravel()
    dirs = (R.T @ np.vstack([und.T, np.ones(len(und))]))
    t = -C[2] / dirs[2]
    world = (C[:, None] + dirs * t)[:2].T
    return {"step": step, "cols": len(xs), "rows": len(ys),
            "x": [round(float(v), 1) for v in world[:, 0]],
            "y": [round(float(v), 1) for v in world[:, 1]]}


def new_overlay(shape):
    return np.zeros((shape[0], shape[1], 4), np.uint8)


def label(ov, text, org, color, scale=0.55):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = org[0]
    if x + tw > ov.shape[1] - 6:
        x = org[0] - 32 - tw
    cv2.putText(ov, text, (x, org[1]), cv2.FONT_HERSHEY_SIMPLEX, scale,
                SHADOW, 4, cv2.LINE_AA)
    cv2.putText(ov, text, (x, org[1]), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def annotate_lowexp(path, K, dist, rvec, tvec, markers):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"cannot read {path}")
    F = flipper(img.shape)
    base = cv2.rotate(
        cv2.cvtColor(cv2.convertScaleAbs(img, alpha=1.6), cv2.COLOR_GRAY2BGR),
        cv2.ROTATE_180)
    ov = new_overlay(img.shape)

    mask = find_glare(img)
    cnts, _ = cv2.findContours(cv2.rotate(mask, cv2.ROTATE_180),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, ORANGE, 2)

    dets = detect_markers(img, mask, expect=len(markers))
    proj = project(markers, K, dist, rvec, tvec, z=MARKER_Z_MM)

    residuals = []
    used = set()
    for m, pp in enumerate(proj):
        d = np.linalg.norm(dets - pp, axis=1)
        for j in np.argsort(d):
            if j not in used:
                used.add(j)
                break
        r = float(np.linalg.norm(dets[j] - pp))
        residuals.append((m, markers[m], r))
        det, pp = F(dets[j]), F(pp)
        cv2.circle(ov, tuple(np.round(det).astype(int)), 12, GREEN, 2)
        cv2.drawMarker(ov, tuple(np.round(pp).astype(int)), MAGENTA,
                       cv2.MARKER_CROSS, 18, 2)
        cv2.line(ov, tuple(np.round(det).astype(int)),
                 tuple(np.round(pp).astype(int)), RED, 1)
        label(ov, f"M{m} ({markers[m][0]:.0f},{markers[m][1]:.0f}) "
              f"r={r:.1f}px", (int(det[0]) + 16, int(det[1]) - 10), WHITE)
    return base, ov, residuals


def annotate_ambient(path, K, dist, rvec, tvec, markers, motors=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"cannot read {path}")
    F = flipper(img.shape)
    base = cv2.rotate(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.ROTATE_180)
    ov = new_overlay(img.shape)
    shape = img.shape

    holes = [(i * PITCH_MM, j * PITCH_MM)
             for i in range(N_COLS) for j in range(N_ROWS)]
    for p in project(holes, K, dist, rvec, tvec):
        if in_frame(p, shape):
            cv2.circle(ov, tuple(np.round(F(p)).astype(int)), 2, CYAN, -1)

    def polyline(pts_xy, color, thick=2, dashed=False):
        px = np.array([F(p) for p in project(pts_xy, K, dist, rvec, tvec)])
        for a, b in zip(px[:-1], px[1:]):
            if not (in_frame(a, shape) or in_frame(b, shape)):
                continue
            if dashed:
                n = max(2, int(np.linalg.norm(b - a) / 14))
                for t0 in range(0, n, 2):
                    p0 = a + (b - a) * t0 / n
                    p1 = a + (b - a) * min(t0 + 1, n) / n
                    cv2.line(ov, tuple(np.round(p0).astype(int)),
                             tuple(np.round(p1).astype(int)), color, thick)
            else:
                cv2.line(ov, tuple(np.round(a).astype(int)),
                         tuple(np.round(b).astype(int)), color, thick)

    def rect(x0, y0, x1, y1, color, dashed=False, samples=40):
        xs = np.linspace(x0, x1, samples)
        ys = np.linspace(y0, y1, samples)
        for seg in ([(x, y0) for x in xs], [(x, y1) for x in xs],
                    [(x0, y) for y in ys], [(x1, y) for y in ys]):
            polyline(seg, color, 2, dashed)

    rect(0, 0, GRID_X_MM, GRID_Y_MM, GREEN)
    rect(RAIL_MIN_X, RAIL_MIN_Y, RAIL_MAX_X, RAIL_MAX_Y, ORANGE, dashed=True)
    polyline([(CENTERLINE_X, y) for y in np.linspace(0, GRID_Y_MM, 40)],
             MAGENTA)

    for m, p in enumerate(project(markers, K, dist, rvec, tvec,
                                  z=MARKER_Z_MM)):
        p = F(p)
        cv2.drawMarker(ov, tuple(np.round(p).astype(int)), RED,
                       cv2.MARKER_TILTED_CROSS, 16, 2)
        label(ov, f"M{m}", (int(p[0]) + 10, int(p[1]) + 5), RED)

    # Motor anchors, if they have been measured. They live on the spool-top
    # plane, not the field plane, so they are projected at that height — the
    # drawn ring is the measured top-face circle, and it should sit on the
    # actual spool in the image. If it floats off, either the anchor or the
    # spool height is wrong.
    if motors:
        z = float(motors["spool_top_height_mm"])
        dias = motors.get("top_face_diameter_mm", {})
        for m, xy in sorted(motors["anchors_mm"].items()):
            r = float(dias.get(m, 76.0)) / 2.0
            ring = [(xy[0] + r * np.cos(t), xy[1] + r * np.sin(t))
                    for t in np.linspace(0, 2 * np.pi, 49)]
            rp = np.array([F(p) for p in project(ring, K, dist, rvec, tvec,
                                                 z=z)])
            if not any(in_frame(p, shape) for p in rp):
                continue
            cv2.polylines(ov, [np.round(rp).astype(np.int32)], True, YELLOW, 2)
            c = F(project([xy], K, dist, rvec, tvec, z=z)[0])
            cv2.drawMarker(ov, tuple(np.round(c).astype(int)), YELLOW,
                           cv2.MARKER_CROSS, 14, 2)
            label(ov, f"motor {m} ({xy[0]:.0f},{xy[1]:.0f})",
                  (int(c[0]) + 12, int(c[1]) - 12), YELLOW)

    o, ox, oy = [F(p) for p in
                 project([(0, 0), (150, 0), (0, 150)], K, dist, rvec, tvec)]
    cv2.arrowedLine(ov, tuple(np.round(o).astype(int)),
                    tuple(np.round(ox).astype(int)), WHITE, 2, tipLength=0.2)
    cv2.arrowedLine(ov, tuple(np.round(o).astype(int)),
                    tuple(np.round(oy).astype(int)), WHITE, 2, tipLength=0.2)
    label(ov, "origin (0,0)", (int(o[0]) - 40, int(o[1]) + 30), WHITE)
    return base, ov


def flatten(base, ov):
    a = ov[:, :, 3:4].astype(np.float32) / 255.0
    return (base.astype(np.float32) * (1 - a)
            + ov[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def b64jpg(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        sys.exit("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def b64png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        sys.exit("png encode failed")
    return base64.b64encode(buf.tobytes()).decode()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>calibration // verify</title>
<style>
:root {{
  --ground: #0e1113; --panel: #151a1d; --line: #232c30;
  --ink: #e8edef; --dim: #8fa0a7; --faint: #5a6a71;
  --grid: #5ad2ff; --det: #50dc50; --proj: #e650e6;
  --glare: #ffa03c; --bad: #ff5a5a; --motor: #f5eb50;
  --mono: ui-monospace, 'JetBrains Mono', 'Cascadia Code', Menlo,
          Consolas, monospace;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--ground); color: var(--ink);
       font: 14px/1.5 system-ui, sans-serif; }}

header {{ display: flex; flex-wrap: wrap; align-items: baseline;
          gap: 10px 28px; padding: 18px 22px 14px;
          border-bottom: 1px solid var(--line); }}
header h1 {{ font: 600 15px var(--mono); letter-spacing: .16em;
             text-transform: uppercase; margin: 0; }}
header h1 span {{ color: var(--faint); }}
.readout {{ font: 13px var(--mono); color: var(--dim);
            font-variant-numeric: tabular-nums; }}
.readout b {{ color: var(--ink); font-weight: 500; }}
.readout .u {{ color: var(--faint); }}

.chips {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 22px; }}
.chip {{ font: 12px var(--mono); padding: 3px 10px; border-radius: 3px;
         border: 1px solid var(--line); color: var(--dim);
         font-variant-numeric: tabular-nums; }}
.chip b {{ font-weight: 500; }}
.chip i {{ font-style: normal; opacity: .55; font-size: 11px; }}
.chip.ok b {{ color: var(--det); }}
.chip.warn b {{ color: var(--glare); }}
.chip.bad b {{ color: var(--bad); }}

section {{ margin: 6px 0 26px; width: min(74%, 1150px); }}
@media (max-width: 1000px) {{ section {{ width: 100%; }} }}
.bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
        padding: 10px 22px; }}
.bar h2 {{ font: 600 12px var(--mono); letter-spacing: .14em;
           text-transform: uppercase; color: var(--dim); margin: 0;
           flex: 1 1 auto; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px;
           font: 12px var(--mono); color: var(--faint); }}
.legend i {{ font-style: normal; }}
.legend i::before {{ content: '\\25A0\\00A0'; }}
.l-det::before {{ color: var(--det); }}
.l-proj::before {{ color: var(--proj); }}
.l-glare::before {{ color: var(--glare); }}
.l-grid::before {{ color: var(--grid); }}
.l-bad::before {{ color: var(--bad); }}
.l-motor::before {{ color: var(--motor); }}

.ctl {{ display: flex; gap: 6px; }}
.ctl button, .ctl label {{ font: 12px var(--mono); color: var(--dim);
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 3px; padding: 4px 12px; cursor: pointer;
    user-select: none; }}
.ctl button:hover, .ctl label:hover {{ color: var(--ink);
    border-color: var(--faint); }}
.ctl :focus-visible {{ outline: 2px solid var(--grid);
    outline-offset: 1px; }}
.ctl input {{ position: absolute; opacity: 0; pointer-events: none; }}
.ctl label.on {{ color: var(--grid); border-color: var(--grid); }}

.frame {{ position: relative; margin: 0 22px; }}
.frame canvas {{ display: block; width: 100%; aspect-ratio: 4 / 3;
                 height: auto; max-height: 80vh; background: #000;
                 cursor: crosshair; border: 1px solid var(--line); }}
.frame canvas.panning {{ cursor: grabbing; }}
.frame::before, .frame::after,
.frame .t1::before, .frame .t1::after {{ content: ''; position: absolute;
    width: 14px; height: 14px; border: 0 solid var(--faint);
    pointer-events: none; z-index: 2; }}
.frame::before {{ top: -1px; left: -1px;
    border-top-width: 2px; border-left-width: 2px; }}
.frame::after {{ top: -1px; right: -1px;
    border-top-width: 2px; border-right-width: 2px; }}
.frame .t1::before {{ bottom: -1px; left: -1px;
    border-bottom-width: 2px; border-left-width: 2px; }}
.frame .t1::after {{ bottom: -1px; right: -1px;
    border-bottom-width: 2px; border-right-width: 2px; }}
.hud {{ position: absolute; top: 10px; right: 12px; z-index: 2;
        font: 12px var(--mono); color: var(--ink);
        background: rgba(14,17,19,.82); border: 1px solid var(--line);
        border-radius: 3px; padding: 4px 10px; pointer-events: none;
        font-variant-numeric: tabular-nums; white-space: pre; }}
.hud .u {{ color: var(--faint); }}
footer {{ padding: 8px 22px 26px; font: 12px var(--mono);
          color: var(--faint); }}
</style></head><body>

<header>
  <h1>calibration <span>//</span> verify</h1>
  <span class="readout">cam <b>{cx:.0f}</b><span class="u">,</span>
    <b>{cy:.0f}</b><span class="u">,</span> <b>{cz:.0f}</b>
    <span class="u">mm</span></span>
  <span class="readout">reprojection <b>{rms:.2f}</b>
    <span class="u">px rms</span></span>
  <span class="readout"><span class="u">{date}</span></span>
</header>

<div class="chips">{chips}</div>

<section>
  <div class="bar">
    <h2>calibration frame</h2>
    <div class="legend"><i class="l-det">detected</i>
      <i class="l-proj">reprojected</i><i class="l-glare">glare mask</i></div>
    <div class="ctl">
      <label id="v0-ov" class="on" tabindex="0">overlay</label>
      <button id="v0-fit">fit</button>
      <button id="v0-100">1:1</button>
    </div>
  </div>
  <div class="frame" id="v0-frame"><span class="t1"></span>
    <div class="hud" id="v0-hud"></div>
    <canvas id="v0-cv"></canvas>
  </div>
</section>

{ambient_section}

<footer>hover reads grid-frame mm through the camera model — holes should
read multiples of 25.4 &middot; drag pans &middot; scroll zooms &middot;
double-click fits</footer>

<script>
const MAPS = {maps_json};
const VIEWS = {views_json};

function viewer(id, baseSrc, ovSrc, map) {{
  const cv = document.getElementById(id + '-cv');
  const ctx = cv.getContext('2d');
  const hud = document.getElementById(id + '-hud');
  const ovBtn = document.getElementById(id + '-ov');
  const frame = document.getElementById(id + '-frame');
  const base = new Image(), ov = new Image();
  let showOv = true, s = 1, x = 0, y = 0, drag = null, raf = 0;
  const dpr = window.devicePixelRatio || 1;

  function draw() {{
    raf = 0;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cv.width / dpr, cv.height / dpr);
    ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * x, dpr * y);
    ctx.imageSmoothingEnabled = s < 2;
    ctx.drawImage(base, 0, 0);
    if (showOv && ov.complete) ctx.drawImage(ov, 0, 0);
  }}
  const req = () => {{ if (!raf) raf = requestAnimationFrame(draw); }};

  function resize() {{
    const r = cv.getBoundingClientRect();
    cv.width = Math.round(r.width * dpr);
    cv.height = Math.round(r.height * dpr);
    req();
  }}
  function fit() {{
    s = Math.min(cv.width / dpr / base.naturalWidth,
                 cv.height / dpr / base.naturalHeight);
    x = (cv.width / dpr - base.naturalWidth * s) / 2;
    y = (cv.height / dpr - base.naturalHeight * s) / 2;
    hud.innerHTML = `<span class="u">zoom</span> ${{Math.round(s * 100)}}%`;
    req();
  }}
  base.onload = () => {{ resize(); fit(); }};
  base.src = baseSrc; ov.src = ovSrc;
  ov.onload = req;
  new ResizeObserver(() => {{ resize(); }}).observe(frame);

  cv.addEventListener('wheel', e => {{
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.18 : 1 / 1.18;
    x = mx - (mx - x) * f; y = my - (my - y) * f; s *= f; req();
  }}, {{ passive: false }});
  cv.addEventListener('pointerdown', e => {{
    drag = [e.clientX - x, e.clientY - y];
    cv.setPointerCapture(e.pointerId); cv.classList.add('panning');
  }});
  cv.addEventListener('pointermove', e => {{
    if (drag) {{ x = e.clientX - drag[0]; y = e.clientY - drag[1]; req(); }}
    const r = cv.getBoundingClientRect();
    const ix = (e.clientX - r.left - x) / s, iy = (e.clientY - r.top - y) / s;
    hud.innerHTML = mmText(map, ix, iy) +
      `  <span class="u">zoom</span> ${{Math.round(s * 100)}}%`;
  }});
  cv.addEventListener('pointerup', e => {{
    drag = null; cv.classList.remove('panning');
  }});
  cv.addEventListener('pointerleave', () => {{
    hud.innerHTML = `<span class="u">zoom</span> ${{Math.round(s * 100)}}%`;
  }});
  cv.addEventListener('dblclick', fit);
  document.getElementById(id + '-fit').onclick = fit;
  document.getElementById(id + '-100').onclick = () => {{
    s = 1;
    x = (cv.width / dpr - base.naturalWidth) / 2;
    y = (cv.height / dpr - base.naturalHeight) / 2; req();
  }};
  const toggle = () => {{
    showOv = !showOv; ovBtn.classList.toggle('on', showOv); req();
  }};
  ovBtn.onclick = toggle;
  ovBtn.onkeydown = e => {{
    if (e.key === ' ' || e.key === 'Enter') {{ e.preventDefault(); toggle(); }}
  }};
  hud.innerHTML = '';
}}

function mmText(map, ix, iy) {{
  const g = (arr, cx, cy) => arr[cy * map.cols + cx];
  let fx = ix / map.step, fy = iy / map.step;
  fx = Math.min(Math.max(fx, 0), map.cols - 1.001);
  fy = Math.min(Math.max(fy, 0), map.rows - 1.001);
  const cx = Math.floor(fx), cy = Math.floor(fy);
  const tx = fx - cx, ty = fy - cy;
  const bil = arr =>
    g(arr, cx, cy) * (1 - tx) * (1 - ty) + g(arr, cx + 1, cy) * tx * (1 - ty)
    + g(arr, cx, cy + 1) * (1 - tx) * ty + g(arr, cx + 1, cy + 1) * tx * ty;
  const mx = bil(map.x), my = bil(map.y);
  if (mx < -400 || mx > 2400 || my < -400 || my > 1400)
    return '<span class="u">off table</span>';
  return `x <b>${{mx.toFixed(0)}}</b> <span class="u">mm</span>  ` +
         `y <b>${{my.toFixed(0)}}</b> <span class="u">mm</span>`;
}}

VIEWS.forEach(v => viewer(v.id, v.base, v.ov, MAPS[v.id]));
</script></body></html>
"""

AMBIENT_BAR = """
<section>
  <div class="bar">
    <h2>ambient frame &mdash; projected geometry</h2>
    <div class="legend"><i class="l-grid">holes</i>
      <i class="l-det">grid border</i><i class="l-proj">centerline</i>
      <i class="l-glare">rails (approx)</i><i class="l-bad">markers</i><i class="l-motor">motor anchors</i></div>
    <div class="ctl">
      <label id="v1-ov" class="on" tabindex="0">overlay</label>
      <button id="v1-fit">fit</button>
      <button id="v1-100">1:1</button>
    </div>
  </div>
  <div class="frame" id="v1-frame"><span class="t1"></span>
    <div class="hud" id="v1-hud"></div>
    <canvas id="v1-cv"></canvas>
  </div>
</section>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lowexp", help="low-exposure calibration frame")
    ap.add_argument("--ambient", help="ambient frame for the geometry overlay")
    ap.add_argument("-o", "--out", default=str(CALIB_DIR / "report.html"))
    args = ap.parse_args()

    K, dist, rvec, tvec, cam_pos, markers, n_fitted = load_all()
    rms = float(np.load(CALIB_DIR / "extrinsics.npz")["rms"])

    base1, ov1, residuals = annotate_lowexp(args.lowexp, K, dist, rvec,
                                            tvec, markers)
    cv2.imwrite(str(CALIB_DIR / "report_lowexp.png"), flatten(base1, ov1))

    # Markers past n_fitted did not constrain the pose — their residual is a
    # genuine held-out error, so it is the one worth reading.
    chips = ""
    for m, pos, r in residuals:
        cls = "ok" if r < 2.0 else ("warn" if r < 4.0 else "bad")
        kind = "fit" if m < n_fitted else "held-out"
        chips += (f'<span class="chip {cls}">M{m} '
                  f'({pos[0]:.0f}, {pos[1]:.0f}) <b>{r:.2f} px</b> '
                  f'<i>{kind}</i></span>')

    maps = {"v0": mm_map(base1.shape, K, dist, rvec, tvec)}
    views = [{"id": "v0", "base": "data:image/jpeg;base64," + b64jpg(base1),
              "ov": "data:image/png;base64," + b64png(ov1)}]

    ambient_html = ""
    if args.ambient:
        base2, ov2 = annotate_ambient(args.ambient, K, dist, rvec, tvec,
                                      markers, load_motors())
        cv2.imwrite(str(CALIB_DIR / "report_ambient.png"),
                    flatten(base2, ov2))
        maps["v1"] = mm_map(base2.shape, K, dist, rvec, tvec)
        views.append({"id": "v1",
                      "base": "data:image/jpeg;base64," + b64jpg(base2),
                      "ov": "data:image/png;base64," + b64png(ov2)})
        ambient_html = AMBIENT_BAR

    html = PAGE.format(date=time.strftime("%Y-%m-%d %H:%M"),
                       cx=cam_pos[0], cy=cam_pos[1], cz=cam_pos[2],
                       rms=rms, chips=chips,
                       ambient_section=ambient_html,
                       maps_json=json.dumps(maps, separators=(",", ":")),
                       views_json=json.dumps(views, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  (flattened PNGs also in {CALIB_DIR})")
    print(f"open with: xdg-open {out}")


if __name__ == "__main__":
    main()
