#!/usr/bin/env python3
"""Bundle a recording and its analysis into a standalone HTML viewer.

WHY THIS EXISTS
    plot_puck_fit.py answers "is the fit any good". It cannot answer "why was
    THAT bounce thrown away", which is the question that actually comes up --
    the fitter measures 11 of 59 wall contacts and the interesting part is the
    48 it could not bracket. A scatter plot has no way to show you a specific
    event in its context.

    So this writes a page where the trajectory is scrubbable, every contact is
    marked with what the fitter decided about it and why, and the summary
    plots are linked to the same frames. Clicking a point in a plot moves the
    playhead to the event it came from.

WHAT IT EMITS
    One self-contained .html: no server, no network, no dependencies. The
    recording is inlined as columnar arrays (~12k frames costs about 500 kB
    once rounded, which is smaller than a single PNG of the same data).

    python vision/bin/export_puck_view.py logs/puck_<ts>.jsonl -o out.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402
from fit_puck import (  # noqa: E402
    MIN_GLIDE, SKIP, SLOPE_FRAMES, WALL_BAND_MM, contact_events, fit_bounces,
    fit_friction, fit_paddle, fit_spin, load, noise_floor, segment, wall_of,
)


def classify_events(d, events, walls, hits):
    """One row per contact, with what the fitter concluded and why.

    The verdict matters more than the coordinates. A bounce the fitter skipped
    is not noise to be hidden -- it is 4 out of every 5 wall contacts in a
    real session, and the reason is always the same shape: another contact
    landed within a few frames, so no uncontaminated glide existed to bracket.
    """
    n = len(d["t"])
    by_lo_wall = {w["lo"]: w for w in walls}
    by_lo_pad = {h["lo"]: h for h in (hits or [])}
    out = []
    for lo, hi in events:
        sp = np.arange(max(0, lo - SLOPE_FRAMES - 4), min(hi + 3, n))
        near = np.minimum(
            np.minimum(d["x"][sp] - geom.RAIL_MIN_X,
                       geom.RAIL_MAX_X - d["x"][sp]),
            np.minimum(d["y"][sp] - geom.RAIL_MIN_Y,
                       geom.RAIL_MAX_Y - d["y"][sp]))
        c = int(sp[int(np.argmin(near))])
        at_rail = float(near.min()) < WALL_BAND_MM
        name, _nrm = wall_of(d["x"][c], d["y"][c])

        w, h = by_lo_wall.get(lo), by_lo_pad.get(lo)
        if w is not None:
            kind, note = "wall", f"measured, e = {w['e_normal']:.3f}"
        elif h is not None:
            kind, note = "mallet", f"measured, e = {h['e']:.3f}"
        elif at_rail and name is None:
            kind, note = "goal", "in the goal mouth — not a cushion"
        elif at_rail:
            kind, note = "wall-skipped", "no clean glide either side to bracket"
        else:
            kind, note = "open", "away from any rail — mallet or hand"
        out.append({"lo": int(lo), "hi": int(hi), "c": c, "kind": kind,
                    "wall": name, "note": note,
                    "t": round(float(d["t"][c]), 3),
                    "x": round(float(d["x"][c]), 1),
                    "y": round(float(d["y"][c]), 1)})
    return out


def build(path):
    d = load(path)
    bounds, cuts, gaps = segment(d)
    events = contact_events(cuts)
    gate = 6.0 * noise_floor(d, bounds)
    fr = fit_friction(d, bounds)
    walls, others = fit_bounces(d, events, gate)
    hits = fit_paddle(d, events, gate) or []
    spin = fit_spin(d, events, gate)

    def col(key, nd):
        v = d.get(key)
        if v is None:
            return None
        v = np.where(np.isfinite(v), v, np.nan)
        return [None if not np.isfinite(a) else round(float(a), nd) for a in v]

    ev = classify_events(d, events, walls, hits)
    fitted = {(a, b) for a, b in fr.get("bounds", [])} if fr else set()

    return {
        "file": Path(path).name,
        "geom": {
            "rail": [geom.RAIL_MIN_X, geom.RAIL_MAX_X,
                     geom.RAIL_MIN_Y, geom.RAIL_MAX_Y],
            "centerline": geom.CENTERLINE_X,
            "goal": geom.GOAL_WIDTH_MM,
            "puck_r": geom.PUCK_RADIUS_MM,
            "mallet_r": geom.MALLET_RADIUS_MM,
            "marker_r": geom.PUCK_MARKER_R_MM,
            "band": WALL_BAND_MM,
        },
        "t": col("t", 4), "x": col("x", 1), "y": col("y", 1),
        "mx": col("mx", 1), "my": col("my", 1),
        "n": [int(a) for a in d["n"]] if "n" in d else None,
        "th": col("th", 4), "w": col("w", 2),
        "vx": col("vx", 1), "vy": col("vy", 1),
        "segments": [{"a": int(a), "b": int(b), "fitted": (a, b) in fitted}
                     for a, b in bounds],
        "events": ev,
        "gaps": [int(g) for g in gaps],
        # Each fitted glide keeps the frame it came from, so a point in the
        # friction scatter can seek the playhead to the glide it measures.
        "friction": dict(
            {k: (None if v is None else
                 (v.tolist() if isinstance(v, np.ndarray) else v))
             for k, v in (fr or {}).items() if k != "bounds"},
            bounds_a=[int((a + b) // 2) for a, b in (fr or {}).get("bounds", [])]),
        "walls": walls,
        "paddle": hits,
        "spin": spin if spin is None else
                {k: (v if k != "rows" else v) for k, v in spin.items()},
        "counts": {"samples": len(d["t"]), "others": others,
                   "span": round(float(d["t"][-1] - d["t"][0]), 1)},
        "consts": {"MIN_GLIDE": MIN_GLIDE, "SKIP": SKIP, "gate": round(gate, 3)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("-o", "--out", default="logs/puck_view.html")
    ap.add_argument("--raw", action="store_true",
                    help="emit body-only HTML (for publishing as an artifact)")
    ap.add_argument("--template", default=str(
        Path(__file__).resolve().parent / "puck_view.html"))
    args = ap.parse_args()

    data = build(args.recording)
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
    html = Path(args.template).read_text()
    if "__PUCK_DATA__" not in html:
        sys.exit(f"{args.template} has no __PUCK_DATA__ placeholder")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = html.replace("__PUCK_DATA__", blob)
    if args.raw:
        out.write_text(body)
    else:
        # Standalone file for opening off disk. The template is body-only so
        # the same source can also be published as an artifact, which supplies
        # its own document shell.
        out.write_text('<!doctype html>\n<html lang="en">\n<head>\n'
                       '<meta charset="utf-8">\n<meta name="viewport" '
                       'content="width=device-width,initial-scale=1">\n'
                       "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} kB, {data['counts']['samples']} frames, "
          f"{len(data['events'])} contacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
