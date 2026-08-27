#!/usr/bin/env python3
"""Do the C++ and Python cable models agree?

shared/cdpr_geometry.h is the canonical forward model used by the firmware.
shared/cable_model.py reimplements the same model in NumPy, because Python
cannot include a C header. Two implementations of one model is exactly the
kind of duplication that drifts silently, so this compares them numerically.

Both return length only up to a constant per-motor offset (encoder zero and
the choice of wrap reference), so the comparison is on pose-to-pose DELTAS,
which is what any homed controller actually acts on.

Usage:  python shared/check_geometry.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared"))
import cable_model as cm  # noqa: E402
import cdpr_geometry as pg  # noqa: E402

# The as-built machine, from the mirror this file is checking.
ANCHORS = np.array(list(zip(pg.MOTOR_X, pg.MOTOR_Y)))
REF = cm.wrap_reference(ANCHORS, np.array([pg.HOME_X, pg.HOME_Y]))

HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include "cdpr_geometry.h"
int main(int argc, char **argv) {
  for (int i = 1; i + 2 < argc; i += 3) {
    float x = atof(argv[i]), y = atof(argv[i + 1]), th = atof(argv[i + 2]);
    for (int m = 0; m < NUM_MOTORS; m++)
      printf("%.6f%s", cableLength(m, x, y, th), m == 3 ? "\n" : " ");
  }
  return 0;
}
"""


CONST_HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include "cdpr_geometry.h"
int main() {
  for (int m = 0; m < NUM_MOTORS; m++)
    printf("MOTOR_X%d %.6f\nMOTOR_Y%d %.6f\nWINDING_SIDE%d %.6f\n"
           "WRAP_REF_ANGLE%d %.6f\n", m, MOTOR_X[m], m, MOTOR_Y[m],
           m, WINDING_SIDE[m], m, WRAP_REF_ANGLE[m]);
  printf("PUCK_RADIUS_MM %.6f\nPUCK_MARKER_R_MM %.6f\nGOAL_WIDTH_MM %.6f\n",
         PUCK_RADIUS_MM, PUCK_MARKER_R_MM, GOAL_WIDTH_MM);
  printf("SPOOL_RADIUS_MM %.6f\nATTACH_R_MM %.6f\nATTACH_CHIRALITY %.6f\n"
         "WS_MIN_X %.6f\nWS_MAX_X %.6f\nWS_MIN_Y %.6f\nWS_MAX_Y %.6f\n"
         "HOME_X %.6f\nHOME_Y %.6f\n"
         "GRID_X_MM %.6f\nGRID_Y_MM %.6f\nCENTERLINE_X %.6f\n",
         SPOOL_RADIUS_MM, ATTACH_R_MM, ATTACH_CHIRALITY, WS_MIN_X, WS_MAX_X,
         WS_MIN_Y, WS_MAX_Y, HOME_X, HOME_Y, GRID_X_MM,
         GRID_Y_MM, CENTERLINE_X);
  return 0;
}
"""


def check_constants():
    """The Python mirror must match the header value for value."""
    with tempfile.TemporaryDirectory() as td:
        src, exe = Path(td) / "c.cpp", Path(td) / "c"
        src.write_text(CONST_HARNESS)
        r = subprocess.run(["g++", "-std=c++17", f"-I{ROOT / 'shared'}",
                            "-o", str(exe), str(src)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit("constant harness failed to compile:\n" + r.stderr)
        out = subprocess.run([str(exe)], capture_output=True, text=True,
                             check=True).stdout
    bad = []
    for line in out.strip().splitlines():
        name, val = line.split()
        val = float(val)
        if name[-1].isdigit() and not name.startswith("GRID"):
            base, idx = name[:-1], int(name[-1])
            got = getattr(pg, base)[idx]
        else:
            got = getattr(pg, name)
        if abs(float(got) - val) > 1e-4:
            bad.append(f"  {name}: header {val} vs python {got}")
    print(f"constants checked: {len(out.strip().splitlines())}")
    if bad:
        sys.exit("FAIL - shared/cdpr_geometry.py has drifted from the "
                 "header:\n" + "\n".join(bad))
    print("OK - Python mirror matches the header\n")


def main():
    check_constants()
    rng = np.random.default_rng(0)
    n = 24
    xs = rng.uniform(ANCHORS[:, 0].min() + 60, 1900, n)
    ys = rng.uniform(60, 880, n)
    ths = rng.uniform(-np.pi, np.pi, n)

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "h.cpp"
        exe = Path(td) / "h"
        src.write_text(HARNESS)
        r = subprocess.run(["g++", "-std=c++17", f"-I{ROOT / 'shared'}",
                            "-o", str(exe), str(src)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit("harness failed to compile:\n" + r.stderr)
        args = []
        for x, y, t in zip(xs, ys, ths):
            args += [f"{x:.6f}", f"{y:.6f}", f"{t:.6f}"]
        out = subprocess.run([str(exe)] + args, capture_output=True,
                             text=True, check=True).stdout
    cpp = np.array([[float(v) for v in ln.split()]
                    for ln in out.strip().splitlines()])

    py = cm.cable_lengths(ANCHORS, np.stack([xs, ys], axis=1), ths,
                          chirality=cm.CHIRALITY, ref=REF)

    # Both are defined up to a per-motor constant: compare deltas vs pose 0.
    dc = cpp - cpp[0]
    dp = py - py[0]
    err = np.abs(dc - dp)

    print(f"{n} random poses across the workspace, per-motor max |delta| "
          "disagreement (mm):")
    for m in range(4):
        print(f"  motor {m}: {err[:, m].max():.6f}")
    worst = err.max()
    print(f"\nworst {worst:.6f} mm")

    # Also report how much the wrap term actually contributes, so a reader
    # can see it is not a rounding-level effect.
    span = (dc.max(axis=0) - dc.min(axis=0))
    print("cable-length span across these poses (mm): "
          + ", ".join(f"{v:.1f}" for v in span))

    if worst > 0.01:
        sys.exit(f"\nFAIL — the two implementations disagree by {worst:.4f} mm. "
                 "One of shared/cdpr_geometry.h or shared/cable_model.py has "
                 "drifted;\nthey must encode the same forward model.")
    print("\nOK — C++ and Python cable models agree")


if __name__ == "__main__":
    main()
