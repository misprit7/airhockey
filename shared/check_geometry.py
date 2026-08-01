#!/usr/bin/env python3
"""Do the C++ and Python cable models agree?

shared/cdpr_geometry.h is the canonical forward model used by the firmware.
ai/bin/calibrate_fit.py reimplements the same model in NumPy, because Python
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
sys.path.insert(0, str(ROOT / "ai" / "bin"))
import calibrate_fit as cf  # noqa: E402

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


def main():
    rng = np.random.default_rng(0)
    n = 24
    xs = rng.uniform(cf.ANCHOR_GUESS[:, 0].min() + 60, 1900, n)
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

    py = cf.model_lengths(cf.ANCHOR_GUESS, np.stack([xs, ys], axis=1), ths,
                          cf.CHIRALITY_CONFIRMED)

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
                 "One of shared/cdpr_geometry.h or ai/bin/calibrate_fit.py has "
                 "drifted;\nthey must encode the same forward model.")
    print("\nOK — C++ and Python cable models agree")


if __name__ == "__main__":
    main()
