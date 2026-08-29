#!/usr/bin/env python3
"""Plot how well the puck fits actually fit.

The numbers fit_puck.py prints are summary statistics, and summary statistics
hide the two things worth knowing: whether a model is the RIGHT SHAPE, and
whether the spread is structure or noise. A deceleration of 283 +/- a lot
looks like a bad measurement; the same data plotted against speed either shows
a clean curve, in which case it is drag and the model is wrong, or a cloud, in
which case it is noise and the model is fine.

    python vision/bin/plot_puck_fit.py logs/puck_<ts>.jsonl [-o out.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402
from fit_puck import (  # noqa: E402
    contact_events,
    G_MM_S2, MIN_GLIDE, MIN_SPEED, SKIP, fit_bounces, fit_friction, load,
    segment,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("-o", "--out", default="logs/puck_fit.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load(args.recording)
    bounds, cuts, _gaps = segment(d)
    fr = fit_friction(d, bounds)
    walls, _others = fit_bounces(d, contact_events(cuts))

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"puck fits — {Path(args.recording).name}", fontsize=13)

    # ── 1. deceleration vs speed, both models ──
    a = ax[0, 0]
    # Straight from the fitter: same acceptance, same points. Recomputing
    # them here is how the plot ended up showing segments the fit had thrown
    # away, which read as "the fit is being dragged by outliers" when the fit
    # had already excluded them.
    per = np.column_stack([fr["speeds"], fr["decels"], fr["weights"]])
    a.scatter(per[:, 0] / 1000, per[:, 1], s=np.clip(per[:, 2], 8, 90),
              alpha=0.55, edgecolor="k", linewidth=0.3, label="glides")
    vv = np.linspace(0, per[:, 0].max() * 1.05, 200)
    a.axhline(fr["decel_mm_s2"], color="crimson", ls="--", lw=1.6,
              label=f"constant {fr['decel_mm_s2']:.0f}")
    if "drag" in fr:
        g = fr["drag"]
        a.plot(vv / 1000, g["a"] + g["b"] * vv ** 2, color="royalblue", lw=2,
               label=f"a+b·v²  ({g['a']:.0f} + {g['b']:.2e}·v²)")
    a.set_xlabel("glide mean speed (m/s)")
    a.set_ylabel("deceleration (mm/s²)")
    a.set_title("friction: is the spread drag, or noise?")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)

    # ── 2. residuals of each model ──
    a = ax[0, 1]
    if "drag" in fr:
        g = fr["drag"]
        rc = per[:, 1] - fr["decel_mm_s2"]
        rq = per[:, 1] - (g["a"] + g["b"] * per[:, 0] ** 2)
        a.axhline(0, color="k", lw=0.8)
        a.scatter(per[:, 0] / 1000, rc, s=22, alpha=0.6, color="crimson",
                  label=f"constant, rms {g['rms_const']:.0f}")
        a.scatter(per[:, 0] / 1000, rq, s=22, alpha=0.6, color="royalblue",
                  label=f"a+b·v², rms {g['rms']:.0f}")
        a.legend(fontsize=8)
    a.set_xlabel("glide mean speed (m/s)")
    a.set_ylabel("residual (mm/s²)")
    a.set_title("a good model leaves no trend behind")
    a.grid(alpha=0.25)

    # ── 3. sample glides with the fitted decay ──
    a = ax[0, 2]
    order = sorted(bounds, key=lambda b: b[1] - b[0], reverse=True)[:6]
    for lo, hi in order:
        t = d["t"][lo + SKIP:hi - SKIP]
        v = np.hypot(d["vx"][lo + SKIP:hi - SKIP], d["vy"][lo + SKIP:hi - SKIP])
        if len(t) < MIN_GLIDE:
            continue
        a.plot(t - t[0], v / 1000, lw=1.2, alpha=0.85)
    a.set_xlabel("time within glide (s)")
    a.set_ylabel("speed (m/s)")
    a.set_title("the six longest glides")
    a.grid(alpha=0.25)

    # ── 4. restitution vs impact speed ──
    a = ax[1, 0]
    cols = {"near(-y)": "tab:blue", "far(+y)": "tab:orange",
            "human(-x)": "tab:green", "robot(+x)": "tab:red"}
    for name, c in cols.items():
        g = [w for w in walls if w["wall"] == name]
        if not g:
            continue
        a.scatter([w["speed_in"] / 1000 for w in g], [w["e_normal"] for w in g],
                  s=34, alpha=0.75, color=c, edgecolor="k", linewidth=0.3,
                  label=f"{name} (n={len(g)})")
    if len(walls) >= 8:
        sp = np.array([w["speed_in"] for w in walls])
        ee = np.array([w["e_normal"] for w in walls])
        k = np.polyfit(sp, ee, 1)
        xs = np.linspace(sp.min(), sp.max(), 50)
        a.plot(xs / 1000, np.polyval(k, xs), "k--", lw=1.6,
               label=f"trend {k[0]*1000:+.3f} /(m/s)")
    a.axhline(0.85, color="grey", ls=":", label="sim's old 0.85")
    a.set_xlabel("impact speed (m/s)")
    a.set_ylabel("e (normal)")
    a.set_title("restitution falls with impact speed")
    a.legend(fontsize=7)
    a.grid(alpha=0.25)

    # ── 5. per-wall distribution ──
    a = ax[1, 1]
    names = [n for n in cols if any(w["wall"] == n for w in walls)]
    data = [[w["e_normal"] for w in walls if w["wall"] == n] for n in names]
    a.boxplot(data, labels=[n.replace("(", "\n(") for n in names])
    for i, dd in enumerate(data):
        a.scatter(np.full(len(dd), i + 1) + np.random.uniform(-.06, .06, len(dd)),
                  dd, s=18, alpha=0.6, color=cols[names[i]], zorder=3)
    a.set_ylabel("e (normal)")
    a.set_title("all four rails should agree — and now do")
    a.grid(alpha=0.25, axis="y")

    # ── 6. tangential: the spin evidence ──
    a = ax[1, 2]
    et = np.array([w["e_tangential"] for w in walls])
    et = et[~np.isnan(et)]
    a.hist(et, bins=18, color="mediumpurple", edgecolor="k", linewidth=0.4)
    a.axvline(1.0, color="k", ls="--", lw=1.6, label="1.0 = frictionless")
    a.axvline(et.mean(), color="crimson", lw=2, label=f"mean {et.mean():.3f}")
    a.set_xlabel("tangential velocity ratio")
    a.set_ylabel("contacts")
    a.set_title("cushion takes tangential momentum:\n"
                "bounce ANGLES are not specular")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    print(f"  {len(per)} glides, {len(walls)} wall contacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
