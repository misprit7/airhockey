#!/usr/bin/env python3
"""Summarise a train_selfplay.py log: the shaper's counters over the run.

train_selfplay prints the reward shaper's `stats` dict every 10k steps
(`    shots {...}`, and `    demo  {...}` when demonstrations are on). This
sums them over three windows -- the first, the middle and the last N
prints -- and shows, per 10k steps: possessions held, drive pay collected,
steps past the shot clock, on-target shots and the controlled multiplier
on them, goals and their multiplier, the mean accel fraction, and the
income per reward term (`pay_*`). Reading the trend across the windows is
how each 3.x run was judged at 500k; the per-opponent W/L/D lines are
printed by the trainer itself (`grep "vs " logs/<run>.log`).

    python ai/bin/run_stats.py logs/3.11-shot-ramp-selfplay.log
    python ai/bin/run_stats.py logs/3.11-shot-ramp-selfplay.log --window 10
"""
from __future__ import annotations

import argparse
import ast
import re


def summarise(rows: list[dict], label: str) -> str:
    S = {k: sum(float(r.get(k, 0)) for r in rows) for k in rows[0]}
    n = S["steps"]
    k10 = 1e4 / n
    pays = " ".join(f"{k[4:]} {S[k] * k10:.0f}" for k in S
                    if k.startswith("pay_") and abs(S[k] * k10) >= 5)
    return (f"{label:12s}: held {S.get('held', 0) * k10:5.1f}/10k drive {S.get('drive_sum', 0) * k10:4.0f} "
            f"overstay {S.get('overstay_steps', 0) * k10:4.0f} | shots {S['shots'] * k10:.0f} "
            f"on-target {S['on_target'] * k10:.1f}/10k ctrl x {S['patience_sum'] / max(S['on_target'], 1):.2f} "
            f"goals {S['goals'] * k10:.1f}/10k goal x {S['goal_patience_sum'] / max(S['goals'], 1):.2f} | "
            f"accel {S['accel_frac_sum'] / n:.2f} | {pays}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--window", type=int, default=5, help="prints per window (each print is 10k steps)")
    args = ap.parse_args()
    agent, demo = [], []
    for line in open(args.log):
        m = re.match(r"\s+(shots|demo)\s+(\{.*?\})", line)
        if m:
            (demo if m.group(1) == "demo" else agent).append(ast.literal_eval(m.group(2)))
    if not agent:
        raise SystemExit(f"no shaper stats in {args.log}")
    n, w = len(agent), max(1, min(args.window, len(agent) // 3 or 1))
    print(f"{args.log}: {n} prints of 10k steps, windows of {w}")
    print(summarise(agent[:w], "agent first"))
    print(summarise(agent[n // 2 - w // 2: n // 2 - w // 2 + w], "agent middle"))
    print(summarise(agent[-w:], "agent last"))
    if demo:
        print(summarise(demo[-w:], "demo last"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
