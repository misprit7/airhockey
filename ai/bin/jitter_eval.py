#!/usr/bin/env python3
"""How much the planner's target moves from tick to tick.

Plays a checkpoint (planner, eval mode) against an opponent and reports
the commanded target's change per tick in mm, split by situation: the
puck near a standstill on the robot's side (the case the table showed as
"jittery when nothing is happening"), the puck away on the far half
(idle), and everything else (in play). It is the sim-side number for the
smoothness tax: 0.2 -> 0.5 took the standstill median from 53 to 23 mm
(3.5 vs 3.9, ai/RETRAIN.md). On the table the same quantity is the
`cmd_x/cmd_y` column of `logs/run_policy/<stamp>.ticks.csv`.

    python ai/bin/jitter_eval.py 3.5-shot-clock-turnover-selfplay 3.11-shot-ramp-selfplay

Defaults: 16 envs x 30 s vs the weak goalie, seed 11.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.eval_play import Driver, header, make_env  # noqa: E402
from airhockey.policy_loader import PLAN_ITERATIONS, load_agent  # noqa: E402


def play(agent, opponent: str, n_envs: int, seconds: float, seed: int, planner: bool) -> str:
    e = make_env(opponent, n_envs, seconds)
    o = e.reset(seed=seed)
    drv = Driver(agent, e, planner=planner)
    lo, hi = e._action_low[:2], e._action_high[:2]
    H = e.table_config.height
    prev = None
    still, away, play_ = [], [], []
    for _ in range(int(round(seconds / e.action_dt))):
        a = drv.act(o)
        tgt = lo + (a[:, :2] + 1.0) / 2.0 * (hi - lo)
        o, r, term, trunc, info = e.step(a)
        drv.done(term, trunc)
        if prev is not None:
            d = np.hypot(*(tgt - prev).T) * 1000.0
            sp = np.hypot(info["puck_vx"], info["puck_vy"])
            dist = np.hypot(info["puck_x"] - info["pad_x"], info["puck_y"] - info["pad_y"])
            near_still = (sp < 0.5) & (dist < 0.35) & (info["puck_y"] < H / 2)
            far = info["puck_y"] > H / 2
            still += d[near_still].tolist()
            away += d[far].tolist()
            play_ += d[~near_still & ~far].tolist()
        prev = tgt

    def f(x):
        return (f"median {np.median(x):4.0f} mm p90 {np.percentile(x, 90):4.0f} mm (n={len(x)})"
                if x else "n=0")
    return (f"{'planner' if planner else 'prior'} vs {opponent}: puck near standstill: {f(still)} | "
            f"puck away: {f(away)} | in play: {f(play_)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--opponents", default="weak_goalie")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--iterations", type=int, default=PLAN_ITERATIONS)
    ap.add_argument("--prior", action="store_true", help="also measure the policy prior alone")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    for run in args.runs:
        agent = load_agent(run, iterations=args.iterations)
        print(header(run, agent), flush=True)
        for opp in args.opponents.split(","):
            print("  " + play(agent, opp, args.envs, args.seconds, args.seed, True), flush=True)
            if args.prior:
                print("  " + play(agent, opp, args.envs, args.seconds, args.seed, False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
