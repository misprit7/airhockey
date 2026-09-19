#!/usr/bin/env python3
"""What a checkpoint does with a puck it has stopped.

Plays a checkpoint (planner, eval mode, training's iterations, the run's
horizon) against each opponent and, per possession on the robot's side,
measures how long the puck sat slow near the paddle, whether a shot
followed the hold and how hard, and whether the possession ended in the
sim's RELAUNCH (the stuck rule or the shot clock's turnover) rather than
the puck leaving. The training log's "held per 10k" says nothing about
whether a hold ends in a strike or in the referee's hands; this does.
It is the row format used throughout ai/RETRAIN.md (2.x-3.x), so a new
checkpoint's line compares with the old ones.

    python ai/bin/hold_eval.py 3.11-shot-ramp-selfplay
    python ai/bin/hold_eval.py 3.9-drive-band-selfplay-500k 3.11-shot-ramp-selfplay --opponents sniper

Defaults: 16 envs x 30 s per opponent, seed 11 (~3-4 min per opponent on
the GPU). "held" = under HELD_SPEED within TRAP_DIST of the paddle for
HOLD_MIN_S in total; "shot after hold" = a forward hit that sped the puck
up by 0.2 m/s after such a hold; "relaunch" = the puck jumped > 0.3 m in
one step (the sim put it back at the centre).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.eval_play import OPPONENTS, Driver, GoalCounter, header, make_env  # noqa: E402
from airhockey.policy_loader import PLAN_ITERATIONS, load_agent  # noqa: E402
from airhockey.rewards import HELD_SPEED, HOLD_MIN_S, TRAP_DIST  # noqa: E402


def play(agent, opponent: str, n_envs: int, seconds: float, seed: int) -> str:
    e = make_env(opponent, n_envs, seconds)
    o = e.reset(seed=seed)
    drv = Driver(agent, e, planner=True)
    goals = GoalCounter(e)
    H = e.table_config.height
    in_prev = e.engine.puck_y < H / 2
    px_prev, py_prev = e.engine.puck_x.copy(), e.engine.puck_y.copy()
    sp_prev = np.hypot(e.engine.puck_vx, e.engine.puck_vy)
    fresh = lambda: dict(slow=0.0, shot=None, end=None, steps=0)   # noqa: E731
    poss = [fresh() for _ in range(n_envs)]
    done = []
    for _ in range(int(round(seconds / e.action_dt))):
        a = drv.act(o)
        o, r, term, trunc, info = e.step(a)
        drv.done(term, trunc)
        goals.update(info)
        px, py, vx, vy = info["puck_x"], info["puck_y"], info["puck_vx"], info["puck_vy"]
        sp = np.hypot(vx, vy)
        d = np.hypot(px - info["pad_x"], py - info["pad_y"])
        in_half = py < H / 2
        jumped = np.hypot(px - px_prev, py - py_prev) > 0.3
        for i in range(n_envs):
            p = poss[i]
            if in_half[i] and not in_prev[i]:
                poss[i] = p = fresh()
            if in_half[i]:
                p["steps"] += 1
                if d[i] < TRAP_DIST and sp[i] < HELD_SPEED:
                    p["slow"] += e.action_dt
                if (d[i] < 0.25 and sp[i] - sp_prev[i] > 0.2 and vy[i] > 0
                        and p["shot"] is None and p["slow"] >= HOLD_MIN_S):
                    p["shot"] = float(sp[i])
            if jumped[i] and p["steps"] > 0 and p["end"] is None:
                p["end"] = "relaunch"
                done.append(p)
                poss[i] = fresh()
            elif not in_half[i] and in_prev[i] and p["steps"] > 0:
                p["end"] = "left"
                done.append(p)
        in_prev = in_half
        px_prev, py_prev, sp_prev = px.copy(), py.copy(), sp.copy()
    held = [p for p in done if p["slow"] >= HOLD_MIN_S]
    if not held:
        return f"vs {opponent:12s}: goals {goals} | possessions {len(done)}, held 0"
    slow = np.array([p["slow"] for p in held])
    shots = [p["shot"] for p in held if p["shot"] is not None]
    rel = sum(p["end"] == "relaunch" for p in held)
    return (f"vs {opponent:12s}: goals {goals} | possessions {len(done)}, held {len(held)} | "
            f"time slow near paddle: median {np.median(slow):.1f}s mean {slow.mean():.1f}s "
            f"max {slow.max():.1f}s | held ending in relaunch {rel / len(held) * 100:.0f}%, "
            f"shot after hold {len(shots) / len(held) * 100:.0f}% | "
            f"shot speed median {np.median(shots) if shots else 0:.1f} m/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names under runs/ (or latest)")
    ap.add_argument("--opponents", default=",".join(OPPONENTS))
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--iterations", type=int, default=PLAN_ITERATIONS)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    for run in args.runs:
        agent = load_agent(run, iterations=args.iterations)
        print(header(run, agent), flush=True)
        for opp in args.opponents.split(","):
            print("  " + play(agent, opp, args.envs, args.seconds, args.seed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
