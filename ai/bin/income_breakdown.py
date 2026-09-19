#!/usr/bin/env python3
"""Where a style's reward comes from, term by term.

Plays up to three ways of moving the paddle on the same table -- the
scripted CushionBot (`airhockey/cushion_bot.py`), a checkpoint's policy
PRIOR alone, and the same checkpoint with the planner -- against each
opponent, scores every step with the CURRENT self-play shaper, and prints
the income per 10k steps split by term (the shaper's `stats["pay_*"]`
counters: shot, goal, hold, drive, clock, penalty, accel, idle, field ...).

Run it BEFORE changing a reward. It is how 2.6 found the per-step defense
income paying ten times the goals for every style (and switching off
whenever the puck was held), and how 3.4 found the taxed shot clock being
paid rather than avoided. Prior-vs-planner on the same checkpoint is also
the test of whether the planner is overriding what the prior learned.

    python ai/bin/income_breakdown.py 3.11-shot-ramp-selfplay
    python ai/bin/income_breakdown.py 3.11-shot-ramp-selfplay --styles prior,planner --opponents sniper

Defaults: 16 envs x 30 s, seed 11, opponents weak_goalie and sniper (the
bot cannot drive an external far side). ~3 min per style per opponent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.cushion_bot import CushionBot  # noqa: E402
from airhockey.eval_play import Driver, GoalCounter, header, make_env  # noqa: E402
from airhockey.policy_loader import PLAN_ITERATIONS, load_agent  # noqa: E402
from airhockey.rewards import (STAGE_SCORING, BatchRewardShaper,  # noqa: E402
                               curriculum_shaper_kwargs)

STYLES = ("bot", "prior", "planner")


def play(style: str, agent, opponent: str, n_envs: int, seconds: float, seed: int) -> str:
    e = make_env(opponent, n_envs, seconds)
    sh = BatchRewardShaper(n_envs, stage=STAGE_SCORING, workspace=e._ws,
                           **curriculum_shaper_kwargs("selfplay"))
    o = e.reset(seed=seed)
    sh.reset(o, info={"puck_y": e.engine.puck_y, "puck_vx": e.engine.puck_vx,
                      "puck_vy": e.engine.puck_vy})
    goals = GoalCounter(e)
    if style == "bot":
        bot = CushionBot(e, np.random.default_rng(seed))
        act = lambda obs: bot.act()          # noqa: E731
        done = lambda term, trunc: None      # noqa: E731
    else:
        drv = Driver(agent, e, planner=(style == "planner"))
        act, done = drv.act, drv.done
    steps = int(round(seconds / e.action_dt))
    for _ in range(steps):
        a = act(o)
        o, r, term, trunc, info = e.step(a)
        done(term, trunc)
        sh.compute(o, r, actions=a, info=info)
        goals.update(info)
    st = sh.stats
    k10 = 1e4 / (steps * n_envs)
    pays = {k[4:]: v * k10 for k, v in st.items() if k.startswith("pay_")}
    total = sum(pays.values())
    terms = " ".join(f"{k} {v:.0f}" for k, v in pays.items() if abs(v) >= 5)
    return (f"{style:7s} vs {opponent:12s}: goals {goals} | held {st['held'] * k10:5.1f}/10k "
            f"hold_steps {st['hold_steps'] * k10:5.0f} | on-target {st['on_target']}/{st['shots']} "
            f"ctrl x {st['patience_sum'] / max(st['on_target'], 1):.2f} "
            f"goal x {st['goal_patience_sum'] / max(st['goals'], 1):.2f} | "
            f"reward/10k {total:6.0f} = {terms}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name under runs/ (or latest); used by the prior and planner styles")
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--opponents", default="weak_goalie,sniper")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--iterations", type=int, default=PLAN_ITERATIONS)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    styles = args.styles.split(",")
    agent = load_agent(args.run, iterations=args.iterations) if set(styles) - {"bot"} else None
    if agent is not None:
        print(header(args.run, agent), flush=True)
    for opp in args.opponents.split(","):
        for style in styles:
            print("  " + play(style, agent, opp, args.envs, args.seconds, args.seed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
