#!/usr/bin/env python3
"""Evaluate one legacy/extended checkpoint on fixed physical skill fixtures."""

import argparse
import json
from pathlib import Path

from eval_foundation import load, torch
from airhockey.policy_benchmark import evaluate_skills


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--per-task", type=int, default=100)
    p.add_argument("--accel", type=float, default=60.0)
    p.add_argument("--wide", action="store_true")
    p.add_argument("--guard", action="store_true")
    p.add_argument("--game-requests", action="store_true")
    p.add_argument("--random-paddle", action="store_true")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--prior-only", action="store_true")
    modes.add_argument("--mpc-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    agent = load(args.checkpoint)
    agent.cfg.iterations, agent.cfg.num_samples = 6, 256
    rows = []
    for planner in (
        [False] if args.prior_only else [True] if args.mpc_only else [False, True]
    ):
        result = evaluate_skills(
            agent,
            legacy=True,
            seed=args.seed,
            per_task=args.per_task,
            accel=args.accel,
            planner=planner,
            wide=args.wide,
            guard=args.guard,
            game_requests=args.game_requests,
            random_paddle=args.random_paddle,
        )
        result["checkpoint"] = str(args.checkpoint)
        rows.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
