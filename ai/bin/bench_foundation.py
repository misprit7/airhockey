#!/usr/bin/env python3
"""Fixed held-out shooting/control/defense benchmark; never connects to hardware."""

import argparse
import json
from pathlib import Path
import time

from train_arrival import ROOT, config, TDMPC2, torch
from airhockey.policy_loader import load_agent
from airhockey.policy_benchmark import evaluate_skills


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-task", type=int, default=100)
    p.add_argument(
        "--output", type=Path, default=ROOT / "logs/foundation/baseline.json"
    )
    p.add_argument("--seed", type=int, default=20261101)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, accel in (
        ("3.11-shot-ramp-selfplay", 40),
        ("3.11-shot-ramp-selfplay", 60),
        ("3.12-accel60-accuracy-selfplay", 60),
        ("4.0-arrival-rms-selfplay", 60),
    ):
        legacy = not name.startswith("4.")
        if legacy:
            agent = load_agent(name)
        else:
            run_dir = ROOT / "runs" / name
            meta = json.loads((run_dir / "run.json").read_text())
            agent = TDMPC2(config(argparse.Namespace(**meta["args"]), run_dir))
            agent.load(run_dir / "agent.pt")
        agent.cfg.num_samples = 256  # same planning budget for every candidate
        for planner in (False, True):
            torch.manual_seed(args.seed)
            start = time.perf_counter()
            result = evaluate_skills(
                agent,
                legacy=legacy,
                accel=accel,
                planner=planner,
                seed=args.seed,
                per_task=args.per_task,
            )
            result.update(run=name, wall_s=time.perf_counter() - start)
            rows.append(result)
            args.output.write_text(json.dumps(rows, indent=2))
            print(json.dumps(result), flush=True)
        del agent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
