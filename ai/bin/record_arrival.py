#!/usr/bin/env python3
"""Backfill/watch arrival training checkpoints without restarting the trainer.

Simulation only. Run from the repository root with system Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

# Reuse the training configuration and sibling TD-MPC2 import setup.
from train_arrival import ROOT, config, TDMPC2, torch
from airhockey.arrival_recording import RECORDINGS_DIR, pending_checkpoints, record_game


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--every", type=int, default=500_000)
    p.add_argument("--directory", type=Path, default=RECORDINGS_DIR)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument(
        "--opponent",
        choices=("self", "sniper", "goalie", "weak_goalie"),
        default="self",
    )
    args = p.parse_args()
    if args.every < 1 or args.duration <= 0:
        p.error("interval and duration must be positive")
    run_dir = args.run.resolve()
    metadata = json.loads((run_dir / "run.json").read_text())
    if metadata.get("action_mode") != "arrival":
        p.error("expected an arrival-action run")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    agent = TDMPC2(config(argparse.Namespace(**metadata["args"]), run_dir))
    # Fixed batch (two views for self-play), reused across immutable checkpoints.
    agent._plan_batch = torch.compile(agent._plan_batch, mode="reduce-overhead")
    with tempfile.TemporaryDirectory(prefix="arrival-recording-") as tmp:
        thermal = Path(tmp) / "thermal.json"
        thermal.write_text(json.dumps(metadata["thermal_model"]))
        while True:
            pending = pending_checkpoints(
                run_dir,
                every=args.every,
                directory=args.directory,
                opponent=args.opponent,
            )
            if pending:
                step, checkpoint = pending[0]
                print(f"[recording] loading {checkpoint}", flush=True)
                agent.load(checkpoint)
                record_game(
                    agent,
                    step,
                    run_dir.name,
                    directory=args.directory,
                    opponent=args.opponent,
                    duration=args.duration,
                    thermal_path=thermal,
                    accel=metadata["agent_accel_range"][1],
                    checkpoint=checkpoint.relative_to(ROOT),
                )
                continue
            if not args.watch or (run_dir / "status.json").exists():
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
