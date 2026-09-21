#!/usr/bin/env python3
"""Generate qualified, randomized skill demonstrations in simulation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.arrival_training import generate_demonstrations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--per-task", type=int, default=256)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--canonical", action="store_true")
    p.add_argument("--wide", action="store_true")
    args = p.parse_args()
    generate_demonstrations(
        args.output,
        per_task=args.per_task,
        population=args.population,
        iterations=args.iterations,
        seed=args.seed,
        canonical=args.canonical,
        wide=args.wide,
    )


if __name__ == "__main__":
    main()
