#!/usr/bin/env python3
"""Summarize a motor-load log. Files only: never contacts or enables the robot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.motor_load import summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="defaults to newest logs/motor_load/*.jsonl")
    args = parser.parse_args()
    path = args.path
    if path is None:
        paths = list(Path("logs/motor_load").glob("*.jsonl"))
        if not paths:
            parser.error("no motor-load recordings yet")
        path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    print(json.dumps(summarize(path), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
