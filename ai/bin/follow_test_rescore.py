#!/usr/bin/env python3
"""Re-judge a tracking-test run from its samples, with the current scoring.

    python ai/bin/follow_test_rescore.py logs/follow_test/20260906-005832.csv
    python ai/bin/follow_test_rescore.py logs/follow_test/*.csv

The CSV carries everything the verdict is computed from, so a scoring
change (a threshold, a fixed artefact) applies to runs already made
without moving the robot again.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey import follow_test as ft  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in ("", "None"):
                    row[k] = None
                elif k in ("seg",):
                    row[k] = int(v)
                elif k in ("name", "kind"):
                    row[k] = v
                elif k.startswith("c") and k[1:].isdigit():
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        path = Path(arg)
        rows = load_rows(path)
        segs = ft.sequence()
        camera = any(r.get("cam_t") is not None for r in rows)
        s = ft.summarize(rows, segs, camera=camera)
        caps = {}
        try:
            import json
            meta = json.loads(path.with_suffix(".json").read_text())
            caps = meta.get("caps", {})
            s["stopped_early"] = meta.get("stopped_early", False)
        except (OSError, ValueError):
            pass
        s["caps"] = caps
        print(f"== {path.name}")
        print("   " + ft.format_summary(s).replace("\n", "\n   "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
