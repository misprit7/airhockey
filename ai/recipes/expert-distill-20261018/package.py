#!/usr/bin/env python3
"""Package a selected checkpoint as runs/<name>/ for review. Simulation only.

    PYTHONPATH=ai python3 ai/recipes/expert-distill-20261018/package.py \
        runs/<run>/agent.pt teacher-dagger-20261018-v1 --summary <holdout summary.json>

Copies agent.pt and its run.json (selected_checkpoint, sha256 and the
evaluation summary added; deployment_ready/physical_qualification stay
false), so `neural:<name>` resolves through the normal deploy loader. It
changes no deployment default and never touches hardware.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("name")
    p.add_argument("--summary", type=Path, action="append", default=[],
                   help="Benchmark summary JSON files to embed (suite.py summary.json, proxy reports)")
    p.add_argument("--note", action="append", default=[])
    a = p.parse_args()
    source = (ROOT / a.checkpoint).resolve() if not a.checkpoint.is_absolute() else a.checkpoint
    target = ROOT / "runs" / a.name
    if target.exists():
        p.error(f"{target} exists; packages are never overwritten")
    meta = json.loads((source.parent / "run.json").read_text())
    target.mkdir(parents=True)
    shutil.copy2(source, target / "agent.pt")
    digest = hashlib.sha256((target / "agent.pt").read_bytes()).hexdigest()
    meta.update(selected_checkpoint=str(source.relative_to(ROOT)), checkpoint_sha256=digest,
                deployment_ready=False, simulation_only=True, physical_qualification=False,
                package_notes=a.note,
                evaluation={str(s): json.loads(Path(s).read_text()) for s in a.summary})
    (target / "run.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(dict(package=str(target.relative_to(ROOT)), sha256=digest,
                          algorithm=meta.get("algorithm"), action_mode=meta.get("action_mode"))))


if __name__ == "__main__":
    main()
