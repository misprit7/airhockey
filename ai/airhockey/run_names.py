"""Run names: ``<major>.<minor>-<description>-<stage>``.

    2.3-control-gate-selfplay
    3.0-horizon8-selfplay
    4.0-strike-primitive-proximity      (a pretrain stage)
    3.3-turnover-selfplay-300k          (a pinned snapshot of a run)

MAJOR bumps when a checkpoint cannot resume from the previous lineage:
a new action space, observation layout, planning horizon or model. MINOR
bumps for every recipe change (reward, opponents, demonstrations) that
resumes within the lineage. The DESCRIPTION is the one thing that
changed, kebab-case. The STAGE is the curriculum stage or ``selfplay``,
optionally followed by ``-<step>k`` for a snapshot pinned from a run.

``ai/RUNS.md`` is the registry: every version, its parent and its one
line. Names starting with ``_`` are scratch (smoke tests, benchmark
copies) and are exempt.

    python -m airhockey.run_names 3 shot-clock selfplay   # -> 3.4-shot-clock-selfplay
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

STAGES = ("proximity", "contact", "scoring", "goalie", "selfplay")
RUN_NAME_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)-(?P<desc>[a-z0-9]+(?:-[a-z0-9]+)*)-"
    r"(?P<stage>" + "|".join(STAGES) + r")(?:-(?P<snap>\d+k))?$")
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def is_scratch(name: str) -> bool:
    return name.startswith("_")


def parse(name: str) -> dict | None:
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["major"], d["minor"] = int(d["major"]), int(d["minor"])
    return d


def check_run_name(name: str) -> None:
    """Raise ValueError unless ``name`` follows the scheme (or is scratch)."""
    if is_scratch(name) or parse(name):
        return
    raise ValueError(
        f"run name {name!r} does not follow <major>.<minor>-<description>-<stage>, "
        f"e.g. 3.4-shot-clock-selfplay (stages: {', '.join(STAGES)}; names starting "
        f"with '_' are scratch). See ai/airhockey/run_names.py and ai/RUNS.md.")


def next_name(major: int, desc: str, stage: str, runs_dir: Path = RUNS_DIR) -> str:
    """The next free minor under ``major``, from the run directories."""
    minors = [p["minor"] for d in runs_dir.iterdir() if d.is_dir()
              for p in [parse(d.name)] if p and p["major"] == major and not p["snap"]]
    minor = max(minors) + 1 if minors else 0
    name = f"{major}.{minor}-{desc}-{stage}"
    check_run_name(name)
    return name


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    print(next_name(int(sys.argv[1]), sys.argv[2], sys.argv[3]))
