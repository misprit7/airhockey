"""Neural-player model versions: ``v<major>.<study>.<model>``.

    v4.1.2      major 4, study 1, model 2

MAJOR   a large or architectural change (network, observation or action
        layout): a checkpoint that cannot simply continue the last one.
STUDY   one finished piece of work on the recipe: it goes up each time
        someone (a person or an AI session) completes a study and trains
        from its conclusions.
MODEL   the models trained within that study, in order.

``ai/models.json`` is the registry: every version, the run directory that
holds it, its parent, which version is ``latest`` (the one to put on the
table at full strength) and the difficulty modes. ``ai/MODELS.md`` is the
same thing for reading. ``runs/`` is not in git, so the registry is the
only durable record of which directory a version is.

    python -m airhockey.models                 # the table
    python -m airhockey.models path latest     # checkpoint file of a version
    python -m airhockey.models next study      # next free version (study | model | major)
    python -m airhockey.models difficulty easy # "<version> <accel m/s^2>"

The table runner takes these names: ``--policy neural:v4.1.2``,
``neural:latest``; ``ai/bin/play.sh --difficulty easy``. TD-MPC2 runs keep
their own scheme (``airhockey/run_names.py``, ``ai/RUNS.md``). No torch here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "ai" / "models.json"
VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def parse(name: str) -> tuple[int, int, int] | None:
    m = VERSION_RE.match(str(name))
    return tuple(int(g) for g in m.groups()) if m else None


def registry(path: Path = REGISTRY) -> dict:
    data = json.loads(Path(path).read_text())
    bad = [name for name in data["models"] if parse(name) is None]
    if bad:
        raise ValueError(f"{path}: not v<major>.<study>.<model>: {bad}")
    for key, target in [("latest", data["latest"])] + [(k, v["model"]) for k, v in data["difficulty"].items()]:
        if target not in data["models"]:
            raise ValueError(f"{path}: {key} names {target}, which is not registered")
    return data


def is_version(name: str, data: dict | None = None) -> bool:
    """A name the registry answers for: ``latest`` or a registered version."""
    return name == "latest" or (parse(name) is not None and name in (data or registry())["models"])


def resolve(name: str, data: dict | None = None, root: Path = ROOT) -> Path:
    """Checkpoint file of ``latest`` or a registered version."""
    data = data or registry()
    version = data["latest"] if name == "latest" else name
    if version not in data["models"]:
        raise KeyError(f"{name} is not a registered model version (ai/models.json)")
    entry = data["models"][version]
    path = Path(root) / "runs" / entry["run"] / entry.get("file", "agent.pt")
    if not path.is_file():
        raise FileNotFoundError(f"{version}: {path} is missing (runs/ is not in git)")
    return path


def difficulty(mode: str, data: dict | None = None) -> tuple[str, float]:
    """(version, accel limit in m/s^2) of a difficulty mode."""
    data = data or registry()
    if mode not in data["difficulty"]:
        raise KeyError(f"unknown difficulty {mode!r}; one of {', '.join(data['difficulty'])}")
    entry = data["difficulty"][mode]
    return entry["model"], float(entry.get("accel", data["models"][entry["model"]]["accel"]))


def next_version(kind: str, data: dict | None = None) -> str:
    """Next free version: a new ``major``, a new ``study`` under the latest
    major, or a new ``model`` in the latest study."""
    versions = sorted(parse(name) for name in (data or registry())["models"])
    major, study, model = versions[-1]
    if kind == "major":
        return f"v{major + 1}.0.0"
    if kind == "study":
        return f"v{major}.{study + 1}.0"
    if kind == "model":
        return f"v{major}.{study}.{model + 1}"
    raise ValueError("kind must be major, study or model")


def table(data: dict | None = None) -> str:
    data = data or registry()
    modes = {}
    for mode, entry in data["difficulty"].items():
        modes.setdefault(entry["model"], []).append(f"{mode}@{difficulty(mode, data)[1]:g}")
    rows = []
    for name in sorted(data["models"], key=parse):
        e = data["models"][name]
        tags = (["latest"] if name == data["latest"] else []) + modes.get(name, [])
        rows.append(f"{name:8} {e['date']}  {e['accel']:>5g} m/s^2  {e.get('parent') or '-':8} "
                    f"{e['run']:40} {' '.join(tags)}\n         {e['summary']}")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    if not argv:
        print(table())
    elif argv[0] == "path" and len(argv) == 2:
        print(resolve(argv[1]))
    elif argv[0] == "next" and len(argv) == 2:
        print(next_version(argv[1]))
    elif argv[0] == "difficulty" and len(argv) == 2:
        version, accel = difficulty(argv[1])
        print(f"{version} {accel:g}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
