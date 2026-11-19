"""The replay menu's index of simulated games (``ai/recordings/*.json``).

A recording's file name is whatever the script that wrote it chose
(``a3-regression-a3-vs-v4-blue-profile``), which tells a newcomer nothing. Each
entry here says in plain fields what the file is: WHO played (``blue`` is the
robot's end, drawn at the bottom; ``red`` is the far end), what KIND of game it
was, and which GROUP it files under -- a registered model version
(``ai/models.json``) when the checkpoint is one, else the training run.

Parsing 3,000 files of ~250 KB takes seconds, so entries are cached on disk
(``logs/recordings_index.json``), keyed by name, mtime and size, and the list
is built off the event loop.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from airhockey import models

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "logs" / "recordings_index.json"
CACHE_VERSION = 5
# What the list needs from a recording's metadata. The full metadata (which
# can carry a whole thermal model) is sent only with the recording itself.
LIST_KEYS = ("algo", "opponent", "match_type", "policy_sides", "accel_cap_m_s2", "seconds",
             "scenario", "selected_outcome", "shot_request", "person", "note", "step", "stage_name")

_STEP_RE = re.compile(r"(.+)_(?:step|update)_([0-9]+)((?:[-_].*)?)")


def checkpoint_parts(stem: str) -> tuple[str, int | None, str]:
    """(run, step, suffix) of a ``<run>_step_<n><suffix>`` name."""
    match = _STEP_RE.fullmatch(stem)
    if match is None:
        return stem, None, ""
    return match[1], int(match[2]), match[3]


def label(stem: str) -> str:
    """'ppo_v4_shaped_step_0100000' -> 'ppo_v4_shaped @ step 100k'."""
    run, step, suffix = checkpoint_parts(stem)
    if step is None:
        return stem
    text = f"{step / 1e6:.1f}M" if step >= 1_000_000 else f"{step // 1000}k" if step >= 1000 else str(step)
    unit = "update " if stem.startswith(run + "_update_") else ""
    return f"{run} @ {unit}{text}{suffix}"


def model_names() -> dict[str, str]:
    """Run directory name -> registered version (``v4.1.2``)."""
    try:
        return {entry["run"]: version for version, entry in models.registry()["models"].items()}
    except (OSError, ValueError, KeyError):
        return {}


def _run_of(checkpoint) -> str | None:
    """'runs/<run>/agent.pt' -> '<run>'."""
    parts = Path(str(checkpoint)).parts if checkpoint else ()
    return parts[-2] if len(parts) >= 2 else None


def _player(name, versions: dict[str, str]) -> str | None:
    """A readable name for whoever drove a mallet."""
    if not name:
        return None
    name = str(name)
    run = _run_of(name) if "/" in name else name
    if run in versions:
        return versions[run]
    if name.startswith("human-proxy"):
        tier = name.removeprefix("human-proxy").strip("-_ ")
        return f"simulated person ({tier})" if tier else "simulated person"
    if name.startswith("expert"):
        return "physics teacher (scripted)"
    return run or name


def describe(path: Path, versions: dict[str, str]) -> dict:
    """One list entry. Never raises on a corrupt file: it still lists."""
    st = path.stat()
    stem = path.stem
    run, step, suffix = checkpoint_parts(stem)
    opponent = None
    if step is None and "_vs_" in stem:
        run, opponent = stem.rsplit("_vs_", 1)
    entry = dict(name=stem, path=path.name, label=label(stem), run=run, step=step,
                 variant=suffix.lstrip("-_"), mtime=st.st_mtime, size=st.st_size,
                 date=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)))
    meta = {}
    try:
        data = json.loads(path.read_text(), parse_constant=lambda _c: None)
        if isinstance(data, dict) and "columns" in data:
            cols = data["columns"]
            n = len(cols[data["fields"][0]])
            entry["frames"] = n
            if n and "time" in cols:
                entry["duration_s"] = round(float(cols["time"][-1]), 1)
            if n and "score_agent" in cols and "score_opponent" in cols:
                entry["score"] = [int(cols["score_agent"][-1]), int(cols["score_opponent"][-1])]
            if isinstance(data.get("metadata"), dict):
                meta = data["metadata"]
        elif isinstance(data, list):
            entry["frames"] = len(data)
    except Exception:            # noqa: BLE001
        entry["unreadable"] = True
    if opponent and "opponent" not in meta:
        meta = dict(meta, opponent=opponent)
    entry["metadata"] = {k: meta[k] for k in LIST_KEYS if meta.get(k) is not None}

    source = meta.get("run_name") or meta.get("run") or _run_of(meta.get("checkpoint")) or run
    version = versions.get(str(source))
    manual = stem.startswith("game_") and not meta
    if manual:
        entry.update(section="manual", group="Games played by mouse", blue="you (mouse)",
                     red="scripted opponent", opponent="scripted opponent", plays="blue",
                     kind="game played in the Simulation tab")
        return entry
    kind = meta.get("match_type") or ("isolated drill" if meta.get("scenario") is not None else None)
    if kind is None:
        kind = "training snapshot" if step is not None else "game"
    red = _player(meta.get("opponent"), versions)
    if red is None and kind == "self-play":
        red = "a copy of itself"
    elif red is None and str(kind).startswith("isolated"):
        red = "a scripted attack"
    elif red is None and str(kind).startswith("resting-puck stall"):
        red = "nobody (a puck placed at rest)"
    model = version or _player(source, versions) or meta.get("algo") or "unknown"
    other = red or "unknown opponent"
    # Paired evaluations play each game twice with the ends swapped; the
    # writer says so in policy_sides ("candidate red; reference blue: ...").
    swapped = str(meta.get("policy_sides", "")).lower().startswith("candidate red")
    entry.update(section="models" if version else "runs", group=version or str(source),
                 group_alias=str(source) if version else None,
                 blue=other if swapped else model, red=model if swapped else other,
                 opponent=other, plays="red" if swapped else "blue", kind=str(kind))
    return entry


_LOCK = threading.Lock()
_MEMORY: dict[str, dict[str, dict]] = {}     # cache file -> {file name: entry}


def _load_cache(cache_path: Path) -> dict[str, dict]:
    try:
        data = json.loads(cache_path.read_text())
        if data.get("version") == CACHE_VERSION and isinstance(data.get("entries"), dict):
            return data["entries"]
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def list_recordings(directory: Path, cache_path: Path = CACHE_PATH) -> list[dict]:
    """Every recording, newest first. Files unchanged since the last call
    (same mtime and size) are not read again, across server restarts too."""
    with _LOCK:
        known = _MEMORY.get(str(cache_path))
        if known is None:
            known = _load_cache(cache_path)
        versions = model_names()
        signature = ",".join(sorted(f"{run}={version}" for run, version in versions.items()))
        out, fresh, changed = [], {}, False
        for path in directory.glob("*.json"):
            try:
                st = path.stat()
            except OSError:
                continue
            cached = known.get(path.name)
            if (cached and cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size
                    and cached.get("_models") == signature):
                entry = cached
            else:
                entry = dict(describe(path, versions), _models=signature)
                changed = True
            fresh[path.name] = entry
            out.append({k: v for k, v in entry.items() if not k.startswith("_")})
        _MEMORY[str(cache_path)] = fresh
        if changed or len(fresh) != len(known):
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(version=CACHE_VERSION, entries=fresh)))
                tmp.replace(cache_path)
            except OSError:
                pass             # a read-only logs/ only costs the next start its cache
        out.sort(key=lambda e: e["mtime"], reverse=True)
        return out
