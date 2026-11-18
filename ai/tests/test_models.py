"""The neural-player version registry (airhockey/models.py, ai/models.json)."""
import json

import pytest

from airhockey import models

DATA = dict(
    latest="v4.1.2",
    difficulty=dict(max=dict(model="v4.1.2"), easy=dict(model="v4.2.1"), hard=dict(model="v4.1.2", accel=60)),
    models={
        "v4.0.0": dict(run="old", date="2026-10-18", accel=100, parent=None, summary="a"),
        "v4.1.2": dict(run="a3", file="agent.pt", date="2026-11-10", accel=100, parent="v4.0.0", summary="b"),
        "v4.2.1": dict(run="v4.2.1", date="2026-11-18", accel=25, parent="v4.1.2", summary="c"),
    })


def test_the_checked_in_registry_is_consistent():
    data = models.registry()
    assert models.parse(data["latest"]) is not None
    for name, entry in data["models"].items():
        assert {"run", "date", "accel", "summary"} <= set(entry), name
        assert entry.get("parent") is None or entry["parent"] in data["models"], name
        assert models.parse(entry["parent"]) < models.parse(name) if entry.get("parent") else True
    for mode in data["difficulty"]:
        version, accel = models.difficulty(mode, data)
        # A mode never asks a model for more than its coordinate units allow.
        assert 5 < accel <= 100 and version in data["models"]
    assert models.difficulty("max", data)[0] == data["latest"]


def test_versions_parse_and_count_up():
    assert models.parse("v4.1.2") == (4, 1, 2)
    assert models.parse("4.1.2") is None and models.parse("v4.1") is None and models.parse("v4.1.2-x") is None
    assert models.next_version("model", DATA) == "v4.2.2"
    assert models.next_version("study", DATA) == "v4.3.0"
    assert models.next_version("major", DATA) == "v5.0.0"
    with pytest.raises(ValueError):
        models.next_version("patch", DATA)


def test_resolve_latest_versions_and_modes(tmp_path):
    for run in ("a3", "v4.2.1"):
        (tmp_path / "runs" / run).mkdir(parents=True)
        (tmp_path / "runs" / run / "agent.pt").write_bytes(b"x")
    assert models.resolve("latest", DATA, tmp_path) == tmp_path / "runs" / "a3" / "agent.pt"
    assert models.resolve("v4.2.1", DATA, tmp_path) == tmp_path / "runs" / "v4.2.1" / "agent.pt"
    with pytest.raises(FileNotFoundError):
        models.resolve("v4.0.0", DATA, tmp_path)
    with pytest.raises(KeyError):
        models.resolve("v9.9.9", DATA, tmp_path)
    assert models.is_version("latest", DATA) and models.is_version("v4.1.2", DATA)
    assert not models.is_version("v9.9.9", DATA) and not models.is_version("runs/a3/agent.pt", DATA)
    assert models.difficulty("max", DATA) == ("v4.1.2", 100.0)
    assert models.difficulty("hard", DATA) == ("v4.1.2", 60.0)
    assert models.difficulty("easy", DATA) == ("v4.2.1", 25.0)
    with pytest.raises(KeyError):
        models.difficulty("impossible", DATA)
    assert "latest max@100 hard@60" in models.table(DATA)


def test_a_registry_naming_an_unregistered_model_is_refused(tmp_path):
    bad = json.loads(json.dumps(DATA))
    bad["latest"] = "v7.0.0"
    path = tmp_path / "models.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        models.registry(path)
    bad["latest"] = "v4.1.2"
    bad["models"]["a3"] = bad["models"]["v4.1.2"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        models.registry(path)
