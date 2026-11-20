"""Replay filename variations must not take the entire recording list down."""

import asyncio
import json

from airhockey import server


def test_recordings_endpoint_accepts_checkpoint_variants_and_non_numeric_names(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(server, "RECORDINGS_INDEX_CACHE", tmp_path / "cache" / "index.json")
    names = [
        "training_step_0001000",
        "neural-edge106-agent_step_924073984-mirror-straight",
        "training_step_1000_original_left",
        "training_step_latest",
        "training_step_",
        "training_step_12bad",
        "training_vs_sniper",
    ]
    payload = {
        "fields": ["time", "score_agent", "score_opponent"],
        "columns": {"time": [0.02], "score_agent": [1], "score_opponent": [0]},
    }
    for name in names:
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))

    response = asyncio.run(server.list_recordings())
    entries = {entry["name"]: entry for entry in json.loads(json.dumps(response, allow_nan=False))}
    assert set(entries) == set(names)
    variant = entries[names[1]]
    assert variant["run"] == "neural-edge106-agent"
    assert variant["step"] == 924073984
    assert variant["variant"] == "mirror-straight"
    assert "mirror-straight" in variant["label"]
    assert variant["score"] == [1, 0]
    assert entries[names[0]]["label"] == "training @ 1k"
    assert entries[names[2]]["variant"] == "original_left"
    for name in names[3:6]:
        assert entries[name]["run"] == name
        assert entries[name]["step"] is None
        assert entries[name]["label"] == name
    assert entries[names[6]]["metadata"]["opponent"] == "sniper"
    # A diagnostic replay can still be selected and read after listing.
    replay = asyncio.run(server.get_recording(f"{names[1]}.json"))
    assert replay["frames"] == [
        {"time": 0.02, "score_agent": 1, "score_opponent": 0}
    ]


def test_entries_say_who_played_what_and_file_under_the_model_version(tmp_path, monkeypatch):
    """A newcomer reads the list, not the file names."""
    from airhockey import recording_index
    versions = {"_neural-table-faithful-20261110-a3": "v4.1.2", "teacher-calm-20261018-v4": "v4.0.0"}
    frames = {"fields": ["time", "score_agent", "score_opponent"],
              "columns": {"time": [0.0, 60.0], "score_agent": [0, 5], "score_opponent": [0, 2]}}

    def write(name, metadata=None):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dict(frames, **({"metadata": metadata} if metadata else {}))))
        return recording_index.describe(path, versions)

    cross = write("a3-regression-a3-vs-v4-blue-profile", {
        "match_type": "cross-play", "opponent": "teacher-calm-20261018-v4", "accel_cap_m_s2": 100.0,
        "checkpoint": "runs/_neural-table-faithful-20261110-a3/agent.pt",
        "run_name": "_neural-table-faithful-20261110-a3",
        "simulation_environment": {"thermal_path": {"holding": list(range(500))}}})
    assert (cross["section"], cross["group"], cross["group_alias"]) == (
        "models", "v4.1.2", "_neural-table-faithful-20261110-a3")
    assert (cross["blue"], cross["red"], cross["kind"]) == ("v4.1.2", "v4.0.0", "cross-play")
    assert cross["score"] == [5, 2] and cross["duration_s"] == 60.0
    # The list carries the fields it shows, not the whole environment.
    assert "simulation_environment" not in cross["metadata"] and cross["metadata"]["accel_cap_m_s2"] == 100.0

    # The same pairing with the ends swapped: the model is drawn in red.
    swapped = write("a3-regression-a3-vs-v4-red-profile", {
        "match_type": "cross-play", "opponent": "teacher-calm-20261018-v4",
        "policy_sides": "candidate red; reference blue: runs/teacher-calm-20261018-v4/agent.pt",
        "checkpoint": "runs/_neural-table-faithful-20261110-a3/agent.pt"})
    assert (swapped["group"], swapped["blue"], swapped["red"]) == ("v4.1.2", "v4.0.0", "v4.1.2")
    assert (swapped["opponent"], swapped["plays"], cross["plays"]) == ("v4.0.0", "red", "blue")

    proxy = write("behav32-v4_vs_human-proxy-table-g0", {"match_type": "human-proxy benchmark",
                                                       "checkpoint": "runs/teacher-calm-20261018-v4/agent.pt"})
    assert (proxy["group"], proxy["red"]) == ("v4.0.0", "simulated person (table-g0)")
    snapshot = write("some-run_step_0500000", {"match_type": "self-play", "run_name": "some-run"})
    assert (snapshot["section"], snapshot["group"], snapshot["red"]) == ("runs", "some-run", "a copy of itself")
    # The evaluator's self-play games name the opponent "self"; that is not a player's name.
    evaluated = write("some-run-selfplay-g0", {"match_type": "self-play", "run_name": "some-run", "opponent": "self"})
    assert evaluated["red"] == "a copy of itself" and evaluated["kind"] == "self-play"
    bare = write("old-run_step_0001000")
    assert (bare["section"], bare["group"], bare["kind"]) == ("runs", "old-run", "training snapshot")
    mouse = write("game_1773964952")
    assert (mouse["section"], mouse["blue"]) == ("manual", "you (mouse)")

    # Unchanged files are served from the cache file, across "restarts" too.
    cache = tmp_path / "cache" / "index.json"
    monkeypatch.setattr(recording_index, "model_names", lambda: versions)
    first = recording_index.list_recordings(tmp_path, cache)
    assert len(first) == 7 and cache.is_file()
    recording_index._MEMORY.clear()
    monkeypatch.setattr(recording_index, "describe", lambda *a: (_ for _ in ()).throw(AssertionError("re-read")))
    assert recording_index.list_recordings(tmp_path, cache) == first
