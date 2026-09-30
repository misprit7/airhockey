"""Replay filename variations must not take the entire recording list down."""

import asyncio
import json

from airhockey import server


def test_recordings_endpoint_accepts_checkpoint_variants_and_non_numeric_names(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(server, "_REC_CACHE", {})
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
