"""Checkpoint actuator metadata, without loading a model or hardware."""
import json

import pytest

from airhockey.policy_loader import trained_limits


def test_legacy_limits(tmp_path):
    assert trained_limits(tmp_path) == (12.0, 40.0)
    (tmp_path / "run.json").write_text('{"horizon": 8}')
    assert trained_limits(tmp_path) == (12.0, 40.0)


def test_metadata_limits(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({
        "agent_speed_range": [8, 10], "agent_accel_range": [50, 60],
    }))
    assert trained_limits(tmp_path) == (10.0, 60.0)


@pytest.mark.parametrize("band", [[60], [60, 40], [0, 60], [60, float("nan")],
                                   [60, float("inf")], [True, 60], "60", None])
def test_invalid_metadata_rejected(tmp_path, band):
    (tmp_path / "run.json").write_text(json.dumps({"agent_accel_range": band}))
    with pytest.raises(ValueError, match="agent_accel_range"):
        trained_limits(tmp_path)


def test_corrupt_metadata_rejected(tmp_path):
    (tmp_path / "run.json").write_text('{"agent_accel_range":')
    with pytest.raises(ValueError):
        trained_limits(tmp_path)
