"""Calibration must never reuse a stopped or stale camera observation."""
from types import SimpleNamespace

import pytest

from airhockey.vision_service import VisionService


@pytest.mark.parametrize('age,running,stopping,expected', [
    (0.05, True, False, True),
    (0.3, True, False, False),
    (30, True, False, False),
    (-0.1, True, False, False),
    (0.05, False, False, False),
    (0.05, True, True, False),
])
def test_calibration_pose_freshness(monkeypatch, age, running, stopping, expected):
    monkeypatch.setattr('airhockey.vision_service.time.time', lambda: 100.0)
    service = VisionService()
    service._thread = SimpleNamespace(is_alive=lambda: running)
    service._pose = (1461.1, 447.3, 2.39)
    service._pose_t = 100.0 - age
    if stopping:
        service._stop.set()
    assert service.latest_pose() == (service._pose if expected else None)


def test_calibration_rejects_nonfinite_pose(monkeypatch):
    monkeypatch.setattr('airhockey.vision_service.time.time', lambda: 100.0)
    service = VisionService()
    service._thread = SimpleNamespace(is_alive=lambda: True)
    service._pose = (1461.1, 447.3, float('nan'))
    service._pose_t = 100.0
    assert service.latest_pose() is None
