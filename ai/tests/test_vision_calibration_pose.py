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


@pytest.mark.parametrize('running,frame_age,pose_age,expected_stall', [
    (True,.05,.05,False), (True,2,.05,True),
    (True,.05,2,False), (False,.05,.05,False),
])
def test_preview_never_presents_stale_frame_or_pose_as_live(monkeypatch, running, frame_age, pose_age, expected_stall):
    monkeypatch.setattr('airhockey.vision_service.time.time', lambda: 100.)
    service=VisionService()
    service._thread=SimpleNamespace(is_alive=lambda: running)
    service._frame_t=100-frame_age;service._pose_t=100-pose_age
    service._pose=(1500.,450.,2.35);service._fps=12.;service._jpeg=b'jpeg'
    service._puck={'x':800.,'y':450.,'theta':0.,'n':4}
    status=service.status()
    assert status['stalled']==expected_stall
    assert (status['pose'] is not None)==(running and frame_age<=1 and pose_age<=.25)
    assert (status['puck'] is not None)==(running and frame_age<=1)
    assert status['fps']==(12. if running and frame_age<=1 else 0.)
    assert service.frame_jpeg()==(b'jpeg' if running and frame_age<=1 else None)


def test_preview_start_clears_previous_image_and_objects(monkeypatch):
    class Thread:
        def __init__(self,**kwargs):pass
        def start(self):pass
        def is_alive(self):return True
    monkeypatch.setattr('airhockey.vision_service.threading.Thread',Thread)
    service=VisionService();service._jpeg=b'old';service._puck={'old':True};service._fps=12
    service.start()
    assert service._jpeg is None and service._puck is None and service._fps==0
    assert service.status()['generation']==1


def test_preview_applies_reported_camera_settings_to_capture_and_detection(monkeypatch):
    import numpy as np
    import camera
    import track_mallet as tm
    service = VisionService(exposure_us=450, gain_db=6, threshold=170)
    captured = {}

    class Stream:
        def __init__(self, exposure, gain):
            captured.update(exposure_us=exposure, gain_db=gain)

        def grab(self):
            service._stop.set()  # Process one synthetic frame, then exit.
            return np.zeros((20, 20), dtype=np.uint8)

        def close(self):
            pass

    def candidates(img, known, *, threshold):
        captured['threshold'] = threshold
        return []

    monkeypatch.setattr(camera, 'Stream', Stream)
    monkeypatch.setattr(tm, 'load_pose', lambda: (None,) * 5)
    monkeypatch.setattr(tm, 'field_marker_pixels', lambda *args: [])
    monkeypatch.setattr(tm, 'find_candidates', candidates)
    monkeypatch.setattr(tm, 'locate', lambda *args, **kwargs: (None, 'no markers'))
    monkeypatch.setattr('airhockey.vision_service._detect_loose', lambda *args: (None, None))
    monkeypatch.setattr(service, '_annotate', lambda *args: b'jpeg')
    service._run()
    assert service.error is None
    assert captured == service.status()['settings']
