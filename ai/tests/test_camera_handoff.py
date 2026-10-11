"""The launcher may release a preview, but must never talk to hardware."""

import errno
import io
import json
from urllib.error import URLError

import pytest

from airhockey import camera_handoff as handoff


def install_responses(monkeypatch, *responses):
    pending = iter(responses)
    calls = []

    def open_request(request, timeout):
        calls.append((request.full_url, request.method, timeout))
        result = next(pending)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(handoff, "urlopen", open_request)
    return calls


def test_running_preview_is_stopped_and_left_off(monkeypatch):
    calls = install_responses(monkeypatch, {"running": True}, {"running": False})
    assert handoff.release_preview_camera()
    assert calls == [
        ("http://127.0.0.1:8420/camera/status", "GET", 2.0),
        ("http://127.0.0.1:8420/camera/stop", "POST", 5.0),
    ]


@pytest.mark.parametrize("response", [
    {"running": False},
    URLError(ConnectionRefusedError(errno.ECONNREFUSED, "no web server")),
])
def test_camera_can_open_without_active_preview(monkeypatch, response):
    calls = install_responses(monkeypatch, response)
    assert not handoff.release_preview_camera()
    assert len(calls) == 1


@pytest.mark.parametrize("response", [
    TimeoutError("stalled server"),
    URLError(TimeoutError("stalled server")),
    {},
    {"running": "false"},
])
def test_uncertain_ownership_does_not_proceed(monkeypatch, response):
    install_responses(monkeypatch, response)
    with pytest.raises(RuntimeError, match="could not check"):
        handoff.release_preview_camera()


@pytest.mark.parametrize("response", [
    {"running": True},
    {},
    TimeoutError("stalled stop"),
    URLError(ConnectionRefusedError(errno.ECONNREFUSED, "server exited")),
])
def test_unconfirmed_stop_does_not_proceed(monkeypatch, response):
    install_responses(monkeypatch, {"running": True}, response)
    with pytest.raises(RuntimeError):
        handoff.release_preview_camera()
