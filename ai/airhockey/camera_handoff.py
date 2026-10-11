"""Release the local web preview before a policy opens the exclusive camera."""

from __future__ import annotations

import errno
import json
from urllib.error import URLError
from urllib.request import Request, urlopen


def release_preview_camera() -> bool:
    """Stop only the UI preview; never send hardware commands or kill owners.

    An absent UI is normal. A reachable UI that cannot confirm release is
    an error, so acquisition fails before measuring or enabling the robot.
    The preview stays off after the policy exits.
    """
    base = "http://127.0.0.1:8420/camera"

    def request(path: str, method: str, timeout: float) -> dict:
        with urlopen(Request(base + path, method=method), timeout=timeout) as response:
            status = json.load(response)
        if not isinstance(status, dict) or type(status.get("running")) is not bool:
            raise ValueError("invalid camera status response")
        return status

    try:
        status = request("/status", "GET", 2.0)
    except URLError as exc:
        if getattr(exc.reason, "errno", None) == errno.ECONNREFUSED:
            return False
        raise RuntimeError(f"could not check web preview camera ownership: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not check web preview camera ownership: {exc}") from exc

    if not status["running"]:
        return False
    try:
        status = request("/stop", "POST", 5.0)
    except (URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"could not release web preview camera: {exc}") from exc
    if status["running"]:
        raise RuntimeError("web preview still owns the camera after stopping it")
    return True
