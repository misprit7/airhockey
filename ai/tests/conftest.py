"""Suite-wide interlocks. No test may start the real motor controller or the
real launcher: the web server's module-level robot session is given launchers
that fail loudly (airhockey/robot_session.py also refuses on its own when
pytest is loaded; this names the offending test)."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_robot_processes(monkeypatch, request):
    try:
        from airhockey import robot_session
    except Exception:       # noqa: BLE001 — tests that never import the server
        return

    def forbidden(*args, **kwargs):
        pytest.fail(f"{request.node.nodeid} tried to start cdpr_master")
    monkeypatch.setattr(robot_session.SESSION, "_master_launcher", forbidden)
    monkeypatch.setattr(robot_session, "_launch_master", forbidden)
