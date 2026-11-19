"""The web UI's robot actions (airhockey/robot_session.py), with no robot:
a fake master client and a stand-in child process for the launcher."""
import json
import sys
import threading
import time

import pytest

from airhockey import robot_session
from airhockey.robot_session import RobotSession


class FakeClient:
    def __init__(self, log, takeup=None):
        self.log, self.takeup, self.release = log, takeup, threading.Event()

    def connect(self):
        self.log.append("connect")

    def take_up_slack(self):
        self.log.append("takeup")
        if self.takeup == "block":
            self.release.wait(5)
            raise ConnectionError("Server closed connection")
        if isinstance(self.takeup, Exception):
            raise self.takeup
        return {"wound_mm": [1.0, 2.0, 3.0, 4.0], "amps": [-0.4, 0.4, -0.4, 0.4]}

    def set_startup_tension(self, mm):
        self.log.append(("tension", mm))

    def disable(self):
        self.log.append("disable")

    def abort(self):
        self.log.append("abort")
        self.release.set()

    def close(self):
        self.log.append("close")


class FakeMaster:
    """Stands in for the cdpr_master process: 'listening' once launched."""

    def __init__(self, log, comes_up=True):
        self.log, self.comes_up, self.listening, self.pid = log, comes_up, False, 4_000_000

    def launch(self, root):
        self.log.append("launch master")
        self.listening = self.comes_up
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs" / "cdpr_master.log").write_text("=== CDPR Master ===\nERROR: No Teensy found (no /dev/ttyACM* device)\n")
        return self

    def poll(self):
        return None if self.comes_up else 1

    returncode = 1

    def send_signal(self, sig):
        self.log.append("signal master")


def session(tmp_path, log, master=True, takeup=None, fake_master=None):
    fake_master = fake_master or FakeMaster(log)
    s = RobotSession(root=tmp_path, client_factory=lambda: FakeClient(log, takeup),
                     master_running=lambda: master or fake_master.listening,
                     master_launcher=fake_master.launch)
    return s


def wait_for(predicate, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_takeup_reports_what_each_cable_wound_in(tmp_path):
    log = []
    s = session(tmp_path, log)
    s.start_takeup()
    assert wait_for(lambda: s.status()["takeup"]["state"] == "done")
    assert s.status()["takeup"]["result"]["wound_mm"] == [1.0, 2.0, 3.0, 4.0]
    assert log == ["connect", "takeup", "close"]


def test_takeup_starts_the_motor_controller_itself(tmp_path):
    """Nobody runs cdpr_master by hand: a button that needs it launches it."""
    log = []
    s = session(tmp_path, log, master=False)
    assert s.status()["master"] is False
    s.start_takeup()
    assert wait_for(lambda: s.status()["takeup"]["state"] == "done")
    assert log == ["launch master", "connect", "takeup", "close"]
    assert s.status()["master"] is True
    # Already running: used as it is, never launched twice.
    log.clear()
    s.start_takeup()
    assert wait_for(lambda: s.status()["takeup"]["state"] == "done")
    assert log == ["connect", "takeup", "close"]
    assert s.ensure_master() is False


def test_a_controller_that_will_not_start_says_why(tmp_path):
    log = []
    s = session(tmp_path, log, master=False, fake_master=FakeMaster(log, comes_up=False))
    s.start_takeup()
    assert wait_for(lambda: s.status()["takeup"]["state"] == "error")
    message = s.status()["takeup"]["message"]
    assert "did not start (exit code 1)" in message and "No Teensy found" in message
    assert "connect" not in log and "takeup" not in log


def test_takeup_needs_a_free_robot(tmp_path):
    log = []
    s = session(tmp_path, log)
    s.hardware_ui = 1
    with pytest.raises(RuntimeError, match="Manual control is on"):
        s.start_takeup()
    with pytest.raises(RuntimeError, match="Manual control is on"):
        s.stop_master()
    assert log == []


def test_only_a_controller_started_here_is_shut_down_from_here(tmp_path):
    import subprocess
    log = []
    s = session(tmp_path, log, master=True)
    assert s.status()["master_owned"] is False
    with pytest.raises(RuntimeError, match="not started from this page"):
        s.stop_master()
    # A real child whose command line says cdpr_master, stopped by SIGINT.
    child = ("import signal, sys, time\n"
             "signal.signal(signal.SIGINT, lambda *a: sys.exit(0))\n"
             "time.sleep(30)  # cdpr_master\n")
    listening = []

    def launch(root):
        proc = subprocess.Popen([sys.executable, "-c", child], start_new_session=True)
        listening.append(proc)
        return proc
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "cdpr_master.log").write_text("Listening on localhost:8421\n")
    s = RobotSession(root=tmp_path, client_factory=lambda: FakeClient(log),
                     master_running=lambda: bool(listening) and listening[0].poll() is None,
                     master_launcher=launch)
    assert s.ensure_master() is True and s.status()["master_owned"] is True
    # A restarted web server still knows it is its own.
    assert RobotSession(root=tmp_path, master_running=lambda: True).status()["master_owned"] is True
    time.sleep(0.3)             # let the child install its handler
    s.stop_master()
    assert listening[0].poll() == 0 and s.status()["master"] is False and s.status()["master_owned"] is False
    assert list((tmp_path / "logs").glob("cdpr_master-*.log"))


def test_takeup_error_and_stop_are_shown_not_raised(tmp_path):
    log = []
    s = session(tmp_path, log, takeup=RuntimeError("slack take-up failed: ERR takeup: motor 1 wound in 150 mm"))
    s.start_takeup()
    assert wait_for(lambda: s.status()["takeup"]["state"] == "error")
    assert "motor 1 wound in 150 mm" in s.status()["takeup"]["message"]
    # Stop closes the socket, which is the master's signal to disable.
    log.clear()
    s = session(tmp_path, log, takeup="block")
    s.start_takeup()
    assert wait_for(lambda: "takeup" in log)
    with pytest.raises(RuntimeError, match="slack take-up is running"):
        s.start_ai("max", 1.5)
    s.stop_takeup()
    assert wait_for(lambda: s.status()["takeup"]["state"] == "stopped")
    assert "abort" in log


CHILD = ("import signal, sys, time\n"
         "signal.signal(signal.SIGINT, lambda *a: (print('braking', flush=True), sys.exit(0)))\n"
         "print('ENABLING the drives', flush=True)\n"
         "time.sleep(30)\n")


def test_ai_runs_the_launcher_and_stops_it_with_sigint(tmp_path, monkeypatch):
    log = []
    s = session(tmp_path, log, master=False)
    commands = []

    def command(difficulty, tension_mm, running):
        commands.append(RobotSession._command(s, difficulty, tension_mm, running))
        return [sys.executable, "-c", CHILD]
    monkeypatch.setattr(s, "_command", command)
    s._marker = "ENABLING"      # what the stand-in child's command line contains
    with pytest.raises(ValueError, match="Unknown difficulty"):
        s.start_ai("impossible", 1.5)
    with pytest.raises(ValueError, match="between 0 and 6"):
        s.start_ai("easy", 9)
    s.start_ai("easy", 2.0)
    # No master running: the launcher starts its own and nothing is sent to one.
    assert commands == [["bash", "ai/bin/play.sh", "--difficulty", "easy", "--tension", "2"]] and log == []
    status = s.status()
    assert status["ai"]["state"] == "running" and status["ai"]["difficulty"] == "easy"
    assert status["ai"]["model"] == robot_session.models.difficulty("easy")[0] and status["ai"]["own_master"]
    assert wait_for(lambda: "ENABLING the drives" in s.status()["log"])
    with pytest.raises(RuntimeError, match="the AI is running"):
        s.start_ai("easy", 2.0)
    with pytest.raises(RuntimeError, match="the AI is running"):
        s.start_takeup()
    # A restarted web server finds the same session from the state file.
    adopted = RobotSession(root=tmp_path, master_running=lambda: False)
    adopted._marker = "ENABLING"
    assert adopted.status()["ai"]["state"] == "running" and adopted.status()["ai"]["difficulty"] == "easy"
    adopted.stop_ai()
    assert wait_for(lambda: s.status()["ai"]["state"] == "exited")
    assert s.status()["ai"]["exit_code"] == 0 and "braking" in s.status()["log"]
    assert adopted.status()["ai"]["state"] == "exited"


def test_ai_on_a_running_master_sets_its_pretension_and_keeps_it(tmp_path, monkeypatch):
    log = []
    s = session(tmp_path, log, master=True)
    seen = []
    monkeypatch.setattr(s, "_command", lambda d, t, running: seen.append(
        RobotSession._command(s, d, t, running)) or [sys.executable, "-c", "pass"])
    s.start_ai("max", 1.5)
    assert seen == [["bash", "ai/bin/play.sh", "--difficulty", "max", "--tension", "1.5", "--use-running-master"]]
    assert log[:3] == ["connect", ("tension", 1.5), "close"]
    assert json.loads((tmp_path / "logs" / "ai_session.json").read_text())["own_master"] is False
    assert wait_for(lambda: s.status()["ai"]["state"] == "exited")
    # The runner leaves a master it did not start with the drives holding;
    # the session ending from the UI must de-energize them.
    assert wait_for(lambda: log[3:] == ["connect", "disable", "close"])
    assert "drives disabled" in (tmp_path / "logs" / "ai_session.log").read_text()


def test_modes_come_from_the_registry():
    modes = robot_session.difficulty_modes()
    assert [m["mode"] for m in modes][0] == "max" and all(5 < m["accel"] <= m["trained_accel"] for m in modes)


def test_master_probe_reads_the_socket_table_without_connecting():
    import socket
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        assert not robot_session.master_listening(port)
        server.listen(1)
        server.settimeout(0.2)
        assert robot_session.master_listening(port)
        with pytest.raises(socket.timeout):      # nobody connected to find that out
            server.accept()


def test_server_routes_need_the_dialogs_confirmation_and_report_refusals(tmp_path, monkeypatch):
    """Nothing that moves the robot starts from a bare POST."""
    import asyncio
    from airhockey import server
    log = []
    fake = session(tmp_path, log, master=False)
    started = []
    monkeypatch.setattr(fake, "start_ai", lambda *a: started.append(a))
    monkeypatch.setattr(server, "ROBOT", fake)
    for body in ({}, {"confirm": "true"}, {"confirm": False}):
        assert asyncio.run(server.robot_takeup(body)) == {"ok": False, "error": "not confirmed"}
        assert asyncio.run(server.robot_ai_start(dict(body, difficulty="max"))) == {"ok": False, "error": "not confirmed"}
    assert started == [] and log == []
    refused = asyncio.run(server.robot_master_stop())
    assert refused["ok"] is False and "not started from this page" in refused["error"]
    assert asyncio.run(server.robot_ai_start({"confirm": True}))["error"] == "difficulty and tension_mm are required"
    reply = asyncio.run(server.robot_ai_start({"confirm": True, "difficulty": "easy", "tension_mm": 2}))
    assert reply["ok"] is True and started == [("easy", 2.0)]
    status = asyncio.run(server.robot_status())
    assert status["ai"]["state"] == "idle" and [m["mode"] for m in status["modes"]][0] == "max"
    # A browser tab holding the hardware blocks both, with the reason.
    fake.hardware_ui = 1
    monkeypatch.undo()
    monkeypatch.setattr(server, "ROBOT", fake)
    blocked = asyncio.run(server.robot_ai_start({"confirm": True, "difficulty": "easy", "tension_mm": 2}))
    assert blocked["ok"] is False and "Manual control is on" in blocked["error"]
