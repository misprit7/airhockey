"""What the web UI's Robot tab can start on the table: slack take-up and the AI.

SLACK TAKE-UP  cdpr_master's ``TAKEUP``: the drives energize, each cable is
               wound in one motor step (0.38 mm) at a time until its own drive's current says
               the cable pulls back, and the drives de-energize. Needs a
               running master and nobody else holding it.
THE AI         ``ai/bin/play.sh`` as a child process, exactly the command a
               terminal would run: it starts cdpr_master if none is running
               (``--use-running-master`` otherwise), and the runner it launches
               takes up slack, measures the mallet with the camera, enables and
               plays. Stop is the same SIGINT that Ctrl-C sends: the runner
               brakes and exits, then the launcher stops a master it started.

One at a time, and never while a browser tab holds the hardware for manual
control (the master serves one client). Nothing here runs unless a person
presses the button: the server only calls ``start_*`` from a POST.

The AI's output goes to ``logs/ai_session.log`` and its pid to
``logs/ai_session.json``, so a restarted web server still finds, shows and can
stop a session it did not start.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

from airhockey import models

ROOT = Path(__file__).resolve().parents[2]
MASTER_PORT = 8421
LOG_TAIL_LINES = 60


def master_listening(port: int = MASTER_PORT) -> bool:
    """Whether something listens on the master's port, WITHOUT connecting.

    The master serves one client: a probe connection is accepted as that
    client when the slot is free (and a real client arriving in the same
    moment is turned away as busy), and logged as a rejected second client
    when it is not. The UI asks once a second, so read the kernel's table.
    """
    try:
        rows = Path("/proc/net/tcp").read_text().splitlines()[1:]
    except OSError:
        with socket.socket() as s:       # not Linux: fall back to a probe
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0
    suffix = f":{port:04X}"
    for row in rows:
        fields = row.split()
        if len(fields) > 3 and fields[1].endswith(suffix) and fields[3] == "0A":
            return True
    return False


def difficulty_modes() -> list[dict]:
    """The modes of ai/models.json, in file order, for the UI's selector."""
    data = models.registry()
    out = []
    for mode in data["difficulty"]:
        version, accel = models.difficulty(mode, data)
        trained = float(data["models"][version]["accel"])
        out.append(dict(mode=mode, model=version, accel=accel, trained_accel=trained,
                        latest=version == data["latest"]))
    return out


def _alive(pid: int, marker: str = "play.sh") -> bool:
    """The recorded pid is still the launcher (not a recycled pid)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return marker in cmdline


class RobotSession:
    def __init__(self, root: Path = ROOT, client_factory=None, master_running=master_listening):
        self.root = Path(root)
        self.log_path = self.root / "logs" / "ai_session.log"
        self.state_path = self.root / "logs" / "ai_session.json"
        self._client_factory = client_factory
        self._master_running = master_running
        self._lock = threading.Lock()
        # Browser tabs currently holding the master for manual control.
        self.hardware_ui = 0
        self._takeup = dict(state="idle")
        self._takeup_client = None
        self._proc: subprocess.Popen | None = None
        self._ai = self._load_state()
        self._marker = "play.sh"

    # ── slack take-up ────────────────────────────────────────────────────

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from airhockey.hardware import CDPRClient   # noqa: PLC0415
        return CDPRClient()

    def start_takeup(self) -> None:
        with self._lock:
            self._refuse_if_busy("take up slack")
            if not self._master_running():
                raise RuntimeError("cdpr_master is not running. Start it (sw/build/cdpr_master), "
                                   "or press Start AI, which starts it and takes up slack itself.")
            self._takeup = dict(state="running", started=time.time())
            client = self._takeup_client = self._client()
        threading.Thread(target=self._run_takeup, args=(client,), daemon=True).start()

    def _run_takeup(self, client) -> None:
        stopped = False
        try:
            client.connect()
            result = client.take_up_slack()
            outcome = dict(state="done", result=result)
        except Exception as e:      # noqa: BLE001 — every failure is shown, none is fatal here
            with self._lock:
                stopped = self._takeup.get("stopping", False)
            outcome = (dict(state="stopped", message="Stopped. The drives are disabled.") if stopped
                       else dict(state="error", message=str(e)))
        finally:
            try:
                client.close()
            except Exception:       # noqa: BLE001
                pass
        with self._lock:
            self._takeup = dict(outcome, started=self._takeup.get("started"), finished=time.time())
            self._takeup_client = None

    def stop_takeup(self) -> None:
        """Closing the socket is the master's stop signal: it disables the drives."""
        with self._lock:
            client = self._takeup_client
            if self._takeup.get("state") != "running" or client is None:
                return
            self._takeup["stopping"] = True
        client.abort()

    # ── the AI ───────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
            return state if isinstance(state, dict) and isinstance(state.get("pid"), int) else {}
        except (OSError, ValueError):
            return {}

    def _command(self, difficulty: str, tension_mm: float, master_running: bool) -> list[str]:
        cmd = ["bash", "ai/bin/play.sh", "--difficulty", difficulty, "--tension", f"{tension_mm:g}"]
        if master_running:
            cmd.append("--use-running-master")
        return cmd

    def ai_running(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        return bool(self._ai.get("pid")) and _alive(self._ai["pid"], self._marker)

    def _refuse_if_busy(self, what: str) -> None:
        if self.ai_running():
            raise RuntimeError(f"Cannot {what}: the AI is running. Stop it first.")
        if self._takeup.get("state") == "running":
            raise RuntimeError(f"Cannot {what}: slack take-up is running.")
        if self.hardware_ui > 0:
            raise RuntimeError(f"Cannot {what}: Manual control is on in a browser tab and holds the "
                               "robot. Turn it off first.")

    def start_ai(self, difficulty: str, tension_mm: float) -> None:
        tension_mm = float(tension_mm)
        if not 0 <= tension_mm <= 6:
            raise ValueError("Startup pretension must be between 0 and 6 mm")
        modes = {m["mode"]: m for m in difficulty_modes()}
        if difficulty not in modes:
            raise ValueError(f"Unknown difficulty {difficulty!r}; one of {', '.join(modes)}")
        with self._lock:
            self._refuse_if_busy("start the AI")
            running = self._master_running()
            if running:
                # A master someone else started keeps its own --tension; tell
                # it the one asked for here. Configuration only, no motion.
                client = self._client()
                try:
                    client.connect()
                    client.set_startup_tension(tension_mm)
                finally:
                    client.close()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            command = self._command(difficulty, tension_mm, running)
            with open(self.log_path, "w") as log:
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                self._proc = subprocess.Popen(
                    command, cwd=self.root, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            self._ai = dict(pid=self._proc.pid, started=time.time(), difficulty=difficulty,
                            model=modes[difficulty]["model"], accel=modes[difficulty]["accel"],
                            tension_mm=tension_mm, command=command, own_master=not running)
            self.state_path.write_text(json.dumps(self._ai))

    def stop_ai(self) -> None:
        """What Ctrl-C does in a terminal: SIGINT to the launcher's process
        group. The runner brakes the mallet and exits; the launcher then stops
        a master it started (which de-energizes the drives)."""
        with self._lock:
            if not self.ai_running():
                return
            self._ai["stopping"] = time.time()
            pid = self._ai["pid"]
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def _log_tail(self) -> list[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16384))
                text = f.read().decode(errors="replace")
        except OSError:
            return []
        # Progress lines rewrite themselves with \r; show their last state.
        lines = [line.rsplit("\r", 1)[-1].rstrip() for line in text.splitlines()]
        return [line for line in lines if line][-LOG_TAIL_LINES:]

    def status(self) -> dict:
        with self._lock:
            running = self.ai_running()
            ai = {k: self._ai.get(k) for k in ("started", "difficulty", "model", "accel", "tension_mm", "own_master")}
            if running:
                ai["state"] = "stopping" if self._ai.get("stopping") else "running"
            elif self._ai.get("pid"):
                ai["state"] = "exited"
                ai["exit_code"] = self._proc.returncode if self._proc is not None else None
            else:
                ai["state"] = "idle"
            takeup = {k: v for k, v in self._takeup.items() if k != "stopping"}
            hardware_ui = self.hardware_ui
        return dict(master=self._master_running(), hardware_ui=hardware_ui, takeup=takeup, ai=ai,
                    log=self._log_tail() if ai["state"] != "idle" else [],
                    log_path=str(self.log_path.relative_to(self.root)))


SESSION = RobotSession()
