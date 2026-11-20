"""What the web UI's Robot tab can start on the table: slack take-up and the AI.

CDPR_MASTER    The bridge to the drives and the Teensy. Nobody has to start
               it by hand: take-up and manual control start it when it is not
               running (``ensure_master``: build if stale, launch detached,
               wait for its port), and it then stays up, idle and with the
               drives de-energized, until "Stop cdpr_master" or a reboot. It
               holds the SC-Hub USB port, so ``activate`` and ``drive_gains``
               need it stopped.
SLACK TAKE-UP  cdpr_master's ``TAKEUP``: the drives energize, each cable is
               wound in one motor step (0.38 mm) at a time until its own
               drive's current says the cable pulls back, and the drives
               de-energize.
THE AI         ``ai/bin/play.sh`` as a child process, exactly the command a
               terminal would run: it starts cdpr_master if none is running
               (``--use-running-master`` otherwise), and the runner it launches
               takes up slack, measures the mallet with the camera, enables and
               plays. Stop is the same SIGINT that Ctrl-C sends: the runner
               brakes and exits, then the launcher stops a master it started.
               On a master that was already running the runner leaves the
               drives holding, so this module sends DISABLE when the session
               ends: the AI stopping always de-energizes the drives.

One at a time, and never while a browser tab holds the hardware for manual
control (the master serves one client). Nothing here runs unless a person
presses the button: the server only calls ``start_*`` from a POST.

The AI's output goes to ``logs/ai_session.log`` and its pid to
``logs/ai_session.json`` (the master's to ``logs/ui_master.json``), so a
restarted web server still finds, shows and can stop what it did not start.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
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
                        latest=version == data["latest"],
                        note=data["difficulty"][mode].get("note")))
    return out


def _alive(pid: int, marker: str = "play.sh") -> bool:
    """The recorded pid is still the launcher (not a recycled pid)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return marker in cmdline


def _refuse_in_tests(what: str) -> None:
    if "pytest" in sys.modules:
        raise RuntimeError(f"refusing to {what} from a test run: inject a fake")


def realtime_available() -> bool:
    """Whether this process may start programs at realtime priority.

    play.sh (the policy loop and the camera tracker) and the motor controller
    launch below ask for SCHED_FIFO 40 and quietly run at normal priority when
    it is refused. The permission is a limit the SERVER inherited when it was
    started, so it depends on how that was done: a login made after
    /etc/security/limits.d/99-airhockey.conf was written has it, a server
    started from an older shell or from a tool's shell does not
    (ai/bin/restart_server.sh starts it through a fresh login). Without it
    another program's batch job made 2-9% of policy ticks more than 30 ms late
    for minutes at a time (2026-11-19) and the robot conceded five times as
    often in those minutes.
    """
    if not shutil.which("chrt"):
        return False
    try:
        return subprocess.run(["chrt", "-f", "40", "true"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _launch_master(root: Path) -> subprocess.Popen:
    """Build cdpr_master if its sources changed, then start it detached.

    The same two steps as ai/bin/play.sh: `make -C sw build/cdpr_master`, and
    realtime priority when the account may ask for it. Launching energizes
    nothing; the drives stay off until ENABLE or TAKEUP.

    Refuses inside a test run. On 2026-11-18 a server test that turned the
    UI's hardware mode on with a fake client reached this function and
    started the real master (idle, never enabled, stopped two minutes later).
    A test must inject its own launcher.
    """
    _refuse_in_tests("start cdpr_master")
    build = subprocess.run(["make", "-C", "sw", "build/cdpr_master"], cwd=root, capture_output=True,
                           text=True, timeout=600)
    if build.returncode != 0:
        tail = "\n".join((build.stdout + build.stderr).strip().splitlines()[-6:])
        raise RuntimeError(f"The motor controller (cdpr_master) did not build:\n{tail}")
    command = [str(root / "sw" / "build" / "cdpr_master")]
    if realtime_available():
        command = ["chrt", "-f", "40", *command]
    # It writes logs/cdpr_master.log itself; stdout would only duplicate it.
    return subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)


class RobotSession:
    MASTER_START_TIMEOUT_S = 25.0

    def __init__(self, root: Path = ROOT, client_factory=None, master_running=master_listening,
                 master_launcher=_launch_master, realtime=realtime_available):
        self.root = Path(root)
        # Asked once: the limit cannot change while this process lives.
        self._realtime_check, self._realtime = realtime, None
        self.log_path = self.root / "logs" / "ai_session.log"
        self.state_path = self.root / "logs" / "ai_session.json"
        self.master_state_path = self.root / "logs" / "ui_master.json"
        self._client_factory = client_factory
        self._master_running = master_running
        self._master_launcher = master_launcher
        self._master_lock = threading.Lock()
        self._master_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # Browser tabs currently holding the master for manual control.
        self.hardware_ui = 0
        self._takeup = dict(state="idle")
        self._takeup_client = None
        self._proc: subprocess.Popen | None = None
        self._ai = self._load_state()
        self._marker = "play.sh"
        if self.ai_running():        # adopted from before a web-server restart
            self._watch_ai()

    # ── cdpr_master ──────────────────────────────────────────────────────

    def _master_pid(self) -> int | None:
        """pid of a master THIS UI started and that is still running."""
        if self._master_proc is not None:
            return self._master_proc.pid if self._master_proc.poll() is None else None
        try:
            pid = json.loads(self.master_state_path.read_text()).get("pid")
        except (OSError, ValueError, AttributeError):
            return None
        return pid if isinstance(pid, int) and _alive(pid, "cdpr_master") else None

    def _master_realtime(self) -> bool | None:
        """Whether the controller this UI started runs at realtime priority
        (None: no such controller). One started by a server without the
        permission keeps normal priority until it is shut down and started
        again."""
        pid = self._master_pid()
        if pid is None:
            return None
        try:
            return os.sched_getscheduler(pid) in (os.SCHED_FIFO, os.SCHED_RR)
        except OSError:
            return None

    def ensure_master(self) -> bool:
        """Start cdpr_master unless one is listening. True if this started it.

        Blocks for the build check and until the port answers (seconds), so
        call it off the event loop. Raises RuntimeError with the master's own
        last log lines when it exits instead of listening -- usually no Teensy
        on USB, or the drives' supply is off.
        """
        with self._master_lock:
            if self._master_running():
                return False
            proc = self._master_launcher(self.root)
            self._master_proc = proc
            try:
                self.master_state_path.parent.mkdir(parents=True, exist_ok=True)
                self.master_state_path.write_text(json.dumps(dict(pid=proc.pid, started=time.time())))
            except OSError:
                pass
            deadline = time.time() + self.MASTER_START_TIMEOUT_S
            while time.time() < deadline:
                if self._master_running():
                    return True
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            raise RuntimeError("The motor controller (cdpr_master) did not start"
                               + (f" (exit code {proc.returncode})" if proc.poll() is not None else " in time")
                               + ". Are the motors powered and the Teensy plugged in? Its log ends:\n"
                               + self._master_log_tail())

    def _master_log_tail(self, lines: int = 6) -> str:
        try:
            text = (self.root / "logs" / "cdpr_master.log").read_text(errors="replace")
        except OSError:
            return "(no logs/cdpr_master.log)"
        return "\n".join([line.rstrip() for line in text.splitlines() if line.strip()][-lines:])

    def stop_master(self) -> None:
        """Stop a master this UI started, once nothing is using it. Its clean
        exit de-energizes the drives; a copy of its log is kept, as play.sh does."""
        with self._lock:
            self._refuse_if_busy("stop cdpr_master")
        with self._master_lock:
            pid = self._master_pid()
            if pid is None:
                raise RuntimeError("The motor controller was not started from this page; "
                                   "stop it where it was started.")
            for _ in range(2):          # the second SIGINT is the master's own forced exit
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    break
                deadline = time.time() + 6.0
                while time.time() < deadline and _alive(pid, "cdpr_master"):
                    if self._master_proc is not None:
                        self._master_proc.poll()
                    time.sleep(0.1)
                if not _alive(pid, "cdpr_master"):
                    break
            self._master_proc = None
            log = self.root / "logs" / "cdpr_master.log"
            try:
                self.master_state_path.unlink(missing_ok=True)
                if log.is_file():
                    shutil.copy(log, log.with_name(f"cdpr_master-{time.strftime('%Y%m%d-%H%M%S')}.log"))
            except OSError:
                pass

    # ── slack take-up ────────────────────────────────────────────────────

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from airhockey.hardware import CDPRClient   # noqa: PLC0415
        return CDPRClient()

    def start_takeup(self) -> None:
        with self._lock:
            self._refuse_if_busy("take up slack")
            self._takeup = dict(state="running", started=time.time(),
                                phase="taking up slack" if self._master_running() else "starting cdpr_master")
            client = self._takeup_client = self._client()
        threading.Thread(target=self._run_takeup, args=(client,), daemon=True).start()

    def _run_takeup(self, client) -> None:
        stopped = False
        try:
            self.ensure_master()
            with self._lock:
                if self._takeup.get("stopping"):
                    raise RuntimeError("stopped before it began")
                self._takeup["phase"] = "taking up slack"
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
        # Stop AI: brake, then return slowly to the centre of the robot's
        # half before the drives go off (run_policy --park home), rather than
        # leave the mallet wherever the rally ended.
        cmd += ["--park", "home"]
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
            if command[:2] == ["bash", "ai/bin/play.sh"]:
                _refuse_in_tests("run play.sh")
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
        self._watch_ai()

    def _watch_ai(self) -> None:
        """When the session ends on a master it did not start, de-energize.

        play.sh stops a master it started, which turns the drives off. On a
        master that was already running the runner brakes and disconnects with
        the drives still holding (by design, for a terminal user who goes on
        working); from a Stop button that would leave the motors energized
        with nobody at the controls.
        """
        def watch():
            while self.ai_running():
                time.sleep(0.25)
            with self._lock:
                needed = not self._ai.get("own_master") and not self._ai.get("disabled")
                self._ai["disabled"] = True
            if needed and self._master_running():
                client = self._client()
                try:
                    client.connect()
                    client.disable()
                    note = "drives disabled"
                except Exception as e:      # noqa: BLE001
                    note = f"COULD NOT DISABLE THE DRIVES ({e}); check cdpr_master"
                finally:
                    try:
                        client.close()
                    except Exception:       # noqa: BLE001
                        pass
                try:
                    with open(self.log_path, "a") as log:
                        log.write(f"[web UI] session ended on a running cdpr_master: {note}\n")
                except OSError:
                    pass
        threading.Thread(target=watch, daemon=True).start()

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
        if self._realtime is None:
            self._realtime = bool(self._realtime_check())
        return dict(master=self._master_running(), master_owned=self._master_pid() is not None,
                    realtime=self._realtime, master_realtime=self._master_realtime(),
                    hardware_ui=hardware_ui, takeup=takeup, ai=ai,
                    log=self._log_tail() if ai["state"] != "idle" else [],
                    log_path=str(self.log_path.relative_to(self.root)))


SESSION = RobotSession()
