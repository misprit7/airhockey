"""Client for the CDPR hardware master (sw/build/cdpr_master).

The master bridges sFoundation motors and the Teensy motion controller,
exposing a TCP interface for position commands and live status.
"""

from __future__ import annotations

import math
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as _geom  # noqa: E402

# Step counts arrive over the wire from the Teensy, so anything decoding
# them needs its step resolution. This mirrors COUNTS_PER_REV in
# fw/include/cdpr_config.h and must be changed with it — it lives here
# rather than in shared/cdpr_geometry.h because it is a property of how the
# Teensy is configured to drive the steppers, not of the machine, and here
# is where the counts are parsed.
TEENSY_COUNTS_PER_REV = 800


def counts_to_cable_mm(counts: float) -> float:
    """Teensy step counts -> millimetres of cable paid out."""
    return counts * (2.0 * math.pi * _geom.SPOOL_RADIUS_MM) \
        / TEENSY_COUNTS_PER_REV


class CDPRClient:
    """Connects to the CDPR master TCP server and sends position commands.

    The master runs as a separate C++ process that manages motor hardware
    (via sFoundation) and the Teensy motion controller (via USB serial).
    Protocol is simple line-based text over TCP on localhost:8421.
    """

    # Every round trip gets a deadline. Without one, a master that stops
    # responding — wedged, or merely SIGSTOP'd by a stray Ctrl-Z — parks this
    # recv forever, and because the server calls straight through to here on
    # its asyncio loop, one frozen process takes the whole UI down with it,
    # camera stream included. Callers already treat an exception as "hardware
    # went away"; the timeout is what lets that path ever run.
    RESPONSE_TIMEOUT_S = 2.0
    # ENABLE is the exception: it drives the Teensy through tensioning and
    # START, and the master allows 10 s for those internally.
    ENABLE_TIMEOUT_S = 20.0
    # LIMITS forwards TWO commands to the Teensy and the master waits up to
    # 5 s for each, so it can legitimately take 10 s. The default 2 s was
    # tighter than the thing it was measuring.
    LIMITS_TIMEOUT_S = 12.0

    def __init__(self, host: str = "127.0.0.1", port: int = 8421):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self.RESPONSE_TIMEOUT_S)
        self._sock.connect((self.host, self.port))

    def close(self) -> None:
        if self._sock:
            try:
                self._send("QUIT")
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def _send(self, cmd: str, timeout: float | None = None) -> str:
        """Send a command and return the response line.

        A timeout drops the connection rather than retrying. The protocol is
        one reply per command with nothing to match them up, so a reply that
        arrives after we stopped waiting would be handed back as the answer to
        whatever we asked NEXT — every later reading off by one command, which
        reads as plausible data rather than as a fault. Closing is the only
        way to resynchronise.
        """
        if self._sock is None:
            raise ConnectionError("not connected to cdpr_master")
        try:
            self._sock.settimeout(timeout or self.RESPONSE_TIMEOUT_S)
            self._sock.sendall((cmd + "\n").encode())
            data = b""
            while b"\n" not in data:
                chunk = self._sock.recv(1024)
                if not chunk:
                    raise ConnectionError("Server closed connection")
                data += chunk
        except socket.timeout:
            self._sock.close()
            self._sock = None
            raise ConnectionError(
                f"cdpr_master did not answer {cmd.split()[0]} within "
                f"{timeout or self.RESPONSE_TIMEOUT_S:g}s; connection dropped.\n"
                f"  Most likely: something ELSE already holds the master — it "
                f"serves one client at a time,\n"
                f"  and the web UI in Hardware mode is the usual culprit. "
                f"Check with:  ss -tnp | grep {self.port}\n"
                f"  Less likely: the master is stopped (State: T in "
                f"/proc/<pid>/status) or its Teensy link is down."
            ) from None
        return data.decode().strip()

    def enable(self, cal_x_mm: float | None = None,
               cal_y_mm: float | None = None,
               cal_theta_deg: float | None = None) -> None:
        """Energize the motors and calibrate the Teensy.

        NOT passive: the master follows this with TENSION and START, so the
        cables take up 2mm of slack and the control loop begins running.

        Pass the MEASURED mallet position if you have it — otherwise the
        master assumes the mallet sits at the centre of the robot half, and
        any error there offsets every later command:
            python vision/bin/track_mallet.py
        """
        t = self.ENABLE_TIMEOUT_S
        if cal_x_mm is None or cal_y_mm is None:
            resp = self._send("ENABLE", timeout=t)
        elif cal_theta_deg is None:
            resp = self._send(f"ENABLE {cal_x_mm:.2f} {cal_y_mm:.2f}", timeout=t)
        else:
            resp = self._send(f"ENABLE {cal_x_mm:.2f} {cal_y_mm:.2f} "
                              f"{cal_theta_deg:.2f}", timeout=t)
        if not resp.startswith("OK"):
            raise RuntimeError(f"enable failed: {resp}")

    def disable(self) -> None:
        """Stop Teensy motion controller and disable motors."""
        resp = self._send("DISABLE", timeout=self.ENABLE_TIMEOUT_S)
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR disable failed: {resp}")

    def command_position(self, x_mm: float, y_mm: float, speed_mm_s: float) -> None:
        """Send a non-blocking position command to the Teensy.

        The Teensy handles trajectory planning. Speed is included for
        interface consistency but ignored by the master.
        """
        resp = self._send(f"CMD {x_mm:.2f} {y_mm:.2f} {speed_mm_s:.1f}")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR cmd failed: {resp}")

    def get_position(self) -> tuple[float, float, float, float]:
        """Get current paddle position and velocity from Teensy status.

        Returns (x_mm, y_mm, vx_mm_s, vy_mm_s).
        """
        resp = self._send("POS")
        if resp.startswith("OK"):
            parts = resp.split()
            return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        raise RuntimeError(f"CDPR pos failed: {resp}")

    def set_limits(self, speed_mm_s: float, accel_mm_s2: float) -> None:
        """Set the Teensy's trajectory speed and acceleration caps."""
        resp = self._send(f"LIMITS {speed_mm_s:.2f} {accel_mm_s2:.2f}",
                          timeout=self.LIMITS_TIMEOUT_S)
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR limits failed: {resp}")

    def reset_peaks(self) -> None:
        """Zero the Teensy's peak usage trackers."""
        resp = self._send("RESETPEAK")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR peak reset failed: {resp}")

    def get_encoders(self) -> dict:
        """Read the DRIVES' encoders, not the Teensy's step counts.

        Returns {'posn': [4 counts], 'res': [4 counts/rev], 'trq': [4 pct]}.
        The Teensy's counts say what it asked for; these say what happened.
        Only their difference distinguishes a wrong model from a motor that
        did not follow, so they are kept separate rather than merged.
        """
        resp = self._send("ENC")
        if resp.startswith("OK"):
            p = resp.split()
            return {
                "posn": [float(v) for v in p[1:5]],
                "res": [int(v) for v in p[5:9]],
                "trq": [float(v) for v in p[9:13]],
            }
        raise RuntimeError(f"CDPR enc failed: {resp}")

    def get_status(self) -> dict:
        """Get full status including motor step counts.

        Returns dict with keys: x, y, vx, vy, c0, c1, c2, c3.
        """
        resp = self._send("STATUS")
        if resp.startswith("OK"):
            parts = resp.split()
            out = {
                "x": float(parts[1]),
                "y": float(parts[2]),
                "vx": float(parts[3]),
                "vy": float(parts[4]),
                "c0": int(parts[5]),
                "c1": int(parts[6]),
                "c2": int(parts[7]),
                "c3": int(parts[8]),
            }
            # Optional: older firmware stops at the step counts.
            if len(parts) >= 12:
                out["speed_limit"] = float(parts[9])
                out["accel_limit"] = float(parts[10])
                out["limit_flags"] = int(parts[11])
            if len(parts) >= 16:
                out["speed_frac"] = float(parts[12])
                out["accel_frac"] = float(parts[13])
                out["speed_peak"] = float(parts[14])
                out["accel_peak"] = float(parts[15])
            return out
        raise RuntimeError(f"CDPR status failed: {resp}")
