"""Client for the CDPR hardware master (sw/build/cdpr_master).

The master bridges sFoundation motors and the Teensy motion controller,
exposing a TCP interface for position commands and live status.
"""

from __future__ import annotations

import socket


class CDPRClient:
    """Connects to the CDPR master TCP server and sends position commands.

    The master runs as a separate C++ process that manages motor hardware
    (via sFoundation) and the Teensy motion controller (via USB serial).
    Protocol is simple line-based text over TCP on localhost:8421.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8421):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((self.host, self.port))

    def close(self) -> None:
        if self._sock:
            try:
                self._send("QUIT")
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def _send(self, cmd: str) -> str:
        """Send a command and return the response line."""
        self._sock.sendall((cmd + "\n").encode())
        data = b""
        while b"\n" not in data:
            chunk = self._sock.recv(1024)
            if not chunk:
                raise ConnectionError("Server closed connection")
            data += chunk
        return data.decode().strip()

    def enable(self) -> None:
        """Enable motors and start Teensy motion controller.

        This enables sFoundation motors, applies tensioning, calibrates
        position at center, and sends CAL/TENSION/START to Teensy.
        """
        resp = self._send("ENABLE")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR enable failed: {resp}")

    def disable(self) -> None:
        """Stop Teensy motion controller and disable motors."""
        resp = self._send("DISABLE")
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

    def get_status(self) -> dict:
        """Get full status including motor step counts.

        Returns dict with keys: x, y, vx, vy, c0, c1, c2, c3.
        """
        resp = self._send("STATUS")
        if resp.startswith("OK"):
            parts = resp.split()
            return {
                "x": float(parts[1]),
                "y": float(parts[2]),
                "vx": float(parts[3]),
                "vy": float(parts[4]),
                "c0": int(parts[5]),
                "c1": int(parts[6]),
                "c2": int(parts[7]),
                "c3": int(parts[8]),
            }
        raise RuntimeError(f"CDPR status failed: {resp}")
