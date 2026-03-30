"""Client for the CDPR hardware server (sw/build/cdpr_server)."""

from __future__ import annotations

import socket


class CDPRClient:
    """Connects to the CDPR TCP server and sends position commands.

    The server runs as a separate C++ process that manages the motor hardware.
    Protocol is simple line-based text over TCP on localhost.
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
        # Read response (single line).
        data = b""
        while b"\n" not in data:
            chunk = self._sock.recv(1024)
            if not chunk:
                raise ConnectionError("Server closed connection")
            data += chunk
        return data.decode().strip()

    def move_to(self, x_mm: float, y_mm: float, speed_mm_s: float) -> tuple[float, float]:
        """Blocking move to absolute position. Waits for completion. Returns (x, y)."""
        resp = self._send(f"MOVE {x_mm:.2f} {y_mm:.2f} {speed_mm_s:.1f}")
        if resp.startswith("OK"):
            parts = resp.split()
            return float(parts[1]), float(parts[2])
        raise RuntimeError(f"CDPR move failed: {resp}")

    def command_position(self, x_mm: float, y_mm: float, speed_mm_s: float) -> tuple[float, float]:
        """Non-blocking position command for real-time streaming. Returns (x, y)."""
        resp = self._send(f"CMD {x_mm:.2f} {y_mm:.2f} {speed_mm_s:.1f}")
        if resp.startswith("OK"):
            parts = resp.split()
            return float(parts[1]), float(parts[2])
        raise RuntimeError(f"CDPR cmd failed: {resp}")

    def get_position(self) -> tuple[float, float]:
        """Get current cart position."""
        resp = self._send("POS")
        if resp.startswith("OK"):
            parts = resp.split()
            return float(parts[1]), float(parts[2])
        raise RuntimeError(f"CDPR pos failed: {resp}")

    def set_position(self, x_mm: float, y_mm: float) -> None:
        """Set the internal position state (no movement)."""
        resp = self._send(f"SETPOS {x_mm:.2f} {y_mm:.2f}")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR setpos failed: {resp}")

    def enable(self) -> None:
        resp = self._send("ENABLE")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR enable failed: {resp}")

    def disable(self) -> None:
        resp = self._send("DISABLE")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR disable failed: {resp}")

    def retract_all(self, mm: float, speed_mm_s: float) -> None:
        resp = self._send(f"RETRACT {mm:.2f} {speed_mm_s:.1f}")
        if not resp.startswith("OK"):
            raise RuntimeError(f"CDPR retract failed: {resp}")
