"""Pluggable motor dynamics models.

These models sit between the RL agent's action (target position) and the
actual paddle position, simulating real-world actuator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MotorDynamics:
    """Base interface for motor dynamics models."""

    def reset(self, x: float, y: float) -> None:
        """Reset to initial position."""
        raise NotImplementedError

    def update(self, target_x: float, target_y: float, dt: float) -> tuple[float, float]:
        """Given a target position, return the actual position after dt seconds."""
        raise NotImplementedError


@dataclass
class IdealDynamics(MotorDynamics):
    """Paddle instantly moves to target. Useful for testing."""

    x: float = 0.0
    y: float = 0.0

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def update(self, target_x: float, target_y: float, dt: float) -> tuple[float, float]:
        self.x = target_x
        self.y = target_y
        return self.x, self.y


@dataclass
class DelayedDynamics(MotorDynamics):
    """First-order low-pass filter dynamics with velocity and acceleration limits.

    Simulates a real motor system where:
    - There's a maximum velocity the paddle can move at
    - There's a maximum acceleration (can't change direction instantly)
    - Response has a time constant (smoothing / lag)
    """

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    max_speed: float = 4.0  # m/s
    max_accel: float = 40.0  # m/s^2
    time_constant: float = 0.02  # seconds (lower = more responsive)

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

    def update(self, target_x: float, target_y: float, dt: float) -> tuple[float, float]:
        # Desired velocity toward target (P controller with time constant)
        dx = target_x - self.x
        dy = target_y - self.y

        desired_vx = dx / max(self.time_constant, dt)
        desired_vy = dy / max(self.time_constant, dt)

        # Clamp desired velocity to max speed
        desired_speed = np.hypot(desired_vx, desired_vy)
        if desired_speed > self.max_speed:
            factor = self.max_speed / desired_speed
            desired_vx *= factor
            desired_vy *= factor

        # Apply acceleration limits
        ax = (desired_vx - self.vx) / dt if dt > 0 else 0.0
        ay = (desired_vy - self.vy) / dt if dt > 0 else 0.0
        accel = np.hypot(ax, ay)
        if accel > self.max_accel:
            factor = self.max_accel / accel
            ax *= factor
            ay *= factor

        self.vx += ax * dt
        self.vy += ay * dt

        # Integrate position
        self.x += self.vx * dt
        self.y += self.vy * dt

        return self.x, self.y


@dataclass
class LearnedDynamics(MotorDynamics):
    """Placeholder for a learned dynamics model.

    This will eventually be a neural network trained on real motor data
    that predicts actual position given commanded position.
    For now it wraps DelayedDynamics with configurable noise.
    """

    inner: DelayedDynamics | None = None
    position_noise_std: float = 0.001  # 1mm noise
    _rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.inner is None:
            self.inner = DelayedDynamics()
        if self._rng is None:
            self._rng = np.random.default_rng()

    def reset(self, x: float, y: float) -> None:
        self.inner.reset(x, y)

    def update(self, target_x: float, target_y: float, dt: float) -> tuple[float, float]:
        x, y = self.inner.update(target_x, target_y, dt)
        # Add noise to simulate imperfect real-world positioning
        x += self._rng.normal(0, self.position_noise_std)
        y += self._rng.normal(0, self.position_noise_std)
        return x, y


class HardwareDynamics(MotorDynamics):
    """Drives the real CDPR through cdpr_master (TCP) -> Teensy (step/dir).

    Coordinate mapping, which is NOT a scale factor:

      sim is 1.0 wide x 2.0 long with the agent owning y in [0, 1.0]; the
      long axis is sim Y. The real table's long axis is grid X. So the axes
      SWAP, and sim y also runs opposite to grid x — sim y=0 is the agent's
      own goal line, which is the robot end of the table (high grid x).

        sim y 0 .. 1   ->  grid x WS_MAX_X .. WS_MIN_X   (robot end -> centre)
        sim x 0 .. 1   ->  grid y WS_MIN_Y .. WS_MAX_Y

    The previous version mapped sim x -> mm x with no swap into a 606x730
    box, which was the prototype's own local frame and is meaningless on
    this table.

    CONFIRM THE SIM-X SIGN ON FIRST USE at low speed: pushing right in the
    UI should move the mallet consistently one way. If it is mirrored, flip
    SIM_X_FLIP below. Getting it wrong is confusing, not dangerous.
    """

    SIM_X_FLIP = False

    def __init__(
        self,
        sim_width: float = 1.0,
        sim_height: float = 2.0,
        speed_mm_s: float = 40.0,
        max_speed_mm_s: float = 120.0,
        host: str = "127.0.0.1",
        port: int = 8421,
        cal_pose_mm: tuple[float, float, float] | None = None,
    ):
        import time as _time
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(
            0, str(_Path(__file__).resolve().parents[2] / "shared"))
        import cdpr_geometry as geom

        from airhockey.hardware import CDPRClient

        self.geom = geom
        self.sim_width = sim_width
        self.sim_half_height = sim_height / 2.0
        # Deliberately conservative: the cable model's winding-side sign is
        # still unverified, so keep commanded motion slow enough to watch.
        self.speed = min(speed_mm_s, max_speed_mm_s)
        self.max_speed = max_speed_mm_s
        self.x = 0.0
        self.y = 0.0
        self.client = CDPRClient(host, port)
        self.client.connect()
        # Measured calibration point if the caller has one (track_mallet.py).
        # Measured (x, y, theta_deg) from track_mallet.py if available.
        self.client.enable(*(cal_pose_mm or (None, None, None)))
        self._time = _time
        self._hw_rate = 10.0  # Hz — command rate to hardware
        self._last_hw_send = 0.0
        self._hw_x_mm = geom.HOME_X
        self._hw_y_mm = geom.HOME_Y
        self._hw_counts = [0, 0, 0, 0]
        self._cmd_x_mm = geom.HOME_X
        self._cmd_y_mm = geom.HOME_Y

    def set_speed(self, mm_s: float) -> None:
        self.speed = max(1.0, min(float(mm_s), self.max_speed))

    def reset(self, x: float, y: float) -> None:
        try:
            self._read_state()
            self.x, self.y = self._mm_to_sim(self._hw_x_mm, self._hw_y_mm)
            print(f"  HW reset: at ({self._hw_x_mm:.1f}, {self._hw_y_mm:.1f})"
                  f" mm = sim ({self.x:.3f}, {self.y:.3f})")
        except Exception as e:
            print(f"  HW reset: failed to read position: {e}, using sim coords")
            self.x = x
            self.y = y

    def update(self, target_x: float, target_y: float, dt: float):
        mm_x, mm_y = self._sim_to_mm(target_x, target_y)
        self._cmd_x_mm, self._cmd_y_mm = mm_x, mm_y
        now = self._time.monotonic()
        if now - self._last_hw_send >= 1.0 / self._hw_rate:
            self._last_hw_send = now
            try:
                self.client.command_position(mm_x, mm_y, self.speed)
                self._read_state()
                self.x, self.y = self._mm_to_sim(self._hw_x_mm, self._hw_y_mm)
            except Exception as e:
                print(f"HardwareDynamics: command failed: {e}")
        return self.x, self.y

    def _read_state(self) -> None:
        """One STATUS round trip instead of POS: the step counts come back
        for free and they are the only record of what the machine believes
        per cable, which is what a position disagreement has to be traced
        through."""
        s = self.client.get_status()
        self._hw_x_mm, self._hw_y_mm = s["x"], s["y"]
        self._hw_counts = [s["c0"], s["c1"], s["c2"], s["c3"]]

    def get_hw_position_mm(self):
        """Last known hardware position in mm (grid frame)."""
        return self._hw_x_mm, self._hw_y_mm

    def hw_state(self) -> dict:
        """Everything the controller believes, in grid mm — for the live
        state view, which exists to be compared against the camera."""
        return {
            "x_mm": round(self._hw_x_mm, 1),
            "y_mm": round(self._hw_y_mm, 1),
            "cmd_x_mm": round(self._cmd_x_mm, 1),
            "cmd_y_mm": round(self._cmd_y_mm, 1),
            "counts": list(self._hw_counts),
            "speed_mm_s": round(self.speed, 1),
        }

    def _sim_to_mm(self, sx: float, sy: float):
        """Sim metres -> grid-frame mm, clamped into the workspace."""
        g = self.geom
        fx = min(max(sx / self.sim_width, 0.0), 1.0)
        fy = min(max(sy / self.sim_half_height, 0.0), 1.0)
        if self.SIM_X_FLIP:
            fx = 1.0 - fx
        mm_x = g.WS_MAX_X - fy * (g.WS_MAX_X - g.WS_MIN_X)
        mm_y = g.WS_MIN_Y + fx * (g.WS_MAX_Y - g.WS_MIN_Y)
        # Clamp here rather than letting the firmware do it silently.
        return g.clamp_to_workspace(mm_x, mm_y)

    def _mm_to_sim(self, mm_x: float, mm_y: float):
        g = self.geom
        fy = (g.WS_MAX_X - mm_x) / (g.WS_MAX_X - g.WS_MIN_X)
        fx = (mm_y - g.WS_MIN_Y) / (g.WS_MAX_Y - g.WS_MIN_Y)
        if self.SIM_X_FLIP:
            fx = 1.0 - fx
        return fx * self.sim_width, fy * self.sim_half_height

