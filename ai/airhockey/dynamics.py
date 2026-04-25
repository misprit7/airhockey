"""Pluggable motor dynamics models.

These models sit between the RL agent's action (target position) and the
actual paddle position, simulating real-world actuator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Drives real CDPR hardware via the cdpr_master.

    Maps between the sim coordinate system (meters, 1.0 x 2.0 table) and
    the CDPR coordinate system (mm, physical table dimensions).

    The agent controls the bottom half of the sim table (y in [0, height/2]).
    The CDPR workspace maps to this region.
    """

    def __init__(
        self,
        cdpr_width_mm: float = 606.0,
        cdpr_height_mm: float = 730.0,
        sim_width: float = 1.0,
        sim_height: float = 2.0,
        speed_mm_s: float = 200.0,
        host: str = "127.0.0.1",
        port: int = 8421,
    ):
        import time as _time
        from airhockey.hardware import CDPRClient

        self.cdpr_width = cdpr_width_mm
        self.cdpr_height = cdpr_height_mm
        self.sim_width = sim_width
        self.sim_half_height = sim_height / 2.0  # agent's half
        self.speed = speed_mm_s
        self.x = 0.0
        self.y = 0.0
        self.client = CDPRClient(host, port)
        self.client.connect()
        self.client.enable()
        self._time = _time
        self._hw_rate = 10.0  # Hz — how often to send commands to hardware
        self._last_hw_send = 0.0
        # Last known hardware position in mm (updated from POS responses)
        self._hw_x_mm = cdpr_width_mm / 2.0
        self._hw_y_mm = cdpr_height_mm / 2.0

    def reset(self, x: float, y: float) -> None:
        # Master calibrates at center on ENABLE. Read current position.
        try:
            mm_x, mm_y, _, _ = self.client.get_position()
            self._hw_x_mm = mm_x
            self._hw_y_mm = mm_y
            self.x, self.y = self._mm_to_sim(mm_x, mm_y)
            print(f"  HW reset: at ({mm_x:.1f}, {mm_y:.1f}) mm = sim ({self.x:.3f}, {self.y:.3f})")
        except Exception as e:
            print(f"  HW reset: failed to read position: {e}, using sim coords")
            self.x = x
            self.y = y

    def update(self, target_x: float, target_y: float, dt: float) -> tuple[float, float]:
        mm_x, mm_y = self._sim_to_mm(target_x, target_y)
        now = self._time.monotonic()
        if now - self._last_hw_send >= 1.0 / self._hw_rate:
            self._last_hw_send = now
            try:
                self.client.command_position(mm_x, mm_y, self.speed)
                # Read back actual position from Teensy status
                act_x, act_y, _, _ = self.client.get_position()
                self._hw_x_mm = act_x
                self._hw_y_mm = act_y
                self.x, self.y = self._mm_to_sim(act_x, act_y)
            except Exception as e:
                print(f"HardwareDynamics: command failed: {e}")
        return self.x, self.y

    def get_hw_position_mm(self) -> tuple[float, float]:
        """Return the last known hardware position in mm."""
        return self._hw_x_mm, self._hw_y_mm

    def _sim_to_mm(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert sim coords (meters) to CDPR coords (mm).

        Maps the full sim area to the inner 2/3 of the CDPR workspace,
        keeping the cart away from the edges.
        """
        x_margin = self.cdpr_width / 6.0
        y_margin = self.cdpr_height / 6.0
        inner_w = self.cdpr_width - 2 * x_margin
        inner_h = self.cdpr_height - 2 * y_margin

        mm_x = x_margin + (sx / self.sim_width) * inner_w
        mm_y = y_margin + (sy / self.sim_half_height) * inner_h

        mm_x = max(x_margin, min(self.cdpr_width - x_margin, mm_x))
        mm_y = max(y_margin, min(self.cdpr_height - y_margin, mm_y))
        return mm_x, mm_y

    def _mm_to_sim(self, mm_x: float, mm_y: float) -> tuple[float, float]:
        """Convert CDPR coords (mm) to sim coords (meters)."""
        x_margin = self.cdpr_width / 6.0
        y_margin = self.cdpr_height / 6.0
        inner_w = self.cdpr_width - 2 * x_margin
        inner_h = self.cdpr_height - 2 * y_margin

        sx = ((mm_x - x_margin) / inner_w) * self.sim_width
        sy = ((mm_y - y_margin) / inner_h) * self.sim_half_height
        return sx, sy
