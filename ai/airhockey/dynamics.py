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
    """Drives real CDPR hardware via the cdpr_server.

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
        speed_mm_s: float = 50.0,
        host: str = "127.0.0.1",
        port: int = 8421,
    ):
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

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        mm_x, mm_y = self._sim_to_mm(x, y)
        self.client.set_position(mm_x, mm_y)

    def update(self, target_x: float, target_y: float, dt: float) -> tuple[float, float]:
        mm_x, mm_y = self._sim_to_mm(target_x, target_y)
        try:
            actual_mm_x, actual_mm_y = self.client.move_to(mm_x, mm_y, self.speed)
            self.x, self.y = self._mm_to_sim(actual_mm_x, actual_mm_y)
        except Exception as e:
            print(f"HardwareDynamics: move failed: {e}")
            # Don't update position on failure — keep last known good position.
        return self.x, self.y

    def _sim_to_mm(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert sim coords (meters) to CDPR coords (mm).

        Clamps output to CDPR workspace with margin to prevent
        commanding positions outside the physical bounds.
        """
        margin = 30.0  # mm, matches CDPRConfig.edge_margin
        mm_x = (sx / self.sim_width) * self.cdpr_width
        mm_y = (sy / self.sim_half_height) * self.cdpr_height
        mm_x = max(margin, min(self.cdpr_width - margin, mm_x))
        mm_y = max(margin, min(self.cdpr_height - margin, mm_y))
        return mm_x, mm_y

    def _mm_to_sim(self, mm_x: float, mm_y: float) -> tuple[float, float]:
        """Convert CDPR coords (mm) to sim coords (meters)."""
        sx = (mm_x / self.cdpr_width) * self.sim_width
        sy = (mm_y / self.cdpr_height) * self.sim_half_height
        return sx, sy
