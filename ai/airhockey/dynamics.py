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
        speed_mm_s: float = 200.0,
        max_speed_mm_s: float = 12000.0,
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
        # The DEFAULT is deliberately conservative — the winding-side sign
        # is still unverified, so commanded motion stays slow enough to
        # watch. The CEILING is the hardware's, so it is never the reason
        # something cannot go faster. See fw/include/cdpr_config.h.
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
        self._enc = None
        self._enc_zero = None
        self._last_enc_read = 0.0
        self._speed_limit = None
        self._accel_limit = None
        self._limit_flags = 0
        self._usage = {}

    def set_speed(self, mm_s: float) -> None:
        self.speed = max(1.0, min(float(mm_s), self.max_speed))

    def reset_peaks(self) -> None:
        self.client.reset_peaks()

    ACCEL_CEILING = 120000.0     # mirrors MAX_ACCEL_MM_S2

    def set_limits(self, speed_mm_s: float, accel_mm_s2: float) -> dict:
        """Push both caps to the Teensy, which is where the profile lives.

        Returns what was actually applied and what the ceilings are, so the
        caller can say so instead of silently clamping — a limit that is
        quietly ignored is worse than one that refuses.
        """
        want_s, want_a = float(speed_mm_s), float(accel_mm_s2)
        accel = max(1.0, min(want_a, self.ACCEL_CEILING))
        self.set_speed(want_s)
        self.client.set_limits(self.speed, accel)
        return {
            "speed": self.speed, "accel": accel,
            "speed_max": self.max_speed, "accel_max": self.ACCEL_CEILING,
            "clamped": (abs(self.speed - want_s) > 0.5
                        or abs(accel - want_a) > 0.5),
        }

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
        self._speed_limit = s.get("speed_limit")
        self._accel_limit = s.get("accel_limit")
        self._limit_flags = s.get("limit_flags", 0)
        self._usage = {k: s.get(k) for k in
                       ("speed_frac", "accel_frac", "speed_peak", "accel_peak")}
        # The drives' own encoders, at a slower cadence — this is a separate
        # serial round trip to four nodes and does not need to keep up with
        # the command rate.
        now = self._time.monotonic()
        if now - self._last_enc_read >= 0.5:
            self._last_enc_read = now
            try:
                e = self.client.get_encoders()
                if self._enc_zero is None:
                    self._enc_zero = list(e["posn"])
                self._enc = e
            except Exception:
                self._enc = None

    def _enc_cable_mm(self):
        """Drive-measured cable travel since ENABLE, in mm, per motor.

        Relative to enable rather than absolute because the drives are not
        homed — only the change since the reference is meaningful. The SIGN
        is raw drive-positive: which way that is on each spool is not yet
        established, and asserting one here would hide the very disagreement
        this exists to expose."""
        if not self._enc or self._enc_zero is None:
            return None
        out = []
        for m in range(4):
            res = self._enc["res"][m]
            if not res:
                out.append(None)
                continue
            revs = (self._enc["posn"][m] - self._enc_zero[m]) / res
            out.append(round(revs * self.geom.SPOOL_CIRCUMFERENCE_MM, 2))
        return out

    def get_hw_position_mm(self):
        """Last known hardware position in mm (grid frame)."""
        return self._hw_x_mm, self._hw_y_mm

    def hw_state(self) -> dict:
        """Everything the controller believes, in grid mm — for the live
        state view, which exists to be compared against the camera."""
        from airhockey.hardware import counts_to_cable_mm

        # What the Teensy asked each cable to do, in the same units as the
        # drive measurement below, so the two columns are comparable.
        step_mm = [round(counts_to_cable_mm(c), 2) for c in self._hw_counts]
        return {
            "x_mm": round(self._hw_x_mm, 1),
            "y_mm": round(self._hw_y_mm, 1),
            "cmd_x_mm": round(self._cmd_x_mm, 1),
            "cmd_y_mm": round(self._cmd_y_mm, 1),
            "counts": list(self._hw_counts),
            "step_mm": step_mm,
            "enc_mm": self._enc_cable_mm(),
            "trq_pct": None if not self._enc else [round(v, 1)
                                                  for v in self._enc["trq"]],
            "speed_mm_s": round(self.speed, 1),
            # What the Teensy is actually enforcing, and which cap the last
            # tick hit. Bit 0/1 = x accel/speed, bit 2/3 = y accel/speed.
            "speed_limit": self._speed_limit,
            "accel_limit": self._accel_limit,
            "limit_flags": self._limit_flags,
            **{k: v for k, v in self._usage.items() if v is not None},
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

