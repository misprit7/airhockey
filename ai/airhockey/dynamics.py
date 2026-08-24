"""Pluggable motor dynamics models.

These models sit between the RL agent's action (target position) and the
actual paddle position, simulating real-world actuator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── The robot's operating limits. ONE definition. ───────────────────────
#
# Sim units are metres, so these are the mm/s figures over 1000. Everything
# that needs a cap imports these rather than carrying its own: before
# 2026-08-23 the same two numbers appeared in six places with THREE
# different values (4.0/40.0 here and in batch_env, 3.0/30.0 in both
# training scripts, and a randomisation range of 2.0-4.5 that did not even
# contain the others), so which cap you got depended on the entry point,
# and raising one was never the one that was binding.
#
# These are SIM limits and are deliberately not the firmware's. The Teensy
# ceiling is 12000 mm/s / 120000 mm/s^2 and clamps independently; that stays
# the single authority for what the hardware will actually do.
MAX_SPEED_M_S = 12.0    # 12000 mm/s
MAX_ACCEL_M_S2 = 20.0   # 20000 mm/s^2

# How these sit against the machine, so a transfer failure is not mysterious:
#
#   SPEED  exactly the firmware clamp (MAX_VELOCITY_MM_S = 12000), and 93% of
#          the motors' own 12968 mm/s of cable (2580 rpm -- the slower of the
#          two drive models -- over a 301.6 mm circumference). On a CDPR every
#          cable moves together, so the system takes the worse of the two
#          models. Nothing here can ask for more than the Teensy will pass.
#
#   ACCEL  well under the firmware ceiling of 120000, and under what the
#          cables can make across most of the workspace: cdpr_config.h's solve
#          puts the centre near 114000. It IS above that solve's worst corner
#          of 9000. But that figure was computed for the old BOX bounds and is
#          order-of-magnitude at best, so read it as "the corners have less
#          than the middle" rather than as a number -- and expect the corners
#          to be where a trained policy first asks for something the rig
#          cannot deliver.

# Domain-randomisation spread, as a FRACTION of the nominal above, so the
# range tracks the nominal instead of silently excluding it. The old
# absolute range would not have contained a 12 m/s nominal at all.
DR_CAP_RANGE = (0.5, 1.125)


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

    # Caps come from MAX_SPEED_M_S / MAX_ACCEL_M_S2 at the top of this
    # file, which is the single definition; see the note there for how
    # they sit against the firmware clamp and the cables.
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    max_speed: float = MAX_SPEED_M_S
    max_accel: float = MAX_ACCEL_M_S2
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

        sim y 0 .. 1   ->  grid x RAIL_MAX_X .. CENTERLINE_X (robot -> centre)
        sim x 0 .. 1   ->  grid y RAIL_MIN_Y .. RAIL_MAX_Y

      Against the TABLE, not the workspace. Mapping onto the workspace made
      the scale change whenever a limit was retuned and silently rescaled
      paddle speed with it; the workspace now enters only as a clamp.

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
        # A DEFAULT, deliberately conservative — the winding-side sign is
        # still unverified, so commanded motion stays slow enough to watch.
        # There is deliberately NO ceiling here: the only clamp in the whole
        # chain lives in the firmware (fw/include/cdpr_config.h), because a
        # limit duplicated in three places is a limit that will disagree with
        # itself, and the one you raise is never the one that was binding.
        self.speed = speed_mm_s
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
        self.speed = float(mm_s)          # the Teensy decides what is legal

    def reset_peaks(self) -> None:
        self.client.reset_peaks()

    def set_limits(self, speed_mm_s: float, accel_mm_s2: float) -> dict:
        """Push both caps to the Teensy and report back what it accepted.

        Nothing is clamped here. The firmware clamps, and then we ASK it
        what it ended up with rather than predicting — so there is no second
        copy of the ceiling to drift out of step with the first, and the
        answer is true by construction instead of by agreement.
        """
        want_s, want_a = float(speed_mm_s), float(accel_mm_s2)
        self.set_speed(want_s)
        before = (self._speed_limit, self._accel_limit)
        self.client.set_limits(want_s, want_a)
        got_s, got_a = self._await_limits(want_s, want_a, before)
        self.speed = got_s
        return {
            "speed": got_s, "accel": got_a,
            "clamped_speed": abs(got_s - want_s) > 0.5,
            "clamped_accel": abs(got_a - want_a) > 0.5,
        }

    # The master does not query the Teensy on demand: STATUS hands back a
    # cache its reader thread refills from the 50 Hz status line. So the
    # obvious "write, then read what took" is a race the write usually
    # loses, and the answer is the caps from BEFORE the change. Reported as
    # 'firmware clamped' and written back into the fields, that reads as the
    # UI corrupting a value nobody touched.
    LIMIT_SETTLE_S = 0.25          # >10 status frames at 20 ms
    LIMIT_POLL_S = 0.005

    def _await_limits(self, want_s: float, want_a: float, before):
        """Read the caps back only once the status can reflect the write.

        Accepts on either of two conditions, because only their combination
        covers a firmware that clamps: the caps now match what was asked, or
        they differ from what was there before it was asked. Matching alone
        would spin until the deadline whenever the firmware legitimately
        refused a value; differing alone would never fire when the request
        was a no-op.
        """
        deadline = self._time.monotonic() + self.LIMIT_SETTLE_S
        have_baseline = None not in before
        while True:
            self._read_state()
            got = (self._speed_limit, self._accel_limit)
            if got[0] is None or got[1] is None:
                return want_s, want_a      # firmware predates the cap readout
            settled = (abs(got[0] - want_s) < 0.5
                       and abs(got[1] - want_a) < 0.5)
            if settled or (have_baseline and got != before):
                return got
            if self._time.monotonic() >= deadline:
                return got
            self._time.sleep(self.LIMIT_POLL_S)

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

    # ── Sim <-> table, NOT sim <-> workspace ────────────────────────────
    #
    # These used to map the sim's whole agent half onto the WORKSPACE
    # rectangle, which is a silent rescale — 1.0 x 1.0 m of sim onto
    # whatever the reachable box happens to be. Three things were wrong
    # with it, and they compound:
    #
    #   1. The scale factor changed whenever the workspace did. Editing
    #      WS_* rescaled the meaning of every sim coordinate in the system.
    #   2. Paddle SPEED was rescaled with it, so a policy trained at
    #      4 m/s in sim produced something else entirely on hardware.
    #   3. The mallet rendered at the corner of the drawn table whenever it
    #      was at the corner of its reachable box, which is mid-table. The
    #      display said "at the wall" when the truth was "at the software
    #      limit", and there was no way to tell those apart by looking.
    #
    # The frame is now the TABLE, which is a physical fact and does not
    # move when a limit is retuned. The workspace re-enters only as a
    # clamp, so an unreachable request stops at the boundary and RENDERS
    # there, in its true place on the table.
    #
    # sim y 0..1  ->  grid x RAIL_MAX_X..CENTERLINE_X  (robot end -> centre)
    # sim x 0..1  ->  grid y RAIL_MIN_Y..RAIL_MAX_Y
    def _sim_to_mm(self, sx: float, sy: float):
        """Sim metres -> grid-frame mm, clamped into the workspace."""
        g = self.geom
        fx = min(max(sx / self.sim_width, 0.0), 1.0)
        fy = min(max(sy / self.sim_half_height, 0.0), 1.0)
        if self.SIM_X_FLIP:
            fx = 1.0 - fx
        mm_x = g.RAIL_MAX_X - fy * (g.RAIL_MAX_X - g.CENTERLINE_X)
        mm_y = g.RAIL_MIN_Y + fx * (g.RAIL_MAX_Y - g.RAIL_MIN_Y)
        # Clamp here rather than letting the firmware do it silently.
        return g.clamp_to_workspace(mm_x, mm_y)

    def _mm_to_sim(self, mm_x: float, mm_y: float):
        g = self.geom
        fy = (g.RAIL_MAX_X - mm_x) / (g.RAIL_MAX_X - g.CENTERLINE_X)
        fx = (mm_y - g.RAIL_MIN_Y) / (g.RAIL_MAX_Y - g.RAIL_MIN_Y)
        if self.SIM_X_FLIP:
            fx = 1.0 - fx
        return fx * self.sim_width, fy * self.sim_half_height

    def workspace_in_sim(self):
        """The reachable box in sim coordinates, for the UI to draw.

        Sent to the browser rather than recomputed there: the front end
        should not own a second copy of this mapping, which is how the two
        coordinate systems got out of step in the first place.
        """
        x0, y0 = self._mm_to_sim(self.geom.WS_MIN_X, self.geom.WS_MIN_Y)
        x1, y1 = self._mm_to_sim(self.geom.WS_MAX_X, self.geom.WS_MAX_Y)
        return {"min_x": min(x0, x1), "max_x": max(x0, x1),
                "min_y": min(y0, y1), "max_y": max(y0, y1)}

