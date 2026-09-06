"""Pluggable motor dynamics models.

These models sit between the RL agent's action (target position) and the
actual paddle position, simulating real-world actuator behavior.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Module level, as in physics.py. These conversions used to do the
# sys.path.insert INSIDE the function, which appends a duplicate entry on every
# call -- fine for the UI's few calls a second, and a growing sys.path for
# anything driving them from a control loop.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as _geom  # noqa: E402


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

# The CONTROL RATE: how often the policy decides, in sim and on the table.
# One constant, derived everywhere (env default action_dt, curriculum
# episode lengths, the trainers, eval, run_policy --cmd-hz), because it was
# "1 / 100" in nine places and "3000 steps = 30 s" in three more.
#
# 50 Hz, down from 100 on 2026-09-06. The planner has to answer inside one
# tick: at 100 Hz three MPPI iterations took 7.7 ms median / 10.4 p95 plus
# ~1 ms of master I/O and the loop fell up to 0.3 s behind the camera. At
# 50 Hz the same planner fits with slack, six iterations fit at all, and a
# five-step horizon looks 100 ms ahead instead of 50 -- nearer the time a
# shot takes to cross the table. Reaction latency grows by 10 ms on
# average, 50 mm of puck travel at 5 m/s, a third of a paddle radius.
# Sensing is unaffected: the camera model ticks on its own 200 Hz clock.
ACTION_HZ = 50.0
ACTION_DT = 1.0 / ACTION_HZ


def table_mm_to_sim(mm_x: float, mm_y: float, sim_width: float = 1.0,
                    sim_half_height: float = 1.0, flip: bool = False):
    """Grid-frame mm -> sim metres.

    A free function, not only a HardwareDynamics method, because the mapping
    is a property of the TABLE and of the sim's chosen dimensions — nothing
    about it involves a motor. The camera needs it with no drives connected
    at all: a puck the robot is not touching still has to render in the right
    place.

    UNCLAMPED, unlike the sim -> mm direction. That is deliberate: the
    workspace is a limit on where the PADDLE may go, and applying it here
    would drag a puck sitting on the human half onto the robot's boundary.
    Grid x below the centreline simply maps to sim y past sim_half_height,
    which is exactly the opponent's half.
    """
    g = _geom
    fy = (g.RAIL_MAX_X - mm_x) / (g.RAIL_MAX_X - g.CENTERLINE_X)
    fx = (mm_y - g.RAIL_MIN_Y) / (g.RAIL_MAX_Y - g.RAIL_MIN_Y)
    if flip:
        fx = 1.0 - fx
    return fx * sim_width, fy * sim_half_height


def sim_to_table_mm(sim_x: float, sim_y: float, sim_width: float = 1.0,
                    sim_half_height: float = 1.0, flip: bool = False):
    """Sim metres -> grid-frame mm. The exact inverse of `table_mm_to_sim`.

    UNCLAMPED, matching the forward direction, so the two compose to the
    identity everywhere rather than only inside the workspace. Callers that
    are about to DRIVE something want HardwareDynamics._sim_to_mm, which is
    this plus the workspace clamp; callers converting a puck, an opponent's
    mallet, or a heuristic bot's view of the table want this.
    """
    g = _geom
    fx = sim_x / sim_width
    fy = sim_y / sim_half_height
    if flip:
        fx = 1.0 - fx
    return (g.RAIL_MAX_X - fy * (g.RAIL_MAX_X - g.CENTERLINE_X),
            g.RAIL_MIN_Y + fx * (g.RAIL_MAX_Y - g.RAIL_MIN_Y))

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

# ── The ROBOT's domain randomisation, in ABSOLUTE units ──────────────────
#
# SPEED IS NOT RANDOMISED. MAX_SPEED_M_S is the firmware's own clamp
# (MAX_VELOCITY_MM_S = 12000 in cdpr_config.h) and the Teensy enforces it
# absolutely -- the one number in the whole actuator model that is not an
# estimate. It used to be sampled 0.5-1.0x "to model a machine that
# underperforms", but a full run showed the machine never gets NEAR the
# clamp anyway: at 20 m/s^2 over the workspace's ~0.3 m runs, v = sqrt(2ad)
# tops out around 3.5 m/s. Speed is not the binding constraint; sampling it
# down only taught the policy about machines that do not exist.
#
# ACCEL IS THE BINDING CONSTRAINT, and it WAS where the spread went:
# 10-60 m/s^2, absolute, because the truth varies with position --
# cdpr_config.h's solve puts the table centre near 114 m/s^2 and the worst
# corner near 9 -- and a single number cannot follow the pose. Since
# 2026-09-02 it is PINNED at the top of that band: the spread made the two
# sides of self-play different machines every episode and cost the policy
# capacity on bodies it will never drive. The observation still carries the
# cap ratios ([13], [14]) as constants, so the band can be reopened without
# changing the network's shape. The firmware ceiling is 120; nothing here
# approaches it.
#
# 2026-09-06: the tracking test (airhockey/follow_test.py) settled what the
# drives follow: 20, 40 and 60 m/s^2 all CLOSE (13 / 21 / 26 mm p90 at
# speed). The 200-440 mm "gap" of the 12/60 play session that first read as
# the drives falling behind was the runner acting on camera frames 0.1-0.3 s
# old; the drives were never the limit. What has NOT been measured is the
# sustained thermal load: drives tripped RMS overload after ~30 s of
# flat-out play at 24.
AGENT_DR_SPEED_M_S = (MAX_SPEED_M_S, MAX_SPEED_M_S)   # pinned to the clamp
#
# 40 since the retrain of 2026-09-06 (ai/RETRAIN.md): the tracking test on
# the rig followed 40 m/s^2 within 21 mm p90 and 60 within 26, and 60 with
# the old policy still jittered on the table after the planner and loop
# fixes -- the user chose to train lower. The observation's cap ratio reads
# 2.0 (nominal MAX_ACCEL_M_S2 stays 20); constant, so the band can reopen.
AGENT_DR_ACCEL_M_S2 = (40.0, 40.0)                      # pinned, see above

# Fraction-of-nominal spread for the OPPONENT (human) side only.
DR_SPEED_RANGE = (0.5, 1.0)

# ── The HUMAN side. Deliberately NOT the robot's limits. ─────────────────
#
# The opponent is a sparring partner, not a second robot, and the point of
# training against it is that the machine will face something less
# constrained than itself. Modelling it with the robot's own caps and the
# robot's own workspace teaches the policy that its opponent is exactly as
# limited as it is, which is the one thing that is certainly false.
#
# Grounded in the 2026-08-29 recording, where the tracked hand-held mallet
# reached 7.33 m/s peak (p99 3.49) and covered 862 x 951 mm against the
# robot's 568 x 620 -- so a human is not much faster at the top end, but has
# 2.3x the reach and gets there without a jerk limit. Headroom over the
# measured peak is deliberate: the recording is one person warming up, not
# the hardest opponent the robot should survive.
OPPONENT_MAX_SPEED_M_S = 15.0    # 1.25x the robot, 2x the measured human peak
OPPONENT_MAX_ACCEL_M_S2 = 80.0   # 4x the robot: a wrist has no jerk limit

# ACCEL is two-sided, because unlike speed it is genuinely uncertain rather
# than clamped. The firmware ceiling is 120000 mm/s^2, six times the nominal
# here, so nothing in this range is near it; what actually bounds
# acceleration is what the cables can deliver, and that VARIES WITH POSITION
# -- cdpr_config.h's solve puts the table centre near 114000 and its worst
# corner near 9000. Sampling above nominal is therefore a claim about the
# cables, not a violation of a limit.
DR_ACCEL_RANGE = (0.5, 1.125)

# Deprecated alias; both terms used to share one range. Kept so an old caller
# fails loudly at import rather than silently getting the speed range.
DR_CAP_RANGE = DR_ACCEL_RANGE


def workspace_in_sim(sim_width: float = 1.0, sim_half_height: float = 1.0,
                     flip: bool = False):
    """The box the PADDLE can actually reach, in sim metres.

    Not the table half. Cables pull only, so the paddle is holdable only well
    inside the anchor hull, and the workspace is further trimmed where the
    drives overloaded just HOLDING position. The result is 35% of the robot's
    half, and it does not include the robot's own goal line -- the nearest it
    gets is sim y 0.099.

    That gap is why this exists. A sim that lets the paddle sit on its goal
    line trains a policy to defend from a place the machine cannot occupy, and
    HardwareDynamics._sim_to_mm clamps silently on the day, so the failure
    looks like a policy that is merely bad rather than one being cut off.
    """
    g = _geom
    x0, y0 = table_mm_to_sim(g.WS_MIN_X, g.WS_MIN_Y, sim_width,
                             sim_half_height, flip)
    x1, y1 = table_mm_to_sim(g.WS_MAX_X, g.WS_MAX_Y, sim_width,
                             sim_half_height, flip)
    return {"min_x": min(x0, x1), "max_x": max(x0, x1),
            "min_y": min(y0, y1), "max_y": max(y0, y1)}


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
class ProfileDynamics(MotorDynamics):
    """The REAL firmware motion law, for a single paddle.

    Wraps the same `fw/include/motion_profile.h` the Teensy runs, built as a
    host library and bound through motion.py -- so the web UI and a scalar
    env move the paddle exactly the way the machine does, jerk limit and
    parking rule included, rather than through a first-order lag that
    resembles it.

    ACCELERATION IS STATE. The profile slews it to bound jerk, so the same
    command produces different motion depending on what the acceleration was.
    That is why this holds a CartState rather than recomputing from position.
    """

    max_speed: float = MAX_SPEED_M_S
    max_accel: float = MAX_ACCEL_M_S2
    ramp_s: float = 0.003          # matches the firmware's RAMP default
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self):
        from airhockey.motion import CartState
        self._cart = CartState(1)

    def reset(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        c = self._cart
        c.x[0], c.y[0] = x * 1000.0, y * 1000.0
        c.vx[0] = c.vy[0] = c.ax[0] = c.ay[0] = 0.0

    def update(self, target_x: float, target_y: float,
               dt: float) -> tuple[float, float]:
        from airhockey.motion import DEFAULT_SIM_DT, advance
        c = self._cart
        substeps = max(1, int(round(dt / DEFAULT_SIM_DT)))
        advance(c,
                np.float32([target_x * 1000.0]), np.float32([target_y * 1000.0]),
                self.max_speed * 1000.0, self.max_accel * 1000.0,
                self.ramp_s, dt / substeps, substeps)
        self.x = float(c.x[0]) / 1000.0
        self.y = float(c.y[0]) / 1000.0
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

        from airhockey.hardware import CDPRClient

        self.geom = _geom
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
        self._hw_x_mm = _geom.HOME_X
        self._hw_y_mm = _geom.HOME_Y
        self._hw_counts = [0, 0, 0, 0]
        self._cmd_x_mm = _geom.HOME_X
        self._cmd_y_mm = _geom.HOME_Y
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

    def absorb_status(self, s: dict) -> None:
        """Take a STATUS reply as the controller's current belief.

        Split from `_read_state` so something else holding the socket --
        the tracking test, which owns it for the duration of its run -- can
        keep this object's view current without a second connection. The
        master serves one client and the protocol is one reply per command,
        so two readers on one socket would swap each other's answers.
        """
        self._hw_x_mm, self._hw_y_mm = s["x"], s["y"]
        self._hw_counts = [s["c0"], s["c1"], s["c2"], s["c3"]]
        self._speed_limit = s.get("speed_limit")
        self._accel_limit = s.get("accel_limit")
        self._limit_flags = s.get("limit_flags", 0)
        self._usage = {k: s.get(k) for k in
                       ("speed_frac", "accel_frac", "speed_peak", "accel_peak")}
        self.x, self.y = self._mm_to_sim(self._hw_x_mm, self._hw_y_mm)

    def _read_state(self) -> None:
        """One STATUS round trip instead of POS: the step counts come back
        for free and they are the only record of what the machine believes
        per cable, which is what a position disagreement has to be traced
        through."""
        self.absorb_status(self.client.get_status())
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
        """Sim metres -> grid-frame mm, clamped into the workspace.

        The mapping itself is `sim_to_table_mm`; this adds only the clamp,
        which is here rather than left to the firmware so an out-of-reach
        request stops visibly instead of silently. The fraction clamp this
        used to do first was redundant -- the mapping is monotone in each
        axis and the workspace is inside the table, so clamping the result
        subsumes it.
        """
        return self.geom.clamp_to_workspace(
            *sim_to_table_mm(sx, sy, self.sim_width, self.sim_half_height,
                             self.SIM_X_FLIP))

    def _mm_to_sim(self, mm_x: float, mm_y: float):
        return table_mm_to_sim(mm_x, mm_y, self.sim_width,
                               self.sim_half_height, self.SIM_X_FLIP)

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

