#!/usr/bin/env python3
"""Drive the physical table from a policy.

    # DEFAULT. Tracks, plans, prints what it would command. Moves nothing.
    python ai/bin/run_policy.py

    # Same, naming a bot from ai/airhockey/heuristics.py
    python ai/bin/run_policy.py --policy heuristic:goalie

    # Neither camera nor robot. Synthetic puck through the whole chain.
    python ai/bin/run_policy.py --selftest

    # MOVES THE ROBOT. Needs sw/build/cdpr_master running, and ONLY that.
    # THE FIRST LIVE RUN OF ANY NEW POLICY GOES THROUGH --gentle. The opening
    # command is the dangerous one: the paddle is wherever ENABLE left it and
    # the policy may ask for the far corner, so that first move is a
    # full-workspace traverse at whatever the caps allow. --gentle holds it
    # to 500 mm/s and 2000 mm/s^2, slow enough that a flipped sign or a bad
    # calibration is something you watch happen rather than something you
    # hear.
    python ai/bin/run_policy.py --live --gentle --policy heuristic:goalie
    python ai/bin/run_policy.py --live --policy heuristic:goalie   # then full

Successor to ai/bin/goalie_demo.py, which hardcodes one policy. The split
here is that the POLICY is a plug: a heuristic bot today, a trained agent
later, with the tracking, the safety clamp and the command protocol shared
so the two are never deployed differently.

Everything above the camera is deliberately camera-free -- report building,
the policy call, the clamp, the cap committer -- so `pytest ai` can exercise
the whole chain without cv2, without vision/calib, and without hardware.
`--selftest` is that path, and ai/tests/test_run_policy.py runs it.

WHY --live RATHER THAN --dry-run
    goalie_demo.py defaults to moving and takes --dry-run to stop. That is
    the wrong way round for a file that will be run by whoever is holding a
    new policy: the failure mode of forgetting a flag should be a robot that
    sits still, not one that lunges. Anything that commands the hardware
    here is behind an explicit --live.

UNITS
    TABLE MILLIMETRES, grid frame, throughout -- the same frame as
    shared/cdpr_geometry.py, vision/bin/puck_stream.py and the Teensy. No
    sim units appear anywhere in this file. The sim<->table conversion is a
    problem for the `sac:` policy loader when it lands, and it belongs
    there, next to the observation it has to build.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai"))
sys.path.insert(0, str(ROOT / "shared"))

import cdpr_geometry as geom  # noqa: E402

# ── Policy interface ────────────────────────────────────────────────────
#
# A bot takes ONE dict and returns four numbers. Both are in table mm.
# This is the dict form of airhockey.heuristics.TrackerReport, which is what
# TrackerReport.coerce accepts:
#
#   {"puck":     [(x_mm, y_mm, t_s), ...]   NEWEST FIRST, may be empty
#    "mallet":   (x_mm, y_mm)               the robot's own. NEVER None.
#    "opponent": (x_mm, y_mm) | None        the human's, None if unseen
#    "t_s":      float}                     the tracker clock, for when the
#                                           puck history is empty
#       ->  (target_x_mm, target_y_mm, speed_mm_s, accel_mm_s2)
#          or a heuristics.Command, which carries the same four.
#
# The key names live here as constants rather than as literals scattered
# through the file, so a rename on the heuristics side is one edit.
OBS_PUCK = "puck"
OBS_MALLET = "mallet"
OBS_OPPONENT = "opponent"
OBS_TIME = "t_s"

# How much puck history the report carries. 200 ms at 200 Hz is up to 40
# samples -- enough for a bot to fit a velocity and see a wall bounce inside
# the window, and short enough that a bounce does not sit in the window long
# enough to corrupt a naive straight-line fit for long.
HISTORY_S = 0.200

# Beyond this with no fix, the puck is not "somewhere near where it was", it
# is unknown. Matches PuckTracker._coast, which gives up at the same point.
STALE_S = 0.150

# Beyond THIS, the puck is not blinking, it is gone -- off the table, under
# an arm, or the tracker has died. An order of magnitude above STALE_S on
# purpose: the two answer different questions and must not be one number.
# See PuckWatchdog.
DEFAULT_PUCK_TIMEOUT_S = 2.0

# The first-live-run preset. Slow enough that a wrong sign or a bad
# calibration is something you watch happen rather than something you hear.
GENTLE_SPEED = 500.0
GENTLE_ACCEL = 2000.0


@dataclass(frozen=True)
class Action:
    """What a policy asked for, before any clamping."""

    x_mm: float
    y_mm: float
    speed_mm_s: float
    accel_mm_s2: float


@dataclass(frozen=True)
class Caps:
    """The envelope a policy is allowed to ask for.

    Defaults follow ai/bin/goalie_demo.py, which is what has actually been
    run on the rig. speed 8000 of the firmware's MAX_VELOCITY_MM_S 12000;
    accel 24000 is deliberately ABOVE both the ~15120 mm/s^2 at which the
    paddle tips (g*r/h, r=50.4, h=32.7) and the ~17400 the cables can make
    near the centreline edge -- see goalie_demo. Drop it if the paddle hops.
    """

    speed_min: float = 100.0
    speed_max: float = 8000.0
    accel_min: float = 400.0
    accel_max: float = 24000.0

    def clamp_speed(self, v: float) -> float:
        return min(max(v, self.speed_min), self.speed_max)

    def clamp_accel(self, a: float) -> float:
        return min(max(a, self.accel_min), self.accel_max)


# ── Tracker report ──────────────────────────────────────────────────────


class ReportBuilder:
    """Accumulates tracker samples into the dict a policy is handed.

    ONLY REAL FIXES GO IN THE HISTORY. PuckTracker.update keeps returning a
    position across a dropout by extrapolating on the last velocity, and
    those coasted points are a straight line by construction. Feeding them
    to a bot that fits a velocity would have the bot re-derive the velocity
    the tracker assumed -- an estimate that looks like a measurement,
    strongest exactly where the tracker is blindest. The IR ring's own
    reflection blanks ~92 x 103 mm at table centre, so this is not a corner
    case; it is the middle of the table.

    A coasted frame therefore leaves the history alone and only ages
    `puck_age`. A bot sees a gap, which is the truth.
    """

    def __init__(self, history_s: float = HISTORY_S):
        self.history_s = history_s
        self.puck: deque[tuple[float, float, float]] = deque()
        self.mallet: tuple[float, float] | None = None
        self.opponent: tuple[float, float] | None = None
        # Last time each was actually SEEN, not last time it was asked for.
        self.t_puck = float("-inf")
        self.t_mallet = float("-inf")
        self.t_opponent = float("-inf")
        # Counters for the status line; the interesting number on the rig is
        # what fraction of frames produced a real fix.
        self.n_frames = 0
        self.n_puck = 0
        self.n_mallet = 0
        self.n_opponent = 0

    def _expire(self, t: float) -> None:
        """Drop samples older than the window, as of time `t`.

        Called on READ as well as on write. Expiring only when a new sample
        arrives would mean a puck that stops being seen leaves its last
        samples in the deque for ever: the history would keep reporting a
        full 40 samples, all of them old, and `TrackerReport`'s contract that
        an empty history means "not visible" would silently stop holding.
        The bot would fit a velocity across a second-old track and act on it
        as if it were current -- which presents as a bad policy, not as a
        stale report.
        """
        cutoff = t - self.history_s
        while self.puck and self.puck[-1][2] < cutoff:
            self.puck.pop()

    def add_puck(self, t: float, x: float, y: float) -> None:
        self.puck.appendleft((float(x), float(y), float(t)))
        self.t_puck = t
        self.n_puck += 1
        self._expire(t)

    def add_mallet(self, t: float, x: float, y: float) -> None:
        self.mallet = (float(x), float(y))
        self.t_mallet = t
        self.n_mallet += 1

    def add_opponent(self, t: float, x: float, y: float) -> None:
        self.opponent = (float(x), float(y))
        self.t_opponent = t
        self.n_opponent += 1

    def frame(self) -> None:
        self.n_frames += 1

    def observation(self, t: float, stale_s: float = STALE_S,
                    mallet_fallback: tuple[float, float] | None = None) -> dict:
        """The dict handed to the policy.

        THE TWO MALLETS ARE TREATED DIFFERENTLY, and the asymmetry is not an
        oversight. The robot's own mallet goes only where THIS PROCESS sent
        it, so when the camera loses sight of it there is a better answer
        than "unknown": the last commanded target, passed in as
        `mallet_fallback`. It is also the field heuristics.TrackerReport
        requires to be a tuple. The OPPONENT gets no fallback and goes None
        once stale, because nothing here influences where a human's mallet
        goes -- last-known would be a guess dressed as a measurement, and
        there is nothing in the dict to say how old it is.

        The puck history needs no staleness rule of its own: old samples
        fall out of the window, and an empty history means "not visible".
        `t_s` is carried so that a bot's timers keep running while it is.
        """
        self._expire(t)
        return {
            OBS_PUCK: list(self.puck),
            OBS_MALLET: (self.mallet if (self.mallet is not None
                                         and t - self.t_mallet <= stale_s)
                         else mallet_fallback or (geom.HOME_X, geom.HOME_Y)),
            OBS_OPPONENT: (self.opponent
                           if t - self.t_opponent <= stale_s else None),
            OBS_TIME: t,
        }

    def staleness(self, t: float) -> dict:
        """Ages in seconds; inf for anything never seen."""
        return {
            "puck": t - self.t_puck,
            "mallet": t - self.t_mallet,
            "opponent": t - self.t_opponent,
        }


class PuckWatchdog:
    """Stop advancing the policy once the puck has been gone too long.

    Not the same thing as a dropout. STALE_S covers the IR ring's blind spot,
    which is 150 ms and ends by itself. This covers the puck being off the
    table, under someone's arm, or the tracker having quietly died -- states
    that do not end, and in which a bot is still perfectly willing to produce
    a target. A goalie with no puck shades toward its last belief and a
    striker holds a swing latch, so the failure is not "the paddle stops", it
    is the paddle acting with confidence on something that stopped being true
    a minute ago.

    Freezing rather than parking, for the same reason the shutdown brakes in
    place: HOME is a full-workspace traverse, and starting one because the
    tracker lost sight of the puck is the opposite of a safe response.

    THE POLICY IS NOT CALLED AT ALL while blind, which matters beyond saving
    the work: bots carry latches and timers keyed off the clock in the
    report, and letting those run through a two-second blind spell would have
    them resume mid-swing on a puck that has since moved.

    Trips on an infinite age too, so a session that starts with no puck in
    frame holds instead of driving to whatever the bot opens with.
    """

    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s = timeout_s
        self.blind = False
        self.n_trips = 0

    def update(self, puck_age_s: float) -> str | None:
        """Advance the state. Returns a message ONLY on a transition.

        Once per transition, not once per tick: at 100 Hz a per-tick warning
        is 6000 lines a minute, which buries the one line that says when it
        started.
        """
        if not self.blind and puck_age_s > self.timeout_s:
            self.blind = True
            self.n_trips += 1
            age = ("never seen" if math.isinf(puck_age_s)
                   else f"unseen for {puck_age_s:.1f}s")
            return (f"puck {age} — HOLDING position, policy paused "
                    f"(resumes by itself when the puck comes back)")
        if self.blind and puck_age_s <= self.timeout_s:
            self.blind = False
            return "puck reacquired — resuming"
        return None


class LagMonitor:
    """How far the loop has fallen behind the camera.

    BlobStream reads blob lines from a PIPE, so a loop that cannot keep up
    does not drop frames -- they queue. The timestamps keep arriving in
    order, just late, and NOTHING IN THE DATA SAYS SO: every puck position
    is real, the history looks healthy, and the command tick is driven off
    the camera clock so it still fires at what looks like 100 Hz. What
    actually happens on the table is a policy acting on a puck that has
    already moved on, which presents as a bad policy rather than as a slow
    loop, and sends you to retrain something that was fine.

    Cheap to measure and impossible to infer: camera time against wall
    time, both differenced from their values on the first frame, so the two
    clocks never need a common epoch.
    """

    WARN_S = 0.025      # 2.5 command ticks; below this it is jitter

    def __init__(self):
        self._t_cam0: float | None = None
        self._t_wall0 = 0.0
        self.lag = 0.0
        self.peak = 0.0
        self.warned = False

    def update(self, t_cam: float, t_wall: float) -> float:
        if self._t_cam0 is None:
            self._t_cam0, self._t_wall0 = t_cam, t_wall
        self.lag = (t_wall - self._t_wall0) - (t_cam - self._t_cam0)
        self.peak = max(self.peak, self.lag)
        return self.lag

    def warn_once(self) -> str | None:
        """The message to print the first time the loop falls behind."""
        if self.warned or self.lag < self.WARN_S:
            return None
        self.warned = True
        return (f"WARNING: {1000 * self.lag:.0f} ms behind the camera and "
                f"growing. Frames are queueing in the blobtrack pipe, so the "
                f"puck the policy sees is that old. Lower --cmd-hz or --fps.")


# ── Safety clamp ────────────────────────────────────────────────────────


def clamp_action(action: Action, caps: Caps,
                 prev: tuple[float, float] | None = None):
    """Force an action inside the workspace and the cap envelope.

    Returns (safe_action, flags). `flags` names every component that had to
    be changed, so the caller can say so out loud rather than silently
    correcting a policy that is asking for something impossible.

    NON-FINITE VALUES ARE REJECTED, NOT CLAMPED. min/max propagate NaN
    rather than bounding it, so a NaN target survives a clamp untouched,
    formats as "nan", and reaches the firmware -- where every comparison
    against it is false, so the workspace clamp there passes it through too.
    A NaN is the one input that would defeat both clamps, so it is caught
    here and the previous target is held instead.
    """
    flags: list[str] = []
    x, y = action.x_mm, action.y_mm

    if not (math.isfinite(x) and math.isfinite(y)):
        flags.append("nonfinite-target")
        x, y = prev if prev is not None else (geom.HOME_X, geom.HOME_Y)

    cx, cy = geom.clamp_to_workspace(x, y)
    if (cx, cy) != (x, y):
        flags.append("workspace")

    v, a = action.speed_mm_s, action.accel_mm_s2
    if not math.isfinite(v):
        flags.append("nonfinite-speed")
        v = caps.speed_max
    if not math.isfinite(a):
        flags.append("nonfinite-accel")
        a = caps.accel_max
    cv, ca = caps.clamp_speed(v), caps.clamp_accel(a)
    if cv != v:
        flags.append("speed")
    if ca != a:
        flags.append("accel")

    return Action(cx, cy, cv, ca), flags


# ── Cap commitment ──────────────────────────────────────────────────────
#
# WHY THIS EXISTS -- the cost of changing the caps, from the source:
#
# sw/bin/cdpr_master.cpp handles LIMITS by forwarding TWO commands to the
# Teensy, SPEED then ACCEL, and calling waitTeensyOK on each. waitTeensyOK
# polls with usleep(1000) between reads, so its resolution is 1 ms and a
# reply that is not already buffered costs at least that. Two USB CDC round
# trips plus two 1 ms polls puts a LIMITS at roughly 2-6 ms.
#
# A CMD is cheaper but not free: it is one Teensy round trip, plus a SECOND
# one whenever the speed in the CMD differs from the last -- the master
# keeps a `static double last_speed` and pushes SPEED when it changes.
#
# At the 100 Hz command rate the whole tick budget is 10 ms. A tick that
# sends LIMITS and a speed-changing CMD is four round trips, which can eat
# the budget entirely. So caps are COMMITTED, not streamed: pushed only when
# the policy's request has moved meaningfully, and never faster than
# MIN_INTERVAL_S. Between commits the committed speed rides along in the
# CMD, where the master's own change detection makes it free.


class CapCommitter:
    """Rate-limited LIMITS pushes, so a per-tick cap change cannot stall.

    Threshold AND interval, not either alone: an interval alone would push
    caps that have not changed, and a threshold alone would let a policy
    that dithers across the threshold push on every tick.
    """

    MIN_INTERVAL_S = 0.200      # 5 Hz ceiling on LIMITS
    REL_TOL = 0.10              # 10% of the committed value ...
    ABS_SPEED_TOL = 250.0       # ... or this, whichever is larger
    ABS_ACCEL_TOL = 1000.0

    def __init__(self, client=None, min_interval_s: float = MIN_INTERVAL_S):
        self.client = client
        self.min_interval_s = min_interval_s
        self.speed: float | None = None
        self.accel: float | None = None
        self._t_last = float("-inf")
        self.n_commits = 0
        self.n_suppressed = 0

    def _worth_it(self, speed: float, accel: float) -> bool:
        assert self.speed is not None and self.accel is not None
        return (abs(speed - self.speed)
                > max(self.ABS_SPEED_TOL, self.REL_TOL * self.speed)
                or abs(accel - self.accel)
                > max(self.ABS_ACCEL_TOL, self.REL_TOL * self.accel))

    def maybe_commit(self, t: float, speed: float, accel: float) -> bool:
        """Push LIMITS if it is both due and worth it. Returns whether it did.

        The FIRST call always commits: until then the Teensy is holding
        whatever the last session left it, which is not something to inherit.
        """
        first = self.speed is None
        if not first:
            if t - self._t_last < self.min_interval_s or not self._worth_it(
                    speed, accel):
                self.n_suppressed += 1
                return False
        if self.client is not None:
            self.client.set_limits(speed, accel)
        self.speed, self.accel = speed, accel
        self._t_last = t
        self.n_commits += 1
        return True


# ── Policies ────────────────────────────────────────────────────────────


class HoldCentre:
    """Fallback bot: sit at the middle of the workspace and do nothing else.

    Here so that this runner is testable and demonstrable on its own, before
    ai/airhockey/heuristics.py exists. It is not a strategy and is not meant
    to become one -- a bot that ignores the puck entirely is the honest
    baseline for checking that the plumbing works.
    """

    name = "hold"

    def __init__(self, caps: Caps):
        self.caps = caps

    def __call__(self, obs: dict) -> tuple[float, float, float, float]:
        return (geom.HOME_X, geom.HOME_Y,
                self.caps.speed_max * 0.25, self.caps.accel_max * 0.25)


class TrackY:
    """Fallback bot: hold the goal line and mirror the puck's y.

    Enough motion to prove the command path end to end without any
    prediction: no intercept, no bounce model, just follow. Deleted the day
    heuristics.py has a real defender.
    """

    name = "tracky"

    def __init__(self, caps: Caps):
        self.caps = caps

    def __call__(self, obs: dict) -> tuple[float, float, float, float]:
        hist = obs[OBS_PUCK]
        y = hist[0][1] if hist else geom.HOME_Y
        # Sit on the goal side of the workspace, which is high grid x.
        return (geom.WS_MAX_X, y,
                self.caps.speed_max, self.caps.accel_max)


BUILTIN_BOTS = {"hold": HoldCentre, "tracky": TrackY}


def _bot_config(caps: Caps):
    """A heuristics.BotConfig whose every cap obeys this run's ceiling.

    Lowering the ceiling with --speed/--accel has to reach the BOT, not just
    the clamp below it. A bot that plans a 12000 mm/s strike and then has it
    cut to 8000 arrives late by the ratio and its own timing model is wrong
    about why -- it thinks it committed to a shot it could make. Telling it
    the real envelope instead makes it plan a shot it can.

    Only the cap fields are touched; everything geometric keeps the
    heuristics defaults, which already come from shared/cdpr_geometry.py.
    """
    import dataclasses      # noqa: PLC0415

    from airhockey.heuristics import BotConfig   # noqa: PLC0415

    base = BotConfig()
    return dataclasses.replace(
        base,
        max_speed_mm_s=min(base.max_speed_mm_s, caps.speed_max),
        max_accel_mm_s2=min(base.max_accel_mm_s2, caps.accel_max),
        idle_speed_mm_s=min(base.idle_speed_mm_s, caps.speed_max),
        idle_accel_mm_s2=min(base.idle_accel_mm_s2, caps.accel_max),
        strike_speed_mm_s=min(base.strike_speed_mm_s, caps.speed_max),
        strike_accel_mm_s2=min(base.strike_accel_mm_s2, caps.accel_max),
    )


def load_policy(spec: str, caps: Caps):
    """Turn a --policy string into a callable(obs) -> Command or 4-tuple.

        heuristic:<name>   a bot from ai/airhockey/heuristics.py
        builtin:<name>     one of BUILTIN_BOTS, above
        sac:<run>          NOT IMPLEMENTED -- see _load_sac
    """
    kind, _, name = spec.partition(":")
    if kind == "sac":
        return _load_sac(name, caps)
    if kind == "builtin":
        if name not in BUILTIN_BOTS:
            raise SystemExit(f"no builtin bot {name!r}; have "
                             f"{sorted(BUILTIN_BOTS)}")
        return BUILTIN_BOTS[name](caps)
    if kind != "heuristic":
        raise SystemExit(f"unknown policy kind {kind!r} "
                         "(heuristic: / builtin: / sac:)")

    # Imported lazily so that --selftest, and therefore `pytest ai`, does not
    # depend on the heuristics module loading.
    from airhockey import heuristics       # noqa: PLC0415

    try:
        return heuristics.make_bot(name, _bot_config(caps))
    except ValueError as e:
        raise SystemExit(str(e)) from None


def list_heuristics() -> list[str]:
    from airhockey import heuristics       # noqa: PLC0415
    return sorted(heuristics.BOTS)


def _load_sac(run: str, caps: Caps):
    """STUB. Deliberately not implemented here.

    Everything needed already exists and none of it is in this file:

      airhockey.policy_loader.load_agent(run)   rebuilds the checkpoint
      airhockey.dynamics.table_mm_to_sim(...)   mm -> the sim frame, with
                                                the axis swap the table needs
      airhockey.dynamics.sim_to_table_mm(...)   its exact, UNCLAMPED inverse,
                                                for the action coming back.
                                                Use this rather than a local
                                                conversion: the two compose
                                                to the identity everywhere,
                                                and the clamp belongs to
                                                clamp_action() above, once.
      airhockey.batch_env.BatchAirHockeyEnv     defines OBS_DIM and the
                                                order of the observation
      airhockey.heuristic_bridge.SimBridge      the SAME translation in the
                                                other direction, already
                                                written and round-trip
                                                tested. Read it first; the
                                                cap-band constants an action
                                                encodes are there.

    The work is an adapter with exactly one job: turn the mm report above
    into the observation vector the checkpoint was TRAINED on, and turn the
    agent's normalised [-1, 1] action back into a target in mm plus caps.
    Two things make that more than a unit conversion and are why it is not
    stubbed in as a guess:

      * the observation layout is a property of the checkpoint, and TD-MPC2
        checkpoints do not carry their hyperparameters (see policy_loader);
        building it from the wrong assumption produces plausible motion
        rather than an error.
      * the agent emits a target POSITION only. Caps are not part of its
        action space, so whoever writes this decides whether they come from
        a flag, from the training config, or from a policy retrained to emit
        them -- and that decision belongs with the training run, not here.
    """
    raise NotImplementedError(
        f"--policy sac:{run} is not implemented. See _load_sac() in "
        f"{__file__} for the three pieces it needs and why the observation "
        f"layout cannot be guessed.")


# ── The tick ────────────────────────────────────────────────────────────


def plan(policy, report: ReportBuilder, t: float, caps: Caps,
         prev: tuple[float, float] | None):
    """One command decision. No I/O, so the tests can call it directly."""
    raw = policy(report.observation(t, mallet_fallback=prev))
    # heuristics bots return a Command; the builtins and anything simpler
    # return the same four numbers as a tuple.
    as_tuple = getattr(raw, "as_tuple", None)
    if callable(as_tuple):
        raw = as_tuple()
    if not (isinstance(raw, (tuple, list)) and len(raw) == 4):
        raise TypeError(f"policy returned {raw!r}; expected a 4-tuple "
                        "(x_mm, y_mm, speed_mm_s, accel_mm_s2) or a "
                        "heuristics.Command")
    return clamp_action(Action(*(float(v) for v in raw)), caps, prev)


# ── Selftest ────────────────────────────────────────────────────────────


def _synthetic_puck(duration_s: float = 2.0, fps: float = 200.0):
    """A puck crossing the table and bouncing off one side rail.

    Straight lines with a specular bounce -- the point is not physical
    fidelity (batch_physics does that) but to sweep the report through a
    velocity reversal and drive the policy across the workspace, so that the
    clamp and the cap committer are exercised on a moving target rather than
    on a stationary one.

    Yields (t, x_mm, y_mm) at the camera's frame rate.

    Aimed THROUGH the middle of the table on purpose: that is where the IR
    ring's reflection blinds the tracker, so this is the trajectory that
    makes the caller's dropout branch run. A puck that misses the patch
    tests the easy half only.
    """
    x, y = 200.0, 900.0
    vx, vy = 3000.0, -1559.0         # mm/s, through table centre at t~0.27s
    dt = 1.0 / fps
    lo = geom.RAIL_MIN_Y + geom.PUCK_RADIUS_MM
    hi = geom.RAIL_MAX_Y - geom.PUCK_RADIUS_MM
    n = int(duration_s * fps)
    for k in range(n):
        x += vx * dt
        y += vy * dt
        if y > hi:
            y = 2 * hi - y
            vy = -vy
        elif y < lo:
            y = 2 * lo - y
            vy = -vy
        if x > geom.RAIL_MAX_X - geom.PUCK_RADIUS_MM:
            vx = -abs(vx)
        elif x < geom.RAIL_MIN_X + geom.PUCK_RADIUS_MM:
            vx = abs(vx)
        yield k * dt, x, y


class _Hostile:
    """A bot that asks for things it must not be given.

    The clamp is the only thing between a policy and the machine, so it is
    tested against a policy that is actively wrong rather than only against
    one that is merely imperfect. Cycles through: far outside the workspace
    on each axis, a speed and accel well over the caps, negative caps, and a
    NaN target -- the one case min/max cannot bound.
    """

    def __init__(self, caps: Caps):
        self.caps = caps
        self.k = 0

    def __call__(self, obs):
        self.k += 1
        bad = [
            (1e6, 1e6, 1e6, 1e6),
            (-1e6, -1e6, -1.0, -1.0),
            (geom.WS_MAX_X + 500.0, geom.WS_MIN_Y - 500.0, 0.0, 0.0),
            (float("nan"), 400.0, float("inf"), float("nan")),
        ]
        if self.k % 5 == 0:
            return bad[(self.k // 5) % len(bad)]
        hist = obs[OBS_PUCK]
        y = hist[0][1] if hist else geom.HOME_Y
        # A target that legitimately sweeps the workspace, so the committer
        # sees real cap changes rather than a constant.
        speed = 1000.0 + 6000.0 * (0.5 + 0.5 * math.sin(self.k * 0.05))
        return (geom.WS_MAX_X, y, speed, 0.6 * speed * 4.0)


def selftest(verbose: bool = True) -> dict:
    """The whole chain on a synthetic puck. No camera, no robot, no cv2.

    Returns a stats dict so a test can assert on it; raises AssertionError
    on anything that would have been unsafe on the rig.
    """
    caps = Caps()
    fps, cmd_hz = 200.0, 100.0      # the rig's real rates
    results = {}

    policies = [("builtin:hold", HoldCentre(caps)),
                ("builtin:tracky", TrackY(caps)),
                ("hostile", _Hostile(caps))]
    # Every REAL bot too. This is the part that proves the deployment path:
    # that the dict above is what TrackerReport.coerce accepts, and that the
    # Command coming back is unwrapped and clamped like anything else. A
    # selftest that only ran the builtins would pass while the one interface
    # that matters was broken.
    try:
        policies += [(f"heuristic:{n}", load_policy(f"heuristic:{n}", caps))
                     for n in list_heuristics()]
    except ImportError:
        print("  (airhockey.heuristics not importable — builtins only)")

    for label, policy in policies:
        report = ReportBuilder()
        committer = CapCommitter(client=None)
        prev: tuple[float, float] | None = None
        next_cmd, period = 0.0, 1.0 / cmd_hz
        cmds: list[Action] = []
        flagged = 0

        for t, x, y in _synthetic_puck(fps=fps):
            report.frame()
            # A dropout wherever the IR ring blinds the tracker: the puck is
            # simply not added, exactly as the real loop does it.
            if not (abs(x - geom.GRID_X_MM / 2) < 46.0
                    and abs(y - geom.GRID_Y_MM / 2) < 51.5):
                report.add_puck(t, x, y)
            report.add_mallet(t, geom.HOME_X, geom.HOME_Y)

            if t < next_cmd:
                continue
            next_cmd = t + period
            action, flags = plan(policy, report, t, caps, prev)
            if flags:
                flagged += 1
            committer.maybe_commit(t, action.speed_mm_s, action.accel_mm_s2)
            prev = (action.x_mm, action.y_mm)
            cmds.append(action)

        assert cmds, f"{label}: no commands produced"
        assert report.n_puck < report.n_frames, \
            f"{label}: the puck was seen on every frame — the synthetic " \
            "trajectory missed the glare patch, so the dropout branch of " \
            "the report never ran"
        for a in cmds:
            assert math.isfinite(a.x_mm) and math.isfinite(a.y_mm), \
                f"{label}: non-finite target {a}"
            assert geom.in_workspace(a.x_mm, a.y_mm), \
                f"{label}: target outside the workspace: {a}"
            assert caps.speed_min <= a.speed_mm_s <= caps.speed_max, \
                f"{label}: speed cap out of range: {a}"
            assert caps.accel_min <= a.accel_mm_s2 <= caps.accel_max, \
                f"{label}: accel cap out of range: {a}"

        # History must stay newest-first and inside its window, or a bot
        # reading hist[0] as "now" is reading something else.
        hist = report.observation(report.t_puck)[OBS_PUCK]
        ts = [s[2] for s in hist]
        assert ts == sorted(ts, reverse=True), f"{label}: history not newest-first"
        assert not ts or ts[0] - ts[-1] <= HISTORY_S + 1e-9, \
            f"{label}: history spans {ts[0] - ts[-1]:.3f}s > {HISTORY_S}"

        # The committer's whole job: LIMITS must not track the command rate.
        span = cmds and (len(cmds) / cmd_hz) or 1.0
        assert committer.n_commits <= math.ceil(
            span / committer.min_interval_s) + 1, \
            f"{label}: {committer.n_commits} LIMITS in {span:.2f}s — " \
            "rate limit is not holding"

        results[label] = {
            "commands": len(cmds),
            "clamped": flagged,
            "limits": committer.n_commits,
            "limits_suppressed": committer.n_suppressed,
            "puck_seen": report.n_puck,
            "frames": report.n_frames,
        }
        if verbose:
            r = results[label]
            print(f"  {label:16s} {r['commands']:4d} cmds  "
                  f"{r['clamped']:3d} clamped  "
                  f"{r['limits']:2d} LIMITS "
                  f"({r['limits_suppressed']:4d} suppressed)  "
                  f"puck seen {100 * r['puck_seen'] / r['frames']:3.0f}%")

    assert results["hostile"]["clamped"] > 0, \
        "the hostile bot was never clamped — the clamp is not being reached"

    results["watchdog"] = _watchdog_scenario(caps, fps, cmd_hz, verbose)

    if verbose:
        print("selftest PASSED — every command inside the workspace and the "
              "caps, including the ones the policy got wrong")
    return results


def _watchdog_scenario(caps: Caps, fps: float, cmd_hz: float,
                       verbose: bool) -> dict:
    """Feed a puck, cut it, feed it again. Commands must freeze in the middle.

    The scenario the watchdog exists for is not a dropout -- the loop above
    already covers those -- it is the feed simply stopping while everything
    else keeps running. That is what a puck knocked off the table looks like,
    and what a dead tracker looks like, and in both cases the bot goes on
    producing targets from a history that empties out under it.

    Mirrors the real loop's structure exactly, including that the policy is
    NOT called while blind.
    """
    policy = load_policy("builtin:tracky", caps)
    watchdog = PuckWatchdog(DEFAULT_PUCK_TIMEOUT_S)
    report = ReportBuilder()
    prev: tuple[float, float] | None = None
    next_cmd, period = 0.0, 1.0 / cmd_hz

    # 1.0 s of puck, 3.0 s of nothing, 1.0 s of puck again.
    feed_gap = (1.0, 4.0)
    during_blind: list[tuple[float, float]] = []
    after: list[tuple[float, float]] = []
    trip_target: tuple[float, float] | None = None
    events = []

    for t, x, y in _synthetic_puck(duration_s=5.0, fps=fps):
        report.frame()
        if not feed_gap[0] <= t < feed_gap[1]:
            report.add_puck(t, x, y)
        report.add_mallet(t, geom.HOME_X, geom.HOME_Y)

        if t < next_cmd:
            continue
        next_cmd = t + period

        was_blind = watchdog.blind
        event = watchdog.update(t - report.t_puck)
        if event:
            events.append((t, event))
        if watchdog.blind:
            if not was_blind:
                trip_target = prev
            if prev is not None:
                during_blind.append(prev)
            continue

        action, _flags = plan(policy, report, t, caps, prev)
        prev = (action.x_mm, action.y_mm)
        if t > feed_gap[1]:
            after.append(prev)

    assert watchdog.n_trips == 1, \
        f"watchdog tripped {watchdog.n_trips} times, expected exactly 1"
    assert len(events) == 2, f"expected a trip and a recovery, got {events}"
    assert not watchdog.blind, "watchdog never recovered after the puck came back"

    # THE ASSERTION THAT MATTERS: nothing moved while blind.
    assert during_blind, "the blind window produced no ticks at all"
    assert trip_target is not None
    assert all(c == trip_target for c in during_blind), \
        f"commands changed while the puck was gone: {set(during_blind)}"

    # And it really did resume, rather than staying frozen for ever.
    assert len(set(after)) > 1, \
        "commands never varied again after the puck came back"

    if verbose:
        print(f"  {'watchdog':16s} tripped at {events[0][0]:.2f}s, held "
              f"{len(during_blind):4d} ticks at "
              f"({trip_target[0]:.0f},{trip_target[1]:.0f}), recovered at "
              f"{events[1][0]:.2f}s")
    return {
        "trips": watchdog.n_trips,
        "held_ticks": len(during_blind),
        "trip_target": trip_target,
        "resumed_targets": len(set(after)),
    }


# ── Live / dry-run loop ─────────────────────────────────────────────────

_stop = False


def _sig(_s, _f):
    global _stop
    _stop = True


def run(args) -> int:
    """The real loop. Imports the camera only here, so --selftest need not."""
    sys.path.insert(0, str(ROOT / "vision" / "bin"))
    from mallet_stream import MalletTracker      # noqa: PLC0415
    from puck_stream import BlobStream, PuckTracker  # noqa: PLC0415
    import track_mallet as tm                    # noqa: PLC0415

    caps = Caps(speed_max=args.speed, accel_max=args.accel)
    policy = load_policy(args.policy, caps)

    client = None
    if args.live:
        print("\n" + "!" * 68)
        print("!!  --live: THE ROBOT WILL MOVE. Stand clear of the table.")
        print(f"!!  policy {args.policy}   caps <= {args.speed:.0f} mm/s, "
              f"{args.accel:.0f} mm/s^2")
        print("!!  ctrl-C stops it and brakes the paddle where it stands.")
        print("!" * 68 + "\n")

        # Measure the paddle BEFORE opening the tracker: one process at a
        # time can hold the Spinnaker device, and ENABLE needs this position
        # -- it is the reference every later cable length is measured from,
        # so an error here offsets the whole session.
        print("measuring the paddle for the enable reference...")
        try:
            mx, my = tm.measure()[:2]
        except Exception as e:      # noqa: BLE001
            sys.exit(f"could not measure the paddle ({e}).\n"
                     "  Is the camera free? Stop the tracker view in the "
                     "web UI.")
        print(f"  paddle at ({mx:.1f}, {my:.1f}) mm")

        from airhockey.hardware import CDPRClient   # noqa: PLC0415
        client = CDPRClient()
        try:
            client.connect()
        except OSError as e:
            sys.exit(f"cannot reach cdpr_master on 8421 ({e}) — is it "
                     "running?\n  sw/build/cdpr_master\n"
                     "Run that ALONE — not alongside sw/build/activate, "
                     "which opens the same USB port.")

        if not args.no_enable:
            print("ENABLING the drives (they will hold position, not move)...")
            try:
                client.enable(mx, my)
            except Exception as e:      # noqa: BLE001
                sys.exit(f"enable failed: {e}")
            print("  enabled")
        client.set_ramp(args.ramp)

    tracker = PuckTracker()
    own = MalletTracker(tracker, markers=3)
    opp = MalletTracker(tracker, markers=1) if args.opponent else None
    report = ReportBuilder()
    committer = CapCommitter(client, min_interval_s=args.limits_interval)
    lag = LagMonitor()
    watchdog = PuckWatchdog(args.puck_timeout)

    stream = BlobStream(fps=args.fps, exposure=args.exposure, gain=args.gain,
                        threshold=args.threshold)
    tag = "" if args.live else "  [DRY RUN — commanding nothing]"
    print(f"tracking {stream.width}x{stream.height} at {args.fps:.0f} Hz, "
          f"commanding at {args.cmd_hz:.0f} Hz{tag} — ctrl-C to stop\n")

    period = 1.0 / args.cmd_hz
    next_cmd = 0.0
    prev: tuple[float, float] | None = None
    action = None
    flags: list[str] = []
    n_clamped = 0
    last_print = 0.0
    t_wall = time.time()
    try:
        for _seq, t, blobs in stream:
            if _stop:
                break
            report.frame()
            lag.update(t, time.time())
            msg = lag.warn_once()
            if msg:
                print(msg)

            puck = tracker.update(t, blobs)
            # n_markers is 0 on a coasted frame. Only a real fix is history;
            # see ReportBuilder for why the coast must not be.
            if puck is not None and tracker.n_markers > 0:
                report.add_puck(t, puck[0], puck[1])
            got = own.update(blobs)
            if got is not None:
                report.add_mallet(t, got[0], got[1])
            if opp is not None:
                got = opp.update(blobs)
                if got is not None:
                    report.add_opponent(t, got[0], got[1])

            if t < next_cmd:
                continue
            next_cmd = t + period

            event = watchdog.update(t - report.t_puck)
            if event:
                print(("" if args.live else "[DRY RUN] ") + event)

            if watchdog.blind:
                # Hold the LAST COMMANDED target, and only if there is one.
                # With nothing commanded yet the paddle is wherever ENABLE
                # found it, and inventing a target here -- HOME, say -- would
                # make "the tracker cannot see the puck" a reason to traverse
                # the workspace. Doing nothing is the correct hold.
                if prev is not None and client is not None:
                    try:
                        client.command_position(prev[0], prev[1],
                                                committer.speed)
                    except Exception as e:      # noqa: BLE001
                        print(f"command failed: {e}")
                        break
                if t - last_print > 0.5:
                    last_print = t
                    _status(t, report, action, committer, flags, n_clamped,
                            report.n_frames / max(time.time() - t_wall, 1e-9),
                            args.live, lag, watchdog)
                continue

            action, flags = plan(policy, report, t, caps, prev)
            if flags:
                n_clamped += 1
            prev = (action.x_mm, action.y_mm)
            committer.maybe_commit(t, action.speed_mm_s, action.accel_mm_s2)
            if client is not None:
                try:
                    # The committed speed rides along: the master re-sends
                    # SPEED to the Teensy only when it changes, so in steady
                    # state this costs nothing and after a LIMITS it costs
                    # one redundant round trip at most 5 times a second.
                    client.command_position(action.x_mm, action.y_mm,
                                            committer.speed)
                except Exception as e:      # noqa: BLE001
                    print(f"command failed: {e}")
                    break

            if t - last_print > 0.5:
                last_print = t
                _status(t, report, action, committer, flags, n_clamped,
                        report.n_frames / max(time.time() - t_wall, 1e-9),
                        args.live, lag, watchdog)
    finally:
        stream.close()
        _shutdown(client, args, prev)
    return 0


def _status(t, report, action, committer, flags, n_clamped, rate, live, lag,
            watchdog):
    st = report.staleness(t)
    # Through observation() rather than off the deque, so the operator is
    # shown exactly the history a policy would be given -- expired the same
    # way. On a blind tick the policy is never called, so reading the raw
    # deque here would keep printing samples that have already aged out.
    hist = report.observation(t)[OBS_PUCK]
    puck = (f"({hist[0][0]:6.0f},{hist[0][1]:5.0f}) n={len(hist):2d}"
            if hist else "--                 ")
    age = "  ".join(
        f"{k} {'  --' if math.isinf(v) else f'{1000 * v:4.0f}ms'}"
        for k, v in st.items())
    # `action` is None until the first unblinded tick — a session that starts
    # with no puck in frame is held by the watchdog before it ever plans.
    if watchdog.blind:
        target = "HOLDING            " if action is None else \
            f"HOLD ({action.x_mm:6.0f},{action.y_mm:5.0f})"
    else:
        target = f"({action.x_mm:6.0f},{action.y_mm:5.0f})"
    caps = ("    --/    --" if committer.speed is None
            else f"{committer.speed:5.0f}/{committer.accel:6.0f}")
    print(f"{rate:6.1f} Hz  lag {1000 * lag.lag:4.0f}ms  puck {puck}  {age}"
          f"  -> {target} @ {caps}"
          f"  {'' if live else 'WOULD SEND  '}"
          f"clamped {n_clamped}"
          + (f"  [{','.join(flags)}]" if flags else ""))


# A stop from the 8000 mm/s cap at the 24000 mm/s^2 one takes v/a = 0.33 s,
# plus the jerk ramp. Half a second covers it with margin.
BRAKE_SETTLE_S = 0.5


def _shutdown(client, args, prev):
    """Brake the paddle, then let go of the master.

    BRAKES AT WHATEVER CAPS ARE ALREADY SET, and does not lower them first.
    That looks backwards and is the important detail here: stopping distance
    is v^2/2a, so dropping the accel cap on the way into a stop LENGTHENS it.
    From 8000 mm/s, braking at 24000 mm/s^2 needs 1.3 m; at a "gentle" 2000
    it would need 16 m, which is eight times the table. A shutdown that set
    soft limits before commanding the stop would be the one that drove the
    paddle into the rail.

    Stopping in place rather than driving home, for a related reason: HOME is
    a full-speed traverse across the table, begun at the moment the operator
    pressed ctrl-C and is least expecting the machine to set off somewhere.
    --park home still does it, but only AFTER the brake has settled, so the
    traverse starts from rest and the soft caps then apply to a move that is
    actually slow.

    The drives are left as they were, exactly as goalie_demo does it: this
    process did not energize them beyond ENABLE, and de-energizing drops the
    paddle's cable tension.
    """
    if client is None:
        return
    try:
        # Where it is NOW. Commanding the current position is what makes the
        # firmware's braking curve run; commanding the last target would let
        # it carry on there.
        try:
            bx, by, _vx, _vy = client.get_position()
            bx, by = geom.clamp_to_workspace(bx, by)
        except Exception:      # noqa: BLE001
            bx, by = prev if prev is not None else (geom.HOME_X, geom.HOME_Y)
        client.command_position(bx, by, 0.0)   # 0 = leave the cap alone
        time.sleep(BRAKE_SETTLE_S)
        where = f"({bx:.0f}, {by:.0f})"

        if args.park == "home":
            client.set_limits(args.park_speed, args.park_accel)
            client.command_position(geom.HOME_X, geom.HOME_Y, args.park_speed)
            time.sleep(0.3)
            where = f"HOME ({geom.HOME_X:.0f}, {geom.HOME_Y:.0f})"

        client.close()
        print(f"\nparked at {where}, disconnected "
              "(drives left as they were)")
    except Exception as e:      # noqa: BLE001
        print(f"\nshutdown command failed: {e}\n"
              "  The paddle may still be moving toward its last target. "
              "Ctrl-C cdpr_master.")


def resolve_limits(args) -> tuple[float, float]:
    """The cap ceilings for this run: explicit flag > --gentle > default.

    --speed/--accel default to None rather than to a number so that "the
    user asked for this" and "nobody said" are distinguishable. Without that
    --gentle would have to either lose to a default the user never typed, or
    silently override a value they did.
    """
    speed = args.speed
    if speed is None:
        speed = GENTLE_SPEED if args.gentle else Caps.speed_max
    accel = args.accel
    if accel is None:
        accel = GENTLE_ACCEL if args.gentle else Caps.accel_max
    return speed, accel


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="COMMAND THE ROBOT. Without this nothing is sent.")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic puck through the whole chain; no camera, "
                         "no robot")
    ap.add_argument("--policy", default="builtin:hold",
                    help="heuristic:<name> | builtin:<name> | sac:<run>. "
                         "--list shows the names. The default sits still on "
                         "purpose.")
    ap.add_argument("--list", action="store_true",
                    help="print the available policies and exit")
    ap.add_argument("--opponent", action="store_true",
                    help="also track the human's single-dot mallet")
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    ap.add_argument("--cmd-hz", type=float, default=100.0,
                    help="command rate to the Teensy; tracking stays at --fps")
    ap.add_argument("--gentle", action="store_true",
                    help=f"first-live-run preset: {GENTLE_SPEED:.0f} mm/s, "
                         f"{GENTLE_ACCEL:.0f} mm/s^2. USE THIS FOR THE FIRST "
                         "LIVE RUN OF ANY NEW POLICY. An explicit --speed or "
                         "--accel still wins.")
    ap.add_argument("--speed", type=float, default=None,
                    help=f"CEILING on the speed cap a policy may ask for, "
                         f"mm/s (default {Caps.speed_max:.0f})")
    ap.add_argument("--accel", type=float, default=None,
                    help=f"CEILING on the accel cap a policy may ask for, "
                         f"mm/s^2 (default {Caps.accel_max:.0f}). That "
                         "default is above the ~15120 at which the paddle "
                         "tips; drop it if the paddle hops.")
    ap.add_argument("--puck-timeout", type=float,
                    default=DEFAULT_PUCK_TIMEOUT_S,
                    help="seconds without a puck fix before the policy is "
                         "paused and the paddle holds. Not the dropout "
                         "window (see STALE_S); this is for a puck that has "
                         "gone away rather than blinked.")
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="jerk-limit ramp, ms")
    ap.add_argument("--limits-interval", type=float,
                    default=CapCommitter.MIN_INTERVAL_S,
                    help="minimum seconds between LIMITS round trips. Each "
                         "costs ~2-6 ms of the command tick; see CapCommitter.")
    ap.add_argument("--park", choices=("stop", "home"), default="stop",
                    help="on exit: brake in place, or brake and THEN traverse "
                         "to HOME. Both brake first; see _shutdown for why "
                         "the brake cannot be the gentle one.")
    ap.add_argument("--park-speed", type=float, default=500.0,
                    help="--park home only: speed for the traverse, mm/s")
    ap.add_argument("--park-accel", type=float, default=2000.0,
                    help="--park home only: accel for the traverse, mm/s^2. "
                         "NOT used for the brake — a low cap there would "
                         "lengthen the stop, not shorten it.")
    ap.add_argument("--no-enable", action="store_true",
                    help="assume the drives are already energized")
    args = ap.parse_args()

    if args.list:
        for n in sorted(BUILTIN_BOTS):
            print(f"  builtin:{n}")
        for n in list_heuristics():
            print(f"  heuristic:{n}")
        print("  sac:<run>          not implemented — see _load_sac()")
        return 0

    if args.selftest:
        selftest()
        return 0

    args.speed, args.accel = resolve_limits(args)
    if args.gentle:
        print(f"--gentle: caps held to {args.speed:.0f} mm/s, "
              f"{args.accel:.0f} mm/s^2")

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
