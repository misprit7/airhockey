"""Heuristic (non-learned) bots for the ROBOT side.

Written against the TRACKER, not against the simulator. Every bot in here is a
function

    (puck position history in mm, own mallet in mm, opponent mallet in mm)
        -> (target_x_mm, target_y_mm, speed_mm_s, accel_mm_s2)

which is exactly what `vision/bin/puck_stream.py` produces on one side and what
`CDPRClient.set_limits` + `command_position` consume on the other. Nothing here
imports the sim, reads a velocity it was handed, or knows the action space of an
RL environment: a bot that cannot be dropped straight into the camera loop is
not a baseline for a policy that has to run there.

The sim glue lives in `airhockey/heuristic_bridge.py`, deliberately in a
separate file so this one stays importable on the table with no gymnasium and
no environment.

WHAT IS MODELLED, AND WHY IT IS MORE THAN demo_goalie.py's STRAIGHT LINES
------------------------------------------------------------------------
`demo_goalie.py` predicts with perfectly elastic specular walls and constant
speed, and argues -- correctly -- that a goalie only needs the arrival POINT,
which a lossless model gets right as long as the puck never bounces. The moment
it bounces that stops being true, because the measured rail is not specular:

    normal component  x  0.785   (wall_restitution)
    tangent component x  0.66    (wall_tangential)

so the outgoing ray is steeper than the incoming one by e/t = 1.19. A one-bounce
prediction made with specular reflection lands in the wrong place, and a
two-bounce one lands somewhere else entirely. `predict_crossing` below walks the
bounces with the measured coefficients instead of folding a triangle wave.

Drag is modelled too, but only for TIMING. decel = mu*g + b*v^2 is collinear
with the velocity, so it cannot bend the path -- it only changes when the puck
arrives. Dropping the small rolling term leaves v dv/ds = -b v^2, i.e.
v(s) = v0 exp(-b s) and t(s) = (exp(b s) - 1) / (b v0), which is closed form and
exact for the dominant term. That matters for the strikers, which are choosing
an intercept TIME, and not at all for a goalie, which is choosing a place.

All measured constants come from TableConfig; none are restated here.

FEED THESE BOTS THE DENSEST PUCK HISTORY YOU HAVE. `estimate_velocity` cuts
its fit at a rail bounce, and the cut can only fire on a segment that
STRADDLES the reversal -- so the resolution of the history sets how precisely
a bounce can be located. The tracker's native 200 Hz ring handles a bounce of
any age. A sparse history does not: `BatchAirHockeyEnv.HISTORY_PUCK_LAGS`
samples at 0/10/20/50/100 ms, and a bounce 40 ms old falls INSIDE the 20-50 ms
segment, where the reversal averages away and the estimate comes back about
50% wrong (measured: 957 mm/s of error against 7 for the raw ring). That is a
hole in the observation rather than in the estimator, and it applies to
anything reading those lags -- a learned policy included.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

from airhockey.physics import TableConfig  # noqa: E402

_TABLE = TableConfig()

# ── Table facts, in the units the tracker and the Teensy both speak ──────
WALL_RESTITUTION = _TABLE.wall_restitution        # normal component survives
WALL_TANGENTIAL = _TABLE.wall_tangential          # tangential component survives
DRAG_B_PER_MM = _TABLE.PUCK_DRAG_B / 1000.0       # SI b (per m) -> per mm
ROLLING_DECEL_MM_S2 = _TABLE.puck_friction * 9810.0

PUCK_RADIUS_MM = geom.PUCK_RADIUS_MM
GOAL_HALF_WIDTH_MM = geom.GOAL_WIDTH_MM / 2.0
GOAL_CENTER_Y_MM = (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y) / 2.0

# Own goal is at HIGH grid x (the robot end), the opponent's at low x, so +vx
# is "closing" throughout this file.
OWN_GOAL_X_MM = geom.RAIL_MAX_X
OPP_GOAL_X_MM = geom.RAIL_MIN_X

# ── Machine ceilings ─────────────────────────────────────────────────────
# Speed is the firmware clamp (fw/include/cdpr_config.h MAX_VELOCITY_MM_S) and
# the Teensy enforces it absolutely. Accel is NOT the firmware's 120000: what
# binds is what four cables can make at a given pose, which cdpr_config.h's own
# solve puts near 114000 at the table centre and near 9000 at the worst corner.
# 60000 is the top of the sim's randomisation band and already optimistic near
# the edges; the default is well below it because a jerk-limited profile that
# never uses its ceiling is what keeps the mallet from tipping (g*r/h ~ 15000).
MAX_SPEED_MM_S = 12000.0
MAX_ACCEL_MM_S2 = 60000.0
DEFAULT_ACCEL_MM_S2 = 20000.0

# Below this much TRAVEL, a sign change between two frames is centroid noise
# rather than a bounce.
#
# In MILLIMETRES, deliberately, and not in mm/s. The noise on a displacement
# between two tracked positions is sqrt(2) x 0.35 = 0.50 mm no matter how far
# apart in time those positions are, so a millimetre threshold means the same
# thing at any sample spacing. A VELOCITY threshold does not: the same 0.50 mm
# is 50 mm/s across a 10 ms gap and 100 mm/s across a 5 ms one, so a constant
# in mm/s silently tightens as the history gets denser. That is not
# hypothetical -- at the tracker's native 200 Hz the old 50 mm/s constant fired
# on 33% of ticks for a slow puck travelling in a straight line, cutting the
# fit to two frames and tripling its own error. 1.5 mm is 3 sigma.
BOUNCE_EPS_MM = 1.5


# ══ The interface ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PuckSample:
    """One tracker frame: where the puck was, and when."""

    x_mm: float
    y_mm: float
    t_s: float


@dataclass(frozen=True)
class TrackerReport:
    """Everything a bot is allowed to look at.

    `puck` is NEWEST FIRST, which is the order a ring buffer reads out and the
    order `puck_stream.py` accumulates. An empty history means the puck is not
    visible -- not that it is at the origin.
    """

    puck: tuple[PuckSample, ...] = ()
    mallet: tuple[float, float] = (geom.HOME_X, geom.HOME_Y)
    opponent: tuple[float, float] | None = None
    # Now, on the tracker's clock. Only needed when the puck is INVISIBLE:
    # otherwise the newest sample's timestamp is the clock, and a bot that took
    # its time from the puck alone would freeze its swing latches for exactly
    # as long as the puck was lost.
    t_s: float | None = None

    @classmethod
    def coerce(cls, obj) -> "TrackerReport":
        """Accept the dict form as well, since that is what crosses a socket."""
        if isinstance(obj, cls):
            return obj
        puck = tuple(
            s if isinstance(s, PuckSample) else PuckSample(*s)
            for s in (obj.get("puck") or ())
        )
        opp = obj.get("opponent")
        return cls(
            puck=puck,
            mallet=tuple(obj["mallet"]),
            opponent=None if opp is None else tuple(opp),
            t_s=obj.get("t_s"),
        )


@dataclass(frozen=True)
class Command:
    """What goes to the Teensy: a target and the caps to get there with."""

    x_mm: float
    y_mm: float
    speed_mm_s: float
    accel_mm_s2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_mm, self.y_mm, self.speed_mm_s, self.accel_mm_s2)


@dataclass(frozen=True)
class PuckEstimate:
    """Position now, velocity fitted over the recent history."""

    x_mm: float
    y_mm: float
    vx_mm_s: float
    vy_mm_s: float
    n_samples: int

    @property
    def speed_mm_s(self) -> float:
        return math.hypot(self.vx_mm_s, self.vy_mm_s)


@dataclass(frozen=True)
class Crossing:
    """Where a predicted puck meets a vertical line, and when."""

    y_mm: float
    eta_s: float
    speed_mm_s: float
    bounces: int


@dataclass
class BotConfig:
    """One config for all the bots; each uses the subset it cares about.

    The radii are configurable because they describe the MACHINE, and the
    simulator's paddle is 80 mm across where the real mallet is 100.8. A bot
    that assumes the wrong contact distance aims through the puck by the
    difference, which is a systematic shot error, not noise.
    """

    # Geometry
    mallet_radius_mm: float = geom.MALLET_RADIUS_MM
    puck_radius_mm: float = PUCK_RADIUS_MM
    ws_min_x: float = geom.WS_MIN_X
    ws_max_x: float = geom.WS_MAX_X
    ws_min_y: float = geom.WS_MIN_Y
    ws_max_y: float = geom.WS_MAX_Y

    # Where the goalie line sits. As close to the goal as the cables allow;
    # the workspace stops 100 mm short of the goal line no matter what.
    defend_margin_mm: float = 15.0

    # Caps
    max_speed_mm_s: float = MAX_SPEED_MM_S
    max_accel_mm_s2: float = MAX_ACCEL_MM_S2
    idle_speed_mm_s: float = 3000.0
    idle_accel_mm_s2: float = 8000.0
    strike_speed_mm_s: float = MAX_SPEED_MM_S
    strike_accel_mm_s2: float = MAX_ACCEL_MM_S2

    # Velocity estimation
    vel_window_s: float = 0.06

    # Engagement, gated on the predicted ARRIVAL TIME rather than on the
    # puck's speed.
    #
    # demo_goalie gates on closing speed -- below 150 mm/s the puck is
    # "drifting, not shot" and the rig rests, which keeps it from twitching at
    # centroid noise all day. That rule costs goals, and the tournament showed
    # where: a puck trickling toward the net at 100 mm/s is not a threat by
    # speed and is certain by geometry. ETA folds speed and distance together
    # and is the quantity that actually decides both questions -- whether the
    # puck is coming, and whether the prediction is worth trusting, since the
    # velocity error the tracker leaves integrates over exactly this long.
    #
    # Hysteresis is still needed, just on ETA: engage inside the first, release
    # outside the second, so a puck hovering at the boundary does not chatter.
    engage_horizon_s: float = 0.9
    release_horizon_s: float = 1.3
    max_horizon_s: float = 1.5
    # Below this the closing speed is noise, not motion, and dividing by it
    # produces an ETA of minutes.
    min_closing_mm_s: float = 20.0
    deadband_mm: float = 3.0

    # How far the resting goalie shades toward the puck's side of the table.
    shade_gain: float = 0.6

    # Urgency: cover `d` in `eta` needs about 4d/eta^2 of accel and 2d/eta of
    # speed on a triangular profile; ask for a bit more than that.
    urgency_safety: float = 1.5

    # Striking
    attack_max_puck_speed_mm_s: float = 1800.0
    attack_min_x_mm: float = geom.CENTERLINE_X
    # How far PAST the contact point the swing is aimed. The single most
    # important number in the striker, and not for the reason it looks like:
    # 150, 250 and 400 all hit the puck about as hard (the mallet is at speed
    # by the contact either way), but they leave the mallet 150, 250 and
    # 400 mm from where it started. Goals conceded went 0.12 -> 0.5 -> 0.7 per
    # game across that range while goals scored barely moved. Follow-through
    # is bought with position, and position is what defends the net.
    follow_through_mm: float = 150.0
    commit_s: float = 0.18
    # Only swing if the contact point is this far inside the workspace, so a
    # clamped target does not quietly turn a shot into a different shot.
    strike_clearance_mm: float = 10.0

    # Intercepting. These three were swept against the scripted opponents and
    # every one of them landed on the CONSERVATIVE end, which is the finding
    # rather than the tuning: an interceptor is only worth its risk when it
    # meets the puck soon (0.3 s, not 0.6) and arrives well early (0.12 s of
    # margin). Reaching further up-table found more intercepts and lost more
    # goals, because the extra ones were the ones it got wrong.
    intercept_step_s: float = 0.02
    intercept_max_s: float = 0.3
    intercept_margin_s: float = 0.12
    # How much later than the puck the mallet may get back to the defence line
    # after a swing that misses. Zero means "only swing when the miss is fully
    # survivable", and the sweep says zero: 60 ms of slack bought a handful of
    # extra swings and cost more than they returned. Conceding is expensive
    # and the swings this rules out are the marginal ones.
    recover_slack_s: float = 0.0


# ══ Prediction maths ═════════════════════════════════════════════════════

def puck_bounds(puck_radius_mm: float = PUCK_RADIUS_MM):
    """The box the puck's CENTRE lives in. Bounces reflect the centre off a
    line one radius in from the rail, not off the rail."""
    r = puck_radius_mm
    return (geom.RAIL_MIN_X + r, geom.RAIL_MAX_X - r,
            geom.RAIL_MIN_Y + r, geom.RAIL_MAX_Y - r)


def fold(value: float, lo: float, hi: float) -> float:
    """Reflect `value` into [lo, hi] as many times as needed.

    The LOSSLESS answer -- a triangle wave in closed form. Kept because it is
    the limit `predict_crossing` must reduce to when both rail coefficients are
    1 and drag is off, which is the cheapest available check that the bounce
    walk is not subtly wrong.
    """
    span = hi - lo
    if span <= 0:
        return lo
    u = (value - lo) % (2.0 * span)
    return lo + (u if u <= span else 2.0 * span - u)


def travel_time_s(distance_mm: float, speed_mm_s: float,
                  drag_b_per_mm: float = DRAG_B_PER_MM) -> float:
    """How long a puck takes to cover `distance_mm` of PATH, with drag.

    v dv/ds = -b v^2 gives v(s) = v0 exp(-b s), hence this. Exact for the
    quadratic term, which is the one that matters: at 6 m/s aerodynamic drag is
    1250 mm/s^2 against 15 of rolling.
    """
    if speed_mm_s <= 0.0:
        return math.inf
    if drag_b_per_mm <= 0.0:
        return distance_mm / speed_mm_s
    return math.expm1(drag_b_per_mm * distance_mm) / (drag_b_per_mm * speed_mm_s)


def travel_distance_mm(time_s: float, speed_mm_s: float,
                       drag_b_per_mm: float = DRAG_B_PER_MM) -> float:
    """Inverse of `travel_time_s`: path covered in `time_s`."""
    if speed_mm_s <= 0.0:
        return 0.0
    if drag_b_per_mm <= 0.0:
        return speed_mm_s * time_s
    return math.log1p(drag_b_per_mm * speed_mm_s * time_s) / drag_b_per_mm


def predict_crossing(
    x: float, y: float, vx: float, vy: float, target_x: float,
    y_lo: float, y_hi: float, *,
    restitution: float = WALL_RESTITUTION,
    tangential: float = WALL_TANGENTIAL,
    drag_b_per_mm: float = DRAG_B_PER_MM,
    max_bounces: int = 6,
) -> Crossing | None:
    """Where and when a puck at (x, y) moving (vx, vy) crosses x = target_x.

    Walks the side-wall bounces one at a time rather than folding, because the
    rail is lossy in BOTH components and the two losses differ: the outgoing
    ray is steeper than the incoming one by restitution/tangential. Returns
    None if the puck is not heading there at all, or needs more than
    `max_bounces` to arrive (a prediction through seven rails is noise).
    """
    speed = math.hypot(vx, vy)
    if speed <= 0.0:
        return None
    dx, dy = vx / speed, vy / speed
    t_total = 0.0
    # A noisy centroid can put the reported puck a millimetre outside the
    # rail. Starting there gives a negative distance-to-wall and the walk
    # steps BACKWARDS; clamp instead.
    y = min(max(y, y_lo), y_hi)

    for bounces in range(max_bounces + 1):
        if dx <= 0.0:
            return None                      # moving away, or parallel
        s_target = (target_x - x) / dx
        if s_target < 0.0:
            return None                      # already past it
        if dy > 0.0:
            s_wall = max((y_hi - y) / dy, 0.0)
        elif dy < 0.0:
            s_wall = max((y_lo - y) / dy, 0.0)
        else:
            s_wall = math.inf

        if s_target <= s_wall:
            return Crossing(
                y_mm=y + dy * s_target,
                eta_s=t_total + travel_time_s(s_target, speed, drag_b_per_mm),
                speed_mm_s=speed * math.exp(-drag_b_per_mm * s_target),
                bounces=bounces,
            )

        x += dx * s_wall
        y = y_hi if dy > 0.0 else y_lo
        t_total += travel_time_s(s_wall, speed, drag_b_per_mm)
        speed *= math.exp(-drag_b_per_mm * s_wall)

        cvx, cvy = dx * speed * tangential, -dy * speed * restitution
        speed = math.hypot(cvx, cvy)
        if speed < 1e-9:
            return None
        dx, dy = cvx / speed, cvy / speed

    return None


def advance_puck(
    x: float, y: float, vx: float, vy: float, dt: float, *,
    puck_radius_mm: float = PUCK_RADIUS_MM,
    restitution: float = WALL_RESTITUTION,
    tangential: float = WALL_TANGENTIAL,
    drag_b_per_mm: float = DRAG_B_PER_MM,
    max_bounces: int = 6,
) -> tuple[float, float, float, float]:
    """Free-flight state `dt` seconds from now: all four rails, with losses.

    The goal mouths are holes, not walls -- a puck heading into one keeps
    going, and the caller finds out by the x it comes back with. Used by the
    strikers, which have to know where a slow puck WILL be rather than where it
    is; the goalie only ever needs `predict_crossing`.
    """
    x_lo, x_hi, y_lo, y_hi = puck_bounds(puck_radius_mm)
    speed = math.hypot(vx, vy)
    if speed <= 0.0 or dt <= 0.0:
        return x, y, vx, vy
    dx, dy = vx / speed, vy / speed

    for _ in range(max_bounces + 1):
        # Nearest rail along the current heading, ignoring the ones behind.
        s_hit, axis = math.inf, None
        for lo, hi, d, ax in ((x_lo, x_hi, dx, "x"), (y_lo, y_hi, dy, "y")):
            pos = x if ax == "x" else y
            if d > 0.0:
                s = (hi - pos) / d
            elif d < 0.0:
                s = (lo - pos) / d
            else:
                continue
            if 0.0 <= s < s_hit:
                s_hit, axis = s, ax

        s_free = travel_distance_mm(dt, speed, drag_b_per_mm)
        if axis is None or s_free <= s_hit:
            v_end = speed * math.exp(-drag_b_per_mm * s_free)
            return x + dx * s_free, y + dy * s_free, dx * v_end, dy * v_end

        x += dx * s_hit
        y += dy * s_hit
        dt -= travel_time_s(s_hit, speed, drag_b_per_mm)
        speed *= math.exp(-drag_b_per_mm * s_hit)

        if axis == "x" and abs(y - GOAL_CENTER_Y_MM) < GOAL_HALF_WIDTH_MM:
            # Into the mouth: no rail there. Coast out of the table and let the
            # caller notice, rather than inventing a bounce off a goal.
            s_free = travel_distance_mm(max(dt, 0.0), speed, drag_b_per_mm)
            v_end = speed * math.exp(-drag_b_per_mm * s_free)
            return x + dx * s_free, y + dy * s_free, dx * v_end, dy * v_end

        if axis == "x":
            cvx, cvy = -dx * speed * restitution, dy * speed * tangential
        else:
            cvx, cvy = dx * speed * tangential, -dy * speed * restitution
        speed = math.hypot(cvx, cvy)
        if speed < 1e-9:
            return x, y, 0.0, 0.0
        dx, dy = cvx / speed, cvy / speed

    return x, y, dx * speed, dy * speed


def reach_time_s(distance_mm: float, max_speed_mm_s: float,
                 max_accel_mm_s2: float) -> float:
    """Mallet time to REACH a point, not to stop on it.

    A strike wants to be moving through the contact, and a block works fine
    while still travelling, so nothing in here needs the decelerating half of a
    trapezoid. Using one would make every intercept look 40% less reachable
    than it is.
    """
    if distance_mm <= 0.0:
        return 0.0
    if max_accel_mm_s2 <= 0.0 or max_speed_mm_s <= 0.0:
        return math.inf
    d_accel = max_speed_mm_s ** 2 / (2.0 * max_accel_mm_s2)
    if distance_mm <= d_accel:
        return math.sqrt(2.0 * distance_mm / max_accel_mm_s2)
    return (max_speed_mm_s / max_accel_mm_s2
            + (distance_mm - d_accel) / max_speed_mm_s)


def estimate_velocity(samples, window_s: float = 0.06,
                      bounce_eps_mm: float = BOUNCE_EPS_MM
                      ) -> PuckEstimate | None:
    """Least-squares slope over the recent history, cut at a bounce.

    Two frames would do arithmetically and be swamped by the 0.35 mm of
    back-projection noise; a window that spans a bounce would average the
    incoming and outgoing legs into a velocity the puck never had, and it does
    so exactly when the goalie most needs the answer. So: take the newest run
    of samples whose consecutive DISPLACEMENTS agree in sign, inside the
    window, and fit that.

    Displacements rather than velocities, so that the reversal test means the
    same thing whether the caller feeds 5 ms frames or 50 ms ones -- see
    BOUNCE_EPS_MM. Signs are identical either way (time only ever runs
    forward); it is the threshold that has to be spacing-free.
    """
    if not samples:
        return None
    if len(samples) == 1:
        s = samples[0]
        return PuckEstimate(s.x_mm, s.y_mm, 0.0, 0.0, 1)

    t0 = samples[0].t_s
    window = [s for s in samples if t0 - s.t_s <= window_s + 1e-9]
    if len(window) < 2:
        window = list(samples[:2])

    # Walk newest -> oldest, stopping at the first segment that reverses
    # either component relative to the newest one.
    def seg(a, b):
        if a.t_s <= b.t_s:
            return None
        return (a.x_mm - b.x_mm, a.y_mm - b.y_mm)

    first = seg(window[0], window[1])
    if first is None:
        s = samples[0]
        return PuckEstimate(s.x_mm, s.y_mm, 0.0, 0.0, 1)
    kept = 2
    for i in range(1, len(window) - 1):
        nxt = seg(window[i], window[i + 1])
        if nxt is None:
            break
        reversed_axis = any(
            a * b < 0.0 and abs(a) > bounce_eps_mm and abs(b) > bounce_eps_mm
            for a, b in zip(first, nxt)
        )
        if reversed_axis:
            break
        kept = i + 2
    fit = window[:kept]

    n = len(fit)
    mt = sum(s.t_s for s in fit) / n
    den = sum((s.t_s - mt) ** 2 for s in fit)
    newest = samples[0]
    if den <= 0.0:
        return PuckEstimate(newest.x_mm, newest.y_mm, 0.0, 0.0, n)
    mx = sum(s.x_mm for s in fit) / n
    my = sum(s.y_mm for s in fit) / n
    vx = sum((s.t_s - mt) * (s.x_mm - mx) for s in fit) / den
    vy = sum((s.t_s - mt) * (s.y_mm - my) for s in fit) / den
    return PuckEstimate(newest.x_mm, newest.y_mm, vx, vy, n)


# ══ Bots ═════════════════════════════════════════════════════════════════

class Bot:
    """Base: workspace clamping, velocity estimation, defensive positioning.

    Callable, and stateful only in ways a real controller has to be -- the
    engage/release hysteresis and the swing commitment latch. `reset()` clears
    all of it, and two bots given the same reports from the same reset state
    produce the same commands.
    """

    name = "bot"

    def __init__(self, cfg: BotConfig | None = None):
        self.cfg = cfg or BotConfig()
        self.reset()

    def reset(self) -> None:
        self.engaged = False
        self.last_target = (self.defend_x, GOAL_CENTER_Y_MM)
        self.last_eta: float | None = None
        self._commit_until = -math.inf
        self._commit_cmd: Command | None = None
        self._t = 0.0

    # -- geometry helpers ------------------------------------------------

    @property
    def defend_x(self) -> float:
        return self.cfg.ws_max_x - self.cfg.defend_margin_mm

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        c = self.cfg
        return (min(max(x, c.ws_min_x), c.ws_max_x),
                min(max(y, c.ws_min_y), c.ws_max_y))

    def inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        c = self.cfg
        return (c.ws_min_x + margin <= x <= c.ws_max_x - margin
                and c.ws_min_y + margin <= y <= c.ws_max_y - margin)

    def contact_distance_mm(self) -> float:
        return self.cfg.mallet_radius_mm + self.cfg.puck_radius_mm

    # -- calling convention ----------------------------------------------

    def __call__(self, report) -> Command:
        rep = TrackerReport.coerce(report)
        if rep.t_s is not None:
            self._t = rep.t_s
        elif rep.puck:
            self._t = rep.puck[0].t_s
        return self.command(rep)

    def command(self, rep: TrackerReport) -> Command:
        raise NotImplementedError

    # -- shared behaviour -------------------------------------------------

    def estimate(self, rep: TrackerReport) -> PuckEstimate | None:
        return estimate_velocity(rep.puck, self.cfg.vel_window_s)

    def raw_crossing(self, est: PuckEstimate) -> Crossing | None:
        """The crossing with NO horizon cap.

        Defence ignores a puck two seconds away, because the prediction is not
        worth acting on that far out. Deciding whether to leave the goal is a
        different question with a different answer: a puck arriving in 2 s is
        exactly the one a swing must be finished before, and treating "too far
        out to defend" as "nothing is coming" is how a striker walks off its
        line into a slow goal.
        """
        c = self.cfg
        if est.vx_mm_s < c.min_closing_mm_s:
            return None
        _, _, y_lo, y_hi = puck_bounds(c.puck_radius_mm)
        return predict_crossing(est.x_mm, est.y_mm, est.vx_mm_s, est.vy_mm_s,
                                self.defend_x, y_lo, y_hi)

    def horizon_capped(self, hit: Crossing | None) -> Crossing | None:
        """Drop a crossing too far out to be worth defending against."""
        return None if hit is None or hit.eta_s > self.cfg.max_horizon_s else hit

    def crossing(self, est: PuckEstimate) -> Crossing | None:
        """Where the puck meets the defence line, if it is coming SOON."""
        return self.horizon_capped(self.raw_crossing(est))

    def update_engagement(self, eta_s: float | None) -> bool:
        c = self.cfg
        if eta_s is None:
            self.engaged = False
        elif self.engaged:
            if eta_s > c.release_horizon_s:
                self.engaged = False
        elif eta_s <= c.engage_horizon_s:
            self.engaged = True
        return self.engaged

    def urgency(self, distance_mm: float, eta_s: float) -> tuple[float, float]:
        """Speed and accel caps to cover `distance_mm` inside `eta_s`."""
        c = self.cfg
        if eta_s <= 1e-3:
            return c.max_speed_mm_s, c.max_accel_mm_s2
        k = c.urgency_safety
        speed = k * 2.0 * distance_mm / eta_s
        accel = k * 4.0 * distance_mm / (eta_s * eta_s)
        return (min(max(speed, c.idle_speed_mm_s), c.max_speed_mm_s),
                min(max(accel, c.idle_accel_mm_s2), c.max_accel_mm_s2))

    def rest_command(self, est: PuckEstimate | None) -> Command:
        """Stand on the line, shaded toward the puck's side.

        Not a fixed rest point: the puck's y is the only information available
        about where the next shot comes from, and a goalie already leaning the
        right way covers a shot it could not otherwise reach. The gain is below
        1 so a puck in the corner does not pull the mallet off its own goal.
        """
        c = self.cfg
        if est is None:
            ty = GOAL_CENTER_Y_MM
        else:
            ty = GOAL_CENTER_Y_MM + c.shade_gain * (est.y_mm - GOAL_CENTER_Y_MM)
        tx, ty = self.clamp(self.defend_x, ty)
        return self.settle(tx, ty, c.idle_speed_mm_s, c.idle_accel_mm_s2)

    def settle(self, x: float, y: float, speed: float, accel: float) -> Command:
        """Apply the deadband and record the target.

        Sub-millimetre corrections are centroid noise, and a jerk-limited
        profile will happily chase them forever; the rig rings instead of
        standing still.
        """
        c = self.cfg
        if (abs(x - self.last_target[0]) < c.deadband_mm
                and abs(y - self.last_target[1]) < c.deadband_mm):
            x, y = self.last_target
        else:
            self.last_target = (x, y)
        return Command(x, y, min(speed, c.max_speed_mm_s),
                       min(accel, c.max_accel_mm_s2))

    def defend(self, est: PuckEstimate, rep: TrackerReport,
               hit: Crossing | None) -> Command:
        """The goalie play: be where the puck will cross the defence line.

        `hit` is passed in rather than solved here, and has no default: the
        strikers need the same crossing to decide whether a swing is safe, and
        a defaulted argument would make "nobody solved it" and "there is
        nothing coming" the same call.
        """
        if not self.update_engagement(None if hit is None else hit.eta_s):
            self.last_eta = None
            return self.rest_command(est)

        self.last_eta = hit.eta_s
        tx, ty = self.clamp(self.defend_x, hit.y_mm)
        mx, my = rep.mallet
        speed, accel = self.urgency(math.hypot(tx - mx, ty - my), hit.eta_s)
        return self.settle(tx, ty, speed, accel)

    # -- aiming ------------------------------------------------------------

    def aim_point(self, rep: TrackerReport) -> tuple[float, float]:
        """A point in the opponent's mouth, biased away from their mallet.

        The mouth is 380 mm and the mallet is 100 wide, so there is almost
        always a side of it that is open; shooting at the centre because the
        centre is easy to write down is shooting at where a goalkeeper stands.
        """
        margin = self.cfg.mallet_radius_mm
        lo = GOAL_CENTER_Y_MM - GOAL_HALF_WIDTH_MM + margin
        hi = GOAL_CENTER_Y_MM + GOAL_HALF_WIDTH_MM - margin
        if rep.opponent is None:
            return OPP_GOAL_X_MM, GOAL_CENTER_Y_MM
        oy = rep.opponent[1]
        return OPP_GOAL_X_MM, (lo if abs(lo - oy) > abs(hi - oy) else hi)

    def strike_command(self, px: float, py: float, aim: tuple[float, float],
                       rep: TrackerReport) -> Command | None:
        """Drive THROUGH a puck at (px, py) toward `aim`, at full tilt.

        Returns None if the contact point is outside the reach: a clamped
        strike target is a different shot from the one that was planned, and
        silently taking it is how a bot ends up nudging the puck sideways into
        its own half.
        """
        c = self.cfg
        ax, ay = aim
        dx, dy = ax - px, ay - py
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        ux, uy = dx / n, dy / n
        reach = self.contact_distance_mm()
        cx, cy = px - ux * reach, py - uy * reach
        if not self.inside(cx, cy, c.strike_clearance_mm):
            return None
        tx, ty = self.clamp(cx + ux * c.follow_through_mm,
                            cy + uy * c.follow_through_mm)
        return Command(tx, ty, c.strike_speed_mm_s, c.strike_accel_mm_s2)

    def recovery_ok(self, cmd: Command, hit: Crossing | None,
                    strike_at_s: float) -> bool:
        """Could the mallet still make the save if this swing misses?

        A swing ends with the mallet a follow-through UP-TABLE of the contact
        point, which is the wrong side of a puck it failed to touch. This is
        the whole difference between a striker that trades goals and one that
        wins: swing when the miss is survivable, defend when it is not.

        The budget is the puck's own ETA at the defence line, against the time
        to finish the swing plus the time to get from where it ends back to
        where the puck is predicted to cross. When the puck is not closing at
        all there is nothing to recover from and this is vacuously true.
        """
        c = self.cfg
        if hit is None:
            return True
        gx, gy = self.clamp(self.defend_x, hit.y_mm)
        back = reach_time_s(math.hypot(cmd.x_mm - gx, cmd.y_mm - gy),
                            c.strike_speed_mm_s, c.strike_accel_mm_s2)
        return strike_at_s + c.commit_s + back <= hit.eta_s + c.recover_slack_s

    def commit(self, cmd: Command, duration_s: float) -> Command:
        """Latch a swing so it completes instead of being re-planned at 100 Hz.

        Re-solving the intercept every tick with a fresh noisy velocity moves
        the target a few mm each time, and the profile spends the whole swing
        re-accelerating toward a point that keeps stepping sideways. The puck
        gets tapped rather than struck.
        """
        self._commit_until = self._t + duration_s
        self._commit_cmd = cmd
        # Swings bypass the positional deadband -- the whole point is to move
        # a long way fast -- but the latched target still has to be what the
        # deadband measures against, or the first defensive command after the
        # swing is compared to a stale rest position.
        self.last_target = (cmd.x_mm, cmd.y_mm)
        return cmd

    def committed(self) -> Command | None:
        if self._commit_cmd is not None and self._t < self._commit_until:
            return self._commit_cmd
        self._commit_cmd = None
        return None


class WallBot(Bot):
    """Baseline. Sits on the line, mirrors the puck's y, predicts nothing.

    Here to be beaten: it is what "just track the puck" is worth, and any bot
    that does not clear it is not paying for its complexity.
    """

    name = "wall"

    def command(self, rep: TrackerReport) -> Command:
        est = self.estimate(rep)
        c = self.cfg
        if est is None:
            return self.rest_command(None)
        tx, ty = self.clamp(self.defend_x, est.y_mm)
        return self.settle(tx, ty, c.max_speed_mm_s, c.max_accel_mm_s2)


class GoalieBot(Bot):
    """Hold the line, predict the crossing with lossy bounces, be there early.

    Never leaves the defence line, so it never scores except by deflection --
    which makes it the reference for what pure defence costs.
    """

    name = "goalie"

    def command(self, rep: TrackerReport) -> Command:
        est = self.estimate(rep)
        if est is None:
            return self.rest_command(None)
        return self.defend(est, rep, self.crossing(est))


class StrikerBot(Bot):
    """A goalie that leaves its line when the puck is slow and in reach.

    The trade is explicit: a shot needs the mallet ~400 mm up-table from the
    goal line, and if the opponent reaches the puck first that distance is the
    open net. So it only commits when the puck is SLOW (nobody is about to
    shoot it), on the robot's side, and the contact point is comfortably inside
    the workspace -- and once committed it finishes the swing.
    """

    name = "striker"

    def command(self, rep: TrackerReport) -> Command:
        held = self.committed()
        if held is not None:
            return held

        est = self.estimate(rep)
        if est is None:
            return self.rest_command(None)

        hit = self.raw_crossing(est)
        cmd = self.try_attack(est, rep, hit)
        if cmd is not None:
            return self.commit(cmd, self.cfg.commit_s)
        return self.defend(est, rep, self.horizon_capped(hit))

    def try_attack(self, est: PuckEstimate, rep: TrackerReport,
                   hit: Crossing | None) -> Command | None:
        c = self.cfg
        if est.speed_mm_s > c.attack_max_puck_speed_mm_s:
            return None
        if est.x_mm < c.attack_min_x_mm:
            return None

        aim = self.aim_point(rep)
        mx, my = rep.mallet
        reach = self.contact_distance_mm()

        # Two passes: guess a strike time, see where the puck goes, recompute
        # how long the mallet actually needs. A slow puck moves little in
        # 150 ms, so the fixed point is reached immediately.
        t_hit = 0.10
        for _ in range(2):
            px, py, _, _ = advance_puck(est.x_mm, est.y_mm, est.vx_mm_s,
                                        est.vy_mm_s, t_hit,
                                        puck_radius_mm=c.puck_radius_mm)
            dx, dy = aim[0] - px, aim[1] - py
            n = math.hypot(dx, dy)
            if n < 1e-6:
                return None
            cx, cy = px - dx / n * reach, py - dy / n * reach
            t_hit = reach_time_s(math.hypot(cx - mx, cy - my),
                                 c.strike_speed_mm_s, c.strike_accel_mm_s2)
            if t_hit > c.intercept_max_s:
                return None
        cmd = self.strike_command(px, py, aim, rep)
        if cmd is None or not self.recovery_ok(cmd, hit, t_hit):
            return None
        return cmd


class InterceptBot(Bot):
    """Meets the puck up-table instead of waiting for it, and shoots it back.

    The others treat the defence line as the only place worth being. This one
    steps forward through the predicted path and takes the EARLIEST point it
    can reach with time to spare, putting the mallet on the goal side of the
    puck there, aimed at the opponent's mouth. Two things fall out:

      * a save becomes a clearance, because the mallet is already oriented
        along the shot instead of square to it;
      * meeting the puck early costs the opponent the whole return trip, which
        is the only way a machine confined to 35% of its half gets tempo.

    The safety is structural rather than a threshold: the contact point is
    always BEHIND the puck relative to the aim, i.e. between the puck and the
    robot's own goal, so a mistimed intercept degrades into a block rather than
    into an open lane. If no intercept clears the margin it is a plain goalie.

    In practice the tuning pushed it most of the way back toward being one --
    a 0.3 s search horizon and 0.12 s of margin -- and it is still the weakest
    of the four. See `ai/bin/eval_heuristics.py` for the numbers and the
    reason: with a 12 m/s mallet against a puck that is usually under 3 m/s,
    there is nothing to gain by leaving early and a whole goal to lose.
    """

    name = "intercept"

    def command(self, rep: TrackerReport) -> Command:
        held = self.committed()
        if held is not None:
            return held

        est = self.estimate(rep)
        if est is None:
            return self.rest_command(None)

        c = self.cfg
        hit = self.raw_crossing(est)
        if hit is not None:
            cmd = self.try_intercept(est, rep, hit)
        elif (est.speed_mm_s <= c.attack_max_puck_speed_mm_s
                and est.x_mm >= c.attack_min_x_mm):
            # Puck loitering in our half and not coming to us: same swing, no
            # time pressure, and nothing to recover from.
            cmd = self.try_intercept(est, rep, None, forward_only=False)
        else:
            cmd = None
        if cmd is not None:
            return self.commit(cmd, c.commit_s)
        return self.defend(est, rep, self.horizon_capped(hit))

    def try_intercept(self, est: PuckEstimate, rep: TrackerReport,
                      hit: Crossing | None,
                      forward_only: bool = True) -> Command | None:
        c = self.cfg
        mx, my = rep.mallet
        reach = self.contact_distance_mm()
        aim = self.aim_point(rep)

        n_steps = int(c.intercept_max_s / c.intercept_step_s)
        for k in range(1, n_steps + 1):
            t = k * c.intercept_step_s
            px, py, _, _ = advance_puck(est.x_mm, est.y_mm, est.vx_mm_s,
                                        est.vy_mm_s, t,
                                        puck_radius_mm=c.puck_radius_mm)
            if forward_only and px > c.ws_max_x:
                break                       # past the line; the goalie has it
            dx, dy = aim[0] - px, aim[1] - py
            n = math.hypot(dx, dy)
            if n < 1e-6:
                continue
            cx, cy = px - dx / n * reach, py - dy / n * reach
            if not self.inside(cx, cy, c.strike_clearance_mm):
                continue
            travel = reach_time_s(math.hypot(cx - mx, cy - my),
                                  c.strike_speed_mm_s, c.strike_accel_mm_s2)
            if travel + c.intercept_margin_s > t:
                continue                    # cannot be there in time
            cmd = self.strike_command(px, py, aim, rep)
            if cmd is not None and self.recovery_ok(cmd, hit, t):
                return cmd
        return None


BOTS: dict[str, type[Bot]] = {
    b.name: b for b in (WallBot, GoalieBot, StrikerBot, InterceptBot)
}


def make_bot(name: str, cfg: BotConfig | None = None) -> Bot:
    try:
        return BOTS[name](cfg)
    except KeyError:
        raise ValueError(
            f"unknown bot {name!r}; have {sorted(BOTS)}") from None
