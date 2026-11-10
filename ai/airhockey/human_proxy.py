"""Scripted stand-in for a PERSON at the far end. Simulation benchmarks only.

The opponent for "how would it do against people": the robot's checkpoint
plays this on the robot's body and seat, and the proxy plays the far end on a
human-capable body. It is a benchmark opponent, never a controller of the
robot, never a policy input and never run at the table.

Body: the env's first-order free far-side body (BatchAirHockeyEnv kind
"human_proxy"), with a speed cap, accel cap and lag drawn per game. Its reach
is the env's half clamp: the mallet centre goes anywhere on the far half up to
the centre line (sim y >= 1.0 + mallet radius from the robot's goal), so it
can strike from the line, which a robot-bodied copy (y >= ~1.10) cannot.

Sensing: ONLY the far side's own camera view (BatchAirHockeyEnv.opponent_obs:
the puck through the same tracker model and latency as the robot's, its own
mallet fresh, the robot's mallet through the camera). No engine state, no
hidden parameters: the bank aim and the defensive forecast use the NOMINAL
rail coefficients, so domain-randomised rails produce real aiming error.
Defending, a person reacts late: the forecast runs on the view from a
reaction delay ago, rotated by a slowly varying direction error. A planned
strike on a slow puck needs no reaction and uses the current view.

Behaviour, every 20 ms tick, in its own frame (its goal at y = 0):
  READY    guard line in front of its goal, shadowing the puck and reading
           the robot's mallet as it closes on the puck; a puck forecast to
           enter the mouth is met on the guard line (sometimes with a soft,
           giving block), and it hurries back when out of position.
  APPROACH a slow puck on its half: go around it to a wind-up point behind
           it on the shot line (half single side-rail banks, half straight, to
           a random point in the mouth, with aim noise), sized for the strike.
  HOLD     optional feint: wait behind the puck 0.2-0.8 s.
  STRIKE   drive through the puck at the speed that releases the planned
           shot speed (contact normal solved for a moving puck).

Calibration targets and how the panel compares with them in the very games
being scored are in ai/bin/eval_human_proxy.py (TARGETS, calibration.checks).
Live sessions with people put their attacks at the centre line at p50 4.5-5.7
m/s, p90 6.9-8.2, max ~10, about half banked; the club tier's strikes are
matched to the people's attacks of at least 3 m/s (p50 5.9, p90 8.0 m/s,
median 24 deg from the table axis).

The TABLE tiers ("table", "table_plus") are the two people who played
runs/teacher-calm-20261018-v4 (2026-10-29 and 2026-10-19), measured from the
replay logs with strikes separated from end-rail rebounds: set shots at the
line p50 7.0 / 7.7 m/s, max ~11, banked 54% / 19-31% and mostly off the
robot's LEFT rail, wide of the mouth more often than not. They add three
things the club and elite tiers do not do, each switched by a tier field that
is None for the older tiers, whose behaviour and random draws are unchanged:
  rail      bank_x0_rail_fraction: the share of planned banks that go off the
            sim x = 0 rail, instead of the geometrically better rail.
  VOLLEY    volley_probability: a counter-swing at a puck still moving in,
            rolled once when the (reaction-delayed) view first shows the
            puck coming. The mallet goes to a wind-up point behind where the puck
            will cross a drawn line and drives through it at a drawn speed,
            timed to the puck; the release speed is whatever that gives.
  serve     serve_x_m / serve_y_m: after a goal the person puts the puck down
            at rest on their half and strikes it (serve_env; the evaluator
            calls it, the env's own restart is a slow puck from the centre).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airhockey.dynamics import ACTION_DT
from airhockey.physics import TableConfig
from airhockey.puck_prediction import incoming_crossings

READY, APPROACH, HOLD, STRIKE = 0, 1, 2, 3
MODE_NAMES = ("ready", "approach", "hold", "strike")
# A tier with volley_probability has a fifth mode (HumanProxy.mode_names).
VOLLEY = 4
# Route codes: 0 straight, -1 bank off the x = 0 rail ("left" from the robot,
# the env's shot route 1), +1 bank off the x = width rail ("right", route 2).
ROUTES = (0, -1, 1)


@dataclass(frozen=True)
class HumanTier:
    """Per-game draw ranges for one kind of player."""

    name: str
    max_speed_m_s: tuple = (4.0, 8.0)
    max_accel_m_s2: tuple = (60.0, 150.0)
    time_constant_s: tuple = (0.010, 0.030)
    reaction_s: tuple = (0.15, 0.25)
    # ("lognormal", median, p90) or ("uniform", low, high): the RELEASE speed
    # the person intends, clipped to release_clip_m_s. Club: the people's
    # deliberate attacks (crossings >= 3 m/s) reach the centre line at p50 5.9,
    # p90 8.0 m/s; a strike leaves at ~0.96x what it intended (its arm's speed
    # cap, the table's mallet restitution) and loses a little more by the line.
    release: tuple = ("lognormal", 6.6, 9.0)
    release_clip_m_s: tuple = (3.0, 11.0)
    bank_fraction: float = 0.5
    # A bank goes off the rail that meets the puck on the ROBOT's half at the
    # shallower angle; this share of banks takes the other rail instead.
    other_rail_fraction: float = 0.2
    aim_noise_deg: tuple = (2.0, 4.0)
    defense_noise_deg: tuple = (2.0, 4.0)
    feint_probability: float = 0.4
    feint_s: tuple = (0.2, 0.8)
    guard_depth_m: tuple = (0.10, 0.20)
    # Lateral follow of the puck while it is away, as a fraction of its
    # offset from the goal's centre line (the puck-to-goal bisector at the
    # guard line is ~0.1 for a puck on the robot's half).
    shadow_gain: tuple = (0.1, 0.3)
    idle_speed_m_s: float = 1.0
    idle_accel_m_s2: float = 30.0
    approach_speed_m_s: tuple = (1.2, 2.0)
    # Faster pucks on its half are blocked (or left to slow) before a strike.
    strike_max_puck_speed: float = 1.5
    # Chance per goal-bound puck of giving with it (a soft block that kills
    # most of its speed) rather than a stiff block, for pucks below the cap.
    trap_probability: tuple = (0.3, 0.7)
    trap_max_speed_m_s: float = 6.0
    # Reading the shooter: while the robot's mallet closes on the puck, move
    # this fraction of the way from the shadow position to where a shot along
    # mallet -> puck would cross the guard line.
    anticipation: tuple = (0.3, 0.7)
    # -- table tiers; None keeps the behaviour and the draw order above.
    # Share of planned banks off the sim x = 0 rail (the robot's LEFT),
    # replacing the preferred-rail geometry and other_rail_fraction.
    bank_x0_rail_fraction: float | None = None
    # ((seen puck speed upper edge m/s, probability), ...): the chance of a
    # counter-swing at a puck first seen coming toward the person.
    volley_probability: tuple | None = None
    volley_mallet_speed_m_s: tuple = (2.0, 5.0)
    # Where it meets the puck, distance from its own goal line.
    volley_contact_y_m: tuple = (0.15, 0.50)
    # After a goal: the puck at rest at sim x, sim y drawn uniformly from
    # these, struck no sooner than a log-uniform wait.
    serve_x_m: tuple | None = None
    serve_y_m: tuple = (1.23, 1.62)
    serve_wait_s: tuple = (0.2, 4.0)


CLUB = HumanTier("club")
ELITE = HumanTier(
    "elite",
    max_speed_m_s=(7.0, 10.0),
    max_accel_m_s2=(100.0, 200.0),
    time_constant_s=(0.010, 0.025),
    release=("uniform", 8.0, 14.0),
    release_clip_m_s=(8.0, 14.0),
    aim_noise_deg=(1.5, 3.0),
    defense_noise_deg=(1.5, 3.0),
    feint_probability=0.5,
    approach_speed_m_s=(1.5, 2.5),
    strike_max_puck_speed=2.5,
    trap_probability=(0.5, 0.8),
    trap_max_speed_m_s=8.0,
    anticipation=(0.5, 0.9),
)
# The person of the four 2026-10-29 sessions. `release` is the INTENDED speed
# (achieved is ~0.96x: set-shot release p50 7.25, p90 9.4, max 10.9 m/s);
# max_speed from the mallet speed at set strikes (p50 4.5, p90 6.0, max 7.2);
# aim noise solved from 38% of straight set shots on target; reaction from the
# onset of mallet motion after a robot shot (p25-p90); guard depth and shadow
# gain are the measured IQR and slope. The feint and volley tables, the aim
# noise and the rail share are tuned until the panel's own play lands in the
# measured ranges (eval_human_proxy TABLE_TARGETS), not fitted; trap,
# anticipation, defence noise, lag and accel could not be isolated and are
# the club values. A planned x = 0 bank falls back to another route when the
# mallet has no room behind the puck, and a wide straight shot banks off
# either rail, so the measured 0.87 needs nearly every plan to be that rail.
TABLE = HumanTier(
    "table",
    max_speed_m_s=(6.0, 7.5),
    reaction_s=(0.12, 0.22),
    release=("lognormal", 7.55, 9.8),
    release_clip_m_s=(4.0, 10.9),
    bank_fraction=0.50,
    aim_noise_deg=(8.0, 12.0),
    feint_probability=0.35,
    feint_s=(0.3, 2.5),
    guard_depth_m=(0.07, 0.22),
    shadow_gain=(0.05, 0.15),
    bank_x0_rail_fraction=0.95,
    volley_probability=((2.0, 0.25), (4.0, 0.50), (6.0, 0.08), (np.inf, 0.0)),
    volley_mallet_speed_m_s=(2.2, 5.2),
    volley_contact_y_m=(0.29, 0.55),
    serve_x_m=(0.39, 0.71),
)
# The stronger person of 2026-10-19: faster and straighter, better aim (48-54%
# on target), quicker to react, and swings at pucks arriving at up to ~8 m/s.
TABLE_PLUS = HumanTier(
    "table_plus",
    max_speed_m_s=(6.5, 8.0),
    max_accel_m_s2=(80.0, 160.0),
    reaction_s=(0.10, 0.18),
    release=("lognormal", 8.1, 10.2),
    release_clip_m_s=(5.0, 11.2),
    bank_fraction=0.25,
    aim_noise_deg=(4.5, 7.5),
    feint_probability=0.30,
    feint_s=(0.3, 1.5),
    shadow_gain=(0.03, 0.10),
    approach_speed_m_s=(1.5, 2.5),
    anticipation=(0.4, 0.8),
    bank_x0_rail_fraction=0.70,
    volley_probability=((2.0, 0.20), (4.0, 0.60), (6.0, 0.38), (8.0, 0.22), (np.inf, 0.06)),
    volley_mallet_speed_m_s=(1.8, 3.8),
    volley_contact_y_m=(0.13, 0.50),
    serve_x_m=(0.39, 0.71),
)
TIERS = {tier.name: tier for tier in (CLUB, ELITE, TABLE, TABLE_PLUS)}


def _norm(v):
    return np.linalg.norm(v, axis=-1, keepdims=True)


def _unit(v, fallback=(0.0, 1.0)):
    length = _norm(v)
    return np.where(length > 1e-9, v / np.maximum(length, 1e-12), np.asarray(fallback))


def _rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c), axis=-1)


def release_speeds(tier, rng, shape):
    """Intended release speeds for `tier`, m/s."""
    kind, a, b = tier.release
    if kind == "lognormal":
        sigma = np.log(b / a) / 1.2815515655446004   # p90 of a standard normal
        draw = a * np.exp(sigma * rng.standard_normal(shape))
    elif kind == "uniform":
        draw = rng.uniform(a, b, shape)
    else:
        raise ValueError(f"unknown release distribution {kind!r}")
    return np.clip(draw, *tier.release_clip_m_s)


def proxy_view(env):
    """The far side's own [N, 12] view: puck, own mallet, rival mallet (its frame).

    `env` is an ArrivalEnv/NeuralTrainingEnv or a robot-shaped
    BatchAirHockeyEnv. Call ONCE per tick: its own-mallet velocity is a
    finite difference between calls.
    """
    base = getattr(env, "base", env)
    return base.opponent_obs()[:, :12].astype(np.float64)


class HumanProxy:
    """Vectorised human stand-in: act(view) -> table-frame command per env."""

    LEAD_S = 0.03            # forecast the puck by the camera age + command hold
    THROUGH_M = 0.12         # drive this far beyond the puck centre
    SETUP_TOLERANCE_M = 0.03
    SETTLE_SPEED_M_S = 1.2
    DECIDE_S = 0.10          # a slow puck must stay playable this long first
    APPROACH_TIMEOUT_S = 1.6
    STRIKE_TIMEOUT_S = 0.40
    COOLDOWN_S = 0.25
    PATIENCE_S = 3.0         # a person does not stall: strike from wherever
    # What a person "feels" the mallet gives back (the sim draws 0.60-0.95).
    ASSUMED_RESTITUTION = 0.78
    NOISE_CORRELATION_S = 0.30
    THREAT_HORIZON_S = 1.2
    GOAL_MARGIN_M = 0.05
    MIN_WINDUP_M = 0.035
    CLEAR_SPEED_M_S = 3.0
    RECOVER_DISTANCE_M = 0.12
    PLANS = 256
    # Forecast lines in its frame: its goal line, then candidate block lines
    # (a mallet centre cannot go below its radius, 0.05).
    FORECAST_LINES = np.array([0.0, 0.06, 0.10, 0.14, 0.18, 0.22])
    # VOLLEY: candidate contact lines (the drawn one snaps to the nearest),
    # a swing that needs the puck this much later than the mallet, and how
    # long after the forecast arrival a miss is given up.
    VOLLEY_LINES = np.round(np.arange(0.10, 0.6001, 0.04), 2)
    VOLLEY_MARGIN_S = 0.04
    VOLLEY_AIM_NOISE = 1.5
    VOLLEY_OVERTIME_S = 0.15
    SERVES = 64

    def __init__(self, n, tier="club", *, seed=0, config=None, dt=ACTION_DT):
        self.n = int(n)
        self.tier = TIERS[tier] if isinstance(tier, str) else tier
        self.cfg = config or TableConfig()
        self.dt = float(dt)
        self.seed = int(seed)
        cfg = self.cfg
        self.width, self.height = cfg.width, cfg.height
        self.contact = cfg.puck_radius + cfg.paddle_radius
        # Reach in its own frame: the whole half up to the centre line.
        self.low = np.array([cfg.paddle_radius, cfg.paddle_radius])
        self.high = np.array([cfg.width - cfg.paddle_radius, cfg.height / 2 - cfg.paddle_radius])
        self.mouth = cfg.goal_width / 2 - cfg.puck_radius
        self._history_len = int(np.ceil(max(self.tier.reaction_s) / self.dt)) + 3
        self.rng = None
        self.volleying = self.tier.volley_probability is not None
        self.mode_names = MODE_NAMES + ("volley",) if self.volleying else MODE_NAMES

    # ------------------------------------------------------------ episode
    def reset(self, view):
        """Draw every game's player and shot sequence; start in READY."""
        t, n, k = self.tier, self.n, self.PLANS
        rng = self.rng = np.random.default_rng(self.seed)

        def u(r, size=n):
            return rng.uniform(r[0], r[1], size)
        # One person per game, drawn once and in a fixed order, so a seed is
        # the same panel of people for every candidate.
        self.max_speed = u(t.max_speed_m_s)
        self.max_accel = u(t.max_accel_m_s2)
        self.time_constant = u(t.time_constant_s)
        self.reaction = u(t.reaction_s)
        self.aim_noise = np.radians(u(t.aim_noise_deg))
        self.defense_noise = np.radians(u(t.defense_noise_deg))
        self.guard_depth = u(t.guard_depth_m)
        self.shadow = u(t.shadow_gain)
        self.approach_speed = np.minimum(u(t.approach_speed_m_s), self.max_speed)
        self.trap_skill = u(t.trap_probability)
        self.anticipation = u(t.anticipation)
        # The i-th shot of a game is the same intended shot for every
        # candidate: route, aim point, release speed, aim error and feint.
        # Route as planned before seeing where the puck is: 0 straight, +1 a
        # bank off the preferred rail, -1 off the other one (resolved per tick).
        bank = rng.random((n, k)) < t.bank_fraction
        if t.bank_x0_rail_fraction is None:
            other = rng.random((n, k)) < t.other_rail_fraction
            self.plan_route = np.where(bank, np.where(other, -1, 1), 0)
        else:
            # A handed player: the rail itself is planned, -1 the x = 0 rail.
            x0 = rng.random((n, k)) < t.bank_x0_rail_fraction
            self.plan_route = np.where(bank, np.where(x0, -1, 1), 0)
        aim = self.mouth - 0.01
        self.plan_aim = self.width / 2 + rng.uniform(-aim, aim, (n, k))
        self.plan_speed = release_speeds(t, rng, (n, k))
        self.plan_aim_error = rng.standard_normal((n, k)) * self.aim_noise[:, None]
        feint = rng.random((n, k)) < t.feint_probability
        self.plan_feint = np.where(feint, rng.uniform(*t.feint_s, (n, k)), 0.0)
        if t.serve_x_m is not None:
            # The i-th serve of a game, likewise the same for every candidate.
            s = self.SERVES
            self.plan_serve = np.stack((rng.uniform(*t.serve_x_m, (n, s)), rng.uniform(*t.serve_y_m, (n, s)),
                                        np.exp(rng.uniform(*np.log(t.serve_wait_s), (n, s)))), axis=-1)
            self.serve_index = np.zeros(n, dtype=int)
            self.serves = np.zeros(n, dtype=int)
        view = np.asarray(view, dtype=np.float64)
        self._hist = np.repeat(view[None, :, :4], self._history_len, axis=0)
        self._head = 0
        self.mode = np.full(n, READY, dtype=np.int8)
        self.timer = np.zeros(n)
        self.playable_time = np.zeros(n)
        self.half_time = np.zeros(n)
        self.cooldown = np.zeros(n)
        self.plan_index = np.zeros(n, dtype=int)
        self.strike_route = np.zeros(n, dtype=int)
        self.strike_clear = np.zeros(n, bool)
        self.strike_direction = np.tile([0.0, 1.0], (n, 1))
        self.strike_speed = np.zeros(n)
        self.noise_state = np.zeros(n)
        self.read_x = np.full(n, self.width / 2)
        self.read_age = np.full(n, np.inf)
        self.trap_roll = np.ones(n)
        self.was_threat = np.zeros(n, bool)
        self.trapping = np.zeros(n, bool)
        self.traps = np.zeros(n, dtype=int)
        self.strikes = np.zeros(n, dtype=int)
        self.mode_ticks = np.zeros((n, len(self.mode_names)), dtype=int)
        if self.volleying:
            self._seen_incoming = np.zeros(n, dtype=int)
            self.volley_y = np.full(n, float(np.mean(t.volley_contact_y_m)))
            self.volley_speed = np.minimum(np.full(n, float(np.mean(t.volley_mallet_speed_m_s))), self.max_speed)
            self.volley_eta = np.zeros(n)
            self.volley_swing = np.zeros(n, bool)
            self.volley_normal = np.tile([0.0, 1.0], (n, 1))
            self.volley_contact = view[:, 4:6].copy()
            self.volley_setup = view[:, 4:6].copy()
            self.volley_attempts = np.zeros(n, dtype=int)
            self.volleys = np.zeros(n, dtype=int)
        self.last_target = view[:, 4:6].copy()
        return self.commands_table(self.last_target, np.minimum(t.idle_speed_m_s, self.max_speed),
                                   np.minimum(t.idle_accel_m_s2, self.max_accel))

    def commands_table(self, target_local, speed, accel):
        """Local-frame target clipped to its reach -> (x, y, speed, accel) in the table frame."""
        x = np.clip(target_local[:, 0], self.low[0], self.high[0])
        y = np.clip(target_local[:, 1], self.low[1], self.high[1])
        return x, self.height - y, np.asarray(speed, float), np.asarray(accel, float)

    # ---------------------------------------------------------- perception
    def _push(self, puck_state):
        self._head = (self._head + 1) % self._history_len
        self._hist[self._head] = puck_state

    def delayed(self, delay=None):
        """Puck [x, y, vx, vy] as seen `delay` s ago (per env, interpolated between ticks)."""
        delay = self.reaction if delay is None else np.broadcast_to(np.asarray(delay, float), (self.n,))
        lag = np.clip(delay / self.dt, 0, self._history_len - 2)
        k0 = np.floor(lag).astype(int)
        frac = (lag - k0)[:, None]
        idx = np.arange(self.n)
        newer = self._hist[(self._head - k0) % self._history_len, idx]
        older = self._hist[(self._head - k0 - 1) % self._history_len, idx]
        return (1 - frac) * newer + frac * older

    def _defense_forecast(self, seen_puck, seen_vel):
        """From the seen (delayed, noisy) puck: is it goal-bound, and where to meet it.

        Goal-bound means its forecast crosses the goal line inside the mouth
        (plus a margin). It is met where it crosses the game's guard line if
        that is in front of the goal; a puck that crosses the guard line wide
        and angles in is met on the deepest line instead.
        """
        lines = self.FORECAST_LINES
        forecast = incoming_crossings(np.column_stack((seen_puck, seen_vel)), lines,
                                      config=self.cfg, max_seconds=self.THREAT_HORIZON_S)
        centre = self.width / 2
        goal_x = forecast[:, 0, 0]
        speed = np.hypot(*seen_vel.T)
        threat = (np.isfinite(goal_x) & (np.abs(goal_x - centre) < self.mouth + self.GOAL_MARGIN_M)
                  & (seen_vel[:, 1] < -0.3) & (speed > 0.6))
        idx = np.arange(self.n)
        primary = np.clip(np.searchsorted(lines, self.guard_depth, side="right") - 1, 1, len(lines) - 1)
        x_primary = forecast[idx, primary, 0]
        band = self.mouth + 0.06
        in_front = np.isfinite(x_primary) & (np.abs(x_primary - centre) < band)
        x_deep = forecast[:, 1, 0]
        block_x = np.where(in_front, x_primary, np.where(np.isfinite(x_deep), x_deep, goal_x))
        block_y = np.where(in_front, lines[primary], lines[1])
        block_x = np.clip(np.where(threat, block_x, centre), centre - band, centre + band)
        arrival = np.where(in_front, forecast[idx, primary, 1], forecast[:, 1, 1])
        return threat, block_x, block_y, arrival

    # ------------------------------------------------------------ geometry
    def _shot_direction(self, puck, route, aim_x):
        """Unit launch direction toward the robot's goal line (its y = height).

        Banks use the NOMINAL lossy-rail ratio: the unfolding of
        neural_possession.goal_shot_velocity, mirrored into this frame.
        """
        r = self.cfg.puck_radius
        wall = np.where(route < 0, r, self.width - r)
        ratio = self.cfg.wall_tangential / self.cfg.wall_restitution
        dx = np.where(route == 0, aim_x - puck[..., 0],
                      wall - puck[..., 0] + ratio * (wall - aim_x))
        return _unit(np.stack((dx, self.height - puck[..., 1]), axis=-1))

    def _room(self, point, direction):
        """Distance from `point` along unit `direction` to the reach box edge (0 if outside)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            to_high = np.where(direction > 1e-9, (self.high - point) / direction, np.inf)
            to_low = np.where(direction < -1e-9, (self.low - point) / direction, np.inf)
        room = np.minimum(to_high, to_low).min(axis=-1)
        inside = ((point >= self.low - 1e-6) & (point <= self.high + 1e-6)).all(axis=-1)
        return np.where(inside, np.maximum(room, 0.0), 0.0)

    def _strike_plan(self, puck, vel, direction, release, max_speed, max_accel):
        """Contact normal, paddle speed, contact point, wind-up and feasibility.

        The mallet is the heavy body: a puck leaves at v + (1+e)((u-v).n) n, so
        the normal that sends it along `direction` at `release` is the unit
        vector of (release*direction - v), struck at that impulse.
        """
        wanted = direction * release[..., None]
        delta = wanted - vel
        normal = _unit(delta)
        impulse = _norm(delta)[..., 0]
        speed = impulse / (1 + self.ASSUMED_RESTITUTION) + (vel * normal).sum(-1)
        speed = np.clip(speed, 0.4, max_speed)
        windup = np.clip(1.3 * speed**2 / (2 * max_accel) + 0.03, 0.05, 0.30)
        contact = puck - self.contact * normal
        room = self._room(contact, -normal)
        feasible = room >= self.MIN_WINDUP_M
        return normal, speed, contact, np.minimum(windup, room), feasible

    def _volley_plan(self, seen_puck, seen_vel, plan):
        """Where and how to meet a moving puck on the drawn line at the drawn mallet speed.

        Forecast from the seen (delayed, noisy) puck with the nominal rails.
        Route and aim are the next planned shot's; the contact normal is the
        one _strike_plan would use for the release speed that this mallet
        speed happens to give (bisected), so the release is an outcome.
        """
        idx = np.arange(self.n)
        lines = self.VOLLEY_LINES
        forecast = incoming_crossings(np.column_stack((seen_puck, seen_vel)), lines,
                                      config=self.cfg, max_seconds=self.THREAT_HORIZON_S)
        line = np.abs(lines[None, :] - self.volley_y[:, None]).argmin(axis=1)
        x, arrival = forecast[idx, line, 0], forecast[idx, line, 1]
        ok = np.isfinite(x) & np.isfinite(arrival)
        puck = np.column_stack((np.where(ok, x, self.width / 2), lines[line]))
        vel = np.where(ok[:, None], forecast[idx, line, 2:4], [0.0, -1.0])
        route = self.plan_route[idx, plan]
        if self.tier.bank_x0_rail_fraction is None:
            route = np.abs(route) * np.where(puck[:, 0] < self.width / 2, -1, 1)
        direction = _rotate(self._shot_direction(puck, route, self.plan_aim[idx, plan]),
                            self.VOLLEY_AIM_NOISE * self.plan_aim_error[idx, plan])
        speed = self.volley_speed
        low, high = np.full(self.n, 0.2), np.full(self.n, 25.0)
        for _ in range(14):
            release = 0.5 * (low + high)
            delta = direction * release[:, None] - vel
            normal = _unit(delta)
            needed = _norm(delta)[:, 0] / (1 + self.ASSUMED_RESTITUTION) + (vel * normal).sum(-1)
            slow = needed < speed
            low, high = np.where(slow, release, low), np.where(slow, high, release)
        contact = puck - self.contact * normal
        room = self._room(contact, -normal)
        windup = np.minimum(np.clip(1.3 * speed**2 / (2 * self.max_accel) + 0.03, 0.05, 0.30), room)
        setup = contact - windup[:, None] * normal
        feasible = ok & (room >= self.MIN_WINDUP_M)
        # From rest at the set-up point: accelerate to the swing speed, then carry it.
        ramp = speed**2 / (2 * self.max_accel)
        swing_time = np.where(windup < ramp, np.sqrt(2 * windup / self.max_accel),
                              speed / self.max_accel + (windup - ramp) / speed)
        return feasible, arrival - self.reaction, normal, contact, setup, direction, swing_time

    def _clear_direction(self, puck, own):
        """Fan direction nearest straight up-table that has wind-up room, and that room."""
        toward_centre = np.where(puck[:, 0] < self.width / 2, 1.0, -1.0)
        best = np.tile([0.0, 1.0], (self.n, 1))
        best_room = np.zeros(self.n)
        found = np.zeros(self.n, bool)
        for degrees in (0, 25, -25, 50, -50, 70, -70, 85, -85):
            angle = np.radians(degrees) * toward_centre
            normal = np.column_stack((np.sin(angle), np.cos(angle)))
            room = self._room(puck - self.contact * normal, -normal)
            ok = ~found & (room >= self.MIN_WINDUP_M)
            best[ok], best_room[ok] = normal[ok], room[ok]
            found |= ok
        # Wedged in a corner: push from where the mallet is, if that is up-table.
        poke = _unit(puck - own)
        wedged = ~found & (poke[:, 1] > 0.2)
        best[wedged] = poke[wedged]
        return best, best_room

    # ----------------------------------------------------------------- act
    def act(self, view):
        """One 20 ms decision from the far side's own view -> (x, y, speed, accel), table frame."""
        t, n, cfg = self.tier, self.n, self.cfg
        view = np.asarray(view, dtype=np.float64)
        if view.ndim != 2 or view.shape[0] != n or view.shape[1] < 12 or not np.isfinite(view[:, :12]).all():
            raise ValueError("expected a finite [N, >=12] far-side view")
        puck, vel = view[:, 0:2], view[:, 2:4]
        own, own_vel = view[:, 4:6], view[:, 6:8]
        rival, rival_vel = view[:, 8:10], view[:, 10:12]
        self._push(view[:, :4])
        idx = np.arange(n)
        half = self.height / 2
        centre = self.width / 2

        # -- defending: what was seen a reaction time ago, with a drifting direction error
        decay = np.exp(-self.dt / self.NOISE_CORRELATION_S)
        self.noise_state = (decay * self.noise_state
                            + np.sqrt(1 - decay**2) * self.rng.standard_normal(n))
        seen = self.delayed()
        seen_vel = _rotate(seen[:, 2:4], self.noise_state * self.defense_noise)
        threat, block_x, block_y, arrival = self._defense_forecast(seen[:, :2], seen_vel)
        now = seen[:, :2] + seen[:, 2:4] * self.reaction[:, None]
        shadow_x = centre + self.shadow * (np.clip(now[:, 0], 0, self.width) - centre)
        shadow_x = np.clip(shadow_x, centre - self.mouth + 0.02, centre + self.mouth - 0.02)
        # Reading the shooter (watched continuously, not a reaction): the
        # robot's mallet closing on the puck from behind it. Where a shot along
        # mallet -> puck would cross the guard line is held for a reaction
        # time after the read ends, until the reaction to the shot takes over.
        rel = puck - rival
        aim = _unit(rel)
        closing = ((rival_vel - vel) * aim).sum(-1) > 0.3
        reading = (puck[:, 1] > half) & (rel[:, 1] < 0) & (np.hypot(*rel.T) < 0.30) & closing
        read_line = incoming_crossings(np.column_stack((puck, 6.0 * aim)), [float(np.median(self.guard_depth))],
                                       config=cfg, max_seconds=1.0)[:, 0, 0]
        read_ok = reading & np.isfinite(read_line) & (np.abs(read_line - centre) < self.mouth + self.GOAL_MARGIN_M)
        self.read_x = np.where(read_ok, np.clip(read_line, centre - self.mouth, centre + self.mouth), self.read_x)
        self.read_age = np.where(read_ok, 0.0, self.read_age + self.dt)
        held = self.read_age < self.reaction + 0.05
        shadow_x = np.where(held, shadow_x + self.anticipation * (self.read_x - shadow_x), shadow_x)
        guard = np.column_stack((np.where(threat, block_x, shadow_x),
                                 np.where(threat, block_y, self.guard_depth)))
        # Out of position (after a strike, say), a person hurries back.
        hurry = held | (np.hypot(*(own - guard).T) > self.RECOVER_DISTANCE_M)
        guard_speed = np.where(threat, self.max_speed,
                               np.where(hurry, self.approach_speed, np.minimum(t.idle_speed_m_s, self.max_speed)))
        guard_accel = np.where(threat, self.max_accel, np.minimum(t.idle_accel_m_s2, self.max_accel))
        # A soft block: decided once per goal-bound puck (a fixed draw every
        # tick keeps the random stream aligned across candidates), then, in
        # place as it arrives, give with the puck at about half its speed.
        rolls = self.rng.random(n)
        fresh = threat & ~self.was_threat
        self.trap_roll[fresh] = rolls[fresh]
        self.was_threat = threat
        inbound = np.hypot(*seen_vel.T)
        time_left = arrival - self.reaction
        placed = np.hypot(*(own - guard).T) < 0.04
        trap = (threat & (self.trap_roll < self.trap_skill) & placed & (inbound < t.trap_max_speed_m_s)
                & (time_left < 0.08) & (time_left > -0.06) & (guard[:, 1] > self.low[1] + 0.04))
        give = guard + 0.10 * _unit(seen_vel)
        guard = np.where(trap[:, None], give, guard)
        guard_speed = np.where(trap, np.clip(0.5 * inbound, 0.3, self.max_speed), guard_speed)
        self.traps += trap & ~self.trapping
        self.trapping = trap

        # -- attacking: the current report, no reaction delay for a planned strike
        speed_now = np.hypot(*vel.T)
        on_half = puck[:, 1] < half - 0.01
        self.half_time = np.where(on_half, self.half_time + self.dt, 0.0)
        lead = np.clip(puck + vel * self.LEAD_S, cfg.puck_radius,
                       [self.width - cfg.puck_radius, self.height - cfg.puck_radius])
        incoming = (vel[:, 1] < -0.8) & (speed_now > 0.8)
        staying = puck[:, 1] + vel[:, 1] * 0.3 < half - 0.02
        playable = on_half & (speed_now < t.strike_max_puck_speed) & ~incoming & staying
        keep = on_half & (speed_now < 1.6 * t.strike_max_puck_speed) & ~incoming
        self.playable_time = np.where(playable, self.playable_time + self.dt, 0.0)

        plan = self.plan_index % self.PLANS
        release = self.plan_speed[idx, plan]
        feint = self.plan_feint[idx, plan]
        # All three routes, aim error applied; the planned one is preferred and
        # the others are the fall-backs when it has no wind-up room. [N, 3, 2]
        directions = np.stack([self._shot_direction(lead, np.full(n, r), self.plan_aim[idx, plan])
                               for r in ROUTES], axis=1)
        # The preferred rail: of the two banks, the one meeting its rail on
        # the robot's half, the shallower if both do (a bank that bounces on
        # one's own half only makes the shot slower and easier to read).
        rail_x = np.array([cfg.puck_radius, self.width - cfg.puck_radius])
        with np.errstate(divide="ignore", invalid="ignore"):
            bounce_y = lead[:, None, 1] + (rail_x - lead[:, None, 0]) * directions[:, 1:, 1] / directions[:, 1:, 0]
        steepness = np.abs(directions[:, 1:, 0]) / np.maximum(directions[:, 1:, 1], 1e-6)
        cost = steepness + 10.0 * ~(bounce_y > half)
        preferred = np.where(cost[:, 0] <= cost[:, 1], -1, 1)
        planned = self.plan_route[idx, plan] * (preferred if t.bank_x0_rail_fraction is None else 1)
        directions = _rotate(directions, self.plan_aim_error[idx, plan][:, None])
        normal, strike_speed, contact, windup, feasible = self._strike_plan(
            lead[:, None, :], vel[:, None, :], directions, np.repeat(release[:, None], 3, 1),
            self.max_speed[:, None], self.max_accel[:, None])
        order = np.where(planned[:, None] == 0, [[0, 1, 2]],
                         np.where(planned[:, None] < 0, [[1, 0, 2]], [[2, 0, 1]]))
        ranked = feasible[idx[:, None], order]
        any_route = ranked.any(axis=1)
        choice = order[idx, np.argmax(ranked, axis=1)]
        locked = self.mode == STRIKE
        # A strike in progress keeps the route it started on.
        route_index = np.where(locked & ~self.strike_clear,
                               np.select([self.strike_route == 0, self.strike_route < 0], [0, 1], 2),
                               choice)
        clear_normal, clear_room = self._clear_direction(lead, own)
        desperate = self.half_time > 2 * self.PATIENCE_S
        use_clear = np.where(locked, self.strike_clear, ~any_route | desperate)
        normal_n = np.where(use_clear[:, None], clear_normal, normal[idx, route_index])
        speed_n = np.where(use_clear, np.minimum(self.CLEAR_SPEED_M_S, self.max_speed),
                           strike_speed[idx, route_index])
        contact_n = np.where(use_clear[:, None], lead - self.contact * clear_normal,
                             contact[idx, route_index])
        windup_n = np.where(use_clear, np.minimum(0.06, clear_room), windup[idx, route_index])
        setup = contact_n - windup_n[:, None] * normal_n

        # Around the puck when it stands between the mallet and the set-up point.
        rel = own - lead
        behind = (rel * normal_n).sum(-1) < -0.6 * self.contact
        seg = setup - own
        along_seg = np.clip(((lead - own) * seg).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-9), 0, 1)
        blocked = np.hypot(*(own + along_seg[:, None] * seg - lead).T) < self.contact + 0.01
        perp = np.column_stack((-normal_n[:, 1], normal_n[:, 0]))
        side = np.where((rel * perp).sum(-1) >= 0, 1.0, -1.0)
        detour = lead + (self.contact + 0.06) * _unit(side[:, None] * perp - 0.4 * normal_n)
        approach_target = np.where((~behind & blocked)[:, None], detour, setup)

        # -- mode transitions (each mask tests the mode it leaves, so they are disjoint)
        self.cooldown = np.maximum(self.cooldown - self.dt, 0.0)
        self.timer += self.dt
        mode = self.mode
        start = (mode == READY) & playable & (self.playable_time >= self.DECIDE_S) & (self.cooldown <= 0)
        abort = ((mode == APPROACH) | (mode == HOLD)) & ~keep
        settled = np.hypot(*own_vel.T) < self.SETTLE_SPEED_M_S
        at_setup = (np.hypot(*(own - setup).T) < self.SETUP_TOLERANCE_M) & settled
        impatient = ((self.half_time > self.PATIENCE_S) | (self.timer > self.APPROACH_TIMEOUT_S)) & behind
        go = (mode == APPROACH) & ~abort & ((at_setup & (feint <= 0)) | impatient | desperate)
        hold = (mode == APPROACH) & ~abort & at_setup & (feint > 0) & ~go
        release_hold = (mode == HOLD) & ~abort & ((self.timer >= feint) | (self.half_time > self.PATIENCE_S))
        along = (vel * self.strike_direction).sum(-1)
        hit = (mode == STRIKE) & (along > 1.0) & (vel[:, 1] > 0.5)
        give_up = (mode == STRIKE) & ~hit & ((self.timer > self.STRIKE_TIMEOUT_S) | ~on_half)
        strike = go | release_hold
        done = hit | give_up
        new_mode = mode.copy()
        new_mode[start] = APPROACH
        new_mode[abort] = READY
        new_mode[hold] = HOLD
        new_mode[strike] = STRIKE
        new_mode[done] = READY
        if self.volleying:
            # Three fixed draws a tick, used or not, as for the soft block.
            draws = self.rng.random((3, n))
            # Once per incoming puck, when it is first seen coming: a person
            # sets up on seeing the robot's strike, before the puck is over
            # the line (their mallet starts 0.06-0.22 s after that contact).
            # (The second tick of it: the first is an interpolation between
            # the view before the strike and the view after.)
            incoming_seen = seen[:, 3] < -0.8
            fresh_puck = incoming_seen & (self._seen_incoming == 1)
            self._seen_incoming = np.where(incoming_seen, self._seen_incoming + 1, 0)
            edges, chance = (np.asarray(v, float) for v in zip(*t.volley_probability))
            chance = chance[np.minimum(np.searchsorted(edges, np.hypot(*seen[:, 2:4].T), side="right"),
                                       len(chance) - 1)]
            want = fresh_puck & (mode == READY) & ~start & (self.cooldown <= 0) & (draws[0] < chance)
            lo, hi = t.volley_contact_y_m
            self.volley_y[want] = lo + (hi - lo) * draws[1][want]
            lo, hi = t.volley_mallet_speed_m_s
            self.volley_speed[want] = np.minimum(lo + (hi - lo) * draws[2], self.max_speed)[want]
            volleying = mode == VOLLEY
            feasible, eta, v_normal, v_contact, v_setup, v_direction, swing_time = self._volley_plan(
                seen[:, :2], seen_vel, plan)
            # Minimum time to the set-up point, coming to rest there.
            distance = np.hypot(*(own - v_setup).T)
            cruise = self.max_speed**2 / self.max_accel
            reach = np.where(distance < cruise, 2 * np.sqrt(distance / self.max_accel),
                             distance / self.max_speed + self.max_speed / self.max_accel)
            begin = want & feasible & (reach < eta - self.VOLLEY_MARGIN_S)
            # While it can still see the puck coming, the plan follows it;
            # after that the swing runs on the last one.
            self.volley_swing[begin] = False
            track = (begin | volleying) & feasible & ~self.volley_swing
            self.volley_eta = np.where(track, eta, self.volley_eta - self.dt)
            self.volley_normal[track], self.volley_contact[track] = v_normal[track], v_contact[track]
            self.volley_setup[track] = v_setup[track]
            self.strike_direction[track] = v_direction[track]
            self.volley_swing |= (begin | volleying) & (self.volley_eta <= swing_time)
            v_hit = volleying & (along > 1.0) & (vel[:, 1] > 0.5)
            v_over = volleying & ~v_hit & (self.volley_eta < -self.VOLLEY_OVERTIME_S)
            new_mode[begin] = VOLLEY
            new_mode[v_hit | v_over] = READY
            self.volley_attempts += begin
            self.volleys += v_hit
            hit = hit | v_hit
            done = done | v_hit | v_over
        self.cooldown[done] = self.COOLDOWN_S
        self.plan_index[done] += 1
        self.strikes += hit & (mode == STRIKE)
        self.timer[new_mode != mode] = 0.0
        self.strike_clear[strike] = use_clear[strike]
        self.strike_route[strike] = np.asarray(ROUTES)[route_index[strike]]
        self.strike_direction[strike] = np.where(use_clear[strike, None], clear_normal[strike],
                                                 directions[idx[strike], route_index[strike]])
        self.strike_speed[strike] = speed_n[strike]
        self.mode = new_mode
        self.mode_ticks[idx, new_mode] += 1

        # -- the command for the mode it is now in
        through = np.minimum(self.contact + self.THROUGH_M, self._room(contact_n, normal_n))
        strike_target = contact_n + through[:, None] * normal_n
        approaching = (new_mode == APPROACH) | (new_mode == HOLD)
        striking = new_mode == STRIKE
        target = np.where(approaching[:, None], approach_target,
                          np.where(striking[:, None], strike_target, guard))
        speed = np.where(approaching, self.approach_speed, np.where(striking, speed_n, guard_speed))
        accel = np.where(approaching, 0.6 * self.max_accel, np.where(striking, self.max_accel, guard_accel))
        if self.volleying:
            # Wait at the wind-up point, then through the contact point at the drawn speed.
            now_volley = new_mode == VOLLEY
            swing = self.volley_swing
            reach_out = np.minimum(self.contact + self.THROUGH_M, self._room(self.volley_contact, self.volley_normal))
            v_target = np.where(swing[:, None], self.volley_contact + reach_out[:, None] * self.volley_normal,
                                self.volley_setup)
            target = np.where(now_volley[:, None], v_target, target)
            speed = np.where(now_volley, np.where(swing, self.volley_speed, self.max_speed), speed)
            accel = np.where(now_volley, self.max_accel, accel)
        self.last_target = target
        return self.commands_table(target, speed, accel)

    def serve(self, mask):
        """The next planned serve for the games in `mask`: sim (x, y) to put the puck down at.

        The person then waits the drawn time before going to it; the strike
        itself is the ordinary planned one. Table tiers only.
        """
        ids = np.flatnonzero(mask)
        x, y, wait = self.plan_serve[ids, self.serve_index[ids] % self.SERVES].T
        self.serve_index[ids] += 1
        self.serves[ids] += 1
        self.cooldown[ids] = np.maximum(self.cooldown[ids], wait)
        return x, y

    # ------------------------------------------------------------- report
    def parameters(self):
        """Per-game draws, for metadata and pairing checks."""
        return dict(
            tier=self.tier.name, seed=self.seed,
            max_speed_m_s=self.max_speed.tolist(), max_accel_m_s2=self.max_accel.tolist(),
            time_constant_s=self.time_constant.tolist(), reaction_s=self.reaction.tolist(),
            aim_noise_deg=np.degrees(self.aim_noise).tolist(),
            defense_noise_deg=np.degrees(self.defense_noise).tolist(),
            guard_depth_m=self.guard_depth.tolist(), shadow_gain=self.shadow.tolist(),
            approach_speed_m_s=self.approach_speed.tolist(), trap_skill=self.trap_skill.tolist(),
            anticipation=self.anticipation.tolist())


def command_env(env, proxy, view=None, *, reset=False):
    """Read the far side's view, decide, and hand the command to the env. Returns the view."""
    view = proxy_view(env) if view is None else view
    base = getattr(env, "base", env)
    if reset:
        x, y, speed, accel = proxy.reset(view)
        base.set_human_proxy_command(x, y, speed, accel, time_constant=proxy.time_constant)
    else:
        x, y, speed, accel = proxy.act(view)
        base.set_human_proxy_command(x, y, speed, accel)
    return view


def serve_env(env, proxy, scored):
    """After a goal against a serving tier: the puck at rest where the person puts it down.

    Call between steps with the games that just had a goal (either way: on
    the table every restart is the person's). Replaces the engine's restart,
    a slow puck leaving the centre line. A no-op for the club and elite tiers.
    """
    scored = np.asarray(scored, bool)
    if proxy.tier.serve_x_m is None or not scored.any():
        return
    engine = getattr(env, "base", env).engine
    engine.puck_x[scored], engine.puck_y[scored] = proxy.serve(scored)
    engine.puck_vx[scored] = engine.puck_vy[scored] = 0.0
