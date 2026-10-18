"""Physics-based teacher for the neural player. Simulation/training only.

The deployed controller remains ONE neural network. This module is a
teacher: its labels are distilled into that network (and it serves as a
sparring partner and benchmark reference). It never runs at the table.

It sees exactly what the actor sees -- the same delayed, noisy 42-feature
frames and shot request -- and emits the same 6-D arrival action through the
same decoder, motion guard, firmware law and thermal model. No simulator
state, hidden release times or trial parameters are read.

Behaviour, in priority order every 20 ms decision:

1. Threat: a puck predicted (drag + lossy rail bounces, latency-compensated)
   to enter the goal mouth under any plausible rail restitution is blocked on
   the deepest line the paddle can reach with margin (minimum-time model,
   current velocity included). Before a bank is seen the paddle stands at the
   centre of the crossings the rail's restitution range allows; afterwards the
   crossings coincide. The block point sits slightly upstream of the path so a
   glancing contact still returns the puck. Re-predicted every tick.
2. Attack: a puck the paddle can reach is struck toward the requested route
   (straight / left bank / right bank) or, without a request, the route that
   avoids the rival mallet. Moving pucks are met with a one-timer whose
   contact normal is solved from the incoming and desired outgoing velocity;
   slow pucks get a set-up point behind the puck and a strike through it.
3. Guard: otherwise wait deep on the puck-to-goal-centre line, moving
   shallower (cooler) as the observed drive load levels rise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from airhockey.arrival import ArrivalDecoder
from airhockey.dynamics import workspace_in_sim
from airhockey.physics import TableConfig
from airhockey.puck_prediction import incoming_crossings

FRAME = 42
ONE_TIMER_FRACTION = 0.995
ONE_TIMER_MARK = 2 * np.sqrt((ONE_TIMER_FRACTION - 0.05) / 0.95) - 1


@dataclass
class ExpertConfig:
    # Sensing: the puck report is ~5-10 ms old and the command lands ~15 ms
    # after the decision; forecast the puck forward by this much.
    latency_s: float = 0.012
    # Threat detection
    threat_min_speed: float = 0.25
    goal_margin: float = 0.03  # widen the mouth for the threat test
    # Paddle-centre depths of the candidate block lines (ascending).
    block_depths: tuple = (0.085, 0.10, 0.12, 0.14, 0.17, 0.20, 0.24, 0.28, 0.33, 0.39,
                           0.46, 0.54, 0.63, 0.72)
    reach_latency_s: float = 0.035
    reach_accel: float = 64.0  # effective m/s^2 for bang-bang reach estimates
    # Blocking. Reach times come from a minimum-time model of the paddle
    # (current velocity included) calibrated through the decoder, guard and
    # firmware: rest-to-rest arrival within 1 cm took 2*sqrt(d/100) + ~18 ms.
    defense_latency_s: float = 0.010  # report age; the command delay is in the reach model
    defense_accel: float = 100.0
    defense_reach_latency_s: float = 0.018
    defense_time_scale: float = 1.0  # arrival duration = scale * minimum time
    block_margin_s: float = 0.03  # slack that makes a line comfortable
    # Upstream offset of the block point, metres per unit tan(incidence): a
    # contact on the side the puck comes from turns its normal toward the
    # far end, which returns the puck even at low paddle restitution.
    block_bias: float = 0.03
    block_bias_max: float = 0.05
    # Restitution scenarios for rail bounces not yet observed (paired
    # normal/tangential; nominal first, then the lowest and highest ratio
    # the rails are randomised over).
    bank_normal: tuple = (0.785, 0.72, 0.86)
    bank_tangential: tuple = (0.90, 0.98, 0.82)
    bounce_window_s: float = 0.03  # velocity report unusable this long after a bounce
    bounce_trust_report: bool = True  # keep a report that already fits a rail coefficient
    defense_fraction: float = 1.0
    # Guard stance. Depth follows the hottest observed load level through
    # these knots (piecewise linear, clamped). Blocking measured 96.6-97.8%
    # anywhere from 0.10 to 0.17 deep and 92% at 0.24; holding at the goal
    # centre settles near 0.97 / 0.92 / 0.82 / 0.73 of the drive limit at
    # 0.10 / 0.12 / 0.17 / 0.24 deep (gain 1.3), so the stance stays deep
    # while cool and the load settles near 0.81 when the drives run hot.
    guard_loads: tuple = (0.72, 0.82, 0.90)
    guard_depths: tuple = (0.11, 0.17, 0.30)
    guard_line_gain: float = 1.0  # 1: centred on the straight-shot fan
    guard_half_width: float = 0.16
    guard_fraction: float = 0.55
    # Attack
    attack_max_y: float = 0.86
    strike_speed: float = 7.0  # desired outgoing puck speed for one-timers
    paddle_restitution: float = 0.78
    one_timer_depths: tuple = (0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74)
    slow_speed: float = 0.9
    max_one_timer_speed: float = 14.0
    one_timer_paddle_speed: float = 2.5  # paddle normal speed the outgoing target allows
    one_timer_min_out: float = 3.0
    one_timer_min_time: float = 0.05
    one_timer_windup: float = 0.03  # s of paddle travel behind the contact point
    one_timer_commit: float = 0.20  # arrive when contact is this close in time
    one_timer_time_offset: float = 0.0
    one_timer_slack: float = 0.0
    one_timer_keep_slack: float = -0.08  # stay with a one-timer under way
    one_timer_margin: float = 0.03  # spare time for a line to count as feasible
    one_timer_latency: float = 0.025
    redirect_threats: bool = True
    redirect_slack: float = 0.05
    # Line strike (slow pucks): set up on the contact normal, then drive the
    # paddle along the ray from where it is through the contact point.
    run_up: float = 0.20  # preferred set-up distance behind the contact point
    run_up_min: float = 0.06
    strike_reach: float = 0.45  # strike from at most this far behind contact
    strike_angle: float = 22.0  # deg between approach ray and contact normal
    setup_cone: float = 18.0  # deg the set-up ray may tilt when the box is short
    strike_tolerance: float = 2.0  # deg from the set-up ray to start a strike
    strike_lateral_speed: float = 0.15  # m/s across the ray at strike start
    strike_latency: float = 0.030  # decision -> paddle accelerating
    strike_accel: float = 92.0  # effective m/s^2 along the strike ray
    command_delay: float = 0.015
    setup_fraction: float = 0.9
    setup_speed: float = 1.0  # m/s along the normal when arriving at the set-up point
    waypoint_speed: float = 1.5
    setup_time: float = 1.0  # arrival duration as a multiple of the reach estimate
    strike_hold: float = 0.10  # hold the strike command this close to contact
    path_margin: float = 0.015  # set-up path must keep this far from contact
    front_room: float = 0.6  # run-up <= this x the room beyond contact
    orbit_step: float = 60.0  # deg per waypoint when circling to the puck's back
    setup_clearance: float = 0.045  # paddle-edge clearance when going around
    # Aiming
    aim_offset: float = 0.0  # straight shots: from goal centre away from the rival
    bank_aim_offset: float = 0.0
    rival_block_radius: float = 0.09


@dataclass
class _State:
    puck: np.ndarray
    vel: np.ndarray
    paddle: np.ndarray
    paddle_vel: np.ndarray
    rival: np.ndarray
    load: np.ndarray
    request: np.ndarray
    accel: np.ndarray = None
    previous: np.ndarray = None


class ExpertPlayer:
    """Vectorised teacher with the NeuralPlayer evaluation interface."""

    history = 16
    shot_conditioned = True

    def __init__(self, options, config: ExpertConfig | None = None):
        self.environment_options = dict(options)
        self.config = config or ExpertConfig()
        ws = workspace_in_sim(bounds_mm=options["workspace_bounds_mm"])
        self.decoder = ArrivalDecoder([ws["min_x"], ws["max_x"], ws["min_y"], ws["max_y"]])
        self.low, self.high = self.decoder.low, self.decoder.high
        self.table = TableConfig()
        self.contact = self.table.puck_radius + self.table.paddle_radius
        self.last_mode = None

    # NeuralPlayer-compatible no-ops
    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    # ----------------------------------------------------------------- utils
    def _state(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        f = obs[:, :FRAME]
        frames = min(self.history, obs.shape[1] // FRAME)
        self._frames = obs[:, :FRAME * frames].reshape(len(obs), frames, FRAME)
        request = obs[:, -3:] if obs.shape[1] == FRAME * self.history + 3 else np.zeros((len(obs), 3))
        return _State(
            puck=f[:, 0:2].copy(), vel=f[:, 2:4] * 6.0, paddle=f[:, 4:6].copy(),
            paddle_vel=f[:, 6:8] * 6.0, rival=f[:, 8:10].copy(), load=f[:, 21:29].copy(),
            request=request, accel=f[:, 29:31] * 60.0, previous=f[:, 15:21].copy())

    def _encode(self, position, velocity, duration, fraction):
        a = np.zeros((len(position), 6))
        a[:, :2] = 2 * (np.clip(position, self.low, self.high) - self.low) / (self.high - self.low) - 1
        a[:, 2:4] = np.clip(velocity / self.decoder.config.velocity_scale, -1, 1)
        cfg = self.decoder.config
        a[:, 4] = 2 * (np.clip(duration, cfg.min_time, cfg.max_time) - cfg.min_time) / (cfg.max_time - cfg.min_time) - 1
        a[:, 5] = 2 * np.sqrt(np.clip((np.asarray(fraction) - 0.05) / 0.95, 0, 1)) - 1
        return np.clip(a, -1, 1)

    def _reach_time(self, start, goal):
        d = np.linalg.norm(goal - start, axis=-1)
        return self.config.reach_latency_s + 2 * np.sqrt(d / self.config.reach_accel)

    def _forecast(self, puck, vel, seconds):
        """Free flight with drag and side-rail bounces for a short horizon."""
        cfg = self.table
        x, y = puck[:, 0].copy(), puck[:, 1].copy()
        vx, vy = vel[:, 0].copy(), vel[:, 1].copy()
        a = cfg.puck_friction * 9.81
        b = cfg.PUCK_DRAG_B
        dt = seconds / 4
        lo, hi = cfg.puck_radius, cfg.width - cfg.puck_radius
        for _ in range(4):
            speed = np.hypot(vx, vy)
            decel = np.minimum(a + b * speed**2, speed / dt)
            scale = np.where(speed > 1e-9, 1 - decel * dt / np.maximum(speed, 1e-9), 0)
            x += vx * dt
            y += vy * dt
            vx *= scale
            vy *= scale
            left, right = x < lo, x > hi
            x = np.where(left, 2 * lo - x, np.where(right, 2 * hi - x, x))
            vx = np.where(left | right, -vx * cfg.wall_restitution, vx)
            vy = np.where(left | right, vy * cfg.wall_tangential, vy)
        return np.column_stack((x, y)), np.column_stack((vx, vy))

    # ------------------------------------------------------------- aiming
    def _shot_direction(self, puck, s: _State, route=None):
        """Unit outgoing direction for a straight / left / right shot from puck.

        Every route aims at the goal centre (the widest margin for the
        full-puck-width test); banks invert the lossy rail law (normal 0.785,
        tangential 0.90) so the rebound, not the incoming ray, finds the mouth.
        """
        cfg, c = self.table, self.config
        n = len(puck)
        r = cfg.puck_radius
        en, et = cfg.wall_restitution, cfg.wall_tangential
        if route is None:
            route = self._choose_route(puck, s)
        side = np.where(s.rival[:, 0] > 0.5, -1.0, 1.0)  # away from the rival
        dy = np.maximum(cfg.height - puck[:, 1], 0.2)
        aim_x = 0.5 + side * c.aim_offset
        straight = np.column_stack((aim_x - puck[:, 0], dy))
        x_left = 0.5 + c.bank_aim_offset
        x_right = 0.5 - c.bank_aim_offset
        k_left = ((x_left - r) * (et / en) + (puck[:, 0] - r)) / dy
        k_right = ((cfg.width - r - x_right) * (et / en) + (cfg.width - r - puck[:, 0])) / dy
        left = np.column_stack((-k_left, np.ones(n)))
        right = np.column_stack((k_right, np.ones(n)))
        d = np.where((route == 1)[:, None], left, np.where((route == 2)[:, None], right, straight))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def _choose_route(self, puck, s: _State):
        """Requested route if any (1 left, 2 right, 3 straight); else open line."""
        req = s.request
        requested = np.where(req.max(axis=1) > 0.5, req.argmax(axis=1) + 1, 0)
        # Without a request: straight unless the rival sits on that line.
        cfg = self.table
        to_goal = np.column_stack((0.5 - puck[:, 0], cfg.height - puck[:, 1]))
        to_goal /= np.linalg.norm(to_goal, axis=1, keepdims=True)
        rel = s.rival - puck
        along = (rel * to_goal).sum(axis=1)
        across = np.abs(rel[:, 0] * to_goal[:, 1] - rel[:, 1] * to_goal[:, 0])
        blocked = (along > 0) & (across < self.config.rival_block_radius + self.contact)
        bank = np.where(s.rival[:, 0] > 0.5, 1, 2)  # bank away from the rival
        free = np.where(blocked, bank, 3)
        return np.where(requested > 0, requested, free)

    def _drift(self, puck, vel, seconds):
        """Per-env free flight (drag, side and back rails) for slow pucks."""
        cfg = self.table
        x, y = puck[:, 0].copy(), puck[:, 1].copy()
        vx, vy = vel[:, 0].copy(), vel[:, 1].copy()
        a, b = cfg.puck_friction * 9.81, cfg.PUCK_DRAG_B
        dt = np.maximum(np.asarray(seconds, dtype=float), 0) / 6
        lo, hi = cfg.puck_radius, cfg.width - cfg.puck_radius
        for _ in range(6):
            speed = np.hypot(vx, vy)
            decel = np.minimum(a + b * speed**2, speed / np.maximum(dt, 1e-9))
            scale = np.where(speed > 1e-9, 1 - decel * dt / np.maximum(speed, 1e-9), 0)
            x += vx * dt
            y += vy * dt
            vx *= scale
            vy *= scale
            left, right, back = x < lo, x > hi, y < lo
            x = np.where(left, 2 * lo - x, np.where(right, 2 * hi - x, x))
            vx = np.where(left | right, -vx * cfg.wall_restitution, vx)
            vy = np.where(left | right, vy * cfg.wall_tangential, vy)
            y = np.where(back, 2 * lo - y, y)
            vy = np.where(back, -vy * cfg.wall_restitution, vy)
        return np.column_stack((x, y)), np.column_stack((vx, vy))

    def _box_ray(self, origin, u, margin=0.004):
        """Distance along unit rows u from origin to the workspace boundary."""
        lo, hi = self.low + margin, self.high - margin
        dist = np.full(len(origin), np.inf)
        for k in (0, 1):
            pos = u[:, k] > 1e-9
            neg = u[:, k] < -1e-9
            t = np.where(pos, (hi[k] - origin[:, k]) / np.where(pos, u[:, k], 1),
                         np.where(neg, (lo[k] - origin[:, k]) / np.where(neg, u[:, k], -1), np.inf))
            dist = np.minimum(dist, t)
        return np.maximum(dist, 0)

    # ------------------------------------------------------------- defense
    def _min_time(self, start, velocity, goal, accel):
        """Minimum time to come to rest at goal under a vector accel cap.

        Along the displacement this is the exact bang-bang time from the
        current velocity (including overshoot when it cannot stop in time);
        velocity across it costs half its braking time on top.
        """
        rel = goal - start
        d = np.linalg.norm(rel, axis=-1)
        u = rel / np.maximum(d, 1e-9)[..., None]
        along = (velocity * u).sum(axis=-1)
        across = np.abs(velocity[..., 0] * u[..., 1] - velocity[..., 1] * u[..., 0])
        stops = (along <= 0) | (along**2 <= 2 * accel * d)
        peak = np.sqrt(np.maximum(accel * d + along**2 / 2, 0))
        direct = (2 * peak - along) / accel
        overshoot = along / accel + 2 * np.sqrt(np.maximum(along**2 / (2 * accel) - d, 0) / accel)
        return np.where(stops, direct, overshoot) + 0.5 * across / accel

    def _scenario_crossings(self, states, lines):
        """incoming_crossings of per-scenario states [m, n, 4]: [m, n, k, 4]."""
        c = self.config
        key = (tuple(c.bank_normal), tuple(c.bank_tangential))
        if getattr(self, "_scenario_key", None) != key:
            self._scenario_key = key
            # No speed clamp: the forecast must follow a puck the table allows.
            self._scenario_tables = [TableConfig(wall_restitution=en, wall_tangential=et, max_puck_speed=40.0)
                                     for en, et in zip(*key)]
        return np.stack([incoming_crossings(state, lines, config=table, max_seconds=2.0)
                         for state, table in zip(states, self._scenario_tables)])

    def _flight(self, pos, vel, seconds, en, et):
        """Short free flight with drag and side-rail bounces; [..., 2] arrays."""
        cfg = self.table
        p, v = pos.copy(), vel.copy()
        a, b = cfg.puck_friction * 9.81, cfg.PUCK_DRAG_B
        steps = max(1, int(np.ceil(seconds / 0.005)))
        dt = seconds / steps
        lo, hi = cfg.puck_radius, cfg.width - cfg.puck_radius
        for _ in range(steps):
            speed = np.linalg.norm(v, axis=-1)
            decel = np.minimum(a + b * speed**2, speed / dt)
            p = p + v * dt
            v = v * np.where(speed > 1e-9, 1 - decel * dt / np.maximum(speed, 1e-9), 0)[..., None]
            for wall, beyond in ((lo, p[..., 0] < lo), (hi, p[..., 0] > hi)):
                # Reflect the overshoot at the reduced normal speed.
                p[..., 0] = np.where(beyond, wall + (wall - p[..., 0]) * en, p[..., 0])
                v[..., 0] = np.where(beyond, -v[..., 0] * en, v[..., 0])
                v[..., 1] = np.where(beyond, v[..., 1] * et, v[..., 1])
        return p, v

    def _defense_states(self, s):
        """Puck state now under each restitution scenario, [m, n, 4].

        The velocity report is a slope over the last ~30 ms of camera
        samples; while those straddle a side-rail bounce it points nowhere
        useful. Detected from an older frame whose flight reaches the rail
        inside that window (and agrees with the current position); unless
        the report already fits a rail coefficient, the post-bounce velocity
        is then that frame's velocity through the bounce, per scenario, at
        the current reported position.
        """
        c, cfg = self.config, self.table
        en = np.asarray(c.bank_normal, dtype=float)[:, None]
        et = np.asarray(c.bank_tangential, dtype=float)[:, None]
        m, n = len(en), len(s.puck)
        pos = np.broadcast_to(s.puck, (m, n, 2)).copy()
        vel = np.broadcast_to(s.vel, (m, n, 2)).copy()
        frames = self._frames
        dt = self.decoder.config.action_dt
        j = 2
        if c.bounce_window_s > 0 and frames.shape[1] > j:
            age = j * dt
            p_old, v_old = frames[:, j, 0:2], frames[:, j, 2:4] * 6.0
            lo, hi = cfg.puck_radius, cfg.width - cfg.puck_radius
            vx = v_old[:, 0]
            to_rail = np.where(vx < -0.3, (lo - p_old[:, 0]) / np.minimum(vx, -1e-9),
                               np.where(vx > 0.3, (hi - p_old[:, 0]) / np.maximum(vx, 1e-9), np.inf))
            since = age - to_rail  # bounce this long before the current report
            p_pred, v_pred = self._flight(np.broadcast_to(p_old, (m, n, 2)), np.broadcast_to(v_old, (m, n, 2)),
                                          age, en, et)
            agree = np.linalg.norm(p_pred[0] - s.puck, axis=1) < 0.02 + 0.1 * np.linalg.norm(v_old, axis=1) * age
            straddle = (since > -0.002) & (since < c.bounce_window_s) & agree
            if c.bounce_trust_report:
                # A report already consistent with some rail coefficient is
                # a clean post-bounce fit: keep it, it knows the actual rail.
                ratio_n = -s.vel[:, 0] / np.where(np.abs(vx) > 1e-9, vx, 1e-9)
                ratio_t = s.vel[:, 1] / np.where(np.abs(v_old[:, 1]) > 1e-9, v_old[:, 1], 1e-9)
                clean = (ratio_n > en.min() - 0.08) & (ratio_n < en.max() + 0.08) & \
                        (ratio_t > et.min() - 0.08) & (ratio_t < et.max() + 0.06)
                straddle &= ~clean
            vel = np.where(straddle[None, :, None], v_pred, vel)
        pos, vel = self._flight(pos, vel, c.defense_latency_s, en, et)
        return np.concatenate((pos, vel), axis=-1)

    def _defense(self, s: _State, puck, vel):
        cfg, c = self.table, self.config
        n = len(puck)
        # The shared forecast is tuned for striking; defend from the report.
        states = self._defense_states(s)  # [m, n, 4]
        puck, vel = states[0, :, :2], states[0, :, 2:]
        incoming = vel[:, 1] < -c.threat_min_speed
        goal_line = self._scenario_crossings(states, np.array([cfg.puck_radius]))[:, :, 0]
        on_goal = np.isfinite(goal_line[..., 1]) & (
            np.abs(goal_line[..., 0] - 0.5) < cfg.goal_width / 2 + c.goal_margin)
        threat = incoming & on_goal.any(axis=0)
        depths = np.asarray(c.block_depths, dtype=float)
        cross = self._scenario_crossings(states, depths)  # [m, n, k, 4]
        x_m, t_m = cross[..., 0], cross[..., 1]
        finite = np.isfinite(t_m)
        nominal = cross[0]
        valid = finite[0] & (puck[:, 1:2] > depths[None, :])
        # Stand at the centre of the plausible crossings (they coincide once
        # the bounce has been seen) and be there by the earliest one.
        x_lo = np.where(valid, np.where(finite, x_m, np.inf).min(axis=0), 0.5)
        x_hi = np.where(valid, np.where(finite, x_m, -np.inf).max(axis=0), 0.5)
        x_mid = (x_lo + x_hi) / 2
        speed_k = np.hypot(nominal[..., 2], nominal[..., 3])
        earliest = np.where(finite, t_m, np.inf).min(axis=0)
        t_contact = earliest - self.contact / np.maximum(np.where(valid, speed_k, 1.0), 0.3)
        tan = np.where(valid, nominal[..., 2] / np.maximum(-nominal[..., 3], 1e-3), 0.0)
        shift = np.clip(-c.block_bias * tan, -c.block_bias_max, c.block_bias_max)
        target = np.stack((np.clip(x_mid + shift, self.low[0], self.high[0]),
                           np.broadcast_to(np.clip(depths, self.low[1], self.high[1]), x_mid.shape)), axis=-1)
        reach = self._min_time(s.paddle[:, None, :], s.paddle_vel[:, None, :], target, c.defense_accel) \
            + c.defense_reach_latency_s
        slack = np.where(valid, t_contact - reach, -np.inf)
        # Deepest line that leaves a comfortable margin; otherwise best slack.
        comfortable = slack > c.block_margin_s
        deepest = np.argmax(comfortable, axis=1)
        best = np.argmax(slack, axis=1)
        pick = np.where(comfortable.any(axis=1), deepest, best)
        rows = np.arange(n)
        block = target[rows, pick]
        has = valid[rows, pick]
        # Puck already past every line: stay between it and the goal.
        fallback = np.column_stack((np.clip(puck[:, 0], 0.5 - cfg.goal_width / 2, 0.5 + cfg.goal_width / 2),
                                    np.full(n, self.low[1])))
        block = np.where(has[:, None], block, fallback)
        duration = c.defense_time_scale * self._min_time(s.paddle, s.paddle_vel, block, c.defense_accel)
        act = self._encode(block, np.zeros((n, 2)), duration, np.full(n, c.defense_fraction))
        self._defense_trace = dict(line=np.where(has, depths[pick], np.nan), slack=slack[rows, pick])
        return threat, act

    # ------------------------------------------------------------- attack
    def _setup_ray(self, C, nrm):
        """Approach direction (within the strike cone) and run-up behind C.

        Straight behind C along the normal unless the box cuts the run-up
        short (rails, back wall); then the tilt within the cone that buys the
        most normal speed, cos(tilt) * sqrt(run-up).
        """
        c = self.config
        def room(u):
            # Behind C for the run-up, and in front of it for braking: the
            # decoder aims the firmware one braking distance (v^2/1.28a) ahead,
            # and a target clipped by the box bends the drive before contact.
            back = self._box_ray(C, -u, margin=0.01)
            front = self._box_ray(C, u, margin=0.004)
            return np.clip(np.minimum(back, c.front_room * front), 0.0, c.run_up)

        best_u, best_L = nrm.copy(), room(nrm)
        score = np.sqrt(best_L)
        limit = c.setup_cone
        for deg in (-limit, -limit * 2 / 3, -limit / 3, limit / 3, limit * 2 / 3, limit):
            a = np.radians(deg)
            u = np.column_stack((np.cos(a) * nrm[:, 0] - np.sin(a) * nrm[:, 1],
                                 np.sin(a) * nrm[:, 0] + np.cos(a) * nrm[:, 1]))
            L = room(u)
            sc = np.cos(a) * np.sqrt(L)
            better = sc > score * 1.03
            best_u = np.where(better[:, None], u, best_u)
            best_L = np.where(better, L, best_L)
            score = np.where(better, sc, score)
        return best_u, best_L

    def _line_strike(self, s: _State, puck, vel):
        """Slow puck: (act, striking) for a set-up / strike along the contact normal.

        A paddle whose centre passes through the contact point C = P - n*R on a
        ray with a positive component along n meets the puck AT C, whatever the
        ray's angle, and the frictionless impulse sends it along n. So the
        strike is a straight drive from the paddle's (latency-advanced) position
        through C to the far edge of the box at the full cap: the firmware
        points its desired velocity at that target, so a paddle at rest or
        already moving along the ray travels a straight line. The set-up point
        sits on the normal behind C with as much run-up as the box allows, and
        is approached around the puck when the paddle is in front of it.
        """
        cfg, c = self.table, self.config
        n = len(puck)
        R = self.contact
        e = c.paddle_restitution
        W = s.paddle_vel + (s.accel if s.accel is not None else 0.0) * 0.01
        X = s.paddle + W * c.command_delay
        # Contact time when striking from here: latency + constant-accel run.
        t = np.full(n, c.strike_latency + 0.05)
        for _ in range(3):
            P, V = self._drift(puck, vel, t)
            d = self._shot_direction(P, s)
            dist = np.linalg.norm(P - d * R - X, axis=1)
            v0 = np.maximum((W * d).sum(axis=1), 0)
            v_c = np.sqrt(v0**2 + 2 * c.strike_accel * np.minimum(dist, c.strike_reach))
            out = np.clip((1 + e) * v_c, 2.0, cfg.max_puck_speed)
            want = d * out[:, None] - V
            nrm = want / np.maximum(np.linalg.norm(want, axis=1, keepdims=True), 1e-9)
            C = P - nrm * R
            dist = np.linalg.norm(C - X, axis=1)
            run = (-v0 + np.sqrt(v0**2 + 2 * c.strike_accel * dist)) / c.strike_accel
            t = c.strike_latency + run
        u = (C - X) / np.maximum(dist[:, None], 1e-9)
        cos_t = (u * nrm).sum(axis=1)
        lateral = W - (W * u).sum(axis=1, keepdims=True) * u
        far = self._box_ray(X, u)
        # Run-up the box allows behind C within the strike cone.
        u_set_now, L = self._setup_ray(C, nrm)
        need = np.minimum(c.run_up_min, 0.8 * L)
        prev = s.previous
        was_striking = (prev[:, 5] > 0.97) & (prev[:, 4] < -0.97) if prev is not None else np.zeros(n, bool)
        # The physics resolves contact after up to v*2.5 ms of overlap, which
        # rotates the normal unless the drive is parallel to it: strike only
        # on the chosen set-up ray (straight behind C unless the box forbids).
        aligned = ((cos_t > np.cos(np.radians(c.strike_angle)))
                   & ((u * u_set_now).sum(axis=1) > np.cos(np.radians(c.strike_tolerance))))
        ready = (aligned & (dist >= need) & (dist <= c.strike_reach)
                 & (np.linalg.norm(lateral, axis=1) < c.strike_lateral_speed)
                 & (far - dist > (dist - R) * 0.0 + 0.02))
        # Keep driving a strike already under way while the ray stays sound.
        striking = ready | (was_striking & (cos_t > np.cos(np.radians(c.strike_angle + 15)))
                            & (far > dist) & (dist < c.strike_reach + 0.1))
        target = X + u * far[:, None]
        strike = self._encode(target, u * 3.0, np.full(n, 0.04), np.ones(n))
        # Within the last few centimetres re-aiming only adds noise (the
        # latency-advanced paddle may already be past C): hold the command.
        if prev is not None:
            u_prev = prev[:, 2:4] / np.maximum(np.linalg.norm(prev[:, 2:4], axis=1, keepdims=True), 1e-6)
            rel = puck - s.paddle
            ahead = (rel * u_prev).sum(axis=1)
            off = np.abs(rel[:, 0] * u_prev[:, 1] - rel[:, 1] * u_prev[:, 0])
            hold = was_striking & (ahead > 0.3 * R) & (ahead < R + c.strike_hold) & (off < R)
            strike = np.where(hold[:, None], prev, strike)
            striking = striking | hold
        # --- set-up: behind the puck's position at the expected contact time
        S0 = C - nrm * L[:, None]
        t_set = np.clip(self._reach_time(X, S0) + c.strike_latency
                        + np.sqrt(2 * np.maximum(L, 1e-3) / c.strike_accel), 0, 0.6)
        Ps, Vs = self._drift(puck, vel, t_set)
        ds = self._shot_direction(Ps, s)
        want = ds * out[:, None] - Vs
        ns = want / np.maximum(np.linalg.norm(want, axis=1, keepdims=True), 1e-9)
        Cs = Ps - ns * R
        u_set, Ls = self._setup_ray(Cs, ns)
        S = np.clip(Cs - u_set * Ls[:, None], self.low, self.high)
        # Orbit the puck (radius R + clearance) toward its back when the
        # straight path to S would touch it.
        seg = S - X
        seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)
        tpar = np.clip(((Ps - X) * seg).sum(axis=1) / seg_len**2, 0, 1)
        closest = X + seg * tpar[:, None]
        collide = np.linalg.norm(closest - Ps, axis=1) < R + c.path_margin
        clear = R + c.setup_clearance
        back = -u_set
        perp = np.column_stack((-back[:, 1], back[:, 0]))
        rel = X - Ps
        phi = np.arctan2((rel * perp).sum(axis=1), (rel * back).sum(axis=1))
        step = np.radians(c.orbit_step)
        phi_wp = phi - np.sign(phi) * np.minimum(np.abs(phi), step)
        wp = Ps + clear * (np.cos(phi_wp)[:, None] * back + np.sin(phi_wp)[:, None] * perp)
        wp = np.clip(wp, self.low, self.high)
        go = np.where(collide[:, None], wp, S)
        # Arrive at S already moving along the approach ray (a flying set-up
        # instead of a settle); a waypoint is passed toward S.
        onward = np.where(collide[:, None], S - wp, u_set)
        onward /= np.maximum(np.linalg.norm(onward, axis=1, keepdims=True), 1e-9)
        v_end = onward * np.where(collide, c.waypoint_speed, c.setup_speed)[:, None]
        duration = np.clip(c.setup_time * self._reach_time(X, go), 0.04, 0.25)
        move = self._encode(go, v_end, duration, np.full(n, c.setup_fraction))
        act = np.where(striking[:, None], strike, move)
        self._debug = dict(striking=striking, dist=dist, cos=cos_t, L=L, collide=collide,
                           lat=np.linalg.norm(lateral, axis=1), v0=(W * u).sum(axis=1), hold=hold if prev is not None else striking)
        return act, striking

    def _one_timer(self, s: _State, puck, vel):
        """Moving puck: (ok, act, slack) for meeting it on its predicted path.

        At each candidate crossing line the contact normal is solved from the
        incoming and the desired outgoing velocity, n = unit(s*d - v_in), and
        the paddle must arrive at P - n*R moving along n at
        v_in.n + |s*d - v_in|/(1+e). The outgoing speed s is the fastest whose
        paddle speed stays inside the arrival's velocity range, so a hard shot
        is returned hard and a slow one is still aimed. The line with the most
        time to spare is used; a contact further away than the arrival horizon
        is waited for at the wind-up point behind the contact point.
        """
        cfg, c = self.table, self.config
        n = len(puck)
        R = self.contact
        e = c.paddle_restitution
        depths = np.asarray(c.one_timer_depths)
        k = len(depths)
        state = np.column_stack((puck, vel))
        cross = incoming_crossings(state, depths, max_seconds=1.0)
        x_k, t_k, vx_k, vy_k = cross[..., 0], cross[..., 1], cross[..., 2], cross[..., 3]
        valid = np.isfinite(t_k) & (puck[:, 1:2] > depths[None, :]) & (t_k > 0.0)
        flat_p = np.stack((x_k, np.broadcast_to(depths, x_k.shape)), -1).reshape(-1, 2)
        flat_v = np.stack((vx_k, vy_k), -1).reshape(-1, 2)
        flat_p = np.where(np.isfinite(flat_p), flat_p, 0.5)
        flat_v = np.where(np.isfinite(flat_v), flat_v, 0.0)
        rep = _State(puck=flat_p, vel=flat_v, paddle=np.repeat(s.paddle, k, 0),
                     paddle_vel=np.repeat(s.paddle_vel, k, 0), rival=np.repeat(s.rival, k, 0),
                     load=np.repeat(s.load, k, 0), request=np.repeat(s.request, k, 0))
        dk = self._shot_direction(flat_p, rep)
        sp_in = np.linalg.norm(flat_v, axis=1)
        s_out = np.clip(e * sp_in + (1 + e) * c.one_timer_paddle_speed, c.one_timer_min_out, cfg.max_puck_speed)
        want = dk * s_out[:, None] - flat_v
        normal = want / np.maximum(np.linalg.norm(want, axis=1, keepdims=True), 1e-9)
        vpn = (flat_v * normal).sum(axis=1) + np.linalg.norm(want, axis=1) / (1 + e)
        pad = flat_p - normal * R
        pad_v = normal * vpn[:, None]
        wind = pad - normal * np.clip(vpn, 0, None)[:, None] * c.one_timer_windup
        m = 0.003
        ok = ((pad >= self.low - m) & (pad <= self.high + m)).all(axis=1) & (normal[:, 1] > 0.2) \
            & (np.abs(pad_v) <= 3.0).all(axis=1) & (vpn > -1.5)
        gap = np.linalg.norm(pad - np.repeat(s.paddle, k, 0), axis=1)
        reach = c.one_timer_latency + 2 * np.sqrt(gap / c.strike_accel)
        slack = np.where(valid.reshape(-1) & ok, t_k.reshape(-1) - reach, -np.inf).reshape(n, k)
        # Meet the puck where the paddle already is: the feasible line with
        # the shortest move (stable from tick to tick, unlike max slack, which
        # always prefers the deepest line and has the paddle chase backward).
        feasible = (slack > c.one_timer_margin) & (t_k > c.one_timer_min_time)
        nearest = np.argmin(np.where(feasible, gap.reshape(n, k), np.inf), axis=1)
        best = np.where(feasible.any(axis=1), nearest, np.argmax(slack, axis=1))
        prev = s.previous
        under_way = (np.abs(prev[:, 5] - ONE_TIMER_MARK) < 0.002) if prev is not None else np.zeros(n, bool)
        if prev is not None:
            # A one-timer under way keeps its line (the plan backs the paddle
            # up for its run-up, which a rest-to-rest reach estimate misreads).
            planned = self.low + (np.clip(prev[:, :2], -1, 1) + 1) / 2 * (self.high - self.low)
            to_plan = np.linalg.norm(pad.reshape(n, k, 2) - planned[:, None, :], axis=2)
            keep = np.argmin(np.where(slack > c.one_timer_keep_slack, to_plan, np.inf), axis=1)
            best = np.where(under_way & np.isfinite(slack).any(axis=1), keep, best)
        rows = np.arange(n)
        idx = rows * k + best
        best_slack = slack[rows, best]
        t_hit = t_k[rows, best] + c.one_timer_time_offset
        # Full cap, marked by a fraction a hair under 1 so a one-timer under
        # way is recognisable from the previous action in the observation.
        mark = np.full(n, ONE_TIMER_FRACTION)
        arrive = self._encode(pad[idx], pad_v[idx], t_hit, mark)
        wait = self._encode(np.clip(wind[idx], self.low, self.high), np.zeros((n, 2)),
                            np.clip(t_hit - c.one_timer_commit, 0.04, 0.25), mark)
        act = np.where((t_hit > c.one_timer_commit)[:, None], wait, arrive)
        prev = s.previous
        under_way = (np.abs(prev[:, 5] - ONE_TIMER_MARK) < 0.002) if prev is not None else np.zeros(n, bool)
        ok = np.isfinite(best_slack) & ((best_slack > c.one_timer_slack)
                                        | (under_way & (best_slack > c.one_timer_keep_slack)))
        return ok, act, np.where(under_way & ok, np.maximum(best_slack, c.redirect_slack + 1e-3), best_slack)

    def _attack(self, s: _State, puck, vel):
        cfg, c = self.table, self.config
        speed = np.linalg.norm(vel, axis=1)
        reachable = puck[:, 1] < c.attack_max_y
        slow = speed < c.slow_speed
        slow_act, _ = self._line_strike(s, puck, vel)
        one_ok, one, slack = self._one_timer(s, puck, vel)
        one_ok &= speed < c.max_one_timer_speed
        attack = (reachable & slow) | one_ok
        act = np.where((slow | ~one_ok)[:, None], slow_act, one)
        self._debug.update(one_ok=one_ok, one_slack=slack)
        return attack, act

    # ------------------------------------------------------------- guard
    def _guard(self, s: _State, puck, vel):
        cfg, c = self.table, self.config
        n = len(puck)
        depth = np.interp(s.load.max(axis=1), c.guard_loads, c.guard_depths)
        # On the puck-to-goal-centre line at this depth: the centre of the
        # fan of straight shots, which banks (slower to arrive) also cross.
        py = np.maximum(puck[:, 1], depth + 0.05)
        x = 0.5 + (puck[:, 0] - 0.5) * c.guard_line_gain * depth / py
        x = np.clip(x, 0.5 - c.guard_half_width, 0.5 + c.guard_half_width)
        target = np.column_stack((x, depth))
        return self._encode(target, np.zeros((n, 2)), np.full(n, 0.12), np.full(n, c.guard_fraction))

    # ------------------------------------------------------------- policy
    def act(self, observation, stochastic=False):
        s = self._state(observation)
        puck, vel = self._forecast(s.puck, s.vel, self.config.latency_s)
        threat, defend = self._defense(s, puck, vel)
        attack, strike = self._attack(s, puck, vel)
        if self.config.redirect_threats:
            # A threat whose one-timer has comfortable time is returned on the
            # requested route instead of blocked; the contact is on the threat's
            # own path, so it still stops the goal.
            threat = threat & ~(self._debug["one_ok"] & (self._debug["one_slack"] > self.config.redirect_slack))
        guard = self._guard(s, puck, vel)
        action = np.where(threat[:, None], defend, np.where(attack[:, None], strike, guard))
        self.last_mode = np.where(threat, 0, np.where(attack, 1, 2))
        return action.astype(np.float32)


PRODUCTION_ENVIRONMENT = "runs/rail30-100-20261001-v1/agent.pt"


def parse_spec(spec: str) -> ExpertConfig:
    """`expert` or `expert:key=value,key=value` (ExpertConfig overrides)."""
    config = ExpertConfig()
    _, _, overrides = str(spec).partition(":")
    for item in filter(None, overrides.split(",")):
        key, _, value = item.partition("=")
        if not hasattr(config, key):
            raise ValueError(f"unknown expert setting {key!r}")
        current = getattr(config, key)
        if isinstance(current, tuple):
            parsed = tuple(float(v) for v in value.split("/"))
        elif isinstance(current, bool):
            parsed = value.lower() in ("1", "true", "yes")
        else:
            parsed = type(current)(value)
        setattr(config, key, parsed)
    return config


def load_expert(spec: str, environment_from=PRODUCTION_ENVIRONMENT):
    """(player, state) in the shape eval_neural_player.load returns."""
    from airhockey.neural_setup import checkpoint_environment
    options = checkpoint_environment(environment_from)
    player = ExpertPlayer(options, parse_spec(spec))
    state = dict(step=0, args=dict(report_sensing=True, continuous_rallies=True), expert=str(spec))
    return player, state
