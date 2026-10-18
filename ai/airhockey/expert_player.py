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
   to enter the goal mouth is blocked on the line that gives the most time
   margin, deepest first. Banks are handled by the same forecast and are
   re-predicted every tick, so a mis-estimated rail is corrected after it.
2. Attack: a puck the paddle can reach is struck toward the requested route
   (straight / left bank / right bank) or, without a request, the route that
   avoids the rival mallet. Moving pucks are met with a one-timer whose
   contact normal is solved from the incoming and desired outgoing velocity;
   slow pucks get a set-up point behind the puck and a strike through it.
3. Guard: otherwise wait on the puck-to-goal line, deep enough to cover banks,
   moving shallower (cooler) as the thermal model's levels rise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from airhockey.arrival import ArrivalDecoder
from airhockey.dynamics import workspace_in_sim
from airhockey.physics import TableConfig
from airhockey.puck_prediction import incoming_crossings

FRAME = 42


@dataclass
class ExpertConfig:
    # Sensing: the puck report is ~5-10 ms old and the command lands ~15 ms
    # after the decision; forecast the puck forward by this much.
    latency_s: float = 0.012
    # Threat detection
    threat_min_speed: float = 0.25
    goal_margin: float = 0.03  # widen the mouth for the threat test
    block_depths: tuple = (0.12, 0.16, 0.20, 0.26, 0.34, 0.44, 0.56)
    reach_latency_s: float = 0.035
    reach_accel: float = 64.0  # effective m/s^2 for bang-bang reach estimates
    # Guard stance
    guard_depth: float = 0.17
    guard_hot_depth: float = 0.36
    guard_hot_start: float = 0.70
    guard_hot_full: float = 0.88
    guard_line_gain: float = 0.65
    guard_half_width: float = 0.16
    guard_fraction: float = 0.55
    # Attack
    attack_max_y: float = 0.86
    strike_speed: float = 7.0  # desired outgoing puck speed for one-timers
    paddle_restitution: float = 0.78
    run_up: float = 0.16
    follow: float = 0.26
    align_tolerance: float = 0.028
    behind_tolerance: float = 0.6  # fraction of contact distance
    one_timer_depths: tuple = (0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74)
    slow_speed: float = 0.9
    max_one_timer_speed: float = 4.5
    # Aiming
    aim_offset: float = 0.10  # from goal centre toward the far post
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
        request = obs[:, -3:] if obs.shape[1] == FRAME * self.history + 3 else np.zeros((len(obs), 3))
        return _State(
            puck=f[:, 0:2].copy(), vel=f[:, 2:4] * 6.0, paddle=f[:, 4:6].copy(),
            paddle_vel=f[:, 6:8] * 6.0, rival=f[:, 8:10].copy(), load=f[:, 21:29].copy(),
            request=request)

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
        """Unit outgoing direction for a straight / left / right shot from puck."""
        cfg = self.table
        n = len(puck)
        r = cfg.puck_radius
        en, et = cfg.wall_restitution, cfg.wall_tangential
        far_y = cfg.height
        rival = s.rival
        # Aim at the far post away from the rival mallet.
        side = np.where(rival[:, 0] > 0.5, -1.0, 1.0)
        aim_x = 0.5 + side * self.config.aim_offset
        if route is None:
            route = self._choose_route(puck, s)
        dy = np.maximum(far_y - puck[:, 1], 0.2)
        straight = np.column_stack((aim_x - puck[:, 0], dy))
        k_left = ((aim_x - r) * (et / en) + (puck[:, 0] - r)) / dy
        k_right = ((cfg.width - r - aim_x) * (et / en) + (cfg.width - r - puck[:, 0])) / dy
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

    # ------------------------------------------------------------- defense
    def _defense(self, s: _State, puck, vel):
        cfg, c = self.table, self.config
        n = len(puck)
        incoming = vel[:, 1] < -c.threat_min_speed
        state = np.column_stack((puck, vel))
        goal_line = incoming_crossings(state, [cfg.puck_radius], max_seconds=2.0)[:, 0]
        threat = incoming & np.isfinite(goal_line[:, 1]) & (
            np.abs(goal_line[:, 0] - 0.5) < cfg.goal_width / 2 + c.goal_margin)
        depths = np.asarray(c.block_depths)
        lines = depths + self.contact
        cross = incoming_crossings(state, lines, max_seconds=2.0)  # [n, k, 4]
        x_k, t_k = cross[..., 0], cross[..., 1]
        valid = np.isfinite(t_k) & (puck[:, 1:2] > lines[None, :])
        target = np.stack((np.clip(x_k, self.low[0], self.high[0]),
                           np.broadcast_to(depths, x_k.shape)), axis=-1)
        reach = self._reach_time(s.paddle[:, None, :], np.where(valid[..., None], target, s.paddle[:, None, :]))
        slack = np.where(valid, t_k - reach, -np.inf)
        # Deepest line that leaves a comfortable margin; otherwise best slack.
        comfortable = slack > 0.03
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
        act = self._encode(block, np.zeros((n, 2)), np.full(n, 0.04), np.ones(n))
        return threat, act

    # ------------------------------------------------------------- attack
    def _attack(self, s: _State, puck, vel):
        cfg, c = self.table, self.config
        n = len(puck)
        speed = np.linalg.norm(vel, axis=1)
        reachable = puck[:, 1] < c.attack_max_y
        slow = speed < c.slow_speed
        # --- slow puck: set up behind it, then strike through it
        future, _ = self._forecast(puck, vel, 0.06)
        d = self._shot_direction(future, s)
        contact_pt = future - d * self.contact
        setup = future - d * (self.contact + c.run_up)
        rel = s.paddle - future
        along = (rel * d).sum(axis=1)
        across = np.abs(rel[:, 0] * d[:, 1] - rel[:, 1] * d[:, 0])
        behind = (along < -self.contact * c.behind_tolerance) & (across < c.align_tolerance + 0.02 * (-along > self.contact + 0.05))
        setup_ok = (setup[:, 0] > self.low[0] - 0.005) & (setup[:, 0] < self.high[0] + 0.005) & \
                   (setup[:, 1] > self.low[1] - 0.005) & (setup[:, 1] < self.high[1] + 0.005)
        # If the ideal set-up point is outside the box, shorten the run-up.
        setup_clipped = np.clip(setup, self.low, self.high)
        through = future + d * c.follow
        strike = self._encode(through, d * 3.0, np.full(n, 0.04), np.ones(n))
        # Route around the puck when the paddle is on the wrong side.
        perp = np.column_stack((-d[:, 1], d[:, 0]))
        side = np.sign((rel * perp).sum(axis=1) + 1e-6)
        seg = setup_clipped - s.paddle
        seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)
        tpar = np.clip(((future - s.paddle) * seg).sum(axis=1) / seg_len**2, 0, 1)
        closest = s.paddle + seg * tpar[:, None]
        collide = np.linalg.norm(closest - future, axis=1) < self.contact + 0.025
        waypoint = future - d * 0.02 + perp * side[:, None] * (self.contact + 0.06)
        go = np.where(collide[:, None], waypoint, setup_clipped)
        move = self._encode(go, np.zeros((n, 2)), np.full(n, 0.04), np.full(n, 0.9))
        slow_act = np.where(behind[:, None], strike, move)
        # --- moving puck: one-timer at the depth with the best timing margin
        depths = np.asarray(c.one_timer_depths)
        state = np.column_stack((puck, vel))
        cross = incoming_crossings(state, depths, max_seconds=1.0)
        x_k, t_k, vx_k, vy_k = cross[..., 0], cross[..., 1], cross[..., 2], cross[..., 3]
        valid = np.isfinite(t_k) & (puck[:, 1:2] > depths[None, :]) & (t_k > 0.05)
        k = len(depths)
        flat_p = np.stack((x_k, np.broadcast_to(depths, x_k.shape)), -1).reshape(-1, 2)
        flat_v = np.stack((vx_k, vy_k), -1).reshape(-1, 2)
        flat_p = np.where(np.isfinite(flat_p), flat_p, 0.5)
        flat_v = np.where(np.isfinite(flat_v), flat_v, 0.0)
        rep = _State(puck=flat_p, vel=flat_v, paddle=np.repeat(s.paddle, k, 0),
                     paddle_vel=np.repeat(s.paddle_vel, k, 0), rival=np.repeat(s.rival, k, 0),
                     load=np.repeat(s.load, k, 0), request=np.repeat(s.request, k, 0))
        dk = self._shot_direction(flat_p, rep)
        want = dk * c.strike_speed - flat_v
        normal = want / np.maximum(np.linalg.norm(want, axis=1, keepdims=True), 1e-9)
        vpn = (flat_v * normal).sum(axis=1) + np.linalg.norm(want, axis=1) / (1 + c.paddle_restitution)
        pad = flat_p - normal * self.contact
        pad_v = normal * vpn[:, None]
        ok = (pad[:, 0] >= self.low[0]) & (pad[:, 0] <= self.high[0]) & \
             (pad[:, 1] >= self.low[1]) & (pad[:, 1] <= self.high[1]) & (normal[:, 1] > 0.25)
        reach = self._reach_time(np.repeat(s.paddle, k, 0), pad)
        slack = (t_k.reshape(-1) - reach)
        slack = np.where(valid.reshape(-1) & ok, slack, -np.inf).reshape(n, k)
        best = np.argmax(slack, axis=1)
        rows = np.arange(n)
        idx = rows * k + best
        one_ok = np.isfinite(slack[rows, best]) & (slack[rows, best] > -0.02) & (speed < c.max_one_timer_speed)
        one = self._encode(pad[idx], np.clip(pad_v[idx], -3, 3), np.maximum(t_k[rows, best] - c.latency_s, 0.04), np.ones(n))
        attack = reachable & ((slow) | one_ok)
        act = np.where((slow | ~one_ok)[:, None], slow_act, one)
        self._debug = dict(behind=behind, one_ok=one_ok, setup_ok=setup_ok)
        return attack, act

    # ------------------------------------------------------------- guard
    def _guard(self, s: _State, puck, vel):
        cfg, c = self.table, self.config
        n = len(puck)
        hot = np.clip((s.load.max(axis=1) - c.guard_hot_start) / (c.guard_hot_full - c.guard_hot_start), 0, 1)
        depth = c.guard_depth + hot * (c.guard_hot_depth - c.guard_depth)
        # On the puck-to-goal-centre line at this depth, pulled toward the
        # centre (guard_line_gain < 1) so a bank off either rail stays covered.
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
