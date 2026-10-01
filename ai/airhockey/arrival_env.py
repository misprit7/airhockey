"""Arrival-action learning environment with measured-load proxy and skill tasks.

The wrapper translates every 50 Hz action through ArrivalDecoder and executes
ordinary commands in BatchAirHockeyEnv. It never assigns state during a step.
Only episode initialization places fixtures. No hardware I/O.
"""

from __future__ import annotations
import numpy as np
from airhockey.arrival import ArrivalDecoder, ArrivalConfig
from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs, _OPP_POLICY_MAP
from airhockey.motion import CartState
from airhockey.skill_benchmark import make_fixtures
from airhockey.thermal import MotorThermal, DEFAULT_MODEL


class ArrivalEnv:
    action_dim = 6
    obs_dim = 45
    PREV_ACTION_IDX = 15

    def __init__(
        self,
        n_envs=32,
        *,
        seed=0,
        accel=60.0,
        workspace_bounds_mm=None,
        realistic=True,
        randomize=True,
        game_fraction=0.0,
        selfplay_fraction=0.0,
        thermal_path=DEFAULT_MODEL,
        load_weight=1.0,
        load_soft_start=0.65,
    ):
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)
        self.game_fraction = game_fraction
        self.selfplay_fraction = selfplay_fraction
        self.load_weight = load_weight
        self.base = BatchAirHockeyEnv(
            n_envs,
            action_mode="profile_a",
            agent_dynamics="profile",
            opponent_dynamics="profile",
            opponent_body="robot",
            opponent_policy="idle",
            max_episode_time=30.0,
            max_score=7,
            domain_randomize=randomize,
            dynamics_max_accel=accel,
            agent_accel_range=(accel, accel),
            workspace_bounds_mm=workspace_bounds_mm,
            **sensing_kwargs(realistic),
        )
        self.base.command_delay_s = (
            0.015  # measured 12 ms host + decoder, rounded at 400 Hz
        )
        self.engine = self.base.engine
        self.cfg = self.base.table_config
        ws = self.base._ws
        self.decoder = ArrivalDecoder(
            [ws[k] for k in ("min_x", "max_x", "min_y", "max_y")],
            ArrivalConfig(action_dt=self.base.action_dt),
        )
        self.loads = [
            MotorThermal(n_envs, thermal_path, seed + 7, randomize, load_soft_start),
            MotorThermal(n_envs, thermal_path, seed + 8, randomize, load_soft_start),
        ]
        self.task = np.zeros(n_envs, dtype=int)
        self.aim = np.full(n_envs, 0.5)
        self.desired_speed = np.full(n_envs, 3.0)
        self.last_action = np.zeros((n_envs, 6), np.float32)
        self.last_opp_action = np.zeros_like(self.last_action)
        self.contacts = np.zeros(n_envs, int)
        self.contact_step = np.zeros(n_envs, bool)
        self.contact_reward = np.zeros(n_envs)
        self.attempt_contacts = np.zeros(n_envs, int)
        self.hold_time = np.zeros(n_envs)
        self.control_best = np.zeros(n_envs)
        self.load_cost = np.zeros(n_envs)
        self.effort = np.zeros(n_envs)
        self.peak_accel = np.zeros(n_envs)
        self.shot_armed = np.ones(n_envs, bool)
        self.shots = np.zeros(n_envs, int)
        self.on_target = np.zeros(n_envs, int)
        self.goal_error = np.full(n_envs, np.nan)
        self.crossing_error = np.full(n_envs, np.nan)
        self.elapsed = np.zeros(n_envs)
        self.base.motion_callback = self._motion
        self.engine.contact_callback = self._contact
        self.engine.goal_callback = self._goal
        self._obs = None

    def _cart(self, dyn, opponent=False):
        c = CartState(self.n_envs)
        for k in ("x", "y", "vx", "vy"):
            getattr(c, k)[:] = dyn[k] * 1000
        c.ax[:] = dyn["cart"].ax
        c.ay[:] = dyn["cart"].ay
        if opponent:
            c.y[:] = self.cfg.height * 1000 - c.y
            c.vy *= -1
            c.ay *= -1
        return c

    def _decode(self, action, opponent=False):
        b = self.base
        dyn = b._opp_dyn if opponent else b._agent_dyn
        previous = np.column_stack((dyn["command_x"], dyn["command_y"]))
        if opponent:
            previous[:, 1] = self.cfg.height - previous[:, 1]
        return self.decoder.decode(
            self._cart(dyn, opponent),
            action,
            0.0,
            previous,
            dyn["command_accel"],
            max_speed=dyn["max_speed"],
            max_accel=dyn["max_accel"],
            delay_s=b.command_delay_s,
        )

    def _features(self, base_obs, opponent=False):
        dyn = self.base._opp_dyn if opponent else self.base._agent_dyn
        c = self._cart(dyn, opponent)
        queued = np.column_stack((dyn["command_x"], dyn["command_y"]))
        if opponent:
            queued[:, 1] = self.cfg.height - queued[:, 1]
        q = 2 * (queued - self.decoder.low) / (self.decoder.high - self.decoder.low) - 1
        task = np.full(self.n_envs, 3) if opponent else self.task
        aim = np.full(self.n_envs, 0.5) if opponent else self.aim
        extras = np.column_stack(
            (
                np.eye(4)[task],
                aim,
                self.desired_speed / 3,
                np.column_stack((c.ax, c.ay)) / 60000,
                q,
                dyn["command_accel"] / dyn["max_accel"],
                np.minimum(self.elapsed / 30, 1),
            )
        )
        return np.column_stack(
            (
                base_obs[:, :15],
                self.last_opp_action if opponent else self.last_action,
                base_obs[:, 18:22],
                self.loads[int(opponent)].features(),
                extras,
            )
        ).astype(np.float32)

    def opponent_obs(self):
        return self._features(self.base.opponent_obs(), True)

    def set_opponent_action(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        if action.shape != (self.n_envs, 6):
            raise ValueError("opponent action must be [N,6]")
        target, cap = self._decode(action, True)
        b = self.base
        b._ext_opp_target_x[:] = target[:, 0]
        b._ext_opp_target_y[:] = self.cfg.height - target[:, 1]
        b._ext_opp_accel_frac[:] = cap / b._opp_dyn["max_accel"]
        self.last_opp_action[:] = action

    def _motion(self, dt, old_agent_v, old_opp_v):
        for side, (dyn, old) in enumerate(
            ((self.base._agent_dyn, old_agent_v), (self.base._opp_dyn, old_opp_v))
        ):
            p = np.column_stack((dyn["x"], dyn["y"]))
            v = np.column_stack((dyn["vx"], dyn["vy"]))
            a = (v - old) / dt
            if side:
                p[:, 1] = self.cfg.height - p[:, 1]
                v[:, 1] *= -1
                a[:, 1] *= -1
            self.loads[side].advance(p, v, a, dt)
            if side == 0:
                self.peak_accel = np.maximum(self.peak_accel, np.linalg.norm(a, axis=1))
                self.load_cost += self.loads[0].penalty(dt)
                self.effort += (a * a).sum(axis=1) * dt

    def _contact(self, event):
        idx = event["indices"]
        if event["body"] != "agent":
            self.shot_armed[idx] = True
            return
        self.contacts[idx] += 1
        self.contact_step[idx] = True
        within = self.elapsed[idx] < 0.40
        self.attempt_contacts[idx[within]] += 1
        out = event["outgoing_before_speed_cap"]
        vx, vy = out.T
        x, y = self.engine.puck_x[idx], self.engine.puck_y[idx]
        crossing = x + vx * (self.cfg.height - y) / np.maximum(vy, 1e-6)
        goal_half = self.cfg.goal_width / 2 - self.cfg.puck_radius
        aimed = (vy > 1.0) & (abs(crossing - self.cfg.width / 2) < goal_half)
        attempt = (vy > 1.0) & self.shot_armed[idx]
        ids = idx[attempt]
        self.shots[ids] += 1
        self.on_target[idx[attempt & aimed]] += 1
        self.shot_armed[ids] = False
        self.goal_error[ids] = abs(crossing[attempt] - self.aim[ids])
        precision = np.exp(-(((crossing - self.aim[idx]) / 0.07) ** 2))
        self.contact_reward[ids] += np.where(
            aimed[attempt], 3 + 7 * precision[attempt], -10
        )

    def _goal(self, event):
        idx = event["indices"][event["agent"]]
        self.crossing_error[idx] = abs(
            event["crossing_x"][event["agent"]] - self.aim[idx]
        )

    def reset(self, *, seed=None, mask=None, fixtures=None, opponent=None):
        if opponent is not None and opponent not in _OPP_POLICY_MAP:
            raise ValueError(f"unknown opponent: {opponent}")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mask = np.ones(self.n_envs, bool) if mask is None else np.asarray(mask, bool)
        ids = np.flatnonzero(mask)
        n = len(ids)
        # Heat is deliberately never cleared here (even when max_score ends a game).
        # Allocate ENVIRONMENT SLOTS, not episode draws. Otherwise 30-second
        # games dominate data despite an "80% skills" episode probability.
        self.task[ids] = ids % 3
        self.task[ids[ids < round(self.n_envs * self.game_fraction)]] = 3
        if fixtures is not None:
            if len(fixtures.task) != n:
                raise ValueError("one fixture per resetting env")
            self.task[ids] = fixtures.task
            f = fixtures
        else:
            bank = make_fixtures(int(self.rng.integers(2**31)), max(n, 1))
            take = self.task[ids].clip(0, 2) * max(n, 1) + np.arange(n)
            f = bank.take(take)
        self.aim[ids] = f.aim
        # A normal game gives both players the same neutral goal-center request.
        # Random precision targets belong only to isolated shooting drills.
        self.aim[ids[self.task[ids] == 3]] = self.cfg.width / 2
        self.desired_speed[ids] = np.where(self.task[ids] == 2, 0, 3)
        skills = self.task[ids] < 3
        si = ids[skills]
        kind = self.rng.choice(["sniper", "weak_goalie", "goalie"], n)
        kind[self.rng.random(n) < self.selfplay_fraction] = "external"
        if opponent is not None:
            kind[:] = opponent
        kind[skills] = "idle"
        self.base._opp_policy_id[ids] = [_OPP_POLICY_MAP[k] for k in kind]
        self.base._opp_free = np.isin(
            self.base._opp_policy_id,
            [_OPP_POLICY_MAP["sniper"], _OPP_POLICY_MAP["weak_goalie"]],
        )
        self.base.reset(seed=seed, mask=mask)
        for j, k in enumerate(("puck_x", "puck_y", "puck_vx", "puck_vy")):
            getattr(self.engine, k)[si] = f.puck[skills, j]
        self.engine.paddle_agent_x[si] = f.paddle[skills, 0]
        self.engine.paddle_agent_y[si] = f.paddle[skills, 1]
        # Idle far paddle sits away from the requested open-goal aim.
        self.engine.paddle_opp_x[si] = 0.08
        self.engine.paddle_opp_y[si] = 1.8
        # Synchronize all state and camera history at the fixture boundary.
        for dyn, prefix in (
            (self.base._agent_dyn, "paddle_agent"),
            (self.base._opp_dyn, "paddle_opp"),
            (self.base._opp_dyn_free, "paddle_opp"),
        ):
            for k in ("x", "y"):
                dyn[k][ids] = getattr(self.engine, prefix + "_" + k)[ids]
                dyn["command_" + k][ids] = dyn[k][ids]
                dyn["v" + k][ids] = 0
            dyn["command_speed"][ids] = dyn["max_speed"][ids]
            dyn["command_accel"][ids] = dyn["max_accel"][ids]
            self.base._clear_profile_accel(dyn, ids)
        b = self.base
        e = self.engine
        if b._perception is not None:
            b._perception.reset(e.puck_x, e.puck_y, ids)
        if b._cam_active:
            for col, name in enumerate(
                (
                    "puck_x",
                    "puck_y",
                    "puck_vx",
                    "puck_vy",
                    "paddle_opp_x",
                    "paddle_opp_y",
                    "paddle_agent_x",
                    "paddle_agent_y",
                )
            ):
                b._cam_ring[:, ids, col] = getattr(e, name)[ids]
            b._cam_ring[:, ids, 8] = 0
        for prefix, body in [
            ("agent", "agent"),
            ("opp", "opp"),
            ("own_opp", "opp"),
            ("rival", "agent"),
        ]:
            for axis in ("x", "y"):
                getattr(b, "_prev_" + prefix + "_" + axis)[ids] = getattr(
                    e, "paddle_" + body + "_" + axis
                )[ids]
        self.last_action[ids] = 0
        self.last_opp_action[ids] = 0
        for v in (
            self.contacts,
            self.attempt_contacts,
            self.hold_time,
            self.control_best,
            self.effort,
            self.peak_accel,
            self.shots,
            self.on_target,
            self.elapsed,
        ):
            v[ids] = 0
        self.goal_error[ids] = np.nan
        self.shot_armed[ids] = True
        self.crossing_error[ids] = np.nan
        self._obs = self._features(b._make_obs_direct())
        return self._obs.copy()

    def step(self, action):
        self.base.referee_active_mask = (self.task == 3) & ~getattr(
            self, "defense_trials", np.zeros(self.n_envs, bool)
        )
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_envs, 6) or not np.isfinite(action).all():
            raise ValueError("arrival action must be finite [N,6]")
        action = np.clip(action, -1, 1)
        target, cap = self._decode(action)
        b = self.base
        low = 2 * (target - b._action_low) / (b._action_high - b._action_low) - 1
        accel = (
            2 * np.sqrt(np.clip((cap / b._agent_dyn["max_accel"] - 0.05) / 0.95, 0, 1))
            - 1
        )
        legacy = np.column_stack((low, accel))
        self.last_action[:] = action
        self.contact_step[:] = False
        self.contact_reward[:] = 0
        self.load_cost[:] = 0
        scores = self.engine.score_agent.copy()
        conceded = self.engine.score_opponent.copy()
        old_gap = np.hypot(
            self.engine.puck_x - self.engine.paddle_agent_x,
            self.engine.puck_y - self.engine.paddle_agent_y,
        )
        obs, raw, term, trunc, info = b.step(legacy)
        self.elapsed += b.action_dt
        gf = self.engine.score_agent - scores
        ga = self.engine.score_opponent - conceded
        self.shot_armed[(gf + ga) > 0] = True
        gap = (
            np.hypot(
                self.engine.puck_x - self.engine.paddle_agent_x,
                self.engine.puck_y - self.engine.paddle_agent_y,
            )
            - self.cfg.puck_radius
            - self.cfg.paddle_radius
        )
        speed = np.hypot(self.engine.puck_vx, self.engine.puck_vy)
        controlled = (speed < 0.6) & (gap < 0.06) & (self.contacts > 0)
        self.hold_time = np.where(controlled, self.hold_time + b.action_dt, 0)
        self.control_best = np.maximum(self.control_best, self.hold_time)
        skills = self.task < 3
        cushion = self.task == 2
        shot = skills & ~cushion
        # Exact goals/contact outcomes; shaping cannot pay endlessly for repeating a hit.
        reward = (
            np.where(cushion, -25 * gf, 100 * gf) - 75 * ga + self.contact_reward - 0.01
        )
        reward += info["penalty"]  # shot-clock and stuck-puck turnovers still count
        potential = 8 * np.clip(
            old_gap - (gap + self.cfg.puck_radius + self.cfg.paddle_radius), -0.1, 0.1
        )
        reward += np.where(
            skills & (self.elapsed < 0.4) & (gf == 0) & (ga == 0), potential, 0
        )
        ending = shot & ((self.elapsed >= 2) | (gf > 0) | (ga > 0))
        ending |= cushion & (self.elapsed >= 0.6)
        reward[ending & shot & (gf == 0)] -= 25
        success = np.where(
            cushion,
            (self.attempt_contacts > 0) & (self.control_best >= 0.2),
            self.engine.score_agent > 0,
        )
        reward[ending & cushion] += np.where(success[ending & cushion], 70, -25)
        reward[ending & cushion] -= (
            10 * np.minimum(speed, 4) + 20 * np.clip(gap, 0, 1)
        )[ending & cushion]
        reward[ending & cushion] += (20 * np.minimum(self.control_best / 0.2, 1))[
            ending & cushion
        ]
        # Explicit failed-attempt feedback even when no collision detector ever fires.
        whiff = (
            skills
            & (self.elapsed >= 0.4)
            & (self.elapsed < 0.4 + b.action_dt)
            & (self.attempt_contacts == 0)
        )
        reward[whiff] -= 15
        reward -= self.load_weight * self.load_cost
        term = np.where(skills, ending, term)
        trunc = np.where(skills, False, trunc)
        info.update(
            task=self.task.copy(),
            success=success.copy(),
            contacts=self.contacts.copy(),
            attempt_contacts=self.attempt_contacts.copy(),
            shots=self.shots.copy(),
            on_target=self.on_target.copy(),
            load_peak=self.loads[0].levels.max(axis=(1, 2)),
            load_cost=self.load_cost.copy(),
            effort=self.effort.copy(),
            peak_accel=self.peak_accel.copy(),
            goal_error=self.crossing_error.copy(),
        )
        self._obs = self._features(obs)
        return self._obs.copy(), reward.astype(np.float32), term, trunc, info
