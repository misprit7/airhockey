"""Vectorized air hockey environment for batch stepping.

Wraps BatchPhysicsEngine to provide the same obs/action interface as
AirHockeyEnv, but processes N environments in a single step() call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.physics import TableConfig


class BatchAirHockeyEnv:
    """Batch air hockey environment — N envs stepped simultaneously.

    Observation per env (14 dims):
        puck_x, puck_y, puck_vx, puck_vy,
        paddle_x, paddle_y, paddle_vx, paddle_vy,
        opp_x, opp_y, opp_vx, opp_vy,
        score_diff, time_remaining

    Action per env: (2,) — normalized [-1, 1] target position.

    Does NOT subclass gym.Env since vectorized envs have a different
    calling convention (no Gymnasium wrappers, returns batched arrays).
    """

    OBS_DIM = 14

    def __init__(
        self,
        n_envs: int,
        table_config: TableConfig | None = None,
        agent_dynamics: str = "ideal",  # "ideal" or "delayed"
        opponent_dynamics: str = "delayed",
        physics_dt: float = 1 / 240,
        action_dt: float = 1 / 60,
        max_episode_time: float = 60.0,
        max_episode_steps: int | None = None,
        max_score: int = 7,
        opponent_policy: str = "idle",
        camera_delay: float | tuple[float, float] = 0.0,
        domain_randomize: bool = False,
        frame_stack: int = 1,  # kept for API compat, must be 1
        score_handicap: bool = False,
        # DelayedDynamics parameters
        dynamics_max_speed: float = 4.0,
        dynamics_max_accel: float = 40.0,
        dynamics_time_constant: float = 0.02,
    ):
        self.n_envs = n_envs
        self.table_config = table_config or TableConfig()
        self.physics_dt = physics_dt
        self.action_dt = action_dt
        self.max_episode_time = max_episode_time
        self.max_episode_steps = max_episode_steps  # None = use time-based truncation
        self.max_score = max_score
        self.opponent_policy = opponent_policy
        self.score_handicap = score_handicap
        self.agent_dynamics_type = agent_dynamics
        self.opponent_dynamics_type = opponent_dynamics
        self.domain_randomize = domain_randomize
        self.frame_stack = 1  # always 1; velocities replace stacking

        self.engine = BatchPhysicsEngine(n_envs, self.table_config,
                                         domain_randomize=domain_randomize)

        # Per-env step counter for step-based truncation
        self._step_count = np.zeros(n_envs, dtype=np.int32)

        cfg = self.table_config
        self.n_substeps = max(1, int(action_dt / physics_dt))
        self.sub_dt = action_dt / self.n_substeps

        # Action rescaling bounds
        self._action_low = np.array([cfg.paddle_radius, cfg.paddle_radius])
        self._action_high = np.array(
            [cfg.width - cfg.paddle_radius, cfg.height / 2 - cfg.paddle_radius]
        )

        # Observation bounds
        vel_max = 10.0
        self.obs_high = np.array([
            cfg.width, cfg.height, vel_max, vel_max,      # puck
            cfg.width, cfg.height, vel_max, vel_max,      # paddle
            cfg.width, cfg.height, vel_max, vel_max,      # opponent
            1.0, 1.0,                                     # context
        ], dtype=np.float32)

        # Camera delay ring buffer.
        # camera_delay: float (uniform) or (min, max) tuple (per-env randomized).
        self._obs_dim = self.OBS_DIM
        if isinstance(camera_delay, tuple):
            self._delay_range = (
                max(0, int(camera_delay[0] / action_dt)),
                max(0, int(camera_delay[1] / action_dt)),
            )
        else:
            d = max(0, int(camera_delay / action_dt))
            self._delay_range = (d, d)
        self._max_delay = self._delay_range[1]
        # Per-env delay in steps [N]
        self._delay_steps = np.full(n_envs, self._max_delay, dtype=np.int32)
        if self._max_delay > 0:
            self._ring_size = self._max_delay + 1
            self._obs_ring = np.zeros(
                (self._ring_size, n_envs, self._obs_dim), dtype=np.float32,
            )
            self._ring_write = 0
            self._env_idx = np.arange(n_envs)

        # Vectorized delayed dynamics state (for agent and opponent)
        self._agent_dyn = self._make_dynamics_state(agent_dynamics, n_envs,
                                                     dynamics_max_speed,
                                                     dynamics_max_accel,
                                                     dynamics_time_constant)
        self._opp_dyn = self._make_dynamics_state(opponent_dynamics, n_envs,
                                                   dynamics_max_speed,
                                                   dynamics_max_accel,
                                                   dynamics_time_constant)

        self._rng = np.random.default_rng()

        # External opponent targets (for "external" policy)
        self._ext_opp_target_x = np.full(n_envs, cfg.width / 2)
        self._ext_opp_target_y = np.full(n_envs, cfg.height * 0.85)

        # Previous paddle positions for velocity estimation
        self._prev_agent_x = np.zeros(n_envs)
        self._prev_agent_y = np.zeros(n_envs)
        self._prev_opp_x = np.zeros(n_envs)
        self._prev_opp_y = np.zeros(n_envs)

        # Puck-stuck detection: reset if speed < threshold for N consecutive steps
        self._puck_slow_count = np.zeros(n_envs, dtype=np.int32)

    @staticmethod
    def _make_dynamics_state(
        dyn_type: str, n: int, max_speed: float, max_accel: float, tc: float
    ) -> dict[str, Any]:
        """Create vectorized dynamics state arrays."""
        return {
            "type": dyn_type,
            "x": np.zeros(n),
            "y": np.zeros(n),
            "vx": np.zeros(n),
            "vy": np.zeros(n),
            "max_speed": np.full(n, max_speed),
            "max_accel": np.full(n, max_accel),
            "time_constant": np.full(n, tc),
        }

    def reset(
        self,
        seed: int | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset environments. Returns observations [N, 14].

        If mask is provided, only resets the specified environments.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.engine.reset(self._rng, mask=mask)

        # Reset per-env step counters and stuck detection
        if mask is None:
            self._step_count[:] = 0
            self._puck_slow_count[:] = 0
        else:
            self._step_count[mask] = 0
            self._puck_slow_count[mask] = 0

        # Apply score handicaps for self-play training
        if self.score_handicap:
            self._apply_score_handicap(mask)

        # Sync dynamics state with paddle positions
        if mask is None:
            idx = slice(None)
            n = self.n_envs
        else:
            idx = mask
            n = int(mask.sum())
            if n == 0:
                return self._make_obs_direct()

        # Domain randomization: per-env motor dynamics
        if self.domain_randomize:
            for dyn in (self._agent_dyn, self._opp_dyn):
                dyn["max_speed"][idx] = self._rng.uniform(2.0, 4.5, size=n)
                dyn["max_accel"][idx] = self._rng.uniform(20.0, 45.0, size=n)
                dyn["time_constant"][idx] = self._rng.uniform(0.01, 0.04, size=n)

        self._agent_dyn["x"][idx] = self.engine.paddle_agent_x[idx]
        self._agent_dyn["y"][idx] = self.engine.paddle_agent_y[idx]
        self._agent_dyn["vx"][idx] = 0.0
        self._agent_dyn["vy"][idx] = 0.0

        # Override opponent position for stationary policies
        cfg = self.table_config
        if self.opponent_policy == "corner":
            corners = np.array([
                [cfg.paddle_radius, cfg.height - cfg.paddle_radius],
                [cfg.width - cfg.paddle_radius, cfg.height - cfg.paddle_radius],
            ])
            picks = self._rng.integers(0, len(corners), size=n)
            self.engine.paddle_opp_x[idx] = corners[picks, 0]
            self.engine.paddle_opp_y[idx] = corners[picks, 1]
        elif self.opponent_policy == "goalie":
            self.engine.paddle_opp_x[idx] = cfg.width / 2
            self.engine.paddle_opp_y[idx] = cfg.height - cfg.paddle_radius

        self._opp_dyn["x"][idx] = self.engine.paddle_opp_x[idx]
        self._opp_dyn["y"][idx] = self.engine.paddle_opp_y[idx]
        self._opp_dyn["vx"][idx] = 0.0
        self._opp_dyn["vy"][idx] = 0.0

        # Init previous positions (zero velocity at start)
        self._prev_agent_x[idx] = self.engine.paddle_agent_x[idx]
        self._prev_agent_y[idx] = self.engine.paddle_agent_y[idx]
        self._prev_opp_x[idx] = self.engine.paddle_opp_x[idx]
        self._prev_opp_y[idx] = self.engine.paddle_opp_y[idx]

        # Pre-fill camera delay buffer for reset envs
        if self._max_delay > 0:
            obs_now = self._make_obs_direct()
            lo, hi = self._delay_range
            if mask is None:
                # Full reset: randomize per-env delays, fill entire ring buffer
                if lo == hi:
                    self._delay_steps[:] = lo
                else:
                    self._delay_steps[:] = self._rng.integers(lo, hi + 1, size=self.n_envs)
                for t in range(self._ring_size):
                    self._obs_ring[t] = obs_now
                self._ring_write = 0
                return obs_now
            else:
                # Partial reset: re-randomize delays for reset envs, fill their slots.
                n_reset = int(mask.sum())
                if lo == hi:
                    self._delay_steps[mask] = lo
                else:
                    self._delay_steps[mask] = self._rng.integers(lo, hi + 1, size=n_reset)
                for t in range(self._ring_size):
                    self._obs_ring[t, mask] = obs_now[mask]
                return obs_now

        return self._make_obs_direct()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all N environments.

        Args:
            actions: [N, 2] normalized actions in [-1, 1].

        Returns:
            obs: [N, 14]
            rewards: [N]
            terminated: [N] bool
            truncated: [N] bool
            info: dict with batched arrays
        """
        # Clip and rescale actions to real positions
        actions = np.clip(actions, -1.0, 1.0)
        target_x = self._action_low[0] + (actions[:, 0] + 1.0) * 0.5 * (self._action_high[0] - self._action_low[0])
        target_y = self._action_low[1] + (actions[:, 1] + 1.0) * 0.5 * (self._action_high[1] - self._action_low[1])

        cfg = self.table_config
        rewards = np.zeros(self.n_envs)

        for _ in range(self.n_substeps):
            dt = self.sub_dt

            # Update agent paddle through dynamics
            ax, ay = self._update_dynamics(self._agent_dyn, target_x, target_y, dt)
            ax, ay = self._clamp_to_half(ax, ay, agent=True)
            self.engine.update_paddle_agent(ax, ay, dt)

            # Update opponent
            ox, oy = self._opponent_action(dt)
            ox, oy = self._clamp_to_half(ox, oy, agent=False)
            self.engine.update_paddle_opponent(ox, oy, dt)

            self.engine.step(dt)

            # Accumulate goal rewards
            rewards += np.where(self.engine.goal_scored == 1, 1.0, 0.0)
            rewards += np.where(self.engine.goal_scored == -1, -1.0, 0.0)

        self._step_count += 1

        # Universal puck-stuck reset: if puck speed < 0.05 for 120 steps (~2s),
        # reset to center heading toward a random side.
        puck_speed = np.hypot(self.engine.puck_vx, self.engine.puck_vy)
        slow = puck_speed < 0.05
        self._puck_slow_count = np.where(slow, self._puck_slow_count + 1, 0)
        stuck = self._puck_slow_count >= 120
        if np.any(stuck):
            n_stuck = int(stuck.sum())
            rng = self._rng
            # Random direction: 50% toward agent, 50% toward opponent
            toward = rng.random(n_stuck) < 0.5
            angle = np.where(
                toward,
                rng.uniform(-np.pi * 0.8, -np.pi * 0.2, size=n_stuck),
                rng.uniform(np.pi * 0.2, np.pi * 0.8, size=n_stuck),
            )
            speed = rng.uniform(0.3, 1.5, size=n_stuck)
            cfg = self.table_config
            self.engine.puck_x[stuck] = cfg.width / 2 + rng.uniform(-0.15, 0.15, size=n_stuck)
            self.engine.puck_y[stuck] = cfg.height / 2
            self.engine.puck_vx[stuck] = speed * np.cos(angle)
            self.engine.puck_vy[stuck] = speed * np.sin(angle)
            self._puck_slow_count[stuck] = 0

        obs = self._make_obs()  # applies camera delay if configured

        # Termination / truncation
        terminated = (
            (self.engine.score_agent >= self.max_score)
            | (self.engine.score_opponent >= self.max_score)
        )
        if self.max_episode_steps is not None:
            truncated = self._step_count >= self.max_episode_steps
        else:
            truncated = self.engine.time >= self.max_episode_time

        info = {
            "score_agent": self.engine.score_agent.copy(),
            "score_opponent": self.engine.score_opponent.copy(),
            "time": self.engine.time.copy(),
            "puck_vx": self.engine.puck_vx.copy(),
            "puck_vy": self.engine.puck_vy.copy(),
        }

        return obs, rewards, terminated, truncated, info

    def auto_reset(
        self, terminated: np.ndarray, truncated: np.ndarray
    ) -> np.ndarray | None:
        """Reset any done environments and return new observations for them.

        Returns None if no envs need resetting.
        """
        done = terminated | truncated
        if not np.any(done):
            return None
        return self.reset(mask=done)

    def _apply_score_handicap(self, mask: np.ndarray | None) -> None:
        """Set initial scores for handicap training.

        70% normal (0-0), 10% agent down (0-3), 10% agent up (3-0), 10% tied (3-3).
        """
        if mask is None:
            n = self.n_envs
            idx = slice(None)
        else:
            n = int(mask.sum())
            if n == 0:
                return
            idx = mask

        rolls = self._rng.random(n)
        # 0.0-0.7: normal (already 0-0 from engine.reset)
        # 0.7-0.8: agent down 0-3
        down = rolls >= 0.7
        down &= rolls < 0.8
        # 0.8-0.9: agent up 3-0
        up = rolls >= 0.8
        up &= rolls < 0.9
        # 0.9-1.0: tied 3-3
        tied = rolls >= 0.9

        if mask is None:
            self.engine.score_agent[down] = 0
            self.engine.score_opponent[down] = 3
            self.engine.score_agent[up] = 3
            self.engine.score_opponent[up] = 0
            self.engine.score_agent[tied] = 3
            self.engine.score_opponent[tied] = 3
        else:
            # Build index arrays for masked envs
            env_indices = np.where(mask)[0]
            self.engine.score_agent[env_indices[down]] = 0
            self.engine.score_opponent[env_indices[down]] = 3
            self.engine.score_agent[env_indices[up]] = 3
            self.engine.score_opponent[env_indices[up]] = 0
            self.engine.score_agent[env_indices[tied]] = 3
            self.engine.score_opponent[env_indices[tied]] = 3

    def set_opponent_actions(
        self, target_x: np.ndarray, target_y: np.ndarray
    ) -> None:
        """Set external opponent targets for all envs."""
        self._ext_opp_target_x = target_x.copy()
        self._ext_opp_target_y = target_y.copy()

    def mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """Mirror observations [N, 14] for opponent perspective.

        Flip y positions, negate y velocities, swap agent/opponent.
        Negate score_diff, time_remaining unchanged.
        """
        cfg = self.table_config
        m = obs.copy()
        # Obs: [puck_x, puck_y, puck_vx, puck_vy,
        #        pad_x, pad_y, pad_vx, pad_vy,
        #        opp_x, opp_y, opp_vx, opp_vy,
        #        score_diff, time_remaining]

        # Puck: flip y, negate vy
        m[:, 1] = cfg.height - obs[:, 1]    # puck_y
        m[:, 3] = -obs[:, 3]                # puck_vy

        # Swap agent/opponent and flip y, negate vy
        m[:, 4] = obs[:, 8]                 # opp_x → pad_x
        m[:, 5] = cfg.height - obs[:, 9]    # opp_y → pad_y (flipped)
        m[:, 6] = obs[:, 10]                # opp_vx → pad_vx
        m[:, 7] = -obs[:, 11]               # opp_vy → pad_vy (negated)
        m[:, 8] = obs[:, 4]                 # pad_x → opp_x
        m[:, 9] = cfg.height - obs[:, 5]    # pad_y → opp_y (flipped)
        m[:, 10] = obs[:, 6]                # pad_vx → opp_vx
        m[:, 11] = -obs[:, 7]               # pad_vy → opp_vy (negated)

        # Context
        m[:, 12] = -obs[:, 12]  # negate score_diff
        # time_remaining unchanged
        return m

    def mirror_action_to_opponent(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert [N, 2] normalized actions from opponent's mirrored perspective
        to real table coordinates in opponent's half."""
        actions = np.clip(actions, -1.0, 1.0)
        cfg = self.table_config
        r = cfg.paddle_radius
        x = r + (actions[:, 0] + 1.0) * 0.5 * (cfg.width - 2 * r)
        # y: mirrored — y=-1 means "near own goal" = back wall (height - r),
        # y=+1 means "opponent's side" = midfield (height/2 + r)
        y = (cfg.height - r) - (actions[:, 1] + 1.0) * 0.5 * (cfg.height / 2 - 2 * r)
        return x, y

    # --- Internal helpers ---

    def _update_dynamics(
        self,
        dyn: dict[str, Any],
        target_x: np.ndarray,
        target_y: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized dynamics update. Returns new (x, y) arrays."""
        if dyn["type"] == "ideal":
            dyn["x"] = target_x.copy()
            dyn["y"] = target_y.copy()
            return dyn["x"].copy(), dyn["y"].copy()

        # Delayed dynamics: P-controller with velocity/acceleration limits
        dx = target_x - dyn["x"]
        dy = target_y - dyn["y"]
        tc = np.maximum(dyn["time_constant"], dt)

        desired_vx = dx / tc
        desired_vy = dy / tc

        # Clamp desired velocity
        desired_speed = np.hypot(desired_vx, desired_vy)
        too_fast = desired_speed > dyn["max_speed"]
        factor = np.where(
            too_fast,
            dyn["max_speed"] / np.maximum(desired_speed, 1e-8),
            1.0,
        )
        desired_vx *= factor
        desired_vy *= factor

        # Acceleration limits
        if dt > 0:
            ax = (desired_vx - dyn["vx"]) / dt
            ay = (desired_vy - dyn["vy"]) / dt
            accel = np.hypot(ax, ay)
            too_much = accel > dyn["max_accel"]
            afactor = np.where(
                too_much,
                dyn["max_accel"] / np.maximum(accel, 1e-8),
                1.0,
            )
            ax *= afactor
            ay *= afactor
            dyn["vx"] += ax * dt
            dyn["vy"] += ay * dt

        # Integrate
        dyn["x"] += dyn["vx"] * dt
        dyn["y"] += dyn["vy"] * dt

        return dyn["x"].copy(), dyn["y"].copy()

    def _opponent_action(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized opponent policies. Returns target (x, y) arrays."""
        cfg = self.table_config

        if self.opponent_policy == "idle":
            return self._update_dynamics(
                self._opp_dyn,
                self.engine.paddle_opp_x.copy(),
                self.engine.paddle_opp_y.copy(),
                dt,
            )

        elif self.opponent_policy == "follow":
            target_x = self.engine.puck_x.copy()
            target_y = np.full(self.n_envs, cfg.height * 0.85)
            return self._update_dynamics(self._opp_dyn, target_x, target_y, dt)

        elif self.opponent_policy == "random":
            target_x = self._rng.uniform(
                cfg.paddle_radius, cfg.width - cfg.paddle_radius, size=self.n_envs
            )
            target_y = self._rng.uniform(
                cfg.height / 2 + cfg.paddle_radius,
                cfg.height - cfg.paddle_radius,
                size=self.n_envs,
            )
            return self._update_dynamics(self._opp_dyn, target_x, target_y, dt)

        elif self.opponent_policy in ("corner", "goalie"):
            # Stationary — position set at reset, don't move
            return self.engine.paddle_opp_x.copy(), self.engine.paddle_opp_y.copy()

        elif self.opponent_policy == "external":
            return self._update_dynamics(
                self._opp_dyn,
                self._ext_opp_target_x,
                self._ext_opp_target_y,
                dt,
            )

        # Fallback: don't move
        return self.engine.paddle_opp_x.copy(), self.engine.paddle_opp_y.copy()

    def _clamp_to_half(
        self, x: np.ndarray, y: np.ndarray, agent: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.table_config
        r = cfg.paddle_radius
        x = np.clip(x, r, cfg.width - r)
        if agent:
            y = np.clip(y, r, cfg.height / 2 - r)
        else:
            y = np.clip(y, cfg.height / 2 + r, cfg.height - r)
        return x, y

    def _make_obs_direct(self) -> np.ndarray:
        """Build [N, 14] observation with positions + velocities + context."""
        e = self.engine
        dt = self.action_dt

        # Paddle velocities from finite differences
        agent_vx = (e.paddle_agent_x - self._prev_agent_x) / dt
        agent_vy = (e.paddle_agent_y - self._prev_agent_y) / dt
        opp_vx = (e.paddle_opp_x - self._prev_opp_x) / dt
        opp_vy = (e.paddle_opp_y - self._prev_opp_y) / dt

        # Update previous positions
        self._prev_agent_x[:] = e.paddle_agent_x
        self._prev_agent_y[:] = e.paddle_agent_y
        self._prev_opp_x[:] = e.paddle_opp_x
        self._prev_opp_y[:] = e.paddle_opp_y

        # Context
        score_diff = (e.score_agent - e.score_opponent) / max(self.max_score, 1)
        if self.max_episode_steps is not None:
            time_remaining = np.clip(self.max_episode_steps - self._step_count, 0, None) / self.max_episode_steps
        else:
            time_remaining = np.clip(self.max_episode_time - e.time, 0.0, None) / self.max_episode_time

        return np.column_stack([
            e.puck_x, e.puck_y, e.puck_vx, e.puck_vy,
            e.paddle_agent_x, e.paddle_agent_y, agent_vx, agent_vy,
            e.paddle_opp_x, e.paddle_opp_y, opp_vx, opp_vy,
            score_diff, time_remaining,
        ]).astype(np.float32)

    def _get_delayed_obs(self, current_obs: np.ndarray) -> np.ndarray:
        """Push current obs into ring buffer, return per-env delayed obs."""
        self._obs_ring[self._ring_write] = current_obs
        # Each env reads from its own delay offset
        read_idx = (self._ring_write - self._delay_steps) % self._ring_size
        delayed = self._obs_ring[read_idx, self._env_idx]
        self._ring_write = (self._ring_write + 1) % self._ring_size
        return delayed

    def _make_obs(self) -> np.ndarray:
        """Build observation array, applying camera delay if configured."""
        current = self._make_obs_direct()
        if self._max_delay > 0:
            return self._get_delayed_obs(current)
        return current
