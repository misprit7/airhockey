"""Vectorized air hockey environment for batch stepping.

Wraps BatchPhysicsEngine to provide the same obs/action interface as
AirHockeyEnv, but processes N environments in a single step() call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.dynamics import (DR_ACCEL_RANGE, DR_SPEED_RANGE,
                                MAX_ACCEL_M_S2, MAX_SPEED_M_S,
                                OPPONENT_MAX_ACCEL_M_S2,
                                OPPONENT_MAX_SPEED_M_S,
                                workspace_in_sim)
from airhockey.motion import DEFAULT_SIM_DT, CartState, advance
from airhockey.perception import PuckPerception
from airhockey.physics import TableConfig

# Per-env opponent policy IDs
OPP_IDLE = 0
OPP_FOLLOW = 1
OPP_GOALIE = 2
OPP_EXTERNAL = 3
OPP_CORNER = 4
OPP_RANDOM = 5

_OPP_POLICY_MAP = {
    "idle": OPP_IDLE,
    "follow": OPP_FOLLOW,
    "goalie": OPP_GOALIE,
    "external": OPP_EXTERNAL,
    "corner": OPP_CORNER,
    "random": OPP_RANDOM,
}


class BatchAirHockeyEnv:
    """Batch air hockey environment — N envs stepped simultaneously.

    Observation per env (15 dims):
        puck_x, puck_y, puck_vx, puck_vy,
        paddle_x, paddle_y, paddle_vx, paddle_vy,
        opp_x, opp_y, opp_vx, opp_vy,
        side, max_speed, max_accel

    Action per env: (2,) — normalized [-1, 1] target position.

    Does NOT subclass gym.Env since vectorized envs have a different
    calling convention (no Gymnasium wrappers, returns batched arrays).
    """

    # 15, not 12. The last three features describe the BODY the observation is
    # driving, which the twelve state features do not determine.
    #
    #   [12] SIDE: 1.0 robot, 0.0 human. The two sides do not have the same
    #        capabilities, and only the robot is confined to the reachable
    #        workspace, so a policy that plays both in self-play cannot act
    #        correctly without knowing which it currently is. Always 1.0 in
    #        production.
    #   [13] MAX SPEED, [14] MAX ACCEL, as a ratio to the robot's nominal
    #        caps. Domain randomisation samples these per env, and they change
    #        the right play rather than just the execution: how early to commit
    #        to an intercept, whether a cross-table save is reachable at all,
    #        how much of a wind-up a shot can afford. A policy that cannot see
    #        them has to average over the range and will consistently
    #        over-commit on a slow draw and under-use a fast one.
    #
    # OWN caps only. The opponent's are deliberately NOT given: in production
    # the opponent is a human whose limits are unknown and unknowable, so a
    # feature for them would be informative in sim and a fixed lie on the
    # table. The policy has to read the opponent off their motion, as it will
    # have to for real.
    OBS_DIM = 15
    ROBOT_SIDE = 1.0
    HUMAN_SIDE = 0.0

    def __init__(
        self,
        n_envs: int,
        table_config: TableConfig | None = None,
        # "profile" is the REAL firmware control law (fw/include/
        # motion_profile.h, built as a host library and bound in motion.py):
        # one velocity profile along the direction of travel, jerk-limited,
        # with the same parking rule the Teensy uses. It is the default
        # because "ideal" teleports the paddle to its target -- a policy
        # trained against that learns to command positions no actuator can
        # reach, and finds out on the hardware.
        agent_dynamics: str = "profile",   # "ideal" | "delayed" | "profile"
        # The opponent is a HUMAN, and a hand is not a stepper under a
        # trapezoidal profile. "delayed" -- a first-order lag with caps -- is
        # the better model, and it is deliberately given the human limits.
        opponent_dynamics: str = "delayed",
        # 12 m/s over a 2 ms step is 24 mm, well inside the 80.7 mm at which
        # puck and paddle touch. At the old 1/240 it was 50 mm and a fast puck
        # could step straight through the paddle without ever registering a
        # collision.
        physics_dt: float = 1 / 500,
        # 100 Hz, not 60. Measured loop latency is 7.7 ms; at 60 Hz one action
        # step is 16.7 ms, so the delay is SHORTER than a step and cannot be
        # represented at all -- the sim would silently model a robot that sees
        # instantly. At 100 Hz it is almost exactly one step. It is also what
        # the real control loop can sustain, since the policy has to run
        # between camera frames.
        action_dt: float = 1 / 100,
        max_episode_time: float = 60.0,
        max_episode_steps: int | None = None,
        max_score: int = 7,
        opponent_policy: str = "idle",
        opponent_mix: dict[str, int] | None = None,
        camera_delay: float | tuple[float, float] = 0.0,
        domain_randomize: bool = False,
        frame_stack: int = 1,  # kept for API compat, must be 1
        score_handicap: bool = False,
        # DelayedDynamics parameters
        dynamics_max_speed: float = MAX_SPEED_M_S,
        dynamics_max_accel: float = MAX_ACCEL_M_S2,
        dynamics_time_constant: float = 0.02,
        # Observe the puck through a model of the real tracker rather
        # than reading the engine: finite-difference velocity over noisy
        # positions, plus the IR ring's blind spot at table centre.
        realistic_perception: bool = False,
        # Bound the AGENT to the box the machine can actually reach, rather
        # than to its half of the table. Off only for ablations; on it, the
        # policy would learn to use 65% of the half that does not exist.
        # The OPPONENT is deliberately left with the full half: it stands in
        # for a human, who can reach anywhere on their side.
        constrain_to_workspace: bool = True,
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

        # Per-env opponent policy
        if opponent_mix is not None:
            total = sum(opponent_mix.values())
            assert total == n_envs, (
                f"opponent_mix sums to {total}, expected n_envs={n_envs}"
            )
            ids = []
            for policy_name, count in opponent_mix.items():
                ids.extend([_OPP_POLICY_MAP[policy_name]] * count)
            self._opp_policy_id = np.array(ids, dtype=np.int8)
        else:
            self._opp_policy_id = np.full(
                n_envs, _OPP_POLICY_MAP[opponent_policy], dtype=np.int8
            )

        self.engine = BatchPhysicsEngine(n_envs, self.table_config,
                                         domain_randomize=domain_randomize)

        # Per-env step counter for step-based truncation
        self._step_count = np.zeros(n_envs, dtype=np.int32)

        cfg = self.table_config
        self.n_substeps = max(1, int(action_dt / physics_dt))
        self.sub_dt = action_dt / self.n_substeps

        # Action rescaling bounds
        self.constrain_to_workspace = constrain_to_workspace
        self._ws = (workspace_in_sim(cfg.width, cfg.height / 2)
                    if constrain_to_workspace else None)
        if self._ws is not None:
            self._action_low = np.array([self._ws["min_x"], self._ws["min_y"]])
            self._action_high = np.array([self._ws["max_x"], self._ws["max_y"]])
        else:
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
            1.0,                                          # side flag
            # Caps, as a ratio to the robot's nominal. The bound is the human
            # side's, since that is the largest either feature ever takes.
            OPPONENT_MAX_SPEED_M_S / MAX_SPEED_M_S,
            OPPONENT_MAX_ACCEL_M_S2 / MAX_ACCEL_M_S2,
        ], dtype=np.float32)

        # Camera delay ring buffer.
        # camera_delay: float (uniform) or (min, max) tuple (per-env randomized).
        self._obs_dim = self.OBS_DIM
        if isinstance(camera_delay, tuple):
            # ROUND, not truncate. The measured 9.9 ms loop against a
            # 10 ms step is 0.99 steps, and int() would call that zero --
            # discarding 99% of a delay that was measured precisely so it
            # would not have to be guessed.
            self._delay_range = (
                max(0, int(round(camera_delay[0] / action_dt))),
                max(0, int(round(camera_delay[1] / action_dt))),
            )
        else:
            d = max(0, int(round(camera_delay / action_dt)))
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
                                                   OPPONENT_MAX_SPEED_M_S,
                                                   OPPONENT_MAX_ACCEL_M_S2,
                                                   dynamics_time_constant)

        self._rng = np.random.default_rng()
        self._perception = (
            PuckPerception(n_envs, cfg.width, cfg.height, action_dt,
                           self._rng)
            if realistic_perception else None)

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
    def _clear_profile_accel(dyn: dict[str, Any], idx) -> None:
        """Drop the jerk slew state on reset.

        Position and velocity are already zeroed above, but the firmware law
        carries ACCELERATION too, and it is not derived from them -- an
        episode starting with a stale acceleration would spend its first
        milliseconds unwinding a manoeuvre from the previous one. Same reason
        CDPR::tick zeroes accX_/accY_ when it parks.
        """
        cart = dyn.get("cart")
        if cart is not None:
            cart.ax[idx] = 0.0
            cart.ay[idx] = 0.0

    @staticmethod
    def _make_dynamics_state(
        dyn_type: str, n: int, max_speed: float, max_accel: float, tc: float
    ) -> dict[str, Any]:
        """Create vectorized dynamics state arrays."""
        # "profile" carries a CartState as well, because the firmware law has
        # ACCELERATION as state -- it slews it to bound jerk, so the same
        # command produces different motion depending on what the
        # acceleration was. The other two types are memoryless in that
        # respect and do not need it.
        extra = {"cart": CartState(n)} if dyn_type == "profile" else {}
        return {
            **extra,
            "type": dyn_type,
            "x": np.zeros(n),
            "y": np.zeros(n),
            "vx": np.zeros(n),
            "vy": np.zeros(n),
            "max_speed": np.full(n, max_speed),
            "max_accel": np.full(n, max_accel),
            "time_constant": np.full(n, tc),
            # The caps this side is NOMINALLY built with, kept as scalars so
            # domain randomisation can scale each side by its own limits.
            # The arrays above get overwritten on every randomised reset.
            "nominal_speed": max_speed,
            "nominal_accel": max_accel,
            # Jerk ramp, seconds. Matches MOTION_ACCEL_RAMP_S in the firmware.
            "ramp_s": 0.003,
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

        # Domain randomization: per-env motor dynamics.
        #
        # Scaled by EACH SIDE'S OWN nominal caps, not by the robot's. Using
        # MAX_SPEED_M_S for both -- as this did -- silently deleted the
        # asymmetry: the opponent is built at 15 m/s / 80 m/s^2 to stand in
        # for a human, and the first randomised reset overwrote that with
        # 6-12 m/s and 10-22.5 m/s^2, i.e. a second robot that is on average
        # SLOWER than the one it is sparring with.
        if self.domain_randomize:
            for dyn in (self._agent_dyn, self._opp_dyn):
                slo, shi = DR_SPEED_RANGE
                alo, ahi = DR_ACCEL_RANGE
                speed, accel = dyn["nominal_speed"], dyn["nominal_accel"]
                dyn["max_speed"][idx] = self._rng.uniform(
                    slo * speed, shi * speed, size=n)
                dyn["max_accel"][idx] = self._rng.uniform(
                    alo * accel, ahi * accel, size=n)
                dyn["time_constant"][idx] = self._rng.uniform(0.01, 0.04, size=n)

        self._agent_dyn["x"][idx] = self.engine.paddle_agent_x[idx]
        self._agent_dyn["y"][idx] = self.engine.paddle_agent_y[idx]
        self._agent_dyn["vx"][idx] = 0.0
        self._agent_dyn["vy"][idx] = 0.0
        self._clear_profile_accel(self._agent_dyn, idx)

        # Override opponent position for stationary policies (per-env)
        cfg = self.table_config
        resetting = np.ones(self.n_envs, dtype=bool) if mask is None else mask

        goalie_reset = resetting & (self._opp_policy_id == OPP_GOALIE)
        if np.any(goalie_reset):
            self.engine.paddle_opp_x[goalie_reset] = cfg.width / 2
            self.engine.paddle_opp_y[goalie_reset] = cfg.height - cfg.paddle_radius

        corner_reset = resetting & (self._opp_policy_id == OPP_CORNER)
        if np.any(corner_reset):
            n_c = int(corner_reset.sum())
            corners = np.array([
                [cfg.paddle_radius, cfg.height - cfg.paddle_radius],
                [cfg.width - cfg.paddle_radius, cfg.height - cfg.paddle_radius],
            ])
            picks = self._rng.integers(0, len(corners), size=n_c)
            self.engine.paddle_opp_x[corner_reset] = corners[picks, 0]
            self.engine.paddle_opp_y[corner_reset] = corners[picks, 1]

        self._opp_dyn["x"][idx] = self.engine.paddle_opp_x[idx]
        self._opp_dyn["y"][idx] = self.engine.paddle_opp_y[idx]
        self._opp_dyn["vx"][idx] = 0.0
        self._opp_dyn["vy"][idx] = 0.0
        self._clear_profile_accel(self._opp_dyn, idx)

        # Init previous positions (zero velocity at start)
        if self._perception is not None:
            self._perception.reset(self.engine.puck_x, self.engine.puck_y, idx)

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
            # Penalize if puck stalled on agent's side (agent should have hit it)
            on_agent_side = stuck & (self.engine.puck_y < self.table_config.height / 2)
            rewards[on_agent_side] -= 0.5

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

    @property
    def external_mask(self) -> np.ndarray:
        """Boolean mask [N] — True for envs using external (self-play) opponent."""
        return self._opp_policy_id == OPP_EXTERNAL

    def mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """Mirror observations [N, 12] for opponent perspective.

        Flip y positions, negate y velocities, swap agent/opponent.
        """
        cfg = self.table_config
        m = obs.copy()
        # Obs: [puck_x, puck_y, puck_vx, puck_vy,
        #        pad_x, pad_y, pad_vx, pad_vy,
        #        opp_x, opp_y, opp_vx, opp_vy]

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
        # FLIP the side flag rather than pinning it to HUMAN. Whoever looks
        # through a mirrored view is the other side -- so mirroring the
        # robot's view yields the human's, and mirroring that yields the
        # robot's again. Pinning it made mirror_obs stop being an involution,
        # which is exactly what its round-trip test is for.
        m[:, 12] = (self.ROBOT_SIDE + self.HUMAN_SIDE) - obs[:, 12]

        # The cap features describe the body being driven, so they follow the
        # side across the mirror. Which caps to write depends on which side the
        # INCOMING view belongs to -- always writing the opponent's would make
        # mirror(mirror(x)) != x, the same way pinning the side flag did.
        # Caps are per-env constants within an episode, so reading them live
        # rather than from the (possibly delayed) obs is safe.
        was_robot = obs[:, 12] > (self.ROBOT_SIDE + self.HUMAN_SIDE) * 0.5
        for col, key, ref in ((13, "max_speed", MAX_SPEED_M_S),
                              (14, "max_accel", MAX_ACCEL_M_S2)):
            m[:, col] = np.where(was_robot,
                                 self._opp_dyn[key],
                                 self._agent_dyn[key]) / ref
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
        if dyn["type"] == "profile":
            # The real firmware control law, via fw/host. Millimetres, because
            # that is what the Teensy works in -- the law itself is
            # scale-agnostic, so sim metres * 1000 keeps position and caps
            # consistent without needing the grid frame here.
            cart = dyn["cart"]
            cart.x[:] = dyn["x"] * 1000.0
            cart.y[:] = dyn["y"] * 1000.0
            cart.vx[:] = dyn["vx"] * 1000.0
            cart.vy[:] = dyn["vy"] * 1000.0
            substeps = max(1, int(round(dt / DEFAULT_SIM_DT)))
            advance(cart,
                    (target_x * 1000.0).astype(np.float32),
                    (target_y * 1000.0).astype(np.float32),
                    dyn["max_speed"] * 1000.0, dyn["max_accel"] * 1000.0,
                    dyn["ramp_s"], dt / substeps, substeps)
            dyn["x"] = cart.x.astype(np.float64) / 1000.0
            dyn["y"] = cart.y.astype(np.float64) / 1000.0
            dyn["vx"] = cart.vx.astype(np.float64) / 1000.0
            dyn["vy"] = cart.vy.astype(np.float64) / 1000.0
            return dyn["x"].copy(), dyn["y"].copy()

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
        """Vectorized per-env opponent policies. Returns target (x, y) arrays."""
        cfg = self.table_config

        # Default: hold current position (idle, goalie, corner)
        target_x = self.engine.paddle_opp_x.copy()
        target_y = self.engine.paddle_opp_y.copy()

        # Follow: track puck x, stay near back wall
        follow = self._opp_policy_id == OPP_FOLLOW
        target_x[follow] = self.engine.puck_x[follow]
        target_y[follow] = cfg.height * 0.85

        # Random: random target each step
        random_mask = self._opp_policy_id == OPP_RANDOM
        n_rand = int(random_mask.sum())
        if n_rand > 0:
            target_x[random_mask] = self._rng.uniform(
                cfg.paddle_radius, cfg.width - cfg.paddle_radius, size=n_rand
            )
            target_y[random_mask] = self._rng.uniform(
                cfg.height / 2 + cfg.paddle_radius,
                cfg.height - cfg.paddle_radius,
                size=n_rand,
            )

        # External: use provided targets
        ext = self._opp_policy_id == OPP_EXTERNAL
        target_x[ext] = self._ext_opp_target_x[ext]
        target_y[ext] = self._ext_opp_target_y[ext]

        return self._update_dynamics(self._opp_dyn, target_x, target_y, dt)

    def _clamp_to_half(
        self, x: np.ndarray, y: np.ndarray, agent: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.table_config
        r = cfg.paddle_radius
        if agent and self._ws is not None:
            # The machine's own limit, not the table's. The firmware clamps
            # here too; doing it in the sim means the policy learns the
            # boundary instead of discovering it on the hardware.
            return (np.clip(x, self._ws["min_x"], self._ws["max_x"]),
                    np.clip(y, self._ws["min_y"], self._ws["max_y"]))
        x = np.clip(x, r, cfg.width - r)
        if agent:
            y = np.clip(y, r, cfg.height / 2 - r)
        else:
            y = np.clip(y, cfg.height / 2 + r, cfg.height - r)
        return x, y

    def _make_obs_direct(self) -> np.ndarray:
        """Build [N, 12] observation with positions + velocities."""
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

        if self._perception is not None:
            px, py, pvx, pvy = self._perception.update(e.puck_x, e.puck_y)
        else:
            px, py, pvx, pvy = e.puck_x, e.puck_y, e.puck_vx, e.puck_vy

        # Caps as a ratio to the robot's nominal, so a nominal robot reads
        # exactly 1.0 on both and anything else is a ratio to the machine as
        # built. The human side reads above 1.0, which is the point.
        return np.column_stack([
            px, py, pvx, pvy,
            e.paddle_agent_x, e.paddle_agent_y, agent_vx, agent_vy,
            e.paddle_opp_x, e.paddle_opp_y, opp_vx, opp_vy,
            np.full(self.n_envs, self.ROBOT_SIDE),
            self._agent_dyn["max_speed"] / MAX_SPEED_M_S,
            self._agent_dyn["max_accel"] / MAX_ACCEL_M_S2,
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
