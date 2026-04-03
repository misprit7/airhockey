"""Gymnasium environment for air hockey."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from airhockey.dynamics import DelayedDynamics, IdealDynamics, MotorDynamics
from airhockey.physics import PhysicsEngine, PhysicsState, TableConfig
from airhockey.recorder import FrameData, Recorder


class AirHockeyEnv(gym.Env):
    """Air hockey environment with 14-dim observation.

    Observation layout (14 dims):
        puck_x, puck_y, puck_vx, puck_vy,
        paddle_x, paddle_y, paddle_vx, paddle_vy,
        opp_x, opp_y, opp_vx, opp_vy,
        score_diff, time_remaining

    Action (2,):
        Target (x, y) position for the agent's paddle.

    Camera delay is simulated by buffering observations and returning
    a delayed version.
    """

    OBS_DIM = 14

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        table_config: TableConfig | None = None,
        agent_dynamics: MotorDynamics | None = None,
        opponent_dynamics: MotorDynamics | None = None,
        physics_dt: float = 1 / 240,  # 240 Hz physics
        action_dt: float = 1 / 60,  # 60 Hz agent control
        camera_delay: float = 0.0,  # seconds of observation delay
        max_episode_time: float = 60.0,  # seconds
        max_episode_steps: int | None = None,
        max_score: int = 7,
        opponent_policy: str = "idle",  # "idle", "follow", "random"
        record: bool = False,
        frame_stack: int = 1,  # kept for API compat, must be 1
    ):
        super().__init__()

        self.table_config = table_config or TableConfig()
        self.agent_dynamics = agent_dynamics or IdealDynamics()
        self.opponent_dynamics = opponent_dynamics or DelayedDynamics()
        self.physics_dt = physics_dt
        self.action_dt = action_dt
        self.camera_delay = camera_delay
        self.max_episode_time = max_episode_time
        self.max_episode_steps = max_episode_steps
        self.max_score = max_score
        self.opponent_policy = opponent_policy
        self.frame_stack = 1  # always 1; velocities replace stacking
        self._step_count = 0

        self.engine = PhysicsEngine(self.table_config)

        # Observation and action spaces
        cfg = self.table_config
        # Bounds: positions [0, width/height], velocities large, context [-1, 1]
        vel_max = 10.0  # generous velocity bound
        high_obs = np.array([
            cfg.width, cfg.height, vel_max, vel_max,      # puck
            cfg.width, cfg.height, vel_max, vel_max,      # paddle
            cfg.width, cfg.height, vel_max, vel_max,      # opponent
            1.0, 1.0,                                     # context
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        # Action: normalized [-1, 1] mapped to paddle position in agent's half
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Real position bounds for rescaling
        self._action_low = np.array([cfg.paddle_radius, cfg.paddle_radius])
        self._action_high = np.array(
            [cfg.width - cfg.paddle_radius, cfg.height / 2 - cfg.paddle_radius]
        )

        # Camera delay buffer
        self._obs_buffer: deque[np.ndarray] = deque()
        self._delay_steps = max(0, int(camera_delay / action_dt))

        # Recording
        self.recorder = Recorder() if record else None
        self._rng: np.random.Generator | None = None

        # Previous paddle positions for velocity estimation
        self._prev_agent_x: float = 0.0
        self._prev_agent_y: float = 0.0
        self._prev_opp_x: float = 0.0
        self._prev_opp_y: float = 0.0

        # Puck-stuck detection: reset if speed < threshold for N consecutive steps
        self._puck_slow_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._puck_slow_count = 0

        state = self.engine.reset(self._rng)

        self.agent_dynamics.reset(state.paddle_agent.x, state.paddle_agent.y)

        # Override opponent position for stationary policies
        cfg = self.table_config
        if self.opponent_policy == "corner":
            corners = [
                (cfg.paddle_radius, cfg.height - cfg.paddle_radius),
                (cfg.width - cfg.paddle_radius, cfg.height - cfg.paddle_radius),
            ]
            cx, cy = corners[self._rng.integers(0, len(corners))]
            state.paddle_opponent.x = cx
            state.paddle_opponent.y = cy
        elif self.opponent_policy == "goalie":
            state.paddle_opponent.x = cfg.width / 2
            state.paddle_opponent.y = cfg.height - cfg.paddle_radius

        self.opponent_dynamics.reset(state.paddle_opponent.x, state.paddle_opponent.y)

        # Init previous positions (zero velocity at start)
        self._prev_agent_x = state.paddle_agent.x
        self._prev_agent_y = state.paddle_agent.y
        self._prev_opp_x = state.paddle_opponent.x
        self._prev_opp_y = state.paddle_opponent.y

        self._obs_buffer.clear()
        obs = self._make_obs(state)
        # Pre-fill delay buffer
        for _ in range(self._delay_steps):
            self._obs_buffer.append(obs.copy())

        self._step_reward = 0.0
        self._cumulative_reward = 0.0

        if self.recorder:
            self.recorder.start_episode()
            self.recorder.record(self._make_frame(state))

        return self._get_delayed_obs(obs), self._make_info(state)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Clip to [-1, 1] then rescale to real position bounds
        action = np.clip(action, -1.0, 1.0)
        real_action = self._action_low + (action + 1.0) * 0.5 * (self._action_high - self._action_low)
        target_x, target_y = float(real_action[0]), float(real_action[1])

        # Run physics substeps
        n_substeps = max(1, int(self.action_dt / self.physics_dt))
        sub_dt = self.action_dt / n_substeps

        state = self.engine.state
        reward = 0.0

        for _ in range(n_substeps):
            # Update agent paddle through dynamics
            ax, ay = self.agent_dynamics.update(target_x, target_y, sub_dt)
            ax, ay = self._clamp_to_half(ax, ay, agent=True)
            self.engine.update_paddle(state.paddle_agent, ax, ay, sub_dt)

            # Update opponent
            ox, oy = self._opponent_action(state, sub_dt)
            ox, oy = self._clamp_to_half(ox, oy, agent=False)
            self.engine.update_paddle(state.paddle_opponent, ox, oy, sub_dt)

            self.engine.step(sub_dt)

            if state.goal_scored == 1:
                reward += 1.0
            elif state.goal_scored == -1:
                reward -= 1.0

        self._step_count += 1
        obs = self._make_obs(state)

        # Universal puck-stuck reset: if puck speed < 0.05 for 120 steps (~2s),
        # reset to center heading toward a random side.
        puck_speed = np.hypot(state.puck.vx, state.puck.vy)
        if puck_speed < 0.05:
            self._puck_slow_count += 1
        else:
            self._puck_slow_count = 0

        if self._puck_slow_count >= 120:
            # Penalize if puck stalled on agent's side (agent should have hit it)
            if state.puck.y < self.table_config.height / 2:
                reward -= 0.5
            self.engine._reset_puck_after_goal(toward_agent=self._rng.random() < 0.5)
            self._puck_slow_count = 0

        # Check termination
        terminated = (
            state.score_agent >= self.max_score
            or state.score_opponent >= self.max_score
        )
        if self.max_episode_steps is not None:
            truncated = self._step_count >= self.max_episode_steps
        else:
            truncated = state.time >= self.max_episode_time

        if self.recorder:
            self.recorder.record(self._make_frame(state))

        return self._get_delayed_obs(obs), reward, terminated, truncated, self._make_info(state)

    def record_reward(self, reward: float) -> None:
        """Called by reward wrapper to log shaped reward into recordings."""
        self._step_reward = reward
        self._cumulative_reward += reward

    def _clamp_to_half(self, x: float, y: float, agent: bool) -> tuple[float, float]:
        """Clamp paddle position to its own half of the table."""
        cfg = self.table_config
        r = cfg.paddle_radius
        x = np.clip(x, r, cfg.width - r)
        if agent:
            y = np.clip(y, r, cfg.height / 2 - r)
        else:
            y = np.clip(y, cfg.height / 2 + r, cfg.height - r)
        return float(x), float(y)

    def set_opponent_action(self, target_x: float, target_y: float) -> None:
        """Set opponent target for 'external' policy mode."""
        self._external_opponent_target = (target_x, target_y)

    def mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """Mirror observation so opponent sees the game from its perspective.

        Flip y positions, negate y velocities, swap agent/opponent.
        Negate score_diff; time_remaining unchanged.
        """
        cfg = self.table_config
        mirrored = obs.copy()
        # Obs: [puck_x, puck_y, puck_vx, puck_vy,
        #        pad_x, pad_y, pad_vx, pad_vy,
        #        opp_x, opp_y, opp_vx, opp_vy,
        #        score_diff, time_remaining]

        # Puck: flip y, negate vy
        mirrored[1] = cfg.height - obs[1]    # puck_y
        mirrored[3] = -obs[3]                # puck_vy

        # Swap agent/opponent and flip y, negate vy
        mirrored[4] = obs[8]                 # opp_x → pad_x
        mirrored[5] = cfg.height - obs[9]    # opp_y → pad_y (flipped)
        mirrored[6] = obs[10]                # opp_vx → pad_vx
        mirrored[7] = -obs[11]               # opp_vy → pad_vy (negated)
        mirrored[8] = obs[4]                 # pad_x → opp_x
        mirrored[9] = cfg.height - obs[5]    # pad_y → opp_y (flipped)
        mirrored[10] = obs[6]                # pad_vx → opp_vx
        mirrored[11] = -obs[7]               # pad_vy → opp_vy (negated)

        # Context
        mirrored[12] = -obs[12]  # negate score_diff
        # time_remaining unchanged
        return mirrored

    def mirror_action_to_opponent(self, action: np.ndarray) -> tuple[float, float]:
        """Convert a [-1,1] normalized action from the opponent's mirrored
        perspective back to real table coordinates for the opponent's half."""
        action = np.clip(action, -1.0, 1.0)
        cfg = self.table_config
        r = cfg.paddle_radius
        # x: same mapping as agent
        x = r + (action[0] + 1.0) * 0.5 * (cfg.width - 2 * r)
        # y: mirrored — y=-1 means "near own goal" = back wall (height - r),
        # y=+1 means "opponent's side" = midfield (height/2 + r)
        y = (cfg.height - r) - (action[1] + 1.0) * 0.5 * (cfg.height / 2 - 2 * r)
        return float(x), float(y)

    def _opponent_action(self, state: PhysicsState, dt: float) -> tuple[float, float]:
        """Simple built-in opponent policies."""
        opp = state.paddle_opponent
        cfg = self.table_config

        if self.opponent_policy == "idle":
            return opp.x, opp.y

        elif self.opponent_policy == "follow":
            # Track puck x, stay near baseline
            target_x = state.puck.x
            target_y = cfg.height * 0.85
            return self.opponent_dynamics.update(target_x, target_y, dt)

        elif self.opponent_policy == "random":
            if self._rng is None:
                self._rng = np.random.default_rng()
            target_x = self._rng.uniform(cfg.paddle_radius, cfg.width - cfg.paddle_radius)
            target_y = self._rng.uniform(cfg.height / 2 + cfg.paddle_radius, cfg.height - cfg.paddle_radius)
            return self.opponent_dynamics.update(target_x, target_y, dt)

        elif self.opponent_policy == "corner":
            # Stationary in a random corner (chosen at reset)
            return opp.x, opp.y

        elif self.opponent_policy == "goalie":
            # Stationary centered in front of goal
            return opp.x, opp.y

        elif self.opponent_policy == "external":
            target = getattr(self, '_external_opponent_target', (opp.x, opp.y))
            return self.opponent_dynamics.update(target[0], target[1], dt)

        return opp.x, opp.y

    def _make_obs(self, state: PhysicsState) -> np.ndarray:
        """Build 14-dim observation with positions + velocities + context."""
        dt = self.action_dt
        # Paddle velocities from finite differences
        agent_vx = (state.paddle_agent.x - self._prev_agent_x) / dt
        agent_vy = (state.paddle_agent.y - self._prev_agent_y) / dt
        opp_vx = (state.paddle_opponent.x - self._prev_opp_x) / dt
        opp_vy = (state.paddle_opponent.y - self._prev_opp_y) / dt

        # Update previous positions
        self._prev_agent_x = state.paddle_agent.x
        self._prev_agent_y = state.paddle_agent.y
        self._prev_opp_x = state.paddle_opponent.x
        self._prev_opp_y = state.paddle_opponent.y

        # Context
        score_diff = (state.score_agent - state.score_opponent) / max(self.max_score, 1)
        if self.max_episode_steps is not None:
            time_remaining = max(0, self.max_episode_steps - self._step_count) / self.max_episode_steps
        else:
            time_remaining = max(0.0, self.max_episode_time - state.time) / self.max_episode_time

        return np.array([
            state.puck.x, state.puck.y, state.puck.vx, state.puck.vy,
            state.paddle_agent.x, state.paddle_agent.y, agent_vx, agent_vy,
            state.paddle_opponent.x, state.paddle_opponent.y, opp_vx, opp_vy,
            score_diff, time_remaining,
        ], dtype=np.float32)

    def _get_delayed_obs(self, current_obs: np.ndarray) -> np.ndarray:
        if self._delay_steps == 0:
            return current_obs
        self._obs_buffer.append(current_obs.copy())
        return self._obs_buffer.popleft()

    def _make_info(self, state: PhysicsState) -> dict[str, Any]:
        return {
            "score_agent": state.score_agent,
            "score_opponent": state.score_opponent,
            "time": state.time,
            "puck_vx": state.puck.vx,
            "puck_vy": state.puck.vy,
        }

    def _make_frame(self, state: PhysicsState) -> FrameData:
        return FrameData(
            time=state.time,
            puck_x=state.puck.x,
            puck_y=state.puck.y,
            puck_vx=state.puck.vx,
            puck_vy=state.puck.vy,
            agent_x=state.paddle_agent.x,
            agent_y=state.paddle_agent.y,
            opponent_x=state.paddle_opponent.x,
            opponent_y=state.paddle_opponent.y,
            score_agent=state.score_agent,
            score_opponent=state.score_opponent,
            reward=getattr(self, '_step_reward', 0.0),
            cumulative_reward=getattr(self, '_cumulative_reward', 0.0),
        )

    def get_recording(self) -> list[FrameData] | None:
        if self.recorder:
            return self.recorder.get_episode()
        return None
