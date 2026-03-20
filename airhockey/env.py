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
    """Air hockey environment.

    Observation (12,):
        [0:2]   puck position (x, y)
        [2:4]   puck velocity (vx, vy)
        [4:6]   agent paddle position (x, y)
        [6:8]   agent paddle velocity (vx, vy)
        [8:10]  opponent paddle position (x, y)
        [10:12] opponent paddle velocity (vx, vy)

    Action (2,):
        Target (x, y) position for the agent's paddle.

    Camera delay is simulated by buffering observations and returning
    a delayed version.
    """

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
        max_score: int = 7,
        opponent_policy: str = "idle",  # "idle", "follow", "random"
        record: bool = False,
    ):
        super().__init__()

        self.table_config = table_config or TableConfig()
        self.agent_dynamics = agent_dynamics or IdealDynamics()
        self.opponent_dynamics = opponent_dynamics or DelayedDynamics()
        self.physics_dt = physics_dt
        self.action_dt = action_dt
        self.camera_delay = camera_delay
        self.max_episode_time = max_episode_time
        self.max_score = max_score
        self.opponent_policy = opponent_policy

        self.engine = PhysicsEngine(self.table_config)

        # Observation and action spaces
        cfg = self.table_config
        high_obs = np.array(
            [
                cfg.width, cfg.height,  # puck pos
                cfg.max_puck_speed, cfg.max_puck_speed,  # puck vel
                cfg.width, cfg.height,  # agent paddle pos
                10.0, 10.0,  # agent paddle vel
                cfg.width, cfg.height,  # opponent paddle pos
                10.0, 10.0,  # opponent paddle vel
            ],
            dtype=np.float32,
        )
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        state = self.engine.reset(self._rng)

        self.agent_dynamics.reset(state.paddle_agent.x, state.paddle_agent.y)
        self.opponent_dynamics.reset(state.paddle_opponent.x, state.paddle_opponent.y)

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

        obs = self._make_obs(state)

        # XXX REMOVE THIS FOR SELF-PLAY OR REAL OPPONENTS XXX
        # Reset puck if it comes to rest in opponent's half (idle opponent can't return it)
        puck = state.puck
        if (puck.y > self.table_config.height / 2
                and abs(puck.vx) < 0.05 and abs(puck.vy) < 0.05):
            self.engine._reset_puck_after_goal(toward_agent=True)
        # XXX END REMOVE XXX

        # Check termination
        terminated = (
            state.score_agent >= self.max_score
            or state.score_opponent >= self.max_score
        )
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

        return opp.x, opp.y

    def _make_obs(self, state: PhysicsState) -> np.ndarray:
        return np.array(
            [
                state.puck.x, state.puck.y,
                state.puck.vx, state.puck.vy,
                state.paddle_agent.x, state.paddle_agent.y,
                state.paddle_agent.vx, state.paddle_agent.vy,
                state.paddle_opponent.x, state.paddle_opponent.y,
                state.paddle_opponent.vx, state.paddle_opponent.vy,
            ],
            dtype=np.float32,
        )

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
