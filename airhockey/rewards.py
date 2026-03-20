"""Minimal reward: just proximity to puck.

Testing if the agent can learn the most basic behavior — move toward the puck.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ShapedRewardWrapper(gym.Wrapper):
    """Reward = closeness to puck. Nothing else."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        cfg = self.env.unwrapped.table_config
        puck_pos = obs[0:2]
        paddle_pos = obs[4:6]

        dist = np.linalg.norm(puck_pos - paddle_pos)
        max_dist = np.hypot(cfg.height, cfg.width)
        closeness = 1.0 - (dist / max_dist)

        shaped_reward = closeness  # 0 to 1 per step

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        self.env.unwrapped.record_reward(shaped_reward)

        return obs, shaped_reward, terminated, truncated, info
