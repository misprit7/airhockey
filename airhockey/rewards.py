"""Reward shaping for air hockey training.

Components:
    1. PROXIMITY (exp(-3*dist)): Exponential closeness to puck. Strongly
       rewards being very close, near-zero for being far away.

    2. GOAL SCORED (+100): Large reward for scoring.

    3. GOAL CONCEDED (-100): Large penalty.

    4. PUCK PROGRESS (potential-based, weight 3.0): Reward for puck moving
       up the table toward opponent's goal.

    5. CONTACT (+5.0): Reward when paddle hits the puck, detected via
       speed change near paddle.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ShapedRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        goal_reward: float = 100.0,
        goal_penalty: float = -5.0,
        puck_progress_weight: float = 3.0,
        contact_reward: float = 5.0,
    ):
        super().__init__(env)
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.puck_progress_weight = puck_progress_weight
        self.contact_reward = contact_reward

        self._prev_puck_y = None
        self._prev_puck_speed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_puck_y = obs[1]
        self._prev_puck_speed = np.linalg.norm(obs[2:4])
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        # 1. Proximity (exponential)
        puck_pos = obs[0:2]
        paddle_pos = obs[4:6]
        dist = np.linalg.norm(puck_pos - paddle_pos)
        shaped_reward += 0.1 * float(np.exp(-3.0 * dist))

        # 2. Goals
        if reward > 0:
            shaped_reward += self.goal_reward
        elif reward < 0:
            shaped_reward += self.goal_penalty

        # 3. Puck progress toward opponent goal (potential-based)
        puck_y = obs[1]
        if self._prev_puck_y is not None:
            delta = puck_y - self._prev_puck_y
            if delta > 0:  # only reward forward progress, never punish
                shaped_reward += self.puck_progress_weight * delta

        # 4. Contact detection
        puck_speed = np.linalg.norm(obs[2:4])
        if self._prev_puck_speed is not None and dist < 0.15:
            speed_change = puck_speed - self._prev_puck_speed
            if speed_change > 0.3:
                shaped_reward += self.contact_reward

        # Update state
        self._prev_puck_y = puck_y
        self._prev_puck_speed = puck_speed

        # Reset potentials after goal
        if reward != 0:
            self._prev_puck_y = obs[1]

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        self.env.unwrapped.record_reward(shaped_reward)

        return obs, shaped_reward, terminated, truncated, info
