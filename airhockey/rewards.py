"""Reward shaping for air hockey training.

Supports curriculum stages:
    Stage 1 (proximity): Just learn to chase the puck
    Stage 2 (contact):   Learn to hit the puck toward the goal
    Stage 3 (scoring):   Learn to score goals

Each stage includes all rewards from previous stages.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# Curriculum stage definitions
STAGE_PROXIMITY = 1
STAGE_CONTACT = 2
STAGE_SCORING = 3


class ShapedRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        stage: int = STAGE_SCORING,
        proximity_weight: float = 0.1,
        proximity_k: float = 3.0,
        contact_reward: float = 5.0,
        directed_hit_weight: float = 2.0,
        puck_progress_weight: float = 3.0,
        goal_reward: float = 100.0,
        goal_penalty: float = -5.0,
    ):
        super().__init__(env)
        self.stage = stage
        self.proximity_weight = proximity_weight
        self.proximity_k = proximity_k
        self.contact_reward = contact_reward
        self.directed_hit_weight = directed_hit_weight
        self.puck_progress_weight = puck_progress_weight
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty

        self._prev_puck_y = None
        self._prev_puck_speed = None
        self._prev_puck_vy = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_puck_y = obs[1]
        self._prev_puck_speed = np.linalg.norm(obs[2:4])
        self._prev_puck_vy = obs[3]
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0
        puck_pos = obs[0:2]
        paddle_pos = obs[4:6]
        dist = np.linalg.norm(puck_pos - paddle_pos)
        puck_speed = np.linalg.norm(obs[2:4])
        puck_vy = obs[3]

        # --- Stage 1: Proximity ---
        shaped_reward += self.proximity_weight * float(np.exp(-self.proximity_k * dist))

        # --- Stage 2: Contact + directed hitting ---
        if self.stage >= STAGE_CONTACT:
            # Contact detection: puck sped up near paddle
            if self._prev_puck_speed is not None and dist < 0.15:
                speed_change = puck_speed - self._prev_puck_speed
                if speed_change > 0.3:
                    shaped_reward += self.contact_reward
                    # Bonus for hitting puck toward opponent (positive vy after hit)
                    if puck_vy > 0:
                        shaped_reward += self.directed_hit_weight * puck_vy

            # Puck progress toward opponent goal (one-way)
            if self._prev_puck_y is not None:
                delta = puck_pos[1] - self._prev_puck_y
                if delta > 0:
                    shaped_reward += self.puck_progress_weight * delta

        # --- Stage 3: Goal scoring ---
        if self.stage >= STAGE_SCORING:
            if reward > 0:
                shaped_reward += self.goal_reward
            elif reward < 0:
                shaped_reward += self.goal_penalty

        # Update state
        self._prev_puck_y = puck_pos[1]
        self._prev_puck_speed = puck_speed
        self._prev_puck_vy = puck_vy

        # Reset potentials after goal
        if reward != 0:
            self._prev_puck_y = obs[1]

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        self.env.unwrapped.record_reward(shaped_reward)

        return obs, shaped_reward, terminated, truncated, info
