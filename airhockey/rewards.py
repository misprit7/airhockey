"""Reward shaping wrapper for air hockey training.

Reward design rationale:
    Uses potential-based reward shaping where possible. Potential-based
    shaping rewards CHANGES in state rather than absolute state values,
    which is guaranteed not to alter the optimal policy.

Components:
    1. GOAL SCORED (+100): Large reward for scoring. Must dominate shaping.

    2. GOAL CONCEDED (-100): Large penalty for letting puck in your goal.

    3. APPROACH PUCK (potential-based, weight 2.0): Reward for DECREASING
       distance to the puck. Unlike rewarding absolute closeness, this
       gives zero reward for sitting near the puck and only rewards
       movement toward it. Avoids the local optimum of camping near the
       puck without engaging.

    4. PUCK TOWARD OPPONENT (potential-based, weight 3.0): Reward for puck
       moving up the table (increasing y). Rewards offensive play and
       pushing the puck toward the goal. Potential-based so only rewards
       change, not absolute position.

    5. PUCK CONTACT (+5.0): Reward when paddle collides with puck. Detected
       via the physics engine's collision state rather than velocity heuristics.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ShapedRewardWrapper(gym.Wrapper):
    """Wraps AirHockeyEnv with shaped rewards for training."""

    def __init__(
        self,
        env: gym.Env,
        goal_reward: float = 100.0,
        goal_penalty: float = -100.0,
        approach_weight: float = 2.0,
        puck_progress_weight: float = 3.0,
        contact_reward: float = 5.0,
    ):
        super().__init__(env)
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.approach_weight = approach_weight
        self.puck_progress_weight = puck_progress_weight
        self.contact_reward = contact_reward

        self._prev_dist_to_puck = None
        self._prev_puck_y = None
        self._prev_puck_speed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist_to_puck = self._dist_to_puck(obs)
        self._prev_puck_y = obs[1]
        self._prev_puck_speed = np.linalg.norm(obs[2:4])
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        # 1. Goal rewards
        if reward > 0:
            shaped_reward += self.goal_reward
        elif reward < 0:
            shaped_reward += self.goal_penalty

        # 2. Potential-based: approach puck (reward for decreasing distance)
        dist = self._dist_to_puck(obs)
        if self._prev_dist_to_puck is not None:
            # Positive when distance decreases (approaching)
            shaped_reward += self.approach_weight * (self._prev_dist_to_puck - dist)

        # 3. Potential-based: puck progress toward opponent goal
        puck_y = obs[1]
        if self._prev_puck_y is not None:
            # Positive when puck moves up (toward opponent)
            shaped_reward += self.puck_progress_weight * (puck_y - self._prev_puck_y)

        # 4. Contact reward: detect paddle-puck collision via speed change near paddle
        puck_speed = np.linalg.norm(obs[2:4])
        if self._prev_puck_speed is not None and dist < 0.15:
            speed_change = puck_speed - self._prev_puck_speed
            if speed_change > 0.3:  # puck sped up near paddle = hit
                shaped_reward += self.contact_reward

        # Update state for next step
        self._prev_dist_to_puck = dist
        self._prev_puck_y = puck_y
        self._prev_puck_speed = puck_speed

        # Reset potentials after a goal (puck resets to center)
        if reward != 0:
            self._prev_dist_to_puck = self._dist_to_puck(obs)
            self._prev_puck_y = obs[1]

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info

    def _dist_to_puck(self, obs: np.ndarray) -> float:
        return float(np.linalg.norm(obs[0:2] - obs[4:6]))
