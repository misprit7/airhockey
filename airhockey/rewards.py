"""Reward shaping wrapper for air hockey training.

Reward design rationale:
    Pure sparse rewards (only goals) are too rare for the agent to learn from
    initially — the puck rarely enters a goal by accident. We add shaped
    rewards that guide the agent toward useful behaviors while keeping goal
    scoring as the dominant signal.

Components:
    1. GOAL SCORED (+10): Large reward for scoring. This is the ultimate
       objective and must dominate all shaping rewards.

    2. GOAL CONCEDED (-10): Large penalty. Teaches the agent that defense
       matters and it can't just ignore the puck.

    3. PUCK VELOCITY TOWARD OPPONENT (+0.1 * vy): Small continuous reward
       proportional to how fast the puck is moving toward the opponent's
       goal (positive vy). Encourages the agent to hit the puck offensively
       rather than just sitting still. Scales with vy so harder hits are
       rewarded more.

    4. CLOSENESS TO PUCK (+0.05 * closeness): Small reward for being near
       the puck when it's in the agent's half. Without this the agent has
       no gradient toward engaging with the puck early in training. Only
       active when puck is in agent's half to avoid chasing it everywhere.

    5. EXISTENCE PENALTY (-0.001/step): Tiny negative reward each step to
       discourage passive play and encourage the agent to end rallies by
       scoring rather than stalling.
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
        goal_reward: float = 10.0,
        goal_penalty: float = -10.0,
        puck_velocity_weight: float = 0.1,
        closeness_weight: float = 0.05,
        existence_penalty: float = -0.001,
    ):
        super().__init__(env)
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.puck_velocity_weight = puck_velocity_weight
        self.closeness_weight = closeness_weight
        self.existence_penalty = existence_penalty

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        # 1. Goal rewards (from base env: +1 for score, -1 for concede)
        if reward > 0:
            shaped_reward += self.goal_reward
        elif reward < 0:
            shaped_reward += self.goal_penalty

        # 2. Puck velocity toward opponent's goal (positive vy = toward opponent)
        puck_vy = obs[3]
        shaped_reward += self.puck_velocity_weight * max(0.0, puck_vy)

        # 3. Closeness to puck when puck is in agent's half
        puck_pos = obs[0:2]
        paddle_pos = obs[4:6]
        table_height = self.env.unwrapped.table_config.height

        if puck_pos[1] < table_height / 2:
            dist = np.linalg.norm(puck_pos - paddle_pos)
            max_dist = np.hypot(table_height / 2, self.env.unwrapped.table_config.width)
            closeness = 1.0 - (dist / max_dist)
            shaped_reward += self.closeness_weight * closeness

        # 4. Existence penalty
        shaped_reward += self.existence_penalty

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info
