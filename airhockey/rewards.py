"""Reward shaping wrapper for air hockey training.

Reward design rationale:
    Pure sparse rewards (only goals) are too rare for the agent to learn from
    initially — the puck rarely enters a goal by accident. We add shaped
    rewards that guide the agent toward useful behaviors while keeping goal
    scoring as the dominant signal.

Components:
    1. GOAL SCORED (+10): Large reward for scoring. The ultimate objective.

    2. GOAL CONCEDED (-10): Large penalty for letting puck in your goal.

    3. CLOSENESS TO PUCK (+0.5 * closeness): Reward for being near the puck.
       This is the critical bootstrapping signal — without it, the agent has
       no gradient to learn from early in training. Always active so the
       agent learns to track the puck everywhere in its half.

    4. PUCK CONTACT (+2.0): One-time reward each time the paddle touches the
       puck. Directly rewards engagement and teaches that touching the puck
       matters.

    5. PUCK VELOCITY TOWARD OPPONENT (+0.3 * vy): Rewards hitting the puck
       toward the opponent's goal. Scales with velocity so harder offensive
       hits earn more.

    6. PUCK IN DANGER ZONE (-0.2): Penalty when puck is close to agent's
       goal. Creates urgency to defend and clear the puck.
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
        closeness_weight: float = 0.5,
        contact_reward: float = 2.0,
        puck_velocity_weight: float = 0.3,
        danger_zone_penalty: float = -0.2,
    ):
        super().__init__(env)
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.closeness_weight = closeness_weight
        self.contact_reward = contact_reward
        self.puck_velocity_weight = puck_velocity_weight
        self.danger_zone_penalty = danger_zone_penalty
        self._prev_puck_vel = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_puck_vel = obs[2:4].copy()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        cfg = self.env.unwrapped.table_config
        shaped_reward = 0.0

        # 1. Goal rewards (from base env: +1 for score, -1 for concede)
        if reward > 0:
            shaped_reward += self.goal_reward
        elif reward < 0:
            shaped_reward += self.goal_penalty

        puck_pos = obs[0:2]
        puck_vel = obs[2:4]
        paddle_pos = obs[4:6]

        # 2. Closeness to puck (always active in agent's half)
        dist = np.linalg.norm(puck_pos - paddle_pos)
        max_dist = np.hypot(cfg.height / 2, cfg.width)
        closeness = 1.0 - min(dist / max_dist, 1.0)
        shaped_reward += self.closeness_weight * closeness

        # 3. Puck contact detection (sudden velocity change near paddle)
        if self._prev_puck_vel is not None:
            vel_change = np.linalg.norm(puck_vel - self._prev_puck_vel)
            if dist < (cfg.puck_radius + cfg.paddle_radius) * 2.0 and vel_change > 0.5:
                shaped_reward += self.contact_reward
        self._prev_puck_vel = puck_vel.copy()

        # 4. Puck velocity toward opponent's goal (positive vy = toward opponent)
        puck_vy = float(puck_vel[1])
        shaped_reward += self.puck_velocity_weight * max(0.0, puck_vy)

        # 5. Puck in danger zone (close to agent's goal)
        if puck_pos[1] < cfg.height * 0.2:
            shaped_reward += self.danger_zone_penalty

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info
