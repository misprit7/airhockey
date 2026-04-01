"""Reward shaping for air hockey training.

4-stage curriculum with progressive opponent difficulty:
    Stage 1 (chase+hit):    Chase and hit the puck (vs idle)
    Stage 2 (game/goalie):  Score past blocker (vs goalie)
    Stage 3 (game/follow):  Score vs reactive opponent (vs follower)
    Stage 4 (self-play):    Pure competitive play (vs opponent pool)

Reward weights are determined by stage defaults but can be overridden per-instance.
Stage 4 drops all auxiliary rewards — only goals and entropy remain.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Curriculum stage constants
# ---------------------------------------------------------------------------
STAGE_CHASE_HIT = 1
STAGE_GAME_GOALIE = 2
STAGE_GAME_FOLLOW = 3
STAGE_SELFPLAY = 4

# Backwards compatibility aliases
STAGE_PROXIMITY = STAGE_CHASE_HIT
STAGE_SCORING = STAGE_GAME_GOALIE

# ---------------------------------------------------------------------------
# Stage configuration tables
# ---------------------------------------------------------------------------
STAGE_DEFAULTS: dict[int, dict[str, float]] = {
    1: {"proximity": 0,    "contact": 3.0, "directed_hit": 2.0, "puck_progress": 0,   "defense": 0,   "shot_placement": 0,   "goal_reward": 0,     "goal_penalty": 0,     "entropy": 0},
    2: {"proximity": 0,    "contact": 0.1, "directed_hit": 0.1, "puck_progress": 0.1, "defense": 0.1, "shot_placement": 0.2, "goal_reward": 160.0, "goal_penalty": -20.0, "entropy": 0},
    3: {"proximity": 0,    "contact": 0,   "directed_hit": 0,   "puck_progress": 0.1, "defense": 0.1, "shot_placement": 0.2, "goal_reward": 160.0, "goal_penalty": -20.0, "entropy": 0},
    4: {"proximity": 0,    "contact": 0,   "directed_hit": 0,   "puck_progress": 0,   "defense": 0,   "shot_placement": 0,   "goal_reward": 130.0, "goal_penalty": -20.0, "entropy": 0},
}

STAGE_OPPONENT: dict[int, str] = {
    1: "idle",
    2: "goalie",
    3: "follow",
    4: "external",
}

STAGE_NAMES: dict[int, str] = {
    1: "CHASE+HIT",
    2: "GAME vs GOALIE",
    3: "GAME vs FOLLOWER",
    4: "SELF-PLAY",
}

STAGE_EPISODE_STEPS: dict[int, int] = {
    1: 600,    # ~10s at 60Hz — chase + hit
    2: 1200,   # ~20s — game vs goalie
    3: 1800,   # ~30s — game vs follower
    4: 1800,   # ~30s — self-play
}

# Default table geometry for shot placement
_GOAL_CX = 0.5   # table_width / 2
_GOAL_CY = 2.0   # table_height


def _resolve(explicit: float | None, key: str, stage: int) -> float:
    """Return explicit weight if provided, else stage default."""
    if explicit is not None:
        return explicit
    return STAGE_DEFAULTS[stage][key]


# ---------------------------------------------------------------------------
# Single-env reward wrapper (gym.Wrapper)
# ---------------------------------------------------------------------------
class ShapedRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        stage: int = STAGE_GAME_GOALIE,
        proximity_weight: float | None = None,
        proximity_k: float = 3.0,
        contact_reward: float | None = None,
        directed_hit_weight: float | None = None,
        puck_progress_weight: float | None = None,
        goal_reward: float | None = None,
        goal_penalty: float | None = None,
        defense_weight: float | None = None,
        shot_placement_weight: float | None = None,
        entropy_weight: float | None = None,
        max_contacts_per_episode: int = 5,
    ):
        super().__init__(env)
        self.stage = stage
        self.proximity_k = proximity_k
        self.proximity_weight = _resolve(proximity_weight, "proximity", stage)
        self.contact_reward = _resolve(contact_reward, "contact", stage)
        self.directed_hit_weight = _resolve(directed_hit_weight, "directed_hit", stage)
        self.puck_progress_weight = _resolve(puck_progress_weight, "puck_progress", stage)
        self.goal_reward = _resolve(goal_reward, "goal_reward", stage)
        self.goal_penalty = _resolve(goal_penalty, "goal_penalty", stage)
        self.defense_weight = _resolve(defense_weight, "defense", stage)
        self.shot_placement_weight = _resolve(shot_placement_weight, "shot_placement", stage)
        self.entropy_weight = _resolve(entropy_weight, "entropy", stage)
        self.max_contacts_per_episode = max_contacts_per_episode

        self._prev_puck_y: float | None = None
        self._prev_puck_speed: float | None = None
        self._contact_count: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_puck_y = obs[1]  # puck_y
        puck_vx = info.get("puck_vx", obs[2])
        puck_vy = info.get("puck_vy", obs[3])
        self._prev_puck_speed = float(np.hypot(puck_vx, puck_vy))
        self._contact_count = 0
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0
        puck_x, puck_y = obs[0], obs[1]
        pad_x, pad_y = obs[4], obs[5]
        # Read velocities from obs (indices 2-3); info overrides if available
        puck_vx = info.get("puck_vx", obs[2])
        puck_vy = info.get("puck_vy", obs[3])
        dist = np.hypot(puck_x - pad_x, puck_y - pad_y)
        puck_speed = np.hypot(puck_vx, puck_vy)

        # Proximity
        if self.proximity_weight > 0:
            shaped_reward += self.proximity_weight * float(np.exp(-self.proximity_k * dist))

        # Contact + directed hit + shot placement
        if self._prev_puck_speed is not None and dist < 0.25:
            speed_change = puck_speed - self._prev_puck_speed
            if speed_change > 0.2 and puck_vy > 0:  # only reward forward hits
                self._contact_count += 1
                contact_ok = self._contact_count <= self.max_contacts_per_episode
                if self.contact_reward > 0 and contact_ok:
                    shaped_reward += self.contact_reward
                if self.directed_hit_weight > 0 and contact_ok:
                    shaped_reward += self.directed_hit_weight * puck_vy
                if self.shot_placement_weight > 0 and puck_vy > 0:
                    dx = _GOAL_CX - puck_x
                    dy = _GOAL_CY - puck_y
                    goal_dist = np.hypot(dx, dy)
                    alignment = (puck_vx * dx + puck_vy * dy) / (puck_speed * goal_dist + 1e-8)
                    shaped_reward += self.shot_placement_weight * float(np.clip(alignment, 0, 1))

        # Puck progress
        if self.puck_progress_weight > 0 and self._prev_puck_y is not None:
            delta = puck_y - self._prev_puck_y
            if delta > 0:
                shaped_reward += self.puck_progress_weight * delta

        # Defense
        if self.defense_weight > 0 and puck_vy < -0.3:
            x_alignment = float(np.exp(-3.0 * abs(puck_x - pad_x)))
            if pad_y < puck_y:
                shaped_reward += self.defense_weight * x_alignment

        # Goals
        if reward > 0 and self.goal_reward > 0:
            shaped_reward += self.goal_reward
        elif reward < 0 and self.goal_penalty != 0:
            shaped_reward += self.goal_penalty

        # Entropy bonus
        if self.entropy_weight > 0:
            shaped_reward += self.entropy_weight * (1.0 - float(np.mean(action ** 2)))

        # Update state
        self._prev_puck_y = puck_y
        self._prev_puck_speed = puck_speed

        # Reset potentials after goal
        if reward != 0:
            self._prev_puck_y = obs[1]  # puck_y
            self._contact_count = 0

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward

        self.env.unwrapped.record_reward(shaped_reward)

        return obs, shaped_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Vectorized batch reward shaper
# ---------------------------------------------------------------------------
class BatchRewardShaper:
    """Vectorized reward shaping on [N, obs_dim] arrays.

    Mirrors ShapedRewardWrapper logic for batch environments.
    Obs layout (14 dims): [puck_x, puck_y, puck_vx, puck_vy,
                           pad_x, pad_y, pad_vx, pad_vy,
                           opp_x, opp_y, opp_vx, opp_vy,
                           score_diff, time_remaining].
    Puck velocities read from obs (indices 2-3); info dict also accepted as fallback.
    """

    def __init__(
        self,
        n_envs: int,
        stage: int = STAGE_GAME_GOALIE,
        frame_stack: int = 1,  # kept for API compat, ignored
        proximity_weight: float | None = None,
        proximity_k: float = 3.0,
        contact_reward: float | None = None,
        directed_hit_weight: float | None = None,
        puck_progress_weight: float | None = None,
        goal_reward: float | None = None,
        goal_penalty: float | None = None,
        defense_weight: float | None = None,
        shot_placement_weight: float | None = None,
        entropy_weight: float | None = None,
        max_contacts_per_episode: int = 5,
    ):
        self.n_envs = n_envs
        self.stage = stage
        self.frame_stack = 1  # always 1 now
        self.proximity_k = proximity_k
        self.proximity_weight = _resolve(proximity_weight, "proximity", stage)
        self.contact_reward = _resolve(contact_reward, "contact", stage)
        self.directed_hit_weight = _resolve(directed_hit_weight, "directed_hit", stage)
        self.puck_progress_weight = _resolve(puck_progress_weight, "puck_progress", stage)
        self.goal_reward = _resolve(goal_reward, "goal_reward", stage)
        self.goal_penalty = _resolve(goal_penalty, "goal_penalty", stage)
        self.defense_weight = _resolve(defense_weight, "defense", stage)
        self.shot_placement_weight = _resolve(shot_placement_weight, "shot_placement", stage)
        self.entropy_weight = _resolve(entropy_weight, "entropy", stage)
        self.max_contacts_per_episode = max_contacts_per_episode

        self._prev_puck_y = np.zeros(n_envs)
        self._prev_puck_speed = np.zeros(n_envs)
        self._contact_count = np.zeros(n_envs, dtype=np.int32)
        self._anneal_decay = 0.0  # 0 = no decay, 1 = full decay
        self._penalty_ramp = 1.0  # 1 = full penalty by default; set_progress() ramps from 0

    def set_progress(self, progress: float) -> None:
        """Set stage progress (0.0 to 1.0) for reward annealing and penalty ramp.

        Penalty ramp: in the first 30% of the stage, goal_penalty linearly
        ramps from 0 to full. This lets the agent learn to score before
        learning to fear conceding (prevents avoidance phase).

        Reward annealing: in the last 40% (progress > 0.6), auxiliary weights
        are linearly decayed to 0. Goal rewards and entropy are NOT annealed.
        """
        # Penalty ramp: 0→1 over first 30% of stage
        self._penalty_ramp = min(1.0, progress / 0.3)

        # Auxiliary decay: 0→1 over last 40% of stage
        if progress <= 0.6:
            self._anneal_decay = 0.0
        else:
            self._anneal_decay = min(1.0, (progress - 0.6) / 0.4)

    def reset(
        self,
        obs: np.ndarray,
        mask: np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        """Initialize state from observations [N, obs_dim].

        info: optional dict with 'puck_vx', 'puck_vy' arrays [N].
        """
        if mask is None:
            idx = slice(None)
        else:
            idx = mask
        self._prev_puck_y[idx] = obs[idx, 1]  # puck_y
        self._contact_count[idx] = 0
        # Read puck velocity from obs (indices 2-3) or info dict
        puck_vx = obs[idx, 2]
        puck_vy = obs[idx, 3]
        if info is not None and "puck_vx" in info:
            puck_vx = info["puck_vx"][idx]
            puck_vy = info["puck_vy"][idx]
        self._prev_puck_speed[idx] = np.hypot(puck_vx, puck_vy)

    def compute(
        self,
        obs: np.ndarray,
        raw_rewards: np.ndarray,
        actions: np.ndarray | None = None,
        info: dict | None = None,
    ) -> np.ndarray:
        """Compute shaped rewards from [N, obs_dim] obs and [N] raw rewards.

        raw_rewards: +1 agent goal, -1 opponent goal, 0 otherwise.
        actions: optional [N, 2] for entropy bonus.
        info: optional dict with 'puck_vx', 'puck_vy' arrays [N].
        Returns: [N] shaped rewards.
        """
        shaped = np.zeros(self.n_envs, dtype=np.float32)
        aux_scale = 1.0 - self._anneal_decay  # annealing multiplier

        puck_x, puck_y = obs[:, 0], obs[:, 1]
        pad_x, pad_y = obs[:, 4], obs[:, 5]
        # Read puck velocity from obs (indices 2-3); info dict overrides if available
        puck_vx, puck_vy = obs[:, 2], obs[:, 3]
        if info is not None and "puck_vx" in info:
            puck_vx, puck_vy = info["puck_vx"], info["puck_vy"]

        dist = np.hypot(puck_x - pad_x, puck_y - pad_y)
        puck_speed = np.hypot(puck_vx, puck_vy)

        # Proximity
        if self.proximity_weight > 0:
            shaped += aux_scale * self.proximity_weight * np.exp(-self.proximity_k * dist)

        # Contact + directed hit + shot placement (only reward forward hits)
        speed_change = puck_speed - self._prev_puck_speed
        hit = (dist < 0.25) & (speed_change > 0.2) & (puck_vy > 0)  # forward hits only

        # Track contacts and enforce per-episode cap
        self._contact_count += hit.astype(np.int32)
        contact_ok = self._contact_count <= self.max_contacts_per_episode

        if self.contact_reward > 0:
            shaped += np.where(hit & contact_ok, aux_scale * self.contact_reward, 0.0)

        if self.directed_hit_weight > 0:
            shaped += np.where(
                hit & contact_ok,
                aux_scale * self.directed_hit_weight * puck_vy,
                0.0,
            )

        if self.shot_placement_weight > 0:
            dx = _GOAL_CX - puck_x
            dy = _GOAL_CY - puck_y
            goal_dist = np.hypot(dx, dy)
            alignment = (puck_vx * dx + puck_vy * dy) / (puck_speed * goal_dist + 1e-8)
            alignment = np.clip(alignment, 0, 1)
            shaped += np.where(
                hit & (puck_vy > 0),
                aux_scale * self.shot_placement_weight * alignment,
                0.0,
            )

        # Puck progress (one-way, only positive delta)
        if self.puck_progress_weight > 0:
            delta_y = puck_y - self._prev_puck_y
            shaped += np.where(delta_y > 0, aux_scale * self.puck_progress_weight * delta_y, 0.0)

        # Defense
        if self.defense_weight > 0:
            approaching = puck_vy < -0.3
            between = pad_y < puck_y
            x_align = np.exp(-3.0 * np.abs(puck_x - pad_x))
            shaped += aux_scale * self.defense_weight * approaching * between * x_align

        # Goals (NOT annealed)
        if self.goal_reward > 0:
            shaped += np.where(raw_rewards > 0, self.goal_reward, 0.0)
        if self.goal_penalty != 0:
            ramped_penalty = self.goal_penalty * self._penalty_ramp
            shaped += np.where(raw_rewards < 0, ramped_penalty, 0.0)

        # Entropy bonus (NOT annealed)
        if self.entropy_weight > 0 and actions is not None:
            shaped += self.entropy_weight * (1.0 - np.mean(actions ** 2, axis=1))

        # Update state
        self._prev_puck_y[:] = puck_y
        self._prev_puck_speed[:] = puck_speed

        # Reset potentials and contact count after goals
        goal_mask = raw_rewards != 0
        if np.any(goal_mask):
            self._prev_puck_y[goal_mask] = obs[goal_mask, 1]  # puck_y
            self._contact_count[goal_mask] = 0

        return shaped.astype(np.float32)
