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
    1: {"proximity": 0,    "contact": 3.0, "directed_hit": 2.0, "puck_progress": 0,   "defense": 0,   "shot_placement": 0,   "goal_reward": 0,     "goal_penalty": 0,     "entropy": 0, "shot_mix": 0.5},
    2: {"proximity": 0,    "contact": 0.1, "directed_hit": 0.1, "puck_progress": 0.1, "defense": 0.1, "shot_placement": 0.2, "goal_reward": 160.0, "goal_penalty": -20.0, "entropy": 0, "shot_mix": 0.5},
    3: {"proximity": 0,    "contact": 0,   "directed_hit": 0,   "puck_progress": 0.1, "defense": 0.1, "shot_placement": 0.2, "goal_reward": 160.0, "goal_penalty": -20.0, "entropy": 0, "shot_mix": 0.5},
    4: {"proximity": 0,    "contact": 0,   "directed_hit": 0,   "puck_progress": 0,   "defense": 0,   "shot_placement": 0,   "goal_reward": 130.0, "goal_penalty": -20.0, "entropy": 0, "shot_mix": 0.5},
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

# Steps are at the 100 Hz action rate. These were written for 60 Hz
# ("600 steps ~ 10 s") and never rescaled when action_dt moved to 1/100 --
# which silently cut every episode to 60% of its intended wall time and made
# each goals-PER-EPISODE gate ~1.7x harder. The first full curriculum run
# spent 840k of its 1M steps held at stage 2 by that inflated bar, while its
# recorded games were winning 5-0.
STAGE_EPISODE_STEPS: dict[int, int] = {
    1: 1000,   # 10 s — chase + hit
    2: 2000,   # 20 s — game vs goalie
    3: 3000,   # 30 s — game vs follower
    4: 3000,   # 30 s — self-play
}

# Default table geometry for shot placement
_GOAL_CX = 0.5   # table_width / 2
_GOAL_CY = 2.0   # table_height
_TABLE_W = 1.0


def _is_bank_shot(px, py, vx, vy):
    """Will this shot touch a side rail before reaching the far goal line?

    Pure geometry on the outgoing velocity: extend the line from (px, py)
    along (vx, vy) to y = table height; if x leaves the table on the way,
    the shot banks. Works elementwise on arrays and on scalars alike.
    """
    t_goal = (_GOAL_CY - py) / np.maximum(vy, 1e-8)
    x_at_goal = px + vx * t_goal
    return (x_at_goal < 0.0) | (x_at_goal > _TABLE_W)


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
        shot_mix_weight: float | None = None,
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
        self.shot_mix_weight = _resolve(shot_mix_weight, "shot_mix", stage)
        self.max_contacts_per_episode = max_contacts_per_episode

        # Running bank-shot fraction, for the shot-mix bonus. Starts at the
        # 50/50 target so the very first shot of a run is not biased either
        # way. Survives episode resets on purpose: the mix worth balancing is
        # the policy's repertoire, not one episode's.
        self._bank_ema: float = 0.5

        self._prev_puck_y: float | None = None
        self._prev_puck_speed: float | None = None
        self._contact_count: int = 0
        self._prev_score_agent: int = 0
        self._prev_score_opp: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_puck_y = obs[1]  # puck_y
        puck_vx = info.get("puck_vx", obs[2])
        puck_vy = info.get("puck_vy", obs[3])
        self._prev_puck_speed = float(np.hypot(puck_vx, puck_vy))
        self._contact_count = 0
        # Baseline from info, not zero: score handicap starts games at 0-3.
        self._prev_score_agent = info.get("score_agent", 0)
        self._prev_score_opp = info.get("score_opponent", 0)
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
                # Shot-mix: nudge toward a repertoire of BOTH bank and
                # straight shots. Each shot earns weight * (fraction of
                # recent shots that were the OTHER kind), so whichever type
                # the policy neglects pays better, with a 50/50 equilibrium.
                # The weight is deliberately small next to a 160-point goal:
                # this is a tiebreaker between near-equal shots, not a
                # reason to bank from in front of an open net.
                if self.shot_mix_weight > 0 and contact_ok:
                    bank = bool(_is_bank_shot(puck_x, puck_y, puck_vx, puck_vy))
                    other_frac = (1.0 - self._bank_ema) if bank else self._bank_ema
                    shaped_reward += self.shot_mix_weight * other_frac
                    self._bank_ema += 0.2 * (float(bank) - self._bank_ema)

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

        # Goals, detected from the SCOREBOARD, not the sign of the base
        # reward. The sign check was a contract ("raw reward != 0 means a
        # goal happened") that the env quietly stopped honouring twice: the
        # stuck-puck penalty (-0.5) has always read as a phantom conceded
        # goal, and the workspace-overshoot penalty made every out-of-reach
        # command bill -20 -- a random policy earned -36k a game while
        # actually up 2-0.
        goal_for = info["score_agent"] > self._prev_score_agent
        goal_against = info["score_opponent"] > self._prev_score_opp
        self._prev_score_agent = info["score_agent"]
        self._prev_score_opp = info["score_opponent"]
        if goal_for and self.goal_reward > 0:
            shaped_reward += self.goal_reward
        elif goal_against and self.goal_penalty != 0:
            shaped_reward += self.goal_penalty

        # Entropy bonus
        if self.entropy_weight > 0:
            shaped_reward += self.entropy_weight * (1.0 - float(np.mean(action ** 2)))

        # Update state
        self._prev_puck_y = puck_y
        self._prev_puck_speed = puck_speed

        # Reset potentials after goal
        if goal_for or goal_against:
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
    Obs layout (12 dims): [puck_x, puck_y, puck_vx, puck_vy,
                           pad_x, pad_y, pad_vx, pad_vy,
                           opp_x, opp_y, opp_vx, opp_vy].
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
        shot_mix_weight: float | None = None,
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
        self.shot_mix_weight = _resolve(shot_mix_weight, "shot_mix", stage)
        self.max_contacts_per_episode = max_contacts_per_episode

        self._prev_puck_y = np.zeros(n_envs)
        self._prev_puck_speed = np.zeros(n_envs)
        self._contact_count = np.zeros(n_envs, dtype=np.int32)
        # Per-env running bank-shot fraction for the shot-mix bonus; starts
        # at the 50/50 target and deliberately survives episode resets --
        # the mix being balanced is the policy's repertoire, not one
        # episode's. See ShapedRewardWrapper for the scalar twin.
        self._bank_ema = np.full(n_envs, 0.5)
        # Scoreboard as of the previous step, for goal detection by delta.
        # A DROP in score (episode reset, or a handicap re-deal) is not a
        # goal; only an increase is. No reset hook needed: after any reset
        # the delta is <= 0 and the baseline self-corrects in one step.
        self._prev_score_agent = np.zeros(n_envs, dtype=np.int64)
        self._prev_score_opp = np.zeros(n_envs, dtype=np.int64)
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
        if info is not None and "puck_y" in info:
            self._prev_puck_y[idx] = info["puck_y"][idx]
        else:
            self._prev_puck_y[idx] = obs[idx, 1]  # puck_y
        self._contact_count[idx] = 0
        # Read puck velocity from info (truth) or obs (indices 2-3)
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

        # TRUE state from info when the env provides it: rewards score what
        # happened on the table, not what a noisy tracker believed -- and
        # history-mode observations do not carry a snapshot at these indices
        # at all. Obs-index fallback serves bare-bones callers only.
        if info is not None and "puck_x" in info:
            puck_x, puck_y = info["puck_x"], info["puck_y"]
            pad_x, pad_y = info["pad_x"], info["pad_y"]
        else:
            puck_x, puck_y = obs[:, 0], obs[:, 1]
            pad_x, pad_y = obs[:, 4], obs[:, 5]
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

        # Shot-mix: each shot earns weight * (recent fraction of the OTHER
        # kind), so the neglected type pays better; 50/50 equilibrium. Small
        # by design ("mostly EV maximise"): a tiebreaker between near-equal
        # shots, never a reason to bank from in front of an open net.
        if self.shot_mix_weight > 0:
            shooting = hit & contact_ok
            if np.any(shooting):
                bank = _is_bank_shot(puck_x, puck_y, puck_vx, puck_vy)
                other_frac = np.where(bank, 1.0 - self._bank_ema, self._bank_ema)
                shaped += np.where(shooting, self.shot_mix_weight * other_frac, 0.0)
                upd = shooting
                self._bank_ema[upd] += 0.2 * (bank[upd].astype(float) - self._bank_ema[upd])

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

        # Goals (NOT annealed), detected from the SCOREBOARD rather than the
        # sign of the raw reward. The sign check assumed "raw reward != 0
        # means a goal" -- but the stuck-puck penalty (-0.5) has always read
        # as a phantom conceded goal, and the workspace-overshoot penalty
        # turned every out-of-reach command into a -20. See
        # ShapedRewardWrapper for the scalar twin of this fix.
        if info is not None and "score_agent" in info:
            goal_for = info["score_agent"] > self._prev_score_agent
            goal_against = info["score_opponent"] > self._prev_score_opp
            self._prev_score_agent[:] = info["score_agent"]
            self._prev_score_opp[:] = info["score_opponent"]
        else:
            # No scoreboard offered (bare-bones callers): the sign is all
            # there is. Only trustworthy if the env's raw reward carries
            # nothing but goals.
            goal_for = raw_rewards > 0
            goal_against = raw_rewards < 0
        if self.goal_reward > 0:
            shaped += np.where(goal_for, self.goal_reward, 0.0)
        if self.goal_penalty != 0:
            ramped_penalty = self.goal_penalty * self._penalty_ramp
            shaped += np.where(goal_against, ramped_penalty, 0.0)

        # Entropy bonus (NOT annealed)
        if self.entropy_weight > 0 and actions is not None:
            shaped += self.entropy_weight * (1.0 - np.mean(actions ** 2, axis=1))

        # Update state
        self._prev_puck_y[:] = puck_y
        self._prev_puck_speed[:] = puck_speed

        # Reset potentials and contact count after goals
        goal_mask = goal_for | goal_against
        if np.any(goal_mask):
            self._prev_puck_y[goal_mask] = puck_y[goal_mask]
            self._contact_count[goal_mask] = 0

        return shaped.astype(np.float32)
