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

from airhockey.dynamics import ACTION_DT

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

# Episode lengths in SECONDS, converted to steps at the control rate. They
# were once steps written for 60 Hz ("600 steps ~ 10 s") and never rescaled
# when action_dt moved to 1/100 -- which silently cut every episode to 60%
# of its intended wall time and made each goals-PER-EPISODE gate ~1.7x
# harder. The first full curriculum run spent 840k of its 1M steps held at
# stage 2 by that inflated bar, while its recorded games were winning 5-0.
# Seconds cannot drift with the rate.
STAGE_EPISODE_S: dict[int, float] = {
    1: 10.0,   # chase + hit
    2: 20.0,   # game vs goalie
    3: 30.0,   # game vs follower
    4: 30.0,   # self-play
}


def steps_for(seconds: float, action_dt: float = ACTION_DT) -> int:
    """Episode seconds -> control steps at the (one) control rate."""
    return int(round(seconds / action_dt))


STAGE_EPISODE_STEPS: dict[int, int] = {k: steps_for(v) for k, v in STAGE_EPISODE_S.items()}

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


# ---------------------------------------------------------------------------
# The shot predictor: where a puck's free path crosses the far goal line
# ---------------------------------------------------------------------------
# One definition of "on target", shared by every term that scores a shot
# (the on-target reward, the shot-type reward, the SAC exchange shaper). The
# path is traced through the MEASURED lossy-wall model -- normal 0.785,
# tangential 0.66, the coefficients the heuristic bots aim with -- so a bank
# is scored where it actually lands rather than where a mirror says. The
# opponent is deliberately ignored: the question is whether it WOULD go in
# with nobody there, which is what makes it a statement about the shot.
# Friction is ignored too; a slow on-target shot is scored optimistically,
# and the speed term is what pays for making it fast.
SHOT_MAX_BOUNCES = 4
SHOT_RAIL_LEFT = -1      # the x = 0 rail, LEFT for a robot facing the far goal
SHOT_RAIL_RIGHT = 1      # the x = width rail
SHOT_RAIL_NONE = 0

# Shot types the policy can be asked for (observation one-hot, see
# BatchAirHockeyEnv): 0 = no preference (all zeros), then the three.
SHOT_TYPE_NONE, SHOT_TYPE_LEFT, SHOT_TYPE_RIGHT, SHOT_TYPE_STRAIGHT = 0, 1, 2, 3
SHOT_TYPE_NAMES = ("none", "left", "right", "straight")

# Possession and control (see BatchRewardShaper): a puck is TRAPPED when,
# on the agent's half, it is brought under TRAP_SPEED within TRAP_DIST of
# the paddle after having moved faster than TRAP_MIN_ARRIVAL during this
# visit -- a puck that drifted to rest on its own is not a trap.
TRAP_SPEED = 0.3
TRAP_DIST = 0.20
TRAP_MIN_ARRIVAL = 0.8


def predict_shot(x, y, vx, vy, width: float = _TABLE_W, height: float = _GOAL_CY,
                 puck_radius: float = 0.0407, e_n: float = 0.785, e_t: float = 0.66,
                 max_bounces: int = SHOT_MAX_BOUNCES):
    """Trace a puck's free path to the far goal line.

    Elementwise over arrays. Returns (x_goal, t_goal, n_bounces, first_rail):
    x_goal is nan where the path never reaches the line (moving away, or
    dead after `max_bounces` rails), t_goal the time to get there,
    n_bounces the side rails touched on the way and first_rail which one
    was touched first (SHOT_RAIL_LEFT / RIGHT / NONE).
    """
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    vx = np.array(vx, dtype=float, copy=True)
    vy = np.array(vy, dtype=float, copy=True)
    t_total = np.zeros_like(x)
    x_goal = np.full_like(x, np.nan)
    n_b = np.zeros(x.shape, dtype=np.int32)
    first = np.zeros(x.shape, dtype=np.int32)
    active = vy > 1e-6
    lo, hi = puck_radius, width - puck_radius
    for _ in range(max_bounces + 1):
        if not active.any():
            break
        with np.errstate(invalid="ignore"):
            t_g = np.where(active, (height - y) / np.maximum(vy, 1e-9), np.inf)
            x_lin = x + vx * t_g
        direct = active & (x_lin >= lo) & (x_lin <= hi)
        x_goal = np.where(direct, x_lin, x_goal)
        t_total = np.where(direct, t_total + t_g, t_total)
        active &= ~direct
        with np.errstate(divide="ignore", invalid="ignore"):
            t_w = np.where(vx > 1e-9, (hi - x) / vx,
                           np.where(vx < -1e-9, (lo - x) / vx, np.inf))
        t_w = np.clip(t_w, 0.0, None)
        bounced = active & np.isfinite(t_w)
        step = np.where(bounced, t_w, 0.0)
        x = x + vx * step
        y = y + vy * step
        t_total = t_total + step
        rail = np.where(vx > 0, SHOT_RAIL_RIGHT, SHOT_RAIL_LEFT)
        first = np.where(bounced & (n_b == 0), rail, first)
        n_b = n_b + bounced.astype(np.int32)
        vx = np.where(bounced, -vx * e_n, vx)
        vy = np.where(bounced, vy * e_t, vy)
        active &= bounced
    # A path still bouncing after the cap never arrives.
    x_goal = np.where(active, np.nan, x_goal)
    return x_goal, t_total, n_b, first


def shot_matches_type(shot_type, n_bounces, first_rail):
    """Elementwise: did a shot with this rail history satisfy the request?

    straight = touched no rail; left / right bank = the FIRST rail touched
    is that side. No preference never matches (nothing to pay for).
    """
    shot_type = np.asarray(shot_type)
    return ((shot_type == SHOT_TYPE_STRAIGHT) & (n_bounces == 0)) | \
           ((shot_type == SHOT_TYPE_LEFT) & (first_rail == SHOT_RAIL_LEFT)) | \
           ((shot_type == SHOT_TYPE_RIGHT) & (first_rail == SHOT_RAIL_RIGHT))


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

        # Proximity, to the closest REACHABLE point of the puck (see
        # BatchRewardShaper.workspace)
        if self.proximity_weight > 0:
            ws = getattr(self.env.unwrapped, "_ws", None)
            if ws is not None:
                rx = min(max(puck_x, ws["min_x"]), ws["max_x"])
                ry = min(max(puck_y, ws["min_y"]), ws["max_y"])
                reach_dist = float(np.hypot(rx - pad_x, ry - pad_y))
            else:
                reach_dist = dist
            shaped_reward += self.proximity_weight * float(np.exp(-self.proximity_k * reach_dist))

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
        workspace: dict | None = None,
        # Idle hygiene for the physical machine, paid ONLY while the puck is
        # on the far half so play is untouched: a pull toward being centred
        # in front of the goal (paddle x at the goal's centre; depth is left
        # to the policy), and a tax on step-to-step action changes, because
        # a target that dithers by a centimetre at 100 Hz is a stepper
        # reversing direction 100 times a second for nothing. Per-step
        # magnitudes are hundredths against goals at 100 and contact at 2,
        # so they only decide what the paddle does when nothing else does.
        home_weight: float = 0.0,
        jitter_weight: float = 0.0,
        # Smoothness DURING PLAY, ungated: a tax on step-to-step action
        # change everywhere. The idle jitter term above only bites with the
        # puck away; on the table the policy flipped its target corner to
        # corner on a quarter of all ticks, puck near or far, and a
        # 60 m/s^2 body followed it (2026-09-02). A strike is one large
        # change and stays cheap against contact (2) and a goal (100); a
        # policy that flips every few ticks pays every time.
        smooth_weight: float = 0.0,
        # Shot OUTCOMES (2026-09-06, ai/RETRAIN.md items 2-4). Paid at the
        # hit, once per possession (a visit of the puck to the agent's half):
        #   on_target_reward   the shot's free path (predict_shot) crosses
        #                      the far goal line inside the mouth
        #   shot_speed_weight  x the puck's speed, on on-target shots only
        #   trap_reward        the first time this visit the puck is brought
        #                      to rest under the paddle after arriving fast
        #   controlled_shot_bonus  multiplies on_target_reward when the
        #                      possession included a trap (1.0 = off)
        #   shot_type_reward   the shot's rail history matches the type the
        #                      env asked for (info["shot_type"])
        on_target_reward: float = 0.0,
        shot_speed_weight: float = 0.0,
        trap_reward: float = 0.0,
        controlled_shot_bonus: float = 1.0,
        shot_type_reward: float = 0.0,
        # Retrain run 2 (2026-09-06). accel_cost_weight: per step, times the
        # accel fraction the action asked for (action_mode "profile_a"), so
        # a high cap is spent on strikes and saves and not on wandering --
        # heat is torque and torque is accel, and the first run tripped a
        # drive after 20 s of continuous full-box traversal. patience_s /
        # patience_floor: the hit rewards (on-target, speed, directed, shot
        # type) are multiplied by floor + (1 - floor) * min(1, t_side /
        # patience_s), t_side being how long the puck has been on the
        # robot's side (info["t_side"]), so an instant slap pays the floor
        # and a controlled shot the whole. Sized against the discount: at
        # 50 Hz and 0.995, 1.5 s of waiting keeps 69% of a reward, and 2x
        # (floor 0.5) is what makes waiting win.
        accel_cost_weight: float = 0.0,
        patience_s: float = 0.0,
        patience_floor: float = 1.0,
        # Run 3: the GOAL a shot produces is scaled by the patience of the
        # hit that produced it. Run 2 scaled only the hit rewards (~10) and
        # left the goal (100) whole, so an instant slap that scored still
        # paid full price and the policy learned to shoot sooner, not
        # later (patience multiplier 0.76 -> 0.62 over the run).
        patience_on_goals: bool = False,
        config=None,
    ):
        self.n_envs = n_envs
        self.stage = stage
        self.accel_cost_weight = accel_cost_weight
        self.patience_s = patience_s
        self.patience_floor = patience_floor
        self.patience_on_goals = patience_on_goals
        # The patience multiplier at the agent's last hit of the current
        # possession; 1.0 when it has not hit (a relaunch, an own goal).
        self._last_hit_patience = np.ones(n_envs)
        from airhockey.physics import TableConfig    # noqa: PLC0415
        cfg = config or TableConfig()
        self._cfg = cfg
        self._H = cfg.height
        self.on_target_reward = on_target_reward
        self.shot_speed_weight = shot_speed_weight
        self.trap_reward = trap_reward
        self.controlled_shot_bonus = controlled_shot_bonus
        self.shot_type_reward = shot_type_reward
        # Possession state: is the puck on the agent's half, has it been
        # trapped this visit, has an on-target shot been paid this visit,
        # and the fastest it moved this visit (the trap's arrival test).
        self._in_half = np.zeros(n_envs, dtype=bool)
        self._trapped = np.zeros(n_envs, dtype=bool)
        self._shot_paid = np.zeros(n_envs, dtype=bool)
        self._visit_max_speed = np.zeros(n_envs)
        # Counters for the trainer's logs: traps, on-target shots and
        # type matches since the last read.
        self.stats = {"traps": 0, "on_target": 0, "type_matched": 0, "shots": 0,
                      "patience_sum": 0.0, "accel_frac_sum": 0.0, "steps": 0,
                      "goal_patience_sum": 0.0, "goals": 0}
        # The reachable box (env._ws), for the proximity term. Measured
        # distance to the puck is dominated by where the PUCK is: it spends
        # most of a game outside the robot's box, and a policy that drives
        # straight at it scores the same as a random one (0.55 m mean
        # distance either way). Distance to the puck's closest REACHABLE
        # point is what the paddle actually controls.
        self.workspace = workspace
        self.home_weight = home_weight
        self.jitter_weight = jitter_weight
        self.smooth_weight = smooth_weight
        self._home_x = _GOAL_CX                  # centred in front of the goal
        if workspace is not None:
            self._home_span = 0.5 * (workspace["max_x"] - workspace["min_x"])
        else:
            self._home_span = _TABLE_W / 2.0
        self._prev_action = np.zeros((n_envs, 2))
        self._has_prev_action = np.zeros(n_envs, dtype=bool)
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
        self._has_prev_action[idx] = False
        py = info["puck_y"][idx] if (info is not None and "puck_y" in info) else obs[idx, 1]
        self._in_half[idx] = py < self._H / 2.0
        self._trapped[idx] = False
        self._shot_paid[idx] = False
        self._visit_max_speed[idx] = 0.0
        self._last_hit_patience[idx] = 1.0
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

        # Proximity, to the closest REACHABLE point of the puck
        if self.proximity_weight > 0:
            if self.workspace is not None:
                ws = self.workspace
                rx = np.clip(puck_x, ws["min_x"], ws["max_x"])
                ry = np.clip(puck_y, ws["min_y"], ws["max_y"])
                reach_dist = np.hypot(rx - pad_x, ry - pad_y)
            else:
                reach_dist = dist
            shaped += aux_scale * self.proximity_weight * np.exp(-self.proximity_k * reach_dist)

        # Patience: how the hit rewards scale with the time the puck has
        # been on the robot's side. 1.0 everywhere when the term is off.
        if self.patience_s > 0 and info is not None and "t_side" in info:
            patience = self.patience_floor + (1.0 - self.patience_floor) * np.minimum(
                1.0, info["t_side"] / self.patience_s)
        else:
            patience = np.ones(self.n_envs)

        # Contact + directed hit + shot placement (only reward forward hits)
        speed_change = puck_speed - self._prev_puck_speed
        hit = (dist < 0.25) & (speed_change > 0.2) & (puck_vy > 0)  # forward hits only

        # Remember how patient each hit was, for the goal it may produce.
        if np.any(hit):
            self._last_hit_patience = np.where(hit, patience, self._last_hit_patience)

        # Track contacts and enforce per-episode cap
        self._contact_count += hit.astype(np.int32)
        contact_ok = self._contact_count <= self.max_contacts_per_episode

        if self.contact_reward > 0:
            shaped += np.where(hit & contact_ok, aux_scale * self.contact_reward, 0.0)

        if self.directed_hit_weight > 0:
            shaped += np.where(
                hit & contact_ok,
                aux_scale * self.directed_hit_weight * puck_vy * patience,
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

        # Possession: a visit of the puck to the agent's half. Entering
        # starts a fresh visit (nothing trapped, nothing paid); leaving --
        # which is what a shot does -- ends it.
        in_half = puck_y < self._H / 2.0
        entered = in_half & ~self._in_half
        if np.any(entered):
            self._trapped[entered] = False
            self._shot_paid[entered] = False
            self._visit_max_speed[entered] = 0.0
            self._last_hit_patience[entered] = 1.0
        self._in_half[:] = in_half
        self._visit_max_speed = np.where(in_half, np.maximum(self._visit_max_speed, puck_speed),
                                         self._visit_max_speed)

        # Trap: the puck brought to rest under the paddle after arriving
        # fast. Once per visit, so bump-stop-bump earns nothing extra.
        if self.trap_reward > 0:
            trap = (in_half & ~self._trapped & (puck_speed < TRAP_SPEED)
                    & (dist < TRAP_DIST) & (self._visit_max_speed > TRAP_MIN_ARRIVAL))
            if np.any(trap):
                shaped += np.where(trap, self.trap_reward, 0.0)
                self._trapped |= trap
                self.stats["traps"] += int(trap.sum())

        # Shot outcome, scored at the hit from the puck's outgoing velocity
        # and paid once per visit. A hit that was already paid this visit
        # (dribbling) earns nothing more until the puck leaves and returns.
        if self.on_target_reward > 0 or self.shot_type_reward > 0:
            scoring = hit & ~self._shot_paid
            if np.any(scoring):
                x_goal, _t, n_b, first = predict_shot(
                    puck_x, puck_y, puck_vx, puck_vy, self._cfg.width, self._H,
                    self._cfg.puck_radius, self._cfg.wall_restitution,
                    self._cfg.wall_tangential)
                on_target = scoring & np.isfinite(x_goal) & \
                    (np.abs(x_goal - self._cfg.width / 2.0) < self._cfg.goal_width / 2.0)
                self.stats["shots"] += int(scoring.sum())
                if np.any(on_target):
                    mult = np.where(self._trapped, self.controlled_shot_bonus, 1.0) * patience
                    shaped += np.where(on_target,
                                       (self.on_target_reward
                                        + self.shot_speed_weight * puck_speed) * mult, 0.0)
                    self.stats["on_target"] += int(on_target.sum())
                    self.stats["patience_sum"] += float(patience[on_target].sum())
                    if self.shot_type_reward > 0 and info is not None and "shot_type" in info:
                        matched = on_target & shot_matches_type(info["shot_type"], n_b, first)
                        shaped += np.where(matched, self.shot_type_reward * patience, 0.0)
                        self.stats["type_matched"] += int(matched.sum())
                    self._shot_paid |= on_target

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
            goal_scale = self._last_hit_patience if self.patience_on_goals else 1.0
            shaped += np.where(goal_for, self.goal_reward * goal_scale, 0.0)
            if self.patience_on_goals and np.any(goal_for):
                self.stats["goal_patience_sum"] += float(self._last_hit_patience[goal_for].sum())
                self.stats["goals"] += int(goal_for.sum())
        if self.goal_penalty != 0:
            ramped_penalty = self.goal_penalty * self._penalty_ramp
            shaped += np.where(goal_against, ramped_penalty, 0.0)

        # Entropy bonus (NOT annealed)
        if self.entropy_weight > 0 and actions is not None:
            shaped += self.entropy_weight * (1.0 - np.mean(actions ** 2, axis=1))

        # Env-side penalties (workspace overshoot fine, stuck puck). They are
        # in raw_rewards too, but raw is otherwise goals-only and superseded
        # by goal_reward/goal_penalty above, so they must be carried here.
        if info is not None and "penalty" in info:
            shaped += info["penalty"]

        # The accel the action asked for, taxed per step (profile_a: the
        # third action slot; absent in the 2-dim mode = nothing to tax).
        if self.accel_cost_weight > 0 and actions is not None:
            a = np.asarray(actions)
            if a.shape[1] >= 3:
                frac = 0.05 + (np.clip(a[:, 2], -1.0, 1.0) + 1.0) * 0.5 * 0.95
                shaped -= self.accel_cost_weight * frac
                self.stats["accel_frac_sum"] += float(frac.sum())
        self.stats["steps"] += self.n_envs

        # Idle hygiene (gated on the puck being on the far half) and
        # smoothness during play (ungated).
        if self.home_weight > 0 or self.jitter_weight > 0 or self.smooth_weight > 0:
            idle = puck_y > _GOAL_CY / 2.0
            if self.home_weight > 0:
                off_home = np.abs(pad_x - self._home_x) / self._home_span
                shaped -= np.where(idle, self.home_weight * off_home, 0.0)
            if (self.jitter_weight > 0 or self.smooth_weight > 0) and actions is not None:
                a = np.asarray(actions)[:, :2]
                delta = np.linalg.norm(a - self._prev_action, axis=1)
                rate = self.smooth_weight + np.where(idle, self.jitter_weight, 0.0)
                shaped -= np.where(self._has_prev_action, rate * delta, 0.0)
                self._prev_action[:] = a
                self._has_prev_action[:] = True

        # Update state
        self._prev_puck_y[:] = puck_y
        self._prev_puck_speed[:] = puck_speed

        # Reset potentials and contact count after goals
        goal_mask = goal_for | goal_against
        if np.any(goal_mask):
            self._prev_puck_y[goal_mask] = puck_y[goal_mask]
            self._contact_count[goal_mask] = 0
            self._in_half[goal_mask] = puck_y[goal_mask] < self._H / 2.0
            self._trapped[goal_mask] = False
            self._shot_paid[goal_mask] = False
            self._visit_max_speed[goal_mask] = 0.0
            self._last_hit_patience[goal_mask] = 1.0

        return shaped.astype(np.float32)


# ---------------------------------------------------------------------------
# Exchange-based rewards (outcomes, not states)
# ---------------------------------------------------------------------------
class ExchangeRewardShaper:
    """Pay for SHOT OUTCOMES, not for standing anywhere.

    The dense shapers above pay continuous income for states -- proximity,
    goal-side alignment, puck progress. Under those, sac_v8's converged
    policy was rational: park goal-side and collect, since a goal needs a
    precise rarely-explored manoeuvre while alignment pays every step.
    Reward decline with a healthy optimizer was the tell.

    This shaper is the Air-Hockey-Sim scheme adapted to our tables: nothing
    pays until the puck crosses the MIDLINE moving away (a shot). At that
    instant the shot is scored once, by trajectory:

      on_target   its path (through the MEASURED lossy-wall model) crosses
                  the opponent goal line inside the mouth, opponent ignored
      beats_opp   additionally, the opponent cannot reach the crossing point
                  in time (crude reach model -- shaping, not physics)
      vel bonus   proportional to shot speed: hard shots beat reaction time,
                  and this is the term that finally pays for STRIKING

    The EXCHANGE then ends -- on the goal, or when the puck comes back over
    three-quarter table (blocked/returned), or on a timeout. The trainer
    truncates those envs, so each shot is its own tight credit-assignment
    unit instead of one event lost in a 20 s episode. Goals and concessions
    keep flat terminal values; a small capped contact reward remains as the
    only bootstrap (a policy that never touches the puck cannot discover
    shooting).

    compute() returns the reward array and sets `self.end_exchange` [N] for
    the trainer to fold into truncation.
    """

    def __init__(self, n_envs: int, config=None,
                 shot_on_target: float = 11.5,
                 shot_beats_opp: float = 2.5,
                 vel_bonus_per_ms: float = 0.5,
                 goal_reward: float = 30.0,
                 goal_penalty: float = -10.0,
                 contact_reward: float = 0.5,
                 max_contacts_per_episode: int = 5,
                 exchange_timeout_steps: int = 200,
                 opp_reach_speed: float = 3.0):
        from airhockey.physics import TableConfig
        cfg = config or TableConfig()
        self.n_envs = n_envs
        self.W = cfg.width
        self.H = cfg.height
        self.r = cfg.puck_radius
        self.goal_half = cfg.goal_width / 2.0
        self.e_n = cfg.wall_restitution
        self.e_t = cfg.wall_tangential
        self.shot_on_target = shot_on_target
        self.shot_beats_opp = shot_beats_opp
        self.vel_bonus_per_ms = vel_bonus_per_ms
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.contact_reward = contact_reward
        self.max_contacts = max_contacts_per_episode
        self.timeout = exchange_timeout_steps
        self.opp_reach_speed = opp_reach_speed

        self._prev_puck_y = np.full(n_envs, cfg.height / 4)
        self._prev_puck_speed = np.zeros(n_envs)
        self._contact_count = np.zeros(n_envs, dtype=np.int32)
        self._shot_in_flight = np.zeros(n_envs, dtype=bool)
        self._shot_age = np.zeros(n_envs, dtype=np.int32)
        self._prev_score_agent = np.zeros(n_envs, dtype=np.int64)
        self._prev_score_opp = np.zeros(n_envs, dtype=np.int64)
        self.end_exchange = np.zeros(n_envs, dtype=bool)

    # API compat with BatchRewardShaper
    def set_progress(self, progress: float) -> None:
        pass

    def reset(self, obs, mask=None, info=None) -> None:
        idx = slice(None) if mask is None else mask
        if info is not None and "puck_y" in info:
            self._prev_puck_y[idx] = info["puck_y"][idx]
        self._prev_puck_speed[idx] = 0.0
        self._contact_count[idx] = 0
        self._shot_in_flight[idx] = False
        self._shot_age[idx] = 0
        if info is not None and "score_agent" in info:
            self._prev_score_agent[idx] = info["score_agent"][idx]
            self._prev_score_opp[idx] = info["score_opponent"][idx]
        else:
            self._prev_score_agent[idx] = 0
            self._prev_score_opp[idx] = 0

    def _predict_goal_crossing(self, x, y, vx, vy):
        """Where the puck's free path crosses the opponent goal line, via the
        shared predict_shot (see it for the wall model)."""
        x_goal, t_total, _n, _first = predict_shot(
            x, y, vx, vy, self.W, self.H, self.r, self.e_n, self.e_t)
        return x_goal, t_total

    def compute(self, obs, raw_rewards, actions=None, info=None):
        assert info is not None and "puck_x" in info, \
            "ExchangeRewardShaper needs true state in info"
        px, py = info["puck_x"], info["puck_y"]
        pvx, pvy = info["puck_vx"], info["puck_vy"]
        pad_x, pad_y = info["pad_x"], info["pad_y"]

        shaped = np.zeros(self.n_envs, dtype=np.float64)
        self.end_exchange = np.zeros(self.n_envs, dtype=bool)

        # Contact bootstrap (capped): the only non-outcome term.
        dist = np.hypot(px - pad_x, py - pad_y)
        speed = np.hypot(pvx, pvy)
        hit = (dist < 0.25) & (speed - self._prev_puck_speed > 0.2) & (pvy > 0)
        self._contact_count += hit.astype(np.int32)
        shaped += np.where(hit & (self._contact_count <= self.max_contacts),
                           self.contact_reward, 0.0)

        # A shot: puck crosses the midline moving away from the agent.
        crossed = ((self._prev_puck_y <= self.H / 2) & (py > self.H / 2)
                   & (pvy > 0.1) & ~self._shot_in_flight)
        if np.any(crossed):
            x_goal, t_goal = self._predict_goal_crossing(px, py, pvx, pvy)
            on_target = crossed & np.isfinite(x_goal) & \
                (np.abs(x_goal - self.W / 2) < self.goal_half)
            shaped += np.where(on_target, self.shot_on_target, 0.0)
            # Crude reach model for the defender; shaping, not physics.
            if "opp_x" in (info or {}):
                ox = info["opp_x"]
            else:
                ox = np.full(self.n_envs, self.W / 2)
            can_reach = np.abs(x_goal - ox) < 0.05 + self.opp_reach_speed * t_goal
            shaped += np.where(on_target & ~can_reach, self.shot_beats_opp, 0.0)
            shaped += np.where(on_target,
                               self.vel_bonus_per_ms * speed, 0.0)
            self._shot_in_flight |= crossed
            self._shot_age[crossed] = 0

        # Terminal outcomes, from the scoreboard.
        goal_for = info["score_agent"] > self._prev_score_agent
        goal_against = info["score_opponent"] > self._prev_score_opp
        self._prev_score_agent[:] = info["score_agent"]
        self._prev_score_opp[:] = info["score_opponent"]
        shaped += np.where(goal_for, self.goal_reward, 0.0)
        shaped += np.where(goal_against, self.goal_penalty, 0.0)
        if "penalty" in info:
            shaped += info["penalty"]

        # Exchange ends: goal either way; or a shot in flight came back over
        # three-quarter table (blocked/returned); or it timed out.
        self._shot_age[self._shot_in_flight] += 1
        returned = self._shot_in_flight & (pvy < 0) & (py < 0.75 * self.H)
        timed_out = self._shot_in_flight & (self._shot_age > self.timeout)
        self.end_exchange = goal_for | goal_against | returned | timed_out
        self._shot_in_flight &= ~self.end_exchange

        self._prev_puck_y[:] = py
        self._prev_puck_speed[:] = speed
        ended = self.end_exchange | goal_for | goal_against
        if np.any(ended):
            self._contact_count[ended] = 0
        return shaped.astype(np.float32)


# ---------------------------------------------------------------------------
# Named curriculum for the pretrain -> self-play pipeline
# ---------------------------------------------------------------------------
# Five stages, each rewarding ONE thing loudly enough to be the signal:
#
#   proximity   get to the puck             (idle opponent)
#   contact     hit it forward              (idle)
#   scoring     put it in                   (idle)
#   goalie      put it past a blocker       (env's stationary goalie)
#   selfplay    win games                   (latest self, see train_selfplay)
#
# Magnitudes were the problem with the old 4-stage table: contact 0.1 next
# to a 160-point goal is nothing, so every dense term was noise and the only
# real signal was a sparse goal -- which is how three trainers learned to
# loiter. Here each stage's target term is O(1..10) per event and the goal
# terms enter only once a stage is ABOUT goals. In self-play a goal is worth
# ~50 clean contacts, not ~1600.
#
# Per-step scale check: proximity 0.1/step over a 10 s episode caps at 50
# (500 steps at 50 Hz); contact 5 x 5 capped hits = 25 max; a goal 100.
#
# Budgets halved for run 2 (2026-09-06): the pretrain only has to get the
# policy doing roughly the right thing before self-play, and run 1's stages
# were winning their recorded games well before their budgets ran out.
CURRICULUM: dict[str, dict] = {
    "proximity": dict(
        opponent="idle", episode_s=10.0, steps=100_000,
        proximity_weight=0.1, contact_reward=0.0, directed_hit_weight=0.0,
        puck_progress_weight=0.0, defense_weight=0.0, shot_placement_weight=0.0,
        goal_reward=0.0, goal_penalty=0.0, entropy_weight=0.0, shot_mix_weight=0.0),
    # From "contact" on, the shot's OUTCOME is what pays (2026-09-06, see
    # ai/RETRAIN.md items 2-4): an on-target shot earns on_target_reward
    # plus shot_speed_weight per m/s, ten times a merely forward hit
    # (directed_hit, cut from 2.0/1.0 to 0.5). From "scoring" on, bringing
    # the puck to rest under the paddle first earns trap_reward and lifts
    # the on-target reward by controlled_shot_bonus, so stop-then-shoot
    # beats the instant slap when both are on target. Self-play adds the
    # shot-type request (shot_type_reward on a match) and the opponent mix.
    "contact": dict(
        opponent="idle", episode_s=10.0, steps=150_000,
        proximity_weight=0.02, contact_reward=5.0, directed_hit_weight=0.5,
        puck_progress_weight=0.0, defense_weight=0.0, shot_placement_weight=0.0,
        goal_reward=0.0, goal_penalty=0.0, entropy_weight=0.0, shot_mix_weight=0.0,
        on_target_reward=10.0, shot_speed_weight=1.0),
    # Run 2 (2026-09-06): the accel the action asks for is taxed from the
    # scoring stage on, and the hit rewards ramp with how long the puck has
    # been on the robot's side (patience_s, floor 0.5) -- run 1 never
    # stopped the puck and drove the paddle 33 m in 20 s on the table.
    "scoring": dict(
        opponent="idle", episode_s=20.0, steps=250_000, fuzz_p=0.2,
        proximity_weight=0.0, contact_reward=2.0, directed_hit_weight=0.5,
        puck_progress_weight=0.5, defense_weight=0.5, shot_placement_weight=1.0,
        goal_reward=100.0, goal_penalty=-20.0, entropy_weight=0.0, shot_mix_weight=0.5,
        on_target_reward=10.0, shot_speed_weight=1.0,
        trap_reward=2.0, controlled_shot_bonus=1.5,
        accel_cost_weight=0.02, patience_s=1.5, patience_floor=0.2, patience_on_goals=True),
    "goalie": dict(
        opponent="goalie", episode_s=20.0, steps=250_000, fuzz_p=0.2,
        proximity_weight=0.0, contact_reward=2.0, directed_hit_weight=0.5,
        puck_progress_weight=0.5, defense_weight=0.5, shot_placement_weight=1.0,
        goal_reward=100.0, goal_penalty=-20.0, entropy_weight=0.0, shot_mix_weight=0.5,
        on_target_reward=10.0, shot_speed_weight=1.0,
        trap_reward=2.0, controlled_shot_bonus=1.5,
        accel_cost_weight=0.02, patience_s=1.5, patience_floor=0.2, patience_on_goals=True),
    "selfplay": dict(
        opponent="external", episode_s=30.0, steps=3_000_000, fuzz_p=0.2,
        proximity_weight=0.0, contact_reward=2.0, directed_hit_weight=0.5,
        puck_progress_weight=0.5, defense_weight=1.0, shot_placement_weight=1.0,
        goal_reward=100.0, goal_penalty=-50.0, entropy_weight=0.0, shot_mix_weight=0.5,
        on_target_reward=10.0, shot_speed_weight=1.0,
        trap_reward=2.0, controlled_shot_bonus=1.5,
        # Run 3: full accel for a whole 30 s episode costs 60 (run 2's 0.02
        # settled the mean fraction at 0.52; the user wants it lower), and
        # patience floors at 0.2 ON THE GOAL AS WELL: a goal from an instant
        # slap pays 20, one from a shot after 1.5 s of control pays 100.
        accel_cost_weight=0.04, patience_s=1.5, patience_floor=0.2, patience_on_goals=True,
        # The env draws a shot type per possession (shot_types=True) and
        # the far side is drawn per episode from opponent_mix.
        shot_type_reward=10.0, shot_types=True,
        opponent_mix={"external": 0.6, "sniper": 0.2, "weak_goalie": 0.2},
        # Idle hygiene (see BatchRewardShaper): at most 0.005/step for
        # sitting off the goal's centre line while the puck is on the far
        # half. Smoothness during play: 0.2 per unit of action change,
        # everywhere -- a full corner-to-corner flip costs 0.57. That is
        # deliberately on the scale of TD-MPC2's value bins: 0.02/unit was
        # tried first and the flip rate did not fall in 300k steps, because
        # the two-hot reward and Q heads (101 symlog bins) cannot resolve a
        # 0.057 difference next to contacts at 2 and goals at 100. One
        # strike per exchange costs a quarter of a contact; flipping every
        # four ticks costs four goals an episode.
        # home_weight 0.05 since the retrain (was 0.005): with the puck away
        # the value landscape was FLAT, and under MPPI a flat landscape is a
        # random walk -- the wander seen on the table and in the replays.
        # At 0.05/step, ten seconds at the box edge costs 25 against a
        # goal's 100; at home it costs nothing. The two-hot value bins
        # resolve ~0.6 near zero, so a discounted 0.05/step (~5) is visible
        # where 0.005 was not.
        home_weight=0.05, jitter_weight=0.0, smooth_weight=0.2),
}
CURRICULUM_ORDER = ["proximity", "contact", "scoring", "goalie", "selfplay"]
# Run 2: position plus an accel fraction (BatchAirHockeyEnv "profile_a").
CURRICULUM_ACTION_MODE = "profile_a"

_SHAPER_KEYS = ("proximity_weight", "contact_reward", "directed_hit_weight",
                "puck_progress_weight", "defense_weight", "shot_placement_weight",
                "goal_reward", "goal_penalty", "entropy_weight", "shot_mix_weight")


# Batch-shaper-only terms (idle hygiene). Passed through only when a stage
# sets them, so the scalar ShapedRewardWrapper -- which has no such terms --
# keeps accepting every pretrain stage's kwargs.
_IDLE_KEYS = ("home_weight", "jitter_weight", "smooth_weight")
# Shot-outcome terms, likewise batch-only and passed through only when set.
_OUTCOME_KEYS = ("on_target_reward", "shot_speed_weight", "trap_reward",
                 "controlled_shot_bonus", "shot_type_reward",
                 "accel_cost_weight", "patience_s", "patience_floor",
                 "patience_on_goals")


def curriculum_episode_steps(name: str, action_dt: float = ACTION_DT) -> int:
    """A stage's episode length in control steps, from its seconds."""
    return steps_for(CURRICULUM[name]["episode_s"], action_dt)


def curriculum_shaper_kwargs(name: str) -> dict:
    """The BatchRewardShaper / ShapedRewardWrapper kwargs for a named stage."""
    spec = CURRICULUM[name]
    out = {k: spec[k] for k in _SHAPER_KEYS}
    out.update({k: spec[k] for k in _IDLE_KEYS if k in spec})
    out.update({k: spec[k] for k in _OUTCOME_KEYS if k in spec})
    return out


def curriculum_env_kwargs(name: str) -> dict:
    """The BatchAirHockeyEnv kwargs a stage sets beyond its opponent:
    the shot-type draw and the opponent mix (self-play only)."""
    spec = CURRICULUM[name]
    # One action space for the whole curriculum: a checkpoint resumes from
    # stage to stage, so every stage must present the same actions.
    out = {"action_mode": CURRICULUM_ACTION_MODE}
    if spec.get("shot_types"):
        out["shot_types"] = True
    if spec.get("opponent_mix"):
        out["opponent_mix_probs"] = dict(spec["opponent_mix"])
    if spec.get("fuzz_p"):
        out["fuzz_p"] = float(spec["fuzz_p"])
    return out
