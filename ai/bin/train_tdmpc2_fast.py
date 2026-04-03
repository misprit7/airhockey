"""Optimized TD-MPC2 training with all speedups combined.

Combines: vectorized batch envs, batched MPPI planning, fast MPPI defaults,
optional self-play, and auto-curriculum — in a single script.

Usage:
    # Pretrain (fast MPPI defaults)
    python bin/train_tdmpc2_fast.py --steps 2000000

    # Auto-curriculum (stages 1-4, auto-advancing on plateau)
    python bin/train_tdmpc2_fast.py --curriculum --steps 5000000

    # Run a specific curriculum stage only
    python bin/train_tdmpc2_fast.py --stage 4 --steps 1000000

    # Self-play (resumes from pretrained agent)
    python bin/train_tdmpc2_fast.py --resume runs/tdmpc2_pretrain/agent.pt --steps 5000000

    # Full MPPI quality (no --fast reduction)
    python bin/train_tdmpc2_fast.py --no-fast

    # Override individual MPPI params
    python bin/train_tdmpc2_fast.py --num-samples 256 --mppi-iterations 4
"""

from __future__ import annotations

import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'

import argparse
import copy
import signal
import sys
import warnings
from collections import deque
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

# Add tdmpc2 to path
TDMPC2_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tdmpc2" / "tdmpc2"
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import cfg_to_dataclass
from common.seed import set_seed
from common.buffer import Buffer
from common import MODEL_SIZE
from tdmpc2 import TDMPC2

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.dynamics import DelayedDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.rewards import (
    ShapedRewardWrapper,
    BatchRewardShaper,
    STAGE_CHASE_HIT, STAGE_SCORING, STAGE_SELFPLAY,
    STAGE_GAME_GOALIE, STAGE_GAME_FOLLOW,
    STAGE_OPPONENT, STAGE_NAMES as REWARD_STAGE_NAMES, STAGE_DEFAULTS,
    STAGE_EPISODE_STEPS,
)
from airhockey.curriculum import CurriculumLRScheduler

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Use canonical stage names/opponents from rewards.py
STAGE_NAMES = REWARD_STAGE_NAMES

STAGE_PLATEAU_CONFIG: dict[int, dict] = {
    1: {"lookback": 5,  "window": 3,  "min_steps": 150_000},
    2: {"lookback": 8,  "window": 5,  "min_steps": 150_000},
    3: {"lookback": 10, "window": 5,  "min_steps": 150_000},
    4: {"lookback": 15, "window": 10, "min_steps": 150_000},
}

# Minimum avg goals per episode before stage advancement is allowed.
# Prevents advancing on shaping rewards alone without actually scoring.
STAGE_MIN_GOALS: dict[int, float] = {
    1: 0,    # no goals needed for chase+hit
    2: 1,    # must avg 1+ goal/ep vs goalie before advancing
    3: 1,    # must avg 1+ goal/ep vs follower
    4: 0,    # self-play never advances
}

STAGE_HORIZON: dict[int, int] = {
    1: 5,   # sequential MLP is faster than transformer at our scale
    2: 5,
    3: 5,
    4: 5,
}


class PlateauDetector:
    """Detect reward plateau for curriculum auto-advancement.

    Tracks rolling avg over last `window` episodes. Every check, compares
    current avg to avg from `lookback` episodes ago. If improvement < threshold,
    plateau is detected.
    """

    def __init__(self, window: int = 100, lookback: int = 200, threshold: float = 0.05,
                 min_steps: int = 0):
        self.window = window
        self.lookback = lookback
        self.threshold = threshold
        self.min_steps = min_steps
        self._total_steps = 0
        self.rewards: list[float] = []

    def add(self, reward: float, steps: int = 0) -> None:
        self.rewards.append(reward)
        self._total_steps += steps

    def reset(self) -> None:
        self.rewards.clear()
        self._total_steps = 0

    def configure(self, window: int | None = None, lookback: int | None = None,
                  threshold: float | None = None, min_steps: int | None = None) -> None:
        """Reconfigure for a new stage. Resets internal state."""
        if window is not None:
            self.window = window
        if lookback is not None:
            self.lookback = lookback
        if threshold is not None:
            self.threshold = threshold
        if min_steps is not None:
            self.min_steps = min_steps
        self.reset()

    def configure_for_stage(self, stage: int) -> None:
        """Apply per-stage plateau config from STAGE_PLATEAU_CONFIG."""
        cfg = STAGE_PLATEAU_CONFIG.get(stage, {})
        self.configure(**cfg)

    def check(self) -> tuple[bool, float, float]:
        """Returns (is_plateau, current_avg, old_avg)."""
        if self._total_steps < self.min_steps:
            return False, 0.0, 0.0

        n = len(self.rewards)
        if n < self.lookback + self.window:
            return False, 0.0, 0.0

        current_avg = float(np.mean(self.rewards[-self.window:]))
        old_start = n - self.lookback - self.window
        old_avg = float(np.mean(self.rewards[old_start:old_start + self.window]))

        if abs(old_avg) < 1.0:
            is_plateau = (current_avg - old_avg) < self.threshold
        else:
            is_plateau = (current_avg - old_avg) / abs(old_avg) < self.threshold

        return is_plateau, current_avg, old_avg


# ---------------------------------------------------------------------------
# Opponent pool for self-play
# ---------------------------------------------------------------------------

class OpponentPool:
    """Opponent pool for self-play with score-differential matchmaking.

    Stores up to `max_size` opponent state dicts on CPU. Selects opponents
    using a mix of challenge-matching (close score differentials) and old
    opponent refresh (prevents forgetting).

    Elo is tracked for monitoring but NOT used for selection.
    """

    K = 32  # Elo K-factor
    INITIAL_ELO = 1000.0

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.snapshots: list[dict] = []    # CPU state dicts
        self.elos: list[float] = []        # per-snapshot Elo (monitoring only)
        self.score_diffs: list[list[float]] = []  # per-snapshot recent score diffs
        self.agent_elo: float = self.INITIAL_ELO

    def add(self, agent) -> int:
        """Save a snapshot of the agent. Returns snapshot index."""
        sd = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in agent.state_dict().items()}
        if len(self.snapshots) >= self.max_size:
            evict = self._worst_match_idx()
            self.snapshots.pop(evict)
            self.elos.pop(evict)
            self.score_diffs.pop(evict)
        self.snapshots.append(sd)
        self.elos.append(self.agent_elo)
        self.score_diffs.append([])
        return len(self.snapshots) - 1

    def sample(self, rng: np.random.Generator) -> tuple[int, str]:
        """Sample an opponent. Returns (index, reason_str)."""
        n = len(self.snapshots)
        if n == 0:
            raise ValueError("Pool is empty")
        if n == 1:
            return 0, "ONLY"

        # 50% chance: uniform random from entire pool
        if rng.random() < 0.5:
            return int(rng.integers(0, n)), "RANDOM"

        # 50% chance: pick by challenge level (closest score diff to 0)
        weights = np.empty(n)
        for i in range(n):
            if self.score_diffs[i]:
                avg_diff = np.mean(self.score_diffs[i])
                weights[i] = np.exp(-abs(avg_diff) / 2.0)
            else:
                weights[i] = 1.0  # untested = high priority

        # Most recent always gets at least 10%
        weights[-1] = max(weights[-1], weights.sum() * 0.1)
        probs = weights / weights.sum()
        return int(rng.choice(n, p=probs)), "CHALLENGE"

    def __len__(self) -> int:
        return len(self.snapshots)

    def load_into(self, opponent, idx: int) -> None:
        """Load snapshot at idx into the opponent agent."""
        opponent.load_state_dict(self.snapshots[idx], strict=False)

    def record_outcome(self, idx: int, agent_goals: int, opp_goals: int) -> None:
        """Record game result with score differential and update Elo."""
        if not (0 <= idx < len(self.elos)):
            return

        # Score differential tracking (keep last 10 games)
        diff = agent_goals - opp_goals
        self.score_diffs[idx].append(diff)
        if len(self.score_diffs[idx]) > 10:
            self.score_diffs[idx] = self.score_diffs[idx][-10:]

        # Elo update (for monitoring)
        elo_opp = self.elos[idx]
        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_opp - self.agent_elo) / 400.0))
        if agent_goals == opp_goals:
            actual_a = 0.5
        else:
            actual_a = 1.0 if agent_goals > opp_goals else 0.0
        self.agent_elo += self.K * (actual_a - expected_a)
        self.elos[idx] += self.K * ((1.0 - actual_a) - (1.0 - expected_a))

    def avg_diff(self, idx: int) -> float:
        """Average score differential against snapshot idx."""
        if 0 <= idx < len(self.score_diffs) and self.score_diffs[idx]:
            return np.mean(self.score_diffs[idx])
        return 0.0

    def get_elo(self, idx: int) -> float:
        return self.elos[idx] if 0 <= idx < len(self.elos) else 0.0

    def elo_summary(self) -> str:
        if not self.elos:
            return "empty pool"
        elos = np.array(self.elos)
        return (f"pool={len(elos)} agent_elo={self.agent_elo:.0f} "
                f"min={elos.min():.0f} median={np.median(elos):.0f} "
                f"max={elos.max():.0f} std={elos.std():.0f}")

    def _worst_match_idx(self) -> int:
        """Find snapshot with worst challenge value for eviction.
        Never evict the most recent snapshot."""
        n = len(self.snapshots)
        if n <= 1:
            return 0
        worst_idx, worst_val = 0, float('inf')
        for i in range(n - 1):
            if self.score_diffs[i]:
                val = np.exp(-abs(np.mean(self.score_diffs[i])) / 2.0)
            else:
                val = 1.0
            if val < worst_val:
                worst_val = val
                worst_idx = i
        return worst_idx

    def __len__(self) -> int:
        return len(self.snapshots)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class FastLogger:
    """TensorBoard + stdout logger with self-play support."""

    def __init__(self, cfg, self_play: bool = False):
        self.cfg = cfg
        self.self_play = self_play
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.work_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(log_dir))
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def log_train(self, step: int, ep_reward: float, ep_length: int, fps: float,
                  score_agent: int = 0, score_opponent: int = 0):
        if self._should_print(step):
            if self.self_play:
                total = self.wins + self.losses + self.draws
                wr = self.wins / max(total, 1) * 100
                print(f"[Train] step={step:,} r={ep_reward:.1f} fps={fps:.0f} "
                      f"W/L/D={self.wins}/{self.losses}/{self.draws} WR={wr:.0f}%")
            else:
                print(f"[Train] step={step:,} r={ep_reward:.1f} len={ep_length} fps={fps:.0f}")

    def log_step(self, step: int, fps: float, n_envs: int):
        """Log periodic step-based metrics (FPS, etc.) independent of episode boundaries."""
        if not self._should_log(step, n_envs):
            return
        self.writer.add_scalar('train/fps', fps, step)
        self.writer.add_scalar('train/total_episodes', self.wins + self.losses + self.draws
                               if self.self_play else 0, step)
        self.writer.flush()
        if self._should_print(step):
            print(f"[Step] step={step:,} fps={fps:.0f}")

    def _should_log(self, step: int, n_envs: int) -> bool:
        """True roughly every 10k steps, accounting for step increments of n_envs."""
        return step % 10_000 < n_envs

    def _should_print(self, step: int) -> bool:
        """True roughly every 50k steps for stdout (less noisy than TensorBoard)."""
        return step % 50_000 < 100

    def log_eval(self, step: int, reward: float, length: int, fps: float):
        print(f"[Eval] step={step:,} reward={reward:.1f} length={length:.0f} fps={fps:.0f}")
        self.writer.add_scalar('eval/episode_reward', reward, step)
        self.writer.add_scalar('eval/episode_length', length, step)

    def record_outcome(self, score_agent: int, score_opponent: int):
        if score_agent > score_opponent:
            self.wins += 1
        elif score_opponent > score_agent:
            self.losses += 1
        else:
            self.draws += 1

    def log_curriculum(self, step: int, stage: int, stage_name: str,
                       current_avg: float = 0.0, old_avg: float = 0.0,
                       transition: bool = False):
        """Log curriculum stage info to TensorBoard."""
        self.writer.add_scalar('curriculum/stage', stage, step)
        self.writer.add_scalar(f'curriculum/stage_{stage}_reward', current_avg, step)
        if transition:
            self.writer.flush()

    def finish(self, agent):
        agent.save(self.work_dir / 'agent.pt')
        print(f"Agent saved to {self.work_dir / 'agent.pt'}")
        self.writer.close()

    # Dummy video interface for TD-MPC2 compatibility
    class Video:
        def init(self, *a, **kw): pass
        def record(self, *a, **kw): pass
        def save(self, *a, **kw): pass

    @property
    def video(self):
        return self.Video()


# ---------------------------------------------------------------------------
# Recording / eval helpers
# ---------------------------------------------------------------------------

def _run_eval(agent, cfg, step, start_time, logger, use_dynamics, frame_stack=1):
    """Run eval episodes."""
    from airhockey.dynamics import IdealDynamics
    dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else IdealDynamics()
    env = ShapedRewardWrapper(
        AirHockeyEnv(
            agent_dynamics=dynamics,
            opponent_policy="idle",
            record=False,
            action_dt=1 / 60,
            max_episode_time=30.0,
            max_score=7,
            frame_stack=frame_stack,
        ),
        stage=STAGE_SCORING,
    )
    ep_rewards = []
    t = 0
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        obs_t = torch.from_numpy(obs).float()
        eval_done, ep_reward, t = False, 0, 0
        while not eval_done:
            action = agent.act(obs_t, t0=t == 0, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action.numpy())
            obs_t = torch.from_numpy(obs).float()
            ep_reward += reward
            eval_done = terminated or truncated
            t += 1
        ep_rewards.append(ep_reward)
    elapsed = time() - start_time
    logger.log_eval(step, np.nanmean(ep_rewards), t, step / max(elapsed, 1))


def _fresh_agent_from_checkpoint(cfg, checkpoint_path):
    """Create a fresh uncompiled GPU agent and load weights from checkpoint.

    Used for synchronous recording/eval on the main thread.
    Uncompiled to avoid CUDA graph state issues.
    """
    fresh_cfg = copy.deepcopy(cfg)
    fresh_cfg.compile = False
    fresh = TDMPC2(fresh_cfg)
    try:
        fresh.load(checkpoint_path)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Warning: could not load checkpoint {checkpoint_path}: {e}")
        print("Using randomly initialized agent for eval/recording.")
    return fresh


def _record_game_pretrain(agent, step, recordings_dir, run_name, stage=STAGE_SCORING, frame_stack=1):
    """Record a pretrain game (vs idle opponent) for web UI."""
    inner = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy="idle",
        record=True,
        action_dt=1 / 60,
        max_episode_time=90.0,
        max_score=21,
        frame_stack=frame_stack,
    )
    wrapped = ShapedRewardWrapper(inner, stage=stage)
    obs, _ = wrapped.reset()
    obs_t = torch.from_numpy(obs).float()
    done, t = False, 0
    while not done:
        with torch.no_grad():
            action = agent.act(obs_t, t0=(t == 0), eval_mode=True)
        obs, _, terminated, truncated, info = wrapped.step(action.numpy())
        obs_t = torch.from_numpy(obs).float()
        done = terminated or truncated
        t += 1
    recording = inner.get_recording()
    if recording:
        rec = Recorder()
        rec._current = recording
        recordings_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "stage": stage,
            "stage_name": f"Stage {stage}: {STAGE_NAMES.get(stage, 'Unknown')}",
            "step": step,
        }
        rec.save(
            recordings_dir / f"{run_name}_s{stage}_step_{step:07d}.json",
            metadata=metadata,
        )
        score = f"{info['score_agent']}-{info['score_opponent']}"
        print(f"Recorded game at step {step:,}: {score}")


def _record_game_selfplay(agent, opponent, step, recordings_dir, run_name, stage=STAGE_SCORING, frame_stack=1):
    """Record a self-play game for web UI."""
    inner = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy="external",
        record=True,
        action_dt=1 / 60,
        max_episode_time=90.0,
        max_score=21,
        frame_stack=frame_stack,
    )
    wrapped = ShapedRewardWrapper(inner, stage=stage)
    obs, _ = wrapped.reset()
    obs_t = torch.from_numpy(obs).float()
    done, t = False, 0
    while not done:
        with torch.no_grad():
            action = agent.act(obs_t, t0=(t == 0), eval_mode=True)
            opp_obs = torch.from_numpy(inner.mirror_obs(obs)).float()
            opp_action = opponent.act(opp_obs, t0=(t == 0), eval_mode=True)
            tx, ty = inner.mirror_action_to_opponent(opp_action.numpy())
            inner.set_opponent_action(tx, ty)
        obs, _, terminated, truncated, info = wrapped.step(action.numpy())
        obs_t = torch.from_numpy(obs).float()
        done = terminated or truncated
        t += 1
    recording = inner.get_recording()
    if recording:
        rec = Recorder()
        rec._current = recording
        recordings_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "stage": stage,
            "stage_name": f"Stage {stage}: {STAGE_NAMES.get(stage, 'Unknown')}",
            "step": step,
        }
        rec.save(
            recordings_dir / f"{run_name}_s{stage}_step_{step:07d}.json",
            metadata=metadata,
        )
        score = f"{info['score_agent']}-{info['score_opponent']}"
        print(f"Recorded game at step {step:,}: {score}")


def _run_benchmark(agent, step, logger, frame_stack=1):
    """Run benchmark evaluation against fixed opponents.

    Plays 5 games each against idle, goalie, and follow opponents.
    Logs win rates, goals per game, and composite score to TensorBoard.
    """
    from airhockey.dynamics import DelayedDynamics

    opponents = ["idle", "goalie", "follow"]
    n_games = 5
    weights = {"idle": 0.1, "goalie": 0.4, "follow": 0.5}
    results = {}

    for opp in opponents:
        wins, goals_for, goals_against = 0, 0, 0
        for game in range(n_games):
            env = AirHockeyEnv(
                agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
                opponent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
                opponent_policy=opp,
                action_dt=1 / 60,
                max_episode_steps=1800,
                max_score=7,
                frame_stack=frame_stack,
            )
            obs, _ = env.reset(seed=step + game)
            obs_t = torch.from_numpy(obs).float()
            done, t = False, 0
            while not done:
                with torch.no_grad():
                    action = agent.act(obs_t, t0=(t == 0), eval_mode=True)
                obs, _, terminated, truncated, info = env.step(action.numpy())
                obs_t = torch.from_numpy(obs).float()
                done = terminated or truncated
                t += 1
            sa, so = info["score_agent"], info["score_opponent"]
            goals_for += sa
            goals_against += so
            if sa > so:
                wins += 1

        wr = wins / n_games
        gpg = goals_for / n_games
        results[opp] = {"win_rate": wr, "goals_per_game": gpg, "goals_against": goals_against / n_games}

        logger.writer.add_scalar(f"benchmark/{opp}_win_rate", wr, step)
        logger.writer.add_scalar(f"benchmark/{opp}_goals_per_game", gpg, step)

    composite = sum(weights[opp] * results[opp]["win_rate"] for opp in opponents)
    logger.writer.add_scalar("benchmark/composite_score", composite, step)

    print(f"\n[Benchmark] step={step:,}")
    for opp in opponents:
        r = results[opp]
        print(f"  vs {opp:8s}: WR={r['win_rate']*100:.0f}%  GF={r['goals_per_game']:.1f}  GA={r['goals_against']:.1f}")
    print(f"  Composite: {composite:.2f}\n")



def _atomic_save(state, path):
    """Save a checkpoint atomically via tmp+rename to prevent corruption on crash."""
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.rename(path)


def _save_training_state(run_dir, step, current_stage, stage_step_start,
                         total_episodes, plateau=None):
    """Save training state alongside agent.pt for checkpoint resume."""
    state = {
        'step': step,
        'current_stage': current_stage,
        'stage_step_start': stage_step_start,
        'total_episodes': total_episodes,
    }
    if plateau is not None:
        state['plateau_rewards'] = list(plateau.rewards)
        state['plateau_total_steps'] = plateau._total_steps
    _atomic_save(state, run_dir / 'training_state.pt')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast TD-MPC2 training (vectorized envs + batched planning)"
    )
    # Training
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--model-size", type=int, default=5, choices=[1, 5, 19, 48])
    parser.add_argument("--n-envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--dynamics", action="store_true", default=True)
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--record-freq", type=int, default=50_000)
    parser.add_argument("--benchmark-freq", type=int, default=500_000,
                        help="Run benchmark evaluation every N steps (0 to disable)")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack in observations (default: 1)")

    # Simple mode (replicate original working config)
    parser.add_argument("--simple", action="store_true",
                        help="Simple training: idle opponent, original rewards, no curriculum")

    # Self-play
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (enables self-play mode)")
    parser.add_argument("--opponent-update-freq", type=int, default=50_000)
    parser.add_argument("--opponent-mix", type=str, default="32:12:12:8",
                        help="Opponent mix for self-play: external:follow:goalie:idle "
                             "counts (e.g. '32:12:12:8'). Set to '0' to disable.")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true",
                        help="Auto-advance through stages 1-4 on plateau detection")
    parser.add_argument("--stage", type=int, default=None, choices=range(1, 5),
                        help="Run a specific curriculum stage only (1-4)")
    parser.add_argument("--min-steps-per-stage", type=int, default=20_000,
                        help="Minimum env steps before checking plateau (per-stage "
                             "min_steps in STAGE_PLATEAU_CONFIG is the real gate)")
    parser.add_argument("--lr-schedule", action="store_true", default=True,
                        help="Per-stage cosine LR decay (default: True)")
    parser.add_argument("--no-lr-schedule", dest="lr_schedule", action="store_false",
                        help="Disable per-stage LR scheduling")

    # MPPI tuning (fast defaults)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--no-fast", action="store_true",
                        help="(deprecated, full quality is now default)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="MPPI trajectory samples (default: 512)")
    parser.add_argument("--mppi-iterations", type=int, default=None,
                        help="MPPI refinement iterations (default: 6)")
    parser.add_argument("--replan-every", type=int, default=1,
                        help="Reuse MPPI plan for K steps before replanning (must be <= horizon)")
    parser.add_argument("--updates-per-step", type=int, default=None,
                        help="Gradient updates per batch of env steps. Defaults to n_envs "
                             "(1:1 update-to-data ratio, matching single-env TD-MPC2).")
    parser.add_argument("--utd-ratio", type=float, default=None,
                        help="Update-to-data ratio as a fraction. Sets updates-per-step = "
                             "utd_ratio * n_envs. E.g. --utd-ratio 1 = 1:1, --utd-ratio 0.5 = 1:2.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (trajectory slices per gradient update). "
                             "Default: 256. Larger batches (512, 1024) enable fewer but bigger "
                             "updates for the same data throughput.")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile on gradient update (default: True). "
                             "~1.9-3.4x speedup with ~33s warmup on first iteration.")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Disable torch.compile for gradient update.")
    parser.add_argument("--prioritized-replay", action="store_true", default=True,
                        help="Use Prioritized Experience Replay (default: True)")
    parser.add_argument("--no-prioritized-replay", dest="prioritized_replay",
                        action="store_false",
                        help="Disable Prioritized Experience Replay")
    args = parser.parse_args()

    # --- Signal handler for graceful shutdown ---
    _shutdown = False
    def _handle_signal(signum, frame):
        nonlocal _shutdown
        _shutdown = True
        print(f"\nSignal {signum} received, will save checkpoint and exit after current step...")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    simple_mode = args.simple
    self_play = args.resume is not None and not simple_mode
    use_curriculum = (args.curriculum or args.stage is not None) and not simple_mode
    if args.run_name is None:
        if simple_mode:
            args.run_name = "simple"
        elif use_curriculum:
            args.run_name = "curriculum"
        elif self_play:
            args.run_name = "selfplay_fast"
        else:
            args.run_name = "tdmpc2_fast"

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = Path("recordings")
    n_envs = args.n_envs

    # --- Parse opponent mix for self-play ---
    opp_mix = None
    if self_play and args.opponent_mix and args.opponent_mix != "0":
        parts = [int(p) for p in args.opponent_mix.split(":")]
        assert len(parts) == 4, (
            f"--opponent-mix expects 4 colon-separated counts "
            f"(external:follow:goalie:idle), got {len(parts)}"
        )
        opp_mix = {
            "external": parts[0],
            "follow": parts[1],
            "goalie": parts[2],
            "idle": parts[3],
        }
        mix_total = sum(parts)
        if mix_total != n_envs:
            print(f"Adjusting n_envs from {n_envs} to {mix_total} to match --opponent-mix")
            n_envs = mix_total

    # --- Auto-resume: detect existing checkpoint in run directory ---
    auto_ckpt = run_dir / 'training_state.pt'
    auto_resumed = False
    if auto_ckpt.exists() and (run_dir / 'agent.pt').exists() and not args.resume:
        print(f"Auto-resuming from {run_dir} (found training_state.pt + agent.pt)")
        args.resume = str(run_dir / 'agent.pt')
        auto_resumed = True
        # Don't change self_play — auto-resume preserves the original mode

    # ------------------------------------------------------------------
    # Build OmegaConf config
    # ------------------------------------------------------------------
    from omegaconf import OmegaConf

    base_cfg = OmegaConf.load(str(TDMPC2_DIR / "config.yaml"))
    overrides = OmegaConf.create({
        "task": "airhockey-selfplay" if self_play else "airhockey",
        "obs": "state",
        "episodic": True,
        "steps": args.steps,
        "model_size": args.model_size,
        "horizon": args.horizon,
        "eval_freq": 100_000,
        "eval_episodes": 3,
        "save_video": False,
        "enable_wandb": False,
        "save_csv": False,
        "work_dir": str(run_dir),
        "compile": args.compile,
        "data_dir": str(run_dir / "data"),
        "exp_name": args.run_name,
        "discount_max": 0.99,
        "rho": 0.85,
        "task_title": "Air Hockey Self-Play" if self_play else "Air Hockey",
        "multitask": False,
        "tasks": ["airhockey-selfplay" if self_play else "airhockey"],
        "task_dim": 0,
        "prioritized_replay": args.prioritized_replay,
        "per_alpha": 0.6,
        "per_beta": 0.4,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)

    # MPPI params: full quality defaults unless --fast
    if not args.no_fast:
        cfg.num_samples = 512
        cfg.iterations = 6
        # horizon comes from CLI default (15) or STAGE_HORIZON — don't override here
    # Individual overrides always win
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.mppi_iterations is not None:
        cfg.iterations = args.mppi_iterations
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    replan_k = args.replan_every
    if replan_k > cfg.horizon:
        parser.error(f"--replan-every ({replan_k}) must be <= horizon ({cfg.horizon})")

    if args.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[args.model_size].items():
            cfg[k] = v

    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

    # Env shape info (from BatchAirHockeyEnv)
    frame_stack = args.frame_stack  # kept for API compat; always 1 internally
    obs_dim = 14  # puck(4) + paddle(4) + opp(4) + context(2)
    action_dim = 2
    episode_length = 1800  # 30s at 60Hz

    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "obs_shape": {"state": [obs_dim]},
        "action_dim": action_dim,
        "episode_length": episode_length,
        "seed_steps": max(1000, 5 * episode_length),
    }))
    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Create agent(s) and batch env
    # ------------------------------------------------------------------
    agent = TDMPC2(cfg)
    # Disable torch.compile for planning — it causes constant recompilation
    # due to torch.randn() and torch.topk() graph breaks. Only _update benefits.
    if args.compile:
        agent._plan_val = agent._plan
    opponent = None
    opp_pool = None
    current_opp_idx = -1

    if auto_resumed and not self_play and args.resume:
        print(f"Auto-resume: loading agent from {args.resume}")
        agent.load(args.resume)
    if self_play:
        print(f"Loading agent from {args.resume}")
        agent.load(args.resume)
        opponent = TDMPC2(cfg)
        opponent.load(args.resume)
        opp_pool = OpponentPool(max_size=100)
        opp_pool.add(opponent)  # Seed pool with initial checkpoint
        current_opp_idx = 0
        if args.compile:
            opponent._plan_val = opponent._plan
        # Opponent uses full MPPI planning for stronger competition.
        # Only ~2x cost since planning is batched on GPU.

    # --- Curriculum state ---
    if use_curriculum:
        if args.stage is not None:
            current_stage = args.stage
            max_stage = args.stage  # Fixed, no advancement
        else:
            current_stage = 1
            max_stage = 4
        stage_step_start = 0  # step at which current stage began
        plateau = PlateauDetector()
        plateau.configure_for_stage(current_stage)
        # Set initial planning horizon for this stage
        cfg.horizon = STAGE_HORIZON.get(current_stage, cfg.horizon)
    else:
        current_stage = None
        stage_step_start = 0

    # --- Per-stage LR scheduling ---
    lr_scheduler = None
    if args.lr_schedule and use_curriculum:
        lr_scheduler = CurriculumLRScheduler(
            optim=agent.optim,
            pi_optim=agent.pi_optim,
            stage=current_stage,
            enc_lr_scale=cfg.enc_lr_scale,
            estimated_stage_steps=STAGE_PLATEAU_CONFIG[current_stage]["min_steps"] * 3,
        )

    # Determine initial opponent policy
    if simple_mode:
        opp_policy = "idle"
    elif self_play:
        opp_policy = "external"
    elif use_curriculum:
        opp_policy = STAGE_OPPONENT[current_stage]
    else:
        opp_policy = "idle"

    dyn_type = "delayed" if args.dynamics else "ideal"
    episode_steps = STAGE_EPISODE_STEPS.get(current_stage, 1800) if use_curriculum else None
    batch_env = BatchAirHockeyEnv(
        n_envs=n_envs,
        agent_dynamics=dyn_type,
        opponent_dynamics=dyn_type,
        opponent_policy=opp_policy,
        opponent_mix=opp_mix,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_episode_steps=episode_steps,
        max_score=7,
        dynamics_max_speed=3.0,
        dynamics_max_accel=30.0,
        frame_stack=frame_stack,
        score_handicap=(current_stage == STAGE_SELFPLAY) if use_curriculum else self_play,
    )
    # Mask for self-play (external) envs — used to subset opponent planning
    sp_mask = batch_env.external_mask if self_play else None

    # Reward shaping — use curriculum stage config if active
    def _make_reward_shaper(stage_num=None):
        if simple_mode:
            # Original Session 2 rewards that produced strong play at 350k steps
            return BatchRewardShaper(
                n_envs, stage=STAGE_SCORING,
                proximity_weight=0.1, contact_reward=5.0,
                directed_hit_weight=2.0, puck_progress_weight=3.0,
                goal_reward=100.0, goal_penalty=-5.0,
                defense_weight=0, shot_placement_weight=0, entropy_weight=0,
                max_contacts_per_episode=999,
            )
        if stage_num is not None and stage_num in STAGE_DEFAULTS:
            return BatchRewardShaper(n_envs, stage=stage_num)
        elif self_play:
            # Use same rewards the agent was pretrained with — the world model's
            # reward head learned these, so MPPI planning depends on them matching.
            return BatchRewardShaper(
                n_envs, stage=STAGE_SCORING,
                proximity_weight=0.1, contact_reward=5.0,
                directed_hit_weight=2.0, puck_progress_weight=3.0,
                goal_reward=100.0, goal_penalty=-5.0,
                defense_weight=0, shot_placement_weight=0, entropy_weight=0,
                max_contacts_per_episode=999,
            )
        else:
            return BatchRewardShaper(n_envs, stage=STAGE_SCORING)

    reward_shaper = _make_reward_shaper(current_stage)

    buffer = Buffer(cfg)
    logger = FastLogger(cfg, self_play=self_play)

    mode_str = "(Self-Play)" if self_play else "(Curriculum)" if use_curriculum else "(Pretrain)"
    print(f"TD-MPC2 Fast Training {mode_str}")
    print(f"  Steps: {args.steps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Model size: {args.model_size}M params")
    print(f"  Planning horizon: {cfg.horizon}")
    print(f"  MPPI samples: {cfg.num_samples}, iterations: {cfg.iterations}")
    if replan_k > 1:
        print(f"  Action chunking: replan every {replan_k} steps")
    # Resolve update count: --utd-ratio takes priority, then --updates-per-step,
    # then default to n_envs (1:1 ratio matching single-env TD-MPC2).
    if args.utd_ratio is not None:
        utd = max(1, int(args.utd_ratio * n_envs))
    elif args.updates_per_step is not None:
        utd = args.updates_per_step
    else:
        utd = n_envs  # 1:1 update-to-data ratio
    print(f"  Updates per step: {utd} (UTD ratio = {utd / n_envs:.2f}, i.e. {utd} updates per {n_envs} env steps)")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Compiled: {args.compile}" + (" (update only, ~33s warmup on first iteration)" if args.compile else ""))
    print(f"  PER: {args.prioritized_replay}" + (f" (alpha={cfg.per_alpha}, beta={cfg.per_beta}→1.0)" if args.prioritized_replay else ""))
    print(f"  Dynamics: {args.dynamics}")
    print(f"  Output: {run_dir}")
    if self_play:
        print(f"  Opponent: full MPPI planning, Elo-rated pool up to 100")
        print(f"  Opponent update: every {args.opponent_update_freq:,} steps")
    if use_curriculum:
        if args.stage is not None:
            print(f"  Curriculum: fixed stage {current_stage} ({STAGE_NAMES[current_stage]})")
        else:
            print(f"  Curriculum: auto stages 1-4, starting at 1 ({STAGE_NAMES[1]})")
            print(f"  Min steps per stage: {args.min_steps_per_stage:,}")
        if lr_scheduler is not None:
            print(f"  LR schedule: cosine decay per stage (initial={lr_scheduler.current_lr:.1e})")
        else:
            print(f"  LR schedule: disabled")
    print()

    # ------------------------------------------------------------------
    # Initialize per-env episode tracking
    # ------------------------------------------------------------------
    obs_all = batch_env.reset()  # [N, obs_dim]

    # Stagger initial episode step counts so envs don't all finish simultaneously.
    # Without this, all N envs complete at the same training step, producing
    # sparse TensorBoard curves (one dot per N-env batch instead of smooth lines).
    if batch_env.max_episode_steps is not None:
        batch_env._step_count[:] = np.linspace(
            0, batch_env.max_episode_steps, n_envs, endpoint=False, dtype=int,
        )
    else:
        batch_env.engine.time[:] = np.linspace(
            0, batch_env.max_episode_time, n_envs, endpoint=False,
        )

    reset_info = {
        "puck_vx": batch_env.engine.puck_vx.copy(),
        "puck_vy": batch_env.engine.puck_vy.copy(),
    }
    reward_shaper.reset(obs_all, info=reset_info)
    obs_t = torch.from_numpy(obs_all).float()

    # Position indices in 14-dim obs (no frame stacking)
    _pb = 0  # base offset (always 0 with velocities, no stacking)

    prev_puck_speed_track = np.hypot(reset_info["puck_vx"], reset_info["puck_vy"])
    stage_wins: deque[float] = deque(maxlen=100)
    stage_goals: deque[int] = deque(maxlen=100)  # goals scored per episode

    # Per-env trajectory buffers (list of TensorDicts per env)
    rand_act = torch.zeros(action_dim)
    tds_list: list[list[TensorDict]] = []
    for i in range(n_envs):
        tds_list.append([TensorDict(
            obs=obs_t[i].unsqueeze(0).cpu(),
            action=torch.full_like(rand_act, float('nan')).unsqueeze(0),
            reward=torch.tensor(float('nan')).unsqueeze(0),
            terminated=torch.tensor(float('nan')).unsqueeze(0),
        batch_size=(1,))])
    t0_all = np.ones(n_envs, dtype=bool)

    # ------------------------------------------------------------------
    # Restore training state from checkpoint (if available)
    # ------------------------------------------------------------------
    step = 0
    total_episodes = 0
    if args.resume:
        resume_dir = Path(args.resume).parent.resolve()
        same_run = resume_dir == run_dir.resolve()
        state_path = resume_dir / 'training_state.pt'
        if state_path.exists() and same_run:
            ts = torch.load(state_path, map_location='cpu', weights_only=False)
            step = ts['step']
            total_episodes = ts.get('total_episodes', 0)
            stage_step_start = ts.get('stage_step_start', 0)
            if use_curriculum:
                saved_stage = ts.get('current_stage')
                if saved_stage is not None:
                    current_stage = saved_stage
                    plateau.configure_for_stage(current_stage)
                    cfg.horizon = STAGE_HORIZON.get(current_stage, cfg.horizon)
                    batch_env.opponent_policy = STAGE_OPPONENT[current_stage]
                    batch_env.max_episode_steps = STAGE_EPISODE_STEPS.get(current_stage, 1800)
                    reward_shaper = _make_reward_shaper(current_stage)
                saved_rewards = ts.get('plateau_rewards')
                if saved_rewards:
                    plateau.rewards = saved_rewards
                    plateau._total_steps = ts.get('plateau_total_steps', 0)
            print(f"Restored training state: step={step:,}, stage={current_stage}, "
                  f"episodes={total_episodes:,}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    start_time = time()
    last_record_step = step
    last_opponent_update = step

    # Step-based metric accumulators (logged every 10k steps, then reset)
    step_goals_scored = 0
    step_goals_conceded = 0
    step_contacts = 0
    step_episodes_completed = 0
    step_wins = 0
    step_shaped_reward = 0.0
    step_puck_dist_acc = 0.0
    step_puck_dist_count = 0

    opponent_version = 0
    pretrained = False

    # CUDA streams for concurrent agent/opponent planning
    if self_play and torch.cuda.is_available():
        agent_stream = torch.cuda.Stream()
        opp_stream = torch.cuda.Stream()
    else:
        agent_stream = None
        opp_stream = None

    # Action chunking state
    chunk_step = 0  # current position within the replan chunk
    cached_plan = None  # [N, H, A] mean plan from last MPPI call

    while step <= cfg.steps:
        # --- Opponent update (self-play) ---
        if self_play and step > 0 and step // args.opponent_update_freq > last_opponent_update // args.opponent_update_freq:
            last_opponent_update = step
            opponent_version += 1
            agent.save(run_dir / "agent.pt")
            # Add current agent to opponent pool, sample next opponent
            opp_pool.add(agent)
            current_opp_idx, select_reason = opp_pool.sample(batch_env._rng)
            opp_pool.load_into(opponent, current_opp_idx)
            opp_elo = opp_pool.get_elo(current_opp_idx)
            avg_diff = opp_pool.avg_diff(current_opp_idx)
            print(f"[Step {step:,}] Selected opponent #{current_opp_idx} "
                  f"({select_reason}, avg_diff={avg_diff:+.1f}), "
                  f"pool={len(opp_pool)}, agent_elo={opp_pool.agent_elo:.0f}")
            # TensorBoard logging
            logger.writer.add_scalar('selfplay/agent_elo', opp_pool.agent_elo, step)
            logger.writer.add_scalar('selfplay/opponent_elo', opp_elo, step)
            logger.writer.add_scalar('selfplay/pool_size', len(opp_pool), step)
            logger.writer.add_scalar('selfplay/opponent_avg_diff', avg_diff, step)
            # Elo distribution summary every 100k steps
            if step % 100_000 < args.opponent_update_freq:
                print(f"[Elo] {opp_pool.elo_summary()}")

        # --- Eval ---
        if step % cfg.eval_freq < n_envs:
            agent.save(run_dir / "agent.pt")
            eval_agent = _fresh_agent_from_checkpoint(cfg, run_dir / "agent.pt")
            _run_eval(eval_agent, cfg, step, start_time, logger, args.dynamics,
                      frame_stack=frame_stack)
            del eval_agent

        # --- Batched planning (with action chunking) ---
        if step > cfg.seed_steps:
            t0_mask = torch.from_numpy(t0_all)  # per-env t0 flags

            # Replan if: first step in chunk, or any env just reset
            need_replan = chunk_step == 0 or np.any(t0_all)
            if need_replan:
                # Concurrent agent + opponent planning via CUDA streams
                if self_play and agent_stream is not None:
                    opp_obs_np = batch_env.mirror_obs(obs_all[sp_mask])
                    opp_obs_t = torch.from_numpy(opp_obs_np).float()
                    t0_sp = torch.from_numpy(t0_all[sp_mask])
                    with torch.cuda.stream(agent_stream):
                        actions_t = agent.act(obs_t, t0=t0_mask)
                    with torch.cuda.stream(opp_stream):
                        with torch.no_grad():
                            opp_actions_t = opponent.act(opp_obs_t, t0=t0_sp, eval_mode=True)
                    torch.cuda.synchronize()
                    opp_actions_np = opp_actions_t.numpy()
                    opp_tx, opp_ty = batch_env.mirror_action_to_opponent(opp_actions_np)
                    batch_env._ext_opp_target_x[sp_mask] = opp_tx
                    batch_env._ext_opp_target_y[sp_mask] = opp_ty
                else:
                    actions_t = agent.act(obs_t, t0=t0_mask)
                # Cache the refined plan for action chunking
                if replan_k > 1 and hasattr(agent, '_prev_mean_batch') and agent._prev_mean_batch is not None:
                    cached_plan = agent._prev_mean_batch.cpu()  # [N, H, A]
                chunk_step = 1
            else:
                # Reuse cached plan: take action at current chunk offset
                actions_t = cached_plan[:, chunk_step].clamp(-1, 1)
                chunk_step += 1

                # Opponent still needs to plan every step even during action chunking
                if self_play:
                    opp_obs_np = batch_env.mirror_obs(obs_all[sp_mask])
                    opp_obs_t = torch.from_numpy(opp_obs_np).float()
                    t0_sp = torch.from_numpy(t0_all[sp_mask])
                    with torch.no_grad():
                        opp_actions_t = opponent.act(opp_obs_t, t0=t0_sp, eval_mode=True)
                    opp_actions_np = opp_actions_t.numpy()
                    opp_tx, opp_ty = batch_env.mirror_action_to_opponent(opp_actions_np)
                    batch_env._ext_opp_target_x[sp_mask] = opp_tx
                    batch_env._ext_opp_target_y[sp_mask] = opp_ty

            if chunk_step >= replan_k:
                chunk_step = 0
        else:
            # Random actions during seed phase
            actions_t = torch.rand(n_envs, action_dim) * 2 - 1  # uniform [-1, 1]

        actions_np = actions_t.numpy() if isinstance(actions_t, torch.Tensor) else actions_t

        # --- Step all envs at once ---
        obs_all, raw_rewards, terminated, truncated, info = batch_env.step(actions_np)

        # Update reward annealing progress
        if use_curriculum:
            steps_in_stage_now = step - stage_step_start
            estimated = STAGE_PLATEAU_CONFIG[current_stage]["min_steps"] * 3
            reward_shaper.set_progress(steps_in_stage_now / estimated)
        elif self_play:
            # Linear decay of auxiliary rewards over 500k steps, goals stay constant
            anneal_steps = 500_000
            reward_shaper._anneal_decay = min(1.0, step / anneal_steps)
            reward_shaper._penalty_ramp = 1.0

        shaped_rewards = reward_shaper.compute(obs_all, raw_rewards, actions=actions_np, info=info)
        obs_t = torch.from_numpy(obs_all).float()

        done = terminated | truncated

        # --- Accumulate step-based metrics (across all envs) ---
        # 14-dim obs: puck at 0-1, paddle at 4-5
        puck_dist_step = np.hypot(
            obs_all[:, 0] - obs_all[:, 4],
            obs_all[:, 1] - obs_all[:, 5],
        )
        puck_speed_now = np.hypot(info["puck_vx"], info["puck_vy"])
        contacts_now = (puck_dist_step < 0.15) & ((puck_speed_now - prev_puck_speed_track) > 0.3)
        prev_puck_speed_track = puck_speed_now
        step_goals_scored += int((raw_rewards > 0).sum())
        step_goals_conceded += int((raw_rewards < 0).sum())
        step_contacts += int(contacts_now.sum())
        step_shaped_reward += float(shaped_rewards.sum())
        step_puck_dist_acc += float(puck_dist_step.sum())
        step_puck_dist_count += n_envs

        # --- Per-env episode tracking ---
        # Pre-compute batched tensors (avoids per-env tensor creation in the hot loop)
        obs_cpu = obs_t.cpu().unsqueeze(1)  # [N, 1, obs_dim]
        actions_cpu = (actions_t if actions_t.ndim > 1 else actions_t.unsqueeze(0)).unsqueeze(1)  # [N, 1, act_dim]
        rewards_t = torch.from_numpy(shaped_rewards).float().unsqueeze(1)  # [N, 1]
        terminated_t = torch.from_numpy(terminated.astype(np.float32)).unsqueeze(1)  # [N, 1]

        for i in range(n_envs):
            tds_list[i].append(TensorDict(
                obs=obs_cpu[i], action=actions_cpu[i],
                reward=rewards_t[i], terminated=terminated_t[i],
            batch_size=(1,)))

            if done[i]:
                # Log episode
                ep_reward = sum(td['reward'].item() for td in tds_list[i][1:])
                elapsed = time() - start_time
                score_a = int(info['score_agent'][i])
                score_o = int(info['score_opponent'][i])
                step_episodes_completed += 1
                if score_a > score_o:
                    step_wins += 1
                if self_play and (sp_mask is None or sp_mask[i]):
                    logger.record_outcome(score_a, score_o)
                    if opp_pool is not None and current_opp_idx >= 0:
                        opp_pool.record_outcome(current_opp_idx, score_a, score_o)
                logger.log_train(step, ep_reward, len(tds_list[i]),
                                 step / max(elapsed, 1), score_a, score_o)

                # Track per-episode stats for curriculum (plateau detector, stage advancement)
                stage_n = current_stage if use_curriculum else 3
                if stage_n >= 2:
                    if score_a > score_o:
                        stage_wins.append(1.0)
                    elif score_o > score_a:
                        stage_wins.append(0.0)
                stage_goals.append(score_a)

                # Add to replay buffer
                buffer.add(torch.cat(tds_list[i]))
                total_episodes += 1

                # Feed plateau detector for curriculum
                if use_curriculum:
                    plateau.add(ep_reward, steps=len(tds_list[i]))

                # Reset trajectory buffer for this env
                tds_list[i] = [TensorDict(
                    obs=obs_cpu[i], action=torch.full((1, action_dim), float('nan')),
                    reward=torch.tensor([float('nan')]),
                    terminated=torch.tensor([float('nan')]),
                batch_size=(1,))]

        # Auto-reset done envs in batch
        if np.any(done):
            new_obs = batch_env.auto_reset(terminated, truncated)
            if new_obs is not None:
                # Update obs for reset envs
                obs_all[done] = new_obs[done]
                obs_t = torch.from_numpy(obs_all).float()
                reset_vel_info = {
                    "puck_vx": batch_env.engine.puck_vx.copy(),
                    "puck_vy": batch_env.engine.puck_vy.copy(),
                }
                reward_shaper.reset(obs_all, mask=done, info=reset_vel_info)
                prev_puck_speed_track[done] = np.hypot(
                    reset_vel_info["puck_vx"][done], reset_vel_info["puck_vy"][done]
                )
                # Fix trajectory buffer: initial obs should be from reset, not terminal
                reset_obs_cpu = obs_t.cpu().unsqueeze(1)  # [N, 1, obs_dim]
                nan_act = torch.full((1, action_dim), float('nan'))
                nan_r = torch.tensor([float('nan')])
                nan_t = torch.tensor([float('nan')])
                for i in np.where(done)[0]:
                    tds_list[i][0] = TensorDict(
                        obs=reset_obs_cpu[i], action=nan_act,
                        reward=nan_r, terminated=nan_t,
                    batch_size=(1,))

        t0_all = done  # Next step is t0 for reset envs

        # --- Curriculum stage advancement ---
        if use_curriculum and current_stage < max_stage:
            steps_in_stage = step - stage_step_start
            if steps_in_stage >= args.min_steps_per_stage and step % 10_000 < n_envs:
                is_plateau, curr_avg, old_avg = plateau.check()
                logger.log_curriculum(step, current_stage, STAGE_NAMES[current_stage],
                                      curr_avg, old_avg)
                if is_plateau:
                    # Check minimum goals requirement for game stages
                    min_goals = STAGE_MIN_GOALS.get(current_stage, 0)
                    avg_goals = sum(stage_goals) / max(len(stage_goals), 1) if stage_goals else 0.0
                    if min_goals > 0 and avg_goals < min_goals:
                        print(f"[Stage {current_stage}] Reward plateaued but avg goals "
                              f"({avg_goals:.1f}) below minimum ({min_goals}). "
                              f"Continuing training...")
                        # Reset plateau detector so it doesn't fire every check
                        plateau.configure_for_stage(current_stage)
                    else:
                        # --- Proceed with stage transition ---
                        # Save checkpoint synchronously before advancing
                        ckpt_path = run_dir / f"agent_stage_{current_stage}.pt"
                        agent.save(ckpt_path)
                        agent.save(run_dir / "agent.pt")  # Always keep latest
                        _save_training_state(run_dir, step, current_stage, stage_step_start,
                                             total_episodes, plateau=plateau)

                        # Record best play at end of this stage
                        # agent.pt was just saved above — load fresh uncompiled copy
                        rec_agent = _fresh_agent_from_checkpoint(cfg, run_dir / "agent.pt")
                        if current_stage == STAGE_SELFPLAY and opponent is not None:
                            opp_ckpt = run_dir / "opponent_tmp.pt"
                            opponent.save(opp_ckpt)
                            rec_opp = _fresh_agent_from_checkpoint(cfg, opp_ckpt)
                            _record_game_selfplay(
                                rec_agent, rec_opp, step, recordings_dir,
                                f"{args.run_name}_s{current_stage}",
                                stage=current_stage, frame_stack=frame_stack,
                            )
                            del rec_opp
                        else:
                            _record_game_pretrain(
                                rec_agent, step, recordings_dir,
                                f"{args.run_name}_s{current_stage}",
                                stage=current_stage, frame_stack=frame_stack,
                            )
                        del rec_agent

                        prev_stage = current_stage
                        current_stage += 1
                        stage_step_start = step
                        plateau.configure_for_stage(current_stage)
                        stage_wins.clear()
                        stage_goals.clear()

                        # Reset step-based accumulators for new stage
                        step_goals_scored = 0
                        step_goals_conceded = 0
                        step_contacts = 0
                        step_episodes_completed = 0
                        step_wins = 0
                        step_shaped_reward = 0.0
                        step_puck_dist_acc = 0.0
                        step_puck_dist_count = 0

                        # Update opponent policy, episode length, and score handicap
                        batch_env.opponent_policy = STAGE_OPPONENT[current_stage]
                        batch_env.max_episode_steps = STAGE_EPISODE_STEPS.get(current_stage, 1800)
                        batch_env.score_handicap = (current_stage == STAGE_SELFPLAY)

                        # Create new reward shaper
                        reward_shaper = _make_reward_shaper(current_stage)

                        # Reset LR scheduler for new stage
                        if lr_scheduler is not None:
                            new_lr = lr_scheduler.set_stage(
                                current_stage,
                                estimated_steps=STAGE_PLATEAU_CONFIG[current_stage]["min_steps"] * 3,
                            )
                            logger.writer.add_scalar('train/learning_rate', new_lr, step)

                        # Update planning horizon for new stage
                        new_horizon = STAGE_HORIZON.get(current_stage, cfg.horizon)
                        if new_horizon != cfg.horizon:
                            old_horizon = cfg.horizon
                            cfg.horizon = new_horizon
                            # Re-init agent's MPPI warm-start buffers for new horizon
                            agent._prev_mean = torch.nn.Buffer(
                                torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
                            )
                            agent._prev_mean_batch = None  # lazily re-initialized
                            if self_play and opponent is not None:
                                opponent.cfg.horizon = new_horizon
                                opponent._prev_mean = torch.nn.Buffer(
                                    torch.zeros(cfg.horizon, cfg.action_dim, device=opponent.device)
                                )
                                opponent._prev_mean_batch = None
                            # Recreate buffer — old data has (old_horizon+1)-step slices
                            # which are incompatible with the new slice length
                            buffer = Buffer(cfg)
                            print(f"  Planning horizon: {old_horizon} → {new_horizon}")
                            print(f"  Replay buffer recreated (slice length {old_horizon+1} → {new_horizon+1})")

                        # Full env reset to apply new opponent policy
                        obs_all = batch_env.reset()
                        # Re-stagger so episodes don't all complete at the same step
                        max_ep = STAGE_EPISODE_STEPS.get(current_stage, 1800)
                        batch_env._step_count[:] = np.linspace(
                            0, max_ep, n_envs, endpoint=False,
                        ).astype(np.int32)
                        stage_reset_info = {
                            "puck_vx": batch_env.engine.puck_vx.copy(),
                            "puck_vy": batch_env.engine.puck_vy.copy(),
                        }
                        reward_shaper.reset(obs_all, info=stage_reset_info)
                        prev_puck_speed_track = np.hypot(
                            stage_reset_info["puck_vx"], stage_reset_info["puck_vy"]
                        )
                        obs_t = torch.from_numpy(obs_all).float()
                        t0_all = np.ones(n_envs, dtype=bool)

                        # Reset per-env trajectory buffers
                        stage_obs_cpu = obs_t.cpu().unsqueeze(1)
                        nan_a = torch.full((1, action_dim), float('nan'))
                        nan_rv = torch.tensor([float('nan')])
                        for i in range(n_envs):
                            tds_list[i] = [TensorDict(
                                obs=stage_obs_cpu[i], action=nan_a,
                                reward=nan_rv, terminated=nan_rv,
                            batch_size=(1,))]

                        print(f"\n{'='*60}")
                        print(f"[Stage {prev_stage} → {current_stage}] "
                              f"Plateau at step {step:,}, avg_reward={curr_avg:.1f}, "
                              f"avg_goals={avg_goals:.1f} "
                              f"→ advancing to {STAGE_NAMES[current_stage]}")
                        print(f"  Saved checkpoint: {ckpt_path}")
                        print(f"  Opponent: {STAGE_OPPONENT[current_stage]}")
                        ep_steps = STAGE_EPISODE_STEPS.get(current_stage, 1800)
                        print(f"  Episode length: {ep_steps} steps (~{ep_steps/60:.0f}s at 60Hz)")
                        if STAGE_HORIZON.get(current_stage, 3) != STAGE_HORIZON.get(prev_stage, 3):
                            print(f"  Note: replay buffer was recreated (horizon changed)")
                        else:
                            print(f"  Note: replay buffer preserved (contains prior stage experience)")
                        print(f"{'='*60}\n")

                        logger.log_curriculum(step, current_stage, STAGE_NAMES[current_stage],
                                              transition=True)

        # --- PER beta annealing (0.4 → 1.0 over training) ---
        if args.prioritized_replay:
            beta = min(1.0, cfg.per_beta + (1.0 - cfg.per_beta) * step / cfg.steps)
            buffer.set_beta(beta)

        # --- Gradient update ---
        if step >= cfg.seed_steps and total_episodes >= 2:
            if not pretrained:
                num_updates = min(step, 5000)
                print(f'Pretraining agent on seed data ({num_updates} updates)...')
                for _ in range(num_updates):
                    agent.update(buffer)
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                pretrained = True
                print('Pretraining done.')
            else:
                train_info = None
                for _ in range(utd):
                    train_info = agent.update(buffer)
                if lr_scheduler is not None:
                    current_lr = lr_scheduler.step()
                    if step % 10_000 < n_envs:
                        logger.writer.add_scalar('train/learning_rate', current_lr, step)
                # World model diagnostics (every 10k steps)
                if train_info is not None and step % 10_000 < n_envs:
                    wm_errors = []
                    for t in range(cfg.horizon):
                        key = f"world_model_error_t{t}"
                        if key in train_info:
                            err = float(train_info[key])
                            logger.writer.add_scalar(f'diagnostics/{key}', err, step)
                            wm_errors.append(err)
                    if wm_errors:
                        logger.writer.add_scalar('diagnostics/world_model_error', sum(wm_errors) / len(wm_errors), step)

        step += n_envs

        # --- Periodic step-based logging (FPS, independent of episode boundaries) ---
        elapsed = time() - start_time
        logger.log_step(step, step / max(elapsed, 1), n_envs)

        # --- Step-based metrics (every 10k steps) ---
        if step % 10_000 < n_envs:
            stage_n = current_stage if use_curriculum else 3
            w = logger.writer
            w.add_scalar(f'stage{stage_n}/goals_per_10k', step_goals_scored, step)
            w.add_scalar(f'stage{stage_n}/conceded_per_10k', step_goals_conceded, step)
            w.add_scalar(f'stage{stage_n}/contacts_per_10k', step_contacts, step)
            w.add_scalar(f'stage{stage_n}/episodes_per_10k', step_episodes_completed, step)
            if step_episodes_completed > 0:
                w.add_scalar(f'stage{stage_n}/win_rate_10k',
                             step_wins / step_episodes_completed, step)
            w.add_scalar('train/reward_per_10k', step_shaped_reward, step)
            if step_puck_dist_count > 0:
                w.add_scalar(f'stage{stage_n}/avg_puck_distance_10k',
                             step_puck_dist_acc / step_puck_dist_count, step)
            # Reset accumulators
            step_goals_scored = 0
            step_goals_conceded = 0
            step_contacts = 0
            step_episodes_completed = 0
            step_wins = 0
            step_shaped_reward = 0.0
            step_puck_dist_acc = 0.0
            step_puck_dist_count = 0

        # --- Checkpoint ---
        if step > 0 and step % 100_000 < n_envs:
            ckpt_path = run_dir / f"agent_step_{step}.pt"
            agent.save(ckpt_path)
            agent.save(run_dir / "agent.pt")
            _save_training_state(run_dir, step, current_stage, stage_step_start,
                                 total_episodes,
                                 plateau=plateau if use_curriculum else None)

        # --- Graceful shutdown on signal ---
        if _shutdown:
            print(f"Shutting down at step {step:,}...")
            agent.save(run_dir / "agent.pt")
            _save_training_state(run_dir, step, current_stage, stage_step_start,
                                 total_episodes,
                                 plateau=plateau if use_curriculum else None)
            print(f"Checkpoint saved to {run_dir}")
            break

        # --- Record ---
        if step > 0 and step // args.record_freq > last_record_step // args.record_freq:
            last_record_step = step
            agent.save(run_dir / "agent.pt")
            rec_agent = _fresh_agent_from_checkpoint(cfg, run_dir / "agent.pt")
            rec_name = args.run_name
            if use_curriculum and current_stage is not None:
                rec_name = f"{args.run_name}_s{current_stage}"
            if self_play:
                opp_ckpt = run_dir / "opponent_tmp.pt"
                opponent.save(opp_ckpt)
                rec_opp = _fresh_agent_from_checkpoint(cfg, opp_ckpt)
                _record_game_selfplay(
                    rec_agent, rec_opp, step, recordings_dir, rec_name,
                    frame_stack=frame_stack,
                )
                del rec_opp
            else:
                _record_game_pretrain(
                    rec_agent, step, recordings_dir, rec_name,
                    frame_stack=frame_stack,
                )
            del rec_agent

        # --- Benchmark ---
        if args.benchmark_freq > 0 and step > 0 and step % args.benchmark_freq < n_envs:
            agent.save(run_dir / "agent.pt")
            bench_agent = _fresh_agent_from_checkpoint(cfg, run_dir / "agent.pt")
            _run_benchmark(bench_agent, step, logger, frame_stack=frame_stack)
            del bench_agent

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    logger.finish(agent)

    # Final recording (synchronous — end of training)
    agent.save(run_dir / "agent.pt")
    _save_training_state(run_dir, step, current_stage, stage_step_start,
                         total_episodes,
                         plateau=plateau if use_curriculum else None)
    final_agent = _fresh_agent_from_checkpoint(cfg, run_dir / "agent.pt")
    final_rec_name = args.run_name
    if use_curriculum and current_stage is not None:
        final_rec_name = f"{args.run_name}_s{current_stage}"
    if self_play:
        opp_ckpt = run_dir / "opponent_tmp.pt"
        opponent.save(opp_ckpt)
        final_opp = _fresh_agent_from_checkpoint(cfg, opp_ckpt)
        _record_game_selfplay(final_agent, final_opp, step, recordings_dir, final_rec_name,
                              frame_stack=frame_stack)
        del final_opp
    else:
        _record_game_pretrain(final_agent, step, recordings_dir, final_rec_name,
                              frame_stack=frame_stack)
    del final_agent

    if self_play:
        agent.save(run_dir / "agent_final.pt")
        total = logger.wins + logger.losses + logger.draws
        print(f"\nSelf-play complete! W={logger.wins} L={logger.losses} D={logger.draws} "
              f"WR={logger.wins / max(total, 1) * 100:.0f}%")
    else:
        print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
