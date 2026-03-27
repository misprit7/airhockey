"""Self-play training with TD-MPC2.

The agent trains against frozen copies of itself. The opponent checkpoint
is refreshed every `opponent_update_freq` steps. The opponent uses the
learned policy (no MPPI planning) for speed.

Usage:
    python bin/train_selfplay.py --resume runs/tdmpc2_v1/agent.pt
    python bin/train_selfplay.py --resume runs/tdmpc2_v1/agent.pt --steps 5000000
"""

from __future__ import annotations

import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'

import argparse
import copy
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

TDMPC2_DIR = Path("/tmp/tdmpc2/tdmpc2")
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import cfg_to_dataclass
from common.seed import set_seed
from common.buffer import Buffer
from common import MODEL_SIZE
from tdmpc2 import TDMPC2

from airhockey.dynamics import DelayedDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.rewards import ShapedRewardWrapper, STAGE_SCORING

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class SelfPlayEnv:
    """Air hockey env with external opponent control for self-play."""

    def __init__(self, use_dynamics: bool = True):
        dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else None
        from airhockey.dynamics import IdealDynamics
        if dynamics is None:
            dynamics = IdealDynamics()

        self.inner = AirHockeyEnv(
            agent_dynamics=dynamics,
            opponent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
            opponent_policy="external",
            record=False,
            action_dt=1 / 60,
            max_episode_time=30.0,
            max_score=7,
        )
        self.reward_wrapper = ShapedRewardWrapper(
            self.inner,
            stage=STAGE_SCORING,
            goal_reward=100.0,
            goal_penalty=-50.0,  # Prioritize defense
        )
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.max_episode_steps = 1800

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def reset(self, task_idx=None):
        obs, info = self.reward_wrapper.reset()
        return torch.from_numpy(obs).float()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        obs, reward, terminated, truncated, info = self.reward_wrapper.step(action)
        done = terminated or truncated
        info = defaultdict(float, info)
        info['success'] = float(info.get('score_agent', 0) > info.get('score_opponent', 0))
        info['terminated'] = torch.tensor(float(terminated))
        return torch.from_numpy(obs).float(), torch.tensor(reward, dtype=torch.float32), done, info

    def set_opponent_action(self, action):
        """Set opponent action from the opponent agent's perspective (mirrored)."""
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        target_x, target_y = self.inner.mirror_action_to_opponent(action)
        self.inner.set_opponent_action(target_x, target_y)

    def get_opponent_obs(self, obs):
        """Get mirrored observation for the opponent."""
        if isinstance(obs, torch.Tensor):
            obs_np = obs.numpy()
        else:
            obs_np = obs
        mirrored = self.inner.mirror_obs(obs_np)
        return torch.from_numpy(mirrored).float()

    def render(self, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)


def record_game(agent, opponent, step, recordings_dir, run_name, use_dynamics=True):
    """Record a self-play game for the web UI."""
    inner_env = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy="external",
        record=True,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    wrapped = ShapedRewardWrapper(inner_env, stage=STAGE_SCORING, goal_reward=100.0, goal_penalty=-50.0)
    obs, _ = wrapped.reset()
    obs_t = torch.from_numpy(obs).float()
    done = False
    t = 0
    while not done:
        with torch.no_grad():
            # Agent plans
            action = agent.act(obs_t, t0=(t == 0), eval_mode=True)
            # Opponent uses policy (no planning)
            opp_obs = inner_env.mirror_obs(obs)
            opp_obs_t = torch.from_numpy(opp_obs).float()
            opp_action = opponent.act(opp_obs_t, t0=(t == 0), eval_mode=True)
            target_x, target_y = inner_env.mirror_action_to_opponent(opp_action.numpy())
            inner_env.set_opponent_action(target_x, target_y)

        obs, _, terminated, truncated, info = wrapped.step(action.numpy())
        obs_t = torch.from_numpy(obs).float()
        done = terminated or truncated
        t += 1

    recording = inner_env.get_recording()
    if recording:
        rec = Recorder()
        rec._current = recording
        filename = f"{run_name}_step_{step:07d}.json"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        rec.save(recordings_dir / filename)
        score = f"{info['score_agent']}-{info['score_opponent']}"
        print(f"Recorded game at step {step:,}: {score}")


def main():
    parser = argparse.ArgumentParser(description="Self-play training with TD-MPC2")
    parser.add_argument("--resume", type=str, required=True, help="Path to pretrained agent.pt")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--model-size", type=int, default=5)
    parser.add_argument("--dynamics", action="store_true", default=True)
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--run-name", type=str, default="selfplay")
    parser.add_argument("--record-freq", type=int, default=50_000)
    parser.add_argument("--opponent-update-freq", type=int, default=100_000,
                        help="Update opponent checkpoint every N steps")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = Path("recordings")
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    # Build config
    from omegaconf import OmegaConf
    base_cfg = OmegaConf.load(str(TDMPC2_DIR / "config.yaml"))
    overrides = OmegaConf.create({
        "task": "airhockey-selfplay",
        "obs": "state",
        "episodic": True,
        "steps": args.steps,
        "model_size": args.model_size,
        "horizon": args.horizon,
        "eval_freq": 200_000,
        "eval_episodes": 3,
        "save_video": False,
        "enable_wandb": False,
        "save_csv": False,
        "work_dir": str(run_dir),
        "compile": False,
        "data_dir": str(run_dir / "data"),
        "exp_name": args.run_name,
        "discount_max": 0.99,
        "rho": 0.7,
        "task_title": "Air Hockey Self-Play",
        "multitask": False,
        "tasks": ["airhockey-selfplay"],
        "task_dim": 0,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)
    if args.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[args.model_size].items():
            cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg = cfg_to_dataclass(cfg)

    set_seed(cfg.seed)

    # Create env
    env = SelfPlayEnv(use_dynamics=args.dynamics)
    cfg = cfg_to_dataclass(OmegaConf.merge(
        OmegaConf.structured(cfg),
        OmegaConf.create({
            "obs_shape": {"state": list(env.observation_space.shape)},
            "action_dim": env.action_space.shape[0],
            "episode_length": env.max_episode_steps,
            "seed_steps": max(1000, 5 * env.max_episode_steps),
        })
    ))

    # Load agent
    print(f"Loading agent from {args.resume}")
    agent = TDMPC2(cfg)
    agent.load(args.resume)

    # Create opponent (frozen copy)
    opponent = TDMPC2(cfg)
    opponent.load(args.resume)
    print(f"Opponent initialized from {args.resume}")

    buffer = Buffer(cfg)

    print(f"\nSelf-Play TD-MPC2 Training")
    print(f"  Steps: {args.steps:,}")
    print(f"  Opponent update: every {args.opponent_update_freq:,} steps")
    print(f"  Goal reward: +100 / Goal conceded: -50")
    print(f"  Output: {run_dir}")
    print()

    # Training loop
    step = 0
    ep_idx = 0
    start_time = time()
    train_metrics = {}
    done = True
    last_record_step = 0
    last_opponent_update = 0
    opponent_version = 0
    wins, losses, draws = 0, 0, 0

    while step <= cfg.steps:
        # Update opponent checkpoint
        if step > 0 and step // args.opponent_update_freq > last_opponent_update // args.opponent_update_freq:
            last_opponent_update = step
            opponent_version += 1
            # Save current agent, load as opponent
            checkpoint_path = run_dir / f"agent_step_{step}.pt"
            agent.save(checkpoint_path)
            agent.save(run_dir / "agent.pt")  # latest for easy resuming
            opponent.load(checkpoint_path)
            print(f"[Step {step:,}] Updated opponent to version {opponent_version}")

        # Reset
        if done:
            if step > 0:
                ep_reward = torch.tensor([td['reward'] for td in tds[1:]]).sum()
                elapsed = time() - start_time
                fps = step / max(elapsed, 1)

                # Track win/loss
                score_a = env.inner.engine.state.score_agent
                score_o = env.inner.engine.state.score_opponent
                if score_a > score_o:
                    wins += 1
                elif score_o > score_a:
                    losses += 1
                else:
                    draws += 1

                writer.add_scalar('train/episode_reward', ep_reward.item(), step)
                writer.add_scalar('train/fps', fps, step)
                writer.add_scalar('train/wins', wins, step)
                writer.add_scalar('train/losses', losses, step)
                writer.add_scalar('train/win_rate', wins / max(wins + losses + draws, 1), step)
                writer.add_scalar('train/opponent_version', opponent_version, step)
                writer.flush()

                if step % 10000 < 2000:
                    wr = wins / max(wins + losses + draws, 1) * 100
                    print(f"[Train] step={step:,} reward={ep_reward.item():.1f} "
                          f"fps={fps:.0f} W/L/D={wins}/{losses}/{draws} WR={wr:.0f}%")

                ep_idx = buffer.add(torch.cat(tds))

            obs = env.reset()
            tds = [TensorDict(
                obs=obs.unsqueeze(0).cpu(),
                action=torch.full_like(env.rand_act(), float('nan')).unsqueeze(0),
                reward=torch.tensor(float('nan')).unsqueeze(0),
                terminated=torch.tensor(float('nan')).unsqueeze(0),
            batch_size=(1,))]

        # Get opponent action (policy only, no MPPI planning)
        with torch.no_grad():
            opp_obs = env.get_opponent_obs(obs)
            opp_action = opponent.act(opp_obs, t0=len(tds) == 1, eval_mode=True)
            env.set_opponent_action(opp_action)

        # Agent action
        if step > cfg.seed_steps:
            action = agent.act(obs, t0=len(tds) == 1)
        else:
            action = env.rand_act()

        obs, reward, done, info = env.step(action)
        tds.append(TensorDict(
            obs=obs.unsqueeze(0).cpu(),
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            terminated=info['terminated'].unsqueeze(0),
        batch_size=(1,)))

        # Update agent
        if step >= cfg.seed_steps:
            if step == cfg.seed_steps:
                num_updates = cfg.seed_steps
                print('Pretraining agent on seed data...')
            else:
                num_updates = 1
            for _ in range(num_updates):
                train_metrics = agent.update(buffer)

        # Record games
        if step > 0 and step // args.record_freq > last_record_step // args.record_freq:
            last_record_step = step
            record_game(agent, opponent, step, recordings_dir, args.run_name, args.dynamics)

        step += 1

    # Final save
    agent.save(run_dir / "agent_final.pt")
    record_game(agent, opponent, step, recordings_dir, args.run_name, args.dynamics)
    writer.close()
    print(f"\nSelf-play training complete! Final model: {run_dir / 'agent_final.pt'}")
    print(f"Final record: W={wins} L={losses} D={draws} WR={wins/max(wins+losses+draws,1)*100:.0f}%")


if __name__ == "__main__":
    main()
