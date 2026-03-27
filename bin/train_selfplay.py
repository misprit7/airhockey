"""Self-play training with TD-MPC2 and parallel environments.

The agent trains against frozen copies of itself. Both agent and opponent
use full MPPI planning. Multiple environments run in parallel.

Usage:
    python bin/train_selfplay.py --resume runs/tdmpc2_pretrain/agent.pt
    python bin/train_selfplay.py --resume runs/tdmpc2_pretrain/agent.pt --steps 5000000 --n-envs 32
"""

from __future__ import annotations

import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

TDMPC2_DIR = Path(__file__).resolve().parent.parent.parent / "tdmpc2" / "tdmpc2"
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import cfg_to_dataclass
from common.seed import set_seed
from common.buffer import Buffer
from common import MODEL_SIZE
from tdmpc2 import TDMPC2

from airhockey.dynamics import DelayedDynamics, IdealDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.rewards import ShapedRewardWrapper, STAGE_SCORING

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def make_selfplay_env(use_dynamics=True):
    dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else IdealDynamics()
    inner = AirHockeyEnv(
        agent_dynamics=dynamics,
        opponent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else IdealDynamics(),
        opponent_policy="external",
        record=False,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    wrapped = ShapedRewardWrapper(inner, stage=STAGE_SCORING, goal_reward=100.0, goal_penalty=-50.0)
    return wrapped, inner


def record_game(agent, opponent, step, recordings_dir, run_name, use_dynamics=True):
    """Record a self-play game for the web UI."""
    inner = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy="external",
        record=True,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    wrapped = ShapedRewardWrapper(inner, stage=STAGE_SCORING, goal_reward=100.0, goal_penalty=-50.0)
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
        rec.save(recordings_dir / f"{run_name}_step_{step:07d}.json")
        score = f"{info['score_agent']}-{info['score_opponent']}"
        print(f"Recorded game at step {step:,}: {score}")


def main():
    parser = argparse.ArgumentParser(description="Self-play training with TD-MPC2")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--model-size", type=int, default=5)
    parser.add_argument("--dynamics", action="store_true", default=True)
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--run-name", type=str, default="selfplay")
    parser.add_argument("--record-freq", type=int, default=50_000)
    parser.add_argument("--opponent-update-freq", type=int, default=50_000)
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
        "obs": "state", "episodic": True,
        "steps": args.steps, "model_size": args.model_size,
        "horizon": args.horizon,
        "eval_freq": 200_000, "eval_episodes": 3,
        "save_video": False, "enable_wandb": False, "save_csv": False,
        "work_dir": str(run_dir), "compile": False,
        "data_dir": str(run_dir / "data"), "exp_name": args.run_name,
        "discount_max": 0.99, "rho": 0.7,
        "task_title": "Air Hockey Self-Play",
        "multitask": False, "tasks": ["airhockey-selfplay"], "task_dim": 0,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)
    if args.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[args.model_size].items():
            cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

    # Create one env to get shapes
    sample_wrapped, sample_inner = make_selfplay_env(args.dynamics)
    obs_shape = sample_inner.observation_space.shape
    action_dim = sample_inner.action_space.shape[0]
    episode_length = 1800

    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "obs_shape": {"state": list(obs_shape)},
        "action_dim": action_dim,
        "episode_length": episode_length,
        "seed_steps": max(1000, 5 * episode_length),
    }))
    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)

    # Load agents
    print(f"Loading agent from {args.resume}")
    agent = TDMPC2(cfg)
    agent.load(args.resume)
    opponent = TDMPC2(cfg)
    opponent.load(args.resume)

    buffer = Buffer(cfg)

    # Create parallel environments
    n_envs = args.n_envs
    envs = []  # list of (wrapped, inner) pairs
    for _ in range(n_envs):
        envs.append(make_selfplay_env(args.dynamics))

    print(f"\nSelf-Play TD-MPC2 Training")
    print(f"  Steps: {args.steps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Opponent update: every {args.opponent_update_freq:,} steps")
    print(f"  Planning horizon: {args.horizon}")
    print(f"  Goal reward: +100 / Goal conceded: -50")
    print(f"  Output: {run_dir}")
    print()

    # Per-env state
    obs_list = []
    tds_list = []
    t0_list = []
    for wrapped, inner in envs:
        obs, _ = wrapped.reset()
        obs_t = torch.from_numpy(obs).float()
        obs_list.append(obs_t)
        rand_act = torch.from_numpy(inner.action_space.sample().astype(np.float32))
        tds_list.append([TensorDict(
            obs=obs_t.unsqueeze(0).cpu(),
            action=torch.full_like(rand_act, float('nan')).unsqueeze(0),
            reward=torch.tensor(float('nan')).unsqueeze(0),
            terminated=torch.tensor(float('nan')).unsqueeze(0),
        batch_size=(1,))])
        t0_list.append(True)

    step = 0
    start_time = time()
    last_record_step = 0
    last_opponent_update = 0
    opponent_version = 0
    wins, losses, draws = 0, 0, 0

    while step <= cfg.steps:
        # Update opponent
        if step > 0 and step // args.opponent_update_freq > last_opponent_update // args.opponent_update_freq:
            last_opponent_update = step
            opponent_version += 1
            ckpt = run_dir / f"agent_step_{step}.pt"
            agent.save(ckpt)
            agent.save(run_dir / "agent.pt")
            opponent.load(ckpt)
            print(f"[Step {step:,}] Updated opponent to version {opponent_version}")

        # Step all envs
        for i, (wrapped, inner) in enumerate(envs):
            obs_t = obs_list[i]
            is_t0 = t0_list[i]

            # Opponent plans
            with torch.no_grad():
                opp_obs = torch.from_numpy(inner.mirror_obs(obs_t.numpy())).float()
                opp_action = opponent.act(opp_obs, t0=is_t0, eval_mode=True)
                tx, ty = inner.mirror_action_to_opponent(opp_action.numpy())
                inner.set_opponent_action(tx, ty)

            # Agent plans
            if step > cfg.seed_steps:
                action = agent.act(obs_t, t0=is_t0)
            else:
                action = torch.from_numpy(inner.action_space.sample().astype(np.float32))

            obs, reward, terminated, truncated, info = wrapped.step(action.numpy())
            done = terminated or truncated
            obs_t_new = torch.from_numpy(obs).float()

            tds_list[i].append(TensorDict(
                obs=obs_t_new.unsqueeze(0).cpu(),
                action=action.unsqueeze(0),
                reward=torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
                terminated=torch.tensor(float(terminated)).unsqueeze(0),
            batch_size=(1,)))

            if done:
                # Log episode
                ep_reward = torch.tensor([td['reward'] for td in tds_list[i][1:]]).sum()
                score_a = inner.engine.state.score_agent
                score_o = inner.engine.state.score_opponent
                if score_a > score_o: wins += 1
                elif score_o > score_a: losses += 1
                else: draws += 1

                writer.add_scalar('train/episode_reward', ep_reward.item(), step)
                writer.add_scalar('train/win_rate', wins / max(wins + losses + draws, 1), step)
                writer.add_scalar('train/opponent_version', opponent_version, step)

                # Add to buffer
                buffer.add(torch.cat(tds_list[i]))

                # Reset
                obs, _ = wrapped.reset()
                obs_t_new = torch.from_numpy(obs).float()
                rand_act = torch.from_numpy(inner.action_space.sample().astype(np.float32))
                tds_list[i] = [TensorDict(
                    obs=obs_t_new.unsqueeze(0).cpu(),
                    action=torch.full_like(rand_act, float('nan')).unsqueeze(0),
                    reward=torch.tensor(float('nan')).unsqueeze(0),
                    terminated=torch.tensor(float('nan')).unsqueeze(0),
                batch_size=(1,))]
                t0_list[i] = True
            else:
                t0_list[i] = False

            obs_list[i] = obs_t_new

        # Update agent (once per batch of env steps)
        total_episodes = wins + losses + draws
        if step >= cfg.seed_steps and total_episodes >= 2:
            if not hasattr(main, '_pretrained'):
                num_updates = min(step, 5000)  # Pretrain on seed data
                print(f'Pretraining agent on seed data ({num_updates} updates)...')
                for _ in range(num_updates):
                    agent.update(buffer)
                main._pretrained = True
                print('Pretraining done.')
            else:
                agent.update(buffer)

        step += n_envs  # Each iteration steps all envs

        # Logging
        elapsed = time() - start_time
        fps = step / max(elapsed, 1)
        if step % 10000 < n_envs * 2:
            wr = wins / max(wins + losses + draws, 1) * 100
            print(f"[Train] step={step:,} fps={fps:.0f} W/L/D={wins}/{losses}/{draws} WR={wr:.0f}% opp_v{opponent_version}")
            writer.add_scalar('train/fps', fps, step)
            writer.flush()

        # Record
        if step > 0 and step // args.record_freq > last_record_step // args.record_freq:
            last_record_step = step
            record_game(agent, opponent, step, recordings_dir, args.run_name, args.dynamics)

    # Final
    agent.save(run_dir / "agent_final.pt")
    agent.save(run_dir / "agent.pt")
    record_game(agent, opponent, step, recordings_dir, args.run_name, args.dynamics)
    writer.close()
    total = wins + losses + draws
    print(f"\nSelf-play complete! W={wins} L={losses} D={draws} WR={wins/max(total,1)*100:.0f}%")
    print(f"Final model: {run_dir / 'agent_final.pt'}")


if __name__ == "__main__":
    main()
