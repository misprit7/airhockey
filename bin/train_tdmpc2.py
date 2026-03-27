"""Train air hockey agent with TD-MPC2.

TD-MPC2 learns a world model and uses MPPI planning at each step,
enabling multi-step lookahead for bank shots and strategic play.

Usage:
    python bin/train_tdmpc2.py
    python bin/train_tdmpc2.py --steps 5000000 --model-size 5
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

# Add tdmpc2 to path
TDMPC2_DIR = Path("/tmp/tdmpc2/tdmpc2")
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2

from airhockey.dynamics import DelayedDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.rewards import ShapedRewardWrapper, STAGE_SCORING

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class AirHockeyTDMPC2Wrapper:
    """Wraps our air hockey env for TD-MPC2 compatibility."""

    def __init__(self, use_dynamics: bool = True, stage: int = STAGE_SCORING):
        dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else None
        from airhockey.dynamics import IdealDynamics
        if dynamics is None:
            dynamics = IdealDynamics()
        self.env = ShapedRewardWrapper(
            AirHockeyEnv(
                agent_dynamics=dynamics,
                opponent_policy="idle",
                record=False,
                action_dt=1 / 60,
                max_episode_time=30.0,
                max_score=7,
            ),
            stage=stage,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = 1800  # 30s at 60Hz

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def reset(self, task_idx=None):
        obs, info = self.env.reset()
        return torch.from_numpy(obs).float()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info = defaultdict(float, info)
        info['success'] = float(info.get('score_agent', 0) > info.get('score_opponent', 0))
        info['terminated'] = torch.tensor(float(terminated))
        return torch.from_numpy(obs).float(), torch.tensor(reward, dtype=torch.float32), done, info

    def render(self, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)  # Dummy for video


class SimpleLogger:
    """Logger with TensorBoard + stdout."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.work_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(log_dir))

    def log(self, metrics, category):
        step = metrics.get('step', 0)
        reward = metrics.get('episode_reward', 0)
        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        if category == 'eval':
            length = metrics.get('episode_length', 0)
            fps = metrics.get('steps_per_second', 0)
            print(f"[Eval] step={step:,} reward={reward:.1f} length={length:.0f} fps={fps:.0f}")
            self.writer.add_scalar('eval/episode_reward', reward, step)
            self.writer.add_scalar('eval/episode_length', length, step)
        elif category == 'train':
            fps = metrics.get('steps_per_second', 0)
            if step % 10000 < 2000:
                print(f"[Train] step={step:,} reward={reward:.1f} fps={fps:.0f}")
            self.writer.add_scalar('train/episode_reward', reward, step)
            self.writer.add_scalar('train/fps', fps, step)
        self.writer.flush()

    def finish(self, agent):
        agent_path = self.work_dir / 'agent.pt'
        agent.save(agent_path)
        print(f"Agent saved to {agent_path}")
        self.writer.close()

    class Video:
        def init(self, *a, **kw): pass
        def record(self, *a, **kw): pass
        def save(self, *a, **kw): pass

    @property
    def video(self):
        return self.Video()


def record_game(agent, env_factory, step, recordings_dir, run_name):
    """Record a game for the web UI."""
    inner_env = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy="idle",
        record=True,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    wrapped = ShapedRewardWrapper(inner_env, stage=STAGE_SCORING)
    obs, _ = wrapped.reset()
    obs_t = torch.from_numpy(obs).float()
    done = False
    t = 0
    while not done:
        with torch.no_grad():
            action = agent.act(obs_t, t0=(t == 0), eval_mode=True)
        obs, _, terminated, truncated, info = wrapped.step(action.numpy())
        obs_t = torch.from_numpy(obs).float()
        done = terminated or truncated
        t += 1

    recording = inner_env.get_recording()
    if recording:
        rec = Recorder()
        rec._current = recording
        step_str = f"{step:07d}"
        filename = f"{run_name}_step_{step_str}.json"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        rec.save(recordings_dir / filename)
        score = f"{info['score_agent']}-{info['score_opponent']}"
        print(f"Recorded game at step {step:,}: {score}")


def main():
    parser = argparse.ArgumentParser(description="Train with TD-MPC2")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--model-size", type=int, default=5, choices=[1, 5, 19, 48])
    parser.add_argument("--dynamics", action="store_true", default=True)
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--run-name", type=str, default="tdmpc2")
    parser.add_argument("--record-freq", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon (steps to look ahead)")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    recordings_dir = Path("recordings")

    # Build config using OmegaConf, bypassing Hydra
    from omegaconf import OmegaConf
    from common import MODEL_SIZE
    from common.parser import cfg_to_dataclass

    base_cfg = OmegaConf.load(str(TDMPC2_DIR / "config.yaml"))
    overrides = OmegaConf.create({
        "task": "airhockey",
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
        "compile": False,
        "data_dir": str(run_dir / "data"),
        "exp_name": args.run_name,
        "discount_max": 0.99,
        "rho": 0.7,
        "task_title": "Air Hockey",
        "multitask": False,
        "tasks": ["airhockey"],
        "task_dim": 0,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)

    # Apply model size params
    if args.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[args.model_size].items():
            cfg[k] = v

    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)

    # Create env
    env = AirHockeyTDMPC2Wrapper(use_dynamics=args.dynamics)
    cfg.obs_shape = {"state": env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)

    print(f"TD-MPC2 Training")
    print(f"  Steps: {args.steps:,}")
    print(f"  Model size: {args.model_size}M params")
    print(f"  Planning horizon: {args.horizon}")
    print(f"  Dynamics: {args.dynamics}")
    print(f"  Output: {run_dir}")
    print()

    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = SimpleLogger(cfg)

    # --- Training loop (adapted from OnlineTrainer) ---
    step = 0
    ep_idx = 0
    start_time = time()
    train_metrics = {}
    done = True
    eval_next = False
    last_record_step = 0

    while step <= cfg.steps:
        # Eval
        if step % cfg.eval_freq == 0:
            eval_next = True

        # Reset
        if done:
            if eval_next:
                ep_rewards = []
                for _ in range(cfg.eval_episodes):
                    obs, eval_done, ep_reward, t = env.reset(), False, 0, 0
                    while not eval_done:
                        torch.compiler.cudagraph_mark_step_begin()
                        action = agent.act(obs, t0=t == 0, eval_mode=True)
                        obs, reward, eval_done, info = env.step(action)
                        ep_reward += reward
                        t += 1
                    ep_rewards.append(ep_reward)
                elapsed = time() - start_time
                logger.log({
                    'step': step,
                    'episode_reward': np.nanmean([r.item() if isinstance(r, torch.Tensor) else r for r in ep_rewards]),
                    'episode_length': t,
                    'steps_per_second': step / max(elapsed, 1),
                }, 'eval')
                eval_next = False

            if step > 0:
                ep_reward = torch.tensor([td['reward'] for td in tds[1:]]).sum()
                elapsed = time() - start_time
                logger.log({
                    'step': step,
                    'episode_reward': ep_reward,
                    'episode_length': len(tds),
                    'steps_per_second': step / max(elapsed, 1),
                }, 'train')
                ep_idx = buffer.add(torch.cat(tds))

            obs = env.reset()
            tds = [TensorDict(
                obs=obs.unsqueeze(0).cpu(),
                action=torch.full_like(env.rand_act(), float('nan')).unsqueeze(0),
                reward=torch.tensor(float('nan')).unsqueeze(0),
                terminated=torch.tensor(float('nan')).unsqueeze(0),
            batch_size=(1,))]

        # Collect
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

        # Update
        if step >= cfg.seed_steps:
            if step == cfg.seed_steps:
                num_updates = cfg.seed_steps
                print('Pretraining agent on seed data...')
            else:
                num_updates = 1
            for _ in range(num_updates):
                train_metrics = agent.update(buffer)

        # Checkpoint every 100k steps
        if step > 0 and step % 100_000 == 0:
            ckpt_path = run_dir / f"agent_step_{step}.pt"
            agent.save(ckpt_path)
            # Also save as latest for easy resuming
            agent.save(run_dir / "agent.pt")

        # Record
        if step > 0 and step // args.record_freq > last_record_step // args.record_freq:
            last_record_step = step
            record_game(agent, None, step, recordings_dir, args.run_name)

        step += 1

    # Final
    logger.finish(agent)
    record_game(agent, None, step, recordings_dir, args.run_name)
    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
