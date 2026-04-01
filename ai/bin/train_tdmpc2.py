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
import copy
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

# Add tdmpc2 to path
TDMPC2_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tdmpc2" / "tdmpc2"
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.dynamics import DelayedDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.rewards import (
    BatchRewardShaper, ShapedRewardWrapper,
    STAGE_SCORING, STAGE_OPPONENT, STAGE_NAMES,
)

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class AirHockeyTDMPC2Wrapper:
    """Wraps our air hockey env for TD-MPC2 compatibility."""

    def __init__(self, use_dynamics: bool = True, stage: int = STAGE_SCORING, frame_stack: int = 1):
        dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else None
        from airhockey.dynamics import IdealDynamics
        if dynamics is None:
            dynamics = IdealDynamics()
        opponent_policy = STAGE_OPPONENT.get(stage, "idle")
        if opponent_policy == "external":
            opponent_policy = "idle"  # eval uses built-in opponents only
        self.env = ShapedRewardWrapper(
            AirHockeyEnv(
                agent_dynamics=dynamics,
                opponent_policy=opponent_policy,
                record=False,
                action_dt=1 / 60,
                max_episode_time=30.0,
                max_score=7,
                frame_stack=frame_stack,
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


def record_game(agent, env_factory, step, recordings_dir, run_name, stage=STAGE_SCORING, frame_stack=1):
    """Record a game for the web UI."""
    opponent_policy = STAGE_OPPONENT.get(stage, "idle")
    if opponent_policy == "external":
        opponent_policy = "idle"  # can't use external agent for recording
    inner_env = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy=opponent_policy,
        record=True,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
        frame_stack=frame_stack,
    )
    wrapped = ShapedRewardWrapper(inner_env, stage=stage)
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
        filename = f"{run_name}_s{stage}_step_{step:07d}.json"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        rec.save(recordings_dir / filename)
        score = f"{info['score_agent']}-{info['score_opponent']}"
        print(f"Recorded game at step {step:,}: {score}")


def _run_eval(agent, cfg, step, start_time, logger, use_dynamics, stage=STAGE_SCORING, frame_stack=1):
    """Run eval episodes in background thread and log results."""
    env = AirHockeyTDMPC2Wrapper(use_dynamics=use_dynamics, stage=stage, frame_stack=frame_stack)
    ep_rewards = []
    t = 0
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


def _save_checkpoint(agent, *paths):
    """Save agent checkpoint to one or more paths in background thread."""
    for p in paths:
        agent.save(p)


def main():
    parser = argparse.ArgumentParser(description="Train with TD-MPC2")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--model-size", type=int, default=5, choices=[1, 5, 19, 48])
    parser.add_argument("--dynamics", action="store_true", default=True)
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--run-name", type=str, default="tdmpc2")
    parser.add_argument("--record-freq", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon (steps to look ahead)")
    parser.add_argument("--fast", action="store_true", help="Use reduced MPPI params for faster planning (num_samples=128, iterations=3, horizon=3)")
    parser.add_argument("--num-samples", type=int, default=None, help="MPPI trajectory samples (default: 512, --fast: 128)")
    parser.add_argument("--mppi-iterations", type=int, default=None, help="MPPI refinement iterations (default: 6, --fast: 3)")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--updates-per-step", type=int, default=1,
                        help="Gradient updates per batch of env steps (default 1; try 4-8 for higher UTD)")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Curriculum stage (1-4, default 2 for backwards compat)")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack in observations (default: 1)")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    recordings_dir = Path("recordings")
    n_envs = args.n_envs

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
        "rho": 0.85,
        "task_title": "Air Hockey",
        "multitask": False,
        "tasks": ["airhockey"],
        "task_dim": 0,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)

    # Apply MPPI overrides (--fast sets defaults, individual flags override)
    if args.fast:
        cfg.num_samples = 128
        cfg.iterations = 3
        cfg.horizon = 3
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.mppi_iterations is not None:
        cfg.iterations = args.mppi_iterations

    # Apply model size params
    if args.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[args.model_size].items():
            cfg[k] = v

    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

    # Create vectorized batch env
    stage = args.stage
    dyn_type = "delayed" if args.dynamics else "ideal"
    opponent_policy = STAGE_OPPONENT[stage]
    frame_stack = args.frame_stack
    batch_env = BatchAirHockeyEnv(
        n_envs=n_envs,
        agent_dynamics=dyn_type,
        opponent_dynamics=dyn_type,
        opponent_policy=opponent_policy,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
        dynamics_max_speed=3.0,
        dynamics_max_accel=30.0,
        frame_stack=frame_stack,
    )
    reward_shaper = BatchRewardShaper(n_envs, stage=stage, frame_stack=frame_stack)

    obs_dim = 14  # positions(4*3) + velocities + context(2)
    cfg.obs_shape = {"state": (obs_dim,)}
    cfg.action_dim = 2
    cfg.episode_length = 1800  # 30s at 60Hz
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)

    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)

    print(f"TD-MPC2 Training")
    print(f"  Stage: {stage} — {STAGE_NAMES[stage]}")
    print(f"  Opponent: {opponent_policy}")
    print(f"  Steps: {args.steps:,}")
    print(f"  Parallel envs: {n_envs} (batch vectorized)")
    print(f"  Model size: {args.model_size}M params")
    print(f"  Planning horizon: {cfg.horizon}")
    print(f"  MPPI samples: {cfg.num_samples}, iterations: {cfg.iterations}")
    print(f"  Dynamics: {args.dynamics}")
    print(f"  Output: {run_dir}")
    print()

    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = SimpleLogger(cfg)

    # Initialize all envs
    obs_np = batch_env.reset()
    init_info = {"puck_vx": batch_env.engine.puck_vx.copy(),
                 "puck_vy": batch_env.engine.puck_vy.copy()}
    reward_shaper.reset(obs_np, info=init_info)
    obs_t = torch.from_numpy(obs_np).float()

    # Per-env episode tracking (needed for buffer which stores full episodes)
    nan_action = torch.full((1, 2), float('nan'))
    nan_scalar = torch.tensor(float('nan')).unsqueeze(0)
    tds_list = []
    t0_list = [True] * n_envs
    for i in range(n_envs):
        tds_list.append([TensorDict(
            obs=obs_t[i].unsqueeze(0).cpu(),
            action=nan_action,
            reward=nan_scalar,
            terminated=nan_scalar,
        batch_size=(1,))])

    # --- Training loop ---
    step = 0
    total_episodes = 0
    start_time = time()
    last_record_step = 0

    # Async recording/eval/checkpointing
    executor = ThreadPoolExecutor(max_workers=2)
    pending_futures = []

    def _cleanup_futures():
        nonlocal pending_futures
        still_pending = []
        for f in pending_futures:
            if f.done():
                exc = f.exception()
                if exc:
                    print(f"Background task failed: {exc}")
            else:
                still_pending.append(f)
        pending_futures = still_pending

    while step <= cfg.steps:
        # Eval
        if step % cfg.eval_freq < n_envs:
            _cleanup_futures()
            if len(pending_futures) < 2:
                agent_copy = copy.deepcopy(agent)
                fut = executor.submit(
                    _run_eval, agent_copy, cfg, step, start_time, logger, args.dynamics, stage,
                    frame_stack=frame_stack,
                )
                pending_futures.append(fut)

        # Collect actions for all envs
        actions_np = np.empty((n_envs, 2), dtype=np.float32)
        actions_t = []
        for i in range(n_envs):
            if step > cfg.seed_steps:
                action = agent.act(obs_t[i], t0=t0_list[i])
            else:
                action = torch.from_numpy(
                    np.random.uniform(-1, 1, size=2).astype(np.float32)
                )
            actions_t.append(action)
            actions_np[i] = action.numpy()

        # Step all envs at once (vectorized physics)
        obs_np, raw_rewards, terminated, truncated, info = batch_env.step(actions_np)
        shaped_rewards = reward_shaper.compute(obs_np, raw_rewards, actions=actions_np, info=info)
        done = terminated | truncated

        # Per-env episode tracking
        for i in range(n_envs):
            obs_i = torch.from_numpy(obs_np[i]).float()
            tds_list[i].append(TensorDict(
                obs=obs_i.unsqueeze(0).cpu(),
                action=actions_t[i].unsqueeze(0),
                reward=torch.tensor(shaped_rewards[i], dtype=torch.float32).unsqueeze(0),
                terminated=torch.tensor(float(terminated[i])).unsqueeze(0),
            batch_size=(1,)))

            if done[i]:
                ep_reward = torch.tensor([td['reward'] for td in tds_list[i][1:]]).sum()
                elapsed = time() - start_time
                logger.log({
                    'step': step,
                    'episode_reward': ep_reward,
                    'episode_length': len(tds_list[i]),
                    'steps_per_second': step / max(elapsed, 1),
                }, 'train')
                buffer.add(torch.cat(tds_list[i]))
                total_episodes += 1

        # Auto-reset done envs and reinitialize their episode tracking
        if np.any(done):
            obs_np = batch_env.auto_reset(terminated, truncated)
            reset_info = {"puck_vx": batch_env.engine.puck_vx.copy(),
                         "puck_vy": batch_env.engine.puck_vy.copy()}
            reward_shaper.reset(obs_np, mask=done, info=reset_info)
            for i in np.where(done)[0]:
                obs_i = torch.from_numpy(obs_np[i]).float()
                tds_list[i] = [TensorDict(
                    obs=obs_i.unsqueeze(0).cpu(),
                    action=nan_action,
                    reward=nan_scalar,
                    terminated=nan_scalar,
                batch_size=(1,))]
                t0_list[i] = True

        # Update t0 flags for non-done envs
        for i in range(n_envs):
            if not done[i]:
                t0_list[i] = False

        obs_t = torch.from_numpy(obs_np).float()

        # Update agent (multiple gradient steps per batch of env steps)
        if step >= cfg.seed_steps and total_episodes >= 2:
            if not hasattr(main, '_pretrained'):
                num_updates = min(step, 5000)
                print(f'Pretraining agent on seed data ({num_updates} updates)...')
                for _ in range(num_updates):
                    agent.update(buffer)
                main._pretrained = True
                print('Pretraining done.')
            else:
                for _ in range(args.updates_per_step):
                    agent.update(buffer)

        step += n_envs

        # Checkpoint every 100k steps
        if step > 0 and step % 100_000 < n_envs:
            _cleanup_futures()
            agent_copy = copy.deepcopy(agent)
            ckpt_path = run_dir / f"agent_step_{step}.pt"
            executor.submit(_save_checkpoint, agent_copy, ckpt_path, run_dir / "agent.pt")

        # Record
        if step > 0 and step // args.record_freq > last_record_step // args.record_freq:
            last_record_step = step
            _cleanup_futures()
            if len(pending_futures) < 2:
                agent_copy = copy.deepcopy(agent)
                fut = executor.submit(
                    record_game, agent_copy, None, step, recordings_dir, args.run_name, stage,
                    frame_stack=frame_stack,
                )
                pending_futures.append(fut)

    # Wait for any pending background tasks
    for f in pending_futures:
        try:
            f.result()
        except Exception as e:
            print(f"Background task failed: {e}")

    # Final
    logger.finish(agent)
    record_game(agent, None, step, recordings_dir, args.run_name, stage, frame_stack=frame_stack)
    executor.shutdown(wait=True)
    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
