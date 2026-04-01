"""Profile training loop components to identify optimization targets."""

from __future__ import annotations

import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'

import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

TDMPC2_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tdmpc2" / "tdmpc2"
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import cfg_to_dataclass
from common.seed import set_seed
from common.buffer import Buffer
from common import MODEL_SIZE
from tdmpc2 import TDMPC2

from airhockey.batch_env import BatchAirHockeyEnv

from airhockey.rewards import BatchRewardShaper, STAGE_SCORING

from omegaconf import OmegaConf
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

N_ENVS = 32


def main():
    base_cfg = OmegaConf.load(str(TDMPC2_DIR / "config.yaml"))
    overrides = OmegaConf.create({
        "task": "airhockey", "obs": "state", "episodic": True,
        "steps": 100000, "model_size": 5, "horizon": 3,
        "eval_freq": 999999, "eval_episodes": 1,
        "save_video": False, "enable_wandb": False, "save_csv": False,
        "work_dir": "/tmp/profile_run", "compile": False,
        "data_dir": "/tmp/profile_run/data", "exp_name": "profile",
        "discount_max": 0.99, "rho": 0.7, "task_title": "Profile",
        "multitask": False, "tasks": ["airhockey"], "task_dim": 0,
        "num_samples": 128, "iterations": 3,
        "obs_shape": {"state": [8]}, "action_dim": 2,
        "episode_length": 1800, "seed_steps": 9000,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)
    for k, v in MODEL_SIZE[5].items():
        cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)

    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    batch_env = BatchAirHockeyEnv(
        n_envs=N_ENVS, agent_dynamics="delayed", opponent_dynamics="delayed",
        opponent_policy="idle", action_dt=1/60, max_episode_time=30.0,
        max_score=7, dynamics_max_speed=3.0, dynamics_max_accel=30.0,
    )
    reward_shaper = BatchRewardShaper(N_ENVS, stage=STAGE_SCORING)

    obs_all = batch_env.reset()
    _reset_info = {
        "puck_vx": batch_env.engine.puck_vx.copy(),
        "puck_vy": batch_env.engine.puck_vy.copy(),
    }
    reward_shaper.reset(obs_all, info=_reset_info)
    obs_t = torch.from_numpy(obs_all).float()
    rand_act = torch.zeros(2)

    tds_list = []
    for i in range(N_ENVS):
        tds_list.append([TensorDict(
            obs=obs_t[i].unsqueeze(0).cpu(),
            action=torch.full_like(rand_act, float('nan')).unsqueeze(0),
            reward=torch.tensor(float('nan')).unsqueeze(0),
            terminated=torch.tensor(float('nan')).unsqueeze(0),
        batch_size=(1,))])

    print("Collecting seed data...")
    step = 0
    total_episodes = 0
    while step <= cfg.seed_steps or total_episodes < 2:
        actions_t = torch.rand(N_ENVS, 2) * 2 - 1
        actions_np = actions_t.numpy()
        obs_all, raw_rewards, terminated, truncated, info = batch_env.step(actions_np)
        shaped_rewards = reward_shaper.compute(obs_all, raw_rewards, info=info)
        obs_t = torch.from_numpy(obs_all).float()
        done = terminated | truncated
        for i in range(N_ENVS):
            tds_list[i].append(TensorDict(
                obs=obs_t[i].unsqueeze(0).cpu(),
                action=actions_t[i].unsqueeze(0),
                reward=torch.tensor(shaped_rewards[i], dtype=torch.float32).unsqueeze(0),
                terminated=torch.tensor(float(terminated[i])).unsqueeze(0),
            batch_size=(1,)))
            if done[i]:
                buffer.add(torch.cat(tds_list[i]))
                total_episodes += 1
                tds_list[i] = [TensorDict(
                    obs=obs_t[i].unsqueeze(0).cpu(),
                    action=torch.full_like(rand_act, float('nan')).unsqueeze(0),
                    reward=torch.tensor(float('nan')).unsqueeze(0),
                    terminated=torch.tensor(float('nan')).unsqueeze(0),
                batch_size=(1,))]
        if np.any(done):
            new_obs = batch_env.auto_reset(terminated, truncated)
            if new_obs is not None:
                obs_all[done] = new_obs[done]
                obs_t = torch.from_numpy(obs_all).float()
                _ri = {
                    "puck_vx": batch_env.engine.puck_vx.copy(),
                    "puck_vy": batch_env.engine.puck_vy.copy(),
                }
                reward_shaper.reset(obs_all, mask=done, info=_ri)
        step += N_ENVS

    print(f"Seed done: {step} steps, {total_episodes} episodes")
    print("Pretraining (500 updates)...")
    for _ in range(500):
        agent.update(buffer)
    print("Done.\n")

    # Profile 50 actual training steps
    t0_all = np.zeros(N_ENVS, dtype=bool)
    timings = {
        "act": [], "env_step": [], "reward": [], "obs_conv": [],
        "td_loop": [], "update": [], "total": [],
    }

    print("Profiling 50 training steps...")
    for s in range(50):
        step_t0 = time()

        # Act (includes CPU->GPU->CPU)
        t = time()
        t0_mask = torch.from_numpy(t0_all)
        actions_t = agent.act(obs_t, t0=t0_mask)
        torch.cuda.synchronize()
        timings["act"].append(time() - t)

        # Env step (numpy)
        t = time()
        actions_np = actions_t.numpy()
        obs_all, raw_rewards, terminated, truncated, info = batch_env.step(actions_np)
        timings["env_step"].append(time() - t)

        # Reward shaping (numpy)
        t = time()
        shaped_rewards = reward_shaper.compute(obs_all, raw_rewards, info=info)
        timings["reward"].append(time() - t)

        # Obs conversion
        t = time()
        obs_t = torch.from_numpy(obs_all).float()
        timings["obs_conv"].append(time() - t)

        # TensorDict per-env loop
        t = time()
        done = terminated | truncated
        for i in range(N_ENVS):
            tds_list[i].append(TensorDict(
                obs=obs_t[i].unsqueeze(0).cpu(),
                action=actions_t[i].unsqueeze(0),
                reward=torch.tensor(shaped_rewards[i], dtype=torch.float32).unsqueeze(0),
                terminated=torch.tensor(float(terminated[i])).unsqueeze(0),
            batch_size=(1,)))
            if done[i]:
                buffer.add(torch.cat(tds_list[i]))
                total_episodes += 1
                tds_list[i] = [TensorDict(
                    obs=obs_t[i].unsqueeze(0).cpu(),
                    action=torch.full_like(rand_act, float('nan')).unsqueeze(0),
                    reward=torch.tensor(float('nan')).unsqueeze(0),
                    terminated=torch.tensor(float('nan')).unsqueeze(0),
                batch_size=(1,))]
        if np.any(done):
            new_obs = batch_env.auto_reset(terminated, truncated)
            if new_obs is not None:
                obs_all[done] = new_obs[done]
                obs_t = torch.from_numpy(obs_all).float()
                _ri = {
                    "puck_vx": batch_env.engine.puck_vx.copy(),
                    "puck_vy": batch_env.engine.puck_vy.copy(),
                }
                reward_shaper.reset(obs_all, mask=done, info=_ri)
        t0_all = done
        timings["td_loop"].append(time() - t)

        # Gradient update
        t = time()
        agent.update(buffer)
        torch.cuda.synchronize()
        timings["update"].append(time() - t)

        timings["total"].append(time() - step_t0)

    print()
    print(f"=== Profiling Results (ms per step, {N_ENVS} envs) ===")
    for k, v in timings.items():
        arr = np.array(v) * 1000
        print(f"{k:12s}: mean={arr.mean():7.2f}  std={arr.std():6.2f}  "
              f"p50={np.median(arr):7.2f}  p95={np.percentile(arr, 95):7.2f}")

    total_arr = np.array(timings["total"]) * 1000
    env_steps_per_sec = N_ENVS / (total_arr.mean() / 1000)
    print(f"\nEffective FPS: {env_steps_per_sec:.0f} env-steps/sec")

    print("\n=== Percentage Breakdown ===")
    total_mean = total_arr.mean()
    for k, v in timings.items():
        if k != "total":
            pct = np.array(v).mean() * 1000 / total_mean * 100
            print(f"{k:12s}: {pct:5.1f}%")


if __name__ == "__main__":
    main()
