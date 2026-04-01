"""Profile GPU utilization in the TD-MPC2 training pipeline.

Measures:
1. GPU utilization during act() and update()
2. CPU↔GPU transfer costs
3. Buffer storage location (GPU vs CPU)
4. Batch sizes hitting GPU
5. Idle time between act() and update()
"""

from __future__ import annotations

import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'

import sys
import warnings
from pathlib import Path
from time import time, perf_counter

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

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def build_cfg(n_envs=32, num_samples=128, iterations=3, horizon=3):
    """Build a TD-MPC2 config for profiling."""
    from omegaconf import OmegaConf

    base_cfg = OmegaConf.load(str(TDMPC2_DIR / "config.yaml"))
    overrides = OmegaConf.create({
        "task": "airhockey",
        "obs": "state",
        "episodic": True,
        "steps": 100_000,
        "model_size": 5,
        "horizon": horizon,
        "eval_freq": 999_999,
        "eval_episodes": 0,
        "save_video": False,
        "enable_wandb": False,
        "save_csv": False,
        "work_dir": "/tmp/tdmpc2_profile",
        "compile": False,
        "data_dir": "/tmp/tdmpc2_profile/data",
        "exp_name": "profile",
        "discount_max": 0.99,
        "rho": 0.7,
        "task_title": "Air Hockey",
        "multitask": False,
        "tasks": ["airhockey"],
        "task_dim": 0,
        "num_samples": num_samples,
        "iterations": iterations,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)

    if 5 in MODEL_SIZE:
        for k, v in MODEL_SIZE[5].items():
            cfg[k] = v

    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.obs_shape = {"state": [14]}
    cfg.action_dim = 2
    cfg.episode_length = 1800
    cfg.seed_steps = max(1000, 5 * 1800)

    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)
    return cfg


def profile_act(agent, obs_t, n_warmup=5, n_measure=20):
    """Profile the act() method (planning) with CUDA timing."""
    t0_mask = torch.ones(obs_t.shape[0], dtype=torch.bool)

    # Warmup
    for _ in range(n_warmup):
        agent.act(obs_t, t0=t0_mask)

    torch.cuda.synchronize()

    # Measure with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]

    for i in range(n_measure):
        start_events[i].record()
        agent.act(obs_t, t0=(i == 0))
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


def profile_update(agent, buffer, n_warmup=5, n_measure=20):
    """Profile the update() method with CUDA timing."""
    # Warmup
    for _ in range(n_warmup):
        agent.update(buffer)

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]

    for i in range(n_measure):
        start_events[i].record()
        agent.update(buffer)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


def profile_transfers(n_envs=32):
    """Profile CPU↔GPU transfer costs."""
    # Simulate obs transfer (float32, [N, 14])
    obs_cpu = torch.randn(n_envs, 14)
    obs_gpu = torch.randn(n_envs, 14, device='cuda')

    # Warmup
    for _ in range(10):
        obs_cpu.to('cuda', non_blocking=True)
        obs_gpu.cpu()
    torch.cuda.synchronize()

    # Measure CPU→GPU
    n_reps = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_reps):
        obs_cpu.to('cuda', non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    cpu_to_gpu_us = start.elapsed_time(end) / n_reps * 1000  # microseconds

    # Measure GPU→CPU
    start.record()
    for _ in range(n_reps):
        obs_gpu.cpu()
    end.record()
    torch.cuda.synchronize()
    gpu_to_cpu_us = start.elapsed_time(end) / n_reps * 1000

    return cpu_to_gpu_us, gpu_to_cpu_us


def fill_buffer_with_seed_data(agent, batch_env, buffer, cfg, n_envs):
    """Fill buffer with random-action episodes (seed phase)."""
    obs_np = batch_env.reset()
    obs_t = torch.from_numpy(obs_np).float()

    nan_action = torch.full((1, 2), float('nan'))
    nan_scalar = torch.tensor(float('nan')).unsqueeze(0)

    tds_list = []
    for i in range(n_envs):
        tds_list.append([TensorDict(
            obs=obs_t[i].unsqueeze(0).cpu(),
            action=nan_action,
            reward=nan_scalar,
            terminated=nan_scalar,
        batch_size=(1,))])

    step = 0
    episodes = 0
    target_episodes = max(10, cfg.seed_steps // 100)

    while episodes < target_episodes:
        actions = torch.rand(n_envs, 2) * 2 - 1
        actions_np = actions.numpy()

        obs_np, raw_rewards, terminated, truncated, info = batch_env.step(actions_np)
        done = terminated | truncated
        obs_t = torch.from_numpy(obs_np).float()

        for i in range(n_envs):
            tds_list[i].append(TensorDict(
                obs=obs_t[i].unsqueeze(0).cpu(),
                action=actions[i].unsqueeze(0),
                reward=torch.tensor(0.0).unsqueeze(0),
                terminated=torch.tensor(float(terminated[i])).unsqueeze(0),
            batch_size=(1,)))

            if done[i]:
                buffer.add(torch.cat(tds_list[i]))
                episodes += 1
                tds_list[i] = [TensorDict(
                    obs=obs_t[i].unsqueeze(0).cpu(),
                    action=nan_action,
                    reward=nan_scalar,
                    terminated=nan_scalar,
                batch_size=(1,))]

        if np.any(done):
            new_obs = batch_env.auto_reset(terminated, truncated)
            if new_obs is not None:
                obs_np[done] = new_obs[done]
                obs_t = torch.from_numpy(obs_np).float()

        step += n_envs

    return obs_t, step


def main():
    n_envs = 32
    cfg = build_cfg(n_envs=n_envs)

    print("=" * 70)
    print("GPU UTILIZATION AUDIT - TD-MPC2 Training Pipeline")
    print("=" * 70)

    # --- GPU Info ---
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    mem_free, mem_total = torch.cuda.mem_get_info(0)
    print(f"Memory: {mem_free / 1e9:.1f} GB free / {mem_total / 1e9:.1f} GB total")

    # --- Agent & Env ---
    agent = TDMPC2(cfg)
    batch_env = BatchAirHockeyEnv(
        n_envs=n_envs,
        agent_dynamics="delayed",
        opponent_dynamics="delayed",
        opponent_policy="idle",
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
        dynamics_max_speed=3.0,
        dynamics_max_accel=30.0,
    )

    # --- Model Parameter Count ---
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model device: {next(agent.model.parameters()).device}")

    # --- Buffer ---
    buffer = Buffer(cfg)

    print(f"\n--- Section 1: Buffer Storage ---")
    print(f"Filling replay buffer with seed data...")
    obs_t, seed_steps = fill_buffer_with_seed_data(agent, batch_env, buffer, cfg, n_envs)
    print(f"Buffer filled with {seed_steps} steps")

    # Check buffer storage device
    sample = buffer._buffer.storage._storage
    if hasattr(sample, 'device'):
        print(f"Buffer storage device: {sample.device}")
    else:
        # Try getting a sample
        batch = buffer.sample()
        if hasattr(batch, 'device'):
            print(f"Buffer sample device: {batch.device}")
        else:
            print(f"Buffer type: {type(buffer._buffer.storage)}")
            try:
                storage = buffer._buffer.storage
                print(f"  Storage type: {type(storage)}")
                print(f"  Capacity: {storage.max_size if hasattr(storage, 'max_size') else 'unknown'}")
            except:
                pass

    mem_after_buffer = torch.cuda.memory_allocated() / 1e6
    print(f"GPU memory allocated after buffer: {mem_after_buffer:.1f} MB")

    # --- Do pretrain warmup ---
    print(f"\nPretraining agent (100 updates for warmup)...")
    for _ in range(100):
        agent.update(buffer)
    print("Pretrain done.")

    # --- Section 2: Transfer Costs ---
    print(f"\n--- Section 2: CPU↔GPU Transfer Costs ---")
    cpu_to_gpu_us, gpu_to_cpu_us = profile_transfers(n_envs)
    print(f"CPU→GPU obs transfer ({n_envs}×12 float32): {cpu_to_gpu_us:.1f} µs")
    print(f"GPU→CPU action transfer: {gpu_to_cpu_us:.1f} µs")

    # --- Section 3: act() Profiling ---
    print(f"\n--- Section 3: act() Profiling (Planning) ---")
    print(f"Config: {cfg.num_samples} samples, {cfg.iterations} iterations, horizon {cfg.horizon}")
    print(f"Batch size: {n_envs} envs × {cfg.num_samples} samples = {n_envs * cfg.num_samples} forward passes")

    act_times = profile_act(agent, obs_t, n_warmup=5, n_measure=30)
    act_mean = np.mean(act_times)
    act_std = np.std(act_times)
    act_min = np.min(act_times)
    print(f"act() time: {act_mean:.1f} ± {act_std:.1f} ms (min: {act_min:.1f} ms)")
    print(f"Per-env act: {act_mean / n_envs:.2f} ms")

    # --- Section 4: update() Profiling ---
    print(f"\n--- Section 4: update() Profiling (Gradient Step) ---")
    print(f"Batch size: {cfg.batch_size} trajectories × {cfg.horizon + 1} steps")

    update_times = profile_update(agent, buffer, n_warmup=5, n_measure=30)
    upd_mean = np.mean(update_times)
    upd_std = np.std(update_times)
    upd_min = np.min(update_times)
    print(f"update() time: {upd_mean:.1f} ± {upd_std:.1f} ms (min: {upd_min:.1f} ms)")

    # --- Section 5: env.step() Profiling ---
    print(f"\n--- Section 5: env.step() Profiling (Physics) ---")
    obs_np = batch_env.reset()
    step_times = []
    for _ in range(100):
        actions = np.random.uniform(-1, 1, size=(n_envs, 2)).astype(np.float32)
        t0 = perf_counter()
        batch_env.step(actions)
        step_times.append((perf_counter() - t0) * 1000)
    env_mean = np.mean(step_times)
    print(f"env.step() time: {env_mean:.2f} ms ({n_envs} envs)")

    # --- Section 6: Full Step Timing ---
    print(f"\n--- Section 6: Full Training Step Breakdown ---")
    obs_np = batch_env.reset()
    obs_t = torch.from_numpy(obs_np).float()

    # Measure one complete training step
    timings = {'env_step': [], 'act': [], 'update': [], 'overhead': []}

    for i in range(30):
        t_total = perf_counter()

        # act()
        t0 = perf_counter()
        torch.cuda.synchronize()
        actions_t = agent.act(obs_t, t0=(i == 0))
        torch.cuda.synchronize()
        timings['act'].append((perf_counter() - t0) * 1000)

        actions_np = actions_t.numpy()

        # env.step()
        t0 = perf_counter()
        obs_np, raw_rewards, terminated, truncated, info = batch_env.step(actions_np)
        obs_t = torch.from_numpy(obs_np).float()
        timings['env_step'].append((perf_counter() - t0) * 1000)

        # update()
        t0 = perf_counter()
        torch.cuda.synchronize()
        agent.update(buffer)
        torch.cuda.synchronize()
        timings['update'].append((perf_counter() - t0) * 1000)

        # Auto-reset
        done = terminated | truncated
        if np.any(done):
            batch_env.auto_reset(terminated, truncated)

        t_total = (perf_counter() - t_total) * 1000
        overhead = t_total - timings['act'][-1] - timings['env_step'][-1] - timings['update'][-1]
        timings['overhead'].append(overhead)

    print(f"  act():     {np.mean(timings['act']):6.1f} ± {np.std(timings['act']):4.1f} ms")
    print(f"  env.step():{np.mean(timings['env_step']):6.1f} ± {np.std(timings['env_step']):4.1f} ms")
    print(f"  update():  {np.mean(timings['update']):6.1f} ± {np.std(timings['update']):4.1f} ms")
    print(f"  overhead:  {np.mean(timings['overhead']):6.1f} ± {np.std(timings['overhead']):4.1f} ms")
    total_step = np.mean(timings['act']) + np.mean(timings['env_step']) + np.mean(timings['update']) + np.mean(timings['overhead'])
    print(f"  TOTAL:     {total_step:6.1f} ms per step ({n_envs} env steps)")
    print(f"  Throughput: {n_envs / total_step * 1000:.0f} env steps/sec")

    act_pct = np.mean(timings['act']) / total_step * 100
    env_pct = np.mean(timings['env_step']) / total_step * 100
    upd_pct = np.mean(timings['update']) / total_step * 100
    print(f"\n  Time split: act={act_pct:.0f}% env={env_pct:.0f}% update={upd_pct:.0f}%")

    # --- Section 7: GPU Memory ---
    print(f"\n--- Section 7: GPU Memory Usage ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    mem_free_now, _ = torch.cuda.mem_get_info(0)
    print(f"Free: {mem_free_now / 1e9:.1f} GB")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    bottleneck = "act()" if act_pct > upd_pct else "update()"
    print(f"Bottleneck: {bottleneck} ({max(act_pct, upd_pct):.0f}% of step time)")
    print(f"GPU memory headroom: {mem_free_now / 1e9:.1f} GB free — room for more envs")
    if env_pct > 20:
        print(f"CPU physics is {env_pct:.0f}% of step time — may benefit from GPU physics")
    if act_pct > 60:
        print("Consider: action chunking (replan every K steps) to amortize planning cost")
    if upd_pct > 60:
        print("Consider: reducing gradient_steps or update frequency")


if __name__ == "__main__":
    main()
