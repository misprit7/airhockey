#!/usr/bin/env python3
"""Simulation-only arrival/RMS curriculum. Profiling precedes the long run.

No deployment registration: experimental checkpoints carry deployment_ready=false.
Run from the repository root, using the system Python used by play.sh.
"""

from __future__ import annotations

# ruff: noqa: E402  # configure torch and sibling TD-MPC2 imports before loading them
import os

os.environ["LAZY_LEGACY_OP"] = "0"
import argparse
import copy
import json
import signal
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ai"), str(ROOT.parent / "tdmpc2/tdmpc2")]
import numpy as np
import torch
from omegaconf import OmegaConf
from common import MODEL_SIZE
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from tdmpc2 import TDMPC2
from torch.utils.tensorboard import SummaryWriter
from airhockey.arrival_env import ArrivalEnv
from airhockey.arrival_training import transfer_encoder, load_demonstrations
from airhockey.sequence_replay import SequenceReplay, EpisodeBatch
from airhockey.skill_benchmark import make_fixtures
from airhockey.run_names import check_run_name
from airhockey.batch_env import _OPP_POLICY_MAP
from airhockey.thermal import DEFAULT_MODEL
from airhockey.arrival_recording import record_game


def config(args, run_dir):
    cfg = OmegaConf.load(ROOT.parent / "tdmpc2/tdmpc2/config.yaml")
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            dict(
                task="airhockey-arrival",
                obs="state",
                episodic=True,
                steps=args.steps,
                model_size=5,
                horizon=8,
                batch_size=256,
                work_dir=str(run_dir),
                data_dir=str(run_dir / "data"),
                exp_name=args.run_name,
                enable_wandb=False,
                save_video=False,
                save_csv=False,
                compile=False,
                discount_max=0.995,
                rho=0.7,
                task_title="Arrival skills and load-aware play",
                multitask=False,
                tasks=["airhockey-arrival"],
                task_dim=0,
                pi_smooth_coef=0.02,
                plan_smooth_coef=0.05,
                iterations=6,
                num_samples=256,
                bc_coef=5.0,
                prev_action_start=15,
                plan_eval_mean=False,
                obs_shape={"state": [ArrivalEnv.obs_dim]},
                action_dim=6,
                episode_length=1500,
                seed_steps=0,
                seed=args.seed,
            )
        ),
    )
    for k, v in MODEL_SIZE[5].items():
        cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    return cfg_to_dataclass(cfg)


def save(agent, path):
    tmp = path.with_suffix(".tmp")
    agent.save(tmp)
    tmp.replace(path)


def time_call(fn, n=15, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


def profile(agent, buffer, env, obs, output):
    """Same horizon/iterations/samples/precision; compile is the only planner change."""
    ot = torch.from_numpy(obs)
    t0 = torch.zeros(len(obs), dtype=torch.bool)

    def plan():
        torch.compiler.cudagraph_mark_step_begin()
        return agent.act(ot, t0=t0)

    rows = {}
    rows["n_envs"] = len(obs)
    rows["env"] = time_call(lambda: env.step(np.zeros((len(obs), 6), np.float32)))
    rows["replay_sample"] = time_call(buffer.sample_with_demo)
    rows["planner_eager"] = time_call(plan)
    agent._plan_batch = torch.compile(agent._plan_batch, mode="reduce-overhead")
    rows["planner_compiled"] = time_call(plan)
    rows["update_eager"] = time_call(lambda: agent.update(buffer))
    eager = agent._update
    try:
        agent._update = torch.compile(eager, mode="reduce-overhead")
        rows["update_compiled"] = time_call(lambda: agent.update(buffer))
    except Exception as exc:
        rows["update_compile_error"] = str(exc)
        agent._update = eager
    # Evaluate the frozen opponent's prior in one batch, no duplicate MPC search.
    agent.cfg.mpc = False
    rows["opponent_prior"] = time_call(lambda: agent.act(ot, t0=t0, eval_mode=True))
    agent.cfg.mpc = True
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print("[profile]", json.dumps(rows), flush=True)
    return rows


@torch.no_grad()
def evaluate(agent, step, output, *, per_task=8, seed=20261020, planner=False):
    """Held-out skills, all attempts counted; distinguish prior from MPC results."""
    f = make_fixtures(seed, per_task)
    n = len(f.task)
    env = ArrivalEnv(n, seed=seed, realistic=True, randomize=True)
    obs = env.reset(seed=seed, fixtures=f)
    finished = np.zeros(n, bool)
    success = np.zeros(n, bool)
    contacts = np.zeros(n, int)
    peak = np.zeros(n)
    effort = np.zeros(n)
    errors = np.full(n, np.nan)
    old = agent.cfg.mpc
    old_mean = agent._prev_mean_batch
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    agent.cfg.mpc = planner
    t0 = torch.ones(n, dtype=torch.bool)
    for tick in range(101):
        torch.compiler.cudagraph_mark_step_begin()
        action = agent.act(torch.from_numpy(obs), t0=t0, eval_mode=True).numpy()
        obs, _, term, trunc, info = env.step(action)
        active = ~finished
        peak[active] = np.maximum(peak[active], info["load_peak"][active])
        just = (term | trunc) & active
        success[just] = info["success"][just]
        contacts[just] = info["attempt_contacts"][just]
        effort[just] = info["effort"][just]
        errors[just] = info["goal_error"][just]
        finished |= just
        t0[:] = False
        if finished.all():
            break
    agent.cfg.mpc = old
    agent._prev_mean_batch = old_mean
    torch.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    result = {"step": step, "seed": seed, "planner": planner, "tasks": {}}
    for k, name in enumerate(("stationary", "moving", "cushion")):
        m = f.task == k
        valid = m & np.isfinite(errors)
        result["tasks"][name] = {
            "attempts": int(m.sum()),
            "successes": int(success[m].sum()),
            "contacts": int((contacts[m] > 0).sum()),
            "mean_effort": float(effort[m].mean()),
            "peak_load": float(peak[m].max()),
            "mean_actual_goal_error_m": float(errors[valid].mean())
            if valid.any()
            else None,
        }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("[eval]", json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--encoder-from", default="runs/3.12-accel60-accuracy-selfplay/agent.pt"
    )
    p.add_argument(
        "--demos", type=Path, default=Path("logs/arrival-training/demonstrations")
    )
    p.add_argument("--steps", type=int, default=6_000_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--pretrain-updates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--compile-update", action="store_true")
    p.add_argument("--no-compile-plan", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--buffer-capacity", type=int, default=1_000_000)
    args = p.parse_args()
    check_run_name(args.run_name)
    if min(args.steps, args.n_envs, args.checkpoint_every, args.eval_every) < 1:
        raise ValueError("counts must be positive")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = config(args, run_dir)
    metadata = dict(
        started=datetime.now().isoformat(),
        deployment_ready=False,
        action_mode="arrival",
        obs_dim=ArrivalEnv.obs_dim,
        action_dim=6,
        model_size=5,
        horizon=8,
        action_hz=50,
        agent_speed_range=[12, 12],
        agent_accel_range=[60, 60],
        thermal_model=json.loads(DEFAULT_MODEL.read_text()),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        curriculum={
            "first_500k": "80% skills, 20% scripted games",
            "500k_to_2m": "50% skills, 50% scripted games",
            "after_2m": "25% skills, 75% games; one third of game opponents frozen self prior",
        },
        optimizer={
            "updates_per_vector_step": 1,
            "batch_size": 256,
            "planning_iterations": 6,
            "planning_samples": 256,
            "opponent_planning": "frozen prior; scripted opponents retained",
        },
        source_hashes={},
    )
    for path in [
        Path(__file__),
        ROOT / "ai/airhockey/arrival_env.py",
        ROOT / "ai/airhockey/thermal.py",
        ROOT / "ai/airhockey/arrival.py",
        ROOT / "ai/airhockey/batch_env.py",
        ROOT / "ai/airhockey/batch_physics.py",
        ROOT / "ai/airhockey/sequence_replay.py",
        ROOT / "ai/airhockey/arrival_training.py",
        ROOT / "ai/airhockey/arrival_recording.py",
        ROOT.parent / "tdmpc2/tdmpc2/tdmpc2.py",
    ]:
        metadata["source_hashes"][str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        dest = run_dir / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    agent = TDMPC2(cfg)
    metadata["migration"] = transfer_encoder(agent, args.encoder_from)
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    demos = load_demonstrations(args.demos)
    buffer = SequenceReplay(
        args.buffer_capacity,
        ArrivalEnv.obs_dim,
        6,
        cfg.horizon,
        cfg.batch_size,
        seed=args.seed,
    )
    for ep in demos:
        buffer.add(ep)
    print(
        f"[init] {len(demos)} successful demo episodes, {buffer.size} rows; encoder transfer only",
        flush=True,
    )
    env = ArrivalEnv(args.n_envs, seed=args.seed, game_fraction=0.2)
    obs = env.reset(seed=args.seed)
    if args.profile_only:
        profile(agent, buffer, env, obs, run_dir / "profile.json")
        return
    if not args.no_compile_plan:
        agent._plan_batch = torch.compile(agent._plan_batch, mode="reduce-overhead")
    if args.compile_update:
        agent._update = torch.compile(agent._update, mode="reduce-overhead")
    writer = SummaryWriter(str(run_dir / "logs"))
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Supervised successes first train the new transition/reward/action semantics.
    warm_start = time.perf_counter()
    for k in range(args.pretrain_updates):
        metrics = agent.update(buffer)
        if k % 100 == 0:
            values = {
                key: float(metrics[key]) for key in ("total_loss", "pi_bc", "pi_loss")
            }
            if not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError("nonfinite pretrain loss")
            print(
                f"[pretrain] update={k} elapsed={time.perf_counter() - warm_start:.1f}s {values}",
                flush=True,
            )
        if stopped:
            break
    save(agent, run_dir / "agent_pretrained.pt")
    save(agent, run_dir / "agent.pt")
    evaluate(agent, 0, run_dir / "eval_prior_0000000.json")
    # A separate fixed-shape prior is inexpensive and cannot mutate main MPC state.
    opp_cfg = copy.deepcopy(cfg)
    opp_cfg.mpc = False
    opponent = TDMPC2(opp_cfg)
    opponent.load(run_dir / "agent.pt")
    cfg.bc_coef = 2.0
    # The compiled update specializes once more for this fixed main-phase BC
    # weight. Preserve the already compiled planner and other graph caches.
    staging = EpisodeBatch(args.n_envs, env.obs_dim, 6, max_steps=1502)
    staging.reset(obs)
    t0 = torch.ones(args.n_envs, dtype=torch.bool)
    step = 0
    updates = 0
    start = time.perf_counter()
    last_log_time, last_log_step = start, 0
    timings = {k: 0.0 for k in ("opponent", "planner", "env", "replay", "update")}
    counts = np.zeros((4, 3), int)
    gf = ga = 0
    peak = 0.0
    heat_sum = 0.0
    episodes = 0
    next_ckpt = args.checkpoint_every
    next_eval = args.eval_every
    with (run_dir / "metrics.jsonl").open("w", buffering=1) as log:
        while step < args.steps and not stopped:
            env.game_fraction = (
                0.2 if step < 500_000 else (0.5 if step < 2_000_000 else 0.75)
            )
            env.selfplay_fraction = 0.0 if step < 2_000_000 else 1 / 3
            ts = time.perf_counter()
            if np.any(env.base._opp_policy_id == _OPP_POLICY_MAP["external"]):
                with torch.no_grad():
                    opp = opponent.act(
                        torch.from_numpy(env.opponent_obs()), eval_mode=True
                    )
                env.set_opponent_action(opp.numpy())
            timings["opponent"] += time.perf_counter() - ts
            ts = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            action = agent.act(torch.from_numpy(obs), t0=t0).numpy()
            timings["planner"] += time.perf_counter() - ts
            ts = time.perf_counter()
            nxt, reward, term, trunc, info = env.step(action)
            done = term | trunc
            timings["env"] += time.perf_counter() - ts
            ts = time.perf_counter()
            staging.append(nxt, action, reward, term)
            peak = max(peak, float(info["load_peak"].max()))
            heat_sum += float(info["load_cost"].sum())
            if done.any():
                for i in np.flatnonzero(done):
                    ep = staging.episode_at(i)
                    buffer.add(ep)
                    task = info["task"][i]
                    counts[task] += [
                        1,
                        int(info["success"][i]),
                        int(info["attempt_contacts"][i] > 0),
                    ]
                    if task == 3:
                        gf += int(info["score_agent"][i])
                        ga += int(info["score_opponent"][i])
                    episodes += 1
                reset = env.reset(mask=done)
                # Keep continuing rows' observation velocities from the original step.
                nxt[done] = reset[done]
                staging.reset(nxt, mask=done)
            obs = nxt
            t0 = torch.from_numpy(done.copy())
            timings["replay"] += time.perf_counter() - ts
            ts = time.perf_counter()
            metrics = agent.update(buffer)
            updates += 1
            timings["update"] += time.perf_counter() - ts
            step += args.n_envs
            # Keep skill successes represented as replay turns over. They remain
            # real 50 Hz trajectories; never mark learner failures as demos.
            if updates % 250 == 0:
                for _ in range(4):
                    buffer.add(demos[int(env.rng.integers(len(demos)))])
            if step % 10000 < args.n_envs:
                torch.cuda.synchronize()
                now = time.perf_counter()
                vals = {
                    k: float(metrics[k])
                    for k in ("total_loss", "pi_bc", "pi_loss", "grad_norm")
                }
                if not np.isfinite(obs).all() or not all(
                    np.isfinite(v) for v in vals.values()
                ):
                    raise RuntimeError("nonfinite training state/loss")
                row = dict(
                    step=step,
                    updates=updates,
                    fps=step / (now - start),
                    recent_fps=(step - last_log_step) / (now - last_log_time),
                    elapsed_s=now - start,
                    episodes=episodes,
                    task_counts=counts.tolist(),
                    goals_for=gf,
                    goals_against=ga,
                    peak_load=peak,
                    load_cost=heat_sum,
                    timing_s=timings.copy(),
                    loss=vals,
                )
                log.write(json.dumps(row) + "\n")
                print("[train]", json.dumps(row), flush=True)
                for k, v in vals.items():
                    writer.add_scalar("loss/" + k, v, step)
                writer.add_scalar("train/fps", row["fps"], step)
                writer.add_scalar("load/peak", peak, step)
                writer.flush()
                last_log_time, last_log_step = now, step
            if step >= next_ckpt:
                save(agent, run_dir / f"agent_step_{step:07d}.pt")
                save(agent, run_dir / "agent.pt")
                opponent.load(run_dir / "agent.pt")
                next_ckpt += args.checkpoint_every
                print(f"[checkpoint] step={step}", flush=True)
            if step >= next_eval:
                evaluate(agent, step, run_dir / f"eval_prior_{step:07d}.json")
                evaluate(
                    agent, step, run_dir / f"eval_planner_{step:07d}.json", planner=True
                )
                record_game(agent, step, args.run_name)
                next_eval += args.eval_every
        save(agent, run_dir / "agent.pt")
        save(agent, run_dir / ("agent_interrupted.pt" if stopped else "agent_final.pt"))
        (run_dir / "status.json").write_text(
            json.dumps(
                dict(
                    step=step,
                    updates=updates,
                    complete=not stopped,
                    elapsed_s=time.perf_counter() - start,
                ),
                indent=2,
            )
        )
    evaluate(agent, step, run_dir / f"eval_prior_{step:07d}.json")
    if not stopped:
        evaluate(agent, step, run_dir / f"eval_planner_{step:07d}.json", planner=True)
        record_game(agent, step, args.run_name)
    writer.close()
    print(f"[done] step={step} interrupted={stopped}", flush=True)


if __name__ == "__main__":
    main()
