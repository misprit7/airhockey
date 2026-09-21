"""Watchable arrival-policy games, using the training simulator and decoder."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from airhockey.arrival_env import ArrivalEnv
from airhockey.recorder import FrameData, Recorder
from airhockey.thermal import DEFAULT_MODEL


RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "recordings"


def recording_path(run_name, step, directory=RECORDINGS_DIR, opponent="self"):
    suffix = "" if opponent == "self" else f"_vs_{opponent}"
    return Path(directory) / f"{run_name}{suffix}_step_{step:07d}.json"


@torch.no_grad()
def record_game(
    agent,
    step,
    run_name,
    *,
    directory=RECORDINGS_DIR,
    seed=20261021,
    duration=30.0,
    opponent="self",
    accel=60.0,
    thermal_path=DEFAULT_MODEL,
    checkpoint=None,
):
    """Record a full game with MPC, realistic sensing and randomized dynamics.

    A fixed held-out seed makes checkpoints comparable. Save atomically so the
    live viewer never sees partial JSON. Preserve the trainer's RNG and planner
    state, including when recording fails. This module has no hardware imports.
    """
    if opponent not in ("self", "sniper", "goalie", "weak_goalie") or duration <= 0:
        raise ValueError("a supported opponent and positive duration are required")
    selfplay = opponent == "self"
    old_mpc, old_mean = agent.cfg.mpc, agent._prev_mean_batch
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            agent.cfg.mpc = True
            # Do not let an in-place planner update touch the training warm start.
            agent._prev_mean_batch = None
            env = ArrivalEnv(
                1,
                seed=seed,
                accel=accel,
                game_fraction=1.0,
                realistic=True,
                randomize=True,
                thermal_path=thermal_path,
            )
            env.base.max_episode_time = duration
            obs = env.reset(seed=seed, opponent="external" if selfplay else opponent)
            rec = Recorder()
            e = env.engine
            total_reward = 0.0
            peak_load = float(env.loads[0].levels.max())

            def frame(reward=0.0):
                rec.record(
                    FrameData(
                        time=float(e.time[0]),
                        puck_x=float(e.puck_x[0]),
                        puck_y=float(e.puck_y[0]),
                        puck_vx=float(e.puck_vx[0]),
                        puck_vy=float(e.puck_vy[0]),
                        agent_x=float(e.paddle_agent_x[0]),
                        agent_y=float(e.paddle_agent_y[0]),
                        opponent_x=float(e.paddle_opp_x[0]),
                        opponent_y=float(e.paddle_opp_y[0]),
                        score_agent=int(e.score_agent[0]),
                        score_opponent=int(e.score_opponent[0]),
                        reward=float(reward),
                        cumulative_reward=total_reward,
                    )
                )

            frame()
            t0 = torch.ones(2 if selfplay else 1, dtype=torch.bool)
            for _ in range(int(np.ceil(duration / env.base.action_dt))):
                torch.compiler.cudagraph_mark_step_begin()
                # One batched call gives each side independent MPC warm starts
                # with the SAME weights and planning budget. Both bodies run
                # the arrival decoder, firmware law, workspace and limits.
                views = np.concatenate((obs, env.opponent_obs())) if selfplay else obs
                action = agent.act(
                    torch.from_numpy(views), t0=t0, eval_mode=True
                ).numpy()
                if selfplay:
                    env.set_opponent_action(action[1:])
                obs, reward, term, trunc, info = env.step(action[:1])
                total_reward += float(reward[0])
                peak_load = max(peak_load, float(info["load_peak"][0]))
                frame(reward[0])
                t0[:] = False
                if term[0] or trunc[0]:
                    break
            path = recording_path(run_name, step, directory, opponent)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            rec.save(
                tmp,
                metadata={
                    "run": run_name,
                    "step": int(step),
                    "algo": "TD-MPC2 arrival",
                    "opponent": opponent,
                    "match_type": "self-play" if selfplay else "scripted opponent",
                    "opponent_planner": selfplay,
                    "opponent_body": "robot"
                    if selfplay or opponent == "goalie"
                    else "free scripted body",
                    "policy_sides": "same checkpoint, MPC on both sides"
                    if selfplay
                    else "learned MPC vs script",
                    "planner": True,
                    "action_mode": "arrival",
                    "fps": 1 / env.base.action_dt,
                    "seed": seed,
                    "realistic_sensing": True,
                    "domain_randomize": True,
                    "accel_cap_m_s2": accel,
                    "peak_modeled_load": peak_load,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "simulation_only": True,
                },
            )
            tmp.replace(path)
            print(
                f"[recording] {path.name}: {e.score_agent[0]}-{e.score_opponent[0]}",
                flush=True,
            )
            return path
    finally:
        agent.cfg.mpc, agent._prev_mean_batch = old_mpc, old_mean
        np.random.set_state(np_state)
        random.setstate(py_state)


def pending_checkpoints(
    run_dir, *, every=500_000, directory=RECORDINGS_DIR, opponent="self"
):
    """Newest missing milestone first; also pick up a completed/interrupted run."""
    import json

    run_dir = Path(run_dir)
    candidates = {}
    for checkpoint in run_dir.glob("agent_step_*.pt"):
        step = int(checkpoint.stem.rsplit("_", 1)[1])
        if step % every == 0:
            candidates[step] = checkpoint
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        final = run_dir / (
            "agent_final.pt" if status["complete"] else "agent_interrupted.pt"
        )
        if final.exists():
            candidates[int(status["step"])] = final
    return [
        (step, checkpoint)
        for step, checkpoint in sorted(candidates.items(), reverse=True)
        if not recording_path(run_dir.name, step, directory, opponent).exists()
    ]
