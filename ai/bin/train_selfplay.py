#!/usr/bin/env python3
"""Self-play training with TD-MPC2 -- the March methodology on the NEW env.

The recipe that produced selfplay_v2 (2026-03-27, the reference "it worked"
run), kept knob for knob:
  * resume from a pretrained agent (train_tdmpc2.py, 500k vs scripted)
  * the opponent is the agent's OWN latest checkpoint, reloaded every 50k
    steps -- no Elo pool, no mix, no handicap
  * stage-2 auxiliary shaping with goals at +100 / -50, constant
  * 30 s games to 7, horizon 5, 5M-parameter model, 32 envs
  * one gradient update per vectorised step (March's UTD)

What changed is only the WORLD it runs in, deliberately: measured physics,
the firmware motion law inside the cable workspace, the human-model
opponent body, 200 Hz camera sensing with latency/noise/blind spot, and
domain randomisation -- i.e. BatchAirHockeyEnv's defaults plus
sensing_kwargs(True). The March envs were 32 scalar AirHockeyEnvs stepped
in a Python loop; this steps one batch env and plans both sides in one
batched MPPI call, which changes speed, not data.

    python ai/bin/train_selfplay.py --resume runs/classic_pretrain/agent.pt
"""

from __future__ import annotations

import os
os.environ['LAZY_LEGACY_OP'] = '0'

import argparse
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

TDMPC2_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tdmpc2" / "tdmpc2"
sys.path.insert(0, str(TDMPC2_DIR))

from common.parser import cfg_to_dataclass  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.buffer import Buffer  # noqa: E402
from common import MODEL_SIZE  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs  # noqa: E402
from airhockey.recorder import FrameData, Recorder  # noqa: E402
from airhockey.rewards import (BatchRewardShaper, STAGE_SCORING,  # noqa: E402
                               CURRICULUM, curriculum_shaper_kwargs)

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# Weights come from the named curriculum's self-play stage so the pretrain
# stages and this one share one table (rewards.CURRICULUM).
_SP = CURRICULUM["selfplay"]
GOAL_REWARD = _SP["goal_reward"]
GOAL_PENALTY = _SP["goal_penalty"]


def make_env(args, n_envs):
    return BatchAirHockeyEnv(
        n_envs=n_envs,
        agent_dynamics="profile" if args.dynamics else "ideal",
        opponent_dynamics="delayed" if args.dynamics else "ideal",
        opponent_policy="external",
        # Symmetric by default: the far side is a copy of the machine, not
        # the human model, so the learner is always playing itself.
        opponent_body="robot" if args.symmetric else "human",
        action_dt=1 / 100,
        max_episode_time=30.0,
        max_score=7,
        domain_randomize=args.domain_randomize,
        **sensing_kwargs(args.realistic_sensing),
    )


def _truth_info(env):
    e = env.engine
    return {"puck_x": e.puck_x, "puck_y": e.puck_y, "puck_vx": e.puck_vx,
            "puck_vy": e.puck_vy, "pad_x": e.paddle_agent_x,
            "pad_y": e.paddle_agent_y, "opp_x": e.paddle_opp_x,
            "opp_y": e.paddle_opp_y, "score_agent": e.score_agent,
            "score_opponent": e.score_opponent}


def drive_opponent(env, opponent, obs, t0_mask):
    """The opponent is the mirrored self: plan on the mirrored view, map the
    action back into the far half."""
    view = env.opponent_obs() if env.opponent_body == "robot" else env.mirror_obs(obs)
    with torch.no_grad():
        opp_obs = torch.from_numpy(view).float()
        opp_act = opponent.act(opp_obs, t0=t0_mask, eval_mode=True)
    tx, ty = env.mirror_action_to_opponent(opp_act.numpy())
    env._ext_opp_target_x[:] = tx
    env._ext_opp_target_y[:] = ty


def record_game(agent, opponent, step, recordings_dir, run_name, args):
    env = make_env(args, 1)
    obs = env.reset(seed=int(step) % 99_991)
    rec = Recorder()
    rec.start_episode()
    e = env.engine
    done, t = False, 0
    t0 = torch.ones(1, dtype=torch.bool)
    while not done and t < 9000:
        drive_opponent(env, opponent, obs, t0)
        with torch.no_grad():
            a = agent.act(torch.from_numpy(obs).float(), t0=t0, eval_mode=True)
        obs, _, term, trunc, _ = env.step(a.numpy())
        t0 = torch.zeros(1, dtype=torch.bool)
        done = bool(term[0] or trunc[0])
        rec.record(FrameData(
            time=float(e.time[0]),
            puck_x=float(e.puck_x[0]), puck_y=float(e.puck_y[0]),
            puck_vx=float(e.puck_vx[0]), puck_vy=float(e.puck_vy[0]),
            agent_x=float(e.paddle_agent_x[0]), agent_y=float(e.paddle_agent_y[0]),
            opponent_x=float(e.paddle_opp_x[0]), opponent_y=float(e.paddle_opp_y[0]),
            score_agent=int(e.score_agent[0]), score_opponent=int(e.score_opponent[0]),
        ))
        t += 1
    recordings_dir.mkdir(parents=True, exist_ok=True)
    rec.save(recordings_dir / f"{run_name}_step_{step:07d}.json",
             metadata={"step": int(step), "algo": "TD-MPC2", "opponent": "self"})
    print(f"Recorded game at step {step:,}: "
          f"{int(e.score_agent[0])}-{int(e.score_opponent[0])}")


def main():
    parser = argparse.ArgumentParser(description="Self-play training with TD-MPC2")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--model-size", type=int, default=5)
    parser.add_argument("--dynamics", action="store_true", default=True)
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--realistic-sensing", action="store_true", default=True)
    parser.add_argument("--no-realistic-sensing", dest="realistic_sensing",
                        action="store_false")
    parser.add_argument("--symmetric", action="store_true", default=True,
                        help="far side is a copy of the robot's body "
                             "(default); --human-opponent restores the "
                             "human model as the sparring partner")
    parser.add_argument("--human-opponent", dest="symmetric",
                        action="store_false")
    parser.add_argument("--domain-randomize", action="store_true", default=True)
    parser.add_argument("--no-domain-randomize", dest="domain_randomize",
                        action="store_false")
    parser.add_argument("--run-name", type=str, default="selfplay")
    parser.add_argument("--record-freq", type=int, default=50_000)
    parser.add_argument("--opponent-update-freq", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--updates-per-iter", type=int, default=1,
                        help="gradient updates per vectorised step; March "
                             "used 1 (i.e. one update per n_envs transitions)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_dir = root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = root / "ai" / "recordings"
    writer = SummaryWriter(str(run_dir / "logs"))

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

    n_envs = args.n_envs
    env = make_env(args, n_envs)
    shaper = BatchRewardShaper(n_envs, stage=STAGE_SCORING,
                               **curriculum_shaper_kwargs("selfplay"))
    episode_length = 3000          # 30 s at the 100 Hz action rate
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "obs_shape": {"state": [env.obs_dim]},
        "action_dim": env.action_dim,
        "episode_length": episode_length,
        "seed_steps": max(1000, 5 * episode_length),
    }))
    cfg = cfg_to_dataclass(cfg)
    set_seed(cfg.seed)

    print(f"Loading agent from {args.resume}")
    agent = TDMPC2(cfg)
    agent.load(args.resume)
    opponent = TDMPC2(cfg)
    opponent.load(args.resume)
    buffer = Buffer(cfg)

    print(f"\nSelf-Play TD-MPC2 Training (March recipe, new environment)")
    print(f"  Steps: {args.steps:,}   Parallel envs: {n_envs}")
    print(f"  Opponent update: every {args.opponent_update_freq:,} steps")
    print(f"  Far side: {'copy of self (robot body, mirrored workspace)' if args.symmetric else 'human model'}")
    print(f"  Planning horizon: {args.horizon}   Goals: +{GOAL_REWARD:.0f} / {GOAL_PENALTY:.0f}")
    print(f"  Sensing: {'on' if args.realistic_sensing else 'off'}   "
          f"DR: {'on' if args.domain_randomize else 'off'}   obs {env.obs_dim} dims")
    print(f"  Output: {run_dir}\n")

    obs = env.reset(seed=cfg.seed)
    shaper.reset(obs, info=_truth_info(env))
    t0_mask = torch.ones(n_envs, dtype=torch.bool)

    def fresh_td(o):
        return TensorDict(
            obs=torch.from_numpy(o).float().unsqueeze(0),
            action=torch.full((1, env.action_dim), float('nan')),
            reward=torch.tensor([float('nan')]),
            terminated=torch.tensor([float('nan')]),
            batch_size=(1,))
    tds = [[fresh_td(obs[i])] for i in range(n_envs)]

    step = 0
    start = time()
    last_record = 0
    last_opp_update = 0
    opp_version = 0
    wins = losses = draws = 0
    pretrained = False

    while step <= cfg.steps:
        if step > 0 and step // args.opponent_update_freq > last_opp_update // args.opponent_update_freq:
            last_opp_update = step
            opp_version += 1
            ckpt = run_dir / f"agent_step_{step:07d}.pt"
            agent.save(ckpt)
            agent.save(run_dir / "agent.pt")
            opponent.load(ckpt)
            print(f"[Step {step:,}] Updated opponent to version {opp_version}")

        drive_opponent(env, opponent, obs, t0_mask)
        if step > cfg.seed_steps:
            with torch.no_grad():
                actions = agent.act(torch.from_numpy(obs).float(), t0=t0_mask)
        else:
            actions = torch.from_numpy(
                np.random.uniform(-1, 1, (n_envs, env.action_dim)).astype(np.float32))
        act_np = actions.numpy().astype(np.float32)

        next_obs, raw, term, trunc, info = env.step(act_np)
        shaped = shaper.compute(next_obs, raw, actions=act_np, info=info)
        done = term | trunc

        for i in range(n_envs):
            tds[i].append(TensorDict(
                obs=torch.from_numpy(next_obs[i]).float().unsqueeze(0),
                action=actions[i].unsqueeze(0),
                reward=torch.tensor([float(shaped[i])], dtype=torch.float32),
                terminated=torch.tensor([float(term[i])]),
                batch_size=(1,)))

        if np.any(done):
            for i in np.where(done)[0]:
                ep_reward = float(sum(float(td['reward'][0]) for td in tds[i][1:]))
                sa, so = int(info["score_agent"][i]), int(info["score_opponent"][i])
                if sa > so: wins += 1
                elif so > sa: losses += 1
                else: draws += 1
                writer.add_scalar('train/episode_reward', ep_reward, step)
                writer.add_scalar('train/win_rate',
                                  wins / max(wins + losses + draws, 1), step)
                writer.add_scalar('train/opponent_version', opp_version, step)
                buffer.add(torch.cat(tds[i]))
            reset_obs = env.auto_reset(term, trunc)
            if reset_obs is not None:
                next_obs = reset_obs
                shaper.reset(next_obs, mask=done, info=_truth_info(env))
                for i in np.where(done)[0]:
                    tds[i] = [fresh_td(next_obs[i])]
        t0_mask = torch.from_numpy(done.copy())
        obs = next_obs

        total_eps = wins + losses + draws
        if step >= cfg.seed_steps and total_eps >= 2:
            if not pretrained:
                n_up = min(step, 5000)
                print(f'Pretraining agent on seed data ({n_up} updates)...')
                for _ in range(n_up):
                    agent.update(buffer)
                pretrained = True
                print('Pretraining done.')
            else:
                for _ in range(args.updates_per_iter):
                    agent.update(buffer)

        step += n_envs

        if step % 10000 < n_envs:
            fps = step / max(time() - start, 1)
            wr = wins / max(total_eps, 1) * 100
            print(f"[Train] step={step:,} fps={fps:.0f} W/L/D={wins}/{losses}/{draws} "
                  f"WR={wr:.0f}% opp_v{opp_version}")
            writer.add_scalar('train/fps', fps, step)
            writer.flush()

        if step > 0 and step // args.record_freq > last_record // args.record_freq:
            last_record = step
            record_game(agent, opponent, step, recordings_dir, args.run_name, args)

    agent.save(run_dir / "agent_final.pt")
    agent.save(run_dir / "agent.pt")
    record_game(agent, opponent, step, recordings_dir, args.run_name, args)
    writer.close()
    total = wins + losses + draws
    print(f"\nSelf-play complete! W={wins} L={losses} D={draws} "
          f"WR={wins / max(total, 1) * 100:.0f}%")
    print(f"Final model: {run_dir / 'agent_final.pt'}")


if __name__ == "__main__":
    main()
