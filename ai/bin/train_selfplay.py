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
from airhockey.dynamics import ACTION_DT  # noqa: E402
from airhockey.recorder import FrameData, Recorder  # noqa: E402
from airhockey.policy_loader import (PLAN_ITERATIONS, PLAN_SMOOTH_COEF,  # noqa: E402
                                     load_checkpoint)
from airhockey.rewards import (BatchRewardShaper, STAGE_SCORING,  # noqa: E402
                               CURRICULUM, curriculum_env_kwargs,
                               curriculum_shaper_kwargs)
from airhockey.batch_env import _OPP_POLICY_MAP  # noqa: E402
from airhockey.cushion_bot import CushionBot  # noqa: E402

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
# TF32 matmuls: 1.5x on the planner and the update on this GPU for a
# precision loss (10-bit mantissa) that a layer-normed RL MLP does not
# notice. PyTorch leaves it off by default; profile_selfplay.py measured
# it on 2026-09-06 -- the two planner calls were 93% of an iteration.
torch.set_float32_matmul_precision("high")

# Weights come from the named curriculum's self-play stage so the pretrain
# stages and this one share one table (rewards.CURRICULUM).
_SP = CURRICULUM["selfplay"]
GOAL_REWARD = _SP["goal_reward"]
GOAL_PENALTY = _SP["goal_penalty"]


def make_env(args, n_envs, stage_kwargs: bool = True):
    # The self-play stage's env settings -- the shot-type request, the
    # opponent mix and the sensing fuzz -- come from rewards.CURRICULUM so
    # the trainer and the table (deploy) agree on what the policy saw.
    extra = curriculum_env_kwargs("selfplay") if stage_kwargs else {}
    if getattr(args, "no_opponent_mix", False):
        extra.pop("opponent_mix_probs", None)
    return BatchAirHockeyEnv(
        n_envs=n_envs,
        **extra,
        agent_dynamics="profile" if args.dynamics else "ideal",
        opponent_dynamics="delayed" if args.dynamics else "ideal",
        opponent_policy="external",
        # Symmetric by default: the far side is a copy of the machine, not
        # the human model, so the learner is always playing itself.
        opponent_body="robot" if args.symmetric else "human",
        action_dt=ACTION_DT,
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
    # Recorded games are always against the copy of self, with requests on.
    env = make_env(args, 1)
    env._opp_mix_ids = None          # no draw: the far side is the checkpoint
    env._opp_policy_id[:] = _OPP_POLICY_MAP["external"]
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
    parser.add_argument("--pi-smooth", type=float, default=0.1,
                        help="temporal-smoothness coefficient on the policy "
                             "prior (TD-MPC2 pi_smooth_coef): pulls the "
                             "prior's mean toward the previous action, which "
                             "the observation carries. 0 disables.")
    parser.add_argument("--plan-smooth", type=float, default=PLAN_SMOOTH_COEF,
                        help="MPPI action-change cost (TD-MPC2 plan_smooth_coef); "
                             "the same constant the table uses, so the planner "
                             "that collects the data is the planner that plays")
    parser.add_argument("--iterations", type=int, default=PLAN_ITERATIONS,
                        help="MPPI iterations for data collection; the table's "
                             "number, for the same reason")
    parser.add_argument("--opp-iterations", type=int, default=None,
                        help="MPPI iterations for the far side's planner "
                             "(default: --iterations); 0 = the prior alone")
    parser.add_argument("--samples", type=int, default=256,
                        help="MPPI samples per env during COLLECTION (eval and "
                             "the table keep the config's 512). At 32 envs the "
                             "planner is compute-bound and 512 -> 256 halves "
                             "its cost, measured 2026-09-06")
    parser.add_argument("--no-compile", dest="compile_plan", action="store_false",
                        default=True,
                        help="skip torch.compile(reduce-overhead) on the "
                             "batched planners (CUDA graphs; ~30 s warm-up, "
                             "then ~1.4x on each planner call)")
    parser.add_argument("--reset-prior", action="store_true",
                        help="re-initialise the policy prior head on resume "
                             "(policy_loader.reset_prior): a prior saturated "
                             "by long training cannot be regularised back")
    parser.add_argument("--demo-envs", type=int, default=0,
                        help="envs played by the scripted CushionBot alongside the "
                             "agent's, whose episodes go into the same replay buffer "
                             "(rewards and all): the stop-hold-shoot chain the policy "
                             "never found on its own. 0 = none")
    parser.add_argument("--demo-until", type=int, default=800_000,
                        help="stop adding demonstrations after this many agent steps")
    parser.add_argument("--no-opponent-mix", action="store_true",
                        help="every far side is the copy of self (the "
                             "curriculum's default mixes in the sniper and "
                             "the weak goalie, see rewards.CURRICULUM)")
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
        "discount_max": 0.995, "rho": 0.7,   # run 2: see rewards.patience_s
        "task_title": "Air Hockey Self-Play",
        "multitask": False, "tasks": ["airhockey-selfplay"], "task_dim": 0,
        "pi_smooth_coef": args.pi_smooth,
        "plan_smooth_coef": args.plan_smooth,
        "iterations": args.iterations,
        "num_samples": args.samples,
    })
    cfg = OmegaConf.merge(base_cfg, overrides)
    if args.model_size in MODEL_SIZE:
        for k, v in MODEL_SIZE[args.model_size].items():
            cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.prev_action_start = BatchAirHockeyEnv.PREV_ACTION_IDX   # obs [15:17]

    n_envs = args.n_envs
    env = make_env(args, n_envs)
    shaper = BatchRewardShaper(n_envs, stage=STAGE_SCORING, workspace=env._ws,
                               **curriculum_shaper_kwargs("selfplay"))
    demo_env = demo_bot = demo_shaper = None
    if args.demo_envs > 0:
        # The demonstrator plays the scripted opponents only (no planner
        # call for its far side); the checkpoint's copy stays the agent's.
        demo_env = make_env(args, args.demo_envs)
        demo_env._opp_mix_ids = np.array([_OPP_POLICY_MAP["sniper"], _OPP_POLICY_MAP["weak_goalie"],
                                          _OPP_POLICY_MAP["goalie"]], dtype=np.int8)
        demo_env._opp_mix_p = np.array([0.4, 0.4, 0.2])
        demo_shaper = BatchRewardShaper(args.demo_envs, stage=STAGE_SCORING, workspace=demo_env._ws,
                                        **curriculum_shaper_kwargs("selfplay"))
    episode_length = int(round(30.0 / ACTION_DT))   # 30 s
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
    load_checkpoint(agent, args.resume)
    if args.reset_prior:
        from airhockey.policy_loader import reset_prior   # noqa: PLC0415
        reset_prior(agent)
        print("  prior head re-initialised (--reset-prior)")
    # The far side gets its OWN cfg: TDMPC2 keeps a reference, and the
    # opponent's iteration count must not change the agent's.
    import copy                                            # noqa: PLC0415
    opp_cfg = copy.deepcopy(cfg)
    opp_iters = args.iterations if args.opp_iterations is None else args.opp_iterations
    opp_cfg.iterations = max(1, opp_iters)
    opp_cfg.mpc = opp_iters > 0
    opponent = TDMPC2(opp_cfg)
    load_checkpoint(opponent, args.resume)
    buffer = Buffer(cfg)
    if args.compile_plan and torch.cuda.is_available():
        # CUDA graphs on the batched planners. Shapes are static (n_envs),
        # weights update in place (the graph reads the same memory), and
        # the opponent's periodic load() copies into the same parameters.
        agent._plan_batch = torch.compile(agent._plan_batch, mode="reduce-overhead")
        if opp_cfg.mpc:
            opponent._plan_batch = torch.compile(opponent._plan_batch, mode="reduce-overhead")

    print(f"\nSelf-Play TD-MPC2 Training (March recipe, new environment)")
    print(f"  Steps: {args.steps:,}   Parallel envs: {n_envs}")
    print(f"  Opponent update: every {args.opponent_update_freq:,} steps")
    print(f"  Far side: {'copy of self (robot body, mirrored workspace)' if args.symmetric else 'human model'}")
    print(f"  Prior smoothness (pi_smooth_coef): {args.pi_smooth}")
    print(f"  Planner: {args.iterations} MPPI iterations x {args.samples} samples, "
          f"action-change cost {args.plan_smooth}; far side "
          f"{'prior only' if not opp_cfg.mpc else f'{opp_cfg.iterations} iterations'}; "
          f"CUDA graphs {'on' if args.compile_plan and torch.cuda.is_available() else 'off'}; "
          f"TF32 on")
    print(f"  Planning horizon: {args.horizon}   Goals: +{GOAL_REWARD:.0f} / {GOAL_PENALTY:.0f}")
    print(f"  Sensing: {'on' if args.realistic_sensing else 'off'}   "
          f"DR: {'on' if args.domain_randomize else 'off'}   obs {env.obs_dim} dims")
    print(f"  Output: {run_dir}\n")

    obs = env.reset(seed=cfg.seed)
    shaper.reset(obs, info=_truth_info(env))
    t0_mask = torch.ones(n_envs, dtype=torch.bool)

    def fresh_td_for(e, o):
        return TensorDict(
            obs=torch.from_numpy(o).float().unsqueeze(0),
            action=torch.full((1, e.action_dim), float('nan')),
            reward=torch.tensor([float('nan')]),
            terminated=torch.tensor([float('nan')]),
            batch_size=(1,))

    def fresh_td(o):
        return fresh_td_for(env, o)
    tds = [[fresh_td(obs[i])] for i in range(n_envs)]
    if demo_env is not None:
        demo_obs = demo_env.reset(seed=cfg.seed + 1)
        demo_env._opp_policy_id[:] = demo_env._rng.choice(demo_env._opp_mix_ids, size=args.demo_envs, p=demo_env._opp_mix_p)
        demo_obs = demo_env.reset(seed=cfg.seed + 1)
        demo_shaper.reset(demo_obs, info=_truth_info(demo_env))
        demo_bot = CushionBot(demo_env, np.random.default_rng(cfg.seed + 2))
        demo_tds = [[fresh_td_for(demo_env, demo_obs[i])] for i in range(args.demo_envs)]
        demo_eps = 0
        print(f"  Demonstrations: {args.demo_envs} envs of CushionBot until step {args.demo_until:,}")

    step = 0
    updating = False
    train_info = None
    start = time()
    last_record = 0
    last_opp_update = 0
    opp_version = 0
    wins = losses = draws = 0
    # Per-opponent-kind tallies: with a mix, one win rate means nothing.
    kind_names = {v: k for k, v in _OPP_POLICY_MAP.items()}
    by_kind: dict[int, list[int]] = {}

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
        # No random seeding: this script always RESUMES a trained agent, and
        # the March recipe's 15k steps of uniform-random targets (a
        # bang-bang random walk at 60 m/s^2) followed by a 5000-update
        # burst on that data was an off-policy shock at every restart -- and
        # every "50k checkpoint" of a resumed run was the starting weights,
        # because nothing had updated yet.
        with torch.no_grad():
            actions = agent.act(torch.from_numpy(obs).float(), t0=t0_mask)
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
                k = int(info["opponent_kind"][i])
                tally = by_kind.setdefault(k, [0, 0, 0, 0, 0])   # W, L, D, GF, GA
                tally[0 if sa > so else 1 if so > sa else 2] += 1
                tally[3] += sa
                tally[4] += so
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

        # Demonstrations: the scripted controller's envs, stepped alongside,
        # their episodes stored exactly like the agent's.
        if demo_env is not None and step <= args.demo_until:
            d_act = demo_bot.act()
            d_next, d_raw, d_term, d_trunc, d_info = demo_env.step(d_act)
            d_shaped = demo_shaper.compute(d_next, d_raw, actions=d_act, info=d_info)
            d_done = d_term | d_trunc
            for i in range(args.demo_envs):
                demo_tds[i].append(TensorDict(
                    obs=torch.from_numpy(d_next[i]).float().unsqueeze(0),
                    action=torch.from_numpy(d_act[i]).float().unsqueeze(0),
                    reward=torch.tensor([float(d_shaped[i])], dtype=torch.float32),
                    terminated=torch.tensor([float(d_term[i])]),
                    batch_size=(1,)))
            if np.any(d_done):
                for i in np.where(d_done)[0]:
                    buffer.add(torch.cat(demo_tds[i]))
                    demo_eps += 1
                r_obs = demo_env.auto_reset(d_term, d_trunc)
                if r_obs is not None:
                    d_next = r_obs
                    demo_shaper.reset(d_next, mask=d_done, info=_truth_info(demo_env))
                    demo_bot.reset(d_done)
                    for i in np.where(d_done)[0]:
                        demo_tds[i] = [fresh_td_for(demo_env, d_next[i])]
            demo_obs = d_next

        total_eps = wins + losses + draws
        # The buffer stores whole episodes, so the first update waits for
        # the first games to finish (~96k env steps at 32 envs x 30 s, or
        # sooner when a game reaches 7). From then on, the recipe's one
        # update per vectorised step.
        if total_eps >= 2:
            if not updating:
                updating = True
                print(f"[Step {step:,}] first games in the buffer -- updates begin")
            for _ in range(args.updates_per_iter):
                train_info = agent.update(buffer)
            if step % 10000 < n_envs and train_info is not None:
                for k in ("pi_loss", "pi_smooth", "pi_scale", "pi_entropy",
                          "value_loss", "reward_loss", "consistency_loss", "total_loss"):
                    if k in train_info.keys():
                        writer.add_scalar(f"loss/{k}", float(train_info[k]), step)

        step += n_envs

        if step % 10000 < n_envs:
            fps = step / max(time() - start, 1)
            wr = wins / max(total_eps, 1) * 100
            print(f"[Train] step={step:,} fps={fps:.0f} W/L/D={wins}/{losses}/{draws} "
                  f"WR={wr:.0f}% opp_v{opp_version}")
            writer.add_scalar('train/fps', fps, step)
            for k, (w_, l_, d_, gf, ga) in sorted(by_kind.items()):
                n_k = max(w_ + l_ + d_, 1)
                name = kind_names.get(k, str(k))
                writer.add_scalar(f'vs_{name}/win_rate', w_ / n_k, step)
                writer.add_scalar(f'vs_{name}/goals_for_per_game', gf / n_k, step)
                writer.add_scalar(f'vs_{name}/goals_against_per_game', ga / n_k, step)
                print(f"    vs {name:12s} {w_}/{l_}/{d_}  GF {gf / n_k:.2f}  GA {ga / n_k:.2f}")
            for k, v in shaper.stats.items():
                writer.add_scalar(f'shots/{k}', v, step)
            print(f"    shots {shaper.stats}")
            for k in shaper.stats:
                shaper.stats[k] = 0
            if demo_env is not None:
                print(f"    demo  {demo_shaper.stats}  episodes stored {demo_eps}  bot {demo_bot.stats}")
                for k, v in demo_shaper.stats.items():
                    writer.add_scalar(f'demo/{k}', v, step)
                    demo_shaper.stats[k] = 0
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
