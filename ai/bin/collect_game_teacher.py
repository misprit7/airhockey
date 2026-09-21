#!/usr/bin/env python3
"""Collect full-game MPC behavior in independent short simulated matches.

Includes positioning and recovery, not only the first defensive contact.
Short chunks are demonstration collection, not a sustained load qualification.
"""

import argparse
import json
from pathlib import Path

from eval_foundation import load, torch, np
from airhockey.legacy_practice import LegacyPracticeEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--seed", type=int, default=20261220)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    teacher = load("runs/3.12-accel60-accuracy-selfplay/agent.pt")
    teacher.cfg.mpc, teacher.cfg.num_samples = True, 256
    teacher._plan_batch = torch.compile(teacher._plan_batch, mode="reduce-overhead")
    observations, actions, phases, results = [], [], [], []
    for batch in range(args.batches):
        seed = args.seed + batch
        env = LegacyPracticeEnv(
            args.games,
            seed=seed,
            motion_guard=True,
            game_fraction=1,
            selfplay_fraction=1,
        )
        env.base.shot_types = False
        env.base.symmetric_referee = True
        env.base.max_episode_time, env.base.max_score = args.seconds + 1, 100000
        obs = env.reset(seed=seed, opponent="external")
        teacher._prev_mean_batch = None
        t0 = torch.ones(2 * args.games, dtype=torch.bool)
        kept, peak = 0, 0.0
        for _ in range(round(args.seconds / env.base.action_dt)):
            both = np.concatenate((obs, env.opponent_obs()))
            torch.compiler.cudagraph_mark_step_begin()
            action = teacher.act(
                torch.from_numpy(both[:, :22]), t0=t0, eval_mode=True
            ).numpy()
            env.set_opponent_action(action[args.games :])
            nxt, _, _, _, _ = env.step(action[: args.games])
            levels = np.concatenate(
                [load.levels.max(axis=(1, 2)) for load in env.loads]
            )
            # Every retained transition is within the provisional load envelope.
            # This does not assert that repeated teacher play stays within it.
            keep = levels < 0.95
            if keep.any():
                observations.append(both[keep].copy())
                actions.append(action[keep].copy())
                o = both[keep]
                phase = np.where(
                    o[:, 3] < -1,
                    0,
                    np.where(
                        (o[:, 1] < 1) & (np.linalg.norm(o[:, 2:4], axis=1) < 1.2),
                        1,
                        np.where(o[:, 1] >= 1, 2, 3),
                    ),
                )
                phases.append(phase)
                kept += int(keep.sum())
            peak = max(peak, float(levels.max()))
            obs, t0 = nxt, torch.zeros_like(t0)
        row = dict(
            seed=seed,
            kept=kept,
            peak_load=peak,
            goals=[
                int(env.engine.score_agent.sum()),
                int(env.engine.score_opponent.sum()),
            ],
        )
        results.append(row)
        print("[collect]", json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=np.concatenate(observations),
        action=np.concatenate(actions),
        phase=np.concatenate(phases),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                teacher="3.12 full MPC, 6 iterations / 256 candidates, elite mean",
                independent_short_games=True,
                deployment_ready=False,
                samples=sum(map(len, actions)),
                batches=results,
                phases=np.bincount(np.concatenate(phases), minlength=4).tolist(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
