#!/usr/bin/env python3
"""Collect successful MPC defensive approaches, simulation only.

Keep frames up to the first real puck contact; don't distill the reference's
unreliable follow-up shots. Held-out benchmark seeds are not used here.
"""

import argparse
import json
from pathlib import Path

from eval_foundation import load, torch, np
from airhockey.legacy_practice import LegacyPracticeEnv
from airhockey.policy_benchmark import fixtures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--per-batch", type=int, default=256)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261125)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    teacher = load("runs/3.12-accel60-accuracy-selfplay/agent.pt")
    teacher.cfg.mpc = True
    teacher.cfg.iterations, teacher.cfg.num_samples = 6, 256
    observations, actions, results = [], [], []
    for batch in range(args.batches):
        n, seed = args.per_batch, args.seed + batch
        bank, _ = fixtures(seed, n, wide=True)
        env = LegacyPracticeEnv(n, seed=seed, motion_guard=True)
        obs = env.reset(seed=seed, fixtures=bank.take(np.arange(3 * n, 4 * n)))
        env.task[:] = 3
        env.desired_speed[:] = 3
        obs[:, 30:34] = [0, 0, 0, 1]
        obs[:, 35] = 1
        teacher._prev_mean_batch = None
        t0 = torch.ones(n, dtype=torch.bool)
        first = np.full(n, 100)
        oo, aa = [], []
        for tick in range(100):
            action = teacher.act(
                torch.from_numpy(obs[:, :22]), t0=t0, eval_mode=True
            ).numpy()
            oo.append(obs.copy())
            aa.append(action)
            obs, _, _, _, _ = env.step(action)
            first[(first == 100) & (env.contacts > 0)] = tick
            t0[:] = False
        success = (env.contacts > 0) & (env.engine.score_opponent == 0)
        keep = (np.arange(100)[:, None] <= first) & success
        observations.append(np.asarray(oo)[keep])
        actions.append(np.asarray(aa)[keep])
        results.append(dict(seed=seed, successes=int(success.sum()), trials=n))
        print("[defense]", results[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, obs=np.concatenate(observations), action=np.concatenate(actions)
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(dict(batches=results, samples=sum(map(len, actions))), indent=2)
    )


if __name__ == "__main__":
    main()
