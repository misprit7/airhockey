#!/usr/bin/env python3
"""Where a self-play training iteration spends its time, and what would help.

One iteration of train_selfplay.py at N envs is: plan for the far side
(opponent.act, eval mode), plan for the agent (agent.act, exploration),
step the batch env, append to the buffer, one gradient update. This times
each on this machine, then re-times the planner under the knobs that
could cut it -- TF32 matmuls, fewer samples, fewer iterations, CUDA
graphs -- and prints the iteration time and env-steps/s each would give.

    python ai/bin/profile_selfplay.py --run retrain40_goalie --n-envs 32

The GPU may be shared with a training run; absolute numbers are then
inflated, the proportions still hold.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs  # noqa: E402
from airhockey.policy_loader import load_agent  # noqa: E402
from airhockey.rewards import curriculum_env_kwargs  # noqa: E402


def timeit(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000.0 * (time.perf_counter() - t))
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="retrain40_goalie")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    N = args.n_envs
    dev = "cuda"
    print(f"GPU {torch.cuda.get_device_name(0)}   matmul precision {torch.get_float32_matmul_precision()}")

    agent = load_agent(args.run, iterations=6)
    agent.model.to(dev)
    agent.device = dev
    for k in ("_prev_mean", "_prev_mean_batch"):
        v = getattr(agent, k, None)
        if torch.is_tensor(v):
            setattr(agent, k, v.to(dev))
    cfg = agent.cfg
    env = BatchAirHockeyEnv(N, opponent_policy="external", opponent_body="robot",
                            domain_randomize=True, **sensing_kwargs(True),
                            **curriculum_env_kwargs("selfplay"))
    obs = env.reset(seed=0)
    t0 = torch.zeros(N, dtype=torch.bool)
    obs_t = torch.from_numpy(obs).float()
    rng = np.random.default_rng(0)

    def env_step():
        env.step(rng.uniform(-1, 1, size=(N, 2)).astype(np.float32))

    def opp_view():
        env.opponent_obs()

    def plan_explore():
        with torch.no_grad():
            agent.act(obs_t, t0=t0, eval_mode=False)

    def plan_eval():
        with torch.no_grad():
            agent.act(obs_t, t0=t0, eval_mode=True)

    rows = []
    rows.append(("env.step (N envs, 8 physics substeps + camera)", timeit(env_step, args.n)))
    rows.append(("opponent_obs()", timeit(opp_view, args.n)))
    rows.append(("agent plan, 6 it x 512 samples, explore", timeit(plan_explore, args.n)))
    rows.append(("opponent plan, 6 it x 512, eval (elite mean)", timeit(plan_eval, args.n)))

    # A gradient update needs a buffer with episodes in it: fill it with
    # random-action play (cheap), then time update() alone.
    from common.buffer import Buffer            # noqa: PLC0415
    from tensordict.tensordict import TensorDict  # noqa: PLC0415
    buffer = Buffer(cfg)
    ep_len = int(cfg.episode_length)
    o = env.reset(seed=1)
    tds = [[TensorDict(obs=torch.from_numpy(o[i]).float().unsqueeze(0),
                       action=torch.full((1, 2), float("nan")),
                       reward=torch.tensor([float("nan")]),
                       terminated=torch.tensor([float("nan")]), batch_size=(1,))]
           for i in range(N)]
    for _ in range(ep_len):
        a = rng.uniform(-1, 1, size=(N, 2)).astype(np.float32)
        o, r, term, trunc, _ = env.step(a)
        for i in range(N):
            tds[i].append(TensorDict(obs=torch.from_numpy(o[i]).float().unsqueeze(0),
                                     action=torch.from_numpy(a[i]).unsqueeze(0),
                                     reward=torch.tensor([float(r[i])]),
                                     terminated=torch.tensor([float(term[i])]),
                                     batch_size=(1,)))
    for i in range(N):
        buffer.add(torch.cat(tds[i]))
    rows.append(("agent.update (one gradient step, batch %d)" % cfg.batch_size,
                 timeit(lambda: agent.update(buffer), args.n)))

    base_iter = sum(ms for _, ms in rows)
    print(f"\n{'component':<52}{'ms':>9}")
    for name, ms in rows:
        print(f"{name:<52}{ms:>9.1f}")
    print(f"{'iteration (sum)':<52}{base_iter:>9.1f}   -> {1000 * N / base_iter:.0f} env-steps/s")

    # ── what would help ──────────────────────────────────────────────
    print("\nplanner variants (agent plan, explore mode), ms per call and the "
          "iteration they imply if BOTH planners use it:")
    other = base_iter - rows[2][1] - rows[3][1]

    def report(label, ms):
        it = other + 2 * ms
        print(f"  {label:<48}{ms:>8.1f} ms   iteration {it:6.1f} ms  -> {1000 * N / it:5.0f} steps/s")

    report("as run: fp32, 6 it, 512 samples", rows[2][1])
    torch.set_float32_matmul_precision("high")
    report("TF32 matmuls", timeit(plan_explore, args.n))
    cfg.num_samples = 256
    report("TF32 + 256 samples", timeit(plan_explore, args.n))
    cfg.num_samples = 512
    cfg.iterations = 3
    report("TF32 + 3 iterations", timeit(plan_explore, args.n))
    cfg.iterations = 6
    try:
        compiled = torch.compile(agent._plan_batch, mode="reduce-overhead")
        t0d = t0.to(dev)
        obs_d = obs_t.to(dev)

        def plan_compiled():
            with torch.no_grad():
                compiled(obs_d, t0d, eval_mode=False).cpu()
        t = time.perf_counter()
        for _ in range(3):
            plan_compiled()
        torch.cuda.synchronize()
        print(f"  (compile + warm-up {time.perf_counter() - t:.0f} s)")
        report("TF32 + CUDA graphs", timeit(plan_compiled, args.n))
        cfg.num_samples = 256
        report("TF32 + CUDA graphs + 256 samples", timeit(plan_compiled, args.n))
        cfg.num_samples = 512
    except Exception as e:                       # noqa: BLE001
        print(f"  CUDA graphs: failed ({type(e).__name__}: {e})")
    report("prior only for the OPPONENT (agent 6 it TF32)",
           0.5 * (timeit(plan_explore, args.n) + 0.3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
