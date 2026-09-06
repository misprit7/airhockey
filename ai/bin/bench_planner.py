#!/usr/bin/env python3
"""How long one planning call takes, across the knobs that set its cost.

THE QUESTION
    The table commands at a fixed rate and the planner has to answer inside
    one tick. Its cost is set by MPPI iterations x samples x horizon, plus
    fixed overhead per call (kernel launches, the GPU->CPU copy of the
    action). This measures each on the machine that will run it, so the
    control rate and planner config can be chosen from numbers.

    Two shapes matter: ONE env (deploy: `agent.act(obs_1d)` -> `plan`) and
    N envs (training collection: `_plan_batch`). Optionally CUDA graphs via
    torch.compile(mode="reduce-overhead") and bf16 autocast, both of which
    change only speed, not what is computed.

    python ai/bin/bench_planner.py                      # default sweep
    python ai/bin/bench_planner.py --run curriculum_selfplay_smooth6 --n 200
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
from airhockey.policy_loader import load_agent, resolve_checkpoint  # noqa: E402


def _bench(fn, n: int, warm: int = 20) -> tuple[float, float, float]:
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(1000.0 * (time.perf_counter() - t))
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))], ts[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="curriculum_selfplay_smooth6")
    ap.add_argument("--n", type=int, default=150, help="timed calls per config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32, help="N envs for the batched shape")
    ap.add_argument("--compile", action="store_true",
                    help="also time torch.compile(reduce-overhead) on _plan")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.run)
    print(f"checkpoint {ckpt}   device {args.device}   "
          f"{torch.cuda.get_device_name(0) if args.device == 'cuda' else ''}")
    agent = load_agent(args.run, iterations=3)
    agent.model.to(args.device)
    agent.device = args.device
    for k in ("_prev_mean", "_prev_mean_batch", "discount"):
        v = getattr(agent, k, None)
        if torch.is_tensor(v):
            setattr(agent, k, v.to(args.device))
    obs_dim = agent.cfg.obs_shape["state"][0]
    obs1 = torch.zeros(obs_dim)
    obs1[12] = 1.0
    obsN = torch.zeros(args.batch, obs_dim)
    obsN[:, 12] = 1.0
    t0N = torch.zeros(args.batch, dtype=torch.bool)

    print(f"\n{'shape':<8}{'iters':>6}{'samples':>9}{'horizon':>9}{'mode':>10}"
          f"{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}")

    def row(shape, iters, samples, horizon, mode, fn):
        agent.cfg.iterations = iters
        agent.cfg.num_samples = samples
        agent.cfg.horizon = horizon
        agent._prev_mean = torch.zeros(horizon, agent.cfg.action_dim, device=args.device)
        agent._prev_mean_batch = None
        p50, p95, mx = _bench(fn, args.n)
        print(f"{shape:<8}{iters:>6}{samples:>9}{horizon:>9}{mode:>10}"
              f"{p50:>9.2f}{p95:>9.2f}{mx:>9.2f}", flush=True)

    # Prior only: the floor.
    agent.cfg.mpc = False
    row("1", 0, 0, 0, "prior", lambda: agent.act(obs1, t0=False, eval_mode=True))
    agent.cfg.mpc = True

    def one():
        return agent.act(obs1, t0=False, eval_mode=True)

    def many():
        return agent.act(obsN, t0=t0N, eval_mode=True)

    for iters in (1, 2, 3, 4, 6):
        row("1", iters, 512, 5, "fp32", one)
    for samples in (128, 256):
        row("1", 3, samples, 5, "fp32", one)
    for horizon in (3, 8):
        row("1", 3, 512, horizon, "fp32", one)

    if args.device == "cuda":
        def one_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return agent.act(obs1, t0=False, eval_mode=True)
        for iters in (3, 6):
            row("1", iters, 512, 5, "bf16", one_bf16)

    for iters in (3, 6):
        row(str(args.batch), iters, 512, 5, "fp32", many)

    if args.compile and args.device == "cuda":
        agent.cfg.iterations, agent.cfg.num_samples, agent.cfg.horizon = 3, 512, 5
        agent._prev_mean = torch.zeros(5, agent.cfg.action_dim, device=args.device)
        t = time.perf_counter()
        plan = torch.compile(agent._plan, mode="reduce-overhead")
        z = torch.zeros(1, obs_dim, device=args.device)
        z[0, 12] = 1.0
        for _ in range(3):
            plan(z, t0=False, eval_mode=True)
        torch.cuda.synchronize()
        print(f"\ncompile + warmup {time.perf_counter() - t:.1f} s")

        def compiled():
            return plan(z, t0=False, eval_mode=True).cpu()
        for iters in (3, 6):
            row("1", iters, 512, 5, "cudagraph", compiled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
