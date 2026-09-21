#!/usr/bin/env python3
"""Supervised feasibility pilot: train encoder and actor on actual observations.

Only the policy prior is fitted. Untrained model/value heads must not be used
for MPC. Simulation-only checkpoints are explicitly marked as such.
"""

import argparse
import json
import time
from pathlib import Path

from train_arrival import ROOT, config, TDMPC2, torch, np, save
from airhockey.arrival_training import transfer_encoder, load_demonstrations
from airhockey.policy_benchmark import evaluate_skills
from airhockey.legacy_practice import (
    LegacyPracticeEnv,
    transfer_full_policy,
    convert_arrival_demonstrations,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="_foundation-bc-fixed")
    p.add_argument(
        "--demos", type=Path, default=ROOT / "logs/arrival-training/demonstrations"
    )
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--position-weight", type=float, default=1.0)
    p.add_argument("--cosine-lr", action="store_true")
    p.add_argument("--adapt-context", action="store_true")
    p.add_argument(
        "--adapt-encoder",
        action="store_true",
        help="Fine tune the entire retained representation at a lower learning rate",
    )
    p.add_argument("--retention", type=Path)
    p.add_argument(
        "--balanced",
        action="store_true",
        help="Balance skills and emphasize the contact approach",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Preserve full 3.12 policy/world model; adapt actor only",
    )
    args = p.parse_args()
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = dict(
        deployment_ready=False,
        action_mode="profile_a" if args.legacy else "arrival",
        inference_mode="prior",
        obs_dim=42 if args.legacy else 45,
        action_dim=3 if args.legacy else 6,
        horizon=8,
        model_size=5,
        agent_accel_range=[60, 60],
        agent_speed_range=[12, 12],
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        note="Feasibility pilot, supervised encoder+actor; model/value heads untrained",
    )
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    cfg = config(args, run_dir)
    if args.legacy:
        cfg.obs_shape = {"state": [42]}
        cfg.action_dim = 3
        cfg.discount_max = 0.99
        cfg.episode_length = 3000
        cfg.plan_smooth_coef = 0.0
        cfg.plan_eval_mean = True
    agent = TDMPC2(cfg)
    if args.legacy:
        meta["transfer"] = transfer_full_policy(
            agent, ROOT / "runs/3.12-accel60-accuracy-selfplay/agent.pt"
        )
        meta["note"] = (
            "Complete legacy model retained, zero-extended observations; actor-only BC pilot"
        )
        save(agent, run_dir / "agent_initial.pt")
    else:
        transfer_encoder(agent, ROOT / "runs/3.11-shot-ramp-selfplay/agent.pt")
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    agent.cfg.mpc = False
    episodes = load_demonstrations(args.demos)
    if args.legacy:
        env = LegacyPracticeEnv(1, realistic=False, randomize=False)
        episodes = convert_arrival_demonstrations(
            episodes, env.base._ws, env.base._action_low, env.base._action_high
        )
    demo_meta = json.loads((args.demos / "summary.json").read_text())
    teacher_fixtures = np.load(args.demos / "teachers.npz")
    obs = torch.as_tensor(
        np.concatenate([e["obs"][:-1] for e in episodes]), device="cuda"
    )
    target = torch.as_tensor(
        np.concatenate([e["action"][1:] for e in episodes]), device="cuda"
    )
    retention = None
    if args.retention:
        retained = np.load(args.retention)
        # Defense is the normal game request, including requested shot speed.
        retained_obs = retained["obs"].copy()
        retained_obs[:, 35] = 1
        retention = tuple(
            torch.as_tensor(v, device="cuda")
            for v in (retained_obs, retained["action"])
        )
    # Deliberately fit actual encoded observations, with gradients to the
    # encoder. 4.0's BC only supervised detached latent model rollouts.
    params = (
        []
        if args.legacy and not args.adapt_encoder
        else list(agent.model._encoder.parameters())
    ) + list(agent.model._pi.parameters())
    if args.adapt_context:
        if args.adapt_encoder:
            raise ValueError("choose either context-only or full encoder adaptation")
        if not args.legacy:
            raise ValueError(
                "context adaptation requires the legacy observation mapping"
            )
        for param in agent.model._encoder.parameters():
            param.requires_grad_(False)
        first = agent.model._encoder["state"][0].weight
        first.requires_grad_(True)
        mask = torch.zeros_like(first)
        mask[:, 22:] = 1
        first.register_hook(lambda grad: grad * mask)
        params.append(first)
        meta["context_adaptation"] = (
            "Learn new input columns; preserve original encoder weights"
        )
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    groups = params
    if args.adapt_encoder:
        groups = [
            dict(params=agent.model._encoder.parameters(), lr=args.lr * 0.1),
            dict(params=agent.model._pi.parameters(), lr=args.lr),
        ]
    opt = torch.optim.Adam(groups, lr=args.lr)
    base_lrs = [g["lr"] for g in opt.param_groups]
    weights = torch.ones(cfg.action_dim, device="cuda")
    weights[:2] = args.position_weight
    weights *= cfg.action_dim / weights.sum()
    buckets = []
    if args.balanced:
        task_start = 30 if args.legacy else 33
        kinds = obs[:, task_start : task_start + 3].argmax(-1)
        early = obs[:, -1] < 0.32 / 30
        for task in range(3):
            for phase in (True, False):
                bucket = torch.where((kinds == task) & (early == phase))[0]
                if not len(bucket):
                    raise ValueError("empty skill/phase demonstration bucket")
                buckets.append(bucket)
    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        if args.cosine_lr:
            for group, base_lr in zip(opt.param_groups, base_lrs):
                group["lr"] = base_lr * (
                    0.1 + 0.9 * (1 + np.cos(np.pi * step / args.steps)) / 2
                )
        idx = torch.randint(len(obs), (512,), device="cuda")
        if args.balanced:
            idx = torch.cat(
                [
                    b[
                        torch.randint(
                            len(b), (120 if i % 2 == 0 else 50,), device="cuda"
                        )
                    ]
                    for i, b in enumerate(buckets)
                ]
            )
        z = agent.model.encode(obs[idx], None)
        if args.legacy and not (args.adapt_context or args.adapt_encoder):
            z = z.detach()
        prediction = agent.model._pi(z)[..., : cfg.action_dim].tanh()
        loss = (((prediction - target[idx]) ** 2) * weights).sum(-1).mean()
        if retention is not None:
            ro, ra = retention
            ri = torch.randint(len(ro), (256,), device="cuda")
            rz = agent.model.encode(ro[ri], None)
            if not (args.adapt_context or args.adapt_encoder):
                rz = rz.detach()
            rp = agent.model._pi(rz)[..., : cfg.action_dim].tanh()
            loss = loss + (((rp - ra[ri]) ** 2) * weights).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10)
        opt.step()
        if step % 250 == 0:
            print(
                f"[fit] step={step} loss={float(loss):.6f} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
        if step in (1000, 3000, args.steps):
            save(agent, run_dir / f"agent_step_{step:07d}.pt")
            save(agent, run_dir / "agent.pt")
            for seed, label, randomized, per_task in (
                (demo_meta["seed"], "training_poses", True, demo_meta["per_task"]),
                (20261102, "heldout", True, 100),
                (20261102, "workspace_validation", True, 100),
            ):
                torch.manual_seed(seed)
                result = evaluate_skills(
                    agent,
                    seed=seed,
                    per_task=per_task,
                    planner=False,
                    legacy=args.legacy,
                    realistic=True,
                    randomize=randomized,
                    wide=label == "workspace_validation",
                    teacher_fixtures=teacher_fixtures
                    if label == "training_poses"
                    else None,
                )
                result.update(update=step, split=label)
                history.append(result)
                print("[eval]", json.dumps(result), flush=True)
            (run_dir / "evaluations.json").write_text(json.dumps(history, indent=2))
    (run_dir / "status.json").write_text(
        json.dumps(dict(complete=True, updates=args.steps))
    )


if __name__ == "__main__":
    main()
