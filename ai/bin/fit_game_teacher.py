#!/usr/bin/env python3
"""Bounded behavior transfer with successful current-skill retention."""

import argparse
import hashlib
import json
from pathlib import Path

from train_arrival import ROOT, save, torch, np
from eval_foundation import load, match
from airhockey.foundation_training import FoundationEnv
from airhockey.sequence_replay import EpisodeBatch
from airhockey.policy_benchmark import evaluate_skills


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--heldout-fraction", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=20261230)
    args = p.parse_args()
    if not 0 <= args.heldout_fraction < 0.5:
        raise ValueError("heldout fraction must be in [0,0.5)")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    agent = load(args.checkpoint)
    agent.cfg.mpc = False
    agent.training_algorithm = "Behavior distillation"
    save(agent, run / "agent_initial.pt")
    meta = dict(
        deployment_ready=False,
        action_mode="profile_a",
        obs_dim=42,
        action_dim=3,
        model_size=5,
        horizon=8,
        action_hz=50,
        agent_accel_range=[60, 60],
        agent_speed_range=[12, 12],
        inference_mode="prior",
        algorithm="Behavior distillation",
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        transfer=(
            "Full source model; train actor/full encoder"
            if args.unfreeze_encoder
            else "Full source model; train actor/new context columns"
        )
        + " on qualified teacher demonstrations and successful source skills",
    )
    meta["source_hashes"] = {}
    for path in [Path(__file__), *sorted((ROOT / "ai/airhockey").glob("*.py"))]:
        dest = run / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        meta["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    meta["teacher_sha256"] = hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    data = np.load(args.teacher)
    game_obs = torch.as_tensor(data["obs"], device="cuda")
    game_action = torch.as_tensor(data["action"], device="cuda")
    split = int(len(game_obs) * (1 - args.heldout_fraction))
    if args.heldout_fraction and "validation_start" in data:
        split = int(data["validation_start"])
    if not 0 < split <= len(game_obs):
        raise ValueError("invalid teacher validation split")
    meta["teacher_training_samples"] = split
    meta["teacher_validation_samples"] = len(game_obs) - split
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    buckets = [
        torch.as_tensor(np.flatnonzero(data["phase"][:split] == k), device="cuda")
        for k in range(4)
    ]
    env = FoundationEnv(64, seed=args.seed, possession_fraction=0.35)
    obs = env.reset(seed=args.seed)
    staging = EpisodeBatch(64, 42, 3, max_steps=102)
    staging.reset(obs)
    retained_obs, retained_action, retained_task = [], [], []
    for tick in range(700):
        action = agent.act(torch.from_numpy(obs), eval_mode=True).numpy()
        nxt, reward, term, trunc, info = env.step(action)
        done = term | trunc
        staging.append(nxt, action, reward, term)
        for i in np.flatnonzero(done):
            task = info["task"][i]
            if task < 3 and info["success"][i] and info["load_peak"][i] < 0.95:
                ep = staging.episode_at(i)
                retained_obs.append(ep["obs"][:-1])
                retained_action.append(ep["action"][1:])
                retained_task.append(np.full(len(ep["obs"]) - 1, task))
        if done.any():
            nxt[done] = env.reset(mask=done)[done]
            staging.reset(nxt, mask=done)
        obs = nxt
    skill_obs = torch.as_tensor(np.concatenate(retained_obs), device="cuda")
    skill_action = torch.as_tensor(np.concatenate(retained_action), device="cuda")
    kinds = np.concatenate(retained_task)
    skill_buckets = [
        torch.as_tensor(np.flatnonzero(kinds == k), device="cuda") for k in range(3)
    ]
    print("[data]", len(game_obs), len(skill_obs), flush=True)
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.model._pi.parameters():
        parameter.requires_grad_(True)
    context = agent.model._encoder["state"][0].weight
    context.requires_grad_(True)
    mask = torch.ones_like(context)
    mask[:, :22] = 0
    if not args.unfreeze_encoder:
        context.register_hook(lambda grad: grad * mask)
    else:
        for parameter in agent.model._encoder.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            dict(params=agent.model._pi.parameters(), lr=1e-5),
            dict(
                params=list(agent.model._encoder.parameters())
                if args.unfreeze_encoder
                else [context],
                lr=3e-5,
            ),
        ]
    )
    weights = torch.tensor([5.0, 5.0, 1.0], device="cuda") / 11
    trainable = [p for p in agent.model.parameters() if p.requires_grad]
    history = []

    def assess(step):
        for possession in (False, True):
            r = evaluate_skills(
                agent,
                legacy=True,
                planner=False,
                wide=True,
                guard=True,
                seed=20261102,
                per_task=100,
                game_requests=True,
                random_paddle=possession,
            )
            r.update(step=step)
            history.append(r)
            print(
                "[evaluate]",
                step,
                possession,
                {k: v["successes"] for k, v in r["tasks"].items()},
                flush=True,
            )
        (run / "evaluations.json").write_text(json.dumps(history, indent=2))

    assess(0)
    for step in range(1, args.steps + 1):
        gi = torch.cat(
            [b[torch.randint(len(b), (128,), device="cuda")] for b in buckets]
        )
        si = torch.cat(
            [b[torch.randint(len(b), (170,), device="cuda")] for b in skill_buckets]
        )
        z = agent.model.encode(torch.cat((game_obs[gi], skill_obs[si])), None)
        pred = agent.model._pi(z)[..., :3].tanh()
        game_loss = (
            ((pred[: len(gi)] - game_action[gi]).square() * weights).sum(-1).mean()
        )
        retention = (
            ((pred[len(gi) :] - skill_action[si]).square() * weights).sum(-1).mean()
        )
        loss = game_loss + 2 * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 250 == 0:
            with torch.no_grad():
                heldout_loss = None
                if split < len(game_obs):
                    heldout = agent.model._pi(
                        agent.model.encode(game_obs[split:], None)
                    )[..., :3].tanh()
                    heldout_loss = float(
                        ((heldout - game_action[split:]).square() * weights)
                        .sum(-1)
                        .mean()
                    )
            print(
                "[fit]",
                step,
                float(game_loss.detach()),
                float(retention.detach()),
                heldout_loss,
                flush=True,
            )
        if step in (500, 1500, args.steps):
            save(agent, run / f"agent_step_{step:07d}.pt")
            save(agent, run / "agent.pt")
            assess(step)
            report = match(
                agent,
                agent,
                games=8,
                seconds=30,
                guard=True,
                seed=20261122,
                record=ROOT / "ai/recordings" / f"{args.run_name}_step_{step:07d}.json",
            )
            (run / f"selfplay_{step:07d}.json").write_text(json.dumps(report, indent=2))
    (run / "status.json").write_text(
        json.dumps(dict(complete=True, updates=args.steps))
    )


if __name__ == "__main__":
    main()
