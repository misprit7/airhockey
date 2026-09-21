#!/usr/bin/env python3
"""Simulation-only direct policy improvement from real environment rollouts.

Retains the full checkpoint; uses its deterministic actor, not latent MPC.
Clipped likelihood updates and KL stopping constrain changes to precise skills.
The inherited world model is not updated or used to rank actions in this run.
"""

import argparse
import copy
import hashlib
import json
import signal
import time
from pathlib import Path

from train_arrival import ROOT, config, save, TDMPC2, torch, np
from eval_foundation import load, match
from airhockey.foundation_training import FoundationEnv, skill_gate
from airhockey.on_policy import advantages, exploration_scale
from airhockey.policy_benchmark import evaluate_skills
from airhockey.thermal import DEFAULT_MODEL
from airhockey.interception import InterceptionController, InterceptionPolicy
from airhockey.batch_env import _OPP_POLICY_MAP


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--optimizer-state", type=Path)
    p.add_argument("--reference-checkpoint", type=Path)
    p.add_argument("--initial-stage", type=int, choices=(0, 1, 2))
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--actor-lr", type=float, default=3e-6)
    p.add_argument("--context-lr", type=float, default=3e-6)
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--interception", action="store_true")
    p.add_argument("--demonstrations", type=Path)
    p.add_argument("--demo-weight", type=float, default=1.0)
    p.add_argument("--precision-exploration", type=float, default=0.03)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--possession-fraction", type=float, default=0.0)
    p.add_argument("--setup-potential-weight", type=float, default=0.0)
    p.add_argument("--load-soft-start", type=float, default=0.65)
    p.add_argument("--setup-exploration", type=float)
    p.add_argument("--urgent-accel-exploration", type=float)
    p.add_argument("--seed", type=int, default=20261210)
    args = p.parse_args()
    for value in (
        args.setup_exploration,
        args.urgent_accel_exploration,
        args.precision_exploration,
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("exploration standard deviations must be positive")
    if not np.isfinite(args.demo_weight) or args.demo_weight < 0:
        raise ValueError("demo weight must be finite and nonnegative")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    cfg = config(args, run)
    cfg.obs_shape, cfg.action_dim, cfg.mpc = {"state": [42]}, 3, False
    agent = TDMPC2(cfg)
    agent.load(args.checkpoint)
    agent.training_algorithm = "PPO"
    controller = InterceptionController() if args.interception else None
    evaluated_policy = InterceptionPolicy(agent) if args.interception else agent
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.model._pi.parameters():
        parameter.requires_grad_(True)
    # Introduce task/load context without changing any original input columns
    # or the rest of the inherited representation.
    context = agent.model._encoder["state"][0].weight
    context.requires_grad_(True)
    mask = torch.ones_like(context)
    mask[:, :22] = 0
    if not args.unfreeze_encoder:
        context.register_hook(lambda grad: grad * mask)
    extra_encoder = [p for p in agent.model._encoder.parameters() if p is not context]
    if args.unfreeze_encoder:
        for parameter in extra_encoder:
            parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": agent.model._pi.parameters(), "lr": args.actor_lr},
            {"params": [context], "lr": args.context_lr},
        ]
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(42, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 1),
    ).cuda()
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    resume_stage = 0
    if args.optimizer_state:
        state = torch.load(
            args.optimizer_state, map_location="cuda", weights_only=False
        )
        expected = state.get("checkpoint_sha256")
        actual = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if expected is not None and expected != actual:
            raise ValueError("optimizer state does not belong to this checkpoint")
        critic.load_state_dict(state["critic"])
        if state.get("unfreeze_encoder", False):
            if not args.unfreeze_encoder:
                raise ValueError(
                    "a fully trained encoder must resume with --unfreeze-encoder"
                )
            optimizer.add_param_group({"params": extra_encoder, "lr": args.context_lr})
        optimizer.load_state_dict(state["actor_optimizer"])
        value_optimizer.load_state_dict(state["critic_optimizer"])
        optimizer.param_groups[0]["lr"] = args.actor_lr
        optimizer.param_groups[1]["lr"] = args.context_lr
        for group in optimizer.param_groups[2:]:
            group["lr"] = args.context_lr
        status = args.optimizer_state.parent / "status.json"
        old_status = json.loads(status.read_text()) if status.exists() else {}
        resume_stage = state.get("stage", old_status.get("stage", 0))
    if args.initial_stage is not None:
        resume_stage = args.initial_stage
    if args.unfreeze_encoder and len(optimizer.param_groups) == 2:
        optimizer.add_param_group({"params": extra_encoder, "lr": args.context_lr})
    trainable = [p for p in agent.model.parameters() if p.requires_grad]
    std = torch.tensor(
        [args.precision_exploration, args.precision_exploration, 0.12], device="cuda"
    )
    demo_obs, demo_action, demo_buckets, demo_metadata = None, None, [], None
    if args.demonstrations:
        data = np.load(args.demonstrations)
        split = (
            int(data["validation_start"])
            if "validation_start" in data
            else len(data["obs"])
        )
        if not 0 < split <= len(data["obs"]):
            raise ValueError("invalid demonstration training split")
        if data["obs"].shape != (len(data["obs"]), 42) or data["action"].shape != (
            len(data["obs"]),
            3,
        ):
            raise ValueError("invalid demonstration shapes")
        if (
            not np.isfinite(data["obs"]).all()
            or not np.isfinite(data["action"]).all()
            or np.max(abs(data["action"])) > 1.00001
        ):
            raise ValueError("invalid demonstration values")
        demo_obs = torch.as_tensor(data["obs"][:split], device="cuda")
        demo_action = torch.as_tensor(data["action"][:split], device="cuda")
        phase = data["phase"][:split] if "phase" in data else np.zeros(split)
        demo_buckets = [
            torch.as_tensor(np.flatnonzero(phase == k), device="cuda")
            for k in np.unique(phase)
        ]
        demo_metadata = dict(
            path=str(args.demonstrations),
            training_samples=split,
            heldout_samples=len(data["obs"]) - split,
            sha256=hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
        )

    def mean(obs):
        return agent.model._pi(agent.model.encode(obs, None))[..., :3]

    def logprob(raw, mu, scale):
        # The tanh Jacobian cancels in the ratio for a fixed sampled action.
        return (-0.5 * ((raw - mu) / scale).square()).sum(-1)

    def exploration(obs):
        return exploration_scale(
            obs, std, args.setup_exploration, args.urgent_accel_exploration
        )

    meta = dict(
        deployment_ready=False,
        action_mode="profile_a",
        obs_dim=42,
        action_dim=3,
        model_size=5,
        horizon=8,
        action_hz=50,
        agent_speed_range=[12, 12],
        agent_accel_range=[60, 60],
        motion_guard=True,
        inference_mode="prior",
        algorithm="PPO",
        controller="observed_interception_v1" if args.interception else None,
        transfer=(
            "Complete checkpoint; actor and full encoder train; world model retained but unused"
            if args.unfreeze_encoder
            else "Complete checkpoint; actor and new context columns train; world model retained but unused"
        ),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        exploration_std=std.tolist(),
        demonstrations=demo_metadata,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        thermal_model=json.loads(DEFAULT_MODEL.read_text()),
        source_hashes={},
    )
    for path in [
        Path(__file__),
        *sorted((ROOT / "ai/airhockey").glob("*.py")),
        ROOT / "fw/include/motion_profile.h",
        ROOT / "fw/host/intercept_motion.cpp",
        ROOT / "fw/host/motion_limits.cpp",
        ROOT / "fw/host/motion_batch.cpp",
    ]:
        dest = run / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        meta["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    save(agent, run / "agent_initial.pt")
    opponent = TDMPC2(copy.deepcopy(cfg))
    opponent.load(args.checkpoint)
    reference = None
    if args.reference_checkpoint:
        reference = load(args.reference_checkpoint)
        reference.cfg.mpc, reference.cfg.num_samples = True, 256
        reference._plan_batch = torch.compile(
            reference._plan_batch, mode="reduce-overhead"
        )
    env = FoundationEnv(
        args.n_envs,
        seed=args.seed,
        game_fraction=(0.25, 0.5, 0.75)[resume_stage],
        selfplay_fraction=(0.5, 0.67, 0.75)[resume_stage],
        possession_fraction=args.possession_fraction,
        setup_potential_weight=args.setup_potential_weight,
        load_soft_start=args.load_soft_start,
    )
    obs = env.reset(seed=args.seed)
    previous_done = np.ones(args.n_envs, bool)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    evaluations, stage, streak, best = [], resume_stage, 0, -np.inf

    def save_training_state(checkpoint, destination):
        torch.save(
            dict(
                critic=critic.state_dict(),
                actor_optimizer=optimizer.state_dict(),
                critic_optimizer=value_optimizer.state_dict(),
                stage=stage,
                unfreeze_encoder=args.unfreeze_encoder,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            ),
            destination,
        )

    def assess(step):
        nonlocal stage, streak, best
        result = evaluate_skills(
            evaluated_policy,
            legacy=True,
            planner=False,
            wide=True,
            guard=True,
            seed=20261102,
            per_task=100,
        )
        passed, failures = skill_gate(result, stage)
        possession = None
        if args.possession_fraction:
            possession = evaluate_skills(
                evaluated_policy,
                legacy=True,
                planner=False,
                wide=True,
                guard=True,
                seed=20261102,
                per_task=100,
                game_requests=True,
                random_paddle=True,
            )
            result["possession"] = possession
            # Before adding more self-play, also require setting up shots from
            # ordinary positions rather than only finishing aligned fixtures.
            if stage:
                thresholds = (0.6, 0.45) if stage == 1 else (0.8, 0.65)
                for name, threshold in zip(("stationary", "moving"), thresholds):
                    row = possession["tasks"][name]
                    rate = row["successes"] / row["attempts"]
                    if rate < threshold:
                        failures["possession_" + name] = dict(
                            actual=rate, required=threshold
                        )
                passed = not failures
        result.update(step=step, stage=stage, gate_failures=failures)
        scores = [v["successes"] / v["attempts"] for v in result["tasks"].values()]
        if possession:
            scores.extend(
                possession["tasks"][name]["successes"]
                / possession["tasks"][name]["attempts"]
                for name in ("stationary", "moving")
            )
        score = float(np.log(np.maximum(scores, 0.01)).mean())
        if score > best:
            best = score
            save(agent, run / "agent_best_skills.pt")
        streak = streak + 1 if passed else 0
        if streak >= 2:
            report = match(
                evaluated_policy,
                evaluated_policy,
                games=8,
                seconds=180,
                guard=True,
                seed=20261121,
            )
            (run / f"load_gate_{step:07d}.json").write_text(
                json.dumps(report, indent=2)
            )
            safe = (
                max(map(max, report["peak_load"])) < 0.95
                and max(map(max, report["peak_accel"])) < 60.1
            )
            result["sustained_load_gate"] = safe
            if safe:
                save(agent, run / "agent_accepted.pt")
                opponent.load(run / "agent_accepted.pt")
                stage = min(2, stage + 1)
                env.game_fraction = (0.25, 0.5, 0.75)[stage]
                env.selfplay_fraction = (0.5, 0.67, 0.75)[stage]
            streak = 0
        evaluations.append(result)
        (run / "evaluations.json").write_text(json.dumps(evaluations, indent=2))
        print("[evaluation]", json.dumps(result), flush=True)

    assess(0)
    step, iteration, next_eval = 0, 0, args.eval_every
    totals = np.zeros((5, 3), int)
    start = time.perf_counter()
    with (run / "metrics.jsonl").open("w", buffering=1) as stream:
        while step < args.steps and not stopped:
            rows = []
            for _ in range(args.rollout):
                ot = torch.as_tensor(obs, device="cuda")
                with torch.no_grad():
                    mu = mean(ot)
                    scale = exploration(ot)
                    raw = mu + scale * torch.randn_like(mu)
                    value = critic(ot).squeeze(-1)
                    lp = logprob(raw, mu, scale)
                    opposite_view = env.opponent_obs()
                    opposite = opponent.act(
                        torch.from_numpy(opposite_view), eval_mode=True
                    ).numpy()
                    if controller is not None:
                        peer_ids = np.flatnonzero(
                            env.base._opp_policy_id == _OPP_POLICY_MAP["external"]
                        )
                        if reference is not None:
                            peer_ids = np.setdiff1d(
                                peer_ids,
                                np.arange(0, round(args.n_envs * env.game_fraction), 2),
                            )
                        opposite[peer_ids], _ = controller(
                            opposite_view[peer_ids], opposite[peer_ids]
                        )
                    if reference is not None:
                        # Half of the external-policy game slots face the
                        # established full planner; the rest face the accepted
                        # peer policy. Scripted slots still use their own bots.
                        ids = np.arange(0, round(args.n_envs * env.game_fraction), 2)
                        view = env.opponent_obs()[ids]
                        if reference.cfg.obs_shape["state"][0] == 22:
                            view = view[:, :22]
                        torch.compiler.cudagraph_mark_step_begin()
                        opposite[ids] = reference.act(
                            torch.from_numpy(view),
                            t0=torch.from_numpy(previous_done[ids]),
                            eval_mode=True,
                        ).numpy()
                env.set_opponent_action(opposite)
                executed = raw.tanh().cpu().numpy()
                applied = np.zeros(args.n_envs, bool)
                if controller is not None:
                    executed, applied = controller(obs, executed)
                nxt, reward, term, trunc, info = env.step(executed)
                done = term | trunc
                previous_done = done.copy()
                with torch.no_grad():
                    nv = critic(torch.as_tensor(nxt, device="cuda")).squeeze(-1)
                rows.append(
                    (
                        ot,
                        raw,
                        lp,
                        value,
                        nv,
                        torch.as_tensor(
                            reward / 100, device="cuda", dtype=torch.float32
                        ),
                        torch.as_tensor(term.copy(), device="cuda"),
                        torch.as_tensor(done.copy(), device="cuda"),
                        torch.as_tensor(~applied, device="cuda"),
                    )
                )
                for i in np.flatnonzero(done):
                    totals[info["task"][i]] += [
                        1,
                        int(info["success"][i]),
                        int(info["contacts"][i] > 0),
                    ]
                if done.any():
                    reset = env.reset(mask=done)
                    nxt[done] = reset[done]
                obs = nxt
                step += args.n_envs
            ot, raw, old_lp, value, nv, reward, term, done, actor_free = [
                torch.stack(v) for v in zip(*rows)
            ]
            adv, returns = advantages(reward, value, nv, term, done)
            ot, raw = ot.flatten(0, 1), raw.flatten(0, 1)
            old_lp, adv, returns = old_lp.flatten(), adv.flatten(), returns.flatten()
            actor_free = actor_free.flatten()
            free_adv = adv[actor_free]
            if free_adv.numel() > 1:
                adv = (adv - free_adv.mean()) / free_adv.std().clamp(min=1e-6)
            kl, updates, actor_stopped = 0.0, 0, False
            demo_loss = torch.zeros((), device="cuda")
            for _ in range(args.epochs):
                for idx in torch.randperm(len(ot), device="cuda").split(1024):
                    prediction = critic(ot[idx]).squeeze(-1)
                    value_loss = 0.5 * (prediction - returns[idx]).square().mean()
                    value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    value_optimizer.step()
                    if actor_stopped:
                        continue
                    # Controller actions were not sampled from this actor.
                    # They contribute to the value targets and earlier credit,
                    # but must not be treated as actor decisions in PPO.
                    idx = idx[actor_free[idx]]
                    if not idx.numel():
                        continue
                    lp = logprob(raw[idx], mean(ot[idx]), exploration(ot[idx]))
                    ratio = (lp - old_lp[idx]).exp()
                    kl = float(((ratio - 1) - (lp - old_lp[idx])).mean().detach())
                    if kl > args.target_kl:
                        actor_stopped = True
                        continue
                    loss = -torch.minimum(
                        ratio * adv[idx], ratio.clamp(0.9, 1.1) * adv[idx]
                    ).mean()
                    if demo_buckets:
                        di = torch.cat(
                            [
                                b[torch.randint(len(b), (64,), device="cuda")]
                                for b in demo_buckets
                            ]
                        )
                        delta = mean(demo_obs[di]).tanh() - demo_action[di]
                        demo_loss = (delta.square() * delta.new_tensor([5, 5, 1])).sum(
                            -1
                        ).mean() / 11
                        loss = loss + args.demo_weight * demo_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                    optimizer.step()
                    updates += 1
            iteration += 1
            row = dict(
                step=step,
                iteration=iteration,
                stage=stage,
                fps=step / (time.perf_counter() - start),
                kl=kl,
                policy_updates=updates,
                actor_control_fraction=float(actor_free.float().mean()),
                demonstration_mse=float(demo_loss.detach()),
                value_loss=float(value_loss.detach()),
                load_peak=float(env.loads[0].levels.max()),
                task_counts=totals.tolist(),
            )
            if (
                not np.isfinite(obs).all()
                or not np.isfinite(
                    [row["kl"], row["value_loss"], row["load_peak"]]
                ).all()
            ):
                raise RuntimeError("nonfinite on-policy training state")
            stream.write(json.dumps(row) + "\n")
            print("[train]", json.dumps(row), flush=True)
            if step >= next_eval:
                checkpoint = run / f"agent_step_{step:07d}.pt"
                save(agent, checkpoint)
                save(agent, run / "agent.pt")
                assess(step)
                save_training_state(checkpoint, run / f"optimizers_step_{step:07d}.pt")
                report = match(
                    evaluated_policy,
                    evaluated_policy,
                    games=8,
                    seconds=30,
                    guard=True,
                    seed=20261122,
                    record=ROOT
                    / "ai/recordings"
                    / f"{args.run_name}_step_{step:07d}.json",
                )
                (run / f"selfplay_{step:07d}.json").write_text(
                    json.dumps(report, indent=2)
                )
                next_eval += args.eval_every
        save(agent, run / "agent.pt")
        save_training_state(run / "agent.pt", run / "optimizers.pt")
        (run / "status.json").write_text(
            json.dumps(
                dict(
                    complete=not stopped,
                    step=step,
                    stage=stage,
                    elapsed_s=time.perf_counter() - start,
                ),
                indent=2,
            )
        )
    print(f"[done] step={step} interrupted={stopped}", flush=True)


if __name__ == "__main__":
    main()
