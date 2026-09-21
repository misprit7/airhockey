#!/usr/bin/env python3
"""Simulation-only full-model continuation with retained skills and gated stages."""

import argparse
import copy
import hashlib
import json
import signal
import time
from pathlib import Path

from train_arrival import ROOT, config, save, TDMPC2, torch, np
from eval_foundation import match
from airhockey.arrival_training import load_demonstrations
from airhockey.foundation_training import FoundationEnv, MixedReplay, skill_gate
from airhockey.legacy_practice import convert_arrival_demonstrations
from airhockey.policy_benchmark import evaluate_skills
from airhockey.sequence_replay import SequenceReplay, EpisodeBatch
from airhockey.thermal import DEFAULT_MODEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--demos", type=Path, default=Path("logs/foundation/demos-wide"))
    p.add_argument(
        "--retention", type=Path, default=Path("logs/foundation/defense-teacher.npz")
    )
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261130)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--rho", type=float, default=0.7)
    p.add_argument("--terminal-fraction", type=float, default=0.0)
    p.add_argument("--actor-lr", type=float, default=0.0001)
    p.add_argument("--anchor-lr", type=float, default=0.00003)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--calibrate-exploration", action="store_true")
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Keep the transferred latent representation fixed while refining control/model heads",
    )
    p.add_argument(
        "--mean-actions",
        action="store_true",
        help="Execute the MPC elite mean during collection; candidate search still explores",
    )
    p.add_argument(
        "--self-imitate",
        action="store_true",
        help="Distill physically successful on-target/defensive planner trajectories",
    )
    p.add_argument(
        "--collect-prior",
        action="store_true",
        help="Actor-critic ablation: execute the fitted policy with small exploration instead of broad MPC search",
    )
    args = p.parse_args()
    if min(args.steps, args.n_envs, args.eval_every) < 1:
        raise ValueError("counts must be positive")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    cfg = config(args, run)
    cfg.obs_shape, cfg.action_dim = {"state": [42]}, 3
    cfg.discount_max, cfg.episode_length = 0.99, 3000
    cfg.plan_smooth_coef, cfg.plan_eval_mean = 0.0, True
    cfg.bc_coef = 2.0
    cfg.rho = args.rho
    cfg.pi_smooth_coef = 0.1
    cfg.lr, cfg.enc_lr_scale = 0.0001, 0.1
    agent = TDMPC2(cfg)
    agent.load(args.checkpoint)
    agent.pi_optim.param_groups[0]["lr"] = args.actor_lr
    if args.freeze_encoder:
        for parameter in agent.model._encoder.parameters():
            parameter.requires_grad_(False)
    if args.calibrate_exploration:
        # Retain every deterministic policy mean and world-model weight.
        # Old acceleration exploration had std ~= exp(2), producing almost
        # binary min/max-cap samples despite successful deterministic shots.
        desired = torch.tensor([0.05, 0.05, 0.2], device=agent.device).log()
        raw = torch.atanh(
            2 * (desired - agent.model.log_std_min) / agent.model.log_std_dif - 1
        )
        with torch.no_grad():
            agent.model._pi[-1].weight[3:].zero_()
            agent.model._pi[-1].bias[3:].copy_(raw)
    # Fresh optimizers, but all learned heads and encoder columns are retained.
    save(agent, run / "agent_initial.pt")
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
        inference_mode="prior" if args.collect_prior else "mpc",
        transfer=(
            "Complete checkpoint and deterministic policy retained; stochastic std outputs recalibrated; fresh optimizers"
            if args.calibrate_exploration
            else "Entire 42-input checkpoint; fresh optimizers, no head resets"
        ),
        curriculum="Advance only after two skill gates and a sustained load gate; maintain 25% practice at final stage",
        opponent="Frozen accepted policy prior with scripted opponents; evaluation uses symmetric full MPC",
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_hashes={},
        thermal_sampling="Independent warm starts for isolated fixtures; continuous heat across goals and resets in game slots",
        thermal_model=json.loads(DEFAULT_MODEL.read_text()),
        torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
        exploration_calibration=[0.05, 0.05, 0.2]
        if args.calibrate_exploration
        else None,
    )
    for path in [
        Path(__file__),
        *sorted((ROOT / "ai/airhockey").glob("*.py")),
        ROOT / "fw/host/motion_limits.cpp",
        ROOT / "fw/include/motion_profile.h",
        ROOT.parent / "tdmpc2/tdmpc2/tdmpc2.py",
    ]:
        dest = run / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        meta["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    env = FoundationEnv(
        args.n_envs, seed=args.seed, game_fraction=0.25, selfplay_fraction=0.5
    )
    episodes = convert_arrival_demonstrations(
        load_demonstrations(args.demos),
        env.base._ws,
        env.base._action_low,
        env.base._action_high,
    )
    demos = SequenceReplay(
        sum(len(e["obs"]) for e in episodes) + 1,
        42,
        3,
        cfg.horizon,
        128,
        seed=args.seed,
    )
    online = SequenceReplay(
        500_000,
        42,
        3,
        cfg.horizon,
        128,
        seed=args.seed + 1,
        terminal_fraction=args.terminal_fraction,
    )
    demos.terminal_fraction = args.terminal_fraction
    for ep in episodes:
        demos.add(ep)
    buffer = MixedReplay(demos, online)
    demo_obs = torch.as_tensor(
        np.concatenate([e["obs"][:-1] for e in episodes]), device="cuda"
    )
    demo_action = torch.as_tensor(
        np.concatenate([e["action"][1:] for e in episodes]), device="cuda"
    )
    retained = np.load(args.retention)
    ro = retained["obs"].copy()
    ro[:, 35] = 1
    retained_obs, retained_action = (
        torch.as_tensor(v, device="cuda") for v in (ro, retained["action"])
    )
    kinds = demo_obs[:, 30:33].argmax(-1)
    early = demo_obs[:, -1] < 0.32 / 30
    buckets = [
        torch.where((kinds == k) & (early == phase))[0]
        for k in range(3)
        for phase in (True, False)
    ]
    aux_params = (
        [] if args.freeze_encoder else list(agent.model._encoder.parameters())
    ) + list(agent.model._pi.parameters())
    aux_opt = torch.optim.Adam(aux_params, lr=args.anchor_lr)
    weights = torch.tensor([10, 10, 1], device="cuda") / 7

    def anchor():
        if args.anchor_lr == 0:
            return torch.zeros((), device=agent.device)
        i = torch.cat(
            [
                b[torch.randint(len(b), (60 if k % 2 == 0 else 25,), device="cuda")]
                for k, b in enumerate(buckets)
            ]
        )
        j = torch.randint(len(retained_obs), (256,), device="cuda")
        o = torch.cat((demo_obs[i], retained_obs[j]))
        a = torch.cat((demo_action[i], retained_action[j]))
        pred = agent.model._pi(agent.model.encode(o, None))[..., :3].tanh()
        loss = ((pred - a).square() * weights).sum(-1).mean()
        if args.self_imitate and online.num_eps:
            oo, aa, _, _, _, successful = online.sample_with_demo()
            online_pred = agent.model._pi(agent.model.encode(oo[0], None))[
                ..., :3
            ].tanh()
            per_sample = ((online_pred - aa[0]).square() * weights).sum(-1)
            mask = successful[0, :, 0]
            loss = loss + (per_sample * mask).sum() / mask.sum().clamp(min=1)
        aux_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(aux_params, 10)
        aux_opt.step()
        aux_opt.zero_grad(set_to_none=True)
        return loss.detach()

    if not args.eager:
        agent._plan_batch = torch.compile(agent._plan_batch, mode="reduce-overhead")
        agent._update = torch.compile(agent._update, mode="reduce-overhead")
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stage, streak, best, step = 0, 0, -np.inf, 0
    oc = copy.deepcopy(cfg)
    oc.mpc = False
    opponent = TDMPC2(oc)
    opponent.load(args.checkpoint)
    evaluations = []
    last_gate_step = -1
    best_mpc = -np.inf

    def assess(step, planner=False):
        nonlocal best, best_mpc, stage, streak, last_gate_step
        # Validation seeds used for selection; final audit uses separate seeds.
        result = evaluate_skills(
            agent,
            legacy=True,
            seed=20261102,
            per_task=100,
            planner=planner,
            wide=True,
            guard=True,
        )
        result.update(step=step, stage=stage)
        passed, failures = skill_gate(result, stage)
        result["gate_failures"] = failures
        scores = [v["successes"] / v["attempts"] for v in result["tasks"].values()]
        score = float(np.mean(np.log(np.maximum(scores, 0.01))))
        if not planner:
            if score > best:
                best = score
                save(agent, run / "agent_best_skills.pt")
        else:
            if score > best_mpc:
                best_mpc = score
                save(agent, run / "agent_best_mpc.pt")
            prior = next(
                (
                    r
                    for r in reversed(evaluations)
                    if r["step"] == step and not r["planner"]
                ),
                None,
            )
            both = (
                (prior is not None and not prior["gate_failures"])
                if args.collect_prior
                else passed
            )
            if step != last_gate_step:
                streak = streak + 1 if both else 0
                last_gate_step = step
            if streak >= 2:
                old_mpc = agent.cfg.mpc
                agent.cfg.mpc = not args.collect_prior
                sustained = match(
                    agent, agent, games=8, seconds=180, seed=20261121, guard=True
                )
                agent.cfg.mpc = old_mpc
                safe = (
                    max(map(max, sustained["peak_load"])) < 0.95
                    and max(map(max, sustained["peak_accel"])) < 60.1
                )
                (run / f"load_gate_{step:07d}.json").write_text(
                    json.dumps(sustained, indent=2)
                )
                result["sustained_load_gate"] = safe
                if safe:
                    save(agent, run / "agent_accepted.pt")
                    opponent.load(run / "agent_accepted.pt")
                    stage = min(stage + 1, 2)
                    env.game_fraction = (0.25, 0.5, 0.75)[stage]
                    env.selfplay_fraction = (0.5, 0.67, 0.75)[stage]
                streak = 0
        evaluations.append(result)
        (run / "evaluations.json").write_text(json.dumps(evaluations, indent=2))
        print("[evaluation]", json.dumps(result), flush=True)

    def preview(step):
        old_mpc = agent.cfg.mpc
        agent.cfg.mpc = not args.collect_prior
        report = match(
            agent,
            agent,
            games=16,
            seconds=30,
            seed=20261122,
            guard=True,
            record=ROOT / "ai/recordings" / f"{args.run_name}_step_{step:07d}.json",
        )
        agent.cfg.mpc = old_mpc
        (run / f"selfplay_{step:07d}.json").write_text(json.dumps(report, indent=2))

    assess(0)
    # Calibrate inherited model/value estimates to the new task rewards while
    # keeping the transferred encoder and actor exactly fixed. Resume their
    # learning rates after this short world-model calibration.
    enc_lr = agent.optim.param_groups[0]["lr"]
    pi_lr = agent.pi_optim.param_groups[0]["lr"]
    agent.optim.param_groups[0]["lr"] = 0.0
    agent.pi_optim.param_groups[0]["lr"] = 0.0
    start = time.perf_counter()
    for k in range(args.warmup):
        metrics = agent.update(buffer)
        if k % 100 == 0:
            print(
                f"[calibrate] update={k} loss={float(metrics['total_loss']):.4f} elapsed={time.perf_counter() - start:.1f}",
                flush=True,
            )
        if stopped:
            break
    agent.optim.param_groups[0]["lr"] = enc_lr
    agent.pi_optim.param_groups[0]["lr"] = pi_lr
    save(agent, run / "agent_calibrated.pt")
    assess(0)
    # Log actual planner performance separately from the policy prior.
    assess(0, planner=True)
    obs = env.reset(seed=args.seed)
    staging = EpisodeBatch(args.n_envs, 42, 3, max_steps=1502)
    staging.reset(obs)
    t0 = torch.ones(args.n_envs, dtype=torch.bool)
    updates, next_eval = 0, args.eval_every
    start = time.perf_counter()
    totals = np.zeros((5, 3), int)
    self_imitated = np.zeros(5, int)
    exploration = np.random.default_rng(args.seed + 71)
    timing = {k: 0.0 for k in ("opponent", "planner", "environment", "update")}
    with (run / "metrics.jsonl").open("w", buffering=1) as stream:
        while step < args.steps and not stopped:
            t = time.perf_counter()
            opposite = opponent.act(
                torch.from_numpy(env.opponent_obs()), eval_mode=True
            ).numpy()
            env.set_opponent_action(opposite)
            timing["opponent"] += time.perf_counter() - t
            t = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            agent.cfg.mpc = not args.collect_prior
            action = agent.act(
                torch.from_numpy(obs),
                t0=t0,
                eval_mode=args.collect_prior or args.mean_actions,
            ).numpy()
            if args.collect_prior:
                action = np.clip(
                    action
                    + exploration.normal(size=action.shape) * [0.025, 0.025, 0.1],
                    -1,
                    1,
                ).astype(np.float32)
            timing["planner"] += time.perf_counter() - t
            t = time.perf_counter()
            nxt, reward, term, trunc, info = env.step(action)
            done = term | trunc
            staging.append(nxt, action, reward, term)
            if done.any():
                for i in np.flatnonzero(done):
                    task = info["task"][i]
                    ep = staging.episode_at(i)
                    eligible = (
                        args.self_imitate
                        and task != 3
                        and info["success"][i]
                        and info["load_peak"][i] < 0.95
                        and (task >= 2 or info["on_target"][i] > 0)
                    )
                    if eligible:
                        ep["demo"][1:] = 1
                        self_imitated[task] += 1
                    online.add(ep)
                    totals[task] += [
                        1,
                        int(info["success"][i]),
                        int(info["contacts"][i] > 0),
                    ]
                reset = env.reset(mask=done)
                nxt[done] = reset[done]
                staging.reset(nxt, mask=done)
            obs, t0 = nxt, torch.from_numpy(done.copy())
            timing["environment"] += time.perf_counter() - t
            t = time.perf_counter()
            metrics = agent.update(buffer)
            aux = anchor()
            timing["update"] += time.perf_counter() - t
            updates += 1
            step += args.n_envs
            if step % 10000 < args.n_envs:
                loss = {
                    k: float(metrics[k]) for k in ("total_loss", "pi_loss", "pi_bc")
                }
                if not np.isfinite(obs).all() or not all(
                    np.isfinite(v) for v in loss.values()
                ):
                    raise RuntimeError("nonfinite training state")
                row = dict(
                    step=step,
                    stage=stage,
                    updates=updates,
                    fps=step / (time.perf_counter() - start),
                    loss=loss,
                    anchor=float(aux),
                    task_counts=totals.tolist(),
                    load_peak=float(env.loads[0].levels.max()),
                    timing_s=timing,
                    guard_unresolved=env.guard_unresolved.sum(axis=1).tolist(),
                    self_imitated_episodes=self_imitated.tolist(),
                )
                stream.write(json.dumps(row) + "\n")
                print("[train]", json.dumps(row), flush=True)
            if step >= next_eval:
                save(agent, run / f"agent_step_{step:07d}.pt")
                save(agent, run / "agent.pt")
                assess(step)
                assess(step, planner=True)
                preview(step)
                next_eval += args.eval_every
        save(agent, run / "agent.pt")
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
    if step < next_eval - args.eval_every or step % args.eval_every:
        assess(step)
    print(f"[done] step={step} stage={stage} interrupted={stopped}", flush=True)


if __name__ == "__main__":
    main()
