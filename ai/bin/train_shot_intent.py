#!/usr/bin/env python3
"""Simulation-only tactical learning over frozen motor skills and arrival control.

A semi-Markov Double DQN chooses a shot only when acquiring a slow puck.
Rewards accrue through navigation, contact and subsequent play until the next
choice or finite episode endpoint. Replay never treats controller ticks as
independent actor decisions. Motor policies and their encoder remain frozen.
"""

import argparse
import copy
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

from eval_foundation import load, match
from airhockey.foundation_training import FoundationEnv
from airhockey.policy_benchmark import evaluate_skills
from airhockey.shot_intent import (
    INTENTS,
    IntentValue,
    OptionAccumulator,
    ShotIntentPolicy,
)
from airhockey.thermal import DEFAULT_MODEL
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


class Replay:
    def __init__(self, capacity=200000):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.arrays = [
            np.zeros((capacity, 42), np.float32),
            np.zeros(capacity, np.int64),
            np.zeros(capacity, np.float32),
            np.zeros(capacity, np.float32),
            np.zeros((capacity, 42), np.float32),
            np.zeros(capacity, np.int64),
        ]

    def add(self, rows):
        n = len(rows[0])
        ids = (self.position + np.arange(n)) % self.capacity
        for buffer, row in zip(self.arrays, rows):
            buffer[ids] = row
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, rng, n=512):
        ids = rng.integers(self.size, size=n)
        return [torch.as_tensor(v[ids], device="cuda") for v in self.arrays]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--fallback", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--eval-every", type=int, default=200000)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261401)
    p.add_argument("--strike-accel", type=float, default=60.0)
    p.add_argument("--early-recovery", action="store_true")
    p.add_argument("--interception-tolerance", type=float)
    p.add_argument("--slow-interception", action="store_true")
    p.add_argument("--fast-prior", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    (run / "fallback").mkdir()
    shutil.copy2(args.fallback, run / "fallback/agent.pt")
    shutil.copy2(args.fallback.parent / "run.json", run / "fallback/run.json")
    selector = IntentValue().cuda()
    optimizer = torch.optim.Adam(selector.parameters(), lr=1e-4)
    old = {}
    if args.resume:
        resume_meta = json.loads((args.resume.parent / "run.json").read_text())
        if not np.array_equal(np.asarray(resume_meta.get("intents")), INTENTS):
            raise ValueError("cannot resume with different shot-intent semantics")
        old = torch.load(args.resume, map_location="cuda", weights_only=False)
        selector.load_state_dict(old["selector"])
        optimizer.load_state_dict(old["optimizer"])
    target = copy.deepcopy(selector).eval()
    if old.get("target"):
        target.load_state_dict(old["target"])
    options = dict(
        controller_options=dict(
            strike_accel=args.strike_accel, early_recovery=args.early_recovery
        ),
        interception_options={},
        fast_prior=args.fast_prior,
    )
    if args.interception_tolerance is not None:
        options["interception_options"]["contact_tolerance"] = (
            args.interception_tolerance
        )
    if args.slow_interception:
        options["interception_options"]["minimum_incoming_speed"] = 0.1

    def make_policy(head):
        return ShotIntentPolicy(load(run / "fallback/agent.pt"), head, **options)

    agent = make_policy(selector)
    peer = make_policy(IntentValue().cuda())
    peer.selector.load_state_dict(selector.state_dict())
    meta = dict(
        deployment_ready=False,
        action_mode="shot_intent",
        obs_dim=42,
        action_dim=len(INTENTS),
        intents=INTENTS.tolist(),
        algorithm=agent.training_algorithm,
        inference_mode="prior",
        action_hz=50,
        agent_speed_range=[12, 12],
        agent_accel_range=[60, 60],
        motion_guard=True,
        gamma=0.999,
        fallback_sha256=hashlib.sha256(args.fallback.read_bytes()).hexdigest(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        transfer="Frozen complete neural motor policy; new tactical value head only",
        episode_end="Finite skill/game episodes, including the timeout, are terminal for tactical returns",
        thermal_model=json.loads(DEFAULT_MODEL.read_text()),
        source_hashes={},
        **options,
    )
    for path in [
        Path(__file__),
        ROOT / "ai/bin/eval_foundation.py",
        *sorted((ROOT / "ai/airhockey").glob("*.py")),
        ROOT / "fw/include/motion_profile.h",
        ROOT / "fw/host/intercept_motion.cpp",
    ]:
        dest = run / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        meta["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    replay = Replay()
    if old.get("replay") is not None:
        # Replay remains valid only when its frozen controller is unchanged.
        resume_meta = json.loads((args.resume.parent / "run.json").read_text())
        if all(
            resume_meta.get(k, {} if k.endswith("options") else False) == v
            for k, v in options.items()
        ):
            replay.add([np.asarray(row) for row in old["replay"]])
    accumulator = OptionAccumulator(args.n_envs)
    env = FoundationEnv(
        args.n_envs,
        seed=args.seed,
        game_fraction=0.5,
        selfplay_fraction=0.75,
        possession_fraction=1,
        realistic=True,
        randomize=True,
        load_soft_start=0.85,
    )
    env.base.max_episode_time = 30
    obs = env.reset(seed=args.seed)
    previous_done = np.ones(args.n_envs, bool)
    stopped = False
    stage, updates, step, best = 0, int(old.get("updates", 0)), 0, -np.inf
    evaluations = []
    choices = np.zeros(len(INTENTS), int)
    start = time.perf_counter()

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save(
            dict(
                selector=selector.state_dict(),
                optimizer=optimizer.state_dict(),
                target=target.state_dict(),
                stage=stage,
                step=step,
                updates=updates,
                replay=[row[: replay.size] for row in replay.arrays],
            ),
            tmp,
        )
        tmp.replace(path)

    def assess():
        nonlocal best, stage
        snapshot = make_policy(copy.deepcopy(selector).eval())
        skills = evaluate_skills(
            snapshot,
            legacy=True,
            wide=True,
            guard=True,
            planner=False,
            game_requests=True,
            random_paddle=True,
            seed=20261125,
            per_task=100,
        )
        rates = {k: v["successes"] / v["attempts"] for k, v in skills["tasks"].items()}
        report = dict(step=step, skills=skills, stage=stage)
        if step:
            rival = make_policy(IntentValue().cuda())
            game = match(
                snapshot, rival, games=8, seconds=60, guard=True, seed=20261126
            )
            report["versus_fixed_arrival"] = game
            goals_for, goals_against = (
                sum(game["goals_for"]),
                sum(game["goals_against"]),
            )
            safe = (
                max(game["peak_load"][0]) < 0.95 and max(game["peak_accel"][0]) < 60.1
            )
            score = goals_for - goals_against
            qualified = (
                safe
                and rates["stationary"] >= 0.9
                and rates["moving"] >= 0.6
                and rates["defense"] >= 0.95
            )
            if qualified and score > best:
                best = score
                save(run / "agent_best.pt")
                # Curriculum advances only after measured skills AND game/load evidence.
                if goals_for > goals_against + 2:
                    stage = 1
                    env.game_fraction = 0.75
                    peer.selector.load_state_dict(selector.state_dict())
                    save(run / "agent_accepted.pt")
            report.update(score_difference=score, qualified=qualified)
            sp = match(
                snapshot,
                snapshot,
                games=8,
                seconds=30,
                guard=True,
                seed=20261127,
                record=ROOT / "ai/recordings" / f"{args.run_name}_step_{step:07d}.json",
            )
            (run / f"selfplay_{step:07d}.json").write_text(json.dumps(sp, indent=2))
        evaluations.append(report)
        (run / "evaluations.json").write_text(json.dumps(evaluations, indent=2))
        print(
            "[evaluation]",
            json.dumps(
                dict(
                    step=step,
                    rates=rates,
                    score_difference=report.get("score_difference"),
                    qualified=report.get("qualified"),
                    stage=stage,
                )
            ),
            flush=True,
        )

    save(run / "agent_initial.pt")
    assess()
    next_eval, next_log = args.eval_every, 16384
    loss = torch.zeros((), device="cuda")
    with (run / "metrics.jsonl").open("w", buffering=1) as stream:
        while step < args.steps and not stopped:
            decision = agent.decisions(obs, previous_done)
            if decision.any():
                replay.add(accumulator.finish(decision, obs))
            with torch.no_grad():
                selected = agent.selected.copy()
                if decision.any():
                    selected[decision] = (
                        selector(torch.as_tensor(obs[decision], device="cuda"))
                        .argmax(-1)
                        .cpu()
                        .numpy()
                    )
                    epsilon = max(0.1, 0.4 - 0.3 * step / 1000000)
                    explore = decision & (rng.random(args.n_envs) < epsilon)
                    selected[explore] = rng.integers(len(INTENTS), size=explore.sum())
                    choices += np.bincount(selected[decision], minlength=len(INTENTS))
                    accumulator.start(decision, obs, selected)
                action = agent.act_selected(
                    torch.from_numpy(obs), selected, decision, previous_done
                ).numpy()
                opposite = peer.act(
                    torch.from_numpy(env.opponent_obs()), t0=previous_done
                ).numpy()
            env.set_opponent_action(opposite)
            nxt, reward, term, trunc, info = env.step(action)
            done = term | trunc
            accumulator.add_reward(reward / 100)
            if done.any():
                replay.add(accumulator.finish(done, nxt, terminal=True))
                reset = env.reset(mask=done)
                nxt[done] = reset[done]
            obs, previous_done = nxt, done.copy()
            step += args.n_envs
            if replay.size >= 512 and (step // args.n_envs) % 8 == 0:
                o, a, r, discount, no, _ = replay.sample(rng)
                with torch.no_grad():
                    selected_next = selector(no).argmax(-1)
                    bootstrap = target(no).gather(1, selected_next[:, None]).squeeze(1)
                    expected = r + discount * bootstrap
                prediction = selector(o).gather(1, a[:, None]).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(prediction, expected)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 5)
                optimizer.step()
                updates += 1
                with torch.no_grad():
                    for dest, src in zip(target.parameters(), selector.parameters()):
                        dest.lerp_(src, 0.01)
            if step >= next_log:
                row = dict(
                    step=step,
                    fps=step / (time.perf_counter() - start),
                    stage=stage,
                    options=replay.size,
                    updates=updates,
                    loss=float(loss.detach()),
                    load_peak=float(env.loads[0].levels.max()),
                    choices=choices.tolist(),
                )
                if (
                    not np.isfinite(obs).all()
                    or not np.isfinite([row["loss"], row["load_peak"]]).all()
                ):
                    raise RuntimeError("nonfinite tactical training state")
                stream.write(json.dumps(row) + "\n")
                print("[train]", json.dumps(row), flush=True)
                next_log += 16384
            if step >= next_eval:
                save(run / f"agent_step_{step:07d}.pt")
                save(run / "agent.pt")
                assess()
                next_eval += args.eval_every
    save(run / "agent.pt")
    (run / "status.json").write_text(
        json.dumps(
            dict(step=step, stopped=stopped, complete=step >= args.steps, stage=stage),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
