"""Training helpers; imported without running training or touching hardware."""

import json
from pathlib import Path
import numpy as np
from airhockey.arrival_env import ArrivalEnv
from airhockey.skill_benchmark import make_fixtures, SkillTrials
from airhockey.sequence_replay import EpisodeBatch


def transfer_encoder(agent, path):
    """Reuse visual/kinematic representation, reset changed action semantics.

    Do not copy the old dynamics, reward, Q or actor: position commands and
    arrival actions represent different transitions. Optimizer state is fresh.
    """
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    target = agent.model._encoder.state_dict()
    copied = []
    for name, value in target.items():
        old = sd.get("_encoder." + name)
        if old is None:
            continue
        if name == "state.0.weight":
            value.zero_()
            pairs = [(i, i) for i in range(15)] + [
                (18, 21),
                (19, 22),
                (20, 23),
                (21, 24),
            ]
            for src, dst in pairs:
                if src < old.shape[1]:
                    value[:, dst].copy_(old[:, src])
        elif old.shape == value.shape:
            value.copy_(old)
        else:
            raise ValueError("encoder model-size mismatch")
        copied.append(name)
    agent.model._encoder.load_state_dict(target)
    if not copied:
        raise ValueError("checkpoint has no compatible encoder")
    return dict(
        source=str(Path(path).resolve()),
        copied_encoder_keys=copied,
        reset=["policy", "dynamics", "reward", "termination", "Q", "optimizer"],
    )


def scheduled_action(env, intent, elapsed):
    """Turn a teacher's fixed arrival intent into legal fresh 50 Hz actions.

    At/after the original deadline command a stop at its follow-through point.
    These are the actual actions stored in replay, not the benchmark's latent
    constant intent. Thus the learned policy sees its own interface during BC.
    """
    p, v, duration, frac = env.decoder.unpack(intent)
    elapsed = np.broadcast_to(elapsed, (len(intent),))
    rem = np.maximum(duration - elapsed, env.decoder.config.min_time)
    action = intent.copy()
    action[:, 4] = 2 * (rem - 0.04) / 0.21 - 1
    finish = elapsed >= duration
    terminal = p + v * (np.linalg.norm(v, axis=1) / (1.28 * frac * 60))[:, None]
    action[finish, :2] = (
        2 * (terminal[finish] - env.decoder.low) / (env.decoder.high - env.decoder.low)
        - 1
    )
    action[finish, 2:4] = 0
    # Recover physically after the attempt; do not teleport or edit velocity.
    recover = elapsed >= 0.32
    if np.any(recover):
        # Keep the fixed terminal braking point, avoiding chasing actual state.
        action[recover, 2:4] = 0
    return np.clip(action, -1, 1).astype(np.float32)


def teacher_rollout(
    fixtures,
    intents,
    *,
    record=False,
    realistic=False,
    seed=0,
    randomize=False,
    require_on_target=False,
):
    env = ArrivalEnv(
        len(fixtures.task), realistic=realistic, randomize=randomize, seed=seed
    )
    obs = env.reset(seed=seed, fixtures=fixtures)
    store = EpisodeBatch(env.n_envs, env.obs_dim, 6, max_steps=105) if record else None
    if store:
        store.reset(obs)
    done = np.zeros(env.n_envs, bool)
    success = np.zeros_like(done)
    reward_sum = np.zeros(env.n_envs)
    episodes = {}
    for tick in range(101):
        action = scheduled_action(env, intents, tick * 0.02)
        obs, reward, term, trunc, info = env.step(action)
        if store:
            store.append(obs, action, reward, term, demo=1.0)
        reward_sum += np.where(done, 0, reward)
        just = (term | trunc) & ~done
        success[just] = info["success"][just]
        legal = info["peak_accel"] <= env.base._agent_dyn["max_accel"] * 1.05
        success[just] &= legal[just]
        reward_sum[just & ~legal] -= 1000
        if require_on_target:
            shots = just & (fixtures.task < 2)
            success[shots] &= (info["contacts"][shots] > 0) & (
                info["on_target"][shots] > 0
            )
        if store:
            for i in np.flatnonzero(just):
                episodes[int(i)] = store.episode_at(i)
        done |= just
        if done.all():
            break
    return reward_sum, success, episodes


def generate_demonstrations(
    output,
    *,
    per_task=32,
    population=24,
    iterations=3,
    seed=20260930,
    canonical=False,
    wide=False,
):
    """CEM on training-interface trajectories, retain only measured successes.

    Fresh seeds are disjoint from the prior arrival benchmark and evaluation.
    Noise replay is re-qualified: clean success alone is insufficient for BC.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    f = make_fixtures(seed, per_task)
    if wide:
        rng_wide = np.random.default_rng(seed + 17)
        trials = SkillTrials()
        lo, hi = trials.decoder.low, trials.decoder.high
        r = trials.cfg.puck_radius + trials.cfg.paddle_radius
        shoot = f.task < 2
        f.puck[shoot, 0] = rng_wide.uniform(lo[0] + 0.03, hi[0] - 0.03, shoot.sum())
        f.puck[shoot, 1] = rng_wide.uniform(
            lo[1] + r + 0.04, hi[1] + r - 0.03, shoot.sum()
        )
        f.paddle[shoot] = np.clip(
            f.puck[shoot, :2]
            + rng_wide.uniform([-0.08, -0.20], [0.08, -0.12], (shoot.sum(), 2)),
            lo + 0.01,
            hi - 0.01,
        )
        f.puck[f.task == 1, 2:] = rng_wide.uniform(
            [-0.65, -0.35], [0.65, 0.25], (per_task, 2)
        )
    n = len(f.task)
    mean = SkillTrials().seed_arrival(f)
    rng = np.random.default_rng(seed + 1)
    # Canonical teachers share timing, velocity rule and effort ceiling rather
    # than selecting unrelated successful action modes that BC averages badly.
    sigma = (
        np.broadcast_to([0.035, 0.025, 0, 0, 0, 0], (n, 6)).copy()
        if canonical
        else np.full((n, 6), 0.20)
    )
    floor = np.array([0.003, 0.003, 0, 0, 0, 0]) if canonical else 0.035
    best = mean.copy()
    best_score = np.full(n, -np.inf)
    for iteration in range(iterations):
        candidates = np.clip(
            mean[:, None, :] + sigma[:, None, :] * rng.normal(size=(n, population, 6)),
            -1,
            1,
        )
        candidates[:, 0] = best
        expanded = f.take(np.repeat(np.arange(n), population))
        scores, success, _ = teacher_rollout(
            expanded, candidates.reshape(-1, 6), seed=seed
        )
        scores = scores.reshape(n, population)
        pick = scores.argmax(axis=1)
        top = scores[np.arange(n), pick]
        improve = top > best_score
        best[improve] = candidates[np.arange(n), pick][improve]
        best_score = np.maximum(best_score, top)
        elite_idx = np.argsort(scores, axis=1)[:, -max(4, population // 5) :]
        elite = np.take_along_axis(candidates, elite_idx[:, :, None], axis=1)
        mean = elite.mean(axis=1)
        sigma = np.maximum(elite.std(axis=1), floor)
        print(
            f"[demos] CEM {iteration + 1}/{iterations}, candidate successes {int(success.sum())}/{len(success)}",
            flush=True,
        )
    episodes = []
    counts = np.zeros((3, 2), int)
    # Multiple independent realistic sensor realizations per optimized fixture.
    for repeat in range(4):
        _, success, eps = teacher_rollout(
            f,
            best,
            record=True,
            realistic=True,
            seed=seed + 100 + repeat,
            randomize=True,
            require_on_target=True,
        )
        for i in range(n):
            counts[f.task[i], 1] += 1
            if success[i]:
                episodes.append(eps[i])
                counts[f.task[i], 0] += 1
    np.savez_compressed(
        output / "teachers.npz",
        task=f.task,
        puck=f.puck,
        paddle=f.paddle,
        aim=f.aim,
        actions=best,
    )
    # Ragged episodes serialize as numbered arrays without pickle.
    data = {f"{i}_{k}": v for i, e in enumerate(episodes) for k, v in e.items()}
    np.savez_compressed(output / "episodes.npz", **data)
    summary = {
        "seed": seed,
        "per_task": per_task,
        "population": population,
        "iterations": iterations,
        "canonical": canonical,
        "wide": wide,
        "qualified_by_task": counts.tolist(),
        "episodes": len(episodes),
        "steps": sum(len(e["obs"]) - 1 for e in episodes),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[demos]", summary, flush=True)
    if len(episodes) < 12 or np.any(counts[:, 0] < 4):
        raise RuntimeError("insufficient successful demonstrations")
    return episodes


def load_demonstrations(path):
    data = np.load(Path(path) / "episodes.npz")
    indices = sorted({int(k.split("_")[0]) for k in data.files})
    return [
        {k: data[f"{i}_{k}"] for k in ("obs", "action", "reward", "terminated", "demo")}
        for i in indices
    ]
