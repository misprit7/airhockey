#!/usr/bin/env python3
"""SAC on the batch env with history observations and profile_v actions.

The Air-Hockey-Sim-shaped recipe on OUR measured environment: a small
model-free policy (one forward pass, trivially real-time at any control
rate) reading raw position history instead of estimated velocities, and
commanding (target, speed-cap, accel-cap) segments the Teensy can execute
verbatim as MOVE + LIMITS.

Deliberately single-stage to start: the bar this exists to clear is
"plays well against a stationary/scripted opponent", so it trains against
one opponent with the matching stage shaping and measures goals directly.

    python ai/bin/train_sac.py --steps 3000000 --opponent goalie
    python ai/bin/train_sac.py --steps 1000000 --opponent idle --stage 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai"))

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs  # noqa: E402
from airhockey.recorder import FrameData, Recorder  # noqa: E402
from airhockey.rewards import BatchRewardShaper  # noqa: E402

import gymnasium as gym  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import VecEnv  # noqa: E402


class BatchVecEnv(VecEnv):
    """SB3 VecEnv over BatchAirHockeyEnv + BatchRewardShaper.

    One adapter, no per-env python objects: SB3 sees N envs, the batch env
    steps them in one call. Auto-resets internally and reports
    terminal_observation the way SB3's replay collection expects.
    """

    def __init__(self, env: BatchAirHockeyEnv, shaper: BatchRewardShaper,
                 total_steps: int, reward_scale: float = 1.0):
        self.env = env
        self.shaper = shaper
        self.total_steps = total_steps
        # SAC is sensitive to value scale: with +160-per-goal shaping the
        # critic starts around Q ~ -20k and the auto entropy coefficient
        # spikes (observed at 11) then collapses to zero. Scaling rewards
        # to O(1) is the standard fix and costs nothing semantically.
        self.reward_scale = reward_scale
        self._elapsed = 0
        obs_dim = env.obs_dim
        observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,),
                                           np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (env.action_dim,), np.float32)
        super().__init__(env.n_envs, observation_space, action_space)
        self._actions = None
        self.render_mode = None
        self._ep_rew = np.zeros(env.n_envs)
        self._ep_len = np.zeros(env.n_envs, dtype=np.int64)

    def reset(self):
        obs = self.env.reset()
        # Shaper state needs truth; first step's info is not available yet,
        # so baseline from the engine directly.
        e = self.env.engine
        info = {"puck_x": e.puck_x, "puck_y": e.puck_y,
                "pad_x": e.paddle_agent_x, "pad_y": e.paddle_agent_y,
                "puck_vx": e.puck_vx, "puck_vy": e.puck_vy,
                "score_agent": e.score_agent, "score_opponent": e.score_opponent}
        self.shaper.reset(obs, info=info)
        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, raw, term, trunc, info = self.env.step(self._actions)
        # Penalty ramp / late-stage anneal follow global progress.
        self.shaper.set_progress(min(1.0, self._elapsed / max(self.total_steps, 1)))
        shaped = self.shaper.compute(obs, raw, actions=self._actions, info=info)
        shaped = shaped * self.reward_scale
        self._elapsed += self.env.n_envs

        dones = term | trunc
        self._ep_rew += shaped
        self._ep_len += 1
        infos = [{} for _ in range(self.num_envs)]
        if np.any(dones):
            for i in np.where(dones)[0]:
                infos[i]["terminal_observation"] = obs[i].copy()
                infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
                # SB3's ep_rew_mean comes from this dict; without it the
                # run trains blind on the one chart that matters.
                infos[i]["episode"] = {"r": float(self._ep_rew[i]),
                                       "l": int(self._ep_len[i])}
                self._ep_rew[i] = 0.0
                self._ep_len[i] = 0
            new_obs = self.env.auto_reset(term, trunc)
            if new_obs is not None:
                obs = new_obs
                e = self.env.engine
                self.shaper.reset(obs, mask=dones, info={
                    "puck_x": e.puck_x, "puck_y": e.puck_y,
                    "pad_x": e.paddle_agent_x, "pad_y": e.paddle_agent_y,
                    "puck_vx": e.puck_vx, "puck_vy": e.puck_vy})
        for i in range(self.num_envs):
            infos[i]["score_agent"] = int(info["score_agent"][i])
            infos[i]["score_opponent"] = int(info["score_opponent"][i])
        return obs, shaped, dones, infos

    # -- VecEnv boilerplate ------------------------------------------------
    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [getattr(self.env, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def seed(self, seed=None):
        self.env.reset(seed=seed)
        return [seed] * self.num_envs


def record_game(model, args, step, run_name):
    """One recorded game for the web UI replay tab, n_envs=1."""
    env = _make_env(args, n_envs=1)
    obs = env.reset(seed=int(step) % 100_000)
    rec = Recorder()
    rec.start_episode()
    e = env.engine
    done = False
    t = 0
    while not done and t < 9000:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term[0] or trunc[0])
        rec.record(FrameData(
            time=float(e.time[0]),
            puck_x=float(e.puck_x[0]), puck_y=float(e.puck_y[0]),
            puck_vx=float(e.puck_vx[0]), puck_vy=float(e.puck_vy[0]),
            agent_x=float(e.paddle_agent_x[0]), agent_y=float(e.paddle_agent_y[0]),
            opponent_x=float(e.paddle_opp_x[0]), opponent_y=float(e.paddle_opp_y[0]),
            score_agent=int(e.score_agent[0]), score_opponent=int(e.score_opponent[0]),
        ))
        t += 1
    recordings = ROOT / "ai" / "recordings"
    recordings.mkdir(exist_ok=True)
    rec.save(recordings / f"{run_name}_step_{step:07d}.json",
             metadata={"step": int(step), "algo": "SAC",
                       "opponent": args.opponent})
    print(f"Recorded game at step {step:,}: "
          f"{int(e.score_agent[0])}-{int(e.score_opponent[0])}")


def collect_demos(vec, bots, bridge, n_transitions):
    """Roll the striker (or any bot) through the training env, via the same
    tested SimBridge the tournament used, gathering (obs, action, reward,
    next_obs, done) at the training reward scale."""
    import torch  # noqa: F401  (parity with caller's device use)
    obs = vec.reset()
    bridge.reset()
    n_steps = int(np.ceil(n_transitions / vec.num_envs))
    O, A, R, NO, D = [], [], [], [], []
    for _ in range(n_steps):
        reports = bridge.reports(obs)
        commands = [bot(rep) for bot, rep in zip(bots, reports)]
        actions = bridge.actions(commands, obs).astype(np.float32)
        vec.step_async(actions)
        next_obs, rew, dones, infos = vec.step_wait()
        O.append(obs.copy()); A.append(actions)
        R.append(rew.copy()); NO.append(next_obs.copy()); D.append(dones.copy())
        obs = next_obs
    return (np.concatenate(O), np.concatenate(A), np.concatenate(R),
            np.concatenate(NO), np.concatenate(D))


def dagger_rounds(model, vec, bots, bridge, args, d_obs, d_act):
    """DAgger: the clone drives, the teacher labels the states the clone
    actually reaches, the aggregate retrains the clone.

    Pure BC failed here measurably (clone 0.15 GF vs teacher 0.80): mse
    0.0007 on the TEACHER'S states says nothing about the states the
    clone's own small errors steer it into, where it has no data at all.
    Labelling exactly those states is the fix.
    """
    O = [d_obs]
    A = [d_act]
    per_round = args.bc_transitions // 2
    for r in range(args.dagger_iters):
        obs = vec.reset()
        bridge.reset()
        ro, ra = [], []
        for _ in range(int(np.ceil(per_round / vec.num_envs))):
            # Teacher labels for the CURRENT states...
            reports = bridge.reports(obs)
            commands = [bot(rep) for bot, rep in zip(bots, reports)]
            labels = bridge.actions(commands, obs).astype(np.float32)
            ro.append(obs.copy())
            ra.append(labels)
            # ...but the CLONE chooses where to go next.
            act, _ = model.predict(obs, deterministic=True)
            vec.step_async(act.astype(np.float32))
            obs, _, _, _ = vec.step_wait()
        O.append(np.concatenate(ro)); A.append(np.concatenate(ra))
        all_o, all_a = np.concatenate(O), np.concatenate(A)
        bc_pretrain(model, all_o, all_a, args.bc_epochs // 2 or 1,
                    batch_size=args.batch_size)
        print(f"  DAgger round {r + 1}/{args.dagger_iters}: "
              f"dataset {len(all_o):,}")
    return np.concatenate(O), np.concatenate(A)


def bc_pretrain(model, demo_obs, demo_act, epochs, batch_size=1024):
    """Supervised warm start: pull the actor's squashed mean onto the
    teacher's actions. The critic is deliberately untouched -- it learns from
    the demo transitions prefilled into the replay buffer instead, so its
    first gradients come from real Bellman targets rather than a guess."""
    import torch
    device = model.device
    obs_t = torch.as_tensor(demo_obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(demo_act, dtype=torch.float32, device=device)
    n = len(obs_t)
    actor = model.actor
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            mean, log_std, _ = actor.get_action_dist_params(obs_t[idx])
            loss = torch.nn.functional.mse_loss(torch.tanh(mean), act_t[idx])
            actor.optimizer.zero_grad()
            loss.backward()
            actor.optimizer.step()
            total += float(loss.detach()) * len(idx)
        if ep == 0 or ep == epochs - 1:
            print(f"  BC epoch {ep + 1}/{epochs}: mse {total / n:.5f}")


def _make_env(args, n_envs=None):
    n = n_envs or args.n_envs
    mix = None
    # The mix only fits the training env's width; the single-env recorder
    # falls back to --opponent. (Every v5 recording died on this assert.)
    if getattr(args, "opponent_mix", None) and n == (args.n_envs or n):
        # "goalie:follow:random" counts, e.g. 32:16:16. Training against a
        # single opponent taught v2-v4 exactly one trick; the striker's
        # tournament showed the failure modes differ per opponent.
        parts = [int(p) for p in args.opponent_mix.split(":")]
        assert sum(parts) == n, f"opponent-mix {parts} must sum to n_envs {n}"
        mix = {"goalie": parts[0], "follow": parts[1], "random": parts[2]}
    return BatchAirHockeyEnv(
        n_envs=n,
        obs_mode="history",
        action_mode="profile_v",
        agent_dynamics="profile",
        opponent_dynamics="delayed",
        opponent_policy=args.opponent,
        opponent_mix=mix,
        action_dt=1 / 100,
        max_episode_steps=2000,          # 20 s exchanges
        max_score=7,
        domain_randomize=args.domain_randomize,
        **sensing_kwargs(args.realistic_sensing),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--opponent", type=str, default="goalie",
                   choices=["idle", "goalie", "follow", "random"])
    p.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4],
                   help="reward shaping stage")
    p.add_argument("--run-name", type=str, default="sac_v1")
    p.add_argument("--record-freq", type=int, default=100_000)
    p.add_argument("--realistic-sensing", action="store_true", default=True)
    p.add_argument("--no-realistic-sensing", dest="realistic_sensing",
                   action="store_false")
    p.add_argument("--domain-randomize", action="store_true", default=True)
    p.add_argument("--no-domain-randomize", dest="domain_randomize",
                   action="store_false")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--buffer", type=int, default=1_000_000)
    p.add_argument("--reward-scale", type=float, default=0.1,
                   help="multiply shaped rewards; SAC wants O(1) values")
    p.add_argument("--ent-coef", type=str, default="0.02",
                   help="SB3 ent_coef. FIXED by default: auto collapsed to "
                        "~0 in both v4 (spike to 11 first) and v5 "
                        "(auto_0.1, target -2), killing exploration")
    p.add_argument("--target-entropy", type=float, default=-2.0,
                   help="entropy target (default -dim(A) was -4)")
    p.add_argument("--defense-weight", type=float, default=0.3,
                   help="override stage defense shaping (stage-2 default 0.1); "
                        "v4 conceded only on reset launches it never learned "
                        "to meet")
    p.add_argument("--opponent-mix", type=str, default=None,
                   help="'g:f:r' env counts, e.g. 32:16:16; overrides --opponent")
    p.add_argument("--bc-bot", type=str, default="striker",
                   help="heuristic teacher for the warm start; 'none' disables")
    p.add_argument("--bc-transitions", type=int, default=200_000)
    p.add_argument("--bc-epochs", type=int, default=25)
    p.add_argument("--dagger-iters", type=int, default=4,
                   help="rounds of clone-drives/teacher-labels; 0 disables")
    p.add_argument("--bc-refresh-epochs", type=int, default=2,
                   help="BC epochs re-anchoring the actor between chunks; "
                        "0 disables")
    p.add_argument("--train-freq", type=int, default=1,
                   help="vec steps between training bursts")
    p.add_argument("--grad-steps", type=int, default=32,
                   help="gradient steps per burst (UTD = this*batch over "
                        "train_freq*n_envs new transitions)")
    args = p.parse_args()

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args)
    shaper = BatchRewardShaper(args.n_envs, stage=args.stage,
                               defense_weight=args.defense_weight)
    vec = BatchVecEnv(env, shaper, total_steps=args.steps,
                      reward_scale=args.reward_scale)

    # Their scale: a few hundred k parameters, LR decaying an order of
    # magnitude or so over training.
    def lr_schedule(progress_remaining):
        return 1e-5 + (args.lr - 1e-5) * progress_remaining

    model = SAC(
        "MlpPolicy", vec,
        learning_rate=lr_schedule,
        buffer_size=args.buffer,
        batch_size=args.batch_size,
        ent_coef=(args.ent_coef if args.ent_coef.startswith("auto")
                  else float(args.ent_coef)),
        target_entropy=args.target_entropy,
        # train_freq counts VEC steps: one vec step is n_envs transitions.
        # The first run used (64, "step") x 64 envs = one training burst per
        # 4,096 transitions -- ~12k gradient steps across 3M transitions,
        # SAC starved to nothing at a very impressive-looking 22k fps.
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.grad_steps,
        learning_starts=20_000,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        tensorboard_log=str(run_dir / "logs"),
        verbose=1,
        device="cuda",
    )

    print(f"SAC {args.run_name}: {args.steps:,} steps, {args.n_envs} envs, "
          f"opponent={args.opponent}, stage={args.stage}, "
          f"sensing={'on' if args.realistic_sensing else 'off'}, "
          f"DR={'on' if args.domain_randomize else 'off'}")
    print(f"  obs {env.obs_dim} dims (history), action {env.action_dim} dims "
          f"(profile_v)")

    # ── Imitation warm start from a heuristic teacher ────────────────────
    # The striker scores ~4x what any trained policy has managed through the
    # identical interface; nobody has to learn air hockey from scratch any
    # more. The actor is pulled onto the teacher's actions; the teacher's
    # transitions also prefill the replay buffer so the critic's first
    # Bellman targets are computed on competent play.
    if args.bc_bot != "none":
        from airhockey.heuristic_bridge import SimBridge
        from airhockey.heuristics import make_bot
        print(f"Collecting {args.bc_transitions:,} demo transitions from "
              f"'{args.bc_bot}'...")
        bridge = SimBridge(env)
        bots = [make_bot(args.bc_bot) for _ in range(args.n_envs)]
        d_obs, d_act, d_rew, d_next, d_done = collect_demos(
            vec, bots, bridge, args.bc_transitions)
        per_step = float(d_rew.mean())
        print(f"  demo reward/step {per_step:.3f} (scaled); "
              f"{len(d_obs):,} transitions")
        n_env = args.n_envs
        for i in range(0, len(d_obs) - n_env + 1, n_env):
            sl = slice(i, i + n_env)
            model.replay_buffer.add(d_obs[sl], d_next[sl], d_act[sl],
                                    d_rew[sl], d_done[sl],
                                    [{} for _ in range(n_env)])
        print(f"BC pretraining actor: {args.bc_epochs} epochs...")
        bc_pretrain(model, d_obs, d_act, args.bc_epochs,
                    batch_size=args.batch_size)
        # The clone itself is an artifact: v5 proved RL can DESTROY it (a
        # 0.0007-mse striker clone came out of 5M steps scoring a quarter
        # of the teacher and conceding 20x). Whatever fine-tuning does,
        # this checkpoint survives it.
        model.save(run_dir / "agent_bc_only")
        print(f"BC-only actor saved to {run_dir}/agent_bc_only.zip")
        if args.dagger_iters > 0:
            print(f"DAgger: {args.dagger_iters} rounds...")
            d_obs, d_act = dagger_rounds(model, vec, bots, bridge, args,
                                         d_obs, d_act)
            model.save(run_dir / "agent_dagger")
            print(f"DAgger actor saved to {run_dir}/agent_dagger.zip")

    remaining = args.steps
    chunk = args.record_freq
    done_steps = 0
    t0 = time.time()
    while remaining > 0:
        n = min(chunk, remaining)
        model.learn(total_timesteps=n, reset_num_timesteps=False,
                    progress_bar=False, log_interval=50)
        done_steps += n
        remaining -= n
        # Re-anchor: a few BC epochs on the demo set between chunks keeps
        # the actor from drifting off the teacher while the critic is still
        # wrong about the world. Crude next to a per-update BC term in the
        # actor loss, but it has the right failure mode: at worst the
        # policy stays a striker clone, which v5 shows beats free-running
        # SAC by a wide margin.
        if args.bc_bot != "none" and args.bc_refresh_epochs > 0 and remaining > 0:
            bc_pretrain(model, d_obs, d_act, args.bc_refresh_epochs,
                        batch_size=args.batch_size)
        model.save(run_dir / "agent")
        fps = done_steps / max(time.time() - t0, 1)
        print(f"[{done_steps:,}/{args.steps:,}] fps={fps:.0f}, saved {run_dir}/agent.zip")
        try:
            record_game(model, args, done_steps, args.run_name)
        except Exception as exc:                      # noqa: BLE001
            print(f"recording failed: {exc}")

    print("Training complete!")


if __name__ == "__main__":
    main()
