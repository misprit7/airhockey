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
                 total_steps: int):
        self.env = env
        self.shaper = shaper
        self.total_steps = total_steps
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
        shaped += raw * 0.0  # keep dtype float32 via shaper; raw kept in infos
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


def _make_env(args, n_envs=None):
    return BatchAirHockeyEnv(
        n_envs=n_envs or args.n_envs,
        obs_mode="history",
        action_mode="profile_v",
        agent_dynamics="profile",
        opponent_dynamics="delayed",
        opponent_policy=args.opponent,
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
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--buffer", type=int, default=1_000_000)
    p.add_argument("--train-freq", type=int, default=1,
                   help="vec steps between training bursts")
    p.add_argument("--grad-steps", type=int, default=32,
                   help="gradient steps per burst (UTD = this*batch over "
                        "train_freq*n_envs new transitions)")
    args = p.parse_args()

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args)
    shaper = BatchRewardShaper(args.n_envs, stage=args.stage)
    vec = BatchVecEnv(env, shaper, total_steps=args.steps)

    # Their scale: a few hundred k parameters, LR decaying an order of
    # magnitude or so over training.
    def lr_schedule(progress_remaining):
        return 1e-5 + (args.lr - 1e-5) * progress_remaining

    model = SAC(
        "MlpPolicy", vec,
        learning_rate=lr_schedule,
        buffer_size=args.buffer,
        batch_size=args.batch_size,
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
