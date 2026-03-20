"""Train an air hockey RL agent with PPO.

PPO is chosen because:
- Stable and forgiving of hyperparameter choices
- Works well with continuous action spaces
- Good sample efficiency for on-policy
- Easy to parallelize with vectorized envs

Usage:
    python bin/train.py
    python bin/train.py --timesteps 1_000_000 --n-envs 8
    python bin/train.py --resume runs/ppo_airhockey_latest/best_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from airhockey.dynamics import DelayedDynamics, IdealDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.rewards import ShapedRewardWrapper


class RecordGameCallback(BaseCallback):
    """Records a full game episode every `record_freq` timesteps.

    Saves recordings to the recordings/ directory so they can be
    viewed in the web UI replay system. Filenames include the step
    count for easy chronological browsing.
    """

    def __init__(
        self,
        record_freq: int,
        opponent: str,
        use_dynamics: bool,
        recordings_dir: Path,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.opponent = opponent
        self.use_dynamics = use_dynamics
        self.recordings_dir = recordings_dir
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._last_record_step = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        # Check if we've crossed a recording boundary
        if step // self.record_freq > self._last_record_step // self.record_freq:
            self._last_record_step = step
            self._record_game(step)
        return True

    def _record_game(self, step: int) -> None:
        env = make_env(
            opponent=self.opponent,
            use_dynamics=self.use_dynamics,
            record=True,
        )
        obs, _ = env.reset(seed=0)  # fixed seed for comparable replays
        terminated, truncated = False, False

        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

        inner_env = env.unwrapped
        recording = inner_env.get_recording()
        if recording:
            rec = Recorder()
            rec._current = recording
            step_str = f"{step:07d}"
            filename = f"train_step_{step_str}.json"
            rec.save(self.recordings_dir / filename)
            score = f"{info['score_agent']}-{info['score_opponent']}"
            if self.verbose:
                print(f"Recorded game at step {step:,}: {score}")


def make_env(
    opponent: str = "follow",
    use_dynamics: bool = False,
    record: bool = False,
) -> gym.Env:
    dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else IdealDynamics()
    env = AirHockeyEnv(
        agent_dynamics=dynamics,
        opponent_policy=opponent,
        record=record,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    env = ShapedRewardWrapper(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train air hockey agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--opponent", type=str, default="idle")
    parser.add_argument("--dynamics", action="store_true", help="Use delayed motor dynamics")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--run-name", type=str, default="ppo_airhockey")
    parser.add_argument("--record-freq", type=int, default=50_000, help="Record a game every N steps")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Training for {args.timesteps:,} steps with {args.n_envs} envs")
    print(f"Opponent: {args.opponent}, Dynamics: {args.dynamics}")
    print(f"Output: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir} --bind_all")

    # Training envs
    vec_env = make_vec_env(
        lambda: make_env(opponent=args.opponent, use_dynamics=args.dynamics),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Eval env (single, for clean metrics)
    eval_env = make_vec_env(
        lambda: make_env(opponent=args.opponent, use_dynamics=args.dynamics),
        n_envs=1,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=str(log_dir))
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cpu",
            tensorboard_log=str(log_dir),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # encourage exploration early on
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )

    # Recordings dir (shared with web UI)
    recordings_dir = Path("recordings")

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(5_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    record_cb = RecordGameCallback(
        record_freq=args.record_freq,
        opponent=args.opponent,
        use_dynamics=args.dynamics,
        recordings_dir=recordings_dir,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, record_cb],
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    print(f"Training complete. Model saved to {run_dir / 'final_model'}")

    # Record a final game
    record_cb._record_game(args.timesteps)
    print(f"Recordings saved to {recordings_dir}/")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
