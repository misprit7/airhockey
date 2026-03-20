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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from airhockey.dynamics import DelayedDynamics, IdealDynamics
from airhockey.env import AirHockeyEnv
from airhockey.rewards import ShapedRewardWrapper


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
    parser.add_argument("--opponent", type=str, default="follow")
    parser.add_argument("--dynamics", action="store_true", help="Use delayed motor dynamics")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--run-name", type=str, default="ppo_airhockey")
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

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    print(f"Training complete. Model saved to {run_dir / 'final_model'}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
