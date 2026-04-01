"""Train an air hockey RL agent with SAC.

Supports curriculum learning with 3 stages:
    Stage 1 (proximity): Learn to chase the puck
    Stage 2 (contact):   Learn to hit the puck toward the goal
    Stage 3 (scoring):   Learn to score goals

Each stage resumes from the previous stage's final model.

Usage:
    # Run full curriculum (all 3 stages)
    python bin/train.py --curriculum

    # Run a single stage
    python bin/train.py --stage 1 --timesteps 500000

    # Resume from a checkpoint
    python bin/train.py --stage 3 --resume runs/stage2/final_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
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
from airhockey.rewards import STAGE_PROXIMITY, STAGE_SCORING, ShapedRewardWrapper

STAGE_NAMES = {
    STAGE_PROXIMITY: "chase+hit",
    STAGE_SCORING: "scoring",
}

STAGE_DEFAULTS = {
    STAGE_PROXIMITY: {"timesteps": 500_000},
    STAGE_SCORING: {"timesteps": 2_000_000},
}


class RecordGameCallback(BaseCallback):
    """Records a full game episode every `record_freq` timesteps."""

    def __init__(
        self,
        record_freq: int,
        stage: int,
        use_dynamics: bool,
        recordings_dir: Path,
        run_name: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.stage = stage
        self.use_dynamics = use_dynamics
        self.recordings_dir = recordings_dir
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._last_record_step = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step // self.record_freq > self._last_record_step // self.record_freq:
            self._last_record_step = step
            self._record_game(step)
        return True

    def _record_game(self, step: int) -> None:
        env = make_env(stage=self.stage, use_dynamics=self.use_dynamics, record=True)
        obs, _ = env.reset(seed=step)
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
            prefix = f"{self.run_name}_" if self.run_name else ""
            filename = f"{prefix}step_{step_str}.json"
            rec.save(self.recordings_dir / filename)
            score = f"{info['score_agent']}-{info['score_opponent']}"
            if self.verbose:
                print(f"Recorded game at step {step:,}: {score}")


def make_env(
    stage: int = STAGE_SCORING,
    use_dynamics: bool = True,
    record: bool = False,
) -> gym.Env:
    dynamics = DelayedDynamics(max_speed=3.0, max_accel=30.0) if use_dynamics else IdealDynamics()
    env = AirHockeyEnv(
        agent_dynamics=dynamics,
        opponent_policy="idle",
        record=record,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    env = ShapedRewardWrapper(env, stage=stage)
    return env


def train_stage(
    stage: int,
    timesteps: int,
    n_envs: int,
    run_name: str,
    resume_from: str | None,
    record_freq: int,
    use_dynamics: bool,
) -> str:
    """Train a single curriculum stage. Returns path to final model."""
    stage_name = STAGE_NAMES[stage]
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    recordings_dir = Path("recordings")

    print(f"\n{'='*60}")
    print(f"Stage {stage}: {stage_name.upper()}")
    print(f"{'='*60}")
    print(f"Timesteps: {timesteps:,} | Envs: {n_envs} | Dynamics: {use_dynamics}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"Output: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir runs --bind_all")
    print()

    vec_env = make_vec_env(
        lambda: make_env(stage=stage, use_dynamics=use_dynamics),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    eval_env = make_vec_env(
        lambda: make_env(stage=stage, use_dynamics=use_dynamics),
        n_envs=1,
    )

    if resume_from:
        model = SAC.load(resume_from, env=vec_env, tensorboard_log=str(log_dir))
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="auto",
            tensorboard_log=str(log_dir),
            learning_rate=5e-3,
            buffer_size=500_000,
            learning_starts=5_000,
            batch_size=4096,
            tau=0.005,
            gamma=0.99,
            train_freq=(128, "step"),
            gradient_steps=8,
            ent_coef="auto",
            policy_kwargs=dict(
                net_arch=[512, 512],
            ),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="sac",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(200_000 // n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
    )
    record_cb = RecordGameCallback(
        record_freq=record_freq,
        stage=stage,
        use_dynamics=use_dynamics,
        recordings_dir=recordings_dir,
        run_name=run_name,
        verbose=1,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb, record_cb],
        progress_bar=True,
    )

    final_path = str(run_dir / "final_model")
    model.save(final_path)
    print(f"Stage {stage} complete. Model saved to {final_path}")

    record_cb._record_game(timesteps)
    print(f"Recordings saved to {recordings_dir}/")

    vec_env.close()
    eval_env.close()

    return final_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Train air hockey agent")
    parser.add_argument("--curriculum", action="store_true", help="Run full 3-stage curriculum")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3], help="Run a single stage")
    parser.add_argument("--timesteps", type=int, default=None, help="Override default timesteps for stage")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--dynamics", action="store_true", default=True, help="Use delayed motor dynamics")
    parser.add_argument("--no-dynamics", dest="dynamics", action="store_false")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--record-freq", type=int, default=50_000)
    args = parser.parse_args()

    if args.curriculum:
        # Run all 3 stages sequentially
        model_path = args.resume
        for stage in [STAGE_PROXIMITY, STAGE_SCORING]:
            stage_name = STAGE_NAMES[stage]
            run_name = args.run_name or "curriculum"
            ts = args.timesteps or STAGE_DEFAULTS[stage]["timesteps"]

            model_path = train_stage(
                stage=stage,
                timesteps=ts,
                n_envs=args.n_envs,
                run_name=f"{run_name}_s{stage}_{stage_name}",
                resume_from=model_path,
                record_freq=args.record_freq,
                use_dynamics=args.dynamics,
            )
        print(f"\nCurriculum complete! Final model: {model_path}")

    elif args.stage is not None:
        stage = args.stage
        stage_name = STAGE_NAMES[stage]
        run_name = args.run_name or f"stage{stage}_{stage_name}"
        ts = args.timesteps or STAGE_DEFAULTS[stage]["timesteps"]

        train_stage(
            stage=stage,
            timesteps=ts,
            n_envs=args.n_envs,
            run_name=run_name,
            resume_from=args.resume,
            record_freq=args.record_freq,
            use_dynamics=args.dynamics,
        )
    else:
        parser.error("Specify --curriculum or --stage N")


if __name__ == "__main__":
    main()
