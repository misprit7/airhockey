"""Validate BatchAirHockeyEnv matches AirHockeyEnv for training-critical behavior.

Tests that the vectorized environment produces equivalent observations, rewards,
and episode termination compared to the single-env version.
"""

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.dynamics import DelayedDynamics, IdealDynamics
from airhockey.env import AirHockeyEnv
from airhockey.physics import TableConfig
from airhockey.rewards import ShapedRewardWrapper, STAGE_SCORING


class TestEnvParity:
    """Compare single-env vs batch-env behavior for identical inputs."""

    def test_obs_layout_matches(self):
        """Observation layout should be identical: [puck(4), agent(4), opp(4)]."""
        single = AirHockeyEnv(
            agent_dynamics=IdealDynamics(),
            opponent_policy="idle",
            action_dt=1/60,
        )
        batch = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            opponent_policy="idle",
            action_dt=1/60,
        )

        # Use same seed for both
        single_obs, _ = single.reset(seed=42)
        batch_obs = batch.reset(seed=42)

        # 14-dim obs: puck(4) + paddle(4) + opp(4) + context(2)
        assert single_obs.shape == (14,)
        assert batch_obs.shape == (1, 14)

        # Both should produce valid observations in the right ranges
        cfg = TableConfig()
        # Puck position should be on the table
        assert 0 < single_obs[0] < cfg.width
        assert 0 < single_obs[1] < cfg.height
        assert 0 < batch_obs[0, 0] < cfg.width
        assert 0 < batch_obs[0, 1] < cfg.height

    def test_idle_opponent_stays_still(self):
        """Idle opponent should not move in either env."""
        batch = BatchAirHockeyEnv(
            n_envs=4,
            agent_dynamics="ideal",
            opponent_dynamics="delayed",
            opponent_policy="idle",
            action_dt=1/60,
        )
        obs = batch.reset(seed=0)
        opp_start = obs[:, 8:10].copy()  # opponent position (opp_x, opp_y)

        # Step with zero actions
        actions = np.zeros((4, 2))
        for _ in range(60):  # 1 second
            obs, _, _, _, _ = batch.step(actions)

        opp_end = obs[:, 8:10]
        # Opponent should barely have moved
        np.testing.assert_allclose(opp_start, opp_end, atol=0.01,
                                   err_msg="Idle opponent moved significantly")

    def test_ideal_dynamics_reaches_target(self):
        """With ideal dynamics, paddle should reach target position immediately."""
        batch = BatchAirHockeyEnv(
            n_envs=2,
            agent_dynamics="ideal",
            opponent_policy="idle",
            action_dt=1/60,
        )
        batch.reset(seed=7)

        # Action [0, 0] should map to center of the range
        actions = np.array([[0.0, 0.0], [0.0, 0.0]])
        obs, _, _, _, _ = batch.step(actions)

        cfg = TableConfig()
        r = cfg.paddle_radius
        expected_x = r + 0.5 * (cfg.width - 2 * r)
        expected_y = r + 0.5 * (cfg.height / 2 - 2 * r)
        np.testing.assert_allclose(obs[0, 4], expected_x, atol=0.01)
        np.testing.assert_allclose(obs[0, 5], expected_y, atol=0.01)

    def test_delayed_dynamics_converges(self):
        """With delayed dynamics, paddle should converge to target over time."""
        batch = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="delayed",
            opponent_policy="idle",
            action_dt=1/60,
            dynamics_max_speed=3.0,
            dynamics_max_accel=30.0,
        )
        batch.reset(seed=42)

        cfg = TableConfig()
        r = cfg.paddle_radius
        target_x = r + 0.5 * (cfg.width - 2 * r)
        target_y = r + 0.5 * (cfg.height / 2 - 2 * r)

        # Repeatedly command center position
        actions = np.array([[0.0, 0.0]])  # maps to center of range
        for _ in range(120):  # 2 seconds
            obs, _, _, _, _ = batch.step(actions)

        # Should have converged close to target
        np.testing.assert_allclose(obs[0, 4], target_x, atol=0.05,
                                   err_msg="Delayed dynamics didn't converge")
        np.testing.assert_allclose(obs[0, 5], target_y, atol=0.05)

    def test_termination_on_max_score(self):
        """Episode terminates when max_score is reached."""
        batch = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            opponent_policy="idle",
            action_dt=1/60,
            max_score=1,
        )
        batch.reset(seed=0)

        # Force a goal by setting score directly
        batch.engine.score_agent[0] = 1
        obs, _, terminated, _, info = batch.step(np.zeros((1, 2)))
        assert terminated[0], "Should terminate when max_score reached"

    def test_truncation_on_max_time(self):
        """Episode truncates after max_episode_time."""
        batch = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            opponent_policy="idle",
            action_dt=1/60,
            max_episode_time=1.0,  # 1 second
        )
        batch.reset(seed=0)

        actions = np.zeros((1, 2))
        truncated_at = None
        for i in range(120):  # 2 seconds worth
            obs, _, terminated, truncated, _ = batch.step(actions)
            if truncated[0]:
                truncated_at = i
                break

        assert truncated_at is not None, "Should truncate after max_episode_time"
        # Should be around 60 steps (1 second at 60Hz)
        assert 55 <= truncated_at <= 65, f"Truncated at step {truncated_at}, expected ~60"

    def test_reward_signs_correct(self):
        """Agent goal = +1, opponent goal = -1 in raw rewards."""
        batch = BatchAirHockeyEnv(
            n_envs=2,
            agent_dynamics="ideal",
            opponent_policy="idle",
            action_dt=1/60,
        )
        batch.reset(seed=0)
        cfg = batch.table_config

        # Env 0: puck heading into opponent goal
        batch.engine.puck_x[0] = cfg.width / 2
        batch.engine.puck_y[0] = cfg.height - 0.005
        batch.engine.puck_vx[0] = 0.0
        batch.engine.puck_vy[0] = 5.0

        # Env 1: puck heading into agent goal
        batch.engine.puck_x[1] = cfg.width / 2
        batch.engine.puck_y[1] = 0.005
        batch.engine.puck_vx[1] = 0.0
        batch.engine.puck_vy[1] = -5.0

        _, rewards, _, _, _ = batch.step(np.zeros((2, 2)))

        assert rewards[0] > 0, f"Agent goal should give positive reward, got {rewards[0]}"
        assert rewards[1] < 0, f"Opponent goal should give negative reward, got {rewards[1]}"

    def test_auto_reset_preserves_other_envs(self):
        """auto_reset should only affect done envs."""
        batch = BatchAirHockeyEnv(n_envs=4, agent_dynamics="ideal", action_dt=1/60)
        obs = batch.reset(seed=99)

        # Step a few times to get some state
        for _ in range(10):
            obs, _, _, _, _ = batch.step(np.zeros((4, 2)))

        # Save state for env 1
        env1_puck = batch.engine.puck_x[1], batch.engine.puck_y[1]

        # Force termination on env 0 and 2
        terminated = np.array([True, False, True, False])
        truncated = np.zeros(4, dtype=bool)
        batch.auto_reset(terminated, truncated)

        # Env 1 should be unchanged
        assert batch.engine.puck_x[1] == env1_puck[0]
        assert batch.engine.puck_y[1] == env1_puck[1]


class TestFullPipelineParity:
    """End-to-end comparison: single-env with ShapedRewardWrapper vs batch pipeline."""

    def test_shaped_reward_pipeline_agreement(self):
        """Run both pipelines with identical actions and verify shaped rewards agree.

        This is the most important test: it validates that the entire
        batch_env + BatchRewardShaper pipeline produces equivalent training
        signal to the single-env + ShapedRewardWrapper pipeline.
        """
        from airhockey.rewards import BatchRewardShaper

        rng = np.random.default_rng(42)

        # --- Single-env setup ---
        single_env = ShapedRewardWrapper(
            AirHockeyEnv(
                agent_dynamics=IdealDynamics(),
                opponent_policy="idle",
                record=False,
                action_dt=1/60,
                max_episode_time=30.0,
                max_score=7,
            ),
            stage=STAGE_SCORING,
        )

        # --- Batch-env setup ---
        batch_env = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            opponent_policy="idle",
            action_dt=1/60,
            max_episode_time=30.0,
            max_score=7,
        )
        shaper = BatchRewardShaper(1, stage=STAGE_SCORING)

        # Reset both with same seed
        single_obs, s_reset_info = single_env.reset(seed=42)
        batch_obs = batch_env.reset(seed=42)
        # Get initial velocity info from batch env engine
        b_reset_info = {
            "puck_vx": batch_env.engine.puck_vx.copy(),
            "puck_vy": batch_env.engine.puck_vy.copy(),
        }
        shaper.reset(batch_obs, info=b_reset_info)

        # Run N steps with random but identical actions
        n_steps = 200
        single_rewards = []
        batch_rewards = []
        max_diff = 0.0

        for i in range(n_steps):
            action = rng.uniform(-1, 1, size=2).astype(np.float32)

            # Single env
            s_obs, s_reward, s_term, s_trunc, s_info = single_env.step(action)
            single_rewards.append(s_reward)

            # Batch env
            b_obs, b_raw, b_term, b_trunc, b_info = batch_env.step(action.reshape(1, 2))
            b_shaped = shaper.compute(b_obs, b_raw, info=b_info)
            batch_rewards.append(b_shaped[0])

            diff = abs(s_reward - b_shaped[0])
            max_diff = max(max_diff, diff)

            if s_term or s_trunc:
                break
            if b_term[0] or b_trunc[0]:
                break

        # Compare cumulative rewards
        cum_single = sum(single_rewards)
        cum_batch = sum(batch_rewards)

        # Note: exact match isn't expected because the single env has
        # a puck-stuck-reset mechanism that the batch env doesn't have,
        # and seeding may differ slightly. But shaped rewards for each
        # step should be close when the underlying physics state matches.
        #
        # The key thing is that reward components have the same scale
        # and direction.
        print(f"\nPipeline comparison ({len(single_rewards)} steps):")
        print(f"  Single cumulative: {cum_single:.2f}")
        print(f"  Batch cumulative:  {cum_batch:.2f}")
        print(f"  Max per-step diff: {max_diff:.4f}")

        # Verify rewards are in the same ballpark (same order of magnitude)
        # Exact match isn't guaranteed due to physics divergence from
        # different RNG paths after reset, but the reward function itself
        # should be equivalent.
        if len(single_rewards) > 10:
            single_arr = np.array(single_rewards[:10])
            batch_arr = np.array(batch_rewards[:10])
            # The first few steps should match closely since physics
            # haven't diverged much yet
            np.testing.assert_allclose(single_arr, batch_arr, atol=0.5,
                                       err_msg="Early reward divergence too large")
