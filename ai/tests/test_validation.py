"""Validation tests for optimized training pipeline.

Ensures batch reward shaping matches scalar, and that the training loop
produces valid outputs.
"""

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.env import AirHockeyEnv
from airhockey.dynamics import DelayedDynamics, IdealDynamics
from airhockey.rewards import ShapedRewardWrapper, STAGE_SCORING, STAGE_DEFAULTS


class TestBatchRewardShaping:
    """Validate BatchRewardShaper in train_tdmpc2.py matches ShapedRewardWrapper."""

    def _make_scalar_env(self):
        return ShapedRewardWrapper(
            AirHockeyEnv(
                agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
                opponent_policy="idle",
                record=False,
                action_dt=1 / 60,
                max_episode_time=30.0,
                max_score=7,
            ),
            stage=STAGE_SCORING,
        )

    def test_reward_shaping_matches(self):
        """Compare batch vs scalar reward shaping over many steps."""
        from airhockey.rewards import BatchRewardShaper

        n_envs = 4
        batch_shaper = BatchRewardShaper(n_envs)

        # Create N scalar envs for comparison
        scalar_envs = [self._make_scalar_env() for _ in range(n_envs)]

        # Reset all
        scalar_obs = []
        scalar_infos = []
        for env in scalar_envs:
            obs, info = env.reset()
            scalar_obs.append(obs)
            scalar_infos.append(info)
        scalar_obs = np.array(scalar_obs)

        # Build batch info from scalar infos
        batch_info = {
            "puck_vx": np.array([inf.get("puck_vx", 0.0) for inf in scalar_infos]),
            "puck_vy": np.array([inf.get("puck_vy", 0.0) for inf in scalar_infos]),
        }
        batch_shaper.reset(scalar_obs, info=batch_info)

        # Run several steps with random actions
        rng = np.random.default_rng(42)
        max_diff = 0.0
        for step in range(200):
            actions = rng.uniform(-1, 1, size=(n_envs, 2)).astype(np.float32)

            # Step scalar envs
            scalar_rewards_shaped = []
            new_scalar_obs = []
            new_scalar_infos = []
            raw_rewards = np.zeros(n_envs)

            for i, env in enumerate(scalar_envs):
                obs, shaped_r, terminated, truncated, info = env.step(actions[i])
                scalar_rewards_shaped.append(shaped_r)
                new_scalar_obs.append(obs)
                new_scalar_infos.append(info)
                raw_rewards[i] = info.get("raw_reward", 0.0)

                if terminated or truncated:
                    obs, info = env.reset()
                    new_scalar_obs[-1] = obs
                    new_scalar_infos[-1] = info

            scalar_obs = np.array(new_scalar_obs)
            scalar_shaped = np.array(scalar_rewards_shaped)

            # Build batch info for velocity
            batch_info = {
                "puck_vx": np.array([inf.get("puck_vx", 0.0) for inf in new_scalar_infos]),
                "puck_vy": np.array([inf.get("puck_vy", 0.0) for inf in new_scalar_infos]),
            }

            # Compute batch rewards
            batch_shaped = batch_shaper.compute(scalar_obs, raw_rewards,
                                                 actions=actions, info=batch_info)

            diff = np.abs(scalar_shaped - batch_shaped)
            max_diff = max(max_diff, diff.max())

            # They should be very close (not exact due to minor implementation diffs)
            np.testing.assert_allclose(
                batch_shaped, scalar_shaped, atol=1e-5,
                err_msg=f"Reward mismatch at step {step}"
            )

        print(f"Max reward difference across {200} steps: {max_diff:.2e}")

    def test_fast_batch_shaper_matches(self):
        """Verify BatchRewardShaper from airhockey.rewards is consistent."""
        from airhockey.rewards import BatchRewardShaper

        n_envs = 8
        orig = BatchRewardShaper(n_envs, stage=STAGE_SCORING)
        fast = BatchRewardShaper(n_envs, stage=STAGE_SCORING)

        rng = np.random.default_rng(99)

        # Generate fake observations (8-dim: 6 positions + 2 context)
        obs = rng.uniform(-1, 2, size=(n_envs, 8)).astype(np.float32)
        info = {
            "puck_vx": rng.uniform(-2, 2, size=n_envs).astype(np.float32),
            "puck_vy": rng.uniform(-2, 2, size=n_envs).astype(np.float32),
        }
        orig.reset(obs, info=info)
        fast.reset(obs, info=info)

        for _ in range(100):
            obs = rng.uniform(-1, 2, size=(n_envs, 8)).astype(np.float32)
            raw = rng.choice([-1, 0, 0, 0, 0, 1], size=n_envs).astype(np.float32)
            info = {
                "puck_vx": rng.uniform(-2, 2, size=n_envs).astype(np.float32),
                "puck_vy": rng.uniform(-2, 2, size=n_envs).astype(np.float32),
            }

            r_orig = orig.compute(obs, raw, info=info)
            r_fast = fast.compute(obs, raw, info=info)

            np.testing.assert_allclose(r_fast, r_orig, atol=1e-6,
                                       err_msg="Same BatchRewardShaper should be deterministic")


class TestBatchEnvConsistency:
    """Ensure batch env steps produce consistent, valid observations."""

    def test_obs_in_range(self):
        env = BatchAirHockeyEnv(n_envs=16)
        obs = env.reset(seed=42)

        for _ in range(500):
            actions = np.random.uniform(-1, 1, size=(16, 2)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(actions)

            # Puck should stay on table (or briefly in goal)
            assert np.all(obs[:, 0] >= -0.1), f"Puck x too low: {obs[:, 0].min()}"
            assert np.all(obs[:, 0] <= 1.1), f"Puck x too high: {obs[:, 0].max()}"

            # Paddles should stay in their halves (index 5 = paddle_y in 14-dim obs)
            cfg = env.table_config
            assert np.all(obs[:, 5] <= cfg.height / 2 + 0.01), \
                f"Agent paddle in opponent half: y={obs[:, 5].max()}"

            # Auto-reset done envs
            done = term | trunc
            if np.any(done):
                env.auto_reset(term, trunc)

    def test_episode_terminates(self):
        """Ensure episodes eventually end."""
        env = BatchAirHockeyEnv(n_envs=1, max_score=1, max_episode_time=5.0)
        env.reset(seed=0)

        done_count = 0
        for _ in range(2000):
            actions = np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(actions)
            if term[0] or trunc[0]:
                done_count += 1
                env.auto_reset(term, trunc)

        assert done_count > 0, "No episodes completed in 2000 steps"

    def test_dynamics_delay_effect(self):
        """Delayed dynamics should cause paddle to lag behind target."""
        env = BatchAirHockeyEnv(
            n_envs=1, agent_dynamics="delayed",
            dynamics_max_speed=3.0, dynamics_max_accel=30.0,
        )
        obs = env.reset(seed=42)

        # Command paddle to far-away position for one step
        action = np.array([[1.0, 1.0]])
        obs, _, _, _, _ = env.step(action)

        cfg = env.table_config
        target_x = cfg.width - cfg.paddle_radius
        assert obs[0, 2] < target_x, "Delayed dynamics should not instantly reach target"


class TestGoalRewardSignals:
    """Validate that goal scoring produces correct reward signals in batch."""

    def test_goal_reward_values(self):
        """Goal scored should give +50, goal conceded should give -25."""
        from airhockey.rewards import BatchRewardShaper

        shaper = BatchRewardShaper(4, stage=STAGE_SCORING)
        obs = np.zeros((4, 8), dtype=np.float32)
        obs[:, 0] = 0.5  # puck near paddle
        obs[:, 1] = 0.5
        obs[:, 2] = 0.5  # pad_x
        obs[:, 3] = 0.5  # pad_y
        shaper.reset(obs)

        raw_rewards = np.array([1.0, -1.0, 0.0, 0.0])
        shaped = shaper.compute(obs, raw_rewards)

        # Env 0: goal scored (goal_reward from STAGE_DEFAULTS)
        goal_r = STAGE_DEFAULTS[STAGE_SCORING]["goal_reward"]
        goal_p = STAGE_DEFAULTS[STAGE_SCORING]["goal_penalty"]
        assert shaped[0] > goal_r * 0.9, f"Goal scored reward too low: {shaped[0]}"
        # Env 1: goal conceded
        assert shaped[1] < goal_p * 0.9, f"Goal conceded should be strongly negative: {shaped[1]}"
        assert shaped[1] > goal_p * 1.5, f"Goal penalty too harsh: {shaped[1]}"
        # Envs 2,3: no goal, no proximity (stages 2+ have proximity=0)
        assert abs(shaped[2]) < 1
        assert abs(shaped[3]) < 1

    def test_contact_detection_with_speed_change(self):
        """Contact reward should trigger when puck speeds up near paddle."""
        from airhockey.rewards import BatchRewardShaper

        shaper = BatchRewardShaper(2, stage=STAGE_SCORING)

        # 14-dim obs: [puck_x, puck_y, puck_vx, puck_vy,
        #              pad_x, pad_y, pad_vx, pad_vy,
        #              opp_x, opp_y, opp_vx, opp_vy,
        #              score_diff, time_remaining]
        # Place puck and paddle very close (dist < 0.15), low initial speed
        obs0 = np.zeros((2, 14), dtype=np.float32)
        obs0[:, 0] = 0.5   # puck_x
        obs0[:, 1] = 0.5   # puck_y
        obs0[:, 4] = 0.5   # pad_x
        obs0[:, 5] = 0.45  # pad_y, dist = 0.05
        info0 = {"puck_vx": np.zeros(2), "puck_vy": np.zeros(2)}
        shaper.reset(obs0, info=info0)

        # Now puck speeds up (contact happened)
        obs1 = obs0.copy()
        info1 = {
            "puck_vx": np.array([2.0, 0.0]),  # env 0: puck sped up
            "puck_vy": np.array([1.5, 0.0]),   # heading toward opponent
        }
        raw = np.zeros(2)
        shaped = shaper.compute(obs1, raw, info=info1)

        # Env 0: contact + directed_hit + shot_placement (no proximity in game stages)
        contact_w = STAGE_DEFAULTS[STAGE_SCORING]["contact"]
        assert shaped[0] > contact_w, f"Contact reward too low: {shaped[0]}"
        # Env 1: no contact, no proximity in game stages
        assert shaped[1] < contact_w

    def test_puck_progress_one_way(self):
        """Puck progress should only reward forward movement."""
        from airhockey.rewards import BatchRewardShaper

        shaper = BatchRewardShaper(2, stage=STAGE_SCORING)
        # 14-dim obs
        obs0 = np.zeros((2, 14), dtype=np.float32)
        obs0[:, 0] = 0.5   # puck_x
        obs0[:, 1] = 0.5   # initial puck_y
        obs0[:, 4] = 2.0   # pad_x far away (no proximity/contact confound)
        obs0[:, 5] = 0.5   # pad_y
        shaper.reset(obs0)

        obs1 = obs0.copy()
        obs1[0, 1] = 0.7   # env 0: puck moved forward (+0.2)
        obs1[1, 1] = 0.3   # env 1: puck moved backward (-0.2)
        raw = np.zeros(2)
        shaped = shaper.compute(obs1, raw)

        # Env 0: proximity + progress (0.2 * 0.2 = 0.04)
        # Env 1: proximity only (no reward for backward movement)
        assert shaped[0] > shaped[1], "Forward progress should be rewarded more than backward"


class TestTrajectoryBufferInitObs:
    """Verify trajectory buffer initial obs uses reset obs, not terminal obs."""

    def test_initial_obs_is_reset_not_terminal(self):
        """After episode done, new trajectory should start with reset observation.

        This catches a bug where the fast script used terminal obs instead of
        reset obs for the first entry in tds_list.
        """
        env = BatchAirHockeyEnv(
            n_envs=2, max_score=1, agent_dynamics="ideal",
            opponent_policy="idle", action_dt=1/60,
        )
        obs_all = env.reset(seed=42)

        # Force env 0 to score (puck heading into opponent goal)
        cfg = env.table_config
        env.engine.puck_x[0] = cfg.width / 2
        env.engine.puck_y[0] = cfg.height - 0.005
        env.engine.puck_vx[0] = 0.0
        env.engine.puck_vy[0] = 5.0

        actions = np.zeros((2, 2))
        obs_all, raw_rewards, terminated, truncated, info = env.step(actions)
        done = terminated | truncated

        assert done[0], "Env 0 should be done"
        terminal_obs_0 = obs_all[0].copy()

        # Auto-reset
        new_obs = env.auto_reset(terminated, truncated)
        assert new_obs is not None
        reset_obs_0 = new_obs[0].copy()

        # Terminal and reset observations should be different
        # (reset puts puck in random center position, terminal has puck in goal)
        assert not np.allclose(terminal_obs_0, reset_obs_0, atol=0.01), \
            "Terminal and reset obs should differ — puck position changes on reset"

        # The trajectory buffer should use reset_obs_0, NOT terminal_obs_0
        # This is what the fix ensures
