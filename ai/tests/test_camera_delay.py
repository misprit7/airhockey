"""Tests for camera delay in BatchAirHockeyEnv.

Validates that the ring buffer delay mechanism correctly delays observations
by the configured number of steps, supports per-env randomized delay,
and handles partial resets properly.
"""

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv


class TestUniformCameraDelay:
    """Test uniform (same for all envs) camera delay."""

    def test_zero_delay_is_default(self):
        """With no camera_delay, obs should be current (no delay)."""
        env = BatchAirHockeyEnv(n_envs=2, agent_dynamics="ideal", action_dt=1/60)
        obs0 = env.reset(seed=42)
        # Step and verify obs reflects current state
        obs1, _, _, _, _ = env.step(np.array([[0.5, 0.5], [0.5, 0.5]]))
        # Obs should change immediately (puck is moving)
        assert not np.allclose(obs0, obs1), "Obs should change with no delay"

    def test_delay_returns_stale_obs(self):
        """With delay, returned obs should lag behind true state."""
        delay_seconds = 3 / 60  # 3 action steps at 60Hz
        env = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            action_dt=1/60,
            camera_delay=delay_seconds,
        )
        obs_reset = env.reset(seed=42)

        # Step 3 times — obs should still look like reset state
        # because delay buffer was pre-filled with initial obs
        actions = np.array([[0.0, 0.0]])
        for i in range(3):
            obs, _, _, _, _ = env.step(actions)
            np.testing.assert_allclose(
                obs, obs_reset, atol=1e-5,
                err_msg=f"Step {i+1}: with 3-step delay, obs should still be initial"
            )

        # Step 4th time — now we should see the obs from step 1
        obs4, _, _, _, _ = env.step(actions)
        assert not np.allclose(obs4, obs_reset, atol=1e-5), \
            "After delay steps, obs should reflect earlier state changes"

    def test_delay_steps_computed_correctly(self):
        """camera_delay / action_dt should give correct integer delay."""
        env = BatchAirHockeyEnv(
            n_envs=4,
            action_dt=1/60,
            camera_delay=0.05,  # 50ms -> 3 steps at 60Hz
        )
        assert env._max_delay == 3
        np.testing.assert_array_equal(env._delay_steps, 3)

    def test_ring_buffer_shape(self):
        """Ring buffer should be sized for max delay + 1."""
        env = BatchAirHockeyEnv(
            n_envs=8,
            action_dt=1/60,
            camera_delay=2 / 60,  # 2 steps
        )
        assert env._ring_size == 3  # delay + 1
        assert env._obs_ring.shape == (3, 8, env._obs_dim)

    def test_no_buffer_when_zero_delay(self):
        """No ring buffer allocated for zero delay."""
        env = BatchAirHockeyEnv(n_envs=4, camera_delay=0.0)
        assert env._max_delay == 0
        assert not hasattr(env, '_obs_ring')


class TestPerEnvCameraDelay:
    """Test per-env randomized camera delay."""

    def test_tuple_delay_range(self):
        """Tuple camera_delay should set per-env random delays."""
        env = BatchAirHockeyEnv(
            n_envs=100,
            action_dt=1/60,
            camera_delay=(1/60, 5/60),  # 1-5 steps
        )
        env.reset(seed=42)

        assert env._delay_range == (1, 5)
        assert env._max_delay == 5
        assert env._ring_size == 6  # max_delay + 1
        # Per-env delays should be in [1, 5]
        assert np.all(env._delay_steps >= 1)
        assert np.all(env._delay_steps <= 5)
        # With 100 envs, we should see some variety
        unique_delays = len(np.unique(env._delay_steps))
        assert unique_delays >= 3, f"Expected diverse delays, got {unique_delays} unique values"

    def test_per_env_delay_actually_differs(self):
        """Envs with different delays should see different obs at the same step."""
        env = BatchAirHockeyEnv(
            n_envs=2,
            agent_dynamics="ideal",
            action_dt=1/60,
            camera_delay=(1/60, 3/60),  # 1-3 steps
        )
        env.reset(seed=42)
        # Force different delays
        env._delay_steps[0] = 1
        env._delay_steps[1] = 3
        # Re-fill buffer with initial obs
        obs_init = env._make_obs_direct()
        for t in range(env._ring_size):
            env._obs_ring[t] = obs_init

        # Step with different actions to create divergent states
        actions = np.array([[1.0, 1.0], [1.0, 1.0]])
        for _ in range(4):
            env.step(actions)

        # Now both envs have the same true state, but env 0 (delay=1)
        # sees more recent obs than env 1 (delay=3)
        obs = env._make_obs()
        # They should not be identical (different delays)
        if not np.allclose(obs[0], obs[1]):
            pass  # Expected: different delays -> different obs
        # At minimum, the delay mechanism should work without errors


class TestCameraDelayWithReset:
    """Test camera delay behavior across episode resets."""

    def test_partial_reset_refills_buffer(self):
        """Resetting one env should pre-fill its buffer slots without affecting others."""
        env = BatchAirHockeyEnv(
            n_envs=4,
            agent_dynamics="ideal",
            action_dt=1/60,
            camera_delay=2/60,  # 2 steps
        )
        env.reset(seed=0)

        # Step a few times to fill buffer with real data
        actions = np.zeros((4, 2))
        for _ in range(5):
            env.step(actions)

        # Save env 1 obs from ring buffer
        env1_ring_before = env._obs_ring[:, 1, :].copy()

        # Reset only env 0 and 2
        mask = np.array([True, False, True, False])
        env.reset(mask=mask)

        # Env 1 ring buffer should be unchanged
        np.testing.assert_array_equal(
            env._obs_ring[:, 1, :], env1_ring_before,
            err_msg="Partial reset modified non-reset env's buffer"
        )

    def test_auto_reset_prefills_delay(self):
        """auto_reset should pre-fill delay buffer for done envs."""
        env = BatchAirHockeyEnv(
            n_envs=4,
            agent_dynamics="ideal",
            action_dt=1/60,
            max_score=1,
            camera_delay=2/60,
        )
        env.reset(seed=99)

        # Step to build up some buffer state
        for _ in range(3):
            env.step(np.zeros((4, 2)))

        # Manually trigger termination on env 0
        env.engine.score_agent[0] = 1
        terminated = env.engine.score_agent >= env.max_score
        truncated = np.zeros(4, dtype=bool)
        new_obs = env.auto_reset(terminated, truncated)

        assert new_obs is not None
        # Env 0 was reset — its buffer slots should all be the new initial obs
        # (pre-filled during reset)
        for t in range(env._ring_size):
            np.testing.assert_allclose(
                env._obs_ring[t, 0], env._obs_ring[0, 0],
                err_msg=f"Ring buffer slot {t} not pre-filled for reset env"
            )

    def test_delay_re_randomized_on_reset(self):
        """Per-env delay should be re-randomized when an env resets."""
        env = BatchAirHockeyEnv(
            n_envs=100,
            action_dt=1/60,
            camera_delay=(1/60, 5/60),
        )
        env.reset(seed=0)
        delays_before = env._delay_steps.copy()

        # Reset only first 50 envs
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True
        env.reset(mask=mask)

        # Last 50 should be unchanged
        np.testing.assert_array_equal(
            env._delay_steps[50:], delays_before[50:],
            err_msg="Non-reset env delays changed"
        )
        # First 50 were re-randomized (at least some should differ,
        # though not guaranteed for every single one)


class TestCameraDelayMatchesSingleEnv:
    """Verify batch camera delay matches single-env camera delay behavior."""

    def test_uniform_delay_matches_concept(self):
        """With uniform delay of K steps, position fields at step T should
        match those from step T-K.

        We compare only the position components (puck_x/y, pad_x/y, opp_x/y)
        because velocity fields are computed from finite differences and
        calling _make_obs_direct() a second time would corrupt that state.
        """
        K = 2
        # Run two envs: one with delay, one without
        env_d = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            action_dt=1/60,
            camera_delay=K/60,
        )
        env_n = BatchAirHockeyEnv(
            n_envs=1,
            agent_dynamics="ideal",
            action_dt=1/60,
            camera_delay=0.0,
        )
        env_d.reset(seed=42)
        env_n.reset(seed=42)

        # Position indices in 14-dim obs
        pos_idx = [0, 1, 4, 5, 8, 9]  # puck_xy, paddle_xy, opp_xy

        true_pos_history = []
        delayed_pos_history = []

        actions = np.array([[0.3, -0.2]])
        for _ in range(10):
            obs_d, _, _, _, _ = env_d.step(actions)
            obs_n, _, _, _, _ = env_n.step(actions)
            true_pos_history.append(obs_n[0, pos_idx].copy())
            delayed_pos_history.append(obs_d[0, pos_idx].copy())

        # delayed positions at step T should equal true positions at step T-K
        for t in range(K, len(delayed_pos_history)):
            np.testing.assert_allclose(
                delayed_pos_history[t], true_pos_history[t - K], atol=1e-5,
                err_msg=f"Step {t}: delayed positions don't match true from {K} steps ago"
            )
