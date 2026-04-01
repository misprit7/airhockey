"""Tests for BatchPhysicsEngine and BatchAirHockeyEnv.

Validates vectorized physics matches single-env physics for identical inputs.
"""

import numpy as np
import pytest

from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.physics import PhysicsEngine, TableConfig


class TestBatchPhysicsEngine:
    """Verify batch engine matches single engine for deterministic inputs."""

    def test_reset_shapes(self):
        engine = BatchPhysicsEngine(8)
        engine.reset()
        assert engine.puck_x.shape == (8,)
        assert engine.puck_vy.shape == (8,)
        assert engine.score_agent.shape == (8,)

    def test_reset_masked(self):
        engine = BatchPhysicsEngine(4)
        rng = np.random.default_rng(42)
        engine.reset(rng)

        # Save env 0 and 2 state, reset only 1 and 3
        px0 = engine.puck_x[0]
        px2 = engine.puck_x[2]
        mask = np.array([False, True, False, True])
        engine.reset(rng, mask=mask)

        assert engine.puck_x[0] == px0
        assert engine.puck_x[2] == px2

    def test_step_matches_single(self):
        """Run both engines with identical state and compare results."""
        cfg = TableConfig()
        single = PhysicsEngine(cfg)
        batch = BatchPhysicsEngine(1, cfg)

        rng = np.random.default_rng(123)
        single_state = single.reset(rng)

        # Copy single state into batch engine
        batch.puck_x[0] = single_state.puck.x
        batch.puck_y[0] = single_state.puck.y
        batch.puck_vx[0] = single_state.puck.vx
        batch.puck_vy[0] = single_state.puck.vy
        batch.paddle_agent_x[0] = single_state.paddle_agent.x
        batch.paddle_agent_y[0] = single_state.paddle_agent.y
        batch.paddle_opp_x[0] = single_state.paddle_opponent.x
        batch.paddle_opp_y[0] = single_state.paddle_opponent.y

        dt = 1 / 240

        # Step both engines with no paddle movement
        for _ in range(100):
            single.step(dt)
            batch.step(dt)

            np.testing.assert_allclose(batch.puck_x[0], single.state.puck.x, atol=1e-10)
            np.testing.assert_allclose(batch.puck_y[0], single.state.puck.y, atol=1e-10)
            np.testing.assert_allclose(batch.puck_vx[0], single.state.puck.vx, atol=1e-10)
            np.testing.assert_allclose(batch.puck_vy[0], single.state.puck.vy, atol=1e-10)

    def test_wall_collisions(self):
        """Puck heading into left wall should bounce."""
        batch = BatchPhysicsEngine(2)
        batch.puck_x[:] = [0.03, 0.5]  # env 0 near left wall
        batch.puck_y[:] = [1.0, 1.0]
        batch.puck_vx[:] = [-2.0, 0.0]
        batch.puck_vy[:] = [0.0, 0.0]

        batch.step(1 / 240)

        # Env 0 should have bounced (positive vx)
        assert batch.puck_vx[0] > 0
        # Env 1 should be unaffected
        assert batch.puck_vx[1] == 0.0

    def test_paddle_collision(self):
        """Puck hitting paddle should bounce off."""
        batch = BatchPhysicsEngine(1)
        cfg = batch.config
        # Place paddle and puck close together
        batch.paddle_agent_x[0] = 0.5
        batch.paddle_agent_y[0] = 0.3
        batch.puck_x[0] = 0.5
        batch.puck_y[0] = 0.3 + cfg.puck_radius + cfg.paddle_radius - 0.001
        batch.puck_vx[0] = 0.0
        batch.puck_vy[0] = -2.0  # heading toward paddle

        batch.step(1 / 240)

        # Puck should have bounced (positive vy or at least changed)
        assert batch.puck_vy[0] > 0

    def test_goal_scoring(self):
        """Puck going past baseline in goal area should score."""
        batch = BatchPhysicsEngine(2)
        cfg = batch.config

        # Env 0: puck heading into agent goal (bottom)
        batch.puck_x[0] = cfg.width / 2  # centered in goal
        batch.puck_y[0] = 0.01
        batch.puck_vx[0] = 0.0
        batch.puck_vy[0] = -3.0

        # Env 1: puck heading into opponent goal (top)
        batch.puck_x[1] = cfg.width / 2
        batch.puck_y[1] = cfg.height - 0.01
        batch.puck_vx[1] = 0.0
        batch.puck_vy[1] = 3.0

        batch.step(1 / 60)

        assert batch.goal_scored[0] == -1  # opponent scored
        assert batch.goal_scored[1] == 1  # agent scored
        assert batch.score_opponent[0] == 1
        assert batch.score_agent[1] == 1


class TestBatchAirHockeyEnv:
    """Test the batch environment wrapper."""

    def test_reset_and_step_shapes(self):
        env = BatchAirHockeyEnv(n_envs=4)
        obs = env.reset(seed=42)
        obs_dim = obs.shape[1]
        assert obs.shape == (4, obs_dim)

        actions = np.zeros((4, 2))
        obs, rew, term, trunc, info = env.step(actions)
        assert obs.shape == (4, obs_dim)
        assert rew.shape == (4,)
        assert term.shape == (4,)
        assert trunc.shape == (4,)

    def test_mirror_obs_roundtrip(self):
        """Mirroring twice should return to original."""
        env = BatchAirHockeyEnv(n_envs=2)
        obs = env.reset(seed=0)
        mirrored = env.mirror_obs(obs)
        back = env.mirror_obs(mirrored)
        np.testing.assert_allclose(obs, back, atol=1e-6)

    def test_auto_reset(self):
        """auto_reset should only reset done envs."""
        env = BatchAirHockeyEnv(n_envs=4, max_score=1)
        env.reset(seed=99)

        # Manually trigger a goal in env 0
        env.engine.score_agent[0] = 1
        terminated = env.engine.score_agent >= env.max_score
        truncated = np.zeros(4, dtype=bool)

        old_puck_x_1 = env.engine.puck_x[1]
        new_obs = env.auto_reset(terminated, truncated)

        assert new_obs is not None
        # Env 0 was reset (score cleared)
        assert env.engine.score_agent[0] == 0
        # Env 1 untouched
        assert env.engine.puck_x[1] == old_puck_x_1

    def test_actions_rescale(self):
        """Actions at -1 and 1 should map to paddle bounds."""
        env = BatchAirHockeyEnv(n_envs=2, agent_dynamics="ideal")
        env.reset(seed=7)

        # Action [-1, -1] = bottom-left, [1, 1] = top-right of agent half
        actions = np.array([[-1.0, -1.0], [1.0, 1.0]])
        env.step(actions)

        cfg = env.table_config
        r = cfg.paddle_radius
        np.testing.assert_allclose(env.engine.paddle_agent_x[0], r, atol=0.01)
        np.testing.assert_allclose(env.engine.paddle_agent_y[0], r, atol=0.01)
        np.testing.assert_allclose(
            env.engine.paddle_agent_x[1], cfg.width - r, atol=0.01
        )
        np.testing.assert_allclose(
            env.engine.paddle_agent_y[1], cfg.height / 2 - r, atol=0.01
        )
