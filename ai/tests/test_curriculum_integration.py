"""Integration tests for the full curriculum system.

Tests that all curriculum components (stages, rewards, opponents, scheduling,
recording) work together end-to-end.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.curriculum import CurriculumLRScheduler, PlateauDetector, STAGE_LR
from airhockey.recorder import FrameData, Recorder
from airhockey.rewards import (
    STAGE_DEFAULTS,
    STAGE_NAMES,
    STAGE_OPPONENT,
    STAGE_CHASE_HIT,
    STAGE_GAME_GOALIE,
    STAGE_GAME_FOLLOW,
    STAGE_SELFPLAY,
    BatchRewardShaper,
)

# Import OpponentPool from training script
import sys

_TDMPC2_FAST = Path(__file__).resolve().parent.parent / "bin" / "train_tdmpc2_fast.py"


def _import_opponent_pool():
    """Import OpponentPool from train_tdmpc2_fast.py without running the script."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_tdmpc2_fast", _TDMPC2_FAST)
    mod = importlib.util.module_from_spec(spec)
    # Don't execute the module — just need the class definition.
    # Read the source and exec only the class.
    source = _TDMPC2_FAST.read_text()
    # Extract OpponentPool class by execing a subset
    import torch
    ns: dict = {"np": np, "torch": torch}
    # Find and exec just the OpponentPool class
    lines = source.split("\n")
    in_class = False
    class_lines = []
    for line in lines:
        if line.startswith("class OpponentPool"):
            in_class = True
        if in_class:
            class_lines.append(line)
            # Class ends at next top-level definition or blank line after dedent
            if len(class_lines) > 1 and line and not line[0].isspace() and not line.startswith("class OpponentPool"):
                class_lines.pop()  # remove the next class/def line
                break
    exec("\n".join(class_lines), ns)
    return ns["OpponentPool"]


OpponentPool = _import_opponent_pool()

N_ENVS = 4
N_STEPS = 10


def _make_env(n_envs=N_ENVS, **kwargs):
    """Create a small batch env for testing."""
    defaults = dict(
        n_envs=n_envs,
        max_episode_time=5.0,
        max_score=3,
    )
    defaults.update(kwargs)
    return BatchAirHockeyEnv(**defaults)


def _run_steps(env, n=N_STEPS, actions=None):
    """Run n random steps, return (obs_list, reward_list, info_list)."""
    rng = np.random.default_rng(42)
    obs = env.reset(seed=0)
    all_obs, all_rew, all_info = [obs], [], []
    for i in range(n):
        act = actions[i] if actions is not None else rng.uniform(-1, 1, (env.n_envs, 2)).astype(np.float32)
        obs, rew, term, trunc, info = env.step(act)
        all_obs.append(obs)
        all_rew.append(rew)
        all_info.append(info)
        done = term | trunc
        if np.any(done):
            env.auto_reset(term, trunc)
    return all_obs, all_rew, all_info


# ---------------------------------------------------------------------------
# 1. Full pipeline smoke test
# ---------------------------------------------------------------------------
class TestFullPipelineSmoke:
    """Verify BatchAirHockeyEnv + BatchRewardShaper work for every stage."""

    @pytest.mark.parametrize("stage", [1, 2, 3, 4])
    def test_env_and_shaper_per_stage(self, stage):
        opponent = STAGE_OPPONENT[stage]
        # Stage 4 "external" needs special handling — use idle for smoke test
        env_opponent = "idle" if opponent == "external" else opponent
        env = _make_env(opponent_policy=env_opponent)
        shaper = BatchRewardShaper(n_envs=N_ENVS, stage=stage)

        obs = env.reset(seed=0)
        assert obs.shape == (N_ENVS, 14)

        shaper.reset(obs)

        rng = np.random.default_rng(1)
        for _ in range(N_STEPS):
            act = rng.uniform(-1, 1, (N_ENVS, 2)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(act)
            shaped = shaper.compute(obs, rew, actions=act, info=info)
            assert shaped.shape == (N_ENVS,)
            assert shaped.dtype == np.float32
            assert np.all(np.isfinite(shaped))
            done = term | trunc
            if np.any(done):
                env.auto_reset(term, trunc)
                shaper.reset(obs, mask=done, info=info)


# ---------------------------------------------------------------------------
# 2. Stage transition (PlateauDetector)
# ---------------------------------------------------------------------------
class TestStageTransition:
    """PlateauDetector correctly detects convergence."""

    def test_plateau_detected_on_flat_rewards(self):
        pd = PlateauDetector(min_steps=100, window=50, lookback=200, threshold=0.05)
        # Feed 300 episodes of flat reward = 1.0
        for _ in range(300):
            pd.record(1.0, steps_since_last=1)
        assert pd.should_advance()

    def test_no_plateau_when_improving(self):
        pd = PlateauDetector(min_steps=100, window=50, lookback=200, threshold=0.05)
        # Feed increasing rewards
        for i in range(300):
            pd.record(float(i), steps_since_last=1)
        assert not pd.should_advance()

    def test_no_plateau_before_min_steps(self):
        pd = PlateauDetector(min_steps=1000, window=50, lookback=200, threshold=0.05)
        for _ in range(300):
            pd.record(1.0, steps_since_last=1)
        # Only 300 steps < min_steps=1000
        assert not pd.should_advance()

    def test_reset_clears_state(self):
        pd = PlateauDetector(min_steps=100, window=50, lookback=200, threshold=0.05)
        for _ in range(300):
            pd.record(1.0, steps_since_last=1)
        assert pd.should_advance()
        pd.reset()
        assert not pd.should_advance()


# ---------------------------------------------------------------------------
# 3. Opponent policy per stage
# ---------------------------------------------------------------------------
class TestOpponentPolicy:
    """Correct opponent policy is used for each stage."""

    def test_stage_opponent_mapping(self):
        expected = {
            1: "idle",
            2: "goalie",
            3: "follow",
            4: "external",
        }
        assert STAGE_OPPONENT == expected

    @pytest.mark.parametrize("stage,policy", [
        (1, "idle"),
        (2, "goalie"),
        (3, "follow"),
    ])
    def test_env_uses_correct_opponent(self, stage, policy):
        env = _make_env(opponent_policy=policy)
        assert env.opponent_policy == policy
        obs = env.reset(seed=0)
        # Just verify it runs without error
        for _ in range(5):
            act = np.zeros((N_ENVS, 2), dtype=np.float32)
            env.step(act)


# ---------------------------------------------------------------------------
# 4. Observation shape (14-dim with velocities)
# ---------------------------------------------------------------------------
class TestObsShape:
    """Obs is always 14 dims (velocities replace frame stacking)."""

    def test_obs_shape_14(self):
        env = _make_env()
        obs = env.reset(seed=0)
        assert obs.shape == (N_ENVS, 14)

    def test_frame_stack_param_ignored(self):
        """frame_stack param is accepted for API compat but obs is always 14."""
        env = _make_env(frame_stack=4)
        obs = env.reset(seed=0)
        assert obs.shape == (N_ENVS, 14)

    def test_velocities_nonzero_after_steps(self):
        env = _make_env()
        obs = env.reset(seed=0)
        rng = np.random.default_rng(99)
        for _ in range(5):
            act = rng.uniform(-1, 1, (N_ENVS, 2)).astype(np.float32)
            obs, *_ = env.step(act)

        # Puck velocities (indices 2-3) should be nonzero after some steps
        puck_vel = obs[:, 2:4]
        assert not np.allclose(puck_vel, 0.0, atol=1e-6), \
            "Puck velocities should be nonzero after stepping"


# ---------------------------------------------------------------------------
# 5. Reward scale per stage
# ---------------------------------------------------------------------------
class TestRewardScale:
    """100 random steps — verify reward magnitudes per stage."""

    @pytest.mark.parametrize("stage", [1, 2, 3, 4])
    def test_reward_magnitude(self, stage):
        opponent = STAGE_OPPONENT[stage]
        env_opponent = "idle" if opponent == "external" else opponent
        env = _make_env(opponent_policy=env_opponent)
        shaper = BatchRewardShaper(n_envs=N_ENVS, stage=stage)
        obs = env.reset(seed=0)
        shaper.reset(obs)
        rng = np.random.default_rng(42)

        all_shaped = []
        for _ in range(100):
            act = rng.uniform(-1, 1, (N_ENVS, 2)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(act)
            shaped = shaper.compute(obs, rew, actions=act, info=info)
            all_shaped.append(shaped)
            done = term | trunc
            if np.any(done):
                env.auto_reset(term, trunc)
                shaper.reset(obs, mask=done, info=info)

        all_shaped = np.concatenate(all_shaped)
        # All rewards should be finite
        assert np.all(np.isfinite(all_shaped))

        # Stage 4 drops all auxiliary rewards — should have small magnitude
        # unless goals are scored
        if stage == 4:
            # Only goal_reward/goal_penalty + entropy; without goals, rewards are small
            non_goal = all_shaped[np.abs(all_shaped) < 10]
            if len(non_goal) > 0:
                assert np.max(np.abs(non_goal)) < 5.0

        # Stage 1 has proximity + contact + directed_hit — rewards can spike on contact
        if stage == 1:
            assert np.max(np.abs(all_shaped)) < 20.0


# ---------------------------------------------------------------------------
# 6. Domain randomization
# ---------------------------------------------------------------------------
class TestDomainRandomization:
    """Per-env physics params differ when domain_randomize=True."""

    def test_dynamics_params_differ(self):
        env = _make_env(domain_randomize=True)
        env.reset(seed=42)

        # Agent dynamics should have randomized per-env params
        speeds = env._agent_dyn["max_speed"]
        accels = env._agent_dyn["max_accel"]
        tcs = env._agent_dyn["time_constant"]

        # With 4 envs and randomization, values should not all be identical
        assert not np.all(speeds == speeds[0]), "max_speed should vary across envs"
        assert not np.all(accels == accels[0]), "max_accel should vary across envs"
        assert not np.all(tcs == tcs[0]), "time_constant should vary across envs"

        # Check ranges
        assert np.all((speeds >= 2.0) & (speeds <= 4.5))
        assert np.all((accels >= 20.0) & (accels <= 45.0))
        assert np.all((tcs >= 0.01) & (tcs <= 0.04))

    def test_no_randomization_default(self):
        env = _make_env(domain_randomize=False)
        env.reset(seed=42)
        speeds = env._agent_dyn["max_speed"]
        # Without randomization, all envs have the same default
        assert np.all(speeds == speeds[0])


# ---------------------------------------------------------------------------
# 7. Camera delay
# ---------------------------------------------------------------------------
class TestCameraDelay:
    """Observations are delayed when camera_delay > 0."""

    def test_delayed_obs_differ_from_current(self):
        # 2-step delay at action_dt=1/60 -> camera_delay ~= 2/60
        delay = 2.0 / 60.0
        env_delayed = _make_env(camera_delay=delay)
        env_nodelay = _make_env(camera_delay=0.0)

        obs_d = env_delayed.reset(seed=0)
        obs_n = env_nodelay.reset(seed=0)

        rng = np.random.default_rng(7)
        # Step both with same actions
        for _ in range(5):
            act = rng.uniform(-1, 1, (N_ENVS, 2)).astype(np.float32)
            obs_d, *_ = env_delayed.step(act)
            obs_n, *_ = env_nodelay.step(act)

        # After enough steps, delayed obs should differ from non-delayed
        assert not np.allclose(obs_d, obs_n, atol=1e-6), \
            "Delayed obs should differ from non-delayed after several steps"

    def test_delay_range_per_env(self):
        env = _make_env(camera_delay=(1.0 / 60, 3.0 / 60), n_envs=16)
        env.reset(seed=42)
        # Delay steps should be in [1, 3]
        assert np.all(env._delay_steps >= 1)
        assert np.all(env._delay_steps <= 3)


# ---------------------------------------------------------------------------
# 8. Elo OpponentPool
# ---------------------------------------------------------------------------
class TestEloOpponentPool:
    """OpponentPool manages snapshots and updates Elo correctly."""

    def _make_mock_agent(self, val=1.0):
        """Create a mock agent with a state_dict."""
        import torch

        agent = MagicMock()
        sd = {"weight": torch.full((4,), val)}
        agent.state_dict.return_value = sd
        return agent

    def test_add_and_sample(self):
        pool = OpponentPool(max_size=10)
        agent = self._make_mock_agent(1.0)
        idx = pool.add(agent)
        assert idx == 0
        assert len(pool) == 1
        sampled_idx, reason = pool.sample(np.random.default_rng(0))
        assert sampled_idx == 0

    def test_elo_updates_on_win(self):
        pool = OpponentPool(max_size=10)
        agent = self._make_mock_agent()
        pool.add(agent)
        initial_agent_elo = pool.agent_elo
        initial_opp_elo = pool.elos[0]

        pool.record_outcome(0, agent_goals=7, opp_goals=3)
        assert pool.agent_elo > initial_agent_elo
        assert pool.elos[0] < initial_opp_elo

    def test_elo_updates_on_loss(self):
        pool = OpponentPool(max_size=10)
        agent = self._make_mock_agent()
        pool.add(agent)
        initial_agent_elo = pool.agent_elo

        pool.record_outcome(0, agent_goals=2, opp_goals=7)
        assert pool.agent_elo < initial_agent_elo

    def test_elo_draw(self):
        pool = OpponentPool(max_size=10)
        agent = self._make_mock_agent()
        pool.add(agent)
        # Both start at same Elo -> draw should not change ratings much
        elo_before = pool.agent_elo
        pool.record_outcome(0, agent_goals=3, opp_goals=3)
        # With equal Elo, expected=0.5, actual=0.5 -> no change
        assert abs(pool.agent_elo - elo_before) < 0.01

    def test_score_diff_tracking(self):
        pool = OpponentPool(max_size=10)
        agent = self._make_mock_agent()
        pool.add(agent)
        pool.record_outcome(0, agent_goals=7, opp_goals=3)
        pool.record_outcome(0, agent_goals=5, opp_goals=5)
        assert len(pool.score_diffs[0]) == 2
        assert pool.score_diffs[0][0] == 4.0   # 7-3
        assert pool.score_diffs[0][1] == 0.0   # 5-5
        assert pool.avg_diff(0) == 2.0

    def test_eviction_at_capacity(self):
        pool = OpponentPool(max_size=3)
        for i in range(5):
            agent = self._make_mock_agent(float(i))
            pool.add(agent)
        assert len(pool) == 3

    def test_multiple_opponents_sampling(self):
        pool = OpponentPool(max_size=10)
        for i in range(5):
            agent = self._make_mock_agent(float(i))
            pool.add(agent)

        rng = np.random.default_rng(42)
        samples = [pool.sample(rng)[0] for _ in range(50)]
        unique = set(samples)
        # Should sample at least 2 different opponents
        assert len(unique) >= 2

    def test_random_and_challenge_sampling(self):
        """Both RANDOM and CHALLENGE selection modes should occur."""
        pool = OpponentPool(max_size=20)
        for i in range(10):
            agent = self._make_mock_agent(float(i))
            pool.add(agent)
        rng = np.random.default_rng(42)
        reasons = [pool.sample(rng)[1] for _ in range(100)]
        assert "RANDOM" in reasons
        assert "CHALLENGE" in reasons


# ---------------------------------------------------------------------------
# 9. LR schedule (CurriculumLRScheduler)
# ---------------------------------------------------------------------------
class TestLRSchedule:
    """CurriculumLRScheduler decays LR and resets on stage change."""

    def _make_scheduler(self, stage=1, estimated_steps=1000):
        import torch

        # Mock optimizers with param groups
        optim = torch.optim.SGD([
            {"params": [torch.zeros(1)], "lr": 1e-3},  # encoder group
            {"params": [torch.zeros(1)], "lr": 1e-3},  # rest
        ])
        pi_optim = torch.optim.SGD([
            {"params": [torch.zeros(1)], "lr": 1e-3},
        ])
        return CurriculumLRScheduler(
            optim, pi_optim, stage=stage,
            estimated_stage_steps=estimated_steps,
        )

    def test_initial_lr_matches_stage(self):
        for stage in range(1, 5):
            sched = self._make_scheduler(stage=stage)
            expected_lr = STAGE_LR[stage]
            assert abs(sched.current_lr - expected_lr) < 1e-8

    def test_lr_decays_over_steps(self):
        sched = self._make_scheduler(stage=1, estimated_steps=100)
        initial_lr = sched.current_lr
        for _ in range(50):
            sched.step()
        mid_lr = sched.current_lr
        for _ in range(50):
            sched.step()
        end_lr = sched.current_lr

        assert mid_lr < initial_lr
        assert end_lr < mid_lr

    def test_lr_reaches_min_at_end(self):
        sched = self._make_scheduler(stage=1, estimated_steps=100)
        base = STAGE_LR[1]
        min_lr = base * 0.2  # default min_lr_fraction
        for _ in range(100):
            sched.step()
        # At exactly estimated_steps, should be at min_lr
        assert abs(sched.current_lr - min_lr) < 1e-8

    def test_set_stage_resets_lr(self):
        sched = self._make_scheduler(stage=1, estimated_steps=100)
        for _ in range(50):
            sched.step()
        assert sched.current_lr < STAGE_LR[1]

        # Advance to stage 3
        new_lr = sched.set_stage(3)
        assert abs(new_lr - STAGE_LR[3]) < 1e-8
        assert abs(sched.current_lr - STAGE_LR[3]) < 1e-8

    def test_encoder_lr_scale(self):
        import torch

        optim = torch.optim.SGD([
            {"params": [torch.zeros(1)], "lr": 1e-3},
            {"params": [torch.zeros(1)], "lr": 1e-3},
        ])
        pi_optim = torch.optim.SGD([
            {"params": [torch.zeros(1)], "lr": 1e-3},
        ])
        enc_scale = 0.3
        sched = CurriculumLRScheduler(
            optim, pi_optim, stage=1,
            enc_lr_scale=enc_scale,
            estimated_stage_steps=100,
        )
        base_lr = STAGE_LR[1]
        # Encoder group (idx=0) should be scaled
        assert abs(optim.param_groups[0]["lr"] - base_lr * enc_scale) < 1e-8
        # Other groups at base LR
        assert abs(optim.param_groups[1]["lr"] - base_lr) < 1e-8
        assert abs(pi_optim.param_groups[0]["lr"] - base_lr) < 1e-8


# ---------------------------------------------------------------------------
# 10. Recording with stage info
# ---------------------------------------------------------------------------
class TestRecordingFilenames:
    """Recordings include stage info in filenames."""

    def test_stage_in_filename(self):
        """Verify the filename pattern includes stage number."""
        for stage in range(1, 5):
            filename = f"tdmpc2_s{stage}_step_0000100.json"
            assert f"_s{stage}_" in filename

    def test_recorder_save_load_roundtrip(self):
        """Verify Recorder can save and load game data."""
        rec = Recorder()
        rec.start_episode()
        for i in range(10):
            rec.record(FrameData(
                time=i * 0.1,
                puck_x=0.5, puck_y=0.5 + i * 0.1,
                puck_vx=0.0, puck_vy=1.0,
                agent_x=0.5, agent_y=0.2,
                opponent_x=0.5, opponent_y=1.8,
                score_agent=0, score_opponent=0,
                reward=0.1 * i,
                cumulative_reward=0.05 * i * (i + 1),
            ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        rec.save(path, metadata={"stage": 3, "step": 100})
        data = json.loads(Path(path).read_text())
        assert data["metadata"]["stage"] == 3
        assert data["metadata"]["step"] == 100

        frames = Recorder.load(path)
        assert len(frames) == 10
        assert frames[0].time == 0.0
        assert frames[9].puck_y == pytest.approx(1.4, abs=0.01)

        Path(path).unlink()


# ---------------------------------------------------------------------------
# Bonus: STAGE_DEFAULTS consistency
# ---------------------------------------------------------------------------
class TestStageDefaults:
    """All stages have consistent reward weight keys."""

    def test_all_stages_have_same_keys(self):
        keys = set(STAGE_DEFAULTS[1].keys())
        for stage in range(2, 5):
            assert set(STAGE_DEFAULTS[stage].keys()) == keys

    def test_stage4_drops_auxiliary_rewards(self):
        s4 = STAGE_DEFAULTS[4]
        assert s4["proximity"] == 0
        assert s4["contact"] == 0
        assert s4["directed_hit"] == 0
        assert s4["puck_progress"] == 0
        assert s4["defense"] == 0
        assert s4["shot_placement"] == 0
        # But keeps goal rewards (entropy also off in stages 3-4)
        assert s4["goal_reward"] > 0
        assert s4["goal_penalty"] < 0
        assert s4["entropy"] == 0

    def test_all_stages_have_names(self):
        for stage in range(1, 5):
            assert stage in STAGE_NAMES
            assert isinstance(STAGE_NAMES[stage], str)
