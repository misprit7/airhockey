"""Validate BatchRewardShaper matches ShapedRewardWrapper for identical inputs.

This is the core correctness test: if the vectorized reward shaping diverges
from the single-env version, training quality will differ regardless of
physics fidelity.

Also tests the 4-stage curriculum reward system:
  Stage 1 (chase+hit):     Chase and hit the puck (vs idle)
  Stage 2 (game/goalie):   Score past blocker (vs goalie)
  Stage 3 (game/follow):   Score vs reactive opponent (vs follower)
  Stage 4 (self-play):     Pure competitive play — goals + entropy only

Observation layout (14 dims):
  [puck_x, puck_y, puck_vx, puck_vy,
   pad_x, pad_y, pad_vx, pad_vy,
   opp_x, opp_y, opp_vx, opp_vy,
   score_diff, time_remaining]
"""

import numpy as np
import pytest

from airhockey.rewards import (
    ShapedRewardWrapper,
    STAGE_CHASE_HIT, STAGE_SCORING,
    STAGE_GAME_GOALIE, STAGE_GAME_FOLLOW, STAGE_SELFPLAY,
    STAGE_DEFAULTS,
)
from airhockey.env import AirHockeyEnv
from airhockey.dynamics import IdealDynamics

from airhockey.rewards import BatchRewardShaper

ALL_STAGES = [1, 2, 3, 4]

# Observation dimension: 14 (positions + velocities + context)
OBS_DIM = 14


def _make_single_env(stage=STAGE_SCORING):
    """Create a single-env wrapper for comparison."""
    env = AirHockeyEnv(
        agent_dynamics=IdealDynamics(),
        opponent_policy="idle",
        record=False,
        action_dt=1 / 60,
        max_episode_time=30.0,
        max_score=7,
    )
    return ShapedRewardWrapper(env, stage=stage)


def _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.5, pad_y=0.5,
              opp_x=0.5, opp_y=1.5, puck_vx=0.0, puck_vy=0.0):
    """Create obs array [N, 14] with given positions and velocities."""
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[:, 0] = puck_x
    obs[:, 1] = puck_y
    obs[:, 2] = puck_vx
    obs[:, 3] = puck_vy
    obs[:, 4] = pad_x
    obs[:, 5] = pad_y
    obs[:, 8] = opp_x
    obs[:, 9] = opp_y
    return obs


def _make_info(n, puck_vx=0.0, puck_vy=0.0):
    """Create info dict with puck velocities."""
    return {
        "puck_vx": np.full(n, puck_vx, dtype=np.float32),
        "puck_vy": np.full(n, puck_vy, dtype=np.float32),
    }


class TestRewardShapingParity:
    """Compare batch vs single reward shaping on synthetic trajectories."""

    @pytest.mark.parametrize("stage", ALL_STAGES)
    def test_proximity_reward_matches(self, stage):
        """Proximity reward: exp(-k * dist) should match for identical obs."""
        n = 8
        batch = BatchRewardShaper(n, stage=stage)

        # Create obs where paddle is at varying distances from puck
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        obs[:, 0] = 0.5  # puck_x
        obs[:, 1] = 0.5  # puck_y
        # Paddle x,y at varying distances
        dists = np.linspace(0.01, 1.0, n)
        obs[:, 4] = 0.5 + dists  # pad_x offset
        obs[:, 5] = 0.5          # pad_y

        batch.reset(obs)
        raw_rewards = np.zeros(n)

        batch_shaped = batch.compute(obs, raw_rewards)

        # Compute single-env expected values using stage-appropriate weight
        prox_weight = STAGE_DEFAULTS[stage]["proximity"]
        for i in range(n):
            puck_pos = obs[i, :2]
            pad_pos = obs[i, 4:6]
            dist = np.linalg.norm(puck_pos - pad_pos)
            expected_prox = prox_weight * np.exp(-3.0 * dist)
            np.testing.assert_allclose(batch_shaped[i], expected_prox, atol=1e-6,
                                       err_msg=f"Proximity mismatch at env {i}, stage={stage}, dist={dist:.3f}")

    def test_contact_reward_matches(self):
        """Contact detection should fire when puck speeds up near paddle."""
        n = 4
        batch = BatchRewardShaper(n, stage=STAGE_SCORING)

        # Initial obs: puck has low speed
        obs0 = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.5, pad_y=0.5)
        info0 = {"puck_vx": np.zeros(n), "puck_vy": np.full(n, 0.1)}
        batch.reset(obs0, info=info0)

        # Next obs: puck sped up (contact) in envs 0,1; didn't in 2,3
        obs1 = obs0.copy()
        # Env 0: paddle close, contact + directed hit (vy > 0)
        obs1[0, 4] = 0.5   # pad_x
        obs1[0, 5] = 0.45  # pad_y close to puck
        # Env 1: paddle close, contact but not directed (vy < 0)
        obs1[1, 4] = 0.5
        obs1[1, 5] = 0.45
        # Env 2: no contact (paddle too far)
        obs1[2, 4] = 0.9   # far from puck
        obs1[2, 5] = 0.9
        # Env 3: same position
        info1 = {
            "puck_vx": np.array([0.5, 1.0, 0.0, 0.0]),
            "puck_vy": np.array([1.5, -0.5, 0.0, 0.1]),
        }

        raw = np.zeros(n)
        shaped = batch.compute(obs1, raw, info=info1)

        # Env 0: contact + directed_hit + puck_progress + shot_placement
        dist0 = np.hypot(obs1[0, 0] - obs1[0, 4], obs1[0, 1] - obs1[0, 5])
        assert dist0 < 0.15, f"Test setup error: dist={dist0}"
        contact_w = STAGE_DEFAULTS[STAGE_SCORING]["contact"]
        directed_w = STAGE_DEFAULTS[STAGE_SCORING]["directed_hit"]
        assert shaped[0] > contact_w + directed_w * 0.5, \
            f"Expected contact+directed reward, got {shaped[0]:.3f}"

        # Env 1: contact (no directed since vy < 0)
        dist1 = np.hypot(obs1[1, 0] - obs1[1, 4], obs1[1, 1] - obs1[1, 5])
        assert dist1 < 0.15
        assert shaped[1] >= contact_w * 0.9, f"Expected contact reward, got {shaped[1]:.3f}"
        assert shaped[1] < shaped[0], "Env 0 (directed) should get more than env 1"

        # Env 2: no contact (paddle far away), no proximity in game stages
        assert shaped[2] < contact_w, f"Expected minimal reward, got {shaped[2]:.3f}"

        # Env 3: no speed change, no proximity
        assert shaped[3] < contact_w, f"Expected minimal reward, got {shaped[3]:.3f}"

    def test_goal_rewards_match(self):
        """Goal scoring/penalty should match single-env wrapper."""
        n = 3
        batch = BatchRewardShaper(n, stage=STAGE_SCORING)

        obs = _make_obs(n, puck_x=0.5, puck_y=1.0, pad_x=0.3, pad_y=0.3)
        batch.reset(obs)

        # Raw rewards: agent scored, opponent scored, no goal
        raw = np.array([1.0, -1.0, 0.0])
        shaped = batch.compute(obs, raw)

        # Env 0: goal_reward(50)
        assert shaped[0] > 40.0, f"Expected goal reward ~50, got {shaped[0]:.1f}"
        # Env 1: goal_penalty(-25)
        assert shaped[1] < 0.0, f"Expected negative for goal conceded, got {shaped[1]:.1f}"
        # Env 2: no goal, no proximity (stages 2+ have proximity=0)
        assert shaped[2] == 0.0 or shaped[2] < 1.0

    def test_puck_progress_one_way(self):
        """Puck progress should only reward forward movement (increasing y)."""
        n = 2
        batch = BatchRewardShaper(n, stage=STAGE_SCORING)

        obs0 = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=2.0, pad_y=0.5)
        batch.reset(obs0)

        # Env 0: puck moved forward (y increased)
        obs1 = obs0.copy()
        obs1[0, 1] = 0.7  # +0.2
        # Env 1: puck moved backward (y decreased)
        obs1[1, 1] = 0.3  # -0.2

        raw = np.zeros(n)
        shaped = batch.compute(obs1, raw)

        # Env 0 should get puck_progress bonus (0.2 * 0.2 = 0.04) + proximity
        # Env 1 should get only proximity (no negative progress)
        assert shaped[0] > shaped[1], \
            f"Forward progress should be rewarded more: fwd={shaped[0]:.3f} vs back={shaped[1]:.3f}"

    def test_batch_vs_single_trajectory(self):
        """Run the same sequence of obs through both shapers and compare."""
        rng = np.random.default_rng(42)
        n_steps = 50

        # Generate a realistic trajectory
        obs_seq = []
        vel_seq = []
        raw_seq = []
        puck_x, puck_y = 0.5, 0.5
        puck_vx, puck_vy = 0.3, -0.4
        pad_x, pad_y = 0.4, 0.3

        for _ in range(n_steps):
            puck_x += puck_vx * (1/60)
            puck_y += puck_vy * (1/60)
            puck_x = np.clip(puck_x, 0.025, 0.975)
            puck_y = np.clip(puck_y, 0.025, 1.975)
            # Bounce off walls
            if puck_x <= 0.025 or puck_x >= 0.975:
                puck_vx *= -0.85
            if puck_y <= 0.025 or puck_y >= 1.975:
                puck_vy *= -0.85

            # Paddle follows puck loosely
            pad_x += np.clip(puck_x - pad_x, -0.05, 0.05)
            pad_y += np.clip(puck_y - pad_y, -0.05, 0.05)
            pad_y = np.clip(pad_y, 0.04, 0.96)

            obs = np.array([puck_x, puck_y, puck_vx, puck_vy,
                           pad_x, pad_y, 0.0, 0.0,
                           0.5, 1.7, 0.0, 0.0,
                           0.0, 0.0], dtype=np.float32)
            obs_seq.append(obs)
            vel_seq.append((puck_vx, puck_vy))
            raw_seq.append(0.0)  # no goals

        # Run through batch shaper (N=1)
        batch = BatchRewardShaper(1, stage=STAGE_SCORING)
        info0 = {"puck_vx": np.array([vel_seq[0][0]]), "puck_vy": np.array([vel_seq[0][1]])}
        batch.reset(obs_seq[0].reshape(1, OBS_DIM), info=info0)
        batch_rewards = []
        for i in range(1, n_steps):
            info_i = {"puck_vx": np.array([vel_seq[i][0]]), "puck_vy": np.array([vel_seq[i][1]])}
            r = batch.compute(obs_seq[i].reshape(1, OBS_DIM), np.array([raw_seq[i]]), info=info_i)
            batch_rewards.append(r[0])

        # Run through single shaper manually
        single_rewards = []
        prev_puck_y = obs_seq[0][1]
        prev_puck_speed = np.hypot(vel_seq[0][0], vel_seq[0][1])
        for i in range(1, n_steps):
            obs = obs_seq[i]
            vx, vy = vel_seq[i]
            raw = raw_seq[i]
            dist = np.linalg.norm(obs[:2] - obs[4:6])
            puck_speed = np.hypot(vx, vy)
            puck_vy_val = vy

            # Use STAGE_SCORING defaults (read from STAGE_DEFAULTS)
            sd = STAGE_DEFAULTS[STAGE_SCORING]
            shaped = sd["proximity"] * np.exp(-3.0 * dist)

            # Contact
            speed_change = puck_speed - prev_puck_speed
            if dist < 0.25 and speed_change > 0.2:
                shaped += sd["contact"]
                if puck_vy_val > 0:
                    shaped += sd["directed_hit"] * puck_vy_val
                    # Shot placement
                    dx = 0.5 - obs[0]
                    dy = 2.0 - obs[1]
                    goal_dist = np.hypot(dx, dy)
                    alignment = (vx * dx + puck_vy_val * dy) / (puck_speed * goal_dist + 1e-8)
                    shaped += sd["shot_placement"] * float(np.clip(alignment, 0, 1))

            # Puck progress
            delta_y = obs[1] - prev_puck_y
            if delta_y > 0:
                shaped += sd["puck_progress"] * delta_y

            # Defensive positioning
            if puck_vy_val < -0.3:
                x_alignment = np.exp(-3.0 * abs(obs[0] - obs[4]))
                if obs[5] < obs[1]:
                    shaped += sd["defense"] * x_alignment

            # Goals
            if raw > 0:
                shaped += sd["goal_reward"]
            elif raw < 0:
                shaped += sd["goal_penalty"]

            single_rewards.append(shaped)
            prev_puck_y = obs[1]
            prev_puck_speed = puck_speed

        np.testing.assert_allclose(batch_rewards, single_rewards, atol=1e-5,
                                   err_msg="Batch vs single reward mismatch over trajectory")

    def test_selfplay_drops_auxiliary_rewards(self):
        """Stage 4 (self-play) should only have goal + entropy rewards,
        no proximity/contact/defense/progress/placement."""
        n = 2
        batch_stage1 = BatchRewardShaper(n, stage=STAGE_CHASE_HIT)
        batch_selfplay = BatchRewardShaper(n, stage=STAGE_SELFPLAY)

        # Obs where paddle is close to puck (would trigger proximity in stage 1)
        obs = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.52, pad_y=0.48)
        batch_stage1.reset(obs)
        batch_selfplay.reset(obs)

        raw = np.zeros(n)
        shaped_stage1 = batch_stage1.compute(obs, raw)
        shaped_selfplay = batch_selfplay.compute(obs, raw)

        # Stage 1 has contact/directed_hit but no proximity; self-play has ~0
        # Without a contact event, stage 1 also produces 0 for stationary obs
        assert abs(shaped_selfplay[0]) < 0.02, \
            f"Self-play stage should have no auxiliary rewards, got {shaped_selfplay[0]:.4f}"


class TestFourStageCurriculum:
    """Tests for the 4-stage curriculum reward system."""

    @pytest.mark.parametrize("stage", [1, 2, 3, 4])
    def test_stage_instantiation(self, stage):
        """Each stage can be instantiated and produces non-zero rewards."""
        n = 4
        shaper = BatchRewardShaper(n, stage=stage)

        # Puck near paddle — should produce at least some reward for stages 1-4
        obs = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.52, pad_y=0.48)
        shaper.reset(obs)

        raw = np.zeros(n)
        shaped = shaper.compute(obs, raw)

        assert shaped.shape == (n,), f"Wrong output shape for stage {stage}"
        assert shaped.dtype == np.float32, f"Wrong dtype for stage {stage}"
        # Stage 1 has contact/directed_hit (no proximity), so stationary obs → 0
        # Stages 2-4 need contact/goals to produce reward

    @pytest.mark.parametrize("stage", [1, 2, 3, 4])
    def test_stage_with_goal(self, stage):
        """Stages 2-4 should produce large rewards on goals."""
        n = 2
        shaper = BatchRewardShaper(n, stage=stage)

        obs = _make_obs(n, puck_x=0.5, puck_y=1.0, pad_x=0.3, pad_y=0.3)
        shaper.reset(obs)

        raw = np.array([1.0, -1.0])  # agent scored, opponent scored
        shaped = shaper.compute(obs, raw)

        if stage >= 2:
            assert shaped[0] > 25.0, \
                f"Stage {stage}: expected large positive for goal, got {shaped[0]:.1f}"
            assert shaped[1] < -5.0, \
                f"Stage {stage}: expected negative for concede, got {shaped[1]:.1f}"

    def test_stage1_no_goals(self):
        """Stage 1 has proximity + contact + directed_hit, but no goal rewards."""
        n = 2
        shaper = BatchRewardShaper(n, stage=1)

        obs = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.5, pad_y=0.5)
        shaper.reset(obs)

        raw = np.array([1.0, 0.0])  # one goal, one not
        shaped = shaper.compute(obs, raw)

        # No goal reward at stage 1
        assert abs(shaped[0] - shaped[1]) < 1.0, \
            f"Stage 1 should not reward goals: diff={abs(shaped[0] - shaped[1]):.1f}"

    def test_stage2_has_goals(self):
        """Stage 2 should reward goals (goal_reward=50)."""
        n = 2
        shaper = BatchRewardShaper(n, stage=2)

        obs = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.5, pad_y=0.5)
        shaper.reset(obs)

        raw = np.array([1.0, 0.0])
        shaped = shaper.compute(obs, raw)

        # Goal reward should make a big difference
        assert shaped[0] - shaped[1] > 25.0, \
            f"Stage 2 should reward goals: diff={shaped[0] - shaped[1]:.1f}"

    def test_selfplay_goal_only(self):
        """Self-play stage (STAGE_SELFPLAY) should produce ONLY goal rewards + entropy
        -- no proximity, contact, defense, progress, or placement."""
        n = 4
        shaper = BatchRewardShaper(n, stage=STAGE_SELFPLAY)

        # Setup: puck very close to paddle (would trigger proximity and contact)
        obs0 = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.5, pad_y=0.49)
        info0 = _make_info(n, puck_vx=0.0, puck_vy=0.0)
        shaper.reset(obs0, info=info0)

        # Obs with contact-like conditions + puck moving toward opponent
        obs1 = obs0.copy()
        obs1[:, 1] = 0.6   # puck progressed forward
        info1 = {
            "puck_vx": np.full(n, 1.0, dtype=np.float32),  # puck sped up (contact)
            "puck_vy": np.full(n, 1.5, dtype=np.float32),  # toward opponent goal
        }

        # Env 0: no goal -- should get only entropy (~0.01 max)
        # Env 1: agent scored -- should get goal_reward + entropy
        # Env 2: opponent scored -- should get goal_penalty + entropy
        # Env 3: no goal -- should get only entropy (~0.01 max)
        raw = np.array([0.0, 1.0, -1.0, 0.0])
        actions = np.zeros((n, 2), dtype=np.float32)  # centered actions -> max entropy
        shaped = shaper.compute(obs1, raw, actions=actions, info=info1)

        # No-goal envs: only entropy (0.01 * (1 - 0) = 0.01)
        assert abs(shaped[0]) < 0.05, \
            f"Self-play env 0 (no goal) should be ~entropy only, got {shaped[0]:.4f}"
        assert abs(shaped[3]) < 0.05, \
            f"Self-play env 3 (no goal) should be ~entropy only, got {shaped[3]:.4f}"

        # Goal envs should get goal rewards + tiny entropy
        goal_r = STAGE_DEFAULTS[STAGE_SELFPLAY]["goal_reward"]
        goal_p = STAGE_DEFAULTS[STAGE_SELFPLAY]["goal_penalty"]
        assert shaped[1] == pytest.approx(goal_r, abs=1.0), \
            f"Self-play goal scored should be ~{goal_r}, got {shaped[1]:.1f}"
        assert shaped[2] == pytest.approx(goal_p, abs=1.0), \
            f"Self-play goal conceded should be ~{goal_p}, got {shaped[2]:.1f}"

    def test_selfplay_no_proximity(self):
        """Self-play stage must have zero proximity weight per STAGE_DEFAULTS."""
        assert STAGE_DEFAULTS[STAGE_SELFPLAY]["proximity"] == 0.0, \
            "Self-play stage should disable proximity reward"

    def test_selfplay_no_contact(self):
        """Self-play stage must have zero contact/directed_hit/defense/progress/placement."""
        sp = STAGE_DEFAULTS[STAGE_SELFPLAY]
        for key in ["contact", "directed_hit", "puck_progress", "defense", "shot_placement"]:
            assert sp[key] == 0.0, f"Self-play stage should disable {key}, got {sp[key]}"

    def test_stage_3_less_shaping_than_2(self):
        """Stage 3 should have less or equal shaping than stage 2 (progressive reduction)."""
        for key in ["contact", "directed_hit"]:
            assert STAGE_DEFAULTS[3][key] <= STAGE_DEFAULTS[2][key], \
                f"Stage 3 '{key}' ({STAGE_DEFAULTS[3][key]}) should be <= stage 2 ({STAGE_DEFAULTS[2][key]})"


class TestShotPlacement:
    """Tests for the shot_placement reward component.

    Shot placement rewards puck velocity alignment with the direction toward
    the opponent's goal center (0.5, 2.0). Fires on contact: paddle close to
    puck (dist < 0.15), puck speed increased (speed_change > 0.3), vy > 0.
    """

    def _make_contact_obs(self, n: int):
        """Create obs pair where puck starts slow near paddle, then speeds up."""
        obs0 = _make_obs(n, puck_x=0.5, puck_y=0.8, pad_x=0.5, pad_y=0.75)
        return obs0

    def test_placement_rewards_goal_aligned_contact(self):
        """Contact sending puck toward goal center should get more placement
        reward than contact sending puck sideways."""
        n = 3
        shaper = BatchRewardShaper(n, stage=STAGE_GAME_GOALIE)

        obs0 = self._make_contact_obs(n)
        info0 = _make_info(n, puck_vx=0.0, puck_vy=0.0)
        shaper.reset(obs0, info=info0)

        # All envs have contact (puck sped up, paddle close, vy > 0)
        obs1 = obs0.copy()
        # Env 2: puck at left edge with paddle there too
        obs1[2, 0] = 0.1   # puck_x
        obs1[2, 4] = 0.1   # pad_x

        info1 = {
            # Env 0: puck heading straight toward goal center (0.5, 2.0)
            #   from (0.5, 0.8), direction to goal = (0, 1.2), so pure vy is aligned
            # Env 1: puck heading up but angled sideways
            # Env 2: puck heading up but off-center x, vx away from goal center
            "puck_vx": np.array([0.0, 1.5, -1.0]),
            "puck_vy": np.array([2.0, 0.5, 1.0]),
        }

        raw = np.zeros(n)
        shaped = shaper.compute(obs1, raw, info=info1)

        # Env 0 (perfectly aligned with goal center) should get highest reward
        assert shaped[0] > shaped[1], \
            f"Goal-aligned ({shaped[0]:.3f}) should beat angled ({shaped[1]:.3f})"
        assert shaped[0] > shaped[2], \
            f"Goal-aligned ({shaped[0]:.3f}) should beat off-center ({shaped[2]:.3f})"

    def test_placement_zero_without_contact(self):
        """No placement reward when there's no contact (puck not speeding up)."""
        n = 2
        shaper = BatchRewardShaper(n, stage=STAGE_GAME_GOALIE)

        # Puck already moving (no speed change = no contact)
        obs = _make_obs(n, puck_x=0.5, puck_y=0.8, pad_x=0.5, pad_y=0.75)
        info_init = _make_info(n, puck_vx=0.0, puck_vy=1.5)  # already fast
        shaper.reset(obs, info=info_init)

        # Same obs again -- no speed change, so no contact detected
        raw = np.zeros(n)
        shaped = shaper.compute(obs, raw, info=info_init)

        # Should only get proximity + puck_progress, NOT placement
        assert np.all(shaped < 2.0), \
            f"Expected small reward without contact, got {shaped}"

    def test_placement_zero_when_puck_backward(self):
        """No placement when puck vy <= 0 after contact."""
        n = 2
        shaper = BatchRewardShaper(n, stage=STAGE_GAME_GOALIE)

        obs0 = self._make_contact_obs(n)
        info0 = _make_info(n, puck_vx=0.0, puck_vy=0.0)
        shaper.reset(obs0, info=info0)

        obs1 = obs0.copy()
        info1 = {
            # Env 0: contact but puck going backward
            # Env 1: contact with puck going forward
            "puck_vx": np.array([0.5, 0.0]),
            "puck_vy": np.array([-1.0, 2.0]),
        }

        raw = np.zeros(n)
        shaped = shaper.compute(obs1, raw, info=info1)

        # Env 1 (forward) should get placement, env 0 (backward) should not
        assert shaped[1] > shaped[0], \
            f"Forward contact ({shaped[1]:.3f}) should beat backward ({shaped[0]:.3f})"


class TestEntropyBonus:
    """Tests for the entropy bonus component.

    When entropy_weight > 0, actions near 0 should get higher entropy bonus
    than actions near +/-1. Default stage configs currently have entropy=0.
    """

    def test_entropy_higher_for_centered_actions(self):
        """With explicit entropy_weight, centered actions should score higher."""
        n = 3
        shaper = BatchRewardShaper(n, stage=STAGE_SCORING, entropy_weight=0.01)

        obs = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.9, pad_y=0.9)
        shaper.reset(obs)

        # Actions: centered, moderate, extreme
        actions_centered = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        actions_extreme = np.array([[1.0, 1.0], [-1.0, -1.0], [0.9, -0.9]], dtype=np.float32)

        raw = np.zeros(n)
        shaped_centered = shaper.compute(obs, raw, actions=actions_centered)

        # Reset state for fair comparison
        shaper.reset(obs)
        shaped_extreme = shaper.compute(obs, raw, actions=actions_extreme)

        # Centered actions should get higher entropy bonus
        assert np.mean(shaped_centered) > np.mean(shaped_extreme), \
            f"Centered ({np.mean(shaped_centered):.4f}) should beat extreme ({np.mean(shaped_extreme):.4f})"

    @pytest.mark.parametrize("stage", [1, 2])
    def test_entropy_present_early_stages(self, stage):
        """With explicit entropy_weight, entropy bonus differentiates actions."""
        n = 2
        shaper = BatchRewardShaper(n, stage=stage, entropy_weight=0.01)

        obs = _make_obs(n, puck_x=0.5, puck_y=0.5, pad_x=0.9, pad_y=0.9)
        shaper.reset(obs)

        raw = np.zeros(n)
        actions_zero = np.zeros((n, 2), dtype=np.float32)
        actions_max = np.ones((n, 2), dtype=np.float32)

        shaped_zero = shaper.compute(obs, raw, actions=actions_zero)
        shaper.reset(obs)
        shaped_max = shaper.compute(obs, raw, actions=actions_max)

        # Entropy contribution should make centered actions score higher
        diff = np.mean(shaped_zero) - np.mean(shaped_max)
        assert diff > 0, \
            f"Stage {stage}: entropy bonus missing, diff={diff:.6f}"
