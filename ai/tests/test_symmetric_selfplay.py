"""Symmetric self-play: the far side is a copy of the machine.

Before 2026-09-02 the self-play opponent was the HUMAN model -- a different
dynamics law, 1.25x the speed, 4x the accel with its own randomisation band,
the whole half instead of the workspace box, and a side flag telling the
network which body it had. The learner then met a sparring partner it had
never been in the body of, and that partner played far worse than the robot.
opponent_body="robot" makes the far side an exact copy, and these tests pin
what "exact" means.
"""
from __future__ import annotations

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
from airhockey.dynamics import (AGENT_DR_ACCEL_M_S2, MAX_ACCEL_M_S2,
                                MAX_SPEED_M_S)


def _robot_env(n, **kw):
    kw.setdefault("opponent_policy", "external")
    kw.setdefault("domain_randomize", True)
    return BatchAirHockeyEnv(n_envs=n, opponent_body="robot", **kw)


def test_the_far_side_is_the_agents_body():
    e = _robot_env(500, **sensing_kwargs(True))
    e.reset(seed=3)
    assert e.opponent_dynamics_type == e.agent_dynamics_type == "profile"
    for key in ("max_speed", "max_accel"):
        np.testing.assert_array_equal(e._opp_dyn[key], e._agent_dyn[key])
    # Mirrored box: same x span, y reflected in the centre line.
    h = e.table_config.height
    assert e._ws_opp["min_x"] == e._ws["min_x"]
    assert e._ws_opp["max_x"] == e._ws["max_x"]
    assert e._ws_opp["min_y"] == pytest.approx(h - e._ws["max_y"])
    assert e._ws_opp["max_y"] == pytest.approx(h - e._ws["min_y"])
    # And the far side STARTS inside it, as the agent does in its own.
    w = e._ws_opp
    assert np.all((e.engine.paddle_opp_x >= w["min_x"]) & (e.engine.paddle_opp_x <= w["max_x"]))
    assert np.all((e.engine.paddle_opp_y >= w["min_y"]) & (e.engine.paddle_opp_y <= w["max_y"]))


def test_far_side_targets_are_capped_at_its_box():
    """An external target in the far corner must leave the paddle on the
    box edge, exactly as the agent's clamp would on its side."""
    e = _robot_env(8, domain_randomize=False)
    e.reset(seed=1)
    cfg = e.table_config
    e.set_opponent_actions(np.full(8, cfg.paddle_radius),
                                    np.full(8, cfg.height - cfg.paddle_radius))
    for _ in range(300):
        e.step(np.zeros((8, 2)))
    w = e._ws_opp
    assert np.all(e.engine.paddle_opp_x >= w["min_x"] - 1e-9)
    assert np.all(e.engine.paddle_opp_y <= w["max_y"] + 1e-9)
    # It did try: the paddle sits ON the edge, not somewhere inside.
    assert np.allclose(e.engine.paddle_opp_x, w["min_x"], atol=2e-3)
    assert np.allclose(e.engine.paddle_opp_y, w["max_y"], atol=2e-3)


def test_both_views_carry_the_robot_flag_and_constant_caps():
    e = _robot_env(64, **sensing_kwargs(True))
    obs = e.reset(seed=5)
    m = e.mirror_obs(obs)
    assert np.all(obs[:, 12] == BatchAirHockeyEnv.ROBOT_SIDE)
    assert np.all(m[:, 12] == BatchAirHockeyEnv.ROBOT_SIDE)
    np.testing.assert_allclose(e.mirror_obs(m), obs, atol=1e-6)
    # Caps pinned: speed at the clamp, accel at the top of the old band, so
    # both features are constants the network can ignore until DR reopens.
    assert AGENT_DR_ACCEL_M_S2[0] == AGENT_DR_ACCEL_M_S2[1]
    np.testing.assert_allclose(obs[:, 13], 1.0)
    np.testing.assert_allclose(obs[:, 14], AGENT_DR_ACCEL_M_S2[1] / MAX_ACCEL_M_S2)
    own = e.opponent_obs()
    np.testing.assert_allclose(own[:, 12:], obs[:, 12:])


def _mirror_state(e, src, dst):
    """Make env `dst` the reflection of env `src` in the centre line, with
    the two paddles swapping roles."""
    en = e.engine
    h = e.table_config.height
    en.puck_x[dst] = en.puck_x[src]
    en.puck_y[dst] = h - en.puck_y[src]
    en.puck_vx[dst] = en.puck_vx[src]
    en.puck_vy[dst] = -en.puck_vy[src]
    en.paddle_agent_x[dst] = en.paddle_opp_x[src]
    en.paddle_agent_y[dst] = h - en.paddle_opp_y[src]
    en.paddle_opp_x[dst] = en.paddle_agent_x[src]
    en.paddle_opp_y[dst] = h - en.paddle_agent_y[src]
    for dyn, x, y in ((e._agent_dyn, en.paddle_agent_x, en.paddle_agent_y),
                      (e._opp_dyn, en.paddle_opp_x, en.paddle_opp_y)):
        dyn["x"][dst] = x[dst]
        dyn["y"][dst] = y[dst]
        dyn["vx"][dst] = 0.0
        dyn["vy"][dst] = 0.0
    for arr, ref in ((e._prev_agent_x, en.paddle_agent_x), (e._prev_agent_y, en.paddle_agent_y),
                     (e._prev_opp_x, en.paddle_opp_x), (e._prev_opp_y, en.paddle_opp_y),
                     (e._prev_own_opp_x, en.paddle_opp_x), (e._prev_own_opp_y, en.paddle_opp_y),
                     (e._prev_rival_x, en.paddle_agent_x), (e._prev_rival_y, en.paddle_agent_y)):
        arr[:] = ref


def test_a_game_and_its_reflection_are_the_same_game():
    """Physics + dynamics + clamps + goals must be symmetric under the
    centre-line reflection, and opponent_obs() must be exactly the view the
    reflected agent gets. Env 1 is env 0 reflected with the roles swapped;
    env 0's agent plays action A and its far side B, env 1 the reverse."""
    e = _robot_env(2, domain_randomize=False, realistic_perception=False)
    e.reset(seed=7)
    en = e.engine
    cfg = e.table_config
    # A definite opening: puck at centre moving toward the agent's side.
    en.puck_x[0], en.puck_y[0] = cfg.width * 0.45, cfg.height * 0.5
    en.puck_vx[0], en.puck_vy[0] = 0.6, -1.4
    en.paddle_agent_x[0], en.paddle_agent_y[0] = cfg.width * 0.5, e._ws["min_y"] + 0.05
    en.paddle_opp_x[0], en.paddle_opp_y[0] = cfg.width * 0.4, e._ws_opp["max_y"] - 0.08
    _mirror_state(e, 0, 0)     # syncs dynamics + prev arrays for env 0 too
    _mirror_state(e, 0, 1)
    rng = np.random.default_rng(0)
    h = cfg.height
    steps = 0
    for k in range(400):
        a = rng.uniform(-1, 1, size=2)          # env 0 agent / env 1 far side
        b = rng.uniform(-1, 1, size=2)          # env 0 far side / env 1 agent
        tx, ty = e.mirror_action_to_opponent(np.stack([b, a]))
        e.set_opponent_actions(tx, ty)
        obs, _, _, _, _ = e.step(np.stack([a, b]))
        if en.score_agent.sum() + en.score_opponent.sum() > 0:
            # The post-goal serve is a random draw per env; the reflection
            # ends there by design, not by asymmetry.
            break
        steps = k + 1
        # Both views are built at the same instant, as the trainer builds
        # them: the far side of env 0 sees exactly what env 1's agent sees.
        view = e.opponent_obs()
        # 1e-3: the law runs in float32 millimetres and the velocities are
        # position differences over 10 ms, which amplifies that rounding to
        # a few 1e-4 m/s.
        np.testing.assert_allclose(view[0], obs[1], atol=1e-3, err_msg=f"step {k}")
        np.testing.assert_allclose(view[1], obs[0], atol=1e-3, err_msg=f"step {k}")
        # And the world itself is the reflection.
        assert abs(en.puck_x[0] - en.puck_x[1]) < 3e-5
        assert abs(en.puck_y[0] - (h - en.puck_y[1])) < 3e-5
        assert abs(en.paddle_agent_x[0] - en.paddle_opp_x[1]) < 1e-5
        assert abs(en.paddle_agent_y[0] - (h - en.paddle_opp_y[1])) < 1e-5
    assert steps >= 50, f"only {steps} steps before a goal; fixture too short"


def test_human_far_side_is_untouched():
    """The default is still the human sparring partner: different law,
    different caps, the whole half, flipped flag."""
    from airhockey.dynamics import OPPONENT_MAX_ACCEL_M_S2
    e = BatchAirHockeyEnv(n_envs=16, opponent_policy="external", domain_randomize=True)
    obs = e.reset(seed=2)
    assert e.opponent_body == "human"
    assert e.opponent_dynamics_type == "delayed"
    assert e._ws_opp is None
    assert e._opp_dyn["nominal_accel"] == OPPONENT_MAX_ACCEL_M_S2
    assert np.all(e.mirror_obs(obs)[:, 12] == BatchAirHockeyEnv.HUMAN_SIDE)
    with pytest.raises(ValueError):
        e.opponent_obs()
