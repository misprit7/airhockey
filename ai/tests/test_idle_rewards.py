"""Idle hygiene in self-play: a tiny pull to being centred in front of the
goal and a tax on action dithering, paid only while the puck is on the far
half.

The point is the physical machine: when the puck is away the policy used to
be free to park on the box edge and jitter its target, which on the table is
four steppers reversing at 100 Hz for nothing. The terms are hundredths per
step so they decide what the paddle does only when nothing else does.
"""
from __future__ import annotations

import numpy as np

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.rewards import (CURRICULUM, BatchRewardShaper,
                               curriculum_shaper_kwargs)

_WS = BatchAirHockeyEnv(n_envs=1)._ws
_H = BatchAirHockeyEnv(n_envs=1).table_config.height


def _shaper(n=4, **kw):
    kw.setdefault("home_weight", 0.005)
    kw.setdefault("jitter_weight", 0.005)
    # Every other term off, so what comes back is the idle terms alone.
    zeros = dict(proximity_weight=0.0, contact_reward=0.0, directed_hit_weight=0.0,
                 puck_progress_weight=0.0, goal_reward=0.0, goal_penalty=0.0,
                 defense_weight=0.0, shot_placement_weight=0.0, entropy_weight=0.0,
                 shot_mix_weight=0.0)
    return BatchRewardShaper(n, workspace=_WS, **zeros, **kw)


def _obs(puck_y, pad_x, n=4):
    o = np.zeros((n, 15), dtype=np.float32)
    o[:, 0] = 0.5
    o[:, 1] = puck_y
    o[:, 4] = pad_x
    o[:, 5] = 0.5 * (_WS["min_y"] + _WS["max_y"])
    return o


_X_HOME = 0.5      # the goal's centre x (table width / 2)


def test_home_term_pulls_to_the_goal_centre_only_when_the_puck_is_away():
    far, near = 0.8 * _H, 0.2 * _H
    for puck_y, expect_pull in ((far, True), (near, False)):
        sh = _shaper()
        sh.reset(_obs(puck_y, _X_HOME))
        at_home = sh.compute(_obs(puck_y, _X_HOME), np.zeros(4), actions=np.zeros((4, 2)))
        sh = _shaper()
        sh.reset(_obs(puck_y, _WS["max_x"]))
        at_edge = sh.compute(_obs(puck_y, _WS["max_x"]), np.zeros(4), actions=np.zeros((4, 2)))
        if expect_pull:
            assert np.all(at_home == 0.0)
            np.testing.assert_allclose(at_edge, -0.005, atol=1e-7)
        else:
            assert np.all(at_home == 0.0) and np.all(at_edge == 0.0)


def test_jitter_term_taxes_action_change_only_when_the_puck_is_away():
    far, near = 0.8 * _H, 0.2 * _H
    rng = np.random.default_rng(0)
    for puck_y, expect_tax in ((far, True), (near, False)):
        sh = _shaper()
        sh.reset(_obs(puck_y, _X_HOME))
        # First step after a reset: nothing to compare against, no tax.
        first = sh.compute(_obs(puck_y, _X_HOME), np.zeros(4), actions=np.ones((4, 2)))
        assert np.all(first == 0.0)
        steady = sh.compute(_obs(puck_y, _X_HOME), np.zeros(4), actions=np.ones((4, 2)))
        assert np.all(steady == 0.0)
        flip = sh.compute(_obs(puck_y, _X_HOME), np.zeros(4), actions=-np.ones((4, 2)))
        if expect_tax:
            np.testing.assert_allclose(flip, -0.005 * np.sqrt(8.0), atol=1e-6)
        else:
            assert np.all(flip == 0.0)


def test_smoothness_taxes_action_change_everywhere_but_one_strike_stays_cheap():
    kw = curriculum_shaper_kwargs("selfplay")
    assert kw["smooth_weight"] > 0
    zeros = dict(proximity_weight=0.0, contact_reward=0.0, directed_hit_weight=0.0,
                 puck_progress_weight=0.0, goal_reward=0.0, goal_penalty=0.0,
                 defense_weight=0.0, shot_placement_weight=0.0, entropy_weight=0.0,
                 shot_mix_weight=0.0, home_weight=0.0, jitter_weight=0.0)
    sh = BatchRewardShaper(4, workspace=_WS, smooth_weight=kw["smooth_weight"], **zeros)
    near = 0.2 * _H                                   # puck on the robot's half: play
    sh.reset(_obs(near, _X_HOME))
    sh.compute(_obs(near, _X_HOME), np.zeros(4), actions=np.ones((4, 2)))
    steady = sh.compute(_obs(near, _X_HOME), np.zeros(4), actions=np.ones((4, 2)))
    flip = sh.compute(_obs(near, _X_HOME), np.zeros(4), actions=-np.ones((4, 2)))
    assert np.all(steady == 0.0)
    np.testing.assert_allclose(flip, -kw["smooth_weight"] * np.sqrt(8.0), atol=1e-6)
    # One full flip (a strike) costs less than half a contact but more
    # than one of TD-MPC2's value bins (~0.2), so the heads can see it; a
    # policy flipping on a quarter of its ticks pays goals per episode.
    flip = kw["smooth_weight"] * np.sqrt(8.0)
    assert 0.2 < flip < 0.5 * kw["contact_reward"]
    assert 0.25 * 3000 * flip > kw["goal_reward"]


def test_idle_terms_are_small_against_play():
    """A second parked at the box edge with the puck away costs about one
    contact; a whole 30 s episode of it costs less than one goal, and less
    than the on-target shots it would forgo. Nothing here can outbid play,
    but since the retrain it IS visible to the value bins (0.05/step, was
    0.005): a flat idle landscape was the wander seen on the table."""
    from airhockey.dynamics import ACTION_DT
    kw = curriculum_shaper_kwargs("selfplay")
    per_step = kw["home_weight"] + kw["jitter_weight"] * np.sqrt(8.0)
    steps_per_s = round(1.0 / ACTION_DT)
    assert 0.02 <= per_step <= 0.1
    assert per_step * steps_per_s <= 1.5 * kw["contact_reward"]
    assert per_step * 30 * steps_per_s < kw["goal_reward"]
    assert per_step * 10 * steps_per_s < kw["on_target_reward"] * kw["controlled_shot_bonus"] * 2
    # And only self-play pays them: the pretrain stages are unchanged.
    for name, stage in CURRICULUM.items():
        if name != "selfplay":
            assert stage.get("home_weight", 0.0) == 0.0
            assert stage.get("jitter_weight", 0.0) == 0.0
            assert stage.get("smooth_weight", 0.0) == 0.0


def test_shaper_without_a_workspace_still_centres_on_the_goal():
    sh = BatchRewardShaper(2, home_weight=0.005, jitter_weight=0.0,
                           proximity_weight=0.0, contact_reward=0.0,
                           directed_hit_weight=0.0, puck_progress_weight=0.0,
                           goal_reward=0.0, goal_penalty=0.0, defense_weight=0.0,
                           shot_placement_weight=0.0, entropy_weight=0.0,
                           shot_mix_weight=0.0)
    far = 0.8 * _H
    o = _obs(far, _X_HOME, n=2)
    sh.reset(o)
    assert np.all(sh.compute(o, np.zeros(2), actions=np.zeros((2, 2))) == 0.0)
    o = _obs(far, 0.0, n=2)                     # the side rail: full pull
    np.testing.assert_allclose(sh.compute(o, np.zeros(2), actions=np.zeros((2, 2))), -0.005, atol=1e-7)
