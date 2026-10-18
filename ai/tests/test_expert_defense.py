"""The physics teacher's goalkeeping on scripted observations (no simulator).

The teacher sees only the 16-frame history observation; these build that
observation by hand, so they pin what the defense and guard do with it.
"""
import numpy as np
import pytest

from airhockey.expert_player import FRAME, ExpertConfig, ExpertPlayer

# rail30 workspace (runs/rail30-100-20261001-v1/run.json); only the box matters.
OPTIONS = dict(workspace_bounds_mm=[1200.0, 1937.5, 61.4, 904.5], accel=100.0)


def player(**overrides):
    return ExpertPlayer(OPTIONS, ExpertConfig(**overrides))


def observation(frames, paddle=(0.5, 0.17), load=0.3):
    """frames: newest-first list of (x, y, vx, vy) puck reports, 20 ms apart."""
    history = 16
    obs = np.zeros((1, FRAME * history + 3))
    for j in range(history):
        x, y, vx, vy = frames[min(j, len(frames) - 1)]
        f = obs[0, j * FRAME:(j + 1) * FRAME]
        f[0:4] = x, y, vx / 6, vy / 6
        f[4:6] = paddle
        f[8:10] = 0.5, 1.8
        f[21:29] = load
    return obs


def decode(expert, action):
    return expert.decoder.unpack(action)[0][0]


def test_minimum_time_matches_bang_bang_and_uses_velocity():
    e = player()
    start, goal = np.array([[0.5, 0.2]]), np.array([[0.7, 0.2]])
    rest = e._min_time(start, np.zeros((1, 2)), goal, 100.0)[0]
    assert rest == pytest.approx(2 * np.sqrt(0.2 / 100.0))
    toward = e._min_time(start, np.array([[1.0, 0.0]]), goal, 100.0)[0]
    away = e._min_time(start, np.array([[-1.0, 0.0]]), goal, 100.0)[0]
    assert toward < rest < away
    # Too fast to stop: overshoot and come back costs more than from rest.
    assert e._min_time(start, np.array([[8.0, 0.0]]), goal, 100.0)[0] > rest


def test_straight_shot_is_a_threat_and_the_block_sits_on_its_path():
    e = player()
    # Puck at (0.6, 1.0) heading for x = 0.45 on the goal line at 8 m/s.
    direction = np.array([0.45 - 0.6, -1.0])
    v = 8 * direction / np.linalg.norm(direction)
    obs = observation([(0.6, 1.0, *v)] * 3)
    s = e._state(obs)
    threat, act = e._defense(s, s.puck, s.vel)
    assert threat[0]
    target = decode(e, act)
    # Head-on to within the small upstream offset, on the path at its depth.
    path_x = 0.6 + direction[0] * (1.0 - target[1])
    assert abs(target[0] - path_x) < 0.02
    assert e.low[1] <= target[1] < 0.3


def test_puck_going_wide_is_not_a_threat():
    e = player()
    obs = observation([(0.9, 1.0, 0.0, -6.0)] * 3)  # straight down the right rail
    s = e._state(obs)
    threat, _ = e._defense(s, s.puck, s.vel)
    assert not threat[0]


def test_unseen_bank_is_covered_at_the_centre_of_the_restitution_spread():
    e = player()
    # Heading for the left rail, no bounce yet: crossings depend on the rail.
    obs = observation([(0.5, 1.2, -4.0, -6.0)] * 3, paddle=(0.4, 0.11))
    s = e._state(obs)
    threat, act = e._defense(s, s.puck, s.vel)
    assert threat[0]
    target = decode(e, act)
    lines = np.array([target[1]])
    states = e._defense_states(s)
    xs = e._scenario_crossings(states, lines)[:, 0, 0, 0]
    assert xs.max() - xs.min() > 0.02  # the rail is genuinely uncertain
    centre = (xs.max() + xs.min()) / 2
    assert abs(target[0] - centre) < e.config.block_bias_max + 1e-6


def test_report_straddling_a_bounce_is_replaced_by_the_bounced_old_velocity():
    e = player()
    dt = 0.02
    # 40 ms ago heading left at (-6, -6) from 0.25 m off the rail ... the rail
    # is reached after ~35 ms; the current report is a meaningless average.
    old = (0.25, 0.9, -6.0, -6.0)
    newer = (0.13, 0.78, -6.0, -6.0)
    hit = (0.25 - e.table.puck_radius) / 6.0
    after = 2 * dt - hit
    x_now = e.table.puck_radius + 6.0 * 0.785 * after
    now = (x_now, 0.9 - 6.0 * 2 * dt, -0.5, -5.6)  # slope straddling the bounce
    s = e._state(observation([now, newer, old]))
    states = e._defense_states(s)
    assert np.all(states[:, 0, 2] > 3.0)  # moving away from the rail again
    assert np.allclose(states[:, 0, 3], -6.0 * np.asarray(e.config.bank_tangential), atol=0.4)


def test_guard_goes_shallower_as_the_drives_heat():
    e = player()
    depths = []
    for load in (0.3, 0.72, 0.82, 0.90, 0.99):
        obs = observation([(0.5, 1.5, 0.0, 0.0)] * 3, load=load)
        s = e._state(obs)
        depths.append(decode(e, e._guard(s, s.puck, s.vel))[1])
    assert depths == sorted(depths)
    assert depths[0] == pytest.approx(e.config.guard_depths[0], abs=1e-6)
    assert depths[-1] == pytest.approx(e.config.guard_depths[-1], abs=1e-6)
