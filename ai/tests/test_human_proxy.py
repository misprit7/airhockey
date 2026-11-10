"""The human-proxy benchmark opponent (airhockey/human_proxy.py).

Pins what makes it a stand-in for a person rather than another script: it
reaches the centre line but no further, stays inside the speed and accel it
was drawn, reads the far side's camera report (never engine state), reacts to
a shot only a reaction time late while a planned strike waits for nothing, and
is the same panel of people for every candidate on a given seed. Also pins
that adding it changed nothing for the existing far-side kinds and that the
fast-defense benchmark's new flags default to its old fixtures.

The table tiers' CUSHION is pinned here too: it stops a puck it saw coming
only a reaction time late, gives with a shot it read from the puck's
departure but still goes across no sooner than it reacts, and in the engine
leaves the puck slow enough for a set shot inside its drawn caps.
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from airhockey.batch_env import (_FREE_BODY_KINDS, _OPP_POLICY_MAP, OPP_HUMAN_PROXY,
                                 OPP_SNIPER, OPP_WEAK_GOALIE, BatchAirHockeyEnv)
from airhockey.human_proxy import (APPROACH, CLUB, CUSHION, ELITE, READY, STRIKE, TABLE, TABLE_PLUS,
                                   HumanProxy, command_env, proxy_view, release_speeds)
from airhockey.neural_possession import goal_shot_velocity
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.physics import TableConfig

BIN = Path(__file__).resolve().parents[1] / "bin"
CFG = TableConfig()
H, W = CFG.height, CFG.width
REACH_Y = H / 2 + CFG.paddle_radius


def _game(n=8, seed=3, tier="club"):
    env = NeuralTrainingEnv(n, stage=3, seed=seed, games=True, report_sensing=True,
                            continuous_rallies=True, shot_conditioned=True)
    env.cfg.max_puck_speed = 16.0
    env.reset(seed=seed, opponent="human_proxy")
    proxy = HumanProxy(n, tier, seed=seed + 1, config=env.cfg)
    command_env(env, proxy, reset=True)
    return env, proxy


def _view(n, puck, vel=(0.0, 0.0), own=(0.5, 0.15), rival=(0.5, 1.8), rival_vel=(0.0, 0.0)):
    view = np.zeros((n, 12))
    view[:, 0:2], view[:, 2:4] = puck, vel
    view[:, 4:6], view[:, 8:10], view[:, 10:12] = own, rival, rival_vel
    return view


def test_new_kind_is_a_free_body_and_old_kinds_keep_their_ids():
    assert {k: _OPP_POLICY_MAP[k] for k in ("idle", "follow", "goalie", "external", "corner",
                                            "random", "sniper", "weak_goalie")} == dict(
        idle=0, follow=1, goalie=2, external=3, corner=4, random=5, sniper=6, weak_goalie=7)
    assert _OPP_POLICY_MAP["human_proxy"] == OPP_HUMAN_PROXY
    assert set(_FREE_BODY_KINDS) == {OPP_SNIPER, OPP_WEAK_GOALIE, OPP_HUMAN_PROXY}
    env, _ = _game(4)
    assert (env.base._opp_policy_id == OPP_HUMAN_PROXY).all() and env.base._opp_free.all()


def test_proxy_commands_never_touch_another_kind_sharing_the_batch():
    e = BatchAirHockeyEnv(4, opponent_body="robot", opponent_mix={"sniper": 2, "human_proxy": 2})
    e.reset(seed=1)
    tc = e._opp_dyn_free["time_constant"].copy()
    e.set_human_proxy_command(np.full(4, 0.3), np.full(4, 1.5), 2.0, 40.0, time_constant=0.05)
    np.testing.assert_array_equal(e._opp_dyn_free["time_constant"][:2], tc[:2])
    np.testing.assert_allclose(e._opp_dyn_free["time_constant"][2:], 0.05)
    for _ in range(25):
        e.step(np.zeros((4, 2), np.float32))
    # The sniper's script still owns its body's caps; the proxy's are as commanded.
    assert np.all(e._opp_dyn_free["max_speed"][2:] == 2.0)
    assert np.all(e._opp_dyn_free["max_accel"][2:] == 40.0)
    assert not np.any(e._opp_dyn_free["max_accel"][:2] == 40.0)
    with pytest.raises(ValueError):
        e.set_human_proxy_command(0.5, 1.5, 0.0, 40.0)


def test_reaches_the_centre_line_but_no_further_inside_its_drawn_caps():
    n = 8
    env, proxy = _game(n, seed=5)
    e, base = env.engine, env.base
    np.testing.assert_allclose(base._opp_dyn_free["time_constant"], proxy.time_constant)
    # A slow puck parked on the person's side of the centre line: struck from the line.
    e.puck_x[:] = np.linspace(0.3, 0.7, n)
    e.puck_y[:] = REACH_Y + 0.03
    e.puck_vx[:] = e.puck_vy[:] = 0.0
    if base._perception is not None:
        base._perception.reset(e.puck_x, e.puck_y)
    base._cam_ring[:, :, 0], base._cam_ring[:, :, 1] = e.puck_x, e.puck_y
    base._cam_ring[:, :, 2:4] = 0.0
    previous = np.column_stack((base._opp_dyn_free["vx"], base._opp_dyn_free["vy"]))
    nearest = np.inf
    for _ in range(400):
        command_env(env, proxy)
        env.step(np.zeros((n, 6), np.float32))
        assert e.paddle_opp_y.min() >= REACH_Y - 1e-9
        nearest = min(nearest, float(e.paddle_opp_y.min()))
        velocity = np.column_stack((base._opp_dyn_free["vx"], base._opp_dyn_free["vy"]))
        assert np.all(np.hypot(*velocity.T) <= proxy.max_speed + 1e-6)
        assert np.all(np.hypot(*(velocity - previous).T) <= proxy.max_accel * env.base.action_dt + 1e-6)
        previous = velocity
    assert nearest < REACH_Y + 0.01, "it never used its reach to the line"
    assert proxy.strikes.sum() >= n // 2


def test_reads_the_camera_report_not_the_engine():
    env, proxy = _game(4, seed=7)
    for _ in range(10):
        command_env(env, proxy)
        env.step(np.zeros((4, 6), np.float32))
    before = proxy_view(env)
    # Move the true puck; the camera has not ticked, so the far side cannot know.
    env.engine.puck_x[:] = 0.05
    env.engine.puck_y[:] = 1.95
    after = proxy_view(env)
    np.testing.assert_array_equal(before[:, :4], after[:, :4])
    np.testing.assert_array_equal(before[:, 8:10], after[:, 8:10])
    assert not np.allclose(after[:, :2], np.column_stack((env.engine.puck_x, H - env.engine.puck_y)))


def test_defending_it_reacts_a_reaction_time_late():
    n = 1
    p = HumanProxy(n, CLUB, seed=2)
    rest = _view(n, puck=(0.30, 1.60))
    p.reset(rest)
    p.reaction[:] = 0.2
    p.defense_noise[:] = 0.0
    p.anticipation[:] = 0.0
    for _ in range(15):
        x0 = float(p.act(rest)[0][0])
    # A 6 m/s shot at the left of its goal mouth, starting now.
    puck, vel = np.array([0.30, 1.50]), np.array([0.06, -6.0])
    targets = []
    for k in range(16):
        targets.append(p.act(_view(n, puck=puck + vel * 0.02 * k, vel=vel))[0][0])
    targets = np.array(targets)
    lag = round(0.2 / 0.02)
    np.testing.assert_allclose(targets[:lag], x0, atol=1e-9)        # has not seen it yet
    crossing = 0.30 + 0.06 * (1.50 - p.guard_depth[0]) / 6.0
    assert abs(targets[lag + 1] - crossing) < 0.02 and abs(x0 - crossing) > 0.05
    assert p.max_speed[0] >= 4.0


def test_a_planned_strike_does_not_wait_for_a_reaction():
    p = HumanProxy(1, CLUB, seed=4)
    slow = _view(1, puck=(0.5, 0.6), own=(0.5, 0.2))
    p.reset(slow)
    p.reaction[:] = 0.25
    modes = []
    for _ in range(8):
        p.act(slow)
        modes.append(int(p.mode[0]))
    first = modes.index(APPROACH)
    assert first * 0.02 <= p.DECIDE_S + 0.02 < p.reaction[0]
    assert modes[0] == READY


def test_bank_aim_goes_through_the_rail_into_the_mouth():
    p = HumanProxy(1, CLUB, seed=0)
    puck = np.array([[0.62, 0.55]])
    for route, wall in ((-1, CFG.puck_radius), (1, W - CFG.puck_radius)):
        d = p._shot_direction(puck, np.array([route]), np.array([0.45]))[0]
        # Straight to the rail, then the nominal lossy bounce, then the goal line.
        t = (wall - puck[0, 0]) / d[0]
        y = puck[0, 1] + t * d[1]
        vx, vy = -CFG.wall_restitution * d[0], CFG.wall_tangential * d[1]
        x_goal = wall + vx * (H - y) / vy
        assert abs(x_goal - 0.45) < 1e-9
        # The same unfolding as the shooting-machine fixtures, mirrored.
        table = goal_shot_velocity(np.array([[puck[0, 0], H - puck[0, 1]]]), 0.45, 1.0, route, CFG)[0]
        np.testing.assert_allclose(d, [table[0], -table[1]], atol=1e-12)


def test_a_seed_is_one_panel_of_people_for_every_candidate():
    view = _view(16, puck=(0.5, 1.5))
    a, b = HumanProxy(16, "club", seed=11), HumanProxy(16, "club", seed=11)
    a.reset(view), b.reset(view)
    assert a.parameters() == b.parameters()
    for key in ("plan_route", "plan_aim", "plan_speed", "plan_aim_error", "plan_feint"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    c = HumanProxy(16, "club", seed=12)
    c.reset(view)
    assert c.parameters() != a.parameters()


def test_release_speeds_follow_the_tiers():
    rng = np.random.default_rng(0)
    club = release_speeds(CLUB, rng, 200000)
    assert club.min() >= 3.0 and club.max() <= 11.0
    assert np.median(club) == pytest.approx(CLUB.release[1], abs=0.05)
    assert np.percentile(club, 90) == pytest.approx(CLUB.release[2], abs=0.1)
    elite = release_speeds(ELITE, rng, 1000)
    assert elite.min() >= 8.0 and elite.max() <= 14.0
    view = _view(64, puck=(0.5, 1.5))
    p = HumanProxy(64, "club", seed=3)
    p.reset(view)
    for lo_hi, values in ((CLUB.max_speed_m_s, p.max_speed), (CLUB.max_accel_m_s2, p.max_accel),
                          (CLUB.reaction_s, p.reaction), (CLUB.time_constant_s, p.time_constant)):
        assert lo_hi[0] <= values.min() and values.max() <= lo_hi[1]
    assert abs((p.plan_route != 0).mean() - CLUB.bank_fraction) < 0.02


def test_fast_defense_defaults_are_the_old_fixtures():
    sys.path.insert(0, str(BIN))
    import eval_fast_defense as fd
    signature = inspect.signature(fd.evaluate).parameters
    assert signature["speed_range"].default == (10., 18.)
    assert signature["bank_fraction"].default == pytest.approx(2 / 3)
    assert signature["release_y_range"].default == (1.08, 1.45)
    for count in (48, 768, 50):
        a, b = np.random.default_rng(31), np.random.default_rng(31)
        legacy = np.tile(np.array([0, -1, 1]), (count + 2) // 3)[:count]
        b.shuffle(legacy)
        np.testing.assert_array_equal(fd.attack_routes(a, count), legacy)
        assert a.random() == b.random()      # same draws consumed: later fixtures unchanged
    half = fd.attack_routes(np.random.default_rng(1), 100, 0.5)
    assert (half == 0).sum() == 50 and (half == -1).sum() == 25 and (half == 1).sum() == 25
    with pytest.raises(ValueError):
        fd.attack_routes(np.random.default_rng(1), 10, 1.5)


# ------------------------------------------------------------- cushion
_CATCHER = replace(TABLE, volley_probability=((np.inf, 0.0),), reaction_s=(0.15, 0.15),
                   defense_noise_deg=(0.0, 0.0))


def test_only_the_table_tiers_cushion():
    for tier in (CLUB, ELITE):
        assert tier.cushion_probability is None and tier.brace_probability is None and tier.chase_s is None
        p = HumanProxy(2, tier, seed=1)
        assert not p.cushioning and p.mode_names == ("ready", "approach", "hold", "strike")
    for tier in (TABLE, TABLE_PLUS):
        p = HumanProxy(2, tier, seed=1)
        assert p.cushioning and p.mode_names[CUSHION] == "cushion"
        # Tuned to how often the people set up a 6-8 m/s arrival, not to stop them all.
        assert 0 < max(chance for _, chance in tier.brace_probability) < 1
    with pytest.raises(ValueError):
        HumanProxy(2, replace(CLUB, brace_probability=((np.inf, 1.0),)), seed=1)


def test_cushion_waits_in_the_way_and_gives_as_the_puck_arrives():
    tier = replace(_CATCHER, brace_probability=None, cushion_contact_y_m=(0.34, 0.34))
    p = HumanProxy(3, tier, seed=5)
    rest = _view(3, puck=(0.45, 1.75))
    p.reset(rest)
    for _ in range(12):
        p.act(rest)
    speed, rows = 3.0, []
    for k in range(26):
        y = 1.75 - speed * 0.02 * (k + 1)
        out = p.act(_view(3, puck=(0.45, y), vel=(0.0, -speed), own=p.last_target.copy()))
        rows.append((int(p.mode[0]), y, p.last_target[0].copy(), out[2][0], bool(p.cushion_give[0])))
    modes = [r[0] for r in rows]
    first = modes.index(CUSHION)
    # Seen a reaction time late, and decided on the second tick it is seen coming.
    assert abs(first * 0.02 - (0.15 + 0.02)) <= 0.021 and (p.cushion_attempts == 1).all()
    give = [r[4] for r in rows].index(True)
    assert give > first
    # Until the give it stands on the puck's path, one contact distance behind the line.
    np.testing.assert_allclose(rows[give - 1][2], [0.45, 0.34 - p.contact], atol=1e-6)
    # The give: back along the path at the speed that leaves the puck dead, as it arrives.
    matched = speed * p.ASSUMED_RESTITUTION / (1 + p.ASSUMED_RESTITUTION)
    assert rows[give][2][1] < rows[give - 1][2][1] - 0.1 and rows[give][2][0] == pytest.approx(0.45)
    assert rows[give][3] == pytest.approx(matched, rel=0.05) and rows[give][3] <= p.max_speed[0]
    remaining = (rows[give][1] - 0.34) / speed
    assert 0.0 < remaining < matched / p.max_accel[0] + 0.06


def _read_shot(read, ticks=12):
    """The robot's mallet closes on a still puck (or not), which then leaves fast and wide; per-tick records."""
    tier = replace(_CATCHER, brace_probability=((np.inf, 1.0),))
    p = HumanProxy(2, tier, seed=5)
    p.reset(_view(2, puck=(0.5, 1.6), rival=(0.5, 1.95)))
    for _ in range(30):
        p.act(_view(2, puck=(0.5, 1.6), own=p.last_target.copy(), rival=(0.5, 1.95)))
    for k in range(5):
        p.act(_view(2, puck=(0.5, 1.6), own=p.last_target.copy(), rival=(0.5, 1.95 - 0.04 * (k + 1)),
                    rival_vel=(0.0, -2.0) if read else (0.0, 0.0)))
    stance = p.last_target.copy()
    rows = []
    for k in range(ticks):
        puck = (0.5 - 1.5 * 0.02 * (k + 1), 1.6 - 6.8 * 0.02 * (k + 1))
        p.act(_view(2, puck=puck, vel=(-1.5, -6.8), own=p.last_target.copy(), rival=(0.5, 1.75)))
        rows.append((int(p.mode[0]), p.last_target[0].copy(), bool(p.cushion_give[0])))
    return p, stance, rows


def test_brace_gives_with_a_read_shot_but_goes_across_no_sooner_than_it_reacts():
    p, stance, rows = _read_shot(read=True)
    # While the robot has the puck it stands out from its goal for the room to give.
    assert stance[0, 1] == pytest.approx(max(p.guard_depth[0], p.brace_depth[0]))
    assert TABLE.brace_depth_m[0] <= p.brace_depth[0] <= TABLE.brace_depth_m[1]
    modes = [r[0] for r in rows]
    # Braced on the second tick the puck is seen leaving: before a reaction time has passed.
    assert modes.index(CUSHION) == 1 and p.cushion_braced[0] and (p.braces == 1).all()
    give = [r[2] for r in rows].index(True)
    assert give * 0.02 < 0.15 and rows[give][1][1] < stance[0, 1] - 0.1
    # It was struck wide of the read: the mallet does not follow it sideways until it has reacted.
    reacted = int(0.15 / 0.02)         # the view from a reaction ago first shows it moving on this tick
    for mode, target, _ in rows[:reacted]:
        assert target[0] == pytest.approx(stance[0, 0], abs=1e-9)
    assert rows[-1][1][0] < stance[0, 0] - 0.02
    # Not read (the robot's mallet never closed on the puck): nothing before the reaction.
    q, stance, rows = _read_shot(read=False)
    assert CUSHION not in [r[0] for r in rows[:reacted]] and (q.braces == 0).all()


def _feed(tier, n=8, seed=3, ticks=110):
    """Pucks sent at the person at 1.8-3.2 m/s in the engine; the proxy and its body's peak speed over cap."""
    env = NeuralTrainingEnv(n, stage=3, seed=seed, games=True, report_sensing=True, continuous_rallies=True,
                            shot_conditioned=True)
    env.reset(seed=seed, opponent="human_proxy")
    proxy = HumanProxy(n, tier, seed=seed + 1, config=env.cfg)
    command_env(env, proxy, reset=True)
    e, free = env.engine, env.base._opp_dyn_free
    over = 0.0
    for k in range(ticks):
        if k < 25:
            e.puck_x[:], e.puck_y[:] = np.linspace(0.3, 0.7, n), 0.95
            e.puck_vx[:], e.puck_vy[:] = 0.0, 0.0
        if k == 25:
            e.puck_vy[:] = np.linspace(1.8, 3.2, n)
        command_env(env, proxy)
        env.step(np.zeros((n, 6), np.float32))
        over = max(over, float((np.hypot(free["vx"], free["vy"]) / proxy.max_speed).max()))
    return proxy, over


def test_cushion_stops_a_moving_puck_in_the_engine_for_a_set_shot():
    catcher, over = _feed(_CATCHER)
    assert catcher.cushion_attempts.sum() >= 6 and catcher.cushions.sum() >= 4
    assert catcher.mode_ticks[:, CUSHION].sum() > 0 and over <= 1 + 1e-6
    # The same people without it block or leave these pucks: fewer are set up and struck in the time.
    blocker, _ = _feed(replace(_CATCHER, cushion_probability=None, brace_probability=None, chase_s=None))
    assert not blocker.cushioning and catcher.strikes.sum() >= 3 > blocker.strikes.sum()


def test_a_slow_puck_drifting_out_is_struck_without_setting_up():
    views = dict(puck=(0.5, 0.62), vel=(0.0, 1.0), own=(0.5, 0.40))
    modes = {}
    for name, tier in (("chase", replace(TABLE, feint_probability=1.0, feint_s=(2.0, 2.0))),
                       ("patient", replace(TABLE, feint_probability=1.0, feint_s=(2.0, 2.0), chase_s=None))):
        p = HumanProxy(1, tier, seed=2)
        p.reset(_view(1, **views))
        modes[name] = []
        for _ in range(12):
            p.act(_view(1, **views))
            modes[name].append(int(p.mode[0]))
    # 0.37 m from the line at 1 m/s: it is gone inside chase_s, and the mallet is already behind it.
    assert STRIKE in modes["chase"] and STRIKE not in modes["patient"]


def test_set_share_by_arrival_counts_what_the_person_sent_back():
    sys.path.insert(0, str(BIN))
    import eval_human_proxy as hp
    crossings = ([dict(speed=1.0, outcome="returned", returned_by="strike")] * 3
                 + [dict(speed=1.5, outcome="returned", returned_by="cushion"),
                    dict(speed=1.2, outcome="returned", returned_by=None),       # off the end rail: not counted
                    dict(speed=1.9, outcome="goal"),
                    dict(speed=7.0, outcome="returned", returned_by="ready"),
                    dict(speed=7.5, outcome="returned", returned_by="strike")])
    table = hp.set_share_by_arrival(crossings, minutes=2.0)
    assert table["0-2"] == dict(arrivals_per_min=3.0, returned_per_min=2.0, set_share=0.75)
    assert table["6-8"]["set_share"] == 0.5 and table["4-6"]["set_share"] is None
    for tier in ("table", "table_plus"):
        assert len(hp.TABLE_TARGETS[tier]["set_share_by_arrival_speed"]) == len(hp.ROBOT_SHOT_BINS)


class _StillRobot:
    """A stand-in candidate: the arrival action of all zeros, every tick."""

    history, shot_conditioned, environment_options = 1, True, {}

    def act(self, observation, stochastic=False):
        return np.zeros((len(observation), 6), np.float32)


def test_benchmark_loop_reports_and_records(tmp_path):
    sys.path.insert(0, str(BIN))
    import eval_human_proxy as hp
    from airhockey.recorder import Recorder
    result = hp.play_tier(_StillRobot(), "club", games=4, seconds=4.0, seed=9, options={},
                          record=tmp_path / "still", record_games=1, candidate="still")
    assert result["goals_against"] >= 0 and result["goals_for"] >= 0
    lo, hi = result["goal_share_ci95"]
    assert 0 <= lo <= hi <= 1
    c = result["calibration"]
    assert c["mallet_nearest_robot_goal_y_m"] >= c["reach_limit_y_m"] - 1e-9
    assert set(c["mode_fraction"]) == {"ready", "approach", "hold", "strike"}
    (path,) = result["recordings"]
    frames = Recorder.load(path)
    assert len(frames) == round(4.0 / 0.02)
    assert min(f.opponent_y for f in frames) >= REACH_Y - 1e-9
    assert Path(path).name == "still_vs_human-proxy-club-g0.json"
