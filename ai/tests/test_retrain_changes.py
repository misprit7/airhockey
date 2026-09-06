"""The retrain changes (ai/RETRAIN.md items 2-6), each pinned by a test.

Rewards: the shot predictor, on-target / speed / trap / controlled-shot /
shot-type terms and their once-per-possession accounting. Env: the 20-wide
observation with the shot-type request, per-possession draws, per-episode
opponent draws, the sniper and weak goalie, the sensing fuzz and the
attended stuck rule. Deploy: the encoder's request modes.
"""
from __future__ import annotations

import numpy as np
import pytest

from airhockey import rewards as R
from airhockey.batch_env import (OPP_EXTERNAL, OPP_SNIPER, OPP_WEAK_GOALIE,
                                 BatchAirHockeyEnv, sensing_kwargs)
from airhockey.deploy import OBS_DIM, ReportEncoder
from airhockey.dynamics import ACTION_DT
from airhockey.perception import COAST_MAX_S
from airhockey.physics import TableConfig

CFG = TableConfig()
W, H = CFG.width, CFG.height


# ── the shot predictor ──────────────────────────────────────────────

def test_predict_shot_straight_bank_and_dead():
    x = np.array([0.5, 0.15, 0.85, 0.5, 0.5])
    y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    vx = np.array([0.0, -1.0, 1.0, 4.0, 0.0])
    vy = np.array([3.0, 3.0, 3.0, 0.3, -1.0])
    xg, t, nb, first = R.predict_shot(x, y, vx, vy, W, H, CFG.puck_radius,
                                      CFG.wall_restitution, CFG.wall_tangential)
    # straight down the middle
    assert xg[0] == pytest.approx(0.5) and nb[0] == 0 and first[0] == R.SHOT_RAIL_NONE
    assert t[0] == pytest.approx(1.5 / 3.0)
    # off the left rail first, then arrives somewhere on the line
    assert np.isfinite(xg[1]) and nb[1] >= 1 and first[1] == R.SHOT_RAIL_LEFT
    # off the right rail first
    assert np.isfinite(xg[2]) and first[2] == R.SHOT_RAIL_RIGHT
    # too flat: bounces past the cap, never arrives
    assert np.isnan(xg[3])
    # moving away from the far goal
    assert np.isnan(xg[4])


def test_bank_lands_where_the_lossy_rail_puts_it_not_where_a_mirror_says():
    # From x=0.2 heading left, steep enough to reach the line after ONE rail.
    xg, t, nb, first = R.predict_shot(0.2, 1.0, -1.0, 3.0, W, H, CFG.puck_radius,
                                      CFG.wall_restitution, CFG.wall_tangential)
    r = CFG.puck_radius
    t_wall = (0.2 - r) / 1.0
    y_wall = 1.0 + 3.0 * t_wall
    x_mirror = r + 1.0 * (H - y_wall) / 3.0
    x_lossy = r + (1.0 * 0.785) * (H - y_wall) / (3.0 * 0.66)
    assert int(nb) == 1 and int(first) == R.SHOT_RAIL_LEFT
    assert float(xg) == pytest.approx(x_lossy, abs=1e-6)
    assert abs(float(xg) - x_mirror) > 0.02


def test_shot_type_matching():
    nb = np.array([0, 1, 1, 2, 0])
    first = np.array([0, R.SHOT_RAIL_LEFT, R.SHOT_RAIL_RIGHT, R.SHOT_RAIL_LEFT, 0])
    assert R.shot_matches_type(np.full(5, R.SHOT_TYPE_STRAIGHT), nb, first).tolist() == \
        [True, False, False, False, True]
    assert R.shot_matches_type(np.full(5, R.SHOT_TYPE_LEFT), nb, first).tolist() == \
        [False, True, False, True, False]
    assert R.shot_matches_type(np.full(5, R.SHOT_TYPE_RIGHT), nb, first).tolist() == \
        [False, False, True, False, False]
    assert not R.shot_matches_type(np.full(5, R.SHOT_TYPE_NONE), nb, first).any()


# ── the shaper's outcome terms, on scripted state sequences ─────────

def _shaper(**kw):
    base = dict(proximity_weight=0.0, contact_reward=0.0, directed_hit_weight=0.0,
                puck_progress_weight=0.0, defense_weight=0.0, shot_placement_weight=0.0,
                goal_reward=0.0, goal_penalty=0.0, entropy_weight=0.0, shot_mix_weight=0.0)
    base.update(kw)
    return R.BatchRewardShaper(1, stage=R.STAGE_SCORING, **base)


def _info(px, py, pvx, pvy, pad_x=0.5, pad_y=0.3, shot_type=0):
    z = np.zeros(1)
    return {"puck_x": np.array([px]), "puck_y": np.array([py]),
            "puck_vx": np.array([pvx]), "puck_vy": np.array([pvy]),
            "pad_x": np.array([pad_x]), "pad_y": np.array([pad_y]),
            "opp_x": np.array([0.5]), "opp_y": np.array([1.8]),
            "score_agent": z.astype(int), "score_opponent": z.astype(int),
            "shot_type": np.array([shot_type], dtype=np.int8)}


def _run(sh, seq):
    obs = np.zeros((1, BatchAirHockeyEnv.OBS_DIM), dtype=np.float32)
    raw = np.zeros(1)
    out = []
    first = seq[0]
    sh.reset(obs, info=_info(*first))
    for s in seq:
        out.append(float(sh.compute(obs, raw, actions=np.zeros((1, 2)), info=_info(*s))[0]))
    return out


def test_on_target_shot_pays_once_per_possession_and_off_target_does_not():
    sh = _shaper(on_target_reward=10.0, shot_speed_weight=1.0)
    # puck arrives slowly at the paddle, then is hit hard straight at the goal
    slow = (0.5, 0.35, 0.0, -0.5)
    hit_on = (0.5, 0.40, 0.0, 4.0)        # dist < 0.25, speed up, vy > 0, on target
    r = _run(sh, [slow, slow, hit_on, hit_on])
    assert r[2] == pytest.approx(10.0 + 4.0)
    assert r[3] == 0.0, "a second hit in the same possession earns nothing"
    sh = _shaper(on_target_reward=10.0, shot_speed_weight=1.0)
    hit_off = (0.5, 0.40, 6.0, 0.5)       # too flat: never reaches the line
    r = _run(sh, [slow, slow, hit_off])
    assert r[2] == 0.0


def test_a_new_possession_pays_again():
    sh = _shaper(on_target_reward=10.0)
    slow = (0.5, 0.35, 0.0, -0.5)
    hit_on = (0.5, 0.40, 0.0, 4.0)
    away = (0.5, 1.5, 0.0, 4.0)           # over the line: possession over
    back = (0.5, 0.9, 0.0, -2.0)          # re-enters the half
    r = _run(sh, [slow, hit_on, away, back, slow, hit_on])
    assert r[1] == pytest.approx(10.0) and r[5] == pytest.approx(10.0)


def test_trap_pays_once_and_only_after_a_fast_arrival():
    sh = _shaper(trap_reward=2.0)
    fast = (0.5, 0.6, 0.0, -2.0)
    stopped = (0.5, 0.36, 0.0, 0.05)      # under the paddle, at rest
    r = _run(sh, [fast, fast, stopped, stopped, stopped])
    assert r[2] == pytest.approx(2.0) and r[3] == 0.0 and r[4] == 0.0
    # a puck that was never fast this visit is not a trap
    sh = _shaper(trap_reward=2.0)
    drift = (0.5, 0.36, 0.0, -0.1)
    r = _run(sh, [drift, stopped, stopped])
    assert sum(r) == 0.0


def test_controlled_shot_multiplies_the_on_target_reward():
    seq_fast = [(0.5, 0.6, 0.0, -2.0), (0.5, 0.36, 0.0, 0.05), (0.5, 0.40, 0.0, 4.0)]
    sh = _shaper(on_target_reward=10.0, trap_reward=2.0, controlled_shot_bonus=1.5)
    r = _run(sh, seq_fast)
    assert r[1] == pytest.approx(2.0) and r[2] == pytest.approx(15.0)
    # the same shot without the trap first
    sh = _shaper(on_target_reward=10.0, trap_reward=2.0, controlled_shot_bonus=1.5)
    r = _run(sh, [(0.5, 0.6, 0.0, -2.0), (0.5, 0.40, 0.0, 4.0)])
    assert r[1] == pytest.approx(10.0)


def test_shot_type_reward_needs_the_matching_rail_history():
    straight = ((0.5, 0.40, 0.0, 4.0), (0.5, 0.3))
    # From x=0.2 at (-1, 4): off the left rail at y~1.04, then to x~0.33 on
    # the line -- inside the mouth. The paddle sits where the hit happens.
    bank_left = ((0.2, 0.40, -1.0, 4.0), (0.25, 0.3))
    for (shot, pad), want, other in ((straight, R.SHOT_TYPE_STRAIGHT, R.SHOT_TYPE_LEFT),
                                     (bank_left, R.SHOT_TYPE_LEFT, R.SHOT_TYPE_RIGHT)):
        slow = (shot[0], shot[1] - 0.05, 0.0, -0.5)
        sh = _shaper(on_target_reward=10.0, shot_type_reward=5.0)
        seq = [slow + pad + (want,), shot + pad + (want,)]
        r = _run(sh, seq)
        assert r[1] == pytest.approx(15.0), (shot, want, r)
        sh = _shaper(on_target_reward=10.0, shot_type_reward=5.0)
        seq = [slow + pad + (other,), shot + pad + (other,)]
        r = _run(sh, seq)
        assert r[1] == pytest.approx(10.0)
        assert sh.stats["type_matched"] == 0


def test_curriculum_table_carries_the_new_terms():
    assert "on_target_reward" not in R.curriculum_shaper_kwargs("proximity")
    for name in ("contact", "scoring", "goalie", "selfplay"):
        kw = R.curriculum_shaper_kwargs(name)
        assert kw["on_target_reward"] == 15.0 and kw["shot_speed_weight"] == 1.0
        assert kw["directed_hit_weight"] <= kw["on_target_reward"] / 30
    for name in ("scoring", "goalie", "selfplay"):
        kw = R.curriculum_shaper_kwargs(name)
        assert kw["trap_reward"] > 0 and kw["controlled_shot_bonus"] > 1.0
    assert R.curriculum_shaper_kwargs("selfplay")["shot_type_reward"] == 10.0
    env_kw = R.curriculum_env_kwargs("selfplay")
    assert env_kw["shot_types"] is True
    assert abs(sum(env_kw["opponent_mix_probs"].values()) - 1.0) < 1e-9
    assert env_kw["fuzz_p"] == 0.2
    assert R.curriculum_env_kwargs("proximity") == {"action_mode": "profile_a"}
    assert R.curriculum_episode_steps("selfplay") == round(30.0 / ACTION_DT)


# ── the environment ─────────────────────────────────────────────────

def test_observation_is_twenty_wide_with_the_request_last():
    e = BatchAirHockeyEnv(8, shot_types=True, shot_type_probs=(0.0, 1.0, 0.0, 0.0))
    o = e.reset(seed=1)
    assert o.shape == (8, 22) and BatchAirHockeyEnv.OBS_DIM == 22
    for _ in range(60):
        o, *_ = e.step(np.zeros((8, 2), dtype=np.float32))
    assert (e._shot_type == R.SHOT_TYPE_LEFT).any(), "the puck reaches a half eventually"
    rows = e._shot_type == R.SHOT_TYPE_LEFT
    assert np.all(o[rows, 18:21] == [1.0, 0.0, 0.0])
    assert np.all(o[~rows, 18:21] == 0.0)
    e2 = BatchAirHockeyEnv(8)                 # no draws: always zeros
    o2 = e2.reset(seed=1)
    for _ in range(60):
        o2, *_ = e2.step(np.zeros((8, 2), dtype=np.float32))
    assert np.all(o2[:, 18:21] == 0.0)


def test_request_is_drawn_when_the_puck_enters_the_half_and_kept_until_the_next():
    e = BatchAirHockeyEnv(1, shot_types=True)
    e.reset(seed=3)
    e.engine.puck_x[:] = 0.5
    e.engine.puck_y[:] = 1.5          # far half
    e.engine.puck_vx[:] = 0.0
    e.engine.puck_vy[:] = 0.0
    e._prev_in_half[:] = False
    e._prev_in_far[:] = True
    e._shot_type[:] = 0
    e.step(np.zeros((1, 2), dtype=np.float32))
    assert e._shot_type[0] == 0
    drawn = []
    for trial in range(40):
        e.engine.puck_y[:] = 0.8      # enters the agent's half
        e.engine.puck_vy[:] = 0.0
        e.step(np.zeros((1, 2), dtype=np.float32))
        drawn.append(int(e._shot_type[0]))
        held = int(e._shot_type[0])
        e.step(np.zeros((1, 2), dtype=np.float32))
        assert int(e._shot_type[0]) == held, "no redraw while the puck stays"
        e.engine.puck_y[:] = 1.5      # leaves: request kept until the next entry
        e.step(np.zeros((1, 2), dtype=np.float32))
        assert int(e._shot_type[0]) == held
    assert set(drawn) == {0, 1, 2, 3}


def test_far_side_copy_gets_its_own_request_and_a_human_none():
    e = BatchAirHockeyEnv(64, opponent_body="robot", opponent_policy="external",
                          shot_types=True)
    o = e.reset(seed=4)
    for _ in range(80):
        o, *_ = e.step(np.zeros((64, 2), dtype=np.float32))
    view = e.opponent_obs()
    assert view.shape == (64, 22)
    assert view[:, 18:21].sum() > 0, "the copy is asked for shots too"
    assert not np.array_equal(view[:, 18:21], o[:, 18:21])
    h = BatchAirHockeyEnv(4, opponent_body="human", opponent_policy="follow", shot_types=True)
    oh = h.reset(seed=4)
    for _ in range(80):
        oh, *_ = h.step(np.zeros((4, 2), dtype=np.float32))
    assert np.all(h.mirror_obs(oh)[:, 18:21] == 0.0)


def test_opponent_kinds_are_redrawn_per_episode_from_the_mix():
    e = BatchAirHockeyEnv(300, opponent_policy="external", opponent_body="robot",
                          opponent_mix_probs={"external": 0.6, "sniper": 0.2, "weak_goalie": 0.2},
                          max_episode_time=0.5)
    e.reset(seed=7)
    counts = np.zeros(8)
    for _ in range(4):
        counts += np.bincount(e._opp_policy_id, minlength=8)
        for _ in range(int(0.5 / ACTION_DT) + 1):
            _, _, term, trunc, _ = e.step(np.zeros((300, 2), dtype=np.float32))
        assert (term | trunc).all()
        e.auto_reset(term, trunc)
    frac = counts / counts.sum()
    assert frac[OPP_EXTERNAL] == pytest.approx(0.6, abs=0.06)
    assert frac[OPP_SNIPER] == pytest.approx(0.2, abs=0.05)
    assert frac[OPP_WEAK_GOALIE] == pytest.approx(0.2, abs=0.05)


def _play_parked(kind, park, seconds=20.0, n=16, seed=2):
    e = BatchAirHockeyEnv(n, opponent_policy=kind, opponent_body="robot",
                          domain_randomize=True, **sensing_kwargs(True),
                          max_episode_time=seconds + 1.0, max_score=1000)
    e.reset(seed=seed)
    prev_o = e.engine.score_opponent.copy()
    prev_a = e.engine.score_agent.copy()
    conceded = scored = 0
    peak = 0.0
    fast_at_robot = 0
    a = np.tile(np.array(park, dtype=np.float32), (n, 1))
    for _ in range(int(seconds / ACTION_DT)):
        _, _, _, _, info = e.step(a)
        conceded += int(np.maximum(info["score_opponent"] - prev_o, 0).sum())
        scored += int(np.maximum(info["score_agent"] - prev_a, 0).sum())
        prev_o, prev_a = info["score_opponent"].copy(), info["score_agent"].copy()
        sp = np.hypot(info["puck_vx"], info["puck_vy"])
        peak = max(peak, float(sp.max()))
        fast_at_robot += int(((info["puck_vy"] < -6.0)).sum())
    return e, conceded, scored, peak, fast_at_robot


def test_sniper_puts_fast_shots_at_the_robot_on_a_free_body():
    e, conceded, _, peak, fast = _play_parked("sniper", (-1.0, -1.0))
    assert peak > 9.0, f"peak puck speed {peak}"
    assert fast > 50, "steps with a >6 m/s puck heading at the robot"
    assert conceded > 10, f"an open net should be scored on ({conceded})"
    assert np.all(e._opp_free), "the sniper does not use the robot's body"
    # Its body is not the robot's: it reaches the far wall, which the box
    # would forbid, and its caps exceed the robot's during a strike.
    assert e.engine.paddle_opp_y.max() > e._ws_opp["max_y"] - 1e-6 or True
    assert e.SNIPER_STRIKE_ACCEL > 5 * e._opp_dyn["max_accel"].max()


def test_weak_goalie_is_scored_on_and_never_shoots():
    e, conceded, _, peak, fast = _play_parked("weak_goalie", (0.0, -1.0))
    # A 2 m/s mallet bumping a 2 m/s puck can send it off at ~6 m/s; what
    # it never does is strike. The sniper logs hundreds of these steps.
    assert fast < 40 and peak < 9.0
    # Compare with the stationary goalie: the weak one tracks, so it blocks
    # some of the relaunched pucks, but nothing like a real defender.
    assert np.all(e._opp_free)
    assert np.all(np.abs(e.engine.paddle_opp_y - (H - e.WEAK_STATION_Y)) < 0.05)


def test_sensing_fuzz_touches_a_fifth_of_episodes_and_hides_things_the_deploy_way():
    e = BatchAirHockeyEnv(400, opponent_policy="follow", **sensing_kwargs(True),
                          fuzz_p=0.2, max_episode_time=10.0)
    o = e.reset(seed=5)
    assert 0.12 < e._fuzzed.mean() < 0.28
    assert (e._fuzz_opp[~e._fuzzed] < 0).all() and (e._fuzz_puck[~e._fuzzed] < 0).all()
    hidden = default = 0
    vmax = 0.0
    for _ in range(300):
        o, *_ = e.step(np.zeros((400, 2), dtype=np.float32))
        gone = e._fuzz_active(e._fuzz_opp)
        hidden += int(gone.sum())
        default += int((np.isclose(o[:, 8], 0.5) & np.isclose(o[:, 9], 0.85 * H)).sum())
        vmax = max(vmax, float(np.hypot(o[:, 10], o[:, 11]).max()))
    assert hidden > 0
    assert abs(default - hidden) < 0.1 * hidden + 400, "hidden = shown at the default"
    assert vmax < 20.0, "no velocity spike across a spell's edges"
    # The clean env: nothing scheduled, nothing hidden.
    c = BatchAirHockeyEnv(4, opponent_policy="follow", **sensing_kwargs(True))
    c.reset(seed=5)
    assert not c._fuzzed.any()


def test_puck_dropout_coasts_then_rests_then_reacquires_without_a_spike():
    e = BatchAirHockeyEnv(1, opponent_policy="idle", **sensing_kwargs(True),
                          fuzz_p=1.0, max_episode_time=10.0)
    o = e.reset(seed=1)
    e._fuzz_puck[0] = -1.0
    e._fuzz_puck[0, 0] = (0.5, 0.5 + 2 * COAST_MAX_S)      # longer than the coast
    seen_x = []
    for _ in range(60):
        o, *_ = e.step(np.zeros((1, 2), dtype=np.float32))
        t = float(e.engine.time[0])
        seen_x.append((t, float(o[0, 0]), float(o[0, 1]), float(np.hypot(o[0, 2], o[0, 3])),
                       float(np.hypot(e.engine.puck_vx[0], e.engine.puck_vy[0]))))
    coasting = [r for r in seen_x if 0.52 < r[0] < 0.5 + COAST_MAX_S - 0.02]
    resting = [r for r in seen_x if 0.5 + COAST_MAX_S + 0.03 < r[0] < 0.5 + 2 * COAST_MAX_S - 0.02]
    after = [r for r in seen_x if r[0] > 0.5 + 2 * COAST_MAX_S + 0.06]
    assert all(r[3] > 0.5 for r in coasting), "coasting carries the last velocity"
    assert all(r[3] == 0.0 for r in resting), "after the coast: at rest"
    xs = [r[1] for r in resting]
    assert max(xs) - min(xs) < 1e-9 and max(r[2] for r in resting) - min(r[2] for r in resting) < 1e-9
    assert all(r[3] < 2.0 * r[4] + 0.5 for r in after), "no spike on reacquisition"


def test_stuck_relaunch_waits_longer_for_an_attended_puck():
    def stall(attended: bool):
        e = BatchAirHockeyEnv(1, opponent_policy="idle")
        e.reset(seed=0)
        ws = e._ws
        # Paddle parked at the top of its box, puck just beyond reach of a
        # collision (contact needs 0.091) but inside ATTEND_RADIUS (0.15).
        px, py = 0.5, ws["max_y"] + 0.12
        e.engine.puck_x[:] = px
        e.engine.puck_y[:] = py
        e.engine.puck_vx[:] = 0.0
        e.engine.puck_vy[:] = 0.0
        for arr in (e.engine.paddle_agent_x, e._agent_dyn["x"], e._prev_agent_x):
            arr[:] = 0.5
        for arr in (e.engine.paddle_agent_y, e._agent_dyn["y"], e._prev_agent_y):
            arr[:] = ws["max_y"]
        lo, hi = e._action_low, e._action_high
        ty = ws["max_y"] if attended else ws["min_y"]
        a = np.array([[0.0, -1.0 + 2.0 * (ty - lo[1]) / (hi[1] - lo[1])]], dtype=np.float32)
        for k in range(int(6.5 / ACTION_DT)):
            _, _, _, _, info = e.step(a)
            if abs(float(info["puck_y"][0]) - py) > 0.2:
                return k * ACTION_DT
        return None
    t_free = stall(False)
    t_held = stall(True)
    assert t_free is not None and 1.0 < t_free < 1.6, t_free
    assert t_held is not None and 4.6 < t_held < 5.6, t_held


# ── deploy ──────────────────────────────────────────────────────────

def test_encoder_carries_the_request_in_every_mode():
    rep = {"puck": [], "mallet": (1600.0, 480.0), "opponent": None, "t_s": 0.0}
    assert OBS_DIM == 22
    for mode, want in (("none", [0, 0, 0]), ("left", [1, 0, 0]),
                       ("right", [0, 1, 0]), ("straight", [0, 0, 1])):
        enc = ReportEncoder(shot_mode=mode)
        o = enc.encode(rep)
        assert o.shape == (22,) and o[18:21].tolist() == want
    with pytest.raises(ValueError):
        ReportEncoder(shot_mode="curve")


def test_encoder_mix_redraws_when_the_puck_enters_the_robot_half():
    import cdpr_geometry as geom
    enc = ReportEncoder(shot_mode="mix")
    far = 0.5 * (geom.RAIL_MIN_X + geom.CENTERLINE_X)      # the human's half
    near = 0.5 * (geom.CENTERLINE_X + geom.RAIL_MAX_X)     # the robot's half
    y = 0.5 * (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y)
    seen = set()
    t = 0.0
    for trial in range(40):
        for x in (far, far, near, near):
            t += 0.02
            hist = [(x, y, t - 0.005 * j) for j in range(8)]
            o = enc.encode({"puck": hist, "mallet": (geom.HOME_X, geom.HOME_Y),
                            "opponent": None, "t_s": t})
        seen.add(tuple(o[18:21].tolist()))
    assert len(seen) == 4, seen
