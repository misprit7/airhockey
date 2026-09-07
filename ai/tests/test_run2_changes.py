"""Run 2 of the retrain (ai/RETRAIN.md): the accel action, its cost, the
time-on-side feature, the patience multiplier, the observation layout
remap for older checkpoints, and the deploy path for a 3-dim policy.
"""
from __future__ import annotations

import numpy as np
import pytest

from airhockey import rewards as R
from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.deploy import OBS_DIM, ReportEncoder, T_SIDE_CLIP_S
from airhockey.dynamics import ACTION_DT
from airhockey.env import AirHockeyEnv
from airhockey.policy_loader import OBS_LAYOUTS, obs_column_map

import cdpr_geometry as geom  # noqa: E402


# ── the accel action ─────────────────────────────────────────────────

def test_accel_fraction_maps_the_action_slot_onto_a_crawl_to_the_cap():
    f = BatchAirHockeyEnv.accel_fraction(np.array([-1.0, 0.0, 1.0, 3.0]))
    # quadratic since run 4: the neutral output is the cheap one
    assert f.tolist() == pytest.approx([0.05, 0.2875, 1.0, 1.0])


def _travel_after(mode, accel_slot, steps=10):
    e = BatchAirHockeyEnv(1, action_mode=mode, opponent_policy="idle")
    e.reset(seed=0)
    x0, y0 = float(e.engine.paddle_agent_x[0]), float(e.engine.paddle_agent_y[0])
    # a target across the box, at the requested accel
    a = np.zeros((1, e.action_dim), dtype=np.float32)
    a[0, 0] = 1.0
    if e.action_dim >= 3:
        a[0, 2] = accel_slot
    for _ in range(steps):
        e.step(a)
    return float(np.hypot(e.engine.paddle_agent_x[0] - x0, e.engine.paddle_agent_y[0] - y0))


def test_a_low_accel_slot_moves_the_paddle_less_in_the_same_time():
    crawl = _travel_after("profile_a", -1.0)
    full = _travel_after("profile_a", 1.0)
    fixed = _travel_after("position", None)
    assert crawl < 0.3 * full, (crawl, full)
    assert full == pytest.approx(fixed, rel=0.05)      # slot 1.0 = the machine's cap


def test_observation_is_22_wide_with_the_layout_the_loader_expects():
    e = BatchAirHockeyEnv(2, action_mode="profile_a", shot_types=True,
                          opponent_policy="external", opponent_body="robot")
    o = e.reset(seed=1)
    assert o.shape == (2, 22) and BatchAirHockeyEnv.OBS_DIM == 22 == OBS_DIM
    groups = {name: (s, w) for name, s, w in OBS_LAYOUTS[22]}
    assert groups["prev_action"] == (BatchAirHockeyEnv.PREV_ACTION_IDX, 3)
    assert groups["shot_type"] == (BatchAirHockeyEnv.SHOT_TYPE_IDX, 3)
    assert groups["t_side"] == (BatchAirHockeyEnv.T_SIDE_IDX, 1)
    a = np.array([[0.2, -0.4, 0.7], [0.0, 0.0, -1.0]], dtype=np.float32)
    o, *_ = e.step(a)
    assert np.allclose(o[:, 15:18], a)
    # the far side's view carries ITS previous action, three wide too
    assert e.opponent_obs().shape == (2, 22)


def test_the_copy_of_the_robot_commands_its_accel_too():
    e = BatchAirHockeyEnv(1, action_mode="profile_a", opponent_policy="external",
                          opponent_body="robot")
    e.reset(seed=0)
    tx, ty = e.mirror_action_to_opponent(np.array([[0.0, 0.0, -1.0]], dtype=np.float32))
    assert e._ext_opp_accel_frac[0] == pytest.approx(0.05)
    tx, ty = e.mirror_action_to_opponent(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    assert e._ext_opp_accel_frac[0] == pytest.approx(1.0)


# ── time on side ─────────────────────────────────────────────────────

def test_time_on_side_counts_up_and_resets_on_a_crossing():
    e = BatchAirHockeyEnv(1, opponent_policy="idle")
    e.reset(seed=0)
    e.engine.puck_x[:] = 0.5
    e.engine.puck_y[:] = 0.6
    e.engine.puck_vx[:] = 0.0
    e.engine.puck_vy[:] = 0.0
    e._t_side[:] = 0.0
    e._prev_in_half[:] = True
    e._prev_in_far[:] = False
    n = 25
    for _ in range(n):
        o, _, _, _, info = e.step(np.zeros((1, 2), dtype=np.float32))
    assert info["t_side"][0] == pytest.approx(n * ACTION_DT)
    assert o[0, 21] == pytest.approx(n * ACTION_DT / BatchAirHockeyEnv.T_SIDE_CLIP)
    e.engine.puck_y[:] = 1.4          # over the line
    o, _, _, _, info = e.step(np.zeros((1, 2), dtype=np.float32))
    assert info["t_side"][0] == 0.0 and o[0, 21] == 0.0
    for _ in range(400):              # saturates at the clip
        o, _, _, _, info = e.step(np.zeros((1, 2), dtype=np.float32))
    assert o[0, 21] == pytest.approx(1.0)


def test_scalar_env_matches_the_batch_layout_and_takes_the_accel_slot():
    env = AirHockeyEnv(action_mode="profile_a")
    o, _ = env.reset()
    assert o.shape == (22,) and env.action_dim == 3
    o, *_ = env.step(np.array([0.1, -0.2, -1.0], dtype=np.float32))
    assert o[15:18].tolist() == pytest.approx([0.1, -0.2, -1.0], abs=1e-6)
    assert 0.0 <= o[21] <= 1.0
    # the 2-dim env still accepts a 3-dim action (the UI plays any checkpoint)
    env2 = AirHockeyEnv()
    env2.reset()
    o2, *_ = env2.step(np.array([0.1, -0.2, 0.5], dtype=np.float32))
    assert o2.shape == (22,) and o2[17] == pytest.approx(0.5)


# ── the shaper: accel cost and patience ──────────────────────────────

def _shaper(**kw):
    base = dict(proximity_weight=0.0, contact_reward=0.0, directed_hit_weight=0.0,
                puck_progress_weight=0.0, defense_weight=0.0, shot_placement_weight=0.0,
                goal_reward=0.0, goal_penalty=0.0, entropy_weight=0.0, shot_mix_weight=0.0)
    base.update(kw)
    return R.BatchRewardShaper(1, stage=R.STAGE_SCORING, **base)


def _info(px, py, pvx, pvy, t_side=0.0, pad_x=0.5, pad_y=0.3):
    z = np.zeros(1)
    return {"puck_x": np.array([px]), "puck_y": np.array([py]),
            "puck_vx": np.array([pvx]), "puck_vy": np.array([pvy]),
            "pad_x": np.array([pad_x]), "pad_y": np.array([pad_y]),
            "opp_x": np.array([0.5]), "opp_y": np.array([1.8]),
            "score_agent": z.astype(int), "score_opponent": z.astype(int),
            "shot_type": np.zeros(1, dtype=np.int8), "t_side": np.array([t_side])}


def test_accel_cost_is_per_step_times_the_fraction_asked_for():
    sh = _shaper(accel_cost_weight=0.02)
    obs = np.zeros((1, 22), dtype=np.float32)
    sh.reset(obs, info=_info(0.5, 1.5, 0.0, 0.0))
    r_full = sh.compute(obs, np.zeros(1), actions=np.array([[0.0, 0.0, 1.0]]), info=_info(0.5, 1.5, 0.0, 0.0))[0]
    r_crawl = sh.compute(obs, np.zeros(1), actions=np.array([[0.0, 0.0, -1.0]]), info=_info(0.5, 1.5, 0.0, 0.0))[0]
    r_2dim = sh.compute(obs, np.zeros(1), actions=np.array([[0.0, 0.0]]), info=_info(0.5, 1.5, 0.0, 0.0))[0]
    assert r_full == pytest.approx(-0.02)
    assert r_crawl == pytest.approx(-0.02 * 0.05)
    assert r_2dim == 0.0
    assert sh.stats["accel_frac_sum"] == pytest.approx(1.05)
    # the neutral output costs less than a third of full
    r_mid = sh.compute(obs, np.zeros(1), actions=np.array([[0.0, 0.0, 0.0]]), info=_info(0.5, 1.5, 0.0, 0.0))[0]
    assert r_mid == pytest.approx(-0.02 * 0.2875)


def test_patience_scales_the_hit_rewards_from_the_floor_to_full():
    def shot_at(t_side):
        sh = _shaper(on_target_reward=10.0, shot_speed_weight=1.0, patience_s=1.5, patience_floor=0.5)
        obs = np.zeros((1, 22), dtype=np.float32)
        slow = _info(0.5, 0.35, 0.0, -0.5, t_side)
        sh.reset(obs, info=slow)
        sh.compute(obs, np.zeros(1), info=slow)
        return float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0, t_side))[0])
    assert shot_at(0.0) == pytest.approx(0.5 * 14.0)
    assert shot_at(0.75) == pytest.approx(0.75 * 14.0)
    assert shot_at(1.5) == pytest.approx(14.0)
    assert shot_at(4.0) == pytest.approx(14.0)


def test_patience_beats_the_discount_by_design():
    """The point of the numbers: at 50 Hz and 0.995, a shot after 1.5 s of
    control must be worth more, discounted, than the instant one."""
    steps = round(1.5 / ACTION_DT)
    discounted_patient = 1.0 * 0.995 ** steps
    assert discounted_patient > 0.05 * 3.0     # floor 0.05, with a wide margin
    kw = R.curriculum_shaper_kwargs("selfplay")
    assert kw["patience_s"] == 1.5
    assert kw["accel_cost_weight"] == 0.04 and kw["patience_on_goals"] is True
    assert kw["control_gate"] is True and kw["cushion_weight"] == 1.5 and kw["hold_income"] == 0.2
    assert kw["on_target_reward"] == 30.0 and kw["trap_reward"] == 10.0 and kw["controlled_shot_bonus"] == 2.0
    assert kw["overstay_cost"] == 0.1 and kw["windup_income"] == 0.2
    assert kw["patience_floor"] == 0.05
    assert R.curriculum_env_kwargs("proximity")["action_mode"] == "profile_a"
    assert R.curriculum_env_kwargs("selfplay")["action_mode"] == "profile_a"
    assert "accel_cost_weight" not in R.curriculum_shaper_kwargs("contact")
    assert sum(R.CURRICULUM[k]["steps"] for k in ("proximity", "contact", "scoring", "goalie")) == 750_000


def test_attended_relaunch_outlasts_the_patience_ramp():
    assert BatchAirHockeyEnv.STUCK_ATTENDED_S > 2 * R.curriculum_shaper_kwargs("selfplay")["patience_s"]


# ── loading older checkpoints ────────────────────────────────────────

def test_obs_column_map_moves_old_groups_to_their_new_columns():
    m = dict(obs_column_map(20, 22))
    assert all(m[i] == i for i in range(17))            # state + prev action x, y
    assert m[17] == 18 and m[18] == 19 and m[19] == 20  # shot type shifted by one
    assert 17 not in m.values() and 21 not in m.values()
    m17 = dict(obs_column_map(17, 22))
    assert m17 == {i: i for i in range(17)}
    m15 = dict(obs_column_map(15, 22))
    assert m15 == {i: i for i in range(15)}
    assert obs_column_map(22, 22) == [(i, i) for i in range(22)]
    with pytest.raises(ValueError):
        obs_column_map(16, 22)


def test_checkpoint_shapes_reads_width_and_action_dim():
    from pathlib import Path
    from airhockey.policy_loader import checkpoint_shapes
    ckpt = Path(__file__).resolve().parents[2] / "runs" / "retrain40_try" / "agent.pt"
    if not ckpt.exists():
        pytest.skip("no retrain40_try checkpoint on this machine")
    obs_dim, action_dim = checkpoint_shapes(ckpt)
    assert (obs_dim, action_dim) == (20, 2)


# ── deploy ───────────────────────────────────────────────────────────

def _rep(x_mm, t):
    hist = [(x_mm, 480.0, t - 0.005 * j) for j in range(8)]
    return {"puck": hist, "mallet": (geom.HOME_X, geom.HOME_Y), "opponent": None, "t_s": t}


def test_encoder_tracks_time_on_side_and_resets_on_a_crossing():
    enc = ReportEncoder()
    near = 0.5 * (geom.CENTERLINE_X + geom.RAIL_MAX_X)     # robot's half
    far = 0.5 * (geom.RAIL_MIN_X + geom.CENTERLINE_X)
    t = 0.0
    for _ in range(50):
        t += 0.02
        o = enc.encode(_rep(near, t))
    assert o.shape == (22,)
    assert o[21] == pytest.approx(49 * 0.02 / T_SIDE_CLIP_S, abs=1e-6)
    t += 0.02
    o = enc.encode(_rep(far, t))
    assert o[21] == 0.0
    assert o[15:18].tolist() == [0.0, 0.0, 0.0]


def test_a_three_dim_policy_commands_its_accel_through_the_runner():
    from airhockey.deploy import TDMPC2Policy
    enc = ReportEncoder()
    # No checkpoint needed for the arithmetic: mimic __call__'s mapping.
    a = np.array([0.0, 0.0, -1.0])
    frac = float(BatchAirHockeyEnv.accel_fraction(a[2]))
    assert frac == pytest.approx(0.05)
    enc.last_action[:] = 0.0
    enc.last_action[:3] = a
    o = enc.encode(_rep(geom.HOME_X, 1.0))
    assert o[17] == pytest.approx(-1.0)
    assert callable(getattr(TDMPC2Policy, "__call__"))


def test_a_goal_from_an_instant_slap_pays_the_floor_and_a_patient_one_pays_whole():
    def goal_after(t_side, on_goals=True):
        sh = _shaper(goal_reward=100.0, patience_s=1.5, patience_floor=0.2,
                     patience_on_goals=on_goals)
        obs = np.zeros((1, 22), dtype=np.float32)
        slow = _info(0.5, 0.35, 0.0, -0.5, t_side)
        sh.reset(obs, info=slow)
        sh.compute(obs, np.zeros(1), info=slow)
        sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0, t_side))   # the hit
        sh.compute(obs, np.zeros(1), info=_info(0.5, 1.5, 0.0, 4.0, t_side))    # in flight
        goal = _info(0.5, 1.0, 0.0, 1.0, 0.0)
        goal["score_agent"] = np.array([1])
        return float(sh.compute(obs, np.zeros(1), info=goal)[0])
    assert goal_after(0.0) == pytest.approx(20.0)
    assert goal_after(0.75) == pytest.approx(60.0)
    assert goal_after(1.5) == pytest.approx(100.0)
    assert goal_after(0.0, on_goals=False) == pytest.approx(100.0)
    # A goal with no hit of the agent's behind it (a relaunch drifting in)
    # is paid whole.
    sh = _shaper(goal_reward=100.0, patience_s=1.5, patience_floor=0.2, patience_on_goals=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    sh.reset(obs, info=_info(0.5, 1.5, 0.0, 1.0))
    goal = _info(0.5, 1.9, 0.0, 1.0); goal["score_agent"] = np.array([1])
    assert float(sh.compute(obs, np.zeros(1), info=goal)[0]) == pytest.approx(100.0)


# ── run 4: the path to control is paid, and control gates the outcome ──

def test_a_touch_that_takes_speed_off_the_puck_is_paid_per_m_per_s():
    sh = _shaper(cushion_weight=1.5)
    obs = np.zeros((1, 22), dtype=np.float32)
    fast = _info(0.5, 0.45, 0.0, -3.0)
    sh.reset(obs, info=fast)
    sh.compute(obs, np.zeros(1), info=fast)
    r = float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, -1.0))[0])   # 3 -> 1 m/s under the paddle
    assert r == pytest.approx(1.5 * 2.0)
    r = float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.38, 0.0, 0.0))[0])    # 1 -> 0
    assert r == pytest.approx(1.5 * 1.0)
    # a bounce that SPEEDS the puck up, or a slowdown far from the paddle, earns nothing
    sh2 = _shaper(cushion_weight=1.5)
    sh2.reset(obs, info=fast); sh2.compute(obs, np.zeros(1), info=fast)
    assert float(sh2.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0))[0]) == 0.0
    sh3 = _shaper(cushion_weight=1.5)
    far = _info(0.5, 0.9, 0.0, -3.0)
    sh3.reset(obs, info=far); sh3.compute(obs, np.zeros(1), info=far)
    assert float(sh3.compute(obs, np.zeros(1), info=_info(0.5, 0.85, 0.0, -1.0))[0]) == 0.0


def test_holding_a_trapped_puck_pays_income_until_the_shot():
    sh = _shaper(trap_reward=3.0, hold_income=0.03)
    obs = np.zeros((1, 22), dtype=np.float32)
    fast = _info(0.5, 0.6, 0.0, -2.0)
    sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
    stopped = _info(0.5, 0.36, 0.0, 0.05)
    r1 = float(sh.compute(obs, np.zeros(1), info=stopped)[0])
    assert r1 == pytest.approx(3.0 + 0.03 * 0.95)
    for _ in range(10):
        r = float(sh.compute(obs, np.zeros(1), info=stopped)[0])
    assert r == pytest.approx(0.03 * 0.95)
    assert sh.stats["hold_steps"] == 11
    # partial control pays partially: a puck at half the scale under the paddle
    assert float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.36, 0.0, 0.5))[0]) == pytest.approx(0.03 * 0.5)
    # once it is shot away the income stops
    assert float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.45, 0.0, 3.0))[0]) == 0.0


def test_on_target_pays_by_shot_speed_and_a_nudge_keeps_the_shot_open():
    def pay(speed, then=None):
        sh = _shaper(on_target_reward=30.0)
        obs = np.zeros((1, 22), dtype=np.float32)
        slow = _info(0.5, 0.35, 0.0, -0.5)
        sh.reset(obs, info=slow); sh.compute(obs, np.zeros(1), info=slow)
        r = float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, speed))[0])
        if then is None:
            return r
        sh.compute(obs, np.zeros(1), info=_info(0.5, 0.42, 0.0, 0.3))      # still on our half, slow
        return r, float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.44, 0.0, then))[0])
    assert pay(1.0) == 0.0, "a nudge is not a shot"
    assert pay(R.SHOT_SPEED_FULL) == pytest.approx(30.0)
    mid = 0.5 * (R.SHOT_SPEED_MIN + R.SHOT_SPEED_FULL)
    assert pay(mid) == pytest.approx(15.0)
    # the nudge did not use up the possession's paid shot
    assert pay(1.0, then=R.SHOT_SPEED_FULL) == (0.0, pytest.approx(30.0))


def test_shot_clock_costs_per_step_after_the_hold_is_established():
    sh = _shaper(hold_income=0.2, overstay_cost=0.1, patience_s=1.5, control_gate=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    fast = _info(0.5, 0.6, 0.0, -2.0)
    sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
    stopped = _info(0.5, 0.36, 0.0, 0.0)
    cap_steps = int(round(R.HOLD_PAY_MAX_S / R.ACTION_DT))
    clock_steps = int(round((R.HOLD_MIN_S + R.SHOT_CLOCK_S) / R.ACTION_DT))
    paid = [float(sh.compute(obs, np.zeros(1), info=stopped)[0]) for _ in range(clock_steps + 10)]
    assert all(r == pytest.approx(0.2) for r in paid[:cap_steps])
    first_neg = next(i for i, r in enumerate(paid) if r < 0)
    assert abs(first_neg - clock_steps) <= 3
    assert all(r == 0.0 for r in paid[cap_steps:first_neg])
    assert all(r == pytest.approx(-0.1) for r in paid[first_neg:])
    assert sh.stats["overstay_steps"] == len(paid) - first_neg
    # ...wound up 0.24 m behind the puck is still within the clock's reach
    back = _info(0.5, 0.40, 0.0, 0.0, pad_x=0.5, pad_y=0.16)
    assert float(sh.compute(obs, np.zeros(1), info=back)[0]) == pytest.approx(-0.1)


def test_windup_income_pays_behind_a_held_puck_on_the_shot_line():
    def run(pad_x, pad_y, windup_income=0.2):
        sh = _shaper(windup_income=windup_income, patience_s=1.5, control_gate=True)
        obs = np.zeros((1, 22), dtype=np.float32)
        fast = _info(0.5, 0.6, 0.0, -2.0)
        sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
        hold_steps = int(round(R.HOLD_MIN_S / R.ACTION_DT))
        for _ in range(hold_steps):
            sh.compute(obs, np.zeros(1), info=_info(0.5, 0.36, 0.0, 0.0))
        wound = _info(0.5, 0.40, 0.0, 0.0, pad_x=pad_x, pad_y=pad_y)
        return [float(sh.compute(obs, np.zeros(1), info=wound)[0]) for _ in range(40)], sh
    cap = int(round(R.WINDUP_PAY_MAX_S / R.ACTION_DT))
    paid, sh = run(0.5, 0.40 - 0.2)                    # 0.2 m straight behind, on the centre line
    assert all(r == pytest.approx(0.2) for r in paid[:cap]) and all(r == 0.0 for r in paid[cap:])
    assert sh.stats["windup_steps"] == cap
    assert run(0.5, 0.40 - 0.05)[0][0] == 0.0, "touching is not wound up"
    assert run(0.5, 0.40 - 0.4)[0][0] == 0.0, "too far back"
    assert run(0.62, 0.40 - 0.2)[0][0] == 0.0, "off the shot line"
    assert run(0.5, 0.40 + 0.2)[0][0] == 0.0, "in front of the puck"
    # not paid unless the puck was held first
    sh = _shaper(windup_income=0.2, patience_s=1.5, control_gate=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    fast = _info(0.5, 0.6, 0.0, -2.0)
    sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
    assert float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 0.0, pad_x=0.5, pad_y=0.2))[0]) == 0.0


def test_hold_income_is_capped_per_possession():
    sh = _shaper(hold_income=0.2, patience_s=1.5, control_gate=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    fast = _info(0.5, 0.6, 0.0, -2.0)
    sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
    stopped = _info(0.5, 0.36, 0.0, 0.0)
    cap_steps = int(round(R.HOLD_PAY_MAX_S / R.ACTION_DT))
    paid = [float(sh.compute(obs, np.zeros(1), info=stopped)[0]) for _ in range(cap_steps + 20)]
    assert all(r == pytest.approx(0.2) for r in paid[:cap_steps])
    assert all(r == 0.0 for r in paid[cap_steps:]), "sitting on the puck pays past the cap"
    # a new possession pays again
    sh.compute(obs, np.zeros(1), info=_info(0.5, 1.5, 0.0, 3.0))
    sh.compute(obs, np.zeros(1), info=fast)
    assert float(sh.compute(obs, np.zeros(1), info=stopped)[0]) == pytest.approx(0.2)


def test_control_gate_pays_the_floor_for_an_uncontrolled_shot_and_full_for_a_held_one():
    hold_steps = int(round(R.HOLD_MIN_S / R.ACTION_DT))

    def shot(held_steps, t_side=3.0):
        sh = _shaper(on_target_reward=15.0, trap_reward=3.0, controlled_shot_bonus=1.5,
                     patience_s=1.5, patience_floor=0.2,
                     control_gate=True, goal_reward=100.0, patience_on_goals=True)
        obs = np.zeros((1, 22), dtype=np.float32)
        fast = _info(0.5, 0.6, 0.0, -2.0, t_side)
        sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
        for _ in range(held_steps):
            sh.compute(obs, np.zeros(1), info=_info(0.5, 0.36, 0.0, 0.05, t_side))
        r_hit = float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0, t_side))[0])
        sh.compute(obs, np.zeros(1), info=_info(0.5, 1.5, 0.0, 4.0, t_side))
        goal = _info(0.5, 1.0, 0.0, 1.0, 0.0); goal["score_agent"] = np.array([1])
        r_goal = float(sh.compute(obs, np.zeros(1), info=goal)[0])
        return r_hit, r_goal
    hit_c, goal_c = shot(hold_steps)
    hit_u, goal_u = shot(0)
    assert hit_u == pytest.approx(0.2 * 15.0) and goal_u == pytest.approx(20.0)
    assert hit_c == pytest.approx(1.5 * 15.0) and goal_c == pytest.approx(100.0)
    # a goal off a BLOCK's rebound (the puck slowed at the touch: not a
    # "hit") is an uncontrolled goal, not a free one
    sh = _shaper(patience_s=1.5, patience_floor=0.2, control_gate=True, goal_reward=100.0,
                 patience_on_goals=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    fast = _info(0.5, 0.6, 0.0, -6.0)
    sh.reset(obs, info=fast); sh.compute(obs, np.zeros(1), info=fast)
    sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0))       # rebound, slower
    sh.compute(obs, np.zeros(1), info=_info(0.5, 1.5, 0.0, 4.0))
    goal = _info(0.5, 1.0, 0.0, 1.0); goal["score_agent"] = np.array([1])
    assert float(sh.compute(obs, np.zeros(1), info=goal)[0]) == pytest.approx(20.0)
    # a touch-and-go stop is not a hold
    assert shot(hold_steps // 3)[0] == pytest.approx(0.2 * 15.0)
    # time on side does NOT substitute for holding, fast arrival or slow
    assert shot(0, t_side=5.0)[0] == pytest.approx(0.2 * 15.0)
    sh = _shaper(on_target_reward=15.0, patience_s=1.5, patience_floor=0.2, control_gate=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    slow = _info(0.5, 0.35, 0.0, -0.5, 2.0)
    sh.reset(obs, info=slow); sh.compute(obs, np.zeros(1), info=slow)
    assert float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0, 2.0))[0]) == pytest.approx(0.2 * 15.0)


def test_a_slow_puck_held_under_the_paddle_counts_as_controlled_without_a_formal_trap():
    # Arrived too slowly for a trap (no trap reward), but held at rest for
    # HOLD_MIN_S: the outcome gate opens all the same.
    hold_steps = int(round(R.HOLD_MIN_S / R.ACTION_DT))
    sh = _shaper(on_target_reward=15.0, trap_reward=3.0, patience_s=1.5, patience_floor=0.05,
                 control_gate=True)
    obs = np.zeros((1, 22), dtype=np.float32)
    slow = _info(0.5, 0.45, 0.0, -0.5)
    sh.reset(obs, info=slow); sh.compute(obs, np.zeros(1), info=slow)
    for _ in range(hold_steps):
        assert float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.36, 0.0, 0.05))[0]) == 0.0
    assert sh.stats["traps"] == 0
    r = float(sh.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0))[0])
    assert r == pytest.approx(15.0)
    # merely SLOWED within reach (above HELD_SPEED) is not control
    sh2 = _shaper(on_target_reward=15.0, patience_s=1.5, patience_floor=0.05, control_gate=True)
    fast = _info(0.5, 0.6, 0.0, -2.0)
    sh2.reset(obs, info=fast); sh2.compute(obs, np.zeros(1), info=fast)
    for _ in range(hold_steps):
        sh2.compute(obs, np.zeros(1), info=_info(0.5, 0.38, 0.0, R.HELD_SPEED + 0.1))
    assert float(sh2.compute(obs, np.zeros(1), info=_info(0.5, 0.40, 0.0, 4.0))[0]) == pytest.approx(0.05 * 15.0)
