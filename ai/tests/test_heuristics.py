"""The heuristic bots, their prediction maths, and the sim bridge.

Three things are worth testing here and they fail in different ways:

  * the PREDICTION, which is arithmetic and can be checked against closed
    forms and against its own lossless limit;
  * the CONVERSIONS, which are the classic place a coordinate system gets
    silently transposed -- so they are checked as round trips rather than
    against hand-copied numbers;
  * the BOTS, whose output has to be inside the workspace and inside the
    machine's caps for every input, including inputs the tracker only
    produces when something has gone wrong (no puck, a frozen puck, a puck
    reported outside the rails).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
from airhockey.dynamics import (sim_to_table_mm, table_mm_to_sim,
                                workspace_in_sim)
from airhockey.heuristic_bridge import SimBridge
from airhockey.heuristics import (BOTS, BotConfig, Command, PuckSample,
                                  TrackerReport, advance_puck,
                                  estimate_velocity, fold, predict_crossing,
                                  puck_bounds, reach_time_s, travel_distance_mm,
                                  travel_time_s, make_bot)

Y_LO, Y_HI = puck_bounds()[2:]


# ── prediction maths ─────────────────────────────────────────────────────

def test_fold_reflects_into_range():
    assert fold(500.0, 100.0, 900.0) == pytest.approx(500.0)
    assert fold(1000.0, 100.0, 900.0) == pytest.approx(800.0)      # one bounce
    assert fold(0.0, 100.0, 900.0) == pytest.approx(200.0)
    for v in (-5000.0, -13.0, 4321.0, 99999.0):
        assert 100.0 <= fold(v, 100.0, 900.0) <= 900.0


def test_predict_crossing_straight_shot():
    hit = predict_crossing(1000.0, 480.0, 4000.0, 0.0, 1900.0, Y_LO, Y_HI)
    assert hit is not None
    assert hit.y_mm == pytest.approx(480.0)
    assert hit.bounces == 0
    assert hit.eta_s > 900.0 / 4000.0        # drag makes it later, not sooner


def test_receding_puck_is_not_a_threat():
    assert predict_crossing(1500.0, 480.0, -3000.0, 200.0,
                            1900.0, Y_LO, Y_HI) is None
    assert predict_crossing(1500.0, 480.0, 0.0, 3000.0,
                            1900.0, Y_LO, Y_HI) is None


def test_puck_already_past_the_line():
    assert predict_crossing(1950.0, 480.0, 3000.0, 0.0,
                            1900.0, Y_LO, Y_HI) is None


def test_lossless_bounce_reduces_to_fold():
    """The whole reason `fold` is still in the file.

    With both rail coefficients at 1 and drag off, walking the bounces must
    give exactly the triangle wave -- so any error in the walk that is not
    also an error in the losses shows up here.
    """
    for vy in (2500.0, -2500.0, 6000.0, -700.0):
        hit = predict_crossing(300.0, 500.0, 3000.0, vy, 1900.0, Y_LO, Y_HI,
                               restitution=1.0, tangential=1.0,
                               drag_b_per_mm=0.0)
        dt = (1900.0 - 300.0) / 3000.0
        assert hit is not None
        assert hit.y_mm == pytest.approx(fold(500.0 + vy * dt, Y_LO, Y_HI),
                                         abs=1e-6)
        assert hit.eta_s == pytest.approx(dt, rel=1e-9)


def test_lossy_bounce_is_steeper_than_specular():
    """The measured rail keeps 78.5% of the normal component and 66% of the
    tangential, so the outgoing ray falls away from the wall FASTER than a
    specular one. A goalie using the specular answer stands 67 mm wrong here,
    which is most of a mallet."""
    args = (900.0, 500.0, 3000.0, 3000.0, 1900.0, Y_LO, Y_HI)
    lossy = predict_crossing(*args)
    specular = predict_crossing(*args, restitution=1.0, tangential=1.0)
    assert lossy.bounces == specular.bounces == 1
    assert lossy.y_mm < specular.y_mm - 40.0


def test_drag_delays_arrival_without_moving_it():
    """Drag is collinear with the velocity, so it can change WHEN but not
    WHERE -- which is exactly why demo_goalie got away with ignoring it and
    why the strikers, which choose a time, cannot."""
    args = (900.0, 400.0, 5000.0, 1500.0, 1900.0, Y_LO, Y_HI)
    with_drag = predict_crossing(*args)
    without = predict_crossing(*args, drag_b_per_mm=0.0)
    assert with_drag.y_mm == pytest.approx(without.y_mm, abs=1e-9)
    assert with_drag.eta_s > without.eta_s
    assert with_drag.speed_mm_s < without.speed_mm_s


def test_travel_time_and_distance_are_inverses():
    for v0 in (300.0, 2000.0, 9000.0):
        for s in (10.0, 250.0, 1800.0):
            t = travel_time_s(s, v0)
            assert travel_distance_mm(t, v0) == pytest.approx(s, rel=1e-9)


def test_advance_puck_matches_crossing_on_a_bounced_path():
    """Two independent walks over the same geometry: stepping forward by TIME
    and solving for a LINE. They have to land in the same place."""
    x, y, vx, vy = 900.0, 500.0, 3000.0, 3000.0
    hit = predict_crossing(x, y, vx, vy, 1900.0, Y_LO, Y_HI)
    px, py, _, _ = advance_puck(x, y, vx, vy, hit.eta_s)
    assert px == pytest.approx(1900.0, abs=0.5)
    assert py == pytest.approx(hit.y_mm, abs=0.5)


def test_advance_puck_stays_on_the_table():
    rng = np.random.default_rng(0)
    x_lo, x_hi, y_lo, y_hi = puck_bounds()
    for _ in range(300):
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        vx, vy = rng.uniform(-8000, 8000, 2)
        px, py, _, _ = advance_puck(x, y, vx, vy, rng.uniform(0.0, 0.5))
        if not x_lo - 1e-6 <= px <= x_hi + 1e-6:
            continue                          # left through a goal mouth
        assert y_lo - 1e-6 <= py <= y_hi + 1e-6


def test_reach_time_is_the_accelerating_solution():
    # Short hop: pure acceleration, d = 1/2 a t^2.
    assert reach_time_s(100.0, 12000.0, 20000.0) == pytest.approx(
        math.sqrt(2 * 100.0 / 20000.0))
    # Long haul: accelerate to the cap, then cruise.
    d = 5000.0
    t = reach_time_s(d, 12000.0, 20000.0)
    d_acc = 12000.0 ** 2 / (2 * 20000.0)
    assert t == pytest.approx(0.6 + (d - d_acc) / 12000.0)
    assert reach_time_s(0.0, 12000.0, 20000.0) == 0.0


# ── velocity estimation ──────────────────────────────────────────────────

def _history(x0, y0, vx, vy, lags=(0, 2, 4, 10, 20), dt=1 / 200):
    """Newest first, exactly as the camera ring reads out."""
    return tuple(PuckSample(x0 - vx * lag * dt, y0 - vy * lag * dt, -lag * dt)
                 for lag in lags)


def test_estimate_velocity_on_a_clean_line():
    est = estimate_velocity(_history(1200.0, 400.0, 3000.0, -800.0))
    assert est.x_mm == pytest.approx(1200.0)
    assert est.vx_mm_s == pytest.approx(3000.0, rel=1e-6)
    assert est.vy_mm_s == pytest.approx(-800.0, rel=1e-6)


def test_estimate_velocity_cuts_at_a_bounce():
    """A window spanning a bounce averages the incoming and outgoing legs into
    a velocity the puck never had -- and does it exactly when the goalie is
    about to need the answer.

    The history here is a real rail bounce 20 ms ago: (3000, +4000) in, and
    (3000x0.66, -4000x0.785) = (1980, -3140) out. Fitting all 40 ms of it
    would report vy near zero for a puck travelling at 3.1 m/s.
    """
    dt = 1 / 200
    y_top = puck_bounds()[3]
    bounce = 4                                   # frames ago
    after = [PuckSample(960.4 + 1980.0 * (bounce - k) * dt,
                        y_top - 3140.0 * (bounce - k) * dt, -k * dt)
             for k in range(bounce + 1)]
    before = [PuckSample(960.4 - 3000.0 * (k - bounce) * dt,
                         y_top - 4000.0 * (k - bounce) * dt, -k * dt)
              for k in range(bounce + 1, 8)]

    est = estimate_velocity(tuple(after + before), window_s=0.06)
    assert est.vy_mm_s == pytest.approx(-3140.0, rel=1e-6)
    assert est.vx_mm_s == pytest.approx(1980.0, rel=1e-6)
    assert est.n_samples == bounce + 1

    naive = estimate_velocity(tuple(after + before), window_s=1e9)
    assert abs(naive.vy_mm_s) < 3140.0, "the cut is doing nothing"


def _noisy_line(rng, spacing_s, n, vx, vy, sigma_mm=0.35):
    """A straight run as the tracker would report it, at a given frame spacing.

    sigma is perception.POS_NOISE_MM -- the measured back-projection noise.
    """
    return tuple(PuckSample(1500.0 - vx * k * spacing_s + rng.normal(0, sigma_mm),
                            500.0 - vy * k * spacing_s + rng.normal(0, sigma_mm),
                            -k * spacing_s)
                 for k in range(n))


@pytest.mark.parametrize("spacing_s,n", [(0.005, 13), (0.010, 7), (0.020, 5)])
def test_bounce_cut_does_not_fire_on_a_straight_line(spacing_s, n):
    """The reason the threshold is in MILLIMETRES and not in mm/s.

    Displacement noise between two frames is sqrt(2)x0.35 = 0.5 mm however far
    apart in time they are, so a millimetre threshold means the same thing at
    any frame spacing. The velocity threshold this replaced did not: the same
    0.5 mm reads as 50 mm/s across a 10 ms gap and 100 mm/s across a 5 ms one,
    so feeding the tracker's native 200 Hz tripped the cut on a THIRD of ticks
    for a slow puck and tripled the estimator's own error.
    """
    rng = np.random.default_rng(0)
    for vx, vy in ((300.0, 0.0), (1000.0, 0.0), (4000.0, 1000.0)):
        ests = [estimate_velocity(_noisy_line(rng, spacing_s, n, vx, vy))
                for _ in range(200)]
        cut = np.mean([e.n_samples <= 2 for e in ests])
        err = np.std([e.vx_mm_s - vx for e in ests])
        assert cut < 0.02, f"cut fired on {cut:.0%} of straight runs"
        assert err < 25.0, f"vx error {err:.0f} mm/s at {spacing_s*1000:.0f} ms"


@pytest.mark.parametrize("spacing_s,n", [(0.005, 13), (0.010, 7)])
def test_bounce_cut_still_fires_on_a_real_bounce(spacing_s, n):
    """Rejecting noise must not cost the thing the cut exists for."""
    rng = np.random.default_rng(1)
    y_top = puck_bounds()[3]
    v_in, bounce = 3000.0, 4
    v_out = -0.785 * v_in

    def history():
        out = []
        for k in range(n):
            # t is time since the bounce, negative before it; y = y_top + v*t
            # on both legs, with the outgoing pair scaled by the measured
            # rail coefficients.
            if k <= bounce:
                t = (bounce - k) * spacing_s
                vx, vy = 0.66 * v_in, v_out
            else:
                t = -(k - bounce) * spacing_s
                vx, vy = v_in, v_in
            out.append(PuckSample(960.4 + vx * t + rng.normal(0, 0.35),
                                  y_top + vy * t + rng.normal(0, 0.35),
                                  -k * spacing_s))
        return tuple(out)

    ests = [estimate_velocity(history(), window_s=1.0) for _ in range(200)]
    assert all(e.n_samples <= bounce + 1 for e in ests), "missed the bounce"
    assert np.mean([e.vy_mm_s for e in ests]) == pytest.approx(v_out, rel=0.02)


def test_bounce_is_localised_at_any_age_given_a_dense_history():
    """A bounce does not wait for a sample boundary.

    The cut can only fire on a segment that straddles the reversal, so what
    matters is that SOME segment is short enough to isolate it. With the
    tracker's native 5 ms ring that is true at every bounce age. It is NOT
    true of a sparse history: BatchAirHockeyEnv.HISTORY_PUCK_LAGS has samples
    at 20 and 50 ms and nothing between, so a bounce 40 ms old falls inside
    one segment, the reversal is averaged away, and the estimate comes back
    ~50% wrong. See the note in the module docstring -- that gap is the
    observation's, not the estimator's, and this test is the guard that the
    estimator itself is not the limitation.
    """
    rng = np.random.default_rng(3)
    y_top = puck_bounds()[3]
    v_in = 2500.0
    v_out = -0.785 * v_in

    def history(age_s):
        out = []
        for k in range(13):                     # 60 ms of 5 ms frames
            t = k * 0.005                       # seconds before now
            dt = age_s - t                      # time since the bounce
            vy, vx = (v_out, 0.66 * 3000.0) if dt >= 0 else (v_in, 3000.0)
            out.append(PuckSample(960.4 + vx * dt + rng.normal(0, 0.35),
                                  y_top + vy * dt + rng.normal(0, 0.35), -t))
        return tuple(out)

    for age_ms in (10, 15, 20, 25, 30, 35, 40, 45, 50):
        ests = [estimate_velocity(history(age_ms / 1000.0)) for _ in range(60)]
        err = np.mean([abs(e.vy_mm_s - v_out) for e in ests])
        assert err < 0.05 * abs(v_out), (
            f"bounce {age_ms} ms old: vy off by {err:.0f} mm/s")


def test_estimate_velocity_degenerate_inputs():
    assert estimate_velocity(()) is None
    one = estimate_velocity((PuckSample(1.0, 2.0, 0.0),))
    assert (one.vx_mm_s, one.vy_mm_s) == (0.0, 0.0)
    # A frozen tracker (the coast has expired) must read as stationary, not
    # as a division by zero.
    frozen = estimate_velocity(tuple(PuckSample(5.0, 6.0, -k / 200)
                                     for k in range(5)))
    assert frozen.vx_mm_s == pytest.approx(0.0)


# ── conversions ──────────────────────────────────────────────────────────

def test_table_mm_sim_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(500):
        mm = (rng.uniform(-200.0, 2200.0), rng.uniform(-200.0, 1200.0))
        back = sim_to_table_mm(*table_mm_to_sim(*mm))
        assert back[0] == pytest.approx(mm[0], abs=1e-9)
        assert back[1] == pytest.approx(mm[1], abs=1e-9)


def test_round_trip_survives_the_flip():
    mm = (1500.0, 300.0)
    assert sim_to_table_mm(*table_mm_to_sim(*mm, flip=True),
                           flip=True) == pytest.approx(mm)


def test_sim_corners_are_the_table_corners():
    """The mapping the whole system's coordinates hang off: sim y 0 is the
    ROBOT's goal line (high grid x) and the axes SWAP."""
    assert table_mm_to_sim(geom.RAIL_MAX_X, geom.RAIL_MIN_Y) == pytest.approx(
        (0.0, 0.0))
    assert table_mm_to_sim(geom.CENTERLINE_X, geom.RAIL_MAX_Y) == pytest.approx(
        (1.0, 1.0))


def test_workspace_in_sim_matches_the_mm_bounds():
    ws = workspace_in_sim()
    lo = table_mm_to_sim(geom.WS_MAX_X, geom.WS_MIN_Y)   # high x -> low sim y
    hi = table_mm_to_sim(geom.WS_MIN_X, geom.WS_MAX_Y)
    assert (ws["min_x"], ws["min_y"]) == pytest.approx((lo[0], lo[1]))
    assert (ws["max_x"], ws["max_y"]) == pytest.approx((hi[0], hi[1]))


# ── the bridge ───────────────────────────────────────────────────────────

def _env(n=4, opponent="goalie", **kw):
    return BatchAirHockeyEnv(
        n_envs=n, opponent_policy=opponent, obs_mode="history",
        action_mode="profile_v", domain_randomize=True, max_score=10 ** 6,
        **sensing_kwargs(True), **kw)


def test_bridge_rejects_the_wrong_env_modes():
    with pytest.raises(ValueError):
        SimBridge(BatchAirHockeyEnv(n_envs=2, obs_mode="kinematic",
                                    action_mode="profile_v"))
    with pytest.raises(ValueError):
        SimBridge(BatchAirHockeyEnv(n_envs=2, obs_mode="history",
                                    action_mode="position"))


def test_action_round_trip_reproduces_the_command():
    """The bridge's action must decode, inside the env, back to the millimetre
    target and the millimetre-per-second caps the bot asked for."""
    env = _env()
    bridge = SimBridge(env)
    obs = env.reset(seed=3)
    cap_v, cap_a = bridge.caps(obs)

    cmds = [Command(1400.0 + 100.0 * i, 300.0 + 80.0 * i,
                    0.4 * cap_v[i], 0.6 * cap_a[i])
            for i in range(env.n_envs)]
    act = bridge.actions(cmds, obs)

    # Positions: exactly the env's own rescale, in sim units, back to mm.
    tx = env._action_low[0] + (act[:, 0] + 1.0) * 0.5 * (
        env._action_high[0] - env._action_low[0])
    ty = env._action_low[1] + (act[:, 1] + 1.0) * 0.5 * (
        env._action_high[1] - env._action_low[1])
    mm_x, mm_y = sim_to_table_mm(tx, ty)
    # 1e-3 mm, not exact: the action array is float32, which is the env's
    # choice and worth about 20 nm of table.
    assert mm_x == pytest.approx([c.x_mm for c in cmds], abs=1e-3)
    assert mm_y == pytest.approx([c.y_mm for c in cmds], abs=1e-3)

    v_frac = 0.05 + (act[:, 2] + 1.0) * 0.5 * 0.95
    a_frac = 0.05 + (act[:, 3] + 1.0) * 0.5 * 0.95
    assert v_frac * cap_v == pytest.approx([c.speed_mm_s for c in cmds], rel=1e-6)
    assert a_frac * cap_a == pytest.approx([c.accel_mm_s2 for c in cmds], rel=1e-6)


def test_bridge_caps_match_the_randomised_machine():
    """The caps come out of the observation, which is what the controller is
    entitled to know -- but they still have to equal the machine's."""
    env = _env(n=8)
    bridge = SimBridge(env)
    obs = env.reset(seed=11)
    cap_v, cap_a = bridge.caps(obs)
    assert cap_v == pytest.approx(env._agent_dyn["max_speed"] * 1000.0, rel=1e-5)
    assert cap_a == pytest.approx(env._agent_dyn["max_accel"] * 1000.0, rel=1e-5)
    assert cap_a.min() < cap_a.max()          # DR actually varied something


def test_bridge_over_asking_gets_the_machine_cap():
    env = _env()
    bridge = SimBridge(env)
    obs = env.reset(seed=5)
    cap_v, cap_a = bridge.caps(obs)
    act = bridge.actions(
        [Command(1500.0, 480.0, 1e9, 1e9)] * env.n_envs, obs)
    v_frac = 0.05 + (act[:, 2] + 1.0) * 0.5 * 0.95
    assert v_frac == pytest.approx(np.ones(env.n_envs))


def test_bridge_reports_are_in_millimetres_and_near_the_truth():
    """Sensing is noisy and late, but not by table-lengths: if the mapping were
    transposed this would be out by hundreds of mm rather than by ten."""
    env = _env(n=6)
    bridge = SimBridge(env)
    obs = env.reset(seed=2)
    for _ in range(40):
        obs, *_ = env.step(np.zeros((env.n_envs, 4), dtype=np.float32))
        bridge.step_index += 1
    reps = bridge.reports(obs)
    truth_x, truth_y = sim_to_table_mm(env.engine.puck_x, env.engine.puck_y)
    for i, rep in enumerate(reps):
        assert len(rep.puck) == len(env.HISTORY_PUCK_LAGS)
        assert rep.puck[0].t_s > rep.puck[1].t_s          # newest FIRST
        assert math.hypot(rep.puck[0].x_mm - truth_x[i],
                          rep.puck[0].y_mm - truth_y[i]) < 60.0
        mx, my = rep.mallet
        assert geom.WS_MIN_X - 1.0 <= mx <= geom.WS_MAX_X + 1.0
        assert geom.WS_MIN_Y - 1.0 <= my <= geom.WS_MAX_Y + 1.0


# ── the bots ─────────────────────────────────────────────────────────────

ALL_BOTS = sorted(BOTS)


@pytest.mark.parametrize("name", ALL_BOTS)
def test_bot_output_stays_inside_the_workspace_and_the_caps(name):
    """Fuzzed over states the tracker really produces, including broken ones.

    A command outside the box is not merely penalised by the env -- on the
    table the firmware clamps it silently, so the bot would be aiming at a
    place it never reaches and no log would say so.
    """
    bot = make_bot(name)
    cfg = bot.cfg
    rng = np.random.default_rng(4)
    for _ in range(400):
        x0 = rng.uniform(-100.0, 2100.0)
        y0 = rng.uniform(-100.0, 1100.0)
        vx, vy = rng.uniform(-9000.0, 9000.0, 2)
        rep = TrackerReport(
            puck=_history(x0, y0, vx, vy),
            mallet=(rng.uniform(geom.WS_MIN_X, geom.WS_MAX_X),
                    rng.uniform(geom.WS_MIN_Y, geom.WS_MAX_Y)),
            opponent=(rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0)),
            t_s=float(rng.uniform(0.0, 100.0)),
        )
        cmd = bot(rep)
        assert geom.WS_MIN_X <= cmd.x_mm <= geom.WS_MAX_X
        assert geom.WS_MIN_Y <= cmd.y_mm <= geom.WS_MAX_Y
        assert 0.0 < cmd.speed_mm_s <= cfg.max_speed_mm_s
        assert 0.0 < cmd.accel_mm_s2 <= cfg.max_accel_mm_s2
        assert math.isfinite(cmd.x_mm) and math.isfinite(cmd.y_mm)


@pytest.mark.parametrize("name", ALL_BOTS)
def test_bot_survives_a_blind_tracker(name):
    """No puck at all is a normal report: the IR ring blinds a patch at table
    centre and the tracker gives up after 150 ms of coasting."""
    bot = make_bot(name)
    cmd = bot(TrackerReport(puck=(), mallet=(geom.HOME_X, geom.HOME_Y), t_s=0.0))
    assert geom.WS_MIN_X <= cmd.x_mm <= geom.WS_MAX_X
    assert geom.WS_MIN_Y <= cmd.y_mm <= geom.WS_MAX_Y


@pytest.mark.parametrize("name", ALL_BOTS)
def test_bot_accepts_the_dict_form(name):
    """The interface the camera workstream will actually hand over."""
    bot = make_bot(name)
    as_dict = {
        "puck": [(1200.0, 480.0, 0.0), (1180.0, 482.0, -0.01),
                 (1160.0, 484.0, -0.02)],
        "mallet": (geom.HOME_X, geom.HOME_Y),
        "opponent": (500.0, 480.0),
        "t_s": 0.0,
    }
    from_dict = bot(as_dict)
    bot.reset()
    from_obj = bot(TrackerReport.coerce(as_dict))
    assert from_dict == from_obj


def test_goalie_stands_where_the_puck_will_arrive():
    bot = make_bot("goalie")
    cmd = bot(TrackerReport(puck=_history(1200.0, 300.0, 4000.0, 0.0),
                            mallet=(geom.HOME_X, geom.HOME_Y), t_s=0.0))
    assert cmd.x_mm == pytest.approx(bot.defend_x)
    assert cmd.y_mm == pytest.approx(300.0, abs=2.0)


def test_goalie_leans_toward_a_drifting_puck_but_does_not_chase_it():
    """Rest is not a fixed point -- but the gain is below 1, so a puck in the
    corner cannot pull the mallet off its own goal."""
    bot = make_bot("goalie")
    centre = (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y) / 2.0
    cmd = bot(TrackerReport(puck=_history(600.0, 900.0, 20.0, 0.0),
                            mallet=(geom.HOME_X, geom.HOME_Y), t_s=0.0))
    assert centre < cmd.y_mm < 900.0
    assert cmd.speed_mm_s <= bot.cfg.idle_speed_mm_s


def test_goalie_hysteresis_does_not_chatter():
    """Engagement is gated on ARRIVAL TIME, with hysteresis: engage inside the
    first horizon, release outside the second, and hold in between."""
    bot = make_bot("goalie")
    mallet = (geom.HOME_X, geom.HOME_Y)
    bot(TrackerReport(puck=_history(1200.0, 300.0, 4000.0, 0.0),
                      mallet=mallet, t_s=0.0))
    assert bot.engaged and bot.last_eta < bot.cfg.engage_horizon_s

    # eta ~ 1.1 s: between the two horizons, so the decision must stick.
    between = _history(1450.0, 300.0, 420.0, 0.0)
    bot(TrackerReport(puck=between, mallet=mallet, t_s=0.1))
    assert bot.cfg.engage_horizon_s < bot.last_eta < bot.cfg.release_horizon_s
    assert bot.engaged, "released between the engage and release horizons"

    bot.reset()
    bot(TrackerReport(puck=between, mallet=mallet, t_s=0.2))
    assert not bot.engaged, "engaged outside the engage horizon"

    bot(TrackerReport(puck=_history(1400.0, 300.0, -50.0, 0.0),
                      mallet=mallet, t_s=0.3))
    assert not bot.engaged


def test_goalie_covers_a_puck_that_is_slow_but_certain():
    """The rule demo_goalie gets wrong. A puck trickling at 100 mm/s is not a
    threat by SPEED and is a goal by geometry once it is close."""
    bot = make_bot("goalie")
    cmd = bot(TrackerReport(puck=_history(1830.0, 620.0, 100.0, 0.0),
                            mallet=(geom.WS_MAX_X - 15.0, 300.0), t_s=0.0))
    assert bot.engaged
    assert cmd.y_mm == pytest.approx(620.0, abs=3.0)


def test_goalie_urgency_scales_with_the_threat():
    """A shot that arrives in 100 ms has to be answered with more of the
    machine than one that arrives in a second."""
    bot = make_bot("goalie")
    far = bot(TrackerReport(puck=_history(1000.0, 200.0, 900.0, 0.0),
                            mallet=(geom.WS_MAX_X - 15.0, 780.0), t_s=0.0))
    bot.reset()
    near = bot(TrackerReport(puck=_history(1750.0, 200.0, 7000.0, 0.0),
                             mallet=(geom.WS_MAX_X - 15.0, 780.0), t_s=0.0))
    assert near.accel_mm_s2 > far.accel_mm_s2
    assert near.speed_mm_s > far.speed_mm_s


def test_striker_attacks_a_slow_puck_in_reach():
    """The commanded target must be up-table of the puck: the mallet drives
    THROUGH it toward the opponent's goal, not up to it."""
    bot = make_bot("striker")
    puck_x, puck_y = 1600.0, 480.0
    cmd = bot(TrackerReport(puck=_history(puck_x, puck_y, -50.0, 0.0),
                            mallet=(geom.WS_MAX_X - 15.0, 480.0),
                            opponent=(300.0, 480.0), t_s=0.0))
    assert cmd.x_mm < puck_x, "did not commit to a strike"
    assert cmd.speed_mm_s == bot.cfg.strike_speed_mm_s


def test_striker_defends_a_fast_puck_instead_of_attacking_it():
    bot = make_bot("striker")
    cmd = bot(TrackerReport(puck=_history(1500.0, 480.0, 6000.0, 0.0),
                            mallet=(geom.HOME_X, geom.HOME_Y),
                            opponent=(300.0, 480.0), t_s=0.0))
    assert cmd.x_mm == pytest.approx(bot.defend_x)


def test_striker_holds_its_swing_for_the_commit_window():
    """Re-solving a swing every 10 ms taps the puck instead of striking it."""
    bot = make_bot("striker")
    rep = TrackerReport(puck=_history(1600.0, 480.0, -50.0, 0.0),
                        mallet=(geom.WS_MAX_X - 15.0, 480.0),
                        opponent=(300.0, 480.0), t_s=0.0)
    first = bot(rep)
    assert first.x_mm < 1600.0
    held = bot(TrackerReport(puck=_history(1500.0, 700.0, 8000.0, 0.0),
                             mallet=(geom.WS_MAX_X - 15.0, 480.0),
                             opponent=(300.0, 480.0), t_s=0.05))
    assert held == first
    later = bot(TrackerReport(puck=_history(1500.0, 700.0, 8000.0, 0.0),
                              mallet=(geom.WS_MAX_X - 15.0, 480.0),
                              opponent=(300.0, 480.0),
                              t_s=bot.cfg.commit_s + 0.05))
    assert later != first


def test_strike_aims_away_from_the_opponent_mallet():
    """The mouth is 380 mm and a mallet is 100 -- there is nearly always an
    open side, and the centre is where a goalkeeper stands."""
    bot = make_bot("striker")
    centre = (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y) / 2.0
    low = bot.aim_point(TrackerReport(mallet=(0.0, 0.0),
                                      opponent=(200.0, centre - 300.0)))[1]
    high = bot.aim_point(TrackerReport(mallet=(0.0, 0.0),
                                       opponent=(200.0, centre + 300.0)))[1]
    assert low > centre and high < centre
    assert bot.aim_point(TrackerReport(mallet=(0.0, 0.0)))[1] == centre


def test_contact_point_is_always_on_the_goal_side_of_the_puck():
    """The structural safety of both strikers: a mistimed swing degrades into
    a block, because the mallet is never up-table of the puck it is aiming
    through."""
    bot = make_bot("intercept")
    rng = np.random.default_rng(9)
    for _ in range(200):
        px = rng.uniform(geom.WS_MIN_X, geom.WS_MAX_X)
        py = rng.uniform(geom.WS_MIN_Y, geom.WS_MAX_Y)
        aim = bot.aim_point(TrackerReport(mallet=(0.0, 0.0)))
        cmd = bot.strike_command(px, py, aim, TrackerReport(mallet=(px, py)))
        if cmd is None:
            continue
        # Contact sits one reach back along the aim, i.e. at higher grid x.
        reach = bot.contact_distance_mm()
        ux = (aim[0] - px) / math.hypot(aim[0] - px, aim[1] - py)
        assert px - ux * reach > px


def test_intercept_meets_the_puck_before_the_line():
    """Its defining behaviour: an incoming puck it has time for is met
    up-table and hit back, not blocked on the line."""
    bot = make_bot("intercept")
    cmd = bot(TrackerReport(puck=_history(1500.0, 480.0, 500.0, 0.0),
                            mallet=(geom.WS_MAX_X - 15.0, 480.0),
                            opponent=(300.0, 480.0), t_s=0.0))
    assert cmd.x_mm < bot.defend_x - 100.0
    assert cmd.speed_mm_s == bot.cfg.strike_speed_mm_s


def test_intercept_stays_home_against_a_puck_it_has_no_time_for():
    """The same shot arriving faster is not an opportunity. Leaving the line
    with 0.4 s of swing to run and 0.5 s of puck left is how a striker turns a
    save into a goal."""
    bot = make_bot("intercept")
    cmd = bot(TrackerReport(puck=_history(1300.0, 480.0, 1200.0, 0.0),
                            mallet=(geom.WS_MAX_X - 15.0, 480.0),
                            opponent=(300.0, 480.0), t_s=0.0))
    assert cmd.x_mm == pytest.approx(bot.defend_x)


def test_intercept_will_not_swing_when_it_cannot_recover():
    """A swing ends with the mallet a follow-through up-table of the contact
    point. Against a puck that is already nearly there, that is the open net,
    so it has to stay home."""
    bot = make_bot("intercept")
    cmd = bot(TrackerReport(puck=_history(1850.0, 700.0, 9000.0, 0.0),
                            mallet=(geom.WS_MIN_X + 5.0, geom.WS_MIN_Y + 5.0),
                            opponent=(300.0, 480.0), t_s=0.0))
    assert cmd.x_mm == pytest.approx(bot.defend_x)


def test_make_bot_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        make_bot("nope")


# ── end to end ───────────────────────────────────────────────────────────

def _play(bot_name, opponent, games=6, seconds=12.0, seed=17):
    env = _env(n=games, opponent=opponent, max_episode_time=seconds + 1.0)
    bridge = SimBridge(env)
    cfg = BotConfig(mallet_radius_mm=env.table_config.paddle_radius * 1000.0)
    bots = [make_bot(bot_name, cfg) for _ in range(games)]
    obs = env.reset(seed=seed)
    bridge.reset()
    info = {}
    over = []
    for _ in range(int(seconds / env.action_dt)):
        cmds = [b(r) for b, r in zip(bots, bridge.reports(obs))]
        obs, _, _, _, info = env.step(bridge.actions(cmds, obs))
        over.append(info["ws_overshoot"].max())
        bridge.step_index += 1
    return info["score_agent"], info["score_opponent"], float(np.max(over))


def test_goalie_concedes_almost_nothing_against_random():
    """The integration test that matters: the whole chain -- camera model,
    latency, history observation, mm conversion, bot, action conversion,
    firmware motion law -- has to hold the net."""
    gf, ga, _ = _play("goalie", "random")
    assert ga.mean() <= 0.2, f"conceded {ga} in 12 s games"


def test_bots_never_command_outside_the_reachable_box_in_play():
    """`ws_overshoot` is the env charging for an unreachable command. A
    heuristic that clamps its own targets should never pay it."""
    for name in ALL_BOTS:
        _, _, overshoot = _play(name, "follow", games=3, seconds=6.0)
        # Not exactly zero: a bot standing ON the boundary, which the goalies
        # do constantly, is float32-rounded a few nanometres past it by the
        # action encoding. A real overshoot is millimetres.
        assert overshoot < 1e-6, f"{name} commanded outside the workspace"


def test_a_striker_actually_scores():
    """Otherwise the attacking code is decoration."""
    gf, _, _ = _play("striker", "random", games=8, seconds=20.0)
    assert gf.sum() >= 3, f"striker scored {gf.sum()} in 8x20 s"


def test_evaluation_is_reproducible():
    """Two identical runs must agree exactly, or the tournament's shared
    fixtures mean nothing and a two-goal gap is the seed."""
    a = _play("intercept", "random", games=4, seconds=8.0)
    b = _play("intercept", "random", games=4, seconds=8.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
