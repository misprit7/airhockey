"""Judging a candidate under table conditions: the table human-proxy tiers,
the evaluators' --sim-profile, and the playability gates.

Pins that all of it is opt-in (the club and elite tiers keep their draws and
their four modes; an evaluator without the flag builds the historical env and
writes the historical report), that the table tiers do the three things the
people did and the older tiers do not (handed banks, a timed counter-swing at
a moving puck, a serve after a goal), and that one gate function gives the
same answer for a simulated game and for a table log of the same motion.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airhockey import playability
from airhockey.human_proxy import (CLUB, ELITE, READY, TABLE, TABLE_PLUS, TIERS, VOLLEY, HumanProxy,
                                   command_env, serve_env)
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.physics import TableConfig
from airhockey.table_fidelity import LiftTally, profile_options, table_fidelity

BIN = Path(__file__).resolve().parents[1] / "bin"
CFG = TableConfig()


def _view(n, puck, vel=(0.0, 0.0), own=(0.5, 0.15), rival=(0.5, 1.8)):
    view = np.zeros((n, 12))
    view[:, 0:2], view[:, 2:4] = puck, vel
    view[:, 4:6], view[:, 8:10] = own, rival
    return view


class _StillRobot:
    history, shot_conditioned, environment_options = 1, True, {}

    def act(self, observation, stochastic=False):
        return np.zeros((len(observation), 6), np.float32)


# ------------------------------------------------------------- proxy tiers
def test_older_tiers_keep_their_draws_modes_and_commands():
    assert TIERS["club"] is CLUB and TIERS["elite"] is ELITE
    for tier in (CLUB, ELITE):
        assert tier.bank_x0_rail_fraction is None and tier.volley_probability is None and tier.serve_x_m is None
    digest = hashlib.sha256()
    for tier in ("club", "elite"):
        p = HumanProxy(6, tier, seed=17)
        p.reset(_view(6, puck=(0.4, 1.5)))
        assert p.mode_names == ("ready", "approach", "hold", "strike") and not p.volleying
        assert p.mode_ticks.shape == (6, 4) and not hasattr(p, "plan_serve")
        for key in ("plan_route", "plan_aim", "plan_speed", "plan_aim_error", "plan_feint"):
            digest.update(np.ascontiguousarray(getattr(p, key)).tobytes())
        # A shot coming in, then a puck left on its half: defence, approach and strike.
        for k in range(40):
            out = p.act(_view(6, puck=(0.4, 1.5 - 0.1 * min(k, 12)), vel=(0.0, -5.0 if k < 12 else 0.0)))
            digest.update(np.ascontiguousarray(np.stack(out)).tobytes())
        digest.update(p.rng.random(4).tobytes())       # the same number of draws consumed
    assert digest.hexdigest() == _OLDER_TIER_DIGEST


# sha256 over the plans, 40 ticks of commands and the generator state above,
# recorded from the proxy before the table tiers were added.
_OLDER_TIER_DIGEST = "c3f9947917c65772601727b5dff819ac5c97daa57bd7783d704889d60cb2dae6"


def test_table_tiers_plan_the_rail_itself():
    p = HumanProxy(64, "table", seed=3)
    p.reset(_view(64, puck=(0.5, 1.5)))
    banks = p.plan_route != 0
    assert abs(banks.mean() - TABLE.bank_fraction) < 0.02
    assert abs((p.plan_route[banks] == -1).mean() - TABLE.bank_x0_rail_fraction) < 0.02
    assert TABLE_PLUS.bank_fraction < TABLE.bank_fraction and TABLE_PLUS.release[1] > TABLE.release[1]
    # A puck by the x = width rail: the geometry prefers that rail, the plan still says x = 0.
    one = HumanProxy(1, replace(TABLE, bank_fraction=1.0, bank_x0_rail_fraction=1.0, feint_probability=0.0), seed=1)
    slow = _view(1, puck=(0.70, 0.60), own=(0.6, 0.2))
    one.reset(slow)
    for _ in range(120):
        one.act(slow)
        if one.strikes.sum() or one.mode[0] == 3:
            break
    assert one.mode[0] == 3 and one.strike_route[0] == -1 and one.strike_direction[0, 0] < 0


def _incoming(proxy, speed, ticks, start=1.75, x=0.45):
    """A puck first at rest by the robot, then struck straight at the person; per-tick records."""
    n = proxy.n
    rest = _view(n, puck=(x, start))
    proxy.reset(rest)
    for _ in range(12):
        proxy.act(rest)
    rows = []
    for k in range(ticks):
        y = start - speed * 0.02 * (k + 1)
        own = proxy.last_target.copy()
        out = proxy.act(_view(n, puck=(x, max(y, 0.05)), vel=(0.0, -speed), own=own))
        rows.append(dict(mode=proxy.mode.copy(), y=y, target=proxy.last_target.copy(), speed=out[2].copy(),
                         swing=proxy.volley_swing.copy(), eta=proxy.volley_eta.copy()))
    return rows


def test_volley_waits_behind_the_line_and_swings_as_the_puck_arrives():
    tier = replace(TABLE, volley_probability=((np.inf, 1.0),), volley_contact_y_m=(0.38, 0.38),
                   volley_mallet_speed_m_s=(3.0, 3.0), reaction_s=(0.15, 0.15), defense_noise_deg=(0.0, 0.0))
    p = HumanProxy(4, tier, seed=5)
    rows = _incoming(p, speed=2.5, ticks=60)
    modes = np.array([r["mode"] for r in rows])
    first = int(np.argmax(modes[:, 0] == VOLLEY))
    # Seen a reaction time late, and decided on the second tick it is seen coming.
    assert modes[first, 0] == VOLLEY and abs(first * 0.02 - (0.15 + 0.02)) <= 0.021
    assert (p.volley_attempts == 1).all() and p.mode_names[VOLLEY] == "volley"
    line = p.VOLLEY_LINES[np.abs(p.VOLLEY_LINES - 0.38).argmin()]
    swinging = np.array([r["swing"][0] for r in rows]) & (modes[:, 0] == VOLLEY)
    assert swinging.any() and not swinging[first]
    start = int(np.argmax(swinging))
    # Before the swing it holds the wind-up point behind the contact line, then drives through at its speed.
    assert rows[start - 1]["target"][0, 1] < line - p.contact
    assert rows[start]["target"][0, 1] > line - p.contact and rows[start]["speed"][0] == pytest.approx(3.0)
    # The swing starts when the puck is a swing-time from the line (within a tick or two of flight).
    remaining = (rows[start]["y"] - line) / 2.5
    assert 0.0 < remaining < 0.20
    # Fast pucks leave no time: no counter-swing, the ordinary block instead.
    fast = HumanProxy(4, tier, seed=5)
    assert not any((r["mode"] == VOLLEY).any() for r in _incoming(fast, speed=9.0, ticks=12))
    assert (fast.volley_attempts == 0).all()


def test_volley_returns_a_moving_puck_in_the_engine():
    n = 6
    tier = replace(TABLE_PLUS, volley_probability=((np.inf, 1.0),))
    env = NeuralTrainingEnv(n, stage=3, seed=3, games=True, report_sensing=True, continuous_rallies=True,
                            shot_conditioned=True)
    env.reset(seed=3, opponent="human_proxy")
    proxy = HumanProxy(n, tier, seed=4, config=env.cfg)
    command_env(env, proxy, reset=True)
    e = env.engine
    # The puck at rest on the centre line first (nothing coming), then sent at the person.
    for k in range(95):
        if k < 25:
            e.puck_x[:], e.puck_y[:] = np.linspace(0.35, 0.65, n), 0.95
            e.puck_vx[:], e.puck_vy[:] = 0.0, 0.0
        if k == 25:
            e.puck_vy[:] = np.linspace(1.8, 3.0, n)
        command_env(env, proxy)
        env.step(np.zeros((n, 6), np.float32))
    assert proxy.volley_attempts.sum() >= n // 2
    assert proxy.volleys.sum() >= max(1, proxy.volley_attempts.sum() // 2)
    assert proxy.mode_ticks[:, VOLLEY].sum() > 0


def test_serve_puts_the_puck_down_at_rest_for_table_tiers_only():
    n = 4
    env = NeuralTrainingEnv(n, stage=3, seed=2, games=True, report_sensing=True, continuous_rallies=True,
                            shot_conditioned=True)
    env.reset(seed=2, opponent="human_proxy")
    scored = np.array([True, False, True, False])
    club = HumanProxy(n, "club", seed=1, config=env.cfg)
    command_env(env, club, reset=True)
    before = (env.engine.puck_x.copy(), env.engine.puck_y.copy(), env.engine.puck_vx.copy())
    serve_env(env, club, scored)
    np.testing.assert_array_equal(before[0], env.engine.puck_x)
    np.testing.assert_array_equal(before[2], env.engine.puck_vx)
    table = HumanProxy(n, "table", seed=1, config=env.cfg)
    command_env(env, table, reset=True)
    serve_env(env, table, scored)
    e = env.engine
    assert (e.puck_vx[scored] == 0).all() and (e.puck_vy[scored] == 0).all()
    assert ((e.puck_x[scored] >= TABLE.serve_x_m[0]) & (e.puck_x[scored] <= TABLE.serve_x_m[1])).all()
    assert ((e.puck_y[scored] >= TABLE.serve_y_m[0]) & (e.puck_y[scored] <= TABLE.serve_y_m[1])).all()
    np.testing.assert_array_equal(e.puck_y[~scored], before[1][~scored])
    assert (table.cooldown[scored] >= TABLE.serve_wait_s[0]).all() and (table.cooldown[~scored] == 0).all()
    assert table.serves.tolist() == [1, 0, 1, 0]
    # The same serves for every candidate on a seed.
    again = HumanProxy(n, "table", seed=1, config=env.cfg)
    again.reset(_view(n, puck=(0.5, 1.5)))
    np.testing.assert_array_equal(again.plan_serve, table.plan_serve)


# --------------------------------------------------------- profile + lifts
def test_profile_options_is_the_flag_to_kwargs_map():
    assert profile_options(None) == {}
    assert profile_options("table-20261029") == table_fidelity("table-20261029")
    parts = profile_options("table-20261029:contact,lift")
    assert set(parts) == {"speed_restitution", "paddle_tangential_range", "mallet_lift"}
    with pytest.raises(ValueError):
        profile_options("table-20261029:contact,heat")
    with pytest.raises(ValueError):
        profile_options("no-such-table")
    json.dumps(profile_options("table-20261029"))


def test_lift_tally_counts_goals_within_a_second_of_a_lift():
    engine = SimpleNamespace(mallet_lift=object(), lift_events=np.zeros((2, 3), dtype=np.int64))
    tally = LiftTally(engine)
    none = np.zeros(3, int)
    engine.lift_events[0, 0] += 1          # under the candidate's mallet, game 0
    engine.lift_events[1, 2] += 1          # under the far mallet, game 2
    tally.after_step(none, none)
    for _ in range(10):
        tally.after_step(none, none)
    tally.after_step(np.array([1, 1, 0]), np.array([0, 0, 1]))       # 0.22 s later
    for _ in range(60):
        tally.after_step(none, none)
    engine.lift_events[0, 1] += 1
    tally.after_step(none, none)
    for _ in range(55):
        tally.after_step(none, none)
    tally.after_step(np.array([0, 1, 0]), none)                      # 1.12 s after that lift: not counted
    assert tally.report() == dict(enabled=True, window_s=1.0, lift_events=2, opponent_lift_events=1,
                                  goals_conceded_through_lift=1, goals_scored_through_opponent_lift=1)


def test_evaluators_default_to_no_profile():
    sys.path.insert(0, str(BIN))
    import eval_fast_defense
    import eval_human_proxy
    import eval_neural_fringe
    import eval_neural_player
    import eval_slow_control
    for function in (eval_fast_defense.evaluate, eval_human_proxy.evaluate, eval_human_proxy.play_tier,
                     eval_neural_fringe.evaluate, eval_neural_player.skills, eval_neural_player.games,
                     eval_slow_control.evaluate):
        assert inspect.signature(function).parameters["sim_profile"].default is None
    assert eval_human_proxy.TIER_GROUPS["both"] == ("club", "elite")
    assert {"club": 101, "elite": 202}.items() <= eval_human_proxy.TIER_SEED.items()
    assert set(eval_human_proxy.TABLE_TARGETS) == {"table", "table_plus"}


def test_proxy_benchmark_under_the_profile_reports_it_and_the_table_checks():
    sys.path.insert(0, str(BIN))
    import eval_human_proxy as hp
    plain = hp.play_tier(_StillRobot(), "club", games=3, seconds=3.0, seed=9, options={})
    assert "lift" not in plain and "table" not in plain["calibration"]
    assert set(plain["calibration"]["mode_fraction"]) == {"ready", "approach", "hold", "strike"}
    seen = []

    def observe(env):
        assert env.engine.mallet_lift is not None and env.engine.speed_restitution is not None
        assert env.base._table_sensing is not None and env.base._command_delay_range == (0.009, 0.019)
        return lambda: seen.append(1)
    result = hp.play_tier(_StillRobot(), "table", games=3, seconds=6.0, seed=9, options={},
                          sim_profile="table-20261029", observe=observe)
    assert len(seen) == round(6.0 / 0.02)
    assert result["lift"]["enabled"] and result["lift"]["goals_against"] == result["goals_against"]
    assert result["lift"]["goals_conceded_through_lift"] <= result["goals_against"]
    c = result["calibration"]
    assert set(c["mode_fraction"]) == {"ready", "approach", "hold", "strike", "volley"}
    targets = {k for k, v in hp.TABLE_TARGETS["table"].items() if isinstance(v, list)}
    assert targets <= set(c["checks"]) and targets <= set(c["table"])
    assert c["table"]["serves"] == result["goals_for"] + result["goals_against"]


# ------------------------------------------------------------- playability
def _trace(ticks=400):
    t = (np.arange(ticks) + 1) * 0.02
    zeros = np.zeros((ticks, 2))
    return dict(t=t, policy=np.ones(ticks, bool), puck=np.tile([0.5, 1.6], (ticks, 1)), puck_v=zeros.copy(),
                rival=np.zeros(ticks), cap=np.full(ticks, 8.0), target=np.tile([0.5, 0.3], (ticks, 1)),
                mallet=np.tile([0.5, 0.3], (ticks, 1)), mallet_v=zeros.copy(), back_y=0.08)


def test_gates_on_a_scripted_trace():
    tr = _trace()
    # Ticks 100-139: the puck comes over; cap 90 on half of them, 50 on the rest.
    tr["puck"][100:140, 1] = 0.5
    tr["cap"][100:120], tr["cap"][120:140] = 90.0, 50.0
    # Far half but busy (rival moving): cap 60, not idle.
    tr["rival"][200:220] = 2.0
    tr["cap"][200:220] = 60.0
    # One reversal at 2 m/s within 100 ms, and one at 4 m/s.
    tr["mallet_v"][300:303, 0], tr["mallet_v"][303:306, 0] = 2.0, -2.0
    tr["mallet_v"][340:343, 0], tr["mallet_v"][343:346, 0] = 4.0, -4.0
    # Parked on the back edge for 40 ticks, and the idle command creeping 1 mm a tick for 50.
    tr["mallet"][360:400, 1] = 0.09
    tr["target"][20:70, 0] += 0.001 * np.arange(50)
    g = playability.gates([playability.measure(tr)])
    assert g["policy_minutes"] == pytest.approx(400 * 0.02 / 60)
    assert g["idle_cap_p50_m_s2"] == 8.0 and g["idle_ticks"] == 400 - 40 - 20
    assert g["far_half_cap_over_40_share"] == pytest.approx(20 / 360) and g["far_half_cap_over_80_share"] == 0
    assert g["all_ticks_cap_over_80_share"] == pytest.approx(20 / 400)
    assert g["reversals_1p5_per_min"] == pytest.approx(2 / (8 / 60))
    assert g["reversals_3_per_min"] == pytest.approx(1 / (8 / 60))
    # dv of 4 m/s (200 m/s^2) and 8 m/s in one tick, plus four steps of 2 or 4 m/s from/to rest.
    assert g["accel_duty_over_60_share"] == pytest.approx(6 / 399)
    assert g["back_edge_dwell_share"] == pytest.approx(40 / 400)
    assert g["idle_command_change_p50_mm"] == 0.0 and g["idle_command_change_p90_mm"] == pytest.approx(1.0)
    assert 0 < g["modeled_load_settle"] < 1 and g["modeled_load_peak"] >= 0


def test_incoming_shot_excursion_and_contact_speed():
    tr = _trace(200)
    # A 6 m/s shot crossing the centre line at tick 100; the mallet lunges 0.25 m forward at 3 m/s.
    for k in range(90, 110):
        tr["puck"][k] = [0.5, playability.CENTRE_Y + 0.06 - 0.12 * (k - 100)]
        tr["puck_v"][k] = [0.0, -6.0]
    tr["puck"][110:] = [0.5, 1.6]
    for k in range(100, 105):
        tr["mallet"][k:, 1] = 0.3 + 0.05 * (k - 99)
        tr["mallet_v"][k] = [0.0, 2.5]
    g = playability.gates([playability.measure(tr)])
    assert g["incoming_shots"] == 1
    assert g["incoming_forward_excursion_mean_mm"] == pytest.approx(250.0)
    assert g["incoming_contact_speed_mean_m_s"] == pytest.approx(2.5)
    slow = _trace(200)
    for k in range(90, 110):
        slow["puck"][k] = [0.5, playability.CENTRE_Y + 0.03 - 0.06 * (k - 100)]
        slow["puck_v"][k] = [0.0, -3.0]
    assert playability.gates([playability.measure(slow)])["incoming_shots"] == 0


def test_table_log_and_sim_trace_of_the_same_motion_score_the_same(tmp_path):
    ticks = 300
    rng = np.random.default_rng(0)
    # A motion in sim units: the mallet shuttles across its box, the puck sits on the far half.
    t = (np.arange(ticks) + 1) * 0.02
    vx = 2.5 * np.sign(np.sin(2 * np.pi * (t - 0.01) / 0.32))
    x = 0.3 + np.cumsum(vx) * 0.02
    cap = rng.choice([8.0, 50.0, 95.0], ticks)
    sim = dict(t=t, policy=np.ones(ticks, bool), puck=playability.metres(np.tile([0.4, 1.5], (ticks, 1))),
               puck_v=np.zeros((ticks, 2)), rival=np.zeros(ticks), cap=cap,
               target=playability.metres(np.column_stack((x, np.full(ticks, 0.3)))),
               mallet=playability.metres(np.column_stack((x, np.full(ticks, 0.3)))),
               mallet_v=playability.metres(np.column_stack((vx, np.zeros(ticks)))), back_y=0.08)
    from airhockey.dynamics import sim_to_table_mm
    mx, my = sim_to_table_mm(x, np.full(ticks, 0.3))
    px, py = sim_to_table_mm(0.4, 1.5)
    scale = playability.SCALE * 1000
    with open(tmp_path / "s.ticks.csv", "w") as handle:
        handle.write("t_cam,flags,puck_x,puck_y,puck_age_ms,opp_x,opp_y,cmd_x,cmd_y,cmd_accel,ctl_mallet_x,ctl_mallet_y,neural_obs\n")
        # A paused tick first: seen, but not the policy's.
        handle.write(f"0.0,puck_hold,{px:.4f},{py:.4f},0.0,300,480,1500,480,2000,1500,480,\n")
        for k in range(ticks):
            loads = "|".join(["0"] * 21 + ["0.4"] * 8)
            handle.write(f"{t[k]:.4f},,{px:.4f},{py:.4f},0.0,300,480,{mx[k]:.5f},{my[k]:.5f},{cap[k] * 1000:.1f},"
                         f"{mx[k]:.5f},{my[k]:.5f},{loads}\n")
    with open(tmp_path / "s.replay.jsonl", "w") as handle:
        handle.write(json.dumps(dict(type="meta", policy="neural:test", workspace_bounds_mm=[1200.0, 1937.5, 61.4, 904.5]),
                                separators=(",", ":")) + "\n")
        for k in range(ticks):
            handle.write(json.dumps(dict(type="controller", t=float(t[k]), sample_monotonic=100.0 + float(t[k]),
                                         x=float(mx[k]), y=float(my[k]), vx=0.0, vy=float(vx[k] * scale[0])),
                                    separators=(",", ":")) + "\n")
    table = playability.table_trace("s", tmp_path)
    assert table["policy"].sum() == ticks and table["policy_name"] == "neural:test"
    np.testing.assert_allclose(table["mallet"][1:], sim["mallet"], atol=1e-6)
    np.testing.assert_allclose(table["mallet_v"][1:], sim["mallet_v"], atol=1e-9)
    a = playability.gates([playability.measure(sim, initial_load=0.4)])
    b = playability.gates([playability.measure(table)])
    assert b.pop("measured_load_peak") == pytest.approx(0.4)
    assert a["idle_ticks"] == ticks and a["reversals_1p5_per_min"] > 0 and a["all_ticks_cap_over_80_share"] > 0.2
    for key, value in a.items():
        if value is None:
            assert b[key] is None, key
        else:
            assert b[key] == pytest.approx(value, rel=1e-6, abs=1e-6), key
    text = playability.format_table([("sim", a), ("table", b)])
    assert "idle cap p50" in text and "reversals >= 1.5 m/s" in text


def test_sim_recorder_reads_one_trace_per_game():
    n = 3
    env = NeuralTrainingEnv(n, stage=3, seed=1, games=True, report_sensing=True, continuous_rallies=True,
                            shot_conditioned=True)
    env.reset(seed=1, opponent="human_proxy")
    proxy = HumanProxy(n, "club", seed=2, config=env.cfg)
    command_env(env, proxy, reset=True)
    recorder = playability.SimRecorder(env)
    action = np.zeros((n, 6), np.float32)
    action[:, 0] = 0.5
    for _ in range(30):
        command_env(env, proxy)
        env.step(action)
        recorder()
    traces = recorder.traces()
    assert len(traces) == n and traces[0]["puck"].shape == (30, 2)
    dyn = env.base._agent_dyn
    np.testing.assert_allclose(traces[1]["mallet"][-1], playability.metres([dyn["x"][1], dyn["y"][1]]))
    assert traces[0]["back_y"] == pytest.approx(env.decoder.low[1] * playability.SCALE[1])
    assert (traces[0]["cap"] > 0).all() and (traces[0]["cap"] <= dyn["max_accel"][0] + 1e-9).all()
    g = playability.gates([playability.measure(tr, initial_load=0.4, thermal_gain=1.3) for tr in traces])
    assert g["traces"] == n and g["policy_minutes"] == pytest.approx(n * 30 * 0.02 / 60)
