"""Opt-in table fidelity: off by default and unchanged, each model as measured."""
import json

import numpy as np
import pytest

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
from airhockey.batch_physics import BatchPhysicsEngine, MalletLift, SpeedRestitution
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.perception import TableDropouts, TableSensing
from airhockey.table_fidelity import PROFILES, table_fidelity

DT = 1 / 400


def _rollout(**options):
    env = NeuralTrainingEnv(16, seed=7, stage=5, report_sensing=True, shot_conditioned=True,
                            accel=100., **options)
    env.reset(seed=7)
    rng = np.random.default_rng(1)
    total = np.zeros(4)
    keys = set()
    for _ in range(150):
        env.set_opponent_action(rng.uniform(-1, 1, (16, 6)))
        obs, reward, term, trunc, info = env.step(rng.uniform(-1, 1, (16, 6)))
        keys |= set(info)
        total += [obs.astype(np.float64).sum(), reward.astype(np.float64).sum(),
                  env.opponent_obs().astype(np.float64).sum(),
                  env.engine.puck_x.sum() + env.engine.puck_y.sum()]
        done = term | trunc
        if done.any():
            env.reset(mask=done)
    return env, total, keys


# ── defaults ────────────────────────────────────────────────────────────

def test_defaults_leave_every_model_off_and_the_rollout_unchanged():
    env, total, keys = _rollout()
    assert env.engine.speed_restitution is None and env.engine.mallet_lift is None
    assert env.base._table_sensing is None and env.base._perception_far is None
    assert env.base._command_delay_env is None and env.base._late_command_p is None
    assert env.base._cam_ring.shape[2] == 9
    assert "lift_events" not in keys
    # Recorded on the commit before these models existed (95fd39b), same
    # seeds: observations, rewards, the far view and the puck all agree.
    assert np.allclose(total, [40536.61517726071, -143.20340720345848,
                               41362.318082097234, 3217.0522959707273], rtol=1e-9, atol=0)


def test_profile_changes_the_rollout_and_reports_lifts():
    env, total, keys = _rollout(**table_fidelity())
    assert {"lift_events", "opponent_lift_events"} <= keys
    assert np.isfinite(total).all()
    assert not np.isclose(total[0], 40536.61517726071)
    assert env.base.command_delay_s == 0.015          # what the decoder plans with
    assert env.base._cam_ring.shape[2] == 14


# ── the helper ──────────────────────────────────────────────────────────

def test_helper_returns_the_named_profile_and_its_parts():
    full = table_fidelity("table-20261029")
    assert set(full) == {"speed_restitution", "paddle_tangential_range", "mallet_lift",
                         "table_sensing", "command_delay_range", "late_command_probability"}
    assert full == table_fidelity()
    json.dumps(full)
    assert set(table_fidelity(lift=False, sensing=False, actuator=False)) == {
        "speed_restitution", "paddle_tangential_range"}
    assert table_fidelity(contact=False, lift=False, sensing=False, actuator=False) == {}
    tuned = table_fidelity(mallet_lift={"off_probability": 0.0}, late_command_probability=None)
    assert tuned["mallet_lift"] == {"off_probability": 0.0}
    assert "late_command_probability" not in tuned
    assert PROFILES["table-20261029"]["lift"] == {"mallet_lift": True}   # overrides do not leak
    with pytest.raises(ValueError):
        table_fidelity("table-19990101")
    with pytest.raises(ValueError):
        table_fidelity(fuzz_p=1.0)


# ── (a) speed-dependent restitution ─────────────────────────────────────

def _head_on(engine, closing, body="agent"):
    """Puck moving -y at `closing` into a paddle at rest; returns outgoing vy."""
    n = engine.n_envs
    r = engine.config.puck_radius + engine.config.paddle_radius
    engine.puck_x[:], engine.puck_y[:] = 0.5, 0.5
    engine.puck_vx[:], engine.puck_vy[:] = 0.0, -closing
    for axis, value in (("x", 0.5), ("y", 0.5 - r + 0.001), ("vx", 0.0), ("vy", 0.0)):
        getattr(engine, f"paddle_{'agent' if body == 'agent' else 'opp'}_{axis}")[:] = value
    prefix = "paddle_agent" if body == "agent" else "paddle_opp"
    engine._collide_paddle(*(getattr(engine, f"{prefix}_{k}") for k in ("x", "y", "vx", "vy")))
    assert n == len(engine.puck_vy)
    return engine.puck_vy.copy()


def test_restitution_falls_with_closing_speed():
    e = BatchPhysicsEngine(4)
    e.enable_speed_restitution(scatter=0.0)
    assert np.allclose(_head_on(e, 1.0), 0.505 * 1.0)
    assert np.allclose(_head_on(e, 3.0), 0.505 * 3.0)
    assert np.allclose(_head_on(e, 6.0), (0.505 + 0.327) / 2 * 6.0)
    assert np.allclose(_head_on(e, 9.0), 0.327 * 9.0)
    assert np.allclose(_head_on(e, 11.0), 0.327 * 11.0)


def test_restitution_is_the_robots_only():
    e = BatchPhysicsEngine(4)
    e.enable_speed_restitution(scatter=0.0)
    e.opponent_robot[:] = [True, True, False, False]
    out = _head_on(e, 9.0, body="opponent")
    assert np.allclose(out[:2], 0.327 * 9.0)                       # a copy of the machine
    assert np.allclose(out[2:], e.config.paddle_restitution * 9.0)  # a hand-held mallet


def test_restitution_draws_cover_the_sessions_and_scatter_is_clipped():
    e = BatchPhysicsEngine(2000, domain_randomize=True)
    e.enable_speed_restitution()
    e.reset(np.random.default_rng(0))
    assert 0.42 <= e.paddle_e_lo.min() and e.paddle_e_lo.max() <= 0.58
    assert 0.27 <= e.paddle_e_hi.min() and e.paddle_e_hi.max() <= 0.42
    assert (e.paddle_e_hi <= e.paddle_e_lo).all()
    ratio = _head_on(e, 9.0) / 9.0
    assert 0.05 <= ratio.min() and ratio.max() <= 0.90
    assert 0.08 < (ratio - e.paddle_e_hi).std() < 0.12
    with pytest.raises(ValueError):
        BatchPhysicsEngine(2).enable_speed_restitution(e_lo=(0.6, 0.4))


# ── (b) mallet lift ─────────────────────────────────────────────────────

CERTAIN = dict(nominal=(1.0, 10.0, 0.0, 0.3, 0.0, 0.0))    # p100 = 1 from 10 m/s^2 up


def _lunge(engine, *, puck_speed=7.0, braking=100.0, offset=0.0, body="agent", steps=120):
    """A puck down the table into a mallet lunging at it at 4 m/s and
    braking at `braking` m/s^2 (0 = a mallet standing still)."""
    n = engine.n_envs
    sign = 1.0 if body == "agent" else -1.0     # the opponent's seat, mirrored
    mid = 1.0
    engine.puck_x[:] = 0.5 + offset
    engine.puck_y[:] = mid - sign * 0.30
    engine.puck_vx[:] = 0.0
    engine.puck_vy[:] = -sign * puck_speed
    y, v = mid - sign * 0.62, sign * (4.0 if braking else 0.0)
    update = engine.update_paddle_agent if body == "agent" else engine.update_paddle_opponent
    other = engine.update_paddle_opponent if body == "agent" else engine.update_paddle_agent
    for axis, value in (("x", 0.5), ("y", y)):
        getattr(engine, f"paddle_{'agent' if body == 'agent' else 'opp'}_{axis}")[:] = value
        getattr(engine, f"paddle_{'opp' if body == 'agent' else 'agent'}_{axis}")[:] = 0.05 if axis == "x" else mid + sign * 0.9
    for _ in range(steps):
        v = sign * max(abs(v) - braking * DT, 0.0)
        y += v * DT
        update(np.full(n, 0.5), np.full(n, y), DT)
        other(np.full(n, 0.05), np.full(n, mid + sign * 0.9), DT)
        engine.step(DT)
    return engine


def test_fast_puck_passes_under_a_braking_mallet():
    e = BatchPhysicsEngine(8)
    e.enable_mallet_lift(**CERTAIN)
    e.reset(np.random.default_rng(0))
    e.score_opponent[:] = 0
    _lunge(e, steps=30)
    assert (e.lift_events[0] == 1).all() and (e.lift_events[1] == 0).all()
    # Heading kept, 45-95% of the speed (less a little drag), still inbound.
    assert np.allclose(e.puck_vx, 0.0)
    assert ((e.puck_vy < -0.40 * 7.0) & (e.puck_vy > -0.95 * 7.0)).all()
    assert (e.puck_y < e.paddle_agent_y).all()
    assert not e.lift_through.any()                # contact is back once clear


@pytest.mark.parametrize("case", [dict(braking=0.0), dict(puck_speed=4.0),
                                  dict(puck_speed=9.0), dict(offset=0.08)])
def test_no_lift_at_rest_off_speed_or_grazing(case):
    e = BatchPhysicsEngine(8)
    e.enable_mallet_lift(**CERTAIN)
    e.reset(np.random.default_rng(0))
    _lunge(e, steps=40, **case)
    assert e.lift_events.sum() == 0
    assert (e.puck_vy > 0).all() or case.get("offset")      # returned up the table


def test_a_mallet_catching_its_own_shot_does_not_lift():
    """The model is of pucks travelling AT the mallet. A mallet accelerating
    through a puck that is leaving it meets every other condition (closing,
    central, in the speed band, a_n along the puck's travel) and must not
    pass over it."""
    e = BatchPhysicsEngine(8)
    e.enable_mallet_lift(nominal=(1.0, 10.0, 1.0, 1.0, 0.0, 0.0))
    e.reset(np.random.default_rng(0))
    n = e.n_envs
    e.puck_x[:], e.puck_y[:] = 0.5, 0.70
    e.puck_vx[:], e.puck_vy[:] = 0.0, 6.5                 # up the table, away
    y, v = 0.70 - 0.12, 7.0
    e.paddle_agent_x[:], e.paddle_agent_y[:] = 0.5, y
    e.paddle_opp_x[:], e.paddle_opp_y[:] = 0.05, 1.9
    touched = np.zeros(n, dtype=bool)
    for _ in range(40):
        v += 100.0 * DT
        y += v * DT
        e.update_paddle_agent(np.full(n, 0.5), np.full(n, y), DT)
        e.update_paddle_opponent(np.full(n, 0.05), np.full(n, 1.9), DT)
        before = e.puck_vy.copy()
        e.step(DT)
        touched |= e.puck_vy > before + 0.5
    assert touched.all()                                   # it did catch the puck
    assert e.lift_events.sum() == 0 and not e.lift_through.any()
    assert (e.puck_y > e.paddle_agent_y).all()             # and the puck is still ahead


def test_a_lifted_puck_gains_nothing_and_is_counted_once():
    e = BatchPhysicsEngine(4000)
    e.enable_mallet_lift()
    e.reset(np.random.default_rng(3))
    e.score_opponent[:] = 0
    fastest = np.zeros(e.n_envs)
    n = e.n_envs
    e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = 0.5, 0.70, 0.0, -7.0
    e.paddle_agent_x[:], e.paddle_agent_y[:] = 0.5, 0.38
    e.paddle_opp_x[:], e.paddle_opp_y[:] = 0.05, 1.9
    y, v = 0.38, 4.0
    for _ in range(120):
        v = max(v - 90.0 * DT, 0.0)
        y += v * DT
        e.update_paddle_agent(np.full(n, 0.5), np.full(n, y), DT)
        e.update_paddle_opponent(np.full(n, 0.05), np.full(n, 1.9), DT)
        e.step(DT)
        live = (e.lift_events[0] > 0) & (e.score_opponent == 0)      # not yet re-served
        fastest = np.maximum(fastest, np.where(live, np.hypot(e.puck_vx, e.puck_vy), 0.0))
    lifted = e.lift_events[0] > 0
    # Pooled fit at 90 m/s^2 of braking: 0.03 + 0.41 * (90 - 30) / 70 = 0.38.
    assert 0.35 < lifted.mean() < 0.42
    assert e.lift_events.max() == 1 and not e.lift_through.any()
    assert fastest.max() < 0.95 * 7.0
    # Under the mallet and out the far side: every one of these is a goal.
    assert (e.score_opponent[lifted] == 1).all() and (e.score_opponent[~lifted] == 0).all()


def test_only_a_robot_bodied_far_side_lifts():
    e = BatchPhysicsEngine(8)
    e.enable_mallet_lift(**CERTAIN)
    e.reset(np.random.default_rng(0))
    e.opponent_robot[:4] = True
    _lunge(e, steps=30, body="opponent")
    assert (e.lift_events[1, :4] == 1).all() and (e.lift_events[1, 4:] == 0).all()
    assert (e.puck_vy[:4] > 0).all() and (e.puck_vy[4:] < 0).all()


def test_lift_draws_include_zero_and_a_zero_episode_never_lifts():
    e = BatchPhysicsEngine(4000, domain_randomize=True)
    e.enable_mallet_lift()
    e.reset(np.random.default_rng(1))
    p100, a0, b0, m_ret, k_hi, shift = e.lift_parameters
    off = p100 == 0
    assert 0.20 < off.mean() < 0.30 and (b0[off] == 0).all()
    assert 0.15 <= p100[~off].min() and p100.max() <= 0.95
    assert 10 <= a0.min() and a0.max() <= 50 and abs(shift).max() <= 0.5
    _lunge(e, steps=30)
    assert e.lift_events[0, off].sum() == 0
    # Braking at 100 m/s^2 on a 7 m/s puck: P = p100 (+- the gate shift).
    assert 0.3 < e.lift_events[0, ~off].mean() < 0.7
    with pytest.raises(ValueError):
        BatchPhysicsEngine(2).enable_mallet_lift(p100=(0.2, 1.5))


def test_env_marks_which_far_bodies_are_the_machine_and_counts_lifts():
    env = NeuralTrainingEnv(8, seed=1, stage=3, games=True, mallet_lift=dict(CERTAIN),
                            speed_restitution=True, randomize=False)
    env.reset(seed=1, opponent="external")
    _, _, _, _, info = env.step(np.zeros((8, 6), np.float32))
    assert env.engine.opponent_robot.all()
    assert info["lift_events"].shape == (8,) and info["lift_events"].sum() == 0
    env.reset(seed=1, opponent="human_proxy")
    env.step(np.zeros((8, 6), np.float32))
    assert not env.engine.opponent_robot.any()


# ── (c) sensing ─────────────────────────────────────────────────────────

def test_end_bands_and_flicker_match_their_calibration():
    rng = np.random.default_rng(0)
    n = 4000
    drops = TableDropouts(n, TableSensing(flicker_nominal=0.0), 2.0)
    for y, speed, unseen in ((0.12, 0.5, 0.45), (0.18, 6.0, 0.65), (0.25, 2.0, 0.05),
                             (2 - 0.16, 0.5, 0.02), (2 - 0.10, 2.0, 0.73), (1.0, 6.0, 0.0)):
        frames = np.array([drops.puck_hidden(np.full(n, y), np.full(n, speed), rng)
                           for _ in range(300)])[100:]
        assert abs(frames.mean() - unseen) < 0.01, (y, speed)
    # Out of the bands the puck is seen again at once.
    drops._end[:] = True
    assert not drops.puck_hidden(np.full(n, 1.0), np.full(n, 6.0), rng).any()
    # Flicker alone: about 3% of open-table frames, mostly one or two frames.
    drops = TableDropouts(n, TableSensing(), 2.0)
    frames = np.array([drops.puck_hidden(np.full(n, 1.0), np.full(n, 1.0), rng)
                       for _ in range(600)])
    assert 0.02 < frames.mean() < 0.04
    lengths = TableDropouts._lengths(TableSensing().flicker_frames, 20000, rng)
    assert 0.66 < (lengths == 1).mean() < 0.71 and lengths.max() <= 98
    assert 0.012 < (lengths >= 10).mean() < 0.025


def _table_env(opponent, n=6):
    env = NeuralTrainingEnv(n, seed=2, stage=3, games=True, report_sensing=True,
                            table_sensing=True)
    env.reset(seed=2, opponent=opponent)
    return env


def test_dropouts_fire_in_the_neural_env_whose_base_episode_never_ends():
    env = _table_env("external", n=32)
    assert env.base.max_episode_time == 1000000
    hidden = opponent = 0
    for _ in range(200):
        env.step(np.zeros((32, 6), np.float32))
        hidden += env.base.puck_hidden.sum()
        opponent += (env.base._cam_ring[:, :, 8] > 0).sum()
    assert hidden > 0 and opponent > 0


def test_hidden_opponent_is_frozen_then_falls_back_to_the_default_pose():
    env = _table_env("external")
    b, e = env.base, env.engine
    b._table_sensing.flicker_hazard[:] = 0
    b._table_sensing.opponent_hazard[:] = 0
    b._camera_tick()
    start = np.column_stack((e.paddle_opp_x, e.paddle_opp_y))
    b._table_sensing._opponent[:] = 60
    e.paddle_opp_x += 0.1                      # it moves while the camera cannot see it
    for frame in range(1, 61):
        b._camera_tick()
        seen = b._cam_ring[(b._cam_write - 1) % b._cam_ring_size]
        expected = start if frame <= 30 else np.tile(b._opp_default, (6, 1))
        assert np.allclose(seen[:, 4:6], expected), frame
        assert (seen[:, 8] > 0).all() and (seen[:, 13] > 0).all()
        assert np.allclose(seen[:, 6], np.where(frame <= 30, e.paddle_agent_x, b._rival_default[0]))
    _, vx, vy = b._camera_read()
    assert not vx.any() and not vy.any()        # at rest across the spell's edges
    b._camera_tick()
    seen = b._cam_ring[(b._cam_write - 1) % b._cam_ring_size]
    assert np.allclose(seen[:, 4], e.paddle_opp_x) and not seen[:, 8].any()


def test_a_person_sees_the_puck_the_camera_lost_and_a_robot_copy_does_not():
    for opponent, sees in (("human_proxy", True), ("external", False)):
        env = _table_env(opponent)
        b, e = env.base, env.engine
        b._table_sensing.flicker_hazard[:] = 0
        b._table_sensing.opponent_hazard[:] = 0
        e.puck_x[:], e.puck_y[:] = 0.3, 1.2
        e.puck_vx[:] = e.puck_vy[:] = 0.0
        b._perception.reset(e.puck_x, e.puck_y)
        b._sync_far_view(slice(None))
        b._table_sensing._flicker[:] = 40       # the camera loses the puck...
        e.puck_x[:] = 0.6                       # ...and it is somewhere else
        for _ in range(12):
            b._camera_tick()
        assert b.puck_hidden.all()
        far = env.opponent_obs()
        assert (abs(far[:, 0] - 0.6) < 0.01).all() == sees
        assert (abs(env._features(b._make_obs_direct())[:, 0] - 0.3) < 0.01).all()


def test_table_sensing_needs_the_camera_model():
    with pytest.raises(ValueError):
        NeuralTrainingEnv(4, realistic=False, table_sensing=True)


# ── (d) command latency ─────────────────────────────────────────────────

def _robot_env(n=4):
    env = BatchAirHockeyEnv(n, action_mode="profile_a", agent_dynamics="profile",
                            opponent_dynamics="profile", opponent_body="robot",
                            opponent_policy="external", domain_randomize=True,
                            **sensing_kwargs(True))
    env.command_delay_s = 0.015
    return env


def test_command_delay_is_drawn_per_episode_and_applied_per_env():
    env = _robot_env(512)
    env.enable_command_jitter(delay_range=(0.009, 0.019))
    env.reset(seed=4)
    delay = env._command_delay_env
    assert 0.009 <= delay.min() and delay.max() <= 0.019 and delay.std() > 0.002
    assert env.command_delay_s == 0.015
    # In force after each env's own delay: at 10 ms the short draws have
    # the new command and the long ones still hold the old.
    dyn = env._agent_dyn
    old = dyn["command_x"].copy()
    new = old + 0.05
    cap = dyn["max_accel"]
    env._update_dynamics(dyn, new, dyn["command_y"].copy(), env.sub_dt,
                         accel_cap=cap, defer_command=0.010 < delay)
    assert np.allclose(dyn["command_x"], np.where(0.010 < delay, old, new))
    with pytest.raises(ValueError):
        env.enable_command_jitter(delay_range=(0.01, 0.02))


def test_a_drawn_delay_lands_on_the_nearest_substep():
    env = _robot_env(6)
    env.enable_command_jitter(delay_range=(0.009, 0.019))
    env.reset(seed=4)
    env._command_delay_env[:] = [0.009, 0.011, 0.0115, 0.0135, 0.0185, 0.019]
    deferred = np.zeros(6)
    update = env._update_dynamics

    def counted(dyn, *args, defer_command=False, **kwargs):
        if dyn is env._agent_dyn:
            deferred[:] += defer_command
        return update(dyn, *args, defer_command=defer_command, **kwargs)

    env._update_dynamics = counted
    env.step(np.zeros((6, 3)))
    assert np.allclose(deferred * env.sub_dt, [0.010, 0.010, 0.0125, 0.0125, 0.0175, 0.020])
    # The assumed delay is on a boundary already, and lands where it did.
    env._command_delay_env[:] = env.command_delay_s
    deferred[:] = 0
    env.step(np.zeros((6, 3)))
    assert np.allclose(deferred * env.sub_dt, 0.015)


def test_latency_stays_nominal_without_domain_randomisation():
    env = BatchAirHockeyEnv(64, action_mode="profile_a", agent_dynamics="profile",
                            opponent_dynamics="profile", opponent_body="robot",
                            opponent_policy="external", **sensing_kwargs(True))
    env.command_delay_s = 0.015
    env.enable_command_jitter(delay_range=(0.009, 0.019), late_probability=(0.001, 0.028))
    env.reset(seed=4)
    assert (env._command_delay_env == 0.015).all()
    assert np.allclose(env._late_command_p, np.sqrt(0.001 * 0.028))


def test_a_late_command_takes_effect_a_tick_later():
    env = _robot_env()
    env.enable_command_jitter(late_probability=(0.001, 0.028))
    env.reset(seed=4)
    assert 0.001 <= env._late_command_p.min() and env._late_command_p.max() <= 0.028
    env._late_command_p[:] = [1.0, 1.0, 0.0, 0.0]
    start = env.engine.paddle_agent_y.copy()
    action = np.tile([0.0, 1.0, 1.0], (4, 1))
    env.step(action)
    moved = np.abs(env.engine.paddle_agent_y - start)
    assert (moved[:2] < 1e-6).all() and (moved[2:] > 1e-4).all()
    env.step(action)
    assert (np.abs(env.engine.paddle_agent_y - start)[:2] > 1e-4).all()
