"""The sim must reproduce what was MEASURED on the table.

These exist because changing `_apply_friction` from a constant deceleration to
mu*g + b*v^2, and `_collide_walls` from specular to lossy, broke nothing in the
existing suite. Eighty-two tests passed either way, which means none of them
constrained the physics -- only its plumbing. A model nobody checks against the
world drifts back to whatever is convenient.

Numbers come from vision/bin/fit_puck.py over two sessions (2026-08-23 and
2026-08-29) with different puck marking and a fitter validated against
synthetic truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.physics import TableConfig

G = 9.81


def engine(n=1, **kw):
    return BatchPhysicsEngine(n_envs=n, config=TableConfig(**kw))


# Deceleration at speed, mm/s^2, straight off the fitted a + b*v^2. Anything
# that silently reverts to a constant coefficient fails the 6 m/s row by 8x.
@pytest.mark.parametrize("speed_ms,expect_mm_s2", [
    (0.3, 17.8), (1.0, 49.5), (3.0, 327.9), (6.0, 1267.5),
])
def test_drag_matches_the_measured_curve(speed_ms, expect_mm_s2):
    e = engine()
    e.puck_vx[:] = speed_ms
    e.puck_vy[:] = 0.0
    dt = 1e-4
    v0 = float(e.puck_vx[0])
    e._apply_friction(dt)
    decel = (v0 - float(e.puck_vx[0])) / dt * 1000.0     # m/s^2 -> mm/s^2
    assert decel == pytest.approx(expect_mm_s2, rel=0.02)


def test_drag_dominates_rolling_where_a_policy_plays():
    """At 6 m/s the v^2 term must be ~80x the rolling one.

    The point of the whole change: a single friction coefficient cannot span
    the range, so if these two ever come out comparable the model has been
    flattened back to a constant.
    """
    c = TableConfig()
    rolling = c.puck_friction * G
    drag = c.PUCK_DRAG_B * 6.0 ** 2
    assert drag / rolling > 50


def test_puck_decelerates_monotonically_and_stops():
    e = engine()
    e.puck_vx[:] = 9.0
    e.puck_vy[:] = 0.0
    last = 9.0
    for _ in range(200000):
        e._apply_friction(1e-4)
        v = float(np.hypot(e.puck_vx[0], e.puck_vy[0]))
        assert v <= last + 1e-12, "drag added energy"
        last = v
    # 20 s from 9 m/s leaves ~1.1 m/s: the v^2 term kills the top end fast and
    # then the tiny rolling term takes over, which is what an air cushion
    # should do and is why a single constant could never describe it.
    assert last < 2.0, f"still at {last:.2f} m/s"
    assert last > 0.3, f"stopped too fast ({last:.2f} m/s) — rolling term too big"


def test_wall_bounce_is_not_specular():
    """A rail keeps only ~2/3 of the tangential component.

    Reflection used to be specular, which puts every bank shot off at the
    wrong angle -- and banking is the skill worth learning.
    """
    c = TableConfig()
    e = engine()
    e.puck_x[:] = c.puck_radius * 0.5          # already through the left rail
    e.puck_y[:] = c.height * 0.5
    e.puck_vx[:] = -4.0
    e.puck_vy[:] = 3.0
    e._collide_walls()
    assert float(e.puck_vx[0]) == pytest.approx(4.0 * c.wall_restitution, rel=1e-6)
    assert float(e.puck_vy[0]) == pytest.approx(3.0 * c.wall_tangential, rel=1e-6)
    # and the outgoing angle is steeper than incidence, which is the visible
    # consequence and the thing a specular model gets wrong
    ang_in = np.degrees(np.arctan2(3.0, 4.0))
    ang_out = np.degrees(np.arctan2(float(e.puck_vy[0]), float(e.puck_vx[0])))
    assert ang_in - ang_out > 3.0


@pytest.mark.parametrize("wall", ["left", "right", "bottom", "top"])
def test_every_rail_takes_tangential_momentum(wall):
    """All four, not just the pair someone happened to test."""
    c = TableConfig()
    e = engine()
    e.puck_x[:], e.puck_y[:] = c.width * 0.5, c.height * 0.5
    # aim off-centre in x so the end rails' goal mouth is missed
    setup = {
        "left":   (c.puck_radius * 0.5, c.height * 0.5, -3.0, 2.0),
        "right":  (c.width - c.puck_radius * 0.5, c.height * 0.5, 3.0, 2.0),
        "bottom": (c.puck_radius * 1.5, c.puck_radius * 0.5, 2.0, -3.0),
        "top":    (c.puck_radius * 1.5, c.height - c.puck_radius * 0.5, 2.0, 3.0),
    }[wall]
    e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = setup
    tan_in = abs(setup[3] if wall in ("left", "right") else setup[2])
    e._collide_walls()
    tan_out = abs(float(e.puck_vy[0] if wall in ("left", "right") else e.puck_vx[0]))
    assert tan_out == pytest.approx(tan_in * c.wall_tangential, rel=1e-6), wall


def test_domain_randomisation_brackets_the_measurements():
    """Randomised envs must straddle reality, not sit beside it.

    The old ranges predated the table: friction was sampled 0.005-0.05, i.e.
    3x to 33x the measured rolling term, so nearly every env modelled a
    surface the puck has never been on.
    """
    e = BatchPhysicsEngine(n_envs=4000, config=TableConfig(),
                           domain_randomize=True)
    e.reset()
    c = TableConfig()
    for name, arr, truth in (
        ("puck_friction", e.puck_friction, c.puck_friction),
        ("drag_b", e.drag_b, c.PUCK_DRAG_B),
        ("wall_restitution", e.wall_restitution, c.wall_restitution),
        ("wall_tangential", e.wall_tangential, c.wall_tangential),
    ):
        assert arr.min() < truth < arr.max(), f"{name} does not bracket {truth}"
        # and the measured value is near the middle, not clinging to an edge
        q = float((arr < truth).mean())
        assert 0.2 < q < 0.8, f"{name}: measured value at quantile {q:.2f}"


def test_a_simulated_glide_refits_to_the_input_drag():
    """Closing the loop: simulate, then recover b the way fit_puck does.

    If the integrator and the model disagree -- an Euler step that overshoots,
    a coefficient applied per-axis instead of along the velocity -- this is
    where it shows, because the recovery uses the same deceleration-vs-speed
    fit that produced the constant in the first place.
    """
    c = TableConfig()
    dt = 1.0 / 1000.0
    speeds, decels = [], []
    for v0 in (1.0, 2.0, 4.0, 6.0, 8.0):
        e = engine()
        e.puck_vx[:], e.puck_vy[:] = v0, 0.0
        vs = []
        for _ in range(120):
            e._apply_friction(dt)
            vs.append(float(np.hypot(e.puck_vx[0], e.puck_vy[0])))
        t = np.arange(len(vs)) * dt
        slope = np.polyfit(t, vs, 1)[0]
        speeds.append(float(np.mean(vs)))
        decels.append(-slope)
    A = np.column_stack([np.ones(len(speeds)), np.array(speeds) ** 2])
    coef, *_ = np.linalg.lstsq(A, np.array(decels), rcond=None)
    assert coef[1] == pytest.approx(c.PUCK_DRAG_B, rel=0.02), (
        f"recovered b {coef[1]:.4e} vs input {c.PUCK_DRAG_B:.4e}")


# ── The paddle must move like the machine, not like a convenience ────────

def test_batch_env_defaults_to_the_firmware_profile():
    """`ideal` teleports the paddle. Training against it produces a policy
    that commands positions no actuator can reach, and it finds out on the
    hardware. This was the default for a long time."""
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=2)
    assert e.agent_dynamics_type == "profile"
    # The OPPONENT is a human and is deliberately NOT on the firmware law --
    # a hand is not a stepper under a trapezoidal profile.
    assert e.opponent_dynamics_type == "delayed"


def test_action_rate_can_represent_the_measured_latency():
    """At 60 Hz a step is 16.7 ms, longer than the whole 7.7 ms loop latency,
    so the delay rounds to zero or one step and the sim silently models a
    robot that sees instantly."""
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.perception import MEASURED_LOOP_MEAN_S
    e = BatchAirHockeyEnv(n_envs=1)
    assert e.action_dt <= MEASURED_LOOP_MEAN_S * 1.5, (
        f"action_dt {e.action_dt * 1000:.1f} ms cannot resolve a "
        f"{MEASURED_LOOP_MEAN_S * 1000:.1f} ms delay")


def test_profile_paddle_does_not_teleport():
    """One action step of the real law covers a bounded distance.

    Ideal dynamics would jump the full 0.4 m instantly; the firmware profile
    accelerates from rest under a jerk limit, so the first 10 ms step moves
    roughly half a millimetre. Anything that reverts the default to `ideal`
    fails here rather than at the table.
    """
    from airhockey.dynamics import ProfileDynamics
    d = ProfileDynamics()
    d.reset(0.5, 0.5)
    x, _y = d.update(0.9, 0.5, 1 / 100)
    moved = x - 0.5
    assert 0.0 < moved < 0.02, f"moved {moved * 1000:.2f} mm in one step"
    # and it does converge, rather than crawling forever
    for _ in range(200):
        x, _y = d.update(0.9, 0.5, 1 / 100)
    assert abs(x - 0.9) < 1e-3, f"never arrived: {x:.4f}"


def test_no_entry_point_carries_its_own_caps():
    """The caps live in dynamics.py. They were duplicated into every training
    script at 3.0 / 30.0, so a policy trained from any of them learned a
    paddle four times slower than the machine."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    bad = []
    for p in list((root / "bin").rglob("*.py")) + list((root / "airhockey").rglob("*.py")):
        txt = p.read_text()
        if "max_speed=3.0" in txt or "max_accel=30.0" in txt:
            bad.append(p.name)
    assert not bad, f"hardcoded caps still in: {bad}"


def test_agent_is_confined_to_the_reachable_workspace():
    """The sim must not offer the paddle ground the machine cannot stand on.

    Cables pull only, so the reachable box is 35% of the robot's half and its
    nearest approach to its own goal line is sim y 0.099. A policy trained on
    the full half learns to defend from y=0; on the hardware
    HardwareDynamics._sim_to_mm clamps silently, so the paddle stops short and
    the failure looks like a bad policy rather than one being cut off.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.dynamics import workspace_in_sim
    e = BatchAirHockeyEnv(n_envs=64, agent_dynamics="ideal")
    ws = workspace_in_sim(e.table_config.width, e.table_config.height / 2)
    e.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        e.step(rng.uniform(-1, 1, (64, 2)).astype(np.float32))
        assert e.engine.paddle_agent_x.min() >= ws["min_x"] - 1e-6
        assert e.engine.paddle_agent_x.max() <= ws["max_x"] + 1e-6
        assert e.engine.paddle_agent_y.min() >= ws["min_y"] - 1e-6
        assert e.engine.paddle_agent_y.max() <= ws["max_y"] + 1e-6


def test_the_opponent_is_not_confined():
    """The opponent stands in for a HUMAN, who can reach their whole side."""
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.dynamics import workspace_in_sim
    e = BatchAirHockeyEnv(n_envs=256, agent_dynamics="ideal",
                          opponent_dynamics="ideal", opponent_policy="random")
    ws = workspace_in_sim(e.table_config.width, e.table_config.height / 2)
    e.reset(seed=5)
    reach = 0.0
    for _ in range(60):
        e.step(np.zeros((256, 2), dtype=np.float32))
        reach = max(reach, float(e.engine.paddle_opp_x.max()))
    assert reach > ws["max_x"] + 0.02, "opponent wrongly clamped to the robot's box"


def test_agent_speed_is_pinned_to_the_firmware_clamp():
    """MAX_SPEED_M_S is the Teensy's own clamp, not an estimate -- the one
    number in the actuator model that is exact. It is neither sampled above
    (the machine cannot go there) nor below (a full run showed accel binds
    long before speed: v = sqrt(2ad) tops out ~3.5 m/s inside the box, so an
    "underperforming machine" is modelled by the accel band, which IS wide).

    Scoped to the AGENT. The opponent stands in for a human and is not behind
    the Teensy; its caps are its own.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.dynamics import AGENT_DR_ACCEL_M_S2, MAX_SPEED_M_S
    e = BatchAirHockeyEnv(n_envs=4000, domain_randomize=True)
    e.reset(seed=11)
    dyn = e._agent_dyn
    assert np.all(dyn["max_speed"] == MAX_SPEED_M_S), "speed must be pinned"
    lo, hi = AGENT_DR_ACCEL_M_S2
    assert (lo, hi) == (10.0, 60.0)
    assert dyn["max_accel"].min() >= lo and dyn["max_accel"].max() <= hi
    assert dyn["max_accel"].std() > 5.0, "accel band not actually sampled"


def test_randomisation_does_not_erase_the_side_asymmetry():
    """DR must scale each side by ITS OWN caps.

    Scaling both by the robot's -- which it did -- meant the opponent was
    built at 15 m/s / 80 m/s^2 and then overwritten on the first randomised
    reset with 6-12 and 10-22.5, making the human sparring partner strictly
    slower and far gentler than the machine it is meant to stretch.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.dynamics import (OPPONENT_MAX_ACCEL_M_S2,
                                    OPPONENT_MAX_SPEED_M_S)
    e = BatchAirHockeyEnv(n_envs=4000, domain_randomize=True)
    e.reset(seed=11)
    opp, agent = e._opp_dyn, e._agent_dyn

    # Accel is where the asymmetry lives now (agent 10-60, human 40-90), and
    # the human's top-end speed still exceeds the robot's clamp.
    assert opp["max_accel"].mean() > agent["max_accel"].mean(), (
        "randomised opponent accelerates less than the robot")
    assert opp["max_speed"].max() > agent["max_speed"].max(), (
        "human top speed should exceed the robot clamp")
    # Each side stays inside its own band.
    assert opp["max_speed"].max() <= OPPONENT_MAX_SPEED_M_S + 1e-9
    assert opp["max_accel"].max() <= 1.125 * OPPONENT_MAX_ACCEL_M_S2 + 1e-9


def test_scalar_env_and_batch_env_agree_on_the_reachable_box():
    """Both envs must bound the paddle identically.

    The batch env was constrained first and the scalar one -- which is what
    the web UI runs -- was missed, so dragging the mouse still put the paddle
    where the robot cannot reach. Two envs disagreeing about the machine's
    limits is the same class of bug as two copies of a constant.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.env import AirHockeyEnv
    b = BatchAirHockeyEnv(n_envs=1)
    sc = AirHockeyEnv()
    np.testing.assert_allclose(sc._action_low, b._action_low, atol=1e-9)
    np.testing.assert_allclose(sc._action_high, b._action_high, atol=1e-9)
    # Everything else the two must not disagree about. They are still separate
    # implementations -- see SIM2REAL.md -- so until the scalar env becomes an
    # adapter over the batch one, this is the guard that keeps a fix applied
    # to one from being missed on the other. That has now happened twice.
    assert sc.action_dt == b.action_dt, "action rate diverged"
    assert type(sc.agent_dynamics).__name__ == "ProfileDynamics", (
        "scalar env is not on the firmware law")
    assert b.agent_dynamics_type == "profile"


def test_scalar_env_clamps_the_agent_to_the_workspace():
    from airhockey.env import AirHockeyEnv
    e = AirHockeyEnv()
    e.reset(seed=1)
    for corner in ((-5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (5.0, -5.0)):
        x, y = e._clamp_to_half(corner[0], corner[1], agent=True)
        assert e._ws["min_x"] - 1e-9 <= x <= e._ws["max_x"] + 1e-9
        assert e._ws["min_y"] - 1e-9 <= y <= e._ws["max_y"] + 1e-9


# ── The two sides are not the same machine ───────────────────────────────

def test_the_human_side_is_less_constrained_than_the_robot():
    """Training against a copy of yourself teaches you your own limits.

    The opponent stands in for a person, and the measured hand-held mallet
    reached 7.33 m/s over 862 x 951 mm against the robot's 568 x 620 box. The
    sparring partner must therefore be faster, sharper and unconfined --
    otherwise the policy learns that its opponent is exactly as limited as it
    is, which is the one thing certainly false.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=8)
    assert e._opp_dyn["max_speed"][0] > e._agent_dyn["max_speed"][0]
    assert e._opp_dyn["max_accel"][0] > e._agent_dyn["max_accel"][0]
    # and a hand is not a stepper under a trapezoidal profile
    assert e.agent_dynamics_type == "profile"
    assert e.opponent_dynamics_type == "delayed"


def test_side_flag_tells_the_policy_which_body_it_is_driving():
    """In self-play one network plays both sides. They have different caps and
    only one is confined, so without this feature the policy cannot act
    correctly for either -- the observation would be identical in both cases.
    Production always sets it to ROBOT."""
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=4)
    out = e.reset(seed=0)
    obs = out[0] if isinstance(out, tuple) else out
    assert obs.shape[1] == BatchAirHockeyEnv.OBS_DIM == 15
    assert np.all(obs[:, 12] == BatchAirHockeyEnv.ROBOT_SIDE)
    mirrored = e.mirror_obs(obs)
    assert np.all(mirrored[:, 12] == BatchAirHockeyEnv.HUMAN_SIDE)
    # mirror is an involution, flag included
    np.testing.assert_allclose(e.mirror_obs(mirrored), obs, atol=1e-6)


def test_opponent_reaches_where_the_robot_cannot():
    """Concretely: the human must be able to stand on their goal line."""
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=128, agent_dynamics="ideal",
                          opponent_dynamics="ideal", opponent_policy="random")
    e.reset(seed=9)
    lowest = 1e9
    for _ in range(80):
        e.step(np.zeros((128, 2), dtype=np.float32))
        lowest = min(lowest, float(e.engine.paddle_opp_y.max()))
    cfg = e.table_config
    assert e.engine.paddle_opp_y.max() > cfg.height - 0.15, \
        "opponent never approaches its own goal line"


def test_cap_features_report_the_body_the_policy_is_driving():
    """Nominal robot reads exactly 1.0 on both, so the features are a ratio to
    the machine as built rather than an arbitrary scale."""
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=4)
    obs = e.reset(seed=0)
    np.testing.assert_allclose(obs[:, 13], 1.0, atol=1e-6)
    np.testing.assert_allclose(obs[:, 14], 1.0, atol=1e-6)


def test_cap_features_track_domain_randomisation():
    """The whole point: with DR on, envs differ and the policy can see it.

    A constant feature would be worse than no feature -- it would look like the
    policy had been told its limits while telling it nothing.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.dynamics import MAX_ACCEL_M_S2, MAX_SPEED_M_S
    e = BatchAirHockeyEnv(n_envs=512, domain_randomize=True)
    obs = e.reset(seed=3)
    # Speed is pinned to the firmware clamp, so its feature is a constant
    # 1.0 on the robot -- it still earns its slot by reading 1.25 on the
    # mirrored human side. Accel carries the per-env variation.
    np.testing.assert_allclose(obs[:, 13], 1.0, atol=1e-6)
    assert obs[:, 14].std() > 0.05, "accel feature is constant under DR"
    # And they are the caps actually in force, not a redundant copy of nominal.
    np.testing.assert_allclose(
        obs[:, 13], e._agent_dyn["max_speed"] / MAX_SPEED_M_S, rtol=1e-6)
    np.testing.assert_allclose(
        obs[:, 14], e._agent_dyn["max_accel"] / MAX_ACCEL_M_S2, rtol=1e-6)


def test_mirrored_view_reports_the_human_caps():
    """Mirroring hands the policy the other body, so it must hand over the
    other body's limits -- and still round-trip."""
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.dynamics import (MAX_ACCEL_M_S2, MAX_SPEED_M_S,
                                    OPPONENT_MAX_ACCEL_M_S2,
                                    OPPONENT_MAX_SPEED_M_S)
    e = BatchAirHockeyEnv(n_envs=8)
    obs = e.reset(seed=0)
    m = e.mirror_obs(obs)
    np.testing.assert_allclose(
        m[:, 13], OPPONENT_MAX_SPEED_M_S / MAX_SPEED_M_S, rtol=1e-6)
    np.testing.assert_allclose(
        m[:, 14], OPPONENT_MAX_ACCEL_M_S2 / MAX_ACCEL_M_S2, rtol=1e-6)
    np.testing.assert_allclose(e.mirror_obs(m), obs, atol=1e-6)


def test_scalar_env_reports_the_same_cap_features():
    """The scalar env drives the UI and must not drift from the batch env --
    this is the third feature both have had to grow independently."""
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.env import AirHockeyEnv
    s, _ = AirHockeyEnv().reset(seed=0)
    b = BatchAirHockeyEnv(n_envs=1).reset(seed=0)
    assert s.shape[0] == b.shape[1] == BatchAirHockeyEnv.OBS_DIM
    np.testing.assert_allclose(s[12:], b[0, 12:], atol=1e-6)


def test_trainers_enable_the_real_sensing_chain():
    """Camera delay, tracker noise and the IR blind spot were all implemented,
    defaulted off in the env, and then never switched on anywhere -- so every
    run trained on perfect, instantaneous, always-visible observation. Same for
    domain randomisation. The env defaults stay off on purpose; the TRAINERS
    are what must not."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("train_tdmpc2.py", "train_tdmpc2_fast.py"):
        txt = (root / "bin" / name).read_text()
        assert "sensing_kwargs(" in txt, f"{name} does not model the camera"
        assert "domain_randomize=" in txt, f"{name} does not randomise"


def test_the_camera_clock_delays_and_the_blind_spot_blinds():
    """Two properties of the 200 Hz camera model, tested separately.

    LATENCY: at constant velocity, the observed puck trails the true one by
    roughly v * latency. The measured band is 5.1-10.3 ms, which at 3 m/s is
    a 15-31 mm trail -- and crucially it must NOT be zero (perfect sight)
    nor a whole action step's worth quantised up (the old ring gave exactly
    30 mm always).

    BLIND SPOT: a straight glide through the glare patch is dead-reckoned
    well -- that is faithful, the real tracker coasts on velocity -- so the
    honest failure mode is a DIRECTION CHANGE while hidden. Flip the puck's
    velocity inside the patch: the tracker must keep believing the old
    course for a while, i.e. large transient error.
    """
    from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
    from airhockey.perception import GLARE_W_MM
    e = BatchAirHockeyEnv(n_envs=4, **sensing_kwargs(True))
    e.reset(seed=0)
    cfg = e.table_config

    # -- latency, far from the glare patch
    e.engine.puck_x[:], e.engine.puck_y[:] = 0.10, 0.4
    e.engine.puck_vx[:], e.engine.puck_vy[:] = 3.0, 0.0
    trails = []
    for _ in range(20):
        obs, *_ = e.step(np.zeros((4, 2), dtype=np.float32))
        trails.append(float(e.engine.puck_x[0]) - float(obs[0, 0]))
    trail = np.median(trails[5:])
    assert 0.010 < trail < 0.045, (
        f"obs trails truth by {trail*1000:.0f} mm at 3 m/s — outside the "
        "5-10 ms latency band (15-31 mm) plus noise margin")

    # -- blind spot: reverse course while hidden in the glare
    e.reset(seed=1)
    e.engine.puck_x[:], e.engine.puck_y[:] = cfg.width / 2 - 0.03, cfg.height / 2
    e.engine.puck_vx[:], e.engine.puck_vy[:] = 1.5, 0.0
    worst = 0.0
    flipped = False
    for _ in range(40):
        inside = abs(float(e.engine.puck_x[0]) - cfg.width / 2) < GLARE_W_MM / 2000.0
        if inside and not flipped:
            e.engine.puck_vx[:] = -1.5     # the bounce nobody saw
            flipped = True
        obs, *_ = e.step(np.zeros((4, 2), dtype=np.float32))
        if flipped:
            worst = max(worst, abs(float(obs[0, 0]) - float(e.engine.puck_x[0])))
    assert flipped, "puck never entered the glare patch — test setup is wrong"
    assert worst > 0.05, (
        f"unseen course reversal cost only {worst*1000:.0f} mm — the "
        "tracker is seeing through the glare")


def test_the_robot_starts_where_it_can_stand():
    """Constraining the ACTION but not the START STATE constrains nothing.

    The engine draws the agent paddle over the whole half, which is 2.8x the
    reachable box, so 58% of episodes began outside it and spent their first
    steps being hauled back by the clamp -- from a pose the machine cannot
    hold, possibly having touched the puck there.
    """
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.env import AirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=4000)
    e.reset(seed=0)
    ws = e._ws
    x, y = e.engine.paddle_agent_x, e.engine.paddle_agent_y
    assert np.all((x >= ws["min_x"]) & (x <= ws["max_x"])), "reset x outside box"
    assert np.all((y >= ws["min_y"]) & (y <= ws["max_y"])), "reset y outside box"
    # and it still spans the box rather than collapsing to a fixed point
    assert x.max() - x.min() > 0.8 * (ws["max_x"] - ws["min_x"])
    assert y.max() - y.min() > 0.8 * (ws["max_y"] - ws["min_y"])

    # Scalar env too -- it drives the UI and has diverged before.
    s = AirHockeyEnv()
    for seed in range(50):
        s.reset(seed=seed)
        p = s.engine.state.paddle_agent
        assert ws["min_x"] <= p.x <= ws["max_x"], f"scalar x {p.x}"
        assert ws["min_y"] <= p.y <= ws["max_y"], f"scalar y {p.y}"


def test_a_stationary_opponent_stays_where_it_was_placed():
    """`state` is a view rebuilt from the arrays, so the corner and goalie
    placements were written to a transient object and reverted on the first
    sync -- a 'stationary in a corner' opponent that was never in the corner."""
    from airhockey.env import AirHockeyEnv
    for policy, check in (
        ("goalie", lambda p, c: abs(p.x - c.width / 2) < 1e-6),
        ("corner", lambda p, c: min(abs(p.x - c.paddle_radius),
                                    abs(p.x - (c.width - c.paddle_radius))) < 1e-6),
    ):
        e = AirHockeyEnv(opponent_policy=policy)
        e.reset(seed=1)
        cfg = e.table_config
        placed = (e.engine.state.paddle_opponent.x,
                  e.engine.state.paddle_opponent.y)
        assert check(e.engine.state.paddle_opponent, cfg), f"{policy}: bad placement"
        for _ in range(50):
            e.step(np.zeros(2, dtype=np.float32))
        now = e.engine.state.paddle_opponent
        assert abs(now.x - placed[0]) < 1e-6 and abs(now.y - placed[1]) < 1e-6, \
            f"{policy} opponent drifted from {placed} to ({now.x}, {now.y})"


# ── Actions speak table coordinates; the box is learned, not baked in ────

def test_unreachable_commands_are_capped_and_fined():
    """The action space spans the full half so one space serves both bodies
    in self-play; the machine's box is enforced by capping the target at the
    closest reachable point and charging for the overshoot."""
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=3, agent_dynamics="ideal")
    e.reset(seed=2)
    # Park the puck away from everything so no goal/contact reward interferes.
    e.engine.puck_x[:], e.engine.puck_y[:] = 0.5, 1.5
    e.engine.puck_vx[:], e.engine.puck_vy[:] = 0.0, 0.0

    # env0: mid-box (reachable). env1: own goal line (unreachable).
    # env2: the half's far corner (unreachable, worst case).
    acts = np.array([[0.0, 0.0], [0.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    _, r, _, _, info = e.step(acts)

    assert r[0] == 0.0, f"reachable command was fined: {r[0]}"
    assert r[1] < 0.0 and r[2] < 0.0, "unreachable commands were free"
    assert r[2] < r[1], "fine is not proportional to the overshoot"
    assert r[2] > -0.05, f"fine {r[2]} is not 'slight'"
    np.testing.assert_allclose(info["ws_overshoot"][0], 0.0)

    # The paddle itself was capped at the closest reachable point.
    ws = e._ws
    np.testing.assert_allclose(e.engine.paddle_agent_y[1], ws["min_y"], atol=1e-6)
    np.testing.assert_allclose(e.engine.paddle_agent_x[2], ws["max_x"], atol=1e-6)
    np.testing.assert_allclose(e.engine.paddle_agent_y[2], ws["min_y"], atol=1e-6)


def test_scalar_env_fines_unreachable_commands_identically():
    """Same rule, same constant, or the UI and training drift apart again."""
    from airhockey.batch_env import BatchAirHockeyEnv
    from airhockey.env import AirHockeyEnv
    assert AirHockeyEnv.WS_PENALTY_PER_UNIT == BatchAirHockeyEnv.WS_PENALTY_PER_UNIT

    s = AirHockeyEnv(agent_dynamics=None, still_puck=True)
    s.reset(seed=2)
    _, r, _, _, _ = s.step(np.array([0.0, -1.0], dtype=np.float32))
    b = BatchAirHockeyEnv(n_envs=1)
    b.reset(seed=2)
    b.engine.puck_x[:], b.engine.puck_y[:] = 0.5, 1.5
    b.engine.puck_vx[:], b.engine.puck_vy[:] = 0.0, 0.0
    _, rb, _, _, _ = b.step(np.array([[0.0, -1.0]], dtype=np.float32))
    assert r < 0.0 and rb[0] < 0.0
    np.testing.assert_allclose(r, rb[0], atol=1e-9)


def test_no_phantom_goal_penalty_for_unreachable_commands():
    """A conceded goal costs -20; commanding past the reachable box must not.

    The shapers detected goals from the SIGN of the base reward, and the
    workspace-overshoot penalty made that sign negative on every out-of-box
    command -- a random policy earned -36k shaped reward while actually
    winning 2-0, and the first training run learned from those numbers.
    """
    from airhockey.dynamics import ProfileDynamics
    from airhockey.env import AirHockeyEnv
    from airhockey.rewards import BatchRewardShaper, ShapedRewardWrapper, STAGE_SCORING

    env = ShapedRewardWrapper(
        AirHockeyEnv(agent_dynamics=ProfileDynamics(), opponent_policy="idle",
                     still_puck=True),
        stage=STAGE_SCORING)
    env.reset(seed=0)
    # Hammer the far corner of the half -- far outside the box -- with the
    # puck parked dead centre: no goals can occur.
    total = 0.0
    for _ in range(100):
        _, r, _, _, info = env.step(np.array([1.0, -1.0], dtype=np.float32))
        total += r
    assert info["score_agent"] == 0 and info["score_opponent"] == 0
    assert total > -2.0, (
        f"{total:.1f} over 100 goalless steps — the -20 goal penalty is "
        "firing on the workspace fine again")

    # And the batch shaper agrees when handed a scoreboard.
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=4)
    obs = e.reset(seed=0)
    shaper = BatchRewardShaper(4, stage=STAGE_SCORING)
    info = {"puck_vx": np.zeros(4), "puck_vy": np.zeros(4),
            "score_agent": np.zeros(4), "score_opponent": np.zeros(4)}
    shaper.reset(obs, info=info)
    raw = np.full(4, -0.007)      # ws fine, no goal anywhere
    shaped = shaper.compute(obs, raw, info=info)
    assert np.all(shaped > -1.0), shaped
    # A real concession -- scoreboard moved -- still costs the full penalty.
    info2 = dict(info, score_opponent=np.ones(4))
    shaped = shaper.compute(obs, np.full(4, -1.0), info=info2)
    assert np.all(shaped <= -19.0), shaped


def test_shot_mix_rewards_the_neglected_shot_type():
    """Bank and straight shots should both stay in the repertoire.

    The bonus pays weight * (recent fraction of the OTHER kind): all-straight
    play makes banks worth more and vice versa, equilibrium 50/50. It must
    stay small -- a tiebreaker, not a reason to bank an open net.
    """
    from airhockey.rewards import BatchRewardShaper, _is_bank_shot

    # Geometry first: from centre, straight up is straight; a shot angled
    # hard sideways leaves the table before the far goal line.
    assert not _is_bank_shot(0.5, 1.0, 0.0, 3.0)
    assert _is_bank_shot(0.5, 1.0, 4.0, 2.0)
    assert _is_bank_shot(0.2, 0.5, -2.0, 2.0)

    sh = BatchRewardShaper(1, stage=2)
    assert 0 < sh.shot_mix_weight <= 1.0, "mix bonus must stay small"
    # After a run of banks, a straight shot must pay more than another bank.
    sh._bank_ema[:] = 0.9
    straight_pay = sh.shot_mix_weight * sh._bank_ema[0]
    bank_pay = sh.shot_mix_weight * (1.0 - sh._bank_ema[0])
    assert straight_pay > bank_pay


# ── History observations and the velocity-carrying action ────────────────

def test_history_obs_are_ordered_frames_with_the_right_spacing():
    """5 puck frames at 0/10/20/50/100 ms behind the newest visible one.

    Motion must read newest-first, and the spacing must be wall-clock true:
    at 2 m/s the 10 ms gap between the first two lags is ~20 mm.
    """
    from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
    e = BatchAirHockeyEnv(n_envs=4, obs_mode="history", action_mode="profile_v",
                          **sensing_kwargs(True))
    e.reset(seed=0)
    # A lane no paddle can reach during the test: hug the left rail heading
    # away from the agent, with the opponent parked in the far corner. The
    # first version sent the puck across the agent's half and the paddle HIT
    # it -- real physics ruining a test that assumed free flight.
    e.engine.paddle_opp_x[:], e.engine.paddle_opp_y[:] = 0.9, 1.9
    e.engine.puck_x[:], e.engine.puck_y[:] = 0.08, 0.8
    e.engine.puck_vx[:], e.engine.puck_vy[:] = 0.0, 2.0
    for _ in range(25):
        obs, *_ = e.step(np.zeros((4, 4), dtype=np.float32))
    assert obs.shape[1] == BatchAirHockeyEnv.HISTORY_OBS_DIM == 27
    py = obs[0, 1:11:2]
    gaps = -np.diff(py)
    assert np.all(gaps > 0), f"history not newest-first: {py}"
    assert 0.012 < gaps[0] < 0.030, f"10 ms gap reads {gaps[0]*1000:.0f} mm at 2 m/s"


def test_profile_v_action_caps_bind_and_stay_inside_the_machine():
    """Dims 2-3 command per-segment speed/accel caps as fractions of the
    machine's. A 5% command must crawl; a 100% command must not exceed what
    the machine could ever do (the Teensy LIMITS clamp on the table)."""
    from airhockey.batch_env import BatchAirHockeyEnv
    e = BatchAirHockeyEnv(n_envs=2, obs_mode="history", action_mode="profile_v")
    e.reset(seed=1)
    for arr in (e.engine.paddle_agent_x, e._agent_dyn["x"]):
        arr[:] = 0.3
    for arr in (e.engine.paddle_agent_y, e._agent_dyn["y"]):
        arr[:] = 0.3
    acts = np.array([[1, 1, 1, 1], [1, 1, -1, -1]], dtype=np.float32)
    peak = 0.0
    for _ in range(30):
        e.step(acts)
        peak = max(peak, float(np.hypot(e._agent_dyn["vx"][0], e._agent_dyn["vy"][0])))
    full = float(e.engine.paddle_agent_x[0]) - 0.3
    crawl = float(e.engine.paddle_agent_x[1]) - 0.3
    assert full > 3 * crawl, f"caps did not bind: full {full:.3f} vs 5% {crawl:.3f}"
    assert peak <= float(e._agent_dyn["max_speed"][0]) + 1e-6, "exceeded machine cap"


def test_history_mode_with_sensing_off_is_clean_truth():
    """Clean-sim history: zero latency, truth frames — the diagnostic
    configuration must not smuggle in any sensing corruption."""
    from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
    e = BatchAirHockeyEnv(n_envs=2, obs_mode="history", action_mode="profile_v",
                          **sensing_kwargs(False))
    e.reset(seed=2)
    assert np.all(e._cam_lag == 0)
    e.engine.puck_x[:], e.engine.puck_y[:] = 0.4, 0.6
    e.engine.puck_vx[:], e.engine.puck_vy[:] = 1.0, 0.0
    obs, *_ = e.step(np.zeros((2, 4), dtype=np.float32))
    # newest frame is exactly truth
    np.testing.assert_allclose(obs[0, 0], e.engine.puck_x[0], atol=1e-6)
