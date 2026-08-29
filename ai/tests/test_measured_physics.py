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
    assert e.opponent_dynamics_type == "profile"


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
