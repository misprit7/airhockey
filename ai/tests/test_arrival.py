"""Executed commands, exact impulses and honest isolated-attempt metrics."""
import numpy as np
import pytest

from airhockey.arrival import ArrivalDecoder
from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.motion import CartState
from airhockey.skill_benchmark import DECISIONS, Fixtures, SkillTrials, TrialPhysics, make_fixtures


def test_decoder_does_not_teleport_or_mutate_queued_command():
    decoder = ArrivalDecoder((0.2, 0.8, 0.1, 0.65))
    cart = CartState(2)
    cart.reset([400, 600], [300, 300])
    cart.vx[:] = [100, -200]
    cart.ax[:] = [3000, -5000]
    before = {k: getattr(cart, k).copy() for k in CartState.__slots__}
    previous = np.array([[0.5, 0.4], [0.4, 0.2]])
    target, accel = decoder.decode(cart, np.array([[1, 1, 1, 1, -1, 1],
        [-1, -1, -1, -1, 1, -1]]), 0, previous, np.array([60, 60]))
    for k in before:
        np.testing.assert_array_equal(getattr(cart, k), before[k])
    assert np.all(target >= decoder.low) and np.all(target <= decoder.high)
    np.testing.assert_allclose(accel, [60, 3])
    np.testing.assert_array_equal(previous, [[0.5, 0.4], [0.4, 0.2]])


def test_new_arrival_interrupts_previous_direction():
    decoder = ArrivalDecoder((0.2, 0.8, 0.1, 0.65))
    cart = CartState(1)
    cart.reset(500, 300)
    previous = np.array([[0.5, 0.3]])
    left, _ = decoder.decode(cart, np.array([[-1, 0, 0, 0, 0, 1]]), 0, previous, np.array([60]))
    right, _ = decoder.decode(cart, np.array([[1, 0, 0, 0, 0, 1]]), 0, previous, np.array([60]))
    assert left[0, 0] < 0.5 < right[0, 0]


def test_reachable_arrival_states_execute_with_small_position_and_velocity_error():
    # Stops, straight strikes and a diagonal terminal velocity, under the
    # real firmware profile and the same delayed 50 Hz command scheduler.
    trials = SkillTrials()
    n = 9
    f = Fixtures(np.zeros(n, dtype=int), np.tile([0.5, 0.9, 0, 0], (n, 1)),
                 np.tile([0.5, 0.25], (n, 1)), np.full(n, 0.5))
    times = np.repeat([0.14, 0.18, 0.22], 3)
    velocities = np.tile([[0, 0], [0, 1.5], [0.3, 1.5]], (3, 1))
    position = 2 * (np.array([0.5, 0.4]) - trials.decoder.low) / (
        trials.decoder.high - trials.decoder.low) - 1
    actions = np.column_stack((np.tile(position, (n, 1)), velocities / 3,
                               2 * (times - 0.04) / 0.21 - 1, np.full(n, 0.7)))
    result = trials.rollout(f, actions, "arrival")
    assert np.max(result["arrival_position_error"]) < 0.003
    assert np.max(result["arrival_velocity_error"]) < 0.15
    assert not np.any(result["backstop_violation"])


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((2, 6), np.nan)])
def test_invalid_arrivals_rejected(bad):
    with pytest.raises(ValueError):
        ArrivalDecoder((0.2, 0.8, 0.1, 0.65)).unpack(bad)


def test_contact_events_are_impulses_not_proximity_and_preserve_physics():
    engines = [BatchPhysicsEngine(2), BatchPhysicsEngine(2)]
    events = []
    engines[0].contact_callback = events.append
    for eng in engines:
        eng.puck_x[:] = eng.paddle_agent_x[:] = 0.5
        eng.puck_y[:] = 0.38
        eng.paddle_agent_y[:] = 0.3
        eng.puck_vy[:] = [-1, 1]  # approaching vs separating, both overlap
        eng.paddle_opp_x[:] = eng.paddle_opp_y[:] = -10
        eng.step(0.0025)
    assert len(events) == 1
    ev = events[0]
    assert ev["body"] == "agent"
    np.testing.assert_allclose(ev["time"], [0.0025])
    np.testing.assert_array_equal(ev["indices"], [0])
    assert ev["incoming"][0, 1] < 0 < ev["outgoing_before_speed_cap"][0, 1]
    np.testing.assert_allclose(ev["normal"], [[0, 1]])
    for key in ("puck_x", "puck_y", "puck_vx", "puck_vy"):
        np.testing.assert_array_equal(getattr(engines[0], key), getattr(engines[1], key))


def test_decoded_commands_reproduce_identical_physics_with_original_interface():
    trials = SkillTrials()
    f = make_fixtures(11, 2)
    arrival = trials.rollout(f, trials.seed_arrival(f), "arrival", record=True)
    position = trials.rollout(f, arrival["commands"], "position", record=True)
    np.testing.assert_allclose(arrival["trace"], position["trace"], atol=2e-5)
    np.testing.assert_array_equal(arrival["contact"], position["contact"])
    np.testing.assert_allclose(arrival["effort"], position["effort"], rtol=1e-4)


def test_whiff_counts_as_failed_attempt_and_has_no_fake_contact():
    trials = SkillTrials()
    f = Fixtures(np.array([0]), np.array([[0.65, 0.55, 0, 0]]),
                 np.array([[0.3, 0.25]]), np.array([0.5]))
    cfg = trials.cfg
    low = np.full(2, cfg.paddle_radius)
    high = np.array([cfg.width, cfg.height / 2]) - low
    action = np.zeros((1, DECISIONS, 3))
    action[:, :, :2] = 2 * (f.paddle[:, None] - low) / (high - low) - 1
    result = trials.rollout(f, action, "position")
    assert not result["contact"][0] and not result["goal"][0]
    assert not result["cushion"][0]
    assert np.isnan(result["contact_time"][0])
    assert result["loss"][0] > 4


def test_fixtures_seeded_inside_workspace_without_initial_overlap():
    trials = SkillTrials()
    f = make_fixtures(5, 100)
    other = make_fixtures(5, 100)
    np.testing.assert_array_equal(f.puck, other.puck)
    assert np.all(f.paddle >= trials.decoder.low)
    assert np.all(f.paddle <= trials.decoder.high)
    assert np.all(np.linalg.norm(f.puck[:, :2] - f.paddle, axis=1) >
                  trials.cfg.puck_radius + trials.cfg.paddle_radius)


def test_trial_goal_is_counted_once_without_inventing_a_serve():
    eng = TrialPhysics(1)
    eng.puck_x[:] = 0.5
    eng.puck_y[:] = 2.001
    eng.puck_vy[:] = 2
    eng.step(0.0025)
    assert eng.goal_scored[0] == 1 and eng.score_agent[0] == 1
    position = (eng.puck_x.copy(), eng.puck_y.copy())
    for _ in range(20):
        eng.step(0.0025)
        assert eng.goal_scored[0] == 0 and eng.score_agent[0] == 1
        np.testing.assert_array_equal(eng.puck_x, position[0])
        np.testing.assert_array_equal(eng.puck_y, position[1])


def test_large_collinear_arrival_jacobian_retains_damping():
    from airhockey.arrival import damped_velocity_step

    jx = np.array([[2000, 2000], [4000, 3999]], dtype=np.float32)
    jy = np.array([[2000, 2000], [4001, 4000]], dtype=np.float32)
    residual = np.array([[1, 1], [1, -1]], dtype=np.float32)
    actual = damped_velocity_step(jx, jy, residual)
    expected = []
    for x, y, r in zip(jx, jy, residual):
        matrix = np.vstack((np.column_stack((x, y)), np.eye(2) * np.sqrt(.1)))
        expected.append(np.linalg.lstsq(matrix, np.r_[r, 0, 0], rcond=None)[0])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)
