import numpy as np

from airhockey.arrival_training import teacher_rollout
from airhockey.legacy_practice import LegacyPracticeEnv, convert_arrival_demonstrations
from airhockey.skill_benchmark import make_fixtures, SkillTrials


def test_legacy_observation_prefix_and_executed_command_semantics():
    env = LegacyPracticeEnv(2, realistic=False, randomize=False)
    obs = env.reset(seed=4)
    assert obs.shape == (2, 42)
    saved = []
    original = env.base._make_obs

    def capture():
        result = original()
        saved.append(result.copy())
        return result

    env.base._make_obs = capture
    action = np.array([[0.1, -0.2, 0.3], [-0.2, 0.1, 0.4]], np.float32)
    obs, *_ = env.step(action)
    np.testing.assert_allclose(obs[:, :22], saved[-1], atol=1e-6)
    np.testing.assert_allclose(obs[:, 15:18], action)


def test_converted_teacher_commands_reproduce_physical_motion():
    f = make_fixtures(20261123, 1)
    _, _, raw = teacher_rollout(f, SkillTrials().seed_arrival(f), record=True, seed=31)
    episodes = [raw[i] for i in range(3)]
    env = LegacyPracticeEnv(3, realistic=False, randomize=False)
    env.reset(seed=31, fixtures=f)
    converted = convert_arrival_demonstrations(
        episodes, env.base._ws, env.base._action_low, env.base._action_high
    )
    for t in range(1, min(len(ep["obs"]) for ep in episodes)):
        obs, *_ = env.step(np.stack([ep["action"][t] for ep in converted]))
        expected = np.stack([ep["obs"][t, :15] for ep in episodes])
        np.testing.assert_allclose(obs[:, :15], expected, atol=0.002)


def test_predictive_guard_avoids_impulsive_boundary_stops():
    env = LegacyPracticeEnv(
        8, game_fraction=1, realistic=False, randomize=False, motion_guard=True
    )
    env.reset(seed=17, opponent="idle")
    rng = np.random.default_rng(33)
    peak = 0.0
    original = env.base.motion_callback

    def measure(dt, old, other):
        nonlocal peak
        original(dt, old, other)
        d = env.base._agent_dyn
        peak = max(
            peak,
            np.linalg.norm(np.column_stack((d["vx"], d["vy"])) - old, axis=1).max()
            / dt,
        )

    env.base.motion_callback = measure
    for _ in range(100):
        env.step(rng.uniform(-1, 1, (8, 3)))
    assert env.guard_interventions[0].sum() > 0
    assert peak < 60.1
    assert env.guard_unresolved[0].sum() == 0


def test_guard_allows_escape_from_inside_conservative_wall_margin():
    from airhockey.motion_guard import guard_command
    from airhockey.motion import CartState

    cart = CartState(1)
    cart.reset([191.6545], [300.0])
    command, cap, _, unresolved = guard_command(
        cart,
        np.array([[0.5, 0.3]]),
        np.array([30.0]),
        np.array([[0.1916545, 0.3]]),
        np.array([30.0]),
        bounds=[0.1911545, 0.8088455, 0.0989553, 0.658289],
        max_accel=np.array([60.0]),
        max_speed=np.array([12.0]),
    )
    assert not unresolved.any()
    assert command[0, 0] > 0.192
    assert cap[0] <= 60


def test_integer_machine_limit_does_not_quantize_runtime_commands():
    integer = LegacyPracticeEnv(
        4, accel=60, randomize=False, realistic=False, motion_guard=True
    )
    floating = LegacyPracticeEnv(
        4, accel=60.0, randomize=False, realistic=False, motion_guard=True
    )
    integer.reset(seed=193)
    floating.reset(seed=193)
    rng = np.random.default_rng(53)
    for _ in range(30):
        action = rng.uniform(-1, 1, (4, 3))
        left, *_ = integer.step(action)
        right, *_ = floating.step(action)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            integer.base._agent_dyn["command_accel"],
            floating.base._agent_dyn["command_accel"],
        )
    assert np.any(integer.base._agent_dyn["command_accel"] % 1 != 0)


def test_guard_survives_recorded_mpc_acceleration_cap_reversals():
    from airhockey.skill_benchmark import Fixtures

    # This sequence produced 244 m/s² with integer queued-cap buffers.
    actions = np.array(
        [
            [0.1486836374, 0.726849556, 0.529900968],
            [-0.313373893, 0.470717132, -0.932637036],
            [0.39479053, 0.926968098, 0.952405453],
            [-0.471735656, 0.238368064, -0.85144341],
            [0.377921641, 0.558940411, 0.971782804],
            [-0.823366165, 0.062018074, -0.745654106],
            [0.038414937, 0.251368433, 0.962894082],
            [0.531465769, -0.873894334, -0.01009085],
            [0.938943505, -0.958553314, 0.477810889],
            [0.963055372, -0.952376664, -0.090845101],
            [-0.950628877, -0.311468452, -0.066474602],
        ],
        np.float32,
    )
    env = LegacyPracticeEnv(
        1, accel=60, randomize=False, realistic=False, motion_guard=True
    )
    f = Fixtures(
        np.array([2]),
        np.array([[0.5, 1.7, 0.0, 0.0]]),
        np.array([[0.5068576965, 0.25]]),
        np.array([0.5]),
    )
    env.reset(seed=91, fixtures=f)
    env.task[:] = 3
    for action in actions:
        env.step(action[None])
    assert env.peak_accel.max() < 60.1
