"""A body limited below the cap a checkpoint's coordinates are fractions of.

The trainer's --body-accel, the table runner's --accel and the evaluators
(a run.json carrying body_accel) share one rule: the commanded cap and the
guard's braking are clipped, observations and every other output keep the
checkpoint's units. Offline; the deploy side runs on fakes.
"""
import json

import numpy as np
import pytest
import torch

from airhockey.dynamics import sim_to_table_mm, table_mm_to_sim
from airhockey.neural_coordinates import limit_cap
from airhockey.neural_deploy import NeuralPolicy, neural_body_accel
from airhockey.neural_player import NeuralPlayer
from airhockey.neural_setup import checkpoint_environment
from airhockey.neural_training import NeuralTrainingEnv

FULL_CAP = np.array([[.1, -.2, .3, -.4, -.5, 1.0]], np.float32)


def cap_of(action, units):
    return (0.05 + 0.95 * ((np.asarray(action)[..., 5] + 1) / 2) ** 2) * units


def test_limit_cap_clips_only_the_cap_slot_and_only_above_the_limit():
    actions = np.array([[.1, .2, .3, .4, .5, 1.0], [.1, .2, .3, .4, .5, -1.0], [0, 0, 0, 0, 0, 0.2]])
    limited = limit_cap(actions, 100.0, 30.0)
    np.testing.assert_array_equal(limited[:, :5], actions[:, :5])
    np.testing.assert_allclose(cap_of(limited, 100.0), np.minimum(cap_of(actions, 100.0), 30.0))
    assert limited is not actions and actions[0, 5] == 1.0
    assert limit_cap(actions, 100.0, None) is actions
    assert limit_cap(actions, 100.0, 100.0) is actions
    tensor = limit_cap(torch.as_tensor(actions), 100.0, 30.0)
    np.testing.assert_allclose(tensor.numpy(), limited)


def test_env_clips_the_learner_and_leaves_units_and_the_far_side_alone():
    kwargs = dict(shot_conditioned=True, realistic=False, randomize=False, accel=100.0)
    plain, limited = NeuralTrainingEnv(2, **kwargs), NeuralTrainingEnv(2, **kwargs, body_accel=30.0)
    for env in (plain, limited):
        env.reset(seed=3)
    actions = np.repeat(FULL_CAP, 2, axis=0)
    # Same state, same observation: the limit is not an input.
    np.testing.assert_array_equal(plain._features(plain.base._make_obs_direct()),
                                  limited._features(limited.base._make_obs_direct()))
    target_plain, cap_plain = plain._decode(actions)
    target, cap = limited._decode(actions)
    assert (cap_plain > 30.0).all() and (cap <= 30.0 + 1e-6).all()
    assert (limited.base._agent_dyn["max_accel"] == 100.0).all()
    # A cap already under the limit decodes exactly as without one.
    calm = actions.copy(); calm[:, 5] = -0.6
    for a, b in zip(plain._decode(calm), limited._decode(calm)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, **kwargs, body_accel=120.0)
    assert NeuralTrainingEnv(2, **kwargs, body_accel=100.0).body_accel is None


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    net = NeuralPlayer(width=16, shot_conditioned=True)
    path = tmp_path / 'agent_step_10.pt'
    torch.save(dict(model=net.state_dict(), width=16, history=1, shot_conditioned=True), path)
    (tmp_path / 'run.json').write_text(json.dumps(dict(
        algorithm='neural_ppo_v1', action_mode='arrival', deployment_ready=False, simulation_only=True,
        args={'workspace': 'rail30', 'accel': 60.0},
        physical_limits=dict(speed_m_s=12, acceleration_m_s2=60))))
    return path


def report(t, own, puck, opponent):
    return dict(t_s=t, mallet=sim_to_table_mm(*own), opponent=sim_to_table_mm(*opponent),
                puck=[(*sim_to_table_mm(*puck), t)], own_fresh=True)


def test_table_runner_under_a_lower_limit_matches_the_limited_sim(checkpoint, monkeypatch):
    """Session accel 30 on a 60-unit checkpoint: the observation is the
    60-unit one and the command is the sim's under body_accel=30."""
    policy = NeuralPolicy(checkpoint, 12000, 30000, shot_mode='straight')
    assert (policy.accel, policy.units_accel, policy.trained_body_accel) == (30.0, 60.0, 60.0)
    env = NeuralTrainingEnv(1, shot_conditioned=True, realistic=False, randomize=False, accel=60.0, body_accel=30.0)
    env.reset(seed=2)
    b, e = env.base, env.engine
    positions = [(0.57, 0.62), (0.45, 0.31), (0.42, 1.6)]
    velocities = [(0.2, -0.5), (0.1, 0.2), (-0.3, 0.1)]
    old = np.asarray(positions) - np.asarray(velocities) * .02
    policy.observe(report(.98, old[1], old[0], old[2]))
    for prefix, pos, vel in zip(('puck', 'paddle_agent', 'paddle_opp'), positions, velocities):
        for axis, value, speed in zip(('x', 'y'), pos, vel):
            getattr(e, prefix + '_' + axis)[:] = value
            getattr(e, prefix + '_v' + axis)[:] = speed
    b._prev_agent_x[:], b._prev_agent_y[:] = old[1]
    b._prev_opp_x[:], b._prev_opp_y[:] = old[2]
    dyn = b._agent_dyn
    for key, value in zip(('x', 'y', 'vx', 'vy'), (*positions[1], *velocities[1])):
        dyn[key][:] = value
    dyn['cart'].ax[:], dyn['cart'].ay[:] = 7000, -2500
    dyn['command_x'][:], dyn['command_y'][:], dyn['command_accel'][:] = .6, .4, 24
    env.last_action[:] = policy.last_action[:] = [.1, -.2, .3, -.4, .5, -.6]
    env.loads[0].observed[:] = np.arange(8).reshape(1, 2, 4) / 10
    policy.loads.model.observed[:] = env.loads[0].observed
    b._shot_type[:] = 3
    policy.cart = env._cart(dyn)
    policy.motion_t = 1.0
    policy.target = np.array([[.6, .4]])
    policy.command_cap = np.array([24.])
    current = report(1.0, positions[1], positions[0], positions[2])
    current['puck'] = [(*sim_to_table_mm(*(np.asarray(positions[0]) - np.asarray(velocities[0]) * k * .005)),
                        1.0 - k * .005) for k in range(7)]
    expected = env._features(b._make_obs_direct())
    monkeypatch.setattr(policy.net, 'act', lambda obs, **_: FULL_CAP.copy())
    command = policy(current)
    np.testing.assert_allclose(policy.last_obs, expected[0], atol=2e-6)
    # The previous-action input stays the policy's own output, not the clipped one.
    np.testing.assert_array_equal(policy.last_action, FULL_CAP[0])
    target, cap = env._decode(FULL_CAP)
    np.testing.assert_allclose(table_mm_to_sim(command.x_mm, command.y_mm), target[0], atol=2e-6)
    assert command.accel_mm_s2 == pytest.approx(cap[0] * 1000, rel=1e-5)
    assert command.accel_mm_s2 <= 30000 + 1e-3


def test_a_checkpoint_trained_under_a_limit_carries_it(checkpoint):
    meta_path = checkpoint.parent / 'run.json'
    meta = json.loads(meta_path.read_text())
    assert neural_body_accel(checkpoint) == 60.0 and 'body_accel' not in checkpoint_environment(checkpoint)
    meta['args']['body_accel'] = 25.0
    meta['physical_limits']['body_acceleration_m_s2'] = 25.0
    meta_path.write_text(json.dumps(meta))
    assert neural_body_accel(checkpoint) == 25.0
    options = checkpoint_environment(checkpoint)
    assert options['accel'] == 60.0 and options['body_accel'] == 25.0
    # Those options build the limited env directly.
    env = NeuralTrainingEnv(1, shot_conditioned=True, **{k: options[k] for k in ('accel', 'body_accel')})
    assert env.body_accel == 25.0
    policy = NeuralPolicy(checkpoint, 12000, 25000)
    assert policy.trained_body_accel == 25.0 and policy.units_accel == 60.0
