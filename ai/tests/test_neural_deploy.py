"""Offline deployment checks. All camera and hardware interfaces are fakes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from airhockey.arrival import copy_cart
from airhockey.arrival_env import ArrivalEnv
from airhockey.dynamics import sim_to_table_mm, table_mm_to_sim
from airhockey.heuristics import Command
from airhockey.motion_guard import predict
from airhockey.neural_deploy import NeuralPolicy, LiveMotorLoad, resolve_neural_checkpoint, neural_limits
from airhockey.neural_player import NeuralPlayer
from airhockey.neural_training import NeuralTrainingEnv
from test_run_policy import rp, _install_fake_camera, _loop_args, _FakeClient


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    net = NeuralPlayer(width=16, shot_conditioned=True)
    path = tmp_path / 'agent_step_10.pt'
    torch.save(dict(model=net.state_dict(), width=16, history=1, shot_conditioned=True), path)
    (tmp_path / 'run.json').write_text(json.dumps(dict(
        algorithm='neural_ppo_v1', action_mode='arrival', deployment_ready=False,
        simulation_only=True, args={'workspace':'rail30'}, physical_limits=dict(speed_m_s=12, acceleration_m_s2=60))))
    return path


@pytest.fixture
def policy(checkpoint):
    return NeuralPolicy(checkpoint, 12000, 60000, shot_mode='straight')


@pytest.mark.parametrize('setting', [
    {'workspace_bounds_mm': [1200, 1937.5, 61.4, 904.5]},
    {'args': {'workspace': 'rail30'}},
])
def test_live_adapter_uses_expanded_checkpoint_coordinates(checkpoint, setting):
    meta_path = checkpoint.parent / 'run.json'
    meta = json.loads(meta_path.read_text())
    meta.update(setting)
    meta_path.write_text(json.dumps(meta))
    assert neural_limits(checkpoint) == (12., 60.)
    policy = NeuralPolicy(checkpoint, 12000, 60000)
    np.testing.assert_allclose(policy.workspace_bounds_mm, [1200,1937.5,61.4,904.5])


def report(t=1.0, own=(0.5, 0.3), puck=(0.55, 0.6)):
    return dict(t_s=t, mallet=sim_to_table_mm(*own), opponent=sim_to_table_mm(0.3, 1.6),
                puck=[(*sim_to_table_mm(*puck), t)], own_fresh=True)


def test_feature_refactor_preserves_original_layout_both_sides():
    env = NeuralTrainingEnv(8, shot_conditioned=True, realistic=False, randomize=False)
    env.reset(seed=59)
    rng = np.random.default_rng(41)
    env.last_action[:] = rng.uniform(-1, 1, (8, 6))
    env.last_opp_action[:] = rng.uniform(-1, 1, (8, 6))
    for side in (False, True):
        env.loads[int(side)].observed[:] = rng.uniform(0, 1.2, (8, 2, 4))
        base = env.base.opponent_obs() if side else env.base._make_obs_direct()
        raw = ArrivalEnv._features(env, base, side)
        old = np.concatenate((raw[:, :21], raw[:, 25:33], raw[:, 39:44]), axis=1)
        rel = np.column_stack((base[:, :2] - base[:, 4:6], base[:, 2:4] - base[:, 6:8]))
        rival = np.column_stack((base[:, 8:10] - base[:, 4:6], base[:, 10:12] - base[:, 6:8]))
        old[:, [2, 3, 6, 7, 10, 11]] /= 6
        rel[:, 2:] /= 6
        rival[:, 2:] /= 6
        expected = np.column_stack((np.clip(np.column_stack((old, rel, rival)), -10, 10),
                                   env.base._shot_onehot(env.base._shot_type_opp if side else env.base._shot_type)))
        np.testing.assert_allclose(env._features(base, side), expected, atol=2e-7)


def test_real_report_features_and_decoded_command_match_sim(policy):
    env = NeuralTrainingEnv(1, shot_conditioned=True, realistic=False, randomize=False)
    env.reset(seed=2)
    b, e = env.base, env.engine
    positions = [(0.57, 0.62), (0.45, 0.31), (0.42, 1.6)]
    velocities = [(0.2, -0.5), (0.1, 0.2), (-0.3, 0.1)]
    old = np.asarray(positions) - np.asarray(velocities) * .02
    first = report(.98, old[1], old[0]); first['opponent'] = sim_to_table_mm(*old[2])
    policy.observe(first)
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
    dyn['max_speed'][:], dyn['max_accel'][:] = 12, 60
    dyn['command_x'][:], dyn['command_y'][:], dyn['command_accel'][:] = .6, .4, 32
    env.last_action[:] = policy.last_action[:] = [.1, -.2, .3, -.4, .5, -.6]
    env.loads[0].observed[:] = np.arange(8).reshape(1, 2, 4) / 10
    policy.loads.model.observed[:] = env.loads[0].observed
    b._shot_type[:] = 3
    policy.cart = env._cart(dyn)
    policy.motion_t = 1.0
    policy.target = np.array([[.6, .4]])
    policy.command_cap = np.array([32.])
    current = report(1.0, positions[1], positions[0]); current['opponent'] = sim_to_table_mm(*positions[2])
    current['puck'] = [(*sim_to_table_mm(*(np.asarray(positions[0]) - np.asarray(velocities[0]) * k * .005)), 1.0-k*.005) for k in range(7)]
    expected = env._features(b._make_obs_direct())
    command = policy(current)
    np.testing.assert_allclose(policy.last_obs, expected[0], atol=2e-6)
    action = policy.net.act(expected)
    np.testing.assert_allclose(policy.last_action, action[0], atol=1e-6)
    target, cap = env._decode(action)
    np.testing.assert_allclose(table_mm_to_sim(command.x_mm, command.y_mm), target[0], atol=2e-6)
    assert command.accel_mm_s2 == pytest.approx(cap[0] * 1000, rel=1e-5)


def test_command_observer_uses_sent_caps_and_latency_including_holds(policy):
    policy.observe(report())
    original = copy_cart(policy.cart)
    old_target = policy.target.copy()
    sent = Command(*sim_to_table_mm(.6, .4), 5000, 22000)
    policy.on_command(1.0, sent)
    policy._advance(1.02)
    # Actual previous target runs for 15 ms, sent target for the remaining 5 ms.
    predict(original, old_target, 60, 12, .015, policy.decoder.bounds)
    predict(original, np.array([[.6, .4]]), 22, 5, .005, policy.decoder.bounds)
    for field in ('x', 'y', 'vx', 'vy', 'ax', 'ay'):
        np.testing.assert_allclose(getattr(policy.cart, field), getattr(original, field), atol=.02)
    np.testing.assert_allclose(policy.target, [[.6, .4]])
    assert policy.command_cap[0] == 22
    policy.on_command(1.02, Command(*sim_to_table_mm(.5, .3), 5000, 22000))
    policy.reset()
    policy._advance(1.04)
    np.testing.assert_allclose(policy.target, [[.5, .3]])
    assert policy.loads.model.h.max() > 0


def test_rms_channel_order_freshness_invalid_partial_and_reset(policy):
    now = time.monotonic()
    snapshot = {'motors': [dict(node=i, rms_pct=dict(valid=True, value=10+20*i, end=now-.02),
                                    rms_slow_pct=dict(valid=i != 2, value=90+i, end=now-.03)) for i in range(4)]}
    policy.update_motor_load({'sample': snapshot, 'logging_ok': True}, now)
    obs = policy.loads.features(now)[0]
    np.testing.assert_allclose(obs[:4], [.1, .3, .5, .7])
    np.testing.assert_allclose(obs[4:], [.9, .91, .8, .93])
    assert policy.loads.fresh.sum() == 7
    before = policy.loads.model.h.copy()
    policy.reset()
    np.testing.assert_array_equal(before, policy.loads.model.h)
    # Replay of the same timestamp must not repeatedly overwrite model cooling.
    policy.loads.model.h[:] = .2
    policy.update_motor_load({'sample': snapshot, 'logging_ok': True}, now)
    assert (policy.loads.model.h == .2).all()
    policy.loads.features(now+1)
    assert not policy.loads.fresh.any()
    policy.update_motor_load({'motors': [dict(node=0, rms_pct=dict(valid=True, value=0, end=now-2))]}, now)
    assert policy.loads.values[0, 0] == .1


def test_shot_request_has_no_none_and_stays_fixed_per_possession(policy):
    policy.shot_mode = 'mix'
    seen = set()
    for i in range(15):
        policy.observe(report(1 + i*.04, puck=(.5, 1.1)))
        first = policy.observe(report(1.02 + i*.04))[-3:].copy()
        seen.add(tuple(first))
        assert first.sum() == 1
    assert len(seen) >= 2
    first = policy.last_obs[-3:].copy()
    for i in range(10):
        np.testing.assert_array_equal(policy.observe(report(1.60+i*.02))[-3:], first)


def test_stale_own_state_holds_without_actor_and_resumes(policy, monkeypatch):
    policy(report())
    act = policy.net.act
    monkeypatch.setattr(policy.net, 'act', lambda _: pytest.fail('actor ran on stale own state'))
    stale = report(1.02); stale['own_fresh'] = False
    a = policy(stale)
    stale['t_s'] = 1.04
    b = policy(stale)
    assert a.as_tuple() == b.as_tuple()
    assert policy.last_flags == ['neural_own_state_stale']
    monkeypatch.setattr(policy.net, 'act', act)
    policy(report(1.06))
    assert policy.last_obs is not None


def test_resolve_caps_and_reject_mismatched_options(checkpoint):
    args = argparse.Namespace(policy=f'neural:{checkpoint}', speed=None, accel=None, gentle=False)
    assert rp.resolve_limits(args) == (12000, 60000)
    assert args.resolved_checkpoint == str(checkpoint)
    args.accel = 40000
    assert rp.resolve_limits(args) == (12000, 40000)
    args.accel = 61000
    with pytest.raises(ValueError): rp.resolve_limits(args)
    args.accel = None; args.ramp = 6
    with pytest.raises(ValueError): rp.resolve_limits(args)
    with pytest.raises(ValueError): resolve_neural_checkpoint('latest')
    with pytest.raises(ValueError): rp.load_policy(f'neural:{checkpoint}', rp.Caps(), cmd_hz=100)


def test_imitation_arrival_artifact_uses_same_actor_and_limit_checks(checkpoint):
    original = NeuralPolicy(checkpoint, 12000, 60000, shot_mode='straight')
    path = checkpoint.parent / 'run.json'
    metadata = json.loads(path.read_text())
    metadata['algorithm'] = 'successful_neural_trajectory_imitation_v1'
    path.write_text(json.dumps(metadata))
    distilled = NeuralPolicy(checkpoint, 12000, 60000, shot_mode='straight')
    observation = np.random.default_rng(71).normal(size=(8, 45)).astype(np.float32)
    np.testing.assert_array_equal(original.net.act(observation), distilled.net.act(observation))
    with pytest.raises(ValueError): NeuralPolicy(checkpoint, 12000, 61000)
    metadata['action_mode'] = 'position'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError): NeuralPolicy(checkpoint, 12000, 60000)
    metadata.update(action_mode='arrival', algorithm='unknown')
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError): NeuralPolicy(checkpoint, 12000, 60000)


def test_report_only_supplies_controller_velocity_when_selected_and_fresh():
    r = rp.ReportBuilder()
    r.set_controller_mallet(1.0, 1600, 400, velocity=(20, -40))
    r.add_mallet(1.0, 1600, 400)
    assert r.observation(1.0)['controller_velocity'] == (20, -40)
    r.add_mallet(1.1, 1600, 400)
    assert r.observation(1.1)['controller_velocity'] is None
    assert r.observation(1.1)['own_fresh']
    assert not r.observation(10)['own_fresh']


def test_neural_runner_with_fake_camera_logs_inputs_and_holds(checkpoint, monkeypatch, tmp_path):
    import csv
    _install_fake_camera(monkeypatch, blank_from_s=.5, duration_s=2)
    # Failing constructors make any accidental access to physical hardware explicit.
    import airhockey.hardware as hardware
    monkeypatch.setattr(hardware, 'CDPRClient', lambda: pytest.fail('hardware opened'))
    assert rp.run(_loop_args(policy=f'neural:{checkpoint}', cmd_hz=50, shot_type='mix',
                             speed=12000, accel=60000, log_dir=str(tmp_path))) == 0
    with next(tmp_path.glob('*.ticks.csv')).open() as f:
        rows = list(csv.DictReader(f))
    active = [row for row in rows if row['neural_obs']]
    assert len(active) >= 10
    assert all(len(row['neural_obs'].split('|')) == 45 for row in active)
    assert all(len(row['neural_action'].split('|')) == 6 for row in active)
    assert any(row['flags'] == 'puck_hold' for row in rows)


def test_neural_live_loop_with_fake_hardware_feedback_load_and_goal_reset(policy, monkeypatch, tmp_path):
    from airhockey import hardware

    def puck_path(t):
        if t < .3:
            return 1900 + 600*t, 480
        return None if t < 1 else (1200 - 100*(t-1), 480)

    _install_fake_camera(monkeypatch, duration_s=1.5, puck_path=puck_path)
    import track_mallet
    monkeypatch.setattr(track_mallet, 'measure', lambda: (1600, 400))

    class Client(_FakeClient):
        loads = 0
        def connect(self): pass
        def set_ramp(self, ramp): pass
        def get_motor_load(self, **kwargs):
            self.loads += 1
            now = time.monotonic()
            return {'sample': {'motors': [dict(node=i, rms_pct=dict(valid=True, value=51+i, end=now),
                               rms_slow_pct=dict(valid=True, value=30+i, end=now)) for i in range(4)]}}
        def command_position(self, x, y, v, a=None):
            self.calls.append(('CMD', x, y, v, a))

    client = Client()
    monkeypatch.setattr(hardware, 'CDPRClient', lambda: client)
    monkeypatch.setattr(rp, '_shutdown', lambda *args: None)
    monkeypatch.setattr(rp, 'load_policy', lambda *args, **kwargs: policy)
    sent, decisions, resets = [], [], []
    original_command, original_reset, original_observe = policy.on_command, policy.reset, policy.observe
    def feedback(t, command, **timing):
        sent.append((t, command))
        original_command(t, command, **timing)
    def reset():
        resets.append(True)
        original_reset()
    def observe(report):
        obs = original_observe(report)
        decisions.append((report['t_s'], obs.copy()))
        return obs
    monkeypatch.setattr(policy, 'on_command', feedback)
    monkeypatch.setattr(policy, 'reset', reset)
    monkeypatch.setattr(policy, 'observe', observe)
    assert rp.run(_loop_args(live=True, no_enable=True, cmd_hz=50, speed=12000,
                            accel=60000, log_dir=str(tmp_path))) == 0
    commands = [c for c in client.calls if c[0] == 'CMD']
    assert len(sent) == len(commands) > 30
    assert client.loads >= 2  # metadata plus live LOAD
    assert policy.loads.fresh.all()
    assert resets
    for (_, feedback), actual in zip(sent, commands):
        assert (feedback.x_mm, feedback.y_mm) == actual[1:3]
        if actual[4] is not None:
            assert feedback.accel_mm_s2 == actual[4]
    events = [json.loads(s) for s in next(tmp_path.glob('*.replay.jsonl')).read_text().splitlines()]
    pause, resume = [e for e in events if e['type'] == 'puck_watchdog']
    assert all(not pause['t'] <= t < resume['t'] for t, _ in decisions)
    first_resumed = next(obs for t, obs in decisions if t >= resume['t'])
    np.testing.assert_array_equal(first_resumed[15:21], 0)
    np.testing.assert_allclose(first_resumed[21:25], [.51, .52, .53, .54])


def test_gentle_caps_and_warmup_leave_clean_runtime_state(policy, checkpoint):
    args = argparse.Namespace(policy=f'neural:{checkpoint}', speed=None, accel=None, gentle=True)
    args.speed, args.accel = rp.resolve_limits(args)
    caps = rp.session_caps(args)
    assert caps.accel_min <= .05 * caps.accel_max
    gentle = NeuralPolicy(checkpoint, caps.speed_max, caps.accel_max)
    assert np.isfinite(gentle.warm_up())
    assert gentle.motion_t is None and not gentle.pending
    assert gentle.command_cap[0] == caps.accel_max / 1000
    assert gentle.last_obs is None
    np.testing.assert_array_equal(gentle.last_action, 0)


def test_controller_sample_projection_advances_position_velocity_and_braking_state(policy):
    policy.observe(report())
    policy.target = np.array([[.8,.4]])
    policy.command_cap[:] = 30
    policy.command_history = [(.5,policy.target.copy(),policy.command_cap.copy(),12)]
    # The timestamped sample is 20 ms old and moving toward the target.
    from airhockey.deploy import mm_velocity_to_sim
    sample = dict(position=sim_to_table_mm(.5,.3), velocity=(-500,1200), age_s=.02)
    expected = copy_cart(policy.cart)
    expected.vx[:], expected.vy[:] = np.array(mm_velocity_to_sim(*sample['velocity']))*1000
    predict(expected,policy.target,30,12,.02,policy.decoder.bounds)
    actual = policy.project_controller(sample,1.0)
    for field in ('x','y','vx','vy','ax','ay'):
        np.testing.assert_allclose(getattr(actual,field),getattr(expected,field),atol=1e-5)
    assert actual.x[0] > policy.cart.x[0]+20
    # Encoder must see the same projected position, not a camera/controller mix.
    inp=report(1.02,own=(.1,.6));inp['controller_sample']=sample
    policy.observe(inp)
    np.testing.assert_allclose(policy.last_obs[4:6], [policy.cart.x[0]/1000,policy.cart.y[0]/1000],atol=1e-7)
    assert policy.state_source=='controller_projected'


def test_projection_honors_command_changes_inside_sample_age(policy):
    policy.observe(report())
    sample=dict(position=report()['mallet'],velocity=(0,0),age_s=.03)
    first=np.array([[.7,.3]]);second=np.array([[.3,.3]])
    policy.command_history=[(.9,first,np.array([30.]),12),(1.01,second,np.array([60.]),12)]
    expected=copy_cart(policy.cart)
    predict(expected,first,30,12,.01,policy.decoder.bounds)
    predict(expected,second,60,12,.02,policy.decoder.bounds)
    got=policy.project_controller(sample,1.03)
    np.testing.assert_allclose(got.x,expected.x,atol=.001)
    np.testing.assert_allclose(got.vx,expected.vx,atol=.01)


def test_replay_keeps_marker_evidence_near_apparent_contact(tmp_path):
    from types import SimpleNamespace
    from airhockey.replay_log import ReplayLog
    p=tmp_path/'contact.jsonl'
    replay=ReplayLog(p,SimpleNamespace(policy='test',live=False,ramp=3))
    report=rp.ReportBuilder();report.add_mallet(1,1600,400);report.add_puck(1,1640,400)
    tracker=SimpleNamespace(rejected_jumps=0,n_markers=2,frame_puck_members=np.array([1,2]))
    replay.tracking_diagnostics(1,tracker,np.array([[1,2,3],[4,5,6],[7,8,9]]),report)
    replay.close()
    e=next(r for r in map(json.loads,p.read_text().splitlines()) if r['type']=='near_contact_tracking')
    assert e['type']=='near_contact_tracking' and e['puck_markers']==2
    assert e['puck_members']==[1,2] and len(e['blobs_px'])==3


def test_command_observer_uses_measured_delivery_instead_of_training_delay(policy):
    policy.observe(report())
    policy.on_command(1.0,Command(*sim_to_table_mm(.65,.4),12000,30000),delivery_delay=.003)
    policy._advance(1.005)
    np.testing.assert_allclose(policy.target,[[.65,.4]])
    assert policy.command_cap[0]==30
    assert policy.command_history[-1][0]==pytest.approx(1.003)


def test_legacy_checkpoint_keeps_original_coordinates(checkpoint):
    meta_path = checkpoint.parent / 'run.json'
    meta = json.loads(meta_path.read_text())
    meta.pop('args')
    meta_path.write_text(json.dumps(meta))
    policy = NeuralPolicy(checkpoint, 12000, 60000)
    np.testing.assert_allclose(policy.workspace_bounds_mm, [1350,1917.5,172.9,793])


def test_firmware_must_cover_checkpoint_before_enabling():
    from airhockey.neural_deploy import verify_firmware_workspace
    required = (1200,1937.5,61.4,904.5)
    verify_firmware_workspace(required, required)
    with pytest.raises(ValueError, match='smaller than the policy'):
        verify_firmware_workspace(required, (1350,1917.5,172.9,793))


def test_checkpoint_thermal_model_is_used(checkpoint):
    meta_path = checkpoint.parent / 'run.json'
    meta = json.loads(meta_path.read_text())
    model_path = Path(__file__).parents[1] / 'recipes/motor-load-20261001.json'
    meta['thermal_model'] = json.loads(model_path.read_text())
    meta_path.write_text(json.dumps(meta))
    policy = NeuralPolicy(checkpoint, 12000, 60000)
    assert policy.loads.model.spatial
    np.testing.assert_allclose(policy.loads.model.limits[0], meta['thermal_model']['fast_limit_amps'])
