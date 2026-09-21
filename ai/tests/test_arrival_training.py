"""Arrival/RMS learning contracts; simulation and synthetic replay only."""

import json
import numpy as np
import pytest
from airhockey.arrival_env import ArrivalEnv
from airhockey.thermal import MotorThermal
from airhockey.sequence_replay import SequenceReplay, EpisodeBatch
from airhockey.arrival_training import transfer_encoder


def test_heat_memory_survives_goal_and_episode_reset():
    e = ArrivalEnv(2, randomize=False, realistic=False)
    e.reset(seed=2)
    e.loads[0].h[:] = 0.7**2
    e.loads[0].observed[:] = 0.7
    before = e.loads[0].h.copy()
    e.reset(mask=np.array([True, False]))
    np.testing.assert_array_equal(before, e.loads[0].h)
    e.engine.puck_x[:] = 0.5
    e.engine.puck_y[:] = 2.001
    e.engine.puck_vy[:] = 2
    e.engine._check_goals()
    np.testing.assert_array_equal(before, e.loads[0].h)


def test_curriculum_allocates_time_by_slots_not_episode_probability():
    e = ArrivalEnv(32, game_fraction=0.25, randomize=False, realistic=False)
    e.reset(seed=2)
    for _ in range(20):
        e.reset(mask=e.task < 3)
        assert (e.task == 3).sum() == 8
    e.game_fraction = 0.5
    e.reset()
    assert (e.task == 3).sum() == 16


def test_thermal_filter_units_and_direction_priors():
    m = MotorThermal(1, randomize=False)
    p = np.array([[0.5, 0.4]])
    z = np.zeros((1, 2))
    assert m.tau[0, 0] == pytest.approx(25.3444, abs=0.001)
    m.advance(p, z, np.array([[60.0, 0.0]]), 0.02)
    assert np.all(m.current_squared > 0), "unseen acceleration axes must not be free"
    expected = (1 - np.exp(-0.02 / m.tau)) * m.current_squared[:, None, :] / m.limits**2
    np.testing.assert_allclose(m.h, expected)
    m.h[:] = 1.1**2
    assert (
        m.penalty(0.02)[0] > 200 * 0.02
    )  # overload charge plus near-limit/energy costs


def test_load_depends_on_actual_motion_not_requested_ceiling():
    e = ArrivalEnv(1, randomize=False, realistic=False)
    e.reset(seed=2)
    p = np.column_stack((e.base._agent_dyn["x"], e.base._agent_dyn["y"]))
    a = np.zeros((1, 6))
    a[:, :2] = 2 * (p - e.decoder.low) / (e.decoder.high - e.decoder.low) - 1
    a[:, 5] = 1  # full cap, but stationary at requested position
    e.step(a)
    assert e.effort[0] < 1e-6
    assert e.loads[0].current_squared[0, 0] < 1e-5


def test_observation_actions_and_partial_reset():
    e = ArrivalEnv(4, randomize=False)
    o = e.reset(seed=3)
    assert o.shape == (4, 45) and np.isfinite(o).all()
    a = np.array([[0.2, -0.3, 0.1, -0.1, 0.4, 0.5]] * 4, np.float32)
    o, _, _, _, _ = e.step(a)
    np.testing.assert_allclose(o[:, 15:21], a)
    np.testing.assert_allclose(o[:, 25:33], e.loads[0].features())
    p = np.column_stack((e.engine.paddle_agent_x, e.engine.paddle_agent_y))
    e.reset(mask=np.array([True, False, False, False]))
    np.testing.assert_array_equal(p[1:, 0], e.engine.paddle_agent_x[1:])
    assert e.contacts[0] == 0


def test_arrival_motion_is_bounded_and_not_teleported():
    e = ArrivalEnv(3, randomize=False, realistic=False)
    e.reset(seed=8)
    old = np.column_stack((e.engine.paddle_agent_x, e.engine.paddle_agent_y))
    e.step(np.ones((3, 6), np.float32))
    new = np.column_stack((e.engine.paddle_agent_x, e.engine.paddle_agent_y))
    assert np.max(np.linalg.norm(new - old, axis=1)) < 0.02
    assert np.all(new >= e.decoder.low) and np.all(new <= e.decoder.high)


def test_cushion_does_not_reward_shooting_a_goal():
    e = ArrivalEnv(1, randomize=False, realistic=False)
    e.reset(seed=1)
    e.task[:] = 2
    e.engine.puck_x[:] = 0.5
    e.engine.puck_y[:] = 1.999
    e.engine.puck_vy[:] = 2
    _, reward, _, _, info = e.step(np.zeros((1, 6)))
    assert reward[0] < -20 and not info["success"][0]


def test_replay_never_crosses_episode_or_ring_overwrite():
    b = SequenceReplay(37, 2, 1, 4, 128, device="cpu", seed=2)
    for ep in range(11):
        t = np.arange(9, dtype=np.float32)
        b.add(
            dict(
                obs=np.column_stack((np.full(9, ep), t)),
                action=t[:, None],
                reward=t,
                terminated=np.zeros(9),
                demo=np.ones(9),
            )
        )
    obs, action, reward, term, task, demo = b.sample_with_demo()
    np.testing.assert_array_equal(
        obs[:, :, 0], np.broadcast_to(obs[0, :, 0], obs[:, :, 0].shape)
    )
    np.testing.assert_array_equal(np.diff(obs[:, :, 1], axis=0), 1)
    np.testing.assert_array_equal(action[:, :, 0], obs[1:, :, 1])
    assert np.isfinite(reward.numpy()).all() and np.all(demo.numpy() == 1)


def test_episode_batch_records_executed_actions_and_terminal_obs():
    b = EpisodeBatch(2, 3, 1, 10)
    b.reset(np.zeros((2, 3)))
    b.append(
        np.ones((2, 3)),
        np.array([[0.5], [0.6]]),
        np.array([1, 2]),
        np.array([True, False]),
    )
    ep = b.episode_at(0)
    assert (
        len(ep["obs"]) == 2 and ep["action"][1, 0] == 0.5 and ep["terminated"][1] == 1
    )
    b.reset(np.ones((2, 3)) * 2, mask=np.array([True, False]))
    assert list(b.length) == [0, 1]


def test_encoder_migration_does_not_reinterpret_old_actions(tmp_path):
    import torch
    from types import SimpleNamespace

    enc = torch.nn.ModuleDict({"state": torch.nn.Sequential(torch.nn.Linear(45, 4))})
    old = torch.arange(4 * 22, dtype=torch.float32).reshape(4, 22)
    p = tmp_path / "old.pt"
    torch.save(
        {
            "model": {
                "_encoder.state.0.weight": old,
                "_encoder.state.0.bias": torch.ones(4),
            }
        },
        p,
    )
    a = SimpleNamespace(model=SimpleNamespace(_encoder=enc))
    migration = transfer_encoder(a, p)
    w = enc["state"][0].weight.detach()
    np.testing.assert_array_equal(w[:, :15], old[:, :15])
    assert torch.all(w[:, 15:21] == 0) and torch.all(w[:, 25:] == 0)
    np.testing.assert_array_equal(w[:, 21:25], old[:, 18:22])
    assert "dynamics" in migration["reset"]


def test_experimental_checkpoint_does_not_replace_deployment_latest(
    tmp_path, monkeypatch
):
    from airhockey import policy_loader as p

    runs = tmp_path / "runs"
    runs.mkdir()
    old = runs / "3.12-old-selfplay"
    old.mkdir()
    (old / "agent.pt").write_text("old")
    new = runs / "4.0-arrival-selfplay"
    new.mkdir()
    (new / "agent.pt").write_text("new")
    (new / "run.json").write_text(json.dumps({"deployment_ready": False}))
    monkeypatch.setattr(p, "_REPO_ROOT", tmp_path)
    assert p.resolve_checkpoint("latest") == old / "agent.pt"
    assert [r["run"] for r in p.list_checkpoints()] == [old.name]
