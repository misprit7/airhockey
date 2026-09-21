"""Replay games must execute arrival actions and leave training state untouched."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airhockey.arrival_recording import pending_checkpoints, record_game
from airhockey.recorder import Recorder
from airhockey.arrival_env import ArrivalEnv
from airhockey.batch_env import _OPP_POLICY_MAP


class FakeAgent:
    def __init__(self, fail=False, n=2):
        self.cfg = SimpleNamespace(mpc=False)
        self._prev_mean_batch = torch.ones(2, 8, 6)
        self.calls = 0
        self.fail = fail
        self.n = n

    def act(self, obs, t0, eval_mode):
        assert self.cfg.mpc and eval_mode
        assert obs.shape == (self.n, 45)
        assert (t0 == (self.calls == 0)).all()
        assert obs[0, 36] == 1  # full game task, not a two-second skill fixture
        self.calls += 1
        self._prev_mean_batch = torch.rand(self.n, 8, 6)
        if self.fail:
            raise RuntimeError("test inference failure")
        return torch.zeros(self.n, 6)


@pytest.mark.parametrize("fail", [False, True])
def test_records_actual_game_and_preserves_training_state(tmp_path, fail):
    agent = FakeAgent(fail)
    mean = agent._prev_mean_batch
    rng = torch.get_rng_state().clone()
    np_rng = np.random.get_state()
    if fail:
        with pytest.raises(RuntimeError, match="test inference failure"):
            record_game(agent, 500000, "test", directory=tmp_path, duration=0.1)
        assert not list(tmp_path.iterdir())
    else:
        path = record_game(agent, 500000, "test", directory=tmp_path, duration=0.1)
        frames = Recorder.load(path)
        assert len(frames) == 6
        assert frames[0].time == 0 and frames[-1].time == pytest.approx(0.1)
        assert (
            frames[0].agent_x != frames[-1].agent_x
            or frames[0].agent_y != frames[-1].agent_y
        )
        data = json.loads(path.read_text())
        assert data["metadata"]["fps"] == 50
        assert data["metadata"]["opponent"] == "self"
        assert data["metadata"]["opponent_planner"]
        assert data["metadata"]["opponent_body"] == "robot"
        assert (
            frames[0].opponent_x != frames[-1].opponent_x
            or frames[0].opponent_y != frames[-1].opponent_y
        )
        assert data["metadata"]["simulation_only"]
        assert not list(tmp_path.glob("*.tmp"))
        assert np.isfinite(list(data["columns"].values())).all()
    assert agent._prev_mean_batch is mean and not agent.cfg.mpc
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])


def test_watcher_uses_immutable_milestones_and_skips_existing(tmp_path):
    run = tmp_path / "arrival-run"
    run.mkdir()
    out = tmp_path / "recordings"
    out.mkdir()
    for name in (
        "agent.pt",
        "agent_step_0500000.pt",
        "agent_step_1000000.pt",
        "agent_step_1100000.pt",
    ):
        (run / name).touch()
    assert [s for s, _ in pending_checkpoints(run, directory=out)] == [1000000, 500000]
    (out / "arrival-run_step_1000000.json").write_text("{}")
    # In-progress writes are never treated as published recordings/checkpoints.
    (out / "arrival-run_step_0500000.tmp").touch()
    (run / "agent_step_1500000.tmp").touch()
    assert [s for s, _ in pending_checkpoints(run, directory=out)] == [500000]
    (run / "agent_interrupted.pt").touch()
    (run / "status.json").write_text(json.dumps({"step": 1234567, "complete": False}))
    pending = pending_checkpoints(run, directory=out)
    assert [s for s, _ in pending] == [1234567, 500000]
    assert pending[0][1].name == "agent_interrupted.pt"
    # Scripted diagnostics must not suppress the default self-play recording.
    assert [
        s for s, _ in pending_checkpoints(run, directory=out, opponent="sniper")
    ] == [1234567, 1000000, 500000]


def test_scripted_recording_has_a_distinct_filename_and_explicit_metadata(tmp_path):
    path = record_game(
        FakeAgent(n=1),
        500000,
        "test",
        directory=tmp_path,
        duration=0.04,
        opponent="sniper",
    )
    assert path.name == "test_vs_sniper_step_0500000.json"
    meta = json.loads(path.read_text())["metadata"]
    assert meta["opponent"] == "sniper" and not meta["opponent_planner"]


def test_recording_opponent_is_selected_before_reset_initializes_its_body():
    env = ArrivalEnv(4, game_fraction=1, realistic=False, randomize=False)
    env.reset(seed=3, opponent="sniper")
    assert (env.base._opp_policy_id == _OPP_POLICY_MAP["sniper"]).all()
    assert env.base._opp_free.all()
    np.testing.assert_allclose(
        env.engine.paddle_opp_y, env.cfg.height - env.base.SNIPER_STATION_Y
    )
    np.testing.assert_allclose(env.base._opp_dyn_free["y"], env.engine.paddle_opp_y)


def test_selfplay_drives_each_robot_with_its_own_action_row(tmp_path, monkeypatch):
    import airhockey.arrival_recording as recording

    captured = []

    def make_env(*args, **kwargs):
        env = ArrivalEnv(*args, **kwargs)
        captured.append(env)
        return env

    class DistinctActions(FakeAgent):
        def act(self, obs, t0, eval_mode):
            super().act(obs, t0, eval_mode)
            # Both see a robot with the same physical limits and a game request.
            np.testing.assert_allclose(obs[0, 12:15], obs[1, 12:15])
            np.testing.assert_allclose(obs[0, 33:37], obs[1, 33:37])
            return torch.tensor(
                [[0.2, -0.3, 0.1, 0.0, -0.2, 0.4], [-0.4, 0.1, -0.2, 0.3, 0.5, 0.7]]
            )

    monkeypatch.setattr(recording, "ArrivalEnv", make_env)
    record_game(DistinctActions(), 1, "test", directory=tmp_path, duration=0.04)
    env = captured[0]
    np.testing.assert_allclose(env.last_action, [[0.2, -0.3, 0.1, 0.0, -0.2, 0.4]])
    np.testing.assert_allclose(env.last_opp_action, [[-0.4, 0.1, -0.2, 0.3, 0.5, 0.7]])
    assert env.base._opp_policy_id[0] == _OPP_POLICY_MAP["external"]
    assert not env.base._opp_free[0]
    assert env.base._agent_dyn["type"] == env.base._opp_dyn["type"] == "profile"
    np.testing.assert_allclose(
        env.base._agent_dyn["max_accel"], env.base._opp_dyn["max_accel"]
    )
