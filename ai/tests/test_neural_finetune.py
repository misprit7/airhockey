"""RL fine-tuning pieces: teacher targets/opponents, optimizer controls, init."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import sys

import numpy as np
import pytest
import torch

from airhockey.neural_finetune import (
    ATTACK, GUARD, THREAT, Teacher, adapt_learning_rate, complete_initialization,
    initialization_model, reference_slots, same_coordinates, std_floor, teacher_mask,
    teacher_opponent_slots, teacher_regression, teacher_weight)
from airhockey.neural_player import NeuralPlayer, log_probability
from airhockey.neural_setup import workspace_bounds

REPO = Path(__file__).resolve().parents[2]
RAIL30 = dict(accel=100.0, workspace_bounds_mm=workspace_bounds("rail30"))
LEGACY = dict(accel=60.0, workspace_bounds_mm=workspace_bounds("legacy"))


def game_layout(n_envs, workers, game_fraction):
    """ParallelNeuralEnv's games: each worker's first round(count*fraction)."""
    count = n_envs // workers
    local = np.arange(n_envs) % count
    return local < round(count * game_fraction)


def test_teacher_slots_leave_reference_slots_and_spread_over_workers():
    game = game_layout(2048, 16, .3)
    reserved = reference_slots(2048, ["a reference"])
    slots = teacher_opponent_slots(game, .3, reserved)
    assert slots.sum() == round(.3 * game.sum())
    assert not (slots & ~game).any() and not (slots & reserved).any()
    per_worker = slots.reshape(16, -1).sum(1)
    assert per_worker.max() - per_worker.min() <= 1
    np.testing.assert_array_equal(slots, teacher_opponent_slots(game, .3, reserved))


def test_teacher_cannot_take_more_than_the_unreserved_game_slots():
    game = game_layout(64, 4, .5)
    reserved = reference_slots(64, ["a reference"])
    assert teacher_opponent_slots(game, .75, reserved).sum() == round(.75 * game.sum())
    with pytest.raises(ValueError, match="free of fixed references"):
        teacher_opponent_slots(game, .8, reserved)
    # Without fixed references every game slot is available.
    assert teacher_opponent_slots(game, 1., reference_slots(64, [])).sum() == game.sum()
    assert not teacher_opponent_slots(game, 0., reserved).any()


def test_teacher_weight_is_linear_in_environment_steps_then_constant():
    assert teacher_weight(0, 1., .2, 100) == 1.
    assert teacher_weight(50, 1., .2, 100) == pytest.approx(.6)
    assert teacher_weight(400, 1., .2, 100) == pytest.approx(.2)
    assert teacher_weight(400, .7, .7, 0) == .7


def test_teacher_masks_follow_expert_modes():
    modes = np.array([THREAT, ATTACK, GUARD, THREAT])
    np.testing.assert_array_equal(teacher_mask(modes, "all"), [1, 1, 1, 1])
    np.testing.assert_array_equal(teacher_mask(modes, "defense"), [1, 0, 1, 1])
    np.testing.assert_array_equal(teacher_mask(modes, "threat"), [1, 0, 0, 1])
    np.testing.assert_array_equal(teacher_mask(torch.as_tensor(modes), "attack").numpy(), [0, 1, 0, 0])
    with pytest.raises(ValueError):
        teacher_mask(modes, "everything")


def test_teacher_regression_uses_only_selected_samples():
    mean = torch.zeros(3, 6, requires_grad=True)
    target = torch.tensor([[.5] * 6, [0.] * 6, [-1.] * 6])
    assert float(teacher_regression(mean, target).detach()) == pytest.approx((.25 + 0 + 1) / 3)
    selected = teacher_regression(mean, target, torch.tensor([True, False, True]))
    assert float(selected.detach()) == pytest.approx((.25 + 1) / 2)
    nothing = teacher_regression(mean, target, torch.zeros(3, dtype=torch.bool))
    nothing.backward()
    assert float(nothing.detach()) == 0 and float(mean.grad.abs().sum()) == 0


def test_adaptive_learning_rate_moves_toward_target_within_bounds():
    assert adapt_learning_rate(1e-4, .1, .02, 1e-6, 1e-3) == pytest.approx(1e-4 / 1.5)
    assert adapt_learning_rate(1e-4, .015, .02, 1e-6, 1e-3, stopped_early=True) == pytest.approx(1e-4 / 1.5)
    assert adapt_learning_rate(1e-4, .002, .02, 1e-6, 1e-3) == pytest.approx(1.5e-4)
    assert adapt_learning_rate(1e-4, .015, .02, 1e-6, 1e-3) == 1e-4
    assert adapt_learning_rate(9e-4, 0., .02, 1e-6, 1e-3) == 1e-3
    assert adapt_learning_rate(1.2e-6, 1., .02, 1e-6, 1e-3) == 1e-6


def test_std_floor_is_one_likelihood_for_rollout_and_update():
    torch.manual_seed(3)
    mean = torch.randn(5, 6)
    log_std = torch.full((5, 6), math.log(.02))
    floored = std_floor(log_std, .1)
    assert torch.allclose(floored.exp(), torch.full((5, 6), .1))
    assert std_floor(log_std, 0) is log_std
    assert torch.equal(std_floor(torch.zeros(2, 6), .1), torch.zeros(2, 6))
    raw = mean + floored.exp() * torch.randn(5, 6)
    assert torch.equal(log_probability(raw, mean, floored),
                       log_probability(raw, mean, std_floor(log_std.clone(), .1)))


def distilled_state(width=32, critic=False, prefix=""):
    torch.manual_seed(11)
    net = NeuralPlayer(width, history=16, shot_conditioned=True, history_deltas=True)
    model = {prefix + k: v for k, v in net.state_dict().items()
             if critic or not (k.startswith("critic.") or k == "log_std")}
    return net, dict(model=model, algorithm="expert_dagger_v1", step=7)


def test_distilled_checkpoint_initializes_without_architecture_fields_or_critic():
    source, state = distilled_state(prefix="_orig_mod.")
    model, architecture = initialization_model(state)
    assert architecture == dict(width=32, history=16, shot_conditioned=True)
    target = NeuralPlayer(32, history=16, shot_conditioned=True, history_deltas=True)
    fresh = {k: v.clone() for k, v in target.state_dict().items()}
    target.load_weights(complete_initialization(model, fresh))
    torch.testing.assert_close(target.critic.weight, fresh["critic.weight"])
    torch.testing.assert_close(target.log_std, fresh["log_std"])
    obs = torch.randn(4, target.obs_dim)
    torch.testing.assert_close(target.actor(target.trunk(obs)), source.actor(source.trunk(obs)))


def test_production_style_checkpoint_passes_through_unchanged():
    source, state = distilled_state(critic=True)
    state.update(width=32, history=16, shot_conditioned=True, algorithm="successful_neural_trajectory_imitation_v1")
    model, architecture = initialization_model(state)
    fresh = NeuralPlayer(32, history=16, shot_conditioned=True).state_dict()
    completed = complete_initialization(model, fresh)
    assert completed.keys() == state["model"].keys()
    assert all(completed[k] is state["model"][k] for k in completed)
    with pytest.raises(ValueError, match="actor.weight"):
        initialization_model(dict(model={k: v for k, v in model.items() if k != "actor.weight"}))


class FixedPlayer:
    history, shot_conditioned = 16, True

    def __init__(self, options, action):
        self.environment_options, self.action, self.seen = options, action, None

    def act(self, obs, stochastic=False):
        assert not stochastic
        self.seen = obs.copy()
        self.last_mode = np.full(len(obs), GUARD)
        return np.repeat(self.action[None], len(obs), 0)


def test_teacher_in_another_workspace_is_seen_in_run_coordinates():
    from airhockey.neural_coordinates import action_coordinates, observation_coordinates

    action = np.array([.3, -.4, .1, .2, -.5, .6], np.float32)
    obs = np.random.default_rng(2).uniform(-1, 1, (3, 42 * 16 + 3)).astype(np.float32)
    same = Teacher(FixedPlayer(RAIL30, action), RAIL30)
    assert same_coordinates(RAIL30, dict(RAIL30)) and not same.translated
    np.testing.assert_array_equal(same.act(obs)[0], action)
    np.testing.assert_array_equal(same.player.seen, obs)
    np.testing.assert_array_equal(same.last_mode, [GUARD] * 3)
    moved = Teacher(FixedPlayer(RAIL30, action), LEGACY)
    assert moved.translated
    np.testing.assert_allclose(moved.act(obs), action_coordinates(np.repeat(action[None], 3, 0), RAIL30, LEGACY), rtol=1e-6)
    np.testing.assert_allclose(moved.player.seen, observation_coordinates(obs, LEGACY, RAIL30, 16), rtol=1e-6)


FAST_DONES = dict(stage=4, realistic=False, randomize=False, fixed_practice_roles=True,
                  practice_defense_fraction=1., defense_min_speed=8., defense_max_speed=12.,
                  game_fraction=.5, game_episode_seconds=.2, seed=7)


def test_worker_auto_reset_reproduces_explicit_resets_exactly():
    from airhockey.neural_vector import ParallelNeuralEnv

    envs = [ParallelNeuralEnv(8, 2, auto_reset=auto, **FAST_DONES) for auto in (False, True)]
    try:
        first = [env.reset() for env in envs]
        np.testing.assert_array_equal(*first)
        rng = np.random.default_rng(5)
        resets = 0
        for _ in range(40):
            action = rng.uniform(-1, 1, (8, 6)).astype(np.float32)
            opponent = rng.uniform(-1, 1, (8, 6)).astype(np.float32)
            results = []
            for env in envs:
                env.set_opponent_action(opponent)
                if env.auto_reset:
                    env.step_async(action)
                    out = env.step_wait()
                else:
                    out = env.step(action)
                done = out[2] | out[3]
                restarted = env.reset(mask=done) if done.any() else None
                results.append((out, restarted, env.opponent_obs(), env.critic_context()))
            (a, ra, oa, ca), (b, rb, ob, cb) = results
            for x, y in zip(a[:4], b[:4]):
                np.testing.assert_array_equal(x, y)
            assert a[4].keys() == b[4].keys()
            for key in a[4]:
                np.testing.assert_array_equal(a[4][key], b[4][key])
            np.testing.assert_array_equal(oa, ob)
            np.testing.assert_array_equal(ca, cb)
            if ra is not None:
                resets += 1
                np.testing.assert_array_equal(ra, rb)
        assert resets >= 3
    finally:
        for env in envs:
            env.close()


def test_auto_reset_must_be_consumed_with_the_done_mask():
    from airhockey.neural_vector import ParallelNeuralEnv

    env = ParallelNeuralEnv(4, 2, auto_reset=True, **FAST_DONES)
    try:
        env.reset()
        for _ in range(60):
            env.set_opponent_action(np.zeros((4, 6), np.float32))
            _, _, term, trunc, _ = env.step(np.zeros((4, 6), np.float32))
            done = term | trunc
            if done.any():
                break
        assert done.any()
        with pytest.raises(RuntimeError, match="not reset"):
            env.step(np.zeros((4, 6), np.float32))
        with pytest.raises(RuntimeError, match="done mask"):
            env.reset(mask=~done)
        env.reset(mask=done)
        env.step(np.zeros((4, 6), np.float32))
    finally:
        env.close()


def save_player(path, net, keep_critic, **fields):
    path.parent.mkdir(parents=True)
    model = {k: v for k, v in net.state_dict().items() if keep_critic or not k.startswith("critic.")}
    torch.save(dict(model=model, **fields), path)
    (path.parent / "run.json").write_text(json.dumps(dict(
        args=dict(accel=100.0, workspace="rail30"), workspace_bounds_mm=workspace_bounds("rail30"),
        physical_limits=dict(speed_m_s=12, acceleration_m_s2=100.0))))


def test_trainer_finetunes_a_distilled_student_with_teacher_and_new_optimizer(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("ai", "shared", "fw"):
        (root / name).symlink_to(REPO / name)
    runs = root / "runs"
    # The teacher's recorded environment (expert_player.PRODUCTION_ENVIRONMENT).
    save_player(runs / "rail30-100-20261001-v1" / "agent.pt", NeuralPlayer(8), True, width=8)
    torch.manual_seed(5)
    student = NeuralPlayer(32, history=16, shot_conditioned=True, history_deltas=True)
    save_player(runs / "_distilled" / "agent.pt", student, False, algorithm="expert_dagger_v1",
                step=99, args=dict(accel=100.0, workspace="rail30"))
    save_player(runs / "_reference" / "agent.pt", student, True, width=32, history=16,
                shot_conditioned=True, step=1)
    thermal = root / "thermal.json"
    shutil.copy(REPO / "ai/recipes/motor-load-20261001.json", thermal)
    spec = importlib.util.spec_from_file_location("trainer_e2e", root / "ai/bin/train_neural_player.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(trainer, "ROOT", root)
    monkeypatch.setattr(sys, "argv", [
        "train_neural_player.py", "--run-name", "_e2e", "--n-envs", "16", "--workers", "2",
        "--rollout", "8", "--epochs", "2", "--minibatch", "32", "--steps", "384", "--stage", "5",
        "--accel", "100", "--workspace", "rail30", "--thermal-model", str(thermal),
        "--history-deltas", "--initialize-from", str(runs / "_distilled" / "agent.pt"),
        "--opponent-checkpoint", str(runs / "_reference" / "agent.pt"), "--game-fraction", ".5",
        "--teacher", "expert", "--teacher-weight", "1", "--teacher-weight-final", ".5",
        "--teacher-anneal-steps", "256", "--teacher-mask", "defense", "--teacher-opponent-fraction", ".5",
        "--value-warmup-updates", "1", "--actor-steps-min", "2", "--adaptive-lr", "--lr", "1e-4",
        "--std-floor", ".1", "--device", "cpu", "--torch-threads", "1", "--save-every", "100000"])
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    threads = torch.get_num_threads()
    try:
        trainer.main()
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
        torch.set_num_threads(threads)
    run = runs / "_e2e"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == [1, 2, 3]
    assert rows[0]["value_warmup"] and rows[0]["actor_updates"] == 0
    assert not rows[1]["value_warmup"] and rows[1]["actor_updates"] >= 2
    assert [r["teacher_weight"] for r in rows] == pytest.approx([.75, .5, .5])
    for r in rows:
        assert np.isfinite(r["teacher_mse"]) and np.isfinite(r["teacher"]["mse_final"])
        assert 1e-6 <= r["next_actor_lr"] <= 3e-4
        assert set(r["game_scores"]) == {"pool", "reference", "teacher"}
        assert min(r["std"]) >= .1 - 1e-6
    meta = json.loads((run / "run.json").read_text())
    assert meta["initialization_algorithm"] == "expert_dagger_v1"
    # Game slots {0..3, 8..11}; references keep id % 4 == 0.
    assert meta["teacher"]["opponent_slots"] == 4 and meta["teacher"]["game_slots"] == 8
    assert meta["teacher"]["training_only"] and not meta["teacher"]["runtime_controller"]
    saved = torch.load(run / "agent.pt", weights_only=False)
    assert saved["teacher_anneal_origin"] == 0 and saved["actor_lr"] == rows[-1]["next_actor_lr"]
    # The initialized actor is the student's; the critic is fresh.
    assert not torch.equal(saved["model"]["critic.weight"], student.state_dict()["critic.weight"])
