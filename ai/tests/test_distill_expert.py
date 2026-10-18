"""DAgger distillation: recorded curriculum, roll-in, buffer, actor-only fit, labels, checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from airhockey.distillation import (PRODUCTION, STUDENT, TEACHER, DistillationEnv, LabelBuffer, RollIn,
                                    StandardizedInputs, _label, actor_parameters, distillation_loss, fit,
                                    heldout_report, learning_rate, parse_mix, widened)
from airhockey.neural_player import NeuralPlayer, PhysicalHistory, PhysicalHistoryLinear

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai/bin"))
PRODUCTION_CHECKPOINT = ROOT / "runs/rail30-100-20261001-v1/agent.pt"
needs_production = pytest.mark.skipif(not PRODUCTION_CHECKPOINT.exists(), reason="production run not present")


def production_options(**changes):
    from train_neural_player import environment_options, stored_arguments
    args = stored_arguments(PRODUCTION_CHECKPOINT.parent / "run.json")
    options = environment_options(args, shot_conditioned=True)
    options.update(changes)
    return options


@needs_production
def test_stored_production_command_reproduces_its_curriculum():
    meta = json.loads((PRODUCTION_CHECKPOINT.parent / "run.json").read_text())
    options = production_options()
    assert options["workspace_bounds_mm"] == meta["workspace_bounds_mm"]
    assert options["accel"] == 100 and options["stage"] == 5 and options["game_fraction"] == .3
    assert options["thermal_gain"] == 1.3 and options["shutdown_level"] == .95 and options["terminate_overload"]
    assert options["report_sensing"] and options["continuous_rallies"] and options["project_rail_contacts"]
    assert options["practice_selfplay_fraction"] == .25 and options["recovery_fraction"] == .5
    assert (options["defense_min_speed"], options["defense_max_speed"]) == (8, 16)
    assert isinstance(options["thermal_path"], Path) and options["thermal_path"].name == "motor-load-20261001.json"
    # A flag added after this run was trained takes its default.
    assert options["defense_block_only"] is False


def test_recipe_argv_and_partial_args_both_parse(tmp_path):
    from train_neural_player import stored_arguments
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(dict(argv=["/usr/bin/python", "ai/bin/train_neural_player.py", "--run-name", "r",
                                            "--stage", "4", "--game-fraction", ".2", "--report-sensing"])))
    args = stored_arguments(recipe)
    assert (args.stage, args.game_fraction, args.report_sensing, args.accel) == (4, .2, True, 60)
    run = tmp_path / "run.json"
    run.write_text(json.dumps(dict(args=dict(run_name="r", stage=5, stationary_replay="logs/x.npz",
                                             thermal_model="/abs/model.json"))))
    args = stored_arguments(run)
    assert args.stage == 5 and args.stationary_replay == Path("logs/x.npz")
    assert args.thermal_model == Path("/abs/model.json") and args.workspace == "legacy"


def test_rollin_blocks_are_coherent_and_redrawn_at_episode_end():
    rng = np.random.default_rng(1)
    rollin = RollIn(4000, rng, block_steps=5, opponent_block_steps=7)
    mix = parse_mix("0.4,0.3,0.3")
    never = np.zeros(4000, bool)
    rollin.update(1.0, mix, never)
    assert rollin.teacher.all()
    rollin.restart()
    rollin.update(0.0, mix, never)
    assert not rollin.teacher.any()
    assert np.allclose(np.bincount(rollin.opponent, minlength=3) / 4000, [.4, .3, .3], atol=.03)
    rollin.restart()
    history = []
    for _ in range(40):
        rollin.update(0.5, mix, never)
        history.append(rollin.teacher.copy())
    history = np.array(history)
    assert .45 < history.mean() < .55
    # The first block is staggered; after it a choice changes only on a 5-decision lattice.
    changes = np.diff(history.astype(int), axis=0) != 0
    gaps = [np.diff(np.flatnonzero(column)) for column in changes.T]
    assert sum(len(g) for g in gaps) > 4000 and all((g % 5 == 0).all() for g in gaps if len(g))
    # An episode end redraws immediately, whatever is left of the block.
    done = np.zeros(4000, bool)
    done[:2000] = True
    before = rollin.teacher.copy()
    rollin.update(0.0, mix, done)
    assert not rollin.teacher[:2000].any() and (rollin.teacher[2000:] == before[2000:]).mean() > .7


def test_label_buffer_keeps_the_most_recent_states():
    buffer = LabelBuffer(10, 3)

    def rows(start, n, heldout=False, round_index=0):
        values = np.arange(start, start + n, dtype=np.float16)[:, None].repeat(3, 1)
        buffer.add(values, np.zeros((n, 6)), mode=1, kind=3, side=0, actor=TEACHER, round=round_index,
                   heldout=heldout)

    rows(0, 6)
    rows(6, 7, heldout=np.arange(7) % 2 == 0, round_index=1)
    assert buffer.size == 10 and buffer.added == 13
    assert sorted(buffer.obs[:, 0].tolist()) == list(range(3, 13))
    held = buffer.obs[buffer.indices(heldout=True), 0].tolist()
    assert sorted(held) == [6, 8, 10, 12]
    assert sorted(buffer.obs[buffer.indices(heldout=False, rounds=[0]), 0].tolist()) == [3, 4, 5]
    rows(100, 25)  # more than the capacity: only the newest survive
    assert sorted(buffer.obs[:, 0].tolist()) == list(range(115, 125)) and buffer.size == 10


def test_fit_moves_only_the_actor_toward_the_teacher():
    torch.manual_seed(0)
    net = NeuralPlayer(32, history=2, shot_conditioned=True, history_deltas=True)
    untouched = {k: v.clone() for k, v in net.state_dict().items()
                 if k.startswith(("log_std", "noise", "value_trunk", "critic"))}
    trained = {k: v.clone() for k, v in net.state_dict().items() if k.startswith(("trunk", "actor"))}
    rng = np.random.default_rng(0)
    obs = rng.normal(0, .5, (6000, net.obs_dim)).astype(np.float16)
    teacher = np.tanh(obs[:, :6].astype(np.float32) * 2 - .3)
    buffer = LabelBuffer(10000, net.obs_dim)
    buffer.add(obs, teacher, mode=rng.integers(0, 3, 6000), kind=0, side=0, actor=TEACHER, round=0,
               heldout=rng.random(6000) < .1)
    before = heldout_report(net, buffer, device="cpu")["mse"]
    optimizer = torch.optim.Adam(actor_parameters(net), lr=3e-3)
    stats = fit(net, optimizer, buffer, epochs=15, batch_size=256, lr=3e-3, final_fraction=.1,
                weights=[1, 1, 1, 1, 1, 1], device="cpu", warmup=10)
    report = heldout_report(net, buffer, device="cpu")
    assert report["mse"] < before / 5 and stats["epoch_losses"][-1] < stats["epoch_losses"][0]
    assert set(report["by_mode"]) == {"threat", "attack", "guard"}
    state = net.state_dict()
    assert all(torch.equal(state[k], v) for k, v in untouched.items())
    assert all(not torch.equal(state[k], v) for k, v in trained.items())


def test_standardized_first_layer_folds_back_exactly_and_trains():
    torch.manual_seed(3)
    net = NeuralPlayer(32, history=3, shot_conditioned=True, history_deltas=True)
    rng = np.random.default_rng(3)
    obs = (rng.normal(0, .5, (4000, net.obs_dim)) * rng.uniform(.001, 2, net.obs_dim)).astype(np.float16)
    teacher = np.tanh(obs[:, [0, 1, 2, 3, 4, 5]].astype(np.float32) * 3)
    buffer = LabelBuffer(5000, net.obs_dim)
    buffer.add(obs, teacher, mode=rng.integers(0, 3, 4000), kind=0, side=0, actor=TEACHER, round=0,
               heldout=rng.random(4000) < .1)
    standardizer = StandardizedInputs.from_buffer(net.trunk[0], buffer)
    probe = torch.as_tensor(obs[:64].astype(np.float32))
    with torch.no_grad():
        np.testing.assert_allclose(standardizer(probe), net.trunk[0](probe), rtol=1e-4, atol=1e-4)
    before = heldout_report(net, buffer, device="cpu")["mse"]
    optimizer = torch.optim.Adam(actor_parameters(net, standardizer), lr=3e-3)
    first = net.trunk[0]
    fit(net, optimizer, buffer, epochs=10, batch_size=256, lr=3e-3, final_fraction=.1, weights=[1] * 6,
        device="cpu", mode_weights=[3, 2, 1], standardizer=standardizer)
    assert net.trunk[0] is first and isinstance(first, PhysicalHistoryLinear)
    with torch.no_grad():
        np.testing.assert_allclose(standardizer(probe), first(probe), rtol=1e-4, atol=1e-4)
    assert heldout_report(net, buffer, device="cpu")["mse"] < before / 3
    assert set(net.state_dict()) == set(NeuralPlayer(32, history=3, shot_conditioned=True).state_dict())


def test_mode_weights_weigh_samples_not_dimensions():
    net = NeuralPlayer(8, history=1, shot_conditioned=True)
    obs = torch.zeros(4, net.obs_dim)
    target = torch.tensor([[1.] * 6, [1.] * 6, [-1.] * 6, [-1.] * 6])
    ones = torch.ones(6)
    errors = torch.tensor([.25, .25, 2.25, 2.25])  # tanh(mean) = 0.5 against +-1
    weights = torch.tensor([3., 3., 1., 1.])
    with torch.no_grad():
        net.actor.weight.zero_()
        net.actor.bias.fill_(float(np.arctanh(.5)))
        assert float(distillation_loss(net, obs, target, ones)) == pytest.approx(float(errors.mean()), rel=1e-4)
        assert float(distillation_loss(net, obs, target, ones, weights)) == pytest.approx(
            float((errors * weights).sum() / weights.sum()), rel=1e-4)


def test_load_bands_even_out_the_observed_load_distribution():
    from airhockey.distillation import load_balance_table
    buffer = LabelBuffer(1000, 42 * 2 + 3)
    obs = np.zeros((1000, buffer.obs_dim), np.float16)
    obs[:900, 21:29] = .85  # crowded hot band
    obs[900:, 21:29] = .62  # sparse band near the guard-depth bend
    obs[:, 25] = 0          # the maximum over the eight channels counts
    buffer.add(obs, np.zeros((1000, 6)), mode=2, kind=0, side=0, actor=TEACHER, round=0, heldout=False)
    table = load_balance_table(buffer, buffer.indices(heldout=False), cap=10)
    hot, sparse = table[int(.85 * 20)], table[int(.62 * 20)]
    assert float(hot * 900) == pytest.approx(float(sparse * 100)) and float(sparse) == pytest.approx(5)
    assert float(load_balance_table(buffer, buffer.indices(heldout=False), cap=2)[int(.62 * 20)]) == 2


def test_learning_rate_warms_up_then_decays_to_the_floor():
    assert learning_rate(0, 100, 1.0, .1, warmup=10) == pytest.approx(.1)
    assert learning_rate(10, 100, 1.0, .1, warmup=10) == pytest.approx(1.0)
    assert learning_rate(100, 100, 1.0, .1, warmup=10) == pytest.approx(.1)


def test_widened_student_acts_identically():
    torch.manual_seed(2)
    net = NeuralPlayer(32, history=2, shot_conditioned=True, history_deltas=True)
    wide = widened(net, 48)
    probe = torch.randn(64, net.obs_dim)
    assert wide.width == 48 and wide.history_deltas
    assert torch.allclose(wide.actor(wide.trunk(probe)), net.actor(net.trunk(probe)), atol=1e-5)


@needs_production
def test_teacher_labels_both_seats_and_marks_external_far_seats():
    from airhockey.batch_env import _OPP_POLICY_MAP
    from airhockey.expert_player import load_expert
    from airhockey.neural_training import NeuralTrainingEnv
    options = production_options(stationary_replay=None, stationary_replay_fraction=0, seed=3)
    env = NeuralTrainingEnv(10, **options)
    teacher, _ = load_expert("expert", PRODUCTION_CHECKPOINT)
    own, far = PhysicalHistory(16), PhysicalHistory(16)
    own.reset(env.reset(seed=3))
    far.reset(env.opponent_obs())
    zeros = np.zeros(10, np.int8)
    out = _label(teacher, own, far, env, _OPP_POLICY_MAP["external"], np.ones(10, bool), zeros, zeros, zeros)
    assert out["obs"].shape == (10, 675) and out["obs"].dtype == np.float16
    np.testing.assert_array_equal(out["label"], teacher.act(own.get()))
    assert out["external"][env.kind == 3].all() and out["external"][:3].all()  # 30% games are external
    np.testing.assert_array_equal(out["far_label"], teacher.act(far.get()[out["external"]]))
    assert set(np.unique(out["mode"])) <= {0, 1, 2}


@needs_production
def test_distillation_workers_step_both_seats():
    options = production_options(stationary_replay=None, stationary_replay_fraction=0, seed=5)
    env = DistillationEnv(8, 2, teacher="expert", environment_from=PRODUCTION_CHECKPOINT, **options)
    try:
        batch = env.reset()
        assert batch.obs.shape == (8, 675) and batch.done.all()
        np.testing.assert_array_equal(batch.far, np.flatnonzero(batch.external))
        assert len(batch.far_obs) == len(batch.far_label) == len(batch.far)
        for _ in range(3):
            far_action = np.zeros((8, 6), np.float32)
            far_action[batch.far] = batch.far_label
            batch = env.step(batch.label, far_action)
        assert np.isfinite(batch.obs.astype(np.float32)).all() and np.abs(batch.label).max() <= 1
        assert batch.kind.shape == (8,) and not batch.done.all()
    finally:
        env.close()


@needs_production
def test_checkpoint_loads_like_production(tmp_path):
    import distill_expert
    from eval_neural_player import load
    from airhockey.neural_setup import checkpoint_environment
    production, production_state = load(PRODUCTION_CHECKPOINT)
    parent = json.loads((PRODUCTION_CHECKPOINT.parent / "run.json").read_text())
    options = production_options()
    state = distill_expert.student_checkpoint(production, init_state=production_state,
                                              base_args=parent["args"], teacher="expert", round=0)
    torch.save(state, tmp_path / "agent.pt")
    meta = distill_expert.distilled_run_metadata(parent, options, algorithm=distill_expert.ALGORITHM,
                                                 args=parent["args"])
    (tmp_path / "run.json").write_text(json.dumps(meta))
    for key in ("workspace_bounds_mm", "thermal_model", "physical_limits", "args"):
        assert meta[key] == parent[key]
    assert "groups" not in meta and "groups" in meta["parent_provenance"]
    net, loaded = load(tmp_path / "agent.pt")
    assert loaded["algorithm"] == "expert_dagger_v1" and loaded["teacher"] == "expert"
    assert net.environment_options == production.environment_options
    assert checkpoint_environment(tmp_path / "agent.pt") == checkpoint_environment(PRODUCTION_CHECKPOINT)
    probe = np.random.default_rng(0).normal(0, .3, (16, 675)).astype(np.float32)
    np.testing.assert_allclose(net.act(probe), production.act(probe), atol=1e-6)
    assert STUDENT != PRODUCTION != TEACHER


def test_production_relabels_only_slow_attack_rows_of_the_round():
    import torch
    from airhockey.distillation import LabelBuffer, MODES, TAGS, relabel_with_network, slow_attack_rows
    from airhockey.neural_player import NeuralPlayer
    buffer = LabelBuffer(8, 675)
    obs = np.zeros((4, 675), np.float32)
    obs[1, 2:4] = (0, 4.0 / 6)  # fast but moving away: calm
    obs[2, 2:4] = (0, -4.0 / 6)  # fast and incoming: keep the teacher
    tags = {k: 0 for k in TAGS}
    tags["mode"] = np.array([MODES.index("attack")] * 3 + [MODES.index("threat")])
    buffer.add(obs, np.full((4, 6), 0.5, np.float32), **tags)
    rows = slow_attack_rows(buffer, [0], 2.5)
    assert rows.tolist() == [0, 1]
    net = NeuralPlayer(64, history=16, shot_conditioned=True)
    assert relabel_with_network(buffer, net, rows, device="cpu") == 2
    expected = net.actor(net.trunk(torch.from_numpy(obs[:2]).half().float())).tanh()
    assert torch.allclose(buffer.label[:2], expected, atol=1e-6)
    assert torch.all(buffer.label[2:4] == 0.5)
