import numpy as np
import pytest

from airhockey.neural_training import NeuralTrainingEnv
from airhockey.skill_benchmark import Fixtures
from airhockey.neural_setup import workspace_bounds


def test_edge_curriculum_samples_both_contact_fringe_sides_without_changing_inputs():
    env = NeuralTrainingEnv(256, stage=5, game_fraction=0, realistic=False,
                            randomize=False, edge_drill_fraction=1, shot_conditioned=True,
                            workspace_bounds_mm=workspace_bounds('legacy'))
    obs = env.reset(seed=5631)
    ids = env.edge_drill
    assert ids.any() and obs.shape == (256, 45)
    x = env.engine.puck_x[ids]
    assert (x < env.decoder.low[0]).any() and (x > env.decoder.high[0]).any()
    radius = env.cfg.puck_radius + env.cfg.paddle_radius
    margin = radius - np.maximum(env.decoder.low[0] + .002 - x,
                                 x - env.decoder.high[0] + .002)
    assert margin.min() >= .0005 - 1e-9
    assert (margin < .003).any()  # Includes the reported replay's narrow reach.
    assert (margin > .01).any()
    assert not env._unreachable_dead_puck()[ids].any()
    assert (env.kind[ids] == 0).all()
    assert (env.base._shot_type[ids] != 0).all()
    np.testing.assert_array_equal(env.engine.puck_vx[ids], 0)
    # Explicit benchmark fixtures must never be replaced by the curriculum.
    fixtures = Fixtures(np.zeros(256, int), np.tile([.5, .4, 0, 0], (256, 1)),
                        np.tile([.5, .2], (256, 1)), np.full(256, .5))
    env.reset(fixtures=fixtures)
    assert not env.edge_drill.any()
    np.testing.assert_allclose(env.engine.puck_x, .5)


def test_edge_recovery_requires_contact_interior_and_safe_speed_and_pays_once():
    env = NeuralTrainingEnv(4, realistic=False, randomize=False, edge_recovery_weight=80)
    env.engine.puck_x[:] = [.5, .1032, .5, .5]
    env.engine.puck_y[:] = .4
    env.engine.puck_vx[:] = [0, 0, 3, 0]
    env.engine.puck_vy[:] = 0
    env._edge_contact[0] = [False, True, True, True]
    np.testing.assert_array_equal(env._edge_recovery_event(0), [False, False, False, True])
    assert not env._edge_recovery_event(0).any()
    # A goal/serve that places the puck inside cannot manufacture recovery credit.
    env._edge_paid[0] = False
    assert not env._edge_recovery_event(0, excluded=np.ones(4, bool)).any()
    np.testing.assert_array_equal(env.edge_recovery_count[0], [0, 0, 0, 1])


def test_wall_bounce_from_reported_replay_state_recovers_without_reset_or_fake_capture():
    env = NeuralTrainingEnv(1, stage=5, game_fraction=0, realistic=True, randomize=False,
                            continuous_rallies=True, edge_recovery_weight=80,
                            workspace_bounds_mm=workspace_bounds('legacy'))
    env.reset(fixtures=Fixtures(np.array([0]), np.array([[.1032, .4306, 0, 0]]),
                               np.array([[.1932, .2113]]), np.array([.5])))
    touched_wall = False
    for tick in range(65):
        action = np.zeros((1, 6))
        action[:, 4:] = [1, -1]
        if tick < 20:
            target = [.235, .4366]
            action[:, 5] = 0
        elif tick < 45:
            target = [env.decoder.low[0], .4366]
            action[:, 2] = 0
            action[:, 4:] = [-1, 1]
        else:
            target = [.5, .2]
        action[:, :2] = 2 * (np.array(target) - env.decoder.low) / (env.decoder.high - env.decoder.low) - 1
        _, _, terminal, truncated, info = env.step(action)
        assert not terminal.any() and not truncated.any()
        touched_wall |= env.engine.puck_x[0] < .047
    assert touched_wall and env.touch_count[0, 0] > 0
    assert env.edge_recovery_count[0, 0] == 1
    assert info['edge_recoveries'][0] == 1
    assert env.capture_count[0, 0] == 0  # Recovery into open space is not control.
    assert env.turnovers.sum() == 0


def test_edge_approach_rewards_feasible_contact_alignment_without_affecting_center():
    env = NeuralTrainingEnv(3, realistic=False, randomize=False, setup_weight=0,
                            edge_approach_weight=60, workspace_bounds_mm=workspace_bounds('legacy'))
    env.engine.puck_x[:] = [.1032, .1032, .5]
    env.engine.puck_y[:] = .4306
    env.engine.puck_vx[:] = env.engine.puck_vy[:] = 0
    env.engine.paddle_agent_x[:] = .1932
    env.engine.paddle_agent_y[:] = [.21, .43, .21]
    potential = env.potential()
    assert potential[1] > potential[0] + 10
    assert potential[2] == 0


def test_edge_practice_keeps_time_for_followthrough_and_preserves_other_reset_ledgers():
    env = NeuralTrainingEnv(2, stage=5, game_fraction=0, realistic=False, randomize=False,
                            edge_drill_fraction=1, possession_followthrough=True)
    fixture = Fixtures(np.zeros(2, int), np.tile([.1032, .43, 0, 0], (2, 1)),
                       np.tile([.5, .2], (2, 1)), np.full(2, .5))
    env.reset(fixtures=fixture)
    env.edge_drill[:] = True
    env.elapsed[:] = 6.1
    action = np.zeros((2, 6))
    action[:, :2] = 2 * (np.array([.5, .2]) - env.decoder.low) / (env.decoder.high - env.decoder.low) - 1
    action[:, 4:] = [1, -1]
    _, _, terminal, _, _ = env.step(action)
    assert not terminal.any()
    assert np.all(env.critic_context()[:, 4] < .6)
    env.elapsed[:] = 12
    _, _, terminal, _, _ = env.step(action)
    assert terminal.all()
    env._edge_paid[:] = True
    env._edge_contact[:] = True
    env.reset(mask=np.array([True, False]))
    assert not env._edge_paid[:, 0].any() and env._edge_paid[:, 1].all()
    assert not env._edge_contact[:, 0].any() and env._edge_contact[:, 1].all()


def test_edge_exploration_is_coherent_local_and_absent_in_full_games():
    import torch
    from airhockey.neural_player import RecoveryExplorationBias, log_probability

    torch.manual_seed(901)
    bias = RecoveryExplorationBias(4, .8, block_steps=12, mode="edge", load_aware=False)
    obs = torch.zeros(4, 45)
    obs[:, :2] = torch.tensor([[.1032, .43], [.5, .43], [.1032, .43], [.1032, .43]])
    obs[3, 3] = -1  # Fast shot: no edge exploration.
    context = torch.zeros(4, 9)
    context[:, 0] = 1
    context[2, 0], context[2, 3] = 0, 1  # Full game.
    first = bias.sample(obs, context, torch.ones(4, dtype=torch.bool)).clone()
    assert first[0].abs().sum() > 0 and first[1:].count_nonzero() == 0
    for _ in range(11):
        torch.testing.assert_close(bias.sample(obs, context, torch.ones(4, dtype=torch.bool)), first)
    assert not torch.equal(bias.sample(obs, context, torch.ones(4, dtype=torch.bool))[0], first[0])
    # PPO conditions both likelihoods on the stored realized offset.
    mean = torch.zeros_like(first)
    log_std = torch.full_like(first, -.7)
    raw = mean + first + log_std.exp() * torch.randn_like(first)
    old = log_probability(raw, mean + first, log_std)
    assert torch.equal(log_probability(raw, mean + first, log_std), old)
    assert not torch.allclose(log_probability(raw, mean, log_std), old)
    bias.reset(torch.tensor([True, False, False, False]))
    assert bias.values[0].count_nonzero() == 0


@pytest.mark.parametrize('kwargs', [dict(edge_drill_fraction=np.nan), dict(edge_drill_fraction=1.1),
                                  dict(edge_recovery_weight=-1), dict(edge_approach_weight=np.inf)])
def test_edge_curriculum_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        NeuralTrainingEnv(1, **kwargs)
