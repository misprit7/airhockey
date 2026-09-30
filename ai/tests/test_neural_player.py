import numpy as np

from airhockey.neural_training import NeuralTrainingEnv


def test_actor_backtracking_bounds_gaussian_change_and_preserves_critic_step():
    import torch
    from airhockey.neural_player import backtrack_actor_step

    actor = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    critic = torch.nn.Parameter(torch.tensor([3.0]))
    before = [actor.detach().clone()]
    with torch.no_grad():
        actor.copy_(torch.tensor([1.0, -0.5]))
        critic.add_(2)
    # Exact KL for two independent fixed-variance Gaussian actions.
    measure = lambda: 0.5 * ((actor - before[0]) / 0.05).square().sum()
    scale, kl = backtrack_actor_step([actor], before, measure, 0.02)
    assert 0 < scale < 1 and kl <= 0.02
    torch.testing.assert_close(actor, scale * torch.tensor([1.0, -0.5]))
    assert critic.item() == 5


def test_actor_backtracking_accepts_small_step_and_restores_failed_proposal():
    import torch
    from airhockey.neural_player import backtrack_actor_step

    actor = torch.nn.Parameter(torch.tensor([0.01]))
    before = [torch.zeros(1)]
    scale, kl = backtrack_actor_step([actor], before, lambda: actor.square().sum(), .02)
    assert scale == 1 and kl < .02
    scale, _ = backtrack_actor_step([actor], before, lambda: float('nan'), .02)
    assert scale == 0 and actor.item() == 0


def test_physical_history_is_newest_first_and_partial_reset_is_isolated():
    from airhockey.neural_player import PhysicalHistory

    h = PhysicalHistory(4)
    start = np.arange(84, dtype=np.float32).reshape(2, 42)
    h.reset(start)
    h.append(start + 1)
    h.append(start + 2)
    np.testing.assert_array_equal(h.get()[0].reshape(4, 42), [start[0]+2, start[0]+1, start[0], start[0]])
    other = h.get()[1].copy()
    h.reset(start + 9, mask=np.array([True, False]))
    np.testing.assert_array_equal(h.get()[0].reshape(4, 42), np.repeat((start[0]+9)[None], 4, axis=0))
    np.testing.assert_array_equal(h.get()[1], other)


def test_adding_physical_history_preserves_policy_and_critic_initially():
    import torch
    from airhockey.neural_player import NeuralPlayer

    torch.set_num_threads(2)
    old, new = NeuralPlayer(32), NeuralPlayer(32, history=4)
    assert new.load_weights(old.state_dict())
    current, past, context = torch.randn(8, 42), torch.randn(8, 126), torch.randn(8, 9)
    expected = old.distribution(current, context)
    actual = new.distribution(torch.cat((current, past), dim=1), context)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert torch.count_nonzero(new.trunk[0].weight[:, 42:]) == 0


def test_temporal_difference_migration_preserves_learned_history_and_source():
    import torch
    from airhockey.neural_player import NeuralPlayer

    torch.set_num_threads(2)
    torch.manual_seed(572)
    raw = NeuralPlayer(32, history=4, shot_conditioned=True)
    with torch.no_grad():
        raw.noise.weight.normal_(0, .05)
        raw.noise.bias.uniform_(-.2, .2)
    source = {k: v.clone() for k, v in raw.state_dict().items()}
    delta = NeuralPlayer(32, history=4, shot_conditioned=True, history_deltas=True)
    assert delta.load_weights(raw.state_dict())
    physical = torch.randn(12, 168)
    request = torch.eye(3).repeat(4, 1)
    x = torch.cat((physical, request), dim=1)
    context = torch.randn(12, 9)
    for actual, expected in zip(delta.distribution(x, context), raw.distribution(x, context)):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for key, expected in source.items():
        torch.testing.assert_close(raw.state_dict()[key], expected, rtol=0, atol=0)
    # Default inference construction must inherit the saved encoding.
    restored = NeuralPlayer(32, history=4, shot_conditioned=True)
    assert not restored.load_weights(delta.state_dict())
    assert restored.history_deltas
    for actual, expected in zip(restored.distribution(x, context), delta.distribution(x, context)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_history_checkpoint_keeps_raw_encoding_exactly():
    import torch
    from airhockey.neural_player import NeuralPlayer

    raw = NeuralPlayer(32, history=4, shot_conditioned=True)
    legacy = dict(raw.state_dict())
    legacy.pop("_history_delta_encoding")
    restored = NeuralPlayer(32, history=4, shot_conditioned=True)
    assert not restored.load_weights(legacy)
    assert not restored.history_deltas
    x = torch.randn(8, 171)
    torch.testing.assert_close(restored(x)[0], raw(x)[0], rtol=0, atol=0)


def test_delta_history_extension_preserves_requests_context_and_zero_motion():
    import torch
    from airhockey.neural_player import NeuralPlayer, PhysicalHistoryLinear

    old = NeuralPlayer(32, history=4, shot_conditioned=True, history_deltas=True)
    extended = NeuralPlayer(32, history=8, shot_conditioned=True)
    assert extended.load_weights(old.state_dict())
    assert extended.history_deltas
    physical, past = torch.randn(10, 168), torch.randn(10, 168)
    request, context = torch.randn(10, 3), torch.randn(10, 9)
    for actual, expected in zip(
            extended.distribution(torch.cat((physical, past, request), dim=1), context),
            old.distribution(torch.cat((physical, request), dim=1), context)):
        torch.testing.assert_close(actual, expected)
    layer = PhysicalHistoryLinear(168+3+9, 7, history=4, deltas=True)
    current, tail = torch.randn(6, 42), torch.randn(6, 12)
    stationary = torch.cat((current.repeat(1, 4), tail), dim=1)
    expected = torch.nn.functional.linear(
        torch.cat((current, torch.zeros(6, 126), tail), dim=1), layer.weight, layer.bias)
    torch.testing.assert_close(layer(stationary), expected, rtol=0, atol=0)


def test_frozen_actor_prefix_survives_adam_momentum_while_new_units_learn():
    import copy
    import torch
    from airhockey.neural_player import ActorPrefixFreeze, NeuralPlayer

    torch.set_num_threads(2)
    torch.manual_seed(573)
    base = NeuralPlayer(16, shot_conditioned=True)
    expanded = NeuralPlayer(32, shot_conditioned=True)
    assert expanded.load_weights(base.state_dict())
    frozen = ActorPrefixFreeze(expanded, 16)
    optimizer = torch.optim.Adam(expanded.parameters(), lr=.001)
    # A resumed optimizer may already contain nonzero momentum. Gradient masks
    # alone do not guarantee a fixed subnetwork under Adam.
    for parameter in expanded.parameters():
        optimizer.state[parameter] = dict(step=torch.tensor(3.),
            exp_avg=torch.full_like(parameter, .1), exp_avg_sq=torch.full_like(parameter, .2))
    x, context = torch.randn(24, 45), torch.randn(24, 9)
    original_value = expanded.critic.weight.detach().clone()
    for _ in range(4):
        mean, value, std = expanded.distribution(x, context)
        loss = (mean-.3).square().mean() + (value-2).square().mean() + (std+.7).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        frozen.restore()
        for parameter, index, initial in frozen.blocks:
            torch.testing.assert_close(parameter[index], initial, rtol=0, atol=0)
    assert torch.count_nonzero(expanded.actor.weight[:, 16:]) > 0
    assert not torch.equal(expanded.critic.weight, original_value)
    torch.testing.assert_close(expanded.trunk(x)[:, :16], base.trunk(x), atol=1e-6, rtol=1e-5)
    # Ordinary checkpoints and inference copies contain no freezing controller.
    restored = NeuralPlayer(32, shot_conditioned=True)
    restored.load_weights(expanded.state_dict())
    torch.testing.assert_close(restored(x)[0], expanded(x)[0], rtol=0, atol=0)
    torch.testing.assert_close(copy.deepcopy(expanded)(x)[0], expanded(x)[0], rtol=0, atol=0)


def test_actor_prefix_freeze_rejects_coupled_or_invalid_prefixes():
    import pytest
    from airhockey.neural_player import ActorPrefixFreeze, NeuralPlayer

    net = NeuralPlayer(32)
    for width in (0, 32, 33, -1, True):
        with pytest.raises(ValueError):
            ActorPrefixFreeze(net, width)
    with pytest.raises(ValueError, match="independent"):
        ActorPrefixFreeze(net, 16)


def test_policy_cannot_observe_training_tasks_or_shot_requests():
    env = NeuralTrainingEnv(4, randomize=False, realistic=False)
    env.reset()
    raw = env.base._make_obs_direct()
    before = env._features(raw)
    env.task[:] = [0, 1, 2, 3]
    env.aim[:] = [0.4, 0.45, 0.55, 0.6]
    env.desired_speed[:] = [0, 2, 4, 6]
    env.elapsed[:] = [1, 2, 3, 4]
    raw[:, 18:22] = 100
    np.testing.assert_array_equal(before, env._features(raw))
    assert before.shape == (4, 42)


def test_partial_reset_preserves_other_observations_and_heat():
    env = NeuralTrainingEnv(4, games=True, stage=3, randomize=False, realistic=False)
    env.reset(opponent="external")
    a = np.full((4, 6), 0.4)
    env.set_opponent_action(a)
    obs, *_ = env.step(a)
    heat = env.loads[0].h.copy()
    after = env.reset(mask=np.array([True, False, False, False]))
    np.testing.assert_array_equal(after[1:], obs[1:])
    np.testing.assert_array_equal(env.loads[0].h, heat)


def test_worst_case_training_load_gain_survives_resets():
    env = NeuralTrainingEnv(4, thermal_gain=1.3)
    env.reset(seed=91)
    env.reset(mask=np.array([True, False, True, False]))
    for load_model in env.loads:
        np.testing.assert_array_equal(load_model.gain, np.full((4, 1), 1.3))


def test_singleton_bank_drill_resets_cover_both_walls():
    env = NeuralTrainingEnv(
        20,
        stage=4,
        fixed_practice_roles=True,
        randomize=False,
        realistic=False,
        seed=431,
    )
    env.reset()
    mask = np.arange(20) == 17  # Fixed bank-defense practice slot, not a game.
    directions = []
    for _ in range(40):
        env.reset(mask=mask)
        directions.append(env.engine.puck_vx[17] > 0)
    assert 8 < sum(directions) < 32


def test_focused_receiving_curriculum_uses_requested_speed_range():
    env = NeuralTrainingEnv(
        100, stage=2, receive_drill=True, randomize=False, realistic=False,
        receiving_min_speed=4, receiving_max_speed=8,
    )
    env.reset(seed=71)
    receiving = env.kind == 1
    assert receiving.sum() == 70
    speeds = -env.engine.puck_vy[receiving]
    assert np.all((speeds >= 4) & (speeds <= 8))
    assert np.count_nonzero(speeds >= 6) > 20


def test_early_volley_goal_cannot_escape_failed_reception_cost():
    from airhockey.skill_benchmark import Fixtures

    env = NeuralTrainingEnv(
        1, stage=2, receive_drill=True, capture_first=True,
        randomize=False, realistic=False, skill_goal_weight=60,
    )
    env.reset(fixtures=Fixtures(
        np.ones(1, int), np.array([[.5, 1.99, 0., 10.]]),
        np.array([[.5, .3]]), np.array([.5]),
    ))
    _, reward, terminal, _, info = env.step(np.zeros((1, 6)))
    assert terminal[0] and info["goals"][0] == 1
    assert info["captures"][0] == 0 and env.elapsed[0] < 1.2
    assert reward[0] < -1


def test_explicit_seed_reproduces_randomized_physics_and_sensing():
    options = dict(stage=5, fixed_practice_roles=True, realistic=True, seed=730)
    first, second = NeuralTrainingEnv(16, **options), NeuralTrainingEnv(16, **options)
    a, b = first.reset(seed=730), second.reset(seed=730)
    np.testing.assert_array_equal(a, b)
    action = np.random.default_rng(911).uniform(-1, 1, (16, 6))
    for _ in range(30):
        first.set_opponent_action(action)
        second.set_opponent_action(action)
        a, ra, *_ = first.step(action)
        b, rb, *_ = second.step(action)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ra, rb)


def test_random_neural_commands_respect_acceleration_and_finite_observations():
    env = NeuralTrainingEnv(16, randomize=False, realistic=False)
    env.reset()
    rng = np.random.default_rng(9101)
    for _ in range(100):
        obs, reward, _, _, info = env.step(rng.uniform(-1, 1, (16, 6)))
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert info["peak_accel"].max() < 60.1


def test_controlled_strike_gets_credit_even_soon_after_cushion_contact():
    env = NeuralTrainingEnv(1, randomize=False, realistic=False)
    env.reset()
    env.engine.puck_x[:] = 0.5
    env.engine.puck_y[:] = 0.5
    env.elapsed[:] = 0.5
    env.contact_clock[0] = 0.46
    env.captured[0] = True
    event = dict(
        body="agent",
        indices=np.array([0]),
        incoming=np.array([[0.0, 0.3]]),
        outgoing_before_speed_cap=np.array([[0.0, 4.0]]),
    )
    env._contact(event)
    assert env.convert_count[0, 0] == 1
    assert env.shot_count[0, 0] == 1
    assert not env.captured[0, 0]
    # Persistent outgoing contact must not keep paying for the same strike.
    event["incoming"][:] = [0.0, 4.0]
    env._contact(event)
    assert env.shot_count[0, 0] == 1


def test_rushed_return_shaping_preserves_controlled_and_slow_strikes():
    rewards = []
    for penalty in (0, 20):
        env = NeuralTrainingEnv(
            3, stage=2, capture_first=True, rushed_shot_penalty=penalty,
            randomize=False, realistic=False,
        )
        env.reset()
        env.engine.puck_x[:] = 0.5
        env.engine.puck_y[:] = 0.5
        env.elapsed[:] = 0.5
        env.captured[0, 1] = True
        env._contact(dict(
            body="agent", indices=np.arange(3),
            incoming=np.array([[0., -5.], [0., -5.], [0., -.3]]),
            outgoing_before_speed_cap=np.array([[0., 5.]] * 3),
        ))
        rewards.append(env.reward_events[0].copy())
        np.testing.assert_array_equal(env.shot_count[0], [1, 1, 1])
    np.testing.assert_allclose(rewards[0] - rewards[1], [20, 0, 0])


def test_variance_upgrade_preserves_old_policy_mean():
    import torch
    from airhockey.neural_player import NeuralPlayer

    torch.set_num_threads(2)
    original = NeuralPlayer(width=32)
    state = {
        k: v.clone()
        for k, v in original.state_dict().items()
        if not k.startswith("noise.")
    }
    restored = NeuralPlayer(width=32)
    assert restored.load_weights(state)
    observations = torch.randn(8, 42)
    torch.testing.assert_close(original(observations)[0], restored(observations)[0])
    _, _, log_std = restored.distribution(observations)
    torch.testing.assert_close(log_std, restored.log_std.expand(8, -1))


def test_training_shutdown_is_a_failure_and_evaluation_keeps_heat():
    for training in [False, True]:
        env = NeuralTrainingEnv(
            2,
            games=True,
            stage=3,
            randomize=False,
            realistic=False,
            terminate_overload=training,
        )
        env.reset(opponent="external")
        env.loads[0].h[0] = 1.02**2
        action = np.zeros((2, 6))
        env.set_opponent_action(action)
        _, reward, terminal, _, info = env.step(action)
        assert bool(terminal[0]) == training
        assert info["load_peak"][0] > 1
        other_heat = env.loads[0].h[1].copy()
        env.reset(mask=np.array([True, False]))
        np.testing.assert_array_equal(env.loads[0].h[1], other_heat)
        if training:
            assert reward[0] < -90
            assert env.loads[0].levels[0].max() <= 0.8
        else:
            assert env.loads[0].levels[0].max() > 1


def test_slow_puck_outside_contact_workspace_is_not_controlled():
    from airhockey.skill_benchmark import Fixtures

    env = NeuralTrainingEnv(1, randomize=False, realistic=False)
    env.reset(
        fixtures=Fixtures(
            np.array([0]),
            np.array([[0.94, 0.4, 0, 0]]),
            np.array([[0.80, 0.4]]),
            np.array([0.5]),
        )
    )
    env.elapsed[:] = 0.4
    env.contact_clock[0] = 0.38
    env.control_time[0] = 0.10
    env.captured[0] = True
    env.step(np.zeros((1, 6)))
    assert env.capture_count[0, 0] == 0
    assert not env.captured[0, 0]
    assert env.control_time[0, 0] == 0


def test_training_context_cannot_change_actor_or_receive_actor_gradients():
    import torch
    from airhockey.neural_player import NeuralPlayer

    torch.set_num_threads(2)
    net = NeuralPlayer(32)
    observations = torch.randn(8, 42)
    a = net.distribution(observations, torch.zeros(8, 9))
    b = net.distribution(observations, torch.randn(8, 9))
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[2], b[2], atol=0, rtol=0)
    b[1].sum().backward()
    assert all(p.grad is None for p in net.actor_parameters())


def test_old_value_input_migration_preserves_zero_context_prediction():
    import torch
    from airhockey.neural_player import NeuralPlayer

    torch.set_num_threads(2)
    net = NeuralPlayer(32)
    state = {k: v.clone() for k, v in net.state_dict().items()}
    state["value_trunk.0.weight"] = state["value_trunk.0.weight"][:, :42].clone()
    restored = NeuralPlayer(32)
    assert restored.load_weights(state)
    obs = torch.randn(8, 42)
    torch.testing.assert_close(net.value(obs), restored.value(obs))
    assert torch.count_nonzero(restored.value_trunk[0].weight[:, 42:]) == 0


def test_training_load_margin_is_distinct_from_actual_overload():
    env = NeuralTrainingEnv(
        1,
        games=True,
        stage=3,
        realistic=False,
        randomize=False,
        terminate_overload=True,
        shutdown_level=0.97,
    )
    env.reset(opponent="external")
    env.loads[0].h[:] = 0.98**2
    env.set_opponent_action(np.zeros((1, 6)))
    _, reward, terminal, _, info = env.step(np.zeros((1, 6)))
    assert terminal[0] and info["shutdown"][0]
    assert info["overload_seconds"][0] == 0
    assert 0.97 < info["load_peak"][0] < 1
    assert reward[0] < -90


def test_persistent_recovery_bias_holds_then_refreshes_and_resets_independently():
    import torch
    from airhockey.neural_player import RecoveryExplorationBias

    torch.manual_seed(6827)
    sampler = RecoveryExplorationBias(2, 1, block_steps=3)
    obs = torch.zeros(2, 45)
    obs[:, 1], obs[:, 3] = .4, .3 / 6
    context = torch.zeros(2, 9)
    context[:, 1] = 1
    touched = torch.zeros(2, dtype=torch.bool)
    first = sampler.sample(obs, context, touched).clone()
    assert torch.count_nonzero(first[:, :4]) == 8
    assert torch.count_nonzero(first[:, 4:]) == 0
    torch.testing.assert_close(sampler.sample(obs, context, touched), first, rtol=0, atol=0)
    sampler.reset(np.array([True, False]))
    third = sampler.sample(obs, context, touched).clone()
    assert not torch.equal(third[0], first[0])
    assert torch.equal(third[1], first[1])
    fourth = sampler.sample(obs, context, touched).clone()
    assert torch.equal(fourth[0], third[0])
    assert not torch.equal(fourth[1], third[1])
    touched[0] = True
    assert torch.count_nonzero(sampler.sample(obs, context, touched)[0]) == 0
    sampler.reset()
    assert not sampler.values.any() and not sampler.remaining.any()


def test_persistent_recovery_bias_excludes_games_incoming_contact_and_hot_states():
    import torch
    from airhockey.neural_player import RecoveryExplorationBias

    sampler = RecoveryExplorationBias(7, 1)
    obs = torch.zeros(7, 45)
    obs[:, 1], obs[:, 3] = .4, .3 / 6
    context = torch.zeros(7, 9)
    context[:, 1] = 1
    context[1] = 0
    context[1, 3] = 1  # Full self-play never receives this exploration.
    obs[2, 3] = -.3 / 6
    obs[3, 1] = 1.2
    obs[4, 3] = 3 / 6
    touched = torch.zeros(7, dtype=torch.bool)
    touched[5] = True
    obs[6, 21:29] = 1
    result = sampler.sample(obs, context, touched)
    assert result[0].abs().sum() > 0
    assert torch.count_nonzero(result[1:]) == 0
    obs[0, 21:29] = .9
    hot = sampler.sample(obs, context, touched)
    torch.testing.assert_close(hot[0], result[0] * .5)


def test_persistent_exploration_conditional_ppo_likelihood_and_gradient():
    import torch
    from airhockey.neural_player import log_probability

    torch.manual_seed(6828)
    mean = torch.randn(32, 6)
    offset = torch.randn_like(mean) * .6
    scale = torch.full_like(mean, -.9)
    raw = mean + offset + scale.exp() * torch.randn_like(mean)
    old_logp = log_probability(raw, mean + offset, scale)
    # Replaying the retained offset gives unit ratios before any update.
    torch.testing.assert_close((log_probability(raw, mean + offset, scale) - old_logp).exp(),
                               torch.ones(32), rtol=0, atol=0)
    assert (log_probability(raw, mean, scale) - old_logp).abs().max() > 1
    new_mean = (mean + .01).requires_grad_()
    logp = log_probability(raw, new_mean + offset, scale)
    expected = torch.distributions.Normal(new_mean + offset, scale.exp()).log_prob(raw).sum(-1)
    torch.testing.assert_close(logp - logp[0], expected - expected[0])
    logp.sum().backward()
    torch.testing.assert_close(new_mean.grad, (raw - new_mean.detach() - offset) / scale.exp().square())
    assert not offset.requires_grad


def test_persistent_recovery_bias_rejects_invalid_configuration():
    import pytest
    from airhockey.neural_player import RecoveryExplorationBias

    for strength in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            RecoveryExplorationBias(2, strength)
    for steps in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            RecoveryExplorationBias(2, 1, block_steps=steps)
