import numpy as np

from airhockey.neural_possession import direct_goal_coverage_cost
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.skill_benchmark import Fixtures


def hold(env, point):
    action = np.zeros((env.n_envs, 6))
    action[:, :2] = 2 * (np.asarray(point) - env.decoder.low) / (env.decoder.high - env.decoder.low) - 1
    action[:, 4] = 1
    action[:, 5] = -1
    return action


def test_continuous_rally_preserves_reachable_puck_past_both_old_reset_deadlines():
    env = NeuralTrainingEnv(1, stage=2, realistic=False, randomize=False,
                            continuous_rallies=True, game_episode_seconds=120)
    env.reset(fixtures=Fixtures(np.array([3]), np.array([[.5, .55, 0., 0.]]),
                               np.array([[.5, .2]]), np.array([.5])))
    action = hold(env, [.5, .2])
    for _ in range(510):
        _, _, terminal, truncated, _ = env.step(action)
        assert not terminal.any() and not truncated.any()
    np.testing.assert_allclose([env.engine.puck_x[0], env.engine.puck_y[0]], [.5, .55])
    assert not env.turnovers.any()


def test_continuous_rally_reserves_dead_puck_only_outside_both_contact_workspaces():
    env = NeuralTrainingEnv(3, stage=2, realistic=False, randomize=False, continuous_rallies=True)
    env.reset(fixtures=Fixtures(np.full(3, 3),
        np.array([[.5, 1., 0., 0.], [.5, .55, 0., 0.], [.5, 1.45, 0., 0.]]),
        np.tile([.5, .2], (3, 1)), np.full(3, .5)))
    action = hold(env, [.5, .2])
    np.testing.assert_array_equal(env._unreachable_dead_puck(), [True, False, False])
    for _ in range(405):
        env.step(action)
    assert env.turnovers[:, 0].sum() == 1
    assert env.turnovers[:, 1:].sum() == 0


def test_unreachable_fringe_is_not_preserved_by_an_outside_tolerance():
    env = NeuralTrainingEnv(2, realistic=False, randomize=False, continuous_rallies=True)
    radius = env.cfg.puck_radius+env.cfg.paddle_radius
    env.engine.puck_x[:] = env.decoder.high[0] + radius + np.array([.003, -.004])
    env.engine.puck_y[:] = .4
    np.testing.assert_array_equal(env._unreachable_dead_puck(), [True, False])


def test_stationary_failure_replay_retains_real_prior_action_and_correlated_heat(tmp_path):
    path = tmp_path/'replay.npz'
    action = np.array([[.1, .2, .3, .4, .5, .6]])
    np.savez(path, puck=np.array([[.5,.5,0.,0.]]), paddle=np.array([[.5,.3]]), previous_action=action, request=np.array([[1.,0.,0.]]))
    env = NeuralTrainingEnv(120, stage=5, realistic=False, randomize=False, shot_conditioned=True,
        fixed_practice_roles=True, stationary_replay=path, stationary_replay_fraction=1,
        stationary_rest_fraction=1, correlated_warm_fraction=1, warm_start_max=.98, shutdown_level=1)
    env.reset(seed=3654)
    selected = env.kind == 0
    assert selected.any()
    np.testing.assert_allclose(env.last_action[selected], np.repeat(action, selected.sum(), axis=0), atol=1e-7)
    assert np.all(env.base._shot_type[selected] == 1)
    levels = env.loads[0].levels[selected]
    np.testing.assert_allclose(levels, np.broadcast_to(levels[:, :1, :1], levels.shape))
    assert np.all((levels >= .75) & (levels <= .98))


def test_hot_full_game_initialization_preserves_heat_at_later_resets():
    env = NeuralTrainingEnv(32, stage=5, games=True, realistic=False,
        warm_game_start=True, warm_start_max=.98, correlated_warm_fraction=1)
    env.reset(seed=4801)
    for load in env.loads:
        assert (load.levels >= .75).all()
        assert (load.levels <= .98).all()
        np.testing.assert_allclose(load.levels, np.broadcast_to(load.levels[:,:1,:1],load.levels.shape))
        load.h[:] = .4321
    env.reset()
    for load in env.loads:
        np.testing.assert_array_equal(load.h,np.full_like(load.h,.4321))


def test_cool_recovery_practice_does_not_remove_hot_full_game_starts():
    env = NeuralTrainingEnv(100, stage=5, game_fraction=.5, realistic=False,
        warm_game_start=True, warm_start_max=.98, correlated_warm_fraction=1,
        cold_practice_fraction=1)
    env.reset(seed=4802)
    for load in env.loads:
        assert (load.levels[env.kind!=3] <= .4).all()
        assert (load.levels[env.kind==3] >= .75).all()


def test_cool_full_game_starts_are_once_only_and_thermal_memory_survives_resets():
    env = NeuralTrainingEnv(40, stage=5, games=True, realistic=False,
        warm_game_start=True, warm_start_max=.97, correlated_warm_fraction=1,
        cold_game_start_fraction=1)
    env.reset(seed=6201)
    for load in env.loads:
        assert (load.levels <= .4).all()
        load.h[:] = .91**2
    env.reset()
    for load in env.loads:
        np.testing.assert_allclose(load.levels, .91)


def test_outgoing_progress_requires_going_around_before_contact_and_cannot_repeat():
    env = NeuralTrainingEnv(3, stage=2, realistic=False, randomize=False,
        recovery_get_ahead_bonus=150)
    env.reset(fixtures=Fixtures(np.ones(3, int), np.tile([.5, .4, 0., .3], (3, 1)),
        np.array([[.5, .25], [.5, .25], [.5, .55]]), np.full(3, .5)))
    env.recovery_drill[:] = True
    assert not env._recovery_approach_event().any()
    # A valid first approach, an approach after pushing from behind, and an
    # already-leading reset are different learning events.
    env.engine.paddle_agent_x[:] = .5
    env.engine.paddle_agent_y[:] = .55
    env.touch_count[0, 1] = 1
    np.testing.assert_array_equal(env._recovery_approach_event(), [True, False, False])
    assert not env._recovery_approach_event().any()


def test_slower_outgoing_curriculum_keeps_its_declared_physical_speed_range():
    env = NeuralTrainingEnv(400, stage=5, realistic=False, randomize=False,
        fixed_practice_roles=True, recovery_fraction=1,
        recovery_min_speed=.1, recovery_max_speed=.45)
    env.reset(seed=6202)
    speed = np.hypot(env.engine.puck_vx, env.engine.puck_vy)[env.recovery_drill]
    assert len(speed) > 30 and (speed >= .1).all() and (speed <= .45).all()


def test_opponent_style_curriculum_persists_across_possessions_without_fixing_own_request():
    env = NeuralTrainingEnv(120, stage=5, games=True, realistic=False,
        shot_conditioned=True, fixed_opponent_style_fraction=1)
    env.reset(seed=5701)
    probabilities = env.base._shot_type_p_opp.copy()
    assert (probabilities.max(axis=1) == 1).all()
    assert set(probabilities.argmax(axis=1)) == {1, 2, 3}
    own = []
    for _ in range(10):
        env.base._draw_shot_types(np.ones(120, bool), agent=False)
        np.testing.assert_array_equal(env.base._shot_type_opp, probabilities.argmax(axis=1))
        env.base._draw_shot_types(np.ones(120, bool), agent=True)
        own.append(env.base._shot_type.copy())
    assert np.any(np.ptp(own, axis=0) > 0)
    env.reset(mask=np.arange(120) == 0)
    np.testing.assert_array_equal(env.base._shot_type_p_opp[1:], probabilities[1:])


def test_recovery_exploration_is_local_and_keeps_shots_defense_and_inputs_intact():
    import torch
    from airhockey.neural_player import recovery_exploration_scale
    obs = torch.zeros(5,45)
    obs[:,1] = torch.tensor([.4,.4,.4,1.4,.4])
    obs[:,5] = torch.tensor([.2,.2,.3,.2,.2])
    obs[:,3] = torch.tensor([.1,0.,0.,.1,-1.])
    scale = torch.full((5,6), -4.)
    result = recovery_exploration_scale(obs,scale,.35)
    torch.testing.assert_close(result[:2,:4].exp(), torch.full((2,4),.35))
    torch.testing.assert_close(result[2:],scale[2:])
    torch.testing.assert_close(result[:,4:],scale[:,4:])
    full = recovery_exploration_scale(obs,scale,.35,timing=True)
    torch.testing.assert_close(full[:2,4:].exp(),torch.full((2,2),.35))
    torch.testing.assert_close(full[2:],scale[2:])
    assert torch.all(scale == -4)
    assert recovery_exploration_scale(obs,scale,0) is scale
    quiet = recovery_exploration_scale(obs,scale,.35,quiet=True)
    torch.testing.assert_close(quiet[2,:4].exp(), torch.full((4,),.175))
    torch.testing.assert_close(quiet[3:],scale[3:])
    mean = torch.full((5,6),1.5,requires_grad=True)
    saturated = recovery_exploration_scale(obs,scale,.35,quiet=True,mean=mean)
    assert (saturated[2,:4].exp() > .5).all()
    torch.testing.assert_close(saturated[0],quiet[0])
    saturated.sum().backward()
    assert torch.isfinite(mean.grad).all() and mean.grad[2,:4].abs().sum() > 0
    obs[:,21:29] = .98
    hot = recovery_exploration_scale(obs,scale,.35,load_aware=True)
    torch.testing.assert_close(hot[:2,:4].exp(), torch.full((2,4),.035), atol=1e-7, rtol=1e-5)
    obs[:,21:29] = 1.01
    torch.testing.assert_close(recovery_exploration_scale(obs,scale,.35,load_aware=True),scale)
    torch.testing.assert_close(recovery_exploration_scale(obs,scale,.35,load_aware=True,quiet=True),scale)
    torch.testing.assert_close(recovery_exploration_scale(obs,scale,.35,load_aware=True,quiet=True,mean=mean),scale)
    torch.testing.assert_close(recovery_exploration_scale(obs,scale,.35,load_aware=True,quiet=True,mean=mean,timing=True),scale)


def test_followthrough_keeps_controlled_puck_in_play_in_receiving_drill():
    for followthrough in (False, True):
        env = NeuralTrainingEnv(1, stage=2, realistic=False, randomize=False,
                                productive_receive_drill=True, possession_followthrough=followthrough)
        env.reset(fixtures=Fixtures(np.array([1]), np.array([[.5, .5, 0., 0.]]),
                                   np.array([[.5, .4]]), np.array([.5])))
        env.contact_clock[0] = 0
        terminated = False
        for _ in range(10):
            _, _, terminal, _, _ = env.step(hold(env, [.5, .4]))
            terminated |= bool(terminal[0])
        assert env.capture_count[0, 0] > 0
        assert terminated == (not followthrough)


def test_hot_effort_exploration_reaches_lower_physical_caps_without_changing_other_actions():
    import torch
    from airhockey.neural_player import thermal_effort_exploration_scale
    obs = torch.zeros(2, 45)
    obs[:, 21:29] = torch.tensor([.5, .95])[:, None]
    scale = torch.full((2, 6), -4.)
    result = thermal_effort_exploration_scale(obs, scale, 2.)
    torch.testing.assert_close(result[0], scale[0])
    torch.testing.assert_close(result[:, :5], scale[:, :5])
    assert abs(result[1, 5].exp().item() - 2.) < 1e-6
    # A two-standard-deviation lower sample now escapes a saturated effort
    # mean of3.5; the original tiny exploration left it at almost60m/s².
    action = (3.5 - 2 * result[:, 5].exp()).tanh()
    cap = 60 * (.05 + .95 * ((action + 1) / 2).square())
    assert cap[0] > 59 and cap[1] < 10
    assert thermal_effort_exploration_scale(obs, scale, 0) is scale
    mask = torch.tensor([True, False])
    torch.testing.assert_close(thermal_effort_exploration_scale(obs, scale, 2., practice_mask=mask), scale)


def test_extra_drill_noise_does_not_replace_full_game_learned_exploration():
    import torch
    from airhockey.neural_player import recovery_exploration_scale, thermal_effort_exploration_scale
    obs = torch.zeros(2, 45)
    obs[:, 1] = .4
    obs[:, 3] = .05
    obs[:, 21:29] = .95
    scale = torch.full((2, 6), -3.)
    practice = torch.tensor([True, False])
    result = recovery_exploration_scale(obs, scale, .5, timing=True, practice_mask=practice)
    result = thermal_effort_exploration_scale(obs, result, 2., practice_mask=practice)
    assert (result[0] > scale[0]).all()
    torch.testing.assert_close(result[1], scale[1])


def test_recovery_exploration_window_preserves_followthrough_and_other_tasks():
    import torch
    from airhockey.neural_player import recovery_exploration_window, recovery_exploration_scale
    context = torch.zeros(7, 9)
    context[:4, 1] = 1
    context[:4, 4] = torch.tensor([0., .06, .08, .2]) / 4
    context[4, 0] = context[5, 2] = context[6, 3] = 1
    expected = torch.tensor([True, True, False, False, False, False, False])
    mask = recovery_exploration_window(context, .08)
    torch.testing.assert_close(mask, expected)
    obs = torch.zeros(7, 45)
    obs[:, 1] = .4
    obs[:, 3] = .05
    scale = torch.full((7, 6), -3.)
    result = recovery_exploration_scale(obs, scale, .5, timing=True, practice_mask=mask)
    assert (result[:2] > scale[:2]).all()
    torch.testing.assert_close(result[2:], scale[2:])
    # Reordered PPO minibatches receive the same Gaussian as collection.
    order = torch.tensor([6, 1, 4, 0, 2, 5, 3])
    torch.testing.assert_close(recovery_exploration_window(context[order], .08), mask[order])
    assert recovery_exploration_window(context, 0) is None
    assert recovery_exploration_window(context, 0, expected) is expected
    torch.testing.assert_close(recovery_exploration_window(context, .08, ~expected), torch.zeros(7, dtype=torch.bool))


def test_recovery_cushion_credit_requires_first_leading_contact_and_real_slowing():
    envs = [NeuralTrainingEnv(5, stage=2, realistic=False, randomize=False,
        recovery_cushion_bonus=bonus) for bonus in (0, 150)]
    event = dict(body="agent", indices=np.arange(5),
        incoming=np.array([[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., -1.]]),
        outgoing_before_speed_cap=np.array([[0., .2], [0., .2], [0., 1.4], [0., .2], [0., .2]]),
        normal=np.array([[0., -1.], [0., 1.], [0., -1.], [0., -1.], [0., 1.]]))
    for env in envs:
        env.reset(seed=567)
        env.recovery_drill[:] = [True, True, True, False, True]
        env._contact(event)
    np.testing.assert_allclose(envs[1].reward_events-envs[0].reward_events,
        [[120., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
    for env in envs:
        env.reward_events[:] = 0
        env._contact(event)
    np.testing.assert_allclose(envs[1].reward_events, envs[0].reward_events)


def test_signed_cushion_feedback_orders_overpowered_contacts_without_repeat_credit():
    envs = [NeuralTrainingEnv(6, stage=2, realistic=False, randomize=False,
        recovery_cushion_bonus=bonus, recovery_cushion_signed=True) for bonus in (0, 100)]
    event = dict(body="agent", indices=np.arange(6),
        incoming=np.tile([0., .6], (6, 1)),
        outgoing_before_speed_cap=np.column_stack((np.zeros(6), [1.2, .8, .6, .3, .3, .3])),
        normal=np.array([[0., -1.]]*5+[[0., 1.]]))
    for env in envs:
        env.reset(seed=567)
        env.recovery_drill[:] = [True, True, True, True, False, True]
        env._contact(event)
    reward = (envs[1].reward_events-envs[0].reward_events)[0]
    assert -100/np.e < reward[0] < reward[1] < 0
    assert abs(reward[2]) < 1e-8
    assert 0 < reward[3] < 100
    np.testing.assert_allclose(reward[4:], 0)
    for env in envs:
        env.reward_events[:] = 0
        env._contact(event)
    np.testing.assert_allclose(envs[1].reward_events, envs[0].reward_events)


def test_recovery_curriculum_has_outgoing_pucks_and_preserves_explicit_fixtures():
    env = NeuralTrainingEnv(400, stage=5, realistic=False, randomize=False,
        fixed_practice_roles=True, productive_receive_drill=True,
        practice_defense_fraction=.4, recovery_fraction=1,
        possession_followthrough=True, shot_conditioned=True)
    obs = env.reset(seed=3451)
    recovery = env.recovery_drill
    assert recovery.sum() > 30
    assert obs.shape == (400, 45)
    assert np.all(env.engine.puck_vy[recovery] > 0)
    assert np.all(env.engine.paddle_agent_y[recovery] < env.engine.puck_y[recovery])
    assert not np.any(recovery & (env.kind != 1))
    before = env.recovery_drill.copy()
    env.reset(mask=np.arange(400) == 399, fixtures=Fixtures(np.array([1]),
        np.array([[.5, .5, 0., -.4]]), np.array([[.5, .3]]), np.array([.5])))
    assert not env.recovery_drill[399]
    np.testing.assert_array_equal(env.recovery_drill[:399], before[:399])
    assert env.engine.puck_vy[399] == -.4


def test_recovery_detour_potential_and_easy_curriculum_teach_leading_side():
    from airhockey.neural_possession import disc_avoiding_distance
    center = np.zeros((3, 2))
    start = np.array([[0., -.2], [.15, 0.], [-.15, 0.]])
    goal = np.tile([0., .15], (3, 1))
    distance = disc_avoiding_distance(start, goal, center, .1)
    assert distance[0] > .35
    np.testing.assert_allclose(distance[1], distance[2])
    env = NeuralTrainingEnv(200, stage=5, realistic=False, randomize=False,
        fixed_practice_roles=True, recovery_fraction=1, recovery_easy_fraction=1)
    env.reset(seed=3751)
    ids = env.recovery_drill
    assert ids.any()
    assert (env.engine.paddle_agent_y[ids] > env.engine.puck_y[ids]).mean() > .8


def test_recovery_curriculum_bridges_side_and_leading_positions():
    env = NeuralTrainingEnv(1000, stage=2, realistic=False, randomize=False,
        fixed_practice_roles=True, recovery_fraction=1, recovery_easy_fraction=1,
        recovery_easy_arc=2*np.pi/3)
    env.reset(seed=3752)
    ids = env.recovery_drill
    offset = np.column_stack((env.engine.paddle_agent_x-env.engine.puck_x,
                              env.engine.paddle_agent_y-env.engine.puck_y))[ids]
    velocity = np.column_stack((env.engine.puck_vx, env.engine.puck_vy))[ids]
    cosine = (offset*velocity).sum(1)/(np.linalg.norm(offset,axis=1)*np.linalg.norm(velocity,axis=1))
    assert (cosine > .7).mean() > .15
    assert (np.abs(cosine) < .4).mean() > .15
    assert (cosine < 0).mean() > .1
    assert (np.linalg.norm(offset,axis=1) > env.cfg.paddle_radius+env.cfg.puck_radius).all()


def test_recovery_evaluation_does_not_credit_later_return_after_possession_loss(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('recovery_evaluation', Path(__file__).parents[1] / 'bin/eval_neural_player.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)

    class EscapeThenReturn(NeuralTrainingEnv):
        def reset(self, **kwargs):
            obs = super().reset(**kwargs)
            self.recovery_ids = kwargs['fixtures'].task == 2
            self.engine.puck_y[self.recovery_ids] = .95
            self.engine.puck_vx[self.recovery_ids] = 0
            self.engine.puck_vy[self.recovery_ids] = 1
            return obs

        def step(self, action):
            # Model a later successful reception. The recovery trial must have
            # already ended when its original puck escaped across midfield.
            self.capture_count[0, self.recovery_ids & (self.elapsed > 1)] = 1
            return super().step(action)

    class Idle:
        history = 1
        shot_conditioned = True
        def act(self, obs, stochastic=False):
            return np.zeros((len(obs), 6))

    monkeypatch.setattr(evaluation, 'NeuralTrainingEnv', EscapeThenReturn)
    result = evaluation.skills(Idle(), per_task=2, seed=914, recovery=True)
    assert result['receiving']['controlled'] == 0
    assert result['receiving']['recovery_metric_version'] == 2


def test_coverage_potential_penalizes_exposed_positions_without_affecting_actor_input():
    env = NeuralTrainingEnv(3, realistic=False, randomize=False, readiness_weight=80)
    puck = np.tile([.5, 1.2], (3, 1))
    pad = np.array([[.5, .25], [.79, .64], [.21, .64]])
    risk = direct_goal_coverage_cost(puck, pad, np.zeros((3, 2)), env.decoder.bounds, env.cfg)
    assert risk[0] < risk[1]
    np.testing.assert_allclose(risk[1], risk[2])
    assert np.all((risk >= 0) & (risk <= .5))
    obs = env.reset(seed=2462)
    env.readiness_weight = 0
    # Readiness is reward shaping, never an extra tactical observation.
    np.testing.assert_array_equal(obs, env._features(env.base._make_obs_direct()))


def test_recovery_rejects_uncaptured_volley_as_a_success():
    env = NeuralTrainingEnv(1, stage=2, realistic=False, randomize=False,
                            possession_followthrough=True, setup_weight=0)
    env.reset(fixtures=Fixtures(np.array([1]), np.array([[.5, 1.01, 0., 2.]]),
                               np.array([[.5, .4]]), np.array([.5])))
    env.recovery_drill[:] = True
    env.fast_aimed_count[0] = 1
    _, reward, terminal, _, _ = env.step(hold(env, [.5, .4]))
    assert terminal[0] and reward[0] < -50


def test_optional_recovery_fallback_requires_fast_aimed_requested_shot_without_fake_capture():
    envs = [NeuralTrainingEnv(5, stage=2, realistic=False, randomize=False,
        shot_conditioned=True, possession_followthrough=True, setup_weight=0,
        recovery_fallback_scale=scale, shot_request_weight=100) for scale in (0, 1)]
    fixture = Fixtures(np.ones(5, int), np.tile([.5, .5, 0., .4], (5, 1)),
                       np.tile([.5, .3], (5, 1)), np.full(5, .5))
    observations = []
    event = dict(body='agent', indices=np.arange(5), incoming=np.tile([0., .4], (5, 1)),
        outgoing_before_speed_cap=np.array([[0., 8], [0., 8], [0., 5], [4., 8], [0., 8]]),
        normal=np.tile([0., 1.], (5, 1)))
    for env in envs:
        observations.append(env.reset(seed=1872, fixtures=fixture))
        env.base._shot_type[:] = [3, 1, 3, 3, 3]
        env.recovery_drill[:] = [True, True, True, True, False]
        env._contact(event)
    np.testing.assert_array_equal(observations[0], observations[1])
    # Wrong route, too slow, off-target and an ordinary task gain no new credit.
    difference = (envs[1].reward_events-envs[0].reward_events)[0]
    assert difference[0] > 100
    np.testing.assert_array_equal(difference[1:], 0)
    np.testing.assert_array_equal(envs[1]._recovery_fast_fallback, [True, False, False, False, False])
    assert not envs[1].captured.any() and not envs[1].capture_count.any()
    results = [env.step(hold(env, [.5, .3])) for env in envs]
    assert not results[0][2][0] and results[1][2][0]
    assert results[1][1][0] > 10
    assert results[1][4]['captures'][0] == 0 and results[1][4]['conversions'][0] == 0
    envs[1].reset(mask=np.arange(5)==0, fixtures=fixture.take(np.array([0])))
    assert not envs[1]._recovery_fast_fallback.any()


def test_recovery_fallback_scale_rejects_invalid_values():
    import pytest
    for scale in (-.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match='recovery fallback scale'):
            NeuralTrainingEnv(1, recovery_fallback_scale=scale)


def test_recovery_potential_rewards_leading_side_until_control_then_shooting_setup():
    env = NeuralTrainingEnv(2, stage=2, realistic=False, randomize=False,
                            recovery_approach_weight=40, setup_weight=6)
    env.reset(fixtures=Fixtures(np.full(2, 1),
        np.tile([.5, .4, 0., .5], (2, 1)),
        np.array([[.5, .27], [.5, .57]]), np.full(2, .5)))
    before = env._features(env.base._make_obs_direct()).copy()
    potential = env.potential()
    assert potential[1] > potential[0] + 5
    env.captured[0] = True
    potential = env.potential()
    assert potential[0] > potential[1]
    np.testing.assert_array_equal(before, env._features(env.base._make_obs_direct()))
    env.captured[0] = False
    env.engine.puck_vy[:] = 0
    assert env.potential()[0] > env.potential()[1]


def test_defense_windup_waits_then_launches_without_exposing_schedule_to_actor():
    env = NeuralTrainingEnv(80, stage=5, realistic=False, randomize=False,
        fixed_practice_roles=True, defense_windup_fraction=1,
        random_practice_opponent=True, practice_selfplay_fraction=1,
        shot_conditioned=True)
    obs = env.reset(seed=3462)
    mask = env.defense_windup.copy()
    assert mask.any() and obs.shape == (80, 45)
    assert np.all(env.kind[mask] == 2)
    assert np.all(np.isfinite(env._windup_release[mask]))
    assert np.all(env.base._shot_type[mask] != 0)
    start = np.column_stack((env.engine.puck_x, env.engine.puck_y))[mask].copy()
    env._windup_release[mask] = .5
    for _ in range(25):
        env.step(hold(env, [.5, .2]))
    np.testing.assert_allclose(np.column_stack((env.engine.puck_x, env.engine.puck_y))[mask], start)
    env.step(hold(env, [.5, .2]))
    assert np.all(env.engine.puck_vy[mask] < -5)
    assert np.all(np.isinf(env._windup_release[mask]))
    index = np.flatnonzero(mask)[0]
    env.reset(mask=np.arange(80) == index, fixtures=Fixtures(np.array([2]),
        np.array([[.5, 1.2, 0., -8.]]), np.array([[.5, .3]]), np.array([.5])))
    assert not env.defense_windup[index] and np.isinf(env._windup_release[index])


def test_neural_reference_leaves_recovery_and_preparation_free_and_has_no_teacher_gradient():
    import torch
    from airhockey.neural_player import NeuralPlayer, established_skill_preservation
    actor, teacher = NeuralPlayer(32, shot_conditioned=True), NeuralPlayer(32, shot_conditioned=True)
    obs = torch.zeros(4, 45)
    obs[:, 1] = torch.tensor([.4, .4, .4, 1.3])
    obs[:, 3] = torch.tensor([0., -1., .1, 0.])
    assert established_skill_preservation(actor, teacher, obs[2:]).item() == 0
    assert established_skill_preservation(actor, teacher, obs[2:], controlled=torch.tensor([True, True])) > 0
    loss = established_skill_preservation(actor, teacher, obs)
    assert loss > 0
    loss.backward()
    assert actor.actor.weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in actor.critic.parameters())
    obs[:,21:29] = .9
    assert established_skill_preservation(actor, teacher, obs, max_load=.85).item() == 0
    obs[:,21:29] = .7
    assert established_skill_preservation(actor, teacher, obs, max_load=.85).item() > 0
    assert established_skill_preservation(actor, teacher, obs[:1], incoming_only=True).item() == 0
    assert established_skill_preservation(actor, teacher, obs[1:2], incoming_only=True).item() > 0
    far_incoming = obs[1:2].clone()
    far_incoming[:,1] = 1.3
    assert established_skill_preservation(actor, teacher, far_incoming).item() == 0
    assert established_skill_preservation(actor, teacher, far_incoming, incoming_only=True).item() > 0


def test_stationary_completion_penalty_requires_an_aimed_shot_and_rest_history_is_physical():
    rewards = []
    for penalty in (0, 120):
        env = NeuralTrainingEnv(2, stage=2, realistic=False, randomize=False,
            possession_followthrough=True, stationary_failure_penalty=penalty,
            stationary_rest_fraction=1, setup_weight=0)
        env.reset(seed=873, fixtures=Fixtures(np.zeros(2, int),
            np.tile([.5, .5, 0., 0.], (2, 1)), np.tile([.5, .2], (2, 1)), np.full(2, .5)))
        np.testing.assert_allclose(env.decoder.unpack(env.last_action)[0], np.tile([.5, .2], (2, 1)), atol=1e-7)
        np.testing.assert_allclose(env.last_action[:, 2:4], 0)
        env.elapsed[:] = 6
        env.aimed_count[0, 1] = 1
        _, reward, terminal, _, _ = env.step(hold(env, [.5, .2]))
        assert terminal.all()
        rewards.append(reward)
    np.testing.assert_allclose(rewards[1]-rewards[0], [-120, 0])


def test_stationary_timeout_penalty_does_not_double_charge_a_conceded_goal():
    rewards = []
    for penalty in (0, 600):
        env = NeuralTrainingEnv(1, stage=2, realistic=False, randomize=False,
            possession_followthrough=True, stationary_failure_penalty=penalty)
        env.reset(seed=874, fixtures=Fixtures(np.zeros(1,int),
            np.array([[.5,.04,0.,-8.]]), np.array([[.5,.4]]), np.full(1,.5)))
        env.elapsed[:] = 6
        _, reward, terminal, _, info = env.step(hold(env,[.5,.4]))
        assert terminal[0] and info['conceded'][0] == 1
        rewards.append(reward)
    np.testing.assert_allclose(rewards[0],rewards[1])


def test_readiness_running_cost_applies_to_waiting_exposure_not_own_possession():
    rewards = []
    for cost in (0, 100):
        env = NeuralTrainingEnv(3, stage=2, realistic=False, randomize=False,
                                readiness_cost_weight=cost, setup_weight=0)
        env.reset(seed=761, fixtures=Fixtures(np.full(3, 3),
            np.array([[.5, 1.4, 0., 0.], [.5, .4, 0., 0.], [.5, 1.4, 0., -8.]]),
            np.tile([.79, .63], (3, 1)), np.full(3, .5)))
        _, reward, _, _, _ = env.step(hold(env, [.79, .63]))
        rewards.append(reward)
    assert rewards[1][0] < rewards[0][0] - .1
    np.testing.assert_allclose(rewards[1][1], rewards[0][1])
    np.testing.assert_allclose(rewards[1][2], rewards[0][2])


def test_recovery_capture_bonus_is_not_repeatable_or_paid_in_other_tasks():
    totals = []
    for bonus in (0, 60):
        env = NeuralTrainingEnv(2, stage=2, realistic=False, randomize=False,
                                possession_followthrough=True, recovery_capture_bonus=bonus)
        env.reset(seed=648, fixtures=Fixtures(np.ones(2, int),
            np.tile([.5, .5, 0., 0.], (2, 1)), np.tile([.5, .4], (2, 1)), np.full(2, .5)))
        env.recovery_drill[:] = [True, False]
        env.contact_clock[0] = 0
        reward_sum = np.zeros(2)
        for repeat in range(2):
            for _ in range(8):
                reward_sum += env.step(hold(env, [.5, .4]))[1]
            env.captured[0] = False
            env.control_time[0] = 0
        totals.append(reward_sum)
    np.testing.assert_allclose(totals[1] - totals[0], [60, 0], atol=1e-4)


def test_direct_shot_pressure_changes_opponent_request_only():
    env = NeuralTrainingEnv(120, stage=2, realistic=False, randomize=False, shot_conditioned=True)
    env.reset(seed=1672)
    env.base._shot_type_p_opp = np.array([0., 0., 0., 1.])
    mask = np.ones(120, bool)
    env.base._draw_shot_types(mask, agent=False)
    env.base._draw_shot_types(mask, agent=True)
    assert np.all(env.base._shot_type_opp == 3)
    assert set(env.base._shot_type) == {1, 2, 3}


def test_pressure_evaluation_keeps_candidate_request_when_swapping_colors():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("possession_evaluation", Path(__file__).parents[1] / "bin/eval_neural_player.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)

    class ObservingPolicy:
        history = 1
        shot_conditioned = True

        def act(self, obs, stochastic=False):
            self.requests = obs[:, -3:].copy()
            return np.zeros((len(obs), 6))

    for swap in (False, True):
        candidate, opponent = ObservingPolicy(), ObservingPolicy()
        evaluation.games(candidate, opponent_net=opponent, swap_sides=swap,
                         n=32, seconds=.02, shot_request="left",
                         opponent_shot_request="straight", continuous_rallies=True)
        for policy, expected in ((candidate, [1, 0, 0]), (opponent, [0, 0, 1])):
            present = policy.requests.sum(axis=1) > 0
            assert present.any()
            np.testing.assert_array_equal(policy.requests[present], np.tile(expected, (present.sum(), 1)))


def test_request_consistency_trains_actor_only_on_far_half_without_changing_inputs():
    import torch
    from airhockey.neural_player import NeuralPlayer, defensive_request_consistency
    torch.manual_seed(2834)
    net = NeuralPlayer(32, shot_conditioned=True)
    observation = torch.randn(32, 45)
    observation[:, -3:] = torch.eye(3)[torch.arange(32) % 3]
    observation[:, 1] = .5
    own = defensive_request_consistency(net, observation)
    assert own.item() == 0
    observation[:, 1] = 1.5
    original = observation.clone()
    loss = defensive_request_consistency(net, observation)
    assert loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.actor_parameters())
    assert all(p.grad is None for p in net.value_parameters())
    torch.testing.assert_close(observation, original)


def test_first_leading_contact_pays_missing_approach_credit_once_without_cushion_bonus():
    envs = [NeuralTrainingEnv(6, stage=2, realistic=False, randomize=False,
        recovery_get_ahead_bonus=bonus) for bonus in (0, 150)]
    event = dict(body='agent', indices=np.arange(6),
        incoming=np.array([[0., .6]]*5 + [[0., -.6]]),
        outgoing_before_speed_cap=np.tile([0., 1.2], (6, 1)),
        normal=np.array([[0., -1.]]*4 + [[0., 1.], [0., 1.]]))
    for env in envs:
        env.reset(seed=568)
        env.recovery_drill[:] = [True, True, True, False, True, True]
        env._recovery_started_behind[:] = [True, True, False, True, True, True]
        env._recovery_ahead_paid[:] = [False, True, False, False, False, False]
        # Successful approach, already paid, already ahead initially, ordinary
        # task, trailing collision, and incoming puck are distinct cases.
        env._contact(event)
    np.testing.assert_allclose((envs[1].reward_events-envs[0].reward_events)[0],
                               [150., 0., 0., 0., 0., 0.])
    assert envs[1]._recovery_ahead_paid[0]
    for env in envs:
        env.reward_events[:] = 0
        # A later leading contact cannot repair an initial trailing collision
        # for another reward, and an already credited approach cannot repeat.
        event['normal'][:] = [0., -1.]
        event['incoming'][:] = [0., .6]
        env._contact(event)
    np.testing.assert_allclose(envs[1].reward_events, envs[0].reward_events)
    envs[1].engine.paddle_agent_x[:] = envs[1].engine.puck_x
    envs[1].engine.paddle_agent_y[:] = envs[1].engine.puck_y + .15
    envs[1].engine.puck_vx[:] = 0
    envs[1].engine.puck_vy[:] = .3
    assert not envs[1]._recovery_approach_event().any()
