import numpy as np
import torch

from airhockey.neural_player import NeuralPlayer, PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.shot_flight import first_goal_crossing, open_goal_outcomes


def test_routes_match_physical_goal_outcomes_and_reflection():
    launches = np.array([[.5, .5, 0, 8], [.5, .5, -5.26, 8], [.5, .5, 5.26, 8],
                         [.5, .5, 2, 8], [.5, .5, 0, .01]])
    x, aimed, banks, rail = first_goal_crossing(launches, return_route=True)
    np.testing.assert_array_equal(aimed, [True, True, True, False, False])
    np.testing.assert_array_equal(open_goal_outcomes(launches), aimed)
    np.testing.assert_array_equal(banks[:3], [0, 1, 1])
    np.testing.assert_array_equal(rail[:3], [0, -1, 1])
    np.testing.assert_allclose(x[1] + x[2], 1)
    old_x, old_aimed = first_goal_crossing(launches)
    np.testing.assert_array_equal(x, old_x)
    np.testing.assert_array_equal(aimed, old_aimed)


def test_conditioning_and_history_upgrade_preserve_actions_value_and_noise():
    torch.set_num_threads(2)
    old = NeuralPlayer(32)
    conditioned = NeuralPlayer(32, shot_conditioned=True)
    assert conditioned.load_weights(old.state_dict())
    obs, request, context = torch.randn(8, 42), torch.randn(8, 3), torch.randn(8, 9)
    for a, b in zip(old.distribution(obs, context), conditioned.distribution(torch.cat((obs, request), 1), context)):
        torch.testing.assert_close(a, b)
    # Make request weights nonzero to test migration of learned conditioning.
    with torch.no_grad():
        conditioned.trunk[0].weight[:, -3:].normal_()
        conditioned.value_trunk[0].weight[:, 42:45].normal_()
    history_net = NeuralPlayer(32, history=4, shot_conditioned=True)
    assert history_net.load_weights(conditioned.state_dict())
    expanded = torch.cat((obs, torch.randn(8, 126), request), 1)
    for a, b in zip(conditioned.distribution(torch.cat((obs, request), 1), context), history_net.distribution(expanded, context)):
        torch.testing.assert_close(a, b)
    history = PhysicalHistory(4)
    history.reset(torch.cat((obs, request), 1).numpy())
    np.testing.assert_array_equal(history.for_policy(old), obs.numpy())
    np.testing.assert_array_equal(history.for_policy(conditioned), torch.cat((obs, request), 1).numpy())
    before = history.get()[1:].copy()
    history.reset(np.zeros((8, 45)), mask=np.arange(8) == 0)
    np.testing.assert_array_equal(history.get()[1:], before)


def test_widening_preserves_actions_and_new_features_can_learn():
    torch.set_num_threads(2)
    old = NeuralPlayer(32, shot_conditioned=True)
    wide = NeuralPlayer(64, history=4, shot_conditioned=True)
    assert wide.load_weights(old.state_dict())
    obs, past, context = torch.randn(16,45), torch.randn(16,126), torch.randn(16,9)
    expanded = torch.cat((obs[:,:42],past,obs[:,42:]),1)
    for a,b in zip(old.distribution(obs,context),wide.distribution(expanded,context)):
        torch.testing.assert_close(a,b)
    wide.actor(wide.trunk(expanded)).sum().backward()
    assert torch.count_nonzero(wide.actor.weight.grad[:,32:]) > 0


def test_request_is_only_new_actor_input_and_stable_within_possession():
    env = NeuralTrainingEnv(96, shot_conditioned=True, randomize=False, realistic=False)
    obs = env.reset(seed=731)
    assert obs.shape == (96, 45)
    assert set(env.base._shot_type) == {1, 2, 3}
    old_request = env.base._shot_type.copy()
    for _ in range(5):
        env.base._update_possessions()
    np.testing.assert_array_equal(env.base._shot_type, old_request)
    env.base._shot_type[:] = 3
    raw = env.base._make_obs_direct()
    before = env._features(raw)
    env.base._shot_type[:] = 1
    env.task[:] = 2
    after = env._features(raw)
    np.testing.assert_array_equal(before[:, :42], after[:, :42])
    np.testing.assert_array_equal(after[:, -3:], np.tile([1, 0, 0], (96, 1)))
    # Crossing to the other half draws the opponent's request, not ours.
    env.engine.puck_y[:] = 1.5
    env.base._update_possessions()
    assert set(env.base._shot_type_opp) == {1, 2, 3}
    np.testing.assert_array_equal(env.base._shot_type, np.ones(96))
    env.engine.puck_y[:] = .5
    env.base._update_possessions()
    assert set(env.base._shot_type) == {1, 2, 3}
    before = env.base._shot_type.copy()
    env.reset(mask=np.arange(96) == 0)
    np.testing.assert_array_equal(env.base._shot_type[1:], before[1:])


def contact_reward(request, velocity, *, controlled=False, fallback=1, incoming=3):
    env = NeuralTrainingEnv(1, stage=5, shot_conditioned=True, capture_first=True,
                            shot_request_weight=40, wrong_shot_penalty=10,
                            fallback_shot_scale=fallback, conversion_weight=25,
                            shot_power_weight=30, off_target_penalty=30,
                            randomize=False, realistic=False)
    env.reset()
    env.engine.puck_x[:] = env.engine.puck_y[:] = .5
    env.base._shot_type[:] = request
    env.captured[0] = controlled
    env._contact(dict(body="agent", indices=np.array([0]), incoming=np.array([[0., -incoming]]),
                      outgoing_before_speed_cap=np.array([velocity], dtype=float)))
    return env.reward_events[0, 0], env


def test_requested_accurate_shots_and_control_bonus_without_punishing_fallback():
    direct, _ = contact_reward(3, [0, 8])
    wrong, _ = contact_reward(1, [0, 8])
    controlled, _ = contact_reward(3, [0, 8], controlled=True)
    old_suppressed, _ = contact_reward(1, [0, 8], fallback=0)
    assert controlled > direct > wrong > old_suppressed
    assert wrong > 0  # A useful emergency shot still earns credit.
    left, env = contact_reward(1, [-5.26, 8], incoming=.3)
    mismatch, _ = contact_reward(2, [-5.26, 8], incoming=.3)
    assert left > mismatch
    assert env.aimed_route_counts[0, 0, 1, 1] == 1
    miss, env = contact_reward(3, [2, 8])
    assert miss < 0 and env.aimed_count[0, 0] == 0


def test_slow_setup_potential_follows_request_but_fast_defense_does_not():
    env = NeuralTrainingEnv(2, shot_conditioned=True, randomize=False, realistic=False)
    env.reset()
    e = env.engine
    e.puck_x[:] = e.puck_y[:] = .5
    e.puck_vx[:] = e.puck_vy[:] = 0
    e.paddle_agent_x[:] = [.43, .57]
    e.paddle_agent_y[:] = .39
    env.base._shot_type[:] = 1
    left = env.potential()
    assert left[1] > left[0]  # A left bank is struck from the puck's right.
    env.base._shot_type[:] = 2
    right = env.potential()
    assert right[0] > right[1]
    np.testing.assert_allclose(left, right[::-1])
    e.puck_vy[:] = -4
    fast_right = env.potential()
    env.base._shot_type[:] = 1
    np.testing.assert_allclose(env.potential(), fast_right)


def test_easy_shot_starts_cover_routes_without_changing_explicit_fixtures():
    from airhockey.skill_benchmark import Fixtures

    env = NeuralTrainingEnv(200, stage=2, fixed_practice_roles=True, shot_conditioned=True,
                            shot_setup_fraction=1, randomize=False, realistic=False)
    env.reset(seed=905)
    e = env.engine
    gap = np.hypot(e.puck_x - e.paddle_agent_x, e.puck_y - e.paddle_agent_y)
    prepared = (env.kind == 0) & (gap < .12)
    assert prepared.sum() > 20
    for code in (1, 2, 3):
        chosen = prepared & (env.base._shot_type == code)
        assert chosen.sum() > 3
        if code == 1:
            assert np.all(e.paddle_agent_x[chosen] > e.puck_x[chosen])
        elif code == 2:
            assert np.all(e.paddle_agent_x[chosen] < e.puck_x[chosen])
    mask = np.arange(200) == 0
    fixture = Fixtures(np.zeros(1, int), np.array([[.5, .5, 0., 0.]]), np.array([[.4, .3]]), np.array([.5]))
    env.reset(mask=mask, fixtures=fixture)
    np.testing.assert_allclose([e.paddle_agent_x[0], e.paddle_agent_y[0]], [.4, .3])


def test_training_reflection_matches_arrival_geometry_and_load_channels():
    from airhockey.neural_symmetry import reflect_physical, reflect_arrival

    env = NeuralTrainingEnv(16, randomize=False, realistic=False)
    obs = env.reset(seed=913)
    reflected = reflect_physical(obs)
    np.testing.assert_allclose(reflect_physical(reflected), obs, atol=1e-7)
    np.testing.assert_allclose(reflected[:,34], reflected[:,0]-reflected[:,4], atol=1e-7)
    np.testing.assert_allclose(reflected[:,38], reflected[:,8]-reflected[:,4], atol=1e-7)
    obs[:,21:29] = np.arange(8)
    np.testing.assert_array_equal(reflect_physical(obs)[0,21:29], [3,2,1,0,7,6,5,4])
    action = np.random.default_rng(913).uniform(-1,1,(16,6))
    mirrored = reflect_arrival(action)
    original, velocity, duration, cap = env.decoder.unpack(action)
    opposite, opposite_v, opposite_t, opposite_cap = env.decoder.unpack(mirrored)
    np.testing.assert_allclose(opposite[:,0], 1-original[:,0])
    np.testing.assert_allclose(opposite[:,1], original[:,1])
    np.testing.assert_allclose(opposite_v, velocity*[-1,1])
    np.testing.assert_array_equal(opposite_t, duration)
    np.testing.assert_array_equal(opposite_cap, cap)


def test_control_potential_requires_recent_reachable_contact_and_grades_slowing():
    env = NeuralTrainingEnv(6, setup_weight=0, control_potential_weight=30,
                            randomize=False, realistic=False)
    env.reset()
    e = env.engine
    e.puck_x[:] = e.paddle_agent_x[:] = .5
    e.puck_y[:] = .4
    e.paddle_agent_y[:] = .3
    e.puck_vx[:] = 0
    e.puck_vy[:] = [0, 2, 6, 0, 0, 0]
    env.contact_clock[0] = env.elapsed
    env.contact_clock[0,3] = -100
    e.puck_y[4] = .9  # Outside the robot's reachable workspace.
    e.puck_y[5] = .3  # Invalid overlap cannot earn control credit.
    p = env.potential()
    assert p[0] > p[1] > p[2] > 0
    np.testing.assert_array_equal(p[3:], 0)


def test_productive_reception_accepts_aimed_fallback_but_not_a_weak_or_missed_return():
    from airhockey.skill_benchmark import Fixtures

    outcomes = []
    for x, incoming in [(.5,6), (.8,6), (.5,2)]:
        env = NeuralTrainingEnv(1, stage=2, productive_receive_drill=True,
                                capture_first=True, fallback_shot_scale=1,
                                randomize=False, realistic=False)
        env.reset(fixtures=Fixtures(np.ones(1,int), np.array([[x,.41,0.,-incoming]]),
                                   np.array([[x,.3]]), np.array([.5])))
        action = np.zeros((1,6))
        action[0,:2] = 2*(np.array([x,.3])-env.decoder.low)/(env.decoder.high-env.decoder.low)-1
        action[0,4:] = -1
        _,reward,terminal,_,_ = env.step(action)
        outcomes.append((terminal[0],reward[0],env.fast_aimed_count[0,0]))
    assert outcomes[0][0] and outcomes[0][1] > 0 and outcomes[0][2] == 1
    assert not outcomes[1][0] and outcomes[1][2] == 0
    assert not outcomes[2][0] and outcomes[2][2] == 0


def test_productive_reception_cannot_claim_a_goal_without_a_successful_touch():
    from airhockey.skill_benchmark import Fixtures

    env = NeuralTrainingEnv(1, stage=2, productive_receive_drill=True,
                            randomize=False, realistic=False, skill_goal_weight=60)
    env.reset(fixtures=Fixtures(np.ones(1,int), np.array([[.5,1.99,0.,10.]]),
                               np.array([[.5,.3]]), np.array([.5])))
    _,reward,terminal,_,info = env.step(np.zeros((1,6)))
    assert terminal[0] and info['goals'][0] == 1
    assert reward[0] < -30


def test_unproductive_return_penalty_exempts_control_fast_shot_and_no_opportunity():
    options = dict(n_envs=4, games=True, stage=5, randomize=False, realistic=False)
    baseline = NeuralTrainingEnv(**options)
    penalized = NeuralTrainingEnv(**options, unproductive_return_penalty=8)
    for env in (baseline, penalized):
        env.reset(seed=741, opponent="external")
        env.engine.puck_x[:] = .5
        env.engine.puck_y[:] = 1.01
        env.engine.puck_vx[:] = env.engine.puck_vy[:] = 0
        env.entry_active[0] = True
        env.entry_speed[0] = 3
        env.entry_reachable_time[0] = [.2, .2, .2, .02]
        env.entry_flags[0] = False
        env.entry_flags[0, 1, 1] = True
        env.entry_flags[0, 2, 3] = True
        env.set_opponent_action(np.zeros((4, 6)))
    _, before, *_ = baseline.step(np.zeros((4, 6)))
    _, after, *_ = penalized.step(np.zeros((4, 6)))
    np.testing.assert_allclose(after - before, [-8, 0, 0, 0], atol=1e-5)
    np.testing.assert_array_equal(penalized.unproductive_returns[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(penalized.entry_outcomes_by_speed.sum(2), penalized.entry_outcomes)
    assert penalized.entry_outcomes_by_speed[0, :, 1, 0].sum() == 4
    assert penalized.productive_entries_by_speed[0, :, 1].sum() == 2
    assert penalized.opportunity_entries_by_speed[0, :, 1].sum() == 3
    assert penalized.productive_opportunities_by_speed[0, :, 1].sum() == 2
