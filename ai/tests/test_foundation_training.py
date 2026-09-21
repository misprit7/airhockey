import numpy as np

from airhockey.foundation_training import FoundationEnv, skill_gate


def test_defense_is_game_request_without_referee_relaunch():
    env = FoundationEnv(16, randomize=False, realistic=False, game_fraction=0.25)
    obs = env.reset(seed=741)
    d = env.defense_trials
    assert d.sum() == 3
    assert np.all(obs[d, 30:34] == [0, 0, 0, 1])
    assert np.all(obs[d, 35] == 1)
    _, _, _, _, info = env.step(np.zeros((16, 3), np.float32))
    assert np.all(info["task"][d] == 4)
    assert not env.base.referee_active_mask[d].any()
    assert env.base.referee_active_mask[:4].all()


def test_gate_rejects_one_bad_skill_and_overload():
    rows = {
        k: dict(
            attempts=100,
            successes=100,
            peak_modeled_load=0.7,
            peak_actual_acceleration=60,
            on_target_contact_trials=100,
        )
        for k in ("stationary", "moving", "cushion", "defense")
    }
    assert skill_gate(dict(tasks=rows), 2)[0]
    rows["defense"]["successes"] = 97
    assert "defense" in skill_gate(dict(tasks=rows), 2)[1]
    rows["defense"]["successes"] = 100
    rows["moving"]["peak_modeled_load"] = 1
    assert "moving_load" in skill_gate(dict(tasks=rows), 2)[1]


def test_selfplay_shot_clock_turns_over_either_side():
    from airhockey.batch_env import BatchAirHockeyEnv

    e = BatchAirHockeyEnv(
        2,
        opponent_body="robot",
        symmetric_referee=True,
        domain_randomize=False,
        opponent_policy="idle",
    )
    e.reset(seed=81)
    e.engine.puck_x[:] = 0.5
    e.engine.puck_y[:] = [0.8, 1.2]
    e.engine.puck_vx[:] = 0.1
    e.engine.puck_vy[:] = 0
    e._prev_in_half[:] = [True, False]
    e._prev_in_far[:] = [False, True]
    e._t_side[:] = e.SHOT_CLOCK_S + 0.1
    _, _, _, _, info = e.step(np.zeros((2, e.action_dim)))
    assert e.engine.puck_vy[0] > 0
    assert e.engine.puck_vy[1] < 0
    assert info["penalty"][0] < 0


def test_independent_drill_heat_does_not_clear_game_heat():
    env = FoundationEnv(16, game_fraction=0.25, randomize=False, realistic=False)
    env.reset(seed=16)
    for load in env.loads:
        load.h[:] = 0.81
    obs = env.reset()
    for load in env.loads:
        np.testing.assert_array_equal(load.h[:4], 0.81)
        assert (load.h[4:] < 0.81).all()
    np.testing.assert_array_equal(obs[4:, 22:30], env.loads[0].features()[4:])


def test_defense_and_game_requests_have_identical_goal_rewards():
    from airhockey.skill_benchmark import Fixtures

    fixture = Fixtures(
        np.array([2]),
        np.array([[0.5, 1.999, 0.0, 2.0]]),
        np.array([[0.5, 0.3]]),
        np.array([0.5]),
    )
    results = []
    for defense in (False, True):
        env = FoundationEnv(1, randomize=False, realistic=False)
        env.reset(seed=12, fixtures=fixture)
        env.task[:] = 3
        env.defense_trials[:] = defense
        env.contacts[:] = 1
        _, reward, term, trunc, info = env.step(np.zeros((1, 3), np.float32))
        results.append(reward)
        assert not term[0]
        if defense:
            assert trunc[0] and info["success"][0]
    np.testing.assert_array_equal(*results)
    assert results[0][0] > 99


def test_possession_curriculum_includes_approach_from_both_sides_of_puck():
    env = FoundationEnv(400, possession_fraction=1, randomize=False, realistic=False)
    env.reset(seed=71)
    shoot = env.task < 2
    e, ws = env.engine, env.base._ws
    assert shoot.sum() == 200
    assert (e.paddle_agent_y[shoot] > e.puck_y[shoot]).sum() > 30
    assert (e.paddle_agent_y[shoot] < e.puck_y[shoot]).sum() > 30
    gap = np.hypot(e.paddle_agent_x - e.puck_x, e.paddle_agent_y - e.puck_y)
    assert (gap[shoot] >= env.cfg.puck_radius + env.cfg.paddle_radius + 0.01).all()
    for axis in ("x", "y"):
        pos = getattr(e, "paddle_agent_" + axis)[shoot]
        assert (pos >= ws["min_" + axis] + 0.01).all()
        assert (pos <= ws["max_" + axis] - 0.01).all()


def test_valid_bank_is_rewarded_and_genuine_miss_is_penalized():
    env = FoundationEnv(1, randomize=False, realistic=False)
    env.reset(seed=11)
    env.task[:] = 3
    env.aim[:] = 0.5
    cfg = env.cfg
    edge = cfg.width - cfg.puck_radius
    vx = (
        3
        * ((edge - 0.5) + cfg.wall_tangential * (edge - 0.5) / cfg.wall_restitution)
        / (cfg.height - 0.5)
    )
    rewards = []
    for x, velocity in ((0.5, [vx, 3]), (0.15, [0, 3])):
        env.engine.puck_x[:], env.engine.puck_y[:] = x, 0.5
        env.shot_armed[:] = True
        env.contact_reward[:] = 0
        env._contact(
            dict(
                indices=np.array([0]),
                body="agent",
                outgoing_before_speed_cap=np.array([velocity]),
                incoming=np.zeros((1, 2)),
            )
        )
        rewards.append(env.contact_reward[0])
    assert rewards[0] > 39
    assert rewards[1] == -40
