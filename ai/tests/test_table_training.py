"""Opt-in table-faithful training: people at the far end, motion costs,
staggered game clocks, drill mix, the trainer's flags. Simulation only."""
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys

import numpy as np
import pytest
import torch

from airhockey.batch_env import _OPP_POLICY_MAP
from airhockey.human_proxy import HumanProxy
from airhockey.neural_player import NeuralPlayer
from airhockey.neural_setup import workspace_bounds
from airhockey.neural_training import MOTION_COLUMNS, NeuralTrainingEnv
from airhockey.neural_vector import ParallelNeuralEnv
from airhockey.table_fidelity import table_fidelity
from airhockey.training_proxies import ProxyPanel, human_proxy_slots, spread_slots

REPO = Path(__file__).resolve().parents[2]
PROXY = _OPP_POLICY_MAP["human_proxy"]
# Game slots 0..7 of 24; drills behind them with fixed roles (defense: id % 20 >= 11).
GAMES = dict(stage=5, seed=4, accel=100.0, game_fraction=1 / 3, shot_conditioned=True,
             continuous_rallies=True, game_episode_seconds=900.0, fixed_practice_roles=True,
             practice_defense_fraction=0.45)


def play(env, steps, seed=0, action=None, reset=True):
    """Random (or fixed) actions on both seats; returns stacked obs, rewards, infos."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        env.set_opponent_action(rng.uniform(-1, 1, (env.n_envs, 6)))
        own = rng.uniform(-1, 1, (env.n_envs, 6)).astype(np.float32)
        obs, reward, term, trunc, info = env.step(own if action is None else action)
        far = env.opponent_obs()
        assert np.isfinite(obs).all() and np.isfinite(reward).all() and np.isfinite(far).all()
        out.append((obs, reward, info, term | trunc))
        if reset and (term | trunc).any():
            env.reset(mask=term | trunc)
            env.opponent_obs()
    return out


def view(n, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros((n, 12))
    v[:, 0], v[:, 1] = rng.uniform(.2, .8, n), rng.uniform(1.2, 1.8, n)
    v[:, 4], v[:, 5] = .5, .15
    v[:, 8], v[:, 9] = .5, 1.7
    return v


# ---------------------------------------------------------------- people
def test_restart_is_a_fresh_person_in_those_rows_only():
    proxy = HumanProxy(6, "table", seed=1)
    proxy.reset(view(6))
    for _ in range(5):
        proxy.act(view(6))
    before = {k: v.copy() for k, v in vars(proxy).items() if isinstance(v, np.ndarray)}
    rows, new = np.array([1, 4]), view(6, seed=3)[[1, 4]]
    command, lag = proxy.restart(rows, new, seed=77)
    fresh = HumanProxy(2, "table", seed=77)
    expected = fresh.reset(new)
    assert all(np.array_equal(a, b) for a, b in zip(command, expected)) and np.array_equal(lag, fresh.time_constant)
    others = np.array([0, 2, 3, 5])
    for name in ("max_speed", "reaction", "plan_speed", "plan_route", "plan_serve", "mode", "strikes", "last_target"):
        assert np.array_equal(getattr(proxy, name)[rows], getattr(fresh, name)), name
        assert np.array_equal(getattr(proxy, name)[others], before[name][others]), name
    assert np.array_equal(proxy._hist[:, rows], fresh._hist)
    assert np.array_equal(proxy._hist[:, others], before["_hist"][:, others])
    assert np.array_equal(proxy.low, before["low"]) and proxy.n == 6
    out = proxy.act(view(6, seed=5))
    assert all(np.isfinite(np.broadcast_to(v, (6,))).all() for v in out)


def test_proxy_slots_are_a_share_of_games_off_the_reference_slots():
    slots = human_proxy_slots(128, 38, 0.3)
    assert slots.sum() == round(0.3 * 38) and not slots[38:].any() and not slots[::4].any()
    assert np.array_equal(slots, human_proxy_slots(128, 38, 0.3))
    assert not human_proxy_slots(128, 38, 0).any()
    assert human_proxy_slots(128, 38, 0.75).sum() == 28      # every game slot with id % 4 != 0
    with pytest.raises(ValueError):
        human_proxy_slots(128, 38, 0.9)
    assert spread_slots(np.arange(10, 30), 0.25).tolist() == [10, 14, 18, 22, 26]
    assert len(spread_slots(np.arange(10), 0)) == 0


def test_panel_draws_a_tier_and_a_new_person_per_game():
    panel = ProxyPanel(40, ("table", "club"), seed=3, strike_speed=(4.0, 9.0))
    command, began, lag = panel.command(view(40))
    assert command.shape == (4, 40) and np.isfinite(command).all()
    assert began.tolist() == list(range(40)) and (lag > 0).all()
    assert set(panel.tier.tolist()) == {0, 1} and panel.games.sum() == 40
    first = panel.tier.copy(), [p.max_speed.copy() for p in panel.proxies]
    panel.restart(np.arange(10), strike=True)
    _, began, _ = panel.command(view(40))
    assert began.tolist() == list(range(10)) and panel.games.sum() == 50
    for j, proxy in enumerate(panel.proxies):
        again = np.flatnonzero((panel.tier[:10] == j))
        assert not np.array_equal(proxy.max_speed[again], first[1][j][again])       # a new person
        assert np.array_equal(proxy.max_speed[10:], first[1][j][10:])                 # the others play on
        assert ((proxy.plan_speed[again] >= 4.0) & (proxy.plan_speed[again] <= 9.0)).all()
    assert (panel.tier[:10] != first[0][:10]).any()
    # A serving tier puts the puck down on its half; the club tier leaves the restart alone.
    scored = np.ones(40, bool)
    rows, x, y = panel.serve(scored)
    assert set(rows.tolist()) == set(np.flatnonzero(panel.tier == 0).tolist())
    assert ((x > 0.3) & (x < 0.8) & (y > 1.2) & (y < 1.7)).all()
    with pytest.raises(ValueError):
        ProxyPanel(4, ("nobody",))


def test_people_take_their_game_slots_and_leave_the_other_games_alone():
    plain = NeuralTrainingEnv(24, **GAMES)
    people = NeuralTrainingEnv(24, **GAMES, human_proxy_share=0.5, human_proxy_tiers=("table", "table_plus"))
    for env in (plain, people):
        env.reset(seed=4)
        env.opponent_obs()
    slots = people._people_game
    assert slots.tolist() == human_proxy_slots(24, 8, 0.5).tolist() and slots.sum() == 4
    assert (people.base._opp_policy_id[slots] == PROXY).all() and people.base._opp_free[slots].all()
    assert (people.base._opp_policy_id[~slots] != PROXY).all()
    assert (plain.base._opp_policy_id != PROXY).all() and plain.people is None
    start = people.engine.paddle_opp_y[slots].copy()
    play(people, 150)
    # The person's mallet is the free body, driven by the proxy's commands
    # and by nothing the trainer sends: it reaches past the robot's box.
    assert people.people.games.sum() >= 4 and np.isfinite(people.base._proxy_target_x).all()
    assert (np.abs(people.engine.paddle_opp_y[slots] - start) > 1e-3).any()
    assert (people.base._opp_policy_id[slots] == PROXY).all()
    assert np.array_equal(people._people_rows, np.flatnonzero(slots))


def test_a_game_ending_brings_a_new_person_and_a_goal_a_serve():
    env = NeuralTrainingEnv(24, **GAMES, human_proxy_share=0.5, human_proxy_tiers=("table",))
    env.reset(seed=4)
    env.opponent_obs()
    play(env, 3, reset=False)
    slot = int(np.flatnonzero(env._people_game)[0])
    row = env._people_row[slot]
    person = env.people.proxies[0].max_speed[row]
    # A goal against the person: the table tier serves from its own half.
    e = env.engine
    e.puck_x[slot], e.puck_y[slot], e.puck_vx[slot], e.puck_vy[slot] = 0.5, 1.97, 0.0, 6.0
    e.paddle_opp_x[slot] = 0.1
    served = False
    for _ in range(6):
        env.set_opponent_action(np.zeros((24, 6)))
        _, _, _, _, info = env.step(np.zeros((24, 6), np.float32))
        env.opponent_obs()
        if info["goals"][slot]:
            served = True
            assert 1.2 < e.puck_y[slot] < 1.7 and e.puck_vx[slot] == 0 and e.puck_vy[slot] == 0
            break
    assert served and env.people.proxies[0].serves[row] == 1
    mask = np.zeros(24, bool)
    mask[slot] = True
    env.reset(mask=mask)
    env.opponent_obs()
    play(env, 1, reset=False)
    assert env.people.games[0] == 5 and env.people.proxies[0].max_speed[row] != person
    assert env.base._opp_policy_id[slot] == PROXY


def test_a_persons_play_is_not_scored_as_a_robots():
    """The far side's shaping income is taken from the learner in games
    against a copy, never in games against a person; goals count in both."""
    def rewards(income, goal=None):
        env = NeuralTrainingEnv(24, **GAMES, human_proxy_share=0.5, human_proxy_tiers=("table", "club"),
                                defense_strike_fraction=0.5)
        env.reset(seed=4)
        env.opponent_obs()
        step = env.base.step

        def stepped(action):
            out = step(action)
            env.reward_events[1] += income      # what a far-side shot or capture would have paid
            if goal is not None:
                env.engine.score_opponent[goal] += 1
            return out
        env.base.step = stepped
        env.set_opponent_action(np.zeros((24, 6)))
        return env, env.step(np.zeros((24, 6), np.float32))[1].astype(np.float64)

    env, plain = rewards(0.0)
    _, charged = rewards(7.0)
    people, games = env._people_game, env.kind == 3
    assert people.sum() == 4 and (games & ~people).sum() == 4 and env.strike_drill.any()
    assert np.allclose((plain - charged)[games & ~people], 7.0, atol=1e-4)
    assert np.array_equal(plain[people], charged[people])
    assert np.array_equal(plain[~games], charged[~games])
    slot = int(np.flatnonzero(people)[0])
    _, conceded = rewards(7.0, goal=slot)
    assert plain[slot] - conceded[slot] > 0.9 * env.game_goal_weight


def test_strike_drills_are_released_by_a_person_in_the_band():
    env = NeuralTrainingEnv(
        40, **{**GAMES, "game_fraction": 0.0}, defense_strike_fraction=0.5, defense_strike_speed=(4.0, 9.0),
        human_proxy_tiers=("table", "club"), defense_windup_fraction=0.85, random_practice_opponent=True,
        defense_block_only=True, defense_clear_reward=300.0)
    env.reset(seed=4)
    env.opponent_obs()
    ids = np.arange(40)
    defense = ids % 20 >= 11
    strike = env._strike_slot
    assert strike.sum() == round(0.5 * defense.sum()) and not strike[~defense].any()
    assert env.strike_drill[strike].all() and not env.strike_drill[~strike].any()
    assert (env.kind[strike] == 2).all() and not env.defense_windup[strike].any()
    e = env.engine
    assert (e.puck_y[strike] > 1.15).all() and (np.hypot(e.puck_vx, e.puck_vy)[strike] == 0).all()
    assert (env.base._opp_policy_id[strike] == PROXY).all() and (env.base._opp_policy_id[~strike] != PROXY).all()
    # The robot waits at its post; the people strike. Speeds at the centre line.
    speeds, previous = [], e.puck_y.copy()
    action = np.zeros((40, 6), np.float32)
    action[:, 1] = -0.8
    for _ in range(900):
        env.set_opponent_action(np.zeros((40, 6)))
        _, _, term, trunc, _ = env.step(action)
        env.opponent_obs()
        crossed = strike & (previous >= 1.0) & (e.puck_y < 1.0) & ~(term | trunc)
        speeds += np.hypot(e.puck_vx, e.puck_vy)[crossed].tolist()
        if (term | trunc).any():
            assert (env.elapsed[(term | trunc) & strike] <= 6.03).all()
            env.reset(mask=term | trunc)
            env.opponent_obs()
        previous = e.puck_y.copy()
    assert len(speeds) >= 20
    assert 3.5 < np.median(speeds) < 8.5 and np.percentile(speeds, 90) < 10.0
    assert env.people.games.sum() > strike.sum()


def test_moving_strike_drills_start_with_the_puck_rolling_to_the_person():
    env = NeuralTrainingEnv(40, **{**GAMES, "game_fraction": 0.0}, defense_strike_fraction=1.0,
                            defense_strike_moving_fraction=1.0)
    env.reset(seed=4)
    strike = env.strike_drill
    assert strike.sum() == 18
    assert (env.engine.puck_vy[strike] >= 1.0).all() and (env.engine.puck_y[strike] <= 1.25).all()


# ------------------------------------------------------------ motion costs
def lunge(env, steps=12):
    """Back and forth across the workspace at the full cap, puck on the far half."""
    env.reset(seed=2)
    e = env.engine
    rewards, motions = [], []
    for t in range(steps):
        e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = 0.5, 1.6, 0.0, 0.0
        action = np.zeros((env.n_envs, 6), np.float32)
        action[:, 1] = 0.9 if (t // 4) % 2 == 0 else -0.9
        action[:, 4], action[:, 5] = -1.0, 1.0
        _, reward, _, _, info = env.step(action)
        rewards.append(reward.astype(np.float64))
        motions.append(info["motion"].astype(np.float64))
    return np.array(rewards), np.array(motions)


def test_motion_is_counted_always_and_charged_only_at_a_weight():
    kw = dict(seed=2, accel=100.0, randomize=False)
    free, counted = lunge(NeuralTrainingEnv(4, **kw))
    assert counted.shape == (12, 4, len(MOTION_COLUMNS))
    column = dict(zip(MOTION_COLUMNS, counted.sum(axis=0).T))
    assert (column["accel_duty_s"] > 0.04).all() and (column["far_half_s"] == pytest.approx(12 * 0.02))
    assert (column["far_cap_s"] > 0.1).all() and (column["reversals"] >= 1).all()
    assert (column["accel_duty"] <= column["accel_duty_s"] + 1e-9).all()
    assert (column["far_cap"] <= column["far_cap_s"] + 1e-9).all()
    for weights in (dict(accel_duty_weight=20.0), dict(reversal_weight=5.0), dict(far_cap_weight=40.0),
                    dict(accel_duty_weight=20.0, reversal_weight=5.0, far_cap_weight=40.0)):
        charged, again = lunge(NeuralTrainingEnv(4, **kw, **weights))
        assert np.array_equal(again, counted)       # a cost never changes the motion itself
        cost = (weights.get("accel_duty_weight", 0) * counted[..., 1] + weights.get("reversal_weight", 0) * counted[..., 2]
                + weights.get("far_cap_weight", 0) * counted[..., 5])
        assert cost.sum() > 0.5 and np.allclose(free - charged, cost, atol=1e-4)


def test_motion_is_the_learners_own_body_never_the_far_sides():
    """A far side thrashing at the full cap while the learner stands still costs nothing."""
    def run(**weights):
        env = NeuralTrainingEnv(8, stage=5, seed=3, accel=100.0, games=True, continuous_rallies=True,
                                randomize=False, **weights)
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        rewards, motions, far = [], [], []
        hold = np.zeros((8, 6), np.float32)
        hold[:, 4], hold[:, 5] = -1.0, 1.0     # one fixed arrival point at the full cap
        for t in range(60):
            e, b = env.engine, env.base
            e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = 0.06, 1.0, 0.0, 0.0
            old = np.column_stack((b._opp_dyn["vx"], b._opp_dyn["vy"]))
            thrash = rng.choice([-0.9, 0.9], (8, 6))
            thrash[:, 5] = 1.0
            env.set_opponent_action(thrash)
            _, reward, _, _, info = env.step(hold)
            far.append(np.linalg.norm(np.column_stack((b._opp_dyn["vx"], b._opp_dyn["vy"])) - old, axis=1) / 0.02)
            rewards.append(reward.astype(np.float64))
            motions.append(info["motion"].astype(np.float64))
        return np.array(rewards), np.array(motions), np.array(far)

    free, motion, far = run()
    charged, again, _ = run(accel_duty_weight=20.0, reversal_weight=5.0)
    settled = slice(20, None)
    assert (far[settled] > 60).mean() > 0.2                         # the far body is over the threshold
    assert np.array_equal(motion, again) and not motion[settled][..., :3].any()
    assert np.allclose(free[settled], charged[settled], atol=1e-5)


def test_motion_thresholds_and_the_own_half():
    kw = dict(seed=2, accel=100.0, randomize=False)
    # Above every threshold the machine can reach: nothing counts as over.
    _, high = lunge(NeuralTrainingEnv(4, **kw, accel_duty_threshold=150.0, far_cap_threshold=100.0, reversal_speed=20.0))
    assert high[..., [0, 1, 2, 4, 5]].sum() == 0 and high[..., 3].sum() > 0
    # The cap is free while the puck is ours to strike.
    env = NeuralTrainingEnv(4, **kw, far_cap_weight=40.0)
    env.reset(seed=2)
    env.engine.puck_x[:], env.engine.puck_y[:], env.engine.puck_vx[:], env.engine.puck_vy[:] = 0.5, 0.7, 0.0, 0.0
    action = np.zeros((4, 6), np.float32)
    action[:, 5] = 1.0
    _, _, _, _, info = env.step(action)
    assert info["motion"][:, 3:6].sum() == 0
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, reversal_weight=-1)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, far_cap_threshold=0)


def test_reversals_are_events_150_ms_apart():
    env = NeuralTrainingEnv(1, seed=0)
    env.reset(seed=0)
    fast = np.array([[2.0, 0.0]])
    signs = [1 if (t // 6) % 2 == 0 else -1 for t in range(40)]
    events = [bool(env._reversal_events(sign * fast)[0]) for sign in signs]
    # A flip every 120 ms is seen for five ticks (against the velocity 100 ms
    # before) and counts once, and not again within 150 ms: the same events
    # as the playability gate's count of the same trace.
    from airhockey.playability import REVERSAL_GAP_S, _starts
    flagged = [t * 0.02 for t in range(5, 40) if signs[t] != signs[t - 5]]
    assert [round(t * 0.02, 2) for t in np.flatnonzero(events)] == [round(t, 2) for t in _starts(flagged, REVERSAL_GAP_S)]
    assert events[6] and not any(events[7:12]) and sum(events) >= 3
    slow = [bool(env._reversal_events(np.array([[1.0 if t % 2 else -1.0, 0.0]]))[0]) for t in range(20)]
    assert not any(slow)


# ---------------------------------------------------------- clocks, drills
def test_staggered_games_start_at_their_own_point_of_the_clock():
    plain = NeuralTrainingEnv(24, **GAMES)
    staggered = NeuralTrainingEnv(24, **GAMES, stagger_games=True)
    first, second = plain.reset(seed=4), staggered.reset(seed=4)
    assert np.array_equal(first, second)
    games = staggered.kind == 3
    assert games.sum() == 8 and (plain.elapsed == 0).all() and (staggered.elapsed[~games] == 0).all()
    clock = staggered.elapsed[games]
    assert len(np.unique(clock)) == 8 and (clock >= 0).all() and (clock < 900).all() and np.ptp(clock) > 300
    a, b = play(plain, 10, reset=False), play(staggered, 10, reset=False)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    # The first game is cut short at the usual length; the next runs in full.
    late = int(np.flatnonzero(games)[np.argmax(clock)])
    staggered.elapsed[late] = 900 - 0.02
    _, _, term, trunc, _ = staggered.step(np.zeros((24, 6), np.float32))
    assert trunc[late] and not term[late]
    staggered.reset(mask=trunc)
    assert staggered.elapsed[late] == 0


def test_left_bank_requests_can_be_weighted():
    assert NeuralTrainingEnv(2, shot_conditioned=True).base._shot_type_p.tolist() == [0, 1 / 3, 1 / 3, 1 / 3]
    env = NeuralTrainingEnv(2, shot_conditioned=True, left_bank_request_weight=2.0)
    assert env.base._shot_type_p.tolist() == [0, 0.5, 0.25, 0.25]
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, shot_conditioned=True, shot_request="left", left_bank_request_weight=2.0)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, left_bank_request_weight=0)


def test_new_options_at_their_defaults_are_the_same_environment():
    explicit = dict(human_proxy_share=0.0, defense_strike_fraction=0.0, left_bank_request_weight=1.0,
                    accel_duty_weight=0.0, reversal_weight=0.0, far_cap_weight=0.0, stagger_games=False,
                    dead_puck_weight=0.0, calm_reachable_scale=None, guard_weight=0.0, guard_unresolved_weight=0.0,
                    far_cap_max_speed=None, accel_effort_weight=0.0, idle_speed_weight=0.0,
                    cross_bank_drill_fraction=0.0, corner_far_start_fraction=0.0,
                    far_cap_brake_seconds=None, dead_puck_cap_weight=0.0, reach_edge_drill_fraction=0.0,
                    back_rail_drill_fraction=0.0, own_corner_weight=0.0, anchor_side_rail_open_y=None,
                    anchor_front_margin=None)
    a, b = NeuralTrainingEnv(24, **GAMES), NeuralTrainingEnv(24, **GAMES, **explicit)
    assert np.array_equal(a.reset(seed=4), b.reset(seed=4))
    for (obs_a, reward_a, info_a, _), (obs_b, reward_b, info_b, _) in zip(play(a, 60), play(b, 60)):
        assert np.array_equal(obs_a, obs_b) and np.array_equal(reward_a, reward_b)
        assert np.array_equal(info_a["motion"], info_b["motion"])
        assert np.array_equal(info_a["anchor_zone"], info_b["anchor_zone"])
    assert a.people is None and not a.strike_drill.any()


def test_a_share_of_workers_can_train_in_the_table_profile():
    options = dict(GAMES, report_sensing=True)
    env = ParallelNeuralEnv(16, 2, auto_reset=True, worker_options=[table_fidelity(), {}], **options)
    try:
        env.reset(seed=4)
        info = None
        for _ in range(3):
            env.set_opponent_action(np.zeros((16, 6), np.float32))
            _, _, term, trunc, info = env.step(np.zeros((16, 6), np.float32))
            if (term | trunc).any():
                env.reset(mask=term | trunc)
        # Only the first worker's environment reports lifts; the rest read zero.
        assert info["lift_events"].shape == (16,) and info["motion"].shape == (16, len(MOTION_COLUMNS))
        assert (info["lift_events"][8:] == 0).all()
    finally:
        env.close()
    with pytest.raises(ValueError):
        ParallelNeuralEnv(16, 2, worker_options=[{}], **options)


# ------------------------------------------------- round 2 (after v5a)
def rest_puck(env, x, y, steps, action=None, puck_speed=0.0):
    """Hold a puck at (x, y) at a given speed while the mallet holds one arrival."""
    e = env.engine
    rewards, motions, clocks = [], [], []
    action = np.zeros((env.n_envs, 6), np.float32) if action is None else action
    for _ in range(steps):
        e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = x, y, puck_speed, 0.0
        _, reward, _, _, info = env.step(action)
        rewards.append(reward.astype(np.float64))
        motions.append(info["motion"].astype(np.float64))
        clocks.append(info["dead_puck_clock"].copy())
    return np.array(rewards), np.array(motions), np.array(clocks)


COLUMN = {name: i for i, name in enumerate(MOTION_COLUMNS)}
QUIET = dict(seed=2, accel=100.0, randomize=False, games=True, continuous_rallies=True, stage=5,
             game_episode_seconds=900.0)


def test_a_puck_lying_within_reach_is_charged_after_the_grace():
    free, charged = NeuralTrainingEnv(2, **QUIET), NeuralTrainingEnv(2, **QUIET, dead_puck_weight=40.0, dead_puck_grace=1.0)
    for env in (free, charged):
        env.reset(seed=2)
    a, counted, clock = rest_puck(free, 0.77, 0.80, 80)
    b, again, _ = rest_puck(charged, 0.77, 0.80, 80)
    dead = counted[..., COLUMN["dead_puck_s"]]
    # 1.6 s lying there: nothing for the first second, then every tick.
    assert np.array_equal(counted, again) and not dead[:48].any() and np.allclose(dead[52:], 0.02)
    assert np.allclose(clock[-1], 1.6, atol=0.021) and np.allclose(a - b, 40.0 * dead, atol=1e-4)
    # A nudge neither buys the grace back nor pauses the cost; a release, the far half or a puck
    # out of reach does. A slow puck that has not lain yet is not on the clock.
    c, nudged, clock = rest_puck(charged, 0.77, 0.80, 5, puck_speed=0.6)
    assert (clock[-1] > 1.6).all() and np.allclose(nudged[..., COLUMN["dead_puck_s"]], 0.02)
    _, nudged, _ = rest_puck(charged, 0.77, 0.80, 2)
    assert (nudged[..., COLUMN["dead_puck_s"]] > 0).all()
    for x, y, speed in ((0.77, 0.80, 1.2), (0.5, 1.4, 0.0), (0.5, 0.97, 0.0)):
        rest_puck(charged, 0.77, 0.80, 60)
        rest_puck(charged, x, y, 1, puck_speed=speed)
        assert (charged.dead_puck_clock == 0).all()
    _, arriving, _ = rest_puck(charged, 0.77, 0.80, 80, puck_speed=0.6)
    assert (charged.dead_puck_clock == 0).all() and not arriving[..., COLUMN["dead_puck_s"]].any()
    charged.dead_puck_clock[:] = 5
    charged.reset(mask=np.array([True, False]))
    assert charged.dead_puck_clock.tolist() == [0, 5]
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, dead_puck_weight=-1)


def test_a_puck_waiting_to_be_struck_is_not_a_quiet_table():
    """Calm accel can be waived for a slow puck within reach, and only there."""
    full = np.zeros((2, 6), np.float32)
    full[:, 5] = 1.0                                   # the full cap, standing still
    def cost(x, y, **options):
        out = []
        for weight in (0.0, 200.0):
            env = NeuralTrainingEnv(2, **QUIET, calm_accel_weight=weight, calm_accel_power=1.0, **options)
            env.reset(seed=2)
            out.append(rest_puck(env, x, y, 10, action=full)[0][3:].sum())
        return out[0] - out[1]
    assert cost(0.6, 0.6) > 1 and cost(0.6, 0.6, calm_reachable_scale=0.0) == pytest.approx(0, abs=1e-4)
    for x, y in ((0.6, 1.5), (0.6, 0.97)):             # the far half; the own half out of reach
        assert cost(x, y, calm_reachable_scale=0.0) == pytest.approx(cost(x, y), abs=1e-4) and cost(x, y) > 1


def sprint(env, steps=30):
    """v5a after a set strike: flying at the front limit at the full cap,
    then the cap dropped to its floor with 3 m/s still on."""
    env.reset(seed=2)
    e = env.engine
    rewards, motions = [], []
    for t in range(steps):
        e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = 0.5, 1.6, 0.0, 0.0
        action = np.zeros((env.n_envs, 6), np.float32)
        action[:, 1] = action[:, 3] = 1.0
        action[:, 4], action[:, 5] = -1.0, (1.0 if t < 6 else -1.0)
        _, reward, _, _, info = env.step(action)
        rewards.append(reward.astype(np.float64))
        motions.append(info["motion"].astype(np.float64))
    return np.array(rewards), np.array(motions)


def test_guard_interventions_and_effort_are_counted_and_charged():
    kw = dict(seed=2, accel=100.0, randomize=False)
    free, counted = sprint(NeuralTrainingEnv(4, **kw))
    column = {name: counted[..., i] for name, i in COLUMN.items()}
    # The guard takes over once the stop no longer fits, and brakes for it.
    assert (column["guard_changed"][4:10].sum(axis=0) >= 1).all() and not column["guard_changed"][:4].any()
    assert column["guard_changed"].sum() > 0 and set(np.unique(column["guard_changed"])) <= {0.0, 1.0}
    assert (column["guard_unresolved"] <= column["guard_changed"]).all()
    # Quadratic above 40 m/s^2: a tick at the full cap is 0.02 * (1 - 0.16).
    assert column["accel_effort"].max() == pytest.approx(0.02 * 0.84, rel=0.1)
    assert ((column["accel_effort"] > 0) >= (column["accel_duty_s"] > 0)).all()
    weights = dict(guard_weight=10.0, guard_unresolved_weight=50.0, accel_effort_weight=100.0)
    charged, again = sprint(NeuralTrainingEnv(4, **kw, **weights))
    cost = 10 * column["guard_changed"] + 50 * column["guard_unresolved"] + 100 * column["accel_effort"]
    assert np.array_equal(again, counted) and np.allclose(free - charged, cost, atol=1e-4)
    # The idle path: metres moved under the calm factor (puck and far mallet
    # still); counted in every slot, charged in full games.
    walked, _ = sprint(NeuralTrainingEnv(4, **kw, idle_speed_weight=20.0))
    assert column["idle_path"].sum(axis=0).min() > 0.5 and np.allclose(free, walked, atol=1e-5)
    game, path = sprint(NeuralTrainingEnv(4, **QUIET))
    paid, _ = sprint(NeuralTrainingEnv(4, **QUIET, idle_speed_weight=20.0))
    assert np.allclose(game - paid, 20 * path[..., COLUMN["idle_path"]], atol=1e-4) and (game - paid).sum() > 20


def test_the_brake_after_a_strike_is_not_an_idle_cap():
    kw = dict(seed=2, accel=100.0, randomize=False)
    _, always = lunge(NeuralTrainingEnv(4, **kw), steps=40)
    _, slow = lunge(NeuralTrainingEnv(4, **kw, far_cap_max_speed=1.0), steps=40)
    cap, gated = always[..., COLUMN["far_cap_s"]], slow[..., COLUMN["far_cap_s"]]
    # The same motion; charged only on ticks begun slower than 1 m/s.
    assert np.array_equal(always[..., :4], slow[..., :4]) and (gated <= cap).all()
    assert 0 < gated.sum() < 0.5 * cap.sum()
    assert np.array_equal(slow[..., COLUMN["far_cap"]] > 0, gated > 0)


def test_cross_bank_drills_start_in_the_hover_that_stalled():
    env = NeuralTrainingEnv(400, **{**GAMES, "game_fraction": 0.0}, cross_bank_drill_fraction=1.0,
                            stationary_rest_fraction=0.5)
    env.reset(seed=4)
    e, rows = env.engine, env.kind == 0
    assert rows.sum() == 80
    x, y, request = e.puck_x[rows], e.puck_y[rows], env.base._shot_type[rows]
    assert (np.abs(x - 0.5) >= 0.05).all() and (np.abs(x - 0.5) <= 0.35).all() and (y >= 0.60).all()
    assert (np.hypot(e.puck_vx, e.puck_vy)[rows] == 0).all()
    # Mostly the bank off the far rail: left (1) from the right side, right (2) from the left.
    cross = np.where(x > 0.5, 1, 2)
    assert 0.6 < (request == cross).mean() < 0.95 and (request > 0).all()
    # The hover: a full-length arrival through the puck at the cap floor, the mallet behind it.
    held = env.last_action[rows]
    hover = (held[:, 4] > 0.85) & (held[:, 5] < -0.7)
    assert 0.4 < hover.mean() < 0.75
    gap = np.hypot(e.paddle_agent_x[rows] - x, e.paddle_agent_y[rows] - y)
    assert (gap > 0.09).all() and (gap[hover] < 0.32).all() and (e.paddle_agent_y[rows][hover] < y[hover]).all()
    toward = np.sign(held[hover, 2]) == np.where(x[hover] > 0.5, -1, 1)
    assert toward.all() and (held[hover, 3] > 0).all()
    # Held as it is, the hover stays a hover: the puck is not touched.
    for _ in range(50):
        action = env.last_action.copy()
        env.set_opponent_action(np.zeros((400, 6), np.float32))
        env.step(action)
    assert env.touch_count[0][rows][hover].mean() < 0.2 and (env.dead_puck_clock[rows][hover] > 0.9).mean() > 0.8


def test_corner_drills_can_start_from_anywhere():
    options = dict({**GAMES, "game_fraction": 0.0}, corner_drill_fraction=1.0)
    near, far = NeuralTrainingEnv(400, **options), NeuralTrainingEnv(400, **options, corner_far_start_fraction=0.5)
    near.reset(seed=4)
    far.reset(seed=4)
    rows = near.kind == 0
    def gap(env):
        e = env.engine
        return np.hypot(e.paddle_agent_x - e.puck_x, e.paddle_agent_y - e.puck_y)[rows]
    assert np.array_equal(near.engine.puck_x, far.engine.puck_x) and gap(near).max() < 0.30
    moved = gap(far) != gap(near)
    assert 0.35 < moved.mean() < 0.65 and gap(far)[moved].max() > 0.6 and gap(far).min() > 0.09


def test_dead_puck_exploration_widens_time_and_cap_where_the_puck_lies():
    from airhockey.neural_finetune import dead_puck_exploration_scale
    log_std = torch.full((3, 6), float(np.log(0.05)))
    dwelling = torch.tensor([True, False, True])
    out = dead_puck_exploration_scale(log_std, dwelling, 1.0, 2.0).exp()
    assert torch.allclose(out[0], torch.tensor([.05, .05, .05, .05, 1.0, 2.0])) and torch.allclose(out[1], log_std[1].exp())
    wide = torch.full((3, 6), float(np.log(2.5)))
    assert torch.equal(dead_puck_exploration_scale(wide, dwelling, 1.0, 2.0), wide)      # a floor, never a cap
    assert dead_puck_exploration_scale(log_std, dwelling, 0.0, 0.0) is log_std
    assert dead_puck_exploration_scale(log_std, None, 1.0, 2.0) is log_std


# ------------------------------------------- round 3: imitation anchors
def test_anchor_zone_is_the_open_table_or_the_rail_strip_where_a_puck_lies():
    from airhockey.neural_training import ANCHOR_NONE, ANCHOR_OPEN, ANCHOR_RAIL, RAIL_ZONE_MARGIN
    env = NeuralTrainingEnv(2, **QUIET)
    env.reset(seed=2)
    low, high = env.decoder.low, env.decoder.high

    def zone(x, y, speed=0.0, steps=3):
        e, info = env.engine, None
        for _ in range(steps):
            e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = x, y, speed, 0.0
            info = env.step(np.zeros((2, 6), np.float32))[4]
        return info["anchor_zone"].tolist(), info["dead_puck_clock"]

    assert zone(0.77, 0.80) == ([ANCHOR_OPEN] * 2, pytest.approx([0.06, 0.06], abs=1e-6))
    assert zone(0.10, 0.80)[0] == [ANCHOR_OPEN] * 2                    # the stall map's rail column: the teacher's
    # The strip the corner and edge drills use: beside a side rail, behind the back edge.
    for x, y in ((low[0] + RAIL_ZONE_MARGIN - .002, 0.5), (high[0] - RAIL_ZONE_MARGIN + .002, 0.5),
                 (0.5, low[1] + RAIL_ZONE_MARGIN - .002), (0.05, 0.05)):
        assert zone(x, y)[0] == [ANCHOR_RAIL] * 2
    # Not lying: in flight, on the far half, out of reach.
    for x, y, speed in ((0.77, 0.80, 2.0), (0.5, 1.4, 0.0), (0.5, 0.97, 0.0)):
        assert zone(x, y, speed)[0] == [ANCHOR_NONE] * 2
    # A corner / edge drill slot is the reference's from its first step to its last, wherever the puck is.
    env.edge_drill[:] = [True, False]
    assert zone(0.5, 1.4)[0] == [ANCHOR_RAIL, ANCHOR_NONE] and zone(0.77, 0.80)[0] == [ANCHOR_RAIL, ANCHOR_OPEN]
    drills = NeuralTrainingEnv(200, **{**GAMES, "game_fraction": 0.0}, corner_drill_fraction=1.0)
    drills.reset(seed=4)
    drills.set_opponent_action(np.zeros((200, 6), np.float32))
    info = drills.step(np.zeros((200, 6), np.float32))[4]
    rows = drills.kind == 0
    assert rows.any() and (info["anchor_zone"][rows] == ANCHOR_RAIL).all() and drills.edge_drill[rows].all()
    # Bookkeeping only: the environment is the same one.
    assert info["anchor_zone"].dtype == np.int8


# ------------------------------------------- round 4: what a3 still did
def test_only_the_stop_after_its_own_strike_is_free_of_the_far_cap():
    """Round 2 freed every fast tick; the walk back to guard is charged again."""
    kw = dict(seed=2, accel=100.0, randomize=False, far_cap_weight=40.0, far_cap_max_speed=1.0)

    def walk(contact_at=None, **options):
        env = NeuralTrainingEnv(4, **kw, **options)
        env.reset(seed=2)
        e, rewards, motions, fast = env.engine, [], [], []
        for t in range(40):
            e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = 0.5, 1.6, 0.0, 0.0
            if t == contact_at:
                env.contact_clock[0] = env.elapsed          # its own strike, this tick
            action = np.zeros((4, 6), np.float32)
            action[:, 1] = 0.9 if (t // 8) % 2 == 0 else -0.9
            action[:, 4], action[:, 5] = -1.0, 1.0
            fast.append(np.hypot(env.base._agent_dyn["vx"], env.base._agent_dyn["vy"]) >= 1.0)
            _, reward, _, _, info = env.step(action)
            rewards.append(reward.astype(np.float64))
            motions.append(info["motion"].astype(np.float64))
        return np.array(rewards), np.array(motions)[..., COLUMN["far_cap_s"]], np.array(fast)

    free, gated, fast = walk()
    _, never, _ = walk(far_cap_brake_seconds=0.25)
    paid, after, _ = walk(contact_at=10, far_cap_brake_seconds=0.25)
    # The speed gate alone: no fast tick is charged. With the window and no contact: every tick is.
    assert fast.mean() > 0.5 and not gated[fast].any() and np.allclose(gated[~fast], 0.02)
    assert np.allclose(never, 0.02)
    # A contact at tick 10 frees the fast ticks of the next 0.25 s, and only those.
    window = np.zeros(40, bool)
    window[10:22] = True
    assert not after[window[:, None] & fast].any() and np.allclose(after[~(window[:, None] & fast)], 0.02)
    assert (window[:, None] & fast).sum() >= 4 * 8
    assert np.allclose(free - paid, 40.0 * 0.02 * (after - gated) / 0.02, atol=1e-4) and (free - paid).sum() > 1
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, far_cap_brake_seconds=0.25)             # needs the speed gate


def test_a_cap_held_up_beside_a_lying_puck_is_charged_after_the_strikes_time():
    full = np.zeros((2, 6), np.float32)
    full[:, 5] = 1.0
    floor = np.zeros((2, 6), np.float32)
    floor[:, 5] = -1.0
    def lie(action, steps=200, **options):
        env = NeuralTrainingEnv(2, **QUIET, **options)
        env.reset(seed=2)
        return rest_puck(env, 0.77, 0.80, steps, action=action)
    a, counted, clock = lie(full)
    b, again, _ = lie(full, dead_puck_cap_weight=100.0, dead_puck_cap_grace=2.5)
    cap = counted[..., COLUMN["dead_puck_cap"]]
    # Nothing for the first 2.5 s (the run into a strike), then 0.95 of the cap's range every tick: 95 a second.
    late = clock[:, 0] > 2.49
    assert np.array_equal(counted, again) and 70 < late.sum() < 80
    assert not cap[~late].any() and np.allclose(cap[late], 0.02 * 0.95, atol=1e-6)
    assert np.allclose(a - b, 100.0 * cap, atol=1e-4) and (a - b)[late].sum(axis=0) == pytest.approx(late.sum() * 1.9, rel=1e-3)
    # The cap at its floor beside the same puck costs nothing; nor does any cap once the puck is released.
    _, calm, _ = lie(floor, dead_puck_cap_weight=100.0)
    assert not calm[..., COLUMN["dead_puck_cap"]].any()
    env = NeuralTrainingEnv(2, **QUIET, dead_puck_cap_weight=100.0)
    env.reset(seed=2)
    rest_puck(env, 0.77, 0.80, 200, action=full)
    _, gone, _ = rest_puck(env, 0.77, 0.80, 5, action=full, puck_speed=1.2)
    assert not gone[..., COLUMN["dead_puck_cap"]].any()
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, dead_puck_cap_weight=-1)


def test_reach_edge_drills_put_the_puck_where_no_other_drill_did():
    options = dict({**GAMES, "game_fraction": 0.0})
    env = NeuralTrainingEnv(600, **options, reach_edge_drill_fraction=1.0)
    plain = NeuralTrainingEnv(600, **options)
    env.reset(seed=4)
    plain.reset(seed=4)
    e, rows = env.engine, env.kind == 0
    low, high = env.decoder.low, env.decoder.high
    radius = env.cfg.puck_radius + env.cfg.paddle_radius
    x, y = e.puck_x[rows], e.puck_y[rows]
    # The stationary drill as it was stops 20 mm short of the front edge of reach and 90 mm off the rails.
    assert plain.engine.puck_y[rows].max() < high[1] + radius - 0.02 + 1e-9
    assert plain.engine.puck_x[rows].min() > 0.16
    front = y > high[1] + radius - 0.05 - 1e-9
    side = ~front
    assert 0.45 < front.mean() < 0.75 and rows.sum() == 120
    # Front: the last 50 mm of reach across the whole width, still touchable.
    assert (y[front] < high[1] + radius - 0.003).all() and x[front].min() < 0.15 and x[front].max() > 0.85
    # Side: against either rail, in the strip the mallet's centre cannot get outside of, forward of y 0.58.
    assert ((x[side] < low[0] + 0.015) | (x[side] > high[0] - 0.015)).all() and (y[side] >= 0.58).all()
    assert (x[side] < 0.5).any() and (x[side] > 0.5).any()
    assert (np.hypot(e.puck_vx, e.puck_vy)[rows] == 0).all() and not env.edge_drill[rows].any()
    gap = np.hypot(e.paddle_agent_x - e.puck_x, e.paddle_agent_y - e.puck_y)[rows]
    assert gap.min() > radius and (gap < 0.40).mean() > 0.35 and (gap > 0.45).mean() > 0.1
    # The other slots are the ones they were.
    assert np.array_equal(env.kind, plain.kind) and np.array_equal(e.puck_y[env.kind == 2], plain.engine.puck_y[plain.kind == 2])
    # Every one of them is within the mallet's reach and on the dead-puck clock.
    env.set_opponent_action(np.zeros((600, 6), np.float32))
    for _ in range(3):
        info = env.step(env.last_action.copy())[4]
    assert (info["dead_puck_clock"][rows] > 0.03).mean() > 0.95
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, reach_edge_drill_fraction=1.5)


def test_forward_side_rail_pucks_are_the_teachers_and_front_edge_pucks_nobodys():
    from airhockey.neural_training import ANCHOR_NONE, ANCHOR_OPEN, ANCHOR_RAIL
    from airhockey.neural_finetune import ATTACK, resting_teacher_mask

    def zone(env, x, y):
        e, info = env.engine, None
        for _ in range(3):
            e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = x, y, 0.0, 0.0
            info = env.step(np.zeros((2, 6), np.float32))[4]
        return info["anchor_zone"].tolist()

    old, new = NeuralTrainingEnv(2, **QUIET), NeuralTrainingEnv(2, **QUIET, anchor_side_rail_open_y=0.55,
                                                                anchor_front_margin=0.06)
    old.reset(seed=2)
    new.reset(seed=2)
    front = new.decoder.high[1]
    # The left rail forward of y 0.55 (v4 leaves 16 of 30 there, the teacher none): the reference's, now the teacher's.
    for x in (0.075, 0.925):
        assert zone(old, x, 0.72) == [ANCHOR_RAIL] * 2 and zone(new, x, 0.72) == [ANCHOR_OPEN] * 2
        assert zone(old, x, 0.40) == zone(new, x, 0.40) == [ANCHOR_RAIL] * 2
    # Past the forward limit by more than 60 mm the teacher hovers at 90 m/s^2: no label.
    assert zone(old, 0.5, front + 0.07) == [ANCHOR_OPEN] * 2 and zone(new, 0.5, front + 0.07) == [ANCHOR_NONE] * 2
    assert zone(new, 0.075, front + 0.07) == [ANCHOR_NONE] * 2
    assert zone(new, 0.5, front + 0.05) == zone(new, 0.5, 0.6) == [ANCHOR_OPEN] * 2
    assert zone(new, 0.5, 0.05) == [ANCHOR_RAIL] * 2 and zone(new, 0.05, 0.05) == [ANCHOR_RAIL] * 2
    # The teacher's cap is a strike's: its window on the clock can close.
    zones = torch.tensor([1, 1, 1, 2, 1])
    clock = torch.tensor([0.5, 1.5, 3.0, 1.5, 1.5])
    modes = torch.tensor([ATTACK, ATTACK, ATTACK, ATTACK, ATTACK + 1])
    assert resting_teacher_mask(zones, clock, modes, 1.0).tolist() == [False, True, True, False, False]
    assert resting_teacher_mask(zones, clock, modes, 1.0, 2.5).tolist() == [False, True, False, False, False]


def test_a_slow_puck_sent_into_its_own_back_corner_is_counted_once():
    env = NeuralTrainingEnv(2, **QUIET)
    charged = NeuralTrainingEnv(2, **QUIET, own_corner_weight=300.0)
    column = COLUMN["own_corner"]

    def run(env, path, touched=True):
        env.reset(seed=2)
        e, rewards, events = env.engine, [], []
        for x, y, speed in path:
            e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = x, y, -speed, 0.0
            env.own_touched[:] = touched
            _, reward, _, _, info = env.step(np.zeros((2, 6), np.float32))
            env.own_touched[:] = touched
            rewards.append(reward.astype(np.float64))
            events.append(info["motion"][:, column].astype(np.float64))
        return np.array(rewards), np.array(events)

    # Dribbled along the back rail at 0.5 m/s into the pocket, where it stops.
    dribble = [(0.30, 0.045, 0.5)] * 3 + [(0.06, 0.045, 0.5)] * 2 + [(0.045, 0.045, 0.0)] * 10
    free, events = run(env, dribble)
    paid, again = run(charged, dribble)
    assert np.array_equal(events, again) and events.sum(axis=0).tolist() == [1, 1] and events[5].tolist() == [1, 1]
    assert np.allclose(free - paid, 300.0 * events, atol=1e-3)
    # Not the learner's doing, or fast enough to bounce back out, or only passing through: nothing.
    assert not run(env, dribble, touched=False)[1].any()
    assert not run(env, [(0.30, 0.045, 3.0)] * 3 + [(0.06, 0.045, 3.0)] * 2 + [(0.045, 0.045, 0.0)] * 10)[1].any()
    assert not run(env, [(0.30, 0.045, 0.5)] * 3 + [(0.06, 0.045, 0.5)] * 2 + [(0.30, 0.30, 0.5)] * 5)[1].any()
    # Lying on the rail short of the pocket: nothing. Nor in a corner / edge drill slot, whose puck starts there.
    assert not run(env, [(0.30, 0.045, 0.0)] * 10)[1].any()
    env.reset(seed=2)
    env.edge_drill[:] = [True, False]
    e = env.engine
    for x, y, speed in dribble:
        e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = x, y, -speed, 0.0
        env.own_touched[:] = True
        events = env.step(np.zeros((2, 6), np.float32))[4]["motion"][:, column]
        env.edge_drill[:] = [True, False]
        if events.any():
            assert events.tolist() == [0, 1]
    assert env._pocket_counted.tolist() == [False, True]
    # A puck a corner drill put in the pocket, touched there by the mallet: nothing either.
    corner = NeuralTrainingEnv(200, **{**GAMES, "game_fraction": 0.0}, corner_drill_fraction=1.0, own_corner_weight=300.0)
    corner.reset(seed=4)
    rows = corner.kind == 0
    assert corner._pocket_inside[rows].mean() > 0.3 and not corner._pocket_owned.any()
    corner.set_opponent_action(np.zeros((200, 6), np.float32))
    x, y = corner.engine.puck_x.copy(), corner.engine.puck_y.copy()
    for _ in range(5):
        corner.engine.puck_x[rows], corner.engine.puck_y[rows] = x[rows], y[rows]
        corner.engine.puck_vx[rows] = corner.engine.puck_vy[rows] = 0
        corner.own_touched[:] = True
        info = corner.step(np.zeros((200, 6), np.float32))[4]
        assert not info["motion"][rows & corner._pocket_inside, column].any()
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, own_corner_weight=-1)


def test_back_rail_drills_start_before_the_corner():
    options = dict({**GAMES, "game_fraction": 0.0})
    env = NeuralTrainingEnv(600, **options, back_rail_drill_fraction=1.0)
    env.reset(seed=4)
    e, rows = env.engine, env.kind == 0
    low, high = env.decoder.low, env.decoder.high
    x, y, speed = e.puck_x[rows], e.puck_y[rows], np.hypot(e.puck_vx, e.puck_vy)[rows]
    assert (x > low[0] + 0.02).all() and (x < high[0] - 0.02).all() and (y < low[1] + 0.021).all()
    assert not env._in_corner_pocket()[rows].any() and not env.edge_drill[rows].any()
    assert 0.3 < (speed > 0).mean() < 0.7 and speed.max() <= 0.8 and (e.puck_vy[rows] == 0).all()
    gap = np.hypot(e.paddle_agent_x - e.puck_x, e.paddle_agent_y - e.puck_y)[rows]
    assert gap.min() > env.cfg.puck_radius + env.cfg.paddle_radius


def test_anchor_errors_keep_a_gradient_where_the_hover_is_saturated():
    from airhockey.neural_finetune import (ATTACK, GUARD, PRE_SQUASH_LIMIT, THREAT, anchor_error, anchor_mean,
                                           rail_reference_mask, resting_teacher_mask, teacher_error)
    # The hover: a full-length arrival at the cap floor. The teacher: go now, at the cap.
    hover = torch.tensor([[0.5, 0.9, -0.7, 1.0, 2.0, -1.8]], requires_grad=True)
    strike = torch.tensor([[0.95, 0.45, -0.26, 0.21, -0.4, 0.89]])
    squashed = anchor_error(hover, strike, "squashed")
    assert torch.allclose(squashed, teacher_error(hover.tanh(), strike))          # the teacher regression's own space
    raw = anchor_error(hover, strike, "pre-squash")
    assert torch.allclose(raw, (hover - torch.atanh(strike)).square().mean(-1))
    g_squashed, = torch.autograd.grad(squashed.sum(), hover)
    g_raw, = torch.autograd.grad(raw.sum(), hover)
    # Both push the time down and the cap up; pre-squash about ten times harder on the saturated time.
    assert (g_squashed[0, 4] > 0) and (g_squashed[0, 5] < 0) and (g_raw[0, 4] > 0) and (g_raw[0, 5] < 0)
    assert g_raw[0, 4] > 8 * g_squashed[0, 4] and g_raw[0, 5].abs() > 3 * g_squashed[0, 5].abs()
    # A label at the edge of the box asks for "at least there": finite, and no pull back from beyond.
    edge = torch.tensor([[1.0, -1.0, 0.0, 0.0, -1.0, 1.0]])
    beyond = torch.tensor([[3.0, -2.5, 0.0, 0.0, -2.2, 2.4]], requires_grad=True)
    error = anchor_error(beyond, edge, "pre-squash")
    assert float(error.detach()) == 0 and not torch.autograd.grad(error.sum(), beyond)[0].any()
    short = anchor_error(torch.zeros(1, 6), edge, "pre-squash")
    assert torch.allclose(short, torch.tensor(4 * np.arctanh(PRE_SQUASH_LIMIT) ** 2 / 6, dtype=torch.float32))
    with pytest.raises(ValueError):
        anchor_error(hover, strike, "logit")
    # Some outputs only (--teacher-resting-dims): the others carry no error and no gradient, and a weight
    # keeps its meaning per dimension (still a mean over six).
    from airhockey.neural_finetune import ANCHOR_DIMS
    assert ANCHOR_DIMS == {"all": None, "time-cap": (4, 5), "cap": (5,)}
    for space in ("squashed", "pre-squash"):
        full, = torch.autograd.grad(anchor_error(hover, strike, space).sum(), hover)
        cap, = torch.autograd.grad(anchor_error(hover, strike, space, dims=ANCHOR_DIMS["cap"]).sum(), hover)
        both, = torch.autograd.grad(anchor_error(hover, strike, space, dims=ANCHOR_DIMS["time-cap"]).sum(), hover)
        assert not cap[0, :5].any() and cap[0, 5] == full[0, 5]
        assert not both[0, :4].any() and torch.equal(both[0, 4:], full[0, 4:])
    assert float(anchor_error(beyond, edge, "pre-squash", dims=(5,)).detach()) == 0
    # Mean over the anchored samples, with a floor on how few may carry it; nothing anchored is exactly zero.
    values = torch.tensor([4.0, 2.0, 6.0, 8.0], requires_grad=True)
    assert anchor_mean(values, torch.tensor([True, False, True, False]), min_count=2).item() == 5
    assert anchor_mean(values, torch.tensor([True, False, False, False]), min_count=2).item() == 2
    none = anchor_mean(values, torch.zeros(4, dtype=torch.bool))
    assert none.item() == 0 and not torch.autograd.grad(none, values)[0].any()
    # Teacher: a puck that has lain on the open table past the start, where the teacher attacks it.
    zone = torch.tensor([1, 1, 1, 2, 0, 1], dtype=torch.int8)
    clock = torch.tensor([1.5, 0.5, 1.5, 1.5, 0.0, 1.5])
    modes = torch.tensor([ATTACK, ATTACK, GUARD, ATTACK, ATTACK, THREAT], dtype=torch.int8)
    assert resting_teacher_mask(zone, clock, modes, 1.0).tolist() == [True, False, False, False, False, False]
    assert resting_teacher_mask(zone, clock, modes, 0.0).tolist() == [True, True, False, False, False, False]
    assert rail_reference_mask(zone).tolist() == [False, False, False, True, False, False]


# ----------------------------------------------------------------- trainer
def load_trainer(root):
    spec = importlib.util.spec_from_file_location("trainer_table", root / "ai/bin/train_neural_player.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_sim_profile_flags_become_environment_options():
    trainer = load_trainer(REPO)
    parse = lambda *argv: trainer.build_parser().parse_args(["--run-name", "x", *argv])
    assert trainer.sim_profile_options(parse()) == {}
    assert trainer.sim_profile_options(parse("--sim-profile", "table-20261029")) == table_fidelity()
    assert trainer.sim_profile_options(parse("--sim-profile", "table-20261029:contact,lift")) == table_fidelity(
        sensing=False, actuator=False)
    changed = trainer.sim_profile_options(parse(
        "--sim-profile", "table-20261029", "--sim-option", 'mallet_lift={"off_probability": 0.1}',
        "--sim-option", "late_command_probability=null"))
    assert changed["mallet_lift"] == {"off_probability": 0.1} and "late_command_probability" not in changed
    NeuralTrainingEnv(2, **changed)
    for bad in (("--sim-option", "mallet_lift=true"), ("--sim-profile", "nowhere"),
                ("--sim-profile", "table-20261029", "--sim-option", "gravity=1")):
        with pytest.raises(ValueError):
            trainer.sim_profile_options(parse(*bad))
    assert trainer.profile_workers(16, 0.5).tolist() == [True, False] * 8
    assert trainer.profile_workers(4, 1.0).all() and trainer.profile_workers(4, 0.25).sum() == 1
    # Every new flag is off by default: the environment arguments of an old command.
    old = trainer.environment_options(parse(), shot_conditioned=False)
    assert old["human_proxy_share"] == 0 and old["defense_strike_fraction"] == 0 and not old["stagger_games"]
    assert old["accel_duty_weight"] == old["reversal_weight"] == old["far_cap_weight"] == 0
    assert old["left_bank_request_weight"] == 1 and old["human_proxy_tiers"] == ("table",)
    assert old["dead_puck_weight"] == old["guard_weight"] == old["accel_effort_weight"] == old["idle_speed_weight"] == 0
    assert old["calm_reachable_scale"] is None and old["far_cap_max_speed"] is None
    assert old["cross_bank_drill_fraction"] == old["corner_far_start_fraction"] == 0
    assert parse().dead_puck_exploration == [0, 0] and parse().dead_puck_exploration_hold == 0


def save_player(path, net, **fields):
    path.parent.mkdir(parents=True)
    torch.save(dict(model=net.state_dict(), **fields), path)
    (path.parent / "run.json").write_text(json.dumps(dict(
        args=dict(accel=100.0, workspace="rail30"), workspace_bounds_mm=workspace_bounds("rail30"),
        physical_limits=dict(speed_m_s=12, acceleration_m_s2=100.0))))


def test_trainer_continues_a_run_with_every_table_option(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("ai", "shared", "fw"):
        (root / name).symlink_to(REPO / name)
    runs = root / "runs"
    save_player(runs / "rail30-100-20261001-v1" / "agent.pt", NeuralPlayer(8), width=8)
    torch.manual_seed(5)
    student = NeuralPlayer(32, history=16, shot_conditioned=True, history_deltas=True)
    save_player(runs / "_reference" / "agent.pt", student, width=32, history=16, shot_conditioned=True, step=1)
    thermal = root / "thermal.json"
    shutil.copy(REPO / "ai/recipes/motor-load-20261001.json", thermal)
    trainer = load_trainer(root)
    monkeypatch.setattr(trainer, "ROOT", root)
    common = [
        "--n-envs", "32", "--workers", "2", "--rollout", "8", "--epochs", "2", "--minibatch", "64",
        "--stage", "5", "--accel", "100", "--workspace", "rail30", "--thermal-model", str(thermal),
        "--history-deltas", "--opponent-checkpoint", str(runs / "_reference" / "agent.pt"), "--game-fraction", ".5",
        "--teacher", "expert", "--teacher-weight", "1", "--teacher-weight-final", ".5",
        "--teacher-anneal-steps", "256", "--teacher-mask", "defense", "--teacher-opponent-fraction", ".25",
        "--actor-steps-min", "2", "--adaptive-lr", "--report-sensing", "--continuous-rallies",
        "--fixed-practice-roles", "--practice-defense-fraction", ".45", "--game-episode-seconds", "900",
        "--std-floor", ".1", "--device", "cpu", "--torch-threads", "1", "--save-every", "100000", "--lr", "1e-4"]

    def train(*argv):
        monkeypatch.setattr(sys, "argv", ["train_neural_player.py", *common, *argv])
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        threads = torch.get_num_threads()
        try:
            trainer.main()
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)
            torch.set_num_threads(threads)
        return [json.loads(line) for line in (runs / argv[1] / "metrics.jsonl").read_text().splitlines()]

    # A run to continue: fresh critic, a warm-up, the teacher pulled in at full weight.
    seeded = train("--run-name", "_first", "--steps", "512", "--initialize-from", str(runs / "_reference" / "agent.pt"),
                   "--value-warmup-updates", "1")
    assert seeded[0]["value_warmup"] and seeded[0]["teacher_weight"] == pytest.approx(.5)
    assert set(seeded[-1]["motion"]) == {
        "accel_duty_share", "reversals_per_minute", "far_half_share", "far_cap_share",
        "accel_duty_cost_per_s", "reversal_cost_per_s", "far_cap_cost_per_s",
        "guard_changed_per_minute", "guard_unresolved_per_minute", "guard_cost_per_s", "accel_effort_cost_per_s",
        "idle_speed_cost_per_s", "dead_puck_share", "dead_puck_cost_per_s"}
    assert seeded[-1]["motion"]["accel_duty_cost_per_s"] == 0 and "lift_events_per_minute" not in seeded[-1]
    assert seeded[-1]["motion"]["guard_cost_per_s"] == seeded[-1]["motion"]["dead_puck_cost_per_s"] == 0
    assert set(seeded[-1]["motion_by_slot"]) == {"receiving", "defense", "games"}    # 16 a worker: no stationary slot
    assert "dead_puck_exploration_share" not in seeded[-1]
    saved = torch.load(runs / "_first" / "agent.pt", weights_only=False)

    rows = train(
        "--run-name", "_table", "--steps", "512", "--resume", str(runs / "_first" / "agent.pt"),
        "--sim-profile", "table-20261029", "--sim-profile-share", ".5",
        "--sim-option", 'mallet_lift={"off_probability": 0.0}',
        "--human-proxy-share", ".25", "--human-proxy-tiers", "table,table_plus,club,elite",
        "--defense-strike-fraction", ".5", "--defense-strike-speed", "4", "9", "--defense-strike-moving-fraction", ".3",
        "--left-bank-request-weight", "2", "--accel-duty-weight", "20", "--reversal-weight", "5",
        "--far-cap-weight", "40", "--stagger-games")
    assert [r["update"] for r in rows] == [1, 2]
    for r in rows:
        assert all(np.isfinite(v) for v in r["motion"].values())
        assert all(np.isfinite(v) for v in r["lift_events_per_minute"].values())
        assert set(r["game_scores"]) == {"pool", "reference", "teacher", "human_proxy"}
        assert np.isfinite(r["explained_variance"]) and np.isfinite(r["final_kl"])
    assert sum(r["motion"]["accel_duty_share"] for r in rows) > 0
    # Continuation: the step count, the critic, the optimiser moments and the
    # adapted rate carry over; no warm-up and no teacher re-pull.
    assert rows[0]["step"] == saved["step"] + 256 and not rows[0]["value_warmup"]
    assert rows[0]["teacher_weight"] == pytest.approx(.5) and rows[0]["actor_lr"] == pytest.approx(saved["actor_lr"])
    initial = torch.load(runs / "_table" / "agent_initial.pt", weights_only=False)
    for key in ("critic.weight", "value_trunk.0.weight", "actor.weight"):
        assert torch.equal(initial["model"][key], saved["model"][key])
    old, new = saved["optimizer"]["state"], initial["optimizer"]["state"]
    assert old and set(old) == set(new)
    assert all(torch.equal(old[k]["exp_avg_sq"], new[k]["exp_avg_sq"]) and old[k]["step"] == new[k]["step"] for k in old)
    meta = json.loads((runs / "_table" / "run.json").read_text())
    assert meta["critic_initialization"] == "checkpoint" and meta["optimizer_transfer"] == "restored"
    assert meta["sim_profile"]["workers"] == [True, False] and meta["sim_profile"]["options"]["mallet_lift"] == {
        "off_probability": 0.0}
    # 8 game slots per worker: 2 people each, off the reference slots.
    assert meta["human_proxy"]["game_slots"] == 4 and meta["human_proxy"]["tiers"] == [
        "table", "table_plus", "club", "elite"]
    assert meta["teacher"]["opponent_slots"] == 4

    # One process takes the whole profile, on top of the options it replaces.
    single = train(
        "--run-name", "_single", "--steps", "128", "--resume", str(runs / "_first" / "agent.pt"),
        "--workers", "1", "--n-envs", "16", "--sim-profile", "table-20261029",
        "--human-proxy-share", ".25", "--defense-strike-fraction", ".5", "--accel-duty-weight", "20")
    assert len(single) == 1 and all(np.isfinite(v) for v in single[0]["lift_events_per_minute"].values())

    # Round 2: every flag at once, with a lift model of three mechanisms.
    second = train(
        "--run-name", "_round2", "--n-envs", "64", "--steps", "1024", "--resume", str(runs / "_first" / "agent.pt"),
        "--sim-profile", "table-20261029", "--sim-profile-share", ".5",
        "--sim-option", 'mallet_lift={"b0": [0.0, 0.3], "flat_probability": 0.25, "closing_probability": 0.25, "far_body": false}',
        "--human-proxy-share", ".25", "--stationary-failure-penalty", "600",
        "--dead-puck-weight", "40", "--dead-puck-exploration", "2", "2", "--dead-puck-exploration-start", "0",
        "--dead-puck-exploration-hold", ".9", "--calm-accel-weight", "200", "--calm-accel-power", "1",
        "--calm-reachable-scale", "0", "--cross-bank-drill-fraction", "1", "--corner-far-start-fraction", ".5",
        "--corner-drill-fraction", ".2", "--guard-weight", "10", "--guard-unresolved-weight", "50",
        "--far-cap-weight", "40", "--far-cap-max-speed", "1", "--accel-effort-weight", "100", "--idle-speed-weight", "20")
    assert [r["update"] for r in second] == [1, 2]
    for r in second:
        assert all(np.isfinite(v) for v in r["motion"].values()) and np.isfinite(r["final_kl"])
        assert set(r["motion_by_slot"]) == {"stationary", "receiving", "defense", "games", "people"}
        assert all(np.isfinite(v) for slot in r["motion_by_slot"].values() for v in slot.values())
        # Resting pucks from the first tick: the wide noise is in the rollout and PPO scores it finitely.
        assert r["dead_puck_exploration_share"] > 0.01 and np.isfinite(r["kl"])
    assert all("anchors" not in r for r in second)

    # Round 3: the teacher on resting open-table pucks, a reference in corner drills, labelled on the
    # learner's own rollout states.
    torch.manual_seed(6)
    save_player(runs / "_anchor" / "agent.pt", NeuralPlayer(32, history=16, shot_conditioned=True, history_deltas=True),
                width=32, history=16, shot_conditioned=True, step=1)
    drills = ["--stationary-failure-penalty", "600", "--dead-puck-weight", "40", "--cross-bank-drill-fraction", ".5",
              "--corner-drill-fraction", ".5", "--skill-reference", str(runs / "_anchor" / "agent.pt"),
              "--n-envs", "64", "--rollout", "32", "--steps", "6144", "--game-fraction", ".25"]
    for space in ("pre-squash", "squashed"):
        third = train("--run-name", "_round3-" + space, "--resume", str(runs / "_first" / "agent.pt"), *drills,
                      "--teacher-resting-weight", "1", "--teacher-resting-start", ".2",
                      "--skill-reference-rail-weight", "1", "--anchor-space", space,
                      *(["--anchor-kl-exempt", "--teacher-resting-dims", "cap"] if space == "squashed" else []))
        key = space.replace("-", "_")
        for r in third:
            assert set(r["anchors"]) == {"teacher_resting", "reference_rail"}
            assert r["anchors"]["teacher_resting"]["dims"] == ([5] if space == "squashed" else "all")
            assert r["anchors"]["reference_rail"]["dims"] == "all"
            # Optionally the KL that limits the step is of the samples no anchor covers.
            assert ("anchor_kl_exempt" in r) == (space == "squashed")
            if space == "squashed":
                both = sum(a["share"] for a in r["anchors"].values())
                assert r["anchor_kl_exempt"]["share"] == pytest.approx(both) and np.isfinite(r["anchor_kl_exempt"]["final_kl_all"])
            for anchor in r["anchors"].values():
                assert anchor["space"] == space and 0 < anchor["share"] < 1 and anchor["samples"] > 0
                assert np.isfinite([anchor["loss"], anchor["loss_first"], anchor["loss_last"]]).all()
                assert np.isfinite([anchor[k + e] for k in ("squashed", "pre_squash") for e in ("_initial", "_final")]).all()
                assert len(anchor["squashed_initial_by_dimension"]) == 6
                # Where its samples are, per slot class, and the policy change on them.
                assert sum(anchor["samples_by_slot"].values()) == anchor["samples"]
                assert set(anchor["share_by_slot"]) <= set(anchor["samples_by_slot"]) == {
                    "stationary", "receiving", "defense", "games", "people"}
                assert all(0 <= share <= 1 for share in anchor["share_by_slot"].values())
                assert np.isfinite(list(anchor["initial_by_slot"].values())).all()
            # ... and the policy change everywhere else, which is what the anchors cost the rest of the play.
            assert np.isfinite(r["anchors_final_kl_outside"]) and r["anchors_final_kl_outside"] >= 0
            assert np.isfinite(list(r["anchors_final_kl_outside_by_slot"].values())).all()
            assert np.isfinite(r["final_kl"])
        # Each update lowers the error it regresses, on the samples it was taken on. (The reference here is
        # another untrained network: its error starts near zero and only has to stay finite.)
        assert np.mean([r["anchors"]["teacher_resting"][key + "_final"] < r["anchors"]["teacher_resting"][key + "_initial"]
                        for r in third]) > 0.6
    # Round 4: every flag at once on top of round 3's; its rates appear, per slot too, and only then.
    fourth = train("--run-name", "_round4", "--resume", str(runs / "_first" / "agent.pt"), *drills,
                   "--teacher-resting-weight", "1", "--teacher-resting-start", ".2", "--teacher-resting-end", "2.5",
                   "--teacher-resting-dims", "cap", "--skill-reference-rail-weight", "1", "--anchor-space", "pre-squash",
                   "--anchor-side-rail-open-y", ".55", "--anchor-front-margin", ".06",
                   "--far-cap-weight", "40", "--far-cap-threshold", "20", "--far-cap-max-speed", "1",
                   "--far-cap-brake-seconds", ".25", "--dead-puck-cap-weight", "100", "--dead-puck-cap-grace", ".5",
                   "--reach-edge-drill-fraction", ".5", "--back-rail-drill-fraction", ".2", "--own-corner-weight", "300")
    added = {"dead_puck_cap_cost_per_s", "own_corner_per_minute", "own_corner_cost_per_s"}
    for r in fourth:
        assert added <= set(r["motion"]) and all(np.isfinite(v) for v in r["motion"].values())
        for slot in r["motion_by_slot"].values():
            assert added | {"far_cap_cost_per_s", "dead_puck_cost_per_s"} <= set(slot) and np.isfinite(list(slot.values())).all()
        assert np.isfinite(r["final_kl"]) and np.isfinite(r["anchors"]["teacher_resting"]["loss"])
    # Pucks lie at the edge of reach from the first tick, under whatever cap the untrained network asks for.
    assert max(r["motion_by_slot"]["stationary"]["dead_puck_cap_cost_per_s"] for r in fourth) > 0
    assert all(not (added & set(r["motion"])) and "far_cap_cost_per_s" not in r["motion_by_slot"]["stationary"]
               for r in third)
    meta = json.loads((runs / "_round4" / "run.json").read_text())
    assert meta["anchors"]["teacher_resting"]["end_s"] == 2.5 and meta["anchors"]["teacher_resting"]["front_margin"] == .06
    for bad in (["--teacher-resting-end", "2"], ["--far-cap-brake-seconds", ".25"],
                ["--teacher-resting-weight", "1", "--teacher-resting-start", "1", "--teacher-resting-end", ".5"]):
        with pytest.raises(SystemExit):
            train("--run-name", "_bad4", "--steps", "128", *bad)
    meta = json.loads((runs / "_round3-pre-squash" / "run.json").read_text())
    assert meta["anchors"]["space"] == "pre-squash" and meta["anchors"]["training_only"]
    assert meta["anchors"]["reference_rail"]["reference"].endswith("_anchor/agent.pt")
    assert not meta["anchors"]["kl_exempt"]
    for bad in (["--teacher-resting-weight", "-1"], ["--skill-reference-rail-weight", "1"], ["--anchor-kl-exempt"]):
        with pytest.raises(SystemExit):
            train("--run-name", "_bad", "--steps", "128", *bad)
    with pytest.raises(SystemExit):
        train("--run-name", "_wide", "--steps", "128", "--dead-puck-exploration", "2", "5")
    assert json.loads((runs / "_single" / "run.json").read_text())["sim_profile"]["workers"] == [True]
    with pytest.raises(SystemExit):
        train("--run-name", "_tiers", "--steps", "128", "--human-proxy-share", ".25", "--human-proxy-tiers", "table,nobody")
    assert not (runs / "_tiers").exists()


def test_recipe_is_v4s_command_continued_with_the_table_options():
    spec = importlib.util.spec_from_file_location("recipe_v5", REPO / "ai/recipes/table-faithful-20261110/finetune.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    trainer = load_trainer(REPO)

    def parsed(*argv):
        cmd = recipe.argv(recipe.build_parser().parse_args(["--run-name", "_x", *argv]))
        return cmd, trainer.build_parser().parse_args(cmd[2:])

    round2 = {"dead_puck_weight", "dead_puck_grace", "dead_puck_exploration", "dead_puck_exploration_start",
              "dead_puck_exploration_hold", "calm_reachable_scale", "guard_weight", "guard_unresolved_weight",
              "far_cap_max_speed", "accel_effort_weight", "accel_effort_threshold", "idle_speed_weight",
              "cross_bank_drill_fraction", "corner_far_start_fraction"}
    round3 = {"teacher_resting_weight", "teacher_resting_start", "teacher_resting_dims", "skill_reference_rail_weight",
              "anchor_space", "anchor_kl_exempt"}
    round4 = {"far_cap_brake_seconds", "dead_puck_cap_weight", "dead_puck_cap_grace", "reach_edge_drill_fraction",
              "back_rail_drill_fraction", "own_corner_weight", "anchor_side_rail_open_y", "anchor_front_margin",
              "teacher_resting_end"}
    cmd, args = parsed()
    # Continuation, and the pitfalls v4's record paid for.
    assert args.resume == Path(recipe.V4) and args.initialize_from is None and args.value_warmup_updates == 0
    assert args.possession_delay_weight == 30 and args.teacher_mask == "defense" and args.teacher_free_guard_accel
    assert args.calm_accel_weight == 200 and args.calm_accel_power == 1
    assert trainer.sim_profile_options(args) == table_fidelity() and args.sim_profile_share == 0.75
    assert args.human_proxy_share == 0.3 and args.human_proxy_tiers == "table,table_plus,club,elite"
    assert args.stagger_games and args.left_bank_request_weight == 2 and args.defense_strike_fraction == 0.4
    assert (args.accel_duty_weight, args.reversal_weight, args.far_cap_weight) == (20, 5, 40)
    new = {"sim_profile", "sim_option", "sim_profile_share", "human_proxy_share", "human_proxy_tiers",
           "defense_strike_fraction", "defense_strike_speed", "defense_strike_moving_fraction",
           "left_bank_request_weight", "accel_duty_weight", "accel_duty_threshold", "reversal_weight",
           "reversal_speed", "far_cap_weight", "far_cap_threshold", "stagger_games"}
    # With every option switched off it is the trainer's command from before they existed.
    plain_cmd, plain = parsed("--sim-profile", "none", "--human-proxy-share", "0", "--defense-strike", "0",
                              "--left-bank", "1", "--accel-duty", "0", "--reversal", "0", "--far-cap", "0", "--shared-clock")
    defaults = trainer.build_parser().parse_args(["--run-name", "_x"])
    assert all(getattr(plain, key) == getattr(defaults, key) for key in new)
    # Arm A carries none of round 2.
    assert all(getattr(args, key) == getattr(defaults, key) for key in round2 | round3)
    assert not any(flag.startswith(("--dead-puck", "--calm-reachable", "--guard", "--accel-effort", "--idle-speed",
                                    "--cross-bank", "--corner-far", "--teacher-resting", "--skill-reference-rail",
                                    "--anchor-space")) for flag in cmd)
    assert args.skill_reference is None
    assert not any(flag.startswith(("--sim-", "--human-proxy", "--defense-strike", "--left-bank", "--accel-duty",
                                    "--reversal", "--far-cap", "--stagger")) for flag in plain_cmd)
    record = REPO / "runs/teacher-calm-20261018-v4/run.json"
    if record.exists():
        # Apart from the continuation and the new options, v4's own arguments.
        old = json.loads(record.read_text())["args"]
        mine = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        changed = {k for k in old if old[k] != mine[k]}
        assert changed == {"run_name", "evaluation_dir", "resume", "initialize_from", "opponent_reference", "minutes",
                           "seed", "value_warmup_updates", "width"}
        assert set(mine) - set(old) == new | round2 | round3 | round4


def test_round2_arm_continues_v5a_with_its_evaluations_changes():
    spec = importlib.util.spec_from_file_location("recipe_v5", REPO / "ai/recipes/table-faithful-20261110/finetune.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    trainer = load_trainer(REPO)
    parse = lambda *argv: trainer.build_parser().parse_args(recipe.argv(recipe.parse(["--run-name", "_x", *argv]))[2:])
    first, second = parse(), parse("--arm", "a2")
    assert second.resume == Path(recipe.V5A) and second.initialize_from is None and second.value_warmup_updates == 0
    # The stall: a cost from the first second, noise that can reach a strike, its own drill, no one-sided requests.
    assert second.dead_puck_weight == 40 and second.dead_puck_exploration == [2, 2] and second.dead_puck_exploration_hold == 0.9
    assert second.dead_puck_grace == second.dead_puck_exploration_start == 1     # the noise begins where the cost does
    assert second.calm_reachable_scale == 0 and second.cross_bank_drill_fraction == 0.25 and second.left_bank_request_weight == 1
    assert second.stationary_failure_penalty == 600 and second.possession_delay_weight == 30
    # Roughness: the guard's brake charged, the far cap only at rest, effort quadratic, shot pay flat from 7 m/s.
    assert (second.guard_weight, second.guard_unresolved_weight, second.far_cap_max_speed) == (10, 50, 1)
    assert second.accel_effort_weight == 100 and second.idle_speed_weight == 20 and second.shot_speed_scale == 7
    assert (second.accel_duty_weight, second.reversal_weight, second.far_cap_weight) == (20, 5, 40)
    assert (second.corner_drill_fraction, second.edge_drill_fraction, second.corner_far_start_fraction) == (.2, .15, .5)
    lift = trainer.sim_profile_options(second)["mallet_lift"]
    assert lift == dict(b0=[0.0, 0.30], a0=[10, 90], flat_probability=0.25, closing_probability=0.25, far_body=False)
    NeuralTrainingEnv(2, **trainer.sim_profile_options(second))
    # Everything else is arm A's command.
    changed = {k for k in vars(first) if getattr(first, k) != getattr(second, k)}
    assert changed == {
        "resume", "seed", "left_bank_request_weight", "shot_speed_scale", "sim_option", "corner_drill_fraction",
        "edge_drill_fraction", "dead_puck_weight", "dead_puck_exploration", "dead_puck_exploration_start",
        "dead_puck_exploration_hold",
        "calm_reachable_scale", "cross_bank_drill_fraction", "corner_far_start_fraction", "guard_weight",
        "guard_unresolved_weight", "far_cap_max_speed", "accel_effort_weight", "idle_speed_weight"}
    # A typed flag still wins over the arm's default.
    assert parse("--arm", "a2", "--dead-puck", "0", "--left-bank", "2").dead_puck_weight == 0


def test_round3_arm_is_a2_with_the_two_anchors():
    spec = importlib.util.spec_from_file_location("recipe_v5", REPO / "ai/recipes/table-faithful-20261110/finetune.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    trainer = load_trainer(REPO)
    command = lambda *argv: recipe.argv(recipe.parse(["--run-name", "_x", *argv]))
    parse = lambda *argv: trainer.build_parser().parse_args(command(*argv)[2:])
    second, third = parse("--arm", "a2"), parse("--arm", "a3")
    # Arm a2's own command has none of it.
    assert not any(flag.startswith(("--teacher-resting", "--skill-reference-rail", "--anchor-space"))
                   for flag in command("--arm", "a2"))
    assert second.skill_reference is None and second.teacher_resting_weight == second.skill_reference_rail_weight == 0
    # Continues a2 by default; the starting point is a flag.
    assert third.resume == Path(recipe.A2) and third.initialize_from is None and third.value_warmup_updates == 0
    assert parse("--arm", "a3", "--resume", recipe.V4).resume == Path(recipe.V4)
    # The teacher where a puck has lain on the open table as long as the cost's grace; v4 in the rail strip.
    assert third.teacher_resting_weight == recipe.TEACHER_RESTING > 0 and third.teacher_resting_start == third.dead_puck_grace
    assert third.skill_reference == Path(recipe.V4) and third.skill_reference_rail_weight == recipe.REFERENCE_RAIL > 0
    assert third.anchor_space == "pre-squash"
    # Only the teacher's accel cap is copied: the aim and the shot stay the learner's own.
    assert third.teacher_resting_dims == recipe.TEACHER_RESTING_DIMS == "cap"
    assert parse("--arm", "a3", "--teacher-resting-dims", "all").teacher_resting_dims == "all"
    # v4 is not the anchor for resting pucks in general (it has the hover), and the defense teacher is as it was.
    assert third.skill_reference_weight == 0 and third.teacher_mask == "defense" and third.teacher_weight_final == 0.2
    changed = {k for k in vars(second) if getattr(second, k) != getattr(third, k)}
    assert changed == {"resume", "seed", "teacher_resting_weight", "teacher_resting_dims", "skill_reference",
                       "skill_reference_rail_weight", "anchor_space"}
    # A typed flag still wins.
    off = parse("--arm", "a3", "--teacher-resting", "0", "--reference-rail", "0")
    assert off.skill_reference is None and off.anchor_space == "squashed" and off.teacher_resting_weight == 0
    assert off.teacher_resting_dims == "all"


def test_round4_arm_is_a3_continued_with_what_its_evaluation_held_against_it():
    spec = importlib.util.spec_from_file_location("recipe_v5", REPO / "ai/recipes/table-faithful-20261110/finetune.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    trainer = load_trainer(REPO)
    command = lambda *argv: recipe.argv(recipe.parse(["--run-name", "_x", *argv]))
    parse = lambda *argv: trainer.build_parser().parse_args(command(*argv)[2:])
    third, fourth = parse("--arm", "a3"), parse("--arm", "a4")
    # Arm a3's own command has none of it.
    new_flags = ("--far-cap-threshold", "--far-cap-brake-seconds", "--dead-puck-cap", "--teacher-resting-end",
                 "--anchor-side-rail-open-y", "--anchor-front-margin", "--reach-edge-drill", "--back-rail-drill",
                 "--own-corner")
    assert not any(flag.startswith(new_flags) for flag in command("--arm", "a3"))
    # Continues a3 itself, critic and all; the starting point is still a flag.
    assert fourth.resume == Path(recipe.A3) and fourth.initialize_from is None and fourth.value_warmup_updates == 0
    assert parse("--arm", "a4", "--resume", recipe.V4).resume == Path(recipe.V4)
    # After its own shot: fast ticks are free for a quarter second after the contact, the cap charged from 20.
    assert (fourth.far_cap_weight, fourth.far_cap_max_speed, fourth.far_cap_brake_seconds) == (40, 1, 0.25)
    assert fourth.far_cap_threshold == 20 and third.far_cap_threshold == 40
    # Beside a lying puck: the run into the strike stays free (calm scale 0, anchor from 1 s), the cap is charged
    # from 2.5 s, where the teacher's cap label ends too.
    assert fourth.calm_reachable_scale == 0 and fourth.teacher_resting_start == 1
    assert fourth.dead_puck_cap_weight == 100 and fourth.dead_puck_cap_grace == fourth.teacher_resting_end == 2.5
    # Coverage: the front edge and forward side rails drilled; those rails the teacher's, the front edge nobody's.
    assert fourth.reach_edge_drill_fraction == 0.2 and fourth.anchor_side_rail_open_y == 0.55
    assert fourth.anchor_front_margin == 0.06
    # The back corner: an entry cost of half a goal and a small drill on the rail that leads there.
    assert fourth.own_corner_weight == 300 == fourth.game_goal_weight / 2 and fourth.back_rail_drill_fraction == 0.05
    changed = {k for k in vars(third) if getattr(third, k) != getattr(fourth, k)}
    assert changed == {"resume", "seed", "far_cap_threshold", "far_cap_brake_seconds", "dead_puck_cap_weight",
                       "teacher_resting_end", "reach_edge_drill_fraction", "anchor_side_rail_open_y",
                       "anchor_front_margin", "back_rail_drill_fraction", "own_corner_weight"}
    # A typed flag still wins; the environment takes the arm's options.
    off = parse("--arm", "a4", "--dead-puck-cap", "0", "--own-corner", "0")
    assert off.dead_puck_cap_weight == 0 and off.own_corner_weight == 0
    NeuralTrainingEnv(2, far_cap_weight=fourth.far_cap_weight, far_cap_max_speed=fourth.far_cap_max_speed,
                      far_cap_brake_seconds=fourth.far_cap_brake_seconds, dead_puck_cap_weight=fourth.dead_puck_cap_weight,
                      reach_edge_drill_fraction=fourth.reach_edge_drill_fraction,
                      anchor_side_rail_open_y=fourth.anchor_side_rail_open_y, anchor_front_margin=fourth.anchor_front_margin)
