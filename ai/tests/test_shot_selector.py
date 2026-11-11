"""Who picks the shot route (2026-11-10): the request governor, the route it
reads back, free requests in training, and the evaluator / table options.

Pins that the governor leaves a balanced network alone and cannot be
starved of a route by one that honours explicit requests, that the realised
route is read from the policy's own inputs with the reward's definitions,
that a free possession pays any aimed route what a matched one earns and
nothing changes with the option off, and that the table runner and the
evaluator hand out the same requests from the same class.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from airhockey.dynamics import sim_to_table_mm
from airhockey.neural_deploy import NeuralPolicy
from airhockey.neural_player import NeuralPlayer
from airhockey.neural_training import NeuralTrainingEnv, route_report
from airhockey.physics import TableConfig
from airhockey.rewards import predict_shot
from airhockey.shot_selector import DEFAULTS, LEFT, NONE, RIGHT, STRAIGHT, RouteClassifier, ShotSelector

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "ai" / "bin"
CFG = TableConfig()


# ------------------------------------------------------------------ governor
def govern(network, possessions=3000, seed=0, **options):
    """One shot per possession by `network(request, rng)`; returns requests, routes, selector."""
    selector = ShotSelector("auto", seed=seed, **options)
    rng = np.random.default_rng(seed + 1)
    requests, routes = [], []
    for _ in range(possessions):
        request = selector.request()
        route = network(request, rng)
        selector.record(route)
        requests.append(request)
        routes.append(route)
    return np.array(requests), np.array(routes), selector


def rolling_shares(routes, width=60, start=60):
    return np.array([np.bincount(routes[i:i + width], minlength=4)[1:] / width
                     for i in range(start, len(routes) - width, 5)])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_network_that_always_shoots_straight_is_made_to_bank(seed):
    requests, routes, selector = govern(lambda request, rng: request or STRAIGHT, seed=seed)
    shares = rolling_shares(routes)
    # Every 60-shot stretch after the first: both banks near the 15% floor
    # or above, straight near the 60% ceiling or below.
    assert shares[:, :2].min() >= 0.12 and shares[:, 2].max() <= 0.68
    # It still gets its own way on about half the possessions.
    assert 0.45 <= (requests == NONE).mean() <= 0.65
    stats = selector.stats()
    assert stats["honoured_share"] == 1 and stats["realised_when_free"] == dict(left=0, right=0, straight=int((requests == 0).sum()))
    assert stats["reasons"]["floor"] > 0 and stats["reasons"]["ceiling"] > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_network_that_never_banks_left_is_asked_for_it(seed):
    requests, routes, _ = govern(lambda request, rng: request or int(rng.integers(2, 4)), seed=seed)
    shares = rolling_shares(routes)
    assert shares[:, 0].min() >= 0.11 and shares.max() <= 0.68
    assert (requests == NONE).mean() >= 0.65
    assert (requests == LEFT).sum() > 2 * (requests == RIGHT).sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_balanced_network_is_left_free(seed):
    requests, routes, selector = govern(lambda request, rng: request or int(rng.integers(1, 4)), seed=seed)
    assert (requests == NONE).mean() >= 0.75
    # What is not free is mostly the standing exploration, a uniform explicit draw.
    reasons = selector.stats()["reasons"]
    assert reasons["explore"] > 2 * (reasons["floor"] + reasons["ceiling"])
    assert abs(reasons["explore"] / len(requests) - DEFAULTS["explore"]) < 0.03
    assert rolling_shares(routes).min() >= 0.12


def test_a_network_deaf_to_requests_is_reported_not_hidden():
    requests, routes, selector = govern(lambda request, rng: STRAIGHT, possessions=600)
    stats = selector.stats()
    # Nothing the request can do: it asks for a bank every possession once the window says so.
    assert (requests[60:] != NONE).mean() > 0.98 and (requests[60:] == STRAIGHT).mean() < 0.1
    assert stats["honoured_share"] < 0.1 and stats["window"]["straight"] == 1


def test_the_window_decides_and_the_most_starved_route_goes_first():
    selector = ShotSelector("auto", explore=0.0, seed=3)
    assert [selector.request() for _ in range(5)] == [NONE] * 5       # nothing known yet
    for _ in range(8):
        selector.record(STRAIGHT)
    assert selector.request() == NONE                                  # eight shots: still under min_shots
    selector.record(RIGHT)
    assert selector.request() == LEFT and selector.reason == "floor"   # 0/9 left, 1/9 right
    for _ in range(3):
        selector.record(LEFT)
    assert selector.request() == RIGHT and selector.reason == "floor"  # 3/12 left, 1/12 right
    for route in (RIGHT, RIGHT):
        selector.record(route)
    # 3 / 3 / 8 of 14: nobody starved, straight under the ceiling.
    assert selector.request() == NONE and selector.reason == "free"
    for _ in range(6):
        selector.record(STRAIGHT)
    # 3 / 3 / 14 of 20 = 70% straight, both banks at 15%: one of them, not straight.
    assert selector.request() in (LEFT, RIGHT) and selector.reason == "ceiling"
    # The window slides: 30 lefts push everything else out.
    for _ in range(30):
        selector.record(LEFT)
    assert selector.shares().tolist() == [1, 0, 0] and selector.request() in (RIGHT, STRAIGHT)


def test_undetermined_shots_are_not_counted_unless_asked():
    plain = ShotSelector("auto", explore=0.0, seed=0)
    plain.current = LEFT
    for _ in range(12):
        plain.record(NONE)
    assert len(plain.recent) == 0 and plain.stats()["undetermined"] == 12
    fallback = ShotSelector("auto", explore=0.0, seed=0, count_requested_when_unclassified=True)
    for _ in range(12):
        fallback.record(NONE, requested=LEFT)
    fallback.record(None, requested=NONE)             # a free shot nobody could read stays uncounted
    assert fallback.shares().tolist() == [1, 0, 0] and len(fallback.recent) == 12


def test_other_modes_and_bad_settings():
    uniform = ShotSelector("random", seed=1)
    assert {uniform.request() for _ in range(60)} == {LEFT, RIGHT, STRAIGHT}
    assert {ShotSelector("free").request() for _ in range(5)} == {NONE}
    assert ShotSelector("left").request() == LEFT and ShotSelector("straight").request() == STRAIGHT
    for bad in (dict(mode="mix"), dict(floor=0.34), dict(ceiling=0.33), dict(window=2), dict(min_shots=31),
                dict(explore=1.5), dict(min_shots=0)):
        with pytest.raises(ValueError):
            ShotSelector(**{"mode": "auto", **bad})


# ---------------------------------------------------------------- classifier
def fly(classifier, launch, velocity, *, mallet=None, start_t=1.0, lose=(), turn_back_at=None, ticks=80,
        lead_in=3):
    """A puck at rest beside the mallet, struck to `velocity` at `launch`, then
    free flight off lossy rails as the 50 Hz policy sees it. Returns the events."""
    x, y = launch
    vx, vy = 0.0, 0.0
    mallet = mallet or (x, y - 0.10)
    low, high = CFG.puck_radius, CFG.width - CFG.puck_radius
    events, t = [], start_t
    seen = (x, y, 0.0, 0.0)
    for k in range(ticks):
        if k == lead_in:
            vx, vy = velocity
        for _ in range(8):                      # 2.5 ms substeps
            x, y = x + vx * 0.0025, y + vy * 0.0025
            if x < low or x > high:
                x = min(max(x, low), high)
                vx, vy = -vx * CFG.wall_restitution, vy * CFG.wall_tangential
            if turn_back_at is not None and y >= turn_back_at:
                vy = -abs(vy)
        if k in lose:
            seen = (seen[0], seen[1], 0.0, 0.0)           # the encoder's frozen report
        else:
            seen = (x, y, vx, vy)
        event = classifier.update(t, seen[:2], seen[2:], mallet)
        if event is not None:
            events.append(event)
        t += 0.02
        if y > 2.0 or y < 0:
            break
    return events


def test_routes_are_the_rewards_definitions():
    for velocity, expected in (((0.0, 5.0), STRAIGHT), ((0.8, 6.0), STRAIGHT), ((-5.0, 5.0), LEFT),
                               ((5.0, 5.0), RIGHT), ((-2.5, 5.0), LEFT), ((6.0, 3.0), RIGHT)):
        events = fly(RouteClassifier(), (0.5, 0.45), velocity)
        assert len(events) == 1 and events[0]["route"] == expected, (velocity, events)
        _, _, bounces, first = predict_shot(0.5, 0.45, *velocity)
        assert expected == (STRAIGHT if bounces == 0 else LEFT if first < 0 else RIGHT)
        assert events[0]["launch_t"] == pytest.approx(1.0 + 3 * 0.02)
    # A bank whose rail lies past the centre line is read from the line, not waited for.
    late = fly(RouteClassifier(), (0.5, 0.80), (2.2, 5.0))
    assert late[0]["route"] == RIGHT and late[0]["how"] == "traced" and late[0]["t"] < 1.0 + 0.02 * 16
    early = fly(RouteClassifier(), (0.2, 0.30), (-5.0, 5.0))
    assert early[0]["route"] == LEFT and early[0]["how"] == "bounce"


def test_what_cannot_be_read_is_not_counted():
    # The tracker loses the puck in flight for longer than the coast.
    lost = fly(RouteClassifier(), (0.5, 0.30), (0.0, 2.0), lose=range(6, 40))
    assert [e["route"] for e in lost] == [NONE] and lost[0]["how"] == "lost"
    # A short dropout is ridden out.
    blink = fly(RouteClassifier(), (0.5, 0.30), (0.0, 3.0), lose=range(6, 10))
    assert [e["route"] for e in blink] == [STRAIGHT]
    # It never reaches the line (turned back on the robot's own half).
    stopped = fly(RouteClassifier(), (0.5, 0.30), (0.0, 3.0), turn_back_at=0.8)
    assert [e["route"] for e in stopped] == [NONE] and stopped[0]["how"] == "stopped"
    # Too slow to be a shot, and a puck nobody was near (a rebound off the back rail): no event at all.
    assert fly(RouteClassifier(), (0.5, 0.45), (0.0, 0.8)) == []
    assert fly(RouteClassifier(), (0.5, 0.45), (0.0, 5.0), mallet=(0.2, 0.9)) == []
    assert fly(RouteClassifier(), (0.5, 0.10), (1.0, 4.0), mallet=(0.5, 0.25)) == []     # mallet in FRONT of it
    # What the person does past the line is theirs: a straight shot sent back off their mallet is still straight.
    returned = fly(RouteClassifier(), (0.5, 0.45), (0.0, 6.0), turn_back_at=1.6)
    assert [e["route"] for e in returned] == [STRAIGHT]


def test_a_gap_in_the_ticks_carries_nothing_across():
    classifier = RouteClassifier()
    assert fly(classifier, (0.5, 0.45), (0.0, 3.0), ticks=6) == []       # in flight
    dropped = classifier.update(9.0, (0.5, 1.5), (0.0, 3.0), (0.5, 0.3))
    assert dropped["route"] == NONE and dropped["how"] == "resync" and classifier.flight is None
    assert [e["route"] for e in fly(classifier, (0.5, 0.45), (-5.0, 5.0), start_t=9.02)] == [LEFT]


# -------------------------------------------------------------------- trainer
GAMES = dict(stage=5, seed=4, accel=100.0, game_fraction=1 / 3, shot_conditioned=True,
             continuous_rallies=True, game_episode_seconds=900.0, fixed_practice_roles=True,
             practice_defense_fraction=0.45, shot_request_weight=100.0, wrong_shot_penalty=20.0,
             fixed_opponent_style_fraction=0.5)


def play(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        env.set_opponent_action(rng.uniform(-1, 1, (env.n_envs, 6)))
        obs, reward, term, trunc, info = env.step(rng.uniform(-1, 1, (env.n_envs, 6)).astype(np.float32))
        out.append((obs, reward, info, env.base._shot_type.copy(), env.base._shot_type_opp.copy()))
        if (term | trunc).any():
            env.reset(mask=term | trunc)
            env.opponent_obs()
    return out


def test_free_share_off_is_the_same_environment():
    a, b = NeuralTrainingEnv(24, **GAMES), NeuralTrainingEnv(24, **GAMES, free_request_share=0.0)
    assert a.base._shot_type_p.tolist() == [0, 1 / 3, 1 / 3, 1 / 3]
    assert np.array_equal(a.reset(seed=4), b.reset(seed=4))
    for (obs_a, reward_a, info_a, own_a, far_a), (obs_b, reward_b, info_b, own_b, far_b) in zip(play(a, 80), play(b, 80)):
        assert np.array_equal(obs_a, obs_b) and np.array_equal(reward_a, reward_b)
        assert np.array_equal(own_a, own_b) and np.array_equal(far_a, far_b)
        assert np.array_equal(info_a["route_events"], info_b["route_events"])


def test_free_share_is_the_learners_and_the_rest_stay_explicit():
    env = NeuralTrainingEnv(48, **{**GAMES, "game_fraction": 1.0}, free_request_share=0.5, left_bank_request_weight=2.0)
    assert env.base._shot_type_p.tolist() == pytest.approx([0.5, 0.25, 0.125, 0.125])
    env.reset(seed=6)
    own, far, free_inputs = [], [], 0
    previous = env.base._shot_type.copy()
    for obs, _, _, own_request, far_request in play(env, 400, seed=1):
        own.extend(own_request[own_request != previous])
        previous = own_request
        far.extend(far_request)
        # The actor's input is the request: all-zero exactly where it is free.
        assert np.array_equal(obs[:, -3:].sum(1) == 0, own_request == 0)
        free_inputs += int((obs[:, -3:].sum(1) == 0).sum())
    own, far = np.array(own), np.array(far)
    assert free_inputs > 0 and 0.25 < (own == 0).mean() < 0.75 and {1, 2, 3} <= set(own.tolist())
    # The far seat (a copy, or an older snapshot that never saw "none") is never left free once it has a request.
    assert set(np.unique(far[far != 0]).tolist()) <= {1, 2, 3}
    assert not (env.base._shot_type_p_opp[:, 0] > 0).any()
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, shot_conditioned=True, free_request_share=1.5)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, shot_conditioned=True, shot_request="left", free_request_share=0.5)
    with pytest.raises(ValueError):
        NeuralTrainingEnv(2, free_request_share=0.5)


def strike(env, requests, outgoing):
    """The shot reward of one prepared strike per env from (0.5, 0.5) under `requests`."""
    n = env.n_envs
    env.reset(seed=11)
    e = env.engine
    e.puck_x[:], e.puck_y[:] = 0.5, 0.5
    env.base._shot_type[:] = requests
    env.reward_events[:] = 0
    env.route_events[:] = 0
    env.captured[0, :] = True
    env.contact_clock[:] = -100
    env._contact(dict(body="agent", indices=np.arange(n), incoming=np.zeros((n, 2)),
                      outgoing_before_speed_cap=np.asarray(outgoing, float), normal=np.tile([0.0, 1.0], (n, 1))))
    return env.reward_events[0].copy()


def test_a_free_possession_pays_any_aimed_route_what_a_matched_one_earns():
    options = dict(stage=5, realistic=False, randomize=False, shot_conditioned=True, shot_request_weight=100.0,
                   wrong_shot_penalty=20.0, shot_power_weight=60.0)
    cfg = TableConfig()
    # Aimed shots from (0.5, 0.5): straight up the middle, and a single bank off either rail into the mouth.
    k = cfg.wall_tangential / cfg.wall_restitution
    reach = (cfg.width / 2 - cfg.puck_radius)
    bank = np.array([reach * (1 + k), cfg.height - 0.5])
    bank = 6.0 * bank / np.linalg.norm(bank)
    shots = np.array([[0.0, 6.0], [-bank[0], bank[1]], [bank[0], bank[1]], [3.0, 1.2]])
    routes = [STRAIGHT, LEFT, RIGHT]
    trained = NeuralTrainingEnv(4, **options, free_request_share=0.5)
    matched = strike(trained, [STRAIGHT, LEFT, RIGHT, STRAIGHT], shots)
    assert trained.last_shot_route[0, :3].tolist() == routes and trained.aimed_count[0, :3].tolist() == [1, 1, 1]
    free = strike(trained, [NONE] * 4, shots)
    # Free: every aimed route earns exactly the matched reward; the off-target one earns no bonus either way.
    np.testing.assert_allclose(free[:3], matched[:3])
    # The log of that step: all four shots free, three of them aimed, one per route.
    events = trained.route_events.sum(0)
    assert events[0, :, 0].sum() == 4 and events[0, 1:, 1].tolist() == [1, 1, 1] and events[1].sum() == 0
    wrong = strike(trained, [LEFT, RIGHT, STRAIGHT, STRAIGHT], shots)
    assert (wrong[:3] < matched[:3] - 100).all()          # no bonus, a quarter of the shaping, the penalty
    assert trained.route_events.sum(0)[1, :, 0].sum() == 4
    # With the option off a missing request earns no bonus, as before (the request is absent until the
    # first entry), and an explicit one is paid the same in both environments.
    historical = NeuralTrainingEnv(4, **options)
    np.testing.assert_allclose(strike(historical, [STRAIGHT, LEFT, RIGHT, STRAIGHT], shots), matched)
    unpaid = strike(historical, [NONE] * 4, shots)
    assert (unpaid[:3] < free[:3] - 50).all()


def test_goals_go_to_the_shot_they_followed_and_the_report_is_finite():
    env = NeuralTrainingEnv(4, stage=5, realistic=False, randomize=False, shot_conditioned=True,
                            free_request_share=0.5, game_fraction=1.0, continuous_rallies=True)
    strike(env, [NONE, LEFT, NONE, NONE], np.tile([0.0, 6.0], (4, 1)))
    assert env._scoring_shot.tolist() == [[0, 3], [1, 3], [0, 3], [0, 3]]
    # The far side touches env 2's puck: that shot can no longer score.
    env._contact(dict(body="opponent", indices=np.array([2]), incoming=np.array([[0.0, 6.0]]),
                      outgoing_before_speed_cap=np.array([[0.0, -3.0]]), normal=np.array([[0.0, -1.0]])))
    assert env._scoring_shot[2].tolist() == [-1, -1]
    e = env.engine
    e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = 0.5, env.cfg.height - 0.06, 0.0, 8.0
    e.paddle_opp_x[:] = 0.1
    total = np.zeros((2, 4, 3))
    for _ in range(3):
        env.set_opponent_action(np.zeros((4, 6)))
        _, _, _, _, info = env.step(np.zeros((4, 6), np.float32))
        total += info["route_events"].sum(0)
    assert e.score_agent.tolist() == [1, 1, 1, 1]
    assert total[0, 3, 2] == 2 and total[1, 3, 2] == 1          # envs 0 and 3 free, env 1 requested, env 2 nobody's
    report = route_report(total + np.array([[[0, 0, 0], [3, 2, 0], [1, 0, 0], [4, 4, 0]], [[0, 0, 0]] * 4]))
    free = report["free"]
    assert free["shots"] == 8 and free["share"] == dict(left=0.375, right=0.125, straight=0.5)
    assert free["aimed_rate"]["left"] == pytest.approx(2 / 3) and free["goal_rate"]["straight"] == 0.5
    assert free["entropy_bits"] == pytest.approx(1.4056, abs=1e-3)
    empty = route_report(np.zeros((2, 4, 3)))["requested"]
    assert empty["entropy_bits"] is None and empty["goal_rate"] == dict(left=None, right=None, straight=None)
    json.dumps(report)


def test_trainer_flag_reaches_the_environment_and_the_recipe():
    spec = importlib.util.spec_from_file_location("train_neural_player_free", BIN / "train_neural_player.py")
    trainer = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BIN))
    spec.loader.exec_module(trainer)
    parser = trainer.build_parser()
    assert parser.parse_args(["--run-name", "x"]).free_request_share == 0
    args = parser.parse_args(["--run-name", "x", "--shot-conditioned", "--free-request-share", "0.5"])
    options = trainer.environment_options(args, shot_conditioned=True)
    assert options["free_request_share"] == 0.5
    spec = importlib.util.spec_from_file_location("recipe_free", REPO / "ai/recipes/table-faithful-20261110/finetune.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    a4 = recipe.argv(recipe.parse(["--run-name", "_x", "--arm", "a4"]))
    assert a4[a4.index("--free-request-share") + 1] == "0.5"
    assert "--free-request-share" not in recipe.argv(recipe.parse(["--run-name", "_x", "--arm", "a3"]))


# ------------------------------------------------------------------ evaluator
class _Striker:
    """Drives at the puck whenever it is on the own half; ignores the request."""
    history, shot_conditioned, environment_options = 1, True, {}

    def act(self, observation, stochastic=False):
        action = np.zeros((len(observation), 6), np.float32)
        own_half = observation[:, 1] < 1.0
        # Arrival point: through the puck toward the far goal, short arrival time, full cap.
        action[:, 0] = np.clip(2 * observation[:, 0] - 1, -1, 1)
        action[:, 1] = np.where(own_half, 1.0, -0.6)
        action[:, 3] = np.where(own_half, 1.0, 0.0)
        action[:, 4], action[:, 5] = -1.0, np.where(own_half, 1.0, -1.0)
        return action


def test_evaluator_request_modes_and_route_report():
    sys.path.insert(0, str(BIN))
    import eval_human_proxy as hp
    assert hp.REQUEST_MODES == ("random", "free", "auto")
    for function in (hp.evaluate, hp.play_tier):
        assert __import__("inspect").signature(function).parameters["shot_request"].default == "random"
    runs = {mode: hp.play_tier(_Striker(), "table", games=4, seconds=30.0, seed=5, options={}, shot_request=mode)
            for mode in hp.REQUEST_MODES}
    for mode, result in runs.items():
        routes, request = result["candidate_routes"], result["shot_request"]
        assert request["mode"] == mode
        assert routes["all"]["shots"] == result["candidate_shots"] == routes["free"]["shots"] + routes["requested"]["shots"]
        assert set(routes["all"]["routes"]) == {"left_bank", "right_bank", "straight"}
        for row in routes["all"]["routes"].values():
            assert {"shots", "share", "on_target_rate", "goals", "goal_rate"} <= set(row)
        assert request["classifier"]["resolved"] + request["classifier"]["dropped"] == int(
            np.sum(request["classifier"]["confusion_env_route_by_classified"]))
        json.dumps(result["candidate_routes"]), json.dumps(request)
    # random: the env's own explicit draw, the governor only watching.
    assert runs["random"]["shot_request"]["possessions"] == 0
    assert runs["random"]["candidate_routes"]["requested"]["shots"] > 0
    # free: never a request once the puck has come over.
    free = runs["free"]["shot_request"]
    assert free["possessions"] > 0 and free["free_share"] == 1 and runs["free"]["candidate_routes"]["requested"]["shots"] == 0
    # auto: the governor's requests, with its settings in the report.
    auto = runs["auto"]["shot_request"]
    assert auto["selector"] == DEFAULTS and auto["possessions"] > 0
    assert sum(auto["requests"].values()) == auto["possessions"] == sum(auto["request_reasons"].values())
    with pytest.raises(ValueError):
        hp.play_tier(_Striker(), "table", games=2, seconds=1.0, seed=5, options={}, shot_request="mix")
    # The route read back from the observation is the env's own on most shots. (This striker wedges pucks
    # against a rail, where the rail touched "first" is a matter of one tick; a trained player's shots
    # agree far more often -- the recipe README has a3's and v4's numbers.)
    seen = np.sum([np.array(r["shot_request"]["classifier"]["confusion_env_route_by_classified"]) for r in runs.values()], axis=0)
    decided = seen[1:, 1:]
    assert decided.sum() >= 15 and np.trace(decided) / decided.sum() >= 0.7


# --------------------------------------------------------------- table runner
@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    net = NeuralPlayer(width=16, shot_conditioned=True)
    path = tmp_path / "agent_step_10.pt"
    torch.save(dict(model=net.state_dict(), width=16, history=1, shot_conditioned=True), path)
    (tmp_path / "run.json").write_text(json.dumps(dict(
        algorithm="neural_ppo_v1", action_mode="arrival", deployment_ready=False, simulation_only=True,
        args={"workspace": "rail30"}, physical_limits=dict(speed_m_s=12, acceleration_m_s2=60))))
    return path


TRACK = []


def report(t, puck, own=(0.5, 0.3)):
    """A tracker report with the puck's fixes of the last 200 ms, newest first (10 ms apart)."""
    if TRACK and not 0 < t - TRACK[0][2] <= 0.1:
        TRACK.clear()
    if TRACK:
        (x0, y0), t0 = (TRACK[0][3], TRACK[0][4]), TRACK[0][2]
        TRACK.insert(0, (*sim_to_table_mm((x0 + puck[0]) / 2, (y0 + puck[1]) / 2), (t0 + t) / 2, 0, 0))
    TRACK.insert(0, (*sim_to_table_mm(*puck), t, *puck))
    del TRACK[20:]
    return dict(t_s=t, mallet=sim_to_table_mm(*own), opponent=sim_to_table_mm(0.3, 1.6),
                puck=[sample[:3] for sample in TRACK], own_fresh=True)


def shoot(policy, t, network, launch=(0.5, 0.45)):
    """Feed the adapter a puck arriving on the robot's half, resting by its mallet, then struck
    at `network(request code)`. Returns the time after it, the request the actor was given
    (0 none .. 3 straight) and the shot the adapter resolved."""
    x, y = launch
    obs = None
    for k in range(4):                       # over the line and back: a new possession
        obs = policy.observe(report(t, (x, 1.2 - 0.25 * k) if k < 2 else (x, y), own=(x, y - 0.10)))
        t += 0.02
    request = obs[-3:].copy()
    code = int(np.argmax(request)) + 1 if request.any() else 0
    assert code == policy.shot
    velocity = network(code)
    resolved = None
    for _ in range(40):
        x, y = x + velocity[0] * 0.02, y + velocity[1] * 0.02
        if not CFG.puck_radius <= x <= CFG.width - CFG.puck_radius:
            x = min(max(x, CFG.puck_radius), CFG.width - CFG.puck_radius)
            velocity = (-velocity[0] * CFG.wall_restitution, velocity[1] * CFG.wall_tangential)
        obs = policy.observe(report(t, (x, y), own=(launch[0], launch[1] - 0.10)))
        np.testing.assert_array_equal(obs[-3:], request)          # fixed for the possession
        t += 0.02
        if policy.last_shot is not None:
            resolved = policy.last_shot
            break
    return t, code, resolved


def test_table_adapter_free_and_auto_requests(checkpoint):
    free = NeuralPolicy(checkpoint, 12000, 60000, shot_mode="free")
    t, request, shot = shoot(free, 1.0, lambda code: (0.0, 5.0))
    assert request == NONE and shot["route"] == STRAIGHT and shot["request"] == NONE
    assert free.selector.stats()["realised_when_free"]["straight"] == 1

    auto = NeuralPolicy(checkpoint, 12000, 60000, shot_mode="auto",
                        selector_options=dict(window=12, floor=0.2, ceiling=0.6, min_shots=4, explore=0.0))
    assert (auto.selector.window, auto.selector.floor, auto.selector.min_shots) == (12, 0.2, 4)
    # A network that only ever shoots straight when left to itself, and honours a request.
    honours = lambda code: {LEFT: (-5.0, 5.0), RIGHT: (5.0, 5.0)}.get(code, (0.0, 5.0))
    t, requests = 1.0, []
    for _ in range(14):
        t, request, shot = shoot(auto, t, honours)
        requests.append(request)
        assert shot is not None and shot["request"] == request and shot["route"] == (request or STRAIGHT)
    # Free until four straight shots are in the window; then the banks are asked for and read back as made.
    assert requests[:4] == [NONE] * 4 and requests[4] in (LEFT, RIGHT) and {LEFT, RIGHT} <= set(requests[4:])
    stats = auto.selector.stats()
    assert stats["realised"]["left"] >= 1 and stats["realised"]["right"] >= 1 and stats["honoured_share"] == 1
    # A session pause (a gap in the reports) keeps the window.
    auto.observe(report(t + 5.0, (0.5, 1.5)))
    assert len(auto.selector.recent) == 12 and auto.routes.flight is None
    with pytest.raises(ValueError):
        NeuralPolicy(checkpoint, 12000, 60000, shot_mode="none")


def test_runner_flags_alignment_line_and_tick_columns(checkpoint):
    from test_run_policy import rp
    assert rp.TICK_COLUMNS[-2:] == ["shot_request", "shot_route"]
    args = argparse.Namespace(shot_window=20, shot_floor=None, shot_ceiling=0.5, shot_min_shots=None, shot_explore=0.1)
    options = rp.selector_options(args)
    assert options == dict(window=20, floor=0.15, ceiling=0.5, min_shots=9, explore=0.1)
    assert rp.selector_options(argparse.Namespace()) == DEFAULTS
    caps = rp.Caps(speed_max=12000, accel_max=60000, accel_min=3000)
    load = lambda mode, **kw: rp.load_policy(f"neural:{checkpoint}", caps, shot_mode=mode, **kw)
    policy = load("auto", selector_options=options)
    line = rp.neural_request_alignment(policy)
    assert "auto" in line and "last 20 shots" in line and line.endswith("<-- DEVIATION")
    assert policy.selector.requests.sum() == 0                       # the warm-up's possessions are not the session's
    assert "DEVIATION" not in rp.neural_request_alignment(load("mix"))
    assert rp.neural_request_alignment(load("free")).endswith("DEVIATION")
    assert rp.neural_request_alignment(load("left")).endswith("DEVIATION")
    # A checkpoint trained with free requests: free and auto reproduce its inputs.
    meta = json.loads((checkpoint.parent / "run.json").read_text())
    meta["args"]["free_request_share"] = 0.5
    (checkpoint.parent / "run.json").write_text(json.dumps(meta))
    trained = load("free")
    line = rp.neural_request_alignment(trained)
    assert "free 50%" in line and "DEVIATION" not in line
    # The tick row carries the request every tick, and the route on the tick a shot resolves.
    assert rp._shot_columns(object()) == {}
    trained.observe(report(0.5, (0.5, 1.5)))
    assert rp._shot_columns(trained) == dict(shot_request="none")
    _, _, shot = shoot(trained, 1.0, lambda code: (-5.0, 5.0))
    assert rp._shot_columns(trained) == dict(shot_request="none", shot_route="left")
    fixed = load("left")
    fixed.last_shot = dict(route=NONE, how="lost")                    # a shot that could not be read says why
    assert rp._shot_columns(fixed) == dict(shot_request="left", shot_route="lost")


def test_session_replay_reads_requests_and_routes_from_a_ticks_file(tmp_path):
    sys.path.insert(0, str(BIN))
    import shot_route_replay as replay
    classifier_input = []
    # Two shots as the policy saw them: a left bank asked for and made, then straight under a right-bank request.
    for start, velocity, request in ((1.0, (-5.0, 5.0), LEFT), (4.0, (0.0, 5.0), RIGHT)):
        class Tap:
            def update(self, t, puck, velocity_seen, mallet):
                classifier_input.append((t, puck, velocity_seen, mallet, request))
        fly(Tap(), (0.5, 0.45), velocity, start_t=start, ticks=30)
    with open(tmp_path / "s1.ticks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["t_cam", "neural_obs"])
        writer.writeheader()
        for t, puck, velocity, mallet, request in classifier_input:
            obs = np.zeros(45)
            obs[0:2], obs[2:4], obs[4:6] = puck, np.array(velocity) / 6, mallet
            obs[42 + request - 1] = 1
            writer.writerow(dict(t_cam=t, neural_obs="|".join(f"{v:.8g}" for v in obs)))
        writer.writerow(dict(t_cam=9.0, neural_obs=""))                 # a blind tick
    result = replay.replay("s1", tmp_path, governor=True)
    counts = np.array(result["request_by_classified_route"]["counts"])
    assert result["resolved"] == 2 and counts[LEFT, LEFT] == 1 and counts[RIGHT, STRAIGHT] == 1
    assert result["request_honoured"]["by_request"] == dict(left="1/1", right="0/1", straight="0/0")
    assert "track_check" not in result and result["governor_bookkeeping"]["shots"] == 2
