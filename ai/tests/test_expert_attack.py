"""The physics teacher's attack: aim geometry, reachable contacts, line strike."""
import numpy as np
import pytest

from airhockey.expert_player import ExpertPlayer, _State, load_expert
from airhockey.physics import TableConfig
from airhockey.shot_flight import first_goal_crossing


def _player():
    player, _ = load_expert("expert")
    return player


def _state(puck, request):
    n = len(puck)
    return _State(puck=puck, vel=np.zeros((n, 2)), paddle=np.tile([0.5, 0.2], (n, 1)),
                  paddle_vel=np.zeros((n, 2)), rival=np.tile([0.5, 1.8], (n, 1)),
                  load=np.zeros((n, 8)), request=np.tile(request, (n, 1)))


@pytest.mark.parametrize("route,rail", [(1, -1), (2, 1), (3, 0)])
def test_every_route_reaches_the_goal_centre_on_its_rail(route, rail):
    player = _player()
    rng = np.random.default_rng(1)
    puck = np.column_stack((rng.uniform(0.12, 0.88, 64), rng.uniform(0.2, 0.85, 64)))
    d = player._shot_direction(puck, _state(puck, np.eye(3)[route - 1]), route=np.full(64, route))
    launch = np.column_stack((puck, d * 8.0))
    crossing, valid, banks, first = first_goal_crossing(launch, None, TableConfig(), return_route=True)
    assert valid.all()
    assert (first == rail).all() and ((banks == 0) == (rail == 0)).all()
    assert np.abs(crossing - 0.5).max() < 0.03


def test_reachable_normal_keeps_contact_inside_the_workspace():
    player = _player()
    R = player.contact
    rail = np.array([[0.045, 0.5], [0.955, 0.4], [0.07, 0.3], [0.5, 0.5]])
    want = np.tile([0.0, 1.0], (len(rail), 1))
    n = player._reachable_normal(rail, want)
    assert np.allclose(np.linalg.norm(n, axis=1), 1)
    contact = rail - n * R
    assert ((contact >= player.low - 1e-6) & (contact <= player.high + 1e-6)).all()
    assert (n[:, 1] > 0.9).all()  # pushed up-table along the rail, never back
    assert np.allclose(n[-1], [0, 1])  # interior pucks keep the requested normal
    # Back wall and corners: reachable only when the pinned rescue is asked for.
    wall = np.array([[0.041, 0.041], [0.959, 0.049], [0.5, 0.05]])
    n = player._reachable_normal(wall, np.tile([0.0, 1.0], (3, 1)), pinned=np.ones(3, bool))
    contact = wall - n * R
    assert ((contact >= player.low - 1e-6) & (contact <= player.high + 1e-6)).all()


def test_line_strike_sends_a_resting_puck_fast_along_the_request():
    from airhockey.neural_player import PhysicalHistory
    from airhockey.neural_training import NeuralTrainingEnv
    from airhockey.skill_benchmark import Fixtures

    player = _player()
    n = 24
    env = NeuralTrainingEnv(n, stage=2, seed=3, realistic=True, report_sensing=True, shot_conditioned=True,
                            shot_request="straight", possession_followthrough=True, **player.environment_options)
    puck = np.zeros((n, 4))
    puck[:, 0] = np.linspace(0.3, 0.7, n)
    puck[:, 1] = 0.5
    paddle = puck[:, :2] - [0.0, 0.25]
    obs = env.reset(seed=3, fixtures=Fixtures(np.zeros(n, int), puck, paddle, np.full(n, 0.5)))
    env.kind[:] = 0
    history = PhysicalHistory(player.history)
    history.reset(env._features(env.base._make_obs_direct()))
    for _ in range(40):
        obs, *_ = env.step(player.act(history.for_policy(player)))
        history.append(obs)
    on_route = (env.aimed_route_counts[0, :, 3, 3] > 0).mean()
    speed = env.shot_speed_sum[0] / np.maximum(env.shot_count[0], 1)
    assert on_route >= 0.8
    assert np.median(speed) >= 7.0
