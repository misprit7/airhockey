import numpy as np
from airhockey.setup_shaping import setup_potential, potential_reward


def observation(paddles):
    obs = np.zeros((len(paddles), 42))
    obs[:, :2] = [0.5, 0.5]
    obs[:, 4:6] = paddles
    obs[:, 33] = 1
    return obs


def test_setup_prefers_going_around_puck_then_lining_up():
    obs = observation([[0.5, 0.68], [0.64, 0.5], [0.5, 0.34]])
    phi = setup_potential(obs)
    assert phi[0] < -0.34  # straight route intersects the puck
    assert phi[0] < phi[1] < phi[2]
    np.testing.assert_allclose(phi[2], 0, atol=1e-8)


def test_setup_is_inactive_for_cushioning_and_incoming_fast_shots():
    obs = observation([[0.5, 0.68], [0.5, 0.68]])
    obs[0, 32:34] = [1, 0]
    obs[1, 3] = -5
    assert np.max(np.abs(setup_potential(obs))) < 1e-12


def test_discounted_shaping_cannot_reward_repeated_positioning():
    gamma, weight = 0.995, 20
    # A circuit and a direct route have the same discounted shaping return.
    for path in ([-0.5, -0.2, -0.5, -0.2, -0.1], [-0.5, -0.1]):
        before, after = np.array(path[:-1]), np.array(path[1:])
        terminal = np.zeros(len(before), bool)
        terminal[-1] = True
        reward = potential_reward(before, after, terminal, weight, gamma)
        assert np.isclose(np.sum(gamma ** np.arange(len(reward)) * reward), 10)


def test_truncation_keeps_potential_for_value_bootstrap():
    reward = potential_reward(
        np.array([-0.3, -0.3]), np.array([-0.2, -0.2]), np.array([False, True]), 20
    )
    np.testing.assert_allclose(reward, [2.02, 6])
