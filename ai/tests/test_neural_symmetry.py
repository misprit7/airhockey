"""Left-right reflection of what the PPO trainer holds, and the mirrored self-imitation built on it."""
import numpy as np
import pytest
import torch

from airhockey.neural_finetune import anchor_error
from airhockey.neural_player import NeuralPlayer
from airhockey.neural_symmetry import (mirror_success_loss, reflect_arrival, reflect_physical,
                                       reflect_policy_observation, reflect_raw_action)


def test_the_stacked_observation_reflects_frame_by_frame_and_swaps_the_bank_requests():
    rng = np.random.default_rng(3)
    history = 4
    frames = rng.uniform(-1, 1, (5, history, 42)).astype(np.float32)
    request = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1]]
    obs = torch.as_tensor(np.column_stack((frames.reshape(5, -1), request)))
    mirrored = reflect_policy_observation(obs, history)
    expected = np.stack([reflect_physical(frames[:, k]) for k in range(history)], axis=1).reshape(5, -1)
    np.testing.assert_allclose(mirrored[:, :42 * history].numpy(), expected, atol=1e-6)
    np.testing.assert_array_equal(mirrored[:, 42 * history:].numpy(), request[:, [1, 0, 2]])
    np.testing.assert_allclose(reflect_policy_observation(mirrored, history).numpy(), obs.numpy(), atol=1e-6)
    # Without a request, and for one frame, the layout is just the frames.
    np.testing.assert_allclose(reflect_policy_observation(obs[:, :42 * history], history).numpy(), expected, atol=1e-6)
    with pytest.raises(ValueError):
        reflect_policy_observation(obs[:, :-1], history)
    action = torch.as_tensor(rng.uniform(-2, 2, (5, 6)).astype(np.float32))
    np.testing.assert_allclose(reflect_raw_action(action).numpy(), reflect_arrival(action.numpy()))


def test_only_its_own_successes_are_imitated_and_in_the_mirrored_state():
    torch.manual_seed(2)
    net = NeuralPlayer(16, history=2, shot_conditioned=True)
    obs = torch.randn(12, net.obs_dim) * 0.3
    raw = torch.randn(12, 6) * 0.5
    advantage = torch.tensor([2.0, -1.0, 0.0, 5.0, 0.5, -3.0, 1.0, -0.2, 0.0, 0.3, -1.0, 4.0])
    loss, counted = mirror_success_loss(net, obs, raw, advantage, cap=3.0)
    chosen = advantage > 0
    assert counted == int(chosen.sum()) == 6
    weight = advantage[chosen].clamp(max=3.0)
    mean = net.actor(net.trunk(reflect_policy_observation(obs[chosen], net.history)))
    by_hand = (weight * anchor_error(mean, reflect_raw_action(raw[chosen]).tanh(), "pre-squash")).sum() / weight.sum()
    assert float(loss.detach()) == pytest.approx(float(by_hand.detach()), rel=1e-6)
    # Nothing to learn from what did not work.
    nothing, none = mirror_success_loss(net, obs, raw, -advantage.abs())
    assert none == 0 and float(nothing) == 0
    # A few steps on it move the actor toward the mirrored action in the mirrored state, and only there.
    optimiser = torch.optim.SGD(net.parameters(), lr=0.05)
    def gap(rows):
        with torch.no_grad():
            mirrored = net.actor(net.trunk(reflect_policy_observation(obs[rows], net.history)))
            return float((mirrored - reflect_raw_action(raw[rows])).square().mean())
    before, ignored = gap(chosen), gap(~chosen)
    for _ in range(60):
        optimiser.zero_grad()
        mirror_success_loss(net, obs, raw, advantage)[0].backward()
        optimiser.step()
    assert gap(chosen) < 0.6 * before and gap(~chosen) > 0.6 * ignored
