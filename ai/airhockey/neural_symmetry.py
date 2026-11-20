"""Training-data reflection for physical neural observations, never a controller.

Cartesian puck dynamics and the normalized arrival workspace are symmetric.
The fitted motor model is NOT exactly symmetric; reflected examples are a
training prior, and resulting actors require unmodified thermal evaluation.
"""
import numpy as np


def reflect_physical(observation):
    result = np.array(observation, copy=True)
    if result.ndim != 2 or result.shape[1] != 42:
        raise ValueError("reflection requires one frame of 42 physical features")
    result[:, [0, 4, 8]] = 1 - result[:, [0, 4, 8]]
    # Velocities, previous arrival x/vx, acceleration x, queued target x,
    # and relative x position/velocity. Y, caps and time remain unchanged.
    result[:, [2, 6, 10, 15, 17, 29, 31, 34, 36, 38, 40]] *= -1
    # Fast and slow load channels swap the corresponding reflected anchors.
    result[:, 21:25] = result[:, [24, 23, 22, 21]]
    result[:, 25:29] = result[:, [28, 27, 26, 25]]
    return result


def reflect_arrival(action):
    result = np.array(action, copy=True)
    if result.ndim != 2 or result.shape[1] != 6:
        raise ValueError("reflection requires six arrival outputs")
    result[:, [0, 2]] *= -1
    return result


# ── The same reflection for what the PPO trainer holds (torch) ────────────

_NEGATED = (2, 6, 10, 15, 17, 29, 31, 34, 36, 38, 40)
_FLIPPED = (0, 4, 8)


def reflect_policy_observation(observation, history):
    """The actor's input (history x 42 physical features, then the 3-wide
    shot request when there is one) as seen on the mirrored table.

    Every frame is reflected as `reflect_physical` does one; the request's
    left bank and right bank change places. A torch tensor in, a new one out.
    """
    import torch
    physical = 42 * history
    extra = observation.shape[-1] - physical
    if extra not in (0, 3):
        raise ValueError("expected history x 42 physical features, optionally followed by a 3-wide shot request")
    frames = observation[..., :physical].reshape(*observation.shape[:-1], history, 42)
    order = torch.arange(42, device=observation.device)
    order[21:25] = torch.tensor([24, 23, 22, 21], device=observation.device)
    order[25:29] = torch.tensor([28, 27, 26, 25], device=observation.device)
    sign = torch.ones(42, device=observation.device, dtype=observation.dtype)
    sign[list(_NEGATED)] = -1
    offset = torch.zeros(42, device=observation.device, dtype=observation.dtype)
    offset[list(_FLIPPED)] = 1
    sign[list(_FLIPPED)] = -1                      # x -> 1 - x
    mirrored = (frames[..., order] * sign + offset).flatten(-2)
    if extra:
        request = observation[..., physical:]
        mirrored = torch.cat((mirrored, request[..., [1, 0, 2]]), dim=-1)
    return mirrored


def reflect_raw_action(action):
    """Six arrival outputs (before or after the tanh) on the mirrored table."""
    result = action.clone()
    result[..., [0, 2]] *= -1
    return result


def mirror_success_loss(net, observation, raw_action, advantage, *, cap=3.0, space="pre-squash"):
    """Imitation of the policy's OWN successes, mirrored (training only).

    For each sample with a positive advantage: the action that was taken,
    reflected, is a target for the actor in the reflected state, weighted by
    the advantage (clipped at `cap`). A skill the policy has on one side of
    the table is carried to the other, with no outside reference; what did
    not work (advantage <= 0) teaches nothing here. The motor model is not
    mirror-symmetric, so this is a prior: the policy's own return on the
    other side still decides.

    Returns (loss, number of samples that counted); the loss is the weighted
    mean of `neural_finetune.anchor_error`, zero when nothing counted.
    """
    from airhockey.neural_finetune import anchor_error
    weight = advantage.clamp(0, cap)
    chosen = weight > 0
    count = int(chosen.sum())
    if not count:
        return observation.new_zeros(()), 0
    mirrored = reflect_policy_observation(observation[chosen], net.history)
    target = reflect_raw_action(raw_action[chosen]).tanh()
    mean = net.actor(net.trunk(mirrored))
    error = anchor_error(mean, target, space)
    weight = weight[chosen]
    return (weight * error).sum() / weight.sum(), count
