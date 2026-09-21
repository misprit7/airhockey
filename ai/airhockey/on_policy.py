"""On-policy returns with distinct terminal and rollout-cutoff semantics."""

import torch


def advantages(reward, value, next_value, terminated, done, gamma=0.995, lam=0.95):
    """Bootstrap time limits, but never leak an advantage through a reset."""
    result = torch.empty_like(reward)
    tail = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * (~terminated[t]) - value[t]
        tail = delta + gamma * lam * (~done[t]) * tail
        result[t] = tail
    return result, result + value


def exploration_scale(obs, base, setup_std=None, urgent_accel_std=None):
    """Explore setup routes/bursts without widening precision shot noise."""
    scale = base.expand(len(obs), -1).clone()
    if setup_std is not None:
        speed = torch.linalg.vector_norm(obs[:, 2:4], dim=-1)
        gap = torch.linalg.vector_norm(obs[:, :2] - obs[:, 4:6], dim=-1)
        awkward = (
            (obs[:, 5] > obs[:, 1] - 0.1)
            | ((obs[:, 4] - obs[:, 0]).abs() > 0.08)
            | (gap > 0.24)
        )
        setup = (speed < 1.2) & (obs[:, 1] < 0.8) & awkward
        scale[setup, :2] = setup_std
    if urgent_accel_std is not None:
        urgent = (obs[:, 3] < -1.5) & (obs[:, 1] < 1.2)
        scale[urgent, 2] = urgent_accel_std
    return scale
