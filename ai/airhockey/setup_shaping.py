"""Potential for approaching the puck without driving through it.

Uses policy observations only. Applied as gamma*Phi(next)-Phi(current), with
zero potential at a true terminal state, so discounted episode returns differ
by a fixed initial-state constant rather than paying repeatedly for positioning.
"""

import numpy as np
from airhockey.physics import TableConfig
from airhockey.dynamics import workspace_in_sim


def setup_potential(obs):
    obs = np.asarray(obs)
    cfg = TableConfig()
    ws = workspace_in_sim()
    puck, paddle = obs[:, :2], obs[:, 4:6]
    direction = np.column_stack((cfg.width / 2 - puck[:, 0], cfg.height - puck[:, 1]))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    goal = np.clip(
        puck - 0.16 * direction,
        [ws["min_x"], ws["min_y"]],
        [ws["max_x"], ws["max_y"]],
    )
    start, finish = paddle - puck, goal - puck
    ds, dg = np.linalg.norm(start, axis=1), np.linalg.norm(finish, axis=1)
    radius = cfg.paddle_radius + cfg.puck_radius + 0.02
    rs, rg = np.maximum(ds, radius), np.maximum(dg, radius)
    angle = np.arccos(
        np.clip(np.sum(start * finish, axis=1) / np.maximum(ds * dg, 1e-9), -1, 1)
    )
    tangent_angles = np.arccos(radius / rs) + np.arccos(radius / rg)
    arc = np.maximum(angle - tangent_angles, 0)
    detour = np.sqrt(rs**2 - radius**2) + np.sqrt(rg**2 - radius**2) + radius * arc
    detour += np.maximum(radius - ds, 0)
    straight = np.linalg.norm(paddle - goal, axis=1)
    distance = np.where(angle > tangent_angles, detour, straight)
    speed = np.linalg.norm(obs[:, 2:4], axis=1)
    weight = np.exp(-np.minimum((speed / 1.2) ** 4, 80))
    weight *= np.clip((0.95 - puck[:, 1]) / 0.15, 0, 1)
    weight *= np.clip((puck[:, 1] - 0.15) / 0.08, 0, 1)
    weight *= (dg > radius) & (obs[:, 32] < 0.5)
    return -weight * np.minimum(distance, 0.8)


def potential_reward(before, after, terminated, weight, gamma=0.995):
    return weight * (gamma * np.where(terminated, 0, after) - before)
