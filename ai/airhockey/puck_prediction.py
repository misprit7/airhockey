"""Nominal free puck flight from observations, without simulator state access."""

import numpy as np

from airhockey.physics import TableConfig


def incoming_crossings(state, lines, *, config=None, max_seconds=1.2):
    """Predict x, elapsed time and velocity at descending horizontal lines.

    Rows of ``state`` are x, y, vx, vy in SI units. ``lines`` is a 1D list
    of y coordinates. Side-rail rebounds use nominal calibrated coefficients;
    paddles and end rails are deliberately excluded. Unreachable crossings
    have infinite time. This is a nominal forecast, not a collision guarantee.
    """
    state, lines = np.asarray(state, float), np.asarray(lines, float)
    if state.ndim != 2 or state.shape[1] != 4 or lines.ndim != 1:
        raise ValueError("expected puck states [N,4] and crossing lines [K]")
    if not np.isfinite(state).all() or not np.isfinite(lines).all():
        raise ValueError("puck states and lines must be finite")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("prediction horizon must be finite and positive")
    cfg = config or TableConfig()
    n, k = len(state), len(lines)
    flight = np.repeat(state, k, axis=0).copy()
    target_y = np.tile(lines, n)
    result = np.full((n * k, 4), np.inf)
    elapsed = np.zeros(n * k)
    active = (flight[:, 3] < 0) & (flight[:, 1] > target_y)
    a, b = max(cfg.puck_friction * 9.81, 1e-12), max(cfg.PUCK_DRAG_B, 1e-12)
    speed = np.linalg.norm(flight[:, 2:], axis=1)
    flight[:, 2:] *= np.minimum(1, cfg.max_puck_speed / np.maximum(speed, 1e-12))[
        :, None
    ]
    for _ in range(12):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        x, y, vx, vy = flight[ids].T
        to_line = (target_y[ids] - y) / vy
        wall = np.where(vx >= 0, cfg.width - cfg.puck_radius, cfg.puck_radius)
        to_wall = np.divide(
            wall - x, vx, out=np.full(len(ids), np.inf), where=abs(vx) > 1e-12
        )
        bank = to_wall < to_line
        scale = np.maximum(np.minimum(to_line, to_wall), 0)
        speed = np.hypot(vx, vy)
        distance = speed * scale
        square = (speed**2 + a / b) * np.exp(-2 * b * distance) - a / b
        next_speed = np.sqrt(np.maximum(square, 0))
        duration = (
            np.arctan(speed * np.sqrt(b / a)) - np.arctan(next_speed * np.sqrt(b / a))
        ) / np.sqrt(a * b)
        elapsed[ids] += duration
        reached = (square > 0) & (elapsed[ids] <= max_seconds)
        flight[ids, :2] += flight[ids, 2:] * scale[:, None]
        flight[ids, 2:] *= (next_speed / np.maximum(speed, 1e-12))[:, None]
        done = ids[reached & ~bank]
        result[done, 0] = flight[done, 0]
        result[done, 1] = elapsed[done]
        result[done, 2:] = flight[done, 2:]
        bounce = ids[reached & bank]
        flight[bounce, 0] = wall[reached & bank]
        flight[bounce, 2] *= -cfg.wall_restitution
        flight[bounce, 3] *= cfg.wall_tangential
        active[ids] = reached & bank
    return result.reshape(n, k, 4)
