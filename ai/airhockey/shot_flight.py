"""Offline shot-direction diagnosis, including calibrated rail rebounds.

Uses privileged simulator launch states only for evaluation. It is neither a
policy input nor a prediction of whether an opponent will block the shot.
"""

import numpy as np

from airhockey.skill_benchmark import TrialPhysics
from airhockey.physics import TableConfig

PARAMETERS = (
    "puck_friction",
    "drag_b",
    "wall_restitution",
    "wall_tangential",
    "end_wall_tangential",
)


def open_goal_outcomes(launch, parameters=None, config=None, seconds=3):
    launch = np.asarray(launch, dtype=float)
    if launch.ndim != 2 or launch.shape[1] != 4:
        raise ValueError("launch must contain x, y, vx, vy per shot")
    if not len(launch):
        return np.zeros(0, bool)
    cfg = config or TableConfig()
    engine = TrialPhysics(len(launch), cfg)
    for j, name in enumerate(("puck_x", "puck_y", "puck_vx", "puck_vy")):
        getattr(engine, name)[:] = launch[:, j]
    if parameters is not None:
        for name in PARAMETERS:
            getattr(engine, name)[:] = parameters[name]
    engine.paddle_agent_x[:] = engine.paddle_agent_y[:] = -10
    engine.paddle_opp_x[:] = engine.paddle_opp_y[:] = -10
    engine._clamp_puck_speed()
    for _ in range(round(seconds / 0.0025)):
        engine.step(0.0025)
        # An end-rail miss is a miss even if later ricochets eventually score.
        miss = (engine.puck_vy < 0) & ~engine.finished
        engine.finished[miss] = True
        engine.puck_vx[miss] = engine.puck_vy[miss] = 0
        if engine.finished.all():
            break
    return engine.score_agent > 0


def first_goal_crossing(launch, parameters=None, config=None, seconds=3):
    """Cheap forward-shot geometry with side-rail damping and friction.

    Returns goal-line x (inf if it stops/needs too many banks) and a conservative
    full-puck-width goal flag. No end-rail ricochets count as aimed shots.
    """
    cfg = config or TableConfig()
    state = np.asarray(launch, dtype=float).copy()
    n = len(state)
    params = parameters or {}
    normal = np.broadcast_to(params.get("wall_restitution", cfg.wall_restitution), (n,))
    tangent = np.broadcast_to(params.get("wall_tangential", cfg.wall_tangential), (n,))
    a = np.broadcast_to(params.get("puck_friction", cfg.puck_friction), (n,)) * 9.81
    b = np.broadcast_to(params.get("drag_b", cfg.PUCK_DRAG_B), (n,))
    # Table friction coefficients are positive; tiny floors also support an
    # effectively frictionless diagnostic without divisions by zero.
    a, b = np.maximum(a, 1e-12), np.maximum(b, 1e-12)
    speed = np.linalg.norm(state[:, 2:], axis=1)
    state[:, 2:] *= np.minimum(1, cfg.max_puck_speed / np.maximum(speed, 1e-12))[
        :, None
    ]
    crossing = np.full(n, np.inf)
    valid = np.zeros(n, bool)
    elapsed = np.zeros(n)
    active = state[:, 3] > 0
    radius = cfg.puck_radius
    mouth = cfg.goal_width / 2 - radius
    for _ in range(5):
        x, y, vx, vy = state.T
        to_goal = (cfg.height - y) / np.maximum(vy, 1e-12)
        wall_x = np.where(vx >= 0, cfg.width - radius, radius)
        to_wall = np.divide(
            wall_x - x, vx, out=np.full(n, np.inf), where=abs(vx) > 1e-12
        )
        bank = to_wall < to_goal
        duration = np.maximum(np.minimum(to_wall, to_goal), 0)
        speed = np.hypot(vx, vy)
        distance = speed * duration
        next_square = (speed**2 + a / b) * np.exp(-2 * b * distance) - a / b
        next_speed = np.sqrt(np.maximum(next_square, 0))
        travel_time = (
            np.arctan(speed * np.sqrt(b / a)) - np.arctan(next_speed * np.sqrt(b / a))
        ) / np.sqrt(a * b)
        elapsed += np.where(active, travel_time, 0)
        reachable = active & (next_square > 0) & (elapsed <= seconds)
        at_goal = reachable & ~bank
        goal_x = x + vx * duration
        entrance_x = goal_x - radius * vx / np.maximum(vy, 1e-12)
        crossing[at_goal] = goal_x[at_goal]
        valid[at_goal] = (abs(goal_x[at_goal] - cfg.width / 2) < mouth) & (
            abs(entrance_x[at_goal] - cfg.width / 2) < mouth
        )
        active = reachable & bank
        state[:, 0] += vx * duration
        state[:, 1] += vy * duration
        state[active, 0] = wall_x[active]
        state[:, 2:] *= (next_speed / np.maximum(speed, 1e-12))[:, None]
        state[:, 2] *= np.where(active, -normal, 1)
        state[:, 3] *= np.where(active, tangent, 1)
    return crossing, valid
