"""Training-only possession/coverage measurements; never supplies actor actions."""
import numpy as np


def disc_avoiding_distance(start, goal, center, radius):
    """Shortest planar approach around a puck-sized disc, for reward shaping."""
    s, g = np.asarray(start)-center, np.asarray(goal)-center
    ds, dg = np.linalg.norm(s, axis=1), np.linalg.norm(g, axis=1)
    delta = g-s
    t = np.clip(-(s*delta).sum(1) / np.maximum((delta*delta).sum(1), 1e-12), 0, 1)
    clear = np.linalg.norm(s+t[:, None]*delta, axis=1) >= radius
    angle = np.arccos(np.clip((s*g).sum(1)/np.maximum(ds*dg, 1e-12), -1, 1))
    tangent = np.arccos(np.clip(radius/np.maximum(ds, radius), 0, 1)) + np.arccos(np.clip(radius/np.maximum(dg, radius), 0, 1))
    detour = np.sqrt(np.maximum(ds*ds-radius*radius, 0)) + np.sqrt(np.maximum(dg*dg-radius*radius, 0))
    detour += radius*np.maximum(angle-tangent, 0) + 2*np.maximum(radius-ds, 0)
    return np.where(clear, np.linalg.norm(delta, axis=1), detour)


def direct_goal_coverage_cost(puck, paddle, velocity, bounds, config,
                            acceleration=60.0, shot_speed=12.0, reaction_s=.035,
                            lateral_uncertainty=0.0):
    """Bounded geometric shortfall for covering possible direct goal shots.

    Consider every goal-mouth target and several reachable interception depths,
    rather than prescribing a home position. The acceleration disk is a generous
    reachability approximation, not a promise of a firmware-executable save.
    This supplies training rewards and evaluation diagnostics only.
    """
    puck, paddle, velocity = (np.asarray(x, float) for x in (puck, paddle, velocity))
    uncertainty = np.broadcast_to(np.asarray(lateral_uncertainty, float), (len(puck),))
    if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
        raise ValueError("release uncertainty must be finite and nonnegative")
    if uncertainty.any():
        # The opponent can change its release position while preparing. Cover
        # that range rather than assuming the visible point is already a shot.
        # This is a training measurement, never an actor target or input.
        sources = np.tile(puck, (3, 1))
        sources[:, 0] += np.concatenate((-uncertainty, np.zeros(len(puck)), uncertainty))
        sources[:, 0] = np.clip(sources[:, 0], config.puck_radius, config.width-config.puck_radius)
        return direct_goal_coverage_cost(sources, np.tile(paddle, (3, 1)),
            np.tile(velocity, (3, 1)), bounds, config, acceleration, shot_speed,
            reaction_s).reshape(3, len(puck)).max(axis=0)
    xmin, xmax, ymin, ymax = bounds
    lines = np.linspace(ymin + .005, ymax - .005, 7)
    mouth = config.goal_width / 2 - config.puck_radius - .005
    goals = np.linspace(config.width / 2 - mouth, config.width / 2 + mouth, 7)
    fraction = (puck[:, None, None, 1] - lines[None, None, :]) / np.maximum(puck[:, None, None, 1], .01)
    x = puck[:, None, None, 0] + fraction * (goals[None, :, None] - puck[:, None, None, 0])
    y = np.broadcast_to(lines, x.shape)
    targets = np.stack((x, y), axis=-1)
    flight = np.linalg.norm(targets - puck[:, None, None, :], axis=-1) / shot_speed
    available = np.maximum(flight - reaction_s, 0)
    coast = paddle[:, None, None, :] + velocity[:, None, None, :] * flight[..., None]
    gap = np.linalg.norm(targets - coast, axis=-1) - config.puck_radius - config.paddle_radius
    shortfall = np.maximum(gap - .5 * acceleration * available**2, 0)
    possible = (x >= xmin - config.puck_radius - config.paddle_radius) & (x <= xmax + config.puck_radius + config.paddle_radius)
    shortfall = np.where(possible, shortfall, 1.)
    return np.clip(shortfall.min(axis=2).max(axis=1), 0, .5)


def goal_shot_velocity(puck, goal_x, speed, bank_side, config,
                       wall_restitution=None, wall_tangential=None):
    """Training-only release aimed at y=0, direct or off one side rail.

    bank_side is -1 for the left rail, +1 for right, and 0 for direct.
    Normal/tangential rail losses change the launch angle; this is not the
    elastic mirror construction. No route is exposed to the defending actor.
    """
    puck = np.asarray(puck, float)
    side = np.asarray(bank_side)
    normal = config.wall_restitution if wall_restitution is None else wall_restitution
    tangent = config.wall_tangential if wall_tangential is None else wall_tangential
    wall = np.where(side < 0, config.puck_radius, config.width-config.puck_radius)
    dx = np.where(side == 0, goal_x-puck[:, 0],
                  wall-puck[:, 0] + np.asarray(tangent)/normal*(wall-goal_x))
    direction = np.column_stack((dx, -puck[:, 1]))
    return direction * (np.asarray(speed)/np.maximum(np.linalg.norm(direction, axis=1), 1e-9))[:, None]
