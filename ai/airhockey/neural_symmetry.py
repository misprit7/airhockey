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
