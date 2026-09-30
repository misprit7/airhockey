"""Shared physical feature layout for neural training and deployment."""
import numpy as np


def neural_features(base_obs, previous_action, loads, acceleration, queued_target,
                    queued_cap, max_accel, low, high, *, shot_conditioned=False, shot_request=None):
    base = np.asarray(base_obs)
    queued = 2 * (np.asarray(queued_target) - low) / (high - low) - 1
    physical = np.column_stack((base[:, :15], previous_action, loads,
                                np.asarray(acceleration) / 60000, queued,
                                np.asarray(queued_cap) / max_accel))
    rel = np.column_stack((base[:, :2] - base[:, 4:6],
                           base[:, 2:4] - base[:, 6:8]))
    rival = np.column_stack((base[:, 8:10] - base[:, 4:6],
                             base[:, 10:12] - base[:, 6:8]))
    physical[:, [2, 3, 6, 7, 10, 11]] /= 6
    rel[:, 2:] /= 6
    rival[:, 2:] /= 6
    result = np.clip(np.column_stack((physical, rel, rival)), -10, 10)
    if shot_conditioned:
        result = np.column_stack((result, base[:, 18:21] if shot_request is None else shot_request))
    return result.astype(np.float32)
