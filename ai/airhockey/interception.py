"""Experimental observation-only defensive controller, for simulation evaluation.

Candidate arrivals use nominal rail response and the real firmware profile.
The external environment must still apply its workspace/motion guard. This
controller does not certify motor load or authorize physical deployment.
"""

import numpy as np
from airhockey.dynamics import workspace_in_sim, MAX_SPEED_M_S, MAX_ACCEL_M_S2
from airhockey.physics import TableConfig
from airhockey.intercept_motion import forecast


class InterceptionController:
    def __init__(self):
        self.cfg = TableConfig()
        ws = workspace_in_sim()
        self.low = np.array([ws["min_x"], ws["min_y"]])
        self.high = np.array([ws["max_x"], ws["max_y"]])
        self.lines = np.repeat([0.17, 0.25, 0.38, 0.52, 0.66], 5)
        self.caps = np.tile([8.0, 15.0, 25.0, 40.0, 60.0], 5)

    def __call__(self, observation, action):
        obs = np.asarray(observation)
        out = np.asarray(action, dtype=np.float32).copy()
        if obs.shape != (len(obs), 42) or out.shape != (len(obs), 3):
            raise ValueError(
                "interception requires observations [N,42] and actions [N,3]"
            )
        if not np.isfinite(obs).all() or not np.isfinite(out).all():
            raise ValueError("interception inputs must be finite")
        applied = np.zeros(len(obs), bool)
        active = (
            (obs[:, 3] < -1)
            & (obs[:, 1] > 0.13)
            & (obs[:, 1] < 1.8)
            & (obs[:, 33] > 0.5)
        )
        ids = np.flatnonzero(active)
        if not len(ids):
            return out, applied
        z = obs[ids]
        if not (
            np.allclose(z[:, 13] * MAX_SPEED_M_S, 12, atol=1e-4)
            and np.allclose(z[:, 14] * MAX_ACCEL_M_S2, 60, atol=1e-4)
        ):
            raise ValueError(
                "this experimental interceptor requires 12 m/s and 60 m/s² modeled caps"
            )
        n, k = len(ids), len(self.lines)
        duration = (self.lines - z[:, 1, None]) / z[:, 3, None]
        px = z[:, 0, None] + z[:, 2, None] * duration
        radius = self.cfg.puck_radius
        for _ in range(3):
            right, left = px > 1 - radius, px < radius
            px = np.where(
                right,
                1
                - radius
                - (px - (1 - radius))
                * self.cfg.wall_restitution
                / self.cfg.wall_tangential,
                px,
            )
            px = np.where(
                left,
                radius
                + (radius - px) * self.cfg.wall_restitution / self.cfg.wall_tangential,
                px,
            )
        target = np.stack((px, np.broadcast_to(self.lines, px.shape)), -1)
        target = np.clip(target, self.low, self.high)
        query = np.ceil((duration - 0.015) / 0.005).astype(int)
        valid = (query >= 1) & (query <= 160)
        error = np.full((n, k), np.inf)
        if valid.any():
            state = np.column_stack((z[:, 4:8] * 1000, z[:, 36:38] * 60000))
            queued_xy = self.low + (z[:, 38:40] + 1) * 0.5 * (self.high - self.low)
            queued = np.column_stack((queued_xy * 1000, z[:, 40] * 60000))
            command = np.concatenate(
                (target * 1000, np.broadcast_to(self.caps * 1000, (n, k))[..., None]),
                -1,
            )
            predicted = forecast(
                np.repeat(state, k, axis=0)[valid.ravel()],
                np.repeat(queued, k, axis=0)[valid.ravel()],
                command[valid],
                query[valid] * 5,
            )
            error[valid] = np.linalg.norm(
                predicted[:, :2] / 1000
                - np.stack((px, np.broadcast_to(self.lines, px.shape)), -1)[valid],
                axis=-1,
            )
        reachable = error < self.cfg.puck_radius + self.cfg.paddle_radius - 0.018
        cost = np.where(reachable, self.caps + duration * 2, 1000 + error * 100)
        chosen = cost.argmin(1)
        finite = np.isfinite(error[np.arange(n), chosen])
        use, selected = ids[finite], chosen[finite]
        lo = np.full(2, self.cfg.paddle_radius)
        hi = np.array([1.0, 1.0]) - self.cfg.paddle_radius
        out[use, :2] = 2 * (target[np.arange(n)[finite], selected] - lo) / (hi - lo) - 1
        out[use, 2] = 2 * np.sqrt((self.caps[selected] / 60 - 0.05) / 0.95) - 1
        applied[use] = True
        return out, applied


class InterceptionPolicy:
    """Inference wrapper. Save/train the underlying model separately."""

    def __init__(self, policy):
        self.policy = policy
        self.cfg = policy.cfg
        self.controller = InterceptionController()
        self.training_algorithm = "PPO with observed interception (experimental)"
        self.inference_mode = "prior"

    @property
    def _prev_mean_batch(self):
        return self.policy._prev_mean_batch

    @_prev_mean_batch.setter
    def _prev_mean_batch(self, value):
        self.policy._prev_mean_batch = value

    def act(self, observation, t0=False, eval_mode=True):
        import torch

        action = self.policy.act(observation, t0=t0, eval_mode=eval_mode).numpy()
        result, _ = self.controller(observation.numpy(), action)
        return torch.from_numpy(result)
