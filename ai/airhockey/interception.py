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
    def __init__(
        self,
        *,
        calibrated_flight=False,
        minimum_incoming_speed=None,
        include_low_line=None,
        contact_tolerance=None,
        threat_only=False,
        acceleration_caps=(8.0, 15.0, 25.0, 40.0, 60.0),
    ):
        self.cfg = TableConfig()
        self.threat_only = threat_only
        self.contact_tolerance = (
            self.cfg.puck_radius + self.cfg.paddle_radius - 0.018
            if contact_tolerance is None
            else contact_tolerance
        )
        if not np.isfinite(self.contact_tolerance) or self.contact_tolerance <= 0:
            raise ValueError("contact tolerance must be finite and positive")
        ws = workspace_in_sim()
        self.low = np.array([ws["min_x"], ws["min_y"]])
        self.high = np.array([ws["max_x"], ws["max_y"]])
        caps = np.asarray(acceleration_caps, float)
        if (
            caps.ndim != 1
            or not len(caps)
            or not np.isfinite(caps).all()
            or ((caps < 3) | (caps > 60)).any()
        ):
            raise ValueError(
                "interception acceleration candidates must be in [3,60] m/s²"
            )
        self.calibrated_flight = calibrated_flight
        self.minimum_incoming_speed = (
            (0.1 if calibrated_flight else 1.0)
            if minimum_incoming_speed is None
            else minimum_incoming_speed
        )
        if (
            not np.isfinite(self.minimum_incoming_speed)
            or self.minimum_incoming_speed <= 0
        ):
            raise ValueError("minimum incoming speed must be finite and positive")
        self.include_low_line = (
            calibrated_flight if include_low_line is None else include_low_line
        )
        if self.include_low_line:
            lines = [0.11, 0.17, 0.25, 0.38, 0.52, 0.66]
        else:
            lines = [0.17, 0.25, 0.38, 0.52, 0.66]
        self.lines = np.repeat(lines, len(caps))
        self.caps = np.tile(caps, len(lines))

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
            (obs[:, 3] < -self.minimum_incoming_speed)
            & (obs[:, 1] > (0.09 if self.include_low_line else 0.13))
            & (obs[:, 1] < 1.8)
            & (obs[:, 33] > 0.5)
        )
        ids = np.flatnonzero(active)
        if len(ids) and not (
            np.allclose(obs[ids, 13] * MAX_SPEED_M_S, 12, atol=1e-4)
            and np.allclose(obs[ids, 14] * MAX_ACCEL_M_S2, 60, atol=1e-4)
        ):
            raise ValueError(
                "this experimental interceptor requires 12 m/s and 60 m/s² modeled caps"
            )
        if self.threat_only and len(ids):
            from airhockey.shot_flight import first_goal_crossing

            launch = obs[ids, :4].copy()
            launch[:, 1] = self.cfg.height - launch[:, 1]
            launch[:, 3] *= -1
            threatened = np.zeros(len(ids), bool)
            # Broad rail-response envelope; never rely on an exact bank estimate
            # to decide a shot cannot enter the goal. Re-evaluate every 20 ms.
            for normal, tangent in [
                (0.72, 0.85),
                (0.72, 0.95),
                (0.86, 0.85),
                (0.86, 0.95),
            ]:
                crossing, _ = first_goal_crossing(
                    launch, dict(wall_restitution=normal, wall_tangential=tangent)
                )
                threatened |= abs(crossing - 0.5) < self.cfg.goal_width / 2 + 0.12
            park = ids[~threatened]
            lo = np.full(2, self.cfg.paddle_radius)
            hi = np.array([1.0, 1.0]) - self.cfg.paddle_radius
            out[park, :2] = 2 * (np.array([0.5, 0.25]) - lo) / (hi - lo) - 1
            out[park, 2] = 2 * np.sqrt((8 / 60 - 0.05) / 0.95) - 1
            applied[park] = True
            ids = ids[threatened]
        if not len(ids):
            return out, applied
        z = obs[ids]
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
        if self.calibrated_flight:
            from airhockey.puck_prediction import incoming_crossings

            crossing = incoming_crossings(z[:, :4], self.lines)
            px, duration = crossing[:, :, 0], crossing[:, :, 1]
        target = np.stack((px, np.broadcast_to(self.lines, px.shape)), -1)
        target = np.clip(target, self.low, self.high)
        query = np.ceil(
            (np.where(np.isfinite(duration), duration, -1) - 0.015) / 0.005
        ).astype(int)
        valid = (query >= 1) & (query <= (240 if self.calibrated_flight else 160))
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
        reachable = error < self.contact_tolerance
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
