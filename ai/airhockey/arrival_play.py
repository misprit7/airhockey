"""Experimental simulation policy combining learned skills and arrival control.

The neural policy handles states outside the setup, recovery and interception
regions. All controller inputs are observations; executed motion remains subject
to the environment's firmware profile and motion guard. This module does not
 certify thermal load or enable physical deployment.
"""

import heapq

import numpy as np

from airhockey.arrival import ArrivalConfig, ArrivalDecoder
from airhockey.dynamics import MAX_ACCEL_M_S2, MAX_SPEED_M_S, workspace_in_sim
from airhockey.intercept_motion import forecast
from airhockey.motion import CartState
from airhockey.physics import TableConfig


class ArrivalPlayController:
    """Fixed direct-shot intent: 3 m/s toward goal center, then recover home.

    Uses a committed 160 ms contact deadline, collision-avoiding setup at
    10 m/s², and the least recovery acceleration predicted to meet a return.
    The experimental body model is explicitly 12 m/s, 60 m/s², 50 Hz, 15 ms
    command delay. It must not be used with other actuator limits silently.
    """

    def __init__(
        self,
        *,
        early_recovery=False,
        strike_accel=60.0,
        contact_duration=0.16,
    ):
        self.early_recovery = early_recovery
        if not np.isfinite(strike_accel) or not 3 <= strike_accel <= 60:
            raise ValueError("strike acceleration must be in [3,60] m/s²")
        self.strike_accel = strike_accel
        if not np.isfinite(contact_duration) or not 0.04 <= contact_duration <= 0.25:
            raise ValueError("contact duration must be in [.04,.25] seconds")
        self.contact_duration = contact_duration
        self.table = TableConfig()
        ws = workspace_in_sim()
        self.decoder = ArrivalDecoder(
            [ws[k] for k in ("min_x", "max_x", "min_y", "max_y")],
            ArrivalConfig(velocity_scale=4.5),
        )
        self.low, self.high = self.decoder.low, self.decoder.high
        self.nav_low, self.nav_high = self.low + 0.003, self.high - 0.003
        self.action_low = np.full(2, self.table.paddle_radius)
        self.action_high = (
            np.array([self.table.width, self.table.height / 2])
            - self.table.paddle_radius
        )
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        self.ring = np.column_stack((np.cos(angles), np.sin(angles)))
        self.remaining = None

    def waypoint(self, start, end, puck, extra_radius=0.0):
        physical = self.table.puck_radius + self.table.paddle_radius + extra_radius
        radius = min(
            physical + 0.04,
            np.linalg.norm(start - puck) - 0.004,
            np.linalg.norm(end - puck) - 0.004,
        )
        radius = max(physical + 0.002, radius)
        ring = puck + self.ring * (radius + 0.025)
        ring = ring[((ring >= self.nav_low) & (ring <= self.nav_high)).all(1)]
        nodes = np.vstack((start, end, ring))
        delta = nodes[None, :, :] - nodes[:, None, :]
        length = np.linalg.norm(delta, axis=-1)
        fraction = np.clip(
            ((puck - nodes)[:, None, :] * delta).sum(-1) / np.maximum(length**2, 1e-12),
            0,
            1,
        )
        closest = nodes[:, None, :] + fraction[:, :, None] * delta
        allowed = np.linalg.norm(closest - puck, axis=-1) >= radius - 1e-8
        if allowed[0, 1]:
            return end
        distance = np.full(len(nodes), np.inf)
        distance[0] = 0
        previous = np.full(len(nodes), -1)
        heap = [(0.0, 0)]
        while heap:
            cost, k = heapq.heappop(heap)
            if k == 1:
                break
            if cost > distance[k]:
                continue
            for j in np.flatnonzero(allowed[k]):
                score = cost + length[k, j]
                if score < distance[j]:
                    distance[j], previous[j] = score, k
                    heapq.heappush(heap, (score, j))
        if previous[1] < 0:
            direction = (start - puck) / max(np.linalg.norm(start - puck), 1e-9)
            return np.clip(
                puck + direction * (physical + 0.05), self.nav_low, self.nav_high
            )
        k = 1
        while previous[k] > 0:
            k = previous[k]
        return nodes[k]

    def encode(self, target, cap):
        target = np.asarray(target)
        cap = np.broadcast_to(np.asarray(cap), target.shape[:-1])
        return np.concatenate(
            (
                2 * (target - self.action_low) / (self.action_high - self.action_low)
                - 1,
                (2 * np.sqrt((cap / 60 - 0.05) / 0.95) - 1)[..., None],
            ),
            axis=-1,
        )

    def direction(self, puck, goal, bank):
        """Direct or single-bank direction under nominal rail response."""
        puck = np.asarray(puck)
        direction = np.stack(
            (goal - puck[..., 0], self.table.height - puck[..., 1]), -1
        )
        if np.any(bank != 0):
            wall = np.where(
                bank < 0,
                self.table.puck_radius,
                self.table.width - self.table.puck_radius,
            )
            dx = wall - puck[..., 0]
            ratio = (
                -self.table.wall_restitution
                / self.table.wall_tangential
                * dx
                / (goal - wall)
            )
            dy = (
                ratio / np.maximum(1 + ratio, 1e-6) * (self.table.height - puck[..., 1])
            )
            # A forecast beyond the intended rail cannot use this bank geometry.
            valid = (bank != 0) & (ratio > 0) & (dy > 0)
            direction = np.where(
                np.asarray(valid)[..., None], np.stack((dx, dy), -1), direction
            )
        return direction / np.maximum(
            np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9
        )

    def __call__(self, observation, action, t0=False, intent=None):
        x = np.asarray(observation)
        out = np.asarray(action, dtype=np.float32).copy()
        if x.shape != (len(x), 42) or out.shape != (len(x), 3):
            raise ValueError(
                "arrival play requires observations [N,42] and actions [N,3]"
            )
        if not np.isfinite(x).all() or not np.isfinite(out).all():
            raise ValueError("arrival play inputs must be finite")
        if intent is None:
            intent = np.tile([0.5, 3.0, 0.0], (len(x), 1))
        intent = np.asarray(intent, float)
        if (
            intent.shape != (len(x), 3)
            or not np.isfinite(intent).all()
            or np.any((intent[:, 0] < 0.35) | (intent[:, 0] > 0.65))
            or np.any((intent[:, 1] < 2) | (intent[:, 1] > 7))
            or not np.isin(intent[:, 2], [-1, 0, 1]).all()
        ):
            raise ValueError("intent requires bounded goal x, puck speed and bank side")
        if not (
            np.allclose(x[:, 13] * MAX_SPEED_M_S, 12, atol=1e-4)
            and np.allclose(x[:, 14] * MAX_ACCEL_M_S2, 60, atol=1e-4)
        ):
            raise ValueError("arrival play requires modeled caps of 12 m/s and 60 m/s²")
        if self.remaining is None or len(self.remaining) != len(x):
            self.remaining = np.zeros(len(x))
        self.remaining[np.broadcast_to(np.asarray(t0, bool), len(x))] = 0
        self.remaining = np.maximum(0, self.remaining - 0.02)
        self.remaining[(x[:, 32] > 0.5) | (x[:, 3] > 1)] = 0
        applied = np.zeros(len(x), bool)
        for i, o in enumerate(x):
            if o[32] > 0.5:
                continue
            if self.remaining[i] > 0:
                continue
            puck, pad, velocity = o[:2], o[4:6], o[2:4]
            speed = np.linalg.norm(velocity)
            if not (
                self.nav_low[1] + 0.08 < puck[1] < self.nav_high[1] + 0.10
                and speed < 0.8
            ):
                continue
            goal, power, bank = intent[i]
            direction = self.direction(puck, goal, bank)
            normal = power * direction - velocity
            normal /= np.linalg.norm(normal)
            windup = 0.17 + 0.03 * (power - 3)
            setup = np.clip(puck - normal * windup, self.nav_low, self.nav_high)
            ready = (
                np.linalg.norm(pad - setup) < 0.008
                and np.linalg.norm(o[6:8] - velocity) < 0.12
            )
            if speed > 0.15:
                gap = puck - pad
                ready = (
                    windup - 0.04 < np.dot(gap, normal) < windup + 0.05
                    and abs(gap[0] * normal[1] - gap[1] * normal[0]) < 0.025
                    and np.linalg.norm(o[6:8] - velocity) < 0.5
                )
            if ready:
                self.remaining[i] = self.contact_duration
            else:
                target = self.waypoint(pad, setup, puck)
                if np.linalg.norm(target - setup) < 0.001:
                    target = target + velocity * (0.03 + speed / 12.8)
                target = np.clip(target, self.nav_low, self.nav_high)
                out[i] = self.encode(target, 10)
                applied[i] = True
        ids = np.flatnonzero(self.remaining > 0)
        if len(ids):
            z = x[ids]
            puck = z[:, :2] + z[:, 2:4] * self.remaining[ids, None]
            direction = self.direction(puck, intent[ids, 0], intent[ids, 2])
            desired, incoming = intent[ids, 1, None] * direction, z[:, 2:4]
            normal = desired - incoming
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            contact = (
                puck - (self.table.paddle_radius + self.table.puck_radius) * normal
            )
            velocity = (desired + self.table.paddle_restitution * incoming) / (
                1 + self.table.paddle_restitution
            )
            arrival = np.zeros((len(ids), 6))
            arrival[:, :2] = 2 * (contact - self.low) / (self.high - self.low) - 1
            arrival[:, 2:4] = velocity / self.decoder.config.velocity_scale
            arrival[:, 4] = 2 * (self.contact_duration - 0.04) / (0.25 - 0.04) - 1
            arrival[:, 5] = 2 * np.sqrt((self.strike_accel / 60 - 0.05) / 0.95) - 1
            cart = CartState(len(ids))
            for j, key in enumerate(("x", "y", "vx", "vy")):
                getattr(cart, key)[:] = z[:, 4 + j] * 1000
            cart.ax[:], cart.ay[:] = z[:, 36] * 60000, z[:, 37] * 60000
            queue = self.low + (z[:, 38:40] + 1) * 0.5 * (self.high - self.low)
            target, cap = self.decoder.decode(
                cart,
                arrival,
                self.contact_duration - self.remaining[ids],
                queue,
                z[:, 40] * 60,
                max_speed=12,
                max_accel=60,
                delay_s=0.015,
            )
            out[ids] = self.encode(target, cap)
            applied[ids] = True
        # Recover while the puck is away. Commands still pass through the guard.
        recover = (x[:, 1] > 0.75) & (x[:, 3] > 0)
        if self.early_recovery:
            recover |= (x[:, 1] > 0.2) & (x[:, 3] > 1)
        ids = np.flatnonzero(recover & (x[:, 33] > 0.5))
        if len(ids):
            z, caps = x[ids], np.array([8.0, 15.0, 25.0, 40.0, 60.0])
            n, k = len(z), len(caps)
            deadline = np.clip(
                np.maximum(1.25 - z[:, 1], 0) / np.maximum(z[:, 3], 0.1) + 1 / 12,
                0.02,
                0.8,
            )
            state = np.column_stack((z[:, 4:8] * 1000, z[:, 36:38] * 60000))
            queue_xy = self.low + (z[:, 38:40] + 1) * 0.5 * (self.high - self.low)
            queue = np.column_stack((queue_xy * 1000, z[:, 40] * 60000))
            command = np.tile(
                np.column_stack((np.full(k, 500), np.full(k, 250), caps * 1000)), (n, 1)
            )
            end = forecast(
                np.repeat(state, k, axis=0),
                np.repeat(queue, k, axis=0),
                command,
                np.repeat(np.ceil((deadline - 0.015) / 0.001).astype(int), k),
            )
            error = np.linalg.norm(end[:, :2] / 1000 - [0.5, 0.25], axis=1).reshape(
                n, k
            )
            speed = np.linalg.norm(end[:, 2:4] / 1000, axis=1).reshape(n, k)
            cost = np.where(
                (error < 0.025) & (speed < 0.5),
                caps[None, :],
                1000 + error * 100 + speed,
            )
            chosen = cost.argmin(1)
            out[ids] = self.encode(np.tile([0.5, 0.25], (n, 1)), caps[chosen])
            applied[ids] = True
        return np.clip(out, -1, 1), applied


class ArrivalPlayPolicy:
    """Inference wrapper around an observation-only neural/interception policy."""

    def __init__(
        self,
        policy,
        *,
        early_recovery=False,
        strike_accel=60.0,
        contact_duration=0.16,
    ):
        self.policy, self.cfg = policy, policy.cfg
        self.controller = ArrivalPlayController(
            early_recovery=early_recovery,
            strike_accel=strike_accel,
            contact_duration=contact_duration,
        )
        self.training_algorithm = (
            "Learned skills with direct arrival control (experimental)"
        )
        self.inference_mode = "prior"

    @property
    def _prev_mean_batch(self):
        return self.policy._prev_mean_batch

    @_prev_mean_batch.setter
    def _prev_mean_batch(self, value):
        self.policy._prev_mean_batch = value

    def cancel_pending(self, mask):
        if self.controller.remaining is not None:
            self.controller.remaining[np.asarray(mask, bool)] = 0

    def act(self, observation, t0=False, eval_mode=True):
        import torch

        action = self.policy.act(observation, t0=t0, eval_mode=eval_mode).numpy()
        result, _ = self.controller(observation.numpy(), action, t0=t0)
        return torch.from_numpy(result)
