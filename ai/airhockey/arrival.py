"""Experimental arrival-state decoder. SI units; no hardware I/O.

An arrival is (x, y, vx, vy, seconds, acceleration fraction). A new arrival
can replace the old one at any 50 Hz decision. Holding an arrival decrements
its time; after its deadline a follow-through target lets the profile brake.
The decoder emits ordinary position/acceleration commands, which MUST still
pass through motion.advance (or the firmware). It never assigns paddle state.

This is a feedback approximation, not a guarantee of reaching an arbitrary
endpoint. Workspace, speed and acceleration limits can make requests infeasible.
The benchmark measures the resulting motion rather than assuming feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airhockey.motion import CartState, DEFAULT_SIM_DT, advance


@dataclass(frozen=True)
class ArrivalConfig:
    min_time: float = 0.04
    max_time: float = 0.25
    velocity_scale: float = 3.0
    action_dt: float = 0.02
    ramp_s: float = 0.003


def copy_cart(cart: CartState) -> CartState:
    result = CartState(len(cart))
    for name in CartState.__slots__:
        getattr(result, name)[:] = getattr(cart, name)
    return result


class ArrivalDecoder:
    """Minimum-energy arrival guidance followed by inverse profile control.

    First predict the paddle through the queued-command latency using the
    actual firmware. Compute cubic boundary-value guidance from that state,
    then invert the profile's braking curve to express velocity as a position
    target. Two local shooting corrections use the C profile to account for
    holding each command for 20 ms. Only the reference is cubic; executed motion uses the C profile.
    Targets and magnitudes are clipped consistently for infeasible requests.
    """

    def __init__(self, bounds, config: ArrivalConfig | None = None):
        self.bounds = np.asarray(bounds, dtype=float)  # xmin, xmax, ymin, ymax
        self.config = config or ArrivalConfig()
        if self.bounds.shape != (4,) or not np.isfinite(self.bounds).all():
            raise ValueError("bounds must contain four finite SI coordinates")
        if self.bounds[0] >= self.bounds[1] or self.bounds[2] >= self.bounds[3]:
            raise ValueError("workspace bounds must be increasing")
        cfg = self.config
        if not (0 < cfg.min_time <= cfg.max_time and cfg.velocity_scale > 0
                and cfg.action_dt > 0 and cfg.ramp_s > 0):
            raise ValueError("arrival scales and times must be positive")
        self.low = self.bounds[[0, 2]]
        self.high = self.bounds[[1, 3]]

    def unpack(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
            raise ValueError("arrival actions must be finite [N, 6]")
        a = np.clip(actions, -1, 1)
        cfg = self.config
        position = self.low + (a[:, :2] + 1) / 2 * (self.high - self.low)
        velocity = a[:, 2:4] * cfg.velocity_scale
        duration = cfg.min_time + (a[:, 4] + 1) / 2 * (cfg.max_time - cfg.min_time)
        fraction = 0.05 + 0.95 * ((a[:, 5] + 1) / 2) ** 2
        return position, velocity, duration, fraction

    def decode(self, cart, actions, elapsed, previous_target, previous_accel,
               *, max_speed=12.0, max_accel=60.0, delay_s=0.0125):
        """Return [N, 2] SI target and [N] acceleration cap.

        `elapsed` is time since this arrival was issued (zero for a new action).
        `cart` is the current actuator estimate, including jerk state, in mm.
        `previous_target/accel` describe the command executing during latency.
        delay_s includes decoder time and must match the caller's scheduler.
        Inputs are not mutated. Feedback is supplied on every call.
        """
        if not np.isfinite(delay_s) or not 0 <= delay_s < self.config.action_dt:
            raise ValueError("delay must be finite and shorter than the action period")
        position, velocity, duration, fraction = self.unpack(actions)
        if len(position) != len(cart):
            raise ValueError("one arrival per cart is required")
        cfg = self.config
        predicted = copy_cart(cart)
        if delay_s:
            ticks = max(1, round(delay_s / DEFAULT_SIM_DT))
            advance(predicted, previous_target[:, 0] * 1000,
                    previous_target[:, 1] * 1000, max_speed * 1000,
                    np.asarray(previous_accel) * 1000, cfg.ramp_s,
                    delay_s / ticks, ticks, bounds=tuple(self.bounds * 1000))
        p = np.column_stack((predicted.x, predicted.y)) / 1000
        v = np.column_stack((predicted.vx, predicted.vy)) / 1000
        remaining = duration - np.asarray(elapsed) - delay_s
        # Near the deadline, extend the reference just enough to pass through
        # the endpoint. After the deadline, keep a fixed braking target.
        original_position = position.copy()
        overrun = np.clip(cfg.action_dt - remaining, 0, cfg.action_dt)
        position = np.clip(position + velocity * overrun[:, None], self.low, self.high)
        t = np.maximum(remaining, cfg.action_dt)[:, None]
        a_cap = fraction * np.asarray(max_accel)
        acceleration = 6 * (position - p) / t**2 - (4 * v + 2 * velocity) / t
        jerk = -12 * (position - p) / t**3 + 6 * (v + velocity) / t**2
        end_velocity = v + acceleration * cfg.action_dt + jerk * cfg.action_dt**2 / 2
        delta_v = end_velocity - v
        delta_v *= np.minimum(1, a_cap * cfg.action_dt /
                             np.maximum(np.linalg.norm(delta_v, axis=1), 1e-9))[:, None]
        end_velocity = v + delta_v
        norm = np.linalg.norm(acceleration, axis=1)
        acceleration *= np.minimum(1, a_cap / np.maximum(norm, 1e-9))[:, None]
        # The command is held for 20 ms. Target the midpoint velocity plus
        # the firmware velocity-loop lag (tau = 2*ramp_s).
        desired_v = v + acceleration * (cfg.action_dt / 2 + 2 * cfg.ramp_s)
        speed = np.linalg.norm(desired_v, axis=1)
        desired_v *= np.minimum(1, np.asarray(max_speed) / np.maximum(speed, 1e-9))[:, None]
        speed = np.linalg.norm(desired_v, axis=1)
        # v_des = 0.8 sqrt(2 a distance); inverse in vector form.
        midpoint = p + v * (cfg.action_dt / 2) + acceleration * (cfg.action_dt**2 / 8)
        target = midpoint + desired_v * (speed / (1.28 * a_cap))[:, None]
        target = np.clip(target, self.low, self.high)

        def endpoint_velocity(candidate):
            sim = copy_cart(predicted)
            ticks = round(cfg.action_dt / DEFAULT_SIM_DT)
            advance(sim, candidate[:, 0] * 1000, candidate[:, 1] * 1000,
                    np.asarray(max_speed) * 1000, a_cap * 1000, cfg.ramp_s,
                    cfg.action_dt / ticks, ticks, bounds=tuple(self.bounds * 1000))
            return np.column_stack((sim.vx, sim.vy)) / 1000

        # Damped least squares handles saturated/flat regions gracefully;
        # cap each correction so a locally bad Jacobian cannot jump the box.
        epsilon = 0.001
        for _ in range(2):
            base = endpoint_velocity(target)
            jx = (endpoint_velocity(target + [epsilon, 0]) - base) / epsilon
            jy = (endpoint_velocity(target + [0, epsilon]) - base) / epsilon
            residual = end_velocity - base
            aa = np.sum(jx * jx, axis=1) + 0.1
            bb = np.sum(jx * jy, axis=1)
            cc = np.sum(jy * jy, axis=1) + 0.1
            rx = np.sum(jx * residual, axis=1)
            ry = np.sum(jy * residual, axis=1)
            determinant = aa * cc - bb**2
            step = np.column_stack(((cc * rx - bb * ry) / determinant,
                                    (aa * ry - bb * rx) / determinant))
            step *= np.minimum(1, 0.05 / np.maximum(np.linalg.norm(step, axis=1), 1e-9))[:, None]
            target = np.clip(target + step, self.low, self.high)
        terminal_speed = np.linalg.norm(velocity, axis=1)
        terminal_target = original_position + velocity * (terminal_speed / (1.28 * a_cap))[:, None]
        target = np.where((remaining <= 0)[:, None], terminal_target, target)
        return np.clip(target, self.low, self.high), a_cap
