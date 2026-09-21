"""Isolated, reproducible skill trials using the production simulation physics.

This is an offline action-search benchmark, not a learned-policy evaluation.
Both representations get the same number of candidate rollouts and the same
task objective. The position baseline can change all three commands at 50 Hz;
the arrival action has six values and a feedback decoder at that same rate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from airhockey.arrival import ArrivalDecoder
from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.dynamics import workspace_in_sim
from airhockey.motion import CartState, advance
from airhockey.physics import TableConfig

ACTION_DT = 0.02
PHYSICS_DT = 0.0025
DECISIONS = 16
TASKS = ("stationary", "moving", "cushion")


@dataclass
class Fixtures:
    task: np.ndarray
    puck: np.ndarray  # x, y, vx, vy
    paddle: np.ndarray  # x, y
    aim: np.ndarray

    def take(self, indices):
        return Fixtures(*(getattr(self, k)[indices] for k in self.__dataclass_fields__))

def make_fixtures(seed, per_task):
    rng = np.random.default_rng(seed)
    task = np.repeat(np.arange(3), per_task)
    n = len(task)
    puck = np.zeros((n, 4))
    puck[:, 0] = rng.uniform(0.34, 0.66, n)
    puck[:, 1] = rng.uniform(0.43, 0.55, n)
    puck[task == 1, 2:] = rng.uniform([-0.45, -0.25], [0.45, 0.15], (per_task, 2))
    paddle = puck[:, :2] + rng.uniform([-0.065, -0.20], [0.065, -0.12], (n, 2))
    # Incoming pucks cross the reachable interior, with enough retreat space.
    c = task == 2
    puck[c, 1] = rng.uniform(0.64, 0.74, per_task)
    puck[c, 2] = rng.uniform(-0.3, 0.3, per_task)
    puck[c, 3] = rng.uniform(-2.3, -1.2, per_task)
    paddle[c, 0] = puck[c, 0] + puck[c, 2] * 0.14 + rng.uniform(-0.03, 0.03, per_task)
    paddle[c, 1] = puck[c, 1] + puck[c, 3] * 0.14 - 0.0911
    return Fixtures(task, puck, paddle, rng.uniform(0.40, 0.60, n))


class TrialPhysics(BatchPhysicsEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finished = np.zeros(self.n_envs, dtype=bool)

    def _reset_puck_subset(self, mask, toward_agent, rng):
        # No invented serves. Leave the scored puck beyond the goal plane.
        self.finished[mask] = True
        self.puck_vx[mask] = self.puck_vy[mask] = 0

    def _check_goals(self):
        finished = self.finished.copy()
        agent, opponent = self.score_agent.copy(), self.score_opponent.copy()
        super()._check_goals()
        self.score_agent[finished] = agent[finished]
        self.score_opponent[finished] = opponent[finished]
        self.goal_scored[finished] = 0


class SkillTrials:
    def __init__(self, accel=60.0, delay=0.015, effort_weight=0.03):
        if not 0 <= delay < ACTION_DT:
            raise ValueError("delay must be shorter than 20 ms")
        if not np.isfinite(accel) or accel <= 0:
            raise ValueError("acceleration must be positive and finite")
        ws = workspace_in_sim()
        self.bounds = np.array([ws[k] for k in ("min_x", "max_x", "min_y", "max_y")])
        self.decoder = ArrivalDecoder(self.bounds)
        self.cfg = TableConfig()
        self.accel = accel
        # Same quantization as BatchAirHockeyEnv: activate at first substep
        # whose start is >= requested delay.
        self.delay = np.ceil(delay / PHYSICS_DT - 1e-9) * PHYSICS_DT
        self.effort_weight = effort_weight

    def seed_arrival(self, f):
        duration = np.full(len(f.task), 0.14)
        puck = f.puck[:, :2] + f.puck[:, 2:] * duration[:, None]
        direction = np.column_stack((f.aim - puck[:, 0], self.cfg.height - puck[:, 1]))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        velocity = direction * 1.7
        c = f.task == 2
        direction[c] = -f.puck[c, 2:] / np.linalg.norm(f.puck[c, 2:], axis=1)[:, None]
        velocity[c] = f.puck[c, 2:] * self.cfg.paddle_restitution / (1 + self.cfg.paddle_restitution)
        pos = puck - direction * (self.cfg.puck_radius + self.cfg.paddle_radius)
        low, high = self.decoder.low, self.decoder.high
        return np.clip(np.column_stack((2 * (pos - low) / (high - low) - 1,
                          velocity / self.decoder.config.velocity_scale,
                          2 * (duration - 0.04) / 0.21 - 1,
                          np.full(len(f.task), 0.5))), -1, 1)

    def rollout(self, f, actions, mode, *, record=False):
        n = len(f.task)
        if mode not in ("arrival", "position"):
            raise ValueError(mode)
        expected = (n, 6) if mode == "arrival" else (n, DECISIONS, 3)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"expected finite actions {expected}")
        actions = np.clip(actions, -1, 1)
        eng = TrialPhysics(n, self.cfg)
        for i, name in enumerate(("puck_x", "puck_y", "puck_vx", "puck_vy")):
            getattr(eng, name)[:] = f.puck[:, i]
        eng.paddle_agent_x[:] = f.paddle[:, 0]
        eng.paddle_agent_y[:] = f.paddle[:, 1]
        eng.paddle_opp_x[:] = eng.paddle_opp_y[:] = -10
        cart = CartState(n)
        cart.reset(f.paddle[:, 0] * 1000, f.paddle[:, 1] * 1000)
        previous_target = f.paddle.copy()
        previous_accel = np.full(n, self.accel)
        contacts = np.zeros(n, dtype=int)
        first_time = np.full(n, np.nan)
        contact_speed = np.zeros(n)
        first_normal = np.zeros((n, 2))
        first_incoming = np.zeros((n, 2))
        first_outgoing = np.zeros((n, 2))
        first_paddle_velocity = np.zeros((n, 2))
        arrival_position_error = np.full(n, np.nan)
        arrival_velocity_error = np.full(n, np.nan)
        if mode == "arrival":
            requested_p, requested_v, requested_t, _ = self.decoder.unpack(actions)
        effort = np.zeros(n)
        peak_accel = np.zeros(n)
        high_accel_time = np.zeros(n)
        travel = np.zeros(n)
        min_gap = np.full(n, 10.0)
        crossing = np.full(n, np.nan)
        goal = np.zeros(n, dtype=bool)
        recovery_speed = np.zeros(n)
        recovery_gap = np.zeros(n)
        commands = []
        traces = []
        contact_radius = self.cfg.puck_radius + self.cfg.paddle_radius

        def contact(event):
            if event["body"] != "agent":
                return
            idx = event["indices"]
            first = contacts[idx] == 0
            out = event["outgoing_before_speed_cap"]
            speed = np.linalg.norm(out, axis=1)
            out = out * np.minimum(1, self.cfg.max_puck_speed / np.maximum(speed, 1e-9))[:, None]
            dest = idx[first]
            first_time[dest] = event["time"][first]
            contact_speed[dest] = np.linalg.norm(out[first], axis=1)
            first_normal[dest] = event["normal"][first]
            first_incoming[dest] = event["incoming"][first]
            first_outgoing[dest] = out[first]
            first_paddle_velocity[dest] = event["paddle_velocity"][first]
            contacts[idx] += 1
        eng.contact_callback = contact
        # 320 ms action attempt, then 200 ms physical braking/recovery; puck
        # flight continues to 2 seconds to measure actual goal-line outcomes.
        for tick in range(100):
            t = tick * ACTION_DT
            if tick < DECISIONS:
                if mode == "arrival":
                    target, a_cap = self.decoder.decode(cart, actions, t, previous_target,
                        previous_accel, max_accel=self.accel, delay_s=self.delay)
                else:
                    a = actions[:, tick]
                    # Original full-half normalization, then firmware workspace clamp.
                    low = np.full(2, self.cfg.paddle_radius)
                    high = np.array([self.cfg.width, self.cfg.height / 2]) - low
                    target = np.clip(low + (a[:, :2] + 1) / 2 * (high - low),
                                     self.decoder.low, self.decoder.high)
                    a_cap = self.accel * (0.05 + 0.95 * ((a[:, 2] + 1) / 2) ** 2)
                low = np.full(2, self.cfg.paddle_radius)
                high = np.array([self.cfg.width, self.cfg.height / 2]) - low
                commands.append(np.column_stack((2 * (target - low) / (high - low) - 1,
                    2 * np.sqrt(np.clip((a_cap / self.accel - 0.05) / 0.95, 0, 1)) - 1)))
            elif tick == DECISIONS:
                # Identical recovery controller and cap for both interfaces.
                target = np.column_stack((cart.x, cart.y)) / 1000
                a_cap = np.full(n, min(20, self.accel))
            for sub in range(8):
                old_x, old_y = eng.puck_x.copy(), eng.puck_y.copy()
                old_v = np.column_stack((cart.vx, cart.vy)) / 1000
                old_p = np.column_stack((cart.x, cart.y)) / 1000
                deferred = sub * PHYSICS_DT < self.delay - 1e-9
                cmd = previous_target if deferred else target
                cap = previous_accel if deferred else a_cap
                # Same 12 profile substeps per 2.5 ms as BatchAirHockeyEnv.
                advance(cart, cmd[:, 0] * 1000, cmd[:, 1] * 1000, 12000,
                        cap * 1000, 0.003, PHYSICS_DT / 12, 12,
                        bounds=tuple(self.bounds * 1000))
                p = np.column_stack((cart.x, cart.y)) / 1000
                new_v = np.column_stack((cart.vx, cart.vy)) / 1000
                if mode == "arrival":
                    sample_t = t + (sub + 1) * PHYSICS_DT
                    due = np.isnan(arrival_position_error) & (sample_t >= requested_t)
                    u = np.clip((requested_t - sample_t + PHYSICS_DT) / PHYSICS_DT, 0, 1)[:, None]
                    arrival_position_error[due] = np.linalg.norm(
                        (old_p + u * (p - old_p) - requested_p)[due], axis=1)
                    arrival_velocity_error[due] = np.linalg.norm(
                        (old_v + u * (new_v - old_v) - requested_v)[due], axis=1)
                actual_a = np.linalg.norm(new_v - old_v, axis=1) / PHYSICS_DT
                # Measured acceleration, including any boundary/speed backstop.
                effort += actual_a**2 * PHYSICS_DT
                peak_accel = np.maximum(peak_accel, actual_a)
                high_accel_time += (actual_a > 40) * PHYSICS_DT
                travel += np.linalg.norm(p - old_p, axis=1)
                eng.update_paddle_agent(p[:, 0], p[:, 1], PHYSICS_DT)
                # Capture first top-plane crossing BEFORE wall resolution.
                # _move_puck is Euler after drag; calculate with same law.
                speed = np.hypot(eng.puck_vx, eng.puck_vy)
                factor = np.maximum(0, 1 - (eng.puck_friction * 9.81
                    + eng.drag_b * speed**2) * PHYSICS_DT / np.maximum(speed, 1e-8))
                next_y = old_y + eng.puck_vy * factor * PHYSICS_DT
                plane = self.cfg.height - self.cfg.puck_radius
                crossed = np.isnan(crossing) & (old_y < plane) & (next_y >= plane)
                frac = (plane - old_y) / np.maximum(next_y - old_y, 1e-9)
                crossing[crossed] = (old_x + eng.puck_vx * factor * PHYSICS_DT * frac)[crossed]
                eng.step(PHYSICS_DT)
                goal |= eng.goal_scored == 1
                if tick < DECISIONS:
                    gap = np.hypot(eng.puck_x - p[:, 0], eng.puck_y - p[:, 1]) - contact_radius
                    min_gap = np.minimum(min_gap, np.maximum(0, gap))
                elif tick < DECISIONS + 10:
                    recovery_speed = np.maximum(recovery_speed, np.hypot(eng.puck_vx, eng.puck_vy))
                    recovery_gap = np.maximum(recovery_gap,
                        np.hypot(eng.puck_x - p[:, 0], eng.puck_y - p[:, 1]) - contact_radius)
            previous_target, previous_accel = target.copy(), a_cap.copy()
            if tick == DECISIONS - 1:
                final_speed = np.hypot(eng.puck_vx, eng.puck_vy)
                final_gap = np.hypot(eng.puck_x - p[:, 0], eng.puck_y - p[:, 1]) - contact_radius
                attempt_contacts = contacts.copy()
            if record:
                traces.append(np.column_stack((eng.puck_x, eng.puck_y,
                    eng.puck_vx, eng.puck_vy, p, new_v)))
        touched = attempt_contacts > 0
        placement = np.where(np.isfinite(crossing), np.abs(crossing - f.aim), 1.0)
        cushion = touched & (recovery_speed < 0.6) & (recovery_gap < 0.06)
        # Bounded shaping never rewards a whiff more than a competent contact.
        shot_loss = (4 * ~touched + 4 * np.minimum(placement, 1)
                     + 0.3 * np.maximum(0, 2.5 - contact_speed) + 1.0 * ~goal)
        cushion_loss = (4 * ~touched + np.minimum(recovery_speed, 6)
                        + 4 * np.minimum(recovery_gap, 1))
        loss = np.where(f.task == 2, cushion_loss, shot_loss)
        loss += 5 * min_gap + self.effort_weight * effort / (60**2 * DECISIONS * ACTION_DT)
        # The firmware containment backstop can instantaneously remove outward
        # velocity. Do not let the optimizer exploit such a boundary stop as
        # free braking beyond the requested acceleration ceiling.
        loss += 50 * np.maximum(0, peak_accel / self.accel - 1.05)
        result = dict(loss=loss, contact=touched, goal=goal, placement_error=placement,
            crossing_x=crossing, cushion=cushion, puck_speed_after_contact=contact_speed,
            final_puck_speed=final_speed, final_gap=final_gap, min_gap=min_gap,
            recovery_peak_puck_speed=recovery_speed, recovery_max_gap=recovery_gap,
            effort=effort, peak_accel=peak_accel, high_accel_time=high_accel_time,
            backstop_violation=peak_accel > self.accel * 1.05,
            travel=travel, contact_time=first_time, contact_normal=first_normal,
            contact_incoming=first_incoming, contact_outgoing=first_outgoing,
            contact_paddle_velocity=first_paddle_velocity,
            arrival_position_error=arrival_position_error, arrival_velocity_error=arrival_velocity_error,
            commands=np.stack(commands, axis=1))
        if record:
            result["trace"] = np.stack(traces, axis=1)
        return result


def search(trials, fixtures, mode, seed, population=48, iterations=4, warm=None):
    """Equal rollout-budget CEM; incumbent is always retained.

    Position exploration mixes smooth and per-tick noise. Its initialization
    is the EXACT sequence emitted by the common analytic arrival seed, so the
    baseline starts with identical executed motion, not a weaker heuristic.
    """
    rng = np.random.default_rng(seed)
    n = len(fixtures.task)
    mean = trials.seed_arrival(fixtures) if warm is None else warm.copy()
    if mode == "position" and warm is None:
        mean = trials.rollout(fixtures, mean, "arrival")["commands"]
    shape = mean.shape[1:]
    incumbent = mean.copy()
    best = trials.rollout(fixtures, mean, mode)["loss"]
    scale = np.ones_like(mean)
    if mode == "position":
        full_span = np.array([trials.cfg.width, trials.cfg.height / 2]) - 2 * trials.cfg.paddle_radius
        scale[:, :, :2] = (trials.decoder.high - trials.decoder.low) / full_span
    std = 0.30 * scale
    repeated = fixtures.take(np.repeat(np.arange(n), population))
    history = []
    for iteration in range(iterations):
        start = time.perf_counter()
        noise = rng.normal(size=(n, population, *shape))
        if mode == "position":
            shared = rng.normal(size=(n, population, 1, 3))
            # Smooth local deviations, retaining unrestricted 50 Hz commands.
            noise = (np.roll(noise, 1, axis=2) + 2 * noise + np.roll(noise, -1, axis=2)) / 4
            noise = (0.5 * noise + 0.5 * shared) / np.sqrt(0.34375)
        candidates = np.clip(mean[:, None] + noise * std[:, None], -1, 1)
        candidates[:, 0] = incumbent
        scores = trials.rollout(repeated, candidates.reshape((-1, *shape)), mode)["loss"].reshape(n, population)
        elite_idx = np.argsort(scores, axis=1)[:, :max(4, population // 8)]
        elite = candidates[np.arange(n)[:, None], elite_idx]
        mean = elite.mean(axis=1)
        std = np.maximum(0.035 * scale, elite.std(axis=1))
        incumbent = elite[:, 0].copy()
        best = scores[np.arange(n), elite_idx[:, 0]]
        history.append({"iteration": iteration, "mean_loss": float(best.mean()),
                        "wall_seconds": time.perf_counter() - start})
        print(f"{mode} iteration {iteration + 1}: loss={best.mean():.3f}, "
              f"{history[-1]['wall_seconds']:.1f}s", flush=True)
    return incumbent, history
