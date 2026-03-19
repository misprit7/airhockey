"""Core 2D air hockey physics engine.

All units are SI: meters, seconds, kg, m/s.
Origin is bottom-left of the table. Y-axis points up (toward opponent's side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class TableConfig:
    """Physical table dimensions and properties."""

    width: float = 1.0  # meters
    height: float = 2.0  # meters
    puck_radius: float = 0.025  # 50mm diameter puck
    paddle_radius: float = 0.04  # 80mm diameter paddle
    puck_mass: float = 0.015  # 15g puck
    paddle_mass: float = 0.17  # 170g paddle
    puck_friction: float = 0.01  # kinetic friction coefficient on air cushion
    wall_restitution: float = 0.85  # energy retained on wall bounce
    paddle_restitution: float = 0.9  # energy retained on paddle hit
    goal_width: float = 0.25  # goal opening width (centered on x-axis)
    max_puck_speed: float = 5.0  # m/s, clamp for stability


@dataclass
class PuckState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class PaddleState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


@dataclass
class PhysicsState:
    puck: PuckState = field(default_factory=PuckState)
    paddle_agent: PaddleState = field(default_factory=PaddleState)
    paddle_opponent: PaddleState = field(default_factory=PaddleState)
    score_agent: int = 0
    score_opponent: int = 0
    goal_scored: int = 0  # 0=none, 1=agent scored, -1=opponent scored
    time: float = 0.0


class PhysicsEngine:
    """Simulates 2D air hockey physics."""

    def __init__(self, config: TableConfig | None = None):
        self.config = config or TableConfig()
        self.state = PhysicsState()

    def reset(
        self,
        rng: np.random.Generator | None = None,
    ) -> PhysicsState:
        """Reset to initial state with puck at center."""
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng()

        self.state = PhysicsState(
            puck=PuckState(
                x=cfg.width / 2,
                y=cfg.height / 2,
                vx=rng.uniform(-0.5, 0.5),
                vy=rng.uniform(-1.0, 1.0),
            ),
            paddle_agent=PaddleState(
                x=cfg.width / 2,
                y=cfg.height * 0.15,
            ),
            paddle_opponent=PaddleState(
                x=cfg.width / 2,
                y=cfg.height * 0.85,
            ),
        )
        return self.state

    def step(self, dt: float) -> PhysicsState:
        """Advance physics by dt seconds. Call after updating paddle positions."""
        self.state.goal_scored = 0
        self._apply_friction(dt)
        self._move_puck(dt)
        self._collide_walls()
        self._collide_paddle(self.state.paddle_agent)
        self._collide_paddle(self.state.paddle_opponent)
        self._clamp_puck_speed()
        self._check_goals()
        self.state.time += dt
        return self.state

    def update_paddle(
        self,
        paddle: PaddleState,
        new_x: float,
        new_y: float,
        dt: float,
    ) -> None:
        """Update paddle position and compute velocity from displacement."""
        old_x, old_y = paddle.x, paddle.y
        paddle.x = new_x
        paddle.y = new_y
        if dt > 0:
            paddle.vx = (new_x - old_x) / dt
            paddle.vy = (new_y - old_y) / dt

    def _apply_friction(self, dt: float) -> None:
        puck = self.state.puck
        speed = puck.speed()
        if speed > 1e-6:
            # Simple friction: deceleration proportional to friction coeff * g
            friction_decel = self.config.puck_friction * 9.81
            new_speed = max(0.0, speed - friction_decel * dt)
            factor = new_speed / speed
            puck.vx *= factor
            puck.vy *= factor

    def _move_puck(self, dt: float) -> None:
        self.state.puck.x += self.state.puck.vx * dt
        self.state.puck.y += self.state.puck.vy * dt

    def _collide_walls(self) -> None:
        cfg = self.config
        puck = self.state.puck
        r = cfg.puck_radius
        e = cfg.wall_restitution

        # Left wall
        if puck.x - r < 0:
            puck.x = r
            puck.vx = abs(puck.vx) * e

        # Right wall
        if puck.x + r > cfg.width:
            puck.x = cfg.width - r
            puck.vx = -abs(puck.vx) * e

        # Bottom wall (agent's side) — check for goal opening
        if puck.y - r < 0:
            goal_left = (cfg.width - cfg.goal_width) / 2
            goal_right = (cfg.width + cfg.goal_width) / 2
            if goal_left < puck.x < goal_right:
                pass  # In goal — handled by _check_goals
            else:
                puck.y = r
                puck.vy = abs(puck.vy) * e

        # Top wall (opponent's side) — check for goal opening
        if puck.y + r > cfg.height:
            goal_left = (cfg.width - cfg.goal_width) / 2
            goal_right = (cfg.width + cfg.goal_width) / 2
            if goal_left < puck.x < goal_right:
                pass  # In goal
            else:
                puck.y = cfg.height - r
                puck.vy = -abs(puck.vy) * e

    def _collide_paddle(self, paddle: PaddleState) -> None:
        """Elastic-ish collision between puck and paddle."""
        puck = self.state.puck
        cfg = self.config

        dx = puck.x - paddle.x
        dy = puck.y - paddle.y
        dist = np.hypot(dx, dy)
        min_dist = cfg.puck_radius + cfg.paddle_radius

        if dist >= min_dist or dist < 1e-8:
            return

        # Normal vector from paddle to puck
        nx = dx / dist
        ny = dy / dist

        # Separate overlapping objects
        overlap = min_dist - dist
        puck.x += nx * overlap
        puck.y += ny * overlap

        # Relative velocity of puck w.r.t. paddle
        rel_vx = puck.vx - paddle.vx
        rel_vy = puck.vy - paddle.vy
        rel_v_normal = rel_vx * nx + rel_vy * ny

        # Only collide if approaching
        if rel_v_normal >= 0:
            return

        # Impulse (puck is much lighter than paddle, treat paddle as infinite mass)
        e = cfg.paddle_restitution
        impulse = -(1 + e) * rel_v_normal

        puck.vx += impulse * nx
        puck.vy += impulse * ny

    def _clamp_puck_speed(self) -> None:
        puck = self.state.puck
        speed = puck.speed()
        if speed > self.config.max_puck_speed:
            factor = self.config.max_puck_speed / speed
            puck.vx *= factor
            puck.vy *= factor

    def _check_goals(self) -> None:
        cfg = self.config
        puck = self.state.puck
        r = cfg.puck_radius

        # Opponent scored (puck past agent's baseline)
        if puck.y - r < -r:
            self.state.score_opponent += 1
            self.state.goal_scored = -1
            self._reset_puck_after_goal(toward_agent=True)

        # Agent scored (puck past opponent's baseline)
        elif puck.y + r > cfg.height + r:
            self.state.score_agent += 1
            self.state.goal_scored = 1
            self._reset_puck_after_goal(toward_agent=False)

    def _reset_puck_after_goal(self, toward_agent: bool) -> None:
        """Reset puck to center, moving toward the side that was scored on."""
        cfg = self.config
        self.state.puck.x = cfg.width / 2
        self.state.puck.y = cfg.height / 2
        self.state.puck.vx = 0.0
        self.state.puck.vy = -0.5 if toward_agent else 0.5
