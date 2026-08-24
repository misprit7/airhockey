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

# The scalar PhysicsEngine that used to live here is gone (2026-08-23). It was
# a second implementation of rules that BatchPhysicsEngine already has, kept
# honest by ~1000 lines of parity tests -- which is a bug waiting for the day
# someone fixes a collision in one of them and not the other. Those tests
# passing is what licensed the removal.
#
# airhockey/scalar_engine.py presents this interface over
# BatchPhysicsEngine(n_envs=1), so AirHockeyEnv is unchanged. The two ENV
# wrappers stay separate on purpose: the batch one takes dynamics as a string
# and returns arrays for training, while AirHockeyEnv's dynamics object is
# hot-swapped at runtime for HardwareDynamics to drive the real robot.
