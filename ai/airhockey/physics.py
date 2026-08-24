"""Core 2D air hockey physics engine.

All units are SI: meters, seconds, kg, m/s.
Origin is bottom-left of the table. Y-axis points up (toward opponent's side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as _geom  # noqa: E402


@dataclass
class TableConfig:
    """Physical table dimensions and properties."""

    width: float = 1.0  # meters
    height: float = 2.0  # meters
    # MEASURED 81.4 mm diameter; from the canonical geometry, not
    # restated. Sim units are metres and the sim table is ~1 m wide,
    # so mm/1000 is the right conversion here.
    puck_radius: float = _geom.PUCK_RADIUS_MM / 1000.0
    paddle_radius: float = 0.04  # 80mm diameter paddle
    puck_mass: float = 0.015  # 15g puck
    paddle_mass: float = 0.17  # 170g paddle
    # MEASURED 2026-08-23 from 78 free glides (vision/bin/fit_puck.py).
    #
    # This is the CONSTANT-model equivalent and is not a good model. The real
    # deceleration is overwhelmingly AERODYNAMIC: only 22 mm/s^2 of rolling
    # (mu 0.0022, which is what an air cushion should give) plus 3.71e-5 * v^2
    # of drag. At 1 m/s that is 59 mm/s^2 and at 6 m/s it is 1358 -- a factor
    # of 23 across the range a policy will see. A single coefficient cannot
    # span that, and a struck puck is the regime that matters. See
    # PUCK_DRAG_B; the quadratic halves the residual (215 vs 350 rms).
    puck_friction: float = 0.0162
    # MEASURED 2026-08-23 from 53 wall contacts, all four rails agreeing
    # (0.756 / 0.777 / 0.756 / 0.811) once goal-mouth events were excluded --
    # a puck arriving at the 380 mm goal does not bounce, and mixing those in
    # had dragged the end rails to ~0.45 with an impossible negative
    # tangential ratio.
    #
    # Also speed-dependent: -0.039 per m/s over the 0.7-7.3 m/s measured, so
    # ~0.79 for a drifting puck and ~0.53 for a hard shot. Not yet modelled.
    wall_restitution: float = 0.785
    paddle_restitution: float = 0.9  # energy retained on paddle hit
    # MEASURED 380 mm, centred on each end rail. Derived from the canonical
    # geometry rather than restated, since it is a fact about the table. The
    # 0.25 that was here predates the table existing and was 34% narrow --
    # which changes how often a shot goes in, and so what the policy learns
    # a shot is worth.
    #
    # Scaled by the real table's width so the fraction of the end rail that
    # is goal matches, which is what actually governs whether a shot scores.
    goal_width: float = (_geom.GOAL_WIDTH_MM
                         / (_geom.RAIL_MAX_Y - _geom.RAIL_MIN_Y))
    max_puck_speed: float = 9.0  # m/s; measured glides reached 8.7

    # Aerodynamic drag, decel = puck_friction*g + PUCK_DRAG_B * v^2, in SI.
    # Measured b = 1.853e-5 per mm; * 1000 for metres.
    PUCK_DRAG_B: float = 3.710e-2


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
