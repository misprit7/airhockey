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
    # ROLLING term only, mu, MEASURED over two sessions. The quadratic model
    # is now actually USED (see _apply_friction) rather than stored beside a
    # constant that ignored it, so this is no longer the constant-model
    # equivalent -- it is the v-independent part alone.
    #
    # WEAKLY IDENTIFIED, deliberately kept anyway: the fit gives 14.5 +- 7.1
    # mm/s^2, consistent with zero at 2 sigma, because it is an intercept
    # extrapolated from glides that are mostly fast. It is small enough that
    # being wrong by its own error bar moves a 1 m/s puck by 15%% and a 6 m/s
    # puck by 0.6%%. To pin it down, record long SLOW glides (200-600 mm/s).
    puck_friction: float = 0.0015
    # MEASURED 2026-08-23 from 53 wall contacts, all four rails agreeing
    # (0.756 / 0.777 / 0.756 / 0.811) once goal-mouth events were excluded --
    # a puck arriving at the 380 mm goal does not bounce, and mixing those in
    # had dragged the end rails to ~0.45 with an impossible negative
    # tangential ratio.
    #
    # Also speed-dependent: -0.039 per m/s over the 0.7-7.3 m/s measured, so
    # ~0.79 for a drifting puck and ~0.53 for a hard shot. Not yet modelled.
    wall_restitution: float = 0.785
    # STILL UNMEASURED. Two recordings tried and neither can answer it: a
    # hand-SWUNG mallet is not a free body, and the arm keeps doing work
    # through the contact. The 2026-08-29 session came back with recoil at
    # 3.54x the puck's impact speed, which no free mass struck by a lighter
    # one can do, so the 0.507 it reported has the swing folded into it.
    #
    # 0.9 is a guess and is left as one on purpose, rather than replaced by a
    # measured-looking number that is really a measurement of somebody's arm.
    # To fix: shoot the puck at a mallet held still, or resting free.
    paddle_restitution: float = 0.9

    # Fraction of TANGENTIAL velocity surviving a rail bounce. 1.0 is a
    # frictionless rail and specular reflection, which is what the sim did
    # until now and is wrong by a third.
    #
    # Measured 0.678 (2026-08-23) and 0.645 (2026-08-29); per-rail on the
    # later session 0.603 / 0.646 / 0.726. This is where the tangential
    # momentum goes, and it does NOT come back as spin -- the puck needs no
    # orientation state, only this coefficient.
    wall_tangential: float = 0.66
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
    # MEASURED peak 9.7 m/s in the 2026-08-29 session, so 9.0 was clamping
    # below what a human actually produces -- and the clamp is invisible: the
    # puck just quietly never goes as fast as it does on the table. Raised
    # with headroom, since this is a safety rail against integrator blow-up,
    # not a physical fact.
    max_puck_speed: float = 12.0

    # Aerodynamic drag, decel = puck_friction*g + PUCK_DRAG_B * v^2, in SI.
    # Measured b per mm, times 1000 for metres.
    #
    # REPRODUCED ACROSS TWO SESSIONS with different puck marking (one dot,
    # then four) and the same validated fitter: 3.539e-5 +- 9.9e-7 on
    # 2026-08-23 and 3.433e-5 +- 7.5e-7 on 2026-08-29. Those agree inside one
    # sigma, which is better evidence than either alone -- the tracking method
    # changed between them and the answer did not. This is their mean.
    #
    # It is the dominant term everywhere a policy plays: at 6 m/s it is
    # 1250 mm/s^2 against 15 of rolling.
    PUCK_DRAG_B: float = 3.48e-2


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
