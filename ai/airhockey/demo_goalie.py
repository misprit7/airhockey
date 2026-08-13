"""Hardcoded goalie. A DEMO, and explicitly a placeholder.

Isolated on purpose. Nothing else in the package should import this, and when
a learned policy lands this file gets deleted rather than refactored. The
puck tracking underneath it (vision/bin/puck_stream.py) is the part meant to
survive.

The model is as crude as it can be while still being a good goalie:

  * the puck travels in straight lines at constant speed
  * side walls are perfectly elastic and lossless
  * the paddle is a point that can be anywhere on its line instantly

None of that is true. Pucks decelerate on the air film, real bounces shed
energy and add spin, and the paddle has the acceleration limit we spent all
that time on. It works anyway because a goalie only has to be in the right
PLACE at the right TIME, and errors in the speed model move the arrival time
without moving the arrival point — which is the one quantity a straight-line
model gets right for free.

What it will visibly fail at, so you are not surprised:
  * a puck struck hard enough to bounce twice before arriving — each bounce
    multiplies any heading error
  * anything with real spin, which curves
  * a puck deflecting off the paddle back into play, where the estimator
    briefly fits a line through the corner
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402


@dataclass
class GoalieConfig:
    # Where the paddle waits and intercepts. The workspace stops well short
    # of the goal mouth, so this is as close to the goal as the cables allow
    # rather than anywhere tactical.
    defend_x: float = geom.WS_MAX_X - 15.0
    rest_y: float = (geom.WS_MIN_Y + geom.WS_MAX_Y) / 2.0

    # Puck radius: bounces reflect the CENTRE off a line one radius in from
    # the wall, not off the wall.
    puck_radius_mm: float = 31.5

    # Below this closing speed the puck is drifting, not shot. Chasing drift
    # makes the rig twitch continuously and wear itself out for nothing.
    min_closing_mm_s: float = 150.0

    # Ignore predictions further out than this. A puck 3 s away will be
    # re-predicted 600 times before it arrives, and the early ones are noise.
    max_horizon_s: float = 1.5

    # Hysteresis. Engage above the first, disengage below the second, so a
    # puck hovering at the threshold does not chatter the paddle between
    # intercept and rest.
    engage_mm_s: float = 250.0
    release_mm_s: float = 120.0

    # Do not command motion for sub-millimetre corrections; the profile will
    # happily chase centroid noise otherwise.
    deadband_mm: float = 3.0


def fold(value, lo, hi):
    """Reflect `value` into [lo, hi] as many times as needed.

    A puck bouncing between two walls traces a triangle wave in y, and this is
    that wave in closed form — no loop over bounces, so an arbitrarily fast
    puck costs the same as a slow one and there is no iteration limit to tune.
    """
    span = hi - lo
    if span <= 0:
        return lo
    u = (value - lo) % (2.0 * span)
    return lo + (u if u <= span else 2.0 * span - u)


def predict_crossing(x, y, vx, vy, target_x, y_lo, y_hi):
    """Where and when the puck crosses `target_x`, bouncing off the side walls.

    Returns (y, seconds) or None if it is not heading there.
    """
    if vx <= 0.0:
        return None
    dt = (target_x - x) / vx
    if dt < 0.0:
        return None
    return fold(y + vy * dt, y_lo, y_hi), dt


class Goalie:
    """Stateful because of the hysteresis; otherwise a pure function."""

    def __init__(self, cfg: GoalieConfig | None = None):
        self.cfg = cfg or GoalieConfig()
        self.engaged = False
        self.last_target = (self.cfg.defend_x, self.cfg.rest_y)
        self.last_eta = None

    def rest(self):
        return (self.cfg.defend_x, self.cfg.rest_y)

    def update(self, puck):
        """puck = (x, y, vx, vy) in mm and mm/s, or None if not visible.

        Returns (target_x, target_y). When there is nothing to defend against
        this is the rest position and it stops changing, which is what keeps
        the rig still instead of jittering on tracker noise.
        """
        c = self.cfg
        if puck is None:
            self.engaged = False
            self.last_eta = None
            self.last_target = self.rest()
            return self.last_target

        x, y, vx, vy = puck
        closing = vx      # +x is toward the robot

        if self.engaged:
            if closing < c.release_mm_s:
                self.engaged = False
        elif closing > c.engage_mm_s:
            self.engaged = True

        if not self.engaged or closing < c.min_closing_mm_s:
            self.last_eta = None
            self.last_target = self.rest()
            return self.last_target

        y_lo = c.puck_radius_mm
        y_hi = geom.GRID_Y_MM - c.puck_radius_mm
        hit = predict_crossing(x, y, vx, vy, c.defend_x, y_lo, y_hi)
        if hit is None or hit[1] > c.max_horizon_s:
            self.last_eta = None
            self.last_target = self.rest()
            return self.last_target

        y_hit, eta = hit
        self.last_eta = eta
        ty = min(max(y_hit, geom.WS_MIN_Y), geom.WS_MAX_Y)
        if abs(ty - self.last_target[1]) < c.deadband_mm:
            return self.last_target
        self.last_target = (c.defend_x, ty)
        return self.last_target


def _selftest():
    c = GoalieConfig()
    lo, hi = c.puck_radius_mm, geom.GRID_Y_MM - c.puck_radius_mm

    assert abs(fold(500.0, lo, hi) - 500.0) < 1e-9, "inside should pass through"
    # one bounce off the far wall
    over = hi + 100.0
    assert abs(fold(over, lo, hi) - (hi - 100.0)) < 1e-6, "single reflection"
    # two bounces returns toward the near wall
    assert lo - 1e-6 <= fold(hi + (hi - lo) + 50.0, lo, hi) <= hi + 1e-6

    # straight shot down the middle
    yh, dt = predict_crossing(200.0, 480.0, 4000.0, 0.0, c.defend_x, lo, hi)
    assert abs(yh - 480.0) < 1e-6 and dt > 0

    # angled shot that must bounce once
    yh, dt = predict_crossing(200.0, 480.0, 3000.0, 3000.0, c.defend_x, lo, hi)
    assert lo <= yh <= hi, f"prediction {yh} escaped the table"

    # receding puck is not a threat
    assert predict_crossing(1500.0, 480.0, -3000.0, 0.0, c.defend_x, lo, hi) is None

    g = Goalie(c)
    assert g.update(None) == g.rest(), "no puck -> rest"
    assert g.update((500.0, 300.0, 20.0, 0.0)) == g.rest(), "drift -> rest"
    t = g.update((500.0, 300.0, 4000.0, 0.0))
    assert t[0] == c.defend_x and abs(t[1] - 300.0) < 1e-6, t
    # hysteresis: still engaged just below the engage threshold
    t2 = g.update((900.0, 300.0, 200.0, 0.0))
    assert g.engaged, "should not disengage between engage and release"
    # and released well below
    g.update((900.0, 300.0, 50.0, 0.0))
    assert not g.engaged and g.last_target == g.rest()
    print("demo_goalie selftest PASSED")


if __name__ == "__main__":
    _selftest()
