"""Who picks the shot route: the request governor and the route it reads back.

The neural player executes a REQUESTED route (one-hot left bank / right bank
/ straight; all-zero = no request, the network's own choice). Two small
pieces, shared by the sim evaluators and the table runner, NumPy only:

`ShotSelector` returns the request for each possession (the puck entering
the robot's half). Modes:

    random    a uniform explicit request (training's draw; the table's "mix")
    free      never a request
    left / right / straight   always that one
    auto      free unless variety is at risk. A sliding window of the last
              `window` completed robot shots, by REALISED route. A route
              under `floor` of the window is requested explicitly (the most
              under-used first); else, a route over `ceiling` gets one of
              the other two requested (the less used); else free. Before
              the window holds `min_shots`, free. In every state an
              explicit uniform request is drawn with probability `explore`,
              so no route's requested skill goes unused.

What auto can and cannot promise: it controls the REQUEST, the network
shoots. Against a network that honours explicit requests no route stays
under the floor and none above the ceiling for long; against one that
ignores them it requests the missing route every possession and
`stats()` shows it (honoured share).

`RouteClassifier` reads the realised route of the robot's shots from what
the policy itself is given every 20 ms tick -- the tracker's puck fix and
30 ms slope velocity and the robot's own mallet, in the robot-local sim
frame (metres; x = 0 is the LEFT rail facing the far goal, y = 0 the
robot's goal line) -- so the same code runs on sim observations, on the
table and on a session's ticks.csv. The definitions are the reward's
(`rewards.shot_matches_type`): straight = no side rail on the way to the
far goal line, left / right bank = the FIRST side rail touched. A side-rail
bounce seen before the centre line decides it; otherwise the puck's state
as it crosses the line is traced on through `rewards.predict_shot` (the
measured lossy rail). The person cannot reach across the line, so nothing
after their touch enters. Undetermined (never counted): the puck is lost
for more than `max_lost_s` in flight, stops or turns back before the line,
or would never reach the far end.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from airhockey.rewards import predict_shot

NONE, LEFT, RIGHT, STRAIGHT = 0, 1, 2, 3
ROUTE_NAMES = ("none", "left", "right", "straight")
MODES = ("random", "free", "auto", "left", "right", "straight")
DEFAULTS = dict(window=30, floor=0.15, ceiling=0.60, min_shots=9, explore=0.15)


class ShotSelector:
    def __init__(self, mode="auto", *, window=DEFAULTS["window"], floor=DEFAULTS["floor"],
                 ceiling=DEFAULTS["ceiling"], min_shots=DEFAULTS["min_shots"],
                 explore=DEFAULTS["explore"], count_requested_when_unclassified=False, seed=None):
        if mode not in MODES:
            raise ValueError(f"shot selector mode must be one of {MODES}, not {mode!r}")
        if int(window) != window or window < 3 or int(min_shots) != min_shots or not 1 <= min_shots <= window:
            raise ValueError("window must be a whole number >= 3 and min_shots in [1, window]")
        if not 0 <= floor < 1 / 3:
            raise ValueError("floor must be in [0, 1/3): three routes cannot all exceed a third")
        if not 1 / 3 < ceiling <= 1:
            raise ValueError("ceiling must be in (1/3, 1]")
        if not 0 <= explore <= 1:
            raise ValueError("explore must be a probability")
        self.mode = mode
        self.window, self.floor, self.ceiling = int(window), float(floor), float(ceiling)
        self.min_shots, self.explore = int(min_shots), float(explore)
        self.count_requested_when_unclassified = bool(count_requested_when_unclassified)
        self.rng = np.random.default_rng(seed)
        self.recent = deque(maxlen=self.window)
        self.current = NONE
        self.reason = "start"
        # requests by route; shots by [request, realised route (0 = undetermined)]
        self.requests = np.zeros(4, int)
        self.reasons = dict(free=0, explore=0, floor=0, ceiling=0, fixed=0, random=0)
        self.shots = np.zeros((4, 4), int)

    def shares(self):
        """Share of each route [left, right, straight] in the window (zeros when empty)."""
        counts = np.bincount(np.asarray(self.recent, int), minlength=4)[1:]
        return counts / max(len(self.recent), 1)

    def _choose(self):
        if self.mode == "free":
            return NONE, "free"
        if self.mode == "random":
            return int(self.rng.integers(1, 4)), "random"
        if self.mode != "auto":
            return ROUTE_NAMES.index(self.mode), "fixed"
        if self.rng.random() < self.explore:
            return int(self.rng.integers(1, 4)), "explore"
        if len(self.recent) < self.min_shots:
            return NONE, "free"
        share = self.shares()
        # Ties broken at random, so two starved routes are served alternately.
        order = np.lexsort((self.rng.random(3), share))
        if share[order[0]] < self.floor:
            return int(order[0]) + 1, "floor"
        if share[order[2]] > self.ceiling:
            return int(order[0]) + 1, "ceiling"
        return NONE, "free"

    def request(self):
        """The request for a new possession: 0 none, 1 left, 2 right, 3 straight."""
        self.current, self.reason = self._choose()
        self.requests[self.current] += 1
        self.reasons[self.reason] += 1
        return self.current

    def record(self, route, requested=None):
        """A completed robot shot. `route` 1..3 as realised, 0 / None when it
        could not be determined; `requested` is the request it was made under
        (default: the current one)."""
        requested = self.current if requested is None else int(requested)
        route = int(route or 0)
        if not 0 <= route <= 3 or not 0 <= requested <= 3:
            raise ValueError("routes and requests are 0..3")
        self.shots[requested, route] += 1
        if route == NONE and self.count_requested_when_unclassified:
            route = requested
        if route != NONE:
            self.recent.append(route)

    def stats(self):
        classified = self.shots[:, 1:]
        explicit = classified[1:]
        free = classified[0]
        return dict(
            mode=self.mode,
            possessions=int(self.requests.sum()),
            requests=dict(zip(ROUTE_NAMES, self.requests.tolist())),
            free_share=float(self.requests[0] / max(self.requests.sum(), 1)),
            reasons=dict(self.reasons),
            shots=int(self.shots.sum()), undetermined=int(self.shots[:, 0].sum()),
            realised=dict(zip(ROUTE_NAMES[1:], classified.sum(0).tolist())),
            realised_when_free=dict(zip(ROUTE_NAMES[1:], free.tolist())),
            honoured_share=(float(np.trace(explicit) / explicit.sum()) if explicit.sum() else None),
            window=dict(zip(ROUTE_NAMES[1:], self.shares().round(4).tolist()), n=len(self.recent)))


class RouteClassifier:
    """Realised route of the robot's shots from the policy's own inputs.

    Call `update` once per control tick; it returns a dict when a shot has
    just been resolved: `route` (1..3, or 0 = undetermined), `launch_t`,
    `t`, `speed` (m/s at the decision), `how` ("bounce" seen before the
    line / "traced" from the line / why it was dropped).
    """

    def __init__(self, *, width=1.0, height=2.0, puck_radius=0.0407, paddle_radius=0.0504,
                 min_shot_vy=1.0, contact_margin=0.07, contact_memory_s=0.10,
                 rail_margin=0.12, max_lost_s=0.20, max_flight_s=1.5):
        self.width, self.height, self.half = float(width), float(height), float(height) / 2
        self.puck_radius = float(puck_radius)
        self.touch = float(puck_radius + paddle_radius + contact_margin)
        self.min_shot_vy = float(min_shot_vy)
        self.contact_memory_s = float(contact_memory_s)
        self.rail_margin = float(rail_margin)
        self.max_lost_s, self.max_flight_s = float(max_lost_s), float(max_flight_s)
        self.reset()

    def reset(self):
        self.prev = None            # (t, x, y, vx, vy) of the last tick
        self.contact_t = -np.inf    # mallet last close to and not ahead of the puck
        self.flight = None

    def _done(self, t, route, how, speed=0.0):
        launch = self.flight["t"]
        self.flight = None
        return dict(route=int(route), launch_t=launch, t=t, speed=float(speed), how=how)

    def update(self, t, puck, velocity, mallet):
        """`puck` (x, y) and `velocity` (vx, vy) as the policy sees them
        (the last real fix; zero velocity while the tracker has none),
        `mallet` (x, y), all in robot-local sim metres."""
        x, y = float(puck[0]), float(puck[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        prev, self.prev = self.prev, (t, x, y, vx, vy)
        if prev is None or not 0 < t - prev[0] <= 0.1:
            # A resume: nothing carries across it.
            dropped = self._done(t, NONE, "resync") if self.flight is not None else None
            self.contact_t = -np.inf
            return dropped
        near = np.hypot(x - mallet[0], y - mallet[1]) <= self.touch and y - mallet[1] > -0.02
        if near:
            self.contact_t = t
        frozen = vx == 0.0 and vy == 0.0 and x == prev[1] and y == prev[2]
        result = None
        struck = (not frozen and y < self.half and vy > self.min_shot_vy and vy - prev[4] > 0.5
                  and t - self.contact_t <= self.contact_memory_s)
        if struck:
            # A second touch of the same shot restarts it: the last one decides.
            self.flight = dict(t=t if self.flight is None else self.flight["t"], seen=t, vx=vx, lost=0.0)
            return None
        f = self.flight
        if f is None:
            return None
        if frozen:
            f["lost"] += t - prev[0]
            if f["lost"] > self.max_lost_s:
                result = self._done(t, NONE, "lost")
            return result
        if t - f["t"] > self.max_flight_s:
            return self._done(t, NONE, "timeout")
        if vy < 0.3:
            return self._done(t, NONE, "stopped")
        speed = np.hypot(vx, vy)
        # A side-rail bounce: the lateral velocity reverses next to a rail.
        if f["vx"] * vx < 0 and min(abs(f["vx"]), abs(vx)) > 0.2:
            if x < self.puck_radius + self.rail_margin and vx > 0:
                return self._done(t, LEFT, "bounce", speed)
            if x > self.width - self.puck_radius - self.rail_margin and vx < 0:
                return self._done(t, RIGHT, "bounce", speed)
        if abs(vx) > 0.2:
            f["vx"] = vx
        f["seen"] = t
        if y >= self.half:
            x_goal, _, bounces, first = predict_shot(x, y, vx, vy, width=self.width, height=self.height,
                                                     puck_radius=self.puck_radius)
            if not np.isfinite(x_goal):
                return self._done(t, NONE, "never_arrives", speed)
            route = STRAIGHT if int(bounces) == 0 else (LEFT if int(first) < 0 else RIGHT)
            return self._done(t, route, "traced", speed)
        return None


def robot_local(puck_mm, velocity_mm_s, mallet_mm):
    """Table-mm reports -> the classifier's frame (for logs; the runner
    already has the encoder's observation)."""
    from airhockey.deploy import mm_velocity_to_sim
    from airhockey.dynamics import table_mm_to_sim
    return (table_mm_to_sim(*puck_mm), mm_velocity_to_sim(*velocity_mm_s, 1.0, 1.0),
            table_mm_to_sim(*mallet_mm))
