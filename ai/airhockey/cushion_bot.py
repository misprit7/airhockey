"""A scripted controller that STOPS the puck, holds it, then shoots.

THE POINT
    Four self-play runs paid for controlling the puck and none learned to,
    because the chain -- meet the puck moving with it, absorb it, hold,
    shoot -- is long and its first link (a retreating touch) looks like a
    mistake from where the policy stands. TD-MPC2 learns from a replay
    buffer, so the chain can simply be put in the buffer: this bot plays
    a share of the training envs and its episodes are stored like any
    other, rewards and all. The value function then knows what a
    controlled possession is worth, and the planner goes and gets it.

HOW IT STOPS A PUCK
    The physics (batch_physics._collide_paddle) applies an impulse on the
    relative normal velocity with restitution e = 0.9, so a puck meeting a
    paddle that is retreating along the puck's direction at e/(1+e) = 0.47
    of the puck's speed leaves with ~zero speed. The bot waits on the
    puck's path with room behind it, and when the puck is about to arrive
    commands a target behind itself so the profile body is already moving
    away at contact. It then keeps the paddle on the puck for a while and
    strikes through it at the far goal.

    Vectorised over the batch env's envs; reads the engine's TRUE state
    (a demonstration may cheat -- the observation stored in the buffer is
    still the env's noisy camera view) and emits actions in the env's own
    action space (position + accel fraction, or position only).
"""
from __future__ import annotations

import numpy as np

from airhockey.batch_env import BatchAirHockeyEnv

WAIT, INTERCEPT, CUSHION, HOLD, SHOOT, BLOCK, WINDUP = 0, 1, 2, 3, 4, 5, 6
PHASE_NAMES = ("wait", "intercept", "cushion", "hold", "shoot", "block", "windup")

# Geometry / tuning, sim metres and seconds.
STATION_ABOVE_FLOOR = 0.25   # wait this far above the box floor: room to retreat
CUSHION_TRIGGER = 0.10       # start retreating when the gap to contact is under this
CUSHION_LEAD_S = 0.08        # ...or when contact is under this many seconds away
RETREAT_M = 0.12             # target this far behind the paddle along the puck's motion
CUSHION_MAX_S = 0.30
HOLD_S = 0.9                 # inside the shaper's paid second (the shot clock)
HOLD_ESCAPE_SPEED = 1.0      # the puck got away: intercept again
SHOOT_S = 0.35
SHOOT_THROUGH = 0.30         # target this far beyond the puck toward the goal
# A real shot needs a WIND-UP: pull back along the shot line, then strike
# through the puck at full accel. Pushing through from contact (the first
# version) left the puck at the paddle's peak speed, ~3 m/s at 40 m/s^2;
# a paddle arriving at v hits a resting puck away at 1.9 v (kinematic
# paddle, e = 0.9), so 0.15 m of run-up gives ~4.5 m/s.
WINDUP_M = 0.15
WINDUP_S = 0.25
APPROACH_SPEED = 0.4         # a puck slower than this on our half is walked up to
# A puck faster than this cannot be absorbed by a 40 m/s^2 body in time,
# and one heading into the goal must not be retreated from at all: BLOCK
# it (sit on its path, no retreat) and pick up the rebound. The first
# version cushioned everything and lost to the sniper 23-61, a
# demonstration whose net value the learner rightly refused.
CUSHION_MAX_SPEED = 3.5
BLOCK_Y_ABOVE_FLOOR = 0.10
BLOCK_S = 0.30


class CushionBot:
    def __init__(self, env: BatchAirHockeyEnv, rng: np.random.Generator | None = None):
        self.env = env
        self.n = env.n_envs
        self.rng = rng or np.random.default_rng()
        self.phase = np.full(self.n, WAIT, dtype=np.int8)
        self.t_phase = np.zeros(self.n)
        self.aim_x = np.full(self.n, env.table_config.width / 2.0)
        cfg = env.table_config
        self.contact = cfg.puck_radius + cfg.paddle_radius
        self.ws = env._ws
        self.H = cfg.height
        self.W = cfg.width
        self.stats = {"cushions": 0, "holds": 0, "shots": 0, "blocks": 0, "windups": 0}

    def reset(self, mask=None) -> None:
        idx = slice(None) if mask is None else mask
        self.phase[idx] = WAIT
        self.t_phase[idx] = 0.0

    # ── helpers ──────────────────────────────────────────────────────

    def _to_action(self, tx, ty, accel_slot) -> np.ndarray:
        e = self.env
        ax = (np.clip(tx, e._action_low[0], e._action_high[0]) - e._action_low[0]) \
            / (e._action_high[0] - e._action_low[0]) * 2.0 - 1.0
        ay = (np.clip(ty, e._action_low[1], e._action_high[1]) - e._action_low[1]) \
            / (e._action_high[1] - e._action_low[1]) * 2.0 - 1.0
        if e.action_dim >= 3:
            return np.column_stack([ax, ay, np.full(self.n, accel_slot) if np.isscalar(accel_slot)
                                    else accel_slot]).astype(np.float32)
        return np.column_stack([ax, ay]).astype(np.float32)

    def _x_at(self, px, py, vx, vy, y_line):
        """Where the puck's straight path (folded on the side rails) crosses y_line."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(vy) > 1e-6, (y_line - py) / vy, 0.0)
        t = np.clip(t, 0.0, 2.0)
        x = px + vx * t
        r = self.env.table_config.puck_radius
        lo, hi = r, self.W - r
        span = hi - lo
        # fold into [lo, hi] as a mirror would
        u = np.mod(x - lo, 2 * span)
        u = np.where(u > span, 2 * span - u, u)
        return lo + u

    # ── the policy ───────────────────────────────────────────────────

    def act(self) -> np.ndarray:
        e = self.env
        eng = e.engine
        dt = e.action_dt
        px, py, vx, vy = eng.puck_x, eng.puck_y, eng.puck_vx, eng.puck_vy
        mx, my = eng.paddle_agent_x, eng.paddle_agent_y
        speed = np.hypot(vx, vy)
        gap = np.hypot(px - mx, py - my) - self.contact
        ws = self.ws
        floor_y = ws["min_y"]
        station_y = min(floor_y + STATION_ABOVE_FLOOR, ws["max_y"] - 0.05)
        on_half = py < self.H / 2.0
        coming = (vy < -0.3) & (py < self.H * 0.75)
        closing = (px - mx) * vx + (py - my) * vy < 0.0
        # Where the puck's path meets our goal line, and whether that is in
        # the mouth: a goal-bound fast puck is blocked, never cushioned.
        x_goal = self._x_at(px, py, vx, vy, 0.0)
        mouth = e.table_config.goal_width / 2.0 + e.table_config.puck_radius
        dangerous = coming & (speed > CUSHION_MAX_SPEED) & (np.abs(x_goal - self.W / 2.0) < mouth)

        self.t_phase += dt
        ph = self.phase
        tx = np.empty(self.n)
        ty = np.empty(self.n)
        acc = np.full(self.n, -0.6)

        # ── transitions ──
        # WAIT -> INTERCEPT when the puck is on our half or coming.
        go = (ph == WAIT) & (on_half | coming)
        ph[go] = INTERCEPT
        self.t_phase[go] = 0.0
        # INTERCEPT -> BLOCK for a dangerous puck (fast, goal-bound).
        bl = (ph == INTERCEPT) & dangerous
        ph[bl] = BLOCK
        self.t_phase[bl] = 0.0
        self.stats["blocks"] += int(bl.sum())
        # BLOCK -> INTERCEPT once it is no longer dangerous (rebound, or
        # passed) and the block has lasted its minimum.
        unbl = (ph == BLOCK) & ~dangerous & (self.t_phase > BLOCK_S)
        ph[unbl] = INTERCEPT
        self.t_phase[unbl] = 0.0
        # INTERCEPT -> CUSHION when about to be hit by a puck that CAN be
        # absorbed: by TIME to contact, so a fast puck gets the same lead
        # as a slow one (the body needs ~35 ms at 40 m/s^2 to reach 1.4 m/s).
        ttc = gap / np.maximum(speed, 1e-6)
        cu = ((ph == INTERCEPT) & ((gap < CUSHION_TRIGGER) | (ttc < CUSHION_LEAD_S)) & closing
              & (speed > 0.6) & (speed <= CUSHION_MAX_SPEED) & ~dangerous)
        ph[cu] = CUSHION
        self.t_phase[cu] = 0.0
        self.stats["cushions"] += int(cu.sum())
        # INTERCEPT -> HOLD when the puck is already slow next to us.
        slow_near = (ph == INTERCEPT) & (gap < 0.06) & (speed <= 0.6) & on_half
        ph[slow_near] = HOLD
        self.t_phase[slow_near] = 0.0
        # CUSHION -> HOLD once the puck is slow, or times out.
        done_cu = (ph == CUSHION) & ((speed < 0.6) | (self.t_phase > CUSHION_MAX_S) | ~closing & (gap > 0.05))
        to_hold = done_cu & (speed < HOLD_ESCAPE_SPEED)
        ph[to_hold] = HOLD
        self.t_phase[to_hold] = 0.0
        self.stats["holds"] += int(to_hold.sum())
        back = done_cu & ~to_hold
        ph[back] = INTERCEPT
        self.t_phase[back] = 0.0
        # HOLD -> WINDUP after HOLD_S; HOLD -> INTERCEPT if it got away.
        escaped = (ph == HOLD) & ((speed > HOLD_ESCAPE_SPEED) | (gap > 0.15))
        ph[escaped] = INTERCEPT
        self.t_phase[escaped] = 0.0
        windup = (ph == HOLD) & (self.t_phase >= HOLD_S)
        ph[windup] = WINDUP
        self.t_phase[windup] = 0.0
        self.stats["windups"] += int(windup.sum())
        if np.any(windup):
            # aim somewhere in the far mouth
            mouth = e.table_config.goal_width / 2.0 - e.table_config.puck_radius
            self.aim_x[windup] = self.W / 2.0 + self.rng.uniform(-mouth, mouth, size=int(windup.sum()))
        # WINDUP -> SHOOT once the paddle is back on the shot line behind
        # the puck (or the wind-up times out); the puck drifting away
        # (> 0.3 m) means intercept again.
        dxa = self.aim_x - px
        dya = self.H - py
        nna = np.maximum(np.hypot(dxa, dya), 1e-6)
        uxa, uya = dxa / nna, dya / nna
        back_x = px - uxa * (self.contact + WINDUP_M)
        back_y = py - uya * (self.contact + WINDUP_M)
        at_back = np.hypot(mx - back_x, my - back_y) < 0.03
        strike = (ph == WINDUP) & (at_back | (self.t_phase >= WINDUP_S))
        ph[strike] = SHOOT
        self.t_phase[strike] = 0.0
        self.stats["shots"] += int(strike.sum())
        lost = (ph == WINDUP) & (gap > 0.3)
        ph[lost] = INTERCEPT
        self.t_phase[lost] = 0.0
        # SHOOT -> WAIT after SHOOT_S, or once the puck has left the half.
        done_sh = (ph == SHOOT) & ((self.t_phase > SHOOT_S) | ~on_half)
        ph[done_sh] = WAIT
        self.t_phase[done_sh] = 0.0
        # Anything -> WAIT when the puck is gone to the far half and not coming.
        gone = (ph != WAIT) & ~on_half & ~coming
        ph[gone] = WAIT
        self.t_phase[gone] = 0.0

        # ── targets per phase ──
        w = ph == WAIT
        tx[w] = np.clip(px[w], ws["min_x"], ws["max_x"])
        ty[w] = station_y
        acc[w] = -0.6

        i = ph == INTERCEPT
        # Meet the puck on its path at our station height (room to retreat
        # behind us); a slow puck is simply walked up to.
        x_meet = self._x_at(px, py, vx, vy, station_y)
        slow = speed < APPROACH_SPEED
        tx[i] = np.where(slow, px, x_meet)[i]
        ty[i] = np.where(slow, py - 0.02, np.minimum(station_y, py - 0.05))[i]
        acc[i] = 1.0

        c = ph == CUSHION
        # Retreat along the puck's direction of travel: the body is then
        # moving with the puck when it arrives.
        n = np.maximum(speed, 1e-6)
        ux, uy = vx / n, vy / n
        tx[c] = (mx + ux * RETREAT_M)[c]
        ty[c] = (my + uy * RETREAT_M)[c]
        acc[c] = 1.0

        h = ph == HOLD
        # Rest against the puck, not in it: the paddle's centre a contact
        # distance short of the puck's, toward our own goal, and once the
        # puck is at rest within reach, STAY PUT. The first version
        # targeted 30 mm inside the puck and pushed it along at 0.3-1 m/s
        # for the whole hold -- a dribble, which the held-puck gate does
        # not count and the planner would not learn from.
        still = (speed < 0.25) & (gap < 0.03)
        tx[h] = np.where(still, mx, px)[h]
        ty[h] = np.where(still, my, py - (self.contact + 0.005))[h]
        acc[h] = -0.3

        b = ph == BLOCK
        # Get onto the path at the height we already have (moving in x only
        # is the shortest trip; an 8 m/s shot crosses the half in ~120 ms
        # and a 40 m/s^2 body covers 0.3 m in that), never below the block
        # floor. The rebound is picked up afterwards.
        y_block = np.clip(my, floor_y + BLOCK_Y_ABOVE_FLOOR, station_y)
        tx[b] = self._x_at(px, py, vx, vy, y_block)[b]
        ty[b] = y_block[b]
        acc[b] = 1.0

        wu = ph == WINDUP
        tx[wu] = np.clip(back_x, ws["min_x"], ws["max_x"])[wu]
        ty[wu] = np.clip(back_y, ws["min_y"], ws["max_y"])[wu]
        acc[wu] = 1.0

        s = ph == SHOOT
        tx[s] = (px + uxa * SHOOT_THROUGH)[s]
        ty[s] = (py + uya * SHOOT_THROUGH)[s]
        acc[s] = 1.0

        return self._to_action(tx, ty, acc)
