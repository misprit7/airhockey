"""Vectorized physics engine for batch environment stepping.

Processes N environments simultaneously using NumPy array operations.
All state is stored as arrays of shape [N] instead of scalar dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from airhockey.physics import TableConfig


@dataclass(frozen=True)
class SpeedRestitution:
    """Measured robot-mallet restitution, falling with closing speed.

    482 clean contacts of the standard paddle over the v4 table sessions
    (2026-10-19, 10-29, 11-08): e is about 0.50 below 3 m/s of closing normal
    speed and about 0.33 above 9, linear between, with a per-contact scatter
    of 0.10. Ranges are (low, high) per-episode draws that cover the session
    confidence intervals; `nominal` is the pooled fit, used when the engine
    does not randomise. Tangential retention is the engine's existing
    paddle_tangential_range (measured 0.80, range 0.70-0.90).
    """
    e_lo: tuple[float, float] = (0.42, 0.58)        # closing <= knee_m_s[0]
    e_hi: tuple[float, float] = (0.27, 0.42)        # closing >= knee_m_s[1]
    nominal: tuple[float, float] = (0.505, 0.327)
    knee_m_s: tuple[float, float] = (3.0, 9.0)
    scatter: float = 0.10                           # per-contact sigma on e
    clip: tuple[float, float] = (0.05, 0.90)


@dataclass(frozen=True)
class MalletLift:
    """Contact failure: a fast puck passes under a braking robot mallet.

    17 pass-throughs in 313 central contacts on the table, every one on a
    6.0-7.9 m/s puck, rising with the mallet's realised braking at contact
    (9% when not braking, 56% at 60 m/s^2 or more) and never against a
    mallet slower than 0.5 m/s. At first touch of a CENTRAL contact (the
    straight-line miss distance under `central_m`):

        d   = puck travel direction, s = puck speed (table frame)
        a_n = (paddle acceleration over the last accel_window_s) . d
        u   = paddle velocity . d            (negative = moving at the puck)
        P   = G(s) * M(u) * (b0 + (p100 - b0) * clip((a_n - a0) / (a_full - a0), 0, 1))
        G   = 0 below speed_gate[0], 1 across speed_gate[1:3], k_hi above [3]
        M   = 1 if u <= retreat_m_s else m_ret

    Ranges are per-episode draws; p100 is exactly 0 in `off_probability` of
    episodes (one evening's log had 9 events and the next had none), and
    `speed_shift` moves the lower gate. `nominal` is the pooled fit. The
    accel knee below ~40 m/s^2 is a physical prior, not a measurement: v4
    answered every 6-8 m/s shot at 97 m/s^2 or more.
    """
    off_probability: float = 0.25
    p100: tuple[float, float] = (0.15, 0.95)
    a0: tuple[float, float] = (10.0, 50.0)
    b0: tuple[float, float] = (0.0, 0.05)
    m_ret: tuple[float, float] = (0.1, 0.5)
    k_hi: tuple[float, float] = (0.0, 0.4)
    speed_shift: tuple[float, float] = (-0.5, 0.5)
    # p100, a0, b0, m_ret, k_hi, speed_shift
    nominal: tuple[float, ...] = (0.44, 30.0, 0.03, 0.3, 0.0, 0.0)
    speed_gate: tuple[float, float, float, float] = (5.5, 6.0, 8.0, 8.5)
    a_full: float = 100.0
    retreat_m_s: float = 0.5
    central_m: float = 0.070
    accel_window_s: float = 0.020       # the controller's 50 Hz difference
    retention: tuple[float, float] = (0.45, 0.95)   # puck speed kept
    timeout_s: float = 0.15             # contact resumes even if still overlapped


class BatchPhysicsEngine:
    """Vectorized 2D air hockey physics for N parallel environments."""

    def __init__(
        self,
        n_envs: int,
        config: TableConfig | None = None,
        domain_randomize: bool = False,
    ):
        self.n_envs = n_envs
        self.config = config or TableConfig()
        self.domain_randomize = domain_randomize
        # Opt-in and checkpointed by the expanded-workspace experiment. Paddle
        # separation can otherwise leave a puck through a rail until next tick.
        self.project_rail_contacts = False

        # Per-env physics parameters — shape [N]
        # Initialized to defaults; randomized on reset when domain_randomize=True
        cfg = self.config
        self.puck_friction = np.full(n_envs, cfg.puck_friction)
        self.wall_restitution = np.full(n_envs, cfg.wall_restitution)
        self.wall_tangential = np.full(n_envs, cfg.wall_tangential)
        self.end_wall_tangential = np.full(n_envs, cfg.end_wall_tangential)
        self.drag_b = np.full(n_envs, cfg.PUCK_DRAG_B)
        self.paddle_restitution = np.full(n_envs, cfg.paddle_restitution)
        self.puck_mass = np.full(n_envs, cfg.puck_mass)
        # Paddle contact randomisation. The default range is the historical
        # guess. Table play (636 contacts, 2026-10-02..11) measured normal
        # restitution 0.47-0.59 in the paddle frame and tangential retention
        # 0.76-0.84, so experiments may opt into a measured range; None keeps
        # the frictionless contact and the default random stream unchanged.
        self.paddle_restitution_range = (0.60, 0.95)
        self.paddle_tangential_range = None
        self.paddle_tangential = np.ones(n_envs)
        # Opt-in table-faithful contact for ROBOT mallets (see the two
        # dataclasses above and airhockey.table_fidelity). None = off: no
        # draw is made and no contact is resolved differently. The agent's
        # paddle is always the machine; `opponent_robot` says which far-side
        # paddles are a copy of it (set by the env) -- a hand-held mallet
        # keeps paddle_restitution and never lifts.
        self.speed_restitution: SpeedRestitution | None = None
        self.mallet_lift: MalletLift | None = None
        self.opponent_robot = np.zeros(n_envs, dtype=bool)
        self.paddle_e_lo = np.full(n_envs, SpeedRestitution.nominal[0])
        self.paddle_e_hi = np.full(n_envs, SpeedRestitution.nominal[1])
        # Per-env lift parameters [6, N] in MalletLift.nominal's order, and
        # the lift state per body (0 agent, 1 opponent): the puck is under
        # the mallet, was overlapping last substep, seconds since a lift.
        self.lift_parameters = np.tile(np.array(MalletLift.nominal)[:, None], (1, n_envs))
        self.lift_through = np.zeros((2, n_envs), dtype=bool)
        self.lift_age = np.full((2, n_envs), np.inf)
        # Cumulative since construction; never cleared by reset().
        self.lift_events = np.zeros((2, n_envs), dtype=np.int64)
        self._lift_timer = np.zeros((2, n_envs))
        self._in_contact = np.zeros((2, n_envs), dtype=bool)
        self._paddle_v_history = [None, None]
        self._dt = 0.0

        # Allocate state arrays — all shape [N]
        self.puck_x = np.zeros(n_envs)
        self.puck_y = np.zeros(n_envs)
        self.puck_vx = np.zeros(n_envs)
        self.puck_vy = np.zeros(n_envs)

        self.paddle_agent_x = np.zeros(n_envs)
        self.paddle_agent_y = np.zeros(n_envs)
        self.paddle_agent_vx = np.zeros(n_envs)
        self.paddle_agent_vy = np.zeros(n_envs)

        self.paddle_opp_x = np.zeros(n_envs)
        self.paddle_opp_y = np.zeros(n_envs)
        self.paddle_opp_vx = np.zeros(n_envs)
        self.paddle_opp_vy = np.zeros(n_envs)

        self.score_agent = np.zeros(n_envs, dtype=np.int32)
        self.score_opponent = np.zeros(n_envs, dtype=np.int32)
        self.goal_scored = np.zeros(n_envs, dtype=np.int32)  # 0=none, 1=agent, -1=opp
        self.time = np.zeros(n_envs)

        # The generator the post-goal restart draws from. It is remembered
        # from reset() rather than created per goal: _check_goals used to make
        # a fresh default_rng() on every call, which seeds from OS entropy and
        # made a seeded env non-reproducible from the first goal onward. That
        # is invisible in training and fatal to any evaluation that compares
        # two controllers on "the same" fixtures.
        self._rng = np.random.default_rng()
        # Optional instrumentation at the actual impulse, before the speed
        # backstop. Disabled in training unless a caller installs a callback.
        self.contact_callback = None
        self.goal_callback = None

    def enable_speed_restitution(self, **overrides) -> None:
        """Opt into SpeedRestitution for robot mallets. Call before reset()."""
        model = replace(SpeedRestitution(), **overrides)
        for name in ("e_lo", "e_hi", "clip"):
            lo, hi = getattr(model, name)
            if not 0 < lo <= hi <= 1:
                raise ValueError(f"{name} must satisfy 0 < low <= high <= 1")
        if not 0 <= model.knee_m_s[0] < model.knee_m_s[1] or model.scatter < 0:
            raise ValueError("invalid restitution knee or scatter")
        self.speed_restitution = model
        self.paddle_e_lo[:], self.paddle_e_hi[:] = model.nominal

    def enable_mallet_lift(self, **overrides) -> None:
        """Opt into MalletLift for robot mallets. Call before reset()."""
        model = replace(MalletLift(), **overrides)
        for name in ("p100", "b0", "m_ret", "k_hi", "retention"):
            lo, hi = getattr(model, name)
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"{name} must satisfy 0 <= low <= high <= 1")
        gate = model.speed_gate
        if (not 0 <= model.off_probability <= 1 or len(model.nominal) != 6
                or not gate[0] < gate[1] <= gate[2] < gate[3]
                or not model.a0[0] <= model.a0[1] < model.a_full
                or min(model.central_m, model.accel_window_s, model.timeout_s) <= 0):
            raise ValueError("invalid mallet lift model")
        self.mallet_lift = model
        self.lift_parameters[:] = np.array(model.nominal)[:, None]

    def reset(
        self,
        rng: np.random.Generator | None = None,
        mask: np.ndarray | None = None,
        still: bool = False,
    ) -> None:
        """Reset environments. If mask is provided, only reset those indices."""
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        if mask is None:
            n = self.n_envs
            idx = slice(None)
        else:
            n = int(mask.sum())
            if n == 0:
                return
            idx = mask

        # Domain randomization: per-env physics parameters
        if self.domain_randomize:
            # Ranges centred on what was MEASURED, with a width that covers
            # the spread actually observed rather than a round guess. The old
            # ones predate the table: friction 0.005-0.05 is 3x to 33x the
            # rolling term, so most sampled envs modelled a surface the puck
            # has never been on.
            #
            # friction: the rolling term is 0.0015 and only known to +-0.0007,
            #           so this spans roughly its own error bar.
            # drag:     two sessions gave 3.43e-2 and 3.54e-2. Widened well
            #           past that, since it is the term that matters most and
            #           the policy should not be brittle to it.
            # wall e:   per-rail 0.756-0.813, and it falls ~0.012 per m/s of
            #           impact speed over a 9 m/s range, so the range has to
            #           cover the speed dependence the sim does not model.
            # tangential: September 20 side-rail fit ~0.90; end rails retain
            # the earlier calibration. This remains a coarse friction model.
            # paddle e: UNMEASURED, so this stays deliberately wide -- the one
            #           parameter randomisation is genuinely papering over.
            self.puck_friction[idx] = rng.uniform(0.0005, 0.0030, size=n)
            self.drag_b[idx] = rng.uniform(0.030, 0.040, size=n)
            self.wall_restitution[idx] = rng.uniform(0.72, 0.86, size=n)
            # One surface-condition draw for both rail pairs preserves seeded
            # initial states across this calibration change.
            rail_draw = rng.random(n)
            self.wall_tangential[idx] = 0.82 + 0.16 * rail_draw
            self.end_wall_tangential[idx] = 0.58 + 0.18 * rail_draw
            self.paddle_restitution[idx] = rng.uniform(*self.paddle_restitution_range, size=n)
            self.puck_mass[idx] = rng.uniform(0.01, 0.04, size=n)
            if self.paddle_tangential_range is not None:
                self.paddle_tangential[idx] = rng.uniform(*self.paddle_tangential_range, size=n)
            if self.speed_restitution is not None:
                m = self.speed_restitution
                self.paddle_e_lo[idx] = rng.uniform(*m.e_lo, size=n)
                self.paddle_e_hi[idx] = np.minimum(rng.uniform(*m.e_hi, size=n),
                                                   self.paddle_e_lo[idx])
            if self.mallet_lift is not None:
                m = self.mallet_lift
                on = rng.random(n) >= m.off_probability
                draws = np.array([rng.uniform(*r, size=n) for r in
                                  (m.p100, m.a0, m.b0, m.m_ret, m.k_hi, m.speed_shift)])
                # Off means off: no baseline rate either.
                draws[0] *= on
                draws[2] *= on
                self.lift_parameters[:, idx] = draws
        if self.mallet_lift is not None:
            self.lift_through[:, idx] = False
            self.lift_age[:, idx] = np.inf
            self._lift_timer[:, idx] = 0.0
            self._in_contact[:, idx] = False
            for history in self._paddle_v_history:
                if history is not None:
                    history[:, idx] = 0.0

        # Randomize puck position and velocity
        angle = rng.uniform(-np.pi * 0.8, -np.pi * 0.2, size=n)
        speed = rng.uniform(0.3, 1.5, size=n)
        px = rng.uniform(cfg.puck_radius, cfg.width - cfg.puck_radius, size=n)
        py = rng.uniform(cfg.height * 0.25, cfg.height * 0.6, size=n)

        # Randomize agent paddle
        ax = rng.uniform(cfg.paddle_radius, cfg.width - cfg.paddle_radius, size=n)
        ay = rng.uniform(cfg.paddle_radius, cfg.height / 2 - cfg.paddle_radius, size=n)

        if mask is None:
            idx = slice(None)
        else:
            idx = mask

        # `still`: a human is at the controls, not a training run. The
        # randomised moving start exists to keep reward signal flowing
        # during training and is wrong when someone is watching to see
        # what the machine does when told to move.
        if still:
            px = np.full(n, cfg.width / 2)
            py = np.full(n, cfg.height / 2)
            speed = np.zeros(n)

        self.puck_x[idx] = px
        self.puck_y[idx] = py
        self.puck_vx[idx] = speed * np.cos(angle)
        self.puck_vy[idx] = speed * np.sin(angle)

        self.paddle_agent_x[idx] = ax
        self.paddle_agent_y[idx] = ay
        self.paddle_agent_vx[idx] = 0.0
        self.paddle_agent_vy[idx] = 0.0

        self.paddle_opp_x[idx] = cfg.width / 2
        self.paddle_opp_y[idx] = cfg.height * 0.85
        self.paddle_opp_vx[idx] = 0.0
        self.paddle_opp_vy[idx] = 0.0

        self.score_agent[idx] = 0
        self.score_opponent[idx] = 0
        self.goal_scored[idx] = 0
        self.time[idx] = 0.0

    def step(self, dt: float) -> None:
        """Advance all N environments by dt seconds."""
        self.goal_scored[:] = 0
        if self.contact_callback is not None:
            self._contact_time = self.time + dt
        if self.mallet_lift is not None:
            self._dt = dt
            self.lift_age += dt
        self._apply_friction(dt)
        self._move_puck(dt)
        self._collide_walls()
        self._collide_paddle(
            self.paddle_agent_x, self.paddle_agent_y,
            self.paddle_agent_vx, self.paddle_agent_vy,
        )
        self._collide_paddle(
            self.paddle_opp_x, self.paddle_opp_y,
            self.paddle_opp_vx, self.paddle_opp_vy,
        )
        if self.project_rail_contacts:
            self._collide_walls()
        self._clamp_puck_speed()
        self._check_goals()
        self.time += dt

    def update_paddle_agent(
        self, new_x: np.ndarray, new_y: np.ndarray, dt: float
    ) -> None:
        """Update agent paddle positions and compute velocities."""
        if dt > 0:
            self.paddle_agent_vx = (new_x - self.paddle_agent_x) / dt
            self.paddle_agent_vy = (new_y - self.paddle_agent_y) / dt
        self.paddle_agent_x = new_x.copy()
        self.paddle_agent_y = new_y.copy()
        if self.mallet_lift is not None and dt > 0:
            self._record_paddle_velocity(0, self.paddle_agent_vx, self.paddle_agent_vy, dt)

    def update_paddle_opponent(
        self, new_x: np.ndarray, new_y: np.ndarray, dt: float
    ) -> None:
        """Update opponent paddle positions and compute velocities."""
        if dt > 0:
            self.paddle_opp_vx = (new_x - self.paddle_opp_x) / dt
            self.paddle_opp_vy = (new_y - self.paddle_opp_y) / dt
        self.paddle_opp_x = new_x.copy()
        self.paddle_opp_y = new_y.copy()
        if self.mallet_lift is not None and dt > 0:
            self._record_paddle_velocity(1, self.paddle_opp_vx, self.paddle_opp_vy, dt)

    # --- Vectorized physics internals ---

    def _record_paddle_velocity(self, side: int, vx, vy, dt: float) -> None:
        """Keep accel_window_s of a body's velocity, newest last, for the
        lift model's realised acceleration (the table's number is a 20 ms
        difference of controller velocity, so this is one too)."""
        steps = max(1, int(round(self.mallet_lift.accel_window_s / dt)))
        history = self._paddle_v_history[side]
        if history is None or len(history) != steps + 1:
            history = self._paddle_v_history[side] = np.zeros((steps + 1, self.n_envs, 2))
        history[:-1] = history[1:]
        history[-1, :, 0] = vx
        history[-1, :, 1] = vy

    def paddle_acceleration(self, side: int) -> np.ndarray:
        """[N, 2] realised acceleration of a body over accel_window_s."""
        history = self._paddle_v_history[side]
        if history is None or self._dt <= 0:
            return np.zeros((self.n_envs, 2))
        return (history[-1] - history[0]) / ((len(history) - 1) * self._dt)

    def _lift_gate(self, side, colliding, dist, dx, dy, pad_vx, pad_vy):
        """Roll the lift at first touch; return `colliding` without the
        envs whose puck is under this body's mallet."""
        model = self.mallet_lift
        min_dist = self.config.puck_radius + self.config.paddle_radius
        through = self.lift_through[side]
        timer = self._lift_timer[side]
        overlap = dist < min_dist
        # Contact comes back once the centres have parted (or, for a puck
        # carried along under the mallet, after timeout_s).
        timer[through] += self._dt
        through &= overlap & (timer <= model.timeout_s)
        first = colliding & ~self._in_contact[side] & ~through
        self._in_contact[side] = overlap
        robot = first if side == 0 else first & self.opponent_robot
        p100, a0, b0, m_ret, k_hi, shift = self.lift_parameters
        candidate = robot & (p100 > 0)
        if candidate.any():
            i = np.flatnonzero(candidate)
            vx, vy = self.puck_vx[i], self.puck_vy[i]
            s = np.hypot(vx, vy)
            safe = np.maximum(s, 1e-9)
            ux, uy = vx / safe, vy / safe
            g0, g1, g2, g3 = model.speed_gate
            gate = np.where(
                s < g2, np.clip((s - g0 - shift[i]) / (g1 - g0), 0, 1),
                1 - (1 - k_hi[i]) * np.clip((s - g2) / (g3 - g2), 0, 1))
            accel = self.paddle_acceleration(side)[i]
            a_n = accel[:, 0] * ux + accel[:, 1] * uy
            along = pad_vx[i] * ux + pad_vy[i] * uy
            braking = np.clip((a_n - a0[i]) / (model.a_full - a0[i]), 0, 1)
            # Central and closing, in the mallet's frame, on a puck that is
            # travelling AT the mallet: the fit is of incoming pucks, and
            # the edge that lifts is the one facing along -d. A mallet that
            # catches up with its own shot accelerates with that edge down.
            wx, wy = vx - pad_vx[i], vy - pad_vy[i]
            miss = np.abs(dx[i] * wy - dy[i] * wx) / np.maximum(np.hypot(wx, wy), 1e-9)
            central = ((miss < model.central_m) & (dx[i] * wx + dy[i] * wy < 0)
                       & (dx[i] * ux + dy[i] * uy < 0))
            p = (gate * np.where(along <= model.retreat_m_s, 1.0, m_ret[i])
                 * (b0[i] + (p100[i] - b0[i]) * braking) * central)
            rolled = p > 0
            fire = np.zeros(len(i), dtype=bool)
            fire[rolled] = self._rng.random(int(rolled.sum())) < p[rolled]
            j = i[fire]
            if len(j):
                keep = self._rng.uniform(*model.retention, size=len(j))
                self.puck_vx[j] *= keep
                self.puck_vy[j] *= keep
                through[j] = True
                timer[j] = 0.0
                self.lift_age[side, j] = 0.0
                self.lift_events[side, j] += 1
        return colliding & ~through

    def _robot_restitution(self, side, closing, resolve):
        """Per-env e for this contact: the speed curve plus scatter where
        the body is the machine, paddle_restitution where it is not."""
        model = self.speed_restitution
        lo, hi = model.knee_m_s
        e = self.paddle_e_lo + (self.paddle_e_hi - self.paddle_e_lo) * np.clip(
            (closing - lo) / (hi - lo), 0, 1)
        robot = resolve if side == 0 else resolve & self.opponent_robot
        if model.scatter > 0 and robot.any():
            e[robot] += self._rng.normal(0.0, model.scatter, int(robot.sum()))
        e = np.clip(e, *model.clip)
        if side == 0:
            return e
        return np.where(self.opponent_robot, e, self.paddle_restitution)

    def _apply_friction(self, dt: float) -> None:
        """Rolling plus AERODYNAMIC drag: decel = mu*g + b*v^2.

        The v^2 term is the whole story above walking pace. A constant
        deceleration -- which is what this was -- is about 3x too much on a
        drifting puck and 8x too little on a struck one, and a struck puck is
        the only regime a policy plays in. Measured over two sessions six days
        apart with different puck marking; see TableConfig.PUCK_DRAG_B.
        """
        speed = np.hypot(self.puck_vx, self.puck_vy)
        friction_decel = self.puck_friction * 9.81 + self.drag_b * speed ** 2
        new_speed = np.maximum(0.0, speed - friction_decel * dt)
        safe_speed = np.maximum(speed, 1e-8)
        factor = np.where(speed > 1e-6, new_speed / safe_speed, 0.0)
        self.puck_vx *= factor
        self.puck_vy *= factor

    def _move_puck(self, dt: float) -> None:
        self.puck_x += self.puck_vx * dt
        self.puck_y += self.puck_vy * dt

    def _collide_walls(self) -> None:
        cfg = self.config
        r = cfg.puck_radius
        e = self.wall_restitution  # per-env [N]
        # Side and end rails have separate calibrations. The September 20
        # data identifies side rails only; do not extrapolate it to ends.
        t = self.wall_tangential  # per-env [N]
        goal_left = (cfg.width - cfg.goal_width) / 2
        goal_right = (cfg.width + cfg.goal_width) / 2

        # Left wall
        hit_left = self.puck_x - r < 0
        self.puck_x = np.where(hit_left, r, self.puck_x)
        self.puck_vx = np.where(hit_left, np.abs(self.puck_vx) * e, self.puck_vx)
        self.puck_vy = np.where(hit_left, self.puck_vy * t, self.puck_vy)

        # Right wall
        hit_right = self.puck_x + r > cfg.width
        self.puck_x = np.where(hit_right, cfg.width - r, self.puck_x)
        self.puck_vx = np.where(hit_right, -np.abs(self.puck_vx) * e, self.puck_vx)
        self.puck_vy = np.where(hit_right, self.puck_vy * t, self.puck_vy)

        # Bottom wall (agent's side) — skip goal opening
        hit_bottom = self.puck_y - r < 0
        in_goal_bottom = (self.puck_x > goal_left) & (self.puck_x < goal_right)
        bounce_bottom = hit_bottom & ~in_goal_bottom
        self.puck_y = np.where(bounce_bottom, r, self.puck_y)
        self.puck_vy = np.where(bounce_bottom, np.abs(self.puck_vy) * e, self.puck_vy)
        self.puck_vx = np.where(bounce_bottom, self.puck_vx * self.end_wall_tangential, self.puck_vx)

        # Top wall (opponent's side) — skip goal opening
        hit_top = self.puck_y + r > cfg.height
        in_goal_top = (self.puck_x > goal_left) & (self.puck_x < goal_right)
        bounce_top = hit_top & ~in_goal_top
        self.puck_y = np.where(bounce_top, cfg.height - r, self.puck_y)
        self.puck_vy = np.where(bounce_top, -np.abs(self.puck_vy) * e, self.puck_vy)
        self.puck_vx = np.where(bounce_top, self.puck_vx * self.end_wall_tangential, self.puck_vx)

    def _collide_paddle(
        self,
        pad_x: np.ndarray, pad_y: np.ndarray,
        pad_vx: np.ndarray, pad_vy: np.ndarray,
    ) -> None:
        cfg = self.config
        dx = self.puck_x - pad_x
        dy = self.puck_y - pad_y
        dist = np.hypot(dx, dy)
        min_dist = cfg.puck_radius + cfg.paddle_radius

        # Mask: colliding and not degenerate
        colliding = (dist < min_dist) & (dist >= 1e-8)
        side = 0 if pad_x is self.paddle_agent_x else 1
        if self.mallet_lift is not None:
            colliding = self._lift_gate(side, colliding, dist, dx, dy, pad_vx, pad_vy)

        if not np.any(colliding):
            return

        # Safe normalize (only where colliding, avoid div-by-zero)
        safe_dist = np.where(colliding, dist, 1.0)
        nx = dx / safe_dist
        ny = dy / safe_dist

        # Separate overlapping objects
        overlap = min_dist - dist
        self.puck_x = np.where(colliding, self.puck_x + nx * overlap, self.puck_x)
        self.puck_y = np.where(colliding, self.puck_y + ny * overlap, self.puck_y)

        # Relative velocity (puck w.r.t. paddle)
        rel_vx = self.puck_vx - pad_vx
        rel_vy = self.puck_vy - pad_vy
        rel_v_normal = rel_vx * nx + rel_vy * ny

        # Only resolve if approaching (rel_v_normal < 0)
        resolve = colliding & (rel_v_normal < 0)

        if not np.any(resolve):
            return

        e = self.paddle_restitution  # per-env [N]
        if self.speed_restitution is not None:
            e = self._robot_restitution(side, -rel_v_normal, resolve)
        impulse = -(1 + e) * rel_v_normal
        out_vx = self.puck_vx + impulse * nx
        out_vy = self.puck_vy + impulse * ny
        if self.paddle_tangential_range is not None:
            # Retain a fraction of the tangential slip in the paddle frame.
            slip = 1 - self.paddle_tangential
            out_vx = out_vx - slip * (rel_vx - rel_v_normal * nx)
            out_vy = out_vy - slip * (rel_vy - rel_v_normal * ny)

        if self.contact_callback is not None:
            idx = np.flatnonzero(resolve)
            self.contact_callback({
                "body": "agent" if pad_x is self.paddle_agent_x else "opponent",
                "indices": idx,
                "time": getattr(self, "_contact_time", self.time)[idx].copy(),
                "normal": np.column_stack((nx[idx], ny[idx])),
                "incoming": np.column_stack((self.puck_vx[idx], self.puck_vy[idx])),
                "outgoing_before_speed_cap": np.column_stack((out_vx[idx], out_vy[idx])),
                "paddle_velocity": np.column_stack((pad_vx[idx], pad_vy[idx])),
            })

        self.puck_vx = np.where(resolve, out_vx, self.puck_vx)
        self.puck_vy = np.where(resolve, out_vy, self.puck_vy)

    def _clamp_puck_speed(self) -> None:
        speed = np.hypot(self.puck_vx, self.puck_vy)
        too_fast = speed > self.config.max_puck_speed
        factor = np.where(too_fast, self.config.max_puck_speed / np.maximum(speed, 1e-8), 1.0)
        self.puck_vx *= factor
        self.puck_vy *= factor

    def _check_goals(self) -> None:
        cfg = self.config
        r = cfg.puck_radius
        rng = self._rng

        # Opponent scored (puck past agent's baseline)
        opp_scored = self.puck_y - r < -r
        # Agent scored (puck past opponent's baseline)
        agent_scored = self.puck_y + r > cfg.height + r

        n_opp = int(opp_scored.sum())
        n_agent = int(agent_scored.sum())

        if self.goal_callback is not None and n_opp + n_agent:
            idx = np.flatnonzero(opp_scored | agent_scored)
            plane = np.where(agent_scored[idx], cfg.height, 0.0)
            # Crossing point on this integration segment, before goal reset.
            back = np.divide(self.puck_y[idx] - plane, self.puck_vy[idx],
                             out=np.zeros(len(idx)), where=abs(self.puck_vy[idx]) > 1e-12)
            self.goal_callback(dict(indices=idx, agent=agent_scored[idx],
                                    crossing_x=self.puck_x[idx] - back * self.puck_vx[idx]))

        if n_opp > 0:
            self.score_opponent[opp_scored] += 1
            self.goal_scored[opp_scored] = -1
            self._reset_puck_subset(opp_scored, toward_agent=True, rng=rng)

        if n_agent > 0:
            self.score_agent[agent_scored] += 1
            self.goal_scored[agent_scored] = 1
            self._reset_puck_subset(agent_scored, toward_agent=False, rng=rng)

    def _reset_puck_subset(
        self, mask: np.ndarray, toward_agent: bool, rng: np.random.Generator
    ) -> None:
        """Reset puck to center for environments where a goal was scored."""
        cfg = self.config
        n = int(mask.sum())
        if n == 0:
            return

        if toward_agent:
            angle = rng.uniform(-np.pi * 0.8, -np.pi * 0.2, size=n)
        else:
            angle = rng.uniform(np.pi * 0.2, np.pi * 0.8, size=n)

        speed = rng.uniform(0.3, 1.5, size=n)
        self.puck_x[mask] = cfg.width / 2 + rng.uniform(-0.15, 0.15, size=n)
        self.puck_y[mask] = cfg.height / 2
        self.puck_vx[mask] = speed * np.cos(angle)
        self.puck_vy[mask] = speed * np.sin(angle)
