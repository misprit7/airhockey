"""Make the simulator's observations look like what the camera actually gives.

The policy currently gets the engine's exact puck velocity. The real robot
gets a least-squares slope over the last 6 tracked positions, each carrying
back-projection noise, with the puck sometimes not visible at all. Those are
different signals, and a policy trained on the first will trust its velocity
estimate in ways the second does not support -- particularly when planning an
intercept, where velocity error integrates straight into a miss.

Three effects, in the order they matter:

  DROPOUT     The IR ring's own reflection blinds a ~92x103 mm patch at table
              CENTRE. The real tracker coasts on last velocity for up to
              150 ms across it. This is not noise: it is a hole in a fixed
              place, in the highest-traffic part of the table, and a policy
              that has never seen it will do something wrong there
              specifically.

  ESTIMATOR   Velocity from a finite-difference slope over noisy positions,
              not from the engine. Its lag and its noise gain are properties
              of the estimator, and the policy should be exposed to both.

  NOISE       Back-projection error, worst away from the camera nadir.

Deliberately NOT modelled here: camera delay, which the environment already
implements as an observation ring buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Loop latency, MEASURED 2026-08-23 ──────────────────────────────────
#
# vision/bin/measure_latency.py, external LED on Teensy A9, 50 trials:
#
#   command path (host -> USB -> Teensy)   0.11 ms one way, very repeatable
#   sensing path (LED lit -> Python)       see below
#
# The first run reported sensing as 9.8 ms with a spread of only 1.1 ms, which
# looked like a clean deterministic pipeline. It was an artefact. The measuring
# loop flashed immediately after reading a frame, so the flash always landed at
# the same point in the frame cycle -- and specifically at the WORST point,
# having just missed a capture, so it waited a full interval every time.
#
# Re-running with the phase deliberately randomised:
#
#   phase-locked      median 9.81  mean 9.78  spread 1.07 ms
#   phase randomised  median 7.40  mean 7.72  spread 5.13 ms   <- reality
#
# The 5.13 ms spread is one frame interval, which is what quantisation should
# produce and is the confirmation that this is the right reading. A puck moves
# at a uniformly random phase relative to the camera clock, so the second row
# is the one to simulate. Taking the first would have overstated the delay by
# 28%.
MEASURED_LOOP_MEAN_S = 0.0077
CAMERA_DELAY_RANGE_S = (0.0051, 0.0103)   # measured min .. max

# Kept because the pipeline floor is a real property: 5.14 ms is the fastest
# the camera path ever delivered, i.e. capture + transfer + centroiding with
# no waiting for the next frame.
PIPELINE_FLOOR_S = 0.0051
FRAME_INTERVAL_S = 1.0 / 200.0

# Matches PuckTracker's least-squares window in vision/bin/puck_stream.py.
# Changing it here without changing it there reintroduces exactly the
# sim/real mismatch this module exists to remove.
SLOPE_WINDOW = 6

# Measured on a stationary puck: about +/-2 mm/s of velocity noise, which at
# the 200 Hz frame rate and a 6-frame window implies roughly this much
# position noise.
POS_NOISE_MM = 0.35

# The IR ring's reflection, table-grid mm. Measured 2026-08-19.
GLARE_W_MM, GLARE_H_MM = 92.0, 103.0
COAST_MAX_S = 0.150


class PuckPerception:
    """Per-env puck position/velocity as the real tracker would report them.

    Works in SIM units (metres), converting internally, because the
    environment is in sim units and the measurements are in mm.
    """

    def __init__(self, n_envs: int, table_w: float, table_h: float,
                 dt: float, rng: np.random.Generator | None = None,
                 glare: bool = True, noise: bool = True):
        self.n = n_envs
        self.dt = dt
        self.glare = glare
        self.noise = noise
        self._rng = rng or np.random.default_rng()
        self._report_history = None

        # Glare patch sits at table centre, where the ring is mounted.
        self.cx, self.cy = table_w / 2.0, table_h / 2.0
        self.hw = GLARE_W_MM / 2000.0        # mm -> m, halved
        self.hh = GLARE_H_MM / 2000.0
        self._glare_cy = self.cy
        self.pos_noise = POS_NOISE_MM / 1000.0

        self._hist_x = np.zeros((SLOPE_WINDOW, n_envs))
        self._hist_y = np.zeros((SLOPE_WINDOW, n_envs))
        self._filled = 0
        self._coast_t = np.zeros(n_envs)
        self._last_vx = np.zeros(n_envs)
        self._last_vy = np.zeros(n_envs)
        self._last_x = np.zeros(n_envs)
        self._last_y = np.zeros(n_envs)

        # Centred time base for the slope; constant, so precompute.
        t = np.arange(SLOPE_WINDOW) * dt
        self._tc = (t - t.mean())[:, None]
        self._den = float((self._tc[:, 0] ** 2).sum())

    def set_glare(self, width_mm: float, height_mm: float, offset_y: float = 0.0) -> None:
        """Resize the blind patch (TableSensing measured a larger one)."""
        self.hw, self.hh = width_mm / 2000.0, height_mm / 2000.0
        self._glare_cy = self.cy + offset_y

    def set_rng(self, rng: np.random.Generator) -> None:
        """Adopt a new generator, e.g. when the environment is reseeded.

        Needed because this object is built once, in the env's constructor,
        from whatever generator existed then -- and `reset(seed=...)` replaces
        that generator rather than reseeding it in place. Without this the
        position noise and every draw downstream of it stayed unseeded, so a
        "seeded" evaluation was reproducible in its puck draws and not in its
        sensing.
        """
        self._rng = rng

    def enable_report_estimator(self):
        """Opt into the live encoder's bounce/gap-aware sample fit.

        Legacy sensing remains the default so existing experiment comparisons
        do not silently change. Enable before reset/training/evaluation.
        """
        from airhockey.report_sensing import ReportVelocityHistory
        from airhockey.deploy import mm_velocity_to_sim
        from airhockey.heuristics import BOUNCE_EPS_MM
        eps = tuple(abs(v) for v in mm_velocity_to_sim(BOUNCE_EPS_MM, BOUNCE_EPS_MM,
                                                      self.cx*2, self.cy))
        self._report_history = ReportVelocityHistory(self.n, self.dt, epsilon=eps)
        self._report_history.reset(self._last_x, self._last_y)

    def reset(self, x: np.ndarray, y: np.ndarray, idx=slice(None)) -> None:
        # x[idx] works for both the full and the partial case (slice(None)
        # just returns a view of x). The previous `idx != slice(None)` guard
        # was not only redundant but broken: numpy takes != on a boolean mask
        # elementwise, so the first PARTIAL reset -- i.e. the first episode to
        # end mid-training -- raised on the ambiguous truth value. Full resets
        # and the tests that only did full resets never touched it.
        self._hist_x[:, idx] = x[idx]
        self._hist_y[:, idx] = y[idx]
        self._coast_t[idx] = 0.0
        self._last_vx[idx] = 0.0
        self._last_vy[idx] = 0.0
        self._last_x[idx] = x[idx]
        self._last_y[idx] = y[idx]
        if self._report_history is not None:
            self._report_history.reset(x, y, idx)

    def visible(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.glare:
            return np.ones(self.n, dtype=bool)
        return ~((np.abs(x - self.cx) < self.hw) & (np.abs(y - self._glare_cy) < self.hh))

    def update(self, true_x: np.ndarray, true_y: np.ndarray,
               hidden: np.ndarray | None = None):
        """Return (x, y, vx, vy) as the tracker would report them.

        `hidden` [N] bool forces a dropout regardless of position -- the
        sensing fuzz's brief losses of the puck (ai/RETRAIN.md item 6),
        handled by the same coast the blind spot gets.
        """
        seen = self.visible(true_x, true_y)
        if hidden is not None:
            seen = seen & ~hidden

        mx = true_x.copy()
        my = true_y.copy()
        if self.noise:
            mx += self._rng.normal(0.0, self.pos_noise, self.n)
            my += self._rng.normal(0.0, self.pos_noise, self.n)

        if self._report_history is not None:
            return self._report_history.update(mx, my, seen)

        # A puck seen again after a dropout restarts its history from the
        # new fix: the real fitter only uses samples inside its 30 ms
        # window, so after a gap the slope is over the fresh samples alone.
        # Fitting across the gap read a 300 ms dropout as ~20 m/s of
        # velocity that never happened (found by the fuzz, 2026-09-06; the
        # IR blind spot had the same artefact).
        reacq = seen & (self._coast_t > 0.0)
        if reacq.any():
            self._hist_x[:, reacq] = mx[reacq]
            self._hist_y[:, reacq] = my[reacq]

        # Roll the history only for envs where the puck was actually seen.
        # Rolling everywhere would feed the estimator its own extrapolation,
        # which makes the coast look like real data and hides the dropout.
        self._hist_x[:-1, seen] = self._hist_x[1:, seen]
        self._hist_y[:-1, seen] = self._hist_y[1:, seen]
        self._hist_x[-1, seen] = mx[seen]
        self._hist_y[-1, seen] = my[seen]

        vx = (self._tc[:, 0] @ (self._hist_x - self._hist_x.mean(0))) / self._den
        vy = (self._tc[:, 0] @ (self._hist_y - self._hist_y.mean(0))) / self._den

        out_x, out_y = mx.copy(), my.copy()
        self._coast_t[seen] = 0.0
        self._last_vx[seen] = vx[seen]
        self._last_vy[seen] = vy[seen]
        self._last_x[seen] = mx[seen]
        self._last_y[seen] = my[seen]

        # Coasting: extrapolate on the last good velocity, exactly as the real
        # tracker does, and give up after COAST_MAX_S rather than coasting for
        # ever -- a stale estimate presented as fresh is worse than none.
        hidden = ~seen
        if hidden.any():
            self._coast_t[hidden] += self.dt
            alive = hidden & (self._coast_t <= COAST_MAX_S)
            self._last_x[alive] += self._last_vx[alive] * self.dt
            self._last_y[alive] += self._last_vy[alive] * self.dt
            out_x[hidden] = self._last_x[hidden]
            out_y[hidden] = self._last_y[hidden]
            # While coasting the report carries the last velocity; once the
            # coast has expired the puck is reported where it was last put
            # and AT REST -- which is what the deploy encoder does with a
            # puck it has not seen (ReportEncoder: last fix, zero velocity).
            # Until 2026-09-06 the sim kept reporting the stale velocity.
            vx = np.where(hidden, np.where(alive, self._last_vx, 0.0), vx)
            vy = np.where(hidden, np.where(alive, self._last_vy, 0.0), vy)

        return out_x, out_y, vx, vy


@dataclass(frozen=True)
class TableSensing:
    """What the camera lost in play on the table (v4 sessions, 19.6 min).

    The puck was seen in 95.65% of in-play frames, with 14 unseen spells of
    50 ms or more a minute -- 30% of them at each end, 17% at the centre.
    Three position-dependent pieces, all per 200 Hz frame:

      GLARE     the centre hole is 160 x 150 mm for the puck centre, not the
                92 x 103 of GLARE_*_MM.
      END BANDS near either end rail the puck is unseen with stationary
                probability `end_bands` (the robot's end, y = 0) or
                `far_end_bands` (the person's), by distance from the rail
                face and speed -- whether or not a mallet is near. A
                two-state chain so spells have the measured length:
                P(unseen -> seen) = end_recover.
      FLICKER   anywhere, spells start at `flicker_hazard` per seen frame
                (drawn per episode; the days differed 3x) with lengths from
                `flicker_frames` -- mostly one or two frames, 0.6% past
                150 ms. Includes dropped camera frames.

    The OPPONENT's mallet is hidden in spells of `opponent_frames` starting
    at opponent_per_min x a per-episode scale (0 in `opponent_off` of
    episodes; sessions ranged from 0 to 42% hidden). As the runner does, a
    hidden mallet is reported at its last fix for `stale_s`, then at the
    default pose, at rest throughout.
    """
    glare_w_mm: float = 160.0
    glare_h_mm: float = 150.0
    glare_offset_y_m: float = -0.005
    # (far edge of the band in metres from the end rail face,
    #  unseen probability at puck speed < 1, 1-4, >= 4 m/s); nearest first.
    end_bands: tuple = ((0.097, (0.375, 0.75, 0.75)),
                        (0.146, (0.45, 0.75, 0.78)),
                        (0.215, (0.10, 0.25, 0.65)),
                        (0.313, (0.05, 0.05, 0.25)))
    far_end_bands: tuple = ((0.083, (0.66, 0.66, 0.66)),
                            (0.132, (0.31, 0.73, 0.66)),
                            (0.201, (0.02, 0.19, 0.35)),
                            (0.300, (0.06, 0.06, 0.06)))
    end_speeds_m_s: tuple[float, float] = (1.0, 4.0)
    end_recover: float = 0.18
    flicker_hazard: tuple[float, float] = (0.004, 0.025)
    flicker_nominal: float = 0.0148
    # (frames low, frames high, share of spells)
    flicker_frames: tuple = ((1, 1, 0.685), (2, 2, 0.213), (3, 5, 0.074),
                             (6, 9, 0.010), (10, 19, 0.008), (20, 29, 0.004),
                             (30, 59, 0.004), (60, 98, 0.002))
    opponent_per_min: float = 89.1
    opponent_frames: tuple = ((1, 1, 0.420), (2, 2, 0.308), (3, 5, 0.159),
                              (6, 9, 0.047), (10, 29, 0.0494), (30, 59, 0.0090),
                              (60, 299, 0.0056), (300, 1420, 0.0022))
    opponent_off: float = 0.25
    opponent_scale: tuple[float, float] = (0.2, 2.5)
    stale_s: float = 0.150


class TableDropouts:
    """Per-env dropout state for TableSensing; one call per camera frame."""

    def __init__(self, n_envs: int, model: TableSensing, table_h: float):
        self.n = n_envs
        self.model = model
        self.table_h = table_h
        # Per end (robot's, person's): band edges, and unseen probability
        # with a last all-zero row -- beyond the bands the chain only recovers.
        self._band_edges = [np.array([edge for edge, _ in bands])
                            for bands in (model.end_bands, model.far_end_bands)]
        self._band_unseen = [np.array([m for _, m in bands] + [(0.0, 0.0, 0.0)])
                             for bands in (model.end_bands, model.far_end_bands)]
        self._speed_edges = np.array(model.end_speeds_m_s)
        self.flicker_hazard = np.full(n_envs, model.flicker_nominal)
        self.opponent_hazard = np.full(
            n_envs, model.opponent_per_min * FRAME_INTERVAL_S / 60.0)
        self._flicker = np.zeros(n_envs, dtype=np.int32)    # frames left
        self._end = np.zeros(n_envs, dtype=bool)
        self._opponent = np.zeros(n_envs, dtype=np.int32)

    def reset(self, rng: np.random.Generator, idx, n: int, randomize: bool) -> None:
        self._flicker[idx] = 0
        self._end[idx] = False
        self._opponent[idx] = 0
        if not randomize:
            return
        m = self.model
        self.flicker_hazard[idx] = rng.uniform(*m.flicker_hazard, size=n)
        scale = rng.uniform(*m.opponent_scale, size=n) * (rng.random(n) >= m.opponent_off)
        self.opponent_hazard[idx] = scale * m.opponent_per_min * FRAME_INTERVAL_S / 60.0

    @staticmethod
    def _lengths(table, k: int, rng: np.random.Generator) -> np.ndarray:
        """k spell lengths in frames: a class by its share, then log-uniform."""
        lo, hi, share = (np.array(col, dtype=float) for col in zip(*table))
        pick = np.minimum(np.searchsorted(np.cumsum(share) / share.sum(), rng.random(k)),
                          len(share) - 1)
        return np.rint(lo[pick] * (hi[pick] / lo[pick]) ** rng.random(k)).astype(np.int32)

    def _spells(self, left, hazard, table, rng) -> np.ndarray:
        start = (left <= 0) & (rng.random(self.n) < hazard)
        k = int(start.sum())
        if k:
            left[start] = self._lengths(table, k, rng)
        hidden = left > 0
        left[hidden] -= 1
        return hidden

    def puck_hidden(self, y: np.ndarray, speed: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        """[N] bool: flicker or an end band hides the puck this frame."""
        m = self.model
        flicker = self._spells(self._flicker, self.flicker_hazard, m.flicker_frames, rng)
        column = np.searchsorted(self._speed_edges, speed, side="right")
        near, far = (unseen[np.searchsorted(edges, distance, side="right"), column]
                     for edges, unseen, distance in zip(
                         self._band_edges, self._band_unseen, (y, self.table_h - y)))
        unseen = np.where(y < self.table_h / 2, near, far)
        lose = np.minimum(unseen * m.end_recover / (1.0 - unseen), 1.0)
        u = rng.random(self.n)
        # The cause is the place: a puck that has left the bands is seen
        # again at once (holding it through the chain's 28 ms mean doubled
        # the long spells behind every shot that reached an end rail).
        self._end = np.where(self._end, (u >= m.end_recover) & (unseen > 0), u < lose)
        return flicker | self._end

    def opponent_hidden(self, rng: np.random.Generator) -> np.ndarray:
        """[N] bool: the camera has no fix on the far mallet this frame."""
        return self._spells(self._opponent, self.opponent_hazard,
                            self.model.opponent_frames, rng)
