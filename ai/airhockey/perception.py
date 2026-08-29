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

        # Glare patch sits at table centre, where the ring is mounted.
        self.cx, self.cy = table_w / 2.0, table_h / 2.0
        self.hw = GLARE_W_MM / 2000.0        # mm -> m, halved
        self.hh = GLARE_H_MM / 2000.0
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

    def visible(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.glare:
            return np.ones(self.n, dtype=bool)
        return ~((np.abs(x - self.cx) < self.hw) & (np.abs(y - self.cy) < self.hh))

    def update(self, true_x: np.ndarray, true_y: np.ndarray):
        """Return (x, y, vx, vy) as the tracker would report them."""
        seen = self.visible(true_x, true_y)

        mx = true_x.copy()
        my = true_y.copy()
        if self.noise:
            mx += self._rng.normal(0.0, self.pos_noise, self.n)
            my += self._rng.normal(0.0, self.pos_noise, self.n)

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
            vx = np.where(hidden, self._last_vx, vx)
            vy = np.where(hidden, self._last_vy, vy)

        return out_x, out_y, vx, vy
