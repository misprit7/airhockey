"""Batched camera-history estimator matching the deployed ReportEncoder.

No policy, interception target, or privileged velocity is supplied here.
Only visible noisy positions and their camera times enter the fit.
"""
import numpy as np


class ReportVelocityHistory:
    def __init__(self, n, dt, *, window=.030, history=.2, epsilon=(.0015, .0015)):
        self.n, self.dt = n, dt
        self.window, self.history = window, history
        self.epsilon = epsilon
        self.time = 0.0
        self.size = max(2, int(np.ceil(window / dt)) + 1)
        self.x = np.zeros((self.size, n))
        self.y = np.zeros_like(self.x)
        self.t = np.full_like(self.x, -np.inf)
        self.last_x = np.zeros(n)
        self.last_y = np.zeros(n)

    def reset(self, x, y, idx=slice(None)):
        self.x[:, idx] = x[idx]
        self.y[:, idx] = y[idx]
        self.t[:, idx] = -np.inf
        self.t[-1, idx] = self.time
        self.last_x[idx], self.last_y[idx] = x[idx], y[idx]

    def update(self, x, y, seen):
        self.time += self.dt
        # ReportBuilder expires real samples after 200 ms; coasting samples
        # never enter its history. Preserve timestamps across shorter gaps.
        valid = self.t >= self.time - self.history - 1e-9
        self.t[~valid] = -np.inf
        for values, sample in ((self.x, x), (self.y, y)):
            values[:-1, seen] = values[1:, seen]
            values[-1, seen] = sample[seen]
        self.t[:-1, seen] = self.t[1:, seen]
        self.t[-1, seen] = self.time
        self.last_x[seen], self.last_y[seen] = x[seen], y[seen]

        count = np.isfinite(self.t).sum(0)
        keep = np.minimum(count, 2)
        alive = count >= 2
        first_x = self.x[-1] - self.x[-2]
        first_y = self.y[-1] - self.y[-2]
        # Compare every older displacement with the newest displacement,
        # exactly as heuristics.estimate_velocity does in production.
        for offset in range(2, self.size):
            j = self.size - 1 - offset
            reversed_axis = np.zeros(self.n, bool)
            for values, first, eps in ((self.x, first_x, self.epsilon[0]),
                                       (self.y, first_y, self.epsilon[1])):
                older = values[j + 1] - values[j]
                reversed_axis |= (first * older < 0) & (abs(first) > eps) & (abs(older) > eps)
            # Use a finite substitute to avoid inf-inf on expired histories.
            newest_t = np.where(count > 0, self.t[-1], 0)
            alive &= ((count > offset) & (self.t[j] >= newest_t-self.window-1e-9)
                      & ~reversed_axis)
            keep += alive
        mask = np.arange(self.size)[:, None] >= self.size - keep
        times = np.where(mask, self.t, 0)
        mean = times.sum(0) / np.maximum(keep, 1)
        centered = np.where(mask, times - mean, 0)
        den = (centered**2).sum(0)
        vx = np.divide((centered*self.x).sum(0), den, out=np.zeros(self.n), where=den > 0)
        vy = np.divide((centered*self.y).sum(0), den, out=np.zeros(self.n), where=den > 0)
        # The deployed encoder uses the latest REAL fix, not extrapolated
        # tracker output. With an expired history it keeps that position and
        # reports zero velocity until a new fix arrives.
        return self.last_x.copy(), self.last_y.copy(), vx, vy
