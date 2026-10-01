"""Provisional per-drive load proxy for simulation. No hardware interfaces.

H is squared current relative to each drive limit. Fast and slow memories
persist across points AND training episode resets. Parameters combine the
recorded fit with explicitly labeled conservative priors to avoid rewarding
unobserved directions/edge holding as free. This is not a drive safety model.
"""

from pathlib import Path
import json
import numpy as np
from airhockey.dynamics import _geom as geom

LEGACY_MODEL = Path(__file__).resolve().parents[1] / "recipes/motor-load-20260920.json"
# Current physical topology. Pass LEGACY_MODEL explicitly for pre-swap studies.
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "recipes/motor-load-20260929.json"


class MotorThermal:
    def __init__(self, n, path=DEFAULT_MODEL, seed=0, randomize=True, soft_start=0.65):
        if not 0 <= soft_start < 1:
            raise ValueError("soft load penalty must start below the load limit")
        self.soft_start = soft_start
        self.energy_weight = 2.0
        self.config = dict(path) if isinstance(path,dict) else json.loads(Path(path).read_text())
        c = self.config
        self.spatial = c.get('schema') == 'spatial-current-v2'
        self.coeff = np.asarray(c["motion_coefficients_amps_squared" if self.spatial else "coefficients_amps_squared"], dtype=float)
        if self.spatial:
            self.centers=np.asarray(c['holding_centers_mm'],dtype=float)
            self.holding=np.asarray(c['holding_amps_squared'],dtype=float)
            if self.centers.shape!=(9,2) or self.holding.shape!=(9,4) or not np.isfinite(self.centers).all() or not np.isfinite(self.holding).all() or (self.holding<0).any():
                raise ValueError('invalid measured holding map')
            self.spatial_low=self.centers.min(0);self.spatial_high=self.centers.max(0)
            self.spacing=(self.spatial_high-self.spatial_low)/2
            if (self.spacing<=0).any():raise ValueError('invalid holding map spacing')
            expected={(x,y) for x in np.linspace(self.spatial_low[0],self.spatial_high[0],3)
                            for y in np.linspace(self.spatial_low[1],self.spatial_high[1],3)}
            if {tuple(x) for x in self.centers}!=expected:
                raise ValueError('holding map must cover the complete regular 3x3 grid')
            prior=c['unmeasured_extension_amps_per_100mm']
            if not np.isfinite(prior) or prior<0:raise ValueError('invalid unmeasured extension prior')
        self.limits = np.asarray([c["fast_limit_amps"], c["slow_limit_amps"]])
        self.tau = np.asarray([c["fast_tau_s"], c["slow_tau_s"]])
        if (
            self.coeff.shape != (4, 5 if self.spatial else 4)
            or self.limits.shape != (2, 4)
            or self.tau.shape != (2, 4)
            or not np.isfinite(self.coeff).all()
            or not np.isfinite(self.limits).all()
            or not np.isfinite(self.tau).all()
            or np.any(self.coeff < 0)
            or np.any(self.limits <= 0)
            or np.any(self.tau <= 0)
        ):
            raise ValueError("invalid motor-load model")
        rng = np.random.default_rng(seed)
        self.gain = rng.uniform(1, 1.3, (n, 1)) if randomize else np.ones((n, 1))
        self.h = np.zeros((n, 2, 4))
        if randomize:
            self.h[:, 0] = rng.uniform(0, 0.65, (n, 4)) ** 2
            self.h[:, 1] = rng.uniform(0, 0.30, (n, 4)) ** 2
        self.current_squared = np.zeros((n, 4))
        self.observed = self.levels.copy()
        self.clock = 0.0
        self._decay = {}

    @property
    def levels(self):
        return np.sqrt(np.maximum(self.h, 0))

    def advance(self, position, velocity, acceleration, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        # Simulation x is grid y; simulation y reverses grid x. Model was
        # fitted in the measured table/grid frame, not simulation coordinates.
        scale = np.array(
            [
                (geom.RAIL_MAX_Y - geom.RAIL_MIN_Y) / 1000,
                (geom.RAIL_MAX_X - geom.CENTERLINE_X) / 1000,
            ]
        )
        v = velocity * scale
        a = acceleration * scale
        if self.spatial:
            grid=np.column_stack((geom.RAIL_MAX_X-position[:,1]*(geom.RAIL_MAX_X-geom.CENTERLINE_X),
                                  geom.RAIL_MIN_Y+position[:,0]*(geom.RAIL_MAX_Y-geom.RAIL_MIN_Y)))
            clipped=np.clip(grid,self.spatial_low,self.spatial_high)
            w=np.maximum(0,1-np.abs(clipped[:,None,:]-self.centers)/self.spacing).prod(axis=2)
            holding=w@self.holding
            extension=np.linalg.norm(grid-clipped,axis=1)/100*self.config['unmeasured_extension_amps_per_100mm']
            grid_a=np.column_stack((-a[:,1],a[:,0]))/60
            features=np.column_stack(((v*v).sum(axis=1)/4,np.maximum(grid_a,0)**2,np.maximum(-grid_a,0)**2))
            self.current_squared=((np.sqrt(holding)+extension[:,None])**2+features@self.coeff.T)*self.gain
            self._integrate(dt)
            return
        f = np.column_stack(
            (
                np.ones(len(v)),
                a[:, 1] ** 2 / 3600,
                a[:, 0] ** 2 / 3600,
                (v * v).sum(axis=1) / 16,
            )
        )
        i2 = f @ self.coeff.T
        grid_y = geom.RAIL_MIN_Y + position[:, 0] * (geom.RAIL_MAX_Y - geom.RAIL_MIN_Y)
        grid_x = geom.RAIL_MAX_X - position[:, 1] * (
            geom.RAIL_MAX_X - geom.CENTERLINE_X
        )
        edge = np.clip((400 - grid_y) / 230, 0, 1) * np.clip(
            (grid_x - 1650) / 208, 0, 1
        )
        hold = (edge[:, None] * np.asarray(self.config["edge_hold_amps"])) ** 2
        prior = (
            (a * a).sum(axis=1)[:, None]
            / 3600
            * np.asarray(self.config["accel_prior_amps_squared"])
        )
        self.current_squared = (np.maximum(i2, hold) + prior) * self.gain
        self._integrate(dt)

    def _integrate(self,dt):
        if dt not in self._decay:
            self._decay[dt] = np.exp(-dt / self.tau)
        decay = self._decay[dt]
        self.h = (
            decay * self.h
            + (1 - decay) * self.current_squared[:, None, :] / self.limits**2
        )
        self.clock += dt
        # Cached approximately 10 Hz observations, matching available telemetry.
        if self.clock >= 0.1 - 1e-9:
            self.observed[:] = np.round(self.levels * 100) / 100
            self.clock %= 0.1

    def penalty(self, dt):
        level = self.levels.max(axis=(1, 2))
        near = np.maximum((level - self.soft_start) / (1 - self.soft_start), 0)
        energy = np.mean(self.current_squared / self.limits[0] ** 2, axis=1)
        return dt * (self.energy_weight * energy + 40 * np.minimum(near, 3) ** 4 + 200 * (level >= 1))

    def features(self):
        # No reset/clipping at shutdown: being over budget remains observable.
        return self.observed.reshape(len(self.h), 8).astype(np.float32)
