"""Experimental simulation cooldown using observable normalized drive loads.

This is a reserve policy over the provisional thermal features, not a drive
safety certification. The environment must still guard motion and execute the
firmware profile. No hardware interfaces are opened here.
"""

import numpy as np

from airhockey.dynamics import MAX_ACCEL_M_S2
from airhockey.physics import TableConfig


class LoadBudgetPolicy:
    """Defend gently while hot; preserve hysteresis across point resets."""

    def __init__(
        self,
        policy,
        *,
        start=0.88,
        resume=0.82,
        defend=True,
        defense_caps=(8.0,),
        defense_ceiling=0.91,
        slow_budget=None,
    ):
        if not np.isfinite([start, resume]).all() or not 0 < resume < start < 1:
            raise ValueError("cooldown thresholds must satisfy 0 < resume < start < 1")
        if policy.cfg.obs_shape["state"] != [42] or policy.cfg.action_dim != 3:
            raise ValueError(
                "load budget requires the 42-feature profile-action policy"
            )
        self.policy, self.cfg = policy, policy.cfg
        self.start, self.resume = start, resume
        if not np.isfinite(defense_ceiling) or not start < defense_ceiling < 1:
            raise ValueError(
                "defensive reserve ceiling must be above cooldown start and below 1"
            )
        self.defense_ceiling = defense_ceiling
        self.slow_budget = None
        if slow_budget is not None:
            if set(slow_budget) != {"start", "resume", "defense_ceiling"}:
                raise ValueError(
                    "slow budget requires start, resume and defense_ceiling"
                )
            s = dict(slow_budget)
            if not np.isfinite(list(s.values())).all() or not (
                0 < s["resume"] < s["start"] < s["defense_ceiling"] < 1
            ):
                raise ValueError(
                    "slow thresholds must satisfy 0 < resume < start < ceiling < 1"
                )
            self.slow_budget = s
        self.fast_cooling = self.slow_cooling = None
        self.defender = None
        self.reserve_defender = None
        if defend:
            from airhockey.interception import InterceptionController

            self.defender = InterceptionController(
                contact_tolerance=0.04,
                minimum_incoming_speed=0.1,
                threat_only=True,
                acceleration_caps=(8.0,),
            )
            self.reserve_defender = InterceptionController(
                contact_tolerance=0.04,
                minimum_incoming_speed=0.1,
                threat_only=True,
                acceleration_caps=defense_caps,
            )
        self.cooling = None
        self.frames, self.cooling_frames = 0, 0
        self.inference_mode = "prior"
        self.training_algorithm = (
            getattr(policy, "training_algorithm", "Policy") + " with RMS cooldown"
        )

    @property
    def _prev_mean_batch(self):
        return self.policy._prev_mean_batch

    @_prev_mean_batch.setter
    def _prev_mean_batch(self, value):
        self.policy._prev_mean_batch = value

    def act(self, observation, t0=False, eval_mode=True):
        import torch

        x = observation.numpy()
        if x.shape != (len(x), 42) or not np.isfinite(x).all() or (x[:, 14] <= 0).any():
            raise ValueError(
                "load budget requires finite 42-feature observations and positive acceleration caps"
            )
        action = self.policy.act(observation, t0=t0, eval_mode=eval_mode).numpy().copy()
        level = x[:, 22:30].max(1)
        if self.cooling is None or len(self.cooling) != len(x):
            self.cooling = np.zeros(len(x), bool)
            self.fast_cooling = np.zeros(len(x), bool)
            self.slow_cooling = np.zeros(len(x), bool)
        if self.slow_budget is None:
            self.cooling[level >= self.start] = True
            self.cooling[level <= self.resume] = False
            reserve_available = level < self.defense_ceiling
        else:
            # Fast and slow drive memories have very different time constants.
            # A fast-load event must not latch cooldown for many minutes merely
            # because a separate slow memory is still below its own trip point.
            fast, slow = x[:, 22:26].max(1), x[:, 26:30].max(1)
            self.fast_cooling[fast >= self.start] = True
            self.fast_cooling[fast <= self.resume] = False
            self.slow_cooling[slow >= self.slow_budget["start"]] = True
            self.slow_cooling[slow <= self.slow_budget["resume"]] = False
            self.cooling = self.fast_cooling | self.slow_cooling
            reserve_available = (fast < self.defense_ceiling) & (
                slow < self.slow_budget["defense_ceiling"]
            )
        ids = np.flatnonzero(self.cooling)
        if len(ids):
            # A vetoed shot must not resume halfway through an arrival later.
            if hasattr(self.policy, "cancel_pending"):
                self.policy.cancel_pending(self.cooling)
            cfg = TableConfig()
            low = np.full(2, cfg.paddle_radius)
            high = np.array([cfg.width, cfg.height / 2]) - cfg.paddle_radius
            action[ids, :2] = 2 * (np.array([0.5, 0.25]) - low) / (high - low) - 1
            fraction = np.clip(8 / (x[ids, 14] * MAX_ACCEL_M_S2), 0.05, 1)
            action[ids, 2] = 2 * np.sqrt((fraction - 0.05) / 0.95) - 1
            if self.defender is not None:
                # Keep a second margin for cached 10 Hz load observations. A
                # warm defender may spend reserve on an urgent interception;
                # above the ceiling only gentle tracking remains available.
                reserve = reserve_available[ids]
                for use, controller in (
                    (ids[reserve], self.reserve_defender),
                    (ids[~reserve], self.defender),
                ):
                    if len(use):
                        action[use], _ = controller(x[use], action[use])
        self.frames += len(x)
        self.cooling_frames += len(ids)
        return torch.from_numpy(action)
