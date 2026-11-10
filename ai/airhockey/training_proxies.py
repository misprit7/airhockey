"""People at the far end of TRAINING games and drills. Simulation only.

The benchmark person (airhockey/human_proxy.py) as a sparring partner inside
NeuralTrainingEnv: a fixed set of env rows whose far side is the free
human-capable body, each game played by a newly drawn person of a tier drawn
from a list. The panel only wraps HumanProxy objects (one per tier, each the
size of the row set); every decision is still HumanProxy.act on the far
side's own camera view. Nothing here runs at the table or reaches the robot.
"""
from __future__ import annotations

import numpy as np

from airhockey.human_proxy import TIERS, HumanProxy


def human_proxy_slots(n_envs, game_count, share):
    """Which of one env instance's slots a person plays: [n_envs] bool.

    `share` is of the instance's full-game slots (ids below `game_count`),
    evenly spread over the ones with id % 4 != 0, so the trainer's fixed
    references (id % 4 == 0) keep their slots. A pure function of the three
    numbers: the trainer calls it per worker to know the slots without asking.
    """
    if not np.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("human proxy share must be in [0,1]")
    ids = np.arange(n_envs)
    eligible = np.flatnonzero((ids < game_count) & (ids % 4 != 0))
    count = int(round(share * game_count))
    if count > len(eligible):
        raise ValueError(f"human proxy share {share:g} needs {count} game slots but only "
                         f"{len(eligible)} of {game_count} are free of fixed references")
    slots = np.zeros(n_envs, bool)
    if count:
        slots[eligible[(np.arange(count) * len(eligible)) // count]] = True
    return slots


def spread_slots(candidates, fraction):
    """An evenly spread `fraction` of the ids in `candidates` (sorted ints)."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("slot fraction must be in [0,1]")
    candidates = np.asarray(candidates, dtype=int)
    count = int(round(fraction * len(candidates)))
    return candidates[(np.arange(count) * len(candidates)) // count] if count else candidates[:0]


class ProxyPanel:
    """One person per game on `n_rows` rows; tiers and people drawn per game.

    restart(rows) ends the games in those rows: the next command() draws
    each a tier's new person (HumanProxy.restart) before deciding. Rows whose
    tier differs are still stepped by the other tiers' objects on a view that
    is not theirs; that state is discarded at their next restart.
    """

    def __init__(self, n_rows, tiers, *, seed=0, config=None, strike_speed=None):
        self.n = int(n_rows)
        self.tiers = tuple(tiers)
        unknown = [tier for tier in self.tiers if tier not in TIERS]
        if not self.tiers or unknown:
            raise ValueError(f"human proxy tiers must be from {sorted(TIERS)}; got {list(tiers)}")
        if strike_speed is not None:
            lo, hi = (float(v) for v in strike_speed)
            if not (np.isfinite([lo, hi]).all() and 0 < lo <= hi):
                raise ValueError("strike speed band must satisfy 0 < low <= high")
            strike_speed = (lo, hi)
        self.strike_speed = strike_speed
        # Its own stream: people never draw from the env's generator.
        self.rng = np.random.default_rng([int(seed) % 2**32, 20261110])
        self.proxies = [HumanProxy(self.n, tier, seed=int(self.rng.integers(2**31)), config=config)
                        for tier in self.tiers]
        self.tier = self.rng.integers(len(self.tiers), size=self.n)
        self.pending = np.ones(self.n, bool)
        self.strike = np.zeros(self.n, bool)
        self.games = np.zeros(len(self.tiers), dtype=int)
        self._started = False

    def restart(self, rows, strike=False):
        """New games in `rows`. `strike` marks drill rows whose planned
        release speeds are redrawn uniformly from the panel's strike band."""
        rows = np.asarray(rows, dtype=int)
        self.tier[rows] = self.rng.integers(len(self.tiers), size=len(rows))
        self.pending[rows] = True
        self.strike[rows] = strike

    def command(self, view):
        """One tick for every row from its [n_rows, >=12] far-side view.

        Returns (x, y, speed, accel) in the table frame, the rows whose game
        began this tick and those people's body time constants.
        """
        view = np.asarray(view, dtype=np.float64)
        if not self._started:
            for proxy in self.proxies:
                proxy.reset(view)
            self._started = True
        began = np.flatnonzero(self.pending)
        lag = np.zeros(len(began))
        for j, proxy in enumerate(self.proxies):
            mine = self.tier[began] == j
            rows = began[mine]
            if not len(rows):
                continue
            _, lag[mine] = proxy.restart(rows, view[rows], int(self.rng.integers(2**31)))
            self.games[j] += len(rows)
            drill = rows[self.strike[rows]]
            if len(drill) and self.strike_speed is not None:
                speed = self.rng.uniform(*self.strike_speed, (len(drill), proxy.PLANS))
                proxy.plan_speed[drill] = np.clip(speed, *proxy.tier.release_clip_m_s)
        self.pending[:] = False
        out = np.zeros((4, self.n))
        for j, proxy in enumerate(self.proxies):
            mine = self.tier == j
            if mine.any():
                for target, value in zip(out, proxy.act(view)):
                    target[mine] = np.broadcast_to(value, (self.n,))[mine]
        return out, began, lag

    def serve(self, scored):
        """After goals in the rows of `scored`: (rows, x, y) where a serving
        tier's person puts the puck down at rest; other tiers' rows are left
        to the engine's restart."""
        scored = np.asarray(scored, bool) & ~self.pending
        rows, xs, ys = [], [], []
        for j, proxy in enumerate(self.proxies):
            mine = scored & (self.tier == j)
            if proxy.tier.serve_x_m is None or not mine.any():
                continue
            x, y = proxy.serve(mine)
            rows.append(np.flatnonzero(mine)), xs.append(x), ys.append(y)
        if not rows:
            return np.zeros(0, dtype=int), np.zeros(0), np.zeros(0)
        return np.concatenate(rows), np.concatenate(xs), np.concatenate(ys)
