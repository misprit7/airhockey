"""One-environment view of the batch physics engine.

WHY THIS EXISTS
    There used to be two physics implementations -- PhysicsEngine (scalar,
    dataclasses) and BatchPhysicsEngine (vectorised, arrays) -- with ~1000
    lines of parity tests whose entire job was to check they agreed. Two
    implementations of the same rules is a bug waiting for the day someone
    fixes a collision in one of them.

    But the two ENV wrappers are not redundant in the same way: the batch env
    takes dynamics as a string and returns batched arrays for training, while
    AirHockeyEnv is a Gymnasium env whose motor dynamics object is hot-swapped
    at runtime -- the web UI substitutes HardwareDynamics to drive the real
    robot mid-session. Collapsing those would break hardware control.

    So the ENGINE is unified and the wrappers are not. This adapter presents
    the old scalar interface over BatchPhysicsEngine(n_envs=1).

    The dataclass state is rebuilt from the arrays on demand rather than
    mirrored, because a mirror that is written in two places is the same class
    of bug this file exists to remove.
"""

from __future__ import annotations

import numpy as np

from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.physics import PaddleState, PhysicsState, PuckState, TableConfig


class ScalarPhysicsEngine:
    """`PhysicsEngine`'s interface, backed by one slot of the batch engine."""

    def __init__(self, config: TableConfig | None = None):
        self.config = config or TableConfig()
        self._batch = BatchPhysicsEngine(1, self.config)
        self._state = PhysicsState()
        self._sync()

    # ── state ───────────────────────────────────────────────────────────
    def _sync(self) -> PhysicsState:
        """Refresh the dataclass view from the arrays, MUTATING in place.

        In place rather than reassigning, so the PuckState/PaddleState objects
        keep their identity for the life of the engine. Callers hold
        references to them across a step -- env.py passes state.paddle_agent
        straight back into update_paddle, which routes on identity -- and
        replacing them each sync would quietly turn one of those into a stale
        object pointing at nothing.
        """
        b, s = self._batch, self._state
        s.puck.x, s.puck.y = float(b.puck_x[0]), float(b.puck_y[0])
        s.puck.vx, s.puck.vy = float(b.puck_vx[0]), float(b.puck_vy[0])
        s.paddle_agent.x = float(b.paddle_agent_x[0])
        s.paddle_agent.y = float(b.paddle_agent_y[0])
        s.paddle_agent.vx = float(b.paddle_agent_vx[0])
        s.paddle_agent.vy = float(b.paddle_agent_vy[0])
        s.paddle_opponent.x = float(b.paddle_opp_x[0])
        s.paddle_opponent.y = float(b.paddle_opp_y[0])
        s.paddle_opponent.vx = float(b.paddle_opp_vx[0])
        s.paddle_opponent.vy = float(b.paddle_opp_vy[0])
        s.score_agent = int(b.score_agent[0])
        s.score_opponent = int(b.score_opponent[0])
        s.goal_scored = int(b.goal_scored[0])
        s.time = float(b.time[0])
        return s

    @property
    def state(self) -> PhysicsState:
        return self._sync()

    # ── the PhysicsEngine surface env.py uses ───────────────────────────
    def reset(self, rng: np.random.Generator, still: bool = False
              ) -> PhysicsState:
        self._batch.reset(rng, still=still)
        return self._sync()

    def step(self, dt: float) -> PhysicsState:
        self._batch.step(dt)
        return self._sync()

    def update_paddle(self, paddle: PaddleState, x: float, y: float,
                      dt: float) -> None:
        """Route by identity, since the batch engine has separate setters.

        Comparing against the CURRENT state objects rather than caching them:
        `state` rebuilds them each access, so a cached reference would go
        stale and silently start routing every paddle to the opponent.
        """
        s = self._state
        if paddle is s.paddle_opponent:
            self._batch.update_paddle_opponent(
                np.array([x]), np.array([y]), dt)
        else:
            self._batch.update_paddle_agent(
                np.array([x]), np.array([y]), dt)
        self._sync()

    def _reset_puck_after_goal(self, toward_agent: bool = True) -> None:
        # The batch call takes a BOOLEAN mask, positionally, and its own
        # rng last -- not an index array.
        self._batch._reset_puck_subset(
            np.array([True]), toward_agent, np.random.default_rng())
        self._sync()
