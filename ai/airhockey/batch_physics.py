"""Vectorized physics engine for batch environment stepping.

Processes N environments simultaneously using NumPy array operations.
All state is stored as arrays of shape [N] instead of scalar dataclasses.
"""

from __future__ import annotations

import numpy as np

from airhockey.physics import TableConfig


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

        # Per-env physics parameters — shape [N]
        # Initialized to defaults; randomized on reset when domain_randomize=True
        cfg = self.config
        self.puck_friction = np.full(n_envs, cfg.puck_friction)
        self.wall_restitution = np.full(n_envs, cfg.wall_restitution)
        self.paddle_restitution = np.full(n_envs, cfg.paddle_restitution)
        self.puck_mass = np.full(n_envs, cfg.puck_mass)

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

    def reset(
        self,
        rng: np.random.Generator | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        """Reset environments. If mask is provided, only reset those indices."""
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng()

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
            self.puck_friction[idx] = rng.uniform(0.005, 0.05, size=n)
            self.wall_restitution[idx] = rng.uniform(0.7, 0.95, size=n)
            self.paddle_restitution[idx] = rng.uniform(0.75, 0.98, size=n)
            self.puck_mass[idx] = rng.uniform(0.01, 0.04, size=n)

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

    def update_paddle_opponent(
        self, new_x: np.ndarray, new_y: np.ndarray, dt: float
    ) -> None:
        """Update opponent paddle positions and compute velocities."""
        if dt > 0:
            self.paddle_opp_vx = (new_x - self.paddle_opp_x) / dt
            self.paddle_opp_vy = (new_y - self.paddle_opp_y) / dt
        self.paddle_opp_x = new_x.copy()
        self.paddle_opp_y = new_y.copy()

    # --- Vectorized physics internals ---

    def _apply_friction(self, dt: float) -> None:
        speed = np.hypot(self.puck_vx, self.puck_vy)
        friction_decel = self.puck_friction * 9.81  # per-env [N]
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
        goal_left = (cfg.width - cfg.goal_width) / 2
        goal_right = (cfg.width + cfg.goal_width) / 2

        # Left wall
        hit_left = self.puck_x - r < 0
        self.puck_x = np.where(hit_left, r, self.puck_x)
        self.puck_vx = np.where(hit_left, np.abs(self.puck_vx) * e, self.puck_vx)

        # Right wall
        hit_right = self.puck_x + r > cfg.width
        self.puck_x = np.where(hit_right, cfg.width - r, self.puck_x)
        self.puck_vx = np.where(hit_right, -np.abs(self.puck_vx) * e, self.puck_vx)

        # Bottom wall (agent's side) — skip goal opening
        hit_bottom = self.puck_y - r < 0
        in_goal_bottom = (self.puck_x > goal_left) & (self.puck_x < goal_right)
        bounce_bottom = hit_bottom & ~in_goal_bottom
        self.puck_y = np.where(bounce_bottom, r, self.puck_y)
        self.puck_vy = np.where(bounce_bottom, np.abs(self.puck_vy) * e, self.puck_vy)

        # Top wall (opponent's side) — skip goal opening
        hit_top = self.puck_y + r > cfg.height
        in_goal_top = (self.puck_x > goal_left) & (self.puck_x < goal_right)
        bounce_top = hit_top & ~in_goal_top
        self.puck_y = np.where(bounce_top, cfg.height - r, self.puck_y)
        self.puck_vy = np.where(bounce_top, -np.abs(self.puck_vy) * e, self.puck_vy)

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
        impulse = -(1 + e) * rel_v_normal

        self.puck_vx = np.where(resolve, self.puck_vx + impulse * nx, self.puck_vx)
        self.puck_vy = np.where(resolve, self.puck_vy + impulse * ny, self.puck_vy)

    def _clamp_puck_speed(self) -> None:
        speed = np.hypot(self.puck_vx, self.puck_vy)
        too_fast = speed > self.config.max_puck_speed
        factor = np.where(too_fast, self.config.max_puck_speed / np.maximum(speed, 1e-8), 1.0)
        self.puck_vx *= factor
        self.puck_vy *= factor

    def _check_goals(self) -> None:
        cfg = self.config
        r = cfg.puck_radius
        rng = np.random.default_rng()

        # Opponent scored (puck past agent's baseline)
        opp_scored = self.puck_y - r < -r
        # Agent scored (puck past opponent's baseline)
        agent_scored = self.puck_y + r > cfg.height + r

        n_opp = int(opp_scored.sum())
        n_agent = int(agent_scored.sum())

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
