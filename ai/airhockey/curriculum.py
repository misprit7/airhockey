"""Curriculum learning utilities for staged reward training."""

from __future__ import annotations

import math
from collections import deque


class PlateauDetector:
    """Detects when training reward has plateaued, signaling readiness to advance.

    Compares recent average reward (last `window` episodes) against a longer
    lookback window. If improvement is below `threshold` fraction and at least
    `min_steps` env steps have elapsed, declares a plateau.
    """

    def __init__(
        self,
        min_steps: int = 150_000,
        window: int = 100,
        lookback: int = 200,
        threshold: float = 0.05,
    ):
        self.min_steps = min_steps
        self.window = window
        self.lookback = lookback
        self.threshold = threshold
        self._rewards: deque[float] = deque(maxlen=lookback)
        self._total_steps = 0

    def reset(self) -> None:
        """Reset all state."""
        self._rewards.clear()
        self._total_steps = 0

    def record(self, episode_reward: float, steps_since_last: int = 1) -> None:
        """Record an episode result."""
        self._rewards.append(episode_reward)
        self._total_steps += steps_since_last

    def should_advance(self) -> bool:
        """Returns True if reward has plateaued and min_steps reached."""
        if self._total_steps < self.min_steps:
            return False
        if len(self._rewards) < self.lookback:
            return False

        recent = self.current_avg()
        # Compare against the older portion of the lookback window
        older = list(self._rewards)[:self.lookback - self.window]
        if not older:
            return False
        older_avg = sum(older) / len(older)

        # Plateau: recent avg hasn't improved much over older avg
        if older_avg <= 0:
            # Avoid division issues — if older avg is non-positive,
            # check absolute improvement instead
            return abs(recent - older_avg) < self.threshold
        return (recent - older_avg) / abs(older_avg) < self.threshold

    def current_avg(self) -> float:
        """Average reward over the most recent `window` episodes."""
        if not self._rewards:
            return 0.0
        recent = list(self._rewards)[-self.window:]
        return sum(recent) / len(recent)

    def check(self) -> tuple[bool, float, float]:
        """Returns (is_plateau, current_avg, older_avg).

        Convenience method combining should_advance() with the averages
        used for the comparison, useful for logging.
        """
        curr = self.current_avg()
        if len(self._rewards) >= self.lookback:
            older_rewards = list(self._rewards)[:self.lookback - self.window]
            older = sum(older_rewards) / len(older_rewards) if older_rewards else curr
        else:
            older = curr
        return self.should_advance(), curr, older

    def configure(
        self,
        min_steps: int | None = None,
        window: int | None = None,
        lookback: int | None = None,
        threshold: float | None = None,
    ) -> None:
        """Reconfigure detector parameters (e.g. on stage transition).

        Resets internal state since the old reward history may not be
        meaningful under new window/lookback sizes.
        """
        if min_steps is not None:
            self.min_steps = min_steps
        if window is not None:
            self.window = window
        if lookback is not None:
            self.lookback = lookback
            self._rewards = deque(self._rewards, maxlen=lookback)
        if threshold is not None:
            self.threshold = threshold
        self.reset()


# ---------------------------------------------------------------------------
# Per-stage learning rate defaults
# ---------------------------------------------------------------------------
STAGE_LR: dict[int, float] = {
    1: 3e-4,   # chase + hit — learn fast
    2: 2e-4,   # game vs goalie (new task, needs high LR)
    3: 1e-4,   # game vs follower
    4: 1e-4,   # self-play
}


class CurriculumLRScheduler:
    """Per-stage cosine learning rate decay for TD-MPC2 optimizers.

    Each curriculum stage starts at a stage-specific initial LR and decays
    via cosine annealing to initial_lr / 5 over estimated_stage_steps.
    On stage transition, the LR resets to the new stage's initial value.

    Handles TD-MPC2's two optimizers (world model + policy) and preserves
    the encoder LR scale ratio for the world model optimizer's first param group.
    """

    def __init__(
        self,
        optim,
        pi_optim,
        stage: int = 1,
        enc_lr_scale: float = 0.3,
        estimated_stage_steps: int = 500_000,
        min_lr_fraction: float = 0.2,
    ):
        self.optim = optim
        self.pi_optim = pi_optim
        self.enc_lr_scale = enc_lr_scale
        self.estimated_stage_steps = estimated_stage_steps
        self.min_lr_fraction = min_lr_fraction

        self._stage = stage
        self._steps_in_stage = 0
        self._base_lr = STAGE_LR.get(stage, 1e-4)
        self._min_lr = self._base_lr * min_lr_fraction
        self._apply_lr(self._base_lr)

    @property
    def current_lr(self) -> float:
        """Current base learning rate (before enc_lr_scale)."""
        return self._compute_lr()

    def _compute_lr(self) -> float:
        """Cosine decay from base_lr to min_lr over estimated_stage_steps."""
        if self.estimated_stage_steps <= 0:
            return self._base_lr
        progress = min(self._steps_in_stage / self.estimated_stage_steps, 1.0)
        # Cosine decay: starts at base_lr, ends at min_lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._min_lr + (self._base_lr - self._min_lr) * cosine

    def _apply_lr(self, lr: float) -> None:
        """Set LR on both optimizers, respecting enc_lr_scale for param group 0."""
        # World model optimizer: group 0 is encoder (scaled), rest are base LR
        for i, pg in enumerate(self.optim.param_groups):
            pg['lr'] = lr * self.enc_lr_scale if i == 0 else lr
        # Policy optimizer: single group at base LR
        for pg in self.pi_optim.param_groups:
            pg['lr'] = lr

    def step(self) -> float:
        """Called once per gradient update. Returns current LR."""
        self._steps_in_stage += 1
        lr = self._compute_lr()
        self._apply_lr(lr)
        return lr

    def set_stage(self, stage: int, estimated_steps: int | None = None) -> float:
        """Reset scheduler for a new curriculum stage. Returns new initial LR."""
        self._stage = stage
        self._steps_in_stage = 0
        self._base_lr = STAGE_LR.get(stage, 1e-4)
        self._min_lr = self._base_lr * self.min_lr_fraction
        if estimated_steps is not None:
            self.estimated_stage_steps = estimated_steps
        self._apply_lr(self._base_lr)
        return self._base_lr
