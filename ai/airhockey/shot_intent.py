"""Experimental learned shot selection over an observation-only motion controller."""

import numpy as np
import torch
from torch import nn

from airhockey.arrival_play import ArrivalPlayController

# Index zero exactly preserves the fixed direct-shot baseline.
INTENTS = np.array(
    [
        [0.5, 3.0, 0],
        [0.42, 3.0, 0],
        [0.58, 3.0, 0],
        [0.5, 5.0, 0],
        [0.42, 5.0, 0],
        [0.58, 5.0, 0],
        [0.5, 5.0, -1],
        [0.5, 5.0, 1],
        [0.5, 7.0, -1],
        [0.5, 7.0, 1],
    ]
)


class IntentValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, len(INTENTS)),
        )
        # A fresh tactical head; the existing neural motor policy is unchanged.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observation):
        return self.net(observation)


class ShotIntentPolicy:
    def __init__(
        self,
        fallback,
        selector=None,
        *,
        controller_options=None,
        interception_options=None,
        fast_prior=False,
    ):
        from airhockey.interception import InterceptionPolicy, InterceptionController

        if interception_options or fast_prior:
            if not isinstance(fallback, InterceptionPolicy):
                raise ValueError(
                    "this tactical configuration requires an interception fallback"
                )
            if interception_options:
                fallback.controller = InterceptionController(**interception_options)
            if fast_prior:
                from airhockey.prior_policy import FrozenPrior

                fallback.policy = FrozenPrior(fallback.policy)
        self.policy, self.cfg = fallback, fallback.cfg
        self.selector = selector if selector is not None else IntentValue().cuda()
        self.controller = ArrivalPlayController(**(controller_options or {}))
        self.selected = self.latched = self.age = None
        self.inference_mode = "prior"
        self.training_algorithm = (
            "Semi-Markov Double DQN shot intents with frozen motor skills"
        )
        self.last_decision = self.last_selected = None

    @property
    def _prev_mean_batch(self):
        return self.policy._prev_mean_batch

    @_prev_mean_batch.setter
    def _prev_mean_batch(self, value):
        self.policy._prev_mean_batch = value

    def decisions(self, observation, t0=False):
        """Advance option lifecycle once per 20 ms action, then mark new choices."""
        x = np.asarray(observation)
        n = len(x)
        if self.selected is None or len(self.selected) != n:
            self.selected = np.zeros(n, int)
            self.latched = np.zeros(n, bool)
            self.age = np.zeros(n)
        reset = np.broadcast_to(np.asarray(t0, bool), n)
        self.latched[reset] = False
        self.selected[reset] = 0
        self.age += 0.02
        released = (x[:, 3] > 1) | (x[:, 1] > 1.2) | (x[:, 32] > 0.5) | (self.age > 4)
        self.latched[released] = False
        ready = (
            (x[:, 32] <= 0.5)
            & (x[:, 1] > self.controller.nav_low[1] + 0.08)
            & (x[:, 1] < self.controller.nav_high[1] + 0.10)
            & (np.linalg.norm(x[:, 2:4], axis=1) < 0.8)
        )
        held = (
            np.zeros(n, bool)
            if self.controller.remaining is None or len(self.controller.remaining) != n
            else self.controller.remaining > 0.02
        )
        return ready & ~self.latched & (~held | reset)

    def act_selected(self, observation, selected, decision, t0=False):
        x = observation.numpy()
        selected = np.asarray(selected)
        if (
            selected.shape != (len(x),)
            or selected.dtype.kind not in "iu"
            or np.any((selected < 0) | (selected >= len(INTENTS)))
        ):
            raise ValueError("shot selections must be valid integer intent indices")
        self.selected[decision] = selected[decision]
        self.latched[decision] = True
        self.age[decision] = 0
        reset = torch.as_tensor(np.broadcast_to(np.asarray(t0, bool), len(x)).copy())
        action = self.policy.act(observation, t0=reset, eval_mode=True).numpy()
        result, _ = self.controller(x, action, t0=t0, intent=INTENTS[self.selected])
        self.last_decision, self.last_selected = decision.copy(), self.selected.copy()
        return torch.from_numpy(result)

    def act(self, observation, t0=False, eval_mode=True):
        decision = self.decisions(observation.numpy(), t0)
        selected = self.selected.copy()
        if decision.any():
            device = next(self.selector.parameters()).device
            with torch.no_grad():
                selected[decision] = (
                    self.selector(observation[decision].to(device))
                    .argmax(-1)
                    .cpu()
                    .numpy()
                )
        return self.act_selected(observation, selected, decision, t0)


class OptionAccumulator:
    """Discount real rewards until the next choice; never learn from fake actions."""

    def __init__(self, n, gamma=0.999):
        self.gamma = gamma
        self.active = np.zeros(n, bool)
        self.obs = np.zeros((n, 42), np.float32)
        self.action = np.zeros(n, np.int64)
        self.reward = np.zeros(n)
        self.discount = np.ones(n)
        self.duration = np.zeros(n, int)

    def start(self, mask, obs, action):
        self.obs[mask], self.action[mask] = obs[mask], action[mask]
        self.reward[mask], self.discount[mask], self.duration[mask] = 0, 1, 0
        self.active[mask] = True

    def add_reward(self, reward):
        self.reward[self.active] += (
            self.discount[self.active] * np.asarray(reward)[self.active]
        )
        self.discount[self.active] *= self.gamma
        self.duration[self.active] += 1

    def finish(self, mask, next_obs, terminal=False):
        ids = np.flatnonzero(np.asarray(mask) & self.active)
        terminal = np.broadcast_to(np.asarray(terminal, bool), len(self.active))
        result = (
            self.obs[ids].copy(),
            self.action[ids].copy(),
            self.reward[ids].astype(np.float32),
            self.discount[ids].astype(np.float32) * ~terminal[ids],
            np.asarray(next_obs)[ids].copy(),
            self.duration[ids].copy(),
        )
        self.active[ids] = False
        return result
