"""Simulation curriculum with isolated defense and measured promotion gates."""

import numpy as np
from airhockey.legacy_practice import LegacyPracticeEnv
from airhockey.policy_benchmark import fixtures, random_paddle_starts
from airhockey.shot_flight import PARAMETERS, first_goal_crossing
from airhockey.setup_shaping import setup_potential, potential_reward


class FoundationEnv(LegacyPracticeEnv):
    def __init__(
        self, *args, possession_fraction=0.0, setup_potential_weight=0.0, **kwargs
    ):
        super().__init__(*args, motion_guard=True, **kwargs)
        if not 0 <= possession_fraction <= 1:
            raise ValueError("possession fraction must be in [0,1]")
        self.possession_fraction = possession_fraction
        if not np.isfinite(setup_potential_weight) or setup_potential_weight < 0:
            raise ValueError("setup potential weight must be finite and nonnegative")
        self.setup_potential_weight = setup_potential_weight
        self.defense_trials = np.zeros(self.n_envs, bool)
        self.base.shot_types = False
        self.base.symmetric_referee = True

    def _contact(self, event):
        idx = event["indices"]
        armed = self.shot_armed[idx].copy()
        prior_reward = self.contact_reward[idx].copy()
        prior_aimed = self.on_target[idx].copy()
        super()._contact(event)
        if event["body"] != "agent":
            return
        outgoing, incoming = event["outgoing_before_speed_cap"], event["incoming"]
        attempt = armed & (outgoing[:, 1] > 1)
        if attempt.any():
            ids = idx[attempt]
            crossing, aimed = first_goal_crossing(
                np.column_stack(
                    (
                        self.engine.puck_x[ids],
                        self.engine.puck_y[ids],
                        outgoing[attempt],
                    )
                ),
                {name: getattr(self.engine, name)[ids] for name in PARAMETERS},
                self.cfg,
            )
            precision = np.exp(-(((crossing - self.aim[ids]) / 0.07) ** 2))
            self.contact_reward[ids] = prior_reward[attempt] + np.where(
                aimed, 3 + 7 * precision, -10
            )
            self.on_target[ids] = prior_aimed[attempt] + aimed
            self.goal_error[ids] = abs(crossing - self.aim[ids])
        # Strengthen accuracy for deliberate accelerating shots. An ordinary
        # defensive deflection is not forced to become a precision countershot.
        shot = (
            armed
            & (self.task[idx] != 2)
            & (outgoing[:, 1] > 1)
            & (
                np.linalg.norm(outgoing, axis=1)
                > np.linalg.norm(incoming, axis=1) + 0.2
            )
        )
        extra = 3 * (self.contact_reward[idx] - prior_reward)
        self.contact_reward[idx[shot]] += extra[shot]

    def reset(self, *, seed=None, mask=None, **kwargs):
        if kwargs.get("fixtures") is not None:
            return super().reset(seed=seed, mask=mask, **kwargs)
        ids = np.arange(self.n_envs) if mask is None else np.flatnonzero(mask)
        n = len(ids)
        bank, _ = fixtures(int(self.rng.integers(2**31)), n, wide=True)
        games = ids < round(self.n_envs * self.game_fraction)
        kinds = ids % 4
        f = bank.take(kinds * n + np.arange(n))
        possession = np.zeros(n, bool)
        if self.possession_fraction:
            possession = (
                (~games) & (kinds < 2) & (self.rng.random(n) < self.possession_fraction)
            )
        if possession.any():
            f.paddle[possession] = random_paddle_starts(
                self.rng,
                f.puck[possession],
                self.base._ws,
                self.cfg.puck_radius + self.cfg.paddle_radius + 0.01,
            )
        f.task[games] = 3
        self.defense_trials[ids] = (~games) & (kinds == 3)
        obs = super().reset(seed=seed, mask=mask, fixtures=f, **kwargs)
        self.task[self.defense_trials] = 3
        self.desired_speed[self.defense_trials] = 3
        obs[self.defense_trials, 30:34] = [0, 0, 0, 1]
        obs[self.defense_trials, 35] = 1
        # Each isolated fixture is a newly sampled situation, not a plausible
        # continuous rally. Back-to-back 600 ms drills otherwise force an
        # impossible duty cycle and overwhelm skill learning with overload.
        # Actual game slots keep heat across points AND episode resets.
        practice = ids[~games]
        for load in self.loads:
            load.h[practice, 0] = self.rng.uniform(0.1, 0.9, (len(practice), 4)) ** 2
            load.h[practice, 1] = self.rng.uniform(0.1, 0.6, (len(practice), 4)) ** 2
            load.observed[practice] = np.round(load.levels[practice] * 100) / 100
        obs[ids, 22:30] = self.loads[0].features()[ids]
        return obs

    def step(self, action):
        potential_before = (
            setup_potential(self._obs) if self.setup_potential_weight else None
        )
        before = self.engine.score_opponent.copy()
        before_for = self.engine.score_agent.copy()
        old_speed = np.hypot(self.engine.puck_vx, self.engine.puck_vy)
        old_contacts = self.contacts.copy()
        obs, reward, term, trunc, info = super().step(action)
        cushion = self.task == 2
        speed = np.hypot(self.engine.puck_vx, self.engine.puck_vy)
        # A cushioning request must not receive the shooting contact bonus.
        reward[cushion] -= self.contact_reward[cushion]
        first_touch = cushion & (old_contacts == 0) & (self.contacts > 0)
        reward[first_touch] += (
            15 * np.clip((old_speed - speed) / np.maximum(old_speed, 0.1), -1, 1)
        )[first_touch]
        gap = np.maximum(
            np.hypot(
                self.engine.puck_x - self.engine.paddle_agent_x,
                self.engine.puck_y - self.engine.paddle_agent_y,
            )
            - self.cfg.puck_radius
            - self.cfg.paddle_radius,
            0,
        )
        reward[cushion] += (
            12 * self.base.action_dt * np.exp(-((speed / 0.6) ** 2) - gap / 0.06)
        )[cushion]
        defense = self.defense_trials
        end = (
            (self.elapsed >= 2 - 1e-8)
            | (self.engine.score_opponent > before)
            | (self.engine.score_agent > before_for)
        )
        success = (self.contacts > 0) & (self.engine.score_opponent == 0)
        # Defense uses the normal game request, so its reward MUST be the
        # normal game reward too. A hidden +75 deadline objective aliases the
        # same observations with conflicting values and corrupts the planner.
        # These are short game rollouts, not actual terminal game states.
        term[defense], trunc[defense] = False, end[defense]
        info["success"][defense] = success[defense]
        info["task"][defense] = 4
        # The firmware containment backstop is not a physically achievable
        # instant stop. A rare guard miss must never earn a goal or save reward.
        unsafe = self.peak_accel > 60.1
        reward[unsafe] = -300 - self.load_cost[unsafe]
        term[unsafe], trunc[unsafe] = True, False
        info["success"][unsafe] = False
        info["motion_violation"] = unsafe
        if self.setup_potential_weight:
            reward += potential_reward(
                potential_before,
                setup_potential(obs),
                term,
                self.setup_potential_weight,
            )
        return obs, reward, term, trunc, info


def skill_gate(result, stage):
    """Validation gates, never driven by step count or training reward."""
    required = (
        dict(stationary=0.85, moving=0.65, cushion=0.25, defense=0.90),
        dict(stationary=0.90, moving=0.75, cushion=0.45, defense=0.95),
        dict(stationary=0.95, moving=0.85, cushion=0.65, defense=0.98),
    )[min(stage, 2)]
    rates = {
        k: v["successes"] / max(1, v["attempts"]) for k, v in result["tasks"].items()
    }
    failures = {
        k: dict(actual=rates[k], required=v)
        for k, v in required.items()
        if rates[k] < v
    }
    for name, row in result["tasks"].items():
        if row["peak_modeled_load"] >= 1:
            failures[name + "_load"] = row["peak_modeled_load"]
        if row["peak_actual_acceleration"] > 60.1:
            failures[name + "_acceleration"] = row["peak_actual_acceleration"]
    for name, threshold in zip(
        ("stationary", "moving"),
        ((0.80, 0.55), (0.85, 0.65), (0.90, 0.80))[min(stage, 2)],
    ):
        row = result["tasks"][name]
        rate = row["on_target_contact_trials"] / max(1, row["attempts"])
        if rate < threshold:
            failures[name + "_on_target"] = dict(actual=rate, required=threshold)
    return not failures, failures


class MixedReplay:
    """Keep demonstrations in a separate immutable buffer as online data grows."""

    def __init__(self, demonstrations, online):
        self.demonstrations, self.online = demonstrations, online

    def sample_with_demo(self):
        import torch

        a = self.demonstrations.sample_with_demo()
        b = (
            self.online.sample_with_demo()
            if self.online.num_eps
            else self.demonstrations.sample_with_demo()
        )
        return tuple(
            None if x is None else torch.cat((x, y), dim=1) for x, y in zip(a, b)
        )
