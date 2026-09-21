"""Preserve the complete legacy policy while adding skill/load observations."""

import numpy as np

from airhockey.policy_benchmark import LegacyTrials


class LegacyPracticeEnv(LegacyTrials):
    """Original 22 features unchanged; append 20 skill/controller/load features."""

    obs_dim = 42
    action_dim = 3

    def __init__(self, *args, motion_guard=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_guard = motion_guard
        self.guard_interventions = np.zeros((2, self.n_envs), int)
        self.guard_unresolved = np.zeros((2, self.n_envs), int)

    def _decode(self, action, opponent=False):
        target, cap = super()._decode(action, opponent)
        if self.motion_guard:
            from airhockey.motion_guard import guard_command
            from airhockey.batch_env import _OPP_POLICY_MAP
            from airhockey.motion import CartState

            dyn = self.base._opp_dyn if opponent else self.base._agent_dyn
            ids = (
                np.flatnonzero(self.base._opp_policy_id == _OPP_POLICY_MAP["external"])
                if opponent
                else np.arange(self.n_envs)
            )
            if not len(ids):
                return target, cap
            previous = np.column_stack((dyn["command_x"], dyn["command_y"]))
            if opponent:
                previous[:, 1] = self.cfg.height - previous[:, 1]
            full = self._cart(dyn, opponent)
            cart = CartState(len(ids))
            for key in CartState.__slots__:
                getattr(cart, key)[:] = getattr(full, key)[ids]
            guarded_target, guarded_cap, changed, unresolved = guard_command(
                cart,
                target[ids],
                cap[ids],
                previous[ids],
                dyn["command_accel"][ids],
                bounds=self.decoder.bounds,
                max_accel=dyn["max_accel"][ids],
                max_speed=dyn["max_speed"][ids],
                delay=self.base.command_delay_s,
                action_dt=self.base.action_dt,
            )
            target[ids], cap[ids] = guarded_target, guarded_cap
            self.guard_interventions[int(opponent), ids] += changed
            self.guard_unresolved[int(opponent), ids] += unresolved
        return target, cap

    def _features(self, base_obs, opponent=False):
        arrival = super()._features(base_obs, opponent)
        return np.concatenate((arrival[:, :18], arrival[:, 21:]), axis=1)

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (self.n_envs, 3) or not np.isfinite(action).all():
            raise ValueError("legacy practice actions must be finite [N,3]")
        return super().step(self.transport(action))

    def set_opponent_action(self, action):
        return super().set_opponent_action(self.transport(action))


def transfer_full_policy(agent, checkpoint):
    """Widen the encoder with zero columns; all existing weights are preserved."""
    import torch

    saved = torch.load(checkpoint, map_location=agent.device, weights_only=False)
    source = saved.get("model", saved)
    target = agent.model.state_dict()
    key = "_encoder.state.0.weight"
    if source[key].shape[1] != 22 or target[key].shape[1] != 42:
        raise ValueError("expected legacy 22 -> practice 42 input mapping")
    migrated = dict(source)
    migrated[key] = torch.zeros_like(target[key])
    migrated[key][:, :22] = source[key]
    agent.model.load_state_dict(migrated, strict=True)
    return dict(
        source=str(checkpoint),
        retained="all heads and original encoder columns",
        new_input_columns="zero initialized",
        optimizer="fresh",
    )


def convert_arrival_demonstrations(episodes, workspace, action_low, action_high):
    """Recover the exact low-level commands actually executed by each teacher.

    At the next observation, queued targets/caps are the commands held during
    the new action interval. This reproduces teacher motion, not an approximate
    six-to-three interpretation of its abstract arrival intent.
    """
    lo = np.array([workspace["min_x"], workspace["min_y"]])
    hi = np.array([workspace["max_x"], workspace["max_y"]])
    result = []
    for episode in episodes:
        obs = episode["obs"]
        xy = lo + (obs[1:, 41:43] + 1) / 2 * (hi - lo)
        cap = 2 * np.sqrt(np.clip((obs[1:, 43] - 0.05) / 0.95, 0, 1)) - 1
        action = np.zeros((len(obs), 3), np.float32)
        action[1:, :2] = 2 * (xy - action_low) / (action_high - action_low) - 1
        action[1:, 2] = cap
        new_obs = np.concatenate((obs[:, :18], obs[:, 21:]), axis=1)
        new_obs[:, 15:18] = action
        result.append(dict(episode, obs=new_obs, action=action))
    return result
