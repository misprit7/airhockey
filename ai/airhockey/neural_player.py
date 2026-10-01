"""A single neural player with no tactical overrides or inference-time teachers.

All six arrival decisions are network outputs. The only deterministic action
processing is the arrival-to-firmware decoder and the kinematic motion guard.
Training task labels, rewards and privileged collision data are never inputs.
An optional three-way shot request conditions the same actor.
"""

import copy
import numpy as np
import torch
from torch import nn


def recovery_exploration_window(context, seconds, practice_mask=None):
    """Training-only mask for a brief exploration burst before cushioning.

    Context contains task one-hot columns and drill age divided by four.
    Rollout collection and PPO reuse the same stored context. No phase cue or
    mask is added to the actor's inputs or deterministic inference.
    """
    if not seconds:
        return practice_mask
    mask = (context[:, 1] > .5) & (context[:, 4] < seconds / 4)
    return mask if practice_mask is None else mask & practice_mask


class RecoveryExplorationBias:
    """Training-only, persistent random offsets before first recovery contact.

    Offsets are exogenous exploration state, not actor outputs or inputs. PPO
    must retain the realized offset and use mean+offset in BOTH likelihoods.
    Its sampling law has no learned parameters, so it cancels from the ratio.
    A deterministic deployed actor never creates or uses this object.
    """

    def __init__(self, n, strength, block_steps=8, device="cpu", load_aware=True, mode="recovery"):
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("persistent exploration strength must be positive and finite")
        if isinstance(block_steps, bool) or not isinstance(block_steps, int) or block_steps < 1:
            raise ValueError("persistent exploration block must contain whole positive steps")
        self.values = torch.zeros((n, 6), device=device)
        self.remaining = torch.zeros(n, dtype=torch.long, device=device)
        if mode not in ("recovery", "edge"):
            raise ValueError("unrecognized persistent exploration mode")
        self.mode = mode
        amplitude = [.8, 1., .3, .3, .2, .2] if mode == "edge" else [.6, .4, .3, .3, 0., 0.]
        self.amplitude = self.values.new_tensor(amplitude) * strength
        if mode == "edge":
            from airhockey.dynamics import workspace_in_sim
            self.workspace = workspace_in_sim()
        self.block_steps = block_steps
        self.load_aware = load_aware

    def reset(self, mask=None):
        if mask is None:
            self.values.zero_()
            self.remaining.zero_()
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.values.device)
            self.values[mask] = 0
            self.remaining[mask] = 0

    @torch.no_grad()
    def sample(self, observation, context, contacted):
        speed = observation[:, 2:4].square().sum(-1).sqrt() * 6
        if self.mode == "edge":
            w = self.workspace
            fringe = ((observation[:, 0] < w['min_x'] + .04)
                      | (observation[:, 0] > w['max_x'] - .04))
            # Random, coherent exploration in quiet side-fringe practice.
            # No prescribed direction, wall tap or staging sequence; full games
            # and deterministic inference do not use these offsets.
            eligible = ((context[:, 0] > .5) & fringe & (speed < .5)
                        & (observation[:, 1] > w['min_y'])
                        & (observation[:, 1] < w['max_y']))
        else:
            eligible = ((context[:, 1] > .5) & ~contacted.bool()
                        & (observation[:, 1] < .8) & (observation[:, 3] * 6 > .05)
                        & (speed < 2))
        self.remaining[~eligible] = 0
        fresh = eligible & (self.remaining == 0)
        self.values[fresh] = torch.randn_like(self.values[fresh]) * self.amplitude
        self.remaining[fresh] = self.block_steps
        self.remaining[eligible] -= 1
        result = self.values * eligible[:, None]
        if self.load_aware:
            reserve = ((1 - observation[:, 21:29].amax(-1)) / .2).clamp(0, 1)
            result = result * reserve[:, None]
        return result


def recovery_exploration_scale(observation, log_std, floor, load_aware=False, quiet=False, mean=None, timing=False, practice_mask=None):
    """Training-only exploration for new recovery/quiet-state maneuvers.

    Apply identically during collection and PPO likelihood evaluation. The
    deterministic deployed actor and its learned variance are not modified.
    """
    if not floor:
        return log_std
    speed = observation[:, 2:4].square().sum(-1).sqrt()*6
    gap = (observation[:, :2]-observation[:, 4:6]).square().sum(-1).sqrt()
    explore = (observation[:, 1] < .8) & (speed < 2) & ((observation[:, 3]*6 > .05) | ((speed < .1) & (gap > .18)))
    nearby_quiet = quiet & (observation[:, 1] < .8) & (speed < .1) & (gap <= .18)
    explore |= nearby_quiet
    minimum = log_std.new_tensor(float(np.log(floor)))
    tail_minimum = minimum
    if quiet:
        minimum = torch.where(nearby_quiet[:,None], minimum-np.log(2), minimum)
        tail_minimum = minimum
        if mean is not None:
            # A raw Gaussian near a saturated tanh output barely moves the
            # physical target. Explore quiet states in a useful action range.
            # This is the same state-dependent Gaussian in rollout and PPO.
            quiet_state = (observation[:, 1] < .8) & (speed < .1)
            amplify = (1 / (1-mean[:,:4].tanh().square()+.1)).clamp(1,4)
            minimum = minimum + torch.where(quiet_state[:,None], amplify.log(), 0)
    if load_aware:
        reserve = ((1-observation[:,21:29].amax(-1))/.2).clamp(0,1)
        minimum = (reserve[:,None]*minimum.exp()).clamp_min(1e-6).log()
        tail_minimum = (reserve[:,None]*tail_minimum.exp()).clamp_min(1e-6).log()
    first = torch.where(explore[:, None], torch.maximum(log_std[:, :4], minimum), log_std[:, :4])
    tail = torch.where(explore[:,None], torch.maximum(log_std[:,4:],tail_minimum),log_std[:,4:]) if timing else log_std[:,4:]
    result = torch.cat((first, tail), dim=-1)
    return result if practice_mask is None else torch.where(practice_mask[:, None], result, log_std)


def defensive_request_consistency(net, observation, mean=None):
    """Training loss: an expired shot request must not choose the defense.

    Requests are redrawn when the puck enters the robot half. While it is in
    the opponent half, compare the actor on the exact same physical observation
    with a different stale request. This supplies neither an action target nor
    any inference-time override; own-half shot conditioning remains trained by
    its normal outcome rewards.
    """
    if mean is None:
        mean = net.actor(net.trunk(observation))
    far = observation[:, 1] > 1.0
    if not net.shot_conditioned or not far.any():
        return mean.sum() * 0
    augmented = observation[far].clone()
    request = torch.roll(augmented[:, -3:], 1, dims=-1)
    request[request.sum(-1) == 0, 0] = 1
    augmented[:, -3:] = request
    alternative = net.actor(net.trunk(augmented))
    return (mean[far].tanh() - alternative.tanh()).square().mean()


def thermal_effort_exploration_scale(observation, log_std, floor, practice_mask=None):
    """Explore effort in hot simulation states even when its tanh is saturated.

    This broadens the training Gaussian only; it supplies neither a lower cap
    nor a runtime action. PPO uses exactly this same likelihood in its update.
    """
    if not floor:
        return log_std
    hot = ((observation[:, 21:29].amax(-1) - .8) / .15).clamp(0, 1)
    minimum = (hot * floor).clamp_min(1e-6).log()
    effort = torch.maximum(log_std[:, 5], minimum)
    result = torch.cat((log_std[:, :5], effort[:, None]), dim=-1)
    return result if practice_mask is None else torch.where(practice_mask[:, None], result, log_std)


def established_skill_preservation(net, reference, observation, mean=None, controlled=None, max_load=None, incoming_only=False):
    """Training-only neural reference for stationary shots/fast own-half returns.

    Uncontrolled slow outgoing recovery and far-half preparation are unconstrained.
    The reference is a previously learned actor, never a deployment ensemble.
    """
    if mean is None:
        mean = net.actor(net.trunk(observation))
    speed = observation[:, 2:4].square().sum(-1).sqrt() * 6
    fast_incoming = (speed > 2) & (observation[:, 3] < 0)
    keep = fast_incoming if incoming_only else (observation[:, 1] < .95) & fast_incoming
    if not incoming_only:
        keep |= (observation[:, 1] < .95) & (speed < .1)
    if controlled is not None and not incoming_only:
        # Privileged capture bookkeeping chooses training examples only; it
        # never enters either actor. Retain the learned strike after recovery.
        keep |= (observation[:, 1] < .95) & controlled.bool()
    if max_load is not None:
        # Do not freeze a reference's unqualified hot-state behavior.
        keep &= observation[:, 21:29].amax(-1) <= max_load
    if not keep.any():
        return mean.sum() * 0
    with torch.no_grad():
        target = (reference.action_mean(observation[keep]) if hasattr(reference,'action_mean')
                  else reference.actor(reference.trunk(observation[keep])).tanh())
    return (mean[keep].tanh() - target).square().mean()


class PhysicalHistory:
    """Newest-first observed frames; resets never leak another episode's state."""

    def __init__(self, frames=1):
        if not isinstance(frames, int) or not 1 <= frames <= 16:
            raise ValueError("physical history must contain 1–16 frames")
        self.frames = frames
        self.values = None
        self.request = None

    def _split(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim != 2 or observation.shape[1] not in (42, 45):
            raise ValueError("42 physical features and optionally three shot-request features required")
        return observation[:, :42], observation[:, 42:] if observation.shape[1] == 45 else None

    def reset(self, observation, mask=None):
        observation, request = self._split(observation)
        if self.values is None:
            self.values = np.repeat(observation[:, None, :], self.frames, axis=1)
        elif mask is None:
            self.values[:] = observation[:, None, :]
        else:
            self.values[mask] = observation[mask, None, :]
        if request is not None:
            if self.request is None or mask is None:
                self.request = request.copy()
            else:
                self.request[mask] = request[mask]
        elif self.request is not None:
            raise ValueError("cannot remove shot-request features during a rollout")
        return self.get()

    def append(self, observation):
        if self.values is None:
            return self.reset(observation)
        observation, request = self._split(observation)
        self.values[:, 1:] = self.values[:, :-1].copy()
        self.values[:, 0] = observation
        self.request = None if request is None else request.copy()
        return self.get()

    def get(self, obs_dim=None):
        result = self.values.reshape(len(self.values), -1)
        if self.request is not None:
            result = np.column_stack((result, self.request))
        return result if obs_dim is None else result[:, :obs_dim]

    def for_policy(self, net):
        """Select physical history and current request for mixed old/new rivals."""
        if net.history > self.frames:
            raise ValueError("insufficient physical history")
        physical = self.values[:, :net.history].reshape(len(self.values), -1)
        if not net.shot_conditioned:
            return physical
        if self.request is None:
            raise ValueError("conditioned policy requires a shot-request observation")
        return np.column_stack((physical, self.request))


class ActorPrefixFreeze:
    """Training-only protection for an existing subnetwork after widening.

    Added hidden units and output connections learn corrections inside the
    same dense neural actor. The saved policy needs no gate or controller.
    Projection also protects frozen entries from pre-existing Adam moments.
    """

    def __init__(self, net, width):
        if isinstance(width, bool) or not isinstance(width, int) or not 0 < width < net.width:
            raise ValueError("frozen actor width must be smaller than the expanded network")
        if any(torch.count_nonzero(layer.weight[:width, width:])
               for layer in (net.trunk[2], net.trunk[4])):
            raise ValueError("actor prefix must be independent; start with function-preserving widening")
        self.blocks = []
        self.handles = []

        def protect(parameter, index):
            mask = torch.ones_like(parameter)
            mask[index] = 0
            self.blocks.append((parameter, index, parameter[index].detach().clone()))
            self.handles.append(parameter.register_hook(lambda grad, mask=mask: grad * mask))

        for layer in (net.trunk[0], net.trunk[2], net.trunk[4]):
            protect(layer.weight, (slice(0, width), slice(None)))
            protect(layer.bias, slice(0, width))
        for head in (net.actor, net.noise):
            protect(head.weight, (slice(None), slice(0, width)))
            protect(head.bias, slice(None))
        protect(net.log_std, slice(None))

    @torch.no_grad()
    def restore(self):
        for parameter, index, initial in self.blocks:
            parameter[index].copy_(initial)


class PhysicalHistoryLinear(nn.Linear):
    """Optional temporal differences; callers still supply physical histories.

    Current physical features, shot request and critic context retain their
    layout. Only past physical frames are expressed relative to the current
    frame. Legacy checkpoints keep the original raw-frame computation.
    """

    def __init__(self, inputs, outputs, history, deltas=False):
        super().__init__(inputs, outputs)
        self.history = history
        self.deltas = deltas

    def forward(self, observation):
        if self.deltas and self.history > 1:
            physical = 42 * self.history
            current = observation[..., :42]
            past = observation[..., 42:physical].reshape(
                *observation.shape[:-1], self.history - 1, 42)
            differences = (past - current.unsqueeze(-2)).flatten(-2)
            observation = torch.cat((current, differences, observation[..., physical:]), dim=-1)
        return nn.functional.linear(observation, self.weight, self.bias)


class NeuralPlayer(nn.Module):
    obs_dim = 42
    action_dim = 6
    critic_context_dim = 9

    def __init__(self, width=256, history=1, shot_conditioned=False, history_deltas=False):
        super().__init__()
        if not isinstance(history, int) or not 1 <= history <= 16:
            raise ValueError("physical history must contain 1–16 frames")
        self.history = history
        self.history_deltas = bool(history_deltas)
        self.register_buffer("_history_delta_encoding", torch.tensor(int(self.history_deltas)))
        self.shot_conditioned = bool(shot_conditioned)
        self.physical_obs_dim = 42 * history
        self.obs_dim = self.physical_obs_dim + 3 * self.shot_conditioned
        self.width = width
        self.trunk = nn.Sequential(
            PhysicalHistoryLinear(self.obs_dim, width, history, self.history_deltas),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
        )
        self.actor = nn.Linear(width, self.action_dim)
        # Value estimation is training-only. Its gradients must not change the
        # actor's representation or swamp precise contact-policy gradients.
        self.value_trunk = copy.deepcopy(self.trunk)
        self.value_trunk[0] = PhysicalHistoryLinear(
            self.obs_dim + self.critic_context_dim, width, history, self.history_deltas)
        self.critic = nn.Linear(width, 1)
        self.noise = nn.Linear(width, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.6))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1)
        nn.init.zeros_(self.noise.weight)
        nn.init.zeros_(self.noise.bias)

    def value(self, obs, context=None):
        if context is None:
            context = obs.new_zeros((*obs.shape[:-1], self.critic_context_dim))
        return self.critic(self.value_trunk(torch.cat((obs, context), dim=-1))).squeeze(
            -1
        )

    def forward(self, obs, context=None):
        features = self.trunk(obs)
        return self.actor(features), self.value(obs, context)

    def distribution(self, obs, context=None):
        features = self.trunk(obs)
        log_std = (self.log_std + self.noise(features)).clamp(-4, 0.5)
        return (
            self.actor(features),
            self.value(obs, context),
            log_std,
        )

    def load_weights(self, state):
        state = dict(state)
        marker = state.pop("_history_delta_encoding", None)
        source_deltas = int(marker.item()) if marker is not None else 0
        if source_deltas not in (0, 1):
            raise ValueError("unrecognized physical history encoding")
        # Inference constructors inherit checkpoint encoding automatically.
        # Training can explicitly migrate a raw-frame checkpoint to differences.
        self.history_deltas = self.history_deltas or bool(source_deltas)
        self.trunk[0].deltas = self.value_trunk[0].deltas = self.history_deltas
        self._history_delta_encoding.fill_(int(self.history_deltas))
        state["_history_delta_encoding"] = self._history_delta_encoding.clone()
        upgraded = "noise.weight" not in state and "noise.bias" not in state
        if upgraded:
            # Old neural-only checkpoints retain their exact deterministic
            # actions; a zero new head starts with their old global variance.
            state["noise.weight"] = torch.zeros_like(self.noise.weight)
            state["noise.bias"] = torch.zeros_like(self.noise.bias)
        if not any(k.startswith("value_trunk.") for k in state):
            for k, v in list(state.items()):
                if k.startswith("trunk."):
                    state["value_" + k] = v.clone()
            upgraded = True
        old_dim = state["trunk.0.weight"].shape[1]
        old_history, old_request = divmod(old_dim, 42)
        if old_history < 1 or old_request not in (0, 3):
            raise ValueError("unrecognized actor observation layout")
        old_physical = 42 * old_history
        if old_history > self.history:
            raise ValueError("cannot discard a checkpoint's learned physical history")
        if old_request and not self.shot_conditioned:
            raise ValueError("cannot discard a checkpoint's learned shot conditioning")
        if old_dim != self.obs_dim:
            padded = state["trunk.0.weight"].new_zeros((state["trunk.0.weight"].shape[0], self.obs_dim))
            padded[:, :old_physical] = state["trunk.0.weight"][:, :old_physical]
            if old_request:
                padded[:, self.physical_obs_dim:] = state["trunk.0.weight"][:, old_physical:]
            state["trunk.0.weight"] = padded
            upgraded = True
        old_input = state["value_trunk.0.weight"]
        if old_input.shape[1] != self.obs_dim + self.critic_context_dim:
            padded = old_input.new_zeros((old_input.shape[0], self.obs_dim + self.critic_context_dim))
            padded[:, :old_physical] = old_input[:, :old_physical]
            if old_request:
                padded[:, self.physical_obs_dim:self.obs_dim] = old_input[:, old_physical:old_dim]
            if old_input.shape[1] == old_dim + self.critic_context_dim:
                padded[:, self.obs_dim:] = old_input[:, old_dim:]
            elif old_input.shape[1] != old_dim:
                raise ValueError("unrecognized value-network observation layout")
            state["value_trunk.0.weight"] = padded
            upgraded = True
        if self.history_deltas and not source_deltas and self.history > 1:
            # W0*x0 + sum(Wi*xi) becomes
            # (W0+sum(Wi))*x0 + sum(Wi*(xi-x0)). Preserve actor, noise and
            # critic predictions, including request/context columns. Clone so
            # migration never mutates a caller's source network/state tensors.
            for key in ("trunk.0.weight", "value_trunk.0.weight"):
                weight = state[key].clone()
                weight[:, :42] = state[key][:, :self.physical_obs_dim].reshape(
                    weight.shape[0], self.history, 42).sum(dim=1)
                state[key] = weight
            upgraded = True  # Parameter coordinates changed: reset Adam moments.
        # Function-preserving widening: existing hidden units keep their
        # connections; new units start with random features and zero influence
        # on existing units/output heads. Their outgoing weights can then learn.
        initialized = self.state_dict()
        for key, value in list(state.items()):
            target = initialized[key]
            if value.shape == target.shape:
                continue
            if value.ndim != target.ndim or any(a > b for a, b in zip(value.shape, target.shape)):
                raise ValueError("cannot discard a checkpoint's learned network capacity")
            widened = target.clone()
            if value.ndim == 2:
                widened[:value.shape[0], :] = 0
                widened[:value.shape[0], :value.shape[1]] = value
            elif value.ndim == 1:
                widened[:len(value)] = value
            else:
                raise ValueError("unsupported parameter shape migration")
            state[key] = widened
            upgraded = True
        self.load_state_dict(state, strict=True)
        return upgraded

    def actor_parameters(self):
        return [
            self.log_std,
            *self.trunk.parameters(),
            *self.actor.parameters(),
            *self.noise.parameters(),
        ]

    def value_parameters(self):
        return [*self.value_trunk.parameters(), *self.critic.parameters()]

    @torch.no_grad()
    def act(self, obs, *, stochastic=False):
        device = next(self.parameters()).device
        features = self.trunk(torch.as_tensor(obs, dtype=torch.float32, device=device))
        mean = self.actor(features)
        if stochastic:
            log_std = (self.log_std + self.noise(features)).clamp(-4, 0.5)
            mean = mean + log_std.exp() * torch.randn_like(mean)
        return mean.tanh().cpu().numpy()


def log_probability(raw, mean, log_std):
    # Jacobian cancels for PPO's same-action likelihood ratios.
    return (-0.5 * ((raw - mean) * (-log_std).exp()).square() - log_std).sum(-1)


@torch.no_grad()
def backtrack_actor_step(parameters, before, measure_kl, limit, max_backtracks=12):
    """Shorten an optimizer proposal that exceeds the sampled PPO KL budget.

    Only the supplied actor parameters move. Adam's gradient moments are kept,
    as with a line search along an optimizer direction; the critic step is kept
    independently. This bounds the measured minibatch change, not every state
    or physical behavior. A completely rejected proposal restores the actor.
    """
    parameters = list(parameters)
    if len(parameters) != len(before) or not np.isfinite(limit) or limit <= 0:
        raise ValueError("actor backtracking needs matching snapshots and a positive KL limit")
    kl = float(measure_kl())
    if np.isfinite(kl) and kl <= limit:
        return 1.0, kl
    direction = [p.detach() - old for p, old in zip(parameters, before)]
    for attempt in range(1, max_backtracks + 1):
        scale = 2.0 ** -attempt
        for parameter, old, delta in zip(parameters, before, direction):
            parameter.copy_(old + scale * delta)
        kl = float(measure_kl())
        if np.isfinite(kl) and kl <= limit:
            return scale, kl
    for parameter, old in zip(parameters, before):
        parameter.copy_(old)
    return 0.0, float(measure_kl())
