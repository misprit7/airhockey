"""DAgger distillation of a teacher into ONE NeuralPlayer. Simulation/training only.

The teacher (`airhockey.expert_player`) labels every visited state and acts
during part of the collection. The saved student is an ordinary NeuralPlayer:
it carries no teacher, needs none at play, and the teacher never runs at the
table.

Collection. `DistillationEnv` workers each own a NeuralTrainingEnv slice built
from a recorded training command (`train_neural_player.environment_options`),
both seats' newest-first physical histories and a teacher. After every reset or
step a worker returns each robot-seat history with the teacher's action for it,
and the same for the far seat wherever that seat is driven by supplied actions
(full games and practice self-play). The main process decides who acts
(`RollIn`: the teacher with probability beta per environment and block of
decisions, else the student; the far seat by a teacher / production / student
mix), runs the networks on the GPU and keeps the labelled states
(`LabelBuffer`). Labels are always the teacher's action for the observation the
acting policy saw.

Training. Mean squared error between tanh(actor mean) and the teacher action on
the trunk and actor head only, optionally weighted per action dimension and per
teacher mode (threat states are rarer and harder than guard states), with the
first layer optionally fitted in standardized input coordinates and folded
back (`StandardizedInputs`). log_std, the noise head and the critic stay as
loaded, for PPO to start from.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import signal
import time
import traceback

import numpy as np
import torch

from airhockey.neural_vector import ParallelNeuralEnv

TEACHER, PRODUCTION, STUDENT = 0, 1, 2
SOURCES = ("teacher", "production", "student")
MODES = ("threat", "attack", "guard")  # ExpertPlayer.last_mode
KINDS = ("stationary", "receiving", "defense", "game")  # NeuralTrainingEnv.kind
SIDES = ("robot", "far")
TAGS = ("mode", "kind", "side", "actor", "round", "heldout")


def _modes(teacher, count):
    mode = getattr(teacher, "last_mode", None)
    if mode is None or len(mode) != count:
        return np.full(count, -1, np.int8)
    return np.asarray(mode, np.int8)


def _label(teacher, own, far, env, external_id, done, goals, conceded, shutdown):
    obs = own.get()
    label = teacher.act(obs)
    mode = _modes(teacher, len(obs))
    external = env.base._opp_policy_id == external_id
    far_obs = far.get()[external]
    if len(far_obs):
        far_label = teacher.act(far_obs)
        far_mode = _modes(teacher, len(far_obs))
    else:
        far_label, far_mode = np.zeros((0, 6), np.float32), np.zeros(0, np.int8)
    return dict(obs=obs.astype(np.float16), label=np.asarray(label, np.float32), mode=mode,
                kind=env.kind.astype(np.int8), external=external,
                far_obs=far_obs.astype(np.float16), far_label=np.asarray(far_label, np.float32),
                far_mode=far_mode, done=done, goals=np.asarray(goals, np.int8),
                conceded=np.asarray(conceded, np.int8), shutdown=np.asarray(shutdown, bool))


def _worker(connection, count, options):
    # The main process owns Ctrl-C: it stops the round and closes the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        options = dict(options)
        settings = options.pop("distillation")
        from airhockey.batch_env import _OPP_POLICY_MAP
        from airhockey.expert_player import load_expert
        from airhockey.neural_player import PhysicalHistory
        from airhockey.neural_training import NeuralTrainingEnv

        env = NeuralTrainingEnv(count, **options)
        teacher, _ = load_expert(settings["teacher"], settings["environment_from"])
        own, far = PhysicalHistory(settings["history"]), PhysicalHistory(settings["history"])
        external_id = _OPP_POLICY_MAP["external"]
        zero = np.zeros(count, np.int8)
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                own.reset(env.reset(seed=options.get("seed", 0)))
                far.reset(env.opponent_obs())
                done, goals, conceded, shutdown = np.ones(count, bool), zero, zero, zero.astype(bool)
            elif command == "step":
                action, opponent = payload
                env.set_opponent_action(opponent)
                obs, _, terminal, truncated, info = env.step(action)
                own.append(obs)
                far.append(env.opponent_obs())
                done = terminal | truncated
                if done.any():
                    # Same partial-reset order as train_neural_player: the far
                    # view is re-read only for the environments that reset.
                    own.reset(env.reset(mask=done), mask=done)
                    far.reset(env.opponent_obs(), mask=done)
                goals, conceded, shutdown = info["goals"], info["conceded"], info["shutdown"]
            else:
                raise ValueError(command)
            connection.send(_label(teacher, own, far, env, external_id, done, goals, conceded, shutdown))
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


@contextmanager
def _thread_limit(threads):
    """Spawned workers read their BLAS/OpenMP thread counts from the environment."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: str(threads) for k in keys})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass
class Labelled:
    """Both seats' observations after a reset/step, each with the teacher's action."""
    obs: np.ndarray        # [N, D] float16 robot-seat physical histories (+ request)
    label: np.ndarray      # [N, 6] teacher action for obs
    mode: np.ndarray       # [N] teacher mode, -1 if the teacher reports none
    kind: np.ndarray       # [N] 0 stationary, 1 receiving, 2 defense, 3 full game
    external: np.ndarray   # [N] far seat driven by supplied actions
    far: np.ndarray        # [E] environment index of each external far seat
    far_obs: np.ndarray    # [E, D] float16 far-seat histories, in the far seat's own frame
    far_label: np.ndarray  # [E, 6]
    far_mode: np.ndarray   # [E]
    done: np.ndarray       # [N] episode ended at this step (obs is the reset state)
    goals: np.ndarray      # [N] robot goals during the step
    conceded: np.ndarray   # [N]
    shutdown: np.ndarray   # [N] thermal training shutdown during the step


class DistillationEnv(ParallelNeuralEnv):
    """ParallelNeuralEnv whose workers return teacher-labelled histories of both seats.

    Same process layout, seeds (seed + i*100003 per worker) and environment
    options as the PPO trainer's. Different protocol: `reset()` and
    `step(action, opponent_action)` return a Labelled batch.
    """

    worker = staticmethod(_worker)

    def __init__(self, n_envs, workers, *, teacher, environment_from, history=16, worker_threads=1, **options):
        settings = dict(teacher=str(teacher), environment_from=str(environment_from), history=int(history))
        with _thread_limit(worker_threads):
            super().__init__(n_envs, workers, distillation=settings, **options)

    def _gather(self):
        parts = self.receive()
        cat = lambda key: np.concatenate([p[key] for p in parts])
        external = cat("external")
        return Labelled(obs=cat("obs"), label=cat("label"), mode=cat("mode"), kind=cat("kind"),
                        external=external, far=np.flatnonzero(external), far_obs=cat("far_obs"),
                        far_label=cat("far_label"), far_mode=cat("far_mode"), done=cat("done"),
                        goals=cat("goals"), conceded=cat("conceded"), shutdown=cat("shutdown"))

    def reset(self):
        for connection in self.connections:
            connection.send(("reset", None))
        return self._gather()

    def step(self, action, opponent_action):
        action = np.asarray(action, np.float32)
        opponent_action = np.asarray(opponent_action, np.float32)
        if action.shape != (self.n_envs, 6) or opponent_action.shape != (self.n_envs, 6):
            raise ValueError("expected [N,6] actions for both seats")
        for connection, own, far in zip(self.connections, np.split(action, self.workers),
                                        np.split(opponent_action, self.workers)):
            connection.send(("step", (own, far)))
        return self._gather()


def parse_floats(text, count=None):
    values = [float(v) for v in str(text).split(",") if v.strip()]
    if not values or (count is not None and len(values) != count) or not np.isfinite(values).all():
        raise ValueError(f"expected {count or 'some'} comma-separated numbers, got {text!r}")
    return values


def parse_mix(text):
    """'teacher,production,student' far-seat weights -> probabilities."""
    mix = np.asarray(parse_floats(text, 3))
    if (mix < 0).any() or mix.sum() <= 0:
        raise ValueError("opponent mix weights must be nonnegative with a positive sum")
    return mix / mix.sum()


class RollIn:
    """Who acts, per environment, in blocks of decisions.

    The robot seat is the teacher with probability beta for a block of
    `block_steps` decisions, else the student; the far seat is drawn from the
    (teacher, production, student) mix for `opponent_block_steps`. A block is
    redrawn when it expires or the episode ends. `restart()` (a new round:
    new beta, new student) redraws every block with staggered first lengths.
    """

    def __init__(self, n, rng, block_steps=25, opponent_block_steps=250):
        if min(block_steps, opponent_block_steps) < 1:
            raise ValueError("blocks must contain at least one decision")
        self.rng = rng
        self.block_steps, self.opponent_block_steps = int(block_steps), int(opponent_block_steps)
        self.teacher = np.ones(n, bool)
        self.opponent = np.zeros(n, np.int8)
        self._left = np.zeros(n, int)
        self._opponent_left = np.zeros(n, int)
        self._fresh = np.ones(n, bool)
        self._opponent_fresh = np.ones(n, bool)

    def restart(self):
        self._left[:] = self._opponent_left[:] = 0
        self._fresh[:] = self._opponent_fresh[:] = True

    def _lengths(self, fresh, block):
        lengths = np.full(len(fresh), block)
        lengths[fresh] = self.rng.integers(1, block + 1, fresh.sum())
        return lengths

    def update(self, beta, mix, done):
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0,1]")
        draw = (self._left <= 0) | done
        if draw.any():
            self.teacher[draw] = self.rng.random(draw.sum()) < beta
            self._left[draw] = self._lengths(self._fresh[draw], self.block_steps)
            self._fresh[draw] = False
        draw = (self._opponent_left <= 0) | done
        if draw.any():
            self.opponent[draw] = self.rng.choice(3, draw.sum(), p=mix)
            self._opponent_left[draw] = self._lengths(self._opponent_fresh[draw], self.opponent_block_steps)
            self._opponent_fresh[draw] = False
        self._left -= 1
        self._opponent_left -= 1


class LabelBuffer:
    """The most recent `capacity` labelled states.

    Observations are float16 (675 x 2 bytes per state); teacher actions float32;
    tags int16 (TAGS: teacher mode, drill kind, seat, who acted / far-seat
    source, round, held-out flag).
    """

    def __init__(self, capacity, obs_dim, device="cpu"):
        if capacity < 1:
            raise ValueError("buffer capacity must be positive")
        self.capacity, self.obs_dim = int(capacity), int(obs_dim)
        self.obs = torch.empty((self.capacity, self.obs_dim), dtype=torch.float16, device=device)
        self.label = torch.empty((self.capacity, 6), dtype=torch.float32, device=device)
        self.tags = torch.empty((self.capacity, len(TAGS)), dtype=torch.int16, device=device)
        self.size = self.cursor = self.added = 0

    def add(self, obs, label, **tags):
        n = len(obs)
        if not n:
            return 0
        if set(tags) != set(TAGS):
            raise ValueError(f"tags must be exactly {TAGS}")
        if np.shape(obs) != (n, self.obs_dim) or np.shape(label) != (n, 6):
            raise ValueError("observation/label shape mismatch")
        values = np.column_stack([np.broadcast_to(np.asarray(tags[k]), (n,)) for k in TAGS]).astype(np.int16)
        obs = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float16))
        label = torch.from_numpy(np.ascontiguousarray(label, dtype=np.float32))
        values = torch.from_numpy(values)
        if n > self.capacity:
            obs, label, values = obs[-self.capacity:], label[-self.capacity:], values[-self.capacity:]
        written = len(obs)
        first = min(written, self.capacity - self.cursor)
        for target, source in ((self.obs, obs), (self.label, label), (self.tags, values)):
            target[self.cursor:self.cursor + first] = source[:first]
            target[:written - first] = source[first:]
        self.cursor = (self.cursor + written) % self.capacity
        self.size = min(self.size + written, self.capacity)
        self.added += n
        return n

    def tag(self, name):
        return self.tags[:self.size, TAGS.index(name)]

    def indices(self, heldout, rounds=None):
        mask = self.tag("heldout") == int(heldout)
        if rounds is not None:
            mask &= torch.isin(self.tag("round"), torch.as_tensor(list(rounds), dtype=torch.int16, device=mask.device))
        return torch.nonzero(mask).squeeze(1)


class StandardizedInputs(torch.nn.Module):
    """Training-time reparameterization of the first layer on standardized inputs.

    The layer computes W x + b on its own inputs x (current frame, past-minus-
    current frames, request). Training W' = W*std and b' = b + W@mean on
    z = (x - mean)/std gives every input column a comparable Adam step, whatever
    its scale (positions, velocities/6, millimetre frame differences, loads).
    `fold()` writes the identical W, b back: the saved network is an unchanged
    NeuralPlayer with no normalization layer. Statistics are fixed once, so
    the Adam moments of W', b' stay meaningful across rounds.
    """

    def __init__(self, layer, mean, std):
        super().__init__()
        self.history, self.deltas = layer.history, layer.deltas
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.weight = torch.nn.Parameter(torch.empty_like(layer.weight))
        self.bias = torch.nn.Parameter(torch.empty_like(layer.bias))
        self.load_from(layer)

    @classmethod
    def from_buffer(cls, layer, buffer, *, samples=100_000, floor=1e-2, generator=None):
        from airhockey.neural_player import history_inputs
        ids = buffer.indices(heldout=False)
        ids = ids[torch.randperm(len(ids), generator=generator)[:samples].to(ids.device)]
        x = history_inputs(buffer.obs[ids].to(layer.weight.device).float(), layer.history, layer.deltas)
        return cls(layer, x.mean(0), x.std(0).clamp_min(floor))

    @torch.no_grad()
    def load_from(self, layer):
        self.weight.copy_(layer.weight * self.std)
        self.bias.copy_(layer.bias + layer.weight @ self.mean)

    def forward(self, observation):
        from airhockey.neural_player import history_inputs
        z = (history_inputs(observation, self.history, self.deltas) - self.mean) / self.std
        return torch.nn.functional.linear(z, self.weight, self.bias)

    @torch.no_grad()
    def fold(self, layer):
        weight = self.weight / self.std
        layer.weight.copy_(weight)
        layer.bias.copy_(self.bias - weight @ self.mean)


def actor_parameters(net, standardizer=None):
    """What distillation trains: the actor trunk and mean head, nothing else.

    With a StandardizedInputs, its W', b' stand in for the first layer's.
    """
    if standardizer is None:
        return [*net.trunk.parameters(), *net.actor.parameters()]
    return [standardizer.weight, standardizer.bias, *net.trunk[1:].parameters(), *net.actor.parameters()]


def widened(net, width):
    """A NeuralPlayer copy at a larger hidden width with identical outputs.

    NeuralPlayer.load_weights gives added units random inputs and zero
    influence; their outgoing weights then learn. Checked on random inputs.
    """
    from airhockey.neural_player import NeuralPlayer
    if width < net.width:
        raise ValueError("cannot narrow a network")
    source = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    wide = NeuralPlayer(width, history=net.history, shot_conditioned=net.shot_conditioned,
                        history_deltas=net.history_deltas)
    wide.load_weights(source)
    wide.environment_options = getattr(net, "environment_options", {})
    narrow = NeuralPlayer(net.width, history=net.history, shot_conditioned=net.shot_conditioned)
    narrow.load_weights(source)
    probe = 0.3 * torch.randn(256, net.obs_dim, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        difference = (wide.actor(wide.trunk(probe)) - narrow.actor(narrow.trunk(probe))).abs().max()
    if difference > 1e-4:
        raise AssertionError(f"widening changed the actor by {float(difference)}")
    return wide.eval()


def distillation_loss(net, obs, target, weights, sample_weights=None):
    """Weighted MSE between tanh(actor mean) and the teacher action.

    `weights` scale the six action dimensions (mean 1); optional per-sample
    weights (e.g. by teacher mode) give a weighted mean over samples.
    """
    error = ((net.actor(net.trunk(obs)).tanh() - target).square() * weights).mean(-1)
    if sample_weights is None:
        return error.mean()
    return (error * sample_weights).sum() / sample_weights.sum()


LOAD_COLUMNS = slice(21, 29)  # current-frame observed motor levels (neural_observation)


def observed_load(obs):
    """Highest of the eight observed motor levels in each current frame."""
    return obs[:, LOAD_COLUMNS].float().amax(1)


def load_balance_table(buffer, ids, cap, bins=20):
    """Per-band weights equalizing the observed-load distribution, at most `cap`.

    The PPO curriculum's warm starts crowd the samples into 0.75-0.94 load and
    leave 0.5-0.75 sparse, which is where the teacher's guard depth bends. The
    band weights make every occupied band carry similar total weight.
    """
    counts = torch.zeros(bins)
    for part in ids.split(1 << 20):
        level = observed_load(buffer.obs[part]).clamp(0, 1 - 1e-6)
        counts += torch.bincount((level * bins).long().cpu(), minlength=bins).float()
    target = counts.sum() / (counts > 0).sum().clamp_min(1)
    return torch.where(counts > 0, (target / counts.clamp_min(1)).clamp(max=cap), torch.zeros(()))


def learning_rate(update, total, lr, final_fraction, warmup=0):
    """Linear warm-up, then cosine from lr to lr*final_fraction over `total` updates."""
    if update < warmup:
        return lr * (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return lr * (final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))


def fit(net, optimizer, buffer, *, epochs, batch_size, lr, final_fraction, weights, device,
        warmup=0, clip=1.0, generator=None, log=None, log_every=500, mode_weights=None,
        standardizer=None, load_balance=0.0):
    """Train the actor on every non-held-out state in the buffer for `epochs` passes.

    `mode_weights` (threat, attack, guard) weight samples by the teacher's mode;
    a sample whose mode is unknown weighs 1. `load_balance` > 0 multiplies in
    observed-load band weights capped at that value. With a StandardizedInputs
    the first layer trains in standardized coordinates and is folded back.
    """
    ids = buffer.indices(heldout=False)
    if not len(ids):
        raise ValueError("no training samples")
    per_epoch = math.ceil(len(ids) / batch_size)
    total = epochs * per_epoch
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    by_mode = None if mode_weights is None else torch.as_tensor(
        [1.0, *mode_weights], dtype=torch.float32, device=device)  # index: mode + 1
    mode_column = TAGS.index("mode")
    bands = load_balance_table(buffer, ids, load_balance).to(device) if load_balance else None
    layer = net.trunk[0]
    if standardizer is not None:
        standardizer.load_from(layer)
        net.trunk[0] = standardizer
    parameters = actor_parameters(net, standardizer)
    net.train()
    started, update, losses = time.monotonic(), 0, []
    try:
        for epoch in range(epochs):
            order = ids[torch.randperm(len(ids), generator=generator).to(ids.device)]
            running = torch.zeros((), device=device)
            for chunk in order.split(batch_size):
                rate = learning_rate(update, total, lr, final_fraction, warmup)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                obs = buffer.obs[chunk].to(device, non_blocking=True).float()
                target = buffer.label[chunk].to(device, non_blocking=True)
                sample_weights = None if by_mode is None else by_mode[
                    buffer.tags[chunk, mode_column].to(device).long().clamp(-1, len(MODES) - 1) + 1]
                if bands is not None:
                    band = bands[(observed_load(obs).clamp(0, 1 - 1e-6) * len(bands)).long()]
                    sample_weights = band if sample_weights is None else sample_weights * band
                loss = distillation_loss(net, obs, target, weights, sample_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # error_if_nonfinite: a NaN must stop the run, not train it.
                torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
                optimizer.step()
                running += loss.detach()
                update += 1
                if log is not None and update % log_every == 0:
                    log(dict(epoch=epoch, update=update, of=total, lr=rate, loss=float(loss.detach())))
            losses.append(float(running) / per_epoch)
    finally:
        if standardizer is not None:
            standardizer.fold(layer)
            net.trunk[0] = layer
    net.eval()
    return dict(updates=update, samples=int(len(ids)), epochs=epochs, epoch_losses=losses,
                seconds=time.monotonic() - started,
                load_band_weights=None if bands is None else [round(float(v), 3) for v in bands])


def physical_scales(decoder):
    """Normalized action units -> physical units for the first five dimensions."""
    cfg = decoder.config
    return dict(x_mm=1000 * (decoder.high[0] - decoder.low[0]) / 2,
                y_mm=1000 * (decoder.high[1] - decoder.low[1]) / 2,
                vx_m_s=cfg.velocity_scale, vy_m_s=cfg.velocity_scale,
                duration_ms=1000 * (cfg.max_time - cfg.min_time) / 2)


def _summary(squared, count, scales):
    if not count:
        return dict(samples=0)
    per_dim = (squared / count).tolist()
    result = dict(samples=int(count), mse=float(np.mean(per_dim)), per_dim=per_dim)
    if scales:
        result["rmse"] = {name: float(np.sqrt(per_dim[i]) * scale) for i, (name, scale) in enumerate(scales.items())}
    return result


@torch.no_grad()
def heldout_report(net, buffer, *, device, scales=None, rounds=None, chunk=65536):
    """Held-out MSE overall, per dimension and split by every tag."""
    ids = buffer.indices(heldout=True, rounds=rounds)
    total = np.zeros(6)
    groups = {name: {} for name in ("mode", "kind", "side", "actor", "round")}
    net.eval()
    for part in ids.split(chunk):
        obs = buffer.obs[part].to(device).float()
        error = (net.actor(net.trunk(obs)).tanh() - buffer.label[part].to(device)).square().double().cpu().numpy()
        tags = buffer.tags[part].cpu().numpy()
        total += error.sum(0)
        for name, table in groups.items():
            column = tags[:, TAGS.index(name)]
            for value in np.unique(column):
                rows = column == value
                squared, count = table.get(int(value), (np.zeros(6), 0))
                table[int(value)] = (squared + error[rows].sum(0), count + int(rows.sum()))
    names = dict(mode={-1: "unknown", **dict(enumerate(MODES))}, kind=dict(enumerate(KINDS)),
                 side=dict(enumerate(SIDES)), actor=dict(enumerate(SOURCES)), round={})
    result = _summary(total, len(ids), scales)
    for name, table in groups.items():
        result["by_" + name] = {names[name].get(k, str(k)): _summary(s, c, scales) for k, (s, c) in sorted(table.items())}
    return result


@torch.no_grad()
def network_actions(net, obs, noise, device, generator=None):
    """Deterministic (noise 0) or Gaussian pre-tanh actions of a network seat."""
    if not len(obs):
        return np.zeros((0, 6), np.float32)
    if not hasattr(net, "trunk"):  # coordinate-translated reference
        return net.act(np.asarray(obs, np.float32), stochastic=False).astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(obs)).to(device).float()
    mean = net.actor(net.trunk(x))
    if noise:
        mean = mean + noise * torch.randn(mean.shape, device=device, generator=generator)
    return mean.tanh().cpu().numpy()


class Collector:
    """Runs the roll-in, stores labelled states and keeps game/thermal counts.

    The environment state persists across rounds: drills cycle every few
    seconds; full games continue (as in PPO's rollouts).
    """

    def __init__(self, env, buffer, student, production, *, rng, heldout, device, rollin,
                 mix, label_opponents=True, student_noise=0.1, opponent_noise=0.1,
                 teacher_noise=0.0, generator=None):
        self.env, self.buffer, self.student, self.production = env, buffer, student, production
        self.rng, self.heldout, self.device, self.rollin = rng, np.asarray(heldout, bool), device, rollin
        self.mix, self.label_opponents = np.asarray(mix, float), bool(label_opponents)
        self.student_noise, self.opponent_noise, self.teacher_noise = student_noise, opponent_noise, teacher_noise
        self.generator = generator
        self.batch = env.reset()

    def run(self, steps, *, beta, round_index, store=True, should_stop=None, log=None, log_every=50):
        n = self.env.n_envs
        rollin = self.rollin
        rollin.restart()
        # Game outcomes per (robot seat: teacher/student) x (far seat source).
        goals, conceded, seconds = (np.zeros((2, 3)) for _ in range(3))
        shutdowns = np.zeros(2)
        mode_counts = np.zeros(len(MODES) + 1)
        kind_counts = np.zeros(len(KINDS))
        stored = robot = far_seat = taken = 0
        started = time.monotonic()
        network_seconds = env_seconds = store_seconds = 0.0
        for _ in range(steps):
            if should_stop is not None and should_stop():
                break
            b = self.batch
            rollin.update(beta, self.mix, b.done)
            teacher, opponent = rollin.teacher, rollin.opponent
            mark = time.monotonic()
            if store:
                stored += self.buffer.add(b.obs, b.label, mode=b.mode, kind=b.kind, side=0,
                                          actor=np.where(teacher, TEACHER, STUDENT), round=round_index,
                                          heldout=self.heldout)
                robot += n
                if self.label_opponents and len(b.far):
                    seat = opponent[b.far]
                    keep = seat != PRODUCTION
                    stored += self.buffer.add(b.far_obs[keep], b.far_label[keep], mode=b.far_mode[keep],
                                              kind=b.kind[b.far][keep], side=1, actor=seat[keep],
                                              round=round_index, heldout=self.heldout[b.far][keep])
                    far_seat += int(keep.sum())
                mode_counts += np.bincount(b.mode.astype(int) + 1, minlength=len(MODES) + 1)[:len(MODES) + 1]
                kind_counts += np.bincount(b.kind, minlength=len(KINDS))[:len(KINDS)]
            store_seconds += time.monotonic() - mark
            mark = time.monotonic()
            action = b.label.copy()
            if self.teacher_noise:
                action[teacher] = np.clip(action[teacher] + self.teacher_noise *
                                          self.rng.standard_normal((teacher.sum(), 6)), -1, 1)
            student = ~teacher
            if student.any():
                action[student] = network_actions(self.student, b.obs[student], self.student_noise,
                                                  self.device, self.generator)
            far_action = np.zeros_like(action)
            if len(b.far):
                seat = opponent[b.far]
                chosen = b.far_label.copy()
                for source, net in ((PRODUCTION, self.production), (STUDENT, self.student)):
                    rows = seat == source
                    if rows.any():
                        chosen[rows] = network_actions(net, b.far_obs[rows], self.opponent_noise,
                                                       self.device, self.generator)
                far_action[b.far] = chosen
            network_seconds += time.monotonic() - mark
            mark = time.monotonic()
            self.batch = nb = self.env.step(action, far_action)
            env_seconds += time.monotonic() - mark
            taken += 1
            game = nb.kind == 3
            actor = (~teacher).astype(int)
            np.add.at(goals, (actor[game], opponent[game]), nb.goals[game])
            np.add.at(conceded, (actor[game], opponent[game]), nb.conceded[game])
            np.add.at(seconds, (actor[game], opponent[game]), 0.02)
            np.add.at(shutdowns, actor, nb.shutdown)
            if log is not None and taken % log_every == 0:
                spent = time.monotonic() - started
                log(dict(step=taken, of=steps, stored=stored, labelled_per_s=stored / max(spent, 1e-9),
                         transitions_per_s=taken * n / max(spent, 1e-9)))
        elapsed = time.monotonic() - started
        pairs = {}
        for a, robot_name in enumerate(("teacher", "student")):
            for o, far_name in enumerate(SOURCES):
                if seconds[a, o]:
                    minutes = seconds[a, o] / 60
                    pairs[f"{robot_name}_vs_{far_name}"] = dict(
                        game_minutes=float(minutes), goals=int(goals[a, o]), conceded=int(conceded[a, o]),
                        goals_per_minute=float(goals[a, o] / minutes), conceded_per_minute=float(conceded[a, o] / minutes))
        return dict(steps=taken, robot_transitions=taken * n, stored=stored, stored_robot=robot,
                    stored_far=far_seat, seconds=elapsed,
                    labelled_per_s=stored / max(elapsed, 1e-9), transitions_per_s=taken * n / max(elapsed, 1e-9),
                    time_split=dict(env=env_seconds, networks=network_seconds, store=store_seconds),
                    teacher_modes={name: int(c) for name, c in zip(("unknown", *MODES), mode_counts)},
                    kinds={name: int(c) for name, c in zip(KINDS, kind_counts)},
                    games=pairs, shutdowns=dict(teacher=int(shutdowns[0]), student=int(shutdowns[1])))
