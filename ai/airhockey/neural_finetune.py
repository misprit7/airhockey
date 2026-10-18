"""Training-only pieces for RL fine-tuning the one neural player.

Nothing here runs at the table. A teacher (``expert_player.ExpertPlayer``, or
anything with its ``act``/``history``/``shot_conditioned``/``last_mode``
interface) supplies regression targets and a sparring partner in simulation.
The optimizer controls change only how PPO moves the single network that is
deployed; they add no runtime component.
"""

import inspect
import math
from pathlib import Path

import numpy as np
import torch

# ExpertPlayer.last_mode codes after act(): 0 threat, 1 attack, 2 guard.
THREAT, ATTACK, GUARD = 0, 1, 2
MODE_NAMES = ("threat", "attack", "guard")
# defense: the teacher is not attacking (blocking a predicted goal or
# guarding); threat: only blocking; attack: only attacking.
TEACHER_MASKS = ("all", "defense", "threat", "attack")


def _coordinates(options):
    from airhockey.neural_coordinates import bounds

    return float(options.get("accel", 60)), bounds(options)


def same_coordinates(source, target):
    (a, box_a), (b, box_b) = _coordinates(source), _coordinates(target)
    return math.isclose(a, b, rel_tol=1e-9) and np.allclose(box_a, box_b, rtol=0, atol=1e-9)


class Teacher:
    """A teacher acting in its own recorded environment, seen in run units.

    Observations and actions are translated like ``CoordinateReference``
    when the teacher's workspace or acceleration cap differs from the run's;
    otherwise they pass through untouched. Always deterministic.
    """

    def __init__(self, player, target_options, spec=None):
        self.player = player
        self.spec = spec
        self.history = int(player.history)
        self.shot_conditioned = bool(player.shot_conditioned)
        self.source = dict(player.environment_options)
        self.target = dict(target_options)
        self.translated = not same_coordinates(self.source, self.target)
        self.last_mode = None

    def act(self, obs, stochastic=False):
        from airhockey.neural_coordinates import action_coordinates, observation_coordinates

        obs = np.asarray(obs, dtype=np.float32)
        if self.translated:
            obs = observation_coordinates(obs, self.target, self.source, self.history)
        action = np.asarray(self.player.act(obs, stochastic=False), dtype=np.float32)
        mode = getattr(self.player, "last_mode", None)
        self.last_mode = None if mode is None else np.array(mode, dtype=np.int8, copy=True)
        if self.translated:
            action = action_coordinates(action, self.source, self.target).astype(np.float32)
        if action.shape != (len(obs), 6):
            raise ValueError("teacher must return one 6-D arrival action per observation")
        return action

    def describe(self):
        accel, box = _coordinates(self.source)
        return dict(spec=self.spec, history=self.history, shot_conditioned=self.shot_conditioned,
                    accel=accel, workspace_sim=box.tolist(), translated=self.translated)


def load_teacher(spec, target_options, root):
    """``expert`` or ``expert:key=value,...`` through ``expert_player.load_expert``."""
    spec = str(spec)
    if not spec.startswith("expert"):
        raise ValueError("teacher must be an expert spec such as 'expert' or 'expert:key=value'")
    from airhockey import expert_player

    kwargs = {}
    default = getattr(expert_player, "PRODUCTION_ENVIRONMENT", None)
    if default is not None and "environment_from" in inspect.signature(expert_player.load_expert).parameters:
        # Resolve against the repository, not the working directory: a missing
        # run.json would otherwise silently give the legacy environment.
        source = Path(root) / default
        if not (source.parent / "run.json").exists():
            raise FileNotFoundError(f"teacher environment {source.parent / 'run.json'} is missing")
        kwargs["environment_from"] = source
    player, _ = expert_player.load_expert(spec, **kwargs)
    return Teacher(player, target_options, spec)


def teacher_mask(modes, which):
    """Which samples regress toward the teacher, from its per-sample modes."""
    if which == "all":
        return torch.ones_like(modes, dtype=torch.bool) if torch.is_tensor(modes) else np.ones(np.shape(modes), bool)
    if which == "defense":
        return modes != ATTACK
    if which == "threat":
        return modes == THREAT
    if which == "attack":
        return modes == ATTACK
    raise ValueError(f"unknown teacher mask {which!r}")


def teacher_weight(steps, initial, final, anneal_steps):
    """Linear in environment steps from ``initial`` to ``final``, then constant."""
    if not anneal_steps:
        return float(initial)
    fraction = min(max(steps / anneal_steps, 0.0), 1.0)
    return float(initial + (final - initial) * fraction)


def masked_mean(values, mask=None):
    """Mean over selected samples; exactly zero (and zero gradient) for none."""
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def teacher_regression(mean, target, mask=None):
    """Mean squared error of tanh(actor mean) to the teacher over masked samples.

    Averaged over the six action dimensions, then over selected samples.
    """
    return masked_mean((mean.tanh() - target).square().mean(-1), mask)


def reference_slots(n_envs, references):
    """Environments the fixed references may use: ``id % 4 == 0``.

    This is the existing rule of train_neural_player (each reference takes
    ``(id // 4) % len(references) == j`` within it).
    """
    ids = np.arange(n_envs)
    return (ids % 4 == 0) if references else np.zeros(n_envs, bool)


def teacher_opponent_slots(game, fraction, reserved=None):
    """A fixed, evenly spread subset of full-game slots for the teacher.

    ``fraction`` is of ALL game slots, chosen only among game slots that are
    not ``reserved`` (the fixed-reference slots), so references keep exactly
    their share and the teacher takes its share from the snapshot pool.
    """
    game = np.asarray(game, bool)
    reserved = np.zeros_like(game) if reserved is None else np.asarray(reserved, bool)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("teacher opponent fraction must be in [0,1]")
    eligible = np.flatnonzero(game & ~reserved)
    count = int(round(fraction * game.sum()))
    if count > len(eligible):
        raise ValueError(
            f"teacher opponent fraction {fraction:g} needs {count} game slots but only "
            f"{len(eligible)} of {int(game.sum())} are free of fixed references")
    slots = np.zeros_like(game)
    if count:
        slots[eligible[(np.arange(count) * len(eligible)) // count]] = True
    return slots


def adapt_learning_rate(lr, kl, target, low, high, stopped_early=False, factor=1.5):
    """KL-targeted multiplicative actor step size for the next update.

    Down when the update overshot (final KL above twice the target, or the
    KL early stop ended it before half its planned minibatch steps), up when
    it moved less than half the target; clamped to [low, high].
    """
    if not (np.isfinite(kl) and np.isfinite(lr)) or lr <= 0 or factor <= 1:
        raise ValueError("adaptive learning rate needs finite positive values and a factor above one")
    if kl > 2 * target or stopped_early:
        lr = lr / factor
    elif kl < target / 2:
        lr = lr * factor
    return float(min(max(lr, low), high))


def std_floor(log_std, floor):
    """Training-only minimum pre-tanh std; apply in rollouts AND PPO alike."""
    if not floor:
        return log_std
    return torch.clamp_min(log_std, math.log(floor))


def _strip_prefix(model):
    for prefix in ("_orig_mod.", "module."):
        if model and all(k.startswith(prefix) for k in model):
            model = {k[len(prefix):]: v for k, v in model.items()}
    return model


def initialization_model(state):
    """The actor weights and architecture of an --initialize-from checkpoint.

    Accepts the production layout (no optimizer) and distilled checkpoints
    regardless of their ``algorithm`` field. Width, history and request
    conditioning come from the checkpoint's fields, else from its weights.
    """
    if "model" not in state:
        raise ValueError(f"checkpoint has no 'model' weights (keys: {sorted(state)})")
    model = _strip_prefix(dict(state["model"]))
    for key in ("trunk.0.weight", "trunk.2.weight", "trunk.4.weight", "actor.weight", "actor.bias"):
        if key not in model:
            raise ValueError(f"checkpoint actor is missing {key}")
    weight = model["trunk.0.weight"]
    history, request = divmod(weight.shape[1], 42)
    architecture = dict(
        width=int(state.get("width", weight.shape[0])),
        history=int(state.get("history", history)),
        shot_conditioned=bool(state.get("shot_conditioned", request == 3)),
    )
    return model, architecture


def complete_initialization(model, fresh):
    """Fill training-only heads a distilled checkpoint may omit.

    The critic is replaced by a fresh one after --initialize-from anyway, and
    a missing global log-std starts at the fresh network's value. The noise
    head and a missing value trunk are already handled by load_weights.
    """
    model = dict(model)
    for key in ("critic.weight", "critic.bias", "log_std"):
        if key not in model:
            model[key] = fresh[key].detach().clone()
    if any(k.startswith("value_trunk.") for k in model) and not all(
            k in model for k in fresh if k.startswith("value_trunk.")):
        for key in fresh:
            if key.startswith("value_trunk."):
                model[key] = fresh[key].detach().clone()
    return model
