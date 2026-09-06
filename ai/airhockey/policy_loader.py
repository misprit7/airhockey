"""Load a trained TD-MPC2 checkpoint for inference.

One place that knows how to turn `runs/<name>/agent.pt` into an agent whose
`act(obs_1d, t0=..., eval_mode=True)` can drive a paddle. Used by the web
UI's sim tab (agent-vs-agent and agent-vs-human play) and intended for the
eventual on-table deployment runner, so the two can never disagree about how
a checkpoint is reconstructed.

The config mirrors ai/bin/train_tdmpc2_fast.py's construction. It has to:
the checkpoint stores weights only, so the architecture hyperparameters
must be rebuilt identically or load_state_dict fails (or worse, doesn't).

Torch and the tdmpc2 repo are imported lazily so importing this module --
and therefore the server -- stays cheap until a policy is actually loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
TDMPC2_DIR = _REPO_ROOT.parent / "tdmpc2" / "tdmpc2"

# Model size used by every training run so far. Stored per-load rather than
# read from the checkpoint because TD-MPC2 checkpoints do not carry their
# hyperparameters.
DEFAULT_MODEL_SIZE = 5


def list_checkpoints() -> list[dict]:
    """Every runs/<name>/agent.pt, newest first."""
    runs = _REPO_ROOT / "runs"
    out = []
    if runs.is_dir():
        for p in runs.glob("*/agent.pt"):
            out.append({"run": p.parent.name, "mtime": p.stat().st_mtime})
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def resolve_checkpoint(run_name: str) -> Path:
    """The checkpoint a run name means.

        <run>     runs/<run>/agent.pt, else the newest runs/<run>/agent_step_*.pt
                  (a run still training has only step files)
        latest    the newest checkpoint of any run, skipping runs whose name
                  starts with "_" (benchmark copies of other runs' files)
    """
    runs = _REPO_ROOT / "runs"
    if run_name == "latest":
        cands = [c for d in runs.iterdir() if d.is_dir() and not d.name.startswith("_")
                 for c in list(d.glob("agent_step_*.pt")) + list(d.glob("agent.pt"))]
        if not cands:
            raise FileNotFoundError(f"no checkpoints under {runs}")
        return max(cands, key=lambda c: c.stat().st_mtime)
    d = runs / run_name
    if (d / "agent.pt").exists():
        return d / "agent.pt"
    steps = sorted(d.glob("agent_step_*.pt"))
    if steps:
        return steps[-1]
    raise FileNotFoundError(d / "agent.pt")


# The MPPI action-change cost, in reward units per squared unit of action
# change, charged from the previously executed action through the horizon.
# ONE constant for training, evaluation and the table: the planner that
# collects the data is the planner that plays. Sized so a corner-to-corner
# flip (8 squared units) costs 4, comparable to one of Q's value bins near
# 50, while a strike-sized change (~2) costs 1.
PLAN_SMOOTH_COEF = 0.0   # parked: see memory; planner cost code stays inert
# MPPI iterations, likewise shared. 6 (TD-MPC2's default) since 2026-09-06:
# the tick is 20 ms at ACTION_HZ = 50, and ai/bin/bench_planner.py on the
# rig's RTX 4090 measured 6 iterations at 12.6 ms eager (p95 14.6) and
# 6.3 ms under CUDA graphs (deploy.TDMPC2Policy compile_plan). At the old
# 100 Hz tick even 3 iterations (7.1 ms) left no room for the master's I/O.
# The cost is launch-bound, not compute-bound: 128 samples cost the same as
# 512, so the knobs that matter are iterations and horizon.
PLAN_ITERATIONS = 6
# Execute the elite MEAN in eval mode, not a sampled elite (local TD-MPC2
# flag plan_eval_mean). Stock MPPI draws one elite trajectory even in eval
# mode, and on a flat value landscape -- the puck parked far away, nothing
# to gain -- the elites are random samples, so the executed target is a
# random point in the box every tick. Measured 2026-09-06 on the smooth6
# checkpoint, static scene: 20% of ticks jumped >300 mm with a sampled
# elite, 1% with the mean at 3 iterations. In play vs the goalie: GF/20 s
# 0.75 -> 1.56 at 3 iterations, 0.88 -> 1.62 at 6, jumps 25% -> 11%.
# Training collection (eval_mode=False) is untouched.
PLAN_EVAL_MEAN = True


def load_agent(run_name: str, iterations: int | None = PLAN_ITERATIONS,
               model_size: int = DEFAULT_MODEL_SIZE,
               ckpt: str | Path | None = None,
               plan_smooth: float = PLAN_SMOOTH_COEF):
    """Build a TDMPC2 agent and load a checkpoint into it.

    run_name resolves through resolve_checkpoint() unless `ckpt` names the
    file directly.

    iterations: MPPI iterations for inference. The training default of 6
    costs ~13 ms per plan, which stalls an interactive 60 fps loop with two
    agents; 3 halves that for a modest quality cost. Pass None to keep the
    training default.
    """
    ckpt = Path(ckpt) if ckpt is not None else resolve_checkpoint(run_name)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    if str(TDMPC2_DIR) not in sys.path:
        sys.path.insert(0, str(TDMPC2_DIR))
    from omegaconf import OmegaConf
    from common import MODEL_SIZE
    from common.parser import cfg_to_dataclass
    from tdmpc2 import TDMPC2

    from airhockey.batch_env import BatchAirHockeyEnv

    base = OmegaConf.load(str(TDMPC2_DIR / "config.yaml"))
    overrides = OmegaConf.create({
        "task": "airhockey", "obs": "state", "episodic": True,
        "steps": 1_000_000, "model_size": model_size, "horizon": 5,
        "eval_freq": 100_000, "eval_episodes": 1, "save_video": False,
        "enable_wandb": False, "save_csv": False,
        "work_dir": str(ckpt.parent), "compile": False,
        "data_dir": str(ckpt.parent / "data"), "exp_name": run_name,
        "discount_max": 0.99, "rho": 0.85, "task_title": run_name,
        "multitask": False, "tasks": ["airhockey"], "task_dim": 0,
        "prioritized_replay": False, "per_alpha": 0.6, "per_beta": 0.4,
    })
    cfg = OmegaConf.merge(base, overrides)
    cfg.num_samples = 512
    cfg.iterations = 6
    for k, v in MODEL_SIZE[model_size].items():
        cfg[k] = v
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "obs_shape": {"state": [BatchAirHockeyEnv.OBS_DIM]},
        "action_dim": 2,
        "episode_length": 3000,
        "seed_steps": 1000,
    }))
    cfg = cfg_to_dataclass(cfg)
    if iterations is not None:
        cfg.iterations = iterations
    cfg.plan_smooth_coef = float(plan_smooth)
    cfg.plan_eval_mean = bool(PLAN_EVAL_MEAN)
    from airhockey.batch_env import BatchAirHockeyEnv   # noqa: PLC0415
    cfg.prev_action_start = BatchAirHockeyEnv.PREV_ACTION_IDX   # obs [15:17]

    agent = TDMPC2(cfg)
    load_checkpoint(agent, ckpt)
    return agent


def load_checkpoint(agent, path) -> None:
    """agent.load(), accepting checkpoints trained on a NARROWER observation.

    New observation features (the previous action, 2026-09-03) get zero
    weight in the encoder's first layer, so the loaded policy computes
    exactly what it did before until training grows those columns. The
    layer norm sits on that layer's OUTPUT, so zero columns change nothing.
    """
    import torch                                        # noqa: PLC0415

    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    key = "_encoder.state.0.weight"
    if key in sd:
        want = agent.model.state_dict()[key].shape[1]
        have = sd[key].shape[1]
        if have < want:
            pad = torch.zeros(sd[key].shape[0], want - have, dtype=sd[key].dtype)
            sd[key] = torch.cat([sd[key], pad], dim=1)
            print(f"[load] observation widened {have} -> {want}: the new inputs "
                  f"start at zero weight")
        elif have > want:
            raise ValueError(f"{path}: trained on {have}-wide observations, "
                             f"this build has {want}")
    agent.load({"model": sd})


def reset_prior(agent) -> None:
    """Re-initialise the policy prior head (the planner and value model stay).

    A prior trained for millions of steps under the tanh-squashed entropy
    bonus ends up with pre-squash means in the hundreds: bang-bang by
    construction, and immovable by any regulariser at a learning-rate-sized
    step. The prior is cheap to relearn from Q; the world model and Q are
    what the checkpoint is worth.
    """
    import torch.nn as nn                                  # noqa: PLC0415

    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    agent.model._pi.apply(init)
    if hasattr(agent, "pi_optim"):
        agent.pi_optim.state.clear()
