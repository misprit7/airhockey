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


def load_agent(run_name: str, iterations: int | None = 3,
               model_size: int = DEFAULT_MODEL_SIZE,
               ckpt: str | Path | None = None):
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

    agent = TDMPC2(cfg)
    agent.load(str(ckpt))
    return agent
