"""Shared harness for the checkpoint diagnostics in `ai/bin` (`hold_eval.py`,
`income_breakdown.py`, `jitter_eval.py`).

One self-play env against one NAMED opponent, and one step of a checkpoint
driving the robot -- planner or policy prior -- with the far side driven by
the same checkpoint when the opponent is "external", exactly as
`train_selfplay.py` does it. The diagnostics differ only in what they
measure, so the play loop lives here once.
"""
from __future__ import annotations

import numpy as np
import torch

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
from airhockey.rewards import curriculum_env_kwargs

OPPONENTS = ("weak_goalie", "sniper", "external")


def make_env(opponent: str, n_envs: int, seconds: float) -> BatchAirHockeyEnv:
    """The self-play stage's env (sensing, DR, shot requests, fuzz, the shot
    clock and turnover rules) against ONE opponent kind instead of the mix."""
    kw = curriculum_env_kwargs("selfplay")
    kw.pop("opponent_mix_probs", None)
    return BatchAirHockeyEnv(n_envs, opponent_policy=opponent, opponent_body="robot",
                             domain_randomize=True, **sensing_kwargs(True), **kw,
                             max_episode_time=seconds + 10.0)


class Driver:
    """Acts for the robot with the checkpoint -- the planner (training's
    iterations and the run's horizon, eval mode) or the prior alone -- and
    drives an external far side with the same checkpoint on its own view."""

    def __init__(self, agent, env: BatchAirHockeyEnv, planner: bool = True):
        self.agent = agent
        self.env = env
        agent.cfg.mpc = planner
        self.t0 = torch.ones(env.n_envs, dtype=torch.bool)

    def act(self, obs: np.ndarray) -> np.ndarray:
        e = self.env
        if bool(np.any(e.external_mask)):
            with torch.no_grad():
                ao = self.agent.act(torch.from_numpy(e.opponent_obs()).float(),
                                    t0=self.t0, eval_mode=True)
            tx, ty = e.mirror_action_to_opponent(ao.cpu().numpy())
            e._ext_opp_target_x[:] = tx
            e._ext_opp_target_y[:] = ty
        with torch.no_grad():
            a = self.agent.act(torch.from_numpy(obs).float(), t0=self.t0, eval_mode=True)
        return np.asarray(a.cpu().numpy(), dtype=np.float32).reshape(e.n_envs, -1)

    def done(self, terminated, truncated) -> None:
        self.t0 = torch.from_numpy(np.asarray(terminated | truncated))


class GoalCounter:
    """Goals for and against, summed over the batch, from the scoreboard."""

    def __init__(self, env: BatchAirHockeyEnv):
        self.prev_a = env.engine.score_agent.copy()
        self.prev_o = env.engine.score_opponent.copy()
        self.gf = 0
        self.ga = 0

    def update(self, info) -> None:
        self.gf += int(np.maximum(info["score_agent"] - self.prev_a, 0).sum())
        self.ga += int(np.maximum(info["score_opponent"] - self.prev_o, 0).sum())
        self.prev_a = info["score_agent"].copy()
        self.prev_o = info["score_opponent"].copy()

    def __str__(self) -> str:
        return f"{self.gf}-{self.ga}"


def header(run: str, agent) -> str:
    cfg = agent.cfg
    return (f"{run}: planner {cfg.iterations} iterations x {cfg.num_samples} samples, "
            f"horizon {cfg.horizon}, action_dim {cfg.action_dim}")
