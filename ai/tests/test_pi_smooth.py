"""The prior's temporal-smoothness regulariser (local TD-MPC2 addition).

A per-step action-change penalty in the reward never reached the prior:
TD-MPC2 trains it by maximising a two-hot Q whose resolution near a value of
50 is ~10 units. pi_smooth_coef pulls the prior's mean toward the previous
action, which the observation carries, directly in the policy loss.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from airhockey.policy_loader import load_agent, resolve_checkpoint


def _latest():
    try:
        return resolve_checkpoint("latest")
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_latest() is None, reason="no checkpoint under runs/")
def test_pi_smooth_pulls_the_prior_toward_the_previous_action():
    from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
    agent = load_agent("latest", iterations=1)
    agent.cfg.pi_smooth_coef = 1.0
    torch.manual_seed(0)
    # Real latents: the prior is only meaningful on encoded observations
    # (SimNorm latents), not on random vectors.
    T, B = 6, 64
    env = BatchAirHockeyEnv(n_envs=B, opponent_policy="goalie", domain_randomize=True,
                            **sensing_kwargs(True))
    obs = env.reset(seed=3)
    frames = []
    for _ in range(T):
        frames.append(torch.from_numpy(obs).float())
        obs = env.step(np.random.uniform(-1, 1, (B, 2)).astype(np.float32))[0]
    with torch.no_grad():
        zs = agent.model.encode(torch.stack(frames).to(agent.device), None)
    prev = torch.zeros(T, B, agent.cfg.action_dim, device=agent.device)
    # A FRESH prior head, as train_selfplay --reset-prior does: the trained
    # heads on these checkpoints emit pre-squash means of ~180 (tanh flat
    # to machine precision), and no learning-rate-sized step moves that.
    from airhockey.policy_loader import reset_prior
    reset_prior(agent)

    def dist():
        with torch.no_grad():
            _, info = agent.model.pi(zs, None)
        return float(((info["mean"] - prev) ** 2).sum(-1).mean())

    def raw_mag():
        with torch.no_grad():
            _, info = agent.model.pi(zs, None)
        return float(info["mean_raw"].abs().max())

    before = dist()
    agent.model.train()
    for _ in range(200):
        info = agent.update_pi(zs, None, prev_action=prev)
    agent.model.eval()
    assert "pi_smooth" in info.keys()
    after = dist()
    assert after < 0.9 * before, f"prior did not move toward the previous action: {before:.3f} -> {after:.3f}"
    assert raw_mag() < 3.0, "the anti-saturation hinge should keep the pre-squash mean bounded"


@pytest.mark.skipif(_latest() is None, reason="no checkpoint under runs/")
def test_pi_smooth_off_leaves_the_loss_alone():
    agent = load_agent("latest", iterations=1)
    agent.cfg.pi_smooth_coef = 0.0
    zs = torch.nn.functional.normalize(torch.randn(3, 16, agent.cfg.latent_dim, device=agent.device), dim=-1)
    agent.model.train()
    info = agent.update_pi(zs, None, prev_action=torch.zeros(3, 16, agent.cfg.action_dim, device=agent.device))
    agent.model.eval()
    assert float(info["pi_smooth"]) == 0.0


@pytest.mark.skipif(_latest() is None, reason="no checkpoint under runs/")
def test_plan_smooth_cost_takes_the_flips_out_of_the_planner():
    """The prior can be smooth and the planner will still flip corner to
    corner, because Q prefers it on a fast body. Charging action change in
    the MPPI objective is what makes the planner smooth too."""
    from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
    jumps = {}
    for coef in (0.0, 0.5):
        agent = load_agent("latest", iterations=2, plan_smooth=coef)
        env = BatchAirHockeyEnv(n_envs=4, opponent_policy="goalie", domain_randomize=True,
                                **sensing_kwargs(True))
        obs = env.reset(seed=9)
        t0 = torch.ones(4, dtype=torch.bool)
        prev = None
        big = 0
        for _ in range(300):
            with torch.no_grad():
                a = agent.act(torch.from_numpy(obs).float(), t0=t0, eval_mode=True).numpy()
            t0 = torch.zeros(4, dtype=torch.bool)
            if prev is not None:
                big += int((np.linalg.norm(a - prev, axis=1) > 0.6).sum())
            prev = a
            obs = env.step(a)[0]
        jumps[coef] = big / (4 * 299)
    assert jumps[0.5] < 0.5 * jumps[0.0], f"planner jumps: {jumps}"
