#!/usr/bin/env python3
"""A trained policy against the scripted opponents, on the tournament's terms.

Same games as `eval_heuristics.py` -- 90 s, no early end on score, realistic
sensing + domain randomisation, the same seed -- so a row here compares
line-for-line with a row there. The heuristic striker's ~4 goals per game
against the goalie is the bar a policy has to clear to be worth deploying.

    python ai/bin/eval_policy.py curriculum_goalie
    python ai/bin/eval_policy.py curriculum_goalie --opponents idle,goalie,follow --games 32
    python ai/bin/eval_policy.py _bench_sp1000k --vs curriculum_goalie   # head-to-head

The scripted opponents cannot show self-play PROGRESS: a learner tuned for
+100/-50 against a live copy of itself learns to concede nothing first, and
its goal rate against a stationary goalie then wobbles rather than climbs.
`--vs` plays two checkpoints against each other on the same terms, the
second one driven through the env's mirrored view exactly as train_selfplay
drives its opponent, which is the comparison that picks a deploy candidate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs  # noqa: E402
from airhockey.policy_loader import load_agent  # noqa: E402

DEFAULT_OPPONENTS = ("goalie", "follow", "random")


def run_match(agent, run_name: str, opponent: str, games: int, seconds: float,
              seed: int, rival=None) -> tuple[np.ndarray, np.ndarray]:
    """`opponent` names a scripted policy, or "external" with `rival` an
    agent that plays the far side through the mirrored observation."""
    import torch

    env = BatchAirHockeyEnv(
        n_envs=games,
        opponent_policy=opponent,
        domain_randomize=True,
        max_score=10 ** 6,
        max_episode_time=seconds + 1.0,
        **sensing_kwargs(True),
    )
    obs = env.reset(seed=seed)
    t0 = torch.ones(games, dtype=torch.bool)
    n_steps = int(round(seconds / env.action_dt))
    wall = time.perf_counter()
    info: dict = {}
    for _ in range(n_steps):
        with torch.no_grad():
            if rival is not None:
                # Same path as train_selfplay.drive_opponent: the rival sees
                # the table from its own end and its action lands there.
                opp_act = rival.act(torch.from_numpy(env.mirror_obs(obs)).float(),
                                    t0=t0, eval_mode=True)
                tx, ty = env.mirror_action_to_opponent(opp_act.numpy())
                env._ext_opp_target_x[:] = tx
                env._ext_opp_target_y[:] = ty
            act = agent.act(torch.from_numpy(obs).float(), t0=t0, eval_mode=True)
        obs, _, _, _, info = env.step(act.numpy())
        t0 = torch.zeros(games, dtype=torch.bool)

    gf = info["score_agent"].astype(float)
    ga = info["score_opponent"].astype(float)
    w, d, l = int((gf > ga).sum()), int((gf == ga).sum()), int((gf < ga).sum())
    print(f"  {run_name:<20s} vs {opponent:<7s}  "
          f"GF {gf.mean():5.2f}  GA {ga.mean():5.2f}  "
          f"{w}-{d}-{l}  ({time.perf_counter() - wall:.1f}s)", flush=True)
    return gf, ga


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name under runs/ (loads runs/<run>/agent.pt)")
    ap.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    ap.add_argument("--games", type=int, default=24,
                    help="parallel games per matchup (default 24)")
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="game length; every game runs the full clock")
    ap.add_argument("--seed", type=int, default=7,
                    help="same default as eval_heuristics so fixtures match")
    ap.add_argument("--iterations", type=int, default=3,
                    help="MPPI iterations at inference (training uses 6)")
    ap.add_argument("--vs", metavar="RUN", default=None,
                    help="head-to-head: this checkpoint plays the far side "
                         "instead of the scripted opponents")
    ap.add_argument("--prior", action="store_true",
                    help="no planning: act from the policy prior alone. This "
                         "is the deployable mode -- 0.1 ms on a CPU against a "
                         "10 ms tick, where even one MPPI iteration is 15 ms "
                         "on a GPU -- so measure it on the same terms")
    args = ap.parse_args()

    agent = load_agent(args.run, iterations=args.iterations)
    if args.prior:
        agent.cfg.mpc = False
    mode = "policy prior, no planning" if args.prior else f"{args.iterations} MPPI iterations"
    print(f"{args.run}: {args.games} games x {args.seconds:.0f} s per opponent, "
          f"seed {args.seed}, {mode}")
    if args.vs:
        rival = load_agent(args.vs, iterations=args.iterations)
        if args.prior:
            rival.cfg.mpc = False
        run_match(agent, args.run, "external", args.games, args.seconds,
                  args.seed, rival=rival)
        print(f"  (far side: {args.vs})")
        return
    for opp in args.opponents.split(","):
        run_match(agent, args.run, opp.strip(), args.games, args.seconds, args.seed)


if __name__ == "__main__":
    main()
