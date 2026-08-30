#!/usr/bin/env python3
"""Tournament for the heuristic bots in `airhockey.heuristics`.

    python ai/bin/eval_heuristics.py                  # the full table
    python ai/bin/eval_heuristics.py --games 32 --seconds 90
    python ai/bin/eval_heuristics.py --bots goalie,intercept --opponents random

Every bot plays the same fixture list against the scripted opponents, through
REALISTIC SENSING (the 200 Hz camera model, its measured 5-10 ms latency, the
IR ring's blind spot at table centre) and with domain randomisation on, because
a heuristic that only works against exact velocities is not a baseline for
anything that has to run on the table.

The bots never see the environment. `SimBridge` turns the history observation
into tracker reports in table millimetres and turns their millimetre commands
back into normalised actions; a bot cannot tell whether it is being driven from
here or from `vision/bin/puck_stream.py`.

FIXTURES ARE SHARED. Every (bot, opponent) pair is seeded identically, so all
bots face the same puck draws, the same randomised machines and the same
opponent behaviour. Without that, a two-goal difference over sixteen games is
mostly the seed.

There is no head-to-head: the table has one robot side, and putting two bots on
it would need the opposite half to run the same motion law and the same
workspace, which is exactly the thing a human opponent does not do. The
scripted opponents are the honest comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs  # noqa: E402
from airhockey.heuristic_bridge import SimBridge  # noqa: E402
from airhockey.heuristics import BOTS, BotConfig, make_bot  # noqa: E402

DEFAULT_OPPONENTS = ("goalie", "follow", "random")

# Conceding costs twice what scoring pays. The machine's job is to be hard to
# beat -- a bot that trades goals evenly with a scripted opponent has learned
# to leave its net, and on a real table the opponent is better than these.
CONCEDE_WEIGHT = 2.0


@dataclass
class MatchResult:
    bot: str
    opponent: str
    games: int
    goals_for: np.ndarray = field(default_factory=lambda: np.zeros(0))
    goals_against: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0
    wall_s: float = 0.0

    @property
    def gf(self) -> float:
        return float(self.goals_for.mean())

    @property
    def ga(self) -> float:
        return float(self.goals_against.mean())

    @property
    def record(self) -> tuple[int, int, int]:
        w = int((self.goals_for > self.goals_against).sum())
        d = int((self.goals_for == self.goals_against).sum())
        return w, d, len(self.goals_for) - w - d

    @property
    def score(self) -> float:
        return self.gf - CONCEDE_WEIGHT * self.ga


def run_match(bot_name: str, opponent: str, games: int, seconds: float,
              seed: int, verbose: bool = True) -> MatchResult:
    """One bot against one scripted opponent, `games` full-length games."""
    env = BatchAirHockeyEnv(
        n_envs=games,
        opponent_policy=opponent,
        obs_mode="history",
        action_mode="profile_v",
        domain_randomize=True,
        # No early stop on score: every game runs the same wall clock, so
        # goals-for and goals-against are rates rather than a race to seven.
        max_score=10 ** 6,
        max_episode_time=seconds + 1.0,
        **sensing_kwargs(True),
    )
    bridge = SimBridge(env)

    # Taken from the env rather than left at BotConfig's default, even though
    # since 1e8c303 the two agree (TableConfig.paddle_radius now comes from
    # geom.MALLET_RADIUS_MM, so both are the measured 50.4 mm). The bots aim
    # THROUGH the puck by mallet+puck radius, so a bot told the wrong body
    # leaves every shot off-line by the difference -- and reading it from the
    # env means the harness cannot silently disagree with the table it is
    # playing on, whichever way that number moves next.
    cfg = BotConfig(mallet_radius_mm=env.table_config.paddle_radius * 1000.0)
    bots = [make_bot(bot_name, cfg) for _ in range(games)]

    obs = env.reset(seed=seed)
    bridge.reset()
    n_steps = int(round(seconds / env.action_dt))
    t0 = time.perf_counter()
    info: dict = {}
    for _ in range(n_steps):
        reports = bridge.reports(obs)
        commands = [b(r) for b, r in zip(bots, reports)]
        obs, _, _, _, info = env.step(bridge.actions(commands, obs))
        bridge.step_index += 1

    res = MatchResult(
        bot=bot_name, opponent=opponent, games=games,
        goals_for=info["score_agent"].astype(float),
        goals_against=info["score_opponent"].astype(float),
        seconds=seconds, wall_s=time.perf_counter() - t0,
    )
    if verbose:
        w, d, ll = res.record
        print(f"  {bot_name:<10s} vs {opponent:<7s}  "
              f"GF {res.gf:5.2f}  GA {res.ga:5.2f}  "
              f"{w}-{d}-{ll}  ({res.wall_s:.1f}s)", flush=True)
    return res


def print_table(results: list[MatchResult], opponents: tuple[str, ...]) -> None:
    bots = []
    for r in results:
        if r.bot not in bots:
            bots.append(r.bot)
    by = {(r.bot, r.opponent): r for r in results}

    head = f"{'bot':<11s}{'opponent':<10s}{'GF':>7s}{'GA':>7s}{'GD':>7s}" \
           f"{'W-D-L':>10s}{'score':>8s}"
    print("\n" + head)
    print("-" * len(head))
    for b in bots:
        for o in opponents:
            r = by.get((b, o))
            if r is None:
                continue
            w, d, ll = r.record
            print(f"{b:<11s}{o:<10s}{r.gf:7.2f}{r.ga:7.2f}"
                  f"{r.gf - r.ga:+7.2f}{f'{w}-{d}-{ll}':>10s}{r.score:8.2f}")
        print("-" * len(head))

    print(f"\nOVERALL (mean over opponents, score = GF - {CONCEDE_WEIGHT:g}xGA)")
    print(f"{'bot':<11s}{'GF':>7s}{'GA':>7s}{'GD':>7s}{'win%':>8s}{'score':>8s}")
    ranked = []
    for b in bots:
        rs = [by[(b, o)] for o in opponents if (b, o) in by]
        gf = float(np.mean([r.gf for r in rs]))
        ga = float(np.mean([r.ga for r in rs]))
        wins = sum(r.record[0] for r in rs)
        played = sum(r.games for r in rs)
        ranked.append((gf - CONCEDE_WEIGHT * ga, -ga, b, gf, ga,
                       100.0 * wins / max(played, 1)))
    ranked.sort(reverse=True)
    for score, _, b, gf, ga, winpct in ranked:
        print(f"{b:<11s}{gf:7.2f}{ga:7.2f}{gf - ga:+7.2f}{winpct:8.1f}{score:8.2f}")

    best = ranked[0]
    print(f"\nWINNER: {best[2]}  (score {best[0]:.2f}, "
          f"{best[4]:.2f} conceded per game, {best[5]:.0f}% of games won)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", default=",".join(BOTS),
                    help=f"comma separated; have {','.join(BOTS)}")
    ap.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    ap.add_argument("--games", type=int, default=24,
                    help="games per (bot, opponent), run in parallel envs")
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="length of one game")
    ap.add_argument("--seed", type=int, default=7,
                    help="shared across bots, so the fixtures are identical")
    args = ap.parse_args()

    bots = tuple(b.strip() for b in args.bots.split(",") if b.strip())
    opponents = tuple(o.strip() for o in args.opponents.split(",") if o.strip())
    for b in bots:
        if b not in BOTS:
            raise SystemExit(f"unknown bot {b!r}; have {sorted(BOTS)}")

    print(f"{len(bots)} bots x {len(opponents)} opponents x {args.games} games "
          f"of {args.seconds:g}s, realistic sensing + DR, seed {args.seed}")
    results = []
    for b in bots:
        for i, o in enumerate(opponents):
            # Same seed per fixture across bots; different per opponent so the
            # three columns are not the same sixteen puck draws three times.
            results.append(run_match(b, o, args.games, args.seconds,
                                     args.seed + 1000 * i))
    print_table(results, opponents)


if __name__ == "__main__":
    main()
