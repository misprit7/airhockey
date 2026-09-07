"""The scripted cushion-and-hold controller that seeds the replay buffer.

It exists to put the stop-hold-shoot chain in front of the value
function, so the one thing it must do is control the puck in a sizeable
share of possessions -- and earn the shaper's control rewards doing it.
"""
from __future__ import annotations

import numpy as np

from airhockey.batch_env import BatchAirHockeyEnv, sensing_kwargs
from airhockey.cushion_bot import BLOCK, CUSHION, HOLD, INTERCEPT, SHOOT, WAIT, CushionBot
from airhockey.rewards import (CONTROL_SPEED, STAGE_SCORING, BatchRewardShaper,
                               curriculum_env_kwargs, curriculum_shaper_kwargs)


def _possessions(opp, seconds=25.0, n=16, seed=11):
    kw = curriculum_env_kwargs("selfplay")
    kw.pop("opponent_mix_probs", None)
    e = BatchAirHockeyEnv(n, opponent_policy=opp, opponent_body="robot",
                          domain_randomize=True, **sensing_kwargs(True), **kw,
                          max_episode_time=seconds + 5)
    sh = BatchRewardShaper(n, stage=STAGE_SCORING, workspace=e._ws,
                           **curriculum_shaper_kwargs("selfplay"))
    o = e.reset(seed=seed)
    bot = CushionBot(e, np.random.default_rng(0))
    sh.reset(o, info={"puck_y": e.engine.puck_y, "puck_vx": e.engine.puck_vx,
                      "puck_vy": e.engine.puck_vy})
    H = e.table_config.height
    in_prev = e.engine.puck_y < H / 2
    poss = [dict(mx=0.0, mn=9.0, steps=0) for _ in range(n)]
    done = []
    phases = set()
    for _ in range(int(seconds / e.action_dt)):
        a = bot.act()
        phases |= set(bot.phase.tolist())
        assert a.shape == (n, e.action_dim) and np.all(np.abs(a) <= 1.0)
        o, r, term, trunc, info = e.step(a)
        sh.compute(o, r, actions=a, info=info)
        in_half = info["puck_y"] < H / 2
        sp = np.hypot(info["puck_vx"], info["puck_vy"])
        d = np.hypot(info["puck_x"] - info["pad_x"], info["puck_y"] - info["pad_y"])
        for i in range(n):
            if in_half[i] and not in_prev[i]:
                poss[i] = dict(mx=0.0, mn=9.0, steps=0)
            if not in_half[i] and in_prev[i] and poss[i]["steps"] > 0:
                done.append(poss[i])
            if in_half[i]:
                poss[i]["steps"] += 1
                poss[i]["mx"] = max(poss[i]["mx"], sp[i])
                if d[i] < 0.20:
                    poss[i]["mn"] = min(poss[i]["mn"], sp[i])
        in_prev = in_half
    fast = [p for p in done if p["mx"] > 0.8]
    return e, bot, sh, fast, phases


def test_bot_controls_the_puck_in_most_possessions_against_a_weak_goalie():
    e, bot, sh, fast, phases = _possessions("weak_goalie")
    assert len(fast) >= 15
    mn = np.array([p["mn"] for p in fast])
    controlled = float(np.mean(mn < CONTROL_SPEED))
    assert controlled > 0.4, f"controlled only {controlled:.0%} of fast possessions"
    assert {WAIT, INTERCEPT, CUSHION, HOLD, SHOOT} <= phases
    assert bot.stats["holds"] > 5 and bot.stats["shots"] > 3
    # and the shaper pays it for exactly that: held steps and on-target shots
    assert sh.stats["hold_steps"] > 100
    assert sh.stats["on_target"] > 0


def test_bot_takes_speed_off_the_snipers_shots():
    e, bot, sh, fast, phases = _possessions("sniper")
    mn = np.array([p["mn"] for p in fast])
    assert len(fast) >= 40
    assert float(np.mean(mn < CONTROL_SPEED)) > 0.15
    assert sh.stats["cushion_sum"] > 50
    # ...and BLOCKS the goal-bound fast ones instead of retreating from
    # them: the first version cushioned everything and lost 23-61, a
    # demonstration whose net value the learner rightly refused.
    assert BLOCK in phases and bot.stats["blocks"] > 20


def test_bot_emits_two_dim_actions_for_a_position_only_env():
    e = BatchAirHockeyEnv(4, opponent_policy="idle", action_mode="position")
    e.reset(seed=0)
    bot = CushionBot(e)
    a = bot.act()
    assert a.shape == (4, 2)
    bot.reset(np.array([True, False, False, False]))
    assert bot.phase[0] == WAIT
