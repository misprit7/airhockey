"""The exchange shaper pays outcomes, not states.

Exists because every dense variant taught the same lesson: under
state-income shaping, a policy that parks goal-side and never shoots is
OPTIMAL (sac_v8: healthy optimizer, declining reward, 0-0 replays). These
tests pin the properties that make passivity earn nothing here.
"""

from __future__ import annotations

import numpy as np

from airhockey.physics import TableConfig
from airhockey.rewards import ExchangeRewardShaper


def _info(n=1, **kw):
    base = dict(puck_x=np.full(n, 0.5), puck_y=np.full(n, 0.5),
                puck_vx=np.zeros(n), puck_vy=np.zeros(n),
                pad_x=np.full(n, 0.5), pad_y=np.full(n, 0.2),
                opp_x=np.full(n, 0.5), opp_y=np.full(n, 1.8),
                score_agent=np.zeros(n, dtype=np.int64),
                score_opponent=np.zeros(n, dtype=np.int64))
    base.update({k: np.asarray(v, dtype=float) if k[:5] != "score"
                 else np.asarray(v) for k, v in kw.items()})
    return base


def _step(sh, info):
    obs = np.zeros((sh.n_envs, 1), dtype=np.float32)
    return sh.compute(obs, np.zeros(sh.n_envs), info=info)


def test_parking_goal_side_earns_nothing():
    """The failure mode this shaper exists to kill: perfect defensive
    positioning, forever, must pay exactly zero."""
    sh = ExchangeRewardShaper(1)
    sh.reset(np.zeros((1, 1)), info=_info())
    total = 0.0
    for _ in range(500):
        total += float(_step(sh, _info(puck_y=[0.6], puck_vy=[0.0]))[0])
    assert total == 0.0


def test_straight_shot_through_the_mouth_pays_on_the_crossing_step():
    sh = ExchangeRewardShaper(1)
    sh.reset(np.zeros((1, 1)), info=_info(puck_y=[0.9]))
    r0 = _step(sh, _info(puck_y=[0.99], puck_vx=[0.0], puck_vy=[4.0]))
    assert r0[0] == 0.0, "no reward before the midline"
    r1 = _step(sh, _info(puck_y=[1.01], puck_vx=[0.0], puck_vy=[4.0]))
    # on-target + beats-opp check + velocity bonus; opponent at centre CAN
    # reach a centre shot, so beats_opp must NOT pay here
    expect = sh.shot_on_target + sh.vel_bonus_per_ms * 4.0
    assert abs(float(r1[0]) - expect) < 1e-6, (float(r1[0]), expect)
    r2 = _step(sh, _info(puck_y=[1.05], puck_vx=[0.0], puck_vy=[4.0]))
    assert r2[0] == 0.0, "a shot is scored once, not per step"


def test_wide_shot_pays_nothing():
    sh = ExchangeRewardShaper(1)
    sh.reset(np.zeros((1, 1)), info=_info(puck_y=[0.99], puck_x=[0.08]))
    # hugging the left rail, dead straight: crosses the goal LINE far
    # outside the mouth
    r = _step(sh, _info(puck_x=[0.08], puck_y=[1.01],
                        puck_vx=[0.0], puck_vy=[4.0]))
    assert float(r[0]) == 0.0


def test_bank_shot_is_scored_where_the_lossy_wall_sends_it():
    """A shot aimed at the rail must be scored through e_n/e_t, not a
    mirror. This one, aimed to bounce once and cut back to the mouth,
    only scores if the prediction bends the way the measured wall does."""
    sh = ExchangeRewardShaper(1)
    c = TableConfig()
    # From just past midline at the left, angled right-and-up steeply
    # enough to hit the right rail then come back toward the mouth.
    x0, y0 = 0.3, 1.01
    vx, vy = 2.0, 2.0
    xg, tg = sh._predict_goal_crossing(np.array([x0]), np.array([y0]),
                                       np.array([vx]), np.array([vy]))
    assert np.isfinite(xg[0])
    # hand-computed: wall hit at x=W-r, then vx=-vx*e_n, vy=vy*e_t
    t1 = (c.width - c.puck_radius - x0) / vx
    y1 = y0 + vy * t1
    vx2, vy2 = -vx * c.wall_restitution, vy * c.wall_tangential
    t2 = (c.height - y1) / vy2
    x_expect = (c.width - c.puck_radius) + vx2 * t2
    assert abs(float(xg[0]) - x_expect) < 1e-9


def test_blocked_shot_ends_the_exchange_with_no_further_pay():
    sh = ExchangeRewardShaper(1)
    sh.reset(np.zeros((1, 1)), info=_info(puck_y=[0.99]))
    _step(sh, _info(puck_y=[1.01], puck_vy=[3.0]))       # shot scored
    assert not sh.end_exchange[0]
    # opponent blocks it; puck comes back below 0.75*H
    r = _step(sh, _info(puck_y=[1.4], puck_vy=[-2.0]))
    assert sh.end_exchange[0], "returned shot must end the exchange"
    assert float(r[0]) == 0.0


def test_goals_and_concessions_keep_terminal_values():
    sh = ExchangeRewardShaper(1)
    sh.reset(np.zeros((1, 1)), info=_info())
    r = _step(sh, _info(score_agent=[1]))
    assert float(r[0]) == sh.goal_reward and sh.end_exchange[0]
    sh.reset(np.zeros((1, 1)), info=_info())
    r = _step(sh, _info(score_opponent=[1]))
    assert float(r[0]) == sh.goal_penalty and sh.end_exchange[0]
