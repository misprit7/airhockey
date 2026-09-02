"""The named curriculum: each stage's target term must be the loud one.

The old 4-stage table had contact at 0.1 beside a 160-point goal, so every
dense term was noise and three trainers learned to loiter on the only real
signal. These pin the shape the curriculum was rebuilt to have.
"""

from __future__ import annotations

import numpy as np

from airhockey.rewards import (CURRICULUM, CURRICULUM_ORDER, BatchRewardShaper,
                               STAGE_SCORING, curriculum_shaper_kwargs)


def test_stage_order_and_opponents():
    assert CURRICULUM_ORDER == ["proximity", "contact", "scoring", "goalie", "selfplay"]
    assert CURRICULUM["proximity"]["opponent"] == "idle"
    assert CURRICULUM["goalie"]["opponent"] == "goalie"
    assert CURRICULUM["selfplay"]["opponent"] == "external"


def test_early_stages_pay_nothing_for_goals():
    for name in ("proximity", "contact"):
        s = CURRICULUM[name]
        assert s["goal_reward"] == 0 and s["goal_penalty"] == 0, name


def test_proximity_stage_only_pays_proximity():
    s = curriculum_shaper_kwargs("proximity")
    assert s["proximity_weight"] > 0
    assert all(v == 0 for k, v in s.items() if k != "proximity_weight")


def test_contact_is_not_noise_next_to_a_goal():
    """A goal should be worth tens of contacts, not thousands."""
    for name in ("scoring", "goalie", "selfplay"):
        s = CURRICULUM[name]
        ratio = s["goal_reward"] / s["contact_reward"]
        assert 20 <= ratio <= 100, f"{name}: goal/contact = {ratio}"


def test_stage_budgets_are_positive_and_total_is_sane():
    total = sum(CURRICULUM[n]["steps"] for n in CURRICULUM_ORDER)
    assert all(CURRICULUM[n]["steps"] > 0 for n in CURRICULUM_ORDER)
    assert 1_000_000 <= total <= 10_000_000


def test_shaper_accepts_every_stage():
    obs = np.zeros((2, 15), dtype=np.float32)
    for name in CURRICULUM_ORDER:
        sh = BatchRewardShaper(2, stage=STAGE_SCORING, **curriculum_shaper_kwargs(name))
        info = {"puck_x": np.full(2, 0.5), "puck_y": np.full(2, 0.5),
                "puck_vx": np.zeros(2), "puck_vy": np.zeros(2),
                "pad_x": np.full(2, 0.5), "pad_y": np.full(2, 0.3),
                "score_agent": np.zeros(2, dtype=np.int64),
                "score_opponent": np.zeros(2, dtype=np.int64)}
        sh.reset(obs, info=info)
        r = sh.compute(obs, np.zeros(2), info=info)
        assert r.shape == (2,) and np.all(np.isfinite(r))
        if name == "proximity":
            # 0.2 m from the puck: 0.1 * exp(-3*0.2) = 0.055 per step
            assert abs(r[0] - 0.1 * np.exp(-0.6)) < 1e-6
