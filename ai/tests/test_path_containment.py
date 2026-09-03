"""The profile body's PATH stays inside the box, not only its target.

On the rig a learned policy's corner-to-corner targets swung the cart model
105 mm past the end rail at 12 m/s / 60 m/s^2: a vector profile turning at
speed has a radius of v^2/a, and only the target was ever clamped. The law
now contains the position too (motionProfileContain), and the simulator's
profile body runs the same bounded law, so what the sim shows the policy is
what the machine will do.
"""
from __future__ import annotations

import numpy as np

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.dynamics import MAX_SPEED_M_S
from airhockey.motion import DEFAULT_SIM_DT, CartState, advance


def test_bounded_law_keeps_the_cart_inside_where_the_unbounded_one_leaves():
    box = (1350.0, 1917.5, 172.9, 793.0)
    out = {}
    for bounded in (False, True):
        c = CartState(1)
        c.reset(1600.0, 480.0)
        worst = 0.0
        for k in range(500):                     # 5 s of 10 ms ticks
            ty = box[2] if (k // 2) % 2 == 0 else box[3]   # flip every 20 ms
            advance(c, np.array([box[1]], np.float32), np.array([ty], np.float32),
                    np.array([12000.0]), np.array([60000.0]), 0.003,
                    DEFAULT_SIM_DT, 50, bounds=box if bounded else None)
            worst = max(worst, float(c.x[0]) - box[1], box[0] - float(c.x[0]),
                        float(c.y[0]) - box[3], box[2] - float(c.y[0]))
        out[bounded] = worst
    assert out[False] > 20.0, f"unbounded law stayed inside ({out[False]:.1f} mm); the case is gone"
    assert out[True] <= 1e-3, f"bounded law left the box by {out[True]:.2f} mm"


def test_sim_profile_body_never_leaves_its_box_under_bang_bang_targets():
    """The env's cart state (not just the drawn paddle) stays in the box at
    the pinned 60 m/s^2 with the worst-case policy: alternate corners."""
    e = BatchAirHockeyEnv(n_envs=4, opponent_policy="idle", domain_randomize=True)
    e.reset(seed=1)
    ws = e._ws
    assert e._agent_dyn["bounds_mm"] is not None
    assert e._agent_dyn["max_speed"][0] == MAX_SPEED_M_S
    worst = 0.0
    for k in range(400):
        a = np.array([[1.0, -1.0] if (k // 2) % 2 == 0 else [-1.0, -1.0]] * 4, np.float32)
        e.step(a)
        x, y = e._agent_dyn["x"], e._agent_dyn["y"]
        worst = max(worst, float(np.max(x - ws["max_x"])), float(np.max(ws["min_x"] - x)),
                    float(np.max(y - ws["max_y"])), float(np.max(ws["min_y"] - y)))
    assert worst <= 1e-6, f"cart left the box by {1000 * worst:.2f} mm"


def test_a_robot_bodied_far_side_is_contained_in_its_mirrored_box():
    e = BatchAirHockeyEnv(n_envs=2, opponent_policy="external", opponent_body="robot",
                          domain_randomize=True)
    e.reset(seed=2)
    assert e._opp_dyn["bounds_mm"] is not None
    w = e._ws_opp
    for k in range(300):
        tx = np.full(2, w["max_x"] + 0.3)
        ty = np.full(2, (w["max_y"] + 0.3) if (k // 2) % 2 == 0 else (w["min_y"] - 0.3))
        e.set_opponent_actions(tx, ty)
        e.step(np.zeros((2, 2), np.float32))
        x, y = e._opp_dyn["x"], e._opp_dyn["y"]
        assert np.all(x <= w["max_x"] + 1e-6) and np.all(x >= w["min_x"] - 1e-6)
        assert np.all(y <= w["max_y"] + 1e-6) and np.all(y >= w["min_y"] - 1e-6)
