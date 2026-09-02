"""The deployment adapter: a tracker report in mm becomes the observation the
checkpoint trained on, and the action comes back as a target in mm.

The encoder is tested with no checkpoint at all against the simulator's own
observation of the same state; the checkpoint tests skip when there is no
run to load.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "shared"))
import cdpr_geometry as geom  # noqa: E402

from airhockey.batch_env import BatchAirHockeyEnv  # noqa: E402
from airhockey.deploy import (PUCK_VELOCITY_WINDOW_S, ReportEncoder,  # noqa: E402
                              mm_velocity_to_sim)
from airhockey.dynamics import sim_to_table_mm, table_mm_to_sim  # noqa: E402


def test_velocity_mapping_is_the_derivative_of_the_position_mapping():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rng.uniform(0, 2000), rng.uniform(0, 1000)
        vx, vy = rng.uniform(-5000, 5000, size=2)
        dt = 0.005
        sx0, sy0 = table_mm_to_sim(x, y)
        sx1, sy1 = table_mm_to_sim(x + vx * dt, y + vy * dt)
        svx, svy = mm_velocity_to_sim(vx, vy)
        assert svx == pytest.approx((sx1 - sx0) / dt, rel=1e-9, abs=1e-9)
        assert svy == pytest.approx((sy1 - sy0) / dt, rel=1e-9, abs=1e-9)


def _report_from_truth(env, hist, t):
    """What the tracker would report for the env's true state, in mm."""
    e = env.engine
    mx, my = sim_to_table_mm(float(e.paddle_agent_x[0]), float(e.paddle_agent_y[0]))
    qx, qy = sim_to_table_mm(float(e.paddle_opp_x[0]), float(e.paddle_opp_y[0]))
    return {"puck": list(hist), "mallet": (mx, my), "opponent": (qx, qy), "t_s": t}


def test_encoder_matches_the_simulators_observation():
    """Feed the encoder reports built from the sim's TRUE state and compare
    with the sim's own (truth) observation of that state, tick by tick.
    Positions must agree to float precision; the puck velocity is a 30 ms
    fit over 10 ms ticks, exact on a straight segment, so it is compared
    away from bounces."""
    # DR on so the cap features are the pinned training constants, which is
    # what the encoder writes; with it off the env reads nominal (1.0).
    env = BatchAirHockeyEnv(n_envs=1, opponent_policy="follow",
                            realistic_perception=False, domain_randomize=True)
    obs = env.reset(seed=3)
    enc = ReportEncoder(env.table_config)
    hist: list[tuple[float, float, float]] = []
    t = 0.0
    n_checked = 0
    for k in range(300):
        e = env.engine
        px, py = sim_to_table_mm(float(e.puck_x[0]), float(e.puck_y[0]))
        hist.insert(0, (px, py, t))
        hist = hist[:40]
        got = enc.encode(_report_from_truth(env, hist, t))
        if k >= 4:
            # Positions and the finite-difference velocities the env itself uses.
            np.testing.assert_allclose(got[[0, 1, 4, 5, 8, 9]], obs[0, [0, 1, 4, 5, 8, 9]],
                                       atol=2e-6, err_msg=f"tick {k}")
            np.testing.assert_allclose(got[[6, 7, 10, 11]], obs[0, [6, 7, 10, 11]],
                                       atol=2e-4, err_msg=f"tick {k}")
            # Puck velocity: exact where the last 30 ms had no bounce or hit,
            # INCLUDING inside the last tick -- a fit over positions trails an
            # instantaneous truth by up to one tick at a reversal, exactly as
            # the sim's own tracker model does.
            recent = hist[:4]
            dxs = [recent[i][0] - recent[i + 1][0] for i in range(3)]
            dys = [recent[i][1] - recent[i + 1][1] for i in range(3)]
            last_sim = mm_velocity_to_sim(dxs[0], dys[0])
            straight = (all(d * dxs[0] >= 0 for d in dxs) and all(d * dys[0] >= 0 for d in dys)
                        and last_sim[0] * obs[0, 2] >= 0 and last_sim[1] * obs[0, 3] >= 0)
            if straight:
                np.testing.assert_allclose(got[[2, 3]], obs[0, [2, 3]], atol=2e-3,
                                           err_msg=f"tick {k}")
                n_checked += 1
            assert got[12] == 1.0
            np.testing.assert_allclose(got[13:], obs[0, 13:])
        obs = env.step(np.array([[0.2 * np.sin(k / 20.0), -0.5]]))[0]
        t += env.action_dt
    assert n_checked > 150


def test_encoder_without_a_puck_holds_the_last_fix_and_parks_an_unseen_opponent():
    enc = ReportEncoder()
    t = 0.0
    rep = {"puck": [(1500.0, 500.0, t)], "mallet": (geom.HOME_X, geom.HOME_Y),
           "opponent": None, "t_s": t}
    o1 = enc.encode(rep)
    o2 = enc.encode({"puck": [], "mallet": (geom.HOME_X, geom.HOME_Y),
                     "opponent": None, "t_s": t + 0.01})
    np.testing.assert_allclose(o2[:2], o1[:2])
    assert o2[2] == o2[3] == 0.0
    np.testing.assert_allclose(o2[8:10], enc._opp_default)
    assert o2[10] == o2[11] == 0.0


def test_a_long_gap_restarts_the_velocities_and_flags_a_fresh_start():
    enc = ReportEncoder()
    rep = lambda t, mx: {"puck": [(1500.0, 500.0, t)], "mallet": (mx, 500.0),  # noqa: E731
                         "opponent": None, "t_s": t}
    enc.encode(rep(0.0, 1500.0))
    enc.fresh = False
    o = enc.encode(rep(0.01, 1510.0))
    assert o[7] != 0.0 and not enc.fresh                # own vy moved (grid x -> sim y)
    o = enc.encode(rep(1.0, 1600.0))                    # a held second later
    assert o[6] == o[7] == 0.0 and enc.fresh


# ── With a checkpoint ──────────────────────────────────────────────────

def _latest():
    from airhockey.policy_loader import resolve_checkpoint
    try:
        return resolve_checkpoint("latest")
    except FileNotFoundError:
        return None


def _report_from_camera(env, hist, t):
    """What the real tracker would report: the puck and the opponent from
    the sim's camera model (latency, noise, blind spot), the own mallet from
    the controller. Building the history from TRUTH instead gave the policy
    a cleaner puck than it trained on and it played differently."""
    e = env.engine
    seen, _, _ = env._camera_read()
    px, py = sim_to_table_mm(float(seen[0, 0]), float(seen[0, 1]))
    hist.insert(0, (px, py, t))
    del hist[40:]
    mx, my = sim_to_table_mm(float(e.paddle_agent_x[0]), float(e.paddle_agent_y[0]))
    qx, qy = sim_to_table_mm(float(seen[0, 4]), float(seen[0, 5]))
    return {"puck": list(hist), "mallet": (mx, my), "opponent": (qx, qy), "t_s": t}


def _scoring_run():
    """A checkpoint whose prior is known to score: the goalie-stage one if
    it exists (18 goals in 12 x 20 s vs idle through this path, matching the
    direct path's 19), else whatever is newest."""
    return "curriculum_goalie" if (_ROOT / "runs" / "curriculum_goalie" / "agent.pt").exists() else "latest"


@pytest.mark.skipif(_latest() is None, reason="no checkpoint under runs/")
def test_deploy_path_scores_in_the_simulator():
    """The adapter, fed what the tracker would report, plays the sim: goals
    against an idle opponent over three 20 s games, and no more conceded
    than scored."""
    from airhockey.deploy import TDMPC2Policy
    pol = TDMPC2Policy(_scoring_run(), 12000.0, 60000.0, plan_iterations=0, device="cpu")
    gf = ga = 0
    for seed in range(3):
        env = BatchAirHockeyEnv(n_envs=1, opponent_policy="idle", domain_randomize=True,
                                realistic_perception=True, camera_delay=(0.0051, 0.0103),
                                max_episode_steps=2000)
        env.reset(seed=seed)
        pol.reset()
        hist: list[tuple[float, float, float]] = []
        t = 0.0
        for _ in range(2000):
            cmd = pol(_report_from_camera(env, hist, t))
            sx, sy = table_mm_to_sim(cmd.x_mm, cmd.y_mm)
            lo, hi = env._action_low, env._action_high
            a = 2.0 * (np.array([sx, sy]) - lo) / (hi - lo) - 1.0
            env.step(a[None, :].astype(np.float32))
            t += env.action_dt
        gf += int(env.engine.score_agent[0])
        ga += int(env.engine.score_opponent[0])
    assert gf >= 1, "the deploy path never scored"
    assert ga <= gf


@pytest.mark.skipif(_latest() is None, reason="no checkpoint under runs/")
def test_runner_drives_a_checkpoint_through_its_own_clamp():
    path = _ROOT / "ai" / "bin" / "run_policy.py"
    spec = importlib.util.spec_from_file_location("run_policy_deploy_test", path)
    rp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rp
    spec.loader.exec_module(rp)
    caps = rp.Caps()
    policy = rp.load_policy("tdmpc2:latest", caps)
    report = rp.ReportBuilder()
    prev = None
    n = 0
    for t, x, y in rp._synthetic_puck(duration_s=1.0):
        report.frame()
        report.add_puck(t, x, y)
        report.add_mallet(t, geom.HOME_X, geom.HOME_Y)
        if n % 2 == 0:
            action, _flags = rp.plan(policy, report, t, caps, prev)
            assert geom.in_workspace(action.x_mm, action.y_mm)
            assert action.speed_mm_s == caps.speed_max
            prev = (action.x_mm, action.y_mm)
        n += 1
    assert policy.last_ms < 5.0
