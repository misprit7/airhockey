"""Common physical skill trials for legacy and arrival policies (simulation only).

Outcomes are measured from contacts and goal events, not training reward. The
legacy adapter executes exactly its original three commands in the same physics.
"""

from __future__ import annotations

import numpy as np

from airhockey.arrival_env import ArrivalEnv
from airhockey.skill_benchmark import Fixtures, make_fixtures


TASKS = ("stationary", "moving", "cushion", "defense")


def random_paddle_starts(rng, puck, workspace, clearance):
    """Sample reachable initial positions without overlapping the puck."""
    low = [workspace["min_x"] + 0.01, workspace["min_y"] + 0.01]
    high = [workspace["max_x"] - 0.01, workspace["max_y"] - 0.01]
    paddle = rng.uniform(low, high, (len(puck), 2))
    for _ in range(100):
        invalid = np.linalg.norm(paddle - puck[:, :2], axis=1) < clearance
        if not invalid.any():
            return paddle
        paddle[invalid] = rng.uniform(low, high, (invalid.sum(), 2))
    raise ValueError("could not sample nonoverlapping paddle positions")


class LegacyTrials(ArrivalEnv):
    """Transport legacy three-wide actions through the instrumented environment."""

    def _decode(self, action, opponent=False):
        b = self.base
        dyn = b._opp_dyn if opponent else b._agent_dyn
        target = b._action_low + (action[:, :2] + 1) * 0.5 * (
            b._action_high - b._action_low
        )
        return target, b.accel_fraction(action[:, 2]) * dyn["max_accel"]

    @staticmethod
    def policy_obs(obs):
        return np.concatenate((obs[:, :18], obs[:, 21:25]), axis=1)

    @staticmethod
    def transport(action):
        out = np.zeros((len(action), 6), np.float32)
        out[:, :3] = action
        return out


def fixtures(seed, per_task, wide=False):
    f = make_fixtures(seed, per_task)
    rng = np.random.default_rng(seed + 1)
    n = per_task
    # Direct on-goal incoming shots, broad speeds/angles. Their free-flight
    # intersection is in the mouth; surviving a miss is not called a save.
    p = np.zeros((n, 4))
    p[:, 0] = rng.uniform(0.15, 0.85, n)
    p[:, 1] = rng.uniform(0.85, 1.65, n)
    aim = rng.uniform(0.43, 0.57, n)
    if wide:
        from airhockey.dynamics import workspace_in_sim
        from airhockey.physics import TableConfig

        ws, cfg = workspace_in_sim(), TableConfig()
        r = cfg.puck_radius + cfg.paddle_radius
        shoot = f.task < 2
        f.puck[shoot, 0] = rng.uniform(
            ws["min_x"] + 0.03, ws["max_x"] - 0.03, shoot.sum()
        )
        f.puck[shoot, 1] = rng.uniform(
            ws["min_y"] + r + 0.04, ws["max_y"] + r - 0.03, shoot.sum()
        )
        f.paddle[shoot] = np.clip(
            f.puck[shoot, :2]
            + rng.uniform([-0.08, -0.20], [0.08, -0.12], (shoot.sum(), 2)),
            [ws["min_x"] + 0.01, ws["min_y"] + 0.01],
            [ws["max_x"] - 0.01, ws["max_y"] - 0.01],
        )
        f.puck[f.task == 1, 2:] = rng.uniform(
            [-0.65, -0.35], [0.65, 0.25], (per_task, 2)
        )
        mouth = cfg.goal_width / 2 - cfg.puck_radius - 0.005
        aim = rng.uniform(0.5 - mouth, 0.5 + mouth, n)
    direction = np.column_stack((aim - p[:, 0], -p[:, 1]))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    p[:, 2:] = direction * rng.uniform(2, 8, (n, 1))
    paddle = np.column_stack((rng.uniform(0.4, 0.6, n), np.full(n, 0.25)))
    if wide:
        paddle[:, 0] = rng.uniform(0.47, 0.53, n)
    # Initialize defense using a fixture-bearing task, then use the game request
    # for its policy observation, since 4.0 had no separate defense request.
    result = Fixtures(
        np.r_[f.task, np.full(n, 2)],
        np.vstack((f.puck, p)),
        np.vstack((f.paddle, paddle)),
        np.r_[f.aim, np.full(n, 0.5)],
    )
    return result, np.repeat(np.arange(4), per_task)


def evaluate_skills(
    agent,
    *,
    legacy=False,
    seed=20261101,
    per_task=100,
    accel=60,
    planner=True,
    realistic=True,
    randomize=True,
    wide=False,
    teacher_fixtures=None,
    guard=False,
    game_requests=False,
    random_paddle=False,
):
    import torch

    if guard and not legacy:
        raise ValueError(
            "this benchmark's command guard currently supports legacy actions only"
        )
    f, task = fixtures(seed, per_task, wide)
    if teacher_fixtures is not None:
        for key in ("puck", "paddle", "aim"):
            getattr(f, key)[: 3 * per_task] = teacher_fixtures[key]
    if random_paddle:
        from airhockey.dynamics import workspace_in_sim
        from airhockey.physics import TableConfig

        ws, cfg = workspace_in_sim(), TableConfig()
        shoot = task < 2
        f.paddle[shoot] = random_paddle_starts(
            np.random.default_rng(seed + 9),
            f.puck[shoot],
            ws,
            cfg.puck_radius + cfg.paddle_radius + 0.01,
        )
    n = len(task)
    extended_policy = legacy and agent.cfg.obs_shape["state"][0] == 42
    practice = legacy and (extended_policy or guard)
    env_cls = LegacyTrials if legacy else ArrivalEnv
    if practice:
        from airhockey.legacy_practice import LegacyPracticeEnv

        env_cls = LegacyPracticeEnv
    env = env_cls(
        n,
        seed=seed,
        accel=accel,
        realistic=realistic,
        randomize=randomize,
    )
    if practice:
        env.motion_guard = guard
    obs = env.reset(seed=seed, fixtures=f)
    # A blocked/stalled puck belongs to this one attempt. Do not let the
    # game referee replace it with another serve during the defense window.
    game = (task == 3) | (game_requests & (task < 2))
    env.defense_trials = game
    env.task[game] = 3
    env.desired_speed[game] = 3
    env.aim[game] = 0.5
    task_start = 30 if practice else 33
    obs[game, task_start : task_start + 4] = [0, 0, 0, 1]
    obs[game, task_start + 4] = 0.5
    obs[game, task_start + 5] = 1
    done = np.zeros(n, bool)
    gf, ga, contacts, aimed = [np.zeros(n, int) for _ in range(4)]
    first_contact = np.full(n, np.nan)
    peak, cost, effort, holds = [np.zeros(n) for _ in range(4)]
    acceleration = np.zeros(n)
    goal_errors = np.full(n, np.nan)
    incoming_speed = np.linalg.norm(f.puck[:, 2:], axis=1)
    original_contact, original_goal = (
        env.engine.contact_callback,
        env.engine.goal_callback,
    )
    bank_aimed = np.zeros(n, int)

    def contact(event):
        original_contact(event)
        if event["body"] != "agent":
            return
        ii = event["indices"]
        use = ~done[ii]
        i = ii[use]
        first = contacts[i] == 0
        first_contact[i[first]] = event["time"][use][first]
        contacts[i] += 1
        vx, vy = event["outgoing_before_speed_cap"][use].T
        x, y = env.engine.puck_x[i], env.engine.puck_y[i]
        crossing = x + vx * (env.cfg.height - y) / np.maximum(vy, 1e-9)
        mouth = env.cfg.goal_width / 2 - env.cfg.puck_radius
        aimed[i] += (vy > 1) & (abs(crossing - env.cfg.width / 2) < mouth)
        from airhockey.shot_flight import PARAMETERS, first_goal_crossing

        forward = vy > 1
        if forward.any():
            ii = i[forward]
            _, on_goal = first_goal_crossing(
                np.column_stack((x[forward], y[forward], vx[forward], vy[forward])),
                {name: getattr(env.engine, name)[ii] for name in PARAMETERS},
                env.cfg,
            )
            bank_aimed[ii] += on_goal

    def goal(event):
        original_goal(event)
        ii = event["indices"]
        use = ~done[ii]
        i = ii[use]
        side = event["agent"][use]
        gf[i] += side
        ga[i] += ~side
        good = i[side]
        goal_errors[good] = abs(event["crossing_x"][use][side] - f.aim[good])

    env.engine.contact_callback, env.engine.goal_callback = contact, goal
    old_mpc, old_mean = agent.cfg.mpc, agent._prev_mean_batch
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    agent.cfg.mpc, agent._prev_mean_batch = planner, None
    t0 = torch.ones(n, dtype=torch.bool)
    try:
        for tick in range(101):
            torch.compiler.cudagraph_mark_step_begin()
            view = env.policy_obs(obs) if legacy and not practice else obs
            if legacy and practice and not extended_policy:
                view = obs[:, :22]
            action = agent.act(torch.from_numpy(view), t0=t0, eval_mode=True).numpy()
            obs, _, term, trunc, info = env.step(
                env.transport(action) if legacy and not practice else action
            )
            use = ~done
            peak[use] = np.maximum(peak[use], info["load_peak"][use])
            cost[use] += info["load_cost"][use]
            effort[use] = info["effort"][use]
            acceleration[use] = info["peak_accel"][use]
            holds[use] = env.control_best[use]
            end = (gf + ga > 0) | term | trunc
            # Cushion trials have the same 600 ms deadline used by 4.0.
            # Defense is assessed up to two seconds or the first actual goal.
            end |= (tick + 1) * env.base.action_dt >= 2.0 - 1e-9
            done |= end
            t0[:] = False
            if done.all():
                break
    finally:
        agent.cfg.mpc, agent._prev_mean_batch = old_mpc, old_mean
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    success = (gf > 0) & (contacts > 0)
    success[task == 2] = (contacts[task == 2] > 0) & (holds[task == 2] >= 0.2)
    success[task == 3] = (contacts[task == 3] > 0) & (ga[task == 3] == 0)
    result = dict(
        seed=seed,
        per_task=per_task,
        accel=accel,
        planner=planner,
        realistic=realistic,
        randomize=randomize,
        tasks={},
        suite="workspace-v2" if wide else "central-v2",
        motion_guard=guard,
        shooting_request="game" if game_requests else "isolated_skill",
        shooting_start="random_nonoverlapping_paddle"
        if random_paddle
        else "aligned_approach",
        planning=dict(
            iterations=agent.cfg.iterations,
            samples=agent.cfg.num_samples,
            horizon=agent.cfg.horizon,
        ),
    )
    for k, name in enumerate(TASKS):
        m = task == k
        t = first_contact[m]
        result["tasks"][name] = dict(
            attempts=int(m.sum()),
            successes=int(success[m].sum()),
            contact_trials=int((contacts[m] > 0).sum()),
            on_target_contact_trials=int((aimed[m] > 0).sum()),
            on_goal_contact_trials_including_banks=int((bank_aimed[m] > 0).sum()),
            goals_for=int(gf[m].sum()),
            goals_against=int(ga[m].sum()),
            first_contact_s=float(np.nanmedian(t)) if np.isfinite(t).any() else None,
            peak_modeled_load=float(peak[m].max()),
            mean_load_cost=float(cost[m].mean()),
            mean_effort=float(effort[m].mean()),
            peak_actual_acceleration=float(acceleration[m].max()),
        )
    for lo, hi in ((2, 4), (4, 6), (6, 8.01)):
        m = (task == 3) & (incoming_speed >= lo) & (incoming_speed < hi)
        result["tasks"]["defense"][f"speed_{lo}_{hi:g}"] = dict(
            attempts=int(m.sum()),
            saves=int(success[m].sum()),
        )
    return result
