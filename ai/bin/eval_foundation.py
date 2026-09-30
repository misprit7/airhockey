#!/usr/bin/env python3
"""Sustained same-body matches and load measurements for legacy-policy pilots."""

import argparse
import json
from pathlib import Path
import time

from train_arrival import config, TDMPC2, torch, np
from airhockey.policy_loader import load_agent
from airhockey.legacy_practice import LegacyPracticeEnv
from airhockey.recorder import Recorder, FrameData
from airhockey.shot_flight import PARAMETERS, open_goal_outcomes


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / "agent.pt"
    meta_path = path.parent / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def finish(policy):
        policy.requires_motion_guard = bool(
            meta.get("motion_guard", False)
            or meta.get("controller") in {"observed_interception_v1", "arrival_play_v1"}
            or meta.get("action_mode") == "shot_intent"
        )
        return policy

    if meta.get("action_mode") == "shot_intent":
        from airhockey.shot_intent import INTENTS, IntentValue, ShotIntentPolicy

        if not np.array_equal(np.asarray(meta.get("intents")), INTENTS):
            raise ValueError(
                "checkpoint shot intents differ from this controller version"
            )
        selector = IntentValue().cuda()
        state = torch.load(path, map_location="cuda", weights_only=False)
        selector.load_state_dict(state["selector"])
        return finish(
            ShotIntentPolicy(
                load(path.parent / "fallback/agent.pt"),
                selector,
                controller_options=meta.get("controller_options"),
                interception_options=meta.get("interception_options"),
                fast_prior=meta.get("fast_prior", False),
            )
        )
    if meta.get("obs_dim") != 42:
        return finish(load_agent(path.parent.name, ckpt=path))
    cfg = config(argparse.Namespace(**meta["args"]), path.parent)
    cfg.obs_shape, cfg.action_dim = {"state": [42]}, 3
    cfg.discount_max, cfg.episode_length = 0.99, 3000
    cfg.plan_smooth_coef = float(
        meta.get("plan_smooth_coef", meta.get("args", {}).get("plan_smooth", 0.0))
    )
    cfg.plan_eval_mean = True
    agent = TDMPC2(cfg)
    agent.load(path)
    agent.training_algorithm = meta.get("algorithm", "TD-MPC2")
    agent.inference_mode = meta.get("inference_mode")
    if meta.get("inference_mode") == "prior":
        agent.cfg.mpc = False
    controller = meta.get("controller")
    if controller == "observed_interception_v1":
        from airhockey.interception import InterceptionPolicy

        return finish(InterceptionPolicy(agent))
    if controller == "arrival_play_v1":
        from airhockey.interception import InterceptionPolicy, InterceptionController
        from airhockey.arrival_play import ArrivalPlayPolicy

        fallback = InterceptionPolicy(agent)
        fallback.controller = InterceptionController(
            **meta.get("interception_options", {})
        )
        policy = ArrivalPlayPolicy(fallback, **meta.get("controller_options", {}))
        if meta.get("thermal_budget") is not None:
            from airhockey.load_budget import LoadBudgetPolicy

            policy = LoadBudgetPolicy(policy, **meta["thermal_budget"])
        return finish(policy)
    if controller is not None:
        raise ValueError(f"unknown checkpoint controller: {controller}")
    return finish(agent)


def match(
    agent,
    rival,
    *,
    games=8,
    seconds=180,
    seed=20261120,
    record=None,
    guard=False,
    shot_requests=False,
    self_play=None,
    policy_labels=None,
    initial_load=None,
):
    guard = guard or any(
        getattr(p, "requires_motion_guard", False) for p in (agent, rival)
    )
    if policy_labels is not None and len(policy_labels) != 2:
        raise ValueError("replay policy labels must identify both sides")
    if initial_load is not None and (
        len(initial_load) != 2
        or any(
            v is not None and (not np.isfinite(v) or not 0 <= v < 1)
            for v in initial_load
        )
    ):
        raise ValueError(
            "initial load requires two normalized levels in [0,1), or None"
        )
    env = LegacyPracticeEnv(
        games,
        seed=seed,
        game_fraction=1,
        selfplay_fraction=1,
        realistic=True,
        randomize=True,
        motion_guard=guard,
    )
    env.base.max_score, env.base.max_episode_time = 1000000, seconds + 1
    env.base.shot_types = shot_requests
    env.base.symmetric_referee = True
    obs = env.reset(seed=seed, opponent="external")
    if initial_load is not None:
        for side, level in enumerate(initial_load):
            if level is not None:
                env.loads[side].h[:] = level**2
                env.loads[side].observed[:] = env.loads[side].levels
        obs[:, 22:30] = env.loads[0].features()
    e = env.engine
    budget_start = {
        id(p): (getattr(p, "frames", 0), getattr(p, "cooling_frames", 0))
        for p in (agent, rival)
    }
    shots, aimed, contacts, blocks = [np.zeros((2, games), int) for _ in range(4)]
    peak, over_time, accel_peak, speed_peak = [np.zeros((2, games)) for _ in range(4)]
    accel_energy, high_accel_time, commanded_cap = [
        np.zeros((2, games)) for _ in range(3)
    ]
    armed = np.ones((2, games), bool)
    return_armed = np.ones((2, games), bool)
    forward_returns = np.zeros((2, games), int)
    original = e.contact_callback
    original_goal = e.goal_callback
    original_motion = env.base.motion_callback
    original_relaunch = env.base._relaunch
    referee_turnovers = np.zeros((2, games), int)
    puck_slow_time = np.zeros(games)
    puck_speed_integral = np.zeros(games)
    thermal_trace = []
    goal_events = []
    launches, launch_ids = [], []
    launch_parameters = {name: [] for name in PARAMETERS}
    returns, return_ids = [], []
    return_parameters = {name: [] for name in PARAMETERS}

    def contact(event):
        original(event)
        side = 0 if event["body"] == "agent" else 1
        ids = event["indices"]
        contacts[side, ids] += 1
        armed[1 - side, ids] = True
        return_armed[1 - side, ids] = True
        outgoing = event["outgoing_before_speed_cap"].copy()
        incoming = event["incoming"].copy()
        outgoing[:, 1] *= 1 if side == 0 else -1
        incoming[:, 1] *= 1 if side == 0 else -1
        y = e.puck_y[ids] if side == 0 else env.cfg.height - e.puck_y[ids]
        x = e.puck_x[ids]
        mouth = env.cfg.goal_width / 2 - env.cfg.puck_radius
        crossing = x + outgoing[:, 0] * (env.cfg.height - y) / np.maximum(
            outgoing[:, 1], 1e-9
        )
        hit = (
            armed[side, ids]
            & (outgoing[:, 1] > 1.5)
            & (
                np.linalg.norm(outgoing, axis=1)
                > np.linalg.norm(incoming, axis=1) + 0.2
            )
        )
        shots[side, ids[hit]] += 1
        # A deliberately aimed, stationary-paddle return need not accelerate
        # the puck. Keep the historical accelerating-shot metric separately.
        returned = return_armed[side, ids] & (outgoing[:, 1] > 1.5)
        forward_returns[side, ids[returned]] += 1
        return_armed[side, ids[returned]] = False
        if returned.any():
            returns.append(
                np.column_stack((x[returned], y[returned], outgoing[returned]))
            )
            return_ids.append(
                np.column_stack((np.full(returned.sum(), side), ids[returned]))
            )
            for name in PARAMETERS:
                return_parameters[name].append(getattr(e, name)[ids[returned]].copy())
        if hit.any():
            launches.append(np.column_stack((x[hit], y[hit], outgoing[hit])))
            launch_ids.append(np.column_stack((np.full(hit.sum(), side), ids[hit])))
            for name in PARAMETERS:
                launch_parameters[name].append(getattr(e, name)[ids[hit]].copy())
        aimed[side, ids[hit & (abs(crossing - 0.5) < mouth)]] += 1
        armed[side, ids[hit]] = False
        inbound_x = x - y * incoming[:, 0] / np.minimum(incoming[:, 1], -1e-9)
        outbound_x = x - y * outgoing[:, 0] / np.minimum(outgoing[:, 1], -1e-9)
        danger = (incoming[:, 1] < -1) & (abs(inbound_x - 0.5) < mouth)
        safe = (outgoing[:, 1] >= 0) | (abs(outbound_x - 0.5) >= mouth)
        blocks[side, ids[danger & safe]] += 1

    def motion(dt, old_agent, old_opp):
        original_motion(dt, old_agent, old_opp)
        for side, (dyn, old) in enumerate(
            ((env.base._agent_dyn, old_agent), (env.base._opp_dyn, old_opp))
        ):
            v = np.column_stack((dyn["vx"], dyn["vy"]))
            actual = np.linalg.norm(v - old, axis=1) / dt
            accel_energy[side] += actual**2 * dt
            high_accel_time[side] += (actual > 40) * dt
            accel_peak[side] = np.maximum(accel_peak[side], actual)
            speed_peak[side] = np.maximum(speed_peak[side], np.linalg.norm(v, axis=1))

    def goal(event):
        original_goal(event)
        armed[:, event["indices"]] = True
        return_armed[:, event["indices"]] = True
        for i, for_agent in zip(event["indices"], event["agent"]):
            goal_events.append(
                dict(
                    time=float(e.time[i]),
                    game=int(i),
                    scorer=0 if for_agent else 1,
                    puck=[
                        float(getattr(e, "puck_" + key)[i])
                        for key in ("x", "y", "vx", "vy")
                    ],
                    paddles=[
                        [
                            float(getattr(e, prefix + key)[i])
                            for key in ("x", "y", "vx", "vy")
                        ]
                        for prefix in ("paddle_agent_", "paddle_opp_")
                    ],
                    commands=[
                        [
                            float(dyn[key][i])
                            for key in ("command_x", "command_y", "command_accel")
                        ]
                        for dyn in (env.base._agent_dyn, env.base._opp_dyn)
                    ],
                    robot_observation=obs[i].tolist(),
                )
            )

    def relaunch(mask, **kwargs):
        ids = np.flatnonzero(mask)
        for side, key in ((0, "to_opponent"), (1, "to_agent")):
            destination = kwargs.get(key)
            if destination is not None:
                referee_turnovers[side, ids[np.asarray(destination, bool)]] += 1
        return original_relaunch(mask, **kwargs)

    e.contact_callback, env.base.motion_callback = contact, motion
    e.goal_callback = goal
    env.base._relaunch = relaunch
    for policy in {agent, rival}:
        policy._prev_mean_batch = None
    rec = Recorder() if record else None
    t0 = torch.ones(games, dtype=torch.bool)
    start = time.perf_counter()
    for tick in range(round(seconds / env.base.action_dt)):
        opposite = env.opponent_obs()
        view = obs if agent.cfg.obs_shape["state"][0] == 42 else obs[:, :22]
        opp_view = (
            opposite if rival.cfg.obs_shape["state"][0] == 42 else opposite[:, :22]
        )
        torch.compiler.cudagraph_mark_step_begin()
        if agent is rival:
            action = agent.act(
                torch.from_numpy(np.concatenate((view, opp_view))),
                t0=torch.cat((t0, t0)),
                eval_mode=True,
            ).numpy()
            a, b = action[:games], action[games:]
        else:
            a = agent.act(torch.from_numpy(view), t0=t0, eval_mode=True).numpy()
            b = rival.act(torch.from_numpy(opp_view), t0=t0, eval_mode=True).numpy()
        env.set_opponent_action(b)
        obs, _, _, _, _ = env.step(a)
        puck_speed = np.hypot(e.puck_vx, e.puck_vy)
        puck_slow_time += (puck_speed < 0.3) * env.base.action_dt
        puck_speed_integral += puck_speed * env.base.action_dt
        t0[:] = False
        for side in range(2):
            dyn = env.base._agent_dyn if side == 0 else env.base._opp_dyn
            commanded_cap[side] += dyn["command_accel"] * env.base.action_dt
            level = env.loads[side].levels.max(axis=(1, 2))
            peak[side] = np.maximum(peak[side], level)
            over_time[side] += (level >= 1) * env.base.action_dt
        if rec and e.time[0] <= 30 + 1e-6:
            rec.record(
                FrameData(
                    time=float(e.time[0]),
                    puck_x=float(e.puck_x[0]),
                    puck_y=float(e.puck_y[0]),
                    puck_vx=float(e.puck_vx[0]),
                    puck_vy=float(e.puck_vy[0]),
                    agent_x=float(e.paddle_agent_x[0]),
                    agent_y=float(e.paddle_agent_y[0]),
                    opponent_x=float(e.paddle_opp_x[0]),
                    opponent_y=float(e.paddle_opp_y[0]),
                    score_agent=int(e.score_agent[0]),
                    score_opponent=int(e.score_opponent[0]),
                )
            )
        if tick % 1500 == 1499:
            thermal_trace.append(
                dict(
                    seconds=float(e.time[0]),
                    fast=[load.levels[:, 0].max(axis=1).tolist() for load in env.loads],
                    slow=[load.levels[:, 1].max(axis=1).tolist() for load in env.loads],
                )
            )
            print(
                f"[match] t={e.time[0]:.0f}s score={e.score_agent.sum()}-{e.score_opponent.sum()} "
                f"peak_robot={peak[0].max():.3f} peak_opponent={peak[1].max():.3f}",
                flush=True,
            )
    flight_goals = np.zeros((2, games), int)
    if launches:
        rows = np.concatenate(launch_ids)
        success = open_goal_outcomes(
            np.concatenate(launches),
            {k: np.concatenate(v) for k, v in launch_parameters.items()},
            env.cfg,
        )
        np.add.at(flight_goals, (rows[:, 0], rows[:, 1]), success.astype(int))
    result = dict(
        games=games,
        seconds=seconds,
        seed=seed,
        initial_load=initial_load,
        wall_s=time.perf_counter() - start,
        goals_for=e.score_agent.tolist(),
        goals_against=e.score_opponent.tolist(),
        goal_events=goal_events,
        shots=shots.tolist(),
        on_target=aimed.tolist(),
        open_goal_flights_including_banks=flight_goals.tolist(),
        forward_returns=forward_returns.tolist(),
        referee_turnovers=referee_turnovers.tolist(),
        puck_fraction_below_0_3m_s=(puck_slow_time / seconds).tolist(),
        mean_puck_speed=(puck_speed_integral / seconds).tolist(),
        thermal_trace=thermal_trace,
        contacts=contacts.tolist(),
        blocks=blocks.tolist(),
        peak_load=peak.tolist(),
        over_limit_seconds=over_time.tolist(),
        peak_accel=accel_peak.tolist(),
        peak_speed=speed_peak.tolist(),
        rms_actual_acceleration=(np.sqrt(accel_energy / seconds)).tolist(),
        fraction_above_40m_s2=(high_accel_time / seconds).tolist(),
        mean_commanded_accel_cap=(commanded_cap / seconds).tolist(),
        motion_guard=guard,
        shot_requests="mixed" if shot_requests else "none",
        symmetric_referee=True,
        guard_interventions=env.guard_interventions.tolist(),
        guard_unresolved=env.guard_unresolved.tolist(),
    )
    return_goals = np.zeros((2, games), int)
    if returns:
        ri = np.concatenate(return_ids)
        success = open_goal_outcomes(
            np.concatenate(returns),
            {k: np.concatenate(v) for k, v in return_parameters.items()},
            env.cfg,
        )
        np.add.at(return_goals, (ri[:, 0], ri[:, 1]), success.astype(int))
        result["return_flight_launches"] = dict(
            state=np.concatenate(returns).tolist(),
            side_and_game=ri.tolist(),
            parameters={
                k: np.concatenate(v).tolist() for k, v in return_parameters.items()
            },
        )
    result["on_goal_returns_including_banks"] = return_goals.tolist()
    if any(hasattr(p, "cooling_frames") for p in (agent, rival)):
        result["cooldown"] = []
        for policy in (agent, rival):
            if not hasattr(policy, "cooling_frames"):
                result["cooldown"].append(None)
                continue
            prior_frames, prior_cooling = budget_start[id(policy)]
            result["cooldown"].append(
                dict(
                    fraction=(policy.cooling_frames - prior_cooling)
                    / max(1, policy.frames - prior_frames),
                    start=policy.start,
                    resume=policy.resume,
                    slow_budget=getattr(policy, "slow_budget", None),
                    shared_across_sides=agent is rival,
                )
            )
    result["shot_definition"] = "armed outgoing vy>1.5 m/s and speed gain>0.2 m/s"
    result["return_definition"] = (
        "first outgoing vy>1.5 m/s per exchange, including passive rebounds"
    )
    if launches:
        result["shot_flight_launches"] = dict(
            state=np.concatenate(launches).tolist(),
            side_and_game=rows.tolist(),
            parameters={
                k: np.concatenate(v).tolist() for k, v in launch_parameters.items()
            },
        )
    if rec:
        self_play = agent is rival if self_play is None else self_play
        labels = policy_labels or [
            getattr(p, "training_algorithm", "TD-MPC2") for p in (agent, rival)
        ]
        modes = ["MPC" if p.cfg.mpc else "prior" for p in (agent, rival)]
        record = Path(record)
        record.parent.mkdir(parents=True, exist_ok=True)
        rec.save(
            record,
            metadata=dict(
                algo=getattr(agent, "training_algorithm", "TD-MPC2") + " pilot",
                fps=50,
                opponent="self" if self_play else "reference",
                match_type="self-play" if self_play else "head-to-head",
                policy_sides=f"robot: {labels[0]} ({modes[0]}); opponent: {labels[1]} ({modes[1]})",
                policy_labels=list(map(str, labels)),
                simulation_only=True,
                accel_cap_m_s2=60,
                shot_requests="mixed" if shot_requests else "none",
            ),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--rival", type=Path)
    p.add_argument("--prior", action="store_true")
    p.add_argument("--rival-prior", action="store_true")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", type=Path)
    p.add_argument("--guard", action="store_true")
    p.add_argument("--compile-mpc", action="store_true")
    p.add_argument(
        "--shot-requests",
        action="store_true",
        help="Random explicit bank/straight requests instead of autonomous play",
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    agent = load(args.checkpoint)
    rival = load(args.rival) if args.rival else agent
    agent.cfg.mpc, agent.cfg.num_samples = (
        not args.prior and getattr(agent, "inference_mode", None) != "prior",
        256,
    )
    if rival is not agent:
        rival.cfg.mpc, rival.cfg.num_samples = (
            not args.rival_prior and getattr(rival, "inference_mode", None) != "prior",
            256,
        )
    if args.compile_mpc:
        for policy in {agent, rival}:
            if policy.cfg.mpc:
                policy._plan_batch = torch.compile(
                    policy._plan_batch, mode="reduce-overhead"
                )
    result = match(
        agent,
        rival,
        games=args.games,
        seconds=args.seconds,
        seed=args.seed,
        record=args.record,
        guard=args.guard,
        shot_requests=args.shot_requests,
        policy_labels=[str(args.checkpoint), str(args.rival or args.checkpoint)],
    )
    result.update(
        checkpoint=str(args.checkpoint),
        rival=str(args.rival) if args.rival else "self",
        planner=agent.cfg.mpc,
        rival_planner=rival.cfg.mpc,
        compiled_mpc=args.compile_mpc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
