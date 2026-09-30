#!/usr/bin/env python3
"""Physical outcomes and complete self-play recordings for the neural-only player."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from airhockey.neural_player import NeuralPlayer, PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.policy_benchmark import (
    fixtures,
    random_paddle_starts,
    bank_defense_launches,
)
from airhockey.recorder import Recorder, FrameData


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    net = NeuralPlayer(state["width"], history=state.get("history", 1), shot_conditioned=state.get("shot_conditioned", False))
    net.load_weights(state["model"])
    net.eval()
    return net, state


def checkpoint_report_sensing(path, state):
    """Keep later evaluations on the sensor model recorded with training."""
    if "report_sensing" in state:
        return bool(state["report_sensing"])
    if "report_sensing" in state.get("args", {}):
        return bool(state["args"]["report_sensing"])
    metadata = Path(path).parent / "run.json"
    return bool(json.loads(metadata.read_text()).get("args", {}).get("report_sensing", False)) if metadata.exists() else False


def checkpoint_continuous_rallies(path, state):
    if "continuous_rallies" in state.get("args", {}):
        return bool(state["args"]["continuous_rallies"])
    metadata = Path(path).parent / "run.json"
    return bool(json.loads(metadata.read_text()).get("args", {}).get("continuous_rallies", False)) if metadata.exists() else False


def set_initial_load(env, level):
    """Evaluation stress condition; never clears heat later in the match."""
    if not 0 <= level < 1:
        raise ValueError("initial load must be in [0,1)")
    for load in env.loads:
        load.h[:] = level**2
        load.observed[:] = level
    return env._features(env.base._make_obs_direct())


def audit_firmware_motion(env):
    """Reproduce each actual firmware integration interval on a separate copy."""
    from airhockey.arrival import copy_cart
    from airhockey.motion_guard import predict

    base = env.base
    original = base._update_dynamics
    audit = dict(
        peak_accel=np.zeros((2, env.n_envs)),
        minimum_margin=np.full((2, env.n_envs), np.inf),
        intervals_over_cap=np.zeros((2, env.n_envs), int),
        reproduction_error=np.zeros(6),
    )

    def update(dyn, target_x, target_y, dt, *args, **kwargs):
        side = 0 if dyn is base._agent_dyn else 1 if dyn is base._opp_dyn else None
        if side is None or dyn["type"] != "profile":
            return original(dyn, target_x, target_y, dt, *args, **kwargs)
        assert abs(dyn["ramp_s"] - 0.003) < 1e-8
        cart = copy_cart(dyn["cart"])
        for key in ("x", "y", "vx", "vy"):
            getattr(cart, key)[:] = dyn[key] * 1000
        result = original(dyn, target_x, target_y, dt, *args, **kwargs)
        peak = np.zeros(env.n_envs, np.float32)
        margin = np.full(env.n_envs, np.inf, np.float32)
        predict(
            cart,
            np.column_stack((dyn["command_x"], dyn["command_y"])),
            dyn["command_accel"],
            dyn["command_speed"],
            dt,
            np.asarray(dyn["bounds_mm"]) / 1000,
            peak,
            margin,
        )
        error = np.array(
            [
                np.max(np.abs(getattr(cart, k) - getattr(dyn["cart"], k)))
                for k in ("x", "y", "vx", "vy", "ax", "ay")
            ]
        )
        audit["reproduction_error"] = np.maximum(audit["reproduction_error"], error)
        if error[:2].max() > 0.05 or error[2:4].max() > 0.5:
            raise AssertionError(
                f"Firmware audit does not reproduce actual integration: {error}"
            )
        audit["peak_accel"][side] = np.maximum(audit["peak_accel"][side], peak / 1000)
        audit["minimum_margin"][side] = np.minimum(
            audit["minimum_margin"][side], margin / 1000
        )
        audit["intervals_over_cap"][side] += peak / 1000 > dyn["max_accel"] * 1.001
        return result

    base._update_dynamics = update
    return audit


def skills(
    net,
    seed=20261901,
    per_task=128,
    random_start=True,
    stochastic=False,
    defense_speed_range=(2, 8),
    bank_defense=False,
    wide_defense=False,
    random_defense=False,
    random_practice_opponent=False,
    initial_load=None,
    ideal_sensing=False,
    details_path=None,
    receiving_speed_range=None,
    thermal_gain=None,
    shot_request="random",
    report_sensing=False,
    recovery=False,
):
    torch.manual_seed(seed)
    f, tasks = fixtures(
        seed, per_task, wide=True, defense_speed_range=defense_speed_range
    )
    if bank_defense:
        f.puck[tasks == 3] = bank_defense_launches(
            seed, per_task, speed_range=defense_speed_range,
            goal_half_width=.14 if wide_defense else .02)
    env = NeuralTrainingEnv(
        len(f.task),
        stage=2,
        seed=seed,
        random_practice_opponent=random_practice_opponent,
        realistic=True,
        report_sensing=report_sensing,
        shot_conditioned=net.shot_conditioned or shot_request != "random",
        shot_request=shot_request,
    )
    if random_start:
        ids = f.task < 2
        f.paddle[ids] = random_paddle_starts(
            np.random.default_rng(seed + 17), f.puck[ids], env.base._ws, 0.11
        )
    if random_defense:
        ids = tasks == 3
        f.paddle[ids] = random_paddle_starts(
            np.random.default_rng(seed + 19), f.puck[ids], env.base._ws, 0.11
        )
    if receiving_speed_range is not None:
        low_speed, high_speed = receiving_speed_range
        if not 0 < low_speed <= high_speed <= 12:
            raise ValueError("receiving speeds must be ordered and within (0,12]")
        rng = np.random.default_rng(seed + 23)
        ids = tasks == 2
        f.puck[ids, 1] = rng.uniform(1, 1.25, per_task)
        f.puck[ids, 2] = rng.uniform(-1, 1, per_task)
        f.puck[ids, 3] = -rng.uniform(low_speed, high_speed, per_task)
        f.paddle[ids, 1] = rng.uniform(0.25, 0.45, per_task)
        travel = (f.puck[ids, 1] - 0.4) / -f.puck[ids, 3]
        f.paddle[ids, 0] = np.clip(
            f.puck[ids, 0]
            + f.puck[ids, 2] * travel
            + rng.uniform(-0.05, 0.05, per_task),
            env.decoder.low[0] + 0.01,
            env.decoder.high[0] - 0.01,
        )
    if recovery:
        rng = np.random.default_rng(seed + 29)
        ids = tasks == 2
        f.puck[ids, 0] = rng.uniform(env.decoder.low[0] + .10, env.decoder.high[0] - .10, per_task)
        f.puck[ids, 1] = rng.uniform(env.decoder.low[1] + .18, env.decoder.high[1] - .12, per_task)
        angle = rng.uniform(-.65, .65, per_task)
        speed = rng.uniform(.15, 1.2, per_task)
        f.puck[ids, 2:] = speed[:, None] * np.column_stack((np.sin(angle), np.cos(angle)))
        f.paddle[ids] = np.clip(f.puck[ids, :2] + rng.uniform([-.12, -.22], [.12, -.12], (per_task, 2)), env.decoder.low + .005, env.decoder.high - .005)
        # Same six-second horizon for every compared policy, with no resets
        # merely for establishing control or reaching the old possession clock.
        env.possession_followthrough = True
    obs = env.reset(seed=seed, fixtures=f)
    if thermal_gain is not None:
        for load_model in env.loads:
            load_model.gain[:] = thermal_gain
    if ideal_sensing:
        # Draw identical physical/camera parameters first. Constructing a
        # different sensing environment would consume a different RNG sequence
        # and confound the diagnostic with different randomized physics.
        env.base._perception = None
        env.base._cam_active = False
        env.base._max_delay = 0
        obs = env._features(env.base._make_obs_direct())
    if initial_load is not None:
        obs = set_initial_load(env, initial_load)
    history = PhysicalHistory(net.history)
    obs = history.reset(obs)
    from airhockey.shot_flight import PARAMETERS, open_goal_outcomes

    defense_ids = np.flatnonzero(tasks == 3)
    launch = f.puck[defense_ids].copy()
    launch[:, 1] = env.cfg.height - launch[:, 1]
    launch[:, 3] *= -1
    verified = open_goal_outcomes(
        launch, {k: getattr(env.engine, k)[defense_ids] for k in PARAMETERS}, env.cfg
    )
    valid = np.ones(len(tasks), bool)
    valid[defense_ids] = verified
    # Source fixture task 3 means isolated incoming defense, not a full game.
    env.kind[:] = np.where(tasks == 3, 2, tasks)
    if recovery:
        # End at the first loss of possession: a later reception after the
        # opponent returns the escaped puck is not a successful recovery.
        env.kind[tasks == 2] = 1
        env.recovery_drill[tasks == 2] = True
    done = np.zeros(env.n_envs, bool)
    rows = [None] * env.n_envs
    first_shot_aimed = np.zeros(env.n_envs, bool)
    first_shot_speed = np.zeros(env.n_envs)
    first_contact_time = np.full(env.n_envs, np.nan)
    first_contact_from_ahead = np.zeros(env.n_envs, bool)
    first_contact_slowed = np.zeros(env.n_envs, bool)
    first_contact_outgoing = np.zeros(env.n_envs, bool)
    trial_request = env.base._shot_type.copy()
    first_shot_requested = np.zeros(env.n_envs, bool)
    controlled_requested_fast = np.zeros(env.n_envs, bool)
    original_contact = env.engine.contact_callback

    def contact(event):
        ids = event["indices"]
        count_before = env.shot_count[0, ids].copy()
        aimed_before = env.aimed_count[0, ids].copy()
        was_controlled = env.captured[0, ids].copy()
        first = env.shot_count[0, ids] == 0
        original_contact(event)
        if event["body"] == "agent":
            requested_fast = (
                was_controlled & (env.shot_count[0, ids] > count_before)
                & (env.aimed_count[0, ids] > aimed_before)
                & (env.last_shot_request[0, ids] != 0)
                & (env.last_shot_route[0, ids] == env.last_shot_request[0, ids])
                & (np.linalg.norm(event["outgoing_before_speed_cap"], axis=1) >= 6)
            )
            controlled_requested_fast[ids[requested_fast]] = True
            new_contact = np.isnan(first_contact_time[ids])
            first_contact_time[ids[new_contact]] = env.engine.time[ids[new_contact]]
            # Normal points from paddle to puck. A negative dot product with
            # its incoming velocity means the paddle met the leading side,
            # instead of pushing an already departing puck from behind.
            ahead = (event['normal'] * event['incoming']).sum(axis=1) < -.05
            first_contact_from_ahead[ids[new_contact]] = ahead[new_contact]
            first_contact_outgoing[ids[new_contact]] = event['incoming'][new_contact, 1] > .05
            first_contact_slowed[ids[new_contact]] = (
                np.linalg.norm(event['outgoing_before_speed_cap'][new_contact], axis=1) < .65)
            first &= env.shot_count[0, ids] > 0
            first_shot_aimed[ids[first]] = env.aimed_count[0, ids[first]] > 0
            first_shot_speed[ids[first]] = env.shot_speed_sum[0, ids[first]]
            first_shot_requested[ids[first]] = (
                (env.last_shot_route[0, ids[first]] == env.last_shot_request[0, ids[first]])
                & (env.last_shot_request[0, ids[first]] != 0) & first_shot_aimed[ids[first]]
            )

    env.engine.contact_callback = contact
    peak_accel = np.zeros(env.n_envs)
    for _ in range(301 if recovery else 201):
        obs, _, term, trunc, info = env.step(net.act(history.for_policy(net), stochastic=stochastic))
        obs = history.append(obs)
        choose = (trial_request == 0) & (env.base._shot_type != 0) & ~done
        trial_request[choose] = env.base._shot_type[choose]
        peak_accel[~done] = np.maximum(peak_accel[~done], info["peak_accel"][~done])
        finish = (term | trunc) & ~done
        for i in np.flatnonzero(finish):
            rows[i] = {
                key: float(info[key][i])
                for key in [
                    "contacts",
                    "captures",
                    "conversions",
                    "shots",
                    "aimed",
                    "shot_speed_sum",
                    "passive_returns",
                    "load_peak",
                    "overload_seconds",
                ]
            }
            rows[i].update(
                goals=int(env.engine.score_agent[i]),
                conceded=int(env.engine.score_opponent[i]),
                peak_accel=float(peak_accel[i]),
                first_shot_aimed=bool(first_shot_aimed[i]),
                first_shot_speed=float(first_shot_speed[i]),
                first_contact_time=float(first_contact_time[i])
                if np.isfinite(first_contact_time[i])
                else None,
                requested_type=int(trial_request[i]),
                first_shot_requested=bool(first_shot_requested[i]),
                controlled_or_aimed_fast=bool(env.capture_count[0, i] or env.fast_aimed_count[0, i]),
                controlled_requested_fast=bool(controlled_requested_fast[i]),
                first_contact_from_ahead=bool(first_contact_from_ahead[i]),
                first_contact_slowed=bool(first_contact_slowed[i]),
                first_contact_outgoing=bool(first_contact_outgoing[i]),
                shot_route_counts=env.shot_route_counts[0, i].tolist(),
                aimed_route_counts=env.aimed_route_counts[0, i].tolist(),
            )
        done |= finish
        if done.all():
            break
    assert done.all()
    result = {}
    for task, name in enumerate(["stationary", "moving", "receiving", "defense"]):
        cohort = [rows[i] for i in np.flatnonzero((tasks == task) & valid)]
        result[name] = dict(
            trials=len(cohort),
            contacted=sum(r["contacts"] > 0 for r in cohort),
            goals=sum(r["goals"] > 0 for r in cohort),
            conceded=sum(r["conceded"] > 0 for r in cohort),
            controlled=sum(r["captures"] > 0 for r in cohort),
            control_to_shot=sum(r["conversions"] > 0 for r in cohort),
            control_to_requested_fast=sum(r["controlled_requested_fast"] for r in cohort),
            first_contacts_from_ahead=sum(r["first_contact_from_ahead"] for r in cohort),
            first_contacts_slowed=sum(r["first_contact_slowed"] for r in cohort),
            first_outgoing_contacts_from_ahead=sum(r["first_contact_from_ahead"] and r["first_contact_outgoing"] for r in cohort),
            saved=sum(r["contacts"] > 0 and r["conceded"] == 0 for r in cohort),
            on_target_shots=sum(r["aimed"] for r in cohort),
            shots=sum(r["shots"] for r in cohort),
            first_shots=sum(r["shots"] > 0 for r in cohort),
            first_shots_on_target=sum(r["first_shot_aimed"] for r in cohort),
            first_shots_on_target_at_least_6m_s=sum(r["first_shot_aimed"] and r["first_shot_speed"] >= 6 for r in cohort),
            first_requested_shots_at_least_6m_s=sum(r["first_shot_requested"] and r["first_shot_speed"] >= 6 for r in cohort),
            first_shots_on_requested_route=sum(r["first_shot_requested"] for r in cohort),
            controlled_or_aimed_fast=sum(r["controlled_or_aimed_fast"] for r in cohort),
            shot_route_counts=np.sum([r["shot_route_counts"] for r in cohort], axis=0).tolist(),
            aimed_route_counts=np.sum([r["aimed_route_counts"] for r in cohort], axis=0).tolist(),
            by_requested_type={
                label: dict(
                    trials=sum(r["requested_type"] == code for r in cohort),
                    first_shots_on_requested_route=sum(r["requested_type"] == code and r["first_shot_requested"] for r in cohort),
                ) for code, label in enumerate(("none", "left", "right", "straight"))
            },
            mean_first_shot_speed=sum(r["first_shot_speed"] for r in cohort)
            / max(1, sum(r["shots"] > 0 for r in cohort)),
            mean_shot_speed=sum(r["shot_speed_sum"] for r in cohort)
            / max(1, sum(r["shots"] for r in cohort)),
            peak_load=max(r["load_peak"] for r in cohort),
            overload_seconds=sum(r["overload_seconds"] for r in cohort),
            peak_accel=max(r["peak_accel"] for r in cohort),
        )
    result["defense"].update(
        speed_range=list(defense_speed_range),
        bank=bank_defense,
        random_paddle=random_defense,
        no_defender_goals=int(verified.sum()),
    )
    if receiving_speed_range is not None:
        result["receiving"]["speed_range"] = list(receiving_speed_range)
    if recovery:
        result["receiving"]["scenario"] = "slow_outgoing_recovery"
        result["receiving"]["recovery_metric_version"] = 2
        result["receiving"]["ends_at_first_possession_loss"] = True
        result["receiving"]["initial_speed_range"] = [.15, 1.2]
    if details_path is not None:
        details_path = Path(details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(
            json.dumps(
                [
                    dict(
                        task=int(tasks[i]),
                        valid=bool(valid[i]),
                        initial_puck=f.puck[i].tolist(),
                        initial_paddle=f.paddle[i].tolist(),
                        paddle_restitution=float(env.engine.paddle_restitution[i]),
                        camera_delay_s=float(env.base._cam_lag[i] * env.base._cam_dt)
                        if env.base._cam_active
                        else 0.0,
                        **row,
                    )
                    for i, row in enumerate(rows)
                ],
                indent=2,
                allow_nan=False,
            )
        )
    return result


def games(
    net,
    *,
    seconds=120,
    n=8,
    seed=20261902,
    record=None,
    checkpoint=None,
    step=0,
    stochastic=False,
    trace=None,
    initial_load=None,
    audit_motion=False,
    opponent_net=None,
    opponent_checkpoint=None,
    swap_sides=False,
    thermal_gain=None,
    shot_request="random",
    report_sensing=False,
    continuous_rallies=False,
    opponent_shot_request=None,
):
    torch.manual_seed(seed)
    conditioned = net.shot_conditioned or (opponent_net is not None and opponent_net.shot_conditioned)
    env = NeuralTrainingEnv(n, stage=3, seed=seed, games=True,
                            report_sensing=report_sensing,
                            continuous_rallies=continuous_rallies,
                            shot_conditioned=conditioned or shot_request != "random", shot_request=shot_request)
    if opponent_shot_request is not None:
        if not conditioned:
            raise ValueError("opponent shot requests require shot-conditioned neural policies")
        code = {"left": 1, "right": 2, "straight": 3}[opponent_shot_request]
        if swap_sides:
            env.base._shot_type_p_opp = env.base._shot_type_p.copy()
            env.base._shot_type_p = np.eye(4)[code]
        else:
            env.base._shot_type_p_opp = np.eye(4)[code]
    obs = env.reset(seed=seed, opponent="external")
    if thermal_gain is not None:
        for load_model in env.loads:
            load_model.gain[:] = thermal_gain
    if initial_load is not None:
        obs = set_initial_load(env, initial_load)
    motion_audit = audit_firmware_motion(env) if audit_motion else None
    rec = Recorder() if record else None
    shot_events = []
    turnover_events = []
    latest_rally_event = ["", -1.0]
    original_contact = env.engine.contact_callback

    def contact(event):
        side = int(event["body"] != "agent")
        ids = event["indices"]
        before = env.shot_count[side, ids].copy()
        controlled = env.captured[side, ids].copy()
        aimed_before = env.aimed_count[side, ids].copy()
        original_contact(event)
        for j in np.flatnonzero(env.shot_count[side, ids] > before):
            i = ids[j]
            incoming = event["incoming"][j]
            outgoing = event["outgoing_before_speed_cap"][j]
            shot_events.append(
                dict(
                    side=side,
                    game=int(i),
                    time=float(env.engine.time[i]),
                    x=float(env.engine.puck_x[i]),
                    local_y=float(
                        env.cfg.height - env.engine.puck_y[i]
                        if side
                        else env.engine.puck_y[i]
                    ),
                    controlled=bool(controlled[j]),
                    aimed=bool(env.aimed_count[side, i] > aimed_before[j]),
                    incoming_speed=float(np.linalg.norm(incoming)),
                    speed=float(min(np.linalg.norm(outgoing), env.cfg.max_puck_speed)),
                    requested_type=int(env.last_shot_request[side, i]),
                    executed_type=int(env.last_shot_route[side, i]),
                )
            )

    env.engine.contact_callback = contact
    original_relaunch = env.base._relaunch

    def relaunch(mask, *args, **kwargs):
        for i in np.flatnonzero(mask):
            e = env.engine
            side = int(e.puck_y[i] > 1)
            if i == 0:
                latest_rally_event[:] = ["Dead puck outside both paddles' reach" if continuous_rallies else "Legacy possession/stall reset", float(e.time[i])]
            turnover_events.append(
                dict(
                    side=side,
                    game=int(i),
                    time=float(e.time[i]),
                    x=float(e.puck_x[i]),
                    local_y=float(
                        env.cfg.height - e.puck_y[i] if side else e.puck_y[i]
                    ),
                    speed=float(np.hypot(e.puck_vx[i], e.puck_vy[i])),
                    reason="unreachable_dead_puck" if continuous_rallies else "legacy_referee",
                )
            )
        return original_relaunch(mask, *args, **kwargs)

    env.base._relaunch = relaunch
    peak_accel = 0
    peak_load_by_channel = np.stack([load.levels.copy() for load in env.loads])
    first_overload_seen = np.zeros((2, n), bool)
    first_overload_events = []
    readiness_samples = np.zeros(2, int)
    readiness_front = np.zeros(2, int)
    readiness_wide = np.zeros(2, int)
    readiness_cost = np.zeros(2)
    reachable_stall_seconds = np.zeros(2)
    prolonged_possession_seconds = np.zeros(2)
    history = PhysicalHistory(max(net.history, opponent_net.history if opponent_net else 1))
    start = time.monotonic()
    traces = []
    for tick in range(round(seconds / 0.02)):
        views = np.concatenate((obs, env.opponent_obs()))
        views = history.append(views)
        if opponent_net is None:
            act = net.act(history.for_policy(net), stochastic=stochastic)
        else:
            blue, red = (opponent_net, net) if swap_sides else (net, opponent_net)
            act = np.concatenate(
                (
                    blue.act(history.for_policy(blue)[:n], stochastic=stochastic and not swap_sides),
                    red.act(history.for_policy(red)[n:], stochastic=stochastic and swap_sides),
                )
            )
        env.set_opponent_action(act[n:])
        previous_score = (int(env.engine.score_agent[0]), int(env.engine.score_opponent[0]))
        obs, reward, _, _, info = env.step(act[:n])
        peak_accel = max(peak_accel, float(info["peak_accel"].max()))
        e = env.engine
        if e.score_agent[0] != previous_score[0] or e.score_opponent[0] != previous_score[1]:
            latest_rally_event[:] = ["Goal: blue" if e.score_agent[0] != previous_score[0] else "Goal: red", float(e.time[0])]
        if tick % 5 == 0:
            from airhockey.neural_possession import direct_goal_coverage_cost
            for side in (0, 1):
                prefix = "paddle_opp" if side else "paddle_agent"
                puck = np.column_stack((e.puck_x, env.cfg.height - e.puck_y if side else e.puck_y))
                pad = np.column_stack((getattr(e, prefix + "_x"), env.cfg.height - getattr(e, prefix + "_y") if side else getattr(e, prefix + "_y")))
                velocity = np.column_stack((getattr(e, prefix + "_vx"), (-1 if side else 1) * getattr(e, prefix + "_vy")))
                defending = puck[:, 1] > env.cfg.height / 2
                readiness_samples[side] += defending.sum()
                readiness_front[side] += (defending & (pad[:, 1] > .45)).sum()
                readiness_wide[side] += (defending & (abs(pad[:, 0] - env.cfg.width / 2) > .2)).sum()
                readiness_cost[side] += direct_goal_coverage_cost(puck, pad, velocity, env.decoder.bounds, env.cfg)[defending].sum()
                own = puck[:, 1] < env.cfg.height / 2
                reachable = np.linalg.norm(puck - np.clip(puck, env.decoder.low, env.decoder.high), axis=1) <= env.cfg.puck_radius + env.cfg.paddle_radius
                reachable_stall_seconds[side] += .1 * (own & reachable & (np.hypot(e.puck_vx, e.puck_vy) < .05) & (env.base._t_side > 4)).sum()
                prolonged_possession_seconds[side] += .1 * (own & (env.base._t_side > 7)).sum()
        levels = np.stack([load.levels for load in env.loads])
        peak_load_by_channel = np.maximum(peak_load_by_channel, levels)
        new_overload = (levels.max(axis=(2, 3)) >= 1) & ~first_overload_seen
        for side, game in np.argwhere(new_overload):
            dyn = env.base._opp_dyn if side else env.base._agent_dyn
            first_overload_events.append(
                dict(
                    side=int(side),
                    game=int(game),
                    time=float(e.time[game]),
                    levels=levels[side, game].tolist(),
                    observed_levels=env.loads[side].observed[game].tolist(),
                    arrival_action=act[side * n + game].tolist(),
                    position=[float(dyn["x"][game]), float(dyn["y"][game])],
                    velocity=[float(dyn["vx"][game]), float(dyn["vy"][game])],
                    commanded_accel=float(dyn["command_accel"][game]),
                )
            )
        first_overload_seen |= new_overload
        if seconds >= 600 and (tick + 1) % 3000 == 0:
            print(
                json.dumps(
                    dict(
                        evaluation_seconds=(tick + 1) * 0.02,
                        peak_load=float(env.peak_load.max()),
                        overload_seconds=float(env.overload_seconds.sum()),
                        goals=int(e.score_agent.sum() + e.score_opponent.sum()),
                        turnovers=int(env.turnovers.sum()),
                    )
                ),
                flush=True,
            )
        if trace:
            commands = []
            for side, dyn in enumerate((env.base._agent_dyn, env.base._opp_dyn)):
                commands.append(
                    [
                        dyn["command_x"][0],
                        env.cfg.height - dyn["command_y"][0]
                        if side
                        else dyn["command_y"][0],
                        dyn["command_accel"][0],
                    ]
                )
            traces.append(
                (
                    views[[0, n]].copy(),
                    act[[0, n]].copy(),
                    np.array(commands),
                    np.stack([load.levels[0] for load in env.loads]),
                    e.time[0],
                )
            )
        if rec:
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
                    shot_type_agent=int(env.base._shot_type[0]),
                    shot_type_opponent=int(env.base._shot_type_opp[0]),
                    rally_event=latest_rally_event[0] if float(e.time[0]) - latest_rally_event[1] < .8 else "",
                )
            )
    result = dict(
        games=n,
        seconds=seconds,
        seed=seed,
        score=[env.engine.score_agent.tolist(), env.engine.score_opponent.tolist()],
        contacts=env.touch_count.tolist(),
        shots=env.shot_count.tolist(),
        on_target=env.aimed_count.tolist(),
        captures=env.capture_count.tolist(),
        control_to_shot=env.convert_count.tolist(),
        mean_shot_speed=(
            env.shot_speed_sum.sum(1) / np.maximum(env.shot_count.sum(1), 1)
        ).tolist(),
        passive_returns_blue=env.passive_returns.tolist(),
        turnovers=env.turnovers.tolist(),
        shot_speed_bins=[0, 2, 4, 6, 8, 12.01],
        shot_speed_histogram=env.shot_speed_histogram.sum(1).tolist(),
        target_thirds_histogram=env.target_histogram.sum(1).tolist(),
        shot_route_names=["none/unresolved", "left bank", "right bank", "straight"],
        shot_route_counts=env.shot_route_counts.sum(1).tolist(),
        aimed_route_counts=env.aimed_route_counts.sum(1).tolist(),
        unproductive_returns=env.unproductive_returns.tolist(),
        entry_outcome_columns=[
            "completed",
            "contacted",
            "controlled",
            "on_target_shot",
            "on_target_at_least_4m_s",
            "untouched_with_100ms_workspace_opportunity",
        ],
        entry_outcomes=env.entry_outcomes.sum(1).tolist(),
        entry_speed_bins_m_s=[0, 2, 5, 8, None],
        entry_outcomes_by_speed=env.entry_outcomes_by_speed.sum(1).tolist(),
        productive_entries_by_speed=env.productive_entries_by_speed.sum(1).tolist(),
        opportunity_entries_by_speed=env.opportunity_entries_by_speed.sum(1).tolist(),
        productive_opportunities_by_speed=env.productive_opportunities_by_speed.sum(1).tolist(),
        productive_entry_definition="control OR an aimed shot of at least 4 m/s; opportunity means at least 100 ms in the geometric workspace",
        peak_load=env.peak_load.tolist(),
        peak_load_by_channel=peak_load_by_channel.tolist(),
        first_overload_events=first_overload_events,
        load_diagnostic_sampling_hz=50,
        overload_seconds=env.overload_seconds.tolist(),
        peak_accel_blue=peak_accel,
        peak_accel=env.motion_accel_peak.tolist(),
        peak_speed=env.motion_speed_peak.tolist(),
        time_above_40=env.high_accel_seconds.tolist(),
        guard_interventions=env.guard_changed.tolist(),
        guard_unresolved=env.guard_unresolved.tolist(),
        wall_seconds=time.monotonic() - start,
        stochastic=stochastic,
        initial_load=initial_load,
        thermal_gain=thermal_gain,
        continuous_rallies=continuous_rallies,
        opponent_shot_request=opponent_shot_request,
        defensive_position=dict(
            sampling_hz=10,
            opponent_possession_samples=readiness_samples.tolist(),
            fraction_above_y_045=(readiness_front / np.maximum(readiness_samples, 1)).tolist(),
            fraction_off_center_over_020=(readiness_wide / np.maximum(readiness_samples, 1)).tolist(),
            mean_direct_coverage_shortfall_m=(readiness_cost / np.maximum(readiness_samples, 1)).tolist(),
        ),
        reachable_stall_player_seconds=reachable_stall_seconds.tolist(),
        possession_over_7s_player_seconds=prolonged_possession_seconds.tolist(),
        final_physical_state=dict(
            puck=np.column_stack((env.engine.puck_x, env.engine.puck_y, env.engine.puck_vx, env.engine.puck_vy)).tolist(),
            blue_paddle=np.column_stack((env.engine.paddle_agent_x, env.engine.paddle_agent_y)).tolist(),
            red_paddle=np.column_stack((env.engine.paddle_opp_x, env.engine.paddle_opp_y)).tolist(),
        ),
        final_policy_observation=views.tolist(),
        final_arrival_action=act.tolist(),
        match_type="cross-play" if opponent_net is not None else "self-play",
        opponent_checkpoint=str(opponent_checkpoint) if opponent_checkpoint else None,
        candidate_side=int(swap_sides),
        shot_events=shot_events,
        turnover_events=turnover_events,
    )
    result["shot_breakdown"] = {}
    for controlled, label in [(True, "after_control"), (False, "without_control")]:
        result["shot_breakdown"][label] = []
        for side in (0, 1):
            events = [
                event
                for event in shot_events
                if event["side"] == side and event["controlled"] == controlled
            ]
            result["shot_breakdown"][label].append(
                dict(
                    shots=len(events),
                    on_target=sum(event["aimed"] for event in events),
                    mean_speed=sum(event["speed"] for event in events)
                    / max(1, len(events)),
                )
            )
    if rec:
        record = Path(record)
        record.parent.mkdir(parents=True, exist_ok=True)
        tmp = record.with_suffix(".tmp")
        rec.save(
            tmp,
            metadata=dict(
                algo="Single neural network",
                opponent=Path(opponent_checkpoint).parent.name
                if opponent_checkpoint
                else "self",
                match_type=result["match_type"],
                policy_sides=(
                    f"candidate {'red' if swap_sides else 'blue'}; "
                    f"reference {'blue' if swap_sides else 'red'}: {opponent_checkpoint}"
                )
                if opponent_net is not None
                else "same neural checkpoint on both sides; no tactical controllers",
                checkpoint=str(checkpoint),
                checkpoint_sha256=(
                    hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                    if checkpoint is not None and Path(checkpoint).is_file()
                    else None
                ),
                run_name=Path(checkpoint).parent.name,
                step=step,
                fps=50,
                simulation_only=True,
                policy_sampling="learned Gaussian" if stochastic else "mean",
                physical_history_frames=net.history,
                shot_conditioned=net.shot_conditioned,
                shot_request=shot_request,
                report_sensing=report_sensing,
                continuous_rallies=continuous_rallies,
                opponent_shot_request=opponent_shot_request,
                shot_type_names=["none", "left bank", "right bank", "straight"],
                accel_cap_m_s2=60,
                thermal_gain=thermal_gain,
                seconds=seconds,
            ),
        )
        tmp.replace(record)
    if motion_audit is not None:
        result["firmware_motion_audit"] = {
            k: v.tolist() for k, v in motion_audit.items()
        }
    if trace:
        trace = Path(trace)
        trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace,
            pre_action_observation=np.stack([t[0] for t in traces]),
            arrival_action=np.stack([t[1] for t in traces]),
            post_step_command_local=np.stack([t[2] for t in traces]),
            post_step_motor_load=np.stack([t[3] for t in traces]),
            post_step_time=np.array([t[4] for t in traces]),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", type=Path)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--per-task", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261901)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--skills-only", action="store_true")
    mode.add_argument("--games-only", action="store_true")
    p.add_argument("--opponent-checkpoint", type=Path)
    p.add_argument("--opponent-shot-request", choices=["left", "right", "straight"], help="Fix the opposing policy's request input for a learned-opponent pressure test")
    p.add_argument("--swap-sides", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--defense-min", type=float, default=2)
    p.add_argument("--defense-max", type=float, default=8)
    p.add_argument("--bank-defense", action="store_true")
    p.add_argument("--wide-defense", action="store_true", help="Bank attacks span the scoring mouth; changes fixture cohort")
    p.add_argument("--report-sensing", action=argparse.BooleanOptionalAction, default=None,
                   help="Use the deployed report estimator (default: checkpoint training metadata)")
    p.add_argument("--continuous-rallies", action=argparse.BooleanOptionalAction, default=None,
                   help="Preserve reachable possessions (default: checkpoint training metadata)")
    p.add_argument("--recovery", action="store_true", help="Replace receiving fixtures with slow outgoing puck recovery")
    p.add_argument("--random-defense", action="store_true")
    p.add_argument("--random-practice-opponent", action="store_true")
    p.add_argument("--initial-load", type=float)
    p.add_argument("--thermal-gain", type=float, help="Fix the load-model gain for stress tests")
    p.add_argument("--shot-request", choices=["random", "left", "right", "straight"], default="random")
    p.add_argument("--request-suite", action="store_true", help="Evaluate identical skill fixtures under each of the three shot requests")
    p.add_argument("--audit-motion", action="store_true")
    p.add_argument("--skill-details", type=Path)
    p.add_argument("--receiving-min", type=float, default=6)
    p.add_argument("--receiving-max", type=float)
    p.add_argument(
        "--ideal-sensing",
        action="store_true",
        help="Diagnostic skill trials only; game sensing remains realistic",
    )
    args = p.parse_args()
    if args.thermal_gain is not None and (
        not np.isfinite(args.thermal_gain) or args.thermal_gain <= 0
    ):
        p.error("--thermal-gain must be finite and positive")
    if args.swap_sides and args.opponent_checkpoint is None:
        p.error("--swap-sides requires --opponent-checkpoint")
    torch.set_num_threads(2)
    net, state = load(args.checkpoint)
    if args.report_sensing is None:
        args.report_sensing = checkpoint_report_sensing(args.checkpoint, state)
    if args.continuous_rallies is None:
        args.continuous_rallies = checkpoint_continuous_rallies(args.checkpoint, state)
    result = dict(
        metrics_version=5,
        wide_defense=args.wide_defense,
        report_sensing=args.report_sensing,
        continuous_rallies=args.continuous_rallies,
        recovery=args.recovery,
        opponent_shot_request=args.opponent_shot_request,
        checkpoint=str(args.checkpoint),
        step=state["step"],
        physical_history_frames=net.history,
        shot_conditioned=net.shot_conditioned,
        shot_request=args.shot_request,
        seed=args.seed,
        stochastic=args.stochastic,
        random_practice_opponent=args.random_practice_opponent,
        initial_load=args.initial_load,
        thermal_gain=args.thermal_gain,
        ideal_skill_sensing=args.ideal_sensing,
        sensing_ablation_version=3,
        physics_seeded=True,
        arrival_solve="float64-positive-determinant-v2",
        evaluation_source_sha256={
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                Path(__file__).resolve(),
                *[
                    Path(__file__).resolve().parents[1] / "airhockey" / name
                    for name in (
                        "neural_player.py",
                        "neural_observation.py",
                        "report_sensing.py",
                        "deploy.py",
                        "heuristics.py",
                        "neural_training.py",
                        "neural_possession.py",
                        "arrival.py",
                        "arrival_env.py",
                        "motion_guard.py",
                        "batch_env.py",
                        "batch_physics.py",
                        "perception.py",
                        "thermal.py",
                        "policy_benchmark.py",
                        "shot_flight.py",
                        "rewards.py",
                        "recorder.py",
                    )
                ],
            ]
        },
    )
    if not args.games_only:
        skill_options = dict(
            seed=args.seed,
            per_task=args.per_task,
            stochastic=args.stochastic,
            defense_speed_range=(args.defense_min, args.defense_max),
            bank_defense=args.bank_defense,
            wide_defense=args.wide_defense,
            random_defense=args.random_defense,
            random_practice_opponent=args.random_practice_opponent,
            initial_load=args.initial_load,
            ideal_sensing=args.ideal_sensing,
            report_sensing=args.report_sensing,
            recovery=args.recovery,
            details_path=args.skill_details,
            receiving_speed_range=(args.receiving_min, args.receiving_max)
            if args.receiving_max is not None
            else None,
            thermal_gain=args.thermal_gain,
        )
        result["skills"] = skills(net, **skill_options, shot_request=args.shot_request)
        if args.request_suite:
            result["requested_skills"] = {
                name: skills(net, **{**skill_options, "details_path": None}, shot_request=name)
                for name in ("straight", "left", "right")
            }
    print(json.dumps(result), flush=True)
    if not args.skills_only:
        opponent = (
            load(args.opponent_checkpoint)[0] if args.opponent_checkpoint else None
        )
        result["selfplay"] = games(
            net,
            seconds=args.seconds,
            n=args.games,
            seed=args.seed + 1,
            record=args.record,
            checkpoint=args.checkpoint,
            step=state["step"],
            stochastic=args.stochastic,
            initial_load=args.initial_load,
            audit_motion=args.audit_motion,
            opponent_net=opponent,
            opponent_checkpoint=args.opponent_checkpoint,
            swap_sides=args.swap_sides,
            thermal_gain=args.thermal_gain,
            shot_request=args.shot_request,
            report_sensing=args.report_sensing,
            trace=args.output.with_suffix(".trace.npz") if args.record else None,
            continuous_rallies=args.continuous_rallies,
            opponent_shot_request=args.opponent_shot_request,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    summary = dict(result)
    if "selfplay" in result:
        summary["selfplay"] = {
            k: v
            for k, v in result["selfplay"].items()
            if k not in ("shot_events", "turnover_events")
        }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
