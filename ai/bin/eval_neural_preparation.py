#!/usr/bin/env python3
"""Simulation-only benchmark of preparation before unpredictable direct attacks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from airhockey.neural_player import NeuralPlayer, PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_possession import direct_goal_coverage_cost
from airhockey.skill_benchmark import Fixtures
from airhockey.shot_flight import PARAMETERS, open_goal_outcomes


def evaluate(path, count=1024, seed=20263139, initial_load=None, lateral_speed=0, speed_range=(8,12),environment_from=None):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    state = torch.load(path, map_location='cpu', weights_only=False)
    net = NeuralPlayer(state['width'], history=state.get('history', 1), shot_conditioned=state.get('shot_conditioned', False))
    net.load_weights(state['model'])
    net.eval()
    from airhockey.neural_setup import checkpoint_environment
    options=checkpoint_environment(environment_from or path,None if environment_from else state)
    if environment_from:
        from airhockey.neural_coordinates import CoordinateReference
        net=CoordinateReference(net,checkpoint_environment(path,state),options)
    env = NeuralTrainingEnv(count, stage=2, seed=seed, report_sensing=True,
        **options,
        shot_conditioned=net.shot_conditioned, possession_followthrough=True,
        defense_clear_reward=100, defense_windup_lateral_speed=lateral_speed)
    env.cfg.max_puck_speed=max(env.cfg.max_puck_speed,speed_range[1])
    rng = np.random.default_rng(seed + 31)
    puck = np.zeros((count, 4))
    puck[:, :2] = rng.uniform([.15, 1.08], [.85, 1.25], (count, 2))
    mouth = env.cfg.goal_width / 2 - env.cfg.puck_radius - .005
    velocity = np.column_stack((rng.uniform(.5-mouth, .5+mouth, count) - puck[:, 0], -puck[:, 1]))
    velocity *= rng.uniform(*speed_range, count)[:, None] / np.linalg.norm(velocity, axis=1, keepdims=True)
    paddle = rng.uniform(env.decoder.low+.005, env.decoder.high-.005, (count, 2))
    delay = rng.uniform(.3, 1.2, count)
    if lateral_speed:
        puck[:, 2] = rng.uniform(-lateral_speed, lateral_speed, count)
    obs = env.reset(seed=seed, fixtures=Fixtures(np.full(count, 2), puck, paddle, np.full(count, .5)))
    env.defense_windup[:] = True
    env._windup_release[:] = delay
    env._windup_velocity[:] = velocity
    env._windup_aim[:] = puck[:, 0] - puck[:, 1]*velocity[:, 0]/velocity[:, 1]
    if net.shot_conditioned:
        env.base._shot_type[:] = rng.integers(1, 4, count)
    if initial_load is not None:
        for model in env.loads:
            model.h[:] = initial_load**2
            model.observed[:] = initial_load
            model.gain[:] = 1.3
    obs = env._features(env.base._make_obs_direct())
    history = PhysicalHistory(net.history)
    history.reset(obs)
    launch = np.column_stack((puck[:, 0], env.cfg.height-puck[:, 1], velocity[:, 0], -velocity[:, 1]))
    valid = open_goal_outcomes(launch, {k: getattr(env.engine, k) for k in PARAMETERS}, env.cfg)
    done = np.zeros(count, bool)
    at_launch = np.zeros((count, 2))
    release_puck = puck[:, :2].copy()
    release_velocity = velocity.copy()
    risk = np.zeros(count)
    rows = [None] * count
    for _ in range(302):
        releasing = (env.elapsed >= env._windup_release) & ~done
        if releasing.any():
            pad = np.column_stack((env.engine.paddle_agent_x, env.engine.paddle_agent_y))
            vel = np.column_stack((env.engine.paddle_agent_vx, env.engine.paddle_agent_vy))
            at_launch[releasing] = pad[releasing]
            current = np.column_stack((env.engine.puck_x, env.engine.puck_y))
            release_puck[releasing] = current[releasing]
            risk[releasing] = direct_goal_coverage_cost(current, pad, vel, env.decoder.bounds, env.cfg)[releasing]
            if lateral_speed:
                ids = np.flatnonzero(releasing)
                direction = np.column_stack((env._windup_aim[ids]-current[ids, 0], -current[ids, 1]))
                release_velocity[ids] = direction * (np.linalg.norm(env._windup_velocity[ids],axis=1) / np.linalg.norm(direction,axis=1))[:,None]
                actual_launch = np.column_stack((current[ids,0], env.cfg.height-current[ids,1],
                                                release_velocity[ids,0], -release_velocity[ids,1]))
                valid[ids] = open_goal_outcomes(actual_launch, {k:getattr(env.engine,k)[ids] for k in PARAMETERS}, env.cfg)
        obs, _, terminal, truncated, info = env.step(net.act(history.for_policy(net)))
        history.append(obs)
        finished = ~done & (terminal | truncated)
        for i in np.flatnonzero(finished):
            rows[i] = dict(valid=bool(valid[i]), saved=bool(env.touch_count[0, i] and not env.engine.score_opponent[i]),
                contacted=bool(env.touch_count[0, i]), conceded=bool(env.engine.score_opponent[i]),
                initial_puck=puck[i].tolist(), incoming_velocity=velocity[i].tolist(), delay_s=float(delay[i]),
                release_puck=release_puck[i].tolist(), release_velocity=release_velocity[i].tolist(),
                launch_paddle=at_launch[i].tolist(), coverage_shortfall_m=float(risk[i]),
                peak_load=float(info['load_peak'][i]), overload_seconds=float(info['overload_seconds'][i]))
        done |= finished
        if done.all():
            break
    assert done.all()
    kept = [row for row in rows if row['valid']]
    return dict(checkpoint=str(path), step=state.get('step'), seed=seed,
        scenario='delayed_direct', report_sensing=True, initial_load=initial_load,
        physics_puck_speed_cap=env.cfg.max_puck_speed,
        incoming_speed_range=list(speed_range),simulation_environment=options,
        lateral_speed=lateral_speed,
        trials=len(kept), saved=sum(x['saved'] for x in kept), contacted=sum(x['contacted'] for x in kept),
        conceded=sum(x['conceded'] for x in kept),
        mean_launch_paddle=np.mean([x['launch_paddle'] for x in kept], axis=0).tolist(),
        forward_at_release=sum(x['launch_paddle'][1]>.60 for x in kept),
        mean_coverage_shortfall_m=float(np.mean([x['coverage_shortfall_m'] for x in kept])),
        peak_load=max(x['peak_load'] for x in kept), overload_seconds=sum(x['overload_seconds'] for x in kept), details=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--trials', default=1024, type=int)
    parser.add_argument('--seed', default=20263139, type=int)
    parser.add_argument('--initial-load', type=float)
    parser.add_argument('--lateral-speed', type=float, default=0)
    parser.add_argument('--speed-min',type=float,default=8)
    parser.add_argument('--speed-max',type=float,default=12)
    parser.add_argument('--environment-from',type=Path)
    args = parser.parse_args()
    if args.trials < 1 or (args.initial_load is not None and not 0 <= args.initial_load <= 1):
        parser.error('positive trial count and initial load in [0,1] required')
    if not np.isfinite(args.lateral_speed) or args.lateral_speed < 0:
        parser.error('lateral speed must be finite and nonnegative')
    if not 0<args.speed_min<=args.speed_max<=20:parser.error('invalid incoming speed range')
    result = evaluate(args.checkpoint, args.trials, args.seed, args.initial_load, args.lateral_speed,
        (args.speed_min,args.speed_max),args.environment_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
