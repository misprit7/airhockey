#!/usr/bin/env python3
"""Deterministic edge/corner recovery tests, with no physical hardware I/O."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from airhockey.neural_player import PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_setup import checkpoint_environment
from airhockey.neural_coordinates import CoordinateReference
from airhockey.skill_benchmark import Fixtures
from eval_neural_player import load


def evaluate(path, count=256, seed=20261015, environment_from=None):
    torch.set_num_threads(2)
    net, state = load(path)
    options = checkpoint_environment(environment_from) if environment_from else net.environment_options
    if environment_from:
        net = CoordinateReference(net, net.environment_options, options)
    env = NeuralTrainingEnv(count, stage=5, game_fraction=0, report_sensing=True,
        shot_conditioned=net.shot_conditioned, possession_followthrough=True,
        edge_recovery_weight=1, **options)
    rng = np.random.default_rng(seed)
    corner = np.arange(count) % 2 == 0
    right = np.arange(count) % 4 >= 2
    r = env.cfg.puck_radius
    puck = np.zeros((count, 4))
    puck[:, 0] = rng.uniform(r+.001, env.decoder.low[0]+.01, count)
    puck[right, 0] = env.cfg.width-puck[right, 0]
    puck[:, 1] = rng.uniform(env.decoder.low[1]+.15, env.decoder.high[1]-.08, count)
    puck[corner, 1] = rng.uniform(r+.001, env.decoder.low[1]+.015, corner.sum())
    paddle = rng.uniform(env.decoder.low+.015, env.decoder.high-.015, (count, 2))
    # Keep initial bodies separate without silently moving the test puck.
    delta = paddle-puck[:, :2]
    overlap = np.linalg.norm(delta, axis=1) < env.cfg.paddle_radius+r+.015
    paddle[overlap] = [.5, .4]
    obs = env.reset(seed=seed, fixtures=Fixtures(np.zeros(count, int), puck, paddle, np.full(count, .5)))
    env.edge_drill[:] = True  # 12 seconds to retrieve, control, and shoot.
    for model in env.loads:
        model.gain[:] = 1.3
        model.h[:] = .5**2
        model.observed[:] = .5
    env.base._shot_type[:] = 1+np.arange(count)%3
    obs = env._features(env.base._make_obs_direct())
    history = PhysicalHistory(net.history)
    history.reset(obs)
    done = np.zeros(count, bool)
    rows = [None]*count
    for _ in range(605):
        obs, _, terminal, truncated, info = env.step(net.act(history.for_policy(net)))
        history.append(obs)
        finished = ~done & (terminal | truncated)
        for i in np.flatnonzero(finished):
            rows[i] = dict(corner=bool(corner[i]), right=bool(right[i]),
                touched=bool(env.touch_count[0, i]), recovered=bool(env.edge_recovery_count[0, i]),
                aimed=bool(env.aimed_count[0, i]), peak_load=float(info['load_peak'][i]),
                overload_seconds=float(info['overload_seconds'][i]),
                initial_puck=puck[i].tolist(), initial_paddle=paddle[i].tolist(),
                final_puck=[float(env.engine.puck_x[i]), float(env.engine.puck_y[i])])
        done |= finished
        if done.all(): break
    if not done.all(): raise RuntimeError('Fringe benchmark did not terminate')
    summary = {}
    for name, mask in [('corner', corner), ('side', ~corner)]:
        group = [rows[i] for i in np.flatnonzero(mask)]
        summary[name] = dict(trials=len(group), **{key:sum(row[key] for row in group)
            for key in ('touched', 'recovered', 'aimed', 'overload_seconds')},
            peak_load=max(row['peak_load'] for row in group))
    return dict(checkpoint=str(path), step=state.get('step'), seed=seed,
        simulation_environment=options, initial_load=.5, thermal_gain=1.3,
        summary=summary, details=rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--trials', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--environment-from', type=Path)
    a=p.parse_args()
    if a.trials<4: p.error('at least four trials required')
    result=evaluate(a.checkpoint,a.trials,a.seed,a.environment_from)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(a.output)
    print(json.dumps(result['summary'],indent=2))
