#!/usr/bin/env python3
"""Train one conditioned actor from existing learned skills; simulation only.

Teachers and reflection exist only in this training program. Saved checkpoints
contain the ordinary NeuralPlayer architecture and require no teacher at play.
The resulting actor must undergo PPO and thermal/full-game qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from airhockey.neural_player import NeuralPlayer
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.neural_symmetry import reflect_physical, reflect_arrival

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RUN = None


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def load(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    net = NeuralPlayer(state['width'], history=state.get('history', 1), shot_conditioned=state.get('shot_conditioned', False))
    net.load_weights(state['model'])
    if net.history != 1:
        raise ValueError('This training-data reflection supports one physical frame')
    return net.eval(), state


def main():
    global ACTIVE_RUN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--bank-checkpoint', type=Path, required=True)
    p.add_argument('--straight-checkpoint', type=Path, required=True)
    p.add_argument('--init-checkpoint', type=Path,
                   help='Optional conditioned student warm start; teachers remain separate')
    p.add_argument('--width', type=int, help='Optional function-preserving student widening')
    p.add_argument('--shared-reception', action='store_true',
                   help='Use the strong bank policy for all receiving/defensive training labels; specialize only slow own-half setup')
    p.add_argument('--steps', type=int, default=2000000)
    p.add_argument('--minutes', type=float, default=25)
    p.add_argument('--n-envs', type=int, default=256)
    p.add_argument('--save-every', type=int, default=250000)
    p.add_argument('--seed', type=int, default=20262306)
    args = p.parse_args()
    if min(args.steps, args.n_envs, args.minutes, args.save_every) <= 0:
        p.error('steps, environments, minutes and save interval must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bank, source = load(args.bank_checkpoint)
    straight, _ = load(args.straight_checkpoint)
    initialization = load(args.init_checkpoint)[1] if args.init_checkpoint else source
    actor = NeuralPlayer(args.width or initialization['width'], shot_conditioned=True)
    actor.load_weights(initialization['model'])
    initial_step = initialization['step']
    optimizer = torch.optim.Adam(actor.actor_parameters(), lr=2e-4)
    run = ROOT / 'runs' / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN = run
    metadata = dict(algorithm='neural_skill_distillation_v1', controller=None,
                    deployment_ready=False, simulation_only=True, obs_dim=45,
                    shot_conditioned=True, physical_limits=dict(speed_m_s=12, acceleration_m_s2=60),
                    args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    teachers='Previously learned neural policies; left/right reflection is training data only',
                    thermal_caveat='Motor fit is asymmetric. Reflected examples require subsequent PPO and unmodified load qualification.',
                    source_hashes={})
    for path in [Path(__file__), *[ROOT/'ai/airhockey'/name for name in
                 ('neural_player.py','neural_training.py','neural_symmetry.py','arrival.py','arrival_env.py','thermal.py')]]:
        dest = run/'source'/path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        metadata['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(run/'run.json', metadata)
    env = NeuralTrainingEnv(args.n_envs, stage=5, seed=args.seed, shot_conditioned=True,
                            capture_first=True, fixed_practice_roles=True, random_practice_opponent=True,
                            practice_selfplay_fraction=.5, receiving_min_speed=1.5, receiving_max_speed=6,
                            warm_start_max=.92, shutdown_level=.94, terminate_overload=True,
                            thermal_gain=1.3)
    obs = env.reset(seed=args.seed)
    # DAgger: query learned teachers on the actual student's visited states.
    # No target trajectory, interception solver, or scripted striking action.
    def targets(x):
        physical = x[:, :42]
        out = bank.act(physical if not bank.shot_conditioned else x)
        left = x[:, 42] > .5
        direct = x[:, 44] > .5
        if args.shared_reception:
            # Dataset labeling only. Keep the strong, load-aware policy's
            # reception/defense examples independent of the shot request.
            # Specialize setup/strike examples once the observed own-half puck
            # is slow, matching the existing prepared-shot reward criterion.
            setup = (physical[:, 1] < 1) & (np.linalg.norm(physical[:,2:4]*6,axis=1) <= .8)
            left &= setup
            direct &= setup
        if left.any():
            mirrored = reflect_physical(physical[left])
            if bank.shot_conditioned:
                mirrored = np.column_stack((mirrored, np.tile([0,1,0], (left.sum(),1))))
            out[left] = reflect_arrival(bank.act(mirrored))
        if direct.any():
            direct_obs = physical[direct] if not straight.shot_conditioned else x[direct]
            out[direct] = straight.act(direct_obs)
        return out

    capacity = 262144
    replay_x = np.empty((capacity,45), np.float32)
    replay_y = np.empty((capacity,6), np.float32)
    filled = cursor = transitions = 0
    start = time.monotonic()
    next_save = args.save_every
    previous_loss = 0.

    def save(name):
        # PPO gets fresh moments; supervised optimizer state is incompatible.
        ppo_optimizer = torch.optim.Adam([
            {'params':actor.actor_parameters(), 'lr':5e-5},
            {'params':actor.value_parameters(), 'lr':.001},
        ], eps=1e-5)
        state = dict(model=actor.state_dict(), optimizer=ppo_optimizer.state_dict(),
                     step=initial_step+transitions, width=actor.width, history=1,
                     shot_conditioned=True, stage=5, args=metadata['args'],
                     algorithm=metadata['algorithm'], distillation_transitions=transitions)
        temp = run/(name+'.tmp')
        torch.save(state, temp)
        temp.replace(run/name)

    save('agent_initial.pt')
    while transitions < args.steps and time.monotonic()-start < args.minutes*60 and not (run/'STOP').exists():
        for _ in range(32):
            rival = env.opponent_obs()
            all_obs = np.concatenate((obs,rival))
            target = targets(all_obs)
            ids = (cursor + np.arange(len(all_obs))) % capacity
            replay_x[ids], replay_y[ids] = all_obs, target
            cursor = (cursor + len(all_obs)) % capacity
            filled = min(capacity, filled + len(all_obs))
            sampled = actor.act(all_obs)
            # Fade teacher roll-ins; most later trajectories are the student's.
            teacher_fraction = max(.05, .8*(1-transitions/max(args.steps*.6,1)))
            use_teacher = rng.random(len(all_obs)) < teacher_fraction
            sampled[use_teacher] = target[use_teacher]
            env.set_opponent_action(sampled[args.n_envs:])
            obs, _, term, trunc, _ = env.step(sampled[:args.n_envs])
            done = term | trunc
            if done.any():
                obs = env.reset(mask=done)
            transitions += args.n_envs
        losses = []
        for _ in range(64):
            ids = rng.integers(filled, size=min(2048,filled))
            x = torch.from_numpy(replay_x[ids])
            y = torch.from_numpy(replay_y[ids])
            predicted = actor.actor(actor.trunk(x)).tanh()
            loss = ((predicted-y).square()*x.new_tensor([4,4,1,1,1,1])).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor_parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
        previous_loss = float(np.mean(losses))
        status = dict(step=initial_step+transitions, distillation_transitions=transitions,
                      elapsed_s=time.monotonic()-start, imitation_loss=previous_loss, running=True)
        write_json(run/'status.json', status)
        if transitions >= next_save:
            save(f"agent_step_{status['step']:09d}.pt")
            next_save = transitions + args.save_every
            print(json.dumps(status),flush=True)
    save(f"agent_step_{initial_step+transitions:09d}.pt")
    save('agent.pt')
    status['running'] = False
    write_json(run/'status.json',status)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if ACTIVE_RUN is not None:
            write_json(ACTIVE_RUN/'status.json', dict(running=False,failed=True,error=f'{type(error).__name__}: {error}'))
        raise
