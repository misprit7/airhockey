#!/usr/bin/env python3
"""Physical-playability gates for a candidate, in the sim and from table logs.

    # a candidate (and a second one beside it) against the table person, table-faithful sim
    python ai/bin/eval_playability.py runs/<run>/agent.pt --reference runs/<other>/agent.pt \
        --tier table --sim-profile table-20261029
    # the same gates from real sessions (read-only; nothing here touches hardware)
    python ai/bin/eval_playability.py --from-table-log 20261029-182715 20261029-183722
    # all four side by side: how each gate transfers
    python ai/bin/eval_playability.py runs/<run>/agent.pt --reference runs/<other>/agent.pt \
        --tier table --sim-profile table-20261029 \
        --from-table-log 20261029-182715 --reference-table-log 20261019-213539

The candidate (a checkpoint or an `expert...` spec) plays the robot's body
against a human-proxy tier exactly as in eval_human_proxy.py (same games,
seeds, heat and rules; `--sim-profile` as there, default none = the historical
simulator), and every tick of every game is scored by airhockey/playability.py.
`--from-table-log <stamp>...` scores logs/run_policy/<stamp>.ticks.csv and
.replay.jsonl with the SAME function, so a sim column and a table column are
like for like; that module's docstring defines each gate and lists what still
differs between the two sources. Prints the table and writes JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_human_proxy as proxy_benchmark
from eval_neural_player import load
from airhockey import playability
from airhockey.human_proxy import TIERS
from airhockey.neural_coordinates import CoordinateReference
from airhockey.neural_setup import checkpoint_environment
from airhockey.table_fidelity import profile_options


def simulate(candidate, *, tier, games, seconds, seed, environment_from, sim_profile, device='cpu',
             thermal_gain=proxy_benchmark.THERMAL_GAIN, initial_load=proxy_benchmark.INITIAL_LOAD):
    """Play the proxy benchmark's games for one candidate and score every tick."""
    torch.set_num_threads(4)
    options = checkpoint_environment(environment_from)
    net, _ = load(candidate)
    if net.environment_options != options:
        net = CoordinateReference(net, net.environment_options, options)
    net.to(device).eval()
    recorders = []

    def observe(env):
        recorders.append(playability.SimRecorder(env))
        return recorders[-1]
    game = proxy_benchmark.play_tier(net, tier, games=games, seconds=seconds, seed=seed, options=options,
                                     candidate=candidate, sim_profile=sim_profile, observe=observe)
    measures = [playability.measure(trace, thermal_model=options.get('thermal_path'), thermal_gain=thermal_gain,
                                    initial_load=initial_load) for trace in recorders[0].traces()]
    summary = dict(goals_for=game['goals_for'], goals_against=game['goals_against'],
                   goal_share=game['goal_share'], goals_for_per_min=game['goals_for_per_min'],
                   goals_against_per_min=game['goals_against_per_min'],
                   # The env's own load model, advanced every 2.5 ms substep.
                   env_peak_load=game['candidate_peak_load'],
                   env_seconds_above_shutdown_level=game['candidate_seconds_above_shutdown_level'],
                   calm_cap_fraction=game['candidate_calm_cap_fraction'],
                   calibration_outside=[k for k, v in game['calibration']['checks'].items() if v[2] is False])
    if 'lift' in game:
        summary['lift'] = game['lift']
    return dict(source='sim', candidate=str(candidate), candidate_sha256=proxy_benchmark.candidate_hash(candidate),
                tier=tier, games=games, seconds=seconds, seed=seed, sim_profile=sim_profile,
                sim_profile_options=profile_options(sim_profile), thermal_gain=thermal_gain,
                initial_load=initial_load, game=summary, gates=playability.gates(measures))


def from_logs(stamps, *, thermal_gain, initial_load=None, log_dir=playability.LOG_DIR):
    """The same gates from run_policy session logs: pooled, and per session."""
    measures, sessions, policies = [], {}, []
    for stamp in stamps:
        trace = playability.table_trace(stamp, log_dir)
        m = playability.measure(trace, thermal_gain=thermal_gain, initial_load=initial_load)
        measures.append(m)
        sessions[stamp] = playability.gates([m])
        policies.append(trace['policy_name'])
    return dict(source='table', stamps=list(stamps), policy=sorted({str(x) for x in policies}),
                thermal_gain=thermal_gain, initial_load='first drive telemetry' if initial_load is None else initial_load,
                gates=playability.gates(measures), sessions=sessions)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('candidate', nargs='?', help="checkpoint path, or an 'expert' / 'expert:key=value,...' spec")
    p.add_argument('--reference', help='a second candidate, shown beside the first on the same games')
    p.add_argument('--tier', choices=sorted(TIERS), default='table', help='human-proxy tier to play (default table)')
    p.add_argument('--sim-profile', help="table-fidelity profile, e.g. table-20261029 "
                                         '(default: none, the historical simulator)')
    p.add_argument('--games', type=int, default=32)
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--seed', type=int, default=20261018, help='shared by both candidates (paired panel)')
    p.add_argument('--environment-from', type=Path, default=proxy_benchmark.BASE,
                   help='run whose physical limits/workspace/thermal model the robot body uses')
    p.add_argument('--thermal-gain', type=float, default=proxy_benchmark.THERMAL_GAIN,
                   help='load-model gain, in the sim and for the load modeled from table logs (default 1.3)')
    p.add_argument('--from-table-log', nargs='+', metavar='STAMP',
                   help='score real sessions logs/run_policy/<stamp>.{ticks.csv,replay.jsonl}, pooled')
    p.add_argument('--reference-table-log', nargs='+', metavar='STAMP',
                   help='a second set of sessions (e.g. another policy), shown beside the first')
    p.add_argument('--log-dir', type=Path, default=playability.LOG_DIR)
    p.add_argument('--output', type=Path, help='JSON report (default logs/playability/<name>-<seed>.json)')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.candidate is None and not a.from_table_log:
        p.error('give a candidate, --from-table-log, or both')
    if a.reference and a.candidate is None:
        p.error('--reference needs a candidate')
    if a.reference_table_log and not a.from_table_log:
        p.error('--reference-table-log needs --from-table-log')
    if a.games < 2 or not np.isfinite(a.seconds) or a.seconds <= 0:
        p.error('--games must be at least 2 and --seconds positive')
    try:
        profile_options(a.sim_profile)
    except ValueError as error:
        p.error(str(error))

    def label(spec):
        return 'expert' if str(spec).startswith('expert') else Path(spec).parent.name
    columns = []
    for spec in filter(None, (a.candidate, a.reference)):
        column = simulate(spec, tier=a.tier, games=a.games, seconds=a.seconds, seed=a.seed,
                          environment_from=a.environment_from, sim_profile=a.sim_profile, device=a.device,
                          thermal_gain=a.thermal_gain)
        columns.append(dict(title=f'sim {label(spec)}'[:32], **column))
    for stamps in filter(None, (a.from_table_log, a.reference_table_log)):
        column = from_logs(stamps, thermal_gain=a.thermal_gain, log_dir=a.log_dir)
        policy = '+'.join(str(x).split(':')[-1] for x in column['policy'])
        columns.append(dict(title=f'table {policy}'[:32], **column))
    name = label(a.candidate) if a.candidate else 'table-' + a.from_table_log[0]
    if a.sim_profile is not None:
        name += '-' + a.sim_profile.replace(':', '-').replace(',', '-')
    output = a.output or Path('logs/playability') / f'{name}-{a.seed}.json'
    result = dict(benchmark='playability', version=1, definitions='ai/airhockey/playability.py',
                  rows=[dict(key=key, label=text.strip()) for key, text, _ in playability.ROWS], columns=columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    for column in columns:
        if column['source'] == 'sim':
            g = column['game']
            lift = g.get('lift')
            print(f"{column['title']}: vs {column['tier']} x{column['games']} games, profile {column['sim_profile']}: "
                  f"goals {g['goals_for']}-{g['goals_against']}"
                  + (f", lift events {lift['lift_events']}, conceded through lift "
                     f"{lift['goals_conceded_through_lift']}" if lift else ''))
        else:
            print(f"{column['title']}: sessions {' '.join(column['stamps'])}")
    print(playability.format_table([(c['title'], c['gates']) for c in columns]))
    print(f'-> {output}', flush=True)


if __name__ == '__main__':
    main()
