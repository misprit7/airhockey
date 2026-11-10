#!/usr/bin/env python3
"""Resting-puck gates: does a candidate strike a puck lying still on its half?

    # a candidate (and a second one beside it), historical sim and table-faithful sim
    python ai/bin/eval_resting_puck.py runs/<run>/agent.pt --reference runs/<other>/agent.pt
    python ai/bin/eval_resting_puck.py runs/<run>/agent.pt --reference runs/<other>/agent.pt \
        --sim-profile table-20261029
    # one part only, with replays of the stalled trials for the web UI
    python ai/bin/eval_resting_puck.py runs/<run>/agent.pt --parts stall --record ai/recordings/<name>

The table has no referee, so a puck the robot hovers beside stays there. This
is the gate before a table run, in three parts (definitions in
airhockey/resting_puck.py; simulation only):

  stall   a puck placed at rest on a grid of the own half x mallet start
          (home / beside) x shot request, 10 s each. Stalled = never struck
          above 1.5 and never left the half. Rates by zone, request and
          cell (zone x request) with 95% Wilson intervals.
          PASS: no cell of at least --min-cell-n trials above --max-cell-stall.
  corner  corner pucks by side x request, 8 cells x 48 mirror-paired trials.
          PASS: every cell touched at least --min-corner-touched times.
  games   games against the table people (table, table_plus): spells of the
          puck at rest on the own half within reach, >= 1 s and >= 4 s, as
          count and seconds per game-minute.
          PASS: at most --max-long-rest spells of 4 s or more.

The candidate is a checkpoint or an `expert...` spec. `--sim-profile` as in
the other evaluators (default none = the historical simulator). Prints the
tables and the pass/fail lines and writes JSON; `--strict` exits 1 on a FAIL.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import eval_human_proxy as proxy_benchmark
from eval_neural_player import load
from airhockey import resting_puck
from airhockey.neural_coordinates import CoordinateReference
from airhockey.neural_setup import checkpoint_environment
from airhockey.table_fidelity import profile_options

PARTS = ('stall', 'corner', 'games')
GAME_TIERS = ('table', 'table_plus')


def prepare(candidate, environment_from, device='cpu'):
    """The candidate on the benchmark body, as the other evaluators build it."""
    options = checkpoint_environment(environment_from)
    net, _ = load(candidate)
    if net.environment_options != options:
        net = CoordinateReference(net, net.environment_options, options)
    net.to(device).eval()
    return net, options


def label(spec):
    return 'expert' if str(spec).startswith('expert') else Path(spec).parent.name


def record_stalls(raw, prefix, count, metadata):
    """Replays of stalled trials, spread over the grid (the first trials if none stalled)."""
    stalled = np.flatnonzero(resting_puck.stalled(raw))
    kind = 'stall' if len(stalled) else 'trial'
    pool = stalled if len(stalled) else np.arange(len(raw['px']))
    chosen = pool[np.unique(np.linspace(0, len(pool) - 1, min(count, len(pool))).round().astype(int))]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in chosen:
        request = resting_puck.REQUEST_NAMES[raw['req'][i]]
        start = 'beside' if raw['beside'][i] else 'home'
        dest = prefix.parent / (f"{prefix.name}_resting-{kind}-x{raw['px'][i]:.2f}-y{raw['py'][i]:.2f}"
                                f"-{request}-{start}.json")
        resting_puck.save_replay(dest, raw['frames'][:, i], dict(
            run_name=prefix.name, match_type='resting-puck stall map', opponent='human proxy (table)',
            stalled=bool(kind == 'stall'), puck=[float(raw['px'][i]), float(raw['py'][i])], request=request,
            mallet_start=start, **metadata))
        paths.append(str(dest))
    return paths


def play_games(net, options, candidate, *, tiers, games, seconds, seed, sim_profile, rest_speed,
               record=None, record_count=0):
    """Proxy-benchmark games (eval_human_proxy.play_tier), watched for rest spells."""
    result = {}
    for tier in tiers:
        watches = []

        def observe(env):
            watches.append(resting_puck.RestWatch(env))
            return watches[-1]
        game = proxy_benchmark.play_tier(net, tier, games=games, seconds=seconds, seed=seed, options=options,
                                         candidate=candidate, sim_profile=sim_profile, observe=observe)
        spells = watches[0].spells(rest_speed)
        result[tier] = dict(resting_puck.rest_report(spells, games=games, seconds=seconds),
                            goals_for=game['goals_for'], goals_against=game['goals_against'],
                            # eval_human_proxy's own stall measure: slow, reachable and > 4 s on the side.
                            reachable_stall_seconds=game['reachable_stall_seconds'])
        if record and record_count:
            frames = np.stack(watches[0].frames)
            order, paths = [], []
            for spell in sorted(spells, key=lambda s: -s['seconds']):
                if spell['game'] not in order:
                    order.append(spell['game'])
            record.parent.mkdir(parents=True, exist_ok=True)
            for i in (order or [0])[:record_count]:
                dest = record.parent / f'{record.name}_resting-games-{tier}-g{i}.json'
                resting_puck.save_replay(dest, frames[:, i], dict(
                    run_name=record.name, match_type='resting-puck games', opponent=f'human proxy ({tier})',
                    checkpoint=str(candidate), seed=seed, game=int(i), sim_profile=sim_profile,
                    rest_spells=[s for s in spells if s['game'] == i and s['seconds'] >= 1.0]))
                paths.append(str(dest))
            result[tier]['recordings'] = paths
    pooled = [s for tier in tiers for s in result[tier]['longest']]
    result['long_spells'] = sum(result[tier]['spells_ge_4s']['count'] for tier in tiers)
    result['longest_s'] = max((s['seconds'] for s in pooled), default=0.0)
    return result


def evaluate(candidate, *, parts=PARTS, seed=20261207, sim_profile=None, environment_from=proxy_benchmark.BASE,
             device='cpu', xs=resting_puck.DEFAULT_XS, ys=resting_puck.DEFAULT_YS, starts=('home', 'beside'),
             requests=(1, 2, 3), stall_seconds=10.0, load=0.45, corner_count=384, corner_seconds=12.1, games=32,
             game_seconds=60.0,
             tiers=GAME_TIERS, rest_speed=resting_puck.REST_SPEED, max_cell_stall=0.05, min_cell_n=20,
             min_corner_touched=46, max_long_rest=0, record=None, record_count=4):
    torch.set_num_threads(2)
    net, options = prepare(candidate, environment_from, device)
    column = dict(candidate=str(candidate), candidate_sha256=proxy_benchmark.candidate_hash(candidate),
                  title=label(candidate), seed=seed, sim_profile=sim_profile,
                  sim_profile_options=profile_options(sim_profile), environment_from=str(environment_from),
                  gates={})
    if 'stall' in parts:
        raw = resting_puck.stall_map(net, options, seed=seed, sim_profile=sim_profile, xs=xs, ys=ys, starts=starts,
                                     requests=requests, seconds=stall_seconds, load=load, keep_frames=bool(record))
        report = resting_puck.stall_report(raw, min_cell_n=min_cell_n)
        if record:
            report['recordings'] = record_stalls(raw, record, record_count, dict(
                checkpoint=str(candidate), seed=seed, sim_profile=sim_profile))
        column['stall'] = report
        worst = report['worst_cell']
        over = [c['cell'] for c in report['cells'] if c['gated'] and c['rate'] > max_cell_stall]
        column['gates']['stall'] = dict(
            passed=not over, threshold=f'no cell of >= {min_cell_n} trials above {max_cell_stall:.0%} stalled',
            cells_over=over, detail=f"{len(over)} of {sum(c['gated'] for c in report['cells'])} cells over"
            + (f"; worst {worst['cell']} {worst['stalled']}/{worst['n']} = {worst['rate']:.0%}" if worst else ''))
    if 'corner' in parts:
        report = resting_puck.corner_recovery(net, options, seed=seed, sim_profile=sim_profile, count=corner_count,
                                              ticks=round(corner_seconds / resting_puck.DT))
        column['corner'] = report
        short = [k for k, c in report['cells'].items() if c['touched'] < min_corner_touched]
        worst = report['cells'][report['worst_cell']]
        column['gates']['corner'] = dict(
            passed=not short, threshold=f"every cell touched >= {min_corner_touched}/{corner_count // 8}",
            cells_short=short, detail=f"{len(short)} of 8 cells short; worst {report['worst_cell']} "
                                      f"{worst['touched']}/{worst['n']}")
    if 'games' in parts:
        report = play_games(net, options, candidate, tiers=tiers, games=games, seconds=game_seconds, seed=seed,
                            sim_profile=sim_profile, rest_speed=rest_speed, record=record,
                            record_count=record_count)
        column['games'] = report
        column['gates']['games'] = dict(
            passed=report['long_spells'] <= max_long_rest, threshold=f'at most {max_long_rest} rest spells >= 4 s',
            detail=f"{report['long_spells']} spells >= 4 s in {len(tiers) * games} games; "
                   f"longest {report['longest_s']:.1f} s")
    column['passed'] = all(g['passed'] for g in column['gates'].values())
    return column


def _cell(c):
    return f"{c['stalled']:>4d}/{c['n']:<4d} {c['rate']:6.1%} [{c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}]"


def format_report(columns):
    """The printed headline: one column per candidate."""
    width = 36
    lines = []

    def table(title, rows):
        lines.append(f"\n{title:34s}" + ''.join(f'{c["title"][:width - 2]:>{width}s}' for c in columns))
        for name, values in rows:
            lines.append(f'  {name:32s}' + ''.join(f'{v:>{width}s}' for v in values))
    if all('stall' in c for c in columns):
        first = columns[0]['stall']
        rows = [('all trials', [_cell(c['stall']['overall']) for c in columns])]
        for group, prefix in (('by_start', 'start '), ('by_request', 'request '), ('by_zone', 'zone ')):
            rows += [(prefix + k, [_cell(c['stall'][group][k]) for c in columns]) for k in first[group]]
        table(f"STALL MAP ({first['trials']} trials, {first['seconds']:g} s)  stalled/n rate [95% CI]", rows)
        table('cells (zone-side / request; * = cross-bank; ~ = too few trials to gate)',
              [(('*' if cell['cross_bank'] else '~' if not cell['gated'] else ' ') + cell['cell'],
                [_cell(c['stall']['cells'][i]) for c in columns]) for i, cell in enumerate(first['cells'])])
        for c in columns:
            lines.append(f"\n{c['title']}:")
            lines.append(resting_puck.format_map(c['stall']))
            if 'stalled_trials' in c['stall']:
                s = c['stall']['stalled_trials']
                lines.append(f"  stalled trials: touched the puck {s['touched_share']:.0%}; accel cap p50 "
                             f"{s['final_accel_cap_p50']} m/s^2 at the end, {s['peak_accel_cap_p50']} at its peak; "
                             f"mallet minus puck at the end {s['mallet_minus_puck_mean']}")
    if all('corner' in c for c in columns):
        first = columns[0]['corner']
        table(f"CORNER PUCKS ({first['trials']} trials, {first['seconds']:g} s)  touched/n, first touch",
              [(k, [f"{c['corner']['cells'][k]['touched']}/{c['corner']['cells'][k]['n']}"
                    + (f"  {c['corner']['cells'][k]['first_touch_s']:.2f} s"
                       if c['corner']['cells'][k]['first_touch_s'] is not None else '') for c in columns])
               for k in first['cells']])
    if all('games' in c for c in columns):
        tiers = [k for k in columns[0]['games'] if isinstance(columns[0]['games'][k], dict)]
        rows = []
        for tier in tiers:
            rows.append((f'{tier}: goals', [f"{c['games'][tier]['goals_for']}-{c['games'][tier]['goals_against']}"
                                            for c in columns]))
            for key, name in (('spells_ge_1s', '>= 1 s'), ('spells_ge_4s', '>= 4 s')):
                rows.append((f'{tier}: rest spells {name}',
                             [f"{c['games'][tier][key]['count']} ({c['games'][tier][key]['per_game_minute']:.2f}/min, "
                              f"{c['games'][tier][key]['seconds_per_game_minute']:.2f} s/min)" for c in columns]))
            rows.append((f'{tier}: longest spell', [
                (lambda s: f"{s[0]['seconds']:.1f} s at ({s[0]['puck'][0]:.2f},{s[0]['puck'][1]:.2f}) {s[0]['zone']}"
                 if s else '-')(c['games'][tier]['longest']) for c in columns]))
        g = columns[0]['games'][tiers[0]]
        table(f"REST SPELLS IN GAMES ({g['games']} games x {len(tiers)} tiers, {g['game_minutes']:g} min each tier)",
              rows)
    lines.append('')
    for c in columns:
        for part, gate in c['gates'].items():
            lines.append(f"{'PASS' if gate['passed'] else 'FAIL'}  {c['title']}: {part}: {gate['threshold']} "
                         f"-- {gate['detail']}")
    return '\n'.join(lines)


def _floats(text):
    return tuple(float(v) for v in text.split(','))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('candidate', help="checkpoint path, or an 'expert' / 'expert:key=value,...' spec")
    p.add_argument('--reference', help='a second candidate, shown beside the first on the same trials')
    p.add_argument('--parts', default=','.join(PARTS), help='comma-separated: stall,corner,games (default all)')
    p.add_argument('--sim-profile', help="table-fidelity profile, e.g. table-20261029 "
                                         '(default: none, the historical simulator)')
    p.add_argument('--seed', type=int, default=20261207, help='shared by both candidates and all parts')
    p.add_argument('--environment-from', type=Path, default=proxy_benchmark.BASE,
                   help='run whose physical limits/workspace/thermal model the robot body uses')
    p.add_argument('--xs', type=_floats, default=resting_puck.DEFAULT_XS,
                   help='stall grid x (default 0.10..0.90 by 0.05)')
    p.add_argument('--ys', type=_floats, default=resting_puck.DEFAULT_YS,
                   help='stall grid y (default 0.10..0.85 by 0.05)')
    p.add_argument('--starts', default='home,beside', help='mallet starts of the stall map')
    p.add_argument('--requests', default='left_bank,right_bank,straight',
                   help='shot requests of the stall map, from ' + ','.join(resting_puck.REQUEST_NAMES))
    p.add_argument('--stall-seconds', type=float, default=10.0)
    p.add_argument('--load', type=float, default=0.45, help='motor load level at the start of a stall trial')
    p.add_argument('--corner-count', type=int, default=384, help='corner trials, 8 equal cells (default 384)')
    p.add_argument('--corner-seconds', type=float, default=12.1, help='length of a corner trial (default 12.1)')
    p.add_argument('--games', type=int, default=32, help='games per tier')
    p.add_argument('--seconds', type=float, default=60.0, help='length of a game')
    p.add_argument('--tiers', default=','.join(GAME_TIERS), help='human-proxy tiers of the games part')
    p.add_argument('--rest-speed', type=float, default=resting_puck.REST_SPEED,
                   help='a puck slower than this is at rest (default 0.05)')
    p.add_argument('--max-cell-stall', type=float, default=0.05, help='PASS: no stall-map cell above this rate')
    p.add_argument('--min-cell-n', type=int, default=20, help='cells with fewer trials are shown, not gated')
    p.add_argument('--min-corner-touched', type=int, default=46, help='PASS: every corner cell touched this often')
    p.add_argument('--max-long-rest', type=int, default=0, help='PASS: at most this many rest spells >= 4 s')
    p.add_argument('--strict', action='store_true', help='exit 1 if any gate fails')
    p.add_argument('--record', type=Path,
                   help='write replays as <dir>/<name>_resting-...json: stalled trials and the games with the '
                        'longest rest spells (e.g. ai/recordings/v5a)')
    p.add_argument('--record-count', type=int, default=4, help='replays per part and tier (default 4)')
    p.add_argument('--output', type=Path, help='JSON report (default logs/resting-puck/<name>-<seed>.json)')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    parts = tuple(a.parts.split(','))
    starts, tiers = tuple(a.starts.split(',')), tuple(a.tiers.split(','))
    if not parts or set(parts) - set(PARTS):
        p.error('--parts takes ' + ','.join(PARTS))
    if set(starts) - {'home', 'beside'} or set(a.requests.split(',')) - set(resting_puck.REQUEST_NAMES):
        p.error('unknown --starts or --requests')
    if set(tiers) - set(proxy_benchmark.TIER_SEED):
        p.error('unknown --tiers')
    if (a.games < 2 or min(a.seconds, a.stall_seconds, a.corner_seconds) <= 0 or a.corner_count % 8
            or not 0 <= a.load < 1):
        p.error('--games >= 2, positive seconds, --corner-count a multiple of 8, --load in [0, 1)')
    if a.min_corner_touched > a.corner_count // 8:
        p.error('--min-corner-touched is more than a corner cell holds')
    try:
        profile_options(a.sim_profile)
    except ValueError as error:
        p.error(str(error))
    requests = tuple(resting_puck.REQUEST_NAMES.index(r) for r in a.requests.split(','))
    columns = []
    for spec in filter(None, (a.candidate, a.reference)):
        record = a.record and a.record.parent / f'{a.record.name}{"-reference" if columns else ""}'
        columns.append(evaluate(
            spec, parts=parts, seed=a.seed, sim_profile=a.sim_profile, environment_from=a.environment_from,
            device=a.device, xs=a.xs, ys=a.ys, starts=starts, requests=requests, stall_seconds=a.stall_seconds,
            load=a.load, corner_count=a.corner_count, corner_seconds=a.corner_seconds, games=a.games,
            game_seconds=a.seconds, tiers=tiers,
            rest_speed=a.rest_speed, max_cell_stall=a.max_cell_stall, min_cell_n=a.min_cell_n,
            min_corner_touched=a.min_corner_touched, max_long_rest=a.max_long_rest, record=record,
            record_count=a.record_count))
    name = label(a.candidate)
    if a.sim_profile is not None:
        name += '-' + a.sim_profile.replace(':', '-').replace(',', '-')
    output = a.output or Path('logs/resting-puck') / f'{name}-{a.seed}.json'
    result = dict(benchmark='resting-puck', version=1, definitions='ai/airhockey/resting_puck.py',
                  sim_profile=a.sim_profile, seed=a.seed, parts=list(parts), columns=columns,
                  passed=all(c['passed'] for c in columns))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f'resting-puck gates, profile {a.sim_profile}, seed {a.seed}')
    print(format_report(columns))
    print(f'-> {output}', flush=True)
    if a.strict and not result['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
