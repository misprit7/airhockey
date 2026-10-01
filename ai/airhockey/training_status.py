"""Read-only training/evaluation discovery for the web UI. Never loads policies."""

import json
import re
import time
from pathlib import Path
from urllib.parse import quote


def read_json(path):
    try:
        value = json.loads(path.read_text(), parse_constant=lambda _: None)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode(errors='replace').strip('\0').split('\0')
            if any(arg.endswith('.py') for arg in argv):
                result[int(path.parent.name)] = argv
        except OSError:
            pass
    return result


def _link(path):
    return '/?replay=' + quote(path.name, safe='')


def training_status(root, *, process_map=None, now=None, limit=12):
    now = time.time() if now is None else now
    process_map = processes() if process_map is None else process_map
    recordings = root / 'ai/recordings'
    runs = []
    for status_path in (root / 'runs').glob('*/status.json'):
        run = status_path.parent
        if 'smoke' in run.name:
            continue
        status = read_json(status_path)
        if not status:
            continue
        metadata = read_json(run / 'run.json')
        args = metadata.get('args', {})
        # Historical request runs predate the explicit evaluation_dir field.
        legacy = re.match(r'_neural-player-requests-(possession\d+)', run.name)
        log_dir = root / metadata.get('evaluation_dir',
            'logs/neural-player/requests/' + (legacy[1] if legacy else run.name))
        trainer_alive = any(
            ('--run-name' in argv and argv[argv.index('--run-name')+1:][:1] == [run.name])
            or any(Path(a).name in ('fit_successes.py', 'fit.py')
                   and (root / a).parent == log_dir for a in argv)
            for pid, argv in process_map.items())
        evaluator_alive = any(
            (any(Path(a).name=='watch_expanded_training.py' for a in argv)
             and '--run' in argv and Path(argv[argv.index('--run')+1]).name==run.name)
            or any(Path(a).name.startswith(('evaluate_', 'audit_')) and a.endswith('.py')
                and (root / a).parent == log_dir for a in argv)
            for argv in process_map.values())
        modified = status_path.stat().st_mtime
        if status.get('failed'):
            state = 'Failed'
        elif trainer_alive and status.get('running', True):
            state = 'Training' if now - modified < 120 else 'Training heartbeat stale'
        elif status.get('running'):
            state = 'Interrupted / process absent'
        else:
            state = 'Finished'
        step = status.get('step')
        initial = status.get('initial_step')
        if initial is None and step is not None and status.get('update') is not None:
            batch = args.get('n_envs', 0) * args.get('rollout', 0)
            if batch:
                initial = step - status['update'] * batch
        completed = max(0, step - initial) if step is not None and initial is not None else None
        target = status.get('target_steps', args.get('steps')) if completed is not None else None
        fraction = min(1, completed / target) if target and completed is not None else None
        if state == 'Finished' and fraction is not None and fraction < 1:
            state = 'Stopped before step target'
        rate = status.get('transitions_per_s', 0)
        selection = read_json(log_dir / 'development-selection.json')
        reviewed = {Path(row['checkpoint']).stem: row for row in selection.get('rows', []) if 'checkpoint' in row}
        checkpoints = []
        for file in sorted([*run.glob('agent_step_*.pt'), *run.glob('agent_update_*.pt')], reverse=True):
            if not re.fullmatch(r'agent_(step|update)_[0-9]+', file.stem):
                continue
            number = int(file.stem.rsplit('_', 1)[1])
            row = reviewed.get(file.stem, {})
            reports = log_dir / file.stem
            ready = [name for name in ('edges', 'skills', 'selfplay', 'preparation', 'fringe', 'endurance') if (reports / (name + '.json')).exists()]
            edge_report = read_json(reports / 'edges.json')
            edge = row.get('edges', edge_report)
            replay = recordings / f'{run.name}_{file.stem}.json'
            diagnostics = sorted(recordings.glob(f'neural-edge*-{file.stem}-*.json'))
            # A matching step in a different run is not the same policy.
            if legacy:
                diagnostics = [p for p in diagnostics if p.name.startswith(f'neural-edge{legacy[1][10:]}-')]
            else:
                diagnostics = []
            evaluation = ('Complete' if row or {'edges','skills','selfplay'}.issubset(ready) else
                          'Evaluating' if evaluator_alive else 'Incomplete / no evaluator')
            screening = ('Not passed' if row.get('eligible') is False else
                         'Passed screen; further review required' if row.get('eligible') else 'Not recorded')
            checkpoints.append(dict(
                name=file.stem, step=number, additional_steps=number-initial if initial is not None and '_step_' in file.stem else None,
                evaluation=evaluation, reports_ready=ready, screening=screening,
                edges={k: edge.get(k) for k in ('restored_interior', 'controlled_after_restore', 'requested_fast_after_restore')},
                edge_trials=edge_report.get('request_trials'), shots=row.get('shots', {}), peak_load=row.get('short_peak'),
                recovery=row.get('recovery'), defense=row.get('defense'),
                fringe=row.get('fringe',read_json(reports/'fringe.json').get('summary')),
                endurance_peak=row.get('endurance_peak'),
                endurance_overload_seconds=row.get('endurance_overload_seconds'),
                sha256=row.get('sha256'), replay=_link(replay) if replay.exists() else None,
                diagnostics=[dict(label=p.stem.split(file.stem+'-',1)[-1], url=_link(p)) for p in diagnostics],
            ))
        count = sum(c['evaluation'] == 'Complete' for c in checkpoints)
        review = read_json(run / 'review.json')
        outcome = review.get('summary')
        if not outcome:
            if checkpoints and len(reviewed) == len(checkpoints) and not any(r.get('eligible') for r in reviewed.values()):
                outcome = 'No checkpoint passed the development screen. No replacement selected.'
            else:
                outcome = 'No production selection recorded for this run.'
        runs.append(dict(
            name=run.name, state=state, training_active=trainer_alive and bool(status.get('running', True)),
            evaluator_active=evaluator_alive, updated_at=modified, heartbeat_age_s=max(0, now-modified),
            elapsed_s=status.get('elapsed_s'), completed_steps=completed, target_steps=target,
            progress=fraction, cumulative_step=step, updates=status.get('update'),
            eta_s=max(0, target-completed)/rate if target and rate and trainer_alive else None,
            save_every=args.get('save_every'), evaluated=count, checkpoint_count=len(checkpoints),
            outcome=outcome, error=status.get('error'), checkpoints=checkpoints,
            latest_replay=next((c['replay'] for c in checkpoints if c['replay']), None),
            initialization=metadata.get('initialization', metadata.get('initial')),
            limits=metadata.get('physical_limits'),
            workspace=args.get('workspace'),
            edge_dwell_band=args.get('edge_dwell_band'),
        ))
    runs.sort(key=lambda r: (r['training_active'] or r['evaluator_active'], r['updated_at']), reverse=True)
    packages = []
    for file in (root / 'runs').glob('*/qualification.json'):
        meta = read_json(file.parent / 'run.json')
        if not (meta.get('deployment_ready') or meta.get('simulation_candidate')):
            continue
        review = read_json(file.parent / 'review.json')
        replay_name = meta.get('replay_file')
        replay = recordings / Path(replay_name).name if isinstance(replay_name, str) else None
        packages.append(dict(name=file.parent.name, note=review.get('summary', 'Packaged policy; see its qualification report for limitations.'),
                             simulation_candidate=bool(meta.get('simulation_candidate')),
                             deployment_ready=bool(meta.get('deployment_ready')),
                             replay=_link(replay) if replay and replay.exists() else None,
                             checkpoint=meta.get('selected_checkpoint'), sha256=meta.get('selected_checkpoint_sha256'),
                             updated_at=file.stat().st_mtime))
    packages.sort(key=lambda p: p['updated_at'], reverse=True)
    return dict(updated_at=now, active_training=sum(r['training_active'] for r in runs),
                active_evaluations=sum(r['evaluator_active'] for r in runs), runs=runs[:limit], packages=packages)
