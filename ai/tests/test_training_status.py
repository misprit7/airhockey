import json
import os

from airhockey.training_status import training_status


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(root, name='_neural-player-requests-possession106', running=False):
    run = root / 'runs' / name
    write(run / 'run.json', {'args': {'steps': 128, 'n_envs': 4, 'rollout': 8, 'save_every': 32}})
    write(run / 'status.json', {'running': running, 'step': 1128, 'update': 4,
                               'elapsed_s': 10, 'transitions_per_s': 12.8})
    (run / 'agent_step_1128.pt').touch()
    return run


def test_resumed_progress_counts_only_this_runs_transitions(tmp_path):
    fixture(tmp_path)
    result = training_status(tmp_path, process_map={})
    run = result['runs'][0]
    assert run['completed_steps'] == 128
    assert run['cumulative_step'] == 1128
    assert run['progress'] == 1
    assert run['state'] == 'Finished'
    assert run['checkpoints'][0]['additional_steps'] == 128
    assert run['checkpoints'][0]['evaluation'] == 'Incomplete / no evaluator'
    assert run['latest_replay'] is None


def test_expanded_training_watcher_is_attached_to_its_run(tmp_path):
    run=fixture(tmp_path)
    process={44:['python','ai/bin/watch_expanded_training.py','--run',str(run),'--output-dir','logs/evaluation']}
    assert training_status(tmp_path,process_map=process)['runs'][0]['evaluator_active']
    process[44][3]='runs/unrelated'
    assert not training_status(tmp_path,process_map=process)['runs'][0]['evaluator_active']


def test_old_running_flag_and_reused_pid_do_not_claim_live_training(tmp_path):
    run = fixture(tmp_path, running=True)
    os.utime(run / 'status.json', (10, 10))
    stale = training_status(tmp_path, process_map={}, now=300)
    assert stale['active_training'] == 0
    assert stale['runs'][0]['state'] == 'Interrupted / process absent'
    write(run / 'status.json', {'running': True, 'pid': 123})
    reused = training_status(tmp_path, process_map={123: ['python3', 'train_neural_player.py', '--run-name', 'another-run']})
    assert reused['active_training'] == 0


def test_running_and_evaluation_processes_are_reported_separately(tmp_path):
    run = fixture(tmp_path, running=True)
    processes = {12: ['python3', 'ai/bin/train_neural_player.py', '--run-name', run.name],
                 13: ['python3', 'logs/neural-player/requests/possession106/evaluate_checkpoints.py']}
    result = training_status(tmp_path, process_map=processes)
    assert result['active_training'] == result['active_evaluations'] == 1
    assert result['runs'][0]['state'] == 'Training'
    assert result['runs'][0]['checkpoints'][0]['evaluation'] == 'Evaluating'


def test_evaluations_link_exact_replays_and_keep_failed_screen_visible(tmp_path):
    run = fixture(tmp_path)
    log = tmp_path / 'logs/neural-player/requests/possession106'
    write(log / 'development-selection.json', {'rows': [{
        'checkpoint': str(run / 'agent_step_1128.pt'), 'eligible': False,
        'edges': {'restored_interior': 5, 'controlled_after_restore': 0},
        'shots': {'stationary': 393}, 'short_peak': .96,
    }]})
    write(log / 'agent_step_1128/edges.json', {'request_trials': 198})
    recordings = tmp_path / 'ai/recordings'
    for name in [f'{run.name}_agent_step_1128.json', 'neural-edge106-agent_step_1128-original-left.json',
                 'neural-edge103-agent_step_1128-original-left.json']:
        write(recordings / name, {})
    (run / 'agent_step_failed.pt').touch()
    result = training_status(tmp_path, process_map={})['runs'][0]
    checkpoint = result['checkpoints'][0]
    assert result['checkpoint_count'] == result['evaluated'] == 1
    assert checkpoint['screening'] == 'Not passed'
    assert checkpoint['edge_trials'] == 198
    assert len(checkpoint['diagnostics']) == 1
    assert '106' in checkpoint['diagnostics'][0]['url']
    assert result['latest_replay'] == f'/?replay={run.name}_agent_step_1128.json'
    assert 'No replacement selected' in result['outcome']


def test_partial_status_and_smoke_runs_do_not_hide_real_run(tmp_path):
    run = fixture(tmp_path)
    fixture(tmp_path, name='_neural-smoke')
    partial = tmp_path / 'runs/partial/status.json'
    partial.parent.mkdir(parents=True)
    partial.write_text('{')
    result = training_status(tmp_path, process_map={})
    assert [r['name'] for r in result['runs']] == [run.name]


def test_explicit_progress_and_early_stop(tmp_path):
    run = fixture(tmp_path)
    write(run / 'status.json', {'running': False, 'step': 1050, 'initial_step': 1000, 'target_steps': 128})
    result = training_status(tmp_path, process_map={})['runs'][0]
    assert result['completed_steps'] == 50
    assert result['state'] == 'Stopped before step target'
    assert result['eta_s'] is None
