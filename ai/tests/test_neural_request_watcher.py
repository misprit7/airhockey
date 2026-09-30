import importlib.util
import json
from pathlib import Path
import sys


def test_final_checkpoint_created_during_evaluation_is_not_skipped(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'bin/watch_neural_requests.py'
    spec = importlib.util.spec_from_file_location('watch_neural_requests_test', path)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    run, reports = tmp_path/'run', tmp_path/'reports'
    run.mkdir()
    (run/'agent_step_1.pt').touch()
    (run/'status.json').write_text(json.dumps({'running': True}))
    evaluated = []
    targets = {}

    def evaluate(command, **kwargs):
        checkpoint = Path(command[2])
        evaluated.append(checkpoint.name)
        if len(evaluated) == 1:
            # Training finishes while the previous snapshot is being tested.
            (run/'agent_step_2.pt').touch()
            (run/'status.json').write_text(json.dumps({'running': False}))
        output = Path(command[command.index('--output')+1])
        targets[output.name] = checkpoint.name
        if '--record' in command:
            recording = Path(command[command.index('--record')+1])
            recording.parent.mkdir(parents=True, exist_ok=True)
            recording.write_text(json.dumps({'checkpoint': checkpoint.name}))
        output.write_text(json.dumps({
            'requested_skills': {},
            'skills': {'receiving': {}, 'defense': {}},
            'selfplay': {'entry_outcomes_by_speed': [], 'entry_speed_bins_m_s': [], 'peak_load': []},
        }))

    monkeypatch.setattr(watcher.subprocess, 'run', evaluate)
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['watch', str(run), '--output-dir', str(reports), '--minutes', '.1'])
    watcher.main()
    assert 'agent_step_2.pt' in evaluated
    assert targets['final-hot.json'] == 'agent_step_2.pt'
    assert targets['final-heldout.json'] == 'agent_step_2.pt'
    assert targets['final-defense-bank-varied.json'] == 'agent_step_2.pt'
    assert json.loads((reports/'latest.json').read_text())['checkpoint'].endswith('agent_step_2.pt')
    recordings = tmp_path/'ai/recordings'
    assert (recordings/'run_step_000000001.json').exists()
    assert (recordings/'run_step_000000002.json').exists()
    assert json.loads((recordings/'neural-requests-wip.json').read_text())['checkpoint'] == 'agent_step_2.pt'


def test_initial_replay_is_published_when_trainer_starts_after_watcher(tmp_path, monkeypatch):
    import torch

    path = Path(__file__).resolve().parents[1] / 'bin/watch_neural_requests.py'
    spec = importlib.util.spec_from_file_location('watch_initial_test', path)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    run, reports = tmp_path/'run', tmp_path/'reports'
    run.mkdir()
    (run/'status.json').write_text(json.dumps({'running': True}))
    calls = []

    def evaluate(command, **kwargs):
        checkpoint = Path(command[2])
        calls.append(checkpoint.name)
        output = Path(command[command.index('--output')+1])
        output.write_text(json.dumps({}))
        recording = Path(command[command.index('--record')+1])
        recording.write_text(json.dumps({'checkpoint': checkpoint.name}))
        # End this mock run after the initial replay; no learned snapshot yet.
        (run/'status.json').write_text(json.dumps({'running': False}))

    def start_trainer(_):
        torch.save({'step': 7}, run/'agent_initial.pt')

    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    monkeypatch.setattr(watcher.subprocess, 'run', evaluate)
    monkeypatch.setattr(watcher.time, 'sleep', start_trainer)
    monkeypatch.setattr(sys, 'argv', ['watch', str(run), '--output-dir', str(reports), '--minutes', '.1'])
    watcher.main()
    assert calls == ['agent_initial.pt']
    assert (tmp_path/'ai/recordings/run_step_000000007.json').exists()
    assert json.loads((tmp_path/'ai/recordings/neural-requests-wip.json').read_text())['checkpoint'] == 'agent_initial.pt'
