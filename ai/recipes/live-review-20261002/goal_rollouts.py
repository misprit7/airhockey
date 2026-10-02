"""Replay six bank concessions with environment physics and recorded commands.

Purely offline: initial camera pose/causal velocity, then commanded robot motion
and measured human input only. No later robot/puck measurement steers a rollout.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0,'ai')
from airhockey.hardware_replay import load_session,simulate
out=Path('logs/analysis/neural-live-20261002')
goals=json.loads((out/'review.json').read_text())['goals']
session=load_session('20261002-114149.replay.jsonl')
results=simulate(session,[g['crossing_t'] for i,g in enumerate(goals) if i!=3],duration=.5)
(out/'goal-physics-rollouts.json').write_text(json.dumps(results,indent=2))
for r in results:print(r['start'],r['reason'],r['errors'])
