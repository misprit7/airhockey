#!/usr/bin/env bash
# Simulation only. Intended to run as a background systemd user service.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=ai PYTHONUNBUFFERED=1

RUN=3.12-accel60-accuracy-selfplay
PARENT=3.11-shot-ramp-selfplay
mkdir -p logs

python3 ai/bin/train_selfplay.py \
    --resume "runs/$PARENT/agent.pt" \
    --run-name "$RUN" --steps 5000000 --n-envs 32 \
    --horizon 8 --max-accel 60 \
    --reward-config ai/recipes/accel60-accuracy.json \
    --record-freq 250000 --opponent-update-freq 50000 \
    --iterations 6 --samples 256 \
    > "logs/$RUN.log" 2>&1

# The same fixtures and cap as the baseline, after training succeeds.
python3 ai/bin/eval_policy.py "$RUN" \
    --opponents sniper,weak_goalie,goalie --games 8 --seconds 30 \
    --iterations 6 --max-accel 60 \
    --json "logs/$RUN-eval.json" > "logs/$RUN-eval.log" 2>&1
