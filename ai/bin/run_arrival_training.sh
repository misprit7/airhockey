#!/usr/bin/env bash
# Simulation only. No camera, controller, motor SDK, or deployment activation.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=ai PYTHONUNBUFFERED=1
exec python3 ai/bin/train_arrival.py \
  --run-name 4.0-arrival-rms-selfplay \
  --encoder-from runs/3.12-accel60-accuracy-selfplay/agent.pt \
  --demos logs/arrival-training/demonstrations \
  --steps 6000000 --n-envs 32 --pretrain-updates 4000 \
  --compile-update --checkpoint-every 100000 --eval-every 500000
