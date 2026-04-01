#!/bin/bash
# Full curriculum training pipeline
# Each stage auto-advances on reward plateau

set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
BASE_ARGS="--n-envs 64 --batch-size 4096 --updates-per-step 2 --num-samples 256 --record-freq 50000"

echo "=== Air Hockey Curriculum Training ==="
echo "Starting 4-stage curriculum..."

$VENV bin/train_tdmpc2_fast.py --curriculum $BASE_ARGS --run-name curriculum_$(date +%Y%m%d_%H%M%S) "$@"

echo "=== Curriculum complete! ==="
