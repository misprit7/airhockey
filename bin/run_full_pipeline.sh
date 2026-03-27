#!/bin/bash
# Full training pipeline: pretrain on idle opponent, then self-play
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Phase 1: Pretrain TD-MPC2 (500k steps) ==="
python bin/train_tdmpc2.py \
    --steps 500000 \
    --model-size 5 \
    --horizon 5 \
    --run-name tdmpc2_pretrain \
    --record-freq 50000

echo ""
echo "=== Phase 2: Self-play (5M steps) ==="
python bin/train_selfplay.py \
    --resume runs/tdmpc2_pretrain/agent.pt \
    --steps 5000000 \
    --model-size 5 \
    --horizon 5 \
    --run-name selfplay_v1 \
    --record-freq 50000 \
    --opponent-update-freq 50000

echo ""
echo "=== Pipeline complete! ==="
