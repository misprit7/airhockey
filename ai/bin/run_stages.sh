#!/bin/bash
# Run curriculum stages one at a time with explicit step counts.
# Useful for manual control — inspect results between stages, adjust hyperparams, etc.
#
# Usage:
#   bash bin/run_stages.sh                  # Run all 4 stages
#   bash bin/run_stages.sh --start 3        # Resume from stage 3

set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
BASE_ARGS="--n-envs 64 --batch-size 4096 --updates-per-step 2 --num-samples 256 --record-freq 50000"
RUN_NAME="stages_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_NAME"

START_STAGE=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START_STAGE="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; RUN_DIR="runs/$RUN_NAME"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Air Hockey Staged Curriculum ==="
echo "Run: $RUN_NAME"
echo "Starting from stage $START_STAGE"
echo ""

# Stage 1: Chase + Hit — learn to chase and hit the puck
if [ "$START_STAGE" -le 1 ]; then
    echo "--- Stage 1: Chase + Hit (500k steps) ---"
    $VENV bin/train_tdmpc2_fast.py --stage 1 --steps 500000 \
        $BASE_ARGS --run-name "${RUN_NAME}_s1"
    echo ""
fi

# Stage 2: Game vs Goalie — learn to score past blocker
if [ "$START_STAGE" -le 2 ]; then
    RESUME="${RUN_DIR}_s1/agent.pt"
    echo "--- Stage 2: Game vs Goalie (1M steps) ---"
    $VENV bin/train_tdmpc2_fast.py --stage 2 --steps 1000000 \
        --resume "$RESUME" $BASE_ARGS --run-name "${RUN_NAME}_s2"
    echo ""
fi

# Stage 3: Game vs Follower — compete against reactive opponent
if [ "$START_STAGE" -le 3 ]; then
    RESUME="${RUN_DIR}_s2/agent.pt"
    echo "--- Stage 3: Game vs Follower (2M steps) ---"
    $VENV bin/train_tdmpc2_fast.py --stage 3 --steps 2000000 \
        --resume "$RESUME" $BASE_ARGS --run-name "${RUN_NAME}_s3"
    echo ""
fi

# Stage 4: Self-play — compete against past selves
if [ "$START_STAGE" -le 4 ]; then
    RESUME="${RUN_DIR}_s3/agent.pt"
    echo "--- Stage 4: Self-play (1M steps) ---"
    $VENV bin/train_tdmpc2_fast.py --stage 4 --steps 1000000 \
        --resume "$RESUME" $BASE_ARGS --run-name "${RUN_NAME}_s4"
    echo ""
fi

echo "=== All stages complete! ==="
echo "Final agent: ${RUN_DIR}_s4/agent.pt"
