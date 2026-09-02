#!/bin/bash
# Five-stage curriculum, then self-play. Each stage resumes the previous
# checkpoint; stage budgets, opponents and reward weights come from
# rewards.CURRICULUM (one table, used by both trainers).
#
#   bash ai/bin/run_full_pipeline.sh            # default budgets
#   PREFIX=curr2 bash ai/bin/run_full_pipeline.sh
#
# train_tdmpc2.py can core-dump at interpreter teardown (background eval
# thread vs torch shutdown) AFTER saving, so each stage is judged by its
# checkpoint, not its exit code.
cd "$(dirname "$0")/../.."
export PYTHONPATH=ai PYTHONUNBUFFERED=1
PREFIX=${PREFIX:-curriculum}

budget() { python3 -c "from airhockey.rewards import CURRICULUM as C; print(C['$1']['steps'])"; }

prev=""
for stage in proximity contact scoring goalie; do
    run="${PREFIX}_${stage}"
    steps=$(budget $stage)
    echo "=== Stage $stage: $steps steps ($run) ==="
    resume=""
    [ -n "$prev" ] && resume="--resume runs/$prev/agent.pt"
    python3 ai/bin/train_tdmpc2.py --curriculum-stage $stage --steps $steps \
        --model-size 5 --horizon 5 --run-name $run --record-freq 50000 \
        --updates-per-step 32 $resume || true
    if [ ! -f runs/$run/agent.pt ]; then
        echo "stage $stage produced no checkpoint — aborting"; exit 1
    fi
    prev=$run
done

steps=$(budget selfplay)
echo "=== Stage selfplay: $steps steps (${PREFIX}_selfplay) ==="
python3 ai/bin/train_selfplay.py --resume runs/$prev/agent.pt --steps $steps \
    --n-envs 32 --model-size 5 --horizon 5 --run-name ${PREFIX}_selfplay \
    --record-freq 50000 --opponent-update-freq 50000
echo "=== Pipeline complete ==="
