#!/bin/bash
# Turn it on and it plays: the master, the camera and the policy, in one
# command, from the REPO ROOT.
#
#   bash ai/bin/play.sh                 # newest checkpoint, full caps
#   bash ai/bin/play.sh --gentle        # FIRST RUN OF ANY NEW CHECKPOINT
#   bash ai/bin/play.sh --dry           # camera + policy, commands nothing
#   POLICY=tdmpc2:curriculum_goalie bash ai/bin/play.sh --plan 1
#   MASTER_ARGS="--tension 1.5" bash ai/bin/play.sh --gentle
#
# Everything after the script's own flags goes to ai/bin/run_policy.py;
# MASTER_ARGS goes to cdpr_master (its default pretension is 0 = slack).
#
# What happens, in order: build anything missing; start sw/build/cdpr_master
# (which must run ALONE -- it opens the SC-Hub USB port; stop `activate` and
# the web UI's hardware mode first); wait for its TCP port; run the policy
# with --live, which measures the paddle with the camera, ENABLEs the drives
# and starts the 100 Hz loop. Ctrl-C brakes the paddle in place, then the
# master is stopped, which de-energizes the drives.
#
# Logs: logs/run_policy/<stamp>.log (everything printed) and
# <stamp>.ticks.csv (what the policy saw and did, per tick); the master's
# own logs/cdpr_master.log is copied to a stamped name on exit.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=ai PYTHONUNBUFFERED=1

POLICY=${POLICY:-tdmpc2:latest}
MASTER_ARGS=${MASTER_ARGS:-}
LIVE=1
for a in "$@"; do
    case "$a" in
        --dry) LIVE=0 ;;
    esac
done
ARGS=()
for a in "$@"; do [ "$a" != "--dry" ] && ARGS+=("$a"); done

[ -x vision/build/blobtrack ] || make -C vision
if [ "$LIVE" = 1 ]; then
    [ -x sw/build/cdpr_master ] || make -C sw
fi

MASTER_PID=""
cleanup() {
    if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "stopping cdpr_master (drives de-energize)..."
        kill -INT "$MASTER_PID" 2>/dev/null
        for _ in $(seq 1 50); do kill -0 "$MASTER_PID" 2>/dev/null || break; sleep 0.1; done
        kill -0 "$MASTER_PID" 2>/dev/null && kill -INT "$MASTER_PID" 2>/dev/null
        wait "$MASTER_PID" 2>/dev/null
    fi
    if [ -f logs/cdpr_master.log ]; then
        cp logs/cdpr_master.log "logs/cdpr_master-$(date +%Y%m%d-%H%M%S).log"
    fi
}
trap cleanup EXIT

if [ "$LIVE" = 1 ]; then
    if python3 - <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(0.3)
sys.exit(0 if s.connect_ex(("127.0.0.1", 8421)) == 0 else 1)
PY
    then
        echo "something already listens on 8421 -- a cdpr_master (or the web UI's"
        echo "hardware mode) is running. Stop it, or run without this launcher."
        exit 1
    fi
    echo "starting cdpr_master ${MASTER_ARGS}..."
    # shellcheck disable=SC2086
    # The master writes logs/cdpr_master.log itself (overwritten per run);
    # cleanup() keeps a stamped copy next to the runner's session log.
    mkdir -p logs
    sw/build/cdpr_master ${MASTER_ARGS} > /dev/null &
    MASTER_PID=$!
    for _ in $(seq 1 100); do
        python3 - <<'PY' 2>/dev/null && break
import socket, sys
s = socket.socket(); s.settimeout(0.3)
sys.exit(0 if s.connect_ex(("127.0.0.1", 8421)) == 0 else 1)
PY
        kill -0 "$MASTER_PID" 2>/dev/null || { echo "cdpr_master exited"; exit 1; }
        sleep 0.2
    done
    # The runner takes over the port and Ctrl-C from here.
    python3 ai/bin/run_policy.py --live --opponent --policy "$POLICY" "${ARGS[@]}"
else
    python3 ai/bin/run_policy.py --opponent --policy "$POLICY" "${ARGS[@]}"
fi
