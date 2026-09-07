#!/bin/bash
# Turn it on and it plays: the master, the camera and the policy, in one
# command, from the REPO ROOT.
#
#   bash ai/bin/play.sh --policy tdmpc2:2.3-control-gate-selfplay
#   bash ai/bin/play.sh --policy tdmpc2:latest --gentle   # FIRST RUN OF ANY NEW CHECKPOINT
#   bash ai/bin/play.sh --policy tdmpc2:latest --dry      # camera + policy, commands nothing
#   bash ai/bin/play.sh --policy tdmpc2:latest --tension 1.5
#
# This script's own flags: --policy <spec> (default tdmpc2:latest), --dry,
# --tension <mm> (cdpr_master's startup pretension; default 0 = slack, which
# is what the tracking test validated). Everything else goes to
# ai/bin/run_policy.py, whose defaults reproduce training (6 planner
# iterations, the run's horizon, shot requests drawn per possession, the sim
# body's caps, 50 Hz); it prints a sim/real alignment block and marks every
# DEVIATION before the first live command.
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

POLICY=tdmpc2:latest
MASTER_ARGS=()
LIVE=1
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry) LIVE=0; shift ;;
        --policy) POLICY=$2; shift 2 ;;
        --policy=*) POLICY=${1#--policy=}; shift ;;
        --tension) MASTER_ARGS+=(--tension "$2"); shift 2 ;;
        --tension=*) MASTER_ARGS+=(--tension "${1#--tension=}"); shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
# Caps: run_policy's defaults are the sim body's (12000 mm/s, 40000 mm/s^2)
# so the table runs what the policy trained on; --gentle or --speed/--accel
# override (and are reported as deviations), --governor trims when the
# paddle falls behind.

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
    echo "starting cdpr_master ${MASTER_ARGS[*]:-}..."
    # The master writes logs/cdpr_master.log itself (overwritten per run);
    # cleanup() keeps a stamped copy next to the runner's session log.
    mkdir -p logs
    sw/build/cdpr_master "${MASTER_ARGS[@]}" > /dev/null &
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
