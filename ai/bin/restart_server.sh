#!/bin/bash
# Restart the web UI's server (port 8420) so that what it starts can run at
# realtime priority. Run from anywhere:
#
#     bash ai/bin/restart_server.sh
#
# WHY NOT JUST `python -m airhockey.server`: the AI's control loop, the camera
# tracker and the motor controller are started BY this server, and they ask for
# realtime priority (SCHED_FIFO 40, see ai/bin/play.sh). They get it only if
# the server itself carries the account's rtprio limit, which is set at LOGIN
# from /etc/security/limits.d/99-airhockey.conf. A server started from a shell
# that was opened before that file existed, from a desktop session that was,
# or from a tool's own shell has limit 0. Nothing fails then: every session
# just runs at normal priority, and any busy program on this PC delays the
# robot. On 2026-11-19 a 15-minute batch job of another project made 2-9% of
# policy ticks more than 30 ms late for six minutes of every fifteen, and the
# robot conceded five times as often in those minutes.
#
# So this starts the server through a fresh login to this machine
# (ssh localhost, a login shell so the camera's environment is set), detached,
# appending to logs/web_server.log. It refuses while the table is in use, and
# it never touches the motor controller: one that is already running keeps
# whatever priority it was started with until "Shut down motor controller".
#
# --wait-idle <seconds>   instead of refusing, wait until the table has been
#                         idle for that long without a break, then restart.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=8420
WAIT_IDLE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --wait-idle) WAIT_IDLE=$2; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)"; exit 2 ;;
    esac
done

server_pid() { pgrep -u "$(id -u)" -f "[p]ython3? -m airhockey.server" | head -1; }

# Why the table is busy, or nothing when it is free. The server's own account
# first (a tab holding the hardware, a take-up, an AI session), then the
# processes, in case the server is not answering.
busy() {
    local state
    state=$(curl -s -m 3 "localhost:$PORT/api/robot" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
if d.get("hardware_ui"):
    print("a browser tab has the hardware on (manual control)")
elif (d.get("takeup") or {}).get("state") == "running":
    print("the cable slack take-up is running")
elif (d.get("ai") or {}).get("state") in ("running", "stopping"):
    print("an AI session is running")
' 2>/dev/null)
    [ -n "$state" ] && { echo "$state"; return; }
    pgrep -u "$(id -u)" -f "[r]un_policy.py" >/dev/null && { echo "a policy session (run_policy.py) is running"; return; }
    pgrep -u "$(id -u)" -f "[a]i/bin/play.sh" >/dev/null && { echo "play.sh is running"; return; }
}

if [ -n "$WAIT_IDLE" ]; then
    echo "waiting until the table has been idle for $WAIT_IDLE s..."
    idle_since=""
    while :; do
        why=$(busy)
        now=$(date +%s)
        if [ -n "$why" ]; then
            idle_since=""
        else
            [ -z "$idle_since" ] && idle_since=$now
            [ $((now - idle_since)) -ge "$WAIT_IDLE" ] && break
        fi
        sleep 5
    done
fi
why=$(busy)
if [ -n "$why" ]; then
    echo "Not restarting: $why. Stop it first (or use --wait-idle <seconds>)."
    exit 1
fi

old=$(server_pid)
if [ -n "$old" ]; then
    kill -TERM "$old"
    for _ in $(seq 1 50); do kill -0 "$old" 2>/dev/null || break; sleep 0.2; done
    if kill -0 "$old" 2>/dev/null; then
        echo "The running server (pid $old) did not stop; nothing was started."
        exit 1
    fi
    echo "stopped the running server (pid $old)"
fi

mkdir -p "$ROOT/logs"
# setsid -f, not "&": the login shell here is zsh, which starts background
# jobs at nice +5 (BG_NICE), and the server passes its niceness on to
# everything it starts that does not get realtime priority.
start="cd $(printf %q "$ROOT") && PYTHONPATH=ai setsid -f nohup python3 -m airhockey.server >> logs/web_server.log 2>&1 < /dev/null"
if ssh -o BatchMode=yes -o ConnectTimeout=5 localhost true 2>/dev/null; then
    ssh -o BatchMode=yes localhost "exec \$SHELL -lc $(printf %q "$start")"
else
    echo "WARNING: cannot log in to localhost with a key (ssh -o BatchMode=yes localhost), so the"
    echo "         server is started from this shell and has this shell's limits (ulimit -r: $(ulimit -r))."
    bash -c "$start"
fi

for _ in $(seq 1 80); do
    curl -s -m 1 "localhost:$PORT/api/robot" >/dev/null && break
    sleep 0.25
done
new=$(server_pid)
if [ -z "$new" ] || ! curl -s -m 2 "localhost:$PORT/api/robot" >/dev/null; then
    echo "The server did not come up; the end of logs/web_server.log:"
    tail -n 15 "$ROOT/logs/web_server.log"
    exit 1
fi
limit=$(awk '/Max realtime priority/ {print $4}' "/proc/$new/limits")
echo "web server running (pid $new), http://localhost:$PORT"
if [ "${limit:-0}" -ge 40 ] 2>/dev/null; then
    echo "realtime priority: available (limit $limit). AI sessions started from now on use it."
    echo "A motor controller that was already running keeps its old priority until it is shut down"
    echo "from the Robot tab and started again."
else
    echo "WARNING: realtime priority is NOT available to the server (limit ${limit:-unknown})."
    echo "         Other programs on this PC can delay the robot. It needs a root-owned"
    echo "         /etc/security/limits.d/99-airhockey.conf:  $(id -un)  -  rtprio  50"
fi
