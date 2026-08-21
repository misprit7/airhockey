#!/usr/bin/env python3
"""Run the hardcoded goalie against the real rig.

    # dry run — tracks and predicts, commands nothing. Start here.
    python ai/bin/goalie_demo.py --dry-run

    # for real — needs sw/build/cdpr_master running, and ONLY that. Do not
    # also run sw/build/activate: both open the same SC-Hub port and the
    # second one errors. cdpr_master energizes the drives itself.
    python ai/bin/goalie_demo.py

This ENERGIZES the drives itself. It has to: cdpr_master serves one client at
a time, so a process holding that slot cannot also ask the web UI to enable.
Turn Hardware OFF in the web UI before running this, or the master will refuse
the connection as busy.

Glue only. The two halves worth keeping are elsewhere:
  vision/bin/puck_stream.py   tracking, survives the RL transition
  ai/airhockey/demo_goalie.py the policy, gets deleted at that point

MOVES THE ROBOT. Nothing here runs on its own — it commands only while it is
in the foreground, and ctrl-C parks the paddle at rest before exiting.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai"))
sys.path.insert(0, str(ROOT / "vision" / "bin"))

from airhockey.demo_goalie import Goalie, GoalieConfig  # noqa: E402
from airhockey.hardware import CDPRClient  # noqa: E402
from puck_stream import BlobStream, PuckTracker  # noqa: E402
import track_mallet as tm  # noqa: E402

_stop = False


def _sig(_s, _f):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="track and predict, but command nothing")
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    ap.add_argument("--speed", type=float, default=8000.0,
                    help="paddle speed cap mm/s. 8000 of the firmware's 12000; "
                         "the motors bind at 12968 of cable")
    ap.add_argument("--accel", type=float, default=24000.0,
                    help="paddle accel cap mm/s^2. NOTE this is ABOVE the "
                         "~15120 at which the paddle tips (g*r/h, r=50.4, "
                         "h=32.7) and above the ~17400 the cables can make "
                         "near the centreline edge of the workspace. Asked "
                         "for deliberately; drop it if the paddle hops.")
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="jerk-limit ramp, ms")
    ap.add_argument("--cmd-hz", type=float, default=100.0,
                    help="command rate to the Teensy; tracking stays at --fps")
    ap.add_argument("--no-enable", action="store_true",
                    help="assume the drives are already energized. Only useful "
                         "if something else enabled them, which cannot be the "
                         "web UI — the master serves one client and this holds "
                         "it")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    goalie = Goalie(GoalieConfig())
    tracker = PuckTracker()

    client = None
    if not args.dry_run:
        # Measure the paddle BEFORE opening the tracker: only one process can
        # hold the Spinnaker device, and ENABLE needs this position — it is
        # the reference every later cable length is measured from, so getting
        # it wrong offsets the whole session.
        print("measuring the paddle for the enable reference...")
        try:
            mx, my = tm.measure()[:2]
        except Exception as e:      # noqa: BLE001
            sys.exit(f"could not measure the paddle ({e}).\n"
                     "  Is the camera free? Stop the tracker view in the web UI.")
        print(f"  paddle at ({mx:.1f}, {my:.1f}) mm")

        client = CDPRClient()
        try:
            client.connect()
        except OSError as e:
            sys.exit(f"cannot reach cdpr_master on 8421 ({e}) — is it running?\n"
                     "  sw/build/cdpr_master\n"
                     "Run that ALONE — not alongside sw/build/activate, which "
                     "opens the same USB port.")

        if not args.no_enable:
            # This ENERGIZES the drives and starts the control loop. It has to
            # happen here rather than from the web UI: the master serves one
            # client at a time and this process is holding it.
            print("ENABLING the drives (they will hold position, not move)...")
            try:
                client.enable(mx, my)
            except Exception as e:      # noqa: BLE001
                sys.exit(f"enable failed: {e}")
            print("  enabled")

        client.set_limits(args.speed, args.accel)
        client.set_ramp(args.ramp)
        print(f"limits -> {args.speed:.0f} mm/s, {args.accel:.0f} mm/s^2, "
              f"ramp {args.ramp:.1f} ms")

    stream = BlobStream(fps=args.fps, exposure=args.exposure, gain=args.gain,
                        threshold=args.threshold)
    print(f"tracking {stream.width}x{stream.height} at {args.fps:.0f} Hz"
          + ("  [DRY RUN]" if args.dry_run else "") + "  — ctrl-C to stop\n")

    period = 1.0 / args.cmd_hz
    next_cmd = 0.0
    n = seen = 0
    t_wall = time.time()
    last_print = 0.0
    try:
        for _seq, t, blobs in stream:
            if _stop:
                break
            n += 1
            puck = tracker.update(t, blobs)
            if puck is not None:
                seen += 1
            tx, ty = goalie.update(puck)

            if t >= next_cmd:
                next_cmd = t + period
                if client is not None:
                    try:
                        client.command_position(tx, ty, args.speed)
                    except Exception as e:      # noqa: BLE001
                        print(f"command failed: {e}")
                        break

            if t - last_print > 0.25:
                last_print = t
                rate = n / max(time.time() - t_wall, 1e-9)
                if puck is None:
                    print(f"{rate:6.1f} Hz  seen {100*seen/max(n,1):3.0f}%  "
                          f"puck --                              rest")
                else:
                    x, y, vx, vy = puck
                    eta = goalie.last_eta
                    print(f"{rate:6.1f} Hz  seen {100*seen/max(n,1):3.0f}%  "
                          f"puck ({x:6.0f},{y:5.0f}) v({vx:7.0f},{vy:6.0f}) "
                          f"|v|{math.hypot(vx,vy):6.0f}  -> "
                          f"({tx:6.0f},{ty:5.0f})"
                          + (f"  eta {eta*1000:5.0f} ms" if eta else "  rest"))
    finally:
        stream.close()
        if client is not None:
            try:
                rx, ry = goalie.rest()
                client.command_position(rx, ry, args.speed)
                time.sleep(0.3)
                client.close()
                print("\nparked at rest, disconnected (drives left as they were)")
            except Exception:      # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
