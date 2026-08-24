#!/usr/bin/env python3
"""Log what the robot was told and what it actually did, at camera rate.

Serves two jobs that want the same data, which is why it is one tool:

  SIM2REAL   commanded target vs camera-measured paddle position is exactly
             what ai/bin/replay_gap.py needs to score the simulator.
  DIAGNOSIS  per-cable torque and step counts against position is the
             instrument for the parked edge-overload problem -- torque
             climbing with distance from the calibration point means a scale
             error, torque jumping at one place means something local.

COMMANDS NOTHING. It opens a second TCP connection to cdpr_master purely to
read STATUS, and reads the camera. Whatever is driving the robot -- you via
the web UI, goalie_demo, a policy -- is unaffected and unaware.

USAGE
    # in one terminal, however you normally drive it:
    sw/build/cdpr_master --tension 0
    # in another:
    python ai/bin/log_hardware.py

    Then move the robot around: the web UI, the goalie demo, whatever. Aim
    for a spread of move sizes and directions, and include some moves toward
    the edges, since that is where the unexplained overload lives.

    Ctrl-C. Then:  python ai/bin/replay_gap.py logs/hw_<timestamp>.jsonl

NOTE cdpr_master serves ONE client at a time. If the web UI or goalie_demo
already holds it, this will be refused with 'ERR busy' -- that is the master
protecting itself, not a bug here. Drive the robot from THIS tool's session
by starting whatever you use after it, or accept camera-only logging (which
still gives puck and paddle, just no torque).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision" / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--no-master", action="store_true",
                    help="camera only; do not connect to cdpr_master")
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    args = ap.parse_args()

    from mallet_stream import MalletTracker  # noqa: E402
    from puck_stream import BlobStream, PuckTracker  # noqa: E402

    out = Path(args.output) if args.output else (
        Path("logs") / f"hw_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.no_master:
        from airhockey.hardware import CDPRClient  # noqa: E402
        client = CDPRClient(args.host, args.port)
        try:
            client.connect()
            print(f"reading cdpr_master at {args.host}:{args.port}")
        except Exception as e:  # noqa: BLE001
            print(f"could not attach to cdpr_master ({e})")
            print("continuing camera-only; no commanded target or torque")
            client = None

    stopping = False

    def stop(_s, _f):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)

    tracker = PuckTracker()
    mallet = MalletTracker(tracker)   # shares the scene rejection
    stream = BlobStream(fps=args.fps, exposure=args.exposure,
                        gain=args.gain, threshold=args.threshold)

    n = 0
    last_status = 0.0
    status = {}
    t0 = None
    print(f"logging to {out}   (Ctrl-C to stop)\n")
    try:
        with out.open("w") as fh:
            for seq, t, blobs in stream:
                if stopping:
                    break
                t0 = t if t0 is None else t0

                # STATUS is a cache the master refills at 50 Hz, so polling it
                # at 200 Hz would trade three round trips for no new
                # information. Sample it at its own rate and carry it forward.
                if client is not None and t - last_status >= 0.02:
                    last_status = t
                    try:
                        status = client.get_status()
                    except Exception:  # noqa: BLE001
                        status = {}

                rec = {"seq": seq, "t": round(t - t0, 6)}
                if status:
                    rec["cmd_x"] = status.get("cmd_x")
                    rec["cmd_y"] = status.get("cmd_y")
                    # 'x'/'y' from the master are where the TEENSY believes
                    # the paddle is -- its own integrated command, not a
                    # measurement. Kept under a different name so it can never
                    # be mistaken for ground truth by replay_gap.
                    rec["teensy_x"] = status.get("x")
                    rec["teensy_y"] = status.get("y")
                    for k in ("c0", "c1", "c2", "c3", "speed_limit",
                              "accel_limit", "limit_flags"):
                        if k in status:
                            rec[k] = status[k]

                # x/y are the CAMERA-measured paddle position and are what
                # replay_gap scores against. Named plainly, unlike
                # teensy_x/teensy_y above, which are the controller's own
                # integrated command and are not a measurement of anything.
                m = mallet.update(blobs)
                if m is not None:
                    rec["x"], rec["y"], rec["n_markers"] = (
                        round(m[0], 2), round(m[1], 2), m[2])

                p = tracker.update(t, blobs)
                if p is not None:
                    px, py, pvx, pvy = p
                    rec.update(puck_x=round(px, 2), puck_y=round(py, 2),
                               puck_vx=round(pvx, 1), puck_vy=round(pvy, 1))

                fh.write(json.dumps(rec) + "\n")
                n += 1
                if n % 200 == 0:
                    cx = rec.get("cmd_x")
                    print(f"\r{n:7d} samples  {t - t0:6.1f} s  "
                          f"cmd {cx if cx is not None else '--'}   ",
                          end="", flush=True)
    finally:
        stream.close()
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    print(f"\n\nwrote {n} samples to {out}")
    print(f"\nnext:  python ai/bin/replay_gap.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
