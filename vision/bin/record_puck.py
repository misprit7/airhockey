#!/usr/bin/env python3
"""Record puck trajectories at 200 Hz for system identification.

WHAT THIS IS FOR
    Three constants in the simulator are placeholders that predate the table
    existing -- puck_friction 0.01, wall_restitution 0.85, paddle_restitution
    0.9. None was ever measured. They govern how the puck decelerates and how
    it leaves a cushion, which is most of what a policy has to predict, and
    measuring them needs no robot motion at all: just someone pushing a puck.

WHAT YOU DO
    1. Nothing needs enabling. Drives can be off, mallet anywhere. Camera
       only.
    2. Run it, then push the puck around for a few minutes. Get a mix of:
         - long straight glides, no wall contact   -> friction
         - square-on hits into each cushion        -> restitution
         - glancing hits at 20-40 degrees          -> tangential / spin
         - shots INTO THE ROBOT MALLET             -> paddle restitution
       Vary the SPEED. Restitution is usually speed-dependent and a single
       speed cannot show that; twenty glides at one speed constrain a point.
    3. Ctrl-C, then run vision/bin/fit_puck.py on what it wrote.

    Leave the robot mallet on the table -- PuckTracker separates the two
    structurally (mallet = a 3-blob cluster, puck = the lone blob), and its
    position is recorded so puck-mallet contacts can be identified. Shooting
    at it deliberately is worth doing: the robot mallet and a hand-held one
    are different impedances and the simulator wrongly uses one coefficient
    for both.

    Five varied minutes beats twenty repetitive ones.

WHAT IT WRITES
    JSON Lines, one object per frame where the puck was actually SEEN:
        {"seq":..., "t":<s>, "x":..., "y":..., "vx":..., "vy":...}
    Position in table-grid mm, velocity mm/s from the tracker's
    least-squares slope.

    Coasted samples are DROPPED, not recorded. PuckTracker extrapolates on
    the last velocity when it loses the puck -- necessary for control, poison
    for identification, because a coasted sample is the constant-velocity
    model's own prediction and fitting friction to it would recover
    "friction = 0" from the estimator rather than from the table. The IR
    ring's reflection blinds a ~92x103 mm patch at table centre for up to
    150 ms, so this is not a rare case. Gaps in `seq` are exactly where that
    happened and are left as gaps.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mallet_stream import MalletTracker  # noqa: E402
from puck_stream import BlobStream, PuckTracker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default=None,
                    help="output .jsonl (default: logs/puck_<timestamp>.jsonl)")
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    args = ap.parse_args()

    out = Path(args.output) if args.output else (
        Path("logs") / f"puck_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    stopping = False

    def stop(_sig, _frm):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)

    tracker = PuckTracker()
    mallet = MalletTracker(tracker)
    stream = BlobStream(fps=args.fps, exposure=args.exposure,
                        gain=args.gain, threshold=args.threshold)

    kept = coasted = 0
    first_t = last_t = None
    last_report = time.monotonic()
    print(f"blobtrack {stream.width}x{stream.height} -> {out}")
    print("push the puck: long glides, square wall hits, glancing hits, "
          "varied speeds. Ctrl-C when done.\n")

    try:
        with out.open("w") as fh:
            for seq, t, blobs in stream:
                if stopping:
                    break
                # Ask the tracker whether it can SEE the puck this frame,
                # rather than accepting whatever update() returns -- update()
                # will happily hand back an extrapolation.
                surviving, _world = tracker.candidates(blobs)
                p = tracker.update(t, blobs)
                if p is None:
                    continue
                if len(surviving) == 0:
                    coasted += 1
                    continue

                x, y, vx, vy = p
                first_t = t if first_t is None else first_t
                # The mallet too, so a puck-mallet contact can be told from a
                # wall contact by WHERE it happened, and so the mallet's
                # recoil separates restitution from effective mass. A human
                # mallet and the robot's are different impedances -- the
                # robot's carries reflected rotor inertia and 2.10 N/mm
                # springs -- so they cannot share one coefficient.
                m = mallet.update(blobs)
                last_t = t
                row = {"seq": seq, "t": round(t, 6),
                       "x": round(x, 2), "y": round(y, 2),
                       "vx": round(vx, 1), "vy": round(vy, 1)}
                if m is not None:
                    row["mx"], row["my"] = round(m[0], 2), round(m[1], 2)
                fh.write(json.dumps(row) + "\n")
                kept += 1

                now = time.monotonic()
                if now - last_report >= 1.0:
                    last_report = now
                    speed = (vx * vx + vy * vy) ** 0.5
                    print(f"\r{kept:7d} kept  {coasted:6d} coasted  "
                          f"{t - first_t:6.1f} s   puck ({x:7.1f},{y:6.1f}) "
                          f"{speed:6.0f} mm/s   ", end="", flush=True)
    finally:
        stream.close()

    span = 0.0 if first_t is None else last_t - first_t
    print(f"\n\nwrote {kept} measured samples over {span:.1f} s to {out}")
    print(f"dropped {coasted} coasted samples (tracker extrapolating, "
          f"mostly the IR blind spot)")
    if kept < 2000:
        print("\nthin dataset -- another run with more variety would help "
              "before fitting")
    print(f"\nnext:  python vision/bin/fit_puck.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
