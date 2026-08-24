#!/usr/bin/env python3
"""Measure end-to-end latency by flashing the Teensy LED and watching for it.

WHY THIS NUMBER MATTERS MORE THAN MOST
    A puck at 5 m/s covers 5 mm per millisecond. Every millisecond between
    the world changing and the policy acting on it is 5 mm of error the
    policy cannot recover, and it compounds: the sim must apply the SAME
    delay or a policy trained there will act on information the real robot
    will not have yet. One measured number fixes that; a guess does not.

WHAT IT SEPARATES
    Two halves, and they need different fixes, so measuring them together
    would be useless:

      COMMAND path   host -> USB -> Teensy. Timed by the serial round trip:
                     the firmware raises the LED and only then replies, so
                     half the round trip is a fair estimate of one direction.
      SENSING path   LED lit -> camera exposes -> blob centroided -> Python
                     sees it. Timed by the gap between the reply arriving and
                     the frame in which the LED appears.

    Total loop latency for control is roughly SENSING + COMMAND, plus
    whatever the policy itself costs.

SETUP — THE ONE PHYSICAL THING YOU MUST DO
    The external LED on A9 (to ground) has to be visible to the camera. Not
    the on-board one: the board sits wherever it is mounted, while this needs
    to be on the playing surface where the camera is looking. It does not need
    to be in focus; it needs to be bright enough to cross the blob threshold
    and be the ONLY thing that changes. So:
      - put the A9 LED on the table where the camera sees it
      - take the PUCK off the table, and leave the mallet still, so nothing
        else appears or moves
      - lights as they normally are for tracking

    Nothing is energised and nothing moves. The drives can be off entirely;
    this does not talk to cdpr_master.

USAGE
    python vision/bin/measure_latency.py                 # 30 flashes
    python vision/bin/measure_latency.py --n 100         # tighter estimate

    Do NOT run cdpr_master at the same time — it holds the Teensy port.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from puck_stream import BlobStream  # noqa: E402

try:
    import serial  # type: ignore
except ImportError:
    serial = None


def find_teensy() -> str | None:
    import glob
    ports = sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="number of flashes")
    ap.add_argument("--flash-ms", type=int, default=40)
    ap.add_argument("--port", default=None)
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--threshold", type=int, default=90)
    args = ap.parse_args()

    if serial is None:
        sys.exit("pyserial not installed:  pip install pyserial")
    port = args.port or find_teensy()
    if port is None:
        sys.exit("no /dev/ttyACM* found — is the Teensy plugged in?")

    print(f"Teensy on {port}")
    ser = serial.Serial(port, 115200, timeout=1.0)
    time.sleep(0.3)
    ser.reset_input_buffer()

    stream = BlobStream(fps=args.fps, exposure=args.exposure,
                        gain=args.gain, threshold=args.threshold)
    print(f"blobtrack {stream.width}x{stream.height}")
    frames = iter(stream)

    # Baseline: what does the scene look like with the LED off? Anything
    # present now is furniture and must not be mistaken for the flash.
    base = []
    for _ in range(60):
        _seq, _t, blobs = next(frames)
        base.append(len(blobs))
    baseline = int(np.median(base))
    print(f"baseline {baseline} blob(s) with the LED off\n")
    if baseline > 2:
        print("NOTE: several blobs already visible. Take the puck off the "
              "table and keep the mallet still, or the flash may not be\n"
              "      separable from what is already there.\n")

    cmd_ms, see_ms = [], []
    for i in range(args.n):
        # Drain anything the camera queued while we were between trials, so
        # the first frame we look at is genuinely after the command.
        ser.reset_input_buffer()
        t0 = time.perf_counter()
        ser.write(f"FLASH {args.flash_ms}\n".encode())
        ser.flush()
        reply = ser.readline().decode(errors="replace").strip()
        t1 = time.perf_counter()
        if not reply.startswith("OK FLASH"):
            print(f"  trial {i}: unexpected reply {reply!r} — is the firmware "
                  "current? FLASH was added 2026-08-23")
            continue
        rt = (t1 - t0) * 1e3

        seen = None
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            _seq, _t, blobs = next(frames)
            if len(blobs) > baseline:
                seen = (time.perf_counter() - t1) * 1e3
                break
        if seen is None:
            print(f"  trial {i}: LED never seen — is it in frame and above "
                  f"threshold {args.threshold}?")
            continue

        cmd_ms.append(rt)
        see_ms.append(seen)
        print(f"\r  {len(cmd_ms):3d}/{args.n}  round trip {rt:6.2f} ms   "
              f"LED seen +{seen:6.2f} ms", end="", flush=True)

        # Let the LED go out and the camera settle before the next trial.
        time.sleep(args.flash_ms / 1000.0 + 0.05)

    stream.close()
    ser.close()
    print("\n")

    if not cmd_ms:
        print("no usable trials.")
        return 1

    c, s = np.array(cmd_ms), np.array(see_ms)
    print("─" * 62)
    print(f"{'':22}{'median':>10}{'mean':>10}{'p90':>10}{'n':>6}")
    print(f"{'command round trip':22}{np.median(c):10.2f}{c.mean():10.2f}"
          f"{np.percentile(c, 90):10.2f}{len(c):6d}")
    print(f"{'command one way (~half)':22}{np.median(c)/2:10.2f}"
          f"{c.mean()/2:10.2f}{np.percentile(c, 90)/2:10.2f}{len(c):6d}")
    print(f"{'sensing (LED->seen)':22}{np.median(s):10.2f}{s.mean():10.2f}"
          f"{np.percentile(s, 90):10.2f}{len(s):6d}")
    total = np.median(c) / 2 + np.median(s)
    print("─" * 62)
    print(f"\nTOTAL LOOP ~= {total:.1f} ms  (sensing + one-way command)")
    print(f"At 5 m/s the puck moves {total * 5:.0f} mm in that time.\n")

    frame_ms = 1000.0 / args.fps
    print(f"Frame interval is {frame_ms:.1f} ms, so the sensing figure "
          f"carries at least +/-{frame_ms/2:.1f} ms of quantisation;")
    print("the median over many trials averages that out, the individual "
          "readings do not.")
    print("\nPut the total into the sim as observation delay — see "
          "camera_delay in the env config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
