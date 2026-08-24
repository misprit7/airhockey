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
      - the table does NOT need to be clear. The script locates the LED
        first by flashing it and taking the blob that appears only while
        lit, so the field markers, spool reflectors and IR glare are all
        fine to leave exactly as they are
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
import select
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
    ap.add_argument("--tol", type=float, default=12.0,
                    help="px radius for matching the LED blob")
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

    def drain():
        """Discard every frame already queued, so the next read is fresh.

        blobtrack free-runs and the pipe buffers. Without this the first frame
        read after a command can predate the command, which makes the measured
        latency arbitrarily small -- the first version of this script reported
        0.05 ms against a 5 ms frame interval, which is the signature of
        exactly that.
        """
        fd = stream.p.stdout
        while select.select([fd], [], [], 0.0)[0]:
            if not fd.readline():
                break

    def blobs_now():
        _seq, _t, b = next(frames)
        return b[:, :2] if len(b) else np.empty((0, 2))

    # ── Find the LED, rather than counting blobs ─────────────────────────
    #
    # The table is not empty and should not have to be: the field markers,
    # the spool retroreflectors and the IR ring's glare are all permanently
    # visible, which is why the baseline was 14 blobs. Counting them and
    # waiting for 15 triggers on the first frame that happens to be noisy.
    #
    # So locate the LED once: flash it, and take the blob that is present
    # while lit and absent while dark. Then every trial looks for THAT
    # position and nothing else.
    print("locating the A9 LED...")
    dark = []
    for _ in range(30):
        dark.append(blobs_now())
    dark_pts = np.vstack([d for d in dark if len(d)]) if any(
        len(d) for d in dark) else np.empty((0, 2))

    led_px = None
    for attempt in range(6):
        drain()
        ser.write(b"FLASH 400\n")
        ser.flush()
        ser.readline()
        time.sleep(0.10)                     # well inside the 400 ms flash
        lit = blobs_now()
        if len(lit) == 0:
            continue
        if len(dark_pts):
            d = np.linalg.norm(lit[:, None, :] - dark_pts[None, :, :],
                               axis=2).min(axis=1)
            new = lit[d > args.tol]
        else:
            new = lit
        time.sleep(0.45)                     # let it go out
        if len(new) == 1:
            led_px = new[0]
            break
        if len(new) > 1:
            print(f"  attempt {attempt}: {len(new)} new blobs, ambiguous")

    if led_px is None:
        print("\nCould not identify the LED.")
        print("  - is the A9 LED in the camera's field of view?")
        print(f"  - is it brighter than threshold {args.threshold}?")
        print("  - try: python vision/bin/measure_latency.py --threshold 60")
        stream.close(); ser.close()
        return 1
    print(f"LED at pixel ({led_px[0]:.0f}, {led_px[1]:.0f})\n")

    cmd_ms, see_ms = [], []
    for i in range(args.n):
        drain()
        ser.reset_input_buffer()
        t0 = time.perf_counter()
        ser.write(f"FLASH {args.flash_ms}\n".encode())
        ser.flush()
        reply = ser.readline().decode(errors="replace").strip()
        t1 = time.perf_counter()
        if not reply.startswith("OK FLASH"):
            print(f"  trial {i}: unexpected reply {reply!r}")
            continue

        seen = None
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            b = blobs_now()
            if len(b) and np.linalg.norm(b - led_px, axis=1).min() < args.tol:
                seen = (time.perf_counter() - t1) * 1e3
                break
        if seen is None:
            print(f"  trial {i}: LED not seen at its known pixel")
            continue

        cmd_ms.append(rt := (t1 - t0) * 1e3)
        see_ms.append(seen)
        print(f"\r  {len(cmd_ms):3d}/{args.n}  round trip {rt:6.2f} ms   "
              f"LED seen +{seen:6.2f} ms", end="", flush=True)

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
    print(f"Frame interval is {frame_ms:.1f} ms, so each sensing reading "
          f"carries up to {frame_ms:.1f} ms of quantisation (the LED can come "
          f"on\njust after a frame was taken). The MEDIAN over many trials "
          "averages that out; individual readings do not.")

    # A sanity floor. A sensing figure below half a frame interval is not a
    # fast camera, it is a bug -- the first version of this script reported
    # 0.05 ms because it triggered on a noisy blob COUNT and read frames that
    # predated the command.
    if np.median(s) < frame_ms / 2:
        print(f"\nWARNING: median sensing {np.median(s):.2f} ms is below half "
              f"a frame interval ({frame_ms/2:.1f} ms).")
        print("That is not physically possible -- treat this run as invalid "
              "and check the LED was really being found.")
    print("\nPut the total into the sim as observation delay — see "
          "camera_delay in the env config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
