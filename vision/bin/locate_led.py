#!/usr/bin/env python3
"""Find the latency LED by differencing a frame with it on against one with
it off.

WHY A SEPARATE TOOL
    blobtrack deliberately never sends frames -- at 200 Hz a 1440x1080 Mono8
    frame is 311 MB/s, which is the whole reason thresholding and centroiding
    happen in C++. So the latency script sees coordinates, not pixels, and
    cannot difference anything.

    `snap --stream` DOES hand back images, one per request. It is far too slow
    for tracking and perfect for this: the LED does not move, so locating it is
    a one-off that can afford a slow, unambiguous method.

WHY DIFFERENCING BEATS WHAT THE LATENCY SCRIPT WAS DOING
    Trying to spot the LED among blob COORDINATES means asking "which blob is
    new", and that fails exactly where you would naturally put the LED: in the
    playing area under the camera, which is where the IR ring's reflection
    already is. A blob 5 px from a glare blob is indistinguishable from the
    glare blob.

    A pixel difference does not care. The LED is the brightest thing that
    CHANGED, whatever it happens to be sitting on top of, and a saturated
    blob merging with the glare shows up plainly as a bright patch that was
    not there before.

USAGE
    # cdpr_master must not be running (it holds the Teensy port)
    python vision/bin/locate_led.py

    It prints the pixel and, once you have it:
    python vision/bin/measure_latency.py --n 50 --led-px X,Y
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import Stream  # noqa: E402

try:
    import serial  # type: ignore
except ImportError:
    serial = None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    ap.add_argument("--exposure", type=float, default=300.0)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--n", type=int, default=5,
                    help="frame pairs to average")
    ap.add_argument("--save", default=None,
                    help="write the difference image here for eyeballing")
    args = ap.parse_args()

    if serial is None:
        sys.exit("pyserial not installed:  pip install pyserial")
    ports = sorted(glob.glob("/dev/ttyACM*"))
    port = args.port or (ports[0] if ports else None)
    if port is None:
        sys.exit("no /dev/ttyACM* found — is the Teensy plugged in?")

    ser = serial.Serial(port, 115200, timeout=1.0)
    time.sleep(0.3)
    ser.reset_input_buffer()
    print(f"Teensy on {port}")

    cam = Stream(args.exposure, args.gain)
    print(f"camera {cam.w}x{cam.h}, averaging {args.n} pairs\n")

    dark_sum = np.zeros((cam.h, cam.w), np.float64)
    lit_sum = np.zeros((cam.h, cam.w), np.float64)
    try:
        for i in range(args.n):
            d = cam.grab()
            if d is None:
                sys.exit("camera returned no frame")
            dark_sum += d

            # Long flash so the LED is certainly on for the whole exposure --
            # grab() costs a round trip and we are not racing anything here.
            ser.write(b"FLASH 900\n")
            ser.flush()
            ser.readline()
            time.sleep(0.15)
            l = cam.grab()
            if l is None:
                sys.exit("camera returned no frame")
            lit_sum += l
            time.sleep(0.9)
            print(f"\r  pair {i + 1}/{args.n}", end="", flush=True)
    finally:
        cam.close()
        ser.close()
    print()

    dark = dark_sum / args.n
    lit = lit_sum / args.n
    diff = lit - dark

    peak = float(diff.max())
    y, x = np.unravel_index(int(np.argmax(diff)), diff.shape)

    # Centroid over the bright region rather than the single peak pixel: a
    # saturated LED has a plateau, and argmax picks an arbitrary corner of it.
    thr = max(peak * 0.5, 10.0)
    ys, xs = np.nonzero(diff > thr)
    if len(xs):
        w = diff[ys, xs]
        cx, cy = float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())
        n_px = len(xs)
    else:
        cx, cy, n_px = float(x), float(y), 0

    print(f"brightest change   {peak:6.1f} counts at ({x}, {y})")
    print(f"bright region      {n_px} px above {thr:.0f}")
    print(f"centroid           ({cx:.1f}, {cy:.1f})")

    if args.save:
        try:
            import cv2
            vis = np.clip(diff, 0, 255).astype(np.uint8)
            cv2.imwrite(args.save, vis)
            print(f"difference image   {args.save}")
        except Exception as e:  # noqa: BLE001
            print(f"could not write {args.save}: {e}")

    if peak < 15:
        print("\nThat is a very small change — the LED may not be firing, or")
        print("may be outside the frame. Check it blinks during this run, and")
        print("that it is inside the camera's view rather than merely on the")
        print("table.")
        return 1

    print(f"\nnext:  python vision/bin/measure_latency.py --n 50 "
          f"--led-px {cx:.0f},{cy:.0f}")
    if n_px > 3000:
        print(f"\nNOTE the lit region is {n_px} px, which is large. blobtrack")
        print("discards blobs above --max-area 4000, so the LED may be being")
        print("thrown away as too big. If measure_latency still cannot see it,")
        print("add:  --max-area 200000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
