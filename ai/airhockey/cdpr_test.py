"""Python CDPR test — mirrors sw/bin/cdpr_test.cpp but goes through the TCP server."""

import math
import signal
import sys
import time

from airhockey.hardware import CDPRClient

# Physical parameters (must match sw/lib/cdpr_config.h)
WIDTH_MM = 606.0
HEIGHT_MM = 730.0

stop = False

def on_sigint(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, on_sigint)


def main():
    client = CDPRClient()

    print("=== CDPR Test (Python, via TCP server) ===\n")
    print(f"Table: {WIDTH_MM:.0f} x {HEIGHT_MM:.0f} mm")

    print("Connecting to CDPR server...")
    try:
        client.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Is build/cdpr_server running?")
        return 1

    print("Connected.\n")

    cx = WIDTH_MM / 2.0
    cy = HEIGHT_MM / 2.0

    print(f"Setting position to center ({cx:.1f}, {cy:.1f})...")
    client.set_position(cx, cy)
    print("Done.\n")

    # Small test moves — 50mm square at 50 mm/s
    step = 50.0
    speed = 50.0

    moves = [
        (step, 0, "right"),
        (0, step, "up"),
        (-step, 0, "left"),
        (0, -step, "down"),
    ]

    x, y = cx, cy

    print(f"Is the cart at the center of the table?")
    input("Press Enter to proceed, Ctrl+C to abort: ")

    print(f"Moving in a {step:.0f}mm square at {speed:.0f} mm/s.\n")

    for dx, dy, desc in moves:
        if stop:
            break
        tx = x + dx
        ty = y + dy
        dist = math.hypot(dx, dy)
        wait_s = dist / speed + 0.5  # expected duration + margin
        print(f"  {desc} to ({tx:.1f}, {ty:.1f})... ", end="", flush=True)
        try:
            actual_x, actual_y = client.command_position(tx, ty, speed)
            time.sleep(wait_s)
            actual_x, actual_y = client.get_position()
            print(f"done (pos={actual_x:.1f}, {actual_y:.1f})")
            x, y = actual_x, actual_y
        except Exception as e:
            print(f"FAILED: {e}")
            break

    print(f"\nFinal position: ({x:.1f}, {y:.1f})")

    client.close()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
