#!/usr/bin/env python3
"""Measure the sim-to-real gap by replaying real commands through the sim.

THE POINT
    Every other statement about simulator fidelity is an opinion until there
    is a number. This is the number: take a recording of what the robot was
    ACTUALLY told and where it ACTUALLY went, drive the simulator with the
    identical command sequence from the identical initial state, and report
    how far apart they are after 0.5, 1 and 2 seconds.

    It is deliberately OPEN LOOP. No feedback, no correction -- errors are
    allowed to accumulate, because that is what happens to a policy planning
    two seconds ahead. A closed-loop comparison would hide exactly the drift
    that breaks transfer.

WHAT IT NEEDS
    A JSONL log with, per sample:
        t          seconds
        cmd_x/cmd_y   what was commanded, grid mm
        x/y        where the paddle actually was, grid mm  (from the camera)
    plus optionally v_max/a_max if the caps changed during the run.

    ai/bin/log_hardware.py writes this. Until you have one, --selftest below
    exercises the whole path against a synthetic log.

READING THE RESULT
    Divergence is reported in millimetres and against two yardsticks that
    already exist: 0.377 mm is one motor step, ~4 mm is the mallet tracking
    error. Below tracking error means the sim is not the limiting factor.
    Growing roughly linearly in time means a velocity/scale error; growing
    quadratically means an acceleration one; a constant offset means the
    initial state or the frame is wrong rather than the dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airhockey.dynamics import MAX_ACCEL_M_S2, MAX_SPEED_M_S  # noqa: E402
from airhockey.motion import DEFAULT_SIM_DT, CartState, advance  # noqa: E402

HORIZONS_S = (0.5, 1.0, 2.0)
MOTOR_STEP_MM = 0.377
TRACKING_ERR_MM = 4.0


def load(path):
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln]
    if len(rows) < 10:
        sys.exit(f"{path}: need at least 10 samples, found {len(rows)}")
    need = ("t", "cmd_x", "cmd_y", "x", "y")
    missing = [k for k in need if k not in rows[0]]
    if missing:
        sys.exit(f"{path}: missing field(s) {missing}")
    return rows


def replay(rows, ramp_s, v_max, a_max, dt=DEFAULT_SIM_DT):
    """Drive one simulated cart with the logged commands; return its path."""
    s = CartState(1)
    s.reset(np.float32(rows[0]["x"]), np.float32(rows[0]["y"]))
    tx = np.zeros(1, dtype=np.float32)
    ty = np.zeros(1, dtype=np.float32)

    sim = np.empty((len(rows), 2))
    sim[0] = (rows[0]["x"], rows[0]["y"])
    for i in range(1, len(rows)):
        gap = rows[i]["t"] - rows[i - 1]["t"]
        if gap <= 0:
            sim[i] = sim[i - 1]
            continue
        tx[0] = rows[i - 1]["cmd_x"]
        ty[0] = rows[i - 1]["cmd_y"]
        # Round rather than truncate: truncating loses up to a full tick per
        # sample, which at 200 Hz accumulates into a systematic lag that would
        # be read as the simulator being slow.
        n = max(1, int(round(gap / dt)))
        advance(s, tx, ty, v_max, a_max, ramp_s, gap / n, n)
        sim[i] = (s.x[0], s.y[0])
    return sim


def report(rows, sim):
    t = np.array([r["t"] for r in rows]) - rows[0]["t"]
    real = np.array([(r["x"], r["y"]) for r in rows])
    err = np.linalg.norm(sim - real, axis=1)

    print(f"{len(rows)} samples over {t[-1]:.2f} s\n")
    print(f"{'horizon':>10}{'divergence':>14}{'vs motor step':>16}"
          f"{'vs tracking':>14}")
    print("-" * 54)
    for h in HORIZONS_S:
        if t[-1] < h:
            print(f"{h:>9.1f}s{'(log too short)':>14}")
            continue
        e = err[np.searchsorted(t, h)]
        print(f"{h:>9.1f}s{e:>13.3f}mm{e / MOTOR_STEP_MM:>15.1f}x"
              f"{e / TRACKING_ERR_MM:>13.2f}x")

    # PEAK and MEAN, not final. A paddle that has settled on its target
    # agrees with any simulator regardless of how wrong the dynamics are,
    # so the last sample of a log that ends at rest says nothing.
    final, peak, mean = err[-1], float(err.max()), float(err.mean())
    print(f"\npeak {peak:.3f} mm, mean {mean:.3f} mm, "
          f"final {final:.3f} mm (final is uninformative if it ended at rest)")

    # Shape of the growth says which kind of parameter is wrong.
    ok = t > 0.1
    if ok.sum() > 20:
        lin = np.polyfit(t[ok], err[ok], 1)
        quad = np.polyfit(t[ok], err[ok], 2)
        r_lin = np.std(err[ok] - np.polyval(lin, t[ok]))
        r_quad = np.std(err[ok] - np.polyval(quad, t[ok]))
        print()
        if err[ok].mean() < MOTOR_STEP_MM:
            print("Divergence is below one motor step. The simulator is not "
                  "what limits fidelity here.")
        elif r_quad < 0.6 * r_lin:
            print("Divergence grows QUADRATICALLY -> an acceleration-side "
                  "error: accel cap, jerk ramp, or the effective mass.")
        elif lin[0] > 0.5:
            print(f"Divergence grows LINEARLY at {lin[0]:.2f} mm/s -> a "
                  "velocity or scale error.\n"
                  "SPOOL_RADIUS_MM is the largest scale factor in the machine "
                  "and is still unverified.")
        else:
            print(f"Divergence is not cleanly growing (slope "
                  f"{lin[0]:+.2f} mm/s, mean {err[ok].mean():.2f} mm) "
                  "-- likely per-move rather than accumulating.\n"
                  "Look at whether error spikes during moves and settles "
                  "between them, which points at the transient (accel cap,\n"
                  "jerk ramp) rather than at a scale error.")
    return peak, mean


def selftest() -> int:
    """Replay a synthetic log the sim itself generated, plus a detuned one.

    The first must come back at essentially zero -- if replaying the
    simulator's own output does not reproduce it, the harness is broken and
    any number it prints about real hardware is meaningless. The second
    confirms it can actually SEE a known error rather than always reporting
    agreement.
    """
    rng = np.random.default_rng(0)
    dt = 0.005
    rows, s = [], CartState(1)
    s.reset(np.float32(1500.0), np.float32(480.0))
    tx = np.zeros(1, dtype=np.float32)
    ty = np.zeros(1, dtype=np.float32)
    cx, cy = 1500.0, 480.0
    for i in range(600):
        if i % 40 == 0:
            cx = rng.uniform(1400, 1900)
            cy = rng.uniform(200, 780)
        rows.append({"t": i * dt, "cmd_x": cx, "cmd_y": cy,
                     "x": float(s.x[0]), "y": float(s.y[0])})
        tx[0], ty[0] = cx, cy
        n = max(1, int(round(dt / DEFAULT_SIM_DT)))
        advance(s, tx, ty, MAX_SPEED_M_S * 1000, MAX_ACCEL_M_S2 * 1000,
                0.003, dt / n, n)

    print("── selftest 1: replay the sim's own output (must be ~0) " + "─" * 8)
    e, _ = report(rows, replay(rows, 0.003, MAX_SPEED_M_S * 1000,
                              MAX_ACCEL_M_S2 * 1000))
    ok1 = e < 0.05
    print(f"\n{'PASS' if ok1 else 'FAIL'}: self-replay peak {e:.4f} mm\n")

    # Detune ACCELERATION, not speed: at a 15 m/s cap the paddle never
    # gets near it on a 400 mm move, so a lower speed cap is a no-op and
    # would make this test pass vacuously. The accel cap does bind.
    print("── selftest 2: replay with a 20% low ACCEL cap (must be seen) "
          + "─" * 1)
    e2, _ = report(rows, replay(rows, 0.003, MAX_SPEED_M_S * 1000,
                               MAX_ACCEL_M_S2 * 800))
    ok2 = e2 > 1.0
    print(f"\n{'PASS' if ok2 else 'FAIL'}: detuned replay peak {e2:.3f} mm "
          f"(must exceed 1 mm or the harness is blind)")
    return 0 if (ok1 and ok2) else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="hardware log .jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ramp", type=float, default=0.003)
    ap.add_argument("--v-max", type=float, default=None,
                    help="mm/s; default: the sim's nominal")
    ap.add_argument("--a-max", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.log:
        ap.error("give a log, or --selftest")

    rows = load(args.log)
    v = args.v_max if args.v_max else MAX_SPEED_M_S * 1000
    a = args.a_max if args.a_max else MAX_ACCEL_M_S2 * 1000
    print(f"replaying {args.log} at v_max {v:.0f} a_max {a:.0f} "
          f"ramp {args.ramp * 1e3:.1f} ms\n")
    report(rows, replay(rows, args.ramp, v, a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
