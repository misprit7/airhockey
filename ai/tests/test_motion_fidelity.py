"""How much does simulating the motion profile at a coarse tick cost?

The firmware ticks at 20 us because it may emit at most one step per tick.
That is a step-rate constraint, not a physics timescale, and the simulator
does not have it -- simulating at 20 us would be 250 substeps per 5 ms
environment step for no benefit.

But "coarser is fine" is an assumption, and the profile has real dynamics at
3 ms (the jerk ramp) and 6 ms (the velocity loop time constant). Three
samples across a 3 ms ramp is not obviously enough. So this measures the
divergence instead of assuming it, and DEFAULT_SIM_DT is whatever that
measurement justifies.

The number that matters is position error against the 20 us reference over a
realistic move, in millimetres, compared against things that already limit
the machine: 0.377 mm of step quantisation, ~4 mm of mallet tracking error.
A sim tick contributing well under the quantisation floor is not the
bottleneck.
"""

from __future__ import annotations

import numpy as np
import pytest

from airhockey.motion import DEFAULT_SIM_DT, FIRMWARE_DT, CartState, advance

RAMP_S = 0.003
V_MAX = 8000.0
A_MAX = 24000.0

# Moves that exercise different parts of the law: one that never leaves the
# braking curve, one that saturates speed, one diagonal, one reversal.
MOVES = [
    ("short 25mm", (1500.0, 480.0), (1525.0, 480.0)),
    ("medium 200mm", (1500.0, 480.0), (1700.0, 480.0)),
    ("long 400mm", (1450.0, 300.0), (1850.0, 300.0)),
    ("diagonal", (1400.0, 250.0), (1850.0, 700.0)),
]

DURATION_S = 0.25


def _run(dt: float, start, target, duration=DURATION_S):
    """Advance one cart for `duration` at tick `dt`; return final state."""
    s = CartState(1)
    s.reset(np.float32(start[0]), np.float32(start[1]))
    tx = np.array([target[0]], dtype=np.float32)
    ty = np.array([target[1]], dtype=np.float32)
    n = int(round(duration / dt))
    advance(s, tx, ty, V_MAX, A_MAX, RAMP_S, dt, n)
    return float(s.x[0]), float(s.y[0]), float(s.vx[0]), float(s.vy[0])


def _err(dt, start, target):
    rx, ry, rvx, rvy = _run(FIRMWARE_DT, start, target)
    x, y, vx, vy = _run(dt, start, target)
    return np.hypot(x - rx, y - ry), np.hypot(vx - rvx, vy - rvy)


@pytest.mark.parametrize("name,start,target", MOVES)
def test_default_tick_is_accurate_enough(name, start, target):
    """The chosen tick must land well inside one motor step of the truth.

    One step is 0.377 mm of cable. A simulator that agrees with the firmware
    to better than that is not what limits sim fidelity -- the unmeasured
    puck constants and the 4 mm tracking error are, by an order of magnitude.
    """
    pos_err, _vel_err = _err(DEFAULT_SIM_DT, start, target)
    assert pos_err < 0.377, (
        f"{name}: {pos_err:.4f} mm at dt={DEFAULT_SIM_DT * 1e3:.1f} ms "
        f"exceeds one motor step")


def test_all_moves_settle_at_the_sim_tick():
    """Coarsening must not destabilise the loop.

    The profile hunted at every ramp longer than ~1 ms once before, when the
    acceleration demand was (vDes - v)/dt and therefore bang-bang. A too-large
    tick could reintroduce exactly that, so check every move actually arrives.
    """
    for name, start, target in MOVES:
        x, y, vx, vy = _run(DEFAULT_SIM_DT, start, target, duration=1.0)
        dist = np.hypot(x - target[0], y - target[1])
        speed = np.hypot(vx, vy)
        assert dist < 1.0 and speed < 5.0, (
            f"{name}: settled {dist:.3f} mm away at {speed:.1f} mm/s")


def test_error_shrinks_with_tick():
    """Divergence must be a discretisation error, not a modelling one.

    If halving the tick did not reduce the error, something other than the
    step size would be wrong and the whole approach would be unsound.
    """
    start, target = (1450.0, 300.0), (1850.0, 300.0)
    errs = [_err(dt, start, target)[0] for dt in (2e-3, 1e-3, 5e-4, 2e-4)]
    assert errs[-1] < errs[0], f"error did not shrink with tick: {errs}"


if __name__ == "__main__":
    print(f"reference tick {FIRMWARE_DT * 1e6:.0f} us, "
          f"{DURATION_S * 1000:.0f} ms of motion\n")
    ticks = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
    print(f"{'move':<14}" + "".join(f"{t * 1e3:>10.2f}ms" for t in ticks))
    print("-" * (14 + 12 * len(ticks)))
    for name, start, target in MOVES:
        row = "".join(f"{_err(t, start, target)[0]:>12.4f}" for t in ticks)
        print(f"{name:<14}{row}")
    print("\nposition error vs the 20 us reference, mm")
    print("one motor step = 0.377 mm; mallet tracking error ~4 mm")
    print(f"\nsubsteps per 5 ms env step: " +
          ", ".join(f"{t * 1e3:.2f}ms={int(0.005 / t)}" for t in ticks))
