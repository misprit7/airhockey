"""The firmware's motion profile, called directly rather than reimplemented.

The simulator advances its paddles through the SAME C code the Teensy runs
(fw/include/motion_profile.h), compiled for the host as a shared library. It
is not a port and not a mirror: there is one implementation of the control
law, so a policy trained here is trained against the controller that will
actually execute it.

That matters more than it sounds. The law is not a simple filter -- it has a
braking curve, a magnitude-limited acceleration demand proportional to
velocity error, a jerk slew with its own state, a parking rule, and a speed
backstop. The sim's previous `DelayedDynamics` was a first-order lag that
reproduced none of it, and carried the same bang-bang relay bug that was
removed from the firmware in 2026-08-12.

BUILD
    make -C fw/host

TICK RATE
    The firmware ticks at 20 us because it may emit at most one step per tick
    -- a step-rate constraint the simulator does not have. Simulating at 20 us
    would cost 250 substeps per 5 ms environment step for no benefit. The sim
    tick instead comes from the profile's own timescales (3 ms jerk ramp, 6 ms
    velocity loop); DEFAULT_SIM_DT below is the chosen value and
    ai/tests/test_motion_fidelity.py is what bounds the error it costs.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "fw" / "host" / "build" / "libmotion.so"

# The firmware's own tick. Reference only -- the sim does not use it except in
# the fidelity test, where it is the ground truth to compare against.
FIRMWARE_DT = 20e-6

# Simulator tick. MEASURED, not reasoned: test_motion_fidelity sweeps this
# against the 20 us reference over four representative moves.
#
# Reasoning alone would have picked 1 ms -- three samples across the 3 ms jerk
# ramp, which sounds adequate. It is not. Straight moves are fine there
# (0.005 mm) but a DIAGONAL diverges 1.32 mm, 3.5x the 0.377 mm motor step,
# because coarsening the tick blunts the acceleration VECTOR's turn and the
# cart takes a slightly different path rather than merely a slightly delayed
# one. Diagonals are the common case in air hockey.
#
# 0.2 ms holds every move under one motor step (worst 0.252 mm) at 25
# substeps per 5 ms environment step -- still 10x cheaper than simulating the
# firmware's own tick.
DEFAULT_SIM_DT = 2e-4


class _Lib:
    """Lazily loaded, and built on first use if it is missing.

    Built rather than merely errored on because the alternative is that
    `pytest ai` fails on a clean checkout with a message about a .so, which
    reads as a broken repo rather than a missing build step.
    """

    _handle = None

    @classmethod
    def get(cls):
        if cls._handle is not None:
            return cls._handle
        if not _LIB.exists():
            subprocess.run(["make", "-C", str(_ROOT / "fw" / "host")],
                           check=True, capture_output=True)
        lib = ctypes.CDLL(str(_LIB))

        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        lib.motion_advance_batch.restype = None
        lib.motion_advance_batch.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
            f32, f32, f32, f32, f32, f32,      # p, v, a  (in-place)
            f32, f32,                          # targets
            f32, f32,                          # per-cart caps
            ctypes.c_float, u8,
        ]
        lib.motion_advance_batch_bounded.restype = None
        lib.motion_advance_batch_bounded.argtypes = (
            lib.motion_advance_batch.argtypes + [ctypes.c_float] * 4)
        lib.motion_trace.restype = None
        # px py vx vy ax ay tx ty vMax aMax rampS = 11 floats after dt.
        lib.motion_trace.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ] + [ctypes.c_float] * 11 + [f32, f32, f32, f32]
        cls._handle = lib
        return lib


class CartState:
    """Position, velocity and acceleration for n carts, in millimetres.

    Acceleration is state, not a derived quantity: the profile slews it to
    bound jerk, so the same command produces different motion depending on
    what the acceleration was. Dropping it would make the environment
    non-Markov in a way nothing in the observation could reveal.
    """

    __slots__ = ("x", "y", "vx", "vy", "ax", "ay", "flags")

    def __init__(self, n: int):
        z = lambda: np.zeros(n, dtype=np.float32)  # noqa: E731
        self.x, self.y = z(), z()
        self.vx, self.vy = z(), z()
        self.ax, self.ay = z(), z()
        self.flags = np.zeros(n, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.x)

    def reset(self, x, y) -> None:
        self.x[:] = x
        self.y[:] = y
        self.vx[:] = self.vy[:] = 0.0
        self.ax[:] = self.ay[:] = 0.0
        self.flags[:] = 0


def advance(state: CartState, target_x, target_y, v_max, a_max,
            ramp_s: float, dt: float, substeps: int,
            bounds: tuple[float, float, float, float] | None = None) -> None:
    """Advance every cart `substeps` ticks of `dt` toward its target, in place.

    Caps are per-cart so domain randomisation can vary them across the batch.
    `bounds` = (x_min, x_max, y_min, y_max) in mm keeps the PATH inside a box
    the way the firmware does (motionProfileContain); without it the cart
    can swing outside the box while turning at speed, which the machine
    cannot do and the paddle would be driven into the rail.
    """
    n = len(state)
    tx = np.ascontiguousarray(target_x, dtype=np.float32)
    ty = np.ascontiguousarray(target_y, dtype=np.float32)
    vm = np.ascontiguousarray(np.broadcast_to(v_max, (n,)), dtype=np.float32)
    am = np.ascontiguousarray(np.broadcast_to(a_max, (n,)), dtype=np.float32)
    lib = _Lib.get()
    if bounds is None:
        lib.motion_advance_batch(
            n, int(substeps), ctypes.c_float(dt),
            state.x, state.y, state.vx, state.vy, state.ax, state.ay,
            tx, ty, vm, am, ctypes.c_float(ramp_s), state.flags)
    else:
        lib.motion_advance_batch_bounded(
            n, int(substeps), ctypes.c_float(dt),
            state.x, state.y, state.vx, state.vy, state.ax, state.ay,
            tx, ty, vm, am, ctypes.c_float(ramp_s), state.flags,
            *[ctypes.c_float(float(b)) for b in bounds])


def trace(ticks: int, every: int, dt: float, start, target,
          v_max: float, a_max: float, ramp_s: float):
    """One cart's trajectory, sampled every `every` ticks. For the fidelity test."""
    n_out = ticks // every
    out = [np.zeros(n_out, dtype=np.float32) for _ in range(4)]
    _Lib.get().motion_trace(
        int(ticks), int(every), ctypes.c_float(dt),
        *[ctypes.c_float(v) for v in
          (start[0], start[1], 0.0, 0.0, 0.0, 0.0,
           target[0], target[1], v_max, a_max, ramp_s)],
        *out)
    return out
