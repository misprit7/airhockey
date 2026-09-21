"""Experimental simulation command guard; never opens a hardware connection.

The existing firmware's containment backstop can zero substantial velocity at
a boundary, especially after an acceleration-cap decrease. Reject commands that
predict this impulse, including the ability to brake on the following tick.
This is a kinematic guard, not a motor-current/thermal guarantee.
"""

import ctypes
from pathlib import Path
import subprocess

import numpy as np

from airhockey.arrival import copy_cart
from airhockey.motion import DEFAULT_SIM_DT


_ROOT = Path(__file__).resolve().parents[2]
_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        path = _ROOT / "fw/host/build/libmotion_limits.so"
        sources = [
            _ROOT / "fw/host/motion_limits.cpp",
            _ROOT / "fw/include/motion_profile.h",
        ]
        if not path.exists() or any(
            p.stat().st_mtime > path.stat().st_mtime for p in sources
        ):
            subprocess.run(
                ["make", "-C", str(_ROOT / "fw/host"), "build/libmotion_limits.so"],
                check=True,
                capture_output=True,
            )
        _LIB = ctypes.CDLL(str(path))
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        _LIB.motion_limits_batch.restype = None
        _LIB.motion_limits_batch.argtypes = (
            [ctypes.c_int, ctypes.c_int, ctypes.c_float]
            + [f32] * 10
            + [ctypes.c_float] * 5
            + [f32, f32]
        )
    return _LIB


def predict(cart, target, cap, speed, duration, bounds, peak=None, margin=None):
    """Advance a COPY supplied by the caller; SI commands, cart in firmware mm."""
    n = len(cart)
    if peak is None:
        peak = np.zeros(n, np.float32)
    if margin is None:
        margin = np.full(n, 1e9, np.float32)
    # Match BatchEnv's 2.5 ms physics intervals, each split into round(dt /
    # DEFAULT_SIM_DT) firmware steps. Using exactly 0.2 ms disagrees slightly
    # at the boundary precisely where a prediction needs to be conservative.
    dt = 0.0025 / round(0.0025 / DEFAULT_SIM_DT)
    ticks = max(1, round(duration / dt))

    def a(x):
        return np.ascontiguousarray(np.broadcast_to(x, (n,)), dtype=np.float32)

    _library().motion_limits_batch(
        n,
        ticks,
        duration / ticks,
        cart.x,
        cart.y,
        cart.vx,
        cart.vy,
        cart.ax,
        cart.ay,
        a(target[:, 0] * 1000),
        a(target[:, 1] * 1000),
        a(np.asarray(speed) * 1000),
        a(np.asarray(cap) * 1000),
        0.003,
        *tuple(np.asarray(bounds) * 1000),
        peak,
        margin,
    )
    return peak


def guard_command(
    cart,
    target,
    cap,
    previous_target,
    previous_cap,
    *,
    bounds,
    max_accel,
    max_speed,
    delay=0.015,
    action_dt=0.02,
):
    """Choose a command whose execution and subsequent emergency braking fit.

    Commands are screened at the host firmware integration rate, so a hard
    boundary stop cannot hide inside an otherwise plausible 2.5 ms average.
    The unmodified request wins when safe; otherwise interpolate from braking.
    """
    state = copy_cart(cart)
    max_accel = np.broadcast_to(max_accel, (len(cart),))
    cap = np.broadcast_to(cap, (len(cart),))
    # Simulate the ACTUAL bounds. Moving the prediction's walls inward changes
    # its braking dynamics and would underestimate the real future velocity.
    bounds = np.asarray(bounds)
    prefix_margin = np.full(len(cart), 1e9, np.float32)
    prefix_peak = predict(
        state,
        previous_target,
        previous_cap,
        max_speed,
        delay,
        bounds,
        margin=prefix_margin,
    )
    low, high = np.asarray(bounds)[[0, 2]], np.asarray(bounds)[[1, 3]]
    current_xy = np.column_stack((cart.x, cart.y))
    initial_room = np.minimum(current_xy - low * 1000, high * 1000 - current_xy).min(
        axis=1
    )
    # Allow a cart already inside the conservative margin to escape. Requiring
    # 1 mm during its queued prefix would reject every command indefinitely.
    required_room = np.minimum(1.0, np.maximum(initial_room, 0)) - 0.05
    target = np.clip(target, low + 0.002, high - 0.002)
    brake_target = np.column_stack((state.x, state.y)) / 1000
    chosen, chosen_cap = brake_target.copy(), np.asarray(max_accel, dtype=float).copy()
    accepted = np.zeros(len(cart), bool)
    unresolved = np.ones(len(cart), bool)
    threshold = np.asarray(max_accel) * 1000 * 1.001
    for fraction in (1.0, 0.5, 0.25, 0.0):
        trial_target = brake_target + fraction * (target - brake_target)
        # Raise, never lower, braking authority as we approach the fallback.
        trial_cap = fraction * cap + (1 - fraction) * max_accel
        trial = copy_cart(state)
        peak = prefix_peak.copy()
        margin = prefix_margin.copy()
        predict(
            trial, trial_target, trial_cap, max_speed, action_dt, bounds, peak, margin
        )
        stop_at = np.column_stack((trial.x, trial.y)) / 1000
        predict(trial, stop_at, max_accel, max_speed, 0.35, bounds, peak, margin)
        safe = (
            (peak <= threshold)
            & (margin >= required_room)
            & (np.hypot(trial.vx, trial.vy) < 10)
        )
        use = unresolved & safe
        chosen[use], chosen_cap[use] = trial_target[use], trial_cap[use]
        if fraction == 1:
            accepted[:] = use
        unresolved &= ~safe
        if not unresolved.any():
            break
    return chosen, chosen_cap, ~accepted, unresolved
