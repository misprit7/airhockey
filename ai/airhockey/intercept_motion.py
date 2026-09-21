"""Host-only variable-horizon forecasts through the unchanged firmware profile."""

import ctypes
from pathlib import Path
import subprocess
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "fw/host/build/libintercept_motion.so"
_HANDLE = None


def forecast(state, queued, command, steps):
    """Return [x,y,vx,vy,ax,ay] in mm after 15 ms queued + steps*1 ms.

    Queued/command rows are [target_x,target_y,acceleration_cap] in mm units.
    Uses the experimental 12 m/s speed ceiling, 3 ms acceleration ramp.
    This is only a predictor; execution still uses the environment's guard.
    """
    global _HANDLE
    state, queued, command = [
        np.ascontiguousarray(a, dtype=np.float32) for a in (state, queued, command)
    ]
    steps = np.asarray(steps)
    n = len(state)
    if (
        state.shape != (n, 6)
        or queued.shape != (n, 3)
        or command.shape != (n, 3)
        or steps.shape != (n,)
    ):
        raise ValueError("invalid forecast shapes")
    if not all(np.isfinite(a).all() for a in (state, queued, command, steps)):
        raise ValueError("forecast inputs must be finite")
    if np.any(steps < 0) or np.any(steps > 100000) or np.any(steps != np.floor(steps)):
        raise ValueError("forecast steps must be integers in [0,100000]")
    if np.any(queued[:, 2] <= 0) or np.any(command[:, 2] <= 0):
        raise ValueError("forecast acceleration caps must be positive")
    steps = np.ascontiguousarray(steps, dtype=np.int32)
    if _HANDLE is None:
        if not _LIB.exists():
            subprocess.run(
                ["make", "-C", str(_ROOT / "fw/host"), "build/libintercept_motion.so"],
                check=True,
                capture_output=True,
            )
        _HANDLE = ctypes.CDLL(str(_LIB))
        fn = _HANDLE.intercept_motion_batch
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            i32,
            f32,
            f32,
            f32,
            ctypes.c_float,
            ctypes.c_float,
            f32,
        ]
        fn.restype = None
    output = np.empty_like(state)
    _HANDLE.intercept_motion_batch(
        n, 15, 0.001, steps, state, queued, command, 12000, 0.003, output
    )
    return output
