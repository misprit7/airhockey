"""NumPy mirror of the cable-length model in cdpr_geometry.h.

This lives in shared/ rather than in ai/bin/ because it is a physical fact
about the machine, not part of the RL project. It used to live inside
ai/bin/calibrate_fit.py, which meant shared/check_geometry.py -- the guard
whose whole job is to protect the canonical geometry -- had to
sys.path.insert its way into an RL training directory to do it. The
canonical module now depends on nothing above it, and both consumers
(calibrate_fit for solving, check_geometry for verifying) import from here.

It exists at all only because Python cannot include a C header. The C in
cdpr_geometry.h is the definition; this is the mirror, and
shared/check_geometry.py is what proves they still agree.

Parameterised rather than hardcoded because the fitter has to vary anchors,
offsets and per-pose orientations to solve for them. Callers wanting the
as-built machine pass the constants from cdpr_geometry.
"""

from __future__ import annotations

import numpy as np

import cdpr_geometry as geom

# Which way the drive's own count goes when a cable RETRACTS. A property of
# how each motor is mounted and wired, not of the cable model.
RETRACT_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])

# Arms 0-3 run CLOCKWISE around the mallet, confirmed on hardware.
CHIRALITY = -1


def wrap_reference(anchors: np.ndarray, home: np.ndarray) -> np.ndarray:
    """Unit vectors from each anchor toward `home`.

    The wrap angle psi is measured against a FIXED per-motor direction so
    that atan2's branch cut sits outside the workspace. The choice only adds
    a constant per-motor offset to the computed length, and every constant
    offset is absorbed by the per-motor offset the fitter solves for -- so
    this is a numerical convenience, not a physical claim.
    """
    ref = home[None, :] - anchors
    return ref / np.linalg.norm(ref, axis=1, keepdims=True)


def cable_lengths(anchors, centers, thetas, *, chirality=CHIRALITY,
                  spool_r=None, attach_r=None, sides=None, ref=None):
    """Encoder-equivalent cable length per (pose, motor), up to a constant.

    `centers` is (P, 2) paddle positions and `thetas` (P,) orientations, so
    the result is (P, 4). Mirrors cableLength() in cdpr_geometry.h: tangency
    (d' = sqrt(d^2 - r^2)) plus the wrap term (-s*r*psi).
    """
    spool_r = geom.SPOOL_RADIUS_MM if spool_r is None else spool_r
    attach_r = geom.ATTACH_R_MM if attach_r is None else attach_r
    sides = np.asarray(geom.WINDING_SIDE) if sides is None else np.asarray(sides)
    if ref is None:
        ref = wrap_reference(anchors, np.array([geom.HOME_X, geom.HOME_Y]))

    centers = np.atleast_2d(centers)
    thetas = np.atleast_1d(thetas)

    phi = thetas[:, None] + chirality * (np.pi / 2) * np.arange(4)[None, :]
    attach = centers[:, None, :] + attach_r * np.stack(
        [np.cos(phi), np.sin(phi)], axis=-1)               # (P, 4, 2)

    delta = attach - anchors[None, :, :]
    d = np.maximum(np.linalg.norm(delta, axis=-1), spool_r + 1e-6)
    dp = np.sqrt(d * d - spool_r ** 2)                     # free wire

    u_hat = delta / d[..., None]
    n_hat = np.stack([-u_hat[..., 1], u_hat[..., 0]], axis=-1)
    tdir = (spool_r / d)[..., None] * u_hat \
        + sides[None, :, None] * (dp / d)[..., None] * n_hat

    cross = ref[None, :, 0] * tdir[..., 1] - ref[None, :, 1] * tdir[..., 0]
    dot = ref[None, :, 0] * tdir[..., 0] + ref[None, :, 1] * tdir[..., 1]
    psi = np.arctan2(cross, dot)
    return dp - sides[None, :] * spool_r * psi


def measured_lengths(counts: np.ndarray, cpr: np.ndarray) -> np.ndarray:
    """Relative cable lengths (mm) from raw drive encoder counts."""
    mm_per_count = (2.0 * np.pi * geom.SPOOL_RADIUS_MM) / cpr
    return -RETRACT_SIGN[None, :] * counts * mm_per_count[None, :]
