#!/usr/bin/env python3
"""Fit CDPR motor anchor positions from recorded calibration poses.

Reads the JSON written by sw/build/calibrate and solves a nonlinear least
squares problem for the 4 motor anchor positions, 4 per-motor length offsets
(encoder zeros are arbitrary), and one free mallet orientation per pose.

Forward model (flat spool, shaft vertical, winding radius r):
    attach = center_p + ATTACH_R * [cos(phi), sin(phi)],
    phi    = theta_p + chirality * 90deg * m
    d      = |anchor_m - attach|          (center-to-attach distance)
    d'     = sqrt(d^2 - r^2)              (free wire, tangent to the spool)
    u      = d' - s_m * r * psi           (wire the ENCODER sees: free length
                                           plus the wrap-angle term; psi is
                                           the tangent point's bearing around
                                           the spool, s_m the winding side)
    u = measured_length[p, m] + offset_m

measured_length = -retract_sign[m] * counts * mm_per_count (retracting motors
shorten the wire). Both attachment chiralities (whether motors 0-3 attach CW
or CCW around the mallet) are tried; the better fit wins. SIDES (which side
of each spool the wire leaves) is a hardware fact — check it at the table and
fix the constants below if the residuals come out large.

Usage:
    python bin/calibrate_fit.py path/to/calib_poses.json
    python bin/calibrate_fit.py --selftest
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

SPOOL_R_MM = geom.SPOOL_RADIUS_MM
SPOOL_CIRC_MM = 2.0 * np.pi * SPOOL_R_MM
ATTACH_R_MM = geom.ATTACH_R_MM
RETRACT_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])  # API count sign that retracts
SIDES = np.array(geom.WINDING_SIDE)              # winding side — still unverified
CHIRALITY_CONFIRMED = -1  # arms 0-3 run CLOCKWISE around the mallet (hardware)

# Initial anchor guess — pulled from shared/cdpr_geometry.py, which is the
# one Python copy of the header. Measured 2026-08-02 from retroreflectors
# on the spool axes, with the plane height fitted to caliper measurements
# against the air-hole grid; source of truth is
# vision/calib/motor_anchors.json.
# GRID frame: origin at the corner hole nearest the human's right corner
# (the old rail-corner origin is at roughly (-19, -33) in this frame).
#
# These are real measurements now, not placeholders, so this fit starts near
# the answer: treat a solution that lands far from here as suspect rather
# than as new information.
ANCHOR_GUESS = np.array(list(zip(geom.MOTOR_X, geom.MOTOR_Y)))

# Fixed per-motor reference directions for the wrap angle psi (avoids atan2
# branch cuts; the constant offset this introduces is absorbed by offset_m).
WS_CENTER = np.array([geom.HOME_X, geom.HOME_Y])  # robot-half centre
REF = WS_CENTER[None, :] - ANCHOR_GUESS
REF = REF / np.linalg.norm(REF, axis=1, keepdims=True)


def measured_lengths(counts, cpr):
    """Relative wire lengths (mm) from raw encoder counts, per pose x motor."""
    mm_per_count = SPOOL_CIRC_MM / cpr
    return -RETRACT_SIGN[None, :] * counts * mm_per_count[None, :]


def model_lengths(anchors, centers, thetas, chirality):
    """Encoder-equivalent wire length u per (pose, motor), up to a constant."""
    phi = thetas[:, None] + chirality * (np.pi / 2) * np.arange(4)[None, :]
    attach = centers[:, None, :] + ATTACH_R_MM * np.stack(
        [np.cos(phi), np.sin(phi)], axis=-1)              # (P, 4, 2)
    delta = attach - anchors[None, :, :]
    d = np.maximum(np.linalg.norm(delta, axis=-1), SPOOL_R_MM + 1e-6)
    dp = np.sqrt(d * d - SPOOL_R_MM ** 2)
    u_hat = delta / d[..., None]
    n_hat = np.stack([-u_hat[..., 1], u_hat[..., 0]], axis=-1)
    tdir = (SPOOL_R_MM / d)[..., None] * u_hat \
        + SIDES[None, :, None] * (dp / d)[..., None] * n_hat  # unit M->T
    cross = REF[None, :, 0] * tdir[..., 1] - REF[None, :, 1] * tdir[..., 0]
    dot = REF[None, :, 0] * tdir[..., 0] + REF[None, :, 1] * tdir[..., 1]
    psi = np.arctan2(cross, dot)
    return dp - SIDES[None, :] * SPOOL_R_MM * psi


def residuals(params, centers, lmeas, chirality):
    anchors = params[:8].reshape(4, 2)
    offsets = params[8:12]
    thetas = params[12:]
    model = model_lengths(anchors, centers, thetas, chirality)
    return (model - (lmeas + offsets[None, :])).ravel()


def pose_models(centers, chirality, grid):
    """Model lengths at guessed anchors for every pose x grid-theta: (P, G, 4)."""
    n_poses, n_grid = len(centers), len(grid)
    cen = np.repeat(centers, n_grid, axis=0)
    ths = np.tile(grid, n_poses)
    return model_lengths(ANCHOR_GUESS, cen, ths, chirality).reshape(
        n_poses, n_grid, 4)


def init_thetas(centers, lmeas, chirality):
    """Coarse thetas via pose-to-pose length DIFFERENCES (offsets cancel)."""
    grid = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    models = pose_models(centers, chirality, grid)     # (P, G, 4)
    n_poses = len(centers)
    best_ths, best_cost = None, np.inf
    for i0 in range(len(grid)):                        # anchor on pose 0's theta
        ths, total = [grid[i0]], 0.0
        for p in range(1, n_poses):
            dm = models[p] - models[0][i0][None, :]    # (G, 4) model diffs
            dmeas = lmeas[p] - lmeas[0]
            costs = ((dm - dmeas[None, :]) ** 2).sum(axis=1)
            j = int(costs.argmin())
            ths.append(grid[j])
            total += costs[j]
        if total < best_cost:
            best_ths, best_cost = ths, total
    return np.array(best_ths)


def make_params(centers, lmeas, chirality, thetas0):
    model = model_lengths(ANCHOR_GUESS, centers, thetas0, chirality)
    offsets0 = (model - lmeas).mean(axis=0)            # per-motor, over poses
    return np.concatenate([ANCHOR_GUESS.ravel(), offsets0, thetas0])


def fit(centers, lmeas, restarts=30, seed=1):
    """Multi-start LM over both chiralities; returns best and other-chirality."""
    rng = np.random.default_rng(seed)
    n_poses = len(centers)
    results = {}
    for chirality in (+1, -1):
        starts = [init_thetas(centers, lmeas, chirality)]
        starts += [rng.uniform(0, 2 * np.pi, n_poses) for _ in range(restarts)]
        best = None
        for thetas0 in starts:
            x0 = make_params(centers, lmeas, chirality, thetas0)
            sol = least_squares(residuals, x0, args=(centers, lmeas, chirality),
                                method="lm" if 4 * n_poses >= len(x0) else "trf")
            if best is None or sol.cost < best.cost:
                best = sol
        results[chirality] = best
    best_ch = min(results, key=lambda c: results[c].cost)
    return best_ch, results[best_ch], results[-best_ch]


def report(centers, lmeas):
    n_poses = len(centers)
    n_eq = 4 * n_poses
    n_unk = 12 + n_poses
    print(f"{n_poses} poses -> {n_eq} equations, {n_unk} unknowns "
          f"(8 anchors + 4 offsets + {n_poses} orientations)")
    print(f"model: flat spool r={SPOOL_R_MM}mm with tangency + wrap term, "
          f"SIDES={SIDES.astype(int).tolist()} (verify at hardware)")
    if n_eq < n_unk:
        sys.exit(f"ERROR: underdetermined by {n_unk - n_eq} — record more poses "
                 "(or re-record positions with the mallet rotated).")
    if n_eq == n_unk:
        print("WARNING: exactly determined — zero redundancy, errors are "
              "undetectable. More poses recommended.")
    if np.ptp(centers[:, 0]) < 100 or np.ptp(centers[:, 1]) < 100:
        print("WARNING: pose centers span <100mm in x or y — near-collinear "
              "poses condition the fit poorly.")

    best_ch, sol, other = fit(centers, lmeas)
    print(f"\nChirality {best_ch:+d} wins "
          f"(cost {sol.cost:.3f} vs {other.cost:.3f} for {-best_ch:+d})")
    if best_ch != CHIRALITY_CONFIRMED:
        print(f"WARNING: hardware says arms run clockwise "
              f"(chirality {CHIRALITY_CONFIRMED:+d}) — a {best_ch:+d} win "
              "suggests a mis-tied wire, wrong SIDES, or bad pose data.")

    anchors = sol.x[:8].reshape(4, 2)
    thetas = np.degrees(sol.x[12:]) % 360
    res = residuals(sol.x, centers, lmeas, best_ch).reshape(n_poses, 4)

    # 1-sigma anchor uncertainty from the Jacobian (needs redundancy).
    dof = sol.jac.shape[0] - sol.jac.shape[1]
    if dof > 0:
        sigma2 = 2 * sol.cost / dof
        cov = sigma2 * np.linalg.pinv(sol.jac.T @ sol.jac)
        std = np.sqrt(np.abs(np.diag(cov)))[:8].reshape(4, 2)
    else:
        std = np.full((4, 2), np.nan)

    print("\nFitted anchors (mm, +/- 1 sigma):")
    for m in range(4):
        d = anchors[m] - ANCHOR_GUESS[m]
        print(f"  Motor {m}: ({anchors[m, 0]:8.1f} +/-{std[m, 0]:5.1f}, "
              f"{anchors[m, 1]:8.1f} +/-{std[m, 1]:5.1f})   "
              f"[{d[0]:+6.1f}, {d[1]:+6.1f} vs guess]")
    print("Per-pose mallet orientation (deg): "
          + ", ".join(f"{t:.1f}" for t in thetas))
    print(f"Residuals: RMS {np.sqrt((res**2).mean()):.3f} mm, "
          f"max |{np.abs(res).max():.3f}| mm")
    if np.abs(res).max() > 2.0:
        print("WARNING: large residuals — check pose positions, slack strings, "
              "or the SIDES winding-direction constants.")

    print("\n// Paste into fw/include/cdpr_config.h:")
    print("constexpr float MOTOR_X[NUM_MOTORS] = {"
          + ", ".join(f"{anchors[m, 0]:.1f}f" for m in range(4)) + "};")
    print("constexpr float MOTOR_Y[NUM_MOTORS] = {"
          + ", ".join(f"{anchors[m, 1]:.1f}f" for m in range(4)) + "};")


def selftest():
    rng = np.random.default_rng(0)
    true_anchors = ANCHOR_GUESS + rng.uniform(-40, 40, (4, 2))
    true_offsets = rng.uniform(500, 2000, 4)
    chirality = -1
    centers = np.array([   # grid frame, mallet pressed against approximate rails
        [1950.0, 18.0],
        [1950.0, 18.0],                 # same corner, different orientation
        [1950.0, 921.8],
        [1950.0, 921.8],
        [1480.6, 18.0],
        [1580.6, 921.8],
    ])
    thetas = rng.uniform(0, 2 * np.pi, len(centers))
    u_true = model_lengths(true_anchors, centers, thetas, chirality)
    lmeas = u_true - true_offsets[None, :]
    lmeas += rng.normal(0, 0.05, lmeas.shape)  # ~encoder/seating noise

    best_ch, sol, _ = fit(centers, lmeas)
    anchors = sol.x[:8].reshape(4, 2)
    err = np.linalg.norm(anchors - true_anchors, axis=1)
    res_rms = np.sqrt((residuals(sol.x, centers, lmeas, best_ch) ** 2).mean())
    print(f"selftest: chirality {best_ch:+d} (true {chirality:+d}), "
          f"residual RMS {res_rms:.3f} mm, anchor errors (mm): "
          + ", ".join(f"{e:.2f}" for e in err))
    assert best_ch == chirality, "wrong chirality selected"
    assert res_rms < 0.15, f"fit did not reach noise floor: {res_rms:.3f}mm"
    assert err.max() < 2.5, f"anchor recovery error too large: {err.max():.2f}mm"
    print("selftest PASSED")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        selftest()
        return
    with open(sys.argv[1]) as f:
        data = json.load(f)
    centers = np.array([[p["x"], p["y"]] for p in data["poses"]])
    counts = np.array([p["counts"] for p in data["poses"]])
    cpr = np.array(data["cpr"], dtype=float)
    lmeas = measured_lengths(counts, cpr)
    report(centers, lmeas)


if __name__ == "__main__":
    main()
