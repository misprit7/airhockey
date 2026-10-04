"""Physical visibility bounds shared by the image and blob trackers.

These are rail bounds, not the controller workspace. Elevated markers need
their own projection planes; clipping all pixels at table height loses valid
paddles near the edges. Three millimetres allows the calibration residual.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom

CALIBRATION_TOLERANCE_MM = 3.0


def inside_playing_area(points, radius=0.0, goal_depth=0.0):
    """Whether a centre/marker fits inside the rails, including its radius.

    Only pucks can pass through goal mouths. goal_depth retains their final
    observations while entering a goal; it never opens the side railings.
    """
    points = np.asarray(points, float).reshape(-1, 2)
    x, y = points.T
    pad = radius - CALIBRATION_TOLERANCE_MM
    side = (y >= geom.RAIL_MIN_Y + pad) & (y <= geom.RAIL_MAX_Y - pad)
    ends = (x >= geom.RAIL_MIN_X + pad) & (x <= geom.RAIL_MAX_X - pad)
    if goal_depth > 0:
        centre_y = (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y) / 2
        mouth = np.abs(y - centre_y) <= geom.GOAL_WIDTH_MM / 2 - pad
        ends |= (mouth & (x >= geom.RAIL_MIN_X - goal_depth)
                 & (x <= geom.RAIL_MAX_X + goal_depth))
    return np.isfinite(points).all(axis=1) & side & ends


def marker_mask(px, K, dist, rvec, tvec, *, robot_only=False):
    """Reject pixels that cannot be an on-table marker at any valid height."""
    from camera import backproject_pixels
    from track_mallet import ARM_Z_MM, MALLET_Z_MM

    px = np.asarray(px, float).reshape(-1, 2)
    valid = np.zeros(len(px), bool)
    if not len(px):
        return valid
    planes = [(ARM_Z_MM, geom.MALLET_RADIUS_MM - geom.ARM_MARKER_R_MM, 0),
              (MALLET_Z_MM, geom.MALLET_RADIUS_MM, 0)]
    if not robot_only:
        planes += [(geom.MALLET_Z_MM, geom.MALLET_RADIUS_MM, 0),
                   (geom.PUCK_MARKER_Z_MM,
                    geom.PUCK_RADIUS_MM - geom.PUCK_MARKER_R_MM,
                    geom.PUCK_RADIUS_MM + geom.PUCK_MARKER_R_MM)]
    for height, inset, goal_depth in planes:
        world = backproject_pixels(px, K, dist, rvec, tvec, height)
        valid |= inside_playing_area(world, inset, goal_depth)
    return valid


def filter_candidates(cands, K, dist, rvec, tvec, *, robot_only=False):
    if not cands:
        return []
    valid = marker_mask([c[1] for c in cands], K, dist, rvec, tvec,
                        robot_only=robot_only)
    return [c for c, keep in zip(cands, valid) if keep]
