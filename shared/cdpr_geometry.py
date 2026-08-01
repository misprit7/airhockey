"""Python mirror of shared/cdpr_geometry.h — the canonical CDPR geometry.

Python cannot include a C header, so these constants are duplicated by
necessity. This module is the ONE Python copy: every Python consumer
(ai/, vision/, calibrate_fit) imports from here rather than hardcoding
its own, and shared/check_geometry.py verifies this file against the
header numerically so the two cannot drift apart silently.

If you change a value in the header, change it here and run:
    python shared/check_geometry.py
"""

import math

NUM_MOTORS = 4

# ── Table geometry (grid frame; origin = corner hole nearest the human's
# right corner, +x toward the robot end, +y human's-right to left) ──
GRID_PITCH_MM = 25.4
GRID_X_MM = 77.0 * GRID_PITCH_MM     # 1955.8
GRID_Y_MM = 37.0 * GRID_PITCH_MM     # 939.8
CENTERLINE_X = 38.5 * GRID_PITCH_MM  # 977.9

RAIL_MIN_X = -32.1
RAIL_MAX_X = 1987.9
RAIL_MIN_Y = -32.6
RAIL_MAX_Y = 972.4

# ── Motor anchors, measured optically 2026-08-01 ──
MOTOR_X = [1067.1, 2030.6, 2031.6, 1064.1]
MOTOR_Y = [1073.3, 1035.6, -87.7, -129.8]

# ── Spool ──
SPOOL_RADIUS_MM = 35.0
SPOOL_CIRCUMFERENCE_MM = 2.0 * math.pi * SPOOL_RADIUS_MM
WINDING_SIDE = [-1.0, 1.0, -1.0, 1.0]   # UNVERIFIED — see the header
RETRACT_CW = [True, False, True, False]

# ── Paddle cable attachment ──
ATTACH_R_MM = 31.1
ATTACH_CHIRALITY = -1.0
MALLET_THETA_RAD = 2.3561945   # 135 deg — see the header for why not 0

# ── Workspace (robot half only) ──
EDGE_MARGIN_MM = 30.0
WS_MIN_X = CENTERLINE_X + EDGE_MARGIN_MM   # 1007.9
WS_MAX_X = RAIL_MAX_X - EDGE_MARGIN_MM     # 1957.9
WS_MIN_Y = RAIL_MIN_Y + EDGE_MARGIN_MM     # -2.6
WS_MAX_Y = RAIL_MAX_Y - EDGE_MARGIN_MM     # 942.4

HOME_X = (CENTERLINE_X + RAIL_MAX_X) / 2.0  # 1482.9
HOME_Y = (RAIL_MIN_Y + RAIL_MAX_Y) / 2.0    # 469.9

WRAP_REF_ANGLE = [-0.967427, -2.340029, 2.348150, 0.961179]

# ── Heights above the playing surface ──
MARKER_Z_MM = 3.3    # field markers, on 3D-printed mounts
MALLET_Z_MM = 67.0   # retroreflector on top of the mallet


def in_workspace(x, y):
    return WS_MIN_X <= x <= WS_MAX_X and WS_MIN_Y <= y <= WS_MAX_Y


def clamp_to_workspace(x, y):
    """Nearest point inside the workspace. Callers driving the machine from
    a UI should clamp rather than let a stray coordinate through — the
    firmware clamps too, but silently, which hides the mistake."""
    return (min(max(x, WS_MIN_X), WS_MAX_X),
            min(max(y, WS_MIN_Y), WS_MAX_Y))
