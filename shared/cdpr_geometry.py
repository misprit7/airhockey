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
GRID_X_MM = 79.0 * GRID_PITCH_MM     # 2006.6
GRID_Y_MM = 38.0 * GRID_PITCH_MM     # 965.2
CENTERLINE_X = 39.5 * GRID_PITCH_MM  # 1003.3

RAIL_MIN_X = -6.7
RAIL_MAX_X = 2013.3
RAIL_MIN_Y = -19.9
RAIL_MAX_Y = 985.1

# ── Motor anchors ──
# ONLY M2 IS TRUSTWORTHY — trilaterated from caliper distances to three
# air holes, rms 0.07 mm. M0/M1/M3 predate the 78x38 -> 80x39 grid-count
# fix and were measured through the wrong camera pose. See the header.
MOTOR_X = [1095.6, 2094.8, 2094.0, 1091.7]
MOTOR_Y = [1107.2, 1066.0, -97.7, -140.6]

# ── Spool ──
# goBILDA 3400 Series hub-mount round-belt pulley, 96 mm PITCH DIAMETER,
# so r = 48 mm. The single largest scale factor in the machine — see the
# header for why the part name needs checking against a wound-turns
# measurement, and for the 96-is-a-diameter-not-a-radius trap.
SPOOL_RADIUS_MM = 48.0
SPOOL_CIRCUMFERENCE_MM = 2.0 * math.pi * SPOOL_RADIUS_MM
WINDING_SIDE = [-1.0, 1.0, -1.0, 1.0]   # UNVERIFIED — see the header
RETRACT_CW = [True, False, True, False]

# ── Paddle cable attachment ──
ATTACH_R_MM = 31.1
ATTACH_CHIRALITY = -1.0
MALLET_THETA_RAD = 2.3561945   # 135 deg — see the header for why not 0

# ── Workspace ──
# The middle half of the robot half — same centre, half the span per axis.
# Not the rails: cables pull only, so the paddle is holdable only inside the
# anchor hull, and tension grows without bound near its boundary. See the
# header for why the centreline itself is unreachable.
WS_MIN_X = 1258.0
WS_MAX_X = 1758.0
WS_MIN_Y = 233.0
WS_MAX_Y = 733.0

HOME_X = (WS_MIN_X + WS_MAX_X) / 2.0   # 1508.0
HOME_Y = (WS_MIN_Y + WS_MAX_Y) / 2.0   # 483.0

WRAP_REF_ANGLE = [-0.986946, -2.359443, 2.360737, 0.982165]

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
