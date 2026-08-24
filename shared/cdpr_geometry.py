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

# Walls MEASURED 2026-08-12 with calipers via the sticker markers; the old
# values assumed a 2020 x 1005 table with the grid centred on it, and it is
# not centred. These are the INSCRIBED rectangle — each wall's closest
# approach along its length — so anything derived from them is safe
# everywhere rather than only where it was measured.
RAIL_MIN_X = -3.8
RAIL_MAX_X = 2017.9
RAIL_MIN_Y = -19.0
RAIL_MAX_Y = 984.9

# Goal mouth, centred on each end rail. A table fact, not a sim
# parameter; TableConfig had 250 mm, which predates the table.
GOAL_WIDTH_MM = 380.0

# Puck MEASURED 2026-08-23: 81.4 mm diameter. The sim had 25 mm radius and
# demo_goalie 31.5 -- two different wrong answers, both too small.
PUCK_RADIUS_MM = 40.7

# ── Motor anchors ──
# All four trilaterated from caliper distances to three air holes each,
# rms 0.01-0.51 mm. See the header for the measurements, the symmetry
# checks, and what the camera was worth by comparison.
MOTOR_X = [1095.7, 2094.6, 2094.0, 1093.4]
MOTOR_Y = [1108.7, 1062.6, -97.7, -141.9]

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
ATTACH_R_MM = 31.5      # cable termination, CALIPERED
ARM_MARKER_R_MM = 26.5  # side markers — optical only, NOT the attachment
ATTACH_CHIRALITY = -1.0
MALLET_THETA_RAD = 2.3561945   # 135 deg — see the header for why not 0

# ── Workspace ──
# The middle half of the robot half — same centre, half the span per axis.
# Not the rails: cables pull only, so the paddle is holdable only inside the
# anchor hull, and tension grows without bound near its boundary. See the
# header for why the centreline itself is unreachable.
# Three bounds are wall - margin - mallet radius. WS_MIN_X cannot be: the
# human-end wall is ~1100 mm outside the anchor hull and unreachable at any
# torque, so it comes from the hull instead. See the header.
MALLET_RADIUS_MM = 50.4     # MEASURED: 100.8 mm diameter
WALL_MARGIN_MM = 50.0       # collision only; force closure binds on 3 sides
HULL_CLEARANCE_MM = 104.3

# Static HOLDING cost, not collision, is what actually bounds this rig.
# 4 cables / 3 DOF -> a 1-D internal-force null space n; every cable must
# stay taut, so max(n)/min(n) is the tension amplification, and it diverges
# at the hull. Parts of WIDE are outside force closure entirely. Threshold
# 7.8 is empirical: the old BOX measured 7.72 and never tripped. See header.
WS_FC_MIN_X, WS_FC_MAX_X = 1350.0, 1917.5
WS_FC_MIN_Y, WS_FC_MAX_Y = 142.9, 823.0
MAX_HOLD_AMPLIFICATION = 7.8

# WIDE — wall-derived. REFERENCE ONLY; not safe to make active.
WS_WIDE_MIN_X = max(MOTOR_X[0], MOTOR_X[3]) + HULL_CLEARANCE_MM
WS_WIDE_MAX_X = RAIL_MAX_X - WALL_MARGIN_MM - MALLET_RADIUS_MM
WS_WIDE_MIN_Y = RAIL_MIN_Y + WALL_MARGIN_MM + MALLET_RADIUS_MM
WS_WIDE_MAX_Y = RAIL_MAX_Y - WALL_MARGIN_MM - MALLET_RADIUS_MM

# Extra y trim, EMPIRICAL — overloads persisted near the y extremes, so both
# y limits come in by hand. Not derived from anything; see header.
WS_Y_TRIM_MM = 30.0

# SAFE — tighter of collision and force closure, per side, less the y trim.
WS_SAFE_MIN_X = max(WS_WIDE_MIN_X, WS_FC_MIN_X)
WS_SAFE_MAX_X = min(WS_WIDE_MAX_X, WS_FC_MAX_X)
WS_SAFE_MIN_Y = max(WS_WIDE_MIN_Y, WS_FC_MIN_Y) + WS_Y_TRIM_MM
WS_SAFE_MAX_Y = min(WS_WIDE_MAX_Y, WS_FC_MAX_Y) - WS_Y_TRIM_MM

# BOX — the conservative middle-half used through bring-up (0.250 m^2)
WS_BOX_MIN_X, WS_BOX_MAX_X = 1258.0, 1758.0
WS_BOX_MIN_Y, WS_BOX_MAX_Y = 233.0, 733.0

# ACTIVE — SAFE. WIDE went live 2026-08-23 and immediately tripped
# RMSOverloadShutdown just HOLDING the mallet near an edge: parts of it are
# outside force closure. See the header for the mechanism.
WS_MIN_X, WS_MAX_X = WS_SAFE_MIN_X, WS_SAFE_MAX_X
WS_MIN_Y, WS_MAX_Y = WS_SAFE_MIN_Y, WS_SAFE_MAX_Y

HOME_X = (WS_MIN_X + WS_MAX_X) / 2.0
HOME_Y = (WS_MIN_Y + WS_MAX_Y) / 2.0

WRAP_REF_ANGLE = [-0.988161, -2.362197, 2.360737, 0.985013]

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
