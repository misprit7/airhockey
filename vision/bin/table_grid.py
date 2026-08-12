"""CNC air-hole grid — the metric ground truth AND the coordinate frame.

FRAME (grid frame, adopted 2026-07-26): the origin is the corner HOLE
nearest the human's right corner. The grid, not the rails, defines
coordinates — the rails are less straight than the CNC pattern. Axis
directions are unchanged from the old rail frame: +x toward the robot end,
+y from the human's right toward their left. The old rail-corner origin
sits at roughly (-19, -33) mm in this frame; rail positions are
approximate, hole positions are exact.

Holes: (i * 25.4, j * 25.4) mm, i in 0..79 (80 columns, i=79 at the robot
end), j in 0..38 (39 rows, j=38 at the far rail).

COUNTED ON THE TABLE 2026-08-10, and confirmed optically: re-solving the
4-corner pose under each hypothesis gives 0.276 px RMS for 80x39 against
0.728 for 78x38, with every neighbour between 2.3 and 10.6 px.

This file previously said 78x38 and claimed the same kind of empirical
support. That check was real but its candidate set was not: it varied ONE
count at a time from 78x38 and compared 78x38, 79x38 and 78x39. 80x39 is
two steps away in both axes and was never tried, so the scan returned the
best of three wrong answers and it looked convincing.

The consequences were large and took a long time to find, because nothing
downstream fails loudly. GRID_X_MM and GRID_Y_MM set where the corner
MARKERS are assumed to be, so a wrong count put two of the four
pose-fitting markers 50.8 mm out in x and two 25.4 mm out in y. Four points
fitting a 6-DOF pose absorb that into the pose rather than into the
residual, so the calibration reported sub-pixel agreement while placing the
mallet tens of millimetres from where it actually was. It also moved
CENTERLINE_X by 25.4 mm, which shifted the workspace and every anchor
derived through the camera.

If you change these counts, everything measured through the camera has to
be redone — extrinsics first, then any anchor that came from it.

Permanent markers: 7 mm circular stickers lying flat on the playing
surface, each flush against a wall (see _WALL_OFFSETS below). Four corner
markers fit the pose; two centre-stripe markers are held out as
validation.

Note what changed with the stickers, because it changes what the numbers
mean. The old mounted markers sat in known grid CELLS, so their positions
were CNC-truth. These are referenced to the walls instead, and the walls
are neither exactly straight nor squarely placed around the grid — the
measurements below show the grid sitting 3.6 mm off centre between the
side walls, and the near wall tilted 0.4 mm across the table. So the
corner positions are now measured data with real uncertainty in them,
where they used to be exact. The trade is deliberate: the mounts stood
proud of the surface and the puck hit them.
"""

PITCH_MM = 25.4
N_COLS = 80   # x direction, i in 0..79
N_ROWS = 39   # y direction, j in 0..38

GRID_X_MM = (N_COLS - 1) * PITCH_MM   # 2006.6 — robot-end corner hole x
GRID_Y_MM = (N_ROWS - 1) * PITCH_MM   # 965.2  — far-rail corner hole y

# Rails, approximate only (grid assumed centered on the 2020x1005 table):
RAIL_MIN_X = -(2020.0 - GRID_X_MM) / 2.0    # -6.7
RAIL_MAX_X = GRID_X_MM - RAIL_MIN_X         # 2013.3
RAIL_MIN_Y = -(1005.0 - GRID_Y_MM) / 2.0    # -19.9
RAIL_MAX_Y = GRID_Y_MM - RAIL_MIN_Y         # 985.1

CENTERLINE_X = (N_COLS - 1) / 2.0 * PITCH_MM  # 1003.3 — between cols 39, 40


def hole_xy(i, j):
    """Grid-frame (x, y) in mm of hole (i, j)."""
    return (i * PITCH_MM, j * PITCH_MM)


_IN = 1.5 * PITCH_MM  # 38.1 — corner-marker inset, both axes
_EDGE = 0.5 * PITCH_MM  # centerline markers sit in the outermost cell

# ── Legacy markers: 0.5 in retroreflective squares on 3D-printed mounts ────
#
# Superseded 2026-08-12 because they stand proud of the surface and the puck
# hits them. Kept because both sets are physically on the table during the
# changeover, so the two calibrations can be compared against each other.
# Delete once the mounts come off.
MOUNTED_MARKERS_XY = [
    (_IN, _IN),                          # human end, near rail
    (GRID_X_MM - _IN, _IN),              # robot end, near rail
    (_IN, GRID_Y_MM - _IN),              # human end, far rail
    (GRID_X_MM - _IN, GRID_Y_MM - _IN),  # robot end, far rail
    (CENTERLINE_X, _EDGE),               # centerline, near rail (nominal)
    (CENTERLINE_X, GRID_Y_MM - _EDGE),   # centerline, far rail (nominal)
]
MOUNTED_MARKER_Z_MM = 3.3   # sat on printed mounts above the playing surface

# ── Current markers: 7 mm circular stickers, flush against the walls ──────
#
# Flat, so nothing for the puck to hit. The cost is that their positions are
# no longer grid-truth: a hole is where the CNC put it, whereas these are
# referenced to the WALLS, which are neither exactly straight nor squarely
# placed around the grid. So the wall offsets below are measured data, not
# derived, and the marker centre is one sticker RADIUS in from the wall.
#
# Measured 2026-08-12 with calipers: distance from the nearest grid line to
# the wall, at each marker. The x column is absent for the centre pair —
# those sit on the painted stripe, not against a side wall.
STICKER_DIAMETER_MM = 7.0
_R = STICKER_DIAMETER_MM / 2.0

# name -> (grid x, grid y, x wall offset or None, y wall offset, x edge gap)
#
# The edge gap is the clearance between the wall and the nearest EDGE of the
# sticker, so flush is zero and the radius gets added in one place for
# everything. robot/far is the exception: a physical irregularity holds it
# ~4 mm off the wall in x.
#
# Stated as a gap rather than as a wall-to-centre distance because the two
# differ by exactly one radius, and getting that backwards is a silent 3.5 mm
# on a marker that FITS the pose. It was in here backwards once already.
_WALL_OFFSETS = [
    ("human/near",  0.0,      0.0,       4.7,  19.0, 0.0),
    ("robot/near",  GRID_X_MM, 0.0,      11.7, 19.4, 0.0),
    ("human/far",   0.0,      GRID_Y_MM,  3.8, 19.7, 0.0),
    ("robot/far",   GRID_X_MM, GRID_Y_MM, 11.3, 19.7, 4.0),
    ("stripe/near", CENTERLINE_X, 0.0,   None, 19.2, 0.0),
    ("stripe/far",  CENTERLINE_X, GRID_Y_MM, None, 19.7, 0.0),
]


def _sticker_xy(gx, gy, xo, yo, gap):
    """Marker centre from its grid line and the wall it rests against.

    The wall lies OUTWARD of the grid line by the measured offset. The
    sticker's near edge sits `gap` back from the wall and its centre a
    further radius, so the centre lands (offset - radius - gap) outward.
    Sign comes from which half of the table the marker is in — every one of
    these is against the wall it is nearest.
    """
    sx = -1.0 if gx < GRID_X_MM / 2 else +1.0
    sy = -1.0 if gy < GRID_Y_MM / 2 else +1.0
    x = gx if xo is None else gx + sx * (xo - _R - gap)
    return (round(x, 2), round(gy + sy * (yo - _R), 2))


STICKER_MARKERS_XY = [_sticker_xy(gx, gy, xo, yo, ins)
                      for _, gx, gy, xo, yo, ins in _WALL_OFFSETS]

# Names in MARKERS_XY order, for overlays and diagnostics. Worth having:
# "F3 is 2 mm out" says nothing about which corner to go and look at, and
# these markers are now hand-placed things that DO need looking at.
MARKER_NAMES = [name for name, *_ in _WALL_OFFSETS]

# Stickers lie on the playing surface; treat as zero like any tape.
STICKER_MARKER_Z_MM = 0.0

# ── The set in use ────────────────────────────────────────────────────────
# Order matters to the solvers: the first four are the pose-fitting corners,
# the last two are held out.
MARKERS_XY = STICKER_MARKERS_XY
MARKER_Z_MM = STICKER_MARKER_Z_MM
