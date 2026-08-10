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

Permanent markers: 0.5 in square retroreflectors on the playing surface
(z ~ tape thickness, treated as 0). Four corner markers, each centered 1.5
pitches diagonally inward from its extreme corner hole (i.e. centered in
the grid cell one square in from the corner), plus two centerline markers
centered in the OUTERMOST cell (0.5 pitch from the edge row — one square
closer to the rails than the corners). Only the corners are grid-truth;
the centerline pair's exact positions are measured by the extrinsics
bootstrap (vision/calib/markers.json) — values below are nominal.
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

# Marker centers (order is irrelevant to the solvers). First 4 (corners)
# are exact; the centerline pair are nominal — real positions come from the
# bootstrap measurement.
_EDGE = 0.5 * PITCH_MM  # centerline markers sit in the outermost cell

MARKERS_XY = [
    (_IN, _IN),                          # human end, near rail
    (GRID_X_MM - _IN, _IN),              # robot end, near rail
    (_IN, GRID_Y_MM - _IN),              # human end, far rail
    (GRID_X_MM - _IN, GRID_Y_MM - _IN),  # robot end, far rail
    (CENTERLINE_X, _EDGE),               # centerline, near rail (nominal)
    (CENTERLINE_X, GRID_Y_MM - _EDGE),   # centerline, far rail (nominal)
]
