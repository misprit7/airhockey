"""CNC air-hole grid — the metric ground truth AND the coordinate frame.

FRAME (grid frame, adopted 2026-07-26): the origin is the corner HOLE
nearest the human's right corner. The grid, not the rails, defines
coordinates — the rails are less straight than the CNC pattern. Axis
directions are unchanged from the old rail frame: +x toward the robot end,
+y from the human's right toward their left. The old rail-corner origin
sits at roughly (-19, -33) mm in this frame; rail positions are
approximate, hole positions are exact.

Holes: (i * 25.4, j * 25.4) mm, i in 0..77 (78 columns, i=77 at the robot
end), j in 0..37 (38 rows, j=37 at the far rail).

(Columns were originally reported as 79, but the corner-marker calibration
residuals settled it empirically — scanning grid-count hypotheses against
4-corner PnP residuals: 78x38 gives 0.98 px RMS vs 3.6 px for 79x38 and
4.7 px for 78x39. 78 also matches the centerline lying BETWEEN two columns
and puts the measured stripe markers 1.8 mm from that centerline.)

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
N_COLS = 78   # x direction, i in 0..77
N_ROWS = 38   # y direction, j in 0..37

GRID_X_MM = (N_COLS - 1) * PITCH_MM   # 1955.8 — robot-end corner hole x
GRID_Y_MM = (N_ROWS - 1) * PITCH_MM   # 939.8  — far-rail corner hole y

# Rails, approximate only (grid assumed centered on the 2020x1005 table):
RAIL_MIN_X = -32.1
RAIL_MAX_X = 1987.9
RAIL_MIN_Y = -32.6
RAIL_MAX_Y = 972.4

CENTERLINE_X = 38.5 * PITCH_MM        # 977.9 — between columns 38 and 39


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
