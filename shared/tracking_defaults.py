"""Shared optical defaults for gameplay, the web preview and paddle measurement.

Change these here, not in individual consumers. Explicit CLI/constructor
overrides apply only to that process. Preview JPEG cadence is independent
of acquisition settings; calibration photography can use its own exposure.
"""

EXPOSURE_US = 300.0
GAIN_DB = 12.0
THRESHOLD = 150
FPS = 200.0
MIN_BLOB_AREA = 4
MAX_BLOB_AREA = 4000
MAX_BLOBS = 256


def add_tracking_arguments(parser):
    """Register camera options with the same defaults in every tracking CLI."""
    parser.add_argument('--fps', type=float, default=FPS)
    parser.add_argument('--exposure', type=float, default=EXPOSURE_US,
                        help=f'camera exposure in us (shared default: {EXPOSURE_US:g})')
    parser.add_argument('--gain', type=float, default=GAIN_DB,
                        help=f'camera gain in dB (shared default: {GAIN_DB:g})')
    parser.add_argument('--threshold', type=int, default=THRESHOLD,
                        help=f'blob brightness threshold (shared default: {THRESHOLD})')
