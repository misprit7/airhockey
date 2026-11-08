"""Swapping the robot paddle is one word in shared/cdpr_geometry.py."""

import cdpr_geometry as geom
import track_mallet

from airhockey.physics import TableConfig


def test_both_paddles_keep_their_measured_values():
    assert geom.PADDLES["standard"] == geom.Paddle(0.170, 65.0, 33.0)
    assert geom.PADDLES["heavy"] == geom.Paddle(0.370, 68.52, 30.55)


def test_the_tracker_and_sim_follow_the_active_paddle():
    assert geom.PADDLE is geom.PADDLES[geom.ACTIVE_PADDLE]
    assert track_mallet.MALLET_Z_MM == geom.PADDLE.centre_marker_z_mm
    assert track_mallet.ARM_Z_MM == geom.PADDLE.arm_marker_z_mm
    assert TableConfig().paddle_mass == geom.PADDLE.mass_kg
