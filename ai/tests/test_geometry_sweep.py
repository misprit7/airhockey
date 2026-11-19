"""The calibration sweep's fit, on a synthetic machine whose motors and
spool are not where the model says, with cables that stretch and go slack."""
import numpy as np
import pytest

from airhockey import geometry_sweep as gs
from airhockey.follow_test import geom


def machine(n=36, seed=0, noise=0.05, shifts=None, radius=None):
    rng = np.random.default_rng(seed)
    xs = np.linspace(geom.WS_MIN_X + 45, geom.WS_MAX_X - 45, 6)
    ys = np.linspace(geom.WS_MIN_Y + 45, geom.WS_MAX_Y - 45, 6)
    camera = np.array([(x, y) for x in xs for y in ys])[:n]
    anchors0 = np.column_stack([geom.MOTOR_X, geom.MOTOR_Y]).astype(float)
    shifts = np.array([[6.0, -4.0], [-3.0, 5.0], [4.0, 2.0], [-5.0, -3.0]]) if shifts is None else shifts
    radius = geom.SPOOL_RADIUS_MM + 0.35 if radius is None else radius
    ref = gs.cable_model.wrap_reference(anchors0, np.array([geom.HOME_X, geom.HOME_Y]))
    true = gs.cable_model.cable_lengths(anchors0 + shifts, camera, np.full(len(camera), geom.MALLET_THETA_RAD),
                                        spool_r=radius, ref=ref)
    amps = rng.uniform(0.0, 4.0, (len(camera), 4))             # some cables slack
    stretch = np.array([1.0, 1.2, 0.9, 1.1])
    offsets = np.array([3.0, -2.0, 1.0, 0.5])
    taut = amps >= gs.SLACK_AMPS
    # A slack cable has paid out MORE than the straight line: it constrains nothing.
    nominal = (geom.SPOOL_RADIUS_MM / radius) * (true - offsets - stretch * amps + np.where(taut, 0, 7.0))
    signed = amps * np.array([-1.0, 1.0, -1.0, 1.0])
    return camera + rng.normal(0, noise, camera.shape), nominal, signed, shifts, radius, stretch


def test_sweep_covers_the_workspace_slowly_and_ends_at_the_centre():
    segs = gs.sequence()
    measure = [s for s in segs if s.name.endswith(": measure")]
    assert len(measure) == 36 and all(s.kind == "hold" for s in measure)
    xs, ys = {round(s.x) for s in measure}, {round(s.y) for s in measure}
    assert len(xs) == 6 and len(ys) == 6
    assert min(xs) < geom.WS_MIN_X + 60 and max(xs) > geom.WS_MAX_X - 60
    assert all(geom.WS_MIN_X < s.x < geom.WS_MAX_X and geom.WS_MIN_Y < s.y < geom.WS_MAX_Y for s in segs)
    assert (segs[-1].x, segs[-1].y) == (pytest.approx(geom.HOME_X), pytest.approx(geom.HOME_Y))
    # The goal-end column (highest x, the hot one) is visited last.
    assert round(measure[-1].x) == max(xs)


def test_fit_recovers_motor_positions_radius_and_stretch_and_recommends_them():
    camera, nominal, amps, shifts, radius, stretch = machine()
    f = gs.fit(camera, nominal, amps)
    got = np.array([[m["dx"], m["dy"]] for m in f["motors"]])
    assert np.abs(got - shifts).max() < 1.0
    assert abs(f["spool_radius"] - radius) < 0.05
    assert np.abs(np.array(f["stretch_mm_per_amp"]) - stretch).max() < 0.15
    assert f["held_out_fitted"] < 0.5 * f["held_out_now"] and f["recommend"]
    assert sum(f["slack"]) > 0          # slack cables were present and left out
    report = gs.format_report({"fit": f})
    assert report.startswith("RECOMMENDED") and "Nothing has been changed" in report


def test_a_machine_that_already_matches_the_model_is_left_alone():
    camera, nominal, amps, *_ = machine(shifts=np.zeros((4, 2)), radius=geom.SPOOL_RADIUS_MM, noise=0.3)
    f = gs.fit(camera, nominal, amps)
    assert not f["recommend"]
    assert gs.format_report({"fit": f}).startswith("NOT RECOMMENDED")


def test_too_few_points_is_an_error_not_a_guess():
    camera, nominal, amps, *_ = machine(n=10)
    with pytest.raises(ValueError, match="only 10 usable points"):
        gs.fit(camera, nominal, amps)


def test_collect_takes_the_end_of_each_hold():
    segs = gs.sequence()[:4]            # two points
    rows, t = [], 0.0
    for i, s in enumerate(segs):
        for k in range(100):
            still = s.kind == "hold" and k >= 30
            rows.append(dict(t=t, seg=i, ctl_x=s.x, ctl_y=s.y, cam_theta_deg=133.0 + i,
                             cam_x=s.x + (2.0 if still else 40.0), cam_y=s.y - 1.0))
            t += 0.01
    dt = np.arange(0, t, 0.01)
    drives = {m: (dt, np.where(dt % 2 >= 1.3, 1.5 + m, 9.0)) for m in range(4)}
    points = gs.collect(rows, segs, drives)
    assert points["camera"].shape == (2, 2)
    np.testing.assert_allclose(points["camera"][0], [segs[1].x + 2.0, segs[1].y - 1.0])
    np.testing.assert_allclose(points["controller"][1], [segs[3].x, segs[3].y])
    np.testing.assert_allclose(points["amps"][0], [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(points["theta_deg"], [134.0, 136.0])      # the mallet's measured angle
    # A drive whose current was not sampled leaves no point to fit.
    assert gs.collect(rows, segs, {0: drives[0]})["camera"].shape == (0, 2)
