"""The tracking test, driven against the firmware's own profile body.

No hardware, no camera: a FakeRig runs the real motion law in virtual time
and lets the test decide where the PADDLE is relative to the controller's
belief -- exactly on it, trailing it, or offset from it -- so the verdicts
can be checked against known ground truth.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pytest

from airhockey import follow_test as ft
from airhockey.hardware import TEENSY_COUNTS_PER_REV
from airhockey.motion import CartState, advance

import cdpr_geometry as geom  # noqa: E402  (path set by follow_test)

BOX = (geom.WS_MIN_X, geom.WS_MAX_X, geom.WS_MIN_Y, geom.WS_MAX_Y)
MM_PER_COUNT = geom.SPOOL_CIRCUMFERENCE_MM / TEENSY_COUNTS_PER_REV
ENC_RES = (6400, 800, 6400, 800)        # the two ClearPath models


class VirtualClock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class FakeRig:
    """Profile body (the controller's belief) + a paddle that may not be on it.

    follow_tau_s > 0  the paddle is a first-order lag of the body: a drive
                      that falls behind at speed and catches up at rest.
    offset_mm         the paddle sits this far from the body always: a
                      wrong cable model, or a slipped cable.
    enc_gain          encoder mm per step mm (1.0 = the step scale is right).
    """

    BODY_DT = 2e-3

    def __init__(self, clock: VirtualClock, speed=12000.0, accel=20000.0,
                 cam_delay_s=0.02, cam_noise_mm=0.3, follow_tau_s=0.0,
                 offset_mm=(0.0, 0.0), enc_gain=1.0,
                 enc_sign=(1, -1, 1, -1), seed=0):
        self.clock = clock
        self.speed, self.accel = speed, accel
        self.state = CartState(1)
        self.state.x[:] = geom.HOME_X
        self.state.y[:] = geom.HOME_Y
        self.target = (geom.HOME_X, geom.HOME_Y)
        self.t_body = clock()
        self.real = (geom.HOME_X, geom.HOME_Y)
        self.cam_delay = cam_delay_s
        self.cam_noise = cam_noise_mm
        self.tau = follow_tau_s
        self.offset = offset_mm
        self.enc_gain = enc_gain
        self.enc_sign = enc_sign
        self.hist: deque = deque([(self.t_body, *self._paddle())])
        self.rng = np.random.default_rng(seed)
        self.commands: list[tuple] = []
        self._home_cable = self._cable(geom.HOME_X, geom.HOME_Y)

    # ── the world ────────────────────────────────────────────────────

    def _paddle(self):
        return self.real[0] + self.offset[0], self.real[1] + self.offset[1]

    def _advance(self) -> None:
        now = self.clock()
        tx = np.array([self.target[0]], dtype=np.float32)
        ty = np.array([self.target[1]], dtype=np.float32)
        while self.t_body + self.BODY_DT <= now + 1e-9:
            advance(self.state, tx, ty, self.speed, self.accel, 0.003,
                    self.BODY_DT, 1, bounds=BOX)
            self.t_body += self.BODY_DT
            bx, by = float(self.state.x[0]), float(self.state.y[0])
            if self.tau > 0:
                k = self.BODY_DT / self.tau
                self.real = (self.real[0] + k * (bx - self.real[0]),
                             self.real[1] + k * (by - self.real[1]))
            else:
                self.real = (bx, by)
            self.hist.append((self.t_body, *self._paddle()))
            while self.hist[0][0] < now - 1.0:
                self.hist.popleft()

    @staticmethod
    def _cable(x, y):
        return [math.hypot(x - geom.MOTOR_X[m], y - geom.MOTOR_Y[m])
                for m in range(4)]

    def _counts(self):
        # Straight-line cable lengths: consistent per motor, which is all
        # the encoder fit needs. Sign: counts grow as cable pays out.
        L = self._cable(float(self.state.x[0]), float(self.state.y[0]))
        return [int(round((L[m] - self._home_cable[m]) / MM_PER_COUNT))
                for m in range(4)]

    # ── the CDPRClient interface ─────────────────────────────────────

    def command_position(self, x, y, v):
        self._advance()
        self.target = (float(x), float(y))
        self.commands.append((x, y, v))

    def get_status(self):
        self._advance()
        c = self._counts()
        return {"x": float(self.state.x[0]), "y": float(self.state.y[0]),
                "vx": float(self.state.vx[0]), "vy": float(self.state.vy[0]),
                "c0": c[0], "c1": c[1], "c2": c[2], "c3": c[3],
                "speed_limit": self.speed, "accel_limit": self.accel,
                "limit_flags": 0}

    def get_encoders(self):
        self._advance()
        c = self._counts()
        return {
            "posn": [self.enc_sign[m] * self.enc_gain * c[m]
                     * ENC_RES[m] / TEENSY_COUNTS_PER_REV for m in range(4)],
            "res": list(ENC_RES),
            "trq": [12.0, 15.0, 11.0, 14.0],
        }

    # ── the camera ───────────────────────────────────────────────────

    def camera(self):
        self._advance()
        t = self.clock()
        want = t - self.cam_delay
        # newest sample at or before the exposure time
        x = y = None
        for ts, hx, hy in reversed(self.hist):
            if ts <= want:
                x, y = hx, hy
                break
        if x is None:
            _, x, y = self.hist[0]
        n = self.rng.normal(0.0, self.cam_noise, 2)
        return (t, x + n[0], y + n[1])


def _run(rig: FakeRig, clock: VirtualClock, tmp_path, camera=True, **kw):
    # 40 Hz camera samples: the stamp only changes when a new frame exists,
    # which is what makes consecutive rows dedupe the way the real one does.
    last = {"t": None, "v": None}

    def cam():
        t = clock()
        if last["t"] is None or t - last["t"] >= 0.025:
            last["t"] = t
            last["v"] = rig.camera()
        return last["v"]

    test = ft.FollowTest(rig, camera=cam if camera else None,
                         log_dir=tmp_path, clock=clock, sleep=clock.sleep,
                         **kw)
    return test, test.run()


# ── the sequence ─────────────────────────────────────────────────────

def test_sequence_stays_inside_the_box_and_has_every_kind():
    segs = ft.sequence()
    for s in segs:
        assert BOX[0] + ft.MARGIN_MM - 1e-6 <= s.x <= BOX[1] - ft.MARGIN_MM + 1e-6
        assert BOX[2] + ft.MARGIN_MM - 1e-6 <= s.y <= BOX[3] - ft.MARGIN_MM + 1e-6
    kinds = {s.kind for s in segs}
    assert kinds == {"hold", "move", "flip"}
    assert segs[0].kind == "hold" and segs[-1].kind == "hold"
    # The corners actually reach the corners of the (shrunk) box.
    xs = {round(s.x, 3) for s in segs if s.name.startswith("corner")}
    assert {round(BOX[0] + ft.MARGIN_MM, 3), round(BOX[1] - ft.MARGIN_MM, 3)} == xs


def test_smaller_box_shortens_the_moves_rather_than_leaving_it():
    small = (1500.0, 1700.0, 400.0, 600.0)
    segs = ft.sequence(box=small, margin=10.0)
    for s in segs:
        assert 1510.0 - 1e-6 <= s.x <= 1690.0 + 1e-6
        assert 410.0 - 1e-6 <= s.y <= 590.0 + 1e-6


def test_travel_time_matches_the_profile_shape():
    # Short move never reaches the cap: triangular.
    d, v, a = 100.0, 12000.0, 20000.0
    assert ft.travel_time(d, v, a, ramp_s=0.0) == pytest.approx(2 * math.sqrt(d / a))
    # Long move at a low speed cap: cruise + two ramps.
    d, v, a = 5000.0, 500.0, 20000.0
    assert ft.travel_time(d, v, a, ramp_s=0.0) == pytest.approx(d / v + v / a)
    assert ft.travel_time(0.0, v, a) == 0.0


def test_gentle_caps_lengthen_the_estimate():
    segs = ft.sequence()
    fast = ft.duration_estimate(segs, 12000.0, 60000.0)
    slow = ft.duration_estimate(segs, 500.0, 2000.0)
    assert slow > fast > 5.0


# ── verdicts ─────────────────────────────────────────────────────────

def test_a_paddle_on_the_body_is_close(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock, cam_delay_s=0.02)
    test, s = _run(rig, clock, tmp_path)
    assert s["verdict"] == "close", ft.format_summary(s)
    assert s["gap_moving_p90_mm"] < 10.0
    assert s["gap_rest_p50_mm"] < 3.0
    # The camera's delay is fitted out, not mistaken for a drive lag.
    assert 10 <= s["lag_ms"] <= 35
    assert s["ctl_peak_speed_mm_s"] > 1500
    assert s["cam_peak_speed_mm_s"] > 1500
    # Encoders: the fit finds the sign of every motor and a gain of one.
    for e, sign in zip(s["encoders"], rig.enc_sign):
        assert e["gain"] == pytest.approx(sign, abs=ft.ENC_GAIN_TOL)
        assert e["rest_mm"] < ft.ENC_REST_MM
    assert s["notes"] == []
    assert s["caps"] == {"speed_mm_s": 12000.0, "accel_mm_s2": 20000.0}
    assert not s["stopped_early"]
    # Logs: samples and summary, where the UI's last line says.
    assert (tmp_path / s["log_csv"].split("/")[-1]).exists()
    assert (tmp_path / s["log_json"].split("/")[-1]).exists()
    with open(s["log_csv"]) as f:
        header = f.readline().strip().split(",")
    assert header == list(ft.ROW_FIELDS)
    # It parks at the centre when done.
    assert rig.commands[-1][:2] == (test.segments[-1].x, test.segments[-1].y)
    text = ft.format_summary(s)
    assert "CLOSE" in text and "log " in text


def test_a_drive_that_falls_behind_at_speed_is_lagging(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock, follow_tau_s=0.05)
    _, s = _run(rig, clock, tmp_path)
    assert s["verdict"] == "lagging", ft.format_summary(s)
    assert s["gap_moving_p90_mm"] > ft.CLOSE_MOVING_P90_MM
    assert s["gap_rest_p50_mm"] < ft.LOST_REST_P50_MM
    # The camera arrives late on the moves, and the summary says by how much.
    assert s["arrive_lag_p50_ms"] > 40


def test_a_paddle_offset_from_the_body_is_lost(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock, offset_mm=(60.0, 0.0))
    _, s = _run(rig, clock, tmp_path)
    assert s["verdict"] == "lost", ft.format_summary(s)
    assert s["gap_rest_p50_mm"] == pytest.approx(60.0, abs=3.0)


def test_a_wrong_step_scale_is_named_per_motor(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock, enc_gain=0.9)
    _, s = _run(rig, clock, tmp_path)
    assert s["verdict"] == "close"          # the camera agrees with the steps
    mism = [n for n in s["notes"] if "step scale mismatch" in n]
    assert len(mism) == 4
    assert all(abs(e["gain"]) == pytest.approx(0.9, abs=0.01) for e in s["encoders"])
    assert "! motor 0" in ft.format_summary(s)


def test_without_a_camera_the_drives_alone_are_scored(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock)
    _, s = _run(rig, clock, tmp_path, camera=False)
    assert s["verdict"].startswith("no camera")
    assert "held their steps" in s["verdict"]
    assert s["camera_hz"] is None
    assert "camera: none" in ft.format_summary(s)


def test_stop_parks_the_paddle_and_marks_the_run(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock)
    holder = {}

    def cam():
        if clock() - holder["t0"] > 4.0:
            holder["test"].stop()
        return rig.camera()

    test = ft.FollowTest(rig, camera=cam, log_dir=tmp_path,
                         clock=clock, sleep=clock.sleep)
    holder["test"] = test
    holder["t0"] = clock()
    s = test.run()
    assert s["stopped_early"]
    assert 3.5 < s["duration_s"] < 5.0
    assert rig.commands[-1][:2] == (test.segments[-1].x, test.segments[-1].y)
    assert "(stopped early)" in ft.format_summary(s)


def test_missing_caps_is_an_error_not_a_run(tmp_path):
    clock = VirtualClock()
    rig = FakeRig(clock, speed=0.0, accel=0.0)
    test = ft.FollowTest(rig, log_dir=tmp_path, clock=clock, sleep=clock.sleep)
    with pytest.raises(RuntimeError, match="caps"):
        test.run()
    assert rig.commands == []


def test_summarize_on_nothing_is_no_data():
    assert ft.summarize([], ft.sequence())["verdict"] == "no data"
    assert ft.format_summary({"verdict": "no data"}) == "no data"
