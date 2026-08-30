"""The deployment runner's safety layer, without a camera or a robot.

ai/bin/run_policy.py is the only thing between a policy and the machine, so
what is tested here is not that a good policy produces good motion -- it is
that a BAD one cannot produce a command that would hurt the rig. Every case
below is about the clamp, the cap committer, or the report contract the
policy reads.

Imported by path because ai/bin is not a package (executables live next to
the code they belong to, per the repo layout) and because running the
selftest through subprocess would test the CLI rather than the logic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "shared"))
import cdpr_geometry as geom  # noqa: E402


def _load():
    path = _ROOT / "ai" / "bin" / "run_policy.py"
    spec = importlib.util.spec_from_file_location("run_policy", path)
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: @dataclass resolves annotations by looking the
    # defining module up in sys.modules, and finds None if it is not there.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load()


# ── The end-to-end selftest ─────────────────────────────────────────────


def test_selftest_passes():
    """The whole report -> policy -> clamp -> command chain, every policy."""
    stats = rp.selftest(verbose=False)
    assert {"builtin:hold", "builtin:tracky", "hostile"} <= set(stats)
    # And every real bot, which is the path that actually ships.
    assert {f"heuristic:{n}" for n in rp.list_heuristics()} <= set(stats)
    # "watchdog" is a scenario, not a policy, and carries different keys.
    for label, s in ((k, v) for k, v in stats.items() if k != "watchdog"):
        assert s["commands"] > 100, f"{label}: only {s['commands']} commands"
        # LIMITS must not track the command rate; that is the point of the
        # committer. Two decades of margin either way would hide a failure.
        assert s["limits"] < s["commands"] / 10, \
            f"{label}: {s['limits']} LIMITS against {s['commands']} commands"
    assert stats["hostile"]["clamped"] > 0
    assert stats["builtin:hold"]["clamped"] == 0, \
        "a bot sitting at HOME should never need clamping"


# ── Clamp ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("x,y", [
    (geom.WS_MAX_X + 500.0, geom.HOME_Y),
    (geom.WS_MIN_X - 500.0, geom.HOME_Y),
    (geom.HOME_X, geom.WS_MAX_Y + 500.0),
    (geom.HOME_X, geom.WS_MIN_Y - 500.0),
    (1e9, -1e9),
])
def test_clamp_pulls_target_into_workspace(x, y):
    caps = rp.Caps()
    a, flags = rp.clamp_action(rp.Action(x, y, 1000.0, 5000.0), caps)
    assert geom.in_workspace(a.x_mm, a.y_mm)
    assert "workspace" in flags


def test_clamp_leaves_a_legal_target_alone():
    caps = rp.Caps()
    a, flags = rp.clamp_action(
        rp.Action(geom.HOME_X, geom.HOME_Y, 1000.0, 5000.0), caps)
    assert (a.x_mm, a.y_mm) == (geom.HOME_X, geom.HOME_Y)
    assert flags == []


def test_nan_target_is_rejected_not_clamped():
    """min/max propagate NaN, so this is the one input a clamp cannot bound.

    It must fall back to the previous target rather than reaching the wire,
    where the firmware's own comparisons against NaN are all false and would
    let it straight through.
    """
    caps = rp.Caps()
    prev = (geom.HOME_X + 10.0, geom.HOME_Y - 10.0)
    a, flags = rp.clamp_action(
        rp.Action(float("nan"), float("inf"), 1000.0, 5000.0), caps, prev)
    assert (a.x_mm, a.y_mm) == prev
    assert "nonfinite-target" in flags
    assert math.isfinite(a.x_mm) and math.isfinite(a.y_mm)


def test_nan_target_with_no_history_falls_back_to_home():
    caps = rp.Caps()
    a, _ = rp.clamp_action(rp.Action(float("nan"), 0.0, 1e3, 1e3), caps, None)
    assert (a.x_mm, a.y_mm) == (geom.HOME_X, geom.HOME_Y)


@pytest.mark.parametrize("speed,accel", [
    (1e9, 1e9), (-1.0, -1.0), (0.0, 0.0),
    (float("nan"), float("nan")), (float("inf"), float("-inf")),
])
def test_caps_are_bounded_whatever_the_policy_asks(speed, accel):
    caps = rp.Caps()
    a, _ = rp.clamp_action(
        rp.Action(geom.HOME_X, geom.HOME_Y, speed, accel), caps)
    assert caps.speed_min <= a.speed_mm_s <= caps.speed_max
    assert caps.accel_min <= a.accel_mm_s2 <= caps.accel_max


def test_caps_ceiling_is_the_one_the_cli_passed():
    """--speed/--accel lower the ceiling; nothing may exceed them."""
    caps = rp.Caps(speed_max=1200.0, accel_max=3000.0)
    a, flags = rp.clamp_action(
        rp.Action(geom.HOME_X, geom.HOME_Y, 8000.0, 24000.0), caps)
    assert a.speed_mm_s == 1200.0 and a.accel_mm_s2 == 3000.0
    assert "speed" in flags and "accel" in flags


# ── Report contract ─────────────────────────────────────────────────────


def test_history_is_newest_first_and_windowed():
    r = rp.ReportBuilder(history_s=0.100)                # raw is default
    for k in range(100):                       # 0.5 s at 200 Hz
        r.add_puck(k * 0.005, float(k), 2.0 * k)
    hist = r.observation(0.495)[rp.OBS_PUCK]
    ts = [s[2] for s in hist]
    assert ts == sorted(ts, reverse=True)
    assert hist[0][0] == 99.0, "hist[0] must be the newest sample"
    assert ts[0] - ts[-1] <= 0.100 + 1e-9


def test_the_whole_ring_is_handed_over_undecimated():
    """Every fix in the window, at the camera's full rate.

    Subsampling to the simulator's lag set was tried and removed: denser is
    strictly more information, and the decimated shape could not localise a
    real bounce that fell between two lags. See HISTORY_S in run_policy.py.
    """
    r = rp.ReportBuilder()
    for k in range(40):
        r.add_puck(k * 0.005, float(k), 0.0)
    hist = r.observation(0.195)[rp.OBS_PUCK]
    assert len(hist) == 40
    assert (hist[0][0], hist[0][2]) == (39.0, pytest.approx(0.195)), \
        "hist[0] must be the newest real fix"


def test_timestamps_strictly_decrease_across_a_dropout():
    """estimate_velocity divides by segment durations; no zero-length ones,
    and no reordering when the tracker drops frames in the middle."""
    r = rp.ReportBuilder()
    for k in range(60):
        t = k * 0.005
        if 0.10 <= t < 0.16:                   # 60 ms hole
            continue
        r.add_puck(t, float(k), 0.0)
    ts = [s[2] for s in r.observation(0.295)[rp.OBS_PUCK]]
    assert len(ts) == len(set(ts))
    assert all(a > b for a, b in zip(ts, ts[1:])), ts


def test_history_reaches_the_bots_intact():
    from airhockey.heuristics import TrackerReport

    r = rp.ReportBuilder()
    for k in range(60):
        r.add_puck(k * 0.005, 1400.0 - 5.0 * k, 500.0 + 2.0 * k)
    r.add_mallet(0.295, geom.HOME_X, geom.HOME_Y)
    rep = TrackerReport.coerce(r.observation(0.295))
    # The 200 ms window is inclusive at both ends, so t=0.095..0.295 at
    # 5 ms spacing is 41 samples -- the whole ring, nothing subsampled.
    assert len(rep.puck) == 41
    assert rep.puck[0].t_s > rep.puck[-1].t_s


def test_history_expires_on_read_not_only_on_write():
    """A puck that stops being seen must not leave a full-looking history.

    Expiring only when a new sample arrives would keep the last 40 samples
    in the deque for ever, so TrackerReport's "empty history means not
    visible" contract would silently stop holding and a bot would fit a
    velocity across a second-old track as if it were current.
    """
    r = rp.ReportBuilder(history_s=0.200)                # raw is default
    for k in range(40):
        r.add_puck(k * 0.005, float(k), 0.0)
    assert len(r.observation(0.195)[rp.OBS_PUCK]) == 40
    # No further samples; just time passing.
    assert len(r.observation(0.300)[rp.OBS_PUCK]) < 40
    assert r.observation(1.000)[rp.OBS_PUCK] == [], \
        "a long-gone puck must report an EMPTY history, not a stale one"
    # The staleness clock is unaffected — it still says when it was last seen.
    assert r.staleness(1.000)["puck"] == pytest.approx(1.000 - 0.195)


def test_coasted_frames_do_not_enter_the_history():
    """A dropout must leave a GAP, not a synthetic straight line.

    The runner only calls add_puck on a real fix; this pins the consequence
    that makes it matter -- the history's span reflects real observations,
    so a bot fitting a velocity is fitting measurements rather than the
    tracker's own extrapolation.
    """
    r = rp.ReportBuilder(history_s=1.0)                  # raw is default
    for k in range(10):
        r.add_puck(k * 0.005, float(k), 0.0)
    # Last real fix was at k=9, i.e. t=0.045. Now 21 frames pass with no
    # fix and nothing is added.
    t_end = 30 * 0.005
    hist = r.observation(t_end)[rp.OBS_PUCK]
    assert len(hist) == 10
    assert r.staleness(t_end)["puck"] == pytest.approx(t_end - 0.045)


def test_controller_position_beats_the_camera_for_the_own_mallet():
    """The sim hands a policy its own paddle fresh; the table must too.

    The controller's POS has no camera latency and never drops out, and it
    is a POSITION rather than the commanded setpoint.
    """
    r = rp.ReportBuilder()
    r.add_mallet(1.0, 1500.0, 400.0)                # camera
    r.set_controller_mallet(1.0, 1502.0, 401.0)     # controller
    assert r.observation(1.0)[rp.OBS_MALLET] == (1502.0, 401.0)


def test_camera_covers_for_the_controller_when_it_is_stale():
    r = rp.ReportBuilder()
    r.set_controller_mallet(0.0, 1502.0, 401.0)
    r.add_mallet(1.0, 1500.0, 400.0)
    assert r.observation(1.0)[rp.OBS_MALLET] == (1500.0, 400.0)


def test_mallet_disagreement_is_the_only_check_on_the_cable_model():
    """Controller position comes from step counts through the cable model;
    the camera's is measured. A steady gap is the model being wrong."""
    r = rp.ReportBuilder()
    assert r.mallet_disagreement(1.0) is None, "nothing seen yet"
    r.add_mallet(1.0, 1500.0, 400.0)
    assert r.mallet_disagreement(1.0) is None, "only one source"
    r.set_controller_mallet(1.0, 1530.0, 440.0)
    assert r.mallet_disagreement(1.0) == pytest.approx(50.0)
    # Either source going stale withdraws the claim.
    assert r.mallet_disagreement(1.0 + rp.STALE_S + 0.01) is None


def test_stale_own_mallet_falls_back_to_the_last_commanded_target():
    """The robot's mallet goes only where this process sent it.

    So when the camera loses it there is a better answer than "unknown", and
    heuristics.TrackerReport requires a tuple regardless.
    """
    r = rp.ReportBuilder()
    r.add_mallet(0.0, 1500.0, 400.0)
    assert r.observation(0.05)[rp.OBS_MALLET] == (1500.0, 400.0)
    stale = r.observation(0.05 + rp.STALE_S, mallet_fallback=(1600.0, 500.0))
    assert stale[rp.OBS_MALLET] == (1600.0, 500.0)


def test_stale_opponent_goes_none_and_gets_no_fallback():
    """Nothing here controls the human's mallet, so last-known is a guess."""
    r = rp.ReportBuilder()
    r.add_opponent(0.0, 500.0, 600.0)
    assert r.observation(0.05)[rp.OBS_OPPONENT] == (500.0, 600.0)
    assert r.observation(0.05 + rp.STALE_S,
                         mallet_fallback=(1.0, 2.0))[rp.OBS_OPPONENT] is None


def test_never_seen_mallet_is_home_and_infinitely_stale():
    r = rp.ReportBuilder()
    obs = r.observation(1.0)
    assert obs[rp.OBS_MALLET] == (geom.HOME_X, geom.HOME_Y)
    assert obs[rp.OBS_OPPONENT] is None
    assert obs[rp.OBS_PUCK] == []
    assert math.isinf(r.staleness(1.0)["opponent"])


def test_observation_carries_the_clock():
    """A bot's timers must keep running while the puck is invisible."""
    r = rp.ReportBuilder()
    assert r.observation(12.5)[rp.OBS_TIME] == 12.5


def test_observation_is_what_trackerreport_accepts():
    """The dict form and heuristics.TrackerReport.coerce must agree.

    These are written in two files by two people; nothing but this test says
    the keys line up, and the failure mode on the table is a KeyError at
    100 Hz with the drives live.
    """
    from airhockey.heuristics import TrackerReport

    r = rp.ReportBuilder()
    r.add_puck(0.10, 1400.0, 500.0)
    r.add_puck(0.11, 1420.0, 505.0)
    r.add_mallet(0.11, 1600.0, 400.0)
    r.add_opponent(0.11, 500.0, 600.0)
    rep = TrackerReport.coerce(r.observation(0.11))
    assert rep.puck[0].x_mm == 1420.0, "newest sample must be first"
    assert rep.puck[0].t_s == 0.11
    assert rep.mallet == (1600.0, 400.0)
    assert rep.opponent == (500.0, 600.0)
    assert rep.t_s == 0.11


# ── Cap committer ───────────────────────────────────────────────────────


def test_first_commit_always_goes_out():
    """The Teensy holds whatever the last session left; do not inherit it."""
    c = rp.CapCommitter(client=None)
    assert c.maybe_commit(0.0, 4000.0, 12000.0) is True
    assert (c.speed, c.accel) == (4000.0, 12000.0)


def test_commit_is_rate_limited_even_when_the_caps_swing():
    c = rp.CapCommitter(client=None, min_interval_s=0.2)
    c.maybe_commit(0.0, 1000.0, 4000.0)
    # 100 Hz of wildly different requests inside one interval: none go out.
    for k in range(1, 20):
        assert c.maybe_commit(k * 0.01, 8000.0, 24000.0) is False
    assert c.n_commits == 1
    assert c.maybe_commit(0.25, 8000.0, 24000.0) is True


def test_small_changes_are_suppressed_even_when_due():
    c = rp.CapCommitter(client=None, min_interval_s=0.2)
    c.maybe_commit(0.0, 4000.0, 12000.0)
    # Under both the relative and the absolute threshold.
    assert c.maybe_commit(10.0, 4100.0, 12200.0) is False
    # Over the relative threshold on speed alone.
    assert c.maybe_commit(20.0, 4800.0, 12200.0) is True


def test_committer_pushes_limits_to_the_client():
    calls = []

    class FakeClient:
        def set_limits(self, s, a):
            calls.append((s, a))

    c = rp.CapCommitter(client=FakeClient())
    c.maybe_commit(0.0, 3000.0, 9000.0)
    c.maybe_commit(0.01, 8000.0, 24000.0)      # too soon
    c.maybe_commit(1.0, 8000.0, 24000.0)
    assert calls == [(3000.0, 9000.0), (8000.0, 24000.0)]


# ── Policy loading and the plan step ────────────────────────────────────


def test_builtin_policies_load_and_return_four_numbers():
    caps = rp.Caps()
    for name in rp.BUILTIN_BOTS:
        pol = rp.load_policy(f"builtin:{name}", caps)
        out = pol({rp.OBS_PUCK: [], rp.OBS_MALLET: None,
                   rp.OBS_OPPONENT: None})
        assert len(out) == 4 and all(math.isfinite(v) for v in out)


def test_every_heuristic_bot_loads_and_returns_a_command():
    from airhockey.heuristics import Command

    caps = rp.Caps()
    names = rp.list_heuristics()
    assert names, "heuristics.BOTS is empty"
    for name in names:
        bot = rp.load_policy(f"heuristic:{name}", caps)
        r = rp.ReportBuilder()
        r.add_puck(0.10, 1400.0, 500.0)
        r.add_puck(0.11, 1420.0, 505.0)
        r.add_mallet(0.11, geom.HOME_X, geom.HOME_Y)
        out = bot(r.observation(0.11))
        assert isinstance(out, Command), f"{name} returned {out!r}"


def test_unknown_heuristic_name_is_rejected():
    with pytest.raises(SystemExit, match="unknown bot"):
        rp.load_policy("heuristic:nosuchbot", rp.Caps())


def test_bot_config_carries_the_cap_ceiling_down_to_the_bot():
    """--speed/--accel must reach the bot's PLANNER, not just the clamp.

    A bot that plans a 12000 mm/s strike and has it cut to 1200 afterwards
    arrives late and its own timing model cannot tell it why.
    """
    cfg = rp._bot_config(rp.Caps(speed_max=1200.0, accel_max=3000.0))
    for field in ("max_speed_mm_s", "idle_speed_mm_s", "strike_speed_mm_s"):
        assert getattr(cfg, field) <= 1200.0, field
    for field in ("max_accel_mm_s2", "idle_accel_mm_s2", "strike_accel_mm_s2"):
        assert getattr(cfg, field) <= 3000.0, field
    # Geometry must be left alone -- it comes from shared/cdpr_geometry.py.
    assert cfg.ws_min_x == geom.WS_MIN_X and cfg.ws_max_x == geom.WS_MAX_X


def test_a_real_bot_never_asks_for_more_than_the_ceiling():
    """With the ceiling pushed down, nothing should need clamping at all."""
    caps = rp.Caps(speed_max=1500.0, accel_max=4000.0)
    for name in rp.list_heuristics():
        bot = rp.load_policy(f"heuristic:{name}", caps)
        r = rp.ReportBuilder()
        for k in range(20):
            r.add_puck(k * 0.005, 1400.0 - 8.0 * k, 500.0 + 3.0 * k)
        r.add_mallet(0.1, geom.HOME_X, geom.HOME_Y)
        _a, flags = rp.plan(bot, r, 0.1, caps, None)
        assert "speed" not in flags and "accel" not in flags, \
            f"{name} exceeded the ceiling it was configured with: {flags}"


def test_plan_unwraps_a_command_object():
    from airhockey.heuristics import Command

    r = rp.ReportBuilder()
    a, _ = rp.plan(lambda obs: Command(geom.HOME_X, geom.HOME_Y, 900.0,
                                       3000.0),
                   r, 0.0, rp.Caps(), None)
    assert (a.x_mm, a.y_mm, a.speed_mm_s, a.accel_mm_s2) == \
        (geom.HOME_X, geom.HOME_Y, 900.0, 3000.0)


def test_sac_policy_is_an_explicit_stub():
    with pytest.raises(NotImplementedError, match="sac:"):
        rp.load_policy("sac:some_run", rp.Caps())


def test_unknown_policy_kind_is_rejected():
    with pytest.raises(SystemExit):
        rp.load_policy("magic:thing", rp.Caps())


def test_plan_rejects_a_policy_that_returns_the_wrong_shape():
    r = rp.ReportBuilder()
    with pytest.raises(TypeError, match="4-tuple"):
        rp.plan(lambda obs: (1.0, 2.0), r, 0.0, rp.Caps(), None)


def test_plan_output_is_always_commandable():
    """Whatever a policy returns, plan() hands back something safe to send."""
    r = rp.ReportBuilder()
    r.add_puck(0.0, 1500.0, 400.0)
    caps = rp.Caps()
    for out in [(1e9, 1e9, 1e9, 1e9),
                (float("nan"), float("nan"), 0.0, -5.0),
                (geom.HOME_X, geom.HOME_Y, 500.0, 2000.0)]:
        a, _ = rp.plan(lambda obs, o=out: o, r, 0.0, caps, None)
        assert geom.in_workspace(a.x_mm, a.y_mm)
        assert caps.speed_min <= a.speed_mm_s <= caps.speed_max
        assert caps.accel_min <= a.accel_mm_s2 <= caps.accel_max


# ── Puck-loss watchdog ──────────────────────────────────────────────────


def test_watchdog_trips_only_after_the_timeout():
    w = rp.PuckWatchdog(timeout_s=2.0)
    assert w.update(0.5) is None and not w.blind
    assert w.update(1.99) is None and not w.blind
    msg = w.update(2.01)
    assert msg is not None and "HOLDING" in msg
    assert w.blind and w.n_trips == 1


def test_watchdog_message_is_once_per_transition_not_once_per_tick():
    """At 100 Hz a per-tick warning is 6000 lines a minute."""
    w = rp.PuckWatchdog(timeout_s=1.0)
    assert w.update(2.0) is not None
    for _ in range(500):
        assert w.update(5.0) is None
    assert w.n_trips == 1


def test_watchdog_recovers_and_can_trip_again():
    w = rp.PuckWatchdog(timeout_s=1.0)
    w.update(2.0)
    msg = w.update(0.0)
    assert msg is not None and "reacquired" in msg
    assert not w.blind
    w.update(2.0)
    assert w.blind and w.n_trips == 2


def test_watchdog_trips_when_the_puck_has_never_been_seen():
    """staleness is inf before the first fix.

    A session that starts with no puck in frame must hold, not drive to
    whatever the bot opens with.
    """
    w = rp.PuckWatchdog(timeout_s=2.0)
    msg = w.update(float("inf"))
    assert w.blind and msg is not None and "never seen" in msg


def test_watchdog_freezes_commands_in_the_selftest():
    """The end-to-end scenario: cut the puck feed, commands must not move."""
    stats = rp.selftest(verbose=False)["watchdog"]
    assert stats["trips"] == 1
    assert stats["held_ticks"] > 50, \
        f"only {stats['held_ticks']} held ticks — the blind window was too short"
    assert stats["resumed_targets"] > 1, "never resumed after reacquisition"


def test_report_staleness_is_what_drives_the_watchdog():
    """The wiring: the loop feeds `t - report.t_puck` in, so pin that path."""
    r = rp.ReportBuilder()
    w = rp.PuckWatchdog(timeout_s=2.0)
    assert math.isinf(r.staleness(0.0)["puck"])
    assert w.update(r.staleness(0.0)["puck"]) is not None   # never seen
    r.add_puck(10.0, 1400.0, 500.0)
    assert w.update(r.staleness(10.0)["puck"]) is not None  # reacquired
    assert w.update(r.staleness(13.0)["puck"]) is not None  # gone again
    assert w.blind


# ── --gentle preset ─────────────────────────────────────────────────────


def _limits_args(gentle=False, speed=None, accel=None):
    import argparse
    return argparse.Namespace(gentle=gentle, speed=speed, accel=accel)


def test_gentle_sets_the_documented_first_run_caps():
    assert rp.resolve_limits(_limits_args(gentle=True)) == \
        (rp.GENTLE_SPEED, rp.GENTLE_ACCEL)
    assert (rp.GENTLE_SPEED, rp.GENTLE_ACCEL) == (500.0, 2000.0)


def test_without_gentle_the_normal_ceilings_apply():
    assert rp.resolve_limits(_limits_args()) == \
        (rp.Caps.speed_max, rp.Caps.accel_max)


def test_an_explicit_flag_beats_gentle():
    """Otherwise --gentle would silently override a number the user typed."""
    assert rp.resolve_limits(_limits_args(gentle=True, speed=300.0)) == \
        (300.0, rp.GENTLE_ACCEL)
    assert rp.resolve_limits(_limits_args(gentle=True, accel=1000.0)) == \
        (rp.GENTLE_SPEED, 1000.0)
    assert rp.resolve_limits(
        _limits_args(gentle=True, speed=300.0, accel=1000.0)) == (300.0, 1000.0)


def test_gentle_caps_actually_bind_a_real_bot():
    caps = rp.Caps(speed_max=rp.GENTLE_SPEED, accel_max=rp.GENTLE_ACCEL)
    for name in rp.list_heuristics():
        bot = rp.load_policy(f"heuristic:{name}", caps)
        r = rp.ReportBuilder()
        for k in range(20):
            r.add_puck(k * 0.005, 1400.0 - 8.0 * k, 500.0 + 3.0 * k)
        r.add_mallet(0.1, geom.HOME_X, geom.HOME_Y)
        a, _ = rp.plan(bot, r, 0.1, caps, None)
        assert a.speed_mm_s <= rp.GENTLE_SPEED
        assert a.accel_mm_s2 <= rp.GENTLE_ACCEL


# ── Lag monitor ─────────────────────────────────────────────────────────


def test_lag_is_zero_when_the_loop_keeps_up():
    m = rp.LagMonitor()
    for k in range(200):
        m.update(1000.0 + k * 0.005, 77.0 + k * 0.005)   # unrelated epochs
    assert abs(m.lag) < 1e-9
    assert m.warn_once() is None


def test_lag_grows_and_warns_once_when_frames_queue():
    """Wall time advancing faster than camera time IS the backlog."""
    m = rp.LagMonitor()
    for k in range(200):
        m.update(k * 0.005, k * 0.006)      # 20% too slow
    assert m.lag == pytest.approx(199 * 0.001)
    assert m.peak == pytest.approx(m.lag)
    msg = m.warn_once()
    assert msg is not None and "behind the camera" in msg
    assert m.warn_once() is None, "the warning must not repeat every frame"


def test_lag_stays_quiet_below_the_threshold():
    m = rp.LagMonitor()
    m.update(0.0, 0.0)
    m.update(0.100, 0.100 + rp.LagMonitor.WARN_S * 0.9)
    assert m.warn_once() is None


# ── Shutdown ────────────────────────────────────────────────────────────


class _FakeClient:
    """Records the command sequence a shutdown produces."""

    def __init__(self, pos=(1600.0, 400.0)):
        self.pos = pos
        self.calls: list[tuple] = []

    def get_position(self):
        return self.pos[0], self.pos[1], 0.0, 0.0

    def set_limits(self, s, a):
        self.calls.append(("LIMITS", s, a))

    def command_position(self, x, y, v):
        self.calls.append(("CMD", x, y, v))

    def close(self):
        self.calls.append(("CLOSE",))


def _park_args(park="stop"):
    import argparse
    return argparse.Namespace(park=park, park_speed=500.0, park_accel=2000.0)


@pytest.fixture
def _nosleep(monkeypatch):
    monkeypatch.setattr(rp.time, "sleep", lambda _s: None)


def test_shutdown_brakes_without_softening_the_accel_cap(_nosleep):
    """Stopping distance is v^2/2a, so a low accel cap LENGTHENS the stop.

    From 8000 mm/s, braking at 24000 mm/s^2 takes 1.3 m; at the 2000 mm/s^2
    park cap it would take 16 m — eight table lengths. A shutdown that set
    soft limits before commanding the stop would be the one that drove the
    paddle into the rail, so there must be no LIMITS before the brake.
    """
    c = _FakeClient(pos=(1600.0, 400.0))
    rp._shutdown(c, _park_args("stop"), prev=(1700.0, 500.0))
    assert c.calls[0] == ("CMD", 1600.0, 400.0, 0.0), \
        "the brake must command the CURRENT position at the existing caps"
    assert not any(k[0] == "LIMITS" for k in c.calls), \
        "no LIMITS may precede the brake"
    assert c.calls[-1] == ("CLOSE",)


def test_shutdown_brakes_at_where_it_is_not_where_it_was_going(_nosleep):
    """Commanding the last TARGET would let it carry on there instead."""
    c = _FakeClient(pos=(1600.0, 400.0))
    rp._shutdown(c, _park_args("stop"), prev=(1400.0, 700.0))
    assert c.calls[0][1:3] == (1600.0, 400.0)


def test_shutdown_park_home_brakes_first_then_traverses(_nosleep):
    """The soft caps apply to the traverse, which starts from rest."""
    c = _FakeClient(pos=(1600.0, 400.0))
    rp._shutdown(c, _park_args("home"), prev=None)
    kinds = [k[0] for k in c.calls]
    assert kinds == ["CMD", "LIMITS", "CMD", "CLOSE"]
    assert c.calls[0] == ("CMD", 1600.0, 400.0, 0.0)
    assert c.calls[1] == ("LIMITS", 500.0, 2000.0)
    assert c.calls[2] == ("CMD", geom.HOME_X, geom.HOME_Y, 500.0)


def test_shutdown_falls_back_to_the_last_target_if_pos_fails(_nosleep):
    class NoPos(_FakeClient):
        def get_position(self):
            raise ConnectionError("master went away")

    c = NoPos()
    rp._shutdown(c, _park_args("stop"), prev=(1400.0, 700.0))
    assert c.calls[0] == ("CMD", 1400.0, 700.0, 0.0)


def test_shutdown_is_a_noop_in_dry_run():
    """No client means nothing to shut down, and no exception either."""
    rp._shutdown(None, _park_args("stop"), prev=None)


# ── The loop itself ─────────────────────────────────────────────────────


def _install_fake_camera(monkeypatch, blank_from_s=None, duration_s=2.0):
    """Replace BlobStream with a synthetic one and return its class.

    Blobs are synthesised in PIXELS by projecting a puck and the robot
    mallet through the REAL calibrated pose, so the trackers downstream do
    exactly the work they do on the table — only the Spinnaker device is
    fake. `blank_from_s` stops emitting the puck's corners from that time
    on, which is what a puck knocked off the table looks like.
    """
    cv2 = pytest.importorskip("cv2")
    if not (_ROOT / "vision" / "calib" / "extrinsics.npz").exists():
        pytest.skip("no camera calibration in vision/calib")

    import numpy as np

    sys.path.insert(0, str(_ROOT / "vision" / "bin"))
    puck_stream = pytest.importorskip("puck_stream")
    tr = puck_stream.PuckTracker()

    def project(points_mm, z):
        obj = np.array([[p[0], p[1], z] for p in points_mm], float)
        px, _ = cv2.projectPoints(obj, tr.rvec, tr.tvec, tr.K, tr.dist)
        return px.reshape(-1, 2)

    r = geom.PUCK_MARKER_R_MM
    ar = geom.ARM_MARKER_R_MM

    class FakeStream:
        """Yields the same blob tuples BlobStream does: (seq, t_s, blobs)."""

        width, height = 1440, 1080
        closed = False

        def __init__(self, **_kw):
            pass

        def __iter__(self):
            for k in range(int(duration_s * 200)):
                t = k * 0.005
                mallet = [(1600.0, 400.0), (1600.0 - ar, 400.0),
                          (1600.0 + ar, 400.0)]
                groups = [project(mallet, 33.0)]
                if blank_from_s is None or t < blank_from_s:
                    px, py = 1400.0 - 300.0 * t, 500.0 + 200.0 * t
                    a = 0.3 + np.arange(4) * (math.pi / 2)
                    groups.insert(0, project(
                        [(px + r * math.cos(v), py + r * math.sin(v))
                         for v in a], geom.PUCK_MARKER_Z_MM))
                pts = np.vstack(groups)
                yield k, t, np.hstack([pts, np.full((len(pts), 1), 30.0)])

        def close(self):
            FakeStream.closed = True

    monkeypatch.setattr(puck_stream, "BlobStream", FakeStream)
    return FakeStream


def _loop_args(**over):
    import argparse
    base = dict(
        live=False, policy="heuristic:goalie", opponent=False, gentle=False,
        fps=200.0, exposure=300.0, gain=12.0, threshold=90, cmd_hz=100.0,
        speed=8000.0, accel=24000.0, ramp=3.0,
        puck_timeout=rp.DEFAULT_PUCK_TIMEOUT_S,
        limits_interval=rp.CapCommitter.MIN_INTERVAL_S,
        park="stop", park_speed=500.0, park_accel=2000.0, no_enable=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_dry_run_loop_executes_end_to_end(monkeypatch, capsys):
    """run() with a FAKE camera and no robot.

    Everything else in this file tests a piece; this runs the actual loop —
    real PuckTracker, real MalletTracker, real camera pose, real bot, real
    clamp. Skipped where the vision stack is not installed: the `ai` package
    does not depend on cv2, and the rest of this file deliberately does not
    either.
    """
    stream = _install_fake_camera(monkeypatch)
    assert rp.run(_loop_args()) == 0
    assert stream.closed, "the camera must be released on the way out"

    out = capsys.readouterr().out
    assert "DRY RUN" in out, "a dry run must say so"
    assert "WOULD SEND" in out, "a dry run must report what it would command"
    # The tracker really resolved the puck, so the status line is not the
    # empty-history one, and the watchdog never fired.
    assert "puck --" not in out
    assert "HOLDING" not in out


def test_dry_run_loop_announces_the_watchdog_when_the_puck_goes(monkeypatch,
                                                                capsys):
    """The puck vanishes mid-run: the loop must say so, in DRY RUN too.

    Through the real trackers this time, not the selftest's synthetic
    report — the puck stops being in frame at all, which is the case that
    reaches the watchdog by way of PuckTracker giving up its coast.
    """
    _install_fake_camera(monkeypatch, blank_from_s=0.5, duration_s=4.0)
    assert rp.run(_loop_args(puck_timeout=1.0)) == 0

    out = capsys.readouterr().out
    assert "HOLDING" in out, "the watchdog must announce itself"
    assert "[DRY RUN]" in out, "and say it is a dry run when it is one"
    assert out.count("HOLDING position") == 1, \
        "the watchdog must announce once, not once per tick"


def test_default_is_dry_run():
    """Forgetting a flag must leave the robot still."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    assert ap.parse_args([]).live is False
    # And the real parser agrees: --live is the only way in.
    src = (_ROOT / "ai" / "bin" / "run_policy.py").read_text()
    assert '"--dry-run"' not in src, \
        "a --dry-run flag would imply live is the default; it must not be"
