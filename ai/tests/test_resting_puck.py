"""The resting-puck gates (airhockey/resting_puck.py, bin/eval_resting_puck.py)
on two scripted policies: one that always drives through the puck toward the
far goal, and one that parks beside it at the accel-cap floor and never
strikes -- the hover the gates exist to catch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from airhockey import resting_puck as rp
from airhockey.neural_training import NeuralTrainingEnv

BIN = Path(__file__).resolve().parents[1] / "bin"
_ENV = NeuralTrainingEnv(1)
LOW, HIGH = _ENV.decoder.low.copy(), _ENV.decoder.high.copy()
GRID = dict(xs=(0.3, 0.7), ys=(0.4, 0.75), requests=(1, 2))


class _Scripted:
    history, shot_conditioned, environment_options = 1, True, {}

    def _action(self, target, velocity, duration, cap):
        action = np.zeros((len(target), 6), np.float32)
        action[:, 0:2] = np.clip(2 * (target - LOW) / (HIGH - LOW) - 1, -1, 1)
        action[:, 2:4] = velocity
        action[:, 4], action[:, 5] = duration, cap
        return action

    def to(self, device):
        return self

    def eval(self):
        return self


class Striker(_Scripted):
    """Arrive at the puck moving up the table, at the full accel cap."""

    def act(self, observation, stochastic=False):
        return self._action(observation[:, 0:2], (0.0, 1.0), -1.0, 1.0)


class Hoverer(_Scripted):
    """Go and stand beside the puck, at rest, at the cap floor once there."""
    offset = (0.105, -0.115)

    def act(self, observation, stochastic=False):
        target = observation[:, 0:2] + self.offset
        near = np.linalg.norm(observation[:, 4:6] - target, axis=1) < 0.03
        return self._action(target, (0.0, 0.0), 1.0, np.where(near, -1.0, 0.0))


def test_wilson_interval():
    assert rp.wilson(0, 48)[0] == 0 and 0.05 < rp.wilson(0, 48)[1] < 0.09
    low, high = rp.wilson(64, 70)
    assert 0.82 < low < 64 / 70 < high < 0.97
    assert np.isnan(rp.wilson(0, 0)).all()


def test_zones_and_cells():
    px = np.array([0.7, 0.3, 0.5, 0.7, 0.1, 0.9, 0.15, 0.85, 0.7, 0.2])
    py = np.array([0.75, 0.65, 0.5, 0.3, 0.5, 0.2, 0.1, 0.15, 0.9, 0.15])
    zone, side = rp.zones(px, py)
    assert zone.tolist() == ["forward", "forward", "mid", "back", "rail", "rail", "corner", "corner",
                             "front_edge", "back"]
    assert side.tolist() == ["right", "left", "centre", "right", "", "", "", "", "", "left"]
    # The headline cell: right side of the forward strip with the left bank asked for.
    px, py, req, beside = rp.stall_conditions(rp.DEFAULT_XS, rp.DEFAULT_YS, ("home", "beside"), (1, 2, 3))
    assert len(px) == 17 * 16 * 2 * 3
    raw = dict(px=px, py=py, req=req, beside=beside, seconds=10.0, load=0.45, t_touch=np.full(len(px), np.nan),
               t_cross=np.full(len(px), np.nan), peak_cap=np.full(len(px), 5.0), final=np.zeros((len(px), 4)),
               final_cap=np.full(len(px), 5.0),
               t_strike=np.where((px > 0.5) & (py >= 0.65) & (req == 1), np.nan, 1.0))
    report = rp.stall_report(raw)
    cell = report["cross_bank"]["forward-right / left_bank"]
    assert cell["n"] == cell["stalled"] == 7 * 5 * 2 and cell["rate"] == 1.0
    assert report["cross_bank"]["forward-left / right_bank"]["stalled"] == 0
    assert set(report["cross_bank"]) == {f"{b}-{s} / {r}" for b in rp.BANDS
                                         for s, r in (("right", "left_bank"), ("left", "right_bank"))}
    assert report["worst_cell"]["cell"] == "forward-right / left_bank"
    # The centre column is reported but too thin to gate.
    centre = next(c for c in report["cells"] if c["cell"] == "forward-centre / left_bank")
    assert centre["n"] == 10 and not centre["gated"]
    assert report["by_zone"]["rail"]["stalled"] == 2 * 5 and "stalled_trials" in report
    assert "0.85" in rp.format_map(report)


def test_rest_spells_need_slow_own_half_and_reach():
    low, high = np.array([0.08, 0.08]), np.array([0.92, 0.8])
    puck = np.tile([0.5, 0.5], (300, 1))
    speed = np.zeros(300)
    speed[50:60] = 1.0                       # a touch splits the spell
    puck[150:200] = [0.5, 1.4]               # far half
    puck[250:] = [0.5, 0.95]                 # own half but past the mallet's reach
    spells = rp.rest_spells(puck, speed, low, high, 0.091, half=1.0)
    assert spells == [(0, 1.0), (60, 1.8), (200, 1.0)]
    report = rp.rest_report([dict(game=0, seconds=s) for s in (0.4, 1.0, 4.5, 6.0)], games=2, seconds=60.0)
    assert report["spells_ge_1s"] == dict(count=3, per_game_minute=1.5, seconds_per_game_minute=5.75)
    assert report["spells_ge_4s"]["count"] == 2 and report["longest"][0]["seconds"] == 6.0


def test_striker_clears_the_grid_and_the_hoverer_stalls_everywhere():
    struck = rp.stall_map(Striker(), {}, seed=3, seconds=3.0, **GRID)
    assert len(struck["px"]) == 16 and not rp.stalled(struck).any()
    assert (struck["strike_speed"][~np.isnan(struck["t_strike"])] > rp.STRIKE_SPEED).all()
    hover = rp.stall_map(Hoverer(), {}, seed=3, seconds=3.0, keep_frames=True, **GRID)
    assert rp.stalled(hover).all() and np.isnan(hover["t_touch"]).all()
    assert hover["frames"].shape == (151, 16, len(rp.FRAME_COLUMNS))
    # The puck never moved and the mallet ended beside it.
    assert np.allclose(hover["final"][:, :2], np.column_stack((hover["px"], hover["py"])), atol=1e-3)
    assert np.allclose(hover["final"][:, 2:] - hover["final"][:, :2], (0.105, -0.115), atol=0.02)
    report = rp.stall_report(hover, min_cell_n=4)
    assert report["overall"]["rate"] == 1.0 and report["by_start"]["beside"]["stalled"] == 8
    assert report["stalled_trials"]["touched_share"] == 0 and report["stalled_trials"]["final_accel_cap_p50"] <= 6


def test_corner_pucks_are_mirror_pairs_by_request():
    touched = rp.corner_recovery(Striker(), {}, seed=5, count=16, ticks=150)
    assert list(touched["cells"]) == [f"{s} / {r}" for s in ("left", "right") for r in rp.REQUEST_NAMES]
    assert all(c == dict(c, n=2, touched=2) for c in touched["cells"].values())
    stand_off = Hoverer()
    stand_off.offset = (0.0, 0.25)           # up the rail from the corner, out of touch
    left = rp.corner_recovery(stand_off, {}, seed=5, count=16, ticks=100)
    assert left["touched"] == 0 and all(c["untouched_min_gap_mm"] > 50 for c in left["cells"].values())
    assert left["cells"][left["worst_cell"]]["first_touch_s"] is None


def test_evaluator_gates_and_report(monkeypatch, tmp_path, capsys):
    sys.path.insert(0, str(BIN))
    import eval_resting_puck as ev
    class CornerShy(Hoverer):
        """Hovers beside an open-table puck and stands off one in a corner."""

        def act(self, observation, stochastic=False):
            self.offset = np.where(observation[:, 1:2] < 0.2, (0.0, 0.25), (0.105, -0.115))
            return super().act(observation)
    policies = {"strike/agent.pt": Striker(), "hover/agent.pt": CornerShy()}
    monkeypatch.setattr(ev, "prepare", lambda candidate, environment_from, device="cpu": (policies[candidate], {}))
    monkeypatch.setattr(ev.proxy_benchmark, "candidate_hash", lambda spec: "scripted")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_resting_puck.py", "hover/agent.pt", "--reference", "strike/agent.pt", "--seed", "3",
        "--xs", "0.3,0.7", "--ys", "0.4,0.75", "--stall-seconds", "2", "--min-cell-n", "2",
        "--corner-count", "8", "--corner-seconds", "3", "--min-corner-touched", "1", "--games", "2", "--seconds", "5",
        "--tiers", "table",
        "--record", str(tmp_path / "rec" / "hover"), "--record-count", "1", "--output", str(output)])
    ev.main()
    text = capsys.readouterr().out
    result = json.loads(output.read_text())
    hover, strike = result["columns"]
    assert hover["title"] == "hover" and strike["title"] == "strike" and result["sim_profile"] is None
    assert not hover["gates"]["stall"]["passed"] and not hover["gates"]["corner"]["passed"]
    assert strike["gates"]["stall"]["passed"] and strike["gates"]["corner"]["passed"]
    assert not result["passed"] and not hover["passed"]
    assert "FAIL  hover: stall" in text and "PASS  strike: stall" in text and "PASS  strike: corner" in text
    assert "*forward-right / left_bank" in text and "REST SPELLS IN GAMES" in text
    games = hover["games"]["table"]
    assert games["games"] == 2 and games["spells_ge_4s"]["count"] == hover["games"]["long_spells"]
    assert hover["gates"]["games"]["passed"] == (hover["games"]["long_spells"] == 0)
    replays = sorted(p.name for p in (tmp_path / "rec").iterdir())
    assert any(n.startswith("hover_resting-stall-") for n in replays)
    assert any(n.startswith("hover_resting-games-table-g") for n in replays)
    assert any(n.startswith("hover-reference_resting-trial-") for n in replays)
    stall = json.loads((tmp_path / "rec" / next(n for n in replays if "resting-stall" in n)).read_text())
    assert stall["metadata"]["stalled"] is True and stall["metadata"]["simulation_only"] is True


def test_rest_watch_reports_spells_per_game():
    still = np.tile([0.4, 0.5, 0, 0, 0.5, 0.2, 0.5, 1.8, 0, 0, 1], (200, 1))

    class Env:
        decoder = type("D", (), dict(low=LOW, high=HIGH))
        cfg = _ENV.cfg
    watch = rp.RestWatch.__new__(rp.RestWatch)
    watch.env, watch.frames = Env, list(np.stack((still, still), axis=1))
    watch.frames[120][1, 2] = 2.0            # game 1 is struck at 2.4 s
    spells = watch.spells()
    assert [(s["game"], s["start_s"], s["seconds"], s["request"]) for s in spells] == [
        (0, 0.0, 4.0, "left_bank"), (1, 0.0, 2.4, "left_bank"), (1, 2.42, 1.58, "left_bank")]
    assert {s["zone"] for s in spells} == {"mid-left"}
    assert rp.rest_report(spells, games=2, seconds=4.0)["spells_ge_4s"]["count"] == 1
