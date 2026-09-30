import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("qualification_summary",
    Path(__file__).parents[1] / "bin/summarize_neural_qualification.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cross_play_never_attributes_opponent_overload_to_candidate():
    game = dict(opponent_checkpoint="old.pt", candidate_side=0, games=2, seconds=180,
        peak_load=[[.9, .95], [1.13, .94]], overload_seconds=[[0, 0], [24.76, 0]],
        score=[[4, 2], [1, 1]], reachable_stall_player_seconds=[10, 20])
    result = module.game_metrics(game)
    assert result["candidate_peak_load"] == .95
    assert result["candidate_overload_seconds"] == 0
    assert result["opponent_overload_seconds"] == 24.76
    assert result["candidate_goals"] == 6
    game["candidate_side"] = 1
    assert module.game_metrics(game)["candidate_overload_seconds"] == 24.76
    del game["candidate_side"]
    with pytest.raises(ValueError):
        module.game_metrics(game)
    del game["opponent_checkpoint"]
    result = module.game_metrics(game)
    assert result["candidate_peak_load"] == 1.13
    assert result["candidate_overload_seconds"] == 24.76
    assert result["game_seconds"] == 360
