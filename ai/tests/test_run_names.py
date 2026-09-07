from pathlib import Path

import pytest

from airhockey.run_names import check_run_name, next_name, parse


def test_scheme():
    assert parse("2.3-control-gate-selfplay") == dict(major=2, minor=3, desc="control-gate",
                                                      stage="selfplay", snap=None)
    assert parse("3.3-turnover-selfplay-300k")["snap"] == "300k"
    assert parse("4.0-strike-primitive-proximity")["stage"] == "proximity"
    for bad in ("run5_selfplay", "2.3-selfplay", "2.3-Control-Gate-selfplay", "2-gate-selfplay",
                "2.3-gate-pretrain", "2.3-gate-selfplay-300"):
        assert parse(bad) is None
        with pytest.raises(ValueError):
            check_run_name(bad)
    check_run_name("_smoke_anything")          # scratch is exempt


def test_next_name_takes_the_next_minor(tmp_path: Path):
    for d in ("3.0-horizon8-selfplay", "3.1-no-demos-selfplay", "3.1-no-demos-selfplay-300k",
              "2.12-drive-squared-selfplay", "_smoke"):
        (tmp_path / d).mkdir()
    assert next_name(3, "shot-clock", "selfplay", tmp_path) == "3.2-shot-clock-selfplay"
    assert next_name(2, "x", "goalie", tmp_path) == "2.13-x-goalie"
    assert next_name(4, "strike-primitive", "proximity", tmp_path) == "4.0-strike-primitive-proximity"
