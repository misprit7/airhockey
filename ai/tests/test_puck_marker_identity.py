"""Partial puck patterns must never steal the robot's three paddle markers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vision/bin'))
from puck_markers import MARK_R, find_puck

# Actual paddle markers from the failed 20261002-110300 reach probe,
# back-projected onto the puck plane. The old three-point fit accepted this.
PADDLE = np.array([[1471.58667465, 502.59265491],
                   [1470.83654973, 465.43511968],
                   [1500.62231133, 484.00085772]])


def corners():
    angles = 0.31 + np.arange(4) * np.pi / 2
    return np.array([812., 431.]) + MARK_R * np.column_stack(
        [np.cos(angles), np.sin(angles)])


@pytest.mark.parametrize('previous', [None, PADDLE.mean(axis=0)])
def test_saved_paddle_is_not_a_puck(previous):
    assert find_puck(PADDLE, prev=previous) is None


@pytest.mark.parametrize('keep', [(0,), (0, 1), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize('previous', [None, np.array([812., 431.])])
def test_incomplete_puck_never_acquires_even_with_history(keep, previous):
    assert find_puck(corners()[list(keep)], prev=previous) is None


def test_real_square_selected_with_paddle_and_nearby_stray():
    # Exercise the four-subset search within a group containing a glare dot.
    world = np.vstack([PADDLE, corners(), [[813., 432.]]])
    centre, _, members, _ = find_puck(world)
    np.testing.assert_allclose(centre, [812., 431.])
    assert set(members) == {3, 4, 5, 6}


def test_distance_gate_still_rejects_impossible_jump():
    assert find_puck(corners(), prev=np.array([0., 0.]), max_distance=50.) is None
