"""Wrapper so `pytest ai` covers the goalie's own checks.

Delete alongside airhockey/demo_goalie.py when the learned policy lands.
"""
from airhockey.demo_goalie import _selftest


def test_demo_goalie_selftest():
    _selftest()
