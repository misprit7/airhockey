#!/usr/bin/env python3
"""Summarize frozen neural qualification reports, attributing each player's loads.

This reports measurements only; it never promotes a checkpoint or runs hardware.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def game_metrics(game):
    cross = bool(game.get("opponent_checkpoint"))
    side = game.get("candidate_side")
    if cross and side not in (0, 1):
        raise ValueError("Cross-play requires an explicit candidate_side")
    indices = [side] if cross else [0, 1]
    peak = np.asarray(game["peak_load"])
    overload = np.asarray(game["overload_seconds"])
    scores = np.asarray(game["score"]).sum(axis=1)
    result = {
        "candidate_side": side if cross else "both",
        "candidate_peak_load": float(peak[indices].max()),
        "candidate_overload_seconds": float(overload[indices].sum()),
        "score_by_side": scores.tolist(),
        "reachable_stall_seconds": float(np.sum(game["reachable_stall_player_seconds"])),
        "game_seconds": game["games"] * game["seconds"],
    }
    if cross:
        result.update(candidate_goals=int(scores[side]), opponent_goals=int(scores[1-side]),
                      opponent_peak_load=float(peak[1-side].max()),
                      opponent_overload_seconds=float(overload[1-side].sum()))
    if "firmware_motion_audit" in game:
        audit = game["firmware_motion_audit"]
        result["candidate_intervals_over_cap"] = int(np.asarray(audit["intervals_over_cap"])[indices].sum())
    return result


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    checkpoint = Path(plan["checkpoint"]).resolve()
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": plan["checkpoint_sha256"],
              "complete": (directory / "complete.json").exists(), "reports": {}}
    for name, _, _ in plan["tasks"]:
        path = directory / (name + ".json")
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if Path(report["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"Mismatched checkpoint: {path}")
        if "selfplay" in report:
            metrics = game_metrics(report["selfplay"])
        elif "scenario" in report:
            metrics = {k: report[k] for k in ("trials", "saved", "peak_load", "overload_seconds")}
        elif "requested_skills" in report:
            metrics = {}
            for kind in ("stationary", "receiving"):
                cohorts = [v[kind] for v in report["requested_skills"].values()]
                keys = ("trials", "controlled", "control_to_shot", "control_to_requested_fast",
                        "first_requested_shots_at_least_6m_s", "first_outgoing_contacts_from_ahead")
                metrics[kind] = {key: sum(c.get(key, 0) for c in cohorts) for key in keys}
                if name == "recovery-v2" and kind == "receiving":
                    if not all(c.get("recovery_metric_version") == 2 for c in cohorts):
                        raise ValueError("Old recovery metrics cannot measure original possession")
        else:
            metrics = {k: report["skills"]["defense"][k] for k in ("trials", "saved", "peak_load", "overload_seconds")}
        result["reports"][name] = metrics
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps([summarize(p) for p in args.directories], indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
