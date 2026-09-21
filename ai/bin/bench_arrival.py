#!/usr/bin/env python3
"""Simulation only: matched arrival-state / position-command skill search.

From repository root:
  PYTHONPATH=ai python3 ai/bin/bench_arrival.py --output logs/arrival-experiment
No checkpoint is registered for deployment and no hardware modules are loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from airhockey.motion import CartState
from airhockey.skill_benchmark import SkillTrials, TASKS, make_fixtures, search


def summarize(fixtures, result):
    out = {}
    for i, task in enumerate(TASKS):
        mask = fixtures.task == i
        contact = result["contact"][mask]
        placement = result["placement_error"][mask]
        crossing = np.isfinite(result["crossing_x"][mask])
        out[task] = {
            "attempts": int(mask.sum()), "contacts": int(contact.sum()),
            "whiffs": int((~contact).sum()), "goals": int(result["goal"][mask].sum()),
            "cushions": int(result["cushion"][mask].sum()),
            "placement_within_50mm": int((crossing & (placement < 0.05)).sum()),
            "top_plane_crossings": int(crossing.sum()),
            "crossing_error_median_mm": float(np.median(placement[crossing]) * 1000) if crossing.any() else None,
            "contact_puck_speed_mean": float(result["puck_speed_after_contact"][mask][contact].mean()) if contact.any() else None,
            "final_puck_speed_mean": float(result["final_puck_speed"][mask].mean()),
            "actual_accel_squared_integral_mean": float(result["effort"][mask].mean()),
            "actual_accel_peak": float(result["peak_accel"][mask].max()),
            "above_40_accel_ms_mean": float(result["high_accel_time"][mask].mean() * 1000),
            "travel_mean_m": float(result["travel"][mask].mean()),
            "backstop_violations": int(result["backstop_violation"][mask].sum()),
            "arrival_position_error_median_mm": float(np.nanmedian(result["arrival_position_error"][mask]) * 1000)
                if np.isfinite(result["arrival_position_error"][mask]).any() else None,
            "arrival_velocity_error_median": float(np.nanmedian(result["arrival_velocity_error"][mask]))
                if np.isfinite(result["arrival_velocity_error"][mask]).any() else None,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=16)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--accel", type=float, default=60)
    parser.add_argument("--effort-weight", type=float, default=0.3)
    args = parser.parse_args()
    if args.per_task < 1 or args.population < 4 or args.iterations < 1:
        parser.error("positive fixture/iteration counts and population >= 4 required")
    args.output.mkdir(parents=True, exist_ok=False)
    trials = SkillTrials(args.accel, effort_weight=args.effort_weight)
    # A disjoint fixture bank. No hyperparameters are fit using this seed.
    fixtures = make_fixtures(args.seed, args.per_task)
    one = fixtures.take([0])
    cart = CartState(1)
    cart.reset(one.paddle[:, 0] * 1000, one.paddle[:, 1] * 1000)
    arrival = trials.seed_arrival(one)
    timing = []
    for iteration in range(550):
        t = time.perf_counter()
        trials.decoder.decode(cart, arrival, 0, one.paddle, np.array([args.accel]),
                              max_accel=args.accel, delay_s=trials.delay)
        if iteration >= 50:
            timing.append(time.perf_counter() - t)
    p99 = float(np.percentile(timing, 99))
    # Include measured decoder cost on top of the measured 12 ms host delay.
    # Same delay for both methods isolates representation from timing changes.
    requested_delay = 0.012 + p99
    trials = SkillTrials(args.accel, delay=requested_delay, effort_weight=args.effort_weight)
    metadata = {
        "kind": "offline trajectory search, not learned policy evaluation",
        "args": {**vars(args), "output": str(args.output)},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path("ai/airhockey/arrival.py"), Path("ai/airhockey/skill_benchmark.py"))},
        "physics": asdict(trials.cfg),
        "accel_ceiling": args.accel, "command_delay_s": trials.delay,
        "decoder_ms_p50": float(np.median(timing) * 1000), "decoder_ms_p99": p99 * 1000,
        "observation": "perfect state; latency modeled, no camera noise",
        "objective": "same contact/placement/control loss and actual-acceleration cost",
        "effort_units": "integral of squared measured acceleration, m^2/s^3; NOT motor heat",
        "fixture_selection": "all generated fixtures retained; no failure filtering",
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_results = {}
    archive = {f"fixture_{key}": getattr(fixtures, key) for key in fixtures.__dataclass_fields__}
    # Common starting trajectory, materialized for the baseline.
    initial = trials.seed_arrival(fixtures)
    warm = trials.rollout(fixtures, initial, "arrival")
    metadata["initial"] = summarize(fixtures, warm)
    for mode in ("position", "arrival"):
        action, history = search(trials, fixtures, mode, args.seed + 1,
            args.population, args.iterations,
            warm=warm["commands"] if mode == "position" else initial)
        result = trials.rollout(fixtures, action, mode, record=True)
        all_results[mode] = result
        metadata[mode] = {"summary": summarize(fixtures, result), "search": history}
        archive[f"{mode}_actions"] = action
        archive.update({f"{mode}_{k}": v for k, v in result.items()})
        np.savez_compressed(args.output / "trials.npz", **archive)
        (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata[mode]["summary"], indent=2), flush=True)
    # Feasibility evidence is the union of successful *executed* trajectories,
    # not an analytic guess; unresolved cases stay in the primary denominator.
    success = {m: np.where(fixtures.task == 2, r["cushion"], r["goal"])
               for m, r in all_results.items()}
    feasible = success["position"] | success["arrival"]
    metadata["demonstrated_feasible"] = {
        task: {"count": int(feasible[fixtures.task == i].sum()),
               "total": int((fixtures.task == i).sum())} for i, task in enumerate(TASKS)}
    metadata["paired_success"] = {task: {
        "arrival_only": int((success["arrival"] & ~success["position"] & (fixtures.task == i)).sum()),
        "position_only": int((success["position"] & ~success["arrival"] & (fixtures.task == i)).sum())}
        for i, task in enumerate(TASKS)}
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
