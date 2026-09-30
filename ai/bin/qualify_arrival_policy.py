#!/usr/bin/env python3
"""One frozen-candidate simulation qualification on previously unused seeds.

Does not select checkpoints, mutate physical defaults, or certify real hardware.
The opponent has the same instantaneous kinematic caps; its modeled load is
reported separately and is not a constraint on candidate qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from eval_foundation import load, match
from airhockey.policy_benchmark import evaluate_skills
from airhockey.thermal import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-task", type=int, default=500)
    p.add_argument("--continuous-seconds", type=int, default=3600)
    p.add_argument("--hot-start-load", type=float, default=0.9)
    args = p.parse_args()
    if not 0 < args.hot_start_load < 0.95:
        p.error(
            "hot-start load must be positive and below the 0.95 qualification margin"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    manifest = dict(
        suite_version=3,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        reference=str(args.reference),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        seeds=list(range(args.seed, args.seed + 9)),
        per_task=args.per_task,
        continuous_seconds=args.continuous_seconds,
        hot_start_load=args.hot_start_load,
        runtime=dict(
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda
        ),
        deployment_ready=False,
        criteria=dict(
            stationary=0.95,
            moving=0.65,
            cushion=0.9,
            defense=0.97,
            random_pose_defense=0.92,
            fast_defense=0.98,
            bank_defense=0.98,
            modeled_load=0.95,
            acceleration=60.1,
            speed=12.01,
            positive_goal_margin_each_role=True,
        ),
        source_hashes={},
        configuration_hashes={},
    )
    for label, checkpoint in (
        ("candidate", args.checkpoint),
        ("reference", args.reference),
    ):
        config = checkpoint.parent / "run.json"
        if config.exists():
            payload = config.read_bytes()
            (args.output / f"{label}-run.json").write_bytes(payload)
            manifest["configuration_hashes"][label] = hashlib.sha256(
                payload
            ).hexdigest()
    for path in [
        Path(__file__),
        ROOT / "ai/bin/eval_foundation.py",
        *sorted((ROOT / "ai/airhockey").glob("*.py")),
        ROOT / "fw/include/motion_profile.h",
        *sorted((ROOT / "fw/host").glob("*.cpp")),
        *sorted((ROOT / "fw/host/build").glob("*.so")),
        ROOT / "shared/cdpr_geometry.py",
        ROOT / "shared/cdpr_geometry.h",
        DEFAULT_MODEL,
    ]:
        dest = args.output / "source" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        manifest["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = dict(
        manifest=manifest, skills={}, matches={}, passed=False, completed=False
    )

    def write():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    def candidate():
        return load(args.checkpoint)

    failures = []
    for name, seed, random_defense, fast, bank in [
        ("skills", args.seed, False, False, False),
        ("stress", args.seed + 1, True, False, False),
        ("fast_defense", args.seed + 7, False, True, False),
        ("bank_defense", args.seed + 8, False, True, True),
    ]:
        r = evaluate_skills(
            candidate(),
            legacy=True,
            wide=True,
            guard=True,
            planner=False,
            game_requests=True,
            random_paddle=True,
            random_defense_paddle=random_defense,
            defense_speed_range=(8, 12) if fast else (2, 8),
            defense_bank=bank,
            seed=seed,
            per_task=args.per_task,
        )
        (args.output / f"{name}.json").write_text(json.dumps(r, indent=2))
        summary["skills"][name] = r["tasks"]
        if fast:
            thresholds = {"defense": 0.98}
        elif random_defense:
            thresholds = {"defense": 0.92}
        else:
            thresholds = {
                "stationary": 0.95,
                "moving": 0.65,
                "cushion": 0.9,
                "defense": 0.97,
            }
        for task, threshold in thresholds.items():
            row = r["tasks"][task]
            if row["successes"] / row["attempts"] < threshold:
                failures.append(f"{name}/{task}")
        for task, row in r["tasks"].items():
            if (
                row["peak_modeled_load"] >= 0.95
                or row["peak_actual_acceleration"] >= 60.1
            ):
                failures.append(f"{name}/{task}/limits")
        print(name, {k: v["successes"] for k, v in r["tasks"].items()}, flush=True)
        write()
    rival = load(args.reference)
    rival.cfg.mpc = True
    rival._plan_batch = torch.compile(rival._plan_batch, mode="reduce-overhead")
    for name, seed, reverse, seconds, hot in [
        ("forward", args.seed + 2, False, 180, False),
        ("reversed", args.seed + 3, True, 180, False),
        ("hot_forward", args.seed + 5, False, 180, True),
        ("hot_reversed", args.seed + 6, True, 180, True),
        ("continuous_selfplay", args.seed + 4, False, args.continuous_seconds, False),
    ]:
        torch.manual_seed(seed)
        a = candidate()
        b = candidate() if name == "continuous_selfplay" else rival
        sides = (b, a) if reverse else (a, b)
        labels = [str(args.checkpoint), str(args.reference)]
        if name == "continuous_selfplay":
            labels[1] = labels[0]
        if reverse:
            labels.reverse()
        initial_load = [None, None]
        if hot:
            initial_load[int(reverse)] = args.hot_start_load
        r = match(
            *sides,
            games=4 if name == "continuous_selfplay" or hot else 8,
            seconds=seconds,
            seed=seed,
            guard=True,
            self_play=name == "continuous_selfplay",
            policy_labels=labels,
            initial_load=initial_load if hot else None,
            record=ROOT
            / "ai/recordings"
            / f"{args.checkpoint.parent.name}_heldout_{name}.json",
        )
        (args.output / f"{name}.json").write_text(json.dumps(r, indent=2))
        selected = [0, 1] if name == "continuous_selfplay" else [int(reverse)]
        goals_for, goals_against = sum(r["goals_for"]), sum(r["goals_against"])
        if reverse:
            goals_for, goals_against = goals_against, goals_for
        row = dict(
            goals_for=goals_for,
            goals_against=goals_against,
            peak_load=max(max(r["peak_load"][i]) for i in selected),
            overload_seconds=sum(sum(r["over_limit_seconds"][i]) for i in selected),
            peak_acceleration=max(max(r["peak_accel"][i]) for i in selected),
            peak_speed=max(max(r["peak_speed"][i]) for i in selected),
            unresolved_guard_forecasts=sum(
                sum(r["guard_unresolved"][i]) for i in selected
            ),
        )
        if (
            row["peak_load"] >= 0.95
            or row["overload_seconds"] > 0
            or row["peak_acceleration"] >= 60.1
            or row["peak_speed"] > 12.01
        ):
            failures.append(name + "/limits")
        if name != "continuous_selfplay" and goals_for <= goals_against:
            failures.append(name + "/score")
        summary["matches"][name] = row
        print(name, json.dumps(row), flush=True)
        write()
    summary.update(
        passed=not failures,
        failures=failures,
        completed=True,
        limitation="Simulation-only; motor-load model is provisional and superhuman performance is not established against humans.",
    )
    write()


if __name__ == "__main__":
    main()
