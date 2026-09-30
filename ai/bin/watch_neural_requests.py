#!/usr/bin/env python3
"""Evaluate request-conditioned simulation checkpoints and publish WIP replays.

This process only reads checkpoints, runs simulation, and writes local artifacts.
It never promotes a model to hardware deployment.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def publish_replay_alias(recording, alias):
    """Keep old deep links working without discarding checkpoint recordings."""
    temp = alias.with_suffix(".tmp")
    shutil.copyfile(recording, temp)
    temp.replace(alias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=130)
    parser.add_argument("--seed", type=int, default=20262351)
    parser.add_argument("--replay-name", default="neural-requests-wip")
    parser.add_argument("--random-practice-opponent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference", type=Path, help="Optional learned reference for final matches in both colors")
    parser.add_argument("--report-sensing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recovery-suite", action="store_true", help="Measure control and follow-through on slow outgoing pucks")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 60 * args.minutes
    seen = set()
    latest = None
    practice_flags = ["--random-practice-opponent"] if args.random_practice_opponent else []

    def evaluate(checkpoint, label, flags):
        report = args.output_dir / f"{label}.json"
        record = Path(flags[flags.index("--record") + 1]) if "--record" in flags else None
        if not report.exists() or (record is not None and not record.exists()):
            with report.with_suffix(".log").open("w") as log:
                subprocess.run([
                    sys.executable, str(ROOT / "ai/bin/eval_neural_player.py"),
                    str(checkpoint), "--output", str(report), "--seed", str(args.seed),
                    *([] if args.report_sensing is None else
                      ["--report-sensing" if args.report_sensing else "--no-report-sensing"]),
                    *flags,
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        return json.loads(report.read_text())

    recordings = ROOT / "ai/recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    alias = recordings / (args.replay_name + ".json")

    initial_published = False
    while time.monotonic() < deadline:
        checkpoints = sorted(args.run.glob("agent_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        # Recheck inside the loop: the watcher can start before the trainer has
        # finished writing its initial checkpoint.
        initial = args.run / "agent_initial.pt"
        if not initial_published and not checkpoints and initial.exists():
            import torch
            initial_step = torch.load(initial, map_location="cpu", weights_only=False)["step"]
            recording = recordings / f"{args.run.name}_step_{initial_step:09d}.json"
            if not recording.exists():
                evaluate(initial, "initial-replay", [
                    "--games-only", "--games", "8", "--seconds", "120",
                    "--record", str(recording),
                ])
            publish_replay_alias(recording, alias)
            initial_published = True
        # Catch up to the newest snapshot if an evaluation was slow.
        if checkpoints and checkpoints[-1] not in seen:
            latest = checkpoints[-1]
            step = int(latest.stem.rsplit("_", 1)[1])
            recording = recordings / f"{args.run.name}_step_{step:09d}.json"
            report = evaluate(latest, f"step-{step}", [
                "--request-suite", "--per-task", "128", "--receiving-min", "2", "--receiving-max", "5",
                "--seconds", "120", "--games", "8",
                "--record", str(recording),
                *practice_flags,
            ])
            # Recover recordings missing from older/cached evaluation reports.
            if not recording.exists():
                evaluate(latest, f"replay-step-{step}", [
                    "--games-only", "--games", "8", "--seconds", "120",
                    "--record", str(recording),
                ])
            publish_replay_alias(recording, alias)
            medium = evaluate(latest, f"medium-step-{step}", [
                "--skills-only", "--per-task", "256", "--receiving-min", "5", "--receiving-max", "8",
                *practice_flags,
            ])
            recovery = None
            if args.recovery_suite:
                recovery = evaluate(latest, f"recovery-step-{step}", [
                    "--skills-only", "--per-task", "256", "--recovery", "--request-suite",
                    *practice_flags,
                ])
            defense = {}
            for name, flags in (
                ("direct_ready", []),
                ("bank_ready", ["--bank-defense"]),
                ("bank_varied", ["--bank-defense", "--random-defense"]),
            ):
                tested = evaluate(latest, f"defense-{name}-step-{step}", [
                    "--skills-only", "--per-task", "512",
                    "--defense-min", "8", "--defense-max", "12",
                    *flags, *practice_flags,
                ])
                defense[name] = tested["skills"]["defense"]
            requests = {
                name: {
                    task: {key: values[key] for key in (
                        "trials", "contacted", "controlled", "controlled_or_aimed_fast",
                        "first_shots_on_requested_route", "mean_first_shot_speed",
                    )} for task, values in suite.items()
                } for name, suite in report["requested_skills"].items()
            }
            write_json(args.output_dir / "latest.json", {
                "checkpoint": str(latest), "report": f"step-{step}.json",
                "deployment_ready": False, "requested_skills": requests,
                "report_sensing": report.get("report_sensing", False),
                "entry_outcomes_by_speed": report["selfplay"]["entry_outcomes_by_speed"],
                "entry_speed_bins_m_s": report["selfplay"]["entry_speed_bins_m_s"],
                "peak_load": report["selfplay"]["peak_load"],
                "medium_receiving": medium["skills"]["receiving"],
                "outgoing_recovery": None if recovery is None else {
                    route: recovery["requested_skills"][route]["receiving"]
                    for route in ("straight", "left", "right")
                },
                "fast_defense": defense,
                "replay": f"http://localhost:8420/?replay={recording.name}",
                "latest_replay": f"http://localhost:8420/?replay={alias.name}",
            })
            seen.add(latest)
            print(json.dumps({"evaluated": str(latest), "requests": requests}), flush=True)
        status_path = args.run / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if not status.get("running", True):
                if "error" in status:
                    raise RuntimeError(status["error"])
                # The final checkpoint may have appeared while an older one
                # was being evaluated. Do not skip it on the completion edge.
                remaining = [p for p in args.run.glob("agent_step_*.pt") if p not in seen]
                if remaining and (latest is None or max(int(p.stem.rsplit("_", 1)[1]) for p in remaining)
                                  > int(latest.stem.rsplit("_", 1)[1])):
                    continue
                break
        time.sleep(10)
    if latest is not None:
        # A separate seed and hot load test; report failures, never reset heat
        # on overload or turn an experimental snapshot into a deployment default.
        evaluate(latest, "final-hot", [
            "--games-only", "--games", "8", "--seconds", "900",
            "--initial-load", ".95", "--thermal-gain", "1.3",
            "--audit-motion", "--seed", str(args.seed + 100),
        ])
        evaluate(latest, "final-heldout", [
            "--request-suite", "--per-task", "256", "--receiving-min", "2", "--receiving-max", "5",
            "--seconds", "180", "--games", "8", "--seed", str(args.seed + 101),
            *practice_flags,
        ])
        if args.recovery_suite:
            evaluate(latest, "final-recovery-heldout", [
                "--skills-only", "--per-task", "512", "--recovery", "--request-suite",
                "--seed", str(args.seed + 106), *practice_flags,
            ])
        evaluate(latest, "final-fast-defense", [
            "--skills-only", "--per-task", "256", "--bank-defense", "--random-defense",
            "--defense-min", "8", "--defense-max", "12", "--receiving-min", "6", "--receiving-max", "8",
            "--seed", str(args.seed + 102), *practice_flags,
        ])
        evaluate(latest, "final-medium-heldout", [
            "--skills-only", "--per-task", "1024",
            "--receiving-min", "5", "--receiving-max", "8",
            "--seed", str(args.seed + 104), *practice_flags,
        ])
        for name, flags in (
            ("direct-ready", []),
            ("bank-ready", ["--bank-defense"]),
            ("bank-varied", ["--bank-defense", "--random-defense"]),
        ):
            evaluate(latest, "final-defense-" + name, [
                "--skills-only", "--per-task", "1024",
                "--defense-min", "8", "--defense-max", "12",
                "--seed", str(args.seed + 105), *flags, *practice_flags,
            ])
            evaluate(latest, "final-defense-hot-" + name, [
                "--skills-only", "--per-task", "1024",
                "--defense-min", "8", "--defense-max", "12",
                "--initial-load", ".9", "--thermal-gain", "1.3",
                "--seed", str(args.seed + 105), *flags, *practice_flags,
            ])
        if args.reference is not None:
            for side in ("blue", "red"):
                evaluate(latest, "final-cross-" + side, [
                    "--games-only", "--games", "4", "--seconds", "180",
                    "--opponent-checkpoint", str(args.reference),
                    "--initial-load", ".8", "--thermal-gain", "1.3",
                    "--seed", str(args.seed + 103),
                    *(["--swap-sides"] if side == "red" else []),
                ])


if __name__ == "__main__":
    main()
