#!/usr/bin/env python3
"""Plot comparable automatic development evaluations; no policy selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(path):
    report = json.loads(path.read_text())
    game = report["selfplay"]
    entries = np.sum(game["entry_outcomes"], axis=0)
    minutes = 2 * game["games"] * game["seconds"] / 60
    shots = np.sum(game["shots"])
    defense = report["skills"]["defense"]
    return dict(
        run=Path(report["checkpoint"]).parent.name,
        sampling="sampled" if report.get("stochastic", False) else "mean",
        step=report["step"],
        report=str(path),
        metrics_version=report["metrics_version"],
        physics_seeded=report.get("physics_seeded", False),
        arrival_solve=report.get("arrival_solve", "legacy-solve"),
        seed=report["seed"],
        games=game["games"],
        seconds=game["seconds"],
        random_practice_opponent=report.get("random_practice_opponent", False),
        defense_bank=defense["bank"],
        defense_speed_range=defense["speed_range"],
        controlled_visits=100 * entries[2] / max(1, entries[0]),
        aimed_visits=100 * entries[3] / max(1, entries[0]),
        powerful_aimed_visits=100 * entries[4] / max(1, entries[0]),
        aimed_shots=100 * np.sum(game["on_target"]) / max(1, shots),
        mean_shot_speed=np.average(
            game["mean_shot_speed"], weights=np.sum(game["shots"], axis=1)
        )
        if shots
        else 0,
        peak_load=np.max(game["peak_load"]),
        overload_seconds=np.sum(game["overload_seconds"]),
        bank_saves=100 * defense["saved"] / max(1, defense["trials"]),
        turnovers_per_player_minute=np.sum(game["turnovers"]) / minutes,
        goals_per_player_minute=np.sum(game["score"]) / minutes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("logs/neural-player"))
    parser.add_argument("--metrics-version", type=int, default=4)
    parser.add_argument("--run", action="append", help="Include this run (repeatable)")
    parser.add_argument("--sampling", choices=("mean", "sampled", "both"), default="both")
    parser.add_argument(
        "--output", type=Path, default=Path("logs/neural-player/progress.png")
    )
    args = parser.parse_args()
    rows = [summarize(p) for p in sorted(args.directory.glob("*-development.json"))]
    rows = [r for r in rows if r["metrics_version"] == args.metrics_version]
    rows = [r for r in rows if not args.run or r["run"] in args.run]
    rows = [r for r in rows if args.sampling == "both" or r["sampling"] == args.sampling]
    if not rows:
        raise SystemExit("No automatic development reports yet")
    rows.sort(key=lambda r: (r["run"], r["step"]))
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        ("aimed_visits", "Own-half visits with an on-target launch (%)"),
        ("controlled_visits", "Own-half visits with controlled possession (%)"),
        ("mean_shot_speed", "Mean launch speed (m/s)"),
        ("peak_load", "Peak modeled motor load (limit = 1)"),
        ("bank_saves", "Verified 8–12 m/s bank saves (%)"),
        ("turnovers_per_player_minute", "Turnovers per player-minute"),
    ]
    runs = sorted({(r["run"], r["sampling"], r["arrival_solve"]) for r in rows})
    for run, sampling, solver in runs:
        cohort = [
            r
            for r in rows
            if (r["run"], r["sampling"], r["arrival_solve"]) == (run, sampling, solver)
        ]
        assert all(
            r["defense_bank"]
            and r["defense_speed_range"] == [8.0, 12.0]
            and r["random_practice_opponent"]
            for r in cohort
        )
        label = run.removeprefix("_neural-player-") + " / " + sampling
        if solver == "legacy-solve":
            label += " / old solve"
        for ax, (key, title) in zip(axes.flat, panels):
            ax.plot(
                [r["step"] / 1e6 for r in cohort],
                [r[key] for r in cohort],
                marker="o",
                label=label,
            )
            ax.set(title=title, xlabel="Cumulative learned transitions (millions)")
            ax.grid(alpha=0.25)
    axes[1, 0].axhline(1, color="red", linestyle="--", linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_rows = (len(labels) + 2) // 3
    legend_height = 0.025 * legend_rows + 0.035
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        f"Development evaluations v{args.metrics_version} • 8 × 180 s continuous self-play per checkpoint"
    )
    fig.tight_layout(rect=(0, legend_height, 1, 0.97))
    fig.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
