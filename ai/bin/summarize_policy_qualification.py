#!/usr/bin/env python3
"""Render completed simulation qualification evidence; no inference or hardware."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qualification", type=Path)
    args = p.parse_args()
    root = args.qualification
    summary = json.loads((root / "summary.json").read_text())
    if not summary["completed"]:
        raise ValueError("qualification is still running")
    skills = summary["skills"]["skills"]
    stress = summary["skills"]["stress"]["defense"]
    labels = [
        "Stationary goals",
        "Moving goals",
        "Cushioning",
        "Home defense",
        "Random-pose defense",
    ]
    rows = [skills[k] for k in ["stationary", "moving", "cushion", "defense"]] + [
        stress
    ]
    for key, label in [
        ("fast_defense", "Fast direct (8–12 m/s)"),
        ("bank_defense", "Fast bank (8–12 m/s)"),
    ]:
        if key in summary["skills"]:
            labels.append(label)
            rows.append(summary["skills"][key]["defense"])
    rates = [100 * r["successes"] / r["attempts"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    bars = ax.barh(labels, rates, color="#26868b")
    ax.bar_label(
        bars,
        labels=[
            f"{r['successes']}/{r['attempts']} ({v:.1f}%)" for r, v in zip(rows, rates)
        ],
        padding=4,
        fontsize=9,
    )
    ax.set_xlim(0, 124)
    ax.set_xlabel("Completion rate (%)")
    ax.invert_yaxis()
    ax.set_title("Unseen skill trials")
    ax = axes[0, 1]
    names = ["forward", "reversed"]
    tick_labels = ["First side\n8 × 180 s", "Opposite side\n8 × 180 s"]
    if "hot_forward" in summary["matches"]:
        names += ["hot_forward", "hot_reversed"]
        tick_labels += ["Hot first\n4 × 180 s", "Hot opposite\n4 × 180 s"]
    matches = [summary["matches"][k] for k in names]
    x = np.arange(len(matches))
    ax.bar(
        x - 0.18,
        [m["goals_for"] for m in matches],
        0.36,
        label="Candidate",
        color="#26868b",
    )
    ax.bar(
        x + 0.18,
        [m["goals_against"] for m in matches],
        0.36,
        label="Old MPC",
        color="#b87946",
    )
    ax.set_xticks(x, tick_labels, fontsize=9)
    ax.set_ylabel("Goals per test set")
    ax.set_title("Held-out reference matches")
    ax.legend()
    long = json.loads((root / "continuous_selfplay.json").read_text())
    trace = long["thermal_trace"]
    minutes = np.array([t["seconds"] for t in trace]) / 60
    ax = axes[1, 0]
    for key, color in [("fast", "#26868b"), ("slow", "#7953a5")]:
        level = np.array([t[key] for t in trace]).reshape(len(trace), -1)
        ax.plot(minutes, level, color=color, alpha=0.15, linewidth=0.6)
        ax.plot(
            minutes,
            level.max(1),
            color=color,
            label=key.capitalize() + " maximum",
            linewidth=1.5,
        )
    ax.axhline(1, color="#bb3434", linestyle="--", label="Modeled overload")
    ax.axhline(0.95, color="#aa7722", linestyle=":", label="Qualification margin")
    ax.set(
        xlabel="Continuous simulated minutes",
        ylabel="Normalized modeled RMS load",
        ylim=(0, 1.08),
    )
    ax.set_title(
        f"Continuous self-play: peak {summary['matches']['continuous_selfplay']['peak_load']:.3f}"
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    fractions = [[], []]
    for name, candidate_side in [("forward", 0), ("reversed", 1)]:
        r = json.loads((root / f"{name}.json").read_text())
        fractions[0].extend(r["fraction_above_40m_s2"][candidate_side])
        fractions[1].extend(r["fraction_above_40m_s2"][1 - candidate_side])
    values = [100 * np.mean(v) for v in fractions]
    bars = ax.bar(["Candidate", "Old MPC"], values, color=["#26868b", "#b87946"])
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_ylabel("Time above 40 m/s² (%)")
    ax.set_ylim(0, max(values) * 1.2 + 0.2)
    ax.set_title("Actual acceleration duty in cold reference games")
    verdict = (
        "PASS"
        if summary["passed"]
        else "NOT QUALIFIED: " + ", ".join(summary["failures"])
    )
    fig.suptitle(
        f"Simulation qualification v{summary['manifest'].get('suite_version', 1)} — "
        + verdict
        + "\nProvisional motor-load model; no physical deployment certification",
        fontsize=12,
    )
    fig.savefig(root / "overview.png", dpi=160)
    plt.close(fig)
    print(root / "overview.png")


if __name__ == "__main__":
    main()
