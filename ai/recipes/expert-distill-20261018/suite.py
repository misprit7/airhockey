#!/usr/bin/env python3
"""Parallel benchmark suite for this experiment's candidates. Simulation only.

    PYTHONPATH=ai:ai/bin python3 ai/recipes/expert-distill-20261018/suite.py \
        --candidate expert --candidate runs/<run>/agent.pt --out logs/neural-player/expert-20261018/suite/<tag>

Each candidate is a checkpoint path or an `expert[:k=v,...]` teacher spec.
`--tier dev` uses development seeds; `--tier holdout` uses the reserved
seeds (20261980-20261999 and 20262000+), larger cohorts, and is meant to be
run once on finalists. Jobs run as separate 2-thread processes.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = "runs/rail30-100-20261001-v1/agent.pt"

TIERS = {
    "dev": dict(defense=(768, 20261073), immediate=(768, 20261052), skills=(128, 20261011),
                control=(384, 20261072), fringe=(256, 20261015), games=(16, 60, 20261300),
                hot=(4, 600, 20261310), proxy=(64, 60, 20261320)),
    "holdout": dict(defense=(3072, 20261981), immediate=(1536, 20261983), skills=(256, 20261984),
                    control=(1536, 20261985), fringe=(512, 20261986), games=(48, 60, 20261987),
                    hot=(16, 900, 20261988), proxy=(128, 60, 20261989)),
}


def jobs(candidate, tier, out, include):
    t = TIERS[tier]
    tag = candidate.replace("/", "_").replace(":", "_").replace(",", "_").replace("=", "-")
    base = out / tag
    py = [sys.executable]
    run = []

    def add(name, argv):
        if include and name.split(":")[0] not in include:
            return
        run.append((name, argv, base / f"{name.replace(':', '-')}.json"))

    n, s = t["defense"]
    add("defense", py + ["ai/bin/eval_fast_defense.py", candidate, "--trials", str(n), "--seed", str(s)])
    n, s = t["immediate"]
    add("immediate", py + ["ai/bin/eval_fast_defense.py", candidate, "--trials", str(n), "--seed", str(s), "--immediate"])
    n, s = t["skills"]
    add("skills", py + ["ai/bin/eval_neural_player.py", candidate, "--skills-only", "--request-suite",
                         "--per-task", str(n), "--receiving-min", "3", "--receiving-max", "8", "--seed", str(s)])
    n, s = t["control"]
    add("control", py + ["ai/bin/eval_slow_control.py", candidate, "--trials", str(n), "--seed", str(s)])
    n, s = t["fringe"]
    add("fringe", py + ["ai/bin/eval_neural_fringe.py", candidate, "--trials", str(n), "--seed", str(s)])
    g, sec, s = t["games"]
    common = ["--games-only", "--games", str(g), "--seconds", str(sec), "--opponent-checkpoint", PRODUCTION,
              "--initial-load", "0.4", "--report-sensing", "--continuous-rallies", "--seed", str(s)]
    add("games:blue", py + ["ai/bin/eval_neural_player.py", candidate] + common)
    add("games:red", py + ["ai/bin/eval_neural_player.py", candidate] + common + ["--swap-sides"])
    g, sec, s = t["hot"]
    add("hot", py + ["ai/bin/eval_neural_player.py", candidate, "--games-only", "--games", str(g),
                      "--seconds", str(sec), "--opponent-checkpoint", PRODUCTION, "--initial-load", "0.8",
                      "--thermal-gain", "1.3", "--report-sensing", "--continuous-rallies", "--seed", str(s)])
    if (ROOT / "ai/bin/eval_human_proxy.py").exists():
        g, sec, s = t["proxy"]
        add("proxy", py + ["ai/bin/eval_human_proxy.py", candidate, "--tier", "both", "--games", str(g),
                            "--seconds", str(sec), "--seed", str(s)])
    return run


def execute(job):
    name, argv, output = job
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    env = dict(os.environ, PYTHONPATH="ai:ai/bin", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    start = time.time()
    with open(log, "w") as fh:
        code = subprocess.call(argv + ["--output", str(output)], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return name, output, code, time.time() - start


def summarize(results):
    """One flat dict of headline numbers per candidate."""
    row = {}
    for name, output, code, seconds in results:
        if code or not output.exists():
            row[name] = f"FAILED ({code})"
            continue
        r = json.loads(output.read_text())
        if name in ("defense", "immediate"):
            s = r["summary"]
            g = r["groups"]
            row[name] = dict(block=round(s["block_rate"], 3), straight=round(g["straight"]["block_rate"], 3),
                             left=round(g["left_bank"]["block_rate"], 3), right=round(g["right_bank"]["block_rate"], 3),
                             depth=round(s["mean_release_depth"] or 0, 3), overload=s["overload_seconds"])
        elif name == "skills":
            req = r.get("requested_skills", {})
            out = {}
            for route, sk in req.items():
                for cohort, values in sk.items():
                    if isinstance(values, dict) and "first_requested_shots_at_least_6m_s" in values:
                        trials = values.get("trials") or values.get("count") or 0
                        out[f"{route}:{cohort}"] = (values["first_requested_shots_at_least_6m_s"], trials)
            row[name] = out
        elif name == "control":
            s = r["summary"]
            row[name] = {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}
        elif name == "fringe":
            row[name] = {k: dict(recovered=v["recovered"], trials=v["trials"], overload=v["overload_seconds"])
                         for k, v in r["summary"].items()}
        elif name.startswith("games") or name == "hot":
            sp = r["selfplay"]
            me = sp["candidate_side"]
            score = [sum(x) for x in sp["score"]]
            row[name] = dict(goals_for=score[me], goals_against=score[1 - me],
                             on_target=sum(sp["on_target"][me]), shots=sum(sp["shots"][me]),
                             mean_shot_speed=round(sp["mean_shot_speed"][me], 2),
                             stall_s=round(sp["reachable_stall_player_seconds"][me], 1),
                             overload_s=round(sum(sp["overload_seconds"][me]), 1),
                             peak_load=round(max(sp["peak_load"][me]), 3))
        elif name == "proxy":
            row[name] = r.get("summary", {k: v for k, v in r.items() if k in ("club", "elite")})
        row[name + "_seconds"] = round(seconds)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tier", choices=sorted(TIERS), default="dev")
    p.add_argument("--only", action="append", default=[], help="Subset of benchmarks, e.g. --only defense --only games")
    p.add_argument("--parallel", type=int, default=10)
    a = p.parse_args()
    all_jobs = {c: jobs(c, a.tier, a.out, set(a.only)) for c in a.candidate}
    flat = [(c, j) for c, js in all_jobs.items() for j in js]
    with ThreadPoolExecutor(a.parallel) as pool:
        done = list(pool.map(lambda cj: (cj[0], execute(cj[1])), flat))
    table = {}
    for c in a.candidate:
        table[c] = summarize([r for cc, r in done if cc == c])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
