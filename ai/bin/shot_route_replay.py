#!/usr/bin/env python3
"""Replay a table session through the shot-route classifier. Read-only on logs.

    python ai/bin/shot_route_replay.py 20261110-203214

Feeds `airhockey.shot_selector.RouteClassifier` the policy's own logged
observation (`neural_obs` in `logs/run_policy/<stamp>.ticks.csv`: puck,
puck velocity, own mallet, and the request that was active) tick by tick,
as the runner does live, and prints the robot's shots by request and
realised route. With the session's `.replay.jsonl` beside it, every
resolved shot is also checked against the 200 Hz puck track: which side
rail, if any, the puck actually reached before the far end or the person
(`track`), a reading that does not go through the classifier's inputs.

`--governor` also runs `ShotSelector('auto')` over the realised routes to
show what it would have asked for (the requests it could not have changed
are the session's own, so this is its bookkeeping, not a counterfactual).
"""
import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ai"), str(ROOT / "shared")]

from airhockey.dynamics import table_mm_to_sim  # noqa: E402
from airhockey.shot_selector import ROUTE_NAMES, RouteClassifier, ShotSelector  # noqa: E402

FRAME = 42       # physical features per history frame; the newest frame is first


def ticks(path):
    """(t, puck, velocity, mallet, request) per tick the policy decided on."""
    csv.field_size_limit(10 ** 9)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("neural_obs"):
                continue
            obs = np.array(row["neural_obs"].split("|"), float)
            conditioned = len(obs) % FRAME == 3
            request = (int(np.argmax(obs[-3:])) + 1 if obs[-3:].any() else 0) if conditioned else 0
            yield float(row["t_cam"]), obs[0:2], obs[2:4] * 6, obs[4:6], request


def puck_track(path):
    """The 200 Hz puck fixes of a replay, robot-local sim metres."""
    times, points = [], []
    with open(path) as f:
        for line in f:
            if '"type":"frame"' not in line:
                continue
            frame = json.loads(line)
            if frame.get("puck"):
                times.append(frame["t"])
                points.append(table_mm_to_sim(*frame["puck"][:2]))
    return np.array(times), np.array(points).reshape(-1, 2)


def track_route(times, points, launch_t, *, radius=0.0407, tolerance=0.02, far_y=1.80):
    """Side rail the puck reached first after `launch_t`, from the raw track:
    1 / 2 = left / right rail before the far end, 3 = neither, 0 = the track
    ends (lost, or the puck turns back) before it gets there."""
    i = bisect.bisect_left(times, launch_t)
    top = -np.inf
    for k in range(i, min(i + 400, len(times))):
        if k > i and times[k] - times[k - 1] > 0.15:
            return 0
        x, y = points[k]
        if x <= radius + tolerance:
            return 1
        if x >= 1 - radius - tolerance:
            return 2
        if y >= far_y:
            return 3
        if y < top - 0.05:
            return 0
        top = max(top, y)
    return 0


def replay(stamp, log_dir, governor=False):
    base = Path(log_dir) / stamp
    classifier = RouteClassifier()
    selector = ShotSelector("auto", seed=0) if governor else None
    shots, how = [], {}
    request_at_launch = 0
    ticked = 0
    for t, puck, velocity, mallet, request in ticks(f"{base}.ticks.csv"):
        ticked += 1
        flying = classifier.flight is not None
        shot = classifier.update(t, puck, velocity, mallet)
        if classifier.flight is not None and not flying:
            request_at_launch = request
        if shot is None:
            continue
        how[shot["how"]] = how.get(shot["how"], 0) + 1
        shots.append(dict(shot, request=request_at_launch))
        if selector is not None and shot["route"]:
            selector.record(shot["route"], request_at_launch)
    table = np.zeros((4, 4), int)                 # [request, classified route]
    for s in shots:
        table[s["request"], s["route"]] += 1
    resolved = table[:, 1:]
    explicit = resolved[1:]
    result = dict(
        session=stamp, ticks=ticked, minutes=round(ticked * 0.02 / 60, 2),
        shots_detected=len(shots), resolved=int(resolved.sum()), dropped=int(table[:, 0].sum()), how=how,
        request_by_classified_route=dict(rows=list(ROUTE_NAMES), columns=list(ROUTE_NAMES), counts=table.tolist()),
        realised_share=dict(zip(ROUTE_NAMES[1:], (resolved.sum(0) / max(resolved.sum(), 1)).round(3).tolist())),
        request_honoured=dict(
            share=round(float(np.trace(explicit) / max(explicit.sum(), 1)), 3), n=int(explicit.sum()),
            by_request={ROUTE_NAMES[k + 1]: f"{explicit[k, k]}/{explicit[k].sum()}" for k in range(3)}))
    replay_path = Path(f"{base}.replay.jsonl")
    if replay_path.exists():
        times, points = puck_track(replay_path)
        versus = np.zeros((4, 4), int)            # [track route, classified route]
        for s in shots:
            versus[track_route(times, points, s["launch_t"] - 0.02), s["route"]] += 1
        both = versus[1:, 1:]
        result["track_check"] = dict(
            rows="rail reached on the 200 Hz track (none = track ends first)", columns="classified",
            counts=versus.tolist(), compared=int(both.sum()),
            agreement=round(float(np.trace(both) / max(both.sum(), 1)), 3))
    if selector is not None:
        result["governor_bookkeeping"] = selector.stats()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stamp", nargs="+", help="session stamp(s), e.g. 20261110-203214")
    p.add_argument("--log-dir", default=str(ROOT / "logs" / "run_policy"))
    p.add_argument("--governor", action="store_true", help="also run the auto selector's bookkeeping over the shots")
    a = p.parse_args()
    for stamp in a.stamp:
        print(json.dumps(replay(stamp, a.log_dir, a.governor), indent=1))


if __name__ == "__main__":
    main()
