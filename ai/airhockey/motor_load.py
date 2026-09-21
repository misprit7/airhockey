"""Read motor-load recordings without opening a robot connection."""
from __future__ import annotations

import json
import math
from pathlib import Path


def records(path):
    """Ignore only an unfinished last line while the master is writing."""
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            yield json.loads(line)


def summarize(path):
    source = None
    last = None
    samples = 0
    first = None
    intervals = []
    acquisition = []
    nodes = [dict(node=i, rms_peak=None, rms_slow_peak=None, valid_rms_samples=0,
                  invalid_rms_samples=0, torque_abs_peak_amps=None) for i in range(4)]
    for record in records(path):
        if record.get("type") == "meta":
            source = record["source"]
        if record.get("type") != "motor_load":
            continue
        now = record["monotonic"]
        if first is None:
            first = now
        if last is not None:
            intervals.append(now - last["monotonic"])
        acquisition.append(record["acquisition_ms"])
        samples += 1
        for motor in record["motors"]:
            node = nodes[motor["node"]]
            # A backed-off failed/stale field is not a new observation. Keep
            # validity counts at snapshot level, peaks only from valid values.
            node["valid_rms_samples" if motor["rms_pct"]["valid"] else "invalid_rms_samples"] += 1
            for field, key in (("rms_pct", "rms_peak"), ("rms_slow_pct", "rms_slow_peak"),
                               ("torque_amps", "torque_abs_peak_amps")):
                value = motor[field]
                if value["valid"] and isinstance(value["value"], (int, float)) and math.isfinite(value["value"]):
                    measured = abs(value["value"]) if field == "torque_amps" else value["value"]
                    node[key] = measured if node[key] is None else max(node[key], measured)
        last = record
    duration = last["monotonic"] - first if last is not None else 0
    return dict(path=str(path), source=source, samples=samples, duration_s=duration,
                achieved_hz=(samples - 1) / duration if duration > 0 else None,
                max_sample_gap_s=max(intervals, default=None),
                max_acquisition_ms=max(acquisition, default=None),
                motors=nodes, latest_monotonic=last["monotonic"] if last else None,
                latest_context=last["context"] if last else None,
                dropped_command_events=last.get("dropped_command_events", 0) if last else 0)
