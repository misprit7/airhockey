"""Read motor-load recordings without opening a robot connection."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


def latest_snapshot(directory, *, now=None):
    """Read-only UI feed, also while another process owns the control socket.

    Bound the read to the final 256 KiB; never scan a growing recording from
    its beginning. Wall time identifies old sessions across host reboots;
    per-field acquisition offsets retain the logger's monotonic precision.
    """
    now = time.time() if now is None else now
    try:
        paths = list(Path(directory).glob('*.jsonl'))
        if not paths:
            return dict(state='unavailable', message='No motor telemetry recorded yet.', motors=[])
        path = max(paths, key=lambda p: p.stat().st_mtime_ns)
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            offset = max(0, size - 262144)
            stream.seek(offset)
            lines = stream.read(262144).split(b'\n')[:-1]
            if offset:
                lines = lines[1:]
        sample = None
        ended = False
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if record.get('type') == 'end':
                ended = True
            if record.get('type') == 'motor_load':
                sample = record
                break
        if sample is None:
            return dict(state='unavailable', source=path.name,
                        message='Waiting for motor telemetry.', motors=[])
        age = now - sample['unix']
        live = not ended and math.isfinite(age) and 0 <= age <= 1
        output = dict(state='live' if live else 'stale', source=path.name,
                      age_s=age if math.isfinite(age) else None,
                      context=sample.get('context', {}), motors=[])
        keys = ('torque_amps', 'rms_pct', 'rms_slow_pct', 'bus_voltage_v', 'status', 'alerts')
        for node in range(4):
            motor = next((m for m in sample.get('motors', []) if m.get('node') == node), {})
            fields = {}
            for key in keys:
                field = motor.get(key, {})
                value = field.get('value')
                values = value if isinstance(value, list) else [value]
                valid = field.get('valid') is True and bool(values) and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                end = field.get('end')
                field_age = age + sample['monotonic'] - end if isinstance(end, (int, float)) else math.inf
                fields[key] = dict(value=value if valid else None, valid=valid,
                                   age_s=field_age if math.isfinite(field_age) else None,
                                   fresh=bool(live and valid and 0 <= field_age <= .5))
            output['motors'].append(dict(node=node, **fields))
        return output
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return dict(state='unavailable', message=f'Motor telemetry unavailable: {exc}', motors=[])


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
