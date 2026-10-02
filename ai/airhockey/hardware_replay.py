"""Read-only hardware replay using the environment's physics and motion law.

Only the initial measured state, issued commands, and human trajectory feed
the rollout. Subsequent puck/robot measurements are used for scoring only.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from airhockey.batch_env import BatchAirHockeyEnv
from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.dynamics import table_mm_to_sim
from airhockey.physics import TableConfig

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "run_policy"
GAP = 0.150
DT = 0.002


def _number(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _point(x, y):
    x, y = _number(x), _number(y)
    return None if x is None or y is None else list(table_mm_to_sim(x, y))


def safe_path(name: str, directory: Path = LOG_DIR) -> Path:
    if Path(name).name != name or not name.endswith((".ticks.csv", ".replay.jsonl")):
        raise ValueError("Invalid recording name")
    p = (directory / name).resolve()
    if p.parent != directory.resolve() or not p.is_file():
        raise FileNotFoundError(name)
    return p


def list_sessions(directory: Path = LOG_DIR):
    files = list(directory.glob("*.replay.jsonl")) + list(directory.glob("*.ticks.csv"))
    rich = {
        p.name.removesuffix(".replay.jsonl")
        for p in files
        if p.name.endswith(".replay.jsonl")
    }
    out = []
    for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        if p.name.endswith(".ticks.csv") and p.name.removesuffix(".ticks.csv") in rich:
            continue
        if p.stat().st_size < 400:
            continue
        out.append(
            dict(
                name=p.name,
                size=p.stat().st_size,
                format="full camera log"
                if p.name.endswith(".jsonl")
                else "legacy decision log",
            )
        )
    return out


def load_session(name: str, directory: Path = LOG_DIR):
    p = safe_path(name, directory)
    st = p.stat()
    return _load(str(p), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=4)
def _load(filename, mtime, size):
    p = Path(filename)
    tracks = {k: [] for k in ("puck", "agent", "human")}
    commands, warnings, meta = [], [], {}
    events = []
    timestamps = []
    clock_offsets, command_monos, event_monos = [], [], []

    def add(kind, t, point):
        if point is not None and t is not None:
            tracks[kind].append([t, *point])

    if p.name.endswith(".replay.jsonl"):
        for line in p.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a live writer may have a partial last row
                continue
            kind, t = r.get("type"), _number(r.get("t"))
            if kind == "meta":
                meta = r
            elif (
                kind == "clock"
                and t is not None
                and _number(r.get("monotonic")) is not None
            ):
                clock_offsets.append(r["monotonic"] - t)
            elif kind == "frame" and t is not None:
                timestamps.append(t)
                for k in tracks:
                    xy = r.get(k)
                    if xy is not None:
                        add(k, t, _point(*xy))
            elif kind == "command" and t is not None:
                xy = _point(r.get("x"), r.get("y"))
                v, a = _number(r.get("speed")), _number(r.get("accel"))
                if (
                    xy is not None
                    and v is not None
                    and a is not None
                    and v > 0
                    and a > 0
                ):
                    commands.append([t, *xy, v / 1000, a / 1000])
                    command_monos.append(_number(r.get("monotonic")))
            elif kind in ("stop", "end") and t is not None:
                events.append(dict(t=t, type=kind))
                event_monos.append(_number(r.get("monotonic")))
        if clock_offsets:
            offset = min(clock_offsets) - float(meta.get("camera_delay_s", 0.0077))
            for cmd, mono in zip(commands, command_monos):
                if mono is not None:
                    cmd[0] = mono - offset
            for event, mono in zip(events, event_monos):
                if mono is not None:
                    event["t"] = mono - offset
        if meta.get("live") is not True:
            warnings.append("Dry-run recording: commands were not sent to the robot.")
        warnings.append(
            "Command/camera clock alignment uses the measured 7.7 ms camera delay; adjust the timing offset to test sensitivity."
        )
    else:
        warnings.append(
            "Legacy recording: decision-rate camera samples, no observations during puck-loss holds, and approximate command timing. A new session records full replay data automatically."
        )
        log = p.with_name(p.name.replace(".ticks.csv", ".log"))
        if log.exists():
            for line in log.open():
                if "   args " in line:
                    try:
                        meta = ast.literal_eval(line.split("   args ", 1)[1])
                    except (ValueError, SyntaxError):
                        pass
                    break
        if meta.get("live") is False:
            warnings.append("Dry-run recording: commands were not sent to the robot.")
        rows = list(csv.DictReader(p.open()))
        offsets = [
            float(r["t_wall"]) - float(r["t_cam"])
            for r in rows
            if r.get("t_wall") and r.get("t_cam")
        ]
        # Closest available legacy estimate: wall time was captured after
        # the command completed. Retain this uncertainty in the UI.
        offset = min(offsets) if offsets else 0
        for r in rows:
            t = _number(r.get("t_cam"))
            if t is None:
                continue
            timestamps.append(t)
            age = _number(r.get("puck_age_ms"))
            if age is not None and age <= 150:
                add("puck", t - age / 1000, _point(r.get("puck_x"), r.get("puck_y")))
            add("agent", t, _point(r.get("cam_mallet_x"), r.get("cam_mallet_y")))
            add("human", t, _point(r.get("opp_x"), r.get("opp_y")))
            xy = _point(r.get("cmd_x"), r.get("cmd_y"))
            v = _number(r.get("limits_speed") or r.get("cmd_speed"))
            a = _number(r.get("cmd_accel"))
            if xy is not None and v is not None and a is not None and v > 0 and a > 0:
                ct = float(r["t_wall"]) - offset if r.get("t_wall") else t
                commands.append([ct, *xy, v / 1000, a / 1000])

    if not timestamps:
        raise ValueError("Recording has no camera samples")
    origin, end = min(timestamps), max(timestamps)
    for k, samples in tracks.items():
        # Last sample wins when a decision log repeats a camera fix.
        samples = {round(t, 6): xy for t, *xy in samples}
        tracks[k] = [[round(t - origin, 6), *xy] for t, xy in sorted(samples.items())]
    commands = [[t - origin, *values] for t, *values in sorted(commands)]
    events = [dict(e, t=e["t"] - origin) for e in events]
    # Replay the recorded configuration, not today's simulation defaults.
    cfg = TableConfig(**meta.get('table_config', {}))
    bounds = meta.get('workspace_bounds_mm')
    if bounds is None:
        checkpoint = meta.get('checkpoint')
        if checkpoint and Path(checkpoint).is_file():
            from airhockey.neural_setup import checkpoint_environment
            bounds = checkpoint_environment(checkpoint)['workspace_bounds_mm']
        else:
            from airhockey.neural_setup import workspace_bounds
            bounds = workspace_bounds('legacy')
    return dict(
        name=p.name,
        duration=end - origin,
        tracks=tracks,
        commands=commands,
        events=events,
        warnings=warnings,
        meta=dict(
            policy=meta.get("policy", "unknown"),
            live=meta.get("live"),
            ramp_s=float(meta.get("ramp", 3)) / 1000,
            table_config=meta.get('table_config', {}),
            workspace_bounds_mm=bounds,
        ),
        table=dict(
            width=cfg.width,
            height=cfg.height,
            goal_width=cfg.goal_width,
            puck_radius=cfg.puck_radius,
            paddle_radius=cfg.paddle_radius,
        ),
    )


def sample(track, times, gap=GAP):
    """Interpolate measured motion only across short, observed intervals."""
    times = np.atleast_1d(times)
    out = np.full((len(times), 2), np.nan)
    if not len(track):
        return out
    a = np.asarray(track, dtype=float)
    ix = np.searchsorted(a[:, 0], times + 1e-9, side="right") - 1
    lo = np.clip(ix, 0, len(a) - 1)
    hi = np.minimum(lo + 1, len(a) - 1)
    span = a[hi, 0] - a[lo, 0]
    exact = np.abs(times - a[lo, 0]) < 1e-6
    after = (lo == len(a) - 1) & (times - a[lo, 0] <= gap)
    between = (hi > lo) & (span <= gap) & (times <= a[hi, 0])
    valid = (ix >= 0) & (exact | after | between)
    frac = np.divide(times - a[lo, 0], span, out=np.zeros(len(times)), where=span > 0)
    out[valid] = (a[lo, 1:] + np.clip(frac, 0, 1)[:, None] * (a[hi, 1:] - a[lo, 1:]))[
        valid
    ]
    out[exact] = a[lo[exact], 1:]
    return out


def initial_velocity(track, t):
    """Causal 40 ms position fit, solely to initialize a new rollout."""
    a = np.asarray(track, dtype=float)
    if not len(a):
        return np.zeros(2)
    a = a[(a[:, 0] <= t + 1e-6) & (a[:, 0] >= t - 0.04)]
    if len(a) < 2 or np.ptp(a[:, 0]) < 0.004:
        return np.zeros(2)
    x = a[:, 0] - a[:, 0].mean()
    return np.clip(
        (x[:, None] * (a[:, 1:] - a[:, 1:].mean(axis=0))).sum(axis=0) / (x @ x), -12, 12
    )


class _ReplayPhysics(BatchPhysicsEngine):
    def _reset_puck_subset(self, mask, toward_agent, rng):
        # No invented serve after a simulated goal. The rollout stops there.
        pass


def simulate(session, starts, duration=10.0, command_offset_ms=0.0):
    """Batch independent rollouts; observations after t0 cannot steer either
    simulated puck or robot. Each output row: t, puck x/y, robot x/y.
    """
    if not starts or len(starts) > 32 or not 0 < duration <= 30:
        raise ValueError("Use 1–32 starts and a duration between 0 and 30 seconds")
    starts = np.asarray(starts, dtype=float)
    if (
        not np.isfinite(starts).all()
        or np.any(starts < 0)
        or np.any(starts > session["duration"])
    ):
        raise ValueError("Start time is outside the recording")
    n = len(starts)
    env = BatchAirHockeyEnv(n_envs=n, domain_randomize=False,
        table_config=TableConfig(**session['meta'].get('table_config', {})),
        workspace_bounds_mm=session['meta'].get('workspace_bounds_mm'))
    e = env.engine = _ReplayPhysics(n, env.table_config)
    dyn = env._agent_dyn
    dyn["ramp_s"] = session["meta"]["ramp_s"]
    tracks = session["tracks"]
    initial = {
        k: sample(tracks[k], starts, gap=0.05 if k == "puck" else GAP) for k in tracks
    }
    active = np.ones(n, dtype=bool)
    results = [
        dict(start=float(t), frames=[], reason="window complete", errors={})
        for t in starts
    ]
    for k, positions in initial.items():
        bad = ~np.isfinite(positions).all(axis=1)
        for i in np.flatnonzero(bad):
            results[i]["reason"] = f"No fresh {k} measurement at start"
        active &= ~bad
    for key, field in [
        ("puck", "puck"),
        ("agent", "paddle_agent"),
        ("human", "paddle_opp"),
    ]:
        pos = np.nan_to_num(initial[key])
        vel = np.array([initial_velocity(tracks[key], t) for t in starts])
        for j, axis in enumerate(("x", "y")):
            getattr(e, f"{field}_{axis}")[:] = pos[:, j]
            getattr(e, f"{field}_v{axis}")[:] = vel[:, j]
            if key == "agent":
                dyn[axis][:] = pos[:, j]
                dyn["v" + axis][:] = vel[:, j]
    commands = np.asarray(session["commands"], dtype=float)
    if not len(commands):
        for r in results:
            r["reason"] = "No recorded commands"
        return results
    command_times = commands[:, 0] + command_offset_ms / 1000
    event_times = [x["t"] for x in session["events"] if x["type"] in ("stop", "end")]
    event_end = np.array(
        [
            min([t for t in event_times if t >= s] + [session["duration"]])
            for s in starts
        ]
    )

    def emit(elapsed):
        for i in np.flatnonzero(active):
            results[i]["frames"].append(
                [
                    round(float(starts[i] + elapsed), 6),
                    float(e.puck_x[i]),
                    float(e.puck_y[i]),
                    float(e.paddle_agent_x[i]),
                    float(e.paddle_agent_y[i]),
                ]
            )

    emit(0)
    for tick in range(int(round(duration / DT))):
        elapsed = tick * DT
        ts = starts + elapsed
        human = sample(tracks["human"], ts + DT)
        missing = active & ~np.isfinite(human).all(axis=1)
        ended = active & (ts + DT > event_end + 1e-6)
        for i in np.flatnonzero(missing | ended):
            results[i]["reason"] = (
                "Human tracking gap"
                if missing[i]
                else "Recording ended or stop recorded"
            )
        active &= ~(missing | ended)
        if not active.any():
            break
        ix = np.searchsorted(command_times, ts, side="right") - 1
        cmd = commands[np.clip(ix, 0, len(commands) - 1)]
        # Before the first command, the robot holds the initialized pose.
        tx = np.where(ix >= 0, cmd[:, 1], initial["agent"][:, 0])
        ty = np.where(ix >= 0, cmd[:, 2], initial["agent"][:, 1])
        ws = env._ws
        tx = np.clip(np.nan_to_num(tx), ws["min_x"], ws["max_x"])
        ty = np.clip(np.nan_to_num(ty), ws["min_y"], ws["max_y"])
        ax, ay = env._update_dynamics(
            dyn, tx, ty, DT, speed_cap=cmd[:, 3], accel_cap=cmd[:, 4]
        )
        e.update_paddle_agent(ax, ay, DT)
        human = np.nan_to_num(human)
        e.update_paddle_opponent(human[:, 0], human[:, 1], DT)
        e.step(DT)
        goal = active & (e.goal_scored != 0)
        if (tick + 1) % 10 == 0 or goal.any():
            emit(elapsed + DT)
        for i in np.flatnonzero(goal):
            results[i]["reason"] = "Simulated goal — no artificial serve"
        active &= ~goal

    # Only here, after the rollout, inspect the measured puck and robot.
    for r in results:
        f = np.asarray(r["frames"])
        if not len(f):
            continue
        for name, cols in [("puck", slice(1, 3)), ("agent", slice(3, 5))]:
            real = sample(tracks[name], f[:, 0], gap=0.05)
            err = np.linalg.norm(real - f[:, cols], axis=1) * 1000
            valid = err[np.isfinite(err)]
            r["errors"][name] = (
                dict(
                    mean_mm=float(valid.mean()),
                    max_mm=float(valid.max()),
                    samples=len(valid),
                )
                if len(valid)
                else None
            )
    return results
