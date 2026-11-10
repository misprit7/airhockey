"""Physical-playability gates, one definition for the sim and for table logs.

A candidate that wins in simulation can still be unplayable on the machine:
runs/teacher-calm-20261018-v4 beat the people on the table but asked for more
than 80 m/s^2 on 13% of its ticks, reversed at speed 32 times a minute and
three times ended 220-400 mm from where it was commanded. These gates put
numbers on that. Every one is computed from a TRACE -- what a 50 Hz log of a
game holds, and nothing else -- so the same function scores a simulated game
(SimRecorder, engine truth sampled once per tick) and a table session
(table_trace, logs/run_policy/<stamp>.ticks.csv + .replay.jsonl). No hardware
I/O: the table mode only reads logs.

A trace is a dict of arrays over the ticks of one game or session, in metres
on the table (x across, y from the robot's goal line; `metres`):
  t          tick time, s
  policy     the policy drove this tick (table: not paused / returning home)
  puck       [T, 2], NaN while unseen;  puck_v [T, 2] m/s, NaN when unknown
  rival      the far mallet's speed, m/s (0 when unseen)
  cap        the accel cap commanded, m/s^2
  target     [T, 2] the commanded position
  mallet     [T, 2] the commanded profile's position (the controller's on the
             table, the cart in the sim);  mallet_v [T, 2] its velocity
  back_y     the workspace's back limit (scalar)

Gates (gates()):
  idle cap p50 / p90        cap on idle ticks: puck on the far half, puck and
                            rival mallet both under 0.6 m/s
  far-half cap > 40, > 80   share of ticks with the puck on the far half
  all ticks cap > 80        share of all policy ticks
  accel duty > 60, rms      |dv/dt| of the profile between consecutive ticks
  per minute                of policy time: each policy tick counts for the
                            time to the next tick (20 ms nominal; under 50 ms)
  reversals / min           the profile's velocity reversed within 100 ms
                            (v . v_100ms_ago < 0) with at least 1.5 (3) m/s on
                            both sides; events 150 ms apart
  incoming >= 5 m/s         per puck crossing the centre line toward the robot
                            at >= 5 m/s: how far FORWARD the mallet went from
                            where it was 100 ms before the crossing, until the
                            puck turned back or 0.5 s passed; and the mallet's
                            speed at the puck's closest seen approach (within
                            0.20 m), the nearest a 50 Hz log gets to contact
  back-edge dwell           share of ticks within 20 mm of the back limit
  modeled load              thermal.MotorThermal driven by the trace at tick
                            rate: peak level, seconds at or above 0.95, and
                            the level this play would settle at (the time-mean
                            current, which does not depend on session length)
  idle command change       target change per tick on idle ticks, mm

Both sources are aligned the same way: a tick is the puck, rival and mallet
the policy decided on and the command it then sent (SimRecorder samples the
engine before each step; a ticks.csv row is the fix, the controller sample
and the command of one loop pass).

Known differences between the two sources, stated rather than hidden: the
sim's puck and rival are engine truth while the table's are camera fixes
(unseen ticks drop out of the idle and far-half sets); the table's policy is
paused while the puck is lost and after goals, the sim's never is; the sim's
cart follows its command exactly, the table's mallet does not (the gates use
the commanded profile on both, so they measure what was ASKED of the machine).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from airhockey.dynamics import _geom as geom
from airhockey.thermal import MotorThermal

# Metres per sim unit: sim x spans the rails across, sim y the robot's half.
SCALE = np.array([(geom.RAIL_MAX_Y - geom.RAIL_MIN_Y) / 1000, (geom.RAIL_MAX_X - geom.CENTERLINE_X) / 1000])
CENTRE_Y = float(SCALE[1])
TICK_S = 0.02
IDLE_SPEED_M_S = 0.6
REVERSAL_WINDOW_TICKS = 5
REVERSAL_GAP_S = 0.15
INCOMING_SPEED_M_S = 5.0
INCOMING_WINDOW_S = 0.5
INCOMING_LEAD_TICKS = 5
CONTACT_RANGE_M = 0.20
BACK_EDGE_M = 0.020
LOAD_LEVEL = 0.95
LOG_DIR = Path("logs/run_policy")


def metres(sim_xy):
    """Sim coordinates (or velocities) -> metres on the table, same axes."""
    return np.asarray(sim_xy, float) * SCALE


def table_metres(x_mm, y_mm):
    """Table mm (x along from the person's end, y across) -> the trace frame."""
    return np.stack(((np.asarray(y_mm, float) - geom.RAIL_MIN_Y) / 1000,
                     (geom.RAIL_MAX_X - np.asarray(x_mm, float)) / 1000), axis=-1)


# ------------------------------------------------------------------ sources
class SimRecorder:
    """Collects one trace per game from a NeuralTrainingEnv.

    Create it after the reset and call it after every step. A tick pairs the
    command of a step with the table as it stood BEFORE that step, which is
    what a ticks.csv row holds: the puck fix and the controller sample the
    policy decided on, then the command it sent.
    """

    def __init__(self, env):
        self.env = env
        self.states = [self._state()]
        self.commands = []

    def _state(self):
        e, dyn = self.env.engine, self.env.base._agent_dyn
        return np.column_stack((
            e.puck_x, e.puck_y, e.puck_vx, e.puck_vy, np.hypot(e.paddle_opp_vx * SCALE[0], e.paddle_opp_vy * SCALE[1]),
            dyn["x"], dyn["y"], dyn["vx"], dyn["vy"]))

    def __call__(self):
        dyn = self.env.base._agent_dyn
        self.commands.append(np.column_stack((dyn["command_accel"], dyn["command_x"], dyn["command_y"])))
        self.states.append(self._state())

    def traces(self):
        state = np.stack(self.states[:-1])             # [T, N, 9], before each step
        command = np.stack(self.commands)              # [T, N, 3], the step's own
        ticks = len(command)
        back = float(self.env.decoder.low[1] * SCALE[1])
        return [dict(t=np.arange(ticks) * TICK_S, policy=np.ones(ticks, bool),
                     puck=metres(state[:, i, 0:2]), puck_v=metres(state[:, i, 2:4]), rival=state[:, i, 4],
                     cap=command[:, i, 0], target=metres(command[:, i, 1:3]), mallet=metres(state[:, i, 5:7]),
                     mallet_v=metres(state[:, i, 7:9]), back_y=back)
                for i in range(command.shape[1])]


def _column(rows, name):
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        value = row.get(name)
        if value not in (None, ""):
            try:
                out[i] = float(value)
            except ValueError:
                pass
    return out


def table_trace(stamp, log_dir=LOG_DIR):
    """One session's trace from <stamp>.ticks.csv and <stamp>.replay.jsonl. Read-only."""
    log_dir = Path(log_dir)
    with open(log_dir / f"{stamp}.ticks.csv", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("t_cam") not in (None, "")]
    t = _column(rows, "t_cam")
    flags = [row.get("flags") or "" for row in rows]
    # Paused ticks (puck lost, returning home, holding) are the runner's, not the policy's.
    policy = np.array([not flag.startswith("puck_") for flag in flags]) & np.isfinite(_column(rows, "cmd_x"))
    puck = table_metres(_column(rows, "puck_x"), _column(rows, "puck_y"))
    age = _column(rows, "puck_age_ms")
    dt = np.diff(t, prepend=np.nan)
    fresh = (age <= 25) & (np.roll(age, 1) <= 25) & (dt > 0) & (dt < 0.05)
    fresh[0] = False
    puck[~(age <= 25)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        puck_v = np.where(fresh[:, None], (puck - np.roll(puck, 1, axis=0)) / dt[:, None], np.nan)
        opp = table_metres(_column(rows, "opp_x"), _column(rows, "opp_y"))
        rival = np.linalg.norm(opp - np.roll(opp, 1, axis=0), axis=1) / dt
    rival[0] = np.nan
    rival = np.where(np.isfinite(rival) & (dt < 0.05), rival, 0.0)
    mallet = table_metres(_column(rows, "ctl_mallet_x"), _column(rows, "ctl_mallet_y"))
    # The controller's own velocity samples (one query per tick) from the replay.
    samples, meta = [], {}
    with open(log_dir / f"{stamp}.replay.jsonl") as handle:
        for line in handle:
            if line.startswith('{"type":"controller"'):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                samples.append((r["t"], r["sample_monotonic"], r["x"], r["y"], r["vx"], r["vy"]))
            elif line.startswith('{"type":"meta"'):
                meta = json.loads(line)
    samples = np.array(samples, float)
    if len(samples) < 2:
        raise ValueError(f"{stamp}: no controller samples in the replay")
    samples = samples[np.argsort(samples[:, 0])]
    nearest = np.clip(np.searchsorted(samples[:, 0], t), 1, len(samples) - 1)
    nearest -= (t - samples[nearest - 1, 0]) < (samples[nearest, 0] - t)
    matched = np.abs(samples[nearest, 0] - t) < 0.015
    # Velocity is d(table_metres)/dt: across = +vy_mm, along = -vx_mm.
    mallet_v = np.where(matched[:, None], np.column_stack((samples[nearest, 5], -samples[nearest, 4])) / 1000, np.nan)
    sample_time = np.where(matched, samples[nearest, 1], np.nan)
    bounds = meta.get("workspace_bounds_mm") or [geom.WS_PROBE_MIN_X, geom.WS_PROBE_MAX_X,
                                                 geom.WS_PROBE_MIN_Y, geom.WS_PROBE_MAX_Y]
    trace = dict(t=t, policy=policy, puck=puck, puck_v=puck_v, rival=rival,
                 cap=_column(rows, "cmd_accel") / 1000,
                 target=table_metres(_column(rows, "cmd_x"), _column(rows, "cmd_y")),
                 mallet=mallet, mallet_v=mallet_v, sample_time=sample_time,
                 back_y=float((geom.RAIL_MAX_X - bounds[1]) / 1000),
                 stamp=str(stamp), policy_name=meta.get("policy"), thermal_model=meta.get("motor_profile"))
    # Drive telemetry as the policy saw it (8 load inputs after 15 base + 6 action).
    measured = np.full((len(rows), 8), np.nan)
    for i, row in enumerate(rows):
        text = row.get("neural_obs")
        if text:
            values = text.split("|")
            if len(values) >= 29:
                measured[i] = [float(v) for v in values[21:29]]
    trace["measured_load"] = measured
    return trace


# -------------------------------------------------------------------- gates
def _starts(times, gap):
    out = []
    for value in times:
        if not out or value - out[-1] > gap:
            out.append(value)
    return out


def modeled_load(trace, thermal_model, *, gain=1.0, initial=None):
    """The sim's motor-load model driven by a trace at tick rate.

    Returns (peak level, seconds at or above LOAD_LEVEL, settling level). The
    last is sqrt of the time-mean squared current over the fast limit: where
    the fast memory would settle if this play went on for ever.
    """
    model = MotorThermal(1, thermal_model, randomize=False) if thermal_model is not None else MotorThermal(
        1, randomize=False)
    model.gain[:] = gain
    if initial is not None:
        # One level for every channel, or the eight the policy observes (fast x4, slow x4).
        model.h[:] = np.broadcast_to(np.asarray(initial, float).reshape(-1, 1) if np.ndim(initial) == 0
                                     else np.asarray(initial, float).reshape(2, 4), (2, 4)) ** 2
    t, policy = trace["t"], trace["policy"]
    position, velocity = trace["mallet"] / SCALE, trace["mallet_v"] / SCALE
    clock = trace.get("sample_time", t)
    peak, above, total, seconds = 0.0, 0.0, np.zeros(4), 0.0
    previous = None
    for k in np.flatnonzero(policy & np.isfinite(velocity).all(axis=1) & np.isfinite(position).all(axis=1)):
        dt = TICK_S if previous is None else float(t[k] - t[previous])
        accel = np.zeros((1, 2))
        if previous is not None and 0 < dt < 0.05:
            step = float(clock[k] - clock[previous])
            accel = (velocity[k] - velocity[previous])[None] / (step if 0.005 < step < 0.05 else dt)
        dt = float(np.clip(dt, 1e-3, 0.05))     # a pause is not integrated: the drives are not modeled at rest
        model.advance(position[k][None], velocity[k][None], accel, dt)
        level = float(model.levels.max())
        peak = max(peak, level)
        above += dt * (level >= LOAD_LEVEL)
        total += model.current_squared[0] * dt
        seconds += dt
        previous = k
    settle = float(np.sqrt((total / max(seconds, 1e-9) / model.limits[0] ** 2).max())) if seconds else 0.0
    return peak, above, settle


def measure(trace, *, thermal_model=None, thermal_gain=1.0, initial_load=None):
    """Raw samples and counts for one trace; pool() adds them up and gates() summarises."""
    t, policy = trace["t"], trace["policy"].astype(bool)
    puck, puck_v = trace["puck"], trace["puck_v"]
    cap, mallet, mallet_v = trace["cap"], trace["mallet"], trace["mallet_v"]
    dt = np.diff(t, prepend=np.nan)
    paired = policy & np.roll(policy, 1) & (dt > 0) & (dt < 0.05)
    paired[0] = False
    seen = np.isfinite(puck).all(axis=1)
    speed = np.linalg.norm(puck_v, axis=1)
    far = policy & seen & (puck[:, 1] > CENTRE_Y)
    idle = far & (speed < IDLE_SPEED_M_S) & (trace["rival"] < IDLE_SPEED_M_S)
    # Realised acceleration: the profile's velocity between consecutive ticks.
    clock = trace.get("sample_time", t)
    step = np.diff(clock, prepend=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = np.linalg.norm(mallet_v - np.roll(mallet_v, 1, axis=0), axis=1) / step
    accel = accel[paired & (step > 0.005) & (step < 0.05) & np.isfinite(accel)]
    # Reversals within 100 ms.
    k = REVERSAL_WINDOW_TICKS
    v = np.nan_to_num(mallet_v)
    own_speed = np.linalg.norm(v, axis=1)
    dot = np.full(len(t), np.nan)
    dot[k:] = (v[k:] * v[:-k]).sum(axis=1)
    before = np.concatenate((np.zeros(k), own_speed[:-k]))
    recent = np.concatenate((np.full(k, np.inf), t[k:] - t[:-k])) < 0.15
    reversals = [len(_starts(t[(dot < 0) & (own_speed >= level) & (before >= level) & policy & recent], REVERSAL_GAP_S))
                 for level in (1.5, 3.0)]
    # Incoming shots at speed.
    crossing = (seen & np.roll(seen, 1) & (puck[:, 1] < CENTRE_Y) & (np.roll(puck[:, 1], 1) >= CENTRE_Y)
                & (speed >= INCOMING_SPEED_M_S) & policy)
    crossing[0] = False
    excursion, contact_speed, peak_speed = [], [], []
    for i in np.flatnonzero(crossing):
        end = i
        while (end + 1 < len(t) and t[end + 1] - t[i] <= INCOMING_WINDOW_S and policy[end + 1]
               and not (seen[end + 1] and puck_v[end + 1, 1] > 0.5)):
            end += 1
        start = max(i - INCOMING_LEAD_TICKS, 0)
        window = slice(start, end + 1)
        if not np.isfinite(mallet[window, 1]).all():
            continue
        excursion.append(float(mallet[window, 1].max() - mallet[start, 1]))
        peak_speed.append(float(own_speed[i:end + 1].max()))
        distance = np.linalg.norm(puck[i:end + 1] - mallet[i:end + 1], axis=1)
        if np.isfinite(distance).any() and np.nanmin(distance) < CONTACT_RANGE_M:
            contact_speed.append(float(own_speed[i + int(np.nanargmin(distance))]))
    target_change = np.linalg.norm(trace["target"] - np.roll(trace["target"], 1, axis=0), axis=1) * 1000
    # Time under the policy: a tick lasts until the next one (a late tick on the
    # table is 40 ms, not 20), a pause or the end of the log closes at TICK_S.
    until_next = np.append(np.diff(t), np.nan)
    until_next = np.where((until_next > 0) & (until_next < 0.05), until_next, TICK_S)
    result = dict(
        policy_seconds=float(until_next[policy].sum()), idle_cap=cap[idle], far_cap=cap[far], cap=cap[policy],
        accel=accel, reversals=reversals, incoming=int(crossing.sum()), excursion=excursion,
        contact_speed=contact_speed, peak_speed=peak_speed,
        back_edge_ticks=int((policy & (mallet[:, 1] < trace["back_y"] + BACK_EDGE_M)).sum()),
        policy_ticks=int(policy.sum()), idle_change=target_change[idle & paired])
    if initial_load is None and "measured_load" in trace:
        known = np.flatnonzero(np.isfinite(trace["measured_load"]).all(axis=1))
        initial_load = trace["measured_load"][known[0]] if len(known) else None
    model = trace.get("thermal_model") if thermal_model is None else thermal_model
    result["load_peak"], result["load_seconds"], result["load_settle"] = modeled_load(
        trace, model, gain=thermal_gain, initial=initial_load)
    if "measured_load" in trace and np.isfinite(trace["measured_load"]).any():
        result["measured_load_peak"] = float(np.nanmax(trace["measured_load"][policy]))
    return result


def _percentile(values, q):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if len(values) else None


def _share(values, level):
    values = np.asarray(values, float)
    return float((values > level).mean()) if len(values) else None


def gates(measures):
    """Pool per-trace measures (measure()) into the reported gates."""
    def cat(key):
        parts = [np.asarray(m[key], float) for m in measures]
        return np.concatenate(parts) if parts else np.zeros(0)
    minutes = sum(m["policy_seconds"] for m in measures) / 60
    ticks = sum(m["policy_ticks"] for m in measures)
    accel, excursion, contact = cat("accel"), cat("excursion"), cat("contact_speed")
    out = dict(
        policy_minutes=minutes, traces=len(measures),
        idle_ticks=int(len(cat("idle_cap"))),
        idle_cap_p50_m_s2=_percentile(cat("idle_cap"), 50), idle_cap_p90_m_s2=_percentile(cat("idle_cap"), 90),
        far_half_cap_over_40_share=_share(cat("far_cap"), 40), far_half_cap_over_80_share=_share(cat("far_cap"), 80),
        all_ticks_cap_over_80_share=_share(cat("cap"), 80),
        accel_duty_over_60_share=_share(accel, 60),
        accel_rms_m_s2=float(np.sqrt(np.mean(accel ** 2))) if len(accel) else None,
        reversals_1p5_per_min=sum(m["reversals"][0] for m in measures) / max(minutes, 1e-9),
        reversals_3_per_min=sum(m["reversals"][1] for m in measures) / max(minutes, 1e-9),
        incoming_shots=int(sum(m["incoming"] for m in measures)),
        incoming_forward_excursion_mean_mm=float(excursion.mean() * 1000) if len(excursion) else None,
        incoming_forward_excursion_p90_mm=None if not len(excursion) else _percentile(excursion, 90) * 1000,
        incoming_contact_speed_mean_m_s=float(contact.mean()) if len(contact) else None,
        incoming_contact_speed_p90_m_s=_percentile(contact, 90),
        incoming_peak_mallet_speed_mean_m_s=float(cat("peak_speed").mean()) if len(cat("peak_speed")) else None,
        back_edge_dwell_share=sum(m["back_edge_ticks"] for m in measures) / max(ticks, 1),
        modeled_load_peak=max((m["load_peak"] for m in measures), default=None),
        modeled_load_seconds_above_0p95=sum(m["load_seconds"] for m in measures),
        modeled_load_settle=float(np.sqrt(sum(m["load_settle"] ** 2 * m["policy_seconds"] for m in measures)
                                          / max(60 * minutes, 1e-9))),
        idle_command_change_p50_mm=_percentile(cat("idle_change"), 50),
        idle_command_change_p90_mm=_percentile(cat("idle_change"), 90))
    measured = [m["measured_load_peak"] for m in measures if "measured_load_peak" in m]
    if measured:
        out["measured_load_peak"] = max(measured)
    return out


# What each row of the printed table is: (key, label, format).
ROWS = (
    ("policy_minutes", "policy minutes", "{:.1f}"),
    ("idle_cap_p50_m_s2", "idle cap p50, m/s^2", "{:.1f}"),
    ("idle_cap_p90_m_s2", "idle cap p90, m/s^2", "{:.1f}"),
    ("far_half_cap_over_40_share", "far-half ticks cap > 40", "{:.3f}"),
    ("far_half_cap_over_80_share", "far-half ticks cap > 80", "{:.3f}"),
    ("all_ticks_cap_over_80_share", "all ticks cap > 80", "{:.3f}"),
    ("accel_duty_over_60_share", "|accel| > 60 duty", "{:.4f}"),
    ("accel_rms_m_s2", "rms accel, m/s^2", "{:.1f}"),
    ("reversals_1p5_per_min", "reversals >= 1.5 m/s, /min", "{:.1f}"),
    ("reversals_3_per_min", "reversals >= 3 m/s, /min", "{:.2f}"),
    ("incoming_shots", "incoming shots >= 5 m/s", "{:d}"),
    ("incoming_forward_excursion_mean_mm", "  forward excursion mean, mm", "{:.0f}"),
    ("incoming_forward_excursion_p90_mm", "  forward excursion p90, mm", "{:.0f}"),
    ("incoming_contact_speed_mean_m_s", "  mallet speed at contact, m/s", "{:.2f}"),
    ("incoming_peak_mallet_speed_mean_m_s", "  peak mallet speed, m/s", "{:.2f}"),
    ("back_edge_dwell_share", "back-edge dwell share", "{:.3f}"),
    ("modeled_load_peak", "modeled load peak", "{:.2f}"),
    ("modeled_load_seconds_above_0p95", "modeled load s >= 0.95", "{:.1f}"),
    ("modeled_load_settle", "modeled load, settling level", "{:.2f}"),
    ("measured_load_peak", "drive RMS peak (telemetry)", "{:.2f}"),
    ("idle_command_change_p50_mm", "idle command change p50, mm", "{:.2f}"),
    ("idle_command_change_p90_mm", "idle command change p90, mm", "{:.2f}"),
)


def format_table(columns):
    """Rows of ROWS, one column per (title, gates dict)."""
    width = max(len(label) for _, label, _ in ROWS)
    widths = [max(len(title), 9) for title, _ in columns]
    lines = [" " * width + "  " + "  ".join(title.rjust(w) for (title, _), w in zip(columns, widths))]
    for key, label, pattern in ROWS:
        if not any(key in g for _, g in columns):
            continue
        cells = []
        for (_, g), w in zip(columns, widths):
            value = g.get(key)
            cells.append(("--" if value is None else pattern.format(value)).rjust(w))
        lines.append(label.ljust(width) + "  " + "  ".join(cells))
    return "\n".join(lines)
