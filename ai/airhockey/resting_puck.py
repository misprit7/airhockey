"""Does the policy strike a puck that is lying still on its own half?

The table has no referee: a puck the robot hovers beside stays there for ever.
Three measurements, all simulation only, used by `ai/bin/eval_resting_puck.py`:

`stall_map`  A puck placed AT REST on a grid of the robot's half, for each
    mallet start (`home` = the guard position, `beside` = already next to the
    puck, 0.10 right and 0.11 behind it) and each shot request, in the
    human-proxy benchmark's env (a `table` person at the far end, continuous
    rallies, report sensing, thermal gain 1.3, load 0.45). A trial is STRUCK
    once the puck moves faster than 1.5 after the mallet has touched it,
    CROSSED once it is over the centre line, and STALLED if neither happens
    in the 10 s. `stall_report` groups trials into zones:

        forward / mid / back   y 0.65-0.85 / 0.35-0.60 / below 0.35, each
                               split left (x < 0.5), centre (x = 0.5), right
        rail                   x <= 0.10 or >= 0.90, above the corners
        corner                 x <= 0.15 or >= 0.85 and y <= 0.15
        front_edge             y > 0.85 (at or past the mallet's reach)

    and a CELL is one zone x one request. The cross-bank cells are a puck on
    one side with the bank off the opposite rail requested (right + left
    bank, left + right bank): where the hover was found.

`corner_recovery`  Pucks at rest in the two back corners, out of the mallet
    centre's reach, 48 per side x request (none / left / right / straight) as
    exact mirror pairs, 12.1 s each. RECOVERED = the mallet touched the puck.

`RestWatch`  Rest spells in ordinary games: runs of ticks with the puck on
    the robot's half, slower than 0.05 and within the mallet's reach (the
    mallet's edge can touch it: a puck wedged in a back corner counts). Each
    spell carries where it lay, by the stall map's zones, and the request.

Positions and speeds are in sim units (the table is 1 x 2; a unit is about a
metre). Every random draw comes from the seed, so a candidate and a reference
see the same trials.
"""
from __future__ import annotations

import numpy as np

from airhockey.human_proxy import HumanProxy, command_env
from airhockey.neural_player import PhysicalHistory
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.recorder import FrameData, Recorder
from airhockey.skill_benchmark import Fixtures
from airhockey.table_fidelity import profile_options

DT = 0.02
THERMAL_GAIN = 1.3
STRIKE_SPEED = 1.5
REST_SPEED = 0.05
HOME = (0.5, 0.2)
BESIDE = (0.10, -0.11)
REQUEST_NAMES = ("none", "left_bank", "right_bank", "straight")
DEFAULT_XS = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))
DEFAULT_YS = tuple(round(0.10 + 0.05 * i, 2) for i in range(16))
BANDS = ("forward", "mid", "back")
SIDES = ("left", "centre", "right")
FRAME_COLUMNS = ("puck_x", "puck_y", "puck_vx", "puck_vy", "agent_x", "agent_y", "opponent_x", "opponent_y",
                 "score_agent", "score_opponent", "shot_type")


def wilson(k, n, z=1.96):
    """95% Wilson interval for k of n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def share(mask, stalled):
    n, k = int(mask.sum()), int((mask & stalled).sum())
    return dict(n=n, stalled=k, rate=round(k / n, 4) if n else None, ci95=wilson(k, n))


def frame_state(env):
    e = env.engine
    return np.column_stack((e.puck_x, e.puck_y, e.puck_vx, e.puck_vy, e.paddle_agent_x, e.paddle_agent_y,
                            e.paddle_opp_x, e.paddle_opp_y, e.score_agent, e.score_opponent,
                            env.base._shot_type)).astype(np.float32)


def save_replay(path, frames, metadata):
    """One trial or game, [T, len(FRAME_COLUMNS)], as a web replay."""
    recorder = Recorder()
    for tick, row in enumerate(frames):
        recorder.record(FrameData(
            time=tick * DT, puck_x=float(row[0]), puck_y=float(row[1]), puck_vx=float(row[2]), puck_vy=float(row[3]),
            agent_x=float(row[4]), agent_y=float(row[5]), opponent_x=float(row[6]), opponent_y=float(row[7]),
            score_agent=int(row[8]), score_opponent=int(row[9]), shot_type_agent=int(row[10])))
    recorder.save(path, metadata=dict(simulation_only=True, fps=round(1 / DT),
                                      shot_type_names=["none", "left bank", "right bank", "straight"], **metadata))


# ------------------------------------------------------------- the stall map
def stall_conditions(xs, ys, starts, requests):
    """One trial per start x request x grid point, x fastest."""
    rows = [(x, y, r, s) for s in starts for r in requests for y in ys for x in xs]
    return (np.array([c[0] for c in rows], float), np.array([c[1] for c in rows], float),
            np.array([c[2] for c in rows], int), np.array([c[3] == "beside" for c in rows]))


def _place(env, puck, mallet):
    """Put the puck (at rest) and the robot's mallet (at rest) down in every env."""
    b, e = env.base, env.engine
    ids = np.arange(env.n_envs)
    e.puck_x[:], e.puck_y[:], e.puck_vx[:], e.puck_vy[:] = puck[:, 0], puck[:, 1], 0, 0
    e.paddle_agent_x[:], e.paddle_agent_y[:] = mallet.T
    e.paddle_agent_vx[:] = e.paddle_agent_vy[:] = 0
    dyn = b._agent_dyn
    for k in ("x", "y"):
        dyn[k][:] = getattr(e, "paddle_agent_" + k)
        dyn["command_" + k][:] = dyn[k]
        dyn["v" + k][:] = 0
    dyn["command_speed"][:] = dyn["max_speed"]
    dyn["command_accel"][:] = 5.0
    b._clear_profile_accel(dyn, ids)
    b._perception.reset(e.puck_x, e.puck_y, ids)
    if b._cam_active:
        for col, name in enumerate(("puck_x", "puck_y", "puck_vx", "puck_vy", "paddle_opp_x", "paddle_opp_y",
                                    "paddle_agent_x", "paddle_agent_y")):
            b._cam_ring[:, ids, col] = getattr(e, name)[ids]
        b._cam_ring[:, ids, 8] = 0
        b._sync_far_view(ids)
    b._prev_agent_x[:] = e.paddle_agent_x
    b._prev_agent_y[:] = e.paddle_agent_y


def stall_map(net, options, *, seed, sim_profile=None, xs=DEFAULT_XS, ys=DEFAULT_YS, starts=("home", "beside"),
              requests=(1, 2, 3), seconds=10.0, load=0.45, tier="table", thermal_gain=THERMAL_GAIN,
              keep_frames=False):
    """Play every trial of the grid at once. Returns per-trial arrays."""
    px, py, req, beside = stall_conditions(xs, ys, starts, requests)
    n = len(px)
    env = NeuralTrainingEnv(n, stage=3, seed=seed, games=True, **{**options, **profile_options(sim_profile)},
                            report_sensing=True, continuous_rallies=True, shot_conditioned=net.shot_conditioned)
    env.reset(seed=seed, opponent="human_proxy")
    b, e = env.base, env.engine
    for model in env.loads:
        model.gain[:] = thermal_gain
    env.loads[0].h[:] = load ** 2
    env.loads[0].observed[:] = load
    env.loads[1].h[:] = 0.4 ** 2
    env.loads[1].observed[:] = 0.4
    proxy = HumanProxy(n, tier, seed=seed + 303, config=env.cfg)
    command_env(env, proxy, reset=True)

    puck = np.column_stack((px, py))
    reach = env.cfg.paddle_radius + env.cfg.puck_radius
    lo, hi = env.decoder.low + 0.005, env.decoder.high - 0.005
    mallet = np.where(beside[:, None], puck + BESIDE, np.tile(HOME, (n, 1)))
    mallet = np.clip(mallet, lo, hi)
    # A start that would overlap the puck (only below y 0.3, never on the forward half of the grid).
    for fallback in (puck + (-BESIDE[0], BESIDE[1]), np.tile((0.5, 0.4), (n, 1))):
        overlap = np.linalg.norm(mallet - puck, axis=1) < reach + 0.015
        mallet[overlap] = np.clip(fallback, lo, hi)[overlap]
    _place(env, puck, mallet)
    b._shot_type[:] = req
    b._t_side[:] = 0
    obs = env._features(b._make_obs_direct())
    history = PhysicalHistory(net.history)
    history.reset(obs)

    dyn = b._agent_dyn
    t_strike, t_cross, t_touch = (np.full(n, np.nan) for _ in range(3))
    strike_speed, peak_cap = np.zeros(n), np.zeros(n)
    live = np.ones(n, bool)               # until the first strike or crossing
    touch0 = env.touch_count[0].copy()
    frames = [frame_state(env)] if keep_frames else None
    one_hot = np.eye(4)
    for tick in range(round(seconds / DT)):
        command_env(env, proxy)
        if tick:
            history.append(obs)
        # The request under test stays up until the trial is decided.
        b._shot_type[live] = req[live]
        if history.request is not None:
            history.request[live] = one_hot[req[live]][:, 1:]
        obs, *_ = env.step(net.act(history.for_policy(net)))
        peak_cap = np.where(live, np.maximum(peak_cap, dyn["command_accel"]), peak_cap)
        speed = np.hypot(e.puck_vx, e.puck_vy)
        now = (tick + 1) * DT
        touched = env.touch_count[0] > touch0
        t_touch = np.where(np.isnan(t_touch) & touched, now, t_touch)
        hit = live & (speed > STRIKE_SPEED) & touched
        t_strike[hit] = now
        strike_speed[hit] = speed[hit]
        crossed = np.isnan(t_cross) & (e.puck_y > env.cfg.height / 2)
        t_cross[crossed] = now
        live &= ~hit & ~crossed
        if keep_frames:
            frames.append(frame_state(env))
    raw = dict(px=px, py=py, req=req, beside=beside, t_strike=t_strike, t_cross=t_cross, t_touch=t_touch,
               strike_speed=strike_speed, peak_cap=peak_cap, final_cap=np.array(dyn["command_accel"], float),
               seconds=seconds, load=load,
               final=np.column_stack((e.puck_x, e.puck_y, e.paddle_agent_x, e.paddle_agent_y)))
    if keep_frames:
        raw["frames"] = np.stack(frames)      # [T + 1, N, columns]
    return raw


def stalled(raw):
    return np.isnan(raw["t_strike"]) & np.isnan(raw["t_cross"])


def zones(px, py):
    """(zone, side) names per trial; side is '' outside forward / mid / back."""
    eps = 1e-9
    px, py = np.asarray(px, float), np.asarray(py, float)
    zone = np.where(py >= 0.65 - eps, "forward", np.where(py >= 0.35 - eps, "mid", "back")).astype("<U10")
    side = np.where(px < 0.5 - eps, "left", np.where(px > 0.5 + eps, "right", "centre")).astype("<U6")
    outer = (px <= 0.15 + eps) | (px >= 0.85 - eps)
    corner = outer & (py <= 0.15 + eps)
    rail = ~corner & ((px <= 0.10 + eps) | (px >= 0.90 - eps))
    zone[rail], zone[corner] = "rail", "corner"
    zone[py > 0.85 + eps] = "front_edge"
    side[~np.isin(zone, BANDS)] = ""
    return zone, side


def stall_report(raw, *, min_cell_n=20):
    """Rates by zone, request, start and cell, each with a 95% Wilson interval."""
    px, py, req, beside = raw["px"], raw["py"], raw["req"], raw["beside"]
    stall = stalled(raw)
    zone, side = zones(px, py)
    everything = np.ones(len(px), bool)
    report = dict(
        trials=len(px), seconds=float(raw["seconds"]), load=float(raw["load"]),
        definition=f"never faster than {STRIKE_SPEED} after a touch and never over the centre line",
        overall=share(everything, stall),
        struck=int((~np.isnan(raw["t_strike"])).sum()),
        crossed_without_strike=int((np.isnan(raw["t_strike"]) & ~np.isnan(raw["t_cross"])).sum()),
        by_start={name: share(beside == flag, stall) for name, flag in (("home", False), ("beside", True))
                  if (beside == flag).any()},
        by_request={REQUEST_NAMES[r]: share(req == r, stall) for r in np.unique(req)},
        by_zone={z: share(zone == z, stall) for z in (*BANDS, "rail", "corner", "front_edge") if (zone == z).any()})
    cells = []
    for z in (*BANDS, "rail", "corner", "front_edge"):
        for s in (SIDES if z in BANDS else ("",)):
            for r in np.unique(req):
                mask = (zone == z) & (side == s) & (req == r)
                if not mask.any():
                    continue
                cell = dict(cell=f"{z}{'-' + s if s else ''} / {REQUEST_NAMES[r]}", zone=z, side=s,
                            request=REQUEST_NAMES[r],
                            cross_bank=(s, int(r)) in (("right", 1), ("left", 2)), **share(mask, stall))
                cell["gated"] = cell["n"] >= min_cell_n
                cells.append(cell)
    report["cells"] = cells
    report["cross_bank"] = {c["cell"]: {k: c[k] for k in ("n", "stalled", "rate", "ci95")}
                            for c in cells if c["cross_bank"]}
    gated = [c for c in cells if c["gated"]]
    worst = max(gated, key=lambda c: c["rate"]) if gated else None
    report["worst_cell"] = worst and {k: worst[k] for k in ("cell", "n", "stalled", "rate", "ci95")}
    xs, ys = np.unique(px), np.unique(py)
    report["map"] = dict(xs=xs.tolist(), ys=ys.tolist(), per_point=int(len(px) / (len(xs) * len(ys))),
                         stall_share=[[round(float(stall[(px == x) & (py == y)].mean()), 3) for x in xs] for y in ys])
    if stall.any():
        final = raw["final"][stall]
        report["stalled_trials"] = dict(
            touched_share=round(float((~np.isnan(raw["t_touch"][stall])).mean()), 3),
            peak_accel_cap_p50=round(float(np.median(raw["peak_cap"][stall])), 1),
            final_accel_cap_p50=round(float(np.median(raw["final_cap"][stall])), 1),
            mallet_minus_puck_mean=np.round((final[:, 2:] - final[:, :2]).mean(0), 3).tolist())
    return report


def format_map(report):
    m = report["map"]
    lines = [f"  stall share per grid point ({m['per_point']} trials each); columns x "
             + " ".join(f"{x:.2f}" for x in m["xs"])]
    for y, row in zip(m["ys"][::-1], m["stall_share"][::-1]):
        lines.append(f"  y={y:.2f}   " + " ".join("   . " if v == 0 else f"{v:5.2f}" for v in row))
    return "\n".join(lines)


# ------------------------------------------------------------ corner pucks
def corner_recovery(net, options, *, seed, sim_profile=None, count=384, ticks=605, thermal_gain=THERMAL_GAIN,
                    load=0.5):
    """Corner pucks by side x request, as mirror pairs: env 2k left, 2k + 1 right."""
    if count % 8:
        raise ValueError("corner trials come in 8 cells of equal size")
    env = NeuralTrainingEnv(count, stage=5, game_fraction=0, report_sensing=True,
                            shot_conditioned=net.shot_conditioned, possession_followthrough=True,
                            edge_recovery_weight=1, **{**options, **profile_options(sim_profile)})
    rng = np.random.default_rng(seed)
    right = np.arange(count) % 2 == 1
    req = (np.arange(count) // 2) % 4
    r = env.cfg.puck_radius
    puck = np.zeros((count, 4))
    puck[:, 0] = rng.uniform(r + .001, env.decoder.low[0] + .01, count)
    puck[:, 1] = rng.uniform(r + .001, env.decoder.low[1] + .015, count)
    paddle = rng.uniform(env.decoder.low + .015, env.decoder.high - .015, (count, 2))
    puck[1::2] = puck[0::2]
    paddle[1::2] = paddle[0::2]
    puck[right, 0] = env.cfg.width - puck[right, 0]
    paddle[right, 0] = env.cfg.width - paddle[right, 0]
    overlap = np.linalg.norm(paddle - puck[:, :2], axis=1) < env.cfg.paddle_radius + r + .015
    paddle[overlap] = [.5, .4]
    env.reset(seed=seed, fixtures=Fixtures(np.zeros(count, int), puck, paddle, np.full(count, .5)))
    env.edge_drill[:] = True
    for model in env.loads:
        model.gain[:] = thermal_gain
        model.h[:] = load ** 2
        model.observed[:] = load
    env.base._shot_type[:] = req
    obs = env._features(env.base._make_obs_direct())
    history = PhysicalHistory(net.history)
    history.reset(obs)
    e = env.engine
    done = np.zeros(count, bool)
    first, gap = np.full(count, np.nan), np.full(count, np.inf)
    for tick in range(ticks):
        obs, _, terminal, truncated, _ = env.step(net.act(history.for_policy(net)))
        history.append(obs)
        live = ~done
        distance = np.hypot(e.paddle_agent_x - e.puck_x, e.paddle_agent_y - e.puck_y)
        gap[live] = np.minimum(gap[live], distance[live])
        first[live & np.isnan(first) & (env.touch_count[0] > 0)] = tick * DT
        done |= live & (terminal | truncated)
        if done.all():
            break
    touched = ~np.isnan(first)
    cells = {}
    for side in (0, 1):
        for q in range(4):
            mask = (right == bool(side)) & (req == q)
            missed = mask & ~touched
            cells[f"{'right' if side else 'left'} / {REQUEST_NAMES[q]}"] = dict(
                n=int(mask.sum()), touched=int((mask & touched).sum()),
                first_touch_s=round(float(np.nanmedian(first[mask])), 2) if (mask & touched).any() else None,
                untouched_min_gap_mm=round(float(np.median(gap[missed] - env.cfg.paddle_radius - r) * 1000), 1)
                if missed.any() else None)
    return dict(trials=count, seconds=round(ticks * DT, 2), load=load, cells=cells,
                touched=int(touched.sum()), worst_cell=min(cells, key=lambda k: cells[k]["touched"]))


# ------------------------------------------------------ rest spells in games
def rest_spells(puck, speed, low, high, reach, *, half, rest_speed=REST_SPEED, dt=DT):
    """Spells of one game as (first tick, seconds): the puck on the own half
    (y < half), slower than rest_speed and within `reach` of the workspace box."""
    puck = np.asarray(puck, float)
    off = np.linalg.norm(puck - np.clip(puck, low, high), axis=1)
    resting = (puck[:, 1] < half) & (np.asarray(speed) < rest_speed) & (off <= reach)
    edges = np.flatnonzero(np.diff(np.r_[False, resting, False].astype(int)))
    return [(int(a), round(float((b - a) * dt), 2)) for a, b in zip(edges[0::2], edges[1::2])]


class RestWatch:
    """Per-tick puck state of every game of a NeuralTrainingEnv; create it after
    the reset and call it after every step (eval_human_proxy.play_tier's `observe`)."""

    def __init__(self, env):
        self.env = env
        self.frames = [frame_state(env)]

    def __call__(self):
        self.frames.append(frame_state(self.env))

    def spells(self, rest_speed=REST_SPEED):
        env = self.env
        frames = np.stack(self.frames)
        out = []
        for game in range(frames.shape[1]):
            f = frames[:, game]
            for first, seconds in rest_spells(f[:, 0:2], np.hypot(f[:, 2], f[:, 3]), env.decoder.low,
                                              env.decoder.high, env.cfg.puck_radius + env.cfg.paddle_radius,
                                              half=env.cfg.height / 2, rest_speed=rest_speed):
                zone, side = zones(f[first, 0:1], f[first, 1:2])
                out.append(dict(game=game, start_s=round(first * DT, 2), seconds=seconds,
                                puck=[round(float(v), 3) for v in f[first, 0:2]],
                                zone=f"{zone[0]}{'-' + side[0] if side[0] else ''}",
                                request=REQUEST_NAMES[int(f[first, 10])]))
        return out


def rest_report(spells, *, games, seconds, thresholds=(1.0, 4.0), longest=5):
    minutes = games * seconds / 60
    report = dict(games=games, game_minutes=round(minutes, 2))
    for limit in thresholds:
        chosen = [s["seconds"] for s in spells if s["seconds"] >= limit]
        report[f"spells_ge_{limit:g}s"] = dict(
            count=len(chosen), per_game_minute=round(len(chosen) / minutes, 3),
            seconds_per_game_minute=round(sum(chosen) / minutes, 3))
    report["longest"] = sorted(spells, key=lambda s: -s["seconds"])[:longest]
    return report
