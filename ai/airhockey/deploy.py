"""A trained TD-MPC2 checkpoint as a table-side policy.

The adapter `ai/bin/run_policy.py` was waiting for: a tracker report in table
millimetres in, a Command in table millimetres out, with the checkpoint's own
observation and action conventions in between. Two objects, split so the
part that has to be exactly right is testable with no checkpoint, no torch
and no camera:

  ReportEncoder   report (mm, camera clock) -> the 20-dim kinematic
                  observation BatchAirHockeyEnv trains on. Same layout, same
                  units (sim metres, sim metres/s), same estimators: puck
                  velocity from the same 30 ms slope the sim's tracker model
                  uses, own velocity as the position difference over one
                  command tick (which is what the env does), opponent
                  velocity likewise.
  TDMPC2Policy    ReportEncoder + the agent + the action rescale back to mm.

WHAT IS AND IS NOT FAITHFUL TO TRAINING. The checkpoint saw its own paddle
fresh (the env hands it the controller's position), the puck and the
opponent through a ~7.7 ms camera, and constant cap features. The table gives
it exactly that: run_policy prefers the controller's POS for the own mallet,
the tracker carries the camera's latency, and the cap features are written
as the constants training used, and the shot-type request ([17:20]) is
whatever --shot-type asks for. The caps are the sim body's since 2026-09-06
(12 m/s, 20 m/s^2, pinned); a run below them is noted at load time -- the
policy then arrives late rather than wrong.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import cdpr_geometry as geom  # noqa: E402

from airhockey.dynamics import (AGENT_DR_ACCEL_M_S2, AGENT_DR_SPEED_M_S,  # noqa: E402
                                MAX_ACCEL_M_S2, MAX_SPEED_M_S,
                                sim_to_table_mm, table_mm_to_sim)
from airhockey.heuristics import (Command, TrackerReport,  # noqa: E402
                                  estimate_velocity)
from airhockey.physics import TableConfig  # noqa: E402
from airhockey.rewards import SHOT_TYPE_NAMES  # noqa: E402

# What to ask the policy for (observation [17:20]): a fixed type for the
# session, or "mix" -- a fresh draw each time the puck enters the robot's
# half, which is how training presented it.
SHOT_MODES = tuple(SHOT_TYPE_NAMES) + ("mix",)

# The sim's tracker model fits the puck velocity over 30 ms (6 frames at
# 200 Hz); the deployed estimate fits the same window so the policy sees the
# same estimator it trained against, noise included.
PUCK_VELOCITY_WINDOW_S = 0.030

# A gap between command ticks longer than this is not a tick, it is a
# resume (the watchdog held the paddle, or the loop stalled): velocities
# differenced across it would be garbage, so they restart from zero and the
# planner is told it is a fresh episode.
RESYNC_GAP_S = 0.100

# Out of play: a fix inside either goal mouth. The sim never shows the
# policy a puck sitting in a goal -- a goal resets it to the centre at once
# -- and on the table the policy chased a scored puck along the rail at full
# caps until a drive overloaded (2026-09-05). Such a fix is reported as the
# sim's post-goal state instead: a puck at the centre, at rest.
GOAL_MOUTH_MARGIN_MM = 30.0     # sideways slack on the mouth
GOAL_LINE_TOL_MM = 5.0          # the sim scores when the CENTRE crosses the line


def puck_out_of_play(x_mm: float, y_mm: float) -> bool:
    beyond_end = (x_mm > geom.RAIL_MAX_X - GOAL_LINE_TOL_MM
                  or x_mm < geom.RAIL_MIN_X + GOAL_LINE_TOL_MM)
    centre_y = 0.5 * (geom.RAIL_MIN_Y + geom.RAIL_MAX_Y)
    in_mouth = abs(y_mm - centre_y) < 0.5 * geom.GOAL_WIDTH_MM + GOAL_MOUTH_MARGIN_MM
    return beyond_end and in_mouth

OBS_DIM = 20


def mm_velocity_to_sim(vx_mm_s: float, vy_mm_s: float,
                       sim_width: float = 1.0, sim_half_height: float = 1.0
                       ) -> tuple[float, float]:
    """The velocity counterpart of dynamics.table_mm_to_sim.

    Positions map grid y -> sim x and grid x -> sim y (reversed), so a
    velocity does the same with the same scale factors and no offset.
    """
    sx_per_mm = sim_width / (geom.RAIL_MAX_Y - geom.RAIL_MIN_Y)
    sy_per_mm = -sim_half_height / (geom.RAIL_MAX_X - geom.CENTERLINE_X)
    return vy_mm_s * sx_per_mm, vx_mm_s * sy_per_mm


class ReportEncoder:
    """Tracker report -> the observation the checkpoint was trained on.

    Stateful only in the ways the env is: previous positions for the
    finite-difference velocities, and the last puck fix for when the history
    is empty. `reset()` clears it.
    """

    def __init__(self, table: TableConfig | None = None, shot_mode: str = "none"):
        if shot_mode not in SHOT_MODES:
            raise ValueError(f"shot_mode must be one of {SHOT_MODES}, not {shot_mode!r}")
        self.shot_mode = shot_mode
        self._rng = np.random.default_rng()
        cfg = table or TableConfig()
        self.width = cfg.width
        self.half_h = cfg.height / 2.0
        self.height = cfg.height
        # Where the env parks an opponent it has not been told about.
        self._opp_default = (cfg.width / 2.0, cfg.height * 0.85)
        self.side = 1.0                                   # BatchAirHockeyEnv.ROBOT_SIDE
        self.cap_features = (AGENT_DR_SPEED_M_S[1] / MAX_SPEED_M_S,
                             AGENT_DR_ACCEL_M_S2[1] / MAX_ACCEL_M_S2)
        self.reset()

    def reset(self) -> None:
        self.fresh = True             # tells the planner to drop its warm start
        self._t_prev: float | None = None
        self._own_prev: tuple[float, float] | None = None
        self._opp_prev: tuple[float, float] | None = None
        self._puck_last: tuple[float, float] | None = None
        self.shot_type = (0 if self.shot_mode == "mix"
                          else SHOT_TYPE_NAMES.index(self.shot_mode))
        self._puck_in_half: bool | None = None
        # The previous action, as the env carries it: zero after a reset,
        # then whatever the policy last emitted (the caller sets it).
        self.last_action = np.zeros(2, dtype=np.float32)

    def _to_sim(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return table_mm_to_sim(x_mm, y_mm, self.width, self.half_h)

    def encode(self, report) -> np.ndarray:
        rep = TrackerReport.coerce(report)
        t = rep.t_s if rep.t_s is not None else (
            rep.puck[0].t_s if rep.puck else 0.0)

        # Tick length, for the finite differences. A resume restarts them.
        dt = None if self._t_prev is None else t - self._t_prev
        resync = dt is None or dt <= 0.0 or dt > RESYNC_GAP_S
        self.fresh = self.fresh or resync
        self._t_prev = t

        # Puck: the tracker's fix and the same 30 ms slope the sim models.
        est = estimate_velocity(rep.puck, window_s=PUCK_VELOCITY_WINDOW_S)
        if est is not None and puck_out_of_play(est.x_mm, est.y_mm):
            est = None
            self._puck_last = None          # a scored puck is not "last seen"
        if est is not None:
            px, py = self._to_sim(est.x_mm, est.y_mm)
            pvx, pvy = mm_velocity_to_sim(est.vx_mm_s, est.vy_mm_s,
                                          self.width, self.half_h)
            self._puck_last = (px, py)
        elif self._puck_last is not None:
            (px, py), (pvx, pvy) = self._puck_last, (0.0, 0.0)
        else:
            px, py, pvx, pvy = self.width / 2.0, self.half_h, 0.0, 0.0

        # Own paddle: fresh, velocity over one tick exactly as the env does it.
        ox, oy = self._to_sim(*rep.mallet)
        if resync or self._own_prev is None:
            ovx = ovy = 0.0
        else:
            ovx = (ox - self._own_prev[0]) / dt
            ovy = (oy - self._own_prev[1]) / dt
        self._own_prev = (ox, oy)

        # Opponent: seen -> its position, differenced when it was also seen
        # last tick; unseen -> where the env parks one, at rest.
        if rep.opponent is not None:
            qx, qy = self._to_sim(*rep.opponent)
            if resync or self._opp_prev is None:
                qvx = qvy = 0.0
            else:
                qvx = (qx - self._opp_prev[0]) / dt
                qvy = (qy - self._opp_prev[1]) / dt
            self._opp_prev = (qx, qy)
        else:
            (qx, qy), qvx, qvy = self._opp_default, 0.0, 0.0
            self._opp_prev = None

        # A possession begins when the puck enters the robot's half; in
        # "mix" mode that is when a new request is drawn, as in training.
        in_half = py < self.half_h
        if self.shot_mode == "mix" and in_half and not self._puck_in_half:
            self.shot_type = int(self._rng.integers(0, 4))
        self._puck_in_half = in_half
        onehot = [1.0 if self.shot_type == k else 0.0 for k in (1, 2, 3)]

        return np.array([px, py, pvx, pvy,
                         ox, oy, ovx, ovy,
                         qx, qy, qvx, qvy,
                         self.side, *self.cap_features,
                         *self.last_action, *onehot], dtype=np.float32)


class TDMPC2Policy:
    """report (mm) -> Command (mm), through a checkpoint.

    plan_iterations=0 acts from the policy prior alone: ~0.1 ms on a CPU,
    the only mode that fits the 10 ms command tick without a GPU in the
    loop. N>0 runs N MPPI iterations on the GPU when there is one; the cost
    is measured at load and printed, and it is the operator's call whether
    it fits (3 iterations was ~26 ms on a shared GPU).
    """

    name = "tdmpc2"

    def __init__(self, run: str, speed_mm_s: float, accel_mm_s2: float,
                 plan_iterations: int = 0, device: str | None = None,
                 ckpt: str | Path | None = None, plan_smooth: float | None = None,
                 compile_plan: bool = True, shot_mode: str = "none"):
        import torch                                        # noqa: PLC0415

        from airhockey.batch_env import BatchAirHockeyEnv   # noqa: PLC0415
        from airhockey.policy_loader import (load_agent,    # noqa: PLC0415
                                             resolve_checkpoint)

        self._torch = torch
        # The env is the authority on the layout and the action rescale; one
        # instance, never stepped.
        env = BatchAirHockeyEnv(n_envs=1)
        assert env.OBS_DIM == OBS_DIM
        self.encoder = ReportEncoder(env.table_config, shot_mode=shot_mode)
        self._low = np.asarray(env._action_low, dtype=float)
        self._high = np.asarray(env._action_high, dtype=float)
        self.width, self.half_h = env.table_config.width, env.table_config.height / 2.0

        self.ckpt = Path(ckpt) if ckpt is not None else resolve_checkpoint(run)
        kw = {} if plan_smooth is None else {"plan_smooth": plan_smooth}
        self.agent = load_agent(run, iterations=max(1, plan_iterations),
                                ckpt=self.ckpt, **kw)
        self.plan = plan_iterations > 0
        if not self.plan:
            self.agent.cfg.mpc = False
        if device is None:
            device = "cuda" if (self.plan and torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            # A 15-in MLP does not want 16 intra-op threads: the pool's
            # wake-ups cost more than the matmuls, and in the 100 Hz loop
            # they contend with the tracker for the same cores.
            torch.set_num_threads(1)
        self.agent.model.to(self.device)
        self.agent.device = self.device
        for k in ("_prev_mean", "_prev_mean_batch"):
            v = getattr(self.agent, k, None)
            if v is not None:
                setattr(self.agent, k, v.to(self.device))

        # CUDA graphs on the single-env planner. The planner's cost is launch
        # overhead, not arithmetic (bench_planner.py: 128 samples cost the
        # same as 512), and replaying a captured graph halves it -- 6 MPPI
        # iterations 12.6 -> 6.3 ms on the 4090. Compile + warm-up takes
        # ~20 s at start-up, which warm_up() below absorbs. The same wrap
        # upstream TD-MPC2 applies under cfg.compile; the shapes here are
        # static (one env), which is what the mode needs.
        self._plan_fn = None
        self.compiled = False
        if self.plan and compile_plan and self.device.type == "cuda":
            self._plan_fn = torch.compile(self.agent._plan, mode="reduce-overhead")
            self.compiled = True

        self.speed_mm_s = float(speed_mm_s)
        self.accel_mm_s2 = float(accel_mm_s2)
        self.last_ms = 0.0
        self.last_obs = None
        self.last_action = None
        self.reset()

    def reset(self) -> None:
        self.encoder.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self._plan_fn is not None:
                # The captured single-env graph; t0 is a Python bool here,
                # so a fresh episode and a continuing one are two graphs.
                a = self._plan_fn(o, t0=bool(self.encoder.fresh), eval_mode=True)
            else:
                t0 = torch.tensor([self.encoder.fresh], dtype=torch.bool,
                                  device=self.device)
                a = self.agent.act(o, t0=t0, eval_mode=True)
        self.encoder.fresh = False
        return np.asarray(a.detach().cpu().numpy(), dtype=float).reshape(-1)[:2]

    def target_mm(self, action: np.ndarray) -> tuple[float, float]:
        a = np.clip(action, -1.0, 1.0)
        sx = self._low[0] + (a[0] + 1.0) * 0.5 * (self._high[0] - self._low[0])
        sy = self._low[1] + (a[1] + 1.0) * 0.5 * (self._high[1] - self._low[1])
        return sim_to_table_mm(sx, sy, self.width, self.half_h)

    def __call__(self, report) -> Command:
        w = time.perf_counter()
        obs = self.encoder.encode(report)
        action = self.act(obs)
        self.encoder.last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        x_mm, y_mm = self.target_mm(action)
        self.last_ms = 1000.0 * (time.perf_counter() - w)
        # Kept for the session log: a bad move is a bad observation or a
        # bad decision, and only this tells them apart.
        self.last_obs = obs
        self.last_action = action
        return Command(float(x_mm), float(y_mm), self.speed_mm_s, self.accel_mm_s2)

    # ── Load-time diagnostics ──────────────────────────────────────────

    def warm_up(self, n: int = 30) -> float:
        """Median ms per decision on a synthetic report, so the operator
        knows before the first live tick whether this mode fits it."""
        t = 0.0
        times = []
        for k in range(n):
            t += 0.01
            hist = [(geom.CENTERLINE_X - 300.0 + 5.0 * j, 500.0 + 2.0 * j, t - 0.005 * j)
                    for j in range(8)]
            rep = {"puck": hist, "mallet": (geom.HOME_X, geom.HOME_Y),
                   "opponent": None, "t_s": t}
            self(rep)
            times.append(self.last_ms)
        self.reset()
        return float(np.median(times[5:]))

    def describe(self, caps_speed: float, caps_accel: float) -> str:
        mode = (f"{self.agent.cfg.iterations} MPPI iterations on {self.device}"
                + (" (CUDA graphs)" if self.compiled else "")
                if self.plan else f"policy prior only on {self.device}")
        trained_v = AGENT_DR_SPEED_M_S[1] * 1000.0
        trained_a = AGENT_DR_ACCEL_M_S2[1] * 1000.0
        lines = [f"tdmpc2: {self.ckpt}", f"  mode: {mode}"]
        if caps_accel < trained_a or caps_speed < trained_v:
            lines.append(
                f"  NOTE: trained at {trained_v:.0f} mm/s, {trained_a:.0f} mm/s^2; "
                f"this run caps at {caps_speed:.0f}, {caps_accel:.0f}. The policy "
                f"will arrive later than it expects -- late, not wrong.")
        return "\n".join(lines)
