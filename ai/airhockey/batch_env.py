"""Vectorized air hockey environment for batch stepping.

Wraps BatchPhysicsEngine to provide the same obs/action interface as
AirHockeyEnv, but processes N environments in a single step() call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from airhockey.batch_physics import BatchPhysicsEngine
from airhockey.dynamics import (ACTION_DT, AGENT_DR_ACCEL_M_S2, AGENT_DR_SPEED_M_S,
                                DR_ACCEL_RANGE, DR_SPEED_RANGE,
                                MAX_ACCEL_M_S2, MAX_SPEED_M_S,
                                OPPONENT_MAX_ACCEL_M_S2,
                                OPPONENT_MAX_SPEED_M_S,
                                workspace_in_sim)
from airhockey.motion import DEFAULT_SIM_DT, CartState, advance
from airhockey.perception import (CAMERA_DELAY_RANGE_S, FRAME_INTERVAL_S,
                                  PuckPerception)
from airhockey.physics import TableConfig


def sensing_kwargs(realistic: bool = True) -> dict[str, Any]:
    """The sensing chain the real robot actually has.

    Kept here rather than written out at each call site because there are two
    trainers and a server, and every feature that has been restated across
    them has eventually diverged.

    NOT the env's own default, deliberately: a library whose observations are
    noisy unless you ask otherwise makes every test that checks physics into a
    test that checks physics through a noise model. Training is where it
    belongs, and training should always have it on.

    Note the delay is quantised by action_dt. At 100 Hz the measured
    5.1-10.3 ms band rounds to one step either way, i.e. a flat 10 ms -- real,
    but 30% above the 7.7 ms mean, and no longer randomised. Sub-step delay
    would need interpolation the ring buffer does not do.
    """
    if not realistic:
        return {"camera_delay": 0.0, "realistic_perception": False}
    return {"camera_delay": CAMERA_DELAY_RANGE_S, "realistic_perception": True}

# Per-env opponent policy IDs
OPP_IDLE = 0
OPP_FOLLOW = 1
OPP_GOALIE = 2
OPP_EXTERNAL = 3
OPP_CORNER = 4
OPP_RANDOM = 5
OPP_SNIPER = 6        # scripted striker on a free body: fast shots at the goal
OPP_WEAK_GOALIE = 7   # scripted, slow, tracks the puck across its line

_OPP_POLICY_MAP = {
    "idle": OPP_IDLE,
    "follow": OPP_FOLLOW,
    "goalie": OPP_GOALIE,
    "external": OPP_EXTERNAL,
    "corner": OPP_CORNER,
    "random": OPP_RANDOM,
    "sniper": OPP_SNIPER,
    "weak_goalie": OPP_WEAK_GOALIE,
}


class BatchAirHockeyEnv:
    """Batch air hockey environment — N envs stepped simultaneously.

    Observation per env (15 dims):
        puck_x, puck_y, puck_vx, puck_vy,
        paddle_x, paddle_y, paddle_vx, paddle_vy,
        opp_x, opp_y, opp_vx, opp_vy,
        side, max_speed, max_accel

    Action per env: (2,) — normalized [-1, 1] target position.

    Does NOT subclass gym.Env since vectorized envs have a different
    calling convention (no Gymnasium wrappers, returns batched arrays).
    """

    # 15, not 12. The last three features describe the BODY the observation is
    # driving, which the twelve state features do not determine.
    #
    #   [12] SIDE: 1.0 robot, 0.0 human. The two sides do not have the same
    #        capabilities, and only the robot is confined to the reachable
    #        workspace, so a policy that plays both in self-play cannot act
    #        correctly without knowing which it currently is. Always 1.0 in
    #        production -- and always 1.0 with opponent_body="robot", where
    #        the far side is a copy of the machine (symmetric self-play).
    #   [13] MAX SPEED, [14] MAX ACCEL, as a ratio to the robot's nominal
    #        caps. Domain randomisation samples these per env, and they change
    #        the right play rather than just the execution: how early to commit
    #        to an intercept, whether a cross-table save is reachable at all,
    #        how much of a wind-up a shot can afford. A policy that cannot see
    #        them has to average over the range and will consistently
    #        over-commit on a slow draw and under-use a fast one.
    #
    # OWN caps only. The opponent's are deliberately NOT given: in production
    # the opponent is a human whose limits are unknown and unknowable, so a
    # feature for them would be informative in sim and a fixed lie on the
    # table. The policy has to read the opponent off their motion, as it will
    # have to for real.
    #   [15] [16] PREVIOUS ACTION of the body being observed, normalised,
    #        zero after a reset. Added 2026-09-03 for the smoothness term:
    #        a reward on step-to-step action change is unlearnable from a
    #        frame that does not carry the previous step. Older 15-wide
    #        checkpoints are loaded with zero weight on these two inputs
    #        (policy_loader.load_checkpoint), so they behave exactly as
    #        before until training grows them.
    #        The previous action is THREE wide since 2026-09-06 (x, y and
    #        the accel fraction of action_mode "profile_a"); in the 2-dim
    #        "position" mode the third slot stays zero.
    #   [18] [19] [20] SHOT TYPE REQUESTED, one-hot [bank left, bank right,
    #        straight]; all zero = no preference. Drawn by the env once per
    #        possession (the puck entering this body's half) when
    #        shot_types=True, else always zero. Left and right are the
    #        observer's own, facing the far goal: the x = 0 rail is LEFT.
    #        The reward for honouring it lives in rewards.BatchRewardShaper
    #        (shot_type_reward); on the table run_policy --shot-type sets it.
    #   [21] TIME ON SIDE: seconds since the puck last crossed the centre
    #        line, clipped at T_SIDE_CLIP and divided by it. Which side it is
    #        on, the puck's y already says. Exists so patience can be
    #        rewarded (rewards: patience_s) and be learnable from a frame.
    OBS_DIM = 22
    PREV_ACTION_IDX = 15      # [15:18]; TD-MPC2's cfg.prev_action_start
    PREV_ACTION_WIDTH = 3
    SHOT_TYPE_IDX = 18        # [18:21]
    T_SIDE_IDX = 21
    T_SIDE_CLIP = 5.0         # seconds; the feature saturates here
    # action_mode "profile_a": the accel fraction maps [-1, 1] -> this range
    # of the machine's cap, so a command is never below a crawl.
    ACCEL_FRAC_MIN = 0.05
    ROBOT_SIDE = 1.0
    HUMAN_SIDE = 0.0

    # Reward charged per sim-unit of commanded overshoot past the reachable
    # box, per step. PROPORTIONAL rather than a flat fine, deliberately: the
    # robot's best defensive station is often ON the box's bottom edge, and a
    # cliff penalty right at the boundary would fine the tiny action noise of
    # standing exactly where the policy should stand. Proportional cost at
    # the boundary is zero, and grows with how far past it the command
    # pretends to go -- a gradient pointing back to reachable ground.
    # Worst case (far corner of the half from the nearest box corner) is
    # ~0.34 sim-units -> -0.007/step, small next to the 0.1 proximity term.
    WS_PENALTY_PER_UNIT = 0.02

    # Stuck-puck relaunch (see step()). Seconds, not steps, so the rule does
    # not change meaning with the control rate. "Attended" = a paddle
    # within ATTEND_RADIUS of the puck: paddle radius 0.05 + puck radius
    # 0.04 + 0.06 of slack, in sim metres.
    STUCK_UNATTENDED_S = 1.2
    STUCK_ATTENDED_S = 5.0     # was 3; the patience ramp (rewards) needs up to 1.5 s of control
    ATTEND_RADIUS = 0.15

    # Scripted far-side opponents on a FREE body (ai/RETRAIN.md item 5).
    # Neither is a copy of the machine: the sniper exists to put fast shots
    # at the robot, which the robot's own body cannot produce at 20 m/s^2,
    # and the weak goalie exists to be scored on. Both run a first-order
    # lag body with the caps below rather than the profile law.
    #
    # SNIPER: waits near its line tracking the puck, and when the puck is
    # on its half, slower than SNIPER_MAX_PUCK_SPEED and in front of it,
    # drives THROUGH the puck toward a random point in the robot's goal
    # mouth (a rail bank on a third of shots) at strike caps for up to
    # SNIPER_STRIKE_S. Puck leaves at 8-12 m/s (paddle restitution 0.9,
    # 15 g puck vs 170 g mallet, max_puck_speed 12).
    SNIPER_STATION_Y = 0.20            # below the far wall
    SNIPER_WAIT_SPEED = 3.0
    SNIPER_WAIT_ACCEL = 30.0
    SNIPER_STRIKE_SPEED = (5.0, 8.0)   # drawn per strike
    SNIPER_STRIKE_ACCEL = 300.0
    SNIPER_STRIKE_S = 0.25
    SNIPER_COOLDOWN_S = 0.40
    SNIPER_MAX_PUCK_SPEED = 2.5
    SNIPER_THROUGH = 0.15              # target this far beyond the puck
    SNIPER_BANK_P = 0.33
    # WEAK GOALIE: tracks puck x along its line with a dead zone, slowly.
    WEAK_STATION_Y = 0.15
    WEAK_SPEED = 2.0
    WEAK_ACCEL = 15.0
    WEAK_DEADZONE = 0.08

    # Sensing fuzz (ai/RETRAIN.md item 6): in a fuzz_p fraction of episodes
    # the camera loses the opponent's mallet for FUZZ_OPP_WINDOWS spells of
    # FUZZ_OPP_S seconds (a hand over it), and the puck for FUZZ_PUCK_WINDOWS
    # spells of FUZZ_PUCK_S. What the policy sees then is what the deploy
    # encoder would give it: the opponent parked at its default, at rest,
    # with the velocity zeroed on both edges of the spell; the puck coasting
    # on the tracker's last velocity for up to 150 ms and then at rest.
    # Both views lose sight of the OTHER paddle in the same spell.
    FUZZ_OPP_WINDOWS = (1, 2)
    FUZZ_OPP_S = (0.3, 1.5)
    FUZZ_PUCK_WINDOWS = (1, 3)
    FUZZ_PUCK_S = (0.05, 0.15)
    FUZZ_MARGIN_S = 0.5              # no spell in the first/last half second

    # History-mode observation layout. Frame lags are relative to the newest
    # frame the env is entitled to see (i.e. after sensing latency), in
    # 5 ms camera frames: 0/10/20/50/100 ms for the puck, 0/20/50 ms for the
    # opponent -- close to the spacing Air-Hockey-Sim used, dense recent,
    # sparse far, enough baseline to read speed AND curvature.
    # The 30 ms sample (lag 6) NARROWS a measured bounce-blindness hole --
    # it does not close it, and the commit that added it overclaimed. With
    # lags 0/10/20/50/100 ms, a bounce aged 20-50 ms was unrecoverable
    # (|vy| error ~957 mm/s against a truth of 1962 at age 40 ms, ~3
    # decision ticks after every rail contact). With 30 ms added, the bad
    # band shrinks to under one tick, worst case ~22% vy error near age
    # 45 ms. The residual is structural to ANY sparse lag set: a reversal
    # near the OLDER end of a straddled segment leaves the segment's net
    # displacement un-flipped, so there is nothing for an estimator to
    # detect -- adding samples only moves and narrows the band. Closing it
    # entirely means a sample every 10 ms out to 50 (obs +2), accepted as
    # a bounded residual for now; a learned consumer also has wall
    # proximity to infer from, which an analytic fitter does not.
    HISTORY_PUCK_LAGS = (0, 2, 4, 6, 10, 20)
    HISTORY_OPP_LAGS = (0, 4, 10)
    # 6*2 puck + 3*2 opp + own(x,y,vx,vy) + prev action(4) + side + caps(2)
    HISTORY_OBS_DIM = 12 + 6 + 4 + 4 + 3

    def __init__(
        self,
        n_envs: int,
        table_config: TableConfig | None = None,
        # "profile" is the REAL firmware control law (fw/include/
        # motion_profile.h, built as a host library and bound in motion.py):
        # one velocity profile along the direction of travel, jerk-limited,
        # with the same parking rule the Teensy uses. It is the default
        # because "ideal" teleports the paddle to its target -- a policy
        # trained against that learns to command positions no actuator can
        # reach, and finds out on the hardware.
        agent_dynamics: str = "profile",   # "ideal" | "delayed" | "profile"
        # The opponent is a HUMAN, and a hand is not a stepper under a
        # trapezoidal profile. "delayed" -- a first-order lag with caps -- is
        # the better model, and it is deliberately given the human limits.
        opponent_dynamics: str = "delayed",
        # 400 Hz: fine enough that a 12 m/s puck moves 30 mm per step (well
        # inside the 80.7 mm contact distance), and an exact multiple of the
        # camera's 200 Hz so the simulated camera can tick every 2nd substep.
        # The previous 1/500 was finer but did NOT divide the camera rate,
        # which is why the delay ended up modelled at the action layer with
        # whole-tick quantisation.
        physics_dt: float = 1 / 400,
        # The action rate is now a free choice: sensing runs on its own
        # 200 Hz camera clock (see below), so nothing about the delay model
        # depends on this number. 100 Hz keeps decisions fresh against a
        # fast puck and is sustainable in real time with a 3-iteration MPPI.
        action_dt: float = ACTION_DT,
        max_episode_time: float = 60.0,
        max_episode_steps: int | None = None,
        max_score: int = 7,
        opponent_policy: str = "idle",
        opponent_mix: dict[str, int] | None = None,
        camera_delay: float | tuple[float, float] = 0.0,
        domain_randomize: bool = False,
        frame_stack: int = 1,  # kept for API compat, must be 1
        score_handicap: bool = False,
        # DelayedDynamics parameters
        dynamics_max_speed: float = MAX_SPEED_M_S,
        dynamics_max_accel: float = MAX_ACCEL_M_S2,
        dynamics_time_constant: float = 0.02,
        # Observe the puck through a model of the real tracker rather
        # than reading the engine: finite-difference velocity over noisy
        # positions, plus the IR ring's blind spot at table centre.
        realistic_perception: bool = False,
        # "kinematic" (default): the 15-dim snapshot obs above.
        # "history": positions over the recent past instead of estimated
        # velocities -- 5 puck frames and 3 opponent frames read straight
        # off the camera ring, own state fresh, previous action included.
        # Motion is left for the network to infer, which sidesteps the
        # estimator entirely and gives a memoryless learner the memory in
        # the observation. (The approach HudsonNock/Air-Hockey-Sim proved
        # on hardware; arrived at independently the moment the camera ring
        # existed to read from.)
        obs_mode: str = "kinematic",       # "kinematic" | "history"
        # "position" (default): 2-dim target the profile chases at machine
        # caps. "profile_a" (the retrain, 2026-09-06): 3-dim (x, y,
        # accel_frac) -- the policy also commands the ACCEL cap for this
        # segment as a fraction of the machine's, speed staying at the
        # clamp; the reward taxes the fraction, so a high one is spent on
        # strikes and saves, not on wandering (heat is torque, torque is
        # accel). "profile_v": 4-dim (x, y, speed_frac, accel_frac).
        # Productionizable by construction: the Teensy takes a runtime
        # ACCEL alongside CMD, so this action IS its command set.
        action_mode: str = "position",     # "position" | "profile_a" | "profile_v"
        # Bound the AGENT to the box the machine can actually reach, rather
        # than to its half of the table. Off only for ablations; on it, the
        # policy would learn to use 65% of the half that does not exist.
        # The OPPONENT is deliberately left with the full half: it stands in
        # for a human, who can reach anywhere on their side.
        constrain_to_workspace: bool = True,
        # Who the far side IS. "human" (default): the sparring partner
        # described above -- its own dynamics law, its own caps, the whole
        # half. "robot": an exact copy of the agent's body -- same law, same
        # caps and the same per-env DR draw, the workspace box mirrored
        # across the centre line -- so self-play is a game between two
        # copies of one machine. The side flag then stays ROBOT_SIDE in both
        # views (there is no other kind of body on the table), and
        # opponent_obs() builds the far side's view natively: own paddle
        # fresh, puck and rival through the camera, exactly as the agent's.
        opponent_body: str = "human",      # "human" | "robot"
        # Shot-type requests in the observation ([17:20]); see OBS_DIM.
        shot_types: bool = False,
        shot_type_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        # Per-EPISODE opponent draw: {policy name: probability}. Each env
        # rolls a fresh opponent kind at every reset, so one batch plays a
        # mix over time rather than a fixed split (opponent_mix, which is
        # counts, stays for the eval harness). "external" is the trainer's
        # own checkpoint.
        opponent_mix_probs: dict[str, float] | None = None,
        # Sensing fuzz: fraction of episodes that get dropout spells (needs
        # realistic_perception; ignored without the camera ring).
        fuzz_p: float = 0.0,
    ):
        self.n_envs = n_envs
        self.table_config = table_config or TableConfig()
        if obs_mode not in ("kinematic", "history"):
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        if action_mode not in ("position", "profile_a", "profile_v"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.action_dim = {"position": 2, "profile_a": 3, "profile_v": 4}[action_mode]
        # Previous action is part of the history observation; kept at the
        # full 4 slots regardless of mode so the layout never shifts.
        self._prev_action = np.zeros((n_envs, 4), dtype=np.float32)
        self.physics_dt = physics_dt
        self.action_dt = action_dt
        self.max_episode_time = max_episode_time
        self.max_episode_steps = max_episode_steps  # None = use time-based truncation
        self.max_score = max_score
        self.opponent_policy = opponent_policy
        self.score_handicap = score_handicap
        self.agent_dynamics_type = agent_dynamics
        self.opponent_dynamics_type = opponent_dynamics
        self.domain_randomize = domain_randomize
        self.frame_stack = 1  # always 1; velocities replace stacking

        # Per-env opponent policy
        self._opp_mix_ids = None
        self._opp_mix_p = None
        if opponent_mix_probs:
            names = list(opponent_mix_probs)
            probs = np.array([opponent_mix_probs[k] for k in names], dtype=float)
            if probs.sum() <= 0:
                raise ValueError("opponent_mix_probs must have positive weight")
            self._opp_mix_ids = np.array([_OPP_POLICY_MAP[k] for k in names], dtype=np.int8)
            self._opp_mix_p = probs / probs.sum()
        if opponent_mix is not None:
            total = sum(opponent_mix.values())
            assert total == n_envs, (
                f"opponent_mix sums to {total}, expected n_envs={n_envs}"
            )
            ids = []
            for policy_name, count in opponent_mix.items():
                ids.extend([_OPP_POLICY_MAP[policy_name]] * count)
            self._opp_policy_id = np.array(ids, dtype=np.int8)
        else:
            self._opp_policy_id = np.full(
                n_envs, _OPP_POLICY_MAP[opponent_policy], dtype=np.int8
            )

        self.engine = BatchPhysicsEngine(n_envs, self.table_config,
                                         domain_randomize=domain_randomize)

        # Per-env step counter for step-based truncation
        self._step_count = np.zeros(n_envs, dtype=np.int32)

        cfg = self.table_config
        self.n_substeps = max(1, int(action_dt / physics_dt))
        self.sub_dt = action_dt / self.n_substeps

        # Action rescaling bounds: the FULL half, for both sides and both
        # settings of constrain_to_workspace.
        #
        # The action space used to map onto the reachable box, so the policy
        # could not even EXPRESS an unreachable target. That changed for two
        # reasons. First, self-play: one network drives both bodies through
        # one action space, and the human side genuinely reaches the whole
        # half -- mapping the shared space onto the robot's box would shrink
        # the human's reach by the robot's limits. Second, the box is a
        # per-machine fact that may move (respooling, anchor changes), and a
        # policy whose action UNITS are the box has to be retrained when it
        # does; one that speaks table coordinates and learns the boundary
        # from a penalty does not.
        #
        # Unreachable commands are clamped to the nearest reachable point and
        # charged WS_PENALTY_PER_UNIT per sim-unit of overshoot in step().
        self.constrain_to_workspace = constrain_to_workspace
        self._ws = (workspace_in_sim(cfg.width, cfg.height / 2)
                    if constrain_to_workspace else None)
        self._action_low = np.array([cfg.paddle_radius, cfg.paddle_radius])
        self._action_high = np.array(
            [cfg.width - cfg.paddle_radius, cfg.height / 2 - cfg.paddle_radius]
        )

        if opponent_body not in ("human", "robot"):
            raise ValueError(f"unknown opponent_body {opponent_body!r}")
        self.opponent_body = opponent_body
        # The far side's reachable box: the agent's, reflected in the centre
        # line. None for a human, who reaches the whole half.
        self._ws_opp = None
        if opponent_body == "robot" and self._ws is not None:
            self._ws_opp = {
                "min_x": self._ws["min_x"], "max_x": self._ws["max_x"],
                "min_y": cfg.height - self._ws["max_y"],
                "max_y": cfg.height - self._ws["min_y"],
            }

        # Observation bounds
        vel_max = 10.0
        self.obs_high = np.array([
            cfg.width, cfg.height, vel_max, vel_max,      # puck
            cfg.width, cfg.height, vel_max, vel_max,      # paddle
            cfg.width, cfg.height, vel_max, vel_max,      # opponent
            1.0,                                          # side flag
            # Caps, as a ratio to the robot's nominal. Bounds are the largest
            # either feature ever takes: the human side's speed, and the
            # human side's accel at the top of its DR band (1.125 x 80).
            OPPONENT_MAX_SPEED_M_S / MAX_SPEED_M_S,
            DR_ACCEL_RANGE[1] * OPPONENT_MAX_ACCEL_M_S2 / MAX_ACCEL_M_S2,
            1.0, 1.0, 1.0,                                # previous action
            1.0, 1.0, 1.0,                                # shot type one-hot
            1.0,                                          # time on side
        ], dtype=np.float32)

        # ── Sensing ─────────────────────────────────────────────────────
        # Two mutually exclusive models.
        #
        # realistic_perception=True: a simulated 200 Hz CAMERA ticks inside
        # the physics loop -- every camera frame runs the perception model
        # (noise, blind spot, 6-frame velocity slope, now spanning the same
        # 30 ms as the real PuckTracker) and snapshots what the tracker
        # would report into a frame ring. The policy reads the newest frame
        # at least `latency` old, with per-env latency drawn from the
        # measured 5.1-10.3 ms band. On the 5 ms frame grid that realises
        # as 1-2 frames -- mean ~7.7 ms, spread one frame interval, exactly
        # what vision/bin/measure_latency.py observed. The delay is no
        # longer quantised by the ACTION rate at all; camera_delay (a
        # (min,max) tuple in seconds) sets the latency band.
        #
        # Only the PUCK and the OPPONENT go through the camera. The robot's
        # own paddle is observed fresh: on the table the controller knows
        # its own commanded position with ~zero latency, so delaying
        # self-state modelled a handicap the real system does not have.
        #
        # realistic_perception=False: legacy whole-obs ring delayed by
        # whole ACTION steps (kept for ablations and tests; with
        # camera_delay=0.0 it is a no-op and observations are truth).
        self._obs_dim = (self.HISTORY_OBS_DIM if obs_mode == "history"
                         else self.OBS_DIM)
        self.obs_dim = self._obs_dim
        self._env_idx = np.arange(n_envs)
        # The camera ring runs whenever anything reads frames from it:
        # realistic sensing (latency + tracker model) or history obs (which
        # are frames by definition -- truth frames when sensing is off).
        self._cam_active = realistic_perception or obs_mode == "history"
        if self._cam_active:
            self._cam_dt = FRAME_INTERVAL_S
            self._cam_every = int(round(self._cam_dt / physics_dt))
            if abs(self._cam_every * physics_dt - self._cam_dt) > 1e-9:
                raise ValueError(
                    f"physics_dt {physics_dt} must divide the camera frame "
                    f"interval {self._cam_dt} for the camera clock to tick "
                    "on substep boundaries")
            if realistic_perception:
                lo, hi = (camera_delay if isinstance(camera_delay, tuple)
                          else (camera_delay, camera_delay))
            else:
                lo = hi = 0.0                    # clean sim: no latency
            self._cam_latency_range = (float(lo), float(hi))
            max_lag = max(1, int(round(hi / self._cam_dt)))
            hist_depth = (max(self.HISTORY_PUCK_LAGS) if obs_mode == "history"
                          else 0)
            self._cam_ring_size = max_lag + hist_depth + 2
            # Per-frame tracker report: puck x,y,vx,vy + opponent x,y, and
            # the agent paddle x,y so a robot-bodied far side can see its
            # rival through the same camera (opponent_obs()).
            # ... plus [8] a flag: the paddles were HIDDEN in this frame
            # (sensing fuzz), so the readers zero the finite-difference
            # velocity across the edges of a spell as the encoder does.
            self._cam_ring = np.zeros((self._cam_ring_size, n_envs, 9))
            self._cam_write = 0
            self._cam_lag = np.zeros(n_envs, dtype=np.int32)
            self._max_delay = 0                  # legacy ring off
            self._delay_range = (0, 0)
        else:
            if isinstance(camera_delay, tuple):
                self._delay_range = (
                    max(0, int(round(camera_delay[0] / action_dt))),
                    max(0, int(round(camera_delay[1] / action_dt))),
                )
            else:
                d = max(0, int(round(camera_delay / action_dt)))
                self._delay_range = (d, d)
            self._max_delay = self._delay_range[1]
            # Per-env delay in steps [N]
            self._delay_steps = np.full(n_envs, self._max_delay, dtype=np.int32)
            if self._max_delay > 0:
                self._ring_size = self._max_delay + 1
                self._obs_ring = np.zeros(
                    (self._ring_size, n_envs, self._obs_dim), dtype=np.float32,
                )
                self._ring_write = 0

        # Vectorized delayed dynamics state (for agent and opponent)
        self._agent_dyn = self._make_dynamics_state(agent_dynamics, n_envs,
                                                     dynamics_max_speed,
                                                     dynamics_max_accel,
                                                     dynamics_time_constant)
        if opponent_body == "robot":
            self.opponent_dynamics_type = agent_dynamics
            self._opp_dyn = self._make_dynamics_state(
                agent_dynamics, n_envs, dynamics_max_speed,
                dynamics_max_accel, dynamics_time_constant)
        else:
            self._opp_dyn = self._make_dynamics_state(
                opponent_dynamics, n_envs, OPPONENT_MAX_SPEED_M_S,
                OPPONENT_MAX_ACCEL_M_S2, dynamics_time_constant)

        # A second far-side body for the scripted free-body opponents
        # (sniper, weak goalie): first-order lag, per-env caps set by the
        # script. Envs whose opponent kind is one of those read their
        # paddle from here; the rest from _opp_dyn. Both are advanced every
        # substep and re-synced to the paddle at reset, so switching kinds
        # between episodes never jumps the paddle.
        self._opp_dyn_free = self._make_dynamics_state(
            "delayed", n_envs, OPPONENT_MAX_SPEED_M_S, OPPONENT_MAX_ACCEL_M_S2,
            dynamics_time_constant)
        self._opp_free = np.zeros(n_envs, dtype=bool)
        # Sniper state: phase (0 wait, 1 strike, 2 cooldown), time in phase,
        # the aim point on the robot's goal line and the strike speed.
        self._sniper_phase = np.zeros(n_envs, dtype=np.int8)
        self._sniper_t = np.zeros(n_envs)
        self._sniper_aim_x = np.full(n_envs, cfg.width / 2.0)
        self._sniper_speed = np.full(n_envs, self.SNIPER_STRIKE_SPEED[0])

        # The firmware keeps the cart's PATH inside the box, not only its
        # target (motionProfileContain); the sim's profile body gets the same
        # box, in mm, so the two agree by construction.
        if self._ws is not None:
            self._agent_dyn["bounds_mm"] = tuple(
                1000.0 * v for v in (self._ws["min_x"], self._ws["max_x"],
                                     self._ws["min_y"], self._ws["max_y"]))
        if self._ws_opp is not None:
            self._opp_dyn["bounds_mm"] = tuple(
                1000.0 * v for v in (self._ws_opp["min_x"], self._ws_opp["max_x"],
                                     self._ws_opp["min_y"], self._ws_opp["max_y"]))

        self._rng = np.random.default_rng()
        # The perception model ticks at the CAMERA rate, not the action
        # rate: its 6-frame slope window then spans 30 ms like the real
        # 200 Hz tracker's, instead of the 60 ms it smeared over when it
        # ran per action step.
        self._perception = (
            PuckPerception(n_envs, cfg.width, cfg.height, self._cam_dt,
                           self._rng)
            if realistic_perception else None)

        # External opponent targets (for "external" policy)
        self._ext_opp_target_x = np.full(n_envs, cfg.width / 2)
        self._ext_opp_target_y = np.full(n_envs, cfg.height * 0.85)

        # Previous paddle positions for velocity estimation
        self._prev_agent_x = np.zeros(n_envs)
        self._prev_agent_y = np.zeros(n_envs)
        self._prev_opp_x = np.zeros(n_envs)
        self._prev_opp_y = np.zeros(n_envs)
        # For opponent_obs(): the far side's own paddle and, with the camera
        # off, its rival. Kept apart from the agent's _prev_* so the two
        # views can be built in either order within a step.
        self._prev_own_opp_x = np.zeros(n_envs)
        self._prev_own_opp_y = np.zeros(n_envs)
        self._prev_rival_x = np.zeros(n_envs)
        self._prev_rival_y = np.zeros(n_envs)
        # The far side's previous normalised action, for its view, and the
        # accel fraction its last command asked for (profile_a).
        self._prev_opp_action = np.zeros((n_envs, 3), dtype=np.float32)
        self._ext_opp_accel_frac = np.ones(n_envs)
        # Seconds since the puck last crossed the centre line.
        self._t_side = np.zeros(n_envs)

        # Sensing fuzz schedule: per env, the start/end times (episode
        # seconds) of the opponent and puck dropout spells; -1 = none.
        self.fuzz_p = float(fuzz_p)
        self._fuzz_opp = np.full((n_envs, self.FUZZ_OPP_WINDOWS[1], 2), -1.0)
        self._fuzz_puck = np.full((n_envs, self.FUZZ_PUCK_WINDOWS[1], 2), -1.0)
        self._fuzzed = np.zeros(n_envs, dtype=bool)
        self._opp_default = (cfg.width / 2.0, cfg.height * 0.85)   # = deploy's
        self._rival_default = (cfg.width / 2.0, cfg.height * 0.15)

        # Shot-type requests, one per side, and the possession edges that
        # trigger a draw (the puck entering a half).
        if shot_types and obs_mode == "history":
            raise NotImplementedError("shot types are kinematic-obs only")
        self.shot_types = bool(shot_types)
        self._shot_type_p = np.asarray(shot_type_probs, dtype=float)
        if self._shot_type_p.shape != (4,) or abs(self._shot_type_p.sum() - 1.0) > 1e-6:
            raise ValueError("shot_type_probs must be four probabilities summing to 1")
        self._shot_type = np.zeros(n_envs, dtype=np.int8)
        self._shot_type_opp = np.zeros(n_envs, dtype=np.int8)
        self._prev_in_half = np.zeros(n_envs, dtype=bool)
        self._prev_in_far = np.zeros(n_envs, dtype=bool)

        # Puck-stuck detection: reset if speed < threshold for N consecutive steps
        self._puck_slow_count = np.zeros(n_envs, dtype=np.int32)
        self._stuck_unattended_steps = max(1, int(round(self.STUCK_UNATTENDED_S / action_dt)))
        self._stuck_attended_steps = max(1, int(round(self.STUCK_ATTENDED_S / action_dt)))

    @staticmethod
    def _clear_profile_accel(dyn: dict[str, Any], idx) -> None:
        """Drop the jerk slew state on reset.

        Position and velocity are already zeroed above, but the firmware law
        carries ACCELERATION too, and it is not derived from them -- an
        episode starting with a stale acceleration would spend its first
        milliseconds unwinding a manoeuvre from the previous one. Same reason
        CDPR::tick zeroes accX_/accY_ when it parks.
        """
        cart = dyn.get("cart")
        if cart is not None:
            cart.ax[idx] = 0.0
            cart.ay[idx] = 0.0

    @staticmethod
    def _make_dynamics_state(
        dyn_type: str, n: int, max_speed: float, max_accel: float, tc: float
    ) -> dict[str, Any]:
        """Create vectorized dynamics state arrays."""
        # "profile" carries a CartState as well, because the firmware law has
        # ACCELERATION as state -- it slews it to bound jerk, so the same
        # command produces different motion depending on what the
        # acceleration was. The other two types are memoryless in that
        # respect and do not need it.
        extra = {"cart": CartState(n)} if dyn_type == "profile" else {}
        return {
            **extra,
            "type": dyn_type,
            "x": np.zeros(n),
            "y": np.zeros(n),
            "vx": np.zeros(n),
            "vy": np.zeros(n),
            "max_speed": np.full(n, max_speed),
            "max_accel": np.full(n, max_accel),
            "time_constant": np.full(n, tc),
            # The caps this side is NOMINALLY built with, kept as scalars so
            # domain randomisation can scale each side by its own limits.
            # The arrays above get overwritten on every randomised reset.
            "nominal_speed": max_speed,
            "nominal_accel": max_accel,
            # Jerk ramp, seconds. Matches MOTION_ACCEL_RAMP_S in the firmware.
            "ramp_s": 0.003,
            # (x_min, x_max, y_min, y_max) in mm for the profile law's path
            # containment; set by the constructor once the boxes are known.
            "bounds_mm": None,
        }

    def _reset_agent_into_workspace(self, mask: np.ndarray | None) -> None:
        """Start the robot somewhere it can actually stand.

        The engine draws the agent paddle uniformly over its HALF, which is
        2.8x the reachable box, so 58% of episodes began outside it. Every one
        of those spent its first steps being dragged back by the action clamp
        -- from a pose the machine cannot hold, having possibly already touched
        the puck there. Constraining the action but not the start state
        constrains nothing.
        """
        if self._ws is None:
            return
        idx = slice(None) if mask is None else mask
        n = self.n_envs if mask is None else int(mask.sum())
        if n == 0:
            return
        ws = self._ws
        self.engine.paddle_agent_x[idx] = self._rng.uniform(
            ws["min_x"], ws["max_x"], size=n)
        self.engine.paddle_agent_y[idx] = self._rng.uniform(
            ws["min_y"], ws["max_y"], size=n)

    def _reset_opponent_into_workspace(self, mask: np.ndarray) -> None:
        """The far side's counterpart, for a robot-bodied opponent."""
        n = int(mask.sum())
        if n == 0:
            return
        ws = self._ws_opp
        self.engine.paddle_opp_x[mask] = self._rng.uniform(
            ws["min_x"], ws["max_x"], size=n)
        self.engine.paddle_opp_y[mask] = self._rng.uniform(
            ws["min_y"], ws["max_y"], size=n)

    def reset(
        self,
        seed: int | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset environments. Returns observations [N, 14].

        If mask is provided, only resets the specified environments.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # The perception model was handed the PREVIOUS generator in the
            # constructor, so reseeding here left the sensing noise running off
            # an unseeded stream -- and once noise differs, so does every
            # collision downstream of it.
            if self._perception is not None:
                self._perception.set_rng(self._rng)

        self.engine.reset(self._rng, mask=mask)
        self._reset_agent_into_workspace(mask)

        # A fresh opponent kind for every env that resets, when a mix is on.
        if self._opp_mix_ids is not None:
            r_idx = slice(None) if mask is None else mask
            n_r = self.n_envs if mask is None else int(mask.sum())
            if n_r:
                self._opp_policy_id[r_idx] = self._rng.choice(
                    self._opp_mix_ids, size=n_r, p=self._opp_mix_p)

        # Reset per-env step counters and stuck detection
        if mask is None:
            self._step_count[:] = 0
            self._puck_slow_count[:] = 0
        else:
            self._step_count[mask] = 0
            self._puck_slow_count[mask] = 0

        # Apply score handicaps for self-play training
        if self.score_handicap:
            self._apply_score_handicap(mask)

        # Sync dynamics state with paddle positions
        if mask is None:
            idx = slice(None)
            n = self.n_envs
        else:
            idx = mask
            n = int(mask.sum())
            if n == 0:
                return self._make_obs_direct()

        # Domain randomization: per-env motor dynamics.
        #
        # The two sides are randomised DIFFERENTLY, on purpose. The AGENT
        # uses absolute bands: speed pinned to the firmware clamp (the one
        # non-estimate in the actuator model) and accel spread wide, because
        # accel is what actually binds -- see AGENT_DR_* in dynamics.py. The
        # OPPONENT stands in for a human and scales by its own nominal caps;
        # using the robot's constants for it -- as this once did -- made the
        # sparring partner slower on average than the machine it stretches.
        if self.domain_randomize:
            a = self._agent_dyn
            a["max_speed"][idx] = self._rng.uniform(
                AGENT_DR_SPEED_M_S[0], AGENT_DR_SPEED_M_S[1], size=n)
            a["max_accel"][idx] = self._rng.uniform(
                AGENT_DR_ACCEL_M_S2[0], AGENT_DR_ACCEL_M_S2[1], size=n)
            a["time_constant"][idx] = self._rng.uniform(0.01, 0.04, size=n)

            o = self._opp_dyn
            slo, shi = DR_SPEED_RANGE
            alo, ahi = DR_ACCEL_RANGE
            o["max_speed"][idx] = self._rng.uniform(
                slo * o["nominal_speed"], shi * o["nominal_speed"], size=n)
            o["max_accel"][idx] = self._rng.uniform(
                alo * o["nominal_accel"], ahi * o["nominal_accel"], size=n)
            o["time_constant"][idx] = self._rng.uniform(0.01, 0.04, size=n)
            if self.opponent_body == "robot":
                # One machine, two copies: the far side gets the SAME draw,
                # not a second sample from the same band.
                for key in ("max_speed", "max_accel", "time_constant"):
                    o[key][idx] = a[key][idx]

        self._agent_dyn["x"][idx] = self.engine.paddle_agent_x[idx]
        self._agent_dyn["y"][idx] = self.engine.paddle_agent_y[idx]
        self._agent_dyn["vx"][idx] = 0.0
        self._agent_dyn["vy"][idx] = 0.0
        self._clear_profile_accel(self._agent_dyn, idx)

        # Override opponent position for stationary policies (per-env)
        cfg = self.table_config
        resetting = np.ones(self.n_envs, dtype=bool) if mask is None else mask

        goalie_reset = resetting & (self._opp_policy_id == OPP_GOALIE)
        if np.any(goalie_reset):
            self.engine.paddle_opp_x[goalie_reset] = cfg.width / 2
            self.engine.paddle_opp_y[goalie_reset] = cfg.height - cfg.paddle_radius

        corner_reset = resetting & (self._opp_policy_id == OPP_CORNER)
        if np.any(corner_reset):
            n_c = int(corner_reset.sum())
            corners = np.array([
                [cfg.paddle_radius, cfg.height - cfg.paddle_radius],
                [cfg.width - cfg.paddle_radius, cfg.height - cfg.paddle_radius],
            ])
            picks = self._rng.integers(0, len(corners), size=n_c)
            self.engine.paddle_opp_x[corner_reset] = corners[picks, 0]
            self.engine.paddle_opp_y[corner_reset] = corners[picks, 1]

        # The free-body scripts start on their stations, whatever the
        # far-side body is, and are exempt from the robot box below.
        self._opp_free = np.isin(self._opp_policy_id, (OPP_SNIPER, OPP_WEAK_GOALIE))
        sniper_reset = resetting & (self._opp_policy_id == OPP_SNIPER)
        if np.any(sniper_reset):
            self.engine.paddle_opp_x[sniper_reset] = cfg.width / 2
            self.engine.paddle_opp_y[sniper_reset] = cfg.height - self.SNIPER_STATION_Y
            self._sniper_phase[sniper_reset] = 0
            self._sniper_t[sniper_reset] = 0.0
        weak_reset = resetting & (self._opp_policy_id == OPP_WEAK_GOALIE)
        if np.any(weak_reset):
            self.engine.paddle_opp_x[weak_reset] = cfg.width / 2
            self.engine.paddle_opp_y[weak_reset] = cfg.height - self.WEAK_STATION_Y

        if self._ws_opp is not None:
            boxed = resetting & ~self._opp_free
            self._reset_opponent_into_workspace(
                boxed & ~goalie_reset & ~corner_reset)
            # Scripted stations sit on the back wall, which a robot-bodied
            # far side cannot reach: hold them at its box edge instead.
            w = self._ws_opp
            self.engine.paddle_opp_x[boxed] = np.clip(
                self.engine.paddle_opp_x[boxed], w["min_x"], w["max_x"])
            self.engine.paddle_opp_y[boxed] = np.clip(
                self.engine.paddle_opp_y[boxed], w["min_y"], w["max_y"])

        for body in (self._opp_dyn, self._opp_dyn_free):
            body["x"][idx] = self.engine.paddle_opp_x[idx]
            body["y"][idx] = self.engine.paddle_opp_y[idx]
            body["vx"][idx] = 0.0
            body["vy"][idx] = 0.0
        self._clear_profile_accel(self._opp_dyn, idx)

        # Init previous positions (zero velocity at start)
        if self._perception is not None:
            self._perception.reset(self.engine.puck_x, self.engine.puck_y, idx)
        if self._cam_active:
            # Fill the camera ring with the reset state so the first reads
            # see a stationary, correctly-placed world rather than frames
            # from the previous episode, and draw each env's latency from
            # the measured band (zero when sensing realism is off).
            e = self.engine
            for f in range(self._cam_ring_size):
                self._cam_ring[f, idx, 0] = e.puck_x[idx]
                self._cam_ring[f, idx, 1] = e.puck_y[idx]
                self._cam_ring[f, idx, 2] = 0.0
                self._cam_ring[f, idx, 3] = 0.0
                self._cam_ring[f, idx, 4] = e.paddle_opp_x[idx]
                self._cam_ring[f, idx, 5] = e.paddle_opp_y[idx]
                self._cam_ring[f, idx, 6] = e.paddle_agent_x[idx]
                self._cam_ring[f, idx, 7] = e.paddle_agent_y[idx]
                self._cam_ring[f, idx, 8] = 0.0
            lo, hi = self._cam_latency_range
            n_r = self.n_envs if mask is None else int(mask.sum())
            if hi > 0:
                lat = self._rng.uniform(lo, hi, size=n_r)
                self._cam_lag[idx] = np.clip(
                    np.round(lat / self._cam_dt).astype(np.int32),
                    1, self._cam_ring_size - 2)
            else:
                self._cam_lag[idx] = 0
        self._prev_action[idx] = 0.0

        self._prev_agent_x[idx] = self.engine.paddle_agent_x[idx]
        self._prev_agent_y[idx] = self.engine.paddle_agent_y[idx]
        self._prev_opp_x[idx] = self.engine.paddle_opp_x[idx]
        self._prev_opp_y[idx] = self.engine.paddle_opp_y[idx]
        self._prev_own_opp_x[idx] = self.engine.paddle_opp_x[idx]
        self._prev_own_opp_y[idx] = self.engine.paddle_opp_y[idx]
        self._prev_rival_x[idx] = self.engine.paddle_agent_x[idx]
        self._prev_rival_y[idx] = self.engine.paddle_agent_y[idx]
        self._prev_opp_action[idx] = 0.0

        self._draw_fuzz(resetting)

        # Shot-type requests restart with the episode: none until the puck
        # first enters a half, which for a puck launched from the centre is
        # a few steps away; one already inside a half is drawn for now.
        half = cfg.height / 2.0
        self._shot_type[idx] = 0
        self._shot_type_opp[idx] = 0
        self._t_side[idx] = 0.0
        self._ext_opp_accel_frac[idx] = 1.0
        self._prev_in_half[idx] = self.engine.puck_y[idx] < half
        self._prev_in_far[idx] = self.engine.puck_y[idx] > half
        self._draw_shot_types(resetting & self._prev_in_half, agent=True)
        self._draw_shot_types(resetting & self._prev_in_far, agent=False)

        # Pre-fill camera delay buffer for reset envs
        if self._max_delay > 0:
            obs_now = self._make_obs_direct()
            lo, hi = self._delay_range
            if mask is None:
                # Full reset: randomize per-env delays, fill entire ring buffer
                if lo == hi:
                    self._delay_steps[:] = lo
                else:
                    self._delay_steps[:] = self._rng.integers(lo, hi + 1, size=self.n_envs)
                for t in range(self._ring_size):
                    self._obs_ring[t] = obs_now
                self._ring_write = 0
                return obs_now
            else:
                # Partial reset: re-randomize delays for reset envs, fill their slots.
                n_reset = int(mask.sum())
                if lo == hi:
                    self._delay_steps[mask] = lo
                else:
                    self._delay_steps[mask] = self._rng.integers(lo, hi + 1, size=n_reset)
                for t in range(self._ring_size):
                    self._obs_ring[t, mask] = obs_now[mask]
                return obs_now

        return self._make_obs_direct()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all N environments.

        Args:
            actions: [N, 2] normalized actions in [-1, 1].

        Returns:
            obs: [N, 14]
            rewards: [N]
            terminated: [N] bool
            truncated: [N] bool
            info: dict with batched arrays
        """
        # Clip and rescale actions to real positions
        actions = np.clip(actions, -1.0, 1.0)
        target_x = self._action_low[0] + (actions[:, 0] + 1.0) * 0.5 * (self._action_high[0] - self._action_low[0])
        target_y = self._action_low[1] + (actions[:, 1] + 1.0) * 0.5 * (self._action_high[1] - self._action_low[1])

        # Velocity-carrying action: dims 2-3 command the caps for this
        # segment as fractions of the MACHINE caps ([-1,1] -> 5%..100%), so
        # a command can be gentle or a full-force strike but never exceed
        # what the DR-sampled machine -- and on the table, the Teensy's
        # LIMITS clamp -- can deliver.
        if self.action_mode == "profile_v":
            v_frac = 0.05 + (actions[:, 2] + 1.0) * 0.5 * 0.95
            a_frac = 0.05 + (actions[:, 3] + 1.0) * 0.5 * 0.95
            agent_speed_cap = v_frac * self._agent_dyn["max_speed"]
            agent_accel_cap = a_frac * self._agent_dyn["max_accel"]
        elif self.action_mode == "profile_a":
            a_frac = self.accel_fraction(actions[:, 2])
            agent_speed_cap = None
            agent_accel_cap = a_frac * self._agent_dyn["max_accel"]
        else:
            agent_speed_cap = agent_accel_cap = None
        self._prev_action[:, :actions.shape[1]] = actions

        cfg = self.table_config
        rewards = np.zeros(self.n_envs)

        # The action space spans the full half; the machine does not. An
        # unreachable command is capped at the closest reachable point --
        # exactly what the firmware would do -- and charged for the overshoot,
        # so the boundary is learned in sim rather than discovered on the
        # hardware as a paddle that silently stops short.
        if self._ws is not None:
            cx = np.clip(target_x, self._ws["min_x"], self._ws["max_x"])
            cy = np.clip(target_y, self._ws["min_y"], self._ws["max_y"])
            ws_overshoot = np.hypot(target_x - cx, target_y - cy)
            rewards -= self.WS_PENALTY_PER_UNIT * ws_overshoot
            target_x, target_y = cx, cy
        else:
            ws_overshoot = np.zeros(self.n_envs)

        for sub in range(self.n_substeps):
            dt = self.sub_dt

            # Update agent paddle through dynamics
            ax, ay = self._update_dynamics(self._agent_dyn, target_x, target_y,
                                           dt, speed_cap=agent_speed_cap,
                                           accel_cap=agent_accel_cap)
            ax, ay = self._clamp_to_half(ax, ay, agent=True)
            self.engine.update_paddle_agent(ax, ay, dt)

            # Update opponent
            ox, oy = self._opponent_action(dt)
            ox, oy = self._clamp_to_half(ox, oy, agent=False)
            self.engine.update_paddle_opponent(ox, oy, dt)

            self.engine.step(dt)

            # Camera tick: the simulated 200 Hz camera lives IN the physics
            # loop, not at the action boundary, so sensing latency is a
            # property of the world rather than of the control rate.
            if self._cam_active and (sub + 1) % self._cam_every == 0:
                self._camera_tick()

            # Accumulate goal rewards
            rewards += np.where(self.engine.goal_scored == 1, 1.0, 0.0)
            rewards += np.where(self.engine.goal_scored == -1, -1.0, 0.0)

        self._step_count += 1

        # Puck-stuck relaunch: a puck at rest with NOBODY near it for
        # STUCK_UNATTENDED_S is dead and gets relaunched from the centre. A
        # puck at rest WITH a paddle on it is being controlled -- the
        # trap-then-shoot play the reward now asks for (ai/RETRAIN.md item
        # 3) -- and gets STUCK_ATTENDED_S before the same relaunch, so
        # control is possible but cannot become holding the puck for ever.
        # Before 2026-09-06 the rule was 120 steps regardless, which at
        # 100 Hz yanked the puck away 1.2 s after a stop and fined the
        # agent for having stopped it.
        puck_speed = np.hypot(self.engine.puck_vx, self.engine.puck_vy)
        slow = puck_speed < 0.05
        self._puck_slow_count = np.where(slow, self._puck_slow_count + 1, 0)
        attended = (
            (np.hypot(self.engine.puck_x - self.engine.paddle_agent_x,
                      self.engine.puck_y - self.engine.paddle_agent_y) < self.ATTEND_RADIUS)
            | (np.hypot(self.engine.puck_x - self.engine.paddle_opp_x,
                        self.engine.puck_y - self.engine.paddle_opp_y) < self.ATTEND_RADIUS))
        limit = np.where(attended, self._stuck_attended_steps,
                         self._stuck_unattended_steps)
        stuck = self._puck_slow_count >= limit
        stuck_penalty = np.zeros(self.n_envs)
        if np.any(stuck):
            # Penalize if puck stalled on agent's side (agent should have hit it)
            on_agent_side = stuck & (self.engine.puck_y < self.table_config.height / 2)
            rewards[on_agent_side] -= 0.5
            stuck_penalty = np.where(on_agent_side, -0.5, 0.0)

            n_stuck = int(stuck.sum())
            rng = self._rng
            # Random direction: 50% toward agent, 50% toward opponent
            toward = rng.random(n_stuck) < 0.5
            angle = np.where(
                toward,
                rng.uniform(-np.pi * 0.8, -np.pi * 0.2, size=n_stuck),
                rng.uniform(np.pi * 0.2, np.pi * 0.8, size=n_stuck),
            )
            speed = rng.uniform(0.3, 1.5, size=n_stuck)
            cfg = self.table_config
            self.engine.puck_x[stuck] = cfg.width / 2 + rng.uniform(-0.15, 0.15, size=n_stuck)
            self.engine.puck_y[stuck] = cfg.height / 2
            self.engine.puck_vx[stuck] = speed * np.cos(angle)
            self.engine.puck_vy[stuck] = speed * np.sin(angle)
            self._puck_slow_count[stuck] = 0

        self._update_possessions()
        obs = self._make_obs()  # applies camera delay if configured

        # Termination / truncation
        terminated = (
            (self.engine.score_agent >= self.max_score)
            | (self.engine.score_opponent >= self.max_score)
        )
        if self.max_episode_steps is not None:
            truncated = self._step_count >= self.max_episode_steps
        else:
            truncated = self.engine.time >= self.max_episode_time

        info = {
            "score_agent": self.engine.score_agent.copy(),
            "score_opponent": self.engine.score_opponent.copy(),
            "time": self.engine.time.copy(),
            "puck_vx": self.engine.puck_vx.copy(),
            "puck_vy": self.engine.puck_vy.copy(),
            # TRUE positions for reward shaping. History-mode obs do not
            # carry a current snapshot at fixed indices, and rewards should
            # be computed on what happened, not on what the noisy tracker
            # believed happened.
            "puck_x": self.engine.puck_x.copy(),
            "puck_y": self.engine.puck_y.copy(),
            "pad_x": self.engine.paddle_agent_x.copy(),
            "pad_y": self.engine.paddle_agent_y.copy(),
            "opp_x": self.engine.paddle_opp_x.copy(),
            "opp_y": self.engine.paddle_opp_y.copy(),
            # How far past the reachable box this step's command pointed.
            # Worth watching in training: if it does not fall over time, the
            # policy is not learning the boundary, just paying the fine.
            "ws_overshoot": ws_overshoot,
            # The NON-goal part of the raw reward (workspace fine + stuck-puck
            # penalty). Every shaper builds its output from zero and detects
            # goals from the scoreboard, so these two only ever lived in the
            # raw array -- which no trainer feeds to the learner. Exposed here
            # so shapers can carry them; until 2026-09-01 no policy ever saw
            # the overshoot fine it was supposedly being trained with.
            "penalty": -self.WS_PENALTY_PER_UNIT * ws_overshoot + stuck_penalty,
            # The shot type asked of the agent this possession (0 = none),
            # for the shaper's shot_type_reward, and who the far side is.
            "shot_type": self._shot_type.copy(),
            "opponent_kind": self._opp_policy_id.copy(),
            "fuzzed": self._fuzzed.copy(),
            "t_side": self._t_side.copy(),
        }

        return obs, rewards, terminated, truncated, info

    def _draw_fuzz(self, mask: np.ndarray) -> None:
        """Schedule this episode's dropout spells for the envs in `mask`."""
        n = int(mask.sum())
        if n == 0:
            return
        self._fuzz_opp[mask] = -1.0
        self._fuzz_puck[mask] = -1.0
        self._fuzzed[mask] = False
        if self.fuzz_p <= 0 or self._perception is None:
            return
        rng = self._rng
        fuzzed = rng.random(n) < self.fuzz_p
        if not fuzzed.any():
            return
        idx = np.nonzero(mask)[0][fuzzed]
        self._fuzzed[idx] = True
        if self.max_episode_steps is not None:
            T = self.max_episode_steps * self.action_dt
        else:
            T = self.max_episode_time
        lo, hi = self.FUZZ_MARGIN_S, max(self.FUZZ_MARGIN_S + 0.1, T - self.FUZZ_MARGIN_S)
        for table, (k_lo, k_hi), (d_lo, d_hi) in (
                (self._fuzz_opp, self.FUZZ_OPP_WINDOWS, self.FUZZ_OPP_S),
                (self._fuzz_puck, self.FUZZ_PUCK_WINDOWS, self.FUZZ_PUCK_S)):
            m = len(idx)
            k = rng.integers(k_lo, k_hi + 1, size=m)
            for j in range(k_hi):
                on = k > j
                start = rng.uniform(lo, hi, size=m)
                dur = rng.uniform(d_lo, d_hi, size=m)
                table[idx, j, 0] = np.where(on, start, -1.0)
                table[idx, j, 1] = np.where(on, start + dur, -1.0)

    def _fuzz_active(self, table: np.ndarray) -> np.ndarray:
        """[N] bool: is any spell in `table` open at each env's time now."""
        t = self.engine.time[:, None]
        return ((t >= table[:, :, 0]) & (t < table[:, :, 1])).any(axis=1)

    @classmethod
    def accel_fraction(cls, a) -> np.ndarray:
        """Action slot [-1, 1] -> fraction of the machine's accel cap.

        QUADRATIC since run 4: slot -1 -> 5%, 0 -> 29%, +1 -> 100%. Runs 2
        and 3 mapped linearly and the policy never used the slot -- it sat
        at the network's neutral output (fraction 0.52) whether the puck
        was near or far. With the neutral output cheap, a save or a strike
        needs the slot raised, which the goal and shot rewards pay for.
        """
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        u = (a + 1.0) * 0.5
        return cls.ACCEL_FRAC_MIN + (1.0 - cls.ACCEL_FRAC_MIN) * u * u

    def _draw_shot_types(self, mask: np.ndarray, agent: bool) -> None:
        """Roll a request for each env in `mask` (no-op unless shot_types)."""
        if not self.shot_types:
            return
        if not agent and self.opponent_body != "robot":
            return          # a human far side is never asked for a shot
        n = int(mask.sum())
        if n == 0:
            return
        draw = self._rng.choice(4, size=n, p=self._shot_type_p).astype(np.int8)
        if agent:
            self._shot_type[mask] = draw
        else:
            self._shot_type_opp[mask] = draw

    def _update_possessions(self) -> None:
        """Track the puck entering each half; a new possession draws a
        fresh request. The request is kept while the puck is away rather
        than cleared -- a hit at the box's top edge can put the puck just
        over the line, and the shaper scores that hit against the request
        it was made under."""
        half = self.table_config.height / 2.0
        in_half = self.engine.puck_y < half
        in_far = self.engine.puck_y > half
        self._draw_shot_types(in_half & ~self._prev_in_half, agent=True)
        self._draw_shot_types(in_far & ~self._prev_in_far, agent=False)
        crossed = (in_half != self._prev_in_half) | (in_far != self._prev_in_far)
        self._t_side = np.where(crossed, 0.0, self._t_side + self.action_dt)
        self._prev_in_half[:] = in_half
        self._prev_in_far[:] = in_far

    def _t_side_feature(self) -> np.ndarray:
        return np.minimum(self._t_side, self.T_SIDE_CLIP) / self.T_SIDE_CLIP

    def _shot_onehot(self, types: np.ndarray) -> np.ndarray:
        """[N, 3] one-hot of [left, right, straight]; type 0 -> zeros."""
        out = np.zeros((self.n_envs, 3), dtype=np.float32)
        hot = types > 0
        out[np.nonzero(hot)[0], types[hot] - 1] = 1.0
        return out

    def auto_reset(
        self, terminated: np.ndarray, truncated: np.ndarray
    ) -> np.ndarray | None:
        """Reset any done environments and return new observations for them.

        Returns None if no envs need resetting.
        """
        done = terminated | truncated
        if not np.any(done):
            return None
        return self.reset(mask=done)

    def _apply_score_handicap(self, mask: np.ndarray | None) -> None:
        """Set initial scores for handicap training.

        70% normal (0-0), 10% agent down (0-3), 10% agent up (3-0), 10% tied (3-3).
        """
        if mask is None:
            n = self.n_envs
            idx = slice(None)
        else:
            n = int(mask.sum())
            if n == 0:
                return
            idx = mask

        rolls = self._rng.random(n)
        # 0.0-0.7: normal (already 0-0 from engine.reset)
        # 0.7-0.8: agent down 0-3
        down = rolls >= 0.7
        down &= rolls < 0.8
        # 0.8-0.9: agent up 3-0
        up = rolls >= 0.8
        up &= rolls < 0.9
        # 0.9-1.0: tied 3-3
        tied = rolls >= 0.9

        if mask is None:
            self.engine.score_agent[down] = 0
            self.engine.score_opponent[down] = 3
            self.engine.score_agent[up] = 3
            self.engine.score_opponent[up] = 0
            self.engine.score_agent[tied] = 3
            self.engine.score_opponent[tied] = 3
        else:
            # Build index arrays for masked envs
            env_indices = np.where(mask)[0]
            self.engine.score_agent[env_indices[down]] = 0
            self.engine.score_opponent[env_indices[down]] = 3
            self.engine.score_agent[env_indices[up]] = 3
            self.engine.score_opponent[env_indices[up]] = 0
            self.engine.score_agent[env_indices[tied]] = 3
            self.engine.score_opponent[env_indices[tied]] = 3

    def set_opponent_actions(
        self, target_x: np.ndarray, target_y: np.ndarray
    ) -> None:
        """Set external opponent targets for all envs."""
        self._ext_opp_target_x = target_x.copy()
        self._ext_opp_target_y = target_y.copy()

    @property
    def external_mask(self) -> np.ndarray:
        """Boolean mask [N] — True for envs using external (self-play) opponent."""
        return self._opp_policy_id == OPP_EXTERNAL

    def mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """Mirror observations [N, 12] for opponent perspective.

        Flip y positions, negate y velocities, swap agent/opponent.
        """
        if self.obs_mode == "history":
            raise NotImplementedError(
                "mirror_obs for history observations lands with SAC "
                "self-play; scripted opponents never look at an obs")
        cfg = self.table_config
        m = obs.copy()
        # Obs: [puck_x, puck_y, puck_vx, puck_vy,
        #        pad_x, pad_y, pad_vx, pad_vy,
        #        opp_x, opp_y, opp_vx, opp_vy]

        # Puck: flip y, negate vy
        m[:, 1] = cfg.height - obs[:, 1]    # puck_y
        m[:, 3] = -obs[:, 3]                # puck_vy

        # Swap agent/opponent and flip y, negate vy
        m[:, 4] = obs[:, 8]                 # opp_x → pad_x
        m[:, 5] = cfg.height - obs[:, 9]    # opp_y → pad_y (flipped)
        m[:, 6] = obs[:, 10]                # opp_vx → pad_vx
        m[:, 7] = -obs[:, 11]               # opp_vy → pad_vy (negated)
        m[:, 8] = obs[:, 4]                 # pad_x → opp_x
        m[:, 9] = cfg.height - obs[:, 5]    # pad_y → opp_y (flipped)
        m[:, 10] = obs[:, 6]                # pad_vx → opp_vx
        m[:, 11] = -obs[:, 7]               # pad_vy → opp_vy (negated)
        # The previous action follows the body too. Which body the incoming
        # view is depends on the flag; with two robot bodies the flag cannot
        # say, so the far side's is written -- that path is for scripted
        # opponents that never look, and opponent_obs() is the real one.
        was_robot = obs[:, 12] > (self.ROBOT_SIDE + self.HUMAN_SIDE) * 0.5
        m[:, 15:18] = np.where(was_robot[:, None], self._prev_opp_action[:, :3],
                               self._prev_action[:, :3])
        # The request follows the body too: the other side's, or none for
        # a human, who is never asked. Time on side is the same from both
        # ends and is left as it is.
        m[:, 18:21] = np.where(was_robot[:, None], self._shot_onehot(self._shot_type_opp),
                               self._shot_onehot(self._shot_type))
        if self.opponent_body == "robot":
            # Two copies of one body: the flag and the caps are the same on
            # both sides, and swapping them would be wrong, not symmetric.
            return m
        # FLIP the side flag rather than pinning it to HUMAN. Whoever looks
        # through a mirrored view is the other side -- so mirroring the
        # robot's view yields the human's, and mirroring that yields the
        # robot's again. Pinning it made mirror_obs stop being an involution,
        # which is exactly what its round-trip test is for.
        m[:, 12] = (self.ROBOT_SIDE + self.HUMAN_SIDE) - obs[:, 12]

        # The cap features describe the body being driven, so they follow the
        # side across the mirror. Which caps to write depends on which side the
        # INCOMING view belongs to -- always writing the opponent's would make
        # mirror(mirror(x)) != x, the same way pinning the side flag did.
        # Caps are per-env constants within an episode, so reading them live
        # rather than from the (possibly delayed) obs is safe.
        was_robot = obs[:, 12] > (self.ROBOT_SIDE + self.HUMAN_SIDE) * 0.5
        for col, key, ref in ((13, "max_speed", MAX_SPEED_M_S),
                              (14, "max_accel", MAX_ACCEL_M_S2)):
            m[:, col] = np.where(was_robot,
                                 self._opp_dyn[key],
                                 self._agent_dyn[key]) / ref
        return m

    def opponent_obs(self) -> np.ndarray:
        """The far side's OWN view, for a robot-bodied opponent.

        mirror_obs() hands the far side the agent's report turned round,
        which gives it its own paddle through the camera (stale) and its
        rival's fresh -- the opposite of what each side actually has. This
        builds the view the way _make_obs_direct builds the agent's: own
        paddle fresh from the controller, puck and rival through the same
        camera at the same latency, then reflected into the far side's frame
        so one policy can drive either end. Kinematic obs only, and the
        legacy whole-obs delay ring is not applied (realistic sensing
        carries the latency).
        """
        if self.opponent_body != "robot":
            raise ValueError("opponent_obs() is for opponent_body='robot'; "
                             "a human far side is viewed via mirror_obs()")
        if self.obs_mode != "kinematic":
            raise NotImplementedError("opponent_obs() is kinematic-only")
        e = self.engine
        cfg = self.table_config
        dt = self.action_dt
        own_vx = (e.paddle_opp_x - self._prev_own_opp_x) / dt
        own_vy = (e.paddle_opp_y - self._prev_own_opp_y) / dt
        self._prev_own_opp_x[:] = e.paddle_opp_x
        self._prev_own_opp_y[:] = e.paddle_opp_y
        if self._perception is not None:
            newest = (self._cam_write - 1) % self._cam_ring_size
            idx = (newest - self._cam_lag) % self._cam_ring_size
            prev = (idx - 1) % self._cam_ring_size
            seen = self._cam_ring[idx, self._env_idx]
            before = self._cam_ring[prev, self._env_idx]
            px, py, pvx, pvy = seen[:, 0], seen[:, 1], seen[:, 2], seen[:, 3]
            rx, ry = seen[:, 6], seen[:, 7]
            edge = (seen[:, 8] > 0) | (before[:, 8] > 0)
            rvx = np.where(edge, 0.0, (seen[:, 6] - before[:, 6]) / self._cam_dt)
            rvy = np.where(edge, 0.0, (seen[:, 7] - before[:, 7]) / self._cam_dt)
        else:
            px, py, pvx, pvy = e.puck_x, e.puck_y, e.puck_vx, e.puck_vy
            rx, ry = e.paddle_agent_x, e.paddle_agent_y
            rvx = (e.paddle_agent_x - self._prev_rival_x) / dt
            rvy = (e.paddle_agent_y - self._prev_rival_y) / dt
        self._prev_rival_x[:] = e.paddle_agent_x
        self._prev_rival_y[:] = e.paddle_agent_y
        h = cfg.height
        return np.column_stack([
            px, h - py, pvx, -pvy,
            e.paddle_opp_x, h - e.paddle_opp_y, own_vx, -own_vy,
            rx, h - ry, rvx, -rvy,
            np.full(self.n_envs, self.ROBOT_SIDE),
            self._opp_dyn["max_speed"] / MAX_SPEED_M_S,
            self._opp_dyn["max_accel"] / MAX_ACCEL_M_S2,
            self._prev_opp_action[:, :3],
            self._shot_onehot(self._shot_type_opp),
            self._t_side_feature(),
        ]).astype(np.float32)

    def mirror_action_to_opponent(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert [N, 2] normalized actions from opponent's mirrored perspective
        to real table coordinates in opponent's half."""
        actions = np.clip(actions, -1.0, 1.0)
        self._prev_opp_action[:, :actions.shape[1]] = actions[:, :3]
        if actions.shape[1] >= 3 and self.action_mode == "profile_a":
            self._ext_opp_accel_frac[:] = self.accel_fraction(actions[:, 2])
        cfg = self.table_config
        r = cfg.paddle_radius
        x = r + (actions[:, 0] + 1.0) * 0.5 * (cfg.width - 2 * r)
        # y: mirrored — y=-1 means "near own goal" = back wall (height - r),
        # y=+1 means "opponent's side" = midfield (height/2 + r)
        y = (cfg.height - r) - (actions[:, 1] + 1.0) * 0.5 * (cfg.height / 2 - 2 * r)
        return x, y

    # --- Internal helpers ---

    def _update_dynamics(
        self,
        dyn: dict[str, Any],
        target_x: np.ndarray,
        target_y: np.ndarray,
        dt: float,
        speed_cap: np.ndarray | None = None,
        accel_cap: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized dynamics update. Returns new (x, y) arrays.

        speed_cap/accel_cap: optional per-env caps for THIS command,
        below the machine limits in dyn["max_speed"/"max_accel"]. This is
        how the velocity-carrying action mode reaches the hardware: the
        Teensy accepts runtime LIMITS alongside MOVE targets, so a policy
        that modulates its caps per command is directly productionizable.
        """
        v_cap = dyn["max_speed"] if speed_cap is None else speed_cap
        a_cap = dyn["max_accel"] if accel_cap is None else accel_cap
        if dyn["type"] == "profile":
            # The real firmware control law, via fw/host. Millimetres, because
            # that is what the Teensy works in -- the law itself is
            # scale-agnostic, so sim metres * 1000 keeps position and caps
            # consistent without needing the grid frame here.
            cart = dyn["cart"]
            cart.x[:] = dyn["x"] * 1000.0
            cart.y[:] = dyn["y"] * 1000.0
            cart.vx[:] = dyn["vx"] * 1000.0
            cart.vy[:] = dyn["vy"] * 1000.0
            substeps = max(1, int(round(dt / DEFAULT_SIM_DT)))
            advance(cart,
                    (target_x * 1000.0).astype(np.float32),
                    (target_y * 1000.0).astype(np.float32),
                    v_cap * 1000.0, a_cap * 1000.0,
                    dyn["ramp_s"], dt / substeps, substeps,
                    bounds=dyn.get("bounds_mm"))
            dyn["x"] = cart.x.astype(np.float64) / 1000.0
            dyn["y"] = cart.y.astype(np.float64) / 1000.0
            dyn["vx"] = cart.vx.astype(np.float64) / 1000.0
            dyn["vy"] = cart.vy.astype(np.float64) / 1000.0
            return dyn["x"].copy(), dyn["y"].copy()

        if dyn["type"] == "ideal":
            dyn["x"] = target_x.copy()
            dyn["y"] = target_y.copy()
            return dyn["x"].copy(), dyn["y"].copy()

        # Delayed dynamics: P-controller with velocity/acceleration limits
        dx = target_x - dyn["x"]
        dy = target_y - dyn["y"]
        tc = np.maximum(dyn["time_constant"], dt)

        desired_vx = dx / tc
        desired_vy = dy / tc

        # Clamp desired velocity
        desired_speed = np.hypot(desired_vx, desired_vy)
        too_fast = desired_speed > v_cap
        factor = np.where(
            too_fast,
            v_cap / np.maximum(desired_speed, 1e-8),
            1.0,
        )
        desired_vx *= factor
        desired_vy *= factor

        # Acceleration limits
        if dt > 0:
            ax = (desired_vx - dyn["vx"]) / dt
            ay = (desired_vy - dyn["vy"]) / dt
            accel = np.hypot(ax, ay)
            too_much = accel > a_cap
            afactor = np.where(
                too_much,
                dyn["max_accel"] / np.maximum(accel, 1e-8),
                1.0,
            )
            ax *= afactor
            ay *= afactor
            dyn["vx"] += ax * dt
            dyn["vy"] += ay * dt

        # Integrate
        dyn["x"] += dyn["vx"] * dt
        dyn["y"] += dyn["vy"] * dt

        return dyn["x"].copy(), dyn["y"].copy()

    def _opponent_action(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized per-env opponent policies. Returns target (x, y) arrays."""
        cfg = self.table_config

        # Default: hold current position (idle, goalie, corner)
        target_x = self.engine.paddle_opp_x.copy()
        target_y = self.engine.paddle_opp_y.copy()

        # Follow: track puck x, stay near back wall
        follow = self._opp_policy_id == OPP_FOLLOW
        target_x[follow] = self.engine.puck_x[follow]
        target_y[follow] = cfg.height * 0.85

        # Random: random target each step
        random_mask = self._opp_policy_id == OPP_RANDOM
        n_rand = int(random_mask.sum())
        if n_rand > 0:
            target_x[random_mask] = self._rng.uniform(
                cfg.paddle_radius, cfg.width - cfg.paddle_radius, size=n_rand
            )
            target_y[random_mask] = self._rng.uniform(
                cfg.height / 2 + cfg.paddle_radius,
                cfg.height - cfg.paddle_radius,
                size=n_rand,
            )

        # External: use provided targets
        ext = self._opp_policy_id == OPP_EXTERNAL
        target_x[ext] = self._ext_opp_target_x[ext]
        target_y[ext] = self._ext_opp_target_y[ext]

        free = self._opp_free
        if np.any(free):
            fx, fy = self._free_body_targets(dt, target_x, target_y)
            target_x = np.where(free, fx, target_x)
            target_y = np.where(free, fy, target_y)

        if self._ws_opp is not None:
            # The same cap the agent's targets get in step(): the profile
            # chases the nearest reachable point, as the firmware would.
            # The free-body scripts are not the machine and keep the half.
            target_x = np.where(free, target_x, np.clip(
                target_x, self._ws_opp["min_x"], self._ws_opp["max_x"]))
            target_y = np.where(free, target_y, np.clip(
                target_y, self._ws_opp["min_y"], self._ws_opp["max_y"]))
        # The copy of the robot commands its accel too (profile_a); the
        # scripted kinds run at the body's cap.
        opp_accel_cap = None
        if self.action_mode == "profile_a":
            opp_accel_cap = np.where(self._opp_policy_id == OPP_EXTERNAL,
                                     self._ext_opp_accel_frac, 1.0) * self._opp_dyn["max_accel"]
        mx, my = self._update_dynamics(self._opp_dyn, target_x, target_y, dt,
                                       accel_cap=opp_accel_cap)
        if not np.any(free):
            return mx, my
        fx, fy = self._update_dynamics(self._opp_dyn_free, target_x, target_y, dt)
        # Keep the idle body on the paddle so a later kind switch is
        # seamless: whichever body is not driving follows the one that is.
        x = np.where(free, fx, mx)
        y = np.where(free, fy, my)
        for body in (self._opp_dyn, self._opp_dyn_free):
            body["x"][:] = x
            body["y"][:] = y
        self._opp_dyn["vx"][:] = np.where(free, self._opp_dyn_free["vx"], self._opp_dyn["vx"])
        self._opp_dyn["vy"][:] = np.where(free, self._opp_dyn_free["vy"], self._opp_dyn["vy"])
        self._opp_dyn_free["vx"][:] = self._opp_dyn["vx"]
        self._opp_dyn_free["vy"][:] = self._opp_dyn["vy"]
        return x, y

    def _free_body_targets(self, dt: float, target_x, target_y):
        """Targets and per-env caps for the sniper and the weak goalie.

        Returns (x, y) targets for every env (garbage where the env is not
        a free-body kind; the caller masks). Sets _opp_dyn_free's caps.
        """
        e = self.engine
        cfg = self.table_config
        W, H = cfg.width, cfg.height
        tx, ty = target_x.copy(), target_y.copy()
        free = self._opp_dyn_free
        ox, oy = e.paddle_opp_x, e.paddle_opp_y

        # ── weak goalie ─────────────────────────────────────────────
        weak = self._opp_policy_id == OPP_WEAK_GOALIE
        if np.any(weak):
            interested = (e.puck_y > H / 2) | (e.puck_vy > 0)
            want_x = np.where(interested, e.puck_x, W / 2)
            move = np.abs(want_x - ox) > self.WEAK_DEADZONE
            tx = np.where(weak, np.where(move, want_x, ox), tx)
            ty = np.where(weak, H - self.WEAK_STATION_Y, ty)
            free["max_speed"][weak] = self.WEAK_SPEED
            free["max_accel"][weak] = self.WEAK_ACCEL

        # ── sniper ──────────────────────────────────────────────────
        sn = self._opp_policy_id == OPP_SNIPER
        if np.any(sn):
            self._sniper_t[sn] += dt
            speed = np.hypot(e.puck_vx, e.puck_vy)
            phase = self._sniper_phase
            # Strike trigger: puck on its half, slow, in front of the paddle.
            can = (sn & (phase == 0) & (e.puck_y > H / 2 + 0.05)
                   & (e.puck_y < oy - 0.02) & (speed < self.SNIPER_MAX_PUCK_SPEED))
            n_c = int(can.sum())
            if n_c:
                r = self._rng
                mouth = cfg.goal_width / 2.0 - cfg.puck_radius
                aim = W / 2 + r.uniform(-mouth, mouth, size=n_c)
                bank = r.random(n_c) < self.SNIPER_BANK_P
                # A bank: aim at the goal's mirror image in the nearer rail
                # (specular; the real rail is lossier, which only makes the
                # sniper miss sometimes, as a human does).
                left = e.puck_x[can] < W / 2
                aim = np.where(bank, np.where(left, -aim, 2 * W - aim), aim)
                self._sniper_aim_x[can] = aim
                self._sniper_speed[can] = r.uniform(*self.SNIPER_STRIKE_SPEED, size=n_c)
                phase[can] = 1
                self._sniper_t[can] = 0.0
            # Strike ends when the time is up or the puck has gone.
            striking = sn & (phase == 1)
            over = striking & ((self._sniper_t > self.SNIPER_STRIKE_S)
                               | (e.puck_y < H / 2) | (speed > 6.0))
            phase[over] = 2
            self._sniper_t[over] = 0.0
            cooled = sn & (phase == 2) & (self._sniper_t > self.SNIPER_COOLDOWN_S)
            phase[cooled] = 0
            striking = sn & (phase == 1)
            # Through the puck, away from the aim point.
            dx = e.puck_x - self._sniper_aim_x
            dy = e.puck_y - 0.0
            n = np.maximum(np.hypot(dx, dy), 1e-6)
            ux, uy = dx / n, dy / n
            sx = e.puck_x - ux * self.SNIPER_THROUGH
            sy = e.puck_y - uy * self.SNIPER_THROUGH
            waiting = sn & ~striking
            station_x = np.clip(e.puck_x, cfg.paddle_radius, W - cfg.paddle_radius)
            tx = np.where(striking, sx, np.where(waiting, station_x, tx))
            ty = np.where(striking, sy, np.where(waiting, H - self.SNIPER_STATION_Y, ty))
            free["max_speed"][sn] = np.where(striking[sn], self._sniper_speed[sn],
                                             self.SNIPER_WAIT_SPEED)
            free["max_accel"][sn] = np.where(striking[sn], self.SNIPER_STRIKE_ACCEL,
                                             self.SNIPER_WAIT_ACCEL)
        # The half is the script's limit; the far wall is the body's.
        r = cfg.paddle_radius
        tx = np.clip(tx, r, W - r)
        ty = np.clip(ty, H / 2 + r, H - r)
        return tx, ty

    def _clamp_to_half(
        self, x: np.ndarray, y: np.ndarray, agent: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.table_config
        r = cfg.paddle_radius
        if not agent and self._ws_opp is not None:
            bx = np.clip(x, self._ws_opp["min_x"], self._ws_opp["max_x"])
            by = np.clip(y, self._ws_opp["min_y"], self._ws_opp["max_y"])
            if np.any(self._opp_free):
                hx = np.clip(x, r, cfg.width - r)
                hy = np.clip(y, cfg.height / 2 + r, cfg.height - r)
                return np.where(self._opp_free, hx, bx), np.where(self._opp_free, hy, by)
            return bx, by
        if agent and self._ws is not None:
            # The machine's own limit, not the table's. The firmware clamps
            # here too; doing it in the sim means the policy learns the
            # boundary instead of discovering it on the hardware.
            return (np.clip(x, self._ws["min_x"], self._ws["max_x"]),
                    np.clip(y, self._ws["min_y"], self._ws["max_y"]))
        x = np.clip(x, r, cfg.width - r)
        if agent:
            y = np.clip(y, r, cfg.height / 2 - r)
        else:
            y = np.clip(y, cfg.height / 2 + r, cfg.height - r)
        return x, y

    def _camera_tick(self) -> None:
        """One 200 Hz camera frame: run the tracker model, store its report.

        With realistic sensing off the "tracker" is truth -- the ring then
        serves as clean position history for history-mode observations.
        """
        e = self.engine
        hidden = None
        if self._perception is not None:
            if self.fuzz_p > 0:
                hidden = self._fuzz_active(self._fuzz_puck)
            px, py, pvx, pvy = self._perception.update(e.puck_x, e.puck_y, hidden=hidden)
        else:
            px, py, pvx, pvy = e.puck_x, e.puck_y, e.puck_vx, e.puck_vy
        w = self._cam_write
        self._cam_ring[w, :, 0] = px
        self._cam_ring[w, :, 1] = py
        self._cam_ring[w, :, 2] = pvx
        self._cam_ring[w, :, 3] = pvy
        if self._perception is not None and self.fuzz_p > 0:
            gone = self._fuzz_active(self._fuzz_opp)
            self._cam_ring[w, :, 4] = np.where(gone, self._opp_default[0], e.paddle_opp_x)
            self._cam_ring[w, :, 5] = np.where(gone, self._opp_default[1], e.paddle_opp_y)
            self._cam_ring[w, :, 6] = np.where(gone, self._rival_default[0], e.paddle_agent_x)
            self._cam_ring[w, :, 7] = np.where(gone, self._rival_default[1], e.paddle_agent_y)
            self._cam_ring[w, :, 8] = gone
        else:
            self._cam_ring[w, :, 4] = e.paddle_opp_x
            self._cam_ring[w, :, 5] = e.paddle_opp_y
            self._cam_ring[w, :, 6] = e.paddle_agent_x
            self._cam_ring[w, :, 7] = e.paddle_agent_y
            self._cam_ring[w, :, 8] = 0.0
        self._cam_write = (w + 1) % self._cam_ring_size

    def _camera_read(self):
        """The tracker report each env is entitled to see right now.

        Newest frame that is at least the env's latency old: per-env lag in
        whole frames, indexed off the write head. Opponent velocity is a
        finite difference of adjacent camera frames -- the same estimator
        the real mallet tracker amounts to.
        """
        newest = (self._cam_write - 1) % self._cam_ring_size
        idx = (newest - self._cam_lag) % self._cam_ring_size
        prev = (idx - 1) % self._cam_ring_size
        seen = self._cam_ring[idx, self._env_idx]        # [N, 9]
        before = self._cam_ring[prev, self._env_idx]
        edge = (seen[:, 8] > 0) | (before[:, 8] > 0)
        opp_vx = np.where(edge, 0.0, (seen[:, 4] - before[:, 4]) / self._cam_dt)
        opp_vy = np.where(edge, 0.0, (seen[:, 5] - before[:, 5]) / self._cam_dt)
        return seen, opp_vx, opp_vy

    def _history_obs(self) -> np.ndarray:
        """Build [N, HISTORY_OBS_DIM]: positions over time, not velocities.

        Everything the policy sees about the world is camera frames at fixed
        lags behind the newest frame it is entitled to (post-latency), so
        motion -- speed, direction, curvature, a bounce mid-window -- is the
        network's inference to make. Own state is fresh (the controller
        knows itself), and the previous action closes the loop on what was
        already commanded.
        """
        e = self.engine
        newest = (self._cam_write - 1) % self._cam_ring_size
        base = (newest - self._cam_lag) % self._cam_ring_size
        cols = []
        for lag in self.HISTORY_PUCK_LAGS:
            idx = (base - lag) % self._cam_ring_size
            frame = self._cam_ring[idx, self._env_idx]
            cols.append(frame[:, 0])
            cols.append(frame[:, 1])
        for lag in self.HISTORY_OPP_LAGS:
            idx = (base - lag) % self._cam_ring_size
            frame = self._cam_ring[idx, self._env_idx]
            cols.append(frame[:, 4])
            cols.append(frame[:, 5])
        dt = self.action_dt
        agent_vx = (e.paddle_agent_x - self._prev_agent_x) / dt
        agent_vy = (e.paddle_agent_y - self._prev_agent_y) / dt
        self._prev_agent_x[:] = e.paddle_agent_x
        self._prev_agent_y[:] = e.paddle_agent_y
        cols += [e.paddle_agent_x, e.paddle_agent_y, agent_vx, agent_vy]
        cols += [self._prev_action[:, k] for k in range(4)]
        cols += [np.full(self.n_envs, self.ROBOT_SIDE),
                 self._agent_dyn["max_speed"] / MAX_SPEED_M_S,
                 self._agent_dyn["max_accel"] / MAX_ACCEL_M_S2]
        return np.column_stack(cols).astype(np.float32)

    def _make_obs_direct(self) -> np.ndarray:
        """Build [N, OBS_DIM] observation with positions + velocities."""
        if self.obs_mode == "history":
            return self._history_obs()
        e = self.engine
        dt = self.action_dt

        # OWN paddle: fresh and true. The controller knows its own commanded
        # position with ~zero latency on the real machine; only the puck and
        # the opponent arrive through the camera.
        agent_vx = (e.paddle_agent_x - self._prev_agent_x) / dt
        agent_vy = (e.paddle_agent_y - self._prev_agent_y) / dt

        if self._perception is not None:
            seen, opp_vx, opp_vy = self._camera_read()
            px, py, pvx, pvy = seen[:, 0], seen[:, 1], seen[:, 2], seen[:, 3]
            opp_x, opp_y = seen[:, 4], seen[:, 5]
        else:
            px, py, pvx, pvy = e.puck_x, e.puck_y, e.puck_vx, e.puck_vy
            opp_x, opp_y = e.paddle_opp_x, e.paddle_opp_y
            opp_vx = (e.paddle_opp_x - self._prev_opp_x) / dt
            opp_vy = (e.paddle_opp_y - self._prev_opp_y) / dt

        # Update previous positions
        self._prev_agent_x[:] = e.paddle_agent_x
        self._prev_agent_y[:] = e.paddle_agent_y
        self._prev_opp_x[:] = e.paddle_opp_x
        self._prev_opp_y[:] = e.paddle_opp_y

        # Caps as a ratio to the robot's nominal, so a nominal robot reads
        # exactly 1.0 on both and anything else is a ratio to the machine as
        # built. The human side reads above 1.0, which is the point.
        return np.column_stack([
            px, py, pvx, pvy,
            e.paddle_agent_x, e.paddle_agent_y, agent_vx, agent_vy,
            opp_x, opp_y, opp_vx, opp_vy,
            np.full(self.n_envs, self.ROBOT_SIDE),
            self._agent_dyn["max_speed"] / MAX_SPEED_M_S,
            self._agent_dyn["max_accel"] / MAX_ACCEL_M_S2,
            self._prev_action[:, :3],
            self._shot_onehot(self._shot_type),
            self._t_side_feature(),
        ]).astype(np.float32)

    def _get_delayed_obs(self, current_obs: np.ndarray) -> np.ndarray:
        """Push current obs into ring buffer, return per-env delayed obs."""
        self._obs_ring[self._ring_write] = current_obs
        # Each env reads from its own delay offset
        read_idx = (self._ring_write - self._delay_steps) % self._ring_size
        delayed = self._obs_ring[read_idx, self._env_idx]
        self._ring_write = (self._ring_write + 1) % self._ring_size
        return delayed

    def _make_obs(self) -> np.ndarray:
        """Build observation array, applying camera delay if configured."""
        current = self._make_obs_direct()
        if self._max_delay > 0:
            return self._get_delayed_obs(current)
        return current
