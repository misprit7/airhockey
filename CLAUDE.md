# Air Hockey RL Project

## Overview
Robotic air hockey table that uses reinforcement learning trained in simulation, then transferred to physical hardware. The robot is a built and running cable-driven parallel robot (CDPR): four ClearPath servos driven step/dir by a Teensy, a FLIR camera tracking the puck at 200 Hz, and TD-MPC2 policies trained in the sim and run through `ai/bin/play.sh`. The current state of the training work and the table candidate are at the top of `ai/RETRAIN.md`; the sim-to-real status is in `SIM2REAL.md`.

## Approach
1. **System identification**: Learn physical dynamics from real hardware (motor response, cable compliance, friction, latency)
2. **Sim training**: Train RL policy in a fast custom simulator with domain randomization
3. **Sim-to-real transfer**: Deploy trained policy on physical robot, optionally fine-tune

## Project Structure
- `ai/` - RL training, simulation, and visualization
  - `airhockey/` - Python package
    - `physics.py` - Core 2D physics engine (puck, paddles, walls, collisions)
    - `batch_physics.py` - Vectorized NumPy physics for N parallel environments
    - `batch_env.py` - Batch environment wrapper (same interface, batched arrays)
    - `dynamics.py` - Pluggable motor dynamics models (ideal, delayed, learned)
    - `env.py` - Gymnasium environment wrapping the physics
    - `rewards.py` - Curriculum reward shaping (4 stages)
    - `curriculum.py` - Per-stage cosine LR scheduler
    - `recorder.py` - Game recording and replay
    - `server.py` - FastAPI WebSocket server for real-time visualization
    - `heuristics.py` - Non-learned bots (wall/goalie/striker/intercept) as
      pure functions of tracker reports in table mm -> (x, y, speed, accel).
      No sim dependency: the same objects run off `vision/bin/puck_stream.py`.
      Wall bounces use the MEASURED rail coefficients rather than specular
      reflection, so a one-bounce prediction lands where the puck does.
    - `deploy.py` - The OTHER direction: a tracker report in table mm -> the
      22-dim observation a checkpoint trained on (same estimators as the
      sim's tracker model; shot request and time on side tracked from the
      reports) -> the action back to a target in mm plus the accel cap.
      `ReportEncoder` is checkpoint-free and tested against the env's own
      observation; `TDMPC2Policy` adds the agent, planning under CUDA
      graphs at training's iterations and the run's horizon (~6-8 ms
      against the 20 ms tick). The prior alone (`--plan 0`) is a
      DEVIATION from training and is reported as one.
    - `heuristic_bridge.py` - SimBridge: BatchAirHockeyEnv history obs <-> the
      mm interface above. Reads observations only, never engine state — a bot
      scored against ground truth is scored on a table that does not exist.
    - `follow_test.py` - The TRACKING TEST (2026-09-06): a fixed move
      sequence (holds, single-axis moves, corners, then 300 ms full-height
      flips and 100 ms twitches in the policy's own style) run at whatever
      caps are applied, sampling three accounts of the paddle at 100 Hz:
      Teensy step counts, the drives' encoders, and the camera. Verdict
      CLOSE / LAGGING (drives fall behind at speed, catch up at rest) /
      LOST (disagree even at rest: model or slipped cable). The camera's
      latency is fitted out but the search is bounded at 30 ms, because a
      wider fit absorbed a 50 ms drive lag and called it close. Also fits
      encoder-mm per step-mm per motor, which is the direct check on the
      unverified 800 counts/rev step input. Run from the web UI's Motion
      limits panel ("Run tracking test", Hardware ON; camera optional);
      samples + summary land in `logs/follow_test/<stamp>.{csv,json}`.
      While it runs it OWNS the master socket — the UI's mouse path,
      limits and peak reset are gated until it finishes. Unit-tested
      against the firmware's profile body in virtual time.
    - `policy_loader.py` - Resolves a run name to a checkpoint (`latest`,
      pinned `-<step>k` snapshots, pre-scheme symlinks), remaps older
      15/17/20-wide observation layouts (`OBS_LAYOUTS`), and builds the
      agent at TRAINING's planner settings: `PLAN_ITERATIONS` 6,
      `PLAN_EVAL_MEAN`, and the horizon the run trained at
      (`trained_horizon`: `runs/<run>/run.json`, else the lineage -- 3.x
      is 8). Eval and the table both go through it.
    - `cushion_bot.py` - Scripted stop / hold / wind-up / strike controller:
      the demonstrator for `train_selfplay --demo-envs` and the "bot"
      style in `bin/income_breakdown.py`. Physics: a paddle retreating at
      0.47x the puck's speed stops it dead (restitution 0.9).
    - `eval_play.py` - The shared play loop for the checkpoint diagnostics
      in `bin/` (env vs one named opponent, planner or prior, the far side
      driven by the same checkpoint when "external").
    - `run_names.py` - The `<major>.<minor>-<description>-<stage>` scheme,
      enforced by both trainers (see "Run names").
    - `perception.py` - The sim's tracker model: 6-frame slope estimator,
      back-projection noise, the IR blind spot and puck dropouts as a coast
      of up to 150 ms and then a frozen zero-velocity report.
    - `web/` - Browser-based visualization UI
  - `bin/` - Training, evaluation, diagnostics, the table
    - `train.py` - SAC curriculum training (legacy)
    - `train_tdmpc2.py` - TD-MPC2 pretrain stages (`--curriculum-stage`),
      vectorized envs; writes `runs/<run>/run.json`
    - `train_tdmpc2_fast.py` - Optimized TD-MPC2 (batched MPPI, all speedups, self-play)
    - `train_selfplay.py` - Self-play with TD-MPC2: TF32, 256 planner
      samples, CUDA graphs on both planners, the opponent mix, optional
      CushionBot demonstrations (`--demo-envs N --demo-until S`) with a
      behaviour-cloning term (`--bc-coef`), per-opponent W/L/D and the
      shaper's counters every 10k steps. `--run-name` is required and
      must follow the run-name scheme.
    - `run_full_pipeline.sh <major>.<minor>-<description>` - Pretrain
      (4 stages, 750k steps) then self-play (3M)
    - `hold_eval.py` - What a checkpoint does with a puck it has stopped:
      hold time, shot after the hold and its speed, possessions ended by
      the sim's relaunch. The number behind every 2.x/3.x decision.
    - `income_breakdown.py` - Reward per term (`stats["pay_*"]`) for the
      bot, the prior alone and the planner on the same table. Run it
      BEFORE changing a reward.
    - `jitter_eval.py` - Planner target change per tick with the puck at a
      standstill / away / in play (the smoothness-tax number).
    - `run_stats.py` - Trend of the shaper's counters over a
      `train_selfplay` log (first / middle / last windows).
    - `bench_planner.py`, `profile_selfplay.py` - Planner cost per config;
      where a self-play iteration spends its time.
    - `profile_loop.py` - Training loop profiler (per-component timing)
    - `eval_heuristics.py` - Tournament harness for the heuristic bots:
      each vs the scripted opponents, realistic sensing + DR, shared fixtures
    - `eval_policy.py` - A trained policy on the SAME terms (90 s games, same
      seed, same opponents) so its rows compare line-for-line with the bots'
    - `follow_test_rescore.py` - Re-judge a tracking-test CSV with the
      current scoring
    - `run_policy.py` - Drives the table from a policy: heuristic bots or a
      TD-MPC2 checkpoint (`--policy tdmpc2:<run>|latest`, via
      `airhockey/deploy.py`). Dry-run by default; `--live` moves the robot.
      DEFAULTS REPRODUCE TRAINING (planner iterations, the run's horizon,
      shot requests drawn per possession, the sim body's caps, 50 Hz, the
      policy paused 0.5 s after the puck is lost) and it prints a sim/real
      alignment block before the first live command, marking every
      DEVIATION (`--gentle`, `--plan 0`, `--accel-floor`, `--cmd-hz`...)
      and the two inherent ones (the opponent; no referee). Sheds stale
      camera frames with a warning; trusts the camera's own-mallet fix only
      when the loop is <=30 ms behind and the fix is within 300 mm of the
      controller's. Every session logs to `logs/run_policy/<stamp>.log`
      (all output) and `<stamp>.ticks.csv` (per tick: what the policy saw,
      the encoded obs, what it asked, what was sent, caps, lag, cost). The
      master keeps `logs/cdpr_master.log`; `play.sh` stamps a copy on exit.
    - `play.sh` - Turn it on and it plays: starts `cdpr_master`, the camera
      and `run_policy.py --live` in one command; Ctrl-C brakes and stops all.
      Its own flags: `--policy <spec>`, `--tension <mm>` (master pretension
      at ENABLE; default 0 = slack, the setting the tracking test
      validated), `--dry`; everything else passes to `run_policy.py`.
      Flags only, never environment variables.
  - `tests/` - Test suite. Run it with the SYSTEM python:
    `PYTHONPATH=ai:vision/bin python3 -m pytest ai` (the venv's tensordict
    0.11 fails 5 TD-MPC2 tests; the system 0.13 passes).
    - `test_batch_physics.py` - Vectorized physics correctness tests
    - `test_validation.py` - Reward shaping equivalence and env consistency tests
    - `test_heuristics.py` - Bot prediction maths, the mm<->sim/action round
      trips, workspace and cap containment, and end-to-end play
    - `test_retrain_changes.py`, `test_run2_changes.py` - every reward term
      and env rule of the 2.x/3.x retrain pinned on scripted state
      sequences: the shot predictor, held gate, speed ramp, drive pay, shot
      clock, turnover, stuck relaunch, layout remap, deploy encoder
    - `test_cushion_bot.py`, `test_run_names.py`, `test_deploy.py`,
      `test_run_policy.py`, `test_follow_test.py`, `test_pi_smooth.py` -
      the demonstrator; the naming scheme and `trained_horizon`; encoder vs
      env parity; the runner's shedder, lag gate, caps and watchdog; the
      tracking test in virtual time; the planner smoothness flags
- `shared/` - Geometry shared by every control path. **Canonical.**
  - `cdpr_geometry.h` - Table frame, motor anchors, spool, paddle attachment,
    and the cable-length model (tangency + wrap). Included by `fw/` and
    `sw/bin/cdpr_master.cpp`. Physical facts go here; how a given controller
    drives a motor does not.
  - `cdpr_geometry.py` - Python mirror of the header (Python can't include
    a C header). Every Python consumer imports from here — never hardcode
    a geometry constant elsewhere.
  - `check_geometry.py` - Verifies the mirror matches the header constant
    for constant, AND that the C++ and NumPy cable models agree numerically
- `fw/` - Teensy 4.1 step/dir firmware. **All motion runs here.**
  - `include/cdpr_config.h` - Stepper specifics only (DIR levels, counts/rev,
    limits); geometry comes from `shared/`
  - `include/motion_profile.h` - The trajectory law: ONE velocity profile
    along the direction of travel, capping the MAGNITUDE of velocity and of
    its change. Deliberately free of `Arduino.h` so it can be compiled and
    exercised on the host. Replaced two independent per-axis trapezoids,
    which ran 41% over both caps on a diagonal and bent the path badly when
    the axes were unequal (80 mm off a 500x150 move).
    Also JERK-LIMITED: acceleration slews over `RAMP` ms rather than
    stepping on in one tick. The paddle is pulled 32.7 mm above the surface
    over a 50.4 mm radius, so it tips at about g*r/h ~ 1.5 g, and an
    instantaneous accel step both applies that moment impulsively and
    overshoots an elastic cable by up to 2x. Parameterised as a ramp TIME
    (jerk = aMax/ramp) so move shape survives a change of accel cap. Set at
    runtime: `RAMP <ms>` over serial. Cost at the 3 ms default is +7% on a
    500 mm move and +39% on a 25 mm one -- tune against the cable's measured
    ringing period, not from the bench.
    Also PATH-CONTAINED (2026-09-03): `motionProfileContain` clamps the
    cart's position to the box every tick and drops the outward velocity
    and acceleration on a clamped axis. Before, only the TARGET was clamped;
    a vector profile turning at speed has radius v²/a and swung 105 mm past
    the end rail at 12 m/s / 60 m/s² on the first live run of a learned
    policy. The simulator's profile body runs the same bounded law
    (`motion.advance(..., bounds=)`), so sim and firmware agree.
  - `test/` - Host tests for the pure-math parts. `make -C fw/test` builds
    and runs them; no Teensy involved. The one that matters is the step
    synchronisation check — it drives the real profile through the real
    cable kinematics and asserts no motor is ever owed more than the one
    step a tick can emit.
- `sw/` - Host-side support for the physical robot
  - `bin/cdpr_master.cpp` - Bridge: energizes the ClearPath servos, forwards
    commands to the Teensy over serial, serves TCP. Runs a DRIVE FAULT
    WATCHDOG (2026-09-02): a thread polls every drive's enabled/alert bits
    every 20 ms and, the moment one has shut down (overload, tracking, bus),
    disables all four and STOPs the Teensy, then answers every CMD with
    `ERR fault ...` until the next ENABLE. Before this, one overloaded
    motor stopped while the other three kept pulling. The Teensy has no
    feedback wire from the drives, so this is the only place it can live.
  - `bin/` - Standalone diagnostics: `test_motor`, `scan_motors`, `activate`,
    `retract_test`, `calibrate` (passive encoder capture)
  - `lib/clearpath.{h,cpp}` - Minimal ClearPath connect/enable/disable
  - `third_party/sFoundation/` - Teknic sFoundation SDK (patched for Linux, .gitignored)
  - NOTE: the host-side CDPR *motion* controller (`lib/cdpr.*`, `cdpr_server`,
    `cdpr_test`) was removed 2026-08-01. Motion is step/dir via the Teensy;
    that code duplicated it and had drifted out of sync.
- `vision/` - Camera calibration and tracking (FLIR Blackfly S via Spinnaker)
  - `bin/` - `camera.py` (frame stream + back-projection helpers),
    `capture_intrinsics.py` (continuous, self-selecting ChArUco capture),
    `calibrate_intrinsics.py`, `check_intrinsics.py` (coverage +
    distortion-extrapolation audit), `calibrate_extrinsics.py`,
    `measure_anchors.py` (motor anchors from retroreflectors on the spool
    axes — supersedes `measure_motors.py`, which fitted ellipses to the
    spool top faces at an assumed height), `measure_motors.py`,
    `track_mallet.py` (mallet position at z=67mm),
    `calib_report.py`, `table_grid.py`, `gen_targets.py`, `snap.cpp`,
    `blobtrack.cpp` + `puck_stream.py` (200 Hz puck tracking — see below),
    `puck_markers.py` (the puck's four-corner marker square; pure geometry,
    no camera, `--selftest`), `mallet_stream.py`, `record_puck.py` +
    `fit_puck.py` + `plot_puck_fit.py` (puck system identification)
  - `calib/` - Solved intrinsics, extrinsics, marker and motor-anchor JSON
  - `Makefile` - Builds sFoundation library and control programs

### Fast puck tracking (200 Hz)
`bin/blobtrack.cpp` -> `build/blobtrack` runs the camera free-running and
streams BLOB COORDINATES rather than frames: at 200 Hz a 1440x1080 Mono8
frame is 311 MB/s down a pipe and puts Python in the hot loop. Thresholding
and centroiding happen in C++; `bin/puck_stream.py` decides which blob is the
puck, because that is a calibration question and calibration lives in Python.

Measured: 200 Hz at full 1440x1080, zero incomplete frames, worst inter-frame
gap 5.00 ms. Camera caps at 226 Hz.

Exposure/gain matter more than they look. 300 us keeps blur to ~1.5 mm at
5 m/s, but at 0 dB the puck marker peaks at 96 against a threshold of 90 and
was detected in 11% of frames. 12 dB of gain saturates it and the background
only goes 2 -> 5, because the scene is dark by construction. Defaults are
300 us / 12 dB / threshold 90 -> 100% detection.

Known blind spot: the IR ring's own reflection is ~92 x 103 mm at table
centre. The tracker coasts on the last velocity for up to 150 ms across it.

**Marker convention (changed 2026-08-26).** The PUCK carries FOUR
retroreflectors in a square 21.85 mm from its centre; a hand-held mallet
carries ONE dot. It used to be the other way round, and the reason for the
swap is that a player's hand wraps the mallet and hides whatever is stuck to
it, while nothing ever touches the puck. Three consequences: a dropout no
longer loses the puck, the reported centre is the centre rather than wherever
one sticker was placed, and four corners give ORIENTATION, so `record_puck.py`
now logs spin instead of leaving it to be inferred.

The puck is found by SOLVING the square (`puck_markers.py`), never by
averaging the corners: the mean of three sits 21.85/3 = 7.3 mm toward the
missing one, and at 200 Hz that step reads as 1460 mm/s of velocity that never
happened — appearing exactly when a corner drops out, i.e. correlated with
glare and with speed. Any three corners have an exact answer instead, because
the widest pair is the diagonal and a diagonal's midpoint is the centre.
Verified end to end through the real camera pose: position error stays at
~1 mm whether 4, 3 or 2 corners are visible (`puck_stream.py --selftest`).

The ROBOT mallet still carries its three markers (a centre plus two at 26.5 mm
radius) and is still found as a cluster — `MalletTracker(markers=3)`, which is
the default. Its 53 mm span is what keeps it from passing the square test, so
`SQUARE_TOL_MM` cannot be loosened much past 5 mm.

### Hardcoded goalie (DEMO — delete when a policy lands)
`airhockey/demo_goalie.py` + `bin/goalie_demo.py` + `tests/test_demo_goalie.py`.
Straight lines, elastic walls, point paddle. Deliberately isolated: nothing
else imports it, and it is meant to be deleted rather than refactored. The
tracking underneath it is the part that survives.
    python ai/bin/goalie_demo.py --dry-run   # tracks + predicts, commands nothing
    python ai/bin/goalie_demo.py             # moves the robot

SUPERSEDED by `airhockey/heuristics.py`, which is the same idea done against
measured physics and evaluated. Two differences worth porting if this file
outlives the transition. It reflects walls SPECULARLY, but the rail keeps 78.5%
of the normal component and 66% of the tangential, so the outgoing ray is 19%
steeper — a one-bounce prediction lands ~67 mm off, most of a mallet. And it
gates on CLOSING SPEED ("below 150 mm/s is drift, ignore it"), which is right
about rig wear and wrong about air hockey: a puck trickling at 100 mm/s next to
the net is a goal by geometry. Gating on predicted ARRIVAL TIME instead took
goals conceded from 0.10 to 0.04 per game.

## Key Design Decisions
- **Physics are general-purpose**: Support configurable camera delay, motor dynamics models, friction, restitution, etc. Goal is to closely match real-world behavior.
- **Control rate**: ONE constant, `dynamics.ACTION_HZ` (50 since 2026-09-06,
  was 100). The env default `action_dt`, curriculum episode lengths
  (written in SECONDS in `rewards.CURRICULUM`), both trainers, eval and
  `run_policy --cmd-hz` derive from it. 50 because the planner must answer
  inside a tick: `ai/bin/bench_planner.py` measured 6 MPPI iterations at
  12.6 ms eager / 6.3 ms under CUDA graphs on the 4090 (cost is launch-bound,
  samples are free); `PLAN_ITERATIONS = 6` for training and the table.
  Sensing latency is unaffected — the camera model ticks on its own 200 Hz.
  `PLAN_EVAL_MEAN = True`: eval and deploy execute the MPPI elite MEAN,
  not a sampled elite (local TD-MPC2 flag `plan_eval_mean`). Stock
  TD-MPC2 samples an elite even in eval mode; on a flat value landscape
  that is a random target every tick, which is what the 2026-09-05 table
  runs showed and the sim reproduces on a static scene. Halves the target
  jumps and doubled goals vs the goalie in sim.
  **Planning horizon** is per lineage: 5 steps (100 ms) for 1.x and 2.x,
  **8** (160 ms) for 3.x -- the change that let a strike from a standstill
  into the plan (2026-09-07). `policy_loader.trained_horizon` reads it from
  `runs/<run>/run.json` (both trainers write it), so eval and the table
  plan at what the run trained at. Until 2026-09-19 the loader listed only
  3.0-3.3 by name, so the `hold_eval` rows for 3.4-3.11 in `ai/RETRAIN.md`
  were measured at horizon 5; 3.11's row is re-measured there at 8.
- **Observation space**: Puck (pos + vel), own paddle (pos + vel), opponent paddle (pos + vel) — all in 2D — then a side flag, two cap ratios (constants since the band was pinned), the PREVIOUS ACTION (2026-09-03: needed for the smoothness term to be learnable from a frame), the SHOT TYPE REQUESTED (2026-09-06: one-hot bank-left / bank-right / straight, all zero = no preference; x = 0 rail is LEFT facing the far goal), and TIME ON SIDE (seconds since the puck last crossed the centre line, clipped at 5 s and divided by it). 22 dims; the previous action is three wide (x, y, accel fraction). Older 15-, 17- and 20-wide checkpoints load with their columns moved into place and zero weight on the new inputs (`policy_loader.OBS_LAYOUTS`, `load_checkpoint`). Camera delay is applied to observations to simulate real sensing latency.
- **Reward and env rules, current** (the self-play stage of
  `rewards.CURRICULUM` as of 3.11, 2026-09-14; the four pretrain stages
  keep their 2026-09-06 terms). `ai/RETRAIN.md` is the run-by-run record
  of how each rule was arrived at, 22 runs; this is where it stands:
  - Accel pinned at **40 m/s²** (`AGENT_DR_ACCEL_M_S2`; the drives follow
    it CLOSE in the tracking test), speed 12 m/s. The cap features stay in
    the observation as constants. Discount 0.995.
  - An ON-TARGET shot (`rewards.predict_shot` traces the puck through the
    measured lossy-wall model; once per possession) pays 30 + 1 per m/s,
    scaled by shot speed -- nothing under 1.5 m/s, full from 3 -- and a
    matching shot type pays 10. A shot or goal pays IN FULL only once the
    puck has been under control this possession (`control_gate`: under
    0.5 m/s within 0.2 m of the paddle for 0.3 s), **5%** otherwise
    (`patience_floor`). Goals +100 / -50.
  - The paddle's DRIVE at a held puck pays 0.25 x speed² per step from
    anywhere behind or beside it (0.15 m off the puck-to-goal line, 0.40 m
    back), 8 per possession: the strike's motion. Nothing else about the
    setup is prescribed -- the cushion, trap, hold and wind-up incomes
    still exist in `BatchRewardShaper` and are OFF.
  - The env is the referee: a puck on the robot's side for more than
    **3 s** (`batch_env.SHOT_CLOCK_S`), or dead there (1.2 s unattended /
    5 s attended), is TURNED OVER -- relaunched toward the opponent at
    **-50** (`STUCK_TURNOVER_PENALTY`). A per-step tax was tried first and
    was paid rather than avoided.
  - Hygiene: the accel fraction taxed 0.04 per step, smoothness 0.5 per
    unit of action change, home pull 0.05 while the puck is away, defense
    0.05 (was 1.0: it paid ten times the goals for every style and
    switched off whenever the puck was held -- why runs 1-7 never stopped
    it).
  - Opponent mix per episode: 60% a copy of itself on the robot's body,
    20% `sniper` (a scripted striker on a FREE body, 5-8 m/s shots), 20%
    `weak_goalie`; a shot type drawn per possession; 20% of episodes get
    sensing fuzz (the opponent's mallet hidden for 0.3-1.5 s spells, shown
    as the deploy encoder's fallback; 50-150 ms puck dropouts through the
    tracker's coast). `rewards.curriculum_env_kwargs` hands these to the
    env.
  - Result in sim (3.11): stops the puck in most possessions, holds
    ~0.8 s, shoots after 78-96% of its holds at 2.3-2.7 m/s; games against
    itself are mostly draws. NOT yet run on the table.
- **Action space**: Target (x, y) position for the paddle, plus (run 2,
  2026-09-06, `action_mode="profile_a"`, 3 dims) the ACCEL CAP for this
  command as a fraction of the machine's (`BatchAirHockeyEnv.accel_fraction`,
  quadratic: slot -1 -> 5%, 0 -> 29%, +1 -> 100%); speed stays at the
  clamp. Under the accel tax the policies settle at a mean fraction of
  ~0.35, i.e. ~13 m/s², and ask for ~3 m/s² on most idle ticks. The
  reward taxes the fraction (`accel_cost_weight`), so a high cap is spent
  on strikes and saves, not on wandering -- heat is torque and torque is
  accel, and run 1 tripped a drive after 20 s of continuous traversal.
  On the table the accel rides on `CMD x y speed accel`; the master
  forwards it to the Teensy as ACCEL only when it changes. Older 2-dim
  checkpoints still load and play (`policy_loader.checkpoint_shapes`).
  The motor dynamics model converts the target to actual paddle movement.
- **Web UI**: Real-time visualization over WebSocket for debugging. Binds to 0.0.0.0 for access over Tailscale. Not used during training. Defaults to replay mode showing most recent recording. Has instant/realistic physics toggle for manual play.
  - **Camera view** (`vision_service.py`) identifies three things and labels
    each in the overlay: the ROBOT paddle (3-marker cluster), the PUCK (its
    four-corner square, drawn at the true 40.7 mm radius with a spoke to each
    claimed corner) and the PLAYER's mallet (a lone blob). All three come out
    of ONE pass of thresholding and connected components — `track_mallet.
    locate()` takes a `cands=` argument so the frame is only labelled once.
  - In **control** mode the canvas draws the camera puck and player mallet
    (`cam_puck_*` / `cam_player_*` in the frame message, sim coordinates via
    `dynamics.table_mm_to_sim`). Deliberately never in sim mode: two pucks on
    one canvas with no way to tell which one the game believes in.
  - The camera is never started automatically — one process at a time can
    hold the Spinnaker device, so the UI holding it would break
    `record_puck.py`, `blobtrack`, and every other vision tool.
- **Recording**: Save game trajectories at intervals during training for later visual replay. Columnar JSON format for ~78% size reduction. Includes per-frame reward and cumulative reward.

## Run names
`<major>.<minor>-<description>-<stage>` (e.g. `3.3-turnover-selfplay`):
major = a new lineage (action space, observation layout, horizon, model),
minor = a recipe change resumed within it, stage = curriculum stage or
`selfplay`, `-<step>k` for a pinned snapshot. `ai/RUNS.md` is the registry
with each run's parent; `airhockey/run_names.py` enforces it in both
trainers; `python -m airhockey.run_names <major> <description> <stage>`
prints the next free name. Pre-scheme names (`runN_selfplay`) are symlinks.
Every run directory carries a `run.json` (horizon, action rate, model size,
parent) that `policy_loader.trained_horizon` reads; a pinned snapshot is a
directory whose `agent.pt` is a symlink to one of the run's
`agent_step_*.pt` files. `runs/` is not in git; the machine at the table is
the only copy of the checkpoints.

## Commands

All commands run from the REPO ROOT — do not `cd`. Keeping one working
directory means unrelated commands can be pasted back to back.
```bash
# Install
pip install -e "./ai[dev]"

# Run visualization server
PYTHONPATH=ai python -m airhockey.server

# Run tests
pytest ai

# Run full training pipeline (pretrain + self-play)
bash ai/bin/run_full_pipeline.sh 4.0-<description>   # runs named <major>.<minor>-<description>-<stage>

# Run SAC curriculum training
python ai/bin/train.py --curriculum

# One pretrain stage of the TD-MPC2 curriculum
python ai/bin/train_tdmpc2.py --curriculum-stage contact --steps 150000 --run-name 4.0-<description>-contact

# Self-play, resumed within the current lineage (3.x = horizon 8). Logs go to
# logs/<run>.log by convention; the trainer prints per-opponent W/L/D and the
# shaper's counters every 10k steps. ~1 h per 1M steps on the 4090.
python -m airhockey.run_names 3 <description> selfplay          # the next free name
python ai/bin/train_selfplay.py --resume runs/3.11-shot-ramp-selfplay/agent.pt --steps 1000000 \
    --n-envs 32 --model-size 5 --horizon 8 --run-name 3.12-<description>-selfplay \
    --record-freq 50000 --opponent-update-freq 50000 > logs/3.12-<description>-selfplay.log 2>&1

# Checkpoint diagnostics in sim (~10 min each at the defaults; they print the
# planner settings they used, horizon included)
python ai/bin/hold_eval.py 3.11-shot-ramp-selfplay                 # stop / hold / shoot / referee, per opponent
python ai/bin/income_breakdown.py 3.11-shot-ramp-selfplay          # reward per term: bot vs prior vs planner
python ai/bin/jitter_eval.py 3.5-shot-clock-turnover-selfplay 3.11-shot-ramp-selfplay
python ai/bin/run_stats.py logs/3.11-shot-ramp-selfplay.log        # a run's counters, first / middle / last

# Fast trainer (batched MPPI, auto-curriculum); older entry point
python ai/bin/train_tdmpc2_fast.py --curriculum --steps 5000000

# Profile training loop components / the planner
python ai/bin/profile_loop.py
python ai/bin/profile_selfplay.py --run 3.11-shot-ramp-selfplay --n-envs 32
python ai/bin/bench_planner.py --run 3.11-shot-ramp-selfplay --compile

# Heuristic-bot tournament (the non-ML baseline a policy has to beat)
python ai/bin/eval_heuristics.py
python ai/bin/eval_heuristics.py --bots goalie,striker --opponents random
python ai/bin/eval_policy.py 3.11-shot-ramp-selfplay --iterations 6   # a checkpoint on the same terms
```

## Hardware
- **Motors**: Teknic ClearPath-SC, NEMA 23 integrated servos — **two different
  models**, confirmed on hardware 2026-08-03 via `sw/build/check_limits`:
  - nodes 0 and 2: `CPM-SCSK-2331P-ELNA` — 310 oz-in (2.19 N·m) peak,
    **4000 rpm**, encoder 0.057° (~6400 counts/rev)
  - nodes 1 and 3: `CPM-SCSK-2331S-RLNA` — 620 oz-in (4.38 N·m) peak,
    **2580 rpm**, encoder 0.450° (800 counts/rev)

  On a CDPR every cable moves together, so the system takes the WORST of each:
  **2580 rpm and 2.19 N·m**. Any sizing calculation that assumes 4000 rpm is
  55% optimistic. The encoder difference is real and per-node — the `ENC` path
  reads `Info.PositioningResolution` per node for exactly this reason.

  NOT verified: that the step/dir INPUT resolution is 800 counts/rev on all
  four. `fw/include/cdpr_config.h` assumes it is. That is a ClearView setting
  independent of encoder resolution, and if the two model types differ there,
  the Teensy drives them at different scales — which would look like cables
  fighting. Worth confirming before blaming the kinematics.
- **Shaft**: Ø9.5 mm (3/8"), 3 mm keyway, key 3×3×10 mm not supplied
  (McMaster 96717A086). Teknic's manual explicitly recommends circumferential
  clamping over set screws.
- **Communication**: SC4-Hub (USB) -> sFoundation C++ API -> motors via proprietary serial
- **Power**: 24-75V DC supply

## Commands (hardware)
```bash
# Build everything
make -C sw                       # sFoundation SDK first time, then binaries
make -C vision                   # snap (Spinnaker capture)
pio run -d fw                    # Teensy firmware
pio run -d fw -t upload          # flash it
make -C fw/test                  # host tests for the motion profile

# Play with the trained policy (from the repo root)
# Defaults reproduce training; the runner prints a sim/real alignment block
# and marks every DEVIATION. Flags only, no environment variables.
bash ai/bin/play.sh --policy tdmpc2:3.11-shot-ramp-selfplay --gentle   # FIRST run of a new checkpoint
bash ai/bin/play.sh --policy tdmpc2:3.11-shot-ramp-selfplay            # as trained
bash ai/bin/play.sh --policy tdmpc2:latest --dry                          # camera + policy, commands nothing
bash ai/bin/play.sh --policy tdmpc2:latest --tension 1.5                  # master pretension, mm (default 0)
python ai/bin/run_policy.py --policy tdmpc2:latest --opponent   # dry-run, no master

# Puck tracking / goalie demo
vision/build/blobtrack --probe                  # report achievable frame rate
python vision/bin/puck_stream.py                # live puck position at 200 Hz
python vision/bin/puck_stream.py --raw          # every surviving blob
python ai/bin/goalie_demo.py --dry-run          # goalie, commands nothing

# Motors. These are MUTUALLY EXCLUSIVE — pick one. Every one of them calls
# PortsOpen on the same SC-Hub USB port, and a second process trying to open
# it just errors. Running activate "alongside" cdpr_master does not work.
#
#   activate     manual: energize by hand, no TCP. ENTER toggles all four,
#                q is the emergency stop, and it de-energizes on exit, so it
#                must STAY RUNNING for the motors to stay on.
#   cdpr_master  everything driven over TCP (web UI, ai/bin/goalie_demo.py).
#                It opens the port itself and energizes on ENABLE, so it does
#                NOT want activate running. Ctrl-C is the stop here, and a
#                second Ctrl-C forces the exit.
#   test_motor   standalone single-motor check.
#
# Nothing moves until commanded, whichever you pick.
sw/build/activate                # ENTER toggles all four
sw/build/cdpr_master             # TCP 8421 -> Teensy bridge; use ALONE
sw/build/test_motor              # or: sw/build/test_motor /dev/ttyACM0

# Camera / calibration
vision/build/snap shots 8 1 --exposure 1000 --gain 0
python vision/bin/capture_intrinsics.py
python vision/bin/calibrate_intrinsics.py --images 'vision/calib_shots/*.png'
python vision/bin/check_intrinsics.py
python vision/bin/calibrate_extrinsics.py vision/extr_shots/*.png
python vision/bin/measure_motors.py --height 36 --seeds "..." vision/ambient/*.png
python vision/bin/calib_report.py vision/extr_shots/shot_000.png --ambient vision/ambient/shot_002.png
python vision/bin/track_mallet.py            # mallet position + CAL line
python vision/bin/track_mallet.py --watch

# Geometry drift guard (C++ header vs both Python mirrors)
python shared/check_geometry.py
```

## World Model Architecture (TD-MPC2)
STOCK upstream TD-MPC2 (MLP dynamics), from the checkout at
`~/dev/p-airhockey/tdmpc2` — NOT the GRU variant this section used to
describe. That fork (GRUCell dynamics, prioritized replay, tuple-returning
act) lived at `/home/rbhagat/projects/tdmpc2`, which does not exist on this
machine; discovered 2026-08-29 when training crashed on its missing pieces.

What the local checkout DOES carry, as LOCAL commits (`git log` in that
repo; it is NOT a github fork with a remote -- do not `git pull --rebase`
them away):
- **batched MPPI planning**: `act()` accepts `(N, obs_dim)` observations
  with a per-env bool `t0` mask and plans every env in one call
  (`_plan_batch`), warm-starting from a persistent `[N, horizon,
  action_dim]` buffer allocated OUTSIDE the planner so
  `torch.compile(mode="reduce-overhead")` (CUDA graphs) can wrap it. Both
  trainers and `deploy.py` use it. `train_tdmpc2_fast.py`'s
  prioritized-replay path degrades to uniform sampling with a warning
  because stock `Buffer` has no `set_beta`.
- `plan_eval_mean` (execute the elite mean in eval mode), `pi_smooth_coef`
  (temporal smoothness regulariser on the prior, in pre-squash space),
  `plan_smooth_coef` + `prev_action_start` (an MPPI action-change cost,
  parked at 0).
- `bc_coef`: behaviour cloning of the prior toward demonstrated actions,
  with a per-step `demo` flag carried through the buffer
  (`sample_with_demo`). Used with `train_selfplay --demo-envs`.

Costs on the 4090 (`ai/bin/bench_planner.py`): a 6-iteration single-env
plan is ~12.6 ms eager / ~6.3 ms under CUDA graphs at horizon 5, ~26%
more at 8 -- why the control rate is 50 Hz and the table plans at
training's settings. At 32 envs the planner is compute-bound: TF32 + 256
samples + CUDA graphs gave ~3x (`ai/bin/profile_selfplay.py`); self-play
collects ~200-300 env-steps/s, ~1 h per 1M steps.

## Tech Stack
- Python, NumPy for physics/env
- Gymnasium for RL environment API
- FastAPI + WebSocket for visualization server
- Vanilla JS + Canvas for web UI
- Stable-Baselines3 for RL training (SAC)
- TD-MPC2 for model-based planning (checkout at ~/dev/p-airhockey/tdmpc2, local batched-MPPI commit)

## Training Learnings
- **Algorithm**: SAC works much better than PPO for this continuous control task.
- **Curriculum learning**: Train on proximity-only reward first (`exp(-3*dist)`), then add full rewards. This bootstraps the agent to move toward the puck before learning what to do with it.
- **Reward design (SAC era, 2026-03; the TD-MPC2 curriculum's terms are in
  `rewards.CURRICULUM` and described under "Key Design Decisions")**:
  - Exponential proximity: `0.1 * exp(-3*dist)` — dense signal that pulls the paddle toward the puck.
  - Goal scored: +100.
  - Goal conceded: -5 (kept low intentionally — a large penalty discourages hitting the puck at all).
  - Puck progress: one-way reward, only credits forward movement toward the opponent's goal.
  - Contact reward: +5 on paddle-puck contact.
- **Action space normalization**: Actions must be normalized to [-1, 1]. This is critical for learning; using raw position coordinates causes action saturation and kills gradients.
- **VecNormalize**: Hurts SAC performance. Do not use it.
- **Symmetric self-play (2026-09-02)**: `BatchAirHockeyEnv(opponent_body="robot")`
  makes the far side an exact copy of the machine — same profile law, same
  caps and DR draw, the workspace box mirrored, side flag ROBOT on both
  views — and `opponent_obs()` builds its view natively (own paddle fresh,
  rival through the camera). `train_selfplay.py` uses it by default
  (`--human-opponent` restores the human model). Before this the sparring
  partner was the human model, which the learner had never been in the
  body of, and it played far worse than the robot. The robot's accel DR
  band is pinned (40 m/s² since the retrain; was 10–60, then 60); the cap
  features stay in the observation as constants so the band can be
  reopened without reshaping the network. `eval_policy.py A --vs B` plays
  two checkpoints on equal bodies; `--human-body` reproduces the
  pre-change ladder.
- **Smoothness (2026-09-03, raised 2026-09-14)**: the self-play stage taxes
  step-to-step action change everywhere (`smooth_weight` 0.5/unit since
  3.6, 0.2 before; a full corner-to-corner flip costs 1.4, a strike's
  target jump ~0.5 against the 30 it earns; 0.02 was invisible to the
  two-hot value heads) plus a tiny idle pull to the goal's centre line
  while the puck is away. The previous action is in the observation so
  the term is learnable from a frame. Measured with `ai/bin/jitter_eval.py`:
  0.2 -> 0.5 took the planner's target change with the puck at a
  standstill from 53 to 23 mm per tick (p90 212 -> 87); the planner still
  moves 2-3x more than the prior alone, and on the table that jitter is
  what tripped a drive's RMS overload on 2026-09-07 together with the
  phantom-puck bug (now the 0.5 s puck timeout).
- **Retrain lessons (2.x-3.x, 2026-09-06 to 09-14; every run's numbers in
  `ai/RETRAIN.md`)**:
  - A per-step income that the wanted behaviour switches off will prevent
    it: the defense term paid ten times the goals and holding the puck
    ended it. Look at `income_breakdown.py` before touching a reward.
  - "Controlled" must be defined strictly (held at rest for a time); a
    gate that accepted "slowed within reach" or "waited N seconds" was
    slapped through.
  - A tax on stalling gets paid whenever stalling is safe (a held puck
    concedes nothing). The env taking the puck at a goal's cost is what
    worked; and the table has no referee, so a policy that stalls in sim
    stalls for ever on the table (3.3, 2026-09-07).
  - Demonstrations in the replay buffer did not transfer through value
    learning alone; cloning transferred them into the prior, and the
    planner then overrode the prior wherever the reward preferred
    otherwise. Compare prior vs planner.
  - A strike from a standstill needed three things at once: its payoff
    inside the planning horizon (8 steps), the drive's motion paid densely
    (0.25 x v²), and an on-target ramp that covers the speed this body
    produces from a hold (2-2.5 m/s at its habitual ~13 m/s²).
  - Judge a checkpoint with `hold_eval.py`, not the training counters,
    and check the 500k and the final checkpoint separately: the same run
    has gained and lost a skill within 1M steps (2.12).
  - Table side: pause the policy 0.5 s after the puck is lost (the sim's
    longest unseen spell), shed stale frames, plan at the run's horizon,
    and remember the sim has no thermal limit -- the drives do.
- **Training throughput**: SAC's bottleneck is gradient updates, not environment stepping. Using `train_freq=32` with `gradient_steps=4` gives roughly 3x speedup over the default.
- **Episode init**: Puck should start heading toward the agent so it encounters the puck quickly and gets reward signal faster.
- **Network size**: 128x128 MLP is sufficient for basic play. Will likely need larger networks for strategic/competitive play.
