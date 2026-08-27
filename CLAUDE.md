# Air Hockey RL Project

## Overview
Robotic air hockey table that uses reinforcement learning trained in simulation, then transferred to physical hardware. The mechanical design is likely a cable-driven parallel robot (CDPR) with 4 motors, though a CoreXY gantry is also under consideration.

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
    - `web/` - Browser-based visualization UI
  - `bin/` - Training scripts
    - `train.py` - SAC curriculum training
    - `train_tdmpc2.py` - TD-MPC2 model-based training (vectorized envs)
    - `train_tdmpc2_fast.py` - Optimized TD-MPC2 (batched MPPI, all speedups, self-play)
    - `train_selfplay.py` - Self-play training with TD-MPC2
    - `run_full_pipeline.sh` - Pretrain + self-play pipeline
    - `profile_loop.py` - Training loop profiler (per-component timing)
  - `tests/` - Test suite
    - `test_batch_physics.py` - Vectorized physics correctness tests
    - `test_validation.py` - Reward shaping equivalence and env consistency tests
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
  - `test/` - Host tests for the pure-math parts. `make -C fw/test` builds
    and runs them; no Teensy involved. The one that matters is the step
    synchronisation check — it drives the real profile through the real
    cable kinematics and asserts no motor is ever owed more than the one
    step a tick can emit.
- `sw/` - Host-side support for the physical robot
  - `bin/cdpr_master.cpp` - Bridge: energizes the ClearPath servos, forwards
    commands to the Teensy over serial, serves TCP
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

## Key Design Decisions
- **Physics are general-purpose**: Support configurable camera delay, motor dynamics models, friction, restitution, etc. Goal is to closely match real-world behavior.
- **Observation space**: Puck (pos + vel), own paddle (pos + vel), opponent paddle (pos + vel) — all in 2D. Camera delay is applied to observations to simulate real sensing latency.
- **Action space**: Target (x, y) position for the paddle. Motor dynamics model converts this to actual paddle movement.
- **Web UI**: Real-time visualization over WebSocket for debugging. Binds to 0.0.0.0 for access over Tailscale. Not used during training. Defaults to replay mode showing most recent recording. Has instant/realistic physics toggle for manual play.
- **Recording**: Save game trajectories at intervals during training for later visual replay. Columnar JSON format for ~78% size reduction. Includes per-frame reward and cumulative reward.

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
bash ai/bin/run_full_pipeline.sh

# Run SAC curriculum training
python ai/bin/train.py --curriculum

# Run TD-MPC2 training (original)
python ai/bin/train_tdmpc2.py --steps 500000

# Run TD-MPC2 fast training (batched MPPI, all speedups)
python ai/bin/train_tdmpc2_fast.py --steps 2000000

# Fast training with full MPPI quality (no speed reduction)
python ai/bin/train_tdmpc2_fast.py --no-fast --steps 2000000

# Auto-curriculum (stages 1-4, auto-advancing on plateau)
python ai/bin/train_tdmpc2_fast.py --curriculum --steps 5000000

# Run a specific curriculum stage only
python ai/bin/train_tdmpc2_fast.py --stage 4 --steps 1000000

# Fast training self-play (resumes from pretrained agent)
python ai/bin/train_tdmpc2_fast.py --resume runs/tdmpc2_pretrain/agent.pt --steps 5000000

# Run self-play (original)
python ai/bin/train_selfplay.py --resume runs/tdmpc2_pretrain/agent.pt

# Profile training loop components
python ai/bin/profile_loop.py
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

## World Model Architecture (TD-MPC2 + GRU)
The world model uses a GRU-based recurrent dynamics model (similar to DreamerV3 RSSM but without stochastic component):
- **Encoder**: MLP (obs_dim → latent_dim=512 via enc_dim=256, 2 layers, SimNorm output)
- **Dynamics**: `nn.GRUCell(latent_dim + action_dim, latent_dim)` + SimNorm. Returns `(z_next, h_next)` — z is the SimNorm of the GRU hidden state. Hidden state h persists per-env across timesteps.
- **Reward**: MLP(latent_dim + action_dim → num_bins=101, two-hot encoding)
- **Policy prior**: MLP(latent_dim → 2×action_dim), Gaussian with squashed output
- **Q-functions**: 5× ensemble MLP(latent_dim + action_dim → num_bins=101)
- **act()** returns `(actions, h_next)` tuple — all callers must unpack both values
- **Training** uses h=None per trajectory slice (deliberate train/inference mismatch, same as DreamerV3)
- External repo at `/home/rbhagat/projects/tdmpc2` — `tdmpc2.py` and `common/world_model.py` are modified

## Tech Stack
- Python, NumPy for physics/env
- Gymnasium for RL environment API
- FastAPI + WebSocket for visualization server
- Vanilla JS + Canvas for web UI
- Stable-Baselines3 for RL training (SAC)
- TD-MPC2 for model-based planning (external repo at /home/rbhagat/projects/tdmpc2)

## Training Learnings
- **Algorithm**: SAC works much better than PPO for this continuous control task.
- **Curriculum learning**: Train on proximity-only reward first (`exp(-3*dist)`), then add full rewards. This bootstraps the agent to move toward the puck before learning what to do with it.
- **Reward design**:
  - Exponential proximity: `0.1 * exp(-3*dist)` — dense signal that pulls the paddle toward the puck.
  - Goal scored: +100.
  - Goal conceded: -5 (kept low intentionally — a large penalty discourages hitting the puck at all).
  - Puck progress: one-way reward, only credits forward movement toward the opponent's goal.
  - Contact reward: +5 on paddle-puck contact.
- **Action space normalization**: Actions must be normalized to [-1, 1]. This is critical for learning; using raw position coordinates causes action saturation and kills gradients.
- **VecNormalize**: Hurts SAC performance. Do not use it.
- **Training throughput**: SAC's bottleneck is gradient updates, not environment stepping. Using `train_freq=32` with `gradient_steps=4` gives roughly 3x speedup over the default.
- **Episode init**: Puck should start heading toward the agent so it encounters the puck quickly and gets reward signal faster.
- **Network size**: 128x128 MLP is sufficient for basic play. Will likely need larger networks for strategic/competitive play.
