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
    `measure_motors.py`, `track_mallet.py` (mallet position at z=67mm),
    `calib_report.py`, `table_grid.py`, `gen_targets.py`, `snap.cpp`
  - `calib/` - Solved intrinsics, extrinsics, marker and motor-anchor JSON
  - `Makefile` - Builds sFoundation library and control programs

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
- **Motors**: Teknic ClearPath-SC CPM-SCSK-2331P-ELNA (NEMA 23 integrated servo, 310 oz-in peak torque, 4000 RPM)
- **Communication**: SC4-Hub (USB) -> sFoundation C++ API -> motors via proprietary serial
- **Power**: 24-75V DC supply

## Commands (hardware)
```bash
# Build everything
make -C sw                       # sFoundation SDK first time, then binaries
make -C vision                   # snap (Spinnaker capture)
pio run -d fw                    # Teensy firmware
pio run -d fw -t upload          # flash it

# Motors: activate must STAY RUNNING (it de-energizes on exit, and q is
# the emergency stop). Nothing moves until commanded.
sw/build/activate                # ENTER toggles all four
sw/build/cdpr_master             # TCP 8421 -> Teensy bridge
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
