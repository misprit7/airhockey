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
    - `dynamics.py` - Pluggable motor dynamics models (ideal, delayed, learned)
    - `env.py` - Gymnasium environment wrapping the physics
    - `rewards.py` - Curriculum reward shaping (3 stages)
    - `recorder.py` - Game recording and replay
    - `server.py` - FastAPI WebSocket server for real-time visualization
    - `web/` - Browser-based visualization UI
  - `bin/` - Training scripts
    - `train.py` - SAC curriculum training
    - `train_tdmpc2.py` - TD-MPC2 model-based training
    - `train_selfplay.py` - Self-play training with TD-MPC2
    - `run_full_pipeline.sh` - Pretrain + self-play pipeline
- `sw/` - Physical robot control software
  - `bin/` - Control programs
    - `test_motor.cpp` - Basic motor communication and movement test
  - `third_party/sFoundation/` - Teknic sFoundation SDK (patched for Linux, .gitignored)
  - `Makefile` - Builds sFoundation library and control programs

## Key Design Decisions
- **Physics are general-purpose**: Support configurable camera delay, motor dynamics models, friction, restitution, etc. Goal is to closely match real-world behavior.
- **Observation space**: Puck (pos + vel), own paddle (pos + vel), opponent paddle (pos + vel) — all in 2D. Camera delay is applied to observations to simulate real sensing latency.
- **Action space**: Target (x, y) position for the paddle. Motor dynamics model converts this to actual paddle movement.
- **Web UI**: Real-time visualization over WebSocket for debugging. Binds to 0.0.0.0 for access over Tailscale. Not used during training. Defaults to replay mode showing most recent recording. Has instant/realistic physics toggle for manual play.
- **Recording**: Save game trajectories at intervals during training for later visual replay. Columnar JSON format for ~78% size reduction. Includes per-frame reward and cumulative reward.

## Commands
```bash
# Install
cd ai && pip install -e ".[dev]"

# Run visualization server
cd ai && python -m airhockey.server

# Run tests
cd ai && pytest

# Run full training pipeline (pretrain + self-play)
cd ai && bash bin/run_full_pipeline.sh

# Run SAC curriculum training
cd ai && python bin/train.py --curriculum

# Run TD-MPC2 training
cd ai && python bin/train_tdmpc2.py --steps 500000

# Run self-play
cd ai && python bin/train_selfplay.py --resume runs/tdmpc2_pretrain/agent.pt
```

## Hardware
- **Motors**: Teknic ClearPath-SC CPM-SCSK-2331P-ELNA (NEMA 23 integrated servo, 310 oz-in peak torque, 4000 RPM)
- **Communication**: SC4-Hub (USB) -> sFoundation C++ API -> motors via proprietary serial
- **Power**: 24-75V DC supply

## Commands (sw/)
```bash
# Build (fetches/builds sFoundation SDK first time)
cd sw && make

# Test motor communication
cd sw && bin/test_motor

# Or specify port manually
cd sw && bin/test_motor /dev/ttyACM0
```

## Tech Stack
- Python, NumPy for physics/env
- Gymnasium for RL environment API
- FastAPI + WebSocket for visualization server
- Vanilla JS + Canvas for web UI
- Stable-Baselines3 for RL training (SAC)
- TD-MPC2 for model-based planning (external repo at ../tdmpc2)

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
