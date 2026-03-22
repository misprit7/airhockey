# Air Hockey RL Project

## Overview
Robotic air hockey table that uses reinforcement learning trained in simulation, then transferred to physical hardware. The mechanical design is likely a cable-driven parallel robot (CDPR) with 4 motors, though a CoreXY gantry is also under consideration.

## Approach
1. **System identification**: Learn physical dynamics from real hardware (motor response, cable compliance, friction, latency)
2. **Sim training**: Train RL policy in a fast custom simulator with domain randomization
3. **Sim-to-real transfer**: Deploy trained policy on physical robot, optionally fine-tune

## Project Structure
- `airhockey/` - Python package
  - `physics.py` - Core 2D physics engine (puck, paddles, walls, collisions)
  - `dynamics.py` - Pluggable motor dynamics models (ideal, delayed, learned)
  - `env.py` - Gymnasium environment wrapping the physics
  - `recorder.py` - Game recording and replay
  - `server.py` - FastAPI WebSocket server for real-time visualization
  - `web/` - Browser-based visualization UI

## Key Design Decisions
- **Physics are general-purpose**: Support configurable camera delay, motor dynamics models, friction, restitution, etc. Goal is to closely match real-world behavior.
- **Observation space**: Puck (pos + vel), own paddle (pos + vel), opponent paddle (pos + vel) — all in 2D. Camera delay is applied to observations to simulate real sensing latency.
- **Action space**: Target (x, y) position for the paddle. Motor dynamics model converts this to actual paddle movement.
- **Web UI**: Real-time visualization over WebSocket for debugging. Binds to 0.0.0.0 for access over Tailscale. Not used during training.
- **Recording**: Save game trajectories at intervals during training for later visual replay.

## Commands
```bash
# Install
pip install -e ".[dev]"

# Run visualization server
python -m airhockey.server

# Run tests
pytest
```

## Tech Stack
- Python, NumPy for physics/env
- Gymnasium for RL environment API
- FastAPI + WebSocket for visualization server
- Vanilla JS + Canvas for web UI
- Stable-Baselines3 for RL training

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
