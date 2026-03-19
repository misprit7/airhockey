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
- Stable-Baselines3 (future) for RL training
