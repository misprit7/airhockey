---
name: project_worklog
description: Chronological work log of the air hockey RL project development
type: project
---

## Work Log

### Session 1 (2026-03-26)

**Discussion: Mechanical Design**
- Discussed CDPR (cable-driven parallel robot) vs CoreXY gantry vs robot arm
- User leaning toward CDPR with 4 motors + wires, but leaving undecided
- Researched existing air hockey robots — most use H-bot/gantry (JJRobots EVO, Nuvation, Microsoft)
- No one has done a CDPR air hockey robot before
- Looked at robot arm options under $20k (UFACTORY xArm 6 ~$5.3k best value, but arms are slow for air hockey)
- User has Teknic ClearPath motors already

**Discussion: Simulation Approach**
- Decided on custom Python/NumPy environment over MuJoCo/PyBullet (2D problem, simplicity wins)
- Plan: system identification → sim training → sim-to-real transfer
- Gymnasium API for RL compatibility, Stable-Baselines3 for training

**Project Setup**
- Created git repo, Python package structure
- Built core physics engine (puck, paddles, walls, collisions, goals, friction)
- Built pluggable motor dynamics models (IdealDynamics, DelayedDynamics, LearnedDynamics placeholder)
- Built Gymnasium environment with camera delay simulation, configurable opponent policies
- Built FastAPI WebSocket server with real-time canvas visualization
- Built recording/replay system for game trajectories
- Made web UI responsive for mobile (collapsible sidebar, touch controls)

**Training Iterations (SAC)**
- PPO failed twice (action space saturation, VecNormalize washing out signal)
- SAC breakthrough: auto-entropy handles exploration naturally
- Proximity-only reward (exp(-3*dist)) validated basic learning
- Full rewards with curriculum: proximity → contact → scoring
- Agent scoring 3-1, 5-1, 6-0 by end of curriculum training

**Key Technical Decisions**
- Action space [-1, 1] normalized — critical for learning
- SAC >> PPO for continuous control
- Goal conceded penalty kept low (-5) to not discourage hitting puck
- Puck starts heading toward agent for faster training signal
- Recording files use columnar JSON format (78% smaller)

### Session 2 (2026-03-27)

**TD-MPC2 Integration (model-based planning)**
- Integrated TD-MPC2 for MPPI planning (512 trajectories × 5 step horizon)
- Dramatically better play than SAC — learned strong play by 350k steps
- No curriculum needed (world model handles exploration internally)
- 5M param model, ~50-70 FPS (planning is the bottleneck, not sim)
- TD-MPC2 repo at /home/xander/dev/p-airhockey/tdmpc2

**SAC Curriculum Training (completed)**
- 3-stage curriculum with 512x512 network on GPU worked well
- LR=0.005 sweet spot for 512x512 (3e-4 too slow, 1e-2 unstable)
- Randomized paddle + puck start positions critical for learning
- Eval callback was eating 87% of training time (fixed: 200k interval)
- batch_size=4096, train_freq=128, gradient_steps=8 on 4090 GPU

**Self-Play Training**
- Agent vs frozen past self, opponent updated every 50k steps
- Both agent and opponent use full MPPI planning (same lookahead)
- 32 parallel environments for GPU utilization (~69 FPS)
- Rewards: +100 goal scored, -50 goal conceded (defense priority)
- External opponent policy mode added to env with mirrored observations

**Repo Reorganization**
- Moved all AI code into ai/ subfolder
- sw/ directory created for physical robot control software
- All commands now run from ai/ directory (cd ai && ...)

## Current State (2026-03-28)

**Repo structure:**
- `ai/` — RL training, simulation, web visualization
- `ai/airhockey/` — Python package (physics, env, rewards, dynamics, recorder, server)
- `ai/bin/` — Training scripts (train.py, train_tdmpc2.py, train_selfplay.py, run_full_pipeline.sh)
- `sw/` — Physical robot control (Teknic ClearPath motors via sFoundation SDK)
- TD-MPC2 external repo at ../tdmpc2

**Best models available:**
- `ai/runs/tdmpc2_pretrain/agent.pt` — TD-MPC2 pretrained on idle opponent (500k steps)
- `ai/runs/selfplay_v1/agent.pt` — Self-play trained (partial, ~400k steps of self-play)
- `ai/runs/curriculum_v6_s3_scoring/final_model.zip` — Best SAC model (3-stage curriculum)

**Training infrastructure:**
- Web UI: port 8420, defaults to replay mode with most recent recording
- TensorBoard: port 6006
- GPU: 4090, batch_size=4096 on GPU is free vs 512
- Training speeds: SAC ~7-10k FPS, TD-MPC2 ~50-70 FPS (planning-bound)

**Next steps:**
- Continue self-play training with both agents using full MPPI planning
- Consider adding "follow" opponent as intermediate before self-play
- Physical hardware: sw/ directory has motor test code for Teknic ClearPath-SC motors
- System identification: need to collect real hardware dynamics data
