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

### Session 3 (2026-03-28)

**Environment Setup**
- Set up uv environment, cloned TD-MPC2 repo to /home/rbhagat/projects/tdmpc2, installed all deps
- Spun up agent team (8 agents) for TD-MPC2 training speedup sprint

**Deep Research into TD-MPC2 Internals**
- MPPI planning: 512 samples, 6 iterations, horizon 5 (default)
- World model: 5M params, data flow: obs CPU→GPU for act(), CPU return, buffer stores on CUDA if memory allows
- Baseline profiling: 32 FPS, act=23ms (50%), update=22.5ms (50%), env=0.14ms

**Built train_tdmpc2_fast.py (all speedups combined)**
- Batched MPPI planning across N parallel envs (one GPU call instead of N sequential)
- Vectorized NumPy physics engine (BatchPhysicsEngine + BatchAirHockeyEnv)
- Fast MPPI defaults: 128 samples, 3 iterations, horizon 3
- Async recording/eval/checkpointing via ThreadPoolExecutor
- Action chunking support (--replan-every K)
- Configurable UTD ratio (--updates-per-step, --utd-ratio)

**Profiling Results (32 envs, fast MPPI)**
- Gradient updates: 124ms (66.5%) — dominant bottleneck, can't be parallelized
- MPPI planning: 61ms (32.8%) — batched across all envs
- Env step: 0.68ms (0.4%), TensorDict loop: 0.57ms (0.3%)
- CPU-GPU data transfers: 0.01ms (0.0%) — NOT a bottleneck
- Effective FPS: ~172 env-steps/sec (fast MPPI), up from 32 FPS baseline

**Key Findings**
- Batched MPPI: planning goes from 736ms (32 sequential calls) to 55-61ms (one batched call) = ~12x speedup on planning
- Fast MPPI (128 samples) matches full MPPI (512 samples) in reward quality at same step count
- With 1:1 UTD ratio: ~2x overall speedup; with reduced UTD: up to 10x+ FPS but worse learning
- GPU: RTX 5080, 91% utilization, 5.7/16.3 GB memory
- torch.compile: limited benefit due to graph breaks (random sampling, data-dependent indices in MPPI)

**Bug Fixes**
- float64 precision issues in batch physics
- Trajectory buffer init obs (used terminal obs instead of reset obs)
- TensorBoard logging frequency
- Curriculum stage guards in BatchRewardShaper

**Validation**
- 10 batch physics tests + 8 validation tests all passing
- Reward shaping equivalence verified between single-env and batch implementations
- Goal rewards, contact detection, puck progress, dynamics delay all tested

### Session 4 (2026-03-29)

**Overnight results: v11 reached self-play**
- v11 curriculum training reached Stage 4 (self-play) at 1.17M steps
- First successful full curriculum run through all stages

**v12 experiments (reverted)**
- Tried 512/6 MPPI + batch_size=8192 → too slow (57 FPS), reverted to fast defaults

**Transformer dynamics (designed, benchmarked, removed)**
- Designed and implemented causal TransformerDynamics for parallel multi-step prediction
- Manual attention blocks, reward head, SimNorm, 3 layers, next_parallel() for batched rollout
- Benchmarked: slower than sequential MLP at our scale (512-dim latent, 2-dim action)
- Transformer overhead dominated for small sequences (horizon 3-5) — not worth it
- All transformer code removed from codebase

**GRU dynamics (implemented, shipped)**
- Replaced MLP dynamics with nn.GRUCell + SimNorm in world model
- model.next(z, a, task, h) returns (z_next, h_next) — z is SimNorm of GRU output
- Hidden state h persists per-env across timesteps via h_all tensor in training loop
- h_all[done] = 0 on episode boundaries, full reset on curriculum stage transitions
- act() accepts h and returns (actions, h_next) — all callers updated across ~15 files
- Training _update() uses h=None per trajectory slice (deliberate, same as DreamerV3 RSSM)
- Seed phase advances h each step so it's warm when planning starts
- Action chunking disabled (incompatible with GRU — would need per-step h advancement)
- Similar architecture to DreamerV3's RSSM but simpler (no stochastic component)

**Compiled rollout optimization**
- Extracted GRU+reward rollout loop into standalone function for torch.compile
- _estimate_value_compiled() compiles the deterministic dynamics loop separately
- pi()+Q() terminal value stays outside compiled region (torch.randn_like causes graph breaks)
- 1.86x speedup on _estimate_value

**Profiling with GRU + 512/6 MPPI**
- act() = 240ms (57%), update() = 180ms (43%), ~87 FPS total

**Reward tuning**
- Forward-only contact reward (no backward hits)
- Goal scored: +160, goal conceded: -20
- Proximity reward removed
- Penalty ramp for defense violations

**Infrastructure changes**
- Synchronous 30s GPU recordings (async CUDA stream approach abandoned — CUDA graph conflicts)
- Step-based TensorBoard metrics (per-episode logging removed for smoother curves)
- 128 tests passing (up from 18 at session start)

**v13 launched**
- First training run with GRU recurrent world model
- Curriculum stages 1-4 with auto-advancement

## Current State (2026-03-29)

**Repo structure:**
- `ai/` — RL training, simulation, web visualization
- `ai/airhockey/` — Python package (physics, env, rewards, dynamics, recorder, server, curriculum)
- `ai/airhockey/batch_physics.py` — Vectorized NumPy physics engine for N envs
- `ai/airhockey/batch_env.py` — Batch environment wrapper (same API as AirHockeyEnv but batched)
- `ai/airhockey/curriculum.py` — Per-stage cosine LR scheduler
- `ai/bin/` — Training scripts
  - `train.py` — SAC curriculum training
  - `train_tdmpc2.py` — TD-MPC2 training (vectorized envs)
  - `train_tdmpc2_fast.py` — Optimized TD-MPC2 (batched MPPI, GRU dynamics, self-play, auto-curriculum)
  - `train_selfplay.py` — Self-play training
  - `run_full_pipeline.sh` — Pretrain + self-play pipeline
  - `profile_loop.py` — Training loop profiler (per-component timing)
- `ai/tests/` — 128 tests (batch physics, reward shaping, env consistency, GRU dynamics)
- `sw/` — Physical robot control (Teknic ClearPath motors via sFoundation SDK)
- TD-MPC2 external repo at /home/rbhagat/projects/tdmpc2

**Modified external files:**
- `tdmpc2/tdmpc2/tdmpc2.py` — GRU hidden state in act()/plan(), compiled rollout, batched MPPI
- `tdmpc2/tdmpc2/common/world_model.py` — GRU dynamics (nn.GRUCell + SimNorm replacing MLP)

**World model architecture:**
- Encoder: MLP (obs_dim → latent_dim=512 via enc_dim=256, 2 layers, SimNorm)
- Dynamics: GRUCell(latent_dim + action_dim, latent_dim) + SimNorm (replaces old MLP)
- Reward: MLP(latent_dim + action_dim → num_bins=101)
- Policy prior: MLP(latent_dim → 2*action_dim, Gaussian with squashed output)
- Q-functions: 5× ensemble MLP(latent_dim + action_dim → num_bins=101)
- Total: ~5M params (model_size=5)

**Training infrastructure:**
- Web UI: port 8420, defaults to replay mode with most recent recording
- TensorBoard: port 6006
- GPU: RTX 5080 (16.3 GB)
- Training speeds: TD-MPC2 fast ~87 FPS (GRU + full MPPI), ~172 FPS (fast MPPI)

**Next steps:**
- Evaluate v13 GRU training quality vs v11 MLP baseline
- Tune GRU hidden state size if needed (currently same as latent_dim=512)
- Continue self-play training once curriculum completes
- Physical hardware: sw/ directory has motor test code for Teknic ClearPath-SC motors
- System identification: need to collect real hardware dynamics data
