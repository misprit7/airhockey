# Arrival and motor-load learning run — 2026-09-20

`4.0-arrival-rms-selfplay` is a six-million-transition simulation experiment.
Start command is preserved in `ai/bin/run_arrival_training.sh`; the active user
service is `airhockey-arrival-rms.service`. Progress is in
`logs/4.0-arrival-rms-selfplay.log` and `runs/4.0-arrival-rms-selfplay/metrics.jsonl`.
Checkpoints save every 100,000 transitions. Prior and planner skill evaluations
run every 500,000 transitions and at completion.

Watchable 30-second MPC self-play games are saved to
`ai/recordings/4.0-arrival-rms-selfplay_step_*.json`, grouped under the run name
in the web UI's Replay tab. Both sides use the SAME checkpoint with independent
MPC plans, equal planning budgets and robot bodies. They use the arrival decoder, realistic sensing,
domain randomization, the load model and the 60 m/s² simulation cap. A fixed
held-out seed makes successive checkpoints comparable. Recordings are 50 Hz;
the viewer reads their frame-rate metadata. These are full games, separate
from the short skill evaluation summaries.

The trainer now records at each evaluation milestone and completion. Since the
already-running process predates that integration, a simulation-only companion
backfills its saved 500k milestones (newest missing first), watches for new ones,
and exits after the final recording:

```bash
python3 ai/bin/record_arrival.py runs/4.0-arrival-rms-selfplay --watch
```

The original companion was `airhockey-arrival-replays.service`. Its first
recordings were against the scripted sniper, not self-play; they are preserved
under explicit `_vs_sniper_step_*.json` filenames. The corrected backfill runs
as `airhockey-arrival-selfplay-replays.service`, logging to
`logs/4.0-arrival-selfplay-replays.log`. Self-play is now the recorder's default;
`--opponent sniper` explicitly requests the separate scripted diagnostic.
The recorder does not restart or mutate training, or
register this experimental policy for physical deployment. To backfill another
completed arrival run, pass its directory and omit `--watch`.

## Completed run: failed skill validation

The run completed 6M transitions / 187,500 online updates. The final held-out
MPC evaluation succeeded on 0/8 stationary shots, 0/8 moving shots and 0/8
cushions. The small evaluation fluctuates between checkpoints, but cushioning
never passed at any 500k milestone. This is not a successful learned-policy
result and must not be promoted for deployment.

This run reused only the old encoder, not the existing playing policy. It also
replaced the older plateau/minimum-goals curriculum with unconditional slot
fractions at 500k and 2M. Neither successful teacher trajectories nor finite
training loss justified advancing despite poor held-out performance. A follow-up
should validate learned skill acquisition before progression, preserve prior
playing behavior through an explicit transfer/distillation experiment, and
calibrate the load/play reward balance in short ablations before a long run.
No follow-up training has been launched as part of the replay/setup audit.

This run does not activate hardware or modify deployment limits. Its metadata
contains `deployment_ready: false`, so it cannot silently become physical
`latest`. Explicit physical loading is rejected until an arrival/load observation
adapter has been implemented and validated. The current physical policy remains
available as before.

## Learning interface and checkpoint transfer

The agent emits six normalized values: arrival position x/y, terminal velocity
x/y, arrival duration, and acceleration fraction. Each 50 Hz decision is a fresh
interruptible arrival intent. The existing decoder emits position/acceleration
commands; the actual firmware C motion law executes them, with 15 ms command
delay and workspace containment. The simulation acceleration ceiling is 60 m/s².

There are 45 observation features: the original 15 state/cap features, six
previous arrival-action values, the existing shot/time features, eight cached
per-motor fast/slow load levels, skill/game request, aim and desired speed,
controller acceleration/queued command state, and episode clock. The load
observations are quantized and cached at approximately 10 Hz, matching available
telemetry. Puck/opponent sensing includes the existing noise, delay and blind
spot model. Controller state is fresh; camera observations are not privileged
truth positions supplied to the policy.

The 3.12 checkpoint supplies its compatible encoder representation only. New
observation columns begin with zero encoder weights. Old previous-position-action
columns are not copied onto arrival-action features. The dynamics, reward, Q,
termination and actor heads, plus optimizer state, start fresh because their
old action semantics are incompatible. Four thousand demonstration updates
bootstrap these new heads before online collection.

## Skills, demonstrations and rewards

A fresh optimizer bank uses seed 20260930, disjoint from held-out seed 20261020.
It searches 32 candidates for four iterations on each of 96 fixtures, using the
actual new training interface at 50 Hz. Its fixed intent is translated into a
sequence of legal fresh arrival actions; the stored actions are the actions
actually executed. Successful trajectories are replayed under four sensor-noise
realizations and requalified by measured outcomes.

The resulting 372 episodes (16,512 transitions) contain 128/128 successful
stationary shots, 128/128 moving shots and 116/128 cushions. Failed trajectories
are excluded from cloning. These are optimized teacher results, not learned
policy performance. Raw teachers, episodes and counts are preserved under
`logs/arrival-training/demonstrations/`.

The curriculum allocates environment slots, not episode probabilities (otherwise
long games would overwhelm short skill episodes):

- Before 500k: approximately 80% skill slots and 20% scripted games.
- 500k–2M: approximately 50% skills and 50% scripted games.
- After 2M: approximately 25% skills and 75% games; one third of game opponents
  use a periodically refreshed frozen policy prior. Other opponents remain
  sniper, weak goalie and goalie. The far-side prior is not a second full MPC
  search; this deliberately changes opponent difficulty and reduces collection
  cost. Evaluation and progress must be interpreted accordingly.

Contacts come from exact collision events. Skill attempts explicitly penalize
whiffs. Shots receive goal outcomes, directional/aim shaping and an off-target
contact penalty. Cushions must contact and keep the puck slow and close for
200 ms; shooting a goal is not counted as cushioning. Goals in full games are
+100/-75; existing shot-clock/stuck-puck turnover penalties remain. Goal placement
metrics use the physical goal crossing before the simulated serve reset.

## Provisional RMS model

`ai/recipes/motor-load-20260920.json` stores the recorded per-motor fit and limits.
Fast exponential memory is about 25.34 s; slow memories use the SDK conversion.
`thermal.py` integrates squared actual velocity-derived acceleration and speed
at physics substeps, including braking. Merely requesting a high cap while
stationary does not incur acceleration effort.

The one-session fit leaves some axes unidentifiable. Training therefore adds
explicit conservative acceleration priors, a measured edge-holding current
floor, and 1–1.3x load gain randomization. These additions are labeled assumptions,
not an independently measured motor model. Heat persists across goals AND episode
resets; initial loads are randomized. Penalties rise steeply above 65% of the
modeled limit, with an extra overload cost. The drive protections are untouched.
This training preference is not an overload guarantee or a deployed runtime guard.

## Performance measurements and validation

Measurements on this RTX 4090, 5M-parameter model, horizon 8, six planner iterations,
256 samples, 32 environments:

- Original compiled planner: about 44 ms/call (61 ms eager). Two planners were
  used per old self-play iteration.
- New compiled planner: about 46 ms/call, versus 62 ms eager.
- Gradient update: about 5 ms compiled, versus 16–37 ms across eager measurements.
- New environment including decoder/load instrumentation: about 3.2 ms/step.
- Episode-safe replay sampling: about 0.054 ms; sampling cost is independent of
  replay capacity. Episodes are appended as blocks, avoiding per-env/per-tick
  TensorDict construction and whole-buffer sequence scans.
- 64 environments took about 100 ms per compiled planner call, so 32 was retained
  to preserve update-to-data ratio without sacrificing useful throughput.

These are microbenchmarks under normal desktop load, not an end-to-end speed
promise. CUDA compilation adds startup overhead. Live `recent_fps` separates
steady collection windows from that overhead. Profiling artifacts are in
`logs/arrival-training/` and `runs/_arrival-profile{32,64}/profile.json`.

The smoke run completed 10,016 transitions and 313 online updates after 500
pretraining updates; losses stayed finite and checkpoints/evaluations completed.
Its learned policy was still weak, especially at cushioning. The experiment
must establish improved held-out skills and sustained match play; successful
teacher trajectories alone do not establish that.

Eight independent 180-second random-action simulations remained finite across
48 episode resets and retained load exactly across resets. They exceeded the
modeled RMS threshold as expected under wasteful motion (peak 1.36x), confirming
that the proxy can penalize sustained overload rather than resetting it away.
This is a software stress test, not evidence that the learned policy is safe.

158 targeted tests passed after integration. The broader suite had 452 passes
and the already documented scripted CushionBot possession-control failure; that
performance threshold was not relaxed. Source snapshots and hashes accompany
the run. The parallel physical-log investigation and host position-freshness
fix are recorded separately under `logs/analysis/paddle-mismatch-20260920/`.
