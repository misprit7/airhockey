# Improving puck control, shooting and sustained play

Latest implementation: arrival actions, RMS observations/costs, optimized skill
demonstrations and the progressive skill/game curriculum are now connected in
`4.0-arrival-rms-selfplay`. See [the training record](ARRIVAL_TRAINING.md). The
older proposal and benchmark evidence below explain the motivation; they are
not a claim that the new learned policy has already improved.

September 20 implementation update: the arrival-state decoder, exact contact
instrumentation and matched offline skill benchmark are now implemented. Across
288 fresh trials per representation, both methods made every contact and scored
all 192 open-goal shots; arrivals used 34–37% less squared-acceleration effort.
See [the experiment record](ARRIVAL_EXPERIMENT.md) for scope, saved evidence and
limitations. This is an offline optimizer comparison, not a new learned policy.

Proposal based on the September 20 physical recordings, current implementation,
and saved 3.11/3.12 evaluations. These are proposed experiments, not implemented
training changes. Hardware limits and checkpoints have not been changed by this
analysis, and no hardware session is authorized by this document.

## Evidence and interpretation

Both saved evaluations use 60 m/s², 6 planner iterations, seed 7, and 8 × 30 s
games against each of sniper, weak goalie and goalie:

| Metric | 3.11 | 3.12 |
| --- | ---: | ---: |
| Goals for / against across all opponents | 24 / 36 | 122 / 17 |
| Detected on-target strikes / detected attempts | 12 / 51 | 108 / 211 |
| Goals against stationary goalie | 5 | 6 |

Sources: `logs/3.12-accel60-accuracy-baseline.json` and
`logs/3.12-accel60-accuracy-selfplay-eval.json`. This small, single-seed benchmark
shows real improvement, mainly against the sniper. It does not show mastery of
placing a shot past a goalie. Its shot metric is a heuristic based on puck speed
increase and proximity; swings that never touch the puck are not counted.

Four implementation details are directly relevant:

- `BatchRewardShaper.compute` detects forward hits using distance < 0.25 m,
  puck speed increase > 0.2 m/s, and positive longitudinal velocity. The new
  off-target penalty applies only after this detector fires and shot speed
  exceeds 1.5 m/s. It supplies no direct failure feedback for a missed swing.
- The acceleration tax is linear in the requested acceleration fraction per
  decision. It does not measure actual acceleration, torque, or cumulative load.
- Explicit learned-model planning covers 8 × 20 ms = 160 ms. Outcomes beyond
  that are represented through the terminal value estimate. Setup and shooting
  can take longer, so accuracy depends strongly on that learned value/model.
- Evaluation executes the weighted mean of elite action sequences. Around
  distinct contact approaches, averaging good candidates could create a bad
  approach. This is a hypothesis to test, not an established cause of misses.

The physical comparison is saved at
`logs/analysis/20260920-113611/report.md`. It finds identity swaps, excessive
simulated tangential rail losses and additional deployment command latency.
Continuous paddle replay is considerably closer (11 mm median / 28 mm p90 on
filtered samples). These discrepancies matter, but do not explain weak play
inside simulation itself.

## Recommended experiment order

### 1. Establish a skill benchmark and reliable event measurements

Build a fixed bank of reachable initial conditions: stationary puck and open
goal; slowly moving puck; incoming puck to cushion; awkward but reachable puck
requiring repositioning; defense; and stationary/moving goalie. Keep held-out
positions, velocities and seeds. Verify task feasibility with an oracle or
trajectory optimizer so unreachable situations do not become failed-skill labels.

Expose actual paddle contact events at physics substeps: time, contact normal,
incoming/outgoing puck velocity, and paddle velocity. Measure attempt-to-contact
rate, goal crossing error, shot speed, trap success, possession conversion,
concessions, recovery time, and movement/load. Count failed attempts explicitly
in isolated strike tasks; in full games, do not label every defensive move or
feint as a shot. Report a Pareto comparison of scoring and load rather than a
single shaped-return ranking.

Run truth observations versus realistic sensing, policy prior versus planner,
and 40 versus 60 m/s² as controlled comparisons. Store failed episodes and the
planner's predictions around contact. This separates observation failures,
contact-model errors, poor candidates, and poor objective/value estimates.

### 2. Train reliable skills before relying on full-game self-play

Progress through stationary directed strikes, moving-puck interception,
cushioning/trapping, repositioning followed by a shot, and then contested play.
Condition strike training on desired goal location and outgoing speed; use short
episodes and actual terminal outcomes. Bootstrap direct shots before introducing
bank shots and full-game shot selection. Keep successful simple tasks mixed in
to prevent forgetting and revisit held-out failures more often.

Treat contact, puck control and shot placement as separate success criteria.
Train physical control outcomes, not simply staying close to the puck or waiting
for a fixed dwell time. Avoid requiring a trap before a tactically good one-touch
shot. Use capped, temporary shaping and preserve actual goal incentives.

There is already a `CushionBot` and demonstration/behavior-cloning support in
`train_selfplay.py`. Reuse that plumbing, but qualify demonstrations by measured
success and feasibility; the existing bot is not automatically a strong teacher.
An offline optimizer can generate better strike examples using the existing
physics and motion profile. Privileged state is appropriate for a teacher or
critic; the deployed actor must still learn from available sensing.

### 3. First action-space experiment: arrival position, velocity and time

The user's preference is to judge this against our hardware and measured
failures, without using other systems' published results as an expected ceiling
or imposing a fixed hierarchy of game modes.

Compare today's `(target_x, target_y, acceleration_fraction)` with a continuous
arrival-state action: `(arrival_x, arrival_y, velocity_x, velocity_y, time,
acceleration_fraction)`. This can express a strike through a point, a cushion
moving with an incoming puck, or a reposition ending at rest. It does not force
the policy to choose a predefined hit/defend state machine. Start the time range
around 40–250 ms as an experiment, then measure whether that range is useful.

A small trajectory decoder must translate the desired arrival state into the
position/acceleration commands the real firmware accepts. Evaluate candidate
segments with the existing motion profile; a polynomial that the firmware cannot
follow would create another sim/real gap. Use the identical decoder and command
timing in training and deployment, keep feedback at 50 Hz, and expose/project
infeasible requests consistently. Measure decoder runtime and include it in
command latency. Longer requested arrivals remain interruptible by new sensing.

Run identical stationary-shot, moving-puck and cushioning scenarios against the
original action space. Only retain the new representation if it improves actual
contact success and placement at comparable movement/load. An offline optimizer
can generate successful examples and establish reachability, but need not become
a mandatory online controller or define the limit of learned play.

Lower-cost TD-MPC2 ablations should accompany that work: deterministic best
candidate or contact-mode-separated candidates versus the current elite mean;
contact-heavy replay/model training; and separately trained longer-horizon
variants if the diagnostics show planning truncation is a bottleneck. Increasing
only inference horizon is not equivalent to learning a longer reliable model,
and any additional planning latency must be included in deployment evaluation.

### 4. Retain 60 m/s² capability with an observable sustained-load budget

The policy already requests an acceleration fraction; selective bursts fit the
existing action interface. What is missing is the cost and state needed to
allocate them well. The earlier 60 m/s² physical run ended with motor 2
`RMSOverloadShutdown`; the later 40 m/s² run did not log that fault.

Model each motor separately. A useful starting approximation is:

```
load_i(t + dt) = exp(-dt / cooling_tau_i) * load_i(t)
               + (1 - exp(-dt / cooling_tau_i)) * (torque_i / continuous_torque_i)^2
```

This is a proposed calibrated proxy, not a claim about Teknic's exact internal
protection algorithm. Estimate torque from cable geometry, position, actual
acceleration and holding/preload forces, then fit against timestamped motor
torque and RMS telemetry where available. Raw torque percentage is not the same
as thermal headroom. The existing `ENC` response exposes per-motor torque; log
it asynchronously/cached so drive polling does not add latency to policy ticks.
The original comparison logs lacked enough current history to calibrate this
model. The user's later 2026-09-20 13:09 session now supplies all four fast/slow
RMS channels and measured currents; see `MOTOR_LOAD_LOGGING.md`. Its provisional
motion model predicts the final 20 seconds with 0.67–2.53 percentage-point RMS
error per motor after fitting the first 40 seconds. The new arrival training
experiment now uses a conservative version of that fit for load state/cost.
Holding current varies substantially with position, and a
more flexible position fit overfits this single minute, so independent and
longer validation remains necessary.

Expose remaining per-motor headroom to the policy and train with a load-budget
constraint or steep near-limit cost. Let isolated high-acceleration saves and
strikes remain valuable while repeated reversals, needless repositioning and
prolonged high-load holding consume budget. Load should carry across points;
randomize warm starts and test multi-minute matches so the policy cannot exploit
30-second episode resets as free cooling.

Mirror any runtime load limiter in the training environment and retain an
independent runtime bound; a learned preference alone cannot ensure no overload.
Do not bypass drive protections. Teknic documents the SC dashboard's RMS-current
measure and shutdown threshold in its
[SC manual](https://teknic.com/files/downloads/Clearpath-SC%20User%20Manual.pdf#page=55).
Selective acceleration is a plausible solution, but a persistent holding-load
or mechanical problem may also require a physical correction. The logs do not
support assuming all overload is caused by useful fast motion.

Before torque calibration, squared actual acceleration plus reversal penalties
can serve as an explicitly provisional training proxy. It must not be presented
as a validated motor thermal limit.

## Concrete first batch

1. Benchmark the current checkpoints with exact contact events and save misses.
2. Train a targeted strike/control curriculum at a 60 m/s² ceiling, comparing
   successful demonstrations against an otherwise identical no-demo run.
3. Compare the current planner against a contact-planning oracle and a
   deterministic candidate-selection ablation on identical held-out scenarios.
4. Add provisional sustained-load state/cost in simulation; sweep its weight and
   plot scoring/contact success against rolling load over long matches.
5. Incorporate the measured rail and latency corrections and re-evaluate every
   candidate, then calibrate thermal behavior using separately authorized
   physical recordings before claiming that 60 m/s² sustained play is resolved.

The highest-confidence immediate gains are better contact measurements and
targeted skill training. The highest-upside design change is explicit contact
planning. The most direct route to occasional 60 m/s² use is learning with
observable, calibrated per-motor load state. A further long self-play run with
larger miss penalties alone is less well targeted to the observed failures.
