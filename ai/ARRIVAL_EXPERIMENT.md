# Arrival-state action experiment — September 20, 2026

Follow-up: the learned arrival/RMS curriculum is now running as
`4.0-arrival-rms-selfplay`; see [the training record](ARRIVAL_TRAINING.md). The
results below remain the earlier optimizer benchmark, not learned-policy scores.

Implemented and evaluated in simulation. No hardware was enabled, no deployment
limits were changed, and no new learned checkpoint was registered.

## Result

At equal offline search budget, arrival actions produced similarly accurate
shots with lower actual acceleration effort. This supports using the decoder
for a targeted learning experiment; it does not establish improved learned
match play or resolve motor RMS shutdowns.

Three fresh fixture seeds, 96 trials per task (288 total per representation):

| Task | Original position commands | Arrival actions | Arrival effort reduction |
| --- | --- | --- | --- |
| Stationary shot | 96/96 contacts and goals | 96/96 contacts and goals | 34.1% |
| Moving-puck shot | 96/96 contacts and goals | 96/96 contacts and goals | 36.9% |
| Cushion and retain | 94/96 successful cushions | 96/96 successful cushions | 34.5% |

All 288 attempts contacted the puck with both representations. All 192 shots
per representation crossed within 50 mm of their requested goal location.
Median placement errors were 2.41/2.34 mm (original stationary/moving) versus
1.13/1.41 mm (arrival). These are idealized simulator measurements with perfect
state, not expected physical accuracy. Mean outgoing shot speed remained near
3 m/s and was slightly higher with arrivals.

Effort is the integral of squared actual velocity-derived acceleration,
sampled every 2.5 ms, including braking/recovery. It is not the requested cap
and is not a calibrated motor heating estimate. Paired fixture bootstrap 95%
intervals for the reduction were 28–40%, 31–42%, and 27–41%, respectively;
these describe this fixture distribution, not general match performance.

Peak measured arrival acceleration on the moving-shot set was 59.83 m/s²;
time above 40 m/s² averaged 5.73 ms per attempt. No selected main-comparison
trajectory triggered the >105%-of-cap backstop-violation metric. The simple
tasks often need less than 40 m/s², so this is not a test of frequent emergency
saves or sustained high-load play.

The effort-weight sweep on one fixed seed (96 trials, identical delay) gave:

| Weight | Original mean effort | Arrival mean effort | Original / arrival shot speed |
| --- | ---: | ---: | ---: |
| 0 | 372.7 | 184.6 | 3.40 / 3.39 m/s |
| 0.3 | 198.0 | 117.0 | 2.95 / 3.06 m/s |
| 3 | 91.6 | 61.8 | 2.58 / 2.63 m/s |

Every sweep run scored all 64 shots. At the strongest penalty, both interfaces
retained 31/32 cushions. Increasing the cost saves effort partly by reducing
shot speed; the outcome/load tradeoff must remain visible during training.

## Implementation and comparison

- `airhockey/arrival.py`: normalized six-component action
  `(arrival_x, arrival_y, velocity_x, velocity_y, time, acceleration_fraction)`.
  Arrival time spans 40–250 ms; terminal velocity components span ±3 m/s.
  Acceleration uses the existing quadratic 5–100% mapping. Requests can be
  interrupted at any 50 Hz decision. After the deadline, a fixed follow-through
  target lets the firmware profile brake.
- The decoder predicts queued-command latency with the actual firmware C
  profile, builds a cubic arrival reference, and makes two local shooting
  corrections through that same profile to produce a position command.
  Actual paddle state is always integrated through the firmware; the cubic
  never assigns simulated positions or velocities. Unreachable requests are
  bounded by workspace/caps, and their execution errors are recorded rather
  than assumed away. This is an approximate controller, not an exact
  feasibility projection.
- Nine reachable stop/strike arrival tests execute within 3 mm and 0.15 m/s of
  their requests. Across the optimized benchmark, median requested-arrival
  errors were 2.14/3.01/1.51 mm and 0.214/0.248/0.113 m/s for stationary shots,
  moving shots and cushions. Not every arbitrary request is reachable.
- Measured single-cart decoder p99 was 0.43–0.54 ms. Each paired comparison
  includes that cost on top of the measured 12 ms host delay, rounded up to a
  physics tick: 12.5 ms for two seeds and 15 ms for the third. Both interfaces
  get the same delay within each comparison.
- `airhockey/batch_physics.py`: optional exact impulse callback containing
  body, environment index, substep time, normal, incoming/outgoing puck
  velocity and paddle velocity. It is disabled by default. Outgoing event
  velocity explicitly precedes the puck-speed backstop.
- `airhockey/skill_benchmark.py`: fixed seeded stationary/moving/cushion trials;
  320 ms attempt, physical recovery, and puck flight to two seconds. No goalie,
  invented serves, shot-clock relaunches or training reset effects. A cushion
  must remain below 0.6 m/s and within a 60 mm surface gap for 200 ms after the
  attempt. Every attempted trial stays in the denominator; whiffs are penalized.
- Offline CEM uses 96 candidates × 8 iterations per fixture for both methods,
  with an identical task objective and effort weight. The incumbent is retained;
  scoring uses the executed trajectory rather than an elite-mean trajectory.
  Baseline position/acceleration commands can change every 20 ms (48 free
  parameters); one arrival action has six parameters with decoder feedback at
  50 Hz. Both start from **the exact same executed command sequence** generated
  by a shared analytic seed. Position exploration has temporal correlation,
  unit marginal variance and matched initial spatial perturbation scale.
  The comparison matches rollout counts, not wall-clock compute. The arrival
  decoder adds simulation work to each candidate.
- Selected trajectories and contact events are saved, including unsuccessful
  cushions. The union of successful executed trajectories demonstrates
  feasibility for every main-comparison fixture; no unsuccessful fixture was
  removed to obtain the reported rates.

## Reproduce and inspect

From the repository root, simulation only:

```bash
PYTHONPATH=ai python3 ai/bin/bench_arrival.py \
  --output logs/arrival-next \
  --per-task 32 --population 96 --iterations 8 \
  --seed 20260926 --accel 60 --effort-weight 0.3
```

Use seeds 20260926–20260928 for the main comparison. The output directory must
be new. The command records measured latency, physics, source hashes, actions,
contact details, 50 Hz replay traces, search histories and per-task results.
Reruns can quantize to a different command delay if host timing differs; the
exact delay is in `config.json` and can be supplied to `SkillTrials` when
replaying saved actions.

Saved local artifacts:

- `logs/arrival-validated-2026092{6,7,8}-effort0.3/{results.json,trials.npz}`
- `logs/arrival-validated-20260926-effort{0,3}/` for the cost sweep
- `logs/arrival-study/summary.json`, `comparison.png`, and `summarize.py`
- `logs/arrival-study/source/` preserves the experiment and shared physics sources

The pilot/development/final-named exploratory directories predate the final
decoder and are not the reported comparison. Only `arrival-validated-*` is used
above. Held-out here means fixture seeds unused for decoder development; the
offline optimizer still sees each test state's true physics. It does **not**
mean a learned model generalized to unseen states.

Validation: 10 new tests pass. Full AI suite: 417 passed, one existing failure
in `test_bot_controls_the_puck_in_most_possessions_against_a_weak_goalie`
(27% versus >40%, already present after the rail/latency corrections).
Lint and diff whitespace checks pass. The behavior threshold was not weakened.

## Implication for the next learned policy

The simulator and existing command interface can already realize accurate
contacts on these tasks. The missing piece is getting a learned controller to
find them reliably. Arrival actions offer a compact, lower-effort representation
and successful executed examples for skill training. They are worth retaining
as an experimental decoder, not yet replacing the deployment interface.

Next compare learned goal-conditioned skill policies using successful optimized
examples against the same training without those examples; retain a position
action baseline. Evaluate sensor noise, harder interceptions and goalies before
self-play promotion. A proper 50 Hz learned planner must be tested separately:
this comparison holds one arrival intent per isolated trial, whereas a learned
policy may revise it every decision. Sustained RMS protection still needs
calibrated motor load observations and longer matches; isolated acceleration
effort reductions do not establish that protection.
