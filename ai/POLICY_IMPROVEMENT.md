# Autonomous policy improvement

Scope: simulation and offline analysis only; physical activation always needs
case-specific permission. Existing deployment policies and limits are preserved.
Every new experimental run is excluded from physical `latest` selection.

Objective: consistent on-goal shots, strong defense, controlled possessions,
competitive self-play, and sustained play within modeled physical/load limits.
"Superhuman" cannot be established by simulator scores alone; use quantified
shot-speed/angle save rates and later separately authorized hardware validation.

## Work sequence

1. Preserve and benchmark 3.11 at 40 and 60, 3.12 at 60, and failed 4.0.
   Common held-out shots, cushioning and incoming on-goal defense; report prior
   and MPC separately, all attempts, actual contacts/goals and load.
2. Small supervised feasibility experiment: learn successful arrival shots on
   fixed fixtures, then verify held-out fixtures. Test direct encoder+actor
   supervision rather than assuming low latent-rollout BC loss means skill.
3. Transfer actual previous-policy behavior and compare legacy-action
   continuation against arrival-action learning with comparable evaluation.
4. Performance-gated curriculum; retain previous skills and evaluate repeatedly.
   Reject/regress experiments on measured play, not finite optimizer loss.
5. Separate motor-load reward diagnostics and short ablations. Sustained load
   and physical command constraints must pass alongside play, not replace it.
6. Extend only successful pilots; preserve checkpoints, reproducible seeds,
   source/config snapshots and equal-body same-checkpoint MPC self-play replays.

Initial artifacts: `logs/foundation/`; common benchmark:
`python3 ai/bin/bench_foundation.py --per-task 100`.

## Findings and experiments (2026-09-20)

- Fixed a measurement error: the game referee could relaunch an untouched puck
  in a short skill trial. Skill trials now disable relaunch, and a scored shot
  must also have actual robot contact. Earlier affected reports are explicitly
  invalidated in `logs/foundation/INVALIDATED.txt`.
- Corrected central baseline, 100 attempts per skill, 6 iterations / 256 MPC
  samples: 3.12 achieves 50 stationary goals, 63 moving goals, 0 cushions and
  97 defensive saves. Failed 4.0 achieves 19 / 18 / 0 / 48. These defense cases
  aim near goal center; `workspace-v1` now tests the full safe goal mouth.
- Cold-only demonstration observations caused a large transfer failure with
  warm motor-load inputs. New qualified demonstrations include sensing noise,
  dynamics randomization and warm initial load; they require contact, on-target
  direction and a real goal. Wide set: 8,104 episodes / 371,159 transitions.
- Comparing arrival BC with full legacy-model continuation favored the latter
  so far. Legacy transfer preserves every original weight exactly and adds zero
  input columns for task/load/controller observations. A direct weight audit
  verified this. No actor, dynamics, reward or Q head is reset.
- Local shooting can improve substantially through imitation, but uniform BC
  also regresses defense and is not sufficient for full-game aim. New pilots
  learn the extra input columns, retain successful reference MPC defense
  approaches, and balance skill/contact-phase sampling. Never select by BC loss.
- Existing profile containment can stop instantly at a wall after a command
  change, exceeding the acceleration cap in simulation. Added an experimental
  predictive command guard using the unchanged host firmware motion law in a
  separate host-only library. Physical firmware and deployment defaults are
  untouched. A guarded 8-game / 180-second diagnostic stayed below modeled load
  0.987 and peak acceleration 60.001 m/s²; rare conservative forecast failures
  remain logged, and this is not a hardware safety guarantee.
- `train_foundation.py` retains the complete model, keeps an immutable demo
  replay alongside online replay, directly supervises real observations, and
  uses skill-plus-sustained-load gates rather than fixed-step promotion. Both
  prior and MPC are measured. Final audit seeds are separate from the validation
  seed used for selection. There is no deployment-ready candidate yet.

Useful artifacts: corrected `logs/foundation/baseline.json`, diagnostic
`match-guard-v3.json`, scratch `_foundation-bc-*` runs, and corresponding
`evaluations.json`. All new runs carry `deployment_ready: false`.

### Further corrections and first RL pilots

- An integer constructor acceleration (e.g. `60`) created integer runtime-cap
  buffers. Repeated cap changes silently truncated commands; the forecast and
  executed controller could disagree enough to cause a wall stop. Dynamics
  limits and queued caps now always use floating point. Regression tests cover
  integer/float equivalence and an actual MPC sequence that produced 244 m/s².
  The captured 400-case MPC diagnostic passes after this fix. Earlier integer-
  cap benchmark numbers are historical diagnostics; rerun final comparisons.
- A guard margin could trap a stationary cart within 1 mm of a wall. Escape
  commands are now allowed while still checking acceleration. The subsequent
  ten-minute / 16-game check had no such persistent traps and acceleration
  stayed at 60, but modeled RMS still reached 1.17: it fails load qualification.
- Random bank/straight requests were enabled in earlier full-game diagnostics,
  unlike the new demos and physical runner's default. `shot_requests` is now
  explicit, defaulting to none. The difference does not explain away the poor
  shot accuracy; both variants failed. Old match artifacts label this setting.
- The old referee applied a shot-clock turnover only to the near side and
  served randomly after far-side stalls. Foundation training/evaluation now
  opt into symmetric turnovers. Existing unrelated environments retain their
  defaults. Older match reports are labeled `symmetric_referee: false`.
- `_foundation-rl-pilot-v2`: full MPC continuation at 50k scored 72 / 61 / 46 /
  93 on the wider stationary/moving/cushion/defense suite, from 38 / 37 / 0 / 94
  after calibration. It regressed at 100k to 36 / 37 / 2 / 92, so the preserved
  50k checkpoint is the continuation candidate, not the last checkpoint.
- `_foundation-rl-prior`: a cheaper actor-critic collection ablation regressed
  over 200k, particularly in its learned model's MPC behavior. Not promoted.
- Both pilots heated isolated 600 ms drills continuously, reaching modeled
  loads above 1.4. New independent drill fixtures sample independent warm load
  states; actual games preserve heat across all goals AND episode resets.
  Sustained-game load tests remain mandatory; resetting drill heat is not a
  way to qualify a continuously played match.
- The inherited acceleration sampling std was about 7.39 in pre-squash action
  units (almost binary min/max cap draws). A controlled probe recalibrates only
  stochastic std outputs to [.05,.05,.2]; exact deterministic-action equality
  is verified. All mean-policy and world-model weights remain intact.
- `_foundation-rl-balanced-load` resumes the preserved 50k model for a 300k
  pilot, with independent drill heat, symmetric referee, calibrated sampling,
  stronger deliberate-shot accuracy reward, and cushion-specific feedback.
  Source/config snapshots are stored per run. No checkpoint has yet met the
  overall skill and sustained-load gates or been approved for deployment.

### Continued validation (not yet a qualified policy)

- Corrected guarded wider-suite 3.12 reference, same seed / 100 per task:
  prior 27 / 24 / 0 / 77; MPC 33 / 50 / 0 / 95. Results are in
  `logs/foundation/reference-v3.json` (floating caps and actual guard on both
  legacy 22-input and extended policies). Earlier baseline adapter silently
  omitted the optional guard for 22-input policies; this is now corrected.
- The preserved 50k checkpoint with calibrated exploration scores 65 / 60 /
  64 / 94 using MPC. The 100k checkpoint does not recover just by calibrating
  sampling; that inference-only ablation is not a promotion candidate.
- `_foundation-rl-balanced-load` regressed at 50k (MPC 59 / 35 / 11 / 52) and
  was stopped. `_foundation-rl-stable` froze the learned encoder, executed
  the MPC mean during collection, and distilled only physically successful
  on-target/control/defense episodes. Its prior learned control (53/100 at
  100k), but MPC still regressed (60 / 41 / 35 / 52). It was stopped too.
- Planner smoothness and conservative-value probes provide modest gains,
  not a solution: logs are `planner-smoothing.json` and
  `planner-value-ablation.json`. The conservative-value implementation lives
  only in the diagnostic script; it has not been adopted in the trainer.
  A termination-head probe found no premature termination on the tested
  initial-state/mean-rollout paths; do not claim it caused these regressions.
- **Confirmed terminal-credit problem:** all 8,104 large positive demo rewards
  occur only at the last slot of valid 8-step replay sequences. There are
  314,431 valid starts, so these windows are rare; rho=.7 further weights that
  final slot by only .08235. `terminal-credit-audit.json` records the counts.
  Replay now optionally mixes in terminal-ending windows with an incremental
  anchor ring (no O(buffer-size) scan); overwrite safety tests pass. Defaults
  remain unchanged for other trainers.
- `_foundation-terminal` is a 1M-step continuation experiment from the preserved
  50k model: rho=1, terminal-window fraction=.5, mean MPC collection, frozen
  encoder and successful-trajectory self-imitation. Curriculum gates still
  apply to the actual execution mode. `_foundation-terminal-calibration` is a
  bounded 2,000-update offline world-model probe with mean policy/encoder fixed,
  using the same terminal-credit correction. Inspect their evaluations before
  drawing conclusions or choosing a longer continuation.

### World-model isolation and direct policy training

- The terminal-credit correction did not cure regression: `_foundation-terminal`
  MPC reached only 18 / 26 / 15 / 47 at 100k. The offline 2,000-update model-only
  probe retained the exact prior but reduced MPC to 20 / 27 / 0 / 34. These are
  negative results, not candidates for promotion.
- Short defense rollouts had a hidden +75 survival objective despite using the
  same observed request as ordinary games. They now retain the normal game
  rewards (including scoring), and end as truncations that bootstrap continuing
  game value. A test verifies equal rewards for equal scoring states.
- `_foundation-world-first` freezes the encoder and complete actor while
  training world/value heads online with corrected defense rewards. Weight
  comparisons at 100k verify exact encoder/actor equality. MPC nevertheless
  regressed to 49 / 40 / 9 / 80. A longer observation window is being retained
  to distinguish transient model adaptation from a persistent failure.
- New `train_foundation_ppo.py` is a controlled alternative: direct on-policy
  training on actual simulation transitions, clipped policy ratios, KL stopping,
  small actor learning rate, and independent critic. The full inherited model
  is preserved, but latent MPC is explicitly unused. Only the actor and the
  encoder's 20 new context columns train; the original 22 input columns and
  other encoder parameters remain fixed. The benchmark/replays use the actual
  deterministic prior used by this experiment.
- `_foundation-ppo-pilot` starts from the best wide behavior-cloned checkpoint.
  At 106,496 steps its stationary/moving/cushion/defense results are
  98 / 67 / 39 / 88, versus 97 / 63 / 38 / 87 initially. On-target moving
  attempts improve 47 -> 56. This is an early validation result; defense and
  sustained load still prevent promotion. Collection initially runs around
  4,000 steps/s, versus roughly 500 for latent MPC training.
- Benchmark protocol `workspace-v2` also disables referee replacement serves
  during isolated defense attempts. `reference-v4.json` reruns the old policy
  under this protocol. Normal-game goal-aim context is now explicitly .5 on
  both sides, matching the far player's existing request; isolated shooting
  drills retain varied aims. Already-running processes retain their captured
  source versions, and final comparisons must use the current protocol.
- GAE tests verify that true terminals do not bootstrap, time-limit truncations
  do bootstrap, and neither leaks the next episode's advantage through reset.
  No checkpoint is deployment-ready. Physical defaults remain unchanged.

### First successful policy improvement and the possession gap

- PPO retains its early gains over a longer run. At 1,204,224 steps it reaches
  100 / 75 / 60 / 95 on the validation skill suite, and its 8-game / 180-second
  load gate peaks at .9427 with acceleration within the cap. It therefore earns
  the first curriculum advance to 50% full-game slots. This is a training-stage
  acceptance, not final qualification: other seeds and longer sessions still
  need testing. Earlier gates peaked at 1.050, 1.006 and .998.
- Independent validation seed 20261105, 200 attempts each, at 400k gave
  197 / 171 / 109 / 174. The variation especially in defense means one
  100-case validation result must not be called a final performance estimate.
- The aligned drill/game gap was investigated explicitly. At 800k, using the
  normal game request still gives 98 stationary goals and 75 moving goals per
  100; arbitrary starting paddle positions reduce these to 36 and 30. This
  points to positioning/setup, not primarily the task-request cue. Contact
  rates are 72 and 81: some misses and many poorly lined-up hits remain.
- Added opt-in `possession_fraction` to training: selected shot fixtures start
  with the paddle anywhere in the reachable workspace, including ahead of the
  puck. Other physics/rewards remain the same. `--random-paddle --game-requests`
  benchmarks that ability separately. Possession-enabled PPO continuations
  retain aligned tests and add setup performance to selection/advancement.
- Full-game diagnostics now also report actual acceleration RMS, time above
  40 m/s² and average requested cap. At 600k, one 180-second sample uses >40
  only 2.4–4.0% of the time, yet acceleration RMS is 21–23 m/s². The load
  problem is not explained solely by rare maximum-acceleration bursts.
- PPO trainer can resume critic and optimizer state from the paired final
  checkpoint; new state files include a checkpoint hash to reject mismatches.
  World models remain unused in this direct-policy branch. Future replay
  metadata names PPO explicitly, and always identifies each side's inference
  mode. Existing running processes use their source snapshots.
- At 1,802,240 steps PPO also passes the second skill/load training gate:
  99 / 77 / 77 / 95. It advances to 75% game slots, but is still not a strong
  full-game opponent: the preliminary head-to-head against 3.12's full planner
  is decisively losing. Neither skill scores nor mirror self-play establish
  competitive match performance. The next continuation adds the established
  full planner to half of the external-policy opponent slots, retaining peer
  self-play and scripted diversity, alongside broader possession starts.
- The world-only TD-MPC2 branch recovered partially at 200–300k, then regressed
  at 400k to 83 / 68 / 1 / 79 despite an unchanged actor. It is stopped at 400k
  rather than consuming the remaining 600k budget. Its preserved checkpoints
  remain available; current effort is directed to the improving on-policy
  branch and the full-game deficits exposed by head-to-head testing.

### Strong-opponent continuation and shot-direction correction

- The 1.2M PPO checkpoint loses 5–81 across eight two-minute games against
  3.12 full MPC; reversed roles give 4–105. This is not merely a side bias.
  The reference exceeds the provisional RMS envelope during these games;
  report that constraint difference rather than treating it as a deployable
  reference. It remains a useful aggressive training opponent.
- `_foundation-ppo-possession-league` resumes the final 2,007,040-step PPO
  model and its critic/optimizers, with 35% of shot fixtures starting from
  arbitrary paddle positions. Half of external-policy game slots use 3.12
  full MPC, half use the accepted peer prior; scripted diversity remains.
  Its initial model is bit-for-bit equal to the source. This run was stopped
  at 835,584 new steps to carry forward two diagnosed corrections below.
- Random starting positions now reject paddle/puck overlap. Earlier arbitrary-
  start diagnostics allowed overlapping initial bodies and are preliminary;
  the corrected protocol labels starts `random_nonoverlapping_paddle`.
- Direct-ray accuracy was incorrectly penalizing legitimate banks. A short
  reference-game audit found many trajectories that miss the straight-ray
  target but go into goal after rails. New `shot_flight.py` separates direct
  aim from a first-end goal trajectory, using the calibrated normal/tangential
  rail response and drag. Later end-wall ricochets do not rescue a missed shot.
  The fast conservative predictor was checked against 5,000 independent
  physics rollouts: 1,385 predicted goals, zero false positive goals; 452
  additional physical goals were conservatively rejected near the mouth or
  by the tighter full-puck-width criterion. This is an evaluation/model check,
  not proof that the physical rail calibration is exact.
- Foundation's accuracy reward now credits such valid bank paths and continues
  to penalize genuine misses. Original direct-ray metrics remain explicit;
  benchmarks additionally report bank-aware trajectories. A regression test
  verifies a modeled bank receives the positive reward and an end-rail miss
  receives -40. Full-flight launch states/parameters are saved by new match
  evaluations so scoring definitions can be audited without rerunning games.
- `_foundation-ppo-possession-league-v2` continues the complete 835k model and
  optimization state for 4M further steps, with nonoverlapping starts and the
  corrected shot reward. It starts at curriculum stage 1 to allocate more
  approach practice; possession performance is now an explicit additional
  progression gate. It is still simulation-only and not deployment-ready.

### Match failures, exploration and representation ablations

- The v2 continuation plateaued and was stopped gracefully at 3,448,832
  additional steps, preserving the complete final model, critic and Adam
  state. Its final starting assessment for the next experiments is 99 / 75 /
  80 / 94 on aligned shooting, moving shooting, cushioning and defense; the
  two arbitrary-start shooting scores are only 32 / 32. These drill results
  do not establish full-game competence.
- At 1,007,616 steps v2 lost 4–119 over eight two-minute matches against 3.12
  full MPC. Candidate peak modeled load was about 0.85, while the aggressive
  reference exceeded 1.15. The candidate used >40 m/s² for only about 1–2%
  of the time. Its actual acceleration RMS was 18–20 m/s². It is conserving
  load but failing to defend/score, so this is not a successful replacement.
- Collected 153,600 full-game MPC teacher transitions (153,600 short-run
  samples, not proof of sustained load qualification). A 3,000-update actor
  distillation with successful-skill retention still lost 1–28 over eight
  one-minute matches. Plain mean-square imitation is not adopted. The
  failed geometric lineup coaches are also diagnostic artifacts only.
- A bounded quiet-MPC test parked at low acceleration when the puck was
  safely moving away. It intervened for 32.1% of frames, but still reached
  modeled load 1.264 in eight three-minute self-play games. Parking alone
  does not solve the RMS problem.
- Two one-million-step continuations begin from the same final v2 model and
  paired optimization state: `_foundation-ppo-explore-context` retains the
  frozen original encoder, while `_foundation-ppo-explore-encoder` allows
  the full encoder to train. Both widen exploration only for awkward slow-
  puck setup and urgent incoming acceleration, retain precise-shot noise,
  and use 50% arbitrary shooting starts. Actor/context learning rates are
  initially 1e-5; the KL limiter is monitored because changing the encoder
  can change actions much faster than updating the actor alone.
- These two experiments move the soft pre-overload penalty onset from 0.65
  to 0.85. They do NOT change the load dynamics, cost at overload, physical
  acceleration cap, or final <0.95 sustained-load qualification requirement.
  This tests whether the previous cost suppressed useful bursts too early.
- A separate observation-only interception ablation tests candidate targets
  through the firmware motion profile with nominal table parameters. It
  cannot read the hidden randomized physics. It is a diagnostic, not a
  deployment change or an accepted policy.

### Observation-based defensive interception and setup credit

- Matched seed/body/opponent tests now isolate the defensive interceptor:
  final v2 alone scores **2–47**, while that same checkpoint with interception
  scores **6–16**, each over eight 60-second games against 3.12 full MPC.
  The interceptor predicts several contact lines and simulates 25 target/cap
  candidates through the firmware profile, including the queued command delay.
  It uses the policy observation and nominal physics, not hidden simulator
  state. It selects a low-cap reachable interception and otherwise minimizes
  miss distance. Prediction is approximate and requires broader validation.
- Candidate peak modeled load is 0.8535 in that one-minute test, with zero
  unresolved motion-guard interventions. Peak acceleration is 60.0011 m/s²
  (floating-point simulation tolerance). This is promising, not sustained
  qualification: the policy still loses and longer/hot-start tests remain.
- The setup-planner / learned-shot / interception combination is being tested
  separately. These wrappers currently live in `logs/foundation/probe_intercept.py`
  and are not wired into physical deployment or advertised as accepted agents.
- Added optional `setup_potential_weight` to the foundation environment and
  PPO trainer. The potential measures a collision-avoiding route behind a slow
  puck, fading out for fast shots and disabled for cushioning requests. The
  shaping is exactly gamma*Phi(next)-Phi(current), with terminal Phi=0 and
  gamma=0.995 matching this PPO trainer. Tests verify the discounted telescoping
  identity and preserve truncation bootstraps. This is intended to supply early
  setup credit without rewarding endless circling; no trained result yet.
- Periodic checkpoints from future PPO runs now also save paired optimizer/
  critic state and the checkpoint hash, so a measured intermediate model can
  be continued without losing optimization state. Current running snapshots
  predate this addition and still save paired state only when they end.
- The full-encoder imitation ablation completed 10,000 updates. Held-out
  action error plateaus around 0.082 and arbitrary-start goals remain only
  17/20 per 100. Encoder adaptation alone did not make mean-square teacher
  imitation effective. Its excellent retained cushioning is not match evidence.

### Training with defensive interception

- The exploration continuations both completed 1,007,616 steps. Their pure
  neural policies still lose: full-encoder variant 1–33 and frozen-original-
  encoder variant 2–23 in the matched eight one-minute reference games.
- `_foundation-ppo-setup-potential` completed 2,007,040 more steps, with full
  encoder learning rate reduced to 1e-6, 75% arbitrary shot starts and setup
  potential weight 20. Its 606,208 checkpoint scores 100 / 79 / 98 / 97 on
  aligned skills and 39 / 40 on arbitrary-start shooting. The final checkpoint
  improves arbitrary stationary shooting to 53/100 but regresses moving
  aligned shooting to 65/100. Neither is a qualified general policy.
- The 606k checkpoint plus the basic interceptor scores **5–6** in matched
  reference games, compared with the earlier v2+interceptor's 6–16. This
  supports retaining that intermediate model instead of assuming latest is best.
- Ported the basic interception forecast to a separate host-only C library,
  `libintercept_motion.so`. The firmware controller itself is unchanged.
  Candidate commands match the original diagnostic exactly on 1,000 sampled
  observations. Median prediction time on this host: 3.92 -> 0.19 ms for one
  robot, 16.57 -> 4.37 ms for 64 simultaneous incoming cases. Unit tests compare
  variable-horizon forecasts with independent calls to the firmware profile.
- `_foundation-ppo-interception` continues the 606k model and paired critic/
  optimizer state for 2M steps. The controller acts in incoming game states;
  PPO learns the remaining actions. Controller-selected actions are explicitly
  excluded from the actor likelihood loss, while rewards and value targets
  retain their consequences. Peer self-play uses the same controller; the
  established reference opponent remains unchanged. Evaluations/replays use
  the composite policy, and checkpoint metadata reconstructs it on loading.
  Models trained by PPO default to prior inference rather than the retained,
  stale latent planner. Physical deployment remains disabled.
- At 401,408 new steps the composite policy scores **9–10** in reference games,
  peak modeled load 0.841. Forward returns: 116, of which 44 would reach the
  empty opposite goal including banks. Good defense has not yet solved attack.
- Initial composite-policy self-play ran eight **900-second** continuous games:
  score 37–32, peak modeled load **0.8434**, zero modeled overload time, peak
  acceleration 60.0006 m/s². There was one unresolved guard forecast; measured
  simulator motion did not exceed the acceleration tolerance. This is longer
  modeled-load evidence, not a physical drive guarantee. It produced 685
  forward returns, 278 on-goal flights. Quiet or stalled play must not be
  mistaken for strong self-play, so new evaluations also count referee
  turnovers, puck speed and slow-puck time, and save fast/slow load traces.
- Forward-return metrics now include passive aimed rebounds separately from
  the historical accelerating-shot definition. All-flight launch states remain
  auditable. Earlier accelerating-shot counts do not include every deliberate
  countershot, and must not be labeled total shot accuracy.
- A settled aimed-countershot diagnostic gives 99/100 defensive saves and
  72/100 goals into the empty opposite goal, but its first match version loses
  3–26. Rejecting predicted early paddle/puck collisions and out-of-workspace
  paths improves this to 5–14, still not clearly better than the simpler
  interceptor. These remain separate diagnostic variants, not the training
  controller or a promoted policy.
- A stricter stationary setup coach finally resolves a useful subproblem:
  collision-avoiding navigation, precise settling, then a direct physical
  stroke scores **90/100 stationary goals from arbitrary paddle starts** at
  navigation cap 10 m/s²; all 100 contact the puck. It is poor on moving pucks
  (19/100), so the next full-game combination applies it only below 0.15 m/s.
  This is being evaluated separately; it is not wired into the training run.

### Arrival-controller and tactical-selector follow-up

The direct motor-policy training has repeatedly preserved aligned contact skills
without learning reliable setup from arbitrary paddle positions. Successful
teacher trajectories alone do not solve this: 10,000 supervised updates achieved
held-out command MSE about 0.0022 but only 37% stationary and 25% moving goals
from arbitrary starts. Low command MSE is not a closed-loop qualification metric.
The demonstration-assisted PPO continuation is still running; through 1.6M steps
its arbitrary-start rates regressed from 42/38 to 38/29. Earlier checkpoints are
preserved. No newer policy is promoted automatically.

A physics-based approach now navigates around the puck, settles behind it, and
executes a 160 ms arrival contact with the ordinary firmware motion profile.
Reachable home recovery and the earlier observed interception controller complete
the policy; a frozen learned policy handles other states and cushioning. It uses
only policy observations, not simulator parameters or future opponent actions.
The reusable implementation is `airhockey/arrival_play.py`. The fixed center-shot
version matches the original probe's commands within 2e-6 across 1,280 test inputs,
including committed contacts. A cushioning request explicitly cancels a shot.

Development seed 20261108, 100 arbitrary-start trials each, direct 3 m/s shots:
98 stationary goals, 66 moving goals; 99/72 trials had on-goal contact. The first
8 x 60 s match against the old full MPC policy won 8–2, but an independent
8 x 180 s replication lost 10–13. Peak candidate modeled load was 0.972 in that
replication (below overload 1, above our conservative promotion threshold 0.95).
This is promising controlled behavior, not evidence of general superiority.
The reversed-side and 15-minute continuous self-play checks are in progress.

An optional incoming-flight predictor includes calibrated rail damping and
friction, plus slow incoming pucks. It matched independent nominal physics
crossings within 1.5 cm and 10 ms in randomized unit tests. It saved 100/100
isolated defense trials but lost its full-game comparison 5–10, versus the
original controller's 8–2 on that seed. It remains opt-in and is not the selected
defense controller. Better prediction alone did not improve match performance.

`train_shot_intent.py` now tests a separate tactical approach: semi-Markov Double
DQN chooses one of ten bounded shot intents (goal location, desired puck speed,
and direct/left-bank/right-bank). The original neural policy and encoder remain
frozen. A zero-initialized tactical head exactly selects the tested center-shot
baseline before learning. Each selected intent remains committed through setup
and contact; the actual discounted rewards accrue until the next decision or
finite episode endpoint. Controller ticks are not mislabeled as policy choices.
A diverse scripted/frozen-self-play mix and random-start skills train together;
more match training and peer updates require measured skill, match, and load
checks. Initial tactical run `_foundation-shot-intent-pilot` failed before its
first rollout due to a reset-mask type mismatch; this is corrected and the active
successor is `_foundation-shot-intent-pilot-v2` (1M pilot steps, 200k evaluation
interval). Checkpoints bundle their frozen fallback policy. All runs remain
simulation-only and explicitly not deployment ready. Final held-out seeds
20261201/02/03 remain unopened.

The demonstration-assisted continuation was stopped gracefully at 2,334,720
additional steps after its 1.6M checkpoint lost 3–14 in an independent reference
match. Final optimizer state and all earlier checkpoints remain available.
The 1M shot-intent pilot completed: at 400k it passed the short skill/load gates
(97/60/97/100 skill successes per 100) and beat the fixed controller 2–0 over
8 x 60 s. At 600k it lost 4–5, and the final checkpoint did not improve that
reference score. These are small development samples, not held-out superiority.

A harder defense suite now optionally randomizes the initial paddle over the
reachable workspace. The original intercept tolerance saved 269/300; requiring
12 mm predicted center error saved 289/300. Friction-predictor variants did not
improve this. The original 60 m/s² shot controller exceeded modeled load in
15-minute continuous self-play: peak 1.055, score 54–57 across 8 games.
Earlier recovery after outgoing contacts fixes much unnecessary motion: five
minutes of self-play with the original interceptor peaked at 0.935, and with
centered slow-puck interception at 0.933. The latter had 646/665 forward returns
and only 18/22 referee turnovers, so low scoring was not simply inactivity.

A settled 3 m/s shot does not need the full acceleration ceiling: over 300
arbitrary-start fixtures, caps 20/30/40 m/s² gave stationary 297/297/297 and
moving 208/216/217 goals, versus 297/218 for cap 60 on that development seed.
The combined candidate uses cap 30 for arrival shots, cap 10 for setup,
reachable low-cap recovery, and up to 60 when interception needs it. It is saved
as `_foundation-efficient-arrival-controller`, explicitly an experimental
controller bundle over frozen learned skills, not newly learned motor weights.
Longer games in both roles and 900-second self-play are being evaluated.

`_foundation-shot-intent-efficient` continues the 400k tactical checkpoint on
that improved controller for 4M additional transitions. Its body/controller
parameters are serialized with the checkpoint; the fallback policy is bundled.
It starts with fresh replay because the controller changed. Future compatible
continuations retain option replay, target values and optimizer state. Frozen
actor-mean inference avoids unused sampling/log-probability work: measured median
16-environment inference fell from 0.212 ms to 0.097 ms without compilation and
0.079 ms compiled. Compilation produced at most 2.9e-5 normalized-action
roundoff in the probes (well below 0.1 mm target displacement). The new run uses
this optional compiled path; the fixed-controller comparisons keep the original
inference path. No firmware or physical deployment defaults were changed.

### Frozen-candidate qualification (in progress; load margin failed)

The fixed efficient controller won the two development reference matches 21–8
and 24–6 on opposite sides (8 x 180 s each); candidate peak modeled loads were
0.910 and 0.897. Eight 900-second continuous self-play games peaked at 0.884,
with zero overload time and actual acceleration within numerical tolerance of
60 m/s². Those games contained 1,149/1,164 forward returns and 65/57 referee
turnovers; strong defense produced a 1–1 aggregate score.

The 4M-transition tactical continuation completed but did not beat the fixed
controller. Its final arbitrary-start rates were 88/57 versus the fixed
controller's roughly 99/70; it is not selected. All checkpoints, replay and
optimizer/target-network state remain available. The selected experimental
bundle is `_foundation-efficient-arrival-controller`, with frozen neural skills,
30 m/s² arrival strokes, early recovery, centered interception up to 60 m/s²,
and the external motion guard. No physical deployment promotion was made.

The candidate was frozen before opening seeds 20261201 onward. On 500 held-out
trials each it scored 493 stationary and 356 moving goals, cushioned 489 pucks,
and saved 500/500 home-position shots at 2–8 m/s. Random-pose defense saved
472/500 on the next seed. A separate 8–12 m/s challenge (seed 20261207) saved
490/500. An earlier requested 8–14 m/s challenge was capped by the simulator's
12 m/s puck ceiling; that file is explicitly annotated and must not be reported
as a realized 14 m/s test. Benchmark input validation now rejects speeds above
the simulator's actual puck ceiling.

The first held-out full match won 29–8 but reached 0.989 modeled load. There was
no overload, yet this FAILED the predeclared conservative 0.95 promotion margin.
Do not relax that gate after seeing the result. The opposite-role and four
one-hour self-play tests continue unchanged. New experiments use development
seeds, and any replacement candidate requires a fresh held-out suite.

One such experiment adds a broad goal-threat envelope before interception:
commands for clear end-rail misses return home quietly, while any shot that
could enter the mouth under a range of rail damping coefficients retains the
full defender. This is opt-in (`threat_only`) and is being evaluated; it is NOT
part of the frozen candidate. More aggressive arrival power, learned cushioning
before setup, swept-circle clearance and forecasted navigation avoidance did not
improve the selected controller; failed variants are preserved under
`logs/foundation`, not enabled. Contact diagnostics explain the remaining
moving-shot issue: 202/221 first contacts during or shortly after a planned
strike were on goal, while 45 earlier setup contacts were all off goal.

### Explicit thermal reserve candidate

The threat-envelope defender won 18–4 on development seed 20261136, peak load
0.889. An independent paired test on seed 20261137 reproduced the original
controller's margin failure: 15–5, peak 0.955. A simple return-home cooldown at
observed level 0.88 until 0.82 reduced peak to 0.880 and won 11–9, active for
2.08% of play. That probe used `probe_load_budget_v1.py`; its lack of pending-shot
cancellation is corrected in the reusable implementation.

`airhockey/load_budget.py` now cancels a vetoed arrival and retains gentle
8 m/s² goal-threat interception during cooldown. Heat hysteresis persists across
points; the external kinematic guard still governs braking. Tests cover fast
and slow memory triggers, point resets, acceleration encoding, pending-action
cancellation and low-budget threat tracking. This is an observed thermal
reserve, not a hardware safety certification.

`_foundation-budgeted-arrival-controller` bundles the unchanged learned motor
weights with threat-aware centered interception, early recovery, 30 m/s²
arrival shots and the observed 0.88/0.82 reserve. A fresh qualification suite
starts at seed 20261501; the earlier held-out seeds are not reused to claim
qualification. The development match in progress leads 14–0 at 90 seconds;
full forward/reversed results are pending. The earlier candidate's one-hour
self-play is still running, peak 0.902 after 51 simulated minutes, but that
candidate already failed its forward-reference load-margin gate.

The self-play replay label now accepts an explicit match type, so two separately
loaded copies of the same policy are not mislabeled as reference matches.
Previously generated known same-policy recordings were corrected; the first
qualification process predates this metadata fix and its final continuous
self-play recording must be corrected after it finishes. These changes affect
labels only, not simulation results.

### Final candidate validation in progress (2026-09-21)

The previous candidate completed four one-hour games at peak load 0.902,
zero overload, score 3–1, and 2,840/2,817 forward returns. It remains failed
because of its earlier 0.989 reference-match peak. Its continuous replay's
self-play metadata is corrected, and its overview plot explicitly shows failure.

The budgeted candidate won development matches 26–1 and 20–6, peak candidate
load 0.869/0.833. Fresh held-out matches won 17–2 and 31–7, peak 0.811/0.876.
Held-out skills scored 492 stationary goals, 347 moving goals, 486 cushions and
500 home-position saves per 500 trials; random-pose defense saved 479/500.
Extra fresh direct-shot defense at 8–12 m/s saved 491/500. A separate bank-shot
challenge saved 500/500; every incoming bank was verified to score without a
defender under the exact randomized trial physics, with no privileged parameters
given to the policy. Its bank aims were near goal center, not the full mouth.

A hot-start skill diagnostic initialized every fast/slow memory to 0.90 and
saved 473/500 direct shots at 2–8 m/s. Requested cooldown acceleration is 8 m/s²,
but the external guard retains up to 60 m/s² braking authority. A sustained
hot-start reference match and final hour-long qualification remain in progress.
See `ai/POLICY_RESULTS.md` for the concise candidate description and simulation
review command. No physical activation or deployment promotion occurred.

### Preserve urgent defense while cooling

The sustained hot-start probe exposed a real weakness hidden by short defense
trials: the 8 m/s² cooldown candidate lost 3–26 with every fast/slow memory
initialized at 0.90. On identical development seed 20261141, allowing defensive
caps 8/15 lost 6–10, 8/25 lost 1–6, and the normal 8/15/25/40/60 feasibility
selection won 8–1. All four variants stayed below their initial 0.90 peak load.
Suppressing attacks while retaining necessary interception acceleration is the
useful thermal compromise; an unconditional low defensive cap is not.

`LoadBudgetPolicy` now accepts `defense_caps` and a separate observed-load
`defense_ceiling`. Defaults preserve the earlier candidate's 8 m/s² behavior.
The new frozen `_foundation-reserve-arrival-controller` uses all five defensive
caps below 0.91, with gentle-only defense at or above 0.91. Attack suppression
still starts at 0.88 and persists until 0.82. The extra ceiling reserves margin
for quantized 10 Hz telemetry; it is not a certified hardware bound. A test
verifies urgent acceleration is available at 0.90 and unavailable at 0.91.

Qualification v2 adds 0.90 hot-start reference games from both sides, keeps all
previous gates, checks skill load/acceleration too, and freezes runtime metadata,
the fitted thermal model, geometry and host binaries as well as policy/source
hashes. New seeds begin at 20261601. Initial results: 493 stationary goals,
340 moving goals, 486 cushions and 500 home saves per 500 trials; random-pose
defense 467/500. Full reference, hot-start and one-hour self-play cases remain
running. Fourteen focused controller/arrival/thermal tests pass. Under concurrent
evaluation, single-state inference was 1.82 ms median / 5.58 ms p99 ordinarily,
and 2.48 ms / 6.11 ms while hot (maximum observed 6.62 ms).

New replay metadata explicitly identifies both policies and their inference
modes, including swapped-side comparisons. Existing qualification recordings
that predate this metadata addition will be annotated after completion.

### Isolated rebound gains did not justify a new defender

An observed-state controller predicted settled paddle contact normals aimed at
the far goal. On development seed 20261143 it increased random-pose defensive
trials with an on-goal return from 97/300 to 212/300, saves 290 to 291. However,
predicting aim through a future rail bounce regressed fast bank saves from
298/300 to 286/300. Restricting aim to paths with no future bank restored those
saves (and preserved 296/300 fast direct saves), but did not meaningfully improve
full-game returns: 375/795 on goal versus 318/683 for the original defender.
The full match won 22–3 versus baseline 24–1, with higher peak load .889 versus
.820. The nominal restitution .9 versus midpoint .775 ablation also produced
only small, mixed skill changes. None is selected. Prototypes, an extracted
module and its tests, mathematical/equivalence checks, and decision evidence are
archived under `logs/foundation`; no aimed-return controller is enabled.

### Independent fast and slow load reserves

The original budgeted candidate completed its v1 qualification: peak .846,
no overload in four one-hour games. It remains rejected because its supplemental
hot-start match lost 3–26. The reserve-defender successor passed cold matches
14–1 and 13–4, and one hot match 2–1, but lost the opposite hot role 3–4. This
fails v2's predeclared score gate even though all observed loads stayed safe.
Its one-hour run continues for the load record; it is not promoted.

A fast-load cooldown had been latched by a different, slow memory still above
the fast release threshold. In the .90 hot-start probe, this disabled attacks
for the entire three minutes even though short-term load had cooled. The new
opt-in `slow_budget` keeps separate hysteresis states: fast .88/.82, slow
.92/.89, with defensive reserve ceilings .91 fast and .94 slow. Legacy/default
behavior is preserved when `slow_budget` is absent. Both latches persist across
points, and unit tests exercise independent triggering and release.

The dual-memory development candidate won its .90 hot-start match 6–3, peak
.900, with cooldown active only 6.26% of play. A .93 initial-load challenge is
running. `_foundation-dual-load-arrival-controller` is frozen for v3 qualification
on seeds starting 20261701. V3 retains prior gates and additionally requires
98% saves on 500 fast direct and 500 fast bank trials. Bank fixtures are checked
against no-defender rollouts under each trial's actual randomized physics;
those parameters never reach the policy. High-speed reporting now includes
8–10 and 10–12 m/s bins. Physical defaults and hardware remain untouched.

### Selected simulation candidate

`_foundation-dual-load-arrival-controller` passed the complete frozen v3 suite.
On 500 held-out attempts per task: stationary goals 497, moving goals 339,
cushioning 488, home defense 500, random-pose defense 478, fast direct defense
494, and fast single-bank defense 497. Both ordinary reference roles won
(25–4 and 30–7), as did both .90 hot-start roles (3–1 and 7–1). Peak candidate
loads were .862/.842 cold and .904/.900 hot. The separate .93 development
stress test won 10–3 without increasing its starting peak load.

Four continuous one-hour self-play games finished at peak load .860, zero
overload time, peak actual acceleration 60.00094 m/s² (numerical tolerance),
and maximum speed 5.740 m/s. The aggregate score was 2–3. Eleven unresolved
guard forecasts were logged, without a measured acceleration-limit violation;
these remain part of the deployment limitations. In ordinary reference games,
the candidate spent 2.79% of time above 40 m/s² versus the reference's 20.34%.

The checkpoint, run configuration, control/physics sources and host libraries
were verified unchanged through qualification. Twenty-one focused regression
tests and lint checks pass. The evaluation loader now automatically retains
the saved motion-guard requirement; all qualification calls had already enabled
that guard explicitly. New progress logs show each side's load separately.

Selection is recorded in the run's `decision.json`; the untouched `run.json`
still has `deployment_ready: false`. This is a validated simulation improvement,
not proof of real-drive safety or superhuman performance against humans. Moving
puck setup remains the main weakness. The concise report and review command
are in `ai/POLICY_RESULTS.md`; the qualification directory contains raw metrics,
frozen source/configuration snapshots, verification evidence and `overview.png`.
No physical hardware was activated, and no deployment defaults were changed.
