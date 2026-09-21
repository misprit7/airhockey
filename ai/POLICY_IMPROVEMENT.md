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
