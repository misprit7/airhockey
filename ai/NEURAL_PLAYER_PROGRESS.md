# Neural-only restart — September 21/22

## September 28: goals-only branch closed; v3 retained

The user rejected the goals-only experiment and requested returning to
`neural:possession-20260926-v3`. No goals-only trainer or evaluator remains
running. Do not resume that branch or run further candidate audits unless
requested. The experiment remains archived; physical deployment defaults and
hardware were not changed. The training explainer has been updated for v3 at
`ai/airhockey/web/training-explainer.html`.

## September 27: goals-only self-play experiment (closed)

User requested restarting training from the current policy with every auxiliary
reward removed. `_neural-player-goals-only-20260927` starts from packaged
`possession-20260926-v3/agent.pt`, also confirmed in the latest physical-session
log. All 12 actor state tensors (including exploration and encoding) match
exactly at initialization; value network and Adam state are fresh, run step
starts at zero, and no actor layers are frozen. The opponent pool starts with
the same actor and then adds this run's snapshots; no external opponents train
it. Shot-request inputs remain compatible, without any reward for obeying them.

Explicit goals-only mode returns exactly +1 for real goals scored, -1 for
conceded, zero otherwise. It bypasses ALL shaping including hard-coded time,
action-change, passive-return, potential and thermal terms. No imitation,
request-consistency or entropy bonus. Full games only, 900s truncation with
ordinary bootstrapping, continuous reachable possessions, no auxiliary drill
or overload terminations. Physical caps remain 12m/s and60m/s²; sensing/decoder
and load model remain, with load measured but not penalized. This experiment
may lose learned thermal restraint; it is simulation-only, not a deployment
replacement or automatic promotion.

128M steps /180minute ceiling,512 environments across4 workers,256-step rollouts,
lr1e-5/value-lr1e-4,gamma.9995,GAE.98,KLbacktracking.01. Checkpoints every8M.
Each gets8×120s self-play replay and8×120s matchups against sourcev3 in both
colors. Dashboard http://localhost:8420/training; live replay alias
http://localhost:8420/?replay=neural-goals-only-20260927.json . Initial alias is
the unchanged starting actor until the first trained checkpoint is evaluated.
Commands, initialization hash/checks, metrics and evaluator are under
`logs/neural-player/requests/goals-only-20260927/`.4096-transition smoke passed;
41reward/actor/dashboard tests passed before launch. No hardware activated.

User rejected the scripted hybrid. New objective: a single learned neural player
that receives and controls the puck, sets up varied accurate powerful shots, and
defends quickly while respecting instantaneous and sustained physical limits.
No tactical controller may be substituted for unsuccessful neural training.

Work started 2026-09-22 02:12 UTC; approximate eight-hour window ends 10:12 UTC.
Simulation only. No physical hardware activation or deployment-default changes.

## Architecture

Fresh random 3×256 ELU neural trunk, six arrival-action outputs, a training-only
value head, and learned exploration variance. No old checkpoint weights or
encoder transfer. Physical observations and prior commands/load estimates only;
no drill, tactical phase, shot request, or privileged simulator state inputs.
Relative positions/velocities are differences of those same observations.
Optional four-frame history supplies only earlier physical observations to the
same actor; it introduces no tactical phase or requested shot.

The arrival decoder translates network outputs to firmware position/acceleration
commands. The kinematic guard protects physical limits. No interception, shot
setup, recovery, or thermal-cooldown tactical override is attached to this policy.
RMS proxy load is observed and penalized, not used to script tactical choices.

## Training and evaluation

`ai/bin/train_neural_player.py` runs PPO directly on the real simulation dynamics.
`ai/airhockey/neural_training.py` supplies curriculum initial states and reward
shaping. The reward includes real goals, on-target shot direction/speed, brief
controlled possession and its subsequent conversion into a shot, missed
opportunities, effort/load, and a small action-change cost. Position shaping is
a discounted potential difference; it does not alter commands.

Stage 0 starts with accessible stationary pucks and slow incoming pucks. Later
stages widen starts and incoming speeds, add defense, then mix complete games
with practice. Stage changes are made after evaluation, not automatically on a
fixed step count. Full-game self-play records both sides using the same neural
checkpoint. Learned shot diversity remains something to establish, not an
assumption from stochastic training.

`ai/bin/eval_neural_player.py` measures contacts, control, control-to-shot,
shot direction and speed, goals, passive returns and load. Skill trials include
random paddle starts; full replay clips cover complete requested durations.
Passing isolated shot tests is not sufficient for selection.

## Current work (updated about 09:04 UTC)

Eight-hour optimization remains active; approximate end 10:12 UTC. Training
freezes by about 09:15, followed by fresh-seed qualification and full replays.
No final candidate is selected yet. No hardware activation or deployment changes.

- Precision13 and league9 finish around 09:04. Constrained12 finishes around
  09:11; offense14 around 09:14. All are one-actor policies with physical inputs.
- Precision400 has the best recent cold reception/accuracy balance: ~74% visits
  controlled, ~97.6% prepared shots on target at 7.06 m/s. Precision417 trades
  some reception accuracy for speed (~69% control, 96.8% prepared, 7.35 m/s).
- Offense400 combines ~68% control, 96.9% prepared accuracy, 7.5 m/s and a 6–0
  result against the load-tested league352 reference in both colors. Offense417
  edged balanced303 3–2, but cold control/accuracy fell (~59% /87.2%). Its short
  hot test recovered ~71% control and96.8%prepared accuracy at7.4m/s, nooverload.
- Hour-long hot tests of precision400 and offense400 are pending. Many earlier
  candidates failed either sustained load or accuracy while hot; both matter.
- Development fallback is league352 mean: 8×3600s hot.95/gain1.3 passed with
  peak.9745, nooverload, ~94%preparedaccuracy. Its sampled policy failed load.
  Independent8×600s firmwareaudit passed, exactreproduction, peak60.000412m/s²,
  minimumworkspace margin.950mm, zeroover-cap intervals. Finalfreshseeds pending.
- New final seeds and the qualification matrix are in
  `logs/neural-player/final-evaluation-plan.json`. All development seeds have
  been used for selection and must not be described as held-out tests.


## Iteration log

- Added four persistent simulator workers. Measured smoke throughput increased
  from about 17k to 53k transitions/s (about 34k for a continuation while other
  training/evaluation was also active). One GPU network learns from all workers.
- At 4M scratch transitions: broad random-start stationary goals 42/128, moving
  25/128, receiving contacts 128/128 but only 5 controlled receptions and zero
  control-to-shot conversions. Full 8×120 s self-play also had zero conversions.
  This is useful early contact learning, not acceptable completed play.
- `_neural-player-possession-stage1` broadened starts/speeds from the fresh
  network's 4,063,232-step checkpoint. Stopped and saved at 7,864,320 steps to
  fix contact accounting: a blanket 200 ms debounce could suppress a strike
  immediately following cushioning. A focused regression test now covers this.
- `_neural-player-possession-stage1b` resumes that saved model/optimizer, with
  stronger control/conversion rewards (10/8); 24M additional transitions planned.
- `_neural-player-capture-scratch` is a second fresh random initialization,
  emphasizing incoming-puck exercises (70%), control reward 20 and conversion
  reward 10. Immediate incoming volleys receive goal credit but not an additional
  setup-shot bonus. This tests whether learning reception first avoids the
  easy-volley strategy acquired by the initial curriculum. No action override.
- Evaluation metrics version 2 includes the immediate controlled-strike fix;
  old reports remain for traceability. Compare checkpoints under the same metric
  version before drawing conclusions. Four focused tests currently pass.
- `_neural-player-receive-scratch` is a third fresh neural initialization with
  a focused reception curriculum: 70% fixed environment slots end when a
  controlled reception succeeds or after 1.2 s; 20% continue shooting practice,
  10% defense. This is only training episode/reward structure. No inference mode
  switch or task input. At about 3.8M transitions its stochastic training
  reception rate reached 64%; this is not a held-out or full-game success claim.
- At about 10M transitions the capture-first full-sequence branch reached 58%
  captures and 14% control-to-shot conversions per receiving training episode.
  Evaluating deterministic play and broader starts before treating that as
  useful possession play.
- Stage 0 finished at 8,028,160 transitions; checkpoint/optimizer retained.
  Eight focused tests (including PPO return semantics) pass. Initial source
  snapshots remain with each run; later reward/metric changes are new runs.
- At 15,073,280 capture-branch transitions, a separate 256-trial evaluation
  controlled 245/256 incoming pucks, converted control into on-target launches
  in 120/256, and saved 252/256 direct 2–8 m/s defense trials. Broad random-start
  shooting remained poor (~34% goals), and 8×120 s self-play on-target fractions
  were only about 25%, with ~2.4 m/s mean launches. These are development seeds,
  not final held-out qualification. Motor load approached 0.95 in self-play.
- `_neural-player-control-strike-stage2` resumes the 15M capture checkpoint with
  broader random starts and stronger rewards for accurate fast shots; control
  reward 5, conversion 15 (partly scaled by shot speed), power reward 8, miss
  penalty 4. Adds learned state-dependent exploration variance with an initially
  zero head, preserving the previous policy mean. Optimizer freshly initialized
  for the new head; all learned actor weights retained. 40M additional planned.
- `_neural-player-control-strike-stage2b` is the same continuation with an
  independent training-only value tower and separate gradient clipping so value
  fitting cannot displace the actor's precise motor representation. Inference
  still uses one actor MLP; no value search or tactical overrides. Older means
  and initial value predictions are preserved during migration.
- Added full-game counters for turnovers, launch-speed bins, goal-mouth thirds,
  and completed own-half visits with contact/control/accurate/powerful shots.
  Workspace-opportunity counts only describe geometric contact space for at least
  100 ms; they do not claim dynamic interception feasibility.
- Prepared competitive self-play rewards: subtract the opponent's skill-event
  bonuses, preventing mutual reward farming through gentle volleys; real goals
  are worth ±25. Training can use a pool of neural snapshots. Also supports a
  longer discount horizon and simulated-overload terminal failures with explicit
  thermal reinitialization only after such failures. Ordinary game/point resets
  retain heat. Evaluation never uses this failure reset and continues logging
  overloads. Ten focused checks pass including thermal-reset semantics.
- First three exploratory branches completed: `_neural-player-possession-stage1b`
  at 31,916,032 total transitions (strong easy volley shooting but no possession,
  rejected as an overall candidate), `_neural-player-capture-scratch` at 24,051,712,
  and the reception-only branch around 30M. Checkpoints and optimizers retained.

## Selection requirements

No candidate is accepted yet. Selection must inspect full self-play plus
physical outcomes: reachable incoming pucks contacted, real controlled
receptions, subsequent on-target launches, shot speed, missed swings, missed
blocks, passive rebounds, turnovers, and sustained motor load. A high isolated
goal score, a favorable matchup against the old agent, or low jitter alone is
insufficient. Keep failed experiments and report unmet objectives honestly.
- Evaluation version 3 excludes slow pucks outside reachable paddle contact
  space from possession. Old control counts could reward losing a puck behind
  the paddle workspace; reward and metrics now agree on this stricter meaning.
  Stationary practice also spans the reachable depth, including near the back.
- Stage2b at 39,452,672: 247/256 broad stationary goals, 230/256 moving goals,
  247/256 controlled receptions with 221 control-to-shot conversions, and
  256/256 direct 2–8 m/s saves. Mean stationary launch speed 5.13 m/s. However,
  8×180 s self-play had only ~29% on-target launches, ~3.5 m/s mean launches,
  and severe sustained overload (peak modeled load 1.29). NOT selected.
- Learned Gaussian sampling on that checkpoint worsened load (peak 1.43) and
  did not improve shot accuracy. Deterministic mean remains the default.
- `_neural-player-control-strike-stage2c` continues from stage2b 39M with the
  corrected possession reward and broad stationary depth; 40M additional planned.
- `_neural-player-selfplay-stage3b` continues neural self-play with preserved
  heat, warm starts, and expensive overload failures. At 40,763,392 transitions,
  8×180 s deterministic self-play had no modeled overload (peak 0.975), but
  mean shots were only 2.7–2.9 m/s and 135 total turnovers occurred. This is
  learned load reduction, not adequate overall play. Skill shooting regressed
  versus the drill-only branch. No candidate selected.
- The stage2b 39M bank-shot challenge saved only 71/256 verified goal-bound
  8–12 m/s bank launches. This exposed a major defense curriculum gap.
- `_neural-player-banks-selfplay-stage4` started 03:05 UTC from stage2c 49,676,288;
  160M additional planned. Half of environment slots are full games against a
  pool of neural snapshots plus a fixed neural volley opponent. Practice includes
  2–12 m/s direct defense and 8–12 m/s banks, with fixed practice slots so short
  failed defensive episodes do not disappear from transition-weighted training.
- Added evaluation shot-event and turnover locations to diagnose full-game
  transfer failures. Development seeds are still 20261901/20261902; final
  qualification must use fresh seeds after freezing any candidate.
- At stage2c 65,011,712, ordinary drills reached 256/256 stationary goals and
  256/256 receiving goals, with 251 controlled receptions and 251 conversions.
  Full self-play still had only 26.5% on-target launches, 22.2% on-target shots
  after control, and peak load 1.34. These scores exposed a context shortcut.
- Opponent-position ablation: changing only the practice opponent's initial
  position from its fixed corner to varied legal workspace positions reduced
  stage2c stationary on-target fraction from 99.2% to 30.0%, receiving from
  100% to 27.9%. Blocks can affect later events; first-launch metrics were added
  to isolate pre-opponent-contact accuracy in future comparisons.
- Added optional randomized physical opponent positions in practice, with
  synchronized dynamics and camera histories. This removes a constant practice
  context cue; it changes initial states, not policy outputs. Full games remain
  against neural policies. Existing 11 focused tests pass.
- Stopped the bank baseline gracefully at 65,536,000. New continuations:
  `_neural-player-generalize-selfplay-stage4` (160M additional, same rewards and
  bank/full-game mix) and `_neural-player-generalize-skills-stage2` (60M additional
  from stage2c 65,011,712). Both randomize practice opponent positions. Comparing
  skill-only transfer with concurrent self-play; neither is selected yet.
- First-launch ablation confirms the shortcut before any opposing contact:
  stationary 256/256 on target in fixed-context practice versus 74/246 with
  randomized opponent positions; receiving 254/256 versus 90/247.
- Diagnostic `_neural-player-context-ablation` algebraically folds the old
  practice opponent constant (workspace min-x, y=1.8, zero velocity) into the
  first-layer biases, retaining the own-state contribution of relative inputs.
  Opponent input weights start at zero but remain fully trainable. Verified
  identical means, variances and values to the source network evaluated at that
  context. This is a neural initialization experiment, NOT a tactical controller
  or selected policy. Fresh optimizer; no hard-coded action decisions.
- Diagnostic self-play (8×180 s): blue completed visits 989, contacts 980,
  controlled 541, on-target 809, on-target >=4 m/s 739; red 986/979/521/793/738.
  Mean launches 5.67/5.55 m/s, eight total turnovers. Random-opponent stationary
  first-shot accuracy 255/255 and receiving 255/256. BUT peak modeled load 1.39,
  so this initializer fails physical-load qualification and is not selected.
- Stopped the earlier fixed-context self-play baseline gracefully around 70M.
  `_neural-player-generalize-selfplay-stage4b` continues from the diagnostic with
  all inputs trainable, randomized practice positions, fast banks, neural
  self-play, load failures and the same outcome rewards. Lower learning rate
  1e-4 and entropy 1e-4; 240M additional planned. Initial checkpoint is 65,011,712
  learned transitions, not a new random model. It still descends entirely from
  this night's fresh neural initialization. Automatic development evaluations
  every ~16M saved transitions; no final candidate selected.
- Curriculum-only stage2 generalization at 97,517,568 independently recovered
  transfer without the diagnostic weight transformation. In 8×180 s self-play,
  blue entries 786/781 contacted/656 controlled/681 aimed/669 aimed >=4 m/s;
  red 789/778/650/700/681. Mean launches ~5.49 m/s, five total turnovers.
  Random-opponent first stationary shots 248/256 aimed, receiving 254/256.
  Peak modeled load 1.21 and fast-bank saves 125/256: still not qualified.
- Generalize stage4 at 81,788,928 saved 209/256 fast banks and peak self-play
  load was 0.915, but only ~23% of visits produced an aimed shot and there were
  208 turnovers. Continue as a load/defense baseline, not a selected candidate.
- Stage4b 73,138,176 warm-start (all load channels initially 0.9), 8×300 s:
  ~81% of visits produced aimed shots, mean launches 6.54 m/s, peak load 1.20.
  Direct 8–12 m/s saves 242/256. Improved load but still fails RMS qualification.
- Stopped the skill-only generalization branch after its ~100M milestone and
  launched `_neural-player-generalize-selfplay-stage4c` from its evaluated
  97,517,568 checkpoint. Same randomized-opponent/bank/game curriculum, load
  weight 1 and configurable shutdown penalty 500 (versus 100). 160M additional
  planned. This branch is entirely ordinary learned-weight continuation, with
  no context-ablation initialization. Existing branches continue for comparison.
- Continued stage4 learning is helping: at 98,041,856, fast-bank saves 222/256,
  self-play aimed visits 314/510 blue and 308/515 red, ~5.24 m/s mean launches,
  peak load 0.964, 61 turnovers. Better than its 81M checkpoint, still incomplete.
- Stage4b 81,264,640 retained strong shooting (~76% aimed visits, 6.7 m/s) but
  peak load 1.187 and fast-bank saves only 114/256. Continuing, not selected.
- Skill-only generalization stopped at 101,842,944. Its last evaluation had
  820/872 and 817/869 aimed visits, ~5.27 m/s launches, just two turnovers,
  but peak load 1.269 and bank saves 115/256. Strong motor skill, inadequate
  sustained load and difficult defense.
- Added independent actor/value learning rates with exact legacy Adam-moment
  migration, and an explicit initial exploration-variance scale. Verified both
  optimizer layouts retain moments and changing variance preserves action means.
- `_neural-player-generalize-selfplay-stage4d` starts from the 101M skill-only
  checkpoint, actor LR 1e-4, value LR 5e-4, initial Gaussian std ×0.35, entropy
  1e-4, load weight 1, shutdown penalty 500. All parameters remain trainable;
  no inference controller changes. 160M additional planned. Four concurrent
  simulation jobs now compare fine-tuning approaches, with periodic replays.
- Added nine training-only critic columns: exercise one-hot, skill-episode time,
  and both sides' possession/reward-paid flags. The actor and exploration head
  still receive exactly 42 physical features. This separates reward-context
  prediction from the actor's required generalization. Old critic input weights
  expand with zero columns; old actor outputs remain unchanged. Added tests for
  no actor leakage/gradients and exact zero-context value migration; 13 focused
  tests pass. Parallel PPO smoke completed successfully, not a candidate.
- Development performance is not monotonic: stage4 at 114,294,784 regressed to
  185/256 bank saves, ~52% aimed visits and peak load 0.993. Its evaluated 98M
  checkpoint remains the stronger load/defense baseline. An 8×3600 s warm-start
  (0.9) continuous-load test of that 98M checkpoint is running.
- Stopped the original stage4 baseline around 118M and launched stage4e from
  its evaluated 98,041,856 checkpoint with the new critic context, fresh
  optimizer for the expanded critic, actor LR 1e-4/value LR 5e-4, initial std
  ×0.7. Original reward weights retained (load .75, shutdown 100). 160M additional
  planned. Stage4b/c/d remain active comparisons. None is a final candidate.
- Prepared configurable real-goal/concession/turnover weights, optional neural
  opponents in practice episodes, reward scaling, and GAE lambda for a later
  competitive curriculum. Current jobs retain their original settings. Simulation
  smoke verifies practice opponents use external neural actions with unchanged
  42-wide actor inputs; 13 focused tests still pass.
- Ideal-sensing defense diagnostic initially used differently constructed
  environments, which changed randomized physics draws. Do NOT treat those
  unpaired reports as a causal sensing comparison. The corrected version first
  initializes identical realistic environments, then disables sensing delay/noise
  only for the diagnostic; metadata sensing_ablation_version=2. Selection and
  automatic development tests remain realistic.
- Hour-long warm qualification of stage4 98,041,856 completed: eight independent
  3600 s self-play games, both bodies initialized at load 0.9, heat continuous.
  Zero modeled overload; peak 0.97307. 18,620 completed own-half visits,
  11,326 with aimed launches (~60.8%), mean launch ~5.17 m/s. 1,785 turnovers
  across 960 player-minutes and 198 total goals. Goal thirds strongly favor
  left/center, so offense/variety remain incomplete. This is a development
  baseline, not final held-out qualification or physical deployment approval.
- The hour report logged 54 unresolved predictive-guard decisions. Physics-tick
  average acceleration remained ~60.00057, but that could miss smaller firmware
  integration spikes. Added an optional evaluation-only audit that reproduces
  every 2.5 ms interval on a separate firmware state copy and measures each
  internal integration step. Two-second smoke reproduced all state components
  exactly with no cap violations. A 16×600 s warm audit is now running.
- Matched sensing ablation on stage4 98M, 512 verified banks: realistic saves
  442/512; ideal sensing on identical physical draws 460/512. Sensing contributes,
  but substantial room remains for policy improvement. Do not use the earlier
  unpaired ideal report (469/512) for this comparison.
- Stage4e 114,294,784: fast-bank saves 227/256, aimed visits ~55%, mean shots
  ~5.5 m/s, peak modeled load 0.952, 64 turnovers in 8×180 s. Stage4d 134M
  achieved peak 0.926 but stalled (188 turnovers). Stage4c 130M restored offense
  (~69% aimed visits, ~6 m/s) but slightly exceeded load (1.008).
- Retired stage4b around 124M: load plateaued near 1.16, bank saves 192/256 at
  its 113M evaluation. It is not selected; checkpoints retained.
- `_neural-player-competitive-stage5` starts from evaluated stage4e 114,294,784.
  Same single actor, no tactical inputs or action overrides. Half of practice
  episodes now face neural opponents; full games still use a neural snapshot
  pool. Real game goals ±100, skill goals +20/concessions -100, own turnovers
  -100, overload failure -1000, load weight 1; skill shaping retained. GAE lambda
  .98, gamma .999, actor LR 1e-4/value LR 5e-4. 160M additional planned. This
  tests whether active defense and stronger real outcomes improve offense
  without making overload or stalling attractive. Other baselines c/d/e continue.
- Correction to stage4b retirement: it had already failed at step 118,358,016
  during rollout with non-finite arrival actions; last saved 113,770,496 model
  is finite. No diagnostic state was available to distinguish simulator inputs
  from an optimizer issue. Marked status failed/not running. It is not resumed.
- Added explicit observation, policy/value output, loss and gradient finite
  checks; bounded extreme PPO likelihood ratios; failure-only debug checkpoints
  and failing input arrays; accurate failed status; worker cleanup on errors.
  Failures are not repaired by replacing NaNs with fabricated observations or
  actions. Existing c/d/e processes retain their loaded earlier code; the new
  competitive run includes these diagnostics. Main training of stage5 actually
  launched after these checks, around 04:20 UTC.
- Firmware-rate audit completed for stage4 98M: 16×600 s warm self-play,
  zero internal-step acceleration violations (max ~60.00074 m/s², numerical
  rounding), minimum workspace margin 0.941 mm, exact reproduction of all six
  firmware state components. No hidden boundary impulse was found. Peak modeled
  load 0.98751 with zero overload. Unresolved guard flags were conservative
  prediction failures, not observed cap violations in this audit.
- Evaluation stdout now omits per-event lists; full shot/turnover events remain
  in the JSON report. This avoids enormous duplicate logs during long tests.
- Training-status caveat: old by-kind summaries selected from a shared last-2000
  episode window. Synchronized 30 s games can fall out of that window between
  batches, leaving only a few early shutdowns and biased game means. Actual PPO
  transition allocation remains 50% game slots; evaluations are unaffected.
  Future trainer processes keep separate per-kind episode windows and report
  sample counts. Do not select checkpoints from the old training means.
- Added explicit neural-vs-neural cross-play evaluation with candidate side
  swapping and honest replay metadata. Reference sampling remains deterministic;
  candidate sampling is configurable. Self-play defaults remain identical neural
  weights on both sides. Cross-play smoke passed. Warm 8×300 s matches in both
  colors are running for stage4c 146M and stage4e 130M against the load-qualified
  98M baseline. A stochastic warm self-play test of stage4e 130M is also running
  to assess learned shot variation without introducing a tactical randomizer.
- Warm cross-play against stage4 98M (8×300 s in each color): stage4c 146M
  won 90–37 aggregate, candidate peak load <=0.953; stage4e 130M won 96–10,
  candidate peak <=0.9944. These are development-seed matchups, not final
  qualification. They support real playing improvement beyond same-policy rallies.
- Stage4e 130M stochastic warm self-play: ~58% aimed visits, ~5.8 m/s mean
  launches, peak load 0.9799, goal-third counts blue 75/163/55 and red 81/172/61.
  Mean-policy and stochastic first-game traces had similar waiting motion:
  far-half mean speed .528/.540 m/s; >40 m/s² duty ~1.69%/1.94%, much below
  the original context-ablation initializer's ~12.4% far-half duty. No explicit
  tactical cooldown is used. Hour-long warm tests of both sampling modes are
  running; the deterministic one includes the firmware-rate audit.

- Around 04:47 UTC, retired stage4c at 180,879,360 transitions to give the
  control-focused stage5b continuation compute. Stage4d 166,854,656 achieves
  1,118/1,126 contacted visits, 893 controlled visits, and 1,025 aimed visits
  in 8×180 s self-play; ~5.49 m/s shots. Controlled shots 871/878 aimed.
  Nevertheless its trajectories are repetitive and goal attempts center/left
  biased, and load peaks at 1.0204. It is a training initializer, NOT a finalist.
  Visual inspection: logs/neural-player/control4d166-filmstrip.png.
- Stage4e 130M beats the previously load-qualified 98M baseline 96–10 with
  mean actions and 64–12 with sampled actions across both colors (8×300 s
  each color). However BOTH modes fail longer warm-load self-play: mean
  peak ≥1.0133; sampled completed 8×3600 s with peak 1.00826 and 2.5 summed
  player-seconds above limit. Short-game pass is insufficient.
- Added configurable training shutdown margin and practice defense fraction.
  Evaluation remains at the actual modeled limit; reported overload is not
  relabeled as shutdown margin. Regression suite: 14 neural/on-policy tests
  pass, including terminal margin vs actual overload distinction.
- Automatic development evaluation now covers both mean and learned sampled
  actions at ~16M intervals; report names include _sampled_ where applicable.
  Sampling never uses a hand-selected tactical target or hand-scaled runtime noise.

- Around 04:56 UTC: found a directional bank-curriculum bug. Benchmark launches
  alternate walls inside a batch; partial training resets often contain just
  one bank episode, which always selected the first wall. Training now randomly
  reflects bank launches, preserving balance for singleton resets. Benchmark
  fixtures are unchanged. Regression confirms both directions across 40
  singleton resets; 11 neural tests pass. Stage5b is continued as stage5c with
  this correction and the same reward/load settings, preserving weights.
- 1,024-bank development breakdown: stage4d166 saves 834 (114 untouched,
  76 contacted but conceded); stage4e146 saves 899 (85 untouched, 40 contacted
  but conceded). Failure rates depend strongly on initial horizontal direction.
  Per-trial diagnostic JSON now records initial states, first contact time and
  outcomes when --skill-details is supplied. No evaluation behavior changes.
- Stage4e130 mean full warm one-hour test completed: peak 1.01373, 9.14 summed
  player-seconds overload. Both mean and sampled modes fail the long load test.

- Sustained 8×1800s warm tests completed: stage4d174 sampled FAILS (peak
  1.02473, 10.88 summed overload seconds); stage4c180 mean passed this cohort
  (peak .98264, zero overload). These used unseeded base physics (pre-v4),
  though they are genuine observed outcomes.
- Seeded v4 development reruns (before stable decoder change): baseline98
  banks227/256, 62% control/61% aimed visits, ~5.2m/s, peak .9521; 4d186
  sampled banks205, 95% control/98% aimed, ~5m/s, peak .9964, just 1 goal
  across 8×180s. Stage4e164 mean banks226, 55% control/60% aimed, ~6.4m/s,
  peak .9629, 34 goals. Stage4c180 mean banks207, ~66% control/65% aimed,
  ~5.9m/s, peak .9736. Trade-off still unresolved; no finalist selected.

- Around 05:10 UTC: control5c184 sampled loses 19–507 against stronger 4g173
  mean across both colors (8×300s each), despite ~97% control in its own
  slow-shot self-play. Control falls to ~35% against faster incoming shots.
  Scorer itself OVERLOADS (peak1.08/1.05), so this is a robustness diagnosis,
  not qualification of that opponent as physically acceptable.
- Retired control6-lowpenalty after saving to free GPU/CPU for attack7, which
  starts from the physically conservative learned catcher. Extended receiving
  curriculum samples speeds up to8 with initial travel distance giving it time
  to observe/accelerate. No runtime intercept suggestions. Configurable reward
  speed saturation and stochastic neural snapshot opponents added; defaults
  preserve old training. Controlled 7m/s strike earns more than5m/s in smoke test.
- New optional --receiving-min/max evaluation: initial receiving puck y1–1.25m,
  vx±1m/s, broad6–8m/s downward speed, paddle near extrapolated contact line
  with y.25–.45. Control6 192M contacts512/512, controls243/512; 4g173 contacts
  511/512, controls240/512. Neither has mastered fast reception yet. Reports:
  fast-receiving-control192.json and fast-receiving-scorer173.json.

- 05:23 UTC performance improvement: host-only motion_limits.cpp exits a
  constant-command prediction once all six state floats exactly repeat.
  All future deterministic steps are then the same fixed point. Compared
  old/new DSOs on 102,400 randomized carts: bitwise identical final state,
  peak acceleration, and minimum workspace margin. Full seeded neural env
  benchmark (256 envs×200steps,3 alternating reps) identical rollout SHA256,
  median 5.987s→4.944s (1.211× throughput). Installed source/library atomically;
  current loaded trainers retain old DSO until next continuation. Firmware law
  and hardware binaries untouched. 29 neural/arrival/motion-fidelity tests pass.
  Reports: motion-predictor-optimization.json and motion-predictor-end-to-end.json.
- Fast receiving monitors are active as airhockey-watch-fast-{control6,
  generalize6,competitive6,attack7}; outputs end -fast-skills.json, mean
  policy,256 per task,6–8m/s receiving. Standard mean/sampled full selfplay
  monitors remain active. Plot script defaults to reproducible metrics v4;
  historical v3 can be plotted explicitly with --metrics-version 3.

- Attack7 first16M continuation (209,190,912) still shows the local-optimum
  problem: mean self-play ~99% controlled/aimed but ~4.93m/s shots and zero
  goals over8×180s; sampled mode92%controlled/87%aimed,~4.8m/s,2goals,peak.950.
  Fast receive129/256; bank230/256 in fast skills. Needs longer learning;
  balanced8 approaches the same objective from the scorer rather than catcher.
- UI now supports /?replay=<filename.json> for an exact recording. It opens
  directly in replay mode. Browser verified9000frames, correct file, slider
  seeking, no JS errors, and ZERO websocket commands with a mocked socket;
  all non-recording API requests were intercepted. No hardware connection
  was made for the test. Desktop/mobile captures saved in logs/neural-player/.
- Added episode duration and goals/conceded per minute to new trainer logs
  so early shutdown episodes cannot make episode-average goals misleading.
- Paired v4 sensing ablations now running for control192 and scorer189,1024
  fast receptions/banks each, real vs ideal sensing. Earlier ablation was
  invalidly paired due omitted base seed; use only new paired-v4-fast reports.

- 05:44 UTC: attack7 225,443,840 now ~95% control/aimed visits,~5.74m/s
  mean shots, bank232/256, short self-play load .936. Sampled~92%control/93%
  aimed,~5.62m/s,bank231,load.973. Still heavily center-biased and0/1total
  goals in mean/sampled self-play. Against fixed 4g173 mean, sampled attack225
  improves from predecessor19–507 to12–28 across both colors,8×300s each.
  Nonetheless its peak load hits1.027 in cross-play; long warm sampled selfplay
  already fails with peak≥1.011. Do not qualify it from short metrics.
- Balanced8 first16M (205,914,112): mean bank237/256,55%control/51%aimed
  visits,~6.89m/s,peak.9621,58goals; sampled bank233,51%control/49%aimed,
  ~6.5m/s,peak.9837,49goals. Needs longer learning of control.
- Properly paired1024-trial sensing diagnosis: scorer189 bank saves913real
  vs950ideal (+3.61pp,paired95%halfwidth1.91pp); control192853vs932
  (+7.71pp,halfwidth2.29pp). Fast reception control worsens under ideal inputs:
  scorer532→413,catcher497→377. These learned policies compensate for their
  training sensing distribution; ideal ablation is not an oracle performance
  ceiling. Keep all training/qualification realistically sensed.
- Decision: allow current four jobs a long uninterrupted stretch. Repeated
  early restarts obscure convergence and reset slow thermal histories. The
  attack branch is visibly improving after32M, so do not replace it prematurely.
  If offensive setup remains stuck after~80M, possible future experiments
  include properly modeled temporally correlated exploration or distillation
  from learned successful trajectories. Neither is implemented; no tactical
  teacher/controllers are part of any current policy.

- 05:52 UTC added a bounded20M/20min fast-reception curriculum alongside the
  four uninterrupted full-game jobs (two workers/512envs):
  `airhockey-neural-player-fastreceive2`, `_neural-player-fast-receive-stage2`,
  from scorer205,914,112. Stage2 receive_drill70% receiving/20%stationary
  shots/10%defense, incoming up to8m/s, receiving ends at successful control
  or1.2s. Capture12, shooting retained (speed reward scales8/7), warm.95,
  shutdown margin.97/penalty1000. This is outcome-based exercise selection,
  not a runtime controller. Fast skill watcher every4M; full mean replay
  watcher every8M. Unit has20min cap; let it finish then evaluate transfer.
- Shot event breakdown: generalize6 205M prepared shots438/458 on target
  (95.6%), mean8.07m/s. Its unprepared returns84/234 on target,mean4.72m/s.
  Attack225 prepared730/740 aimed,mean5.76m/s. Balanced8 first205M prepared
  431/482 aimed,mean7.97m/s. Distinguish shot accuracy from failure to first
  control incoming puck. Evaluator now includes shot_breakdown by control.
- Attack225 warm sampled8×1800s completed FAIL: peak1.02482,31.54 summed
  player-seconds overload. Do not select it.

- Added repeatable --opponent-reference to future PPO runs. Fixed25% slots
  cycle through several frozen learned styles; remaining75% use recent neural
  snapshots. Existing --opponent-checkpoint stays supported as first reference.
  Successful512-transition GPU smoke with3 distinct references, no hardware.
  continue_run.py now correctly expands list-valued CLI options. No active
  long run was restarted for this change.
- Suggested next phase after the bounded fast-reception run completes (~06:12)
  and current30min monitor ends (~06:17): evaluate its improved catches AND
  preserved offense, then continue a strong checkpoint against a broader fixed
  neural league (fast scorer, catcher, older baseline), with realistic full
  games and practice. Consider stronger training load margin/penalty since
  current .97/1000 still permits rare true overload. Do not launch blindly;
  inspect actual fastreceive outputs/transfer first. Keep main long runs
  learning; avoid repeated early restarts of thermal histories.

- 06:16 UTC: competitive6 retired cleanly at232,914,944 after aimed-visit
  rate fell to~30%. Short fast-receive curriculum completed225,968,128:
  fast catches137/256 vs139/256 parent, no reliable gain; final self-play
  67% controlled/54% aimed. Its222,167,040 intermediate transfers better:
  70.5% controlled/69.2% aimed,~5.85m/s,bank239/256,peak.962,30goals.
- Started `_neural-player-league-stage9` from fastreceive222 checkpoint,
  220M additional/180min.1024env4workers, stage5, actorLR1e-4/value1e-3,
  capture10/conversion20,power15,speed scales8/7,offtarget10,goals200,
  skillgoal60/concede150,turnover100,defensepractice.4,neuralpractice.5,
  receiving max8. Training shutdown margin.94/penalty3000,load1.5,warm.92.
  Three frozen neural references: scorer254M,control213M,baseline98M;
  remaining75% current/recent stochastic neural snapshots. No tactical inputs
  or controllers. Mean/sampled/fast development watchers active. Expect finish
  around09:18 UTC, leaving time for final validation before~10:12.
-1024fast reception physics breakdown scorer205: capture rates by paddle e
  quintile [.60,.67,.74,.81,.88,.95] =50.3,50.5,51.2,55.2,50.3%; delay5ms
  51.0%,10ms51.9%. No strong concentration by these physics parameters.
- First six consecutive scorer254 replay visits plotted (scorer254-visits.png).
  Several show clear learned slowdown, preparation then7–8m/s strikes; one
  ends with puck stranded against the unreachable side wall and one is a
  passive bounce. These plots are diagnostic, not a cherry-picked demo or
  evidence that control is solved. Generalize6 now59% controlled visits,
  mean~6.85m/s,bank243/256; stochastic counterpart still hitsload1.0018.

- 06:27 UTC: all three further warm8×1800s mean tests failed strictload:
  scorer254peak1.00441/1.50summedplayer-secondsoverload; receive222
  1.00065/.20s; attack2571.00343/1.70s. Long safety remains unsolved.
  Development tournament1 now tests all six pairings among scorer254,
  receive222,attack257,balanced238 in both colors,8×300s/warm.9.
- Retired generalize6 cleanly after~100M additional uninterrupted training;
  current results still~half controlled visits and long thermal failures.
  Started `_neural-player-reception-stage10` from receive222,40M/45min,
  512env2workers, receiving exercise70% restrictedto4–8m/s (previous
  fast exercise .6–8 includedmanytrivial arrivals),20%stationary10%defense.
  ActorLR5e-5,value1e-3,capture20/conversion20,power15,offtarget10,
  speed scales8/7,shutdownmargin.94/penalty3000,warm.92,load1.5.
  No tactical actions/phaseinputs; bounded specialist curriculum must pass
  full-game transfer before continuation. Fastwatch4M, fullwatch16M.
  Receiving min-speed option validated,13neuraltests pass.
- Evaluator adds per-drive/channel load peaks and first-overload state,
  observedload,action diagnostics at50Hz. Existing metricsunchanged; smoke
  forced simulated overload correctlyrecordsoneeventperplayer/game.
  Progressplot supports run/sampling filters and external legend.

- 06:36 UTC: tournament1 completed (8×300s percolor,warm.9,mean policies):
  scorer254 vs receive222106–20; scorer vs attack25740–106;
  scorer vs balanced23822–74; receive vs attack10–31;
  receive vs balanced4–134; attack vs balanced73–100.
  Balanced238 strongest in this small cohort; attack257 materiallystronger
  competitively than earlier attack225. These are development results.
- League238 mean passes seeded warm8×1800selfplay:peak.932955,zerooverload.
  Most limiting channel fastmotor2,slowchannels remain~.9003max. Further
  cross-play vs balanced238/attack257 bothcolors nowqueued,alongwith
  balanced238 longwarm test. A selfplayloadpassdoesnotqualifycross-play.
- Counterfactual load-input check (common scorer254recordedphysicalstates):
  league238meanrequestedaccel13.51m/s²atallchannels.2→10.89at.98;
  scorer25414.92→12.70,attack27416.90→13.03. Learnedresponseexists,
  but this diagnostic cannot establish closed-loop safety. Report saved.

- 06:47 UTC: found/fixed receive_drill early-goal escape: generic skill
  terminal ended a successful volley before1.2s, while failed-catch cost only
  ran atcapture/time limit. A zero-catch earlygoal got~0 reward beforecontact
  bonuses instead of failed-receptioncost. Include all terminal receptions
  in fail accounting. Ordinary game goalrewards unchanged. Closed-loop
  regression scores an early uncaught puck and confirmsnegative drillreward;
  15neuraltests pass. Reception10 retired, restart10b from itsbestfastcatch
  checkpoint234,356,736 (155/256),30M/35min,otherwiseidenticalsettings.
  Next238Mcheckpoint hadregressed126/256fastcatches,so earliergainunstable.
- Prepared configurable rushed-shot penalty(default0,notactiveinmainruns),
  for nextfull-game continuation. Itpenalizes positiveforwardreturns from
  fastincomingpuck withoutcontrol; controlled/slowspeedstrikes unchanged.
  Thisisrewardshapingonly,neveranactionoverride. Regressionchecks20point
  differenceonlyforrushedcase; existingmetricdefinitionsunchanged.
- League238 crossplay warm.9: vsattack25725–42,vsbalanced23815–69;
  candidatepeak≤.9392 across all fourgamesets. Saferbutcompetitivelyweaker.

- 06:57 UTC: worst-gain/hotstart stress invalidates both earlier fallback
  safety impressions. At1680s balanced238peak1.0224/17.08player-seconds
  overload; league238peak1.01008/2.0s. Both testscontinue to3600s. These
  useinitial.95 andfixedgain1.3, upperendoftrainingmodel. Neitherqualified.
- Added optional --thermal-gain totrain/eval. Itpinsmodeledloadgain only,
  withallotherphysicalrandomizationunchanged; defaultretains1–1.3random.
  Tested resetpersistence;16neuraltests pass. Noactivejobwasrestartedforit.
- Nextplanned continuation after reception10b (~07:15): fullstage5 league,
  strong fixedlearnedrefsbalanced238,attack257,league238, andrecents;
  worstcasegain1.3,load2,shutdownmargin.92/penalty3000,warm.90;
  capture20/conversion20,power15/scales8/7,goals150–200,rushedpenalty20,
  receivedspeed4–8. Needinspectactual30Mreceptionresults andfulltransfer
  beforechoosinginitializer. Run~120min to09:15,thenfrozenfreshvalidation.
  Thisisproposed,notyetlaunched. Keep otherthreelearninguninterrupted.
- Added stress watchers toattack7/balanced8/league9: every32M andfinal,
  mean4×900s,initial.95,gain1.3,developmentonly. Addedcrosswatchleague9
  againstfrozenbalanced238:8×300s eachcolor,warm.9,every32M/final.
  Fresh finalseeds stillreserved; current developmentseed20261901.

- 07:08 UTC: implemented optional physical-frame history within the same
  neuralactor (4×42inputs,current plusprevious3×20ms frames). No tactical
  phase/desiredshotinput. Zero-paddingfirst-layer weights preservesstarting
  actor andcritic/contextpredictions; newpastfeatureweightslearnwithPPO.
  Buffersupdateonceperphysicalstep andresetonlycompletedslots. Frozenolder
  neuralopponents consumeonlytheircurrentframe;newopponentssamehistory.
  Checkpointrecords historycount;default/inheritedsingleframestillworks.
 18neuraltests pass,includinghistoryorder/resetisolation/initialbehavior;
 1024-transition two-workerGPU smoke andmixed1/4-frame evaluationbothcolors
  pass. Smokerunnotacandidate. Newhistorybranchresetsoptimizerbecauseinput
  matrixshapechanges,whilekeepingthisnight'slearnedactor/valueweights.
- Retiredattack7 cleanlyafter~130M additionallearning (~2h uninterrupted).
  Started `_neural-player-history-stage11` fromstrongbalanced238M:
  history4,1024×4,180M/120min (end~09:08),actorLR7.5e-5/value.001,
  entropy.0002,capture20/conversion20,rushedpenalty20,goals200,
  skill60/concede150,power15/scales8/7,miss10,receive4–8,practice.5neural,
  defensepractice.4,thermalgain1.3/load2,shutdownmargin.92/penalty3000,
  warm.9. Fixedrefsbalanced238/attack257/league238 +6recentneuralpool.
  Mean/sampledfullreplay,fastskills,hotstress,andcrossvsbalancedwatchers.

- 07:16 UTC: reception10b completed264,372,224 after30M additional.
  Bestfast testsofar262,799,360171/256; finaltransferpending. Full250M
  report66%control/61%aimed,~5.5m/s,bank220,load.9945—notbetteroverall
  thanstrongbalancedbranch. Savedbutnotusedasnextfullgameinitializer.
- Balanced303Mbeatbalanced238M52–5 acrossbothcolors8×300/warm.9,
  peakcandidate.96975/.96313. Thisisamaterialcompetitivegainfromlonger
  neuralselfplay,evenwhensame-network goalcountsfall.
- Started `_neural-player-constrained-stage12` frombalanced303M,
  history1,180M/115min (end~09:11),sameworstgain1.3/.92margin/3000
  penalty/load2/rushed20/capture20/conv20/goals200/power15 settingsas
  history11. Fixedrefsbalanced303/attack257/league238 +6recentpool.
  Differentinitializer/seed/optimizercontinuationmeanthisisNOTaclean
  causalhistoryablation; thesearetwopracticalcandidatebranches.
  Mean/sampledfullreplays,fastskills,hotstress,crossvsbalanced303watchers.
  History11crosswatchremainsvsbalanced238;dontcomparetheirrawcrossscores
  asiftheysharedopponent. Finalqualificationwillusecommonfrozenopponents.
- Gain1.3/initial.95/hourstress completed: league238peak1.01008/2.0s,
  balanced2381.03725/173.5summedplayer-secondsoverload. Allfirstfailures
  arefastmemorymotor2. Neitherqualifiesforworst-gainlongsafety.

- 07:48 UTC: all39neural/arrival/motion-fidelity/on-policy tests pass,
  pluschangedtrackedfileswhitespacecheck. No subsequentimplementationchanges.
- League303completed8×3600s hot.95/fixedgain1.3:PASSEDpeak.965582,
  zerooverload. 8×600s sameconditionactualfirmwareaudit:PASSED,
  zeroover-capintervals,exact6-state reproduction,minworkspace.865mm,
  peakaccel60.000618m/s²(floatnumericaltolerance). Thisisadevelopment
  fallback,notfresh-seedfinalqualificationorhardwarevalidation.
- Balanced303samehourstressFAILEDpeak1.02628/106.18player-secondsoverload.
  Thus52–5competitivewinbyitdoesnotmakeitaselectablesafecandidate.
- Upcomingbalanced8completionaround08:05–08:15freesone4-worker slot.
  Consider~50–55min refinementofbestprovenbranchwithloweractorLR3e-5,
  strongeron-targetshotpower/missweights(e.g30/30)IFitsactualprepared
  shotsremain~80–86%accurateandtoo weak. No newrewardcodeisneeded;
  thoseflagsalreadyexist. Baselineleague303isprovenhottestmodel-safe,
  newhistory/constrainedbranchesmayimproveafterlongerlearning. Choose
  usingthen-currentevidence;neitherthisrefinementnoritsweightsisfinalyet.
  Keepotherlongjobsuninterrupted;freezeallcandidatesaround09:05–09:15.

## 08:08 UTC — common-opponent comparison and final precision continuation

Comparison2 used the same balanced303 opponent, 8×300 s per color, warm load
0.9, mean actions. Candidate scores: league303 3–43, league352 2–10, history303
7–69, constrained352 3–12. Candidate peak loads were respectively .9696, .931,
.959, .912 with zero candidate overload. Some opponent overloads occurred;
aggregate load must not be misattributed to the candidate. League352 prepared
shots were 618/648 on target (~95.4%) at ~6.42 m/s against this opponent.
The remaining offensive weakness is real, despite substantially better defense.

A 55-minute low-learning-rate precision continuation is queued to start after
balanced8 naturally completes. Parent league352; actor LR2.5e-5, value5e-4,
initial exploration std×.5, entropy1e-4, power30 and miss30, strong balanced303
as one frozen neural opponent. Other parent settings including thermal margins
remain intact. No tactical override. New run `_neural-player-precision-stage13`.
Separate mean and sampled hour-long hot(.95)/gain1.3 load tests of league352 are
running. These are still development tests, not fresh-seed final qualification.

First six consecutive ≥.2 s blue visits in the league352 replay were plotted,
not selected as highlights. Five show slowing followed by a stronger strike;
one remains a weak uncontrolled return. The replay does demonstrate learned
reception/setup behavior, but not reliable control or diverse adversarial shots.

## 08:19 UTC — load failures remain a hard selection constraint

Balanced8 completed at389,677,056. Its development replay improved to542/587
prepared shots on target,mean7.94m/s,632/959 visits controlled,243/256 bank
saves. However final short hot/gain1.3 stress FAILED: peak1.04913 and656.18
summed player-seconds overloaded in4×900s. Excluded; cancelled its redundant
hour-long test after this result. Strong shooting alone is insufficient.

Constrained368 passed the matching short stress: peak.95064,zerooverload.
League368 beats balanced23818–2 over both colors,peakcandidate.91491; again
this reference is weaker than balanced303 used by the other cross watcher.

A single-step load-input counterfactual for league352 on18,000 recorded
physical states changed only its eight cached load features. Setting all
channels0→.97 lowered mean requested acceleration15.07→12.91m/s² and share
of requests>40m/s² from9.08%→6.93%. Requests near60 remained possible. This
is learned input dependence, not proof of closed-loop safety; full stress
tests remain required. Artifact: league352-load-input-counterfactual.json.

## 08:24 UTC — mean and sampled behavior need separate load qualification

League352 passed8×3600s hot.95/gain1.3 with mean actions: peak.974502,
zerooverload,73.25% visits controlled,99.17% contacted; prepared11002/11690
on target(~94.12%) at6.53m/s. This long hot condition produced17goals total.
Sampling its learned Gaussian FAILED the matching stress: peak1.00196 and
1.02 summed player-seconds overload. Do not qualify sampled inference using
the mean-policy result. The load model is still provisional.

Constrained384 development: 606/866 visits controlled(~70%),483/531prepared
shots on target(~91%),mean6.79m/s,239/256bank saves,peak.848. Common-opponent
comparison3 now includes this checkpoint,league384,history335,andprecision368
against balanced303 inbothcolors, plus an independent firmware audit of
league352. Precision368 is learning without a large early collapse; it retains
about64%control and65%aimedvisits,with higher shot speed than its parent.

## 08:32 UTC — final competitive refinement

Comparison3 vs the same balanced303 reference (8×300s ineachcolor,load.9):
constrained3843–15,league38412–8,history3351–34,precision3680–12. All candidates
had zero overload; peakcandidate load .915,.973,.962,.940 respectively.
Precision368 prepared727/747on-target(~97.3%),~6.94m/s inthese matches, but
failed to score. Accuracy is insufficient evidence of offensive strength.

History11 was gracefully stopped after ~112M additional transitions and its
final checkpoint preserved. It has not demonstrated a practical advantage.
No causal history-only claim: initializer/optimizer differ across branches.
Its slot now runs `_neural-player-offense-stage14` for43min (~09:15end), from
the competitively stronger league384. Goals600,shotpower30,miss30,speedscale10,
conversion scale9; LR5e-5,initial std×.75. Pinthermalgain1.3,load2,retain.94
shutdown margin/3000penalty. References balanced303,league352,attack257 plus
recent stochastic neural snapshots. This is reward shaping and opponent
selection only: the actor still receives no tactical instructions.
Precision13,league9,andconstrained12 continue unchanged.

## 08:54 UTC — accuracy must survive sustained warm operation

League384 and constrained400 both passed8×3600s hot.95/gain1.3 load tests:
peak.97799 and.95828 respectively,zerooverload. But their prepared-shot
accuracy fell to6335/9438(~67.1%) and6896/10769(~64.0%). These are not strong
all-condition candidates despite good cold replays.

Precision384 short hotstress retained96.6%preparedaccuracy,80.8%controlled
visits,6.33m/s preparedshots andpeak.96446. League400 short hotstress also
retained95.4%accuracy at7.0m/s,70.3%controlled,peak.96117. This warm-quality
variation is part of selection, not just the overload pass/fail.

Precision400 development improved to725/979controlledvisits(~74.1%),717/979
aimedvisits,651/667preparedshotson-target(~97.6%) at7.06m/s. Offense400 showed
583/863controlled(~67.6%),503/519preparedaccurate(~96.9%),7.50m/s. Both now
have hour-long hotstress evaluations underway. These still use development
seeds; final qualification will use the reserved fresh seeds after selection.

## 2026-09-23: user-requested shot conditioning and useful fallback returns

The user likes the neural WIP but reports a nearly exclusive right-wall bank
and passive returns on medium-speed incoming pucks. They explicitly requested
restoring a straight/left-bank/right-bank observation. This supersedes the prior
preference to omit shot requests; the policy remains a single neural actor.

Implemented optional three-input conditioning after the physical observation
history, preserving old checkpoints through zero-initialized new columns.
Requests are uniform and fixed per possession. Route rewards use the calibrated
friction/restitution flight predictor. Accurate requested routes earn a bonus;
wrong prepared routes are penalized. Immediate accurate returns now earn useful
fallback credit, with additional reward for control followed by a shot. A modest
penalty covers completed reachable visits with neither control nor a fast aimed
shot. Added requested-route confusion matrices, matched-fixture request suites,
full-game outcome bins by incoming speed, and replay request labels.

Validation: 48 focused tests passed (neural policies and requests, arrival,
firmware motion fidelity, PPO accounting, and shot flight). CPU training smoke
completed 1,048,576 new transitions from offense-stage14 step 439,484,416.
CUDA is unavailable in this session; CPU throughput is approximately 12–14k
transitions/second with 1024 environments and four simulator workers.

Baseline, seed 20262351: stationary 64/64 contacts; first aimed requested shots
0/64 straight, 0/64 left, 56/64 right. Receiving at 2–5 m/s: 61/64 controlled,
62/64 controlled or fast aimed, first requested successes 0/64 straight,
0/64 left, 59/64 right. This confirms severe route collapse; the receiving
fixtures are easier than the problematic full-game exchanges. The one-million-
step smoke checkpoint remains strongly right-biased and is not an improvement
claim.

Active continuation: `_neural-player-requests-stage15`, bounded to 60M new
transitions / 120 minutes, preserves parent limits and conservative thermal
settings. Run settings and source snapshots are in its run directory. Automatic
checkpoint evaluations and full self-play WIP replay refreshes are handled by
`ai/bin/watch_neural_requests.py`; reports are under
`logs/neural-player/requests/stage15/`. No hardware activation, physical default
changes, or deployment promotion. Final qualification is pending.

### Conditioning follow-up: escape the learned right-bank preference

At roughly 4M and 8M new steps, ordinary request-reward continuation still
produced almost exclusively right banks. A focused practice branch (curriculum16)
also failed to establish the missing routes and was explicitly interrupted at
step 454,033,408; its last periodic checkpoint remains preserved. Its stale
running flag was corrected after the process exited and progress stopped.

Added a request-dependent slow-puck setup potential and an optional easy-start
curriculum (`--shot-setup-fraction`). This changes training reset positions only,
never explicit evaluation fixtures or actor actions. Curriculum18 tests these
starts; it is bounded to 16M steps / 25 minutes. Its prospective self-play19
continuation requires at least 25% combined stationary/receiving success on EACH
of straight and left, retained right-bank success, reception, and defense.

An audit of earlier fully neural checkpoints found that control-strike-stage2c
79,560,704 retained accurate direct shots (30/32 stationary and 31/32 receiving
first shots on the direct requested route). Distill20 trains the SAME ordinary
45-input NeuralPlayer from this previously learned straight behavior, the
current offense-stage14 right-bank behavior, and reflected neural training
examples for the left route. Teacher policies and reflection exist only during
training. DAgger collects labels on student-visited states and fades teacher
roll-ins; saved actors need no teacher or tactical controller. Reflection is an
inductive prior: the measured motor-load fit is asymmetric and is not claimed
to be physically reflection invariant. Unmodified thermal qualification remains
required. This branch is bounded to 2M transitions / 25 minutes, followed by
selfplay21 only if the route and retained-skill gates pass. Selfplay21 is bounded
to 40M transitions / 75 minutes. Intermediate and final evaluations are automatic;
no checkpoint is promoted to physical deployment.

Focused verification now passes 51 tests. The mixed old/new policy smoke
matches worked in both colors. A short warm-load firmware audit reproduced
all six motion states exactly, with no over-cap integration intervals; this is
not a replacement for the scheduled sustained-load checks. Full-game outcome
comparisons are currently noisy and no consistent reception improvement or
three-route solution has yet been declared.

Distill20 at 507,904 new transitions (step 439,992,320) finally changed route
behavior on ordinary evaluation fixtures. Requested aimed first shots /128:
straight 117 stationary /111 receiving; left 73/76; right 79/98. Shot speeds
averaged 5.4–6.8 m/s across these cohorts; receiving contact 128/128 and ordinary
defense saves 128/128 in the random-request suite. These are development results,
not held-out qualification. The full-game test FAILED load: peak 1.0529 and
67.5 summed player-seconds at/above the modeled limit across 8×120-second games.
It also regressed full-game control versus the original policy. Distillation
is therefore skill initialization only, not a candidate for physical use. The
queued selfplay21 phase retains load weight 2, thermal gain 1.3, shutdown margin
.94 with penalty 3000, and warm starts to .92 to relearn constrained play.


### Request conditioning: varied-opponent correction and modern26

The first straight teacher (control-strike-stage2c 79,560,704) overfit the fixed
practice opponent: only 10/64 stationary and 12/64 receiving requested direct
shots with randomized opponent positions. Its impressive fixed-opponent score
was insufficient. Distill20/selfplay21 and shared24/wide25 are not selected.
Selfplay21 at 12M lost 26–118 to the previous offense policy across both colors.
Watchers now randomize practice opponent positions by default, test 5–8 m/s
receiving separately at each checkpoint, and rescan for a final checkpoint if
training ends while an older checkpoint is being evaluated.

Modern26 uses the later, fully neural attack-stage7 318,898,176 straight teacher,
the offense-stage14 439,484,416 bank teacher, and reflected neural examples.
Its exported 512-wide actor has 45 inputs (42 physical + three-way request), six
arrival outputs, and no teachers or tactical selector at inference. Width
expansion preserves the old function before training. All-state neural teacher
labels are used in this branch, without the shared-reception phase selector.

Modern26 final is step 443,498,496, after 2,007,040 distillation transitions.
Development requested aimed first shots /128 (stationary, then 2–5 m/s receiving):
straight 115/104; left 113/119; right 117/114. Opponent positions are randomized.
Against the previous WIP, 4×180-second games in each color yielded 33–18 as blue
and 29–16 as red, combined 62–34. It FAILED sustained load: 8×900 seconds with
initial load .95 and thermal gain 1.3 peaked at 1.03794, with 35.82 summed
player-seconds above the modeled limit. A separate 8×180 held-out self-play
cohort also exceeded load (1.01226; 2.66 summed player-seconds). Not hardware-ready.

Medium receiving is not improved by neural skill transfer alone. In a matched
256-trial 5–8 m/s varied-opponent cohort, original WIP achieved 175 productive
receives (control OR aimed shot >=4 m/s), versus 166 for modern26 at 0.762M.
A new optional productive-receive curriculum ends on measured control (+60),
fast aimed fallback (+35), or timeout/failure (-30 plus distance/speed cost).
It removes positive contact-shot bonuses within these short exercises so the
large request bonus cannot make an immediate volley more valuable than control.
Full games and ordinary strike exercises retain their existing rewards. No
curriculum task label enters the actor. Receiving practice now covers 5–8 m/s.

55 focused tests pass, including actual collision fallback vs weak/off-target
return tests, no-touch-goal failure, mixed conditioning/history, width expansion,
request reward and feature semantics, motion fidelity, and watcher completion.
Receive27 was stopped after its first PPO updates exposed excessive policy KL
(~0.9 despite low learning rate): the trainer checked KL only after a whole
epoch (64 minibatches). The trainer now stops actor updates within an epoch while
continuing its independent critic fit. A replacement run will start again from
modern26, preserving the useful learned routes rather than the drifted branch.
No robot activation or deployment-default changes have occurred.


Receive28 is the replacement: 8M transitions / 25-minute bound, 512-wide actor,
actor LR 5e-6, value LR 5e-4, two PPO epochs, target KL .02 checked before each
minibatch update, exploration std scaled .5, load weight 3, thermal gain 1.3,
shutdown margin .92, warm starts to .90. It retains 50% full-game slots and uses
strong attack318 plus original offense439 as frozen league references. The
CPU smoke with an extremely low KL threshold performed exactly one actor
update and continued the independent critic, then exited normally. Initial
Receive28 policy KL is near .02 rather than Receive27's .9.

Modern26 held-out requested-shot counts /256 stationary then receiving:
straight 226/231, left 225/231, right 235/223 (~89% each combined). Its final
5–8 m/s development receiving score is 177/256 productive, 132 controlled,
256 contacted, vs original 175 productive and 133 controlled: no meaningful
receiving improvement yet. Full self-play has different generated trajectories:
modern26 productive visits by entry-speed bin [148,117,114,19] over
[181,144,305,52]; original [168,40,138,34] over [204,52,251,89]. These are not
paired incoming trajectories and must not be interpreted as a causal reception
comparison. Prepared modern26 shots (controlled OR incoming <=.8 m/s) were
286/381 aimed, 283/381 aimed on the requested route, mean speed 7.09 m/s.

The modern26 long hot test independently reproduced all six firmware motion
states exactly, reported zero intervals over the acceleration tolerance and
minimum workspace margin ~0.95 mm. Its thermal failure remains disqualifying.


### Receiving refinement: results and control29 continuation

Receive28 completed 8,126,464 new PPO transitions at step 451,624,960 in 890 s
(~9.1k transitions/s on CPU). It retained three-route accuracy and reduced load,
but did NOT improve medium receiving on the larger test. Fresh-seed 1,024-trial
5–8 m/s results (same fixtures/randomized opponents, seed 20262455): original
WIP 728 productive /584 controlled; Receive28 at 4M 691/542; final 687/546.
The small development cohort had suggested an improvement (183/256 at 4M vs
175/256 original), which failed to generalize. Do not claim a receiving gain.

The 4M checkpoint's 8×900-second hot test (initial .95, thermal gain 1.3) peaked
at 1.003446 with .86 summed player-seconds over limit: substantially reduced,
still a failed thermal check. The final checkpoint's qualification is ongoing.

Control29 resumes Receive28 final, bounded to 32M further transitions /75 min.
Its only reward changes from Receive28 are capture weight 15→30 and a bounded
control-quality potential of weight 30. The potential smoothly values slowing
a recently touched puck close to the paddle in reachable space. It is applied
as gamma*Phi(next)-Phi(current), with zero terminal potential, so holding or
repeatedly entering the same state does not farm a per-step bonus. No target
velocity, action override, task flag, or tactical selector enters inference.
The existing receive exercise prefers measured control (+60) and accepts a
fast aimed fallback (+35). Full games continue in half the environment slots.

All other learning/load settings are retained, including actor LR 5e-6,
within-epoch KL .02 stopping, load weight 3, shutdown .92, thermal gain 1.3.
Exploration std is NOT halved again on this continuation (std_scale=1).
New seed 20262312; automatic development/qualification watcher seed 20262361.
The watcher now also schedules a 1,024-trial fresh-seed medium-receiving test
at completion. Control29 replay: neural-requests-control-wip.json, initially
showing the resumed policy and then refreshed at evaluated checkpoints.
56 focused tests pass. The new run has progressed through its first PPO update
with finite outputs and KL .0125. No physical system activation or promotion.


Receive28 final hot qualification completed: 8×900-second continuous self-play
matches, initial load .95, thermal gain 1.3, seed 20262451. Peak modeled load
0.99276144, zero summed player-seconds over the modeled limit. It passes this
simulation stress test; this is not physical validation or evidence that the
receiving regression is solved. Score totals [200,208]. Firmware audit has zero
reproduction error and zero integration intervals over acceleration tolerance.
Control29 is progressing normally (1.31M new transitions by 143 s, KL .0133).


### Defense requirement audit and defense30 (2026-09-23)

Control29 completed 32,112,640 transitions at step 483,737,600 in 3,316.9 s.
It won 47–18 combined against offense439 over four 180-second matches in each
color (25–7 blue,22–11 red), and passed 8×900-second hot self-play: maximum
modeled load .975588, zero overload time. Its held-out requested-shot success
was straight 421/512, left 394/512, right 437/512 (stationary+receiving). These
are useful results but NOT a final candidate; hard defense remains inadequate.
Its prior 256-trial hard bank/random-paddle cohort saved 167. Its larger medium
receiving result 734/1024 uses seed 20262465, not the original baseline's seed
20262455, so no paired improvement is claimed.

New matched defense audit, seed 20262390, 512 trials each, 8–12 m/s launch
speeds, randomized practice opponent, realistic sensing and motion:
- Direct, centered ready paddle:504 saved/512,508 contacted,7 conceded.
- Bank, centered ready paddle:466/512 saved,476 contacted,32 conceded.
- Bank, varied paddle position:373/512 saved,400 contacted,125 conceded.
Saved requires contact and no concession through the trial; no-touch survivals
are not counted. Free-flight goal validation is retained. Random starts are
not asserted to be dynamically recoverable. No superhuman-defense claim.

Found and fixed a curriculum allocation bug: productive receiving ignored
practice_defense_fraction=.4 and imposed only10% defense practice slots, i.e.
5% of all slots with half reserved for games. All those slots had odd indices,
so the bank/direct selector gave them only banks. Productive receiving now
honors the requested defense fraction, and attack type is sampled independently
of slot index. Optional defense settings vary paddle starts, raise incoming
speed, and reward measured control or a touched puck cleared into the opponent
half. A no-touch miss or conceded puck cannot earn the clear reward. No changes
to supplied benchmark fixtures or to full-game termination/defense actions.

Defense30 resumes Control29 final,32M new transitions/75-minute bound, seed
20262331. Approximately50% full games,30% defense,10% stationary shooting,10%
receiving; defense8–12 m/s, half direct/half banks,60% varied paddle starts.
Defensive clear/control bonus100, practice concede penalty600. Other control,
request,load and PPO settings are retained. No repeated exploration scaling.
New flags: --defense-min-speed,--random-defense-start-fraction,
--defense-clear-reward. Automatic watcher seed20262390 tests three fast-defense
cohorts per checkpoint (512 each), repeats with1,024 fresh-seed trials at the
end, and adds matching hot-motor tests(initial.9,gain1.3). Existing shot,
receiving,continuous-load and two-color cross-play tests remain scheduled.

Defense is a selection requirement, not just an auxiliary metric. Target near
99% saves from ready position on both direct and bank attacks, with strong
varied-position recovery and no load violations; investigate physical
reachability of residual failures before claiming a ceiling. No completed
candidate yet establishes this.58 focused tests pass, including curriculum
fractions,attack diversity,physical reset isolation,and defensive reward
anti-exploits. Defense30 has completed its first PPO update with finite outputs,
KL .018 and ~9k CPU transitions/s. No hardware activation or deployment change.


### Replay publishing repair

Defense30 completed at515,850,240 after3,261.7s (~54min), with all evaluations
finished. It is not still training. Its final fresh-seed1024-trial defense
results remain short of the target:1003 direct-ready,931 bank-ready,718
bank-varied saves. Long hot self-play passed (peak .960623,zero overload).
Combined cross-play vs offense439 is47–46. Do not promote it as the final
well-rounded/strong-defense candidate.

The user could not find training replays because the watcher overwrote a
single generically named WIP file. The server groups by filename, so no actual
run/step group existed. The watcher now preserves `<run>_step_<step>.json` and
atomically refreshes the old WIP alias. It also publishes an initial-policy
replay and rechecks for a late-arriving initial checkpoint. Cached evaluations
with missing recordings are regenerated. Metadata now includes run_name.

Rebuilt all9 defense30 recordings (initial+8 checkpoints), each a complete
120-second/6000-frame self-play game using8 simultaneous games,seed20262390.
The existing final WIP recording was preserved under its canonical filename;
earlier checkpoints were re-evaluated in simulation, with new reports under
logs/neural-player/requests/defense30/replay-backfill. The UI run group is
_neural-player-requests-defense30, final link:
http://localhost:8420/?replay=_neural-player-requests-defense30_step_515850240.json

Both publishing regression tests pass (checkpoint retention/final-checkpoint
race,and watcher starting before trainer initialization). The server's isolated
metadata helpers identify all9 files under the correct run, with matching steps
and6000frames each. Direct HTTP verification from this tool environment could
not connect to localhost:8420; no server restart or hardware connection was
attempted. Existing UI polling reads the new filenames without a server change.


### September 25 physical-session audit and defensepower31

The user reports strong in-person play and requests more defense/shot consistency.
Audited three September 24 defense30 sessions: no commanded workspace/cap
violations, max fast RMS81%, all8 live channels fresh, zero fault samples.
Found cached controller state used without age projection; moving-state p95
error34–39mm becomes2–3mm in an offline projection check against later controller
samples. Adapter now projects position/velocity before observation/decoding.
Launcher also isolates master from terminal Ctrl-C, avoiding the cleanup second
SIGINT racing its disable sequence. Fake-process signal-order test passes.
No physical system was activated. Full findings and plots:
logs/analysis/neural-live-20260924/README.md.

Defensepower31 resumes deployed515850240,16M transitions/45min bounded,1024envs,
4CPUworkers (~9–10k/s; CUDA unavailable).50% games/30% defense/10% shot/10%
receiving. New wide defensive goal coverage addresses center-only practice;
3–12m/s banks/directs,60% varied starting positions. Shot power/request speed
credit exponent2,scale8m/s,powerweight50; load/control/accuracy penalties retained.
Frozen opponents: strong straight neural318898176 and currently deployed515850240,
plus evolving neural snapshot pool. Watcher publishes canonical run/step self-play
replays and final cross-play/hot-load checks. Fresh paired comparison seed20262593
includes requested/on-target first shots>=6m/s and wide bank defense. Training
results and checkpoint selection remain pending; old physical command stays pinned.


Defensepower31 training completed at 531,972,096 after 1,630.6 seconds
(16,121,856 additional transitions). Paired results through 12M show stationary
requested-shot accuracy ~79% versus 72.7%, and requested/on-target first shots
at least 6 m/s ~51.4% versus 48.0%. Wide-bank defense remains ~77%, with no
established improvement. Full tables and final evaluation artifacts:
logs/neural-player/requests/defensepower31/README.md.

Runtime validation: 152 relevant tests pass; explicit defense30 offline policy
preflight passes. Near-contact raw-marker logging is included for the next
physical session. Controller observer command timing now uses send/ACK midpoint
relative to the POS query, while arrival planning retains its trained 15 ms
assumption. The actor retains training's finite-difference velocity convention;
the decoder uses projected instantaneous velocity. No hardware was activated.


Because defensepower31 did not improve defense, started defensefocus32 from its
12M checkpoint (528039936), selected for highest combined fast-requested count
1599/3072 on the development fixtures. Changes are practice allocation only:
25% self-play, ~52.5% defensive drills, ~15% stationary shots, ~7.5% receiving;
90% varied defensive starts. Same neural architecture, rewards, load penalties,
physics and optimizer. Bounded at 12M additional transitions / 35 minutes.
A paired baseline sensing ablation saves408/512 bank-varied with ideal sensing,
versus360/512 through the realistic perception stack: a material but partial gap.
Progress, replay and comparison details: logs/neural-player/requests/defensefocus32/README.md.


Found a concrete observation mismatch during that ablation: production cuts its
velocity fit at bounces and uses real fixes through gaps; legacy simulated
perception averages across bounces and extrapolates hidden positions. Added
opt-in --report-sensing, tested directly against the live estimator; default
legacy behavior remains unchanged for historical reproducibility. Frozen baseline
bank-varied saves360/512 legacy,392/512 report-matching,408/512 ideal. These are
sensor ablations, not learning gains. Forcing full acceleration gives353/512.

Stopped defensefocus32 after its saved4M checkpoint532103168 to prioritize this
correction. Tool interruption exited130; status records that the last partial
update532234240 is unsaved. The checkpoint/replays and paired comparisons remain.
Started reportdefense33 from the retained defensepower31 step528039936, same
focused practice allocation, but corrected report sensing. Bounded12M/35min,
4M snapshots; baseline and candidates are both evaluated with --report-sensing.
Latest replay: http://localhost:8420/?replay=neural-reportdefense-wip.json.


Reportdefense33 first4M checkpoint532103168 is the best fully tested candidate
so far. Paired report-sensing fast requested shots: stationary718→795/1536,
receiving832→919/1536 versus deployed515850240. Fresh bank-varied789→802/1024;
hot bank-varied789→784, so defensive gains are modest/mixed. Two-color cross-play
42–26 over16 three-minute games;8 five-minute hot games from.95/gain1.3 peak.96514,
zero overload and zero firmware intervals over cap. Retained copy:
runs/_neural-player-requests-reportdefense33/agent_candidate_532103168.pt.
See logs/neural-player/requests/reportdefense33/CANDIDATE.md.161 tests pass and
its offline production preflight passes. No promotion or physical activation.
The8M checkpoint improves shots slightly but loses the4M defensive gain; final
snapshot evaluation remains pending.


Reportdefense33 training completed at540098560 after1208.9s (12,058,624 new
transitions). Final paired fast-requested count1704/3072 vs4M candidate1714;
wide-bank406/508 vs408. Retain the reviewed4M candidate532103168. Later snapshots
did not establish a better balance. Long final-snapshot evaluation continues
under the automatic watcher; README.md updates as held-out/hot/cross-play reports
finish. Existing physical checkpoint remains515850240; no hardware activation.

### September 25 unrestricted runtime and fastdefense34

The host GPU is accessible again. The web UI is reachable on8420 and the
reviewed reportdefense33 step532103168 replay was fetched successfully over
HTTP, including metadata confirming both sides use that same neural checkpoint.

Reportdefense33 final qualification has completed: cold direct/bank-ready/
bank-varied saves1010/958/804 of1024, versus deployed baseline1007/955/789.
Hot saves1009/976/786 versus1006/962/789. Eight900-second hot games reached
modeled load0.99059 without overload or firmware over-cap intervals. Small
two-color cross-play totals12–9 do not establish superiority to the retained
intermediate candidate. Keep reviewed532103168 pinned; these final results do
not imply a checkpoint promotion.

Started `_neural-player-requests-fastdefense34` from that retained candidate.
One curriculum change: incoming defense speeds8–12 instead of3–12m/s, retaining
wide direct/bank attacks and90% varied paddle starts, shot practice, self-play,
report sensing, architecture and all limits/load penalties. CUDA pilot1,048,576
transitions took38.49s (~27,240/s); main run bounded24M/30min with4M snapshots.
The pilot is a timing check; main training resumes the reviewed candidate again.
Automatic watcher publishes120-second neural self-play at
http://localhost:8420/?replay=neural-fastdefense-wip.json .
Paired comparisons include all requested routes plus1024 matched cold/hot
bank-varied trials, reporting rescued and lost saves. Results/configuration:
`logs/neural-player/requests/fastdefense34/README.md`. This is a WIP experiment,
not a new final candidate or a physical deployment change.

### September25 user feedback: interrupted possession, drifting recovery, readiness

User requests a couple of hours of careful improvement and production
preparation. Confirmed legacy three-second possession clock forcibly replaces
controlled pucks in full games/replays, plus capture-terminating receiving
drills. Implemented opt-in continuous rallies: only goals or pucks stopped8s
outside both robots' contact workspaces are re-served. Added replay event labels.
New training receiving exercises continue through shooting, include60% slow
outgoing starts0.15–1.2m/s, and reject an uncaught volley as successful recovery.
Training game episodes extend30→120s, with a gradual possession-delay cost.

Added bounded training-only goal-coverage potential using several goal-mouth
targets and interception depths; no target, phase, forecast, or new input goes
to the actor. New evaluation records defensive position/coverage, reachable
stalls, and control-to-requested-fast outcomes. Optional opponent-only shot
requests provide a learned straight-shot pressure test. Actor/decoder/caps stay
unchanged, and default historical evaluation behavior is retained.

Stopped fastdefense34 cleanly via STOP at549142528; obsolete evaluators stopped
after preserving reports. Its first4M536166400 is the strongest tested offense
snapshot (fast requested854/944 stationary/receiving vs reviewed33's795/919;
cold/hot bank-varied785/770 vs780/772). Selected it only as an initializer.

Started `_neural-player-requests-possession35`:48M/60min,8M snapshots, CUDA,
readiness potential80, recovery follow-through, same thermal settings. Smoke
262,144 transitions10.15s;176 relevant tests passed before later additions.
Offline `play.sh --check-policy` on its smoke checkpoint passed:45 inputs/six
outputs,50Hz,1.17ms decoder+guard,12m/s and60m/s². No hardware opened.

Paired recovery seed20263035,1536 trials: reviewed33 control441, conversion375,
controlled/requested/>=6m/s351; initializer34 control424, conversion372, fast349.
Phase35 first8M544292864 control457, conversion406, fast374. Improvement is small;
straight-route control actually declines versus reviewed33. First self-play
still spends~95% of defensive time above y=.45, so not a solved readiness issue.

Started `_neural-player-requests-possession36` from35 step552419328 (16M):64M/80min,
8M snapshots. Adds ongoing readiness cost100 and once-per-possession recovery
capture bonus60, retains complete recovery-to-shot episodes and all limits.
Phase35 continues as a comparison branch. Readiness cost excludes fast incoming
returns and own-half possession; tests verify it and nonrepeatable capture
credit. Both branches have autonomous replay/evaluation/comparison workers.

Logs/config/process IDs:
`logs/neural-player/requests/possession35/` and `.../possession36/`.
WIP links: `?replay=neural-possession-wip.json` and
`?replay=neural-possession-readiness-wip.json` onlocalhost8420.
Comparison scripts use matched new rally rules and sensors for all references;
watchers add fresh final/hot/cross-play checks. Neither branch is qualified yet.
User's selected reviewed candidate stays33; production defaults stay unchanged.

Correction to older prose: actual NeuralTrainingEnv fixes half of training
environments to full games. `practice_selfplay_fraction` chooses active neural
opponents within practice slots; it does not set the full-game fraction.
Phase35/36 allocation is50% games,20% defense,10% stationary,20% receiving
(12% of all environments are outgoing recovery drills).


## September26 possession/recovery/preparation follow-up

User reported premature resets, poor slow outgoing recovery, and high/off-center waiting positions. Found legacy3s possession referee and capture-ending receiving drills. New opt-in continuous rallies preserve reachable pucks and only re-serve unreachable dead pucks after8s; new receiving drills continue through control and shooting. Visible replay rally labels distinguish goals/dead serves. Ordinary historical simulator defaults remain unchanged.

Phases35/36 tested outgoing recovery drills (12% of all environments), coverage shaping and waiting exposure cost. Actual allocation is50% full games,20% defense,10% stationary,20% receiving; practice_selfplay_fraction changes opponents within practice, not this allocation. First gains were small and mixed. The behind-puck setup potential conflicts with getting ahead of outgoing pucks; phase38 blends it away during slow uncontrolled outward motion and shapes approaching the leading side (weight40), with no runtime target/action override.

Far-half frozen-observation diagnosis: changing only the expired shot request shifted desired lateral arrival by median~36cm. Phase37 adds training-only actor consistency on such counterfactual observations; own-half shot requests remain distinct. First8M checkpoint reduced this to~32cm; not yet a solved behavior. All three later branches start from36step568672256 for comparison.

New preparation benchmark lets the actor choose a position during a hidden0.3–1.2s pause before an8–12m/s direct launch. Reviewed33 saved391/1024 versus1006/1024 on a separate immediate-launch centered-start cohort. These are different scenarios, not a paired estimate; they expose why existing reaction benchmarks overstated defensive readiness. Phase39 adds delayed-launch exercises to60% of defense practice. Single neural network remains the only tactical actor; shooting-machine fixture timing is training-only and not an observation.

Runs37/38/39 each target64M additional transitions with8M snapshots. Evaluation compares recovery, requested fast shots, cold/hot reaction defense, continuous selfplay, preparation and request invariance. Reviewed33 remains pinned until qualification. 183 relevant regression tests passed after these changes. No hardware activation or deployment-default changes.


### Later qualification findings and42 continuation (September26 ~04:10UTC)

- Phase36 final570114048 fails long hot audit (peak1.0485,11.26 player-seconds overload); excluded from deployment consideration. Descendants/earlier checkpoints need independent tests.
- Phase37 stopped at602226688 after32M: preparation689/1024 vs reviewed33 391. Phase38 stopped602357760. Phase40 adds learned reference preservation; phase41 increases recovery-capture credit and applies preservation after control. Phase40 early8M hot8×300 passes peak.9963, zerooverload/overcap, but32% of player time was stalled with reachable stationary pucks, so this is NOT a successful fullgame result.
- The old3s referee hid absorbing stationary states. Stationary practice paid control credit but only charged2 for timeout without shooting. Phase42 adds120 failure penalty and physically consistent rest-command histories on50% of stationary resets.
- CRITICAL recovery metric correction: previous v1 evaluation could count a subsequent reception after the original puck had escaped. Version2 sets the recovery kind/mask and ends at FIRST possession loss. All old recovery counts are superseded; use logs/neural-player/requests/possession42/strict-recovery. Training already ended at escape. Fresh qualification reference is recovery-v2.json.
- Direct outgoing first-contact intercepts were only3/1024 in the reviewed33 diagnostic; many apparent leading contacts were later opponent returns. New metric first_outgoing_contacts_from_ahead excludes those.
- Phase42 starts from41step617431040 (preparation725/1024), teaches50% easier ahead-of-puck recovery starts, and shapes collision-avoiding approach distance rather than a straight line through the puck. Waiting exposure cost300. Preserves only recorded SUCCESSFUL learned shots (979 shots,15310 prior examples) instead of anchoring arbitrary current states to an older actor that can itself stall. Single deployed neural actor remains unchanged; all teachers/curriculum are training-only.
- New trials pending. No final candidate selected or hardware activated.


### Focused recovery and load correction, ~04:25UTC

Strict development recoveryv2 reviewed33:41/1536 controlled,36converted,30controlled requested>=6m/s,9first outgoing-leading contacts. Fresh qualification baseline is24/1536controlled,15controlled requestedfast. This exposes how much v1 overstated original-possession recovery. First42step625557504:47controlled/40converted/37fast,17outgoing-leading contacts,675first requestedfast vs610reference. Small recovery gain, not solved.

Further long hot failures:38finalpeak1.1059 with587.24summed overload seconds;40finalpeak1.2259 with35.04s. Exclude both final checkpoints.37finalpassedpeak.956 but~5042reachable-stall player-seconds in8×900s, so passing heat alone is not proof of strong play.40early8M cold.3/gain1.3 8×300s also stalled~1143/4800playerseconds; not merely hot cooldown.

Collected six genuinely reachable stationary failure states from that cold evaluation, retaining their actual prior arrival actions and requests. A seventh was an unreachable fringe puck; removed the extra5mm dead-puck reach tolerance and accounted for the guard's1mm inset.43 adds failure-state resets,15%fullgames/17%stationary/17%defense/51%outgoing recovery (70%easy ahead starts), warmmax.98/correlated heat25%/shutdown1.0/loadcost6. Previously training terminated at.92 and therefore never learned the.95evaluation states.43saved/stopped630145024 after~4.6M.

Key exploration measurement: outgoing recovery raw std median~.03 on arrival dimensions, only~1cm target variation. This is much smaller than the9.1cm collision radius that must be circumnavigated.44 continues43 with stdfloor.35 on arrival position/velocity ONLY for slow outgoing own-half and distant stationary observations. Same likelihood scale used in rollouts andPPO. No inference noise or tactical action override. Runs48Madditional/70minbound,8Msnapshots. MainPID3119734,watch3119735,preparation3119736,strict comparator3119737. Logs underpossession44. New WIP aliasneural-possession-exploration-wip.42stilltraining parallel;41nearcompletion;39complete. No new finalcandidate or hardwareactivation.

### Recovery refinement and fresh evaluation, ~04:50 UTC Sep26

42 completed665534464. Fresh frozen qualification seeds: preparation1528/2048 vs33's809, hot.9 preparation1500vs727. Stationary requested>=6m/s807/1536vs774; incoming3–8m/s requested>=6m/s1009vs940; captured-then-requested-fast665vs637 stationary and966vs918 incoming. Strict outgoing recovery v2 only30/1536controlledvs24,18controlled-requestedfastvs15. Do not label recovery solved. Continuous8×180s games30goalsvs12, but reachable-stall time611vs504playerseconds; quiet-state failures remain. Full qualification/RMS/cross-play pending underpossession42/qualification-final.

45 continues44step646397952 for24M with load-aware exploration(.35 below.8heat, taper to0 at1), and disc-avoiding slow shot setup potentialweight12. First4Mstep650461184 passed development8×300s hot.95/gain1.3 peak.9969,zerooverload,24goals,945stallplayerseconds. This is not the final900s stress qualification.

Quiet-state diagnostic on44cold selfplay: many stalled nearby pucks had center gap.13–.16m and actor std.02–.04. Prior exploration only included stationary gap>.18m, excluding those failures. Added opt-in --quiet-exploration at half the recovery floor, still load-aware and training-only. Effective scale is identical in rollout and PPO likelihood. All default/inference behavior unchanged.

46 starts45step654524416 for32M with that quiet exploration and training-only self-imitation. Collected12,288 hard outgoing trials with temporally persistent Gaussian perturbations of the learned actor;47first-leading-contact then capture successes before first possession loss yielded2,236prefix examples. No trajectory planner/tactical teacher. Mix with15,310prior learned successful requested-shot examples approximately50:50, replaylossweight30. The eventual actor remains one deterministic45input/6output network. Collection provenance and exact source hashes inpossession46/recovery-replay.json andmixed-replay.json. This is a development experiment, not a qualified candidate. Final selection must evaluate its exact weights against fresh seeds and physical/RMS checks. No hardware activated.

### Selection gates and gradual curriculum, ~05:00 UTC

42 final665534464 failed the fresh15-minute hot audit:peak1.062677,399.24summed overload seconds,first event500.24s. Acceleration/guard reproduction passed exactly, zero intervals above cap+0.1%. Excluded from production despite skill improvements. In the same expanded continuous audit, baseline33 reached1.00459 for0.52summed seconds; prior short/resetting evaluations did not establish universal RMS safety.

45 early4M650461184 fresh preliminary results: preparation1423/2048vs809baseline,hot1385vs727,stationaryfast834/1536vs774,incomingfast943vs940. Continuous8×180s48goalsvs12,stalls311vs504playerseconds. Strict recovery22vs24controlled: not improved. Its full fresh900s audit is pending atqualification-early4m.45finished670515200, with full exact-final qualification scheduled automatically.47 has its own exact-final qualification scheduled.

46 rejected after~7.6M(final662126592). At4M658587648, first outgoing-leading contacts increased34vs9baseline, but captures48vs41 and requested-fast conversions18vs30, while first requested-fast shots288vs610. Watcher incoming requested-fast123/384vsparent323 and stationary126vs223. Too much loss of established shooting; no production selection. Rejected-branch watcher stopped. Data and reports retained for honest provenance.

47 returns to45step662650880, adds only quiet exploration and angular recovery starts +/-120degrees,85%nearbyangular/15%hardtrailing. Starts are too abrupt a jump for reliable acquisition, so a separate **gradual sequence**48–51starts from45final670515200 at30/60/90/120degrees respectively,16Madditional perstage,8Msnapshots. Same learned shot replayweight10; NO noisy recovery-imitation data. Driverlogs/neural-player/requests/possession-curriculum/train_stages.py,PID3527830; phase48trainer3527831. WIPaliasneural-possession-gradual-wip. Each stage resumes predecessor's exact final checkpoint. STOPsentinel in curriculum directory prevents subsequent stages. No firmware/deployment-default changes or hardware activation.191regression tests passed after curriculum/exploration changes; node syntax and diff whitespace checks passed.

### Full-game heat distribution correction, ~05:10 UTC

45early650461184 failed the fresh900s stress:peak1.13712,704.64summed overload seconds.45final670515200 failedpeak1.15015,24.64seconds. Both excluded. A300s development pass was insufficient; selection continues to require full900s exact-checkpoint testing.

Found a concrete distribution gap: `warm_start_max` and correlated heat randomized only short skill drills/shutdown resets. Initial full games retained the thermal constructor's much cooler fast0–.65/slow0–.30 levels. Their physical states also reset every120s, while the new evaluation correctly continued quiet possessions past several minutes. Heat itself already persisted; the bug was exposure to hot **long-game states**, not a heat reset at goals.

Added opt-in `--warm-game-start`: apply configured heat distribution on each environment's first game initialization only, preserve it at later goal/truncation resets. Regression test verifies correlated hot initialization and exact heat preservation on subsequent reset. Upcoming49–51 use this flag,900s full-game horizons,and35%fullgames while continuing60/90/120degree recovery starts.48continuesunchanged30degree stage. Replaced the curriculum manager without stopping48's independent trainer: new managerPID3609237,scriptpossession-curriculum/continue_stages.py,waiting48then49–51. Originalmanager3527830terminated. Source and run metadata retain actual flags for each stage.

Angular diagnostic baseline45final(seed20264148,128trials each):0degree36controlled/23controlledrequestedfast;30degree37/23;60degree13/8;90degree8/6;120degree8/4. This supports using small difficulty steps. These angular metrics are curriculum diagnostics, not the hard behind-start recovery benchmark. CurrentWIPneural-possession-gradual-wip verifiedHTTP200 withcontinuousrallies; no final candidate selected or hardware activated.

### Refinement and qualification, ~05:40 UTC

49early694763520 and52early674578432 completed development8x900s(seed20263141,.95initial,1.3gain):peaks.966535/.965309,zerooverload. Exact49final702758912 subsequently failed independentfresh900s(seed20263545):1.10608peak/20.96overloadseconds. Do not generalize an earlier snapshot's heat pass to later weights.

52final686637056 passed that fresh hot audit(peak.994187,zerooverload), preparation1585/2048vs809baseline,direct1008/1024vs1009. But **do not promote**: incoming requested-fast767/1536vs940baseline,hardrecovery22controlledvs24 andrequested-fast520vs646,selfplaystalls689vs504playerseconds,crossrandomaggregate14:18. It fails overall play despite defense/heat improvements.47final686768128 also failedfreshhot(1.153147peak,846.32overloadseconds).

53 is now the main additional refinement,trainer3965588,watch3965589,prep3965590,strict3965591. It starts49final702758912, widens512to1024 function-preservingly,32Madditional,save4M,lr1.5e-5. Verified initial actions EXACTLY match49 on914 real replay observations(maxdifference0). Offline production preflight of its INITIAL model passed45inputs/6outputs,50Hz,12m/s60m/s2,decoder+guard1.44msmedian; no camera/master/hardware activated. Final weights still need their own preflight and qualification.

53data:65,536 explored hard recovery trials from52earlyactor produced340first-leading-contact then measuredcapture successes(17,882prefixexamples), plus47older successes(2,236examples). Separate full-skill collection from49final produced2,015firstrequestedfastsuccesses(left592,right672,straight751),88,620observations including up to2.56s BEFORE successful shots, including far-half incoming approach. Former320ms/own-half-only preservation omitted reception setup. New mixeddataset roughlybalances fullsuccessfulshot sequences and recoveryprefixes by resampling; sources/hashes saved. Finalruntime remains one network, no tactical controller/teacher.

Training reference now optionally supports `--skill-reference-incoming-only`, covering FAST INCOMING observed pucks even before they cross the half; quiet states and stationary/outgoing far-half preparation remain free. Reference maxload.85 avoids preserving unqualified hot behavior.53uses learned49referenceweight100 andmixedreplayweight10. Other runs' default reference behavior is unchanged. Regression covers far incoming masking and absence of teacher gradients.

Reward incentive correction for53andupcoming54/55: stationary timeouts600(vs120); delayweight6(vs2); recoverycapture600(vs240). Withgamma.999/actiondt.02,the old maximum waiting cost.4/step has discounted upperbound400<goalconcession600, allowing stalling to be preferable. Newmaximum1.2/step gives1200upperbound. Stationary timeout penalty now excludes goals already penalized separately, avoiding double-charging conceded goals and restoring the wait incentive. Tests pass (193full-suite before latest incoming-only mask refinement;23targeted passed after it). No physical deployment defaults changed.

Pure gradual curriculum continues51at120degrees(trainer3921838). A following manager3918577 waits51thenruns54at150degrees and55at180degrees,16M each,withhot900s/.35fullgames and stronger above rewards;55reduces easy fraction to.6(remaining.4original trailing starts). Driverpossession-curriculum/complete_arc.py. Maincurrentmanager3609237runs49–51;original3527830stoppedpreviously. Qualificationmanager3783206handles49/51/52;manager3918578handles53/55aftertraining. All simulation only. WIPgradualalias remainsvalid;53hasneural-possession-refinement-wip. No final balanced productioncandidate selected yet.


### Follow-up qualification and recovery experiments, September26 ~06:15UTC

All195/196 relevant regression checks passed after the first-contact cushion reward and per-game opponent-style curriculum respectively; trainer CLI help was also executed (caught/fixed an argparse percent-sign help-string failure before55 actually launched).55 is now running normally, with source/flags/checkpoints recorded.

51 final735002624 improved fresh preparation1599/2048vs809baseline, hot1479vs727; stationary requestedfast839/1536vs774; incoming935vs940. Continuous8x180s selfplay40goals and186.8stallseconds vs12goals/503.6baseline. It passed fresh8x900s hot selfplay(.995057peak,zerooverload), but the initial summary **incorrectly attributed the old opponent's cross-straight overload to it** (corrected below). The separate extra-seed hot audit was stopped once this failure was established. All cross-opponent load channels must be inspected, not merely the dedicated selfplay heat audit.

53 final734871552 rejected: stationary requestedfast400/1536vs774baseline and incoming342vs940. Larger capacity plus noisy recovery self-imitation did not preserve shooting. Hard recovery24controlled/6controlledrequestedfastvs24/15baseline. No promotion.

54 final751124480 is receiving full qualification.55 continues54 for24M using .5cold-short-practice fraction, all-six recovery exploration(.5floor), and extra quiet-state exploration when tanh saturates. Full900s games retain hot initialization. This explores arrival duration and acceleration, whose original recovery standard deviations were only~.0325/.0183 with median~25m/s2 chosen effort. Inference remains deterministic and unchanged.

56 starts51 for32M with these exploration changes plus150 first-leading-contact cushioning credit proportional to actual speed reduction; only first contact of outgoing recovery exercise can earn it. Capture still separately earns600 and exercise continues to a shot. Test verifies no behind-contact/speedup/repeated-contact reward. No runtime tactical actions.

58 is a further32M branch from51. It removes conflicting slow-outgoing trailing-approach examples from successful-shot preservation:9,631of15,310 removed,5,679retained. Velocities are explicitly converted from neural normalized units by*6; selection is outgoingvy>.05m/s,speed<2m/s,paddlebehindalongvelocity,centergap>.13m.57 was stopped early because its first experimental filter missed this unit conversion; never a candidate.58 also exposes50%fullgames to one fixed opponent shot style per game (uniform left/right/straight), with random styles in remaining games. This teaches repeated straight-shot pressure that revealed51's heat failure. Adds reviewed33 to existing neural opponent references. It remains one learned player with unchanged inputs and six actions. New per-row opponent request distributions preserve the existing shared-distribution RNG path; test verifies style persists across possessions and own requests still vary.

Correction to earlier stall denominators: reachable-stall sums count only the puck-owning side, so divide by total GAME time (e.g.7200s for8x900), not twice that amount. Heat passing by prolonged idle behavior is insufficient. No final candidate selected. No physical hardware activated; no firmware or deployment-default changes for these experiments.

### Correction: attribute cross-play loads to the evaluated actor (~06:14UTC)

51 was incorrectly rejected by taking max load across BOTH players in cross-straight-blue. The old opponent, red, reached1.133109; candidateblue peaked.896368 withzerooverload. Across all four cross suites the candidate hadzerooverload(peaksrandomblue.978393,randomred.919494,straightblue.896368,straightred.916014). Thus51 remains a viable general-play fallback after its.995057/zero900s selfplay pass. SUPERSEDED-ASSESSMENT.md retains the mistaken reasoning; ASSESSMENT.md corrects it. The extra independent hot audit(seed20265926) was restarted. This attribution error does not affect the other rejected hotSELFPLAY runs, where both sides use candidateweights, or53's shooting regression. Fixed opponent style diversity remains useful coverage but is not evidence of51overloading. Recovery still unresolved.


### Exact candidate gates and59 load-margin refinement (~06:25UTC)

51's second independent8x900s hot selfplay(seed20265926) genuinely failed:peak1.014561,3.20summed overloadseconds. Candidate-side cross-play was safe, but this additional same-actor selfplay failure excludes735002624.54final751124480 likewise failed hot selfplay1.129277/24.90s and cold180s1.002259/.28s. These are true candidate failures, not old-opponent attribution.

55 completed775241728 after24,117,248additional transitions in1123s. Fresh results so far: preparation1681/2048vs809baseline,hot1512vs727,stationary requestedfast837/1536vs774,incoming1008vs940. Strict recovery34controlled/18controlledrequestedfastvs24/15: modest, still inadequate as a reliable get-ahead maneuver. Continuous8x180s selfplay28goals,129.6stallseconds/1440game-seconds vs12goals/503.6baseline;peak.971488/zerooverload. Full hot/crossqualification plus second independent900s hot audit are pending.49early694763520 is also receiving exact full qualification; its prior development hot pass alone does not qualify it.

56 continues with cushion credit;58 with correctly filtered replay and fixed opponent styles.59starts58's immutable16Msnapshot751255552 for24M.59 keeps the same single512-wide actor and60m/s²cap, but terminates simulated learning episodes at.98modeledload (real modeled shutdown remains1.0 in qualification), warmmax.97,shutdownpenalty6000,loadweight8,correlatedwarmfraction.5,and50%fullgames. This teaches a reserve below shutdown rather than claiming that one passing900s seed is sufficient. Trainer311134,watch311135,strict311136,prep311137,qualifier311138; sources/flags are snapshotted in run59. Currentrecovery WIPaliasneural-possession-recovery-wip points to58; load refinement hasneural-possession-load-refinement-wip. No physical motion or defaultcapchanges.

New ai/bin/summarize_neural_qualification.py attributes cross-play goals/load to report.candidate_side and separately reports old opponent load. A regression test covers both colors, absent-role rejection, and same-actor selfplay;196previous relevant checks+1new test passed. A local packaging script inpossession-curriculum/package_candidate.py refuses unfinished qualifications, changed checkpoint hashes, candidate load/cap failures, shooting regression, weaker preparation or worse reachable stalls. It additionally requires an independent secondhot audit. Packaging has not run: no final candidate yet. Production loader for51 was checked entirely offline (45inputs/6outputs,50Hz,1.42msmedian decoder+guard); final packaged weights will receive their own check.


### Sustained cold-start audit and withheld package (~07:10 UTC)

IMPORTANT: `runs/possession-20260926/agent.pt` and `possession-20260926-fallback` are **WITHHELD**, not final production selections. Their exact source49early694763520 passed three8x900s hot-start(.95) suites (peaks.966535/.988383/.994085), but failed the added sustained COOL-start(.3) conservative-gain1.3 audit, seed20266026:peak1.059487,600.72overloadseconds. Its replay metadata now saysWITHHELD and both run.json files have deployment_ready=false/simulation_only=true. `ai/POSSESSION_RESULTS_20260926.md` is markedDRAFT/WITHHELD and must be rewritten around the eventual winner. No final run command has been handed off after this failure. No physical hardware was activated.

This cold-to-hot exposure is a material selection requirement, not interchangeable with initially-hot tests. `package_candidate.py` now requires --cold-audit and checks its exact checkpoint,8x900s duration,.3initialload,1.3gain,zerooverload/capviolations alongside two distinct hot seeds. Reaction defense preservation is also checked on direct/varied-bank/hot-bank suites; source hashes and candidate_side attribution remain mandatory.

55 final775241728 secondhot failed1.116578/37.18s despite firsthot pass and strong preparation/shots.56final767115264 firsthot.999441/zero but secondhot1.044781/18.54s.58final767115264 firsthot1.016969/7.18s,secondhot1.194318/827.28s.59final775372800 passed twohot(.988400/.970645) but incoming requestedfast891/1536vs940 and selfplaystall489.1s/1440vs503.6: not good enough overall.

60 starts59step759382016 and finishes775503872 after16M. New --hot-effort-exploration2 broadens raw effort noise in hot TRAINING observations, including saturated tanh outputs; other action dimensions unchanged and PPO uses identical likelihood. Deterministic inference is unchanged. It passed twohot(.988872/.982356) and sustainedcold(.980034/zero) audits. Preparation1767/2048,hot1641;stationaryfast829/1536,incoming957;selfplay40goals/167.9stallseconds. But variedbank722/1024vs787baseline,hot754vs774. Stronger direct preparation traded away bank defense. Long cold games still stall4514.9of7200seconds.198regression tests passed before subsequent curriculum additions.

61 restores bank responses through training-only learned-reference preservation (49early694763520,weight50,incoming-only,maxload.9), starting60final for8M. Final783630336; no runtime teacher/controller. Preparation1711/2048,hot1550,direct1005/1024,bank789,hotbank795;stationaryfast811/1536,incoming959. Strict recovery26controls/14controlledrequestedfast vs24/15baseline. Additionalhot passed.973465, sustainedcold passed.946673. However standard8x180s selfplaystall598.8seconds exceedsbaseline503.6, despite24goals. Thus still not a complete balanced candidate. Full reports under61/qualification-final,qualification-extra,qualification-cold-sustained.61 had restored the specific bank loss; this is not proof that its overall play improved.

### Longer continuous exposure and explicit slow recovery curriculum (~07:20 UTC)

A16Mstage at1024envs gives only~315simulatedseconds per environment before the next run recreates all physical states, even with a nominal900s horizon. Later exposure also suffered exploratory shutdown resets. In62, completed game episodes averaged~33s and all those completed episodes were shutdowns; unfinished long games are not represented in that statistic, so this is NOT the duration of every environment.

62 added: initial cold_game_start_fraction.5 (onlyfirstgame initialization; ordinary resets preserve heat); slower outgoing practice.1–.45m/s; once-only150get-ahead progress credit before firstcontact for a trailing start (no actor target/action/input added); and20stationary failure replay starts, combining6old states with14reachable quiet failures from49/60sustainedcold. Real positions,puckvelocity,priorarrivalaction,request are retained; the existing reset zeros paddlevelocity/queuedprofile, so these are diverse practice starts, not exact full-state resumes. Tests cover once-only/no-free-leading/no-after-contact credit, speed bounds, and cold initial heat preservation.62 used512envs to lengthen exposure, but extra recovery/thermal noise floors still applied to fullgames. Stopped/saved795492352 after~11.86M to fix that schedule; all62watch/qualification parents stopped, no final promotion.

CURRENT:63 starts62final795492352 for48M,512envs,4workers,rollout128,epochs2,lr8e-6,save8M,90minbound. It adds --practice-only-exploration: extra recovery/quiet/hot-effort floors apply only to shortdrills. Fullgames retain the learned Gaussian variance, allowing sustained trajectories closer to deterministic deployment. The training-only kind mask comes from existing critic context; it never enters the actor and is applied identically to rollout/PPO likelihood. Defaults preserve older experiments. Expected~31minutes of simulator time per fullgame environment over this run, before accounting for early shutdowns; ordinary goal heat stays continuous. Do not restart this run merely at checkpoint boundaries.

63PIDs:train992894,watch992895,strict992896,prep992897,fullqual992898,extrahot992899,coldaudit992900. Source and flags are snapshotted under runs/_neural-player-requests-possession63. Full qualification plus independent hot20265926 and cold20266026 audits wait for its immutable final checkpoint. WIPaliasneural-possession-long-game-wip now tracks63. Final selection remains pending. The latest targeted49tests passed after practice-only-noise changes; earlier60targeted passed for the slow/leading/cold curriculum. No physical defaults or hardware activation changed.

### September26 07:43UTC — continuous exposure and preservation conflict

63 continues62step795492352 for48M with512environments, so individual games
can actually reach900seconds. It restricts the additional recovery/quiet/hot-effort
exploration floors to practice slots; full games retain the learned Gaussian.
The same mask is used in rollout and PPO likelihood; runtime inference is unchanged.
At8M, preparation1736/2048 vs reviewed33's809 and incoming requested fast
shots1016/1536vs940. Fresh strict recovery34controlled/19controlled-fastvs24/15
remains weak. That exact checkpoint803553280 was rejected for promotion:
primary sustained hot audit peaked1.009878 with3.3overloadseconds, despite
independent hot and cool audits passing (.999537 and.995868). It is not a finalcandidate.

The old provisional49 package/report/replays are explicitlyWITHHELD after the
sustained cool audit failure; the contradictory provisional results document was
archived and replaced with current selection status. No physical default changed.

Inspection of the58 preservation dataset found2472slow outgoing examples among5679
retained states,2453of them with the paddle behind the puck. The earlier filter
removed distant approach examples but retained these near-contact punching actions,
which conflict with cushioning/getting around the outgoing puck.64 starts from
63step819675136 with the same settings and3207remaining preservation examples:
all slow outgoing states (vy>.05m/s,speed<2m/s) removed.32Madditional transitions;
63 continues independently for a comparison. No action target is added at runtime.
The dataset filter is a hypothesis under test, not evidence of recovery success.

202 relevant regression tests pass after the continuous-game exploration change,
including rollout/PPO variance-mask parity and deployment isolation. Finalexact
weights must still pass shooting, banks, preparation and sustained hot/cool audits.

### September 26 final selection — possession-20260926-v2

Selected run63 step827736064, 32M into the continuous 48M run, pinned byte-for-byte
as `runs/possession-20260926-v2/agent.pt`. SHA256:
`7247a6d7351a8891a1df0973d5b62fe4554f705f3be123c35d6217f614c28e3c`.

Five sets of eight 900-second games at thermal gain 1.3 passed: peak loads
0.974930, 0.961298, 0.983651, 0.958838, 0.966109; zero overload and firmware
acceleration-cap violations. Preparation saves 1744/2048 versus 809; hot saves
1564 versus 727; direct saves 1013/1024 versus 1009; bank saves 785 versus 787;
hot bank saves 792 versus 774. Stationary requested fast shots 806/1536 versus
774; incoming 965 versus 940. Short self-play: 47 goals and 124 stall seconds
versus 12 goals and 503.6 seconds.

Strict outgoing recovery 32/1536 versus 24, and controlled requested fast shots
18 versus 15, remain weak. Controlled requested fast shots from stationary
starts 588 versus 637 and incoming starts 898 versus 918 are explicit tradeoffs.
Long hot/cool games still stall substantially; the primary cool-start suite
accumulated 5362.2 stall seconds over 7200 game seconds. Do not label all requested
behaviors solved.

Run63 final step843530240 passed the three declared thermal audits but increased
short-game stalls to 260 seconds and strict control returned to 24/1536. Run64
final step851722240 failed both sustained cool seeds: primary peak 1.000650 with
0.26 overload seconds, confirmation peak 1.166670 with 21.44 seconds. No rejected
checkpoint is promoted. Development watchers for runs63/64 were stopped after
selection; completed records remain local.

Final offline package preflight passed: 45 inputs / 6 outputs, 1.28 ms median
at 50 Hz, 12 m/s and 60 m/s² session caps. No hardware I/O or motion was started.
Pinned replay `neural-possession-candidate-20260926-v2.json` serves 9000 frames
covering 180 seconds, with the same selected actor on both sides, qualification
seed 20263543, continuous rallies, and metadata matching the packaged SHA256.
HTTP 200 verified. All 202 relevant tests, UI syntax, and diff-whitespace checks
pass. Measured comparisons and limitations are in `ai/POSSESSION_RESULTS_20260926.md`.

### Continuing possession work after the pinned v2 selection

The completed selection turn made progress but did not complete the broader
learned-possession objective. V2 remains pinned and no hardware is authorized.

Paired diagnostics under `logs/neural-player/requests/possession-diagnostics`
use outgoing speeds 0.15/0.3/0.6/0.9 m/s and initial paddle placements behind,
beside, and ahead. V2 controlled 126/512 leading starts and converted 96; from
behind it almost always struck immediately rather than going around. Starting
in front therefore does not fully resolve the weakness. On a separate stochastic
ablation, learned variance gave 57 controls / 41 conversions from 256 leading
starts; broad extra floors gave 58 / 20, and geometry-restricted floors 61 / 12.
The proposed geometric exploration mask was NOT added: it failed to improve
follow-through. No runtime rule was introduced.

Run65 used all-leading recovery starts with ±0.15-radian variation, 0.4–0.9 m/s
pucks, learned exploration variance, capture bonus 150 (formerly 600), 25% games,
and the filtered preservation dataset from64. It completed 8M steps at835796992.
Final paired leading control was141/512 and conversion115, versus126/96 for V2.
Separate requested-fast shot counts: stationary432/768 vs409; incoming478 vs451.
The 4M snapshot was worse, so intermediate results must not be mistaken for final
learning behavior. This remains a short developmental stage, not qualified for
production or evidence that getting around the puck is solved.

Run66 continues65final for24M, with8M saves. New optional training-only
`--recovery-cushion-signed` distinguishes mildly and severely overpowered first
leading contacts using a bounded smooth reward. It leaves old defaults, actor
inputs/actions and runtime inference unchanged. First-contact credit cannot be
repeated or collected outside the recovery drill.203 relevant tests passed.
Its evaluator measures paired leading starts, requested shots and self-play,
and publishes `neural-possession-cushioning-smooth-wip.json` at checkpoints.
Broaden the approach curriculum only after verifying control AND follow-through.

First run66 checkpoint843857920 improved the paired leading diagnostic to153
controls and125 conversions /512, versus141/115 for65final and126/96 for pinned
V2. Separate requested-fast counts429stationary/467incoming versus409/451V2;
short self-play37goals/119.3stall seconds versus20/75.8V2 on seed20262935.
These mixed results do not qualify a new candidate. Keep the easy curriculum
running; do not broaden based on training reward alone. Sensor tracing ruled
out a reset-velocity explanation: the actor observes the correct outgoing
velocity before actively moving into the puck too hard. No sensing patch made.
Run66 remains active (trainer1802818, evaluator1802819); inspect live processes
and its status/checkpoints before any restart. Target24M additional transitions.

Run66 completed 24M at step859848704. Final leading-start control was 175/512,
conversion 140/512 (V2:126/96). Requested-fast shots on the separate768-request
suite were434 stationary and472 incoming (V2:409/451). Short self-play had34goals,
0.6reachable-stall seconds and peak load0.970533 over eight120-second games;
V2 on that seed had20goals/75.8stall seconds. These remain development results.

Added angular-recovery-v2 diagnostics: both approach directions at0,±45,±90,
±135,180degrees, speeds0.3/0.6m/s. They separately count any controlled conversion
and the complete control→requested on-target ≥6m/s shot. Run66's16M checkpoint
improved the collinear case35→45controls and23→32controlled-requested-fast/128;
±45degrees remained one or zero controls. This is evidence of a narrow learned
cushioning skill, not general getting-around behavior.

Run67 starts66final and changes only the initial-angle spread to±0.65radians
(about37degrees), for16M more transitions. Trainer2012285; read the process
metadata for the evaluator PID. Its checks include the angular diagnostic,
leading starts, normal shots, full strict recovery-v2 and continuous self-play.
Do not restart a live run or promote from short-game load results. Production
package possession-20260926-v2 remains unchanged; no hardware authorized.

### Run67 outcome and targeted exploration experiment

Run67 completed 16M at step875905024. It retained leading control169/512 and
conversion137 (V2:126/96), but broad strict recovery remained only18/1536 controls
and10 controlled requested fast shots (V2:32/18). Requested fast returns improved
to688 versus652. Separate ordinary fast-shot counts433/469 versus409/451.
Short self-play29goals/85.8stall seconds versus20/75.8 for V2. No promotion.

The finer angular grid (0,±15,±30,±45,±60,±90,±135,180 degrees) confirms that gains
are concentrated near collinear leading starts. At the8M checkpoint, ±30-degree
control was4/256 compared with16/256 for V2, despite improved head-on performance.
Do not substitute the easy diagnostic for generalized recovery. A consolidated
comparison is generated by `compare_recovery_curriculum.py` under the curriculum
log directory and retains both improvements and regressions.

A frozen run67 exploration ablation at ±30 degrees found learned variance gave
9 controls /2 controlled requested fast shots from256 trials; a whole-drill .5
floor gave23/3, pre-contact-only noise23/13, and an initial80ms window23/12.
The initial window better preserved head-on follow-through than the pre-contact
mask, and can be derived from existing stored critic context without extra state.

Implemented optional `--recovery-exploration-seconds`: extra recovery exploration
is restricted to the beginning of receiving drills. The identical helper derives
the rollout and PPO masks from stored context; no actor input/action or inference
change. Full games and stationary/defense drills retain their learned variance.
204 relevant tests pass, including phase boundaries, task isolation and minibatch
ordering. The production weights and caps remain unchanged.

Run68 starts67final, same ±0.65-radian curriculum,16M transitions/8M saves, with
recovery exploration .5 only for80ms. Trainer2224697, evaluator2224698. Inspect
live handles before restarting. Its angular evaluator uses the finer grid plus
strict recovery, ordinary shots and full self-play. It is experimental, not a
production selection. Goal remains active; no hardware authorization or motion.


## September 26: possession curriculum, runs 65–69

Production remains pinned to `neural:possession-20260926-v2`; see
`POSSESSION_RESULTS_20260926.md` for the qualified comparison and command.
No hardware was enabled. The continuing research has not solved general
outgoing recovery, and later checkpoints are not automatically better.

Run 67's final model improved delayed-shot preparation to 1824/2048 saves
(1678 hot), against pinned v2's 1744 (1564 hot). Its independently seeded
15-minute hot-game audit nevertheless reached load 1.007687 for 1.0 seconds,
so it is withheld. Cold sustained play stayed below the limit but stalled
for 4709/7200 game-seconds. Broad outgoing recovery was only 18/1536 controls
and 10 controlled requested fast shots, versus pinned v2's 32 and 18.

Run 68 tested extra exploration confined to the first 80 ms of receiving drills.
The rollout and PPO likelihood use the same context-derived mask; this adds no
actor input or inference-time behavior. The 204 relevant regression tests pass.
The completed 16M-transition run improved narrow leading catches but failed to
generalize: its final broad recovery result was 13 controls and 9 controlled
requested fast shots. Short self-play had 39 goals and 1.3 stalled seconds out
of 960 game-seconds, which does not substitute for broad recovery or sustained
thermal qualification. Replay: `neural-possession-approach-burst-wip.json`.

Run 69 tests successful neural trajectory imitation rather than another identical
PPO continuation. It collects oblique recoveries from run 68's exploration,
accepting only leading contact, measured control, and a requested on-target shot
at least 6 m/s before first possession loss. It fits these examples into pinned
v2's actor while retaining its learned outputs on other skills and hot states.
Separate fixture batches are held out, then each frozen model is evaluated on
new physical starts, broad recovery, ordinary shots and self-play. No planner
provides an action; the policy architecture remains the same single network.
Collection, training and evaluation plans/results are under
`logs/neural-player/requests/possession69/`.


### September 26: completed imitation ablations and active run 72

Run 69's corrected, workspace-filtered supervised update600 retained fast shots
(430 stationary,467 incoming out of768 requested trials) and improved the narrow
leading-start catch test to154/512. Full outgoing recovery still regressed to
23/1536 controls and16 controlled requested fast shots (pinned v2:32 and18).
It is a research initialization, not a qualified production replacement.

Run70 added previously collected trailing-start neural success prefixes but
strongly degraded shooting: its final2400-update model scored304 stationary and
260 incoming requested fast shots, versus pinned409/451. General control stayed
32/1536, so it is withheld. Run71 tests a stronger preservation loss and frees
slow post-cushion states; its evaluations remain development-only. None of these
supervised models has been promoted or used on hardware.

The reward probe uses the actual run68 training settings. The exploration
sampler produced no catches from512 trailing starts. Leading starts averaged
-83 return, behind volleys-53, and controlled leading sequences+315. These are
separate initial-state cohorts, not a causal comparison, but show how costly
failed catches can inhibit learning. Saturated action outputs were checked and
were uncommon in the successful trailing dataset; no action-space change was
made on that hypothesis.

Run72 is a32M-transition PPO continuation from corrected run69 update600. It
mixes leading/trailing outgoing starts50/50 at0.15–1.2m/s, restores30% fast
incoming receiving exercises, raises once-only recovery capture bonus150->400,
and broadens practice-only initial exploration to0.9 for160ms. Qualified v2
fast-incoming outputs and successful held-in recovery trajectories provide
training-only preservation. Snapshots and diagnostic replays are published
every8M transitions. Fresh final-comparison seeds were declared before choosing
any run72 snapshot. Full thermal and defense qualification remain mandatory.

Current process records: logs/neural-player/requests/possession72/*-process.json.
The pinned production package and its SHA remain unchanged. The evaluator now
labels recordings generically as a single neural network (not every checkpoint
was trained only by PPO) and includes the exact checkpoint SHA. Three related
qualification/watcher tests and byte compilation passed after that metadata edit;
204 relevant tests passed after the earlier training exploration change.


### September 26: physical-history ablation (run 73)

Run 72's first 8M snapshot (835796992) preserved fast shooting: 417 stationary
and 472 incoming requested fast shots /768, versus pinned v2's409/451. Broad
recovery was33/1536 controls and19 controlled requested fast shots, versus32/18.
That is only a small development-set change, not proof of a meaningful gain.
It made30 goals and stalled121.8/960 game-seconds in short self-play. No sustained
thermal or full-defense qualification has yet been run on this checkpoint.

Run 73 is a paired architectural experiment with run72's curriculum/settings:
16 physical frames replace the one-frame input, using existing supported history
code. The hypothesis is that past motion helps separate a newly cushioned puck
from settled possession; this has not been established as the cause of failure.
No phase, capture, target-trajectory or task label enters the actor. The expanded
network initially preserves the previous policy's actions within6.5e-7 because
older-frame input weights start at zero. Replay histories were verified against
the exact deployment history class without crossing trajectory boundaries.

Run73 trains16M transitions with8M snapshots; run72 continues toward32M.
The actual training run's agent_initial.pt passed the offline play launcher check:
675 inputs,6 arrival outputs,50Hz,1.27ms median including decoder/guard,
12m/s and60m/s². This only establishes adapter compatibility and latency.
Direct research initialization artifacts were correctly rejected by the launcher
because they lacked recognized deployment metadata; that gate was not relaxed.
No hardware was accessed. The qualified v2 package remains unchanged.

Process records and exact commands are under possession72 and possession73 in
logs/neural-player/requests. Run72's fresh-seed baseline qualification completed;
use those declared seeds for the eventual selected checkpoint. Development
comparison JSON covers runs65–73. Runs70/71 are withheld for shooting regressions.


### September 26: completed runs 72/73; active runs 74/75

The previous goal turn made concrete progress: completed both training runs and
paired diagnostics, reproduced a sustained-load failure, implemented and tested
an optional temporal-difference input representation, and started two focused
follow-ups. The goal remains incomplete because broad learned recovery is weak.
There is no external blocker. Do not mark it complete based on narrow drill gains.

Run72 final859783168: leading control159/512, broad control26/1536, controlled
requested fast shots14, stationary/incoming requested fast shots387/445 out of768.
Short self-play:38 goals,0.5 stalled seconds out of960, peak load.9383. Its stronger
middle843857920 nevertheless failed sustained cold games: peak1.024283 and9.54s
overload. Hot games passed at.977014 but stalled5666/7200 game-seconds. No model
from this branch is promoted. The cold failure was reproduced at exactly348.82s
(red,game5,seed20266027). The diagnostic trace shows recurring high-acceleration
bursts around.95–1.0 normalized load, not sufficient cooling between them.

Run73 final843792384: leading161/512, broad22/1536, controlled requested fast16,
stationary/incoming shots402/459. Short games:36 goals,1.5 stalled seconds,
peak.9790. At its first8M snapshot, replacing actual history with repeated current
frames improved leading control/conversion156/118->171/137. Raw history has not
shown useful generalization; this is not evidence that all temporal information
is unhelpful.

Run74 repeats run73's initialization, seed20262573, curriculum and16M budget,
with past physical frames represented as differences from the current frame.
The actor still receives only physical observations and the optional shot request.
`PhysicalHistoryLinear` preserves the raw path exactly for old checkpoints.
Explicit training migration analytically transforms first-layer weights, preserves
actor/noise/critic predictions and resets optimizer moments. A checkpoint buffer
restores encoding automatically through the existing inference loaders.
207 relevant tests pass; the21 player tests were rerun after strengthening the
learned-noise migration assertion. The actual run74 initial checkpoint passed
an offline launcher check:675 inputs,6 outputs,50Hz,1.34ms including decoder/guard,
12m/s and60m/s². No hardware access or default-limit change occurred.

Run75 resumes run72 middle843857920 to learn extra thermal reserve. Only the
training trip threshold changes.98->.95, paired with warm-start maximum.97->.94
to prevent immediate episode failure. Physical caps remain12m/s and60m/s²;
audits still use RMS threshold1.0. Every8M snapshot gets both hot and cold8×900s
audits from the independent background audit manager. This is learned policy
training, not a runtime thermal throttle.

Current process records are under logs/neural-player/requests/possession74 and
possession75. Their training/evaluation PIDs were verified live. Run75's separate
audit-manager-process.json records the sustained-test watcher. The comparison
script covers runs65–75. Use run72's predeclared fresh qualification seeds after
choosing a candidate on development evidence. The production package remains
neural:possession-20260926-v2, SHA7247a6d7351a8891a1df0973d5b62fe4554f705f3be123c35d6217f614c28e3c.

Run74 completed without a broad recovery improvement. Its8M/final leading
controls were151/174 out of512; strict outgoing controls38/26 and controlled
requested-fast completions18/13 out of1536 request trials. Stationary/incoming
requested-fast shots were397/465 then401/439 out of768. Delta history allowed
more actor updates per rollout than raw history but has not established better
play. Neither memory experiment is promoted.

Run75's earlier training trip (.95) reduced but did not eliminate sustained
cold-game overload. First8M/final hot peaks .992348/.992089 passed; cold peaks
1.019654/1.009252 failed with9.86/3.44 aggregate overload seconds. The first8M
full qualification completed; preparation1698/2048, hot1492/2048, both below
pinned v2's1744/1564. Short-game activity is insufficient for promotion.

The exact failed run75 first-checkpoint trajectory was reproduced (red game6,
first trip288.88s). In428 reference-preservation-eligible frames with observed
load>=.94, the v2 teacher requested high effort120 times; the student121, with
mean action MSE7.21e-5. Eligibility uses the exact float32 Torch training mask.
This suggests conflicting objectives near the .95 training cutoff, not proof
of causation. Run76 starts from run75 final859914240 and changes only the
preservation maximum load .99->.80, with16M additional transitions and both
hot/cold8×900s audits at every8M snapshot. Full evidence is under possession76.

Run77 tests protection against destructive skill interference. It starts afresh
from pinned v2, widens512->1024, adds16 physical history frames with temporal
differences, and freezes the original actor features/output connections during
training. New units learn corrections; the saved policy remains one dense NN.
There is no runtime gate, teacher, planner, or tactical controller. Critic is
fully trainable. Gradient masking plus post-Adam projection prevents old optimizer
momentum from moving protected entries. Resume inherits the frozen width;
initial exploration rescaling happens before the protected snapshot is taken.

Run77's mixed recovery replay has65406 valid training frames, excluding batches
24–31 and invalid workspace trajectories. It mixes complete angled catch/shot
sequences with old trailing-start control prefixes; the latter are not claimed
as completed fast shots. The3207 older precision anchors are relabeled with the
exact pinned v2 policy and repeated4x. Recovery histories match PhysicalHistory;
anchor histories repeat the current frame because originals were unrecorded.
The32M run otherwise uses run76's curriculum and thermal settings, plus the
expanded architecture and replay. Snapshots every8M have matched skill/game
metrics and sustained hot/cold audits. No automatic promotion is configured.

209 relevant tests pass, including prefix protection under nonzero Adam momentum,
checkpoint reload and inference deepcopy. The23 player tests passed again after
moving protection initialization behind exploration rescaling. Actual expanded
initial checkpoint parity across2524 examples, with randomized previous frames,
has maximum action difference9.24e-7 versus pinned v2. Offline deployment preflight
passed:675 inputs,6 arrival outputs,50Hz,1.47ms median including decoder/guard.
Caps remain12m/s and60m/s². No hardware was accessed or activated. Process records,
training plan and parity evidence are in logs/neural-player/requests/possession77.
The production candidate remains neural:possession-20260926-v2.

Run76 completed at875970560. First8M/final strict controls15/21 and controlled
requested-fast completions11/10 out of1536; shots401/449 then408/440 out of768.
Leading controls166/162 out of512; short-game stalls54.3/106.2s out of960.
Both sustained audit pairs passed without modeled overload:8M hot/cold peaks
.977073/.999860, final .964627/.991224. Final hot/cold stalled5255.6/4447.0s out
of7200 each. This is evidence that freeing hot-state responses can help this
branch's load behavior, but recovery/activity still do not justify promotion.

Run77 first8M protected weights were checked directly against initial weights:
all1,401,874 protected entries remained bitwise unchanged. New output weights
moved at most.001714. Leading control126/512 matched the pinned source; shots
401/446 versus409/451 in the paired development suite. Recent PPO updates used
all64 minibatches with KL.005–.011, below the.02 cutoff. This motivates checking
whether the inherited8e-6 fine-tuning rate is too conservative for added units.

Run78 is a paired16M experiment from the exact saved run77 initial checkpoint,
same seed20262577/data/curriculum, changing only actor learning rate to3.2e-5.
The32M low-rate run77 continues as a comparison. Both get the same8M/16M metrics
and hot/cold sustained audits. This is an optimization hypothesis, not evidence
that a larger network has improved recovery. Process records/plans are under
possession77 and possession78. Production remains pinned to v2; no hardware.

Run77 first8M strict recovery completed:34/1536 controlled,18 controlled requested
fast completions,660 first requested fast shots. Pinned v2 is32/18 on the same
strict suite; this is effectively unchanged and not a demonstrated recovery gain.
The full relevant regression suite was rerun after the final training edit:
209 passed in8.33s. Pinned v2 SHA and replay/UI HTTP200 were rechecked unchanged.
Runs77/78 were verified live; their remaining snapshots and sustained audits
are still pending. No newer checkpoint has been promoted.

Persistent exploration investigation: the old successful trailing-start collector
used8-frame raw-action perturbations, whereas PPO used independent frame noise.
Paired2304-trial probes (angles135,-135,180; speeds.15,.3,.6;256 per cell) compare
the same perturbation marginals before first contact. Seed20267826 independent/
persistent:7/32 first-leading contacts,1/11 leading-contact captures,32/35 total
captures,18/14 controlled requested-fast completions. Seed20267827:9/20,1/8,
27/34,13/20. Both use deterministic actions after contact. Thus coherent
exploration finds more useful approaches, but this does not establish improved
deterministic play. Early-only persistent noise was not better (first seed,
65 total captures versus79 independent;1 versus3 leading-contact captures).
Evidence: possession-diagnostics/persistent-exploration-probe*.json.

Added optional RecoveryExplorationBias and trainer flags. Exogenous offsets are
held8 decisions, refreshed until first contact only in slow outgoing receiving
drills, scaled by observed load reserve, and cleared independently at episode
resets. Full games/deployed inference never use them. PPO retains the actual
bias with each transition and evaluates both likelihoods under mean+bias, rather
than silently scoring exploratory actions under an unbiased Gaussian. Sampling
law has no learned parameters. Four new tests cover persistence/reset isolation,
contact/game/heat gates, likelihood ratios/gradients and invalid settings.
213 relevant tests pass. Actual2048-transition smoke training had initial KL
1.14e-6 and9.83e-7, with bias active on27.2%/13.4% of its two rollouts.

Run79 starts from the exact run77 expanded initial policy, same seed20262577,
8e-6 LR, replay/curriculum/frozen original512units,16M budget. It replaces the
first.16s independent .9 floor with persistent offsets of std[.6,.4,.3,.3,0,0]
through first contact, retaining learned Gaussian noise. Compare against run77;
this changes the exploration strategy, not only its temporal correlation. Every
8M snapshot gets deterministic skills/self-play and hot/cold audits. Process
records and full plan are under possession79. Production remains pinned v2.

Run77 at16M: strict26/1536 controlled,14 controlled requested-fast; shots423/452
out of768; leading123/512. Its sustained cold test passed at.969412, but hot failed
at1.027573 with170.42 aggregate overload seconds. First8M cold also failed at
1.003603 with1.14s overload; hot passed. Run78's higher LR first8M had28/1536
strict controls,14 completions,shots411/431 and leading115/512. No promotion.
A separate reward-credit diagnostic is checking whether20ms sampling misses
get-ahead milestones before first leading contact; no result or fix claimed yet.

The recovery progress diagnostic found26/32 first leading contacts had received
no geometric get-ahead credit (seed20267826,2304 trailing-start trials). The
milestone requires a sampled leading position outside the collision radius;
oblique approaches and contacts between control decisions can miss it. The
probe does not distinguish these causes. The contact callback now also pays
an unpaid get-ahead reward on the first leading-side contact, only for recovery
exercises that began behind. It shares the existing once-only ledger; already
paid/ahead starts, games, incoming pucks, trailing first contacts and later taps
receive no new credit. Separate cushioning/control rewards are unchanged.

The exact paired rerun preserved every physical/outcome field, with26 previously
unpaid contacts now paid and zero still unpaid after the callback. This validates
reward accounting only. Evidence is persistent-exploration-credit-probe.json and
persistent-exploration-credit-fixed.json.214 relevant tests pass in9.82s,
including the new first-contact/once-only/non-recovery regression.

Run80 is the explicit reward-change comparison to run79: identical expanded
initial weights, seed,16M budget,8e-6 LR, persistent exploration and remaining
settings. Their archived neural_training.py sources verify old versus new
callback:09da0035a440708d8709b12e0d62ca6edb076bf3c20ea413f72ae6676943905f
and d73f224b14eff7acf554cef8726cbcf2ce2101646769dba508986833085c3696.
Both were confirmed live. Run80 initially had rollout KL8.52e-7 and bias active
on11.5% of decisions. Training plans/PIDs are in possession79/possession80;
no hardware or physical deployment defaults changed.

Run78 finished16M at843792384. Strict control34/1536 and controlled requested-fast20
are insufficient to offset the shot regression362/388 out of768; it is WITHHELD.
Run77 at24M: strict28/1536,15 completions,shots421/470; leading127/512;
hot/cold sustained peaks.964395/.999140 with zero overload, but4125/3322 stalled
seconds out of7200. Its final32M snapshot is still pending. Neither experiment
has displaced the qualified v2 package.

Run77 training finished at859783168 (32M additional transitions); final evaluation
and sustained audits remain pending. Run78 final also failed sustained cold play:
peak1.017576,4.64 aggregate overload seconds,4172.7 stalled seconds; hot passed
at.973772 with4498.0 stalled seconds. Its existing WITHHELD decision now has an
additional load failure. Run79 produced first8M snapshot835796992 and its
watchers started evaluating it. Runs79/80 training PIDs3595279/3669015 were both
confirmed live; current artifacts stay experimental and the goal remains active.


Run81 filled an angular curriculum gap: easy recovery starts formerly covered
only+-37.3degrees around the leading side, then jumped to difficult rear starts.
The new easy arc reaches+-90degrees. An8192-env reset diagnostic counted2167
recovery cases: old angle bins[0,37.3,60,90,120,150,180] had
[1062,4,0,53,372,676], new[463,241,362,53,372,676]. This is initial-state coverage,
not a claim the old policy never visited side states. First8M self-play had31
goals and no reachable stalls, but sustained hot failed at1.005385/1.12s overload.
Final16M:strict32/1536 controls20 controlled-fast,shots401/462,43 goals21.7s
short stalls. Hot/cold peaks.985654/.983573 passed, but4188/4788s long stalls.
No promotion. Run80 final also failed cold at1.000613/0.10s overload. Run77 final
passed both load audits(.985879/.975886), but retained3626/3806s stalls.

Offline recovery search established a local feasibility result. Persistent
exploration offsets over six time blocks were optimized with common camera
noise/delays across candidates and reproduced incumbent scores across iterations.
This was nominal physics with report sensing, not broad randomized validation.
For a central departing puck at.6m/s/right-bank request, iteration05 parameters
controlled59/64 and completed54 controlled requested>=6m/s shots on separate
noise seed20269026. Parent v2 had0 controls. Search is data generation only;
no selected neural actor or production runtime runs it.

Run82 distilled successful rollouts from separate seed20269126 into a protected
expanded v2 network, using complete16-frame observed histories and v2 action
anchors. Selected update600 learned the local behavior: fresh seed20269027
53/64 controls,46 controlled-fast. Parent had0 controls but64/64 fast requested
volleys versus48/64 for the new actor. Nearby grid gains were right-request only.
Broader stationary shooting fell409->288/768; incoming451->434. Cold sustained
load failed at1.003770/0.54s overload. It is explicitly WITHHELD; low imitation
loss is insufficient evidence of preservation. Details:possession82/WITHHELD.md.

The original search acted before reset camera histories showed puck motion,
creating potentially conflicting labels for stationary versus moving starts.
A revised search gate uses observed outgoing velocity/speed/own-half position,
with no true-contact gating. The same fixed parameters still controlled55/64
and completed53 controlled requested-fast shots. Run83 repeats the imitation
recipe using separately generated data with this observed gate, to test whether
this fixes the stationary-shot regression. No physical hardware or deployment
defaults changed; pinned v2 remains the only qualified package.


Run83's observed-motion-gated data alone did not fix the broader regression:
shots284/438 versus pinned409/451, despite56/64 local controls48 controlled-fast.
Both sustained audits failed:hot peak1.030452/9.44s overload,cold1.007705/2.88s.
The camera-start conflict remains a plausible data concern, not a demonstrated
cause of the shooting regression. Run83 is withheld.

Collected77569 additional full-history skill observations from both v2 and
run83 trajectories (separate training seed20269426), labeled by v2 actions.
Run84 adds these to the existing27518 anchors, increases preservation weight
4->32 and fits20000 updates. Pure-NN development update20000 restores ordinary
shots410/459 out768 while retaining56/64 controls51 controlled-fast in central
right-request/.6m/s case. Select it for broader tests. Short self-play39 goals,
0 reachable stalls,peak.928089; leading-start112/512 controls80 conversions is
below pinned126/96, so this is not yet a general recovery improvement. Remaining
strict recovery and sustained-load reports pending; no promotion.

A training-only cross-request proposal reuses the successful right-request
neural recovery maneuver while keeping actual request in the environment/saved
observations, then follows the existing actor on that actual request when the
observed recovery condition is absent. It also uses the fixed searched offsets
for the initial.96s. This is an offline source of complete successful examples,
not deployed inference. Central.6m/s proposal results across left/right/straight:
45/55/48 controls and21/53/35 controlled requested-fast out64 perrequest. Run85
will distill successes from separate noise realizations at both.3/.6m/s, sampling
all requests equally. Its saved player remains one dense NN with no runtime
proposal switch/search/offset. Broad qualification is still required.


Run85 learned all three requested recoveries near the center:54-case grid338/864
controls258 controlled requested-fast (parent16/16). Ordinary shots418/448 out768
vs409/451. Default randomized recovery remained30/1536 controls17 completions.
Its cold sustained test failed at1.111116 with860.72s aggregate overload; hot
passed.973629. Run84 also failed cold1.034829/329.28s. Both are WITHHELD.

Run86 adds65536 synthetic warm variants of recovery training observations,
labeled by v2, with no runtime gate. Warm load histories are correlated,
fast-only,slow-only or single-channel peaks in.8–1.05; these are training inputs,
not verified safe actions. It passed development hot/cold audits(.984656/.992917,
zero overload). Full qualification preserved shots807/968 out1536 vs pinned806/965
and preparation1752/hot1576 vs1744/1564 out2048. However independent cold
seed20269728 FAILED:peak1.017420,314.90s overload. Do not promote it. Other reports
continue for comparison only. Warm augmentation did not establish general load
safety. At84/85 final overloaded states, v2's action was nearly identical to the
new actors, so imitating the parent cannot be assumed to teach a safe escape.

Run87 is active32M PPO from85 update20000. It mixes20 old quiet-puck examples
with29 unique quiet end states from84/85 development audits,50/50 group weight
(1160 weighted entries). It increases load cost8->16, warm practice correlation
.5->.75, stationary replay fraction.5->.8, uses existing hot effort exploration.6,
and keeps cooler skill/recovery imitation atweight32 over163895 full-history
frames. Actor LR1.6e-5, saved every8M, same single actor/frozen original512 units.
Hot reference constraints are omitted for quiet states so PPO can learn a
response. The old hot/cold seeds informed this replay and are now development,
not independent qualification. Exact plan/commands/PIDs:possession87. It produced
first8M snapshot835796992; evaluations/audits are automatic. No hardware used.

Offline search on12 random physical departures x3 requests expanded the local
recovery examples. With4 common sensor realizations and32 candidates,8 iterations
of observed-motion-gated offsets ended93/144 controls85 controlled requested-fast.
Separate sensor seed20269826:290/576 controls220 completions vs53 parent controls.
Separate training seed20269827/128 realizations yielded1568 complete successful
leading-contact sequences;1192 in first96 realizations train, remaining32 validate.
These are nominal searched proposals, not learned-policy success rates.

Run88 distills these plus the central examples into one NN, with ordinary skill
anchors and additional warm counterfactuals.138291 training frames44419 holdout,
236159 anchor frames,40000 supervised updates. Source initial actor remains the
same expanded v2; it does not contain run87's ongoing PPO improvements. Early
learned-only nearby-grid results377/864 controls278 controlled-fast, ordinary
shots409/438 out768. Broader/load evaluation pending. Pinned v2 remains unchanged.

Production metadata now accepts the explicitly named successful neural trajectory
imitation algorithm with arrival action mode, using the same actor/limits path
as PPO. Unknown algorithms/action modes and excess caps still fail.111 offline
deployment/launcher/runner tests pass. Run86 offline --check-policy passed at
50Hz,675 inputs,6 outputs,12m/s,60m/s2,~1.90ms median inference/decoder. This is
compatibility only; it is WITHHELD by the independent load failure.

A paired CUDA recovery diagnostic is separating initial-heat effects from
physical/sensor randomization. Most new successful demonstrations started with
uniform.3 load; standard skill evaluation starts with independently varied
channels up to.8. Do not attribute the remaining recovery gap to heat until
that comparison completes. No evaluation defaults were made easier.


Heat-only recovery ablation is complete and REJECTS initial heat as the main
cause: CUDA v2 default/cold.3 controls30/27 out1536, completions17/17;86 controls
33/26,completions19/18. Cold alone did not improve recovery. Do not repeat the
earlier temperature hypothesis as a finding.

Found a direct context shortcut in the demonstration recipe. Run86 central
nominal tests with fixed rival had controls[55,59,31,50,49,35] of64 percase. Holding
puck/paddle, camera noise, requests and load fixed but randomizing only idle
rival position reduced ALL SIX to0 controls/completions. Source skill fixtures
put the rival at x=.08,y=1.8; ordinary skill anchors and broad evaluation had
varied reachable rivals. Evidence:86/central-random-opponent versus its central
report. Recovery training needs that variation, rather than more copies of the
same incidental cue. Default evaluation has not been weakened.

Added offline recorded-action replay to the diagnostic generator. It re-executes
stored commands through physics with varied rival positions and initial RMS
profiles, verifies every pre-action own puck/paddle/command feature to1e-5,
and remeasures control/requested-fast/load outcomes. Changed rival/load features
are excluded explicitly; no runtime replay/controller is added. Original cold
central replay reproduces exactly. Context-varied central replay also has max
physical error0, retains1100 controls/779 controlled requested-fast over1536
trials,peakload.838436. Broad-source context replay is pending. Run89 will replace
88's demonstration inputs with these verified variants, preserving whole-group
train/holdout splits. Its fitter rejects insufficient rival-position spread.

Run87 first8M has shots405/421 out768,33/1536 strict controls16 completions,
22 self-play goals218.9 stalled seconds. Development hot/cold audits passed at
.973432/.990701. This is not promotion evidence, especially with reduced incoming
shot success and development failure states used in training. Training continues.
Run88 initial broad result remains32/1536 controls18 completions; its fixed-rival
training shares the shortcut, so local nominal gains are not a deployable fix.


## Context generalization and outcome-based recovery (runs89–93)

Broad recorded-action augmentation reproduced all own physical features exactly
(max error0), retaining2337 controls1767 controlled requested-fast outcomes with
randomized idle rival positions and independent initial load channels. Together
with central variants this removes the demonstrated fixed-rival input shortcut.
All original/augmented pairs keep identical whole-realization training folds;
validation sensor realizations are excluded from supervised and PPO replay.

Run89 learned104/1536 strict controls57 controlled-fast completions, but regressed
ordinary shots374/428 out768. Run90 mixes originals and varied examples equally:
137 strict controls83 completions (pinned32/18),183/512 leading controls141
conversions. Shots395/447 versus409/451 pinned; short selfplay29goals79.8stallseconds.
Its cold sustained audit FAILED1.007329 with1.0overload seconds; hot.974789 passed.
Both remain WITHHELD. A control gain alone is not an overall-qualified policy.
Run91 adds71711 ordinary and20179 fast-receiving true-history skill anchors from
parent/learner states. It returned126 controls71 completions,shots397/426, failing
the incoming-shot development gate. Source90 was selected for next PPO, not91.

Run87 final32M shots378/415 regress; strict38/19, finalcold/hot.997376/.975294 passed,
but16Mfailed. Run88 fixed-rival broadgeometry remains32controls18completions; its
local gains do not generalize. All87–91 withholding reasons are recorded locally.

Recovery drills had explicitly removed all positive uncaptured-shot credit.
Added optional recovery_fallback_scale (default0 retains old curricula): a fast
>=6m/s on-target requested route may earn scaled positive shot/completion credit
without capture, avoiding the failure charge for a useful return. Wrong-route,
slow and off-target returns retain their treatment. Internal completion ledger
never enters actor observations or reports fake capture. Existing control-only
mode is unchanged. Two focused regressions plus217 relevant tests passed; a real
2048-transition CLI smoke with the option passed. Deployment metadata supports
explicit successful neural-trajectory imitation artifacts via the same arrival
actor/limits validation; no inference teacher or controller is added.

Run92 starts90's context-aware NN and lowers intermediate capture/get-ahead/
cushion bonuses400/150/200->100/0/0, enables fallback1, keeps potential shaping and
competitive play.48M transitions at8e-6, saved8M, loadweight16 and gain1.3; skill
replay262140 weighted training-only frames at weight8, half request-balanced
successful recovery and half established skill trajectories. No synthetic hot
reference constrains PPO.900s games, warm/cold starts and failed quiet-state
practice remain. Automatic skills/recovery/selfplay and sustained development
load audits. Fresh20270427/20270428 reserved for a frozen selection. Exact plan,
command and PIDs inpossession92. WIP aliasneural-possession-outcome-recovery-wip.

Run93 separately searches32 additional randomized physical departing-puck starts
x3 requests, varying rival and initial load from the start. This is offline
training-data exploration using simulator outcomes, not deployable performance.
It uses nominal dynamics/report sensing and shared sensor realizations across
candidate actions. Further selection/data validation is required before use.
Pinned v2 and physical defaults remain unchanged; no hardware is activated.


Run92 stopped gracefully at842285056 (14.55M new transitions) after its extra
capture bonus still favored unfinished or weak control.8M had159 strict controls
93 controlled-fast but only258 requested-fast outgoing returns; final169/99/248.
Ordinary shots8M388/439, final383/447.8M hot FAILED1.030891/10.56soverload, cold
passed.976339; final audits remain pending. Snapshots/replays are retained.

Run95 starts the same90 checkpoint and seed, changing only the extra recovery
capture bonus100->0, leaving ordinary capture credit30. Bare event ordering now
favors useful completed returns: weak1m/s controlled shot~101.59;6m/s fast fallback
~122.38;8m/s fallback188;controlled6m/s195.34;controlled8m/s265.66. Capture thenloss
becomes30-60=-30. These omit load/movement/potential costs. Exact plan in95;48M
budget and automatic independent skill/load evaluation. No runtime action rule.

Run93 data search stopped at its strict incumbent-reproduction assertion.
Diagnosis showed all saved actor observations/actions reproduced EXACTLY; reward
alone differed by up to.243. The diagnostic changed shot request after reset but
left initial shaping potential from the old random request. This matters when
very slow vy<.2 leaves some shot-dependent setup shaping in the recovery blend.
Reconstructing old potential predicts all reward differences within2.1e-8.
Fix: refresh potential after diagnostic request/context overrides. Main PPO reset
already computes it after requests, so its code was not changed. Strict assertion
is retained. Corrected93search and later independent-noise collection restarted;
failed data search is not silently treated as completed optimization.

Run94 tests fixed scaling of the learned output correction in the same NN.
Quarter/half/three-quarter scales controls33/32/59,completions19/12/31,requested-fast
463/329/273. Ordinary shots413/442,403/440,396/451. No useful balanced candidate.
All94artifacts are WITHHELD; this is weight regularization, never runtime mixing.

Run96 collects full successful learned trajectories on512 new randomized physical
starts x3requests, with varied dynamics, rivals,heat/report sensing. Source90
contributes76 controlled requested-fast outcomes; otherwise v2 contributes578
successful requested-fast returns. Physical start observations match exactly
between paired actors. Training split first384physicalstarts, validation128,
allrequests kept together:502train152validation sequences. Single NN starts90,
learns new controls/new fallbacks/older searched controls equally with balanced
requests, retains ordinary/warm anchors.40k supervised updates, three development
snapshots. This addresses the previous blanket exclusion of all outgoing states
from preservation without suppressing successful control. Selection is offline;
no runtime switch/controller is added. Full qualification is still required.


Run96 development is complete:10k controls67/1536,controlled-fast44,first-fast523,
ordinary414/446;20k61/38/548,shots411/450;40k64/33/534,shots409/444. Declared gate
selects10k (>=64controls) but its score567 is below650 for sustained qualification.
Selfplay40goals66.8stallseconds. No promotion. Run98 waits for corrected93's
broader examples, then fits from96-10k with half fallback,one-sixth each of new
actual controls,older searched controls,new32-geometry controls. It keeps all
holdout distinctions and deploys one actor. Exact plan in98.

Run95 first8M:148controls82controlled-fast234first-fast,shots385/452;self30goals
12.7stallseconds. Development hot/cold PASS .992525/.987053,zerooverloads, but
3928.3/4946 long-stall seconds.16M179/110/242,shots397/442,self38goals143.4stall;
hot/cold.978389/.987576pass,4705.7/4127.3stall.48M training still running.
Run97 is a16M controlled ablation, same90 start/seed95 reward settings but removes
persistent random recovery action offsets. These were useful for initial
exploration; their continued benefit once recovery is learned is uncertain.
Likelihood handling is correct; this tests training distribution, not a bug.

Development head-to-head uses actual pinnedv2, seed20271026,8x180s per color/style,
initial load.8/gain1.3. Aggregated bothcolors:95-8M mixed28:19,straight14:22;
96-10k mixed30:30,straight17:21. Matching baseline v2vsitself mixed16:16,straight
11:21. Thus raw straight-only losses do not alone establish a defensive
regression: the pinned actor also loses when only its opponent uses straight.
95-8M preparation1743/2048,hoth1553 versusv2 1744/1564. All cross-play candidate
loads<1,zerooverloads. Samples are small and these are development comparisons.

Freeze95-8M for complete qualification based on preserved preparation, passing
both development load audits,4.6x controls and positive mixed head-to-head.
It retains the outgoing first-fast tradeoff and slightly lower development
stationary-shot count; those are not hidden. Full suite plus freshhot20270427/
freshcold20270428 started; exact SHA/reasons/commands in95/qualification-8m.
These seeds are now USED for this frozen checkpoint and must not be called
fresh for later selection. No automatic promotion. Pinned v2 remains unchanged.


95-8M full shooting805/968 out1536 matchespinned806/965 despite its smaller
385/452 development result. Preparation1743/1553,direct1003,banks776/hot792;
fullself46goals194.6stallseconds. Independenthot20270427 FAILED1.072740 with542.72
overloadseconds; independentcold20270428 passed.976671. Bothzero capviolations.
It is WITHHELD. Full reports cannot override the sustained hot failure.

The95-8M straight-only development baseline finished: pinned11:21 versusnew14:22,
whereas mixedbaseline16:16 versusnew28:19. Replays/match outcomes therefore provide
useful context beyond isolated first-shot counts, but do not cancel the load issue.

Run97 completed16M at843792384. Its8M ordinaryshots378/431 and shortselfplay already
exceeded load1.055057 for11.54seconds. This snapshot is excluded. Final evaluations
continue; removing persistent recovery offsets is not established as an improvement.

Run93 correctedsearch completed8iterations with strict incumbent reproduction
checks intact. On4training contexts percase final143/384controlled requested-fast.
New validationcontexts:301/1536controls172completedfast vs128parent controls.
Collection64contexts:1242/6144controls684controlledfast vs497parent controls. These
remain training proposals under nominal physics; they are not learned NN results.
98 fit/evaluation launched after successful collection. A preprocessing slowdown
was fixed by caching NPZ arrays once instead of decompressing257MB for each group.
The first98process stopped before run/checkpoint creation, then restarted with
identical examples/seed/settings; no trained work was discarded. Cached fitter
PID/process records are under98/process-cached.json.96fitter receives same cache
fix for future reproducibility; its already-completed results are unchanged.

Run99 starts frozen95-8M and trains16M with warm starts up to.99 and termination
at actual modeledshutdown1.0 instead ofinitial<=.94/termination.95. Previous
training had essentially no continuation in [.95,1), although qualification
starts.95. This is a real coverage gap, not yet proven as the sole failure cause.
Loadweight16 and all other settings retained; no new controller/safety derating.
No fresh-audit endstates added to replay. Physical drive limits/trips unchanged.
Fresh20271527/20271528 reserved for a frozen eventual selection. See99/PLAN.md.
Production remains pinnedv2. No hardware is activated.


After the usage-limit pause, all95/97/98/99 training and development evaluations
had completed. No training process remained running.95final: ordinary387/435,
strict176controls103controlled-fast216first-fast; development hot/cold pass, but
no new independent qualification.97final399/453,161/101/217, developmentloadpass.
98selected10k396/433,78/44/522; below its predeclared650 combinedrecoverygate.

99first8M843857920: ordinary386/446,strict182/109/220; shortself21goals193.1stalls,
hot.967071/cold.985505 zero overload/capviolations.99final851853312:360/426,
187/123/209; cold1.002294 with.3soverload, so final is excluded.
Freeze99-8M before fresh results: full suite plus independenthot20271527 and
cold20271528; exactSHA/reasons/commands in99/qualification-8m/selection.json.
Those seeds are now USED for this frozen checkpoint. Also frozen cross-play
against actual pinnedv2 and matchingv2baseline, bothcolors/random-or-straight
requests, seed20271226 in99/cross-pinned. No promotion implied; substantial
outgoing first-fast tradeoff remains visible. Production/hardware unchanged.


99-8M independent qualification FAILED: hot exceeded1.10 and cold1.002683/0.32s.
Full data collection finishes for retained evidence only. Frozen cross-pinned
seed20271226:99mixed28:34 vsbaseline25:25;99vsstraight20:23 vsbaseline14:13.
The previous95mixed28:19 therefore did not generalize to this checkpoint/seed.
Recovery drill gains alone do not establish better full play.

100 briefly tested unfreezing the actor prefix. Single optimizer steps at8e-6
caused KL>1 versus target.02; stopped early and excluded. Existing PPO only
checked KL before the next step. Added optional --backtrack-kl to shorten an
oversized proposal before accepting it, retaining critic steps and Adam gradient
moments. This is a measured minibatch bound, not a global policy/physical bound.
Two new focused tests and a real512-transition PPO smoke pass;219 relevant
regression tests pass.101 restarts unchanged99-8M with unfrozen actor,lr8e-7,
backtracking enabled. Other rewards/settings retained;16M steps/8M snapshots.

Separately freeze97final843792384 for full qualification plus freshhot20272601
andcold20272602. It preserves ordinary shooting better than99, has161controls/
101controlledfast/217firstfast and passed both development loads. All remaining
load/defense/full-play conditions must still pass; no promotion implied.


97final full paired skills: stationary821/incoming971 requested fast out1536,
versus pinned806/965; preparation1744/hot1567 versus1744/1564; direct1009,
banks783/hot790 versus1013/785/792. Short fullself174.3stallseconds versus124.
Freshhot20272601 peak.962049 andcold20272602 .973017,zerooverload/capviolations.
Primary full hot audit still pending. Small cross-pinned seed20272703: mixed
27:29 versusbaseline15:15; vsstraight14:15 versusbaseline15:6. Expanded frozen
32games×180s/color/style,seed20272803,launched to assess the defense concern;
earlier results retained. Not yet production selected.
101 first8M851918848 saved and entered development evaluation; lowerlr/backtracking
now permits64actorupdates/rollout with ordinary sampledKL around.006–.012.
100 stoppedweights dropped ordinaryshots to258/251, illustrating the damage
from oversized updates.101 behavior still needs closed-loop evaluation.


Selected97final843792384 packaged as possession-20260926-v3, SHA
a3cd692c80c4ca42f6f200475aa5ccf967793df27853d9770c56be265886c2dc. Full qualification
and five sustained audits pass; gates also compared against actualv2 rather
than only older33. Expanded cross-pinned32games/color/style: mixed112:123 vs
baseline104:104;straight81:63 vsbaseline51:56. No overall dominance claim.
Source97's earlier8M remains excluded; its top-levelWITHHELD note replaced by
SELECTION.md with exact identity. Production command/replay/results documented
in ai/NEURAL_PLAYER.md and ai/POSSESSION_RESULTS_20260926_V3.md. v2 retained.

Exact packaged selfplay matches source score/load/stall arrays;9000frames/180s,
SHA-matched metadata,sameNNbothsides,standardseed20263543. UI and recordingAPI200.
Offline--check-policy passed2.01ms,675inputs/6outputs,50Hz,caps12/60.219regression
tests plus real PPO backtracking smoke pass. No hardware activated.

101 completed16M at859914240.8M:406/439shots,157controls103controlledfast212firstfast;
shortself24goals26.4stalls;hot1.018648/5.84soverload,cold1.010973/2.10s. Not selected.
Its final diagnostics are still finishing independently with no automatic
promotion. Selectedv3 is a qualified control-oriented improvement, with fewer
immediate outgoing fast returns, remaining quiet stalls and no established
win-rate advantage overv2. Further progress must address those limits.


User identified the pinnedv3 replay's left-edge stall. Verifiedpuck settles near
x.1032,y.4306 at~20s and remains to180s; paddle stays nearx.1932,y.21. Paddle-center
xmin.1911545, combinedradius.0911, so the puck is contactable even withguardinset.
No unreachable-puck re-serve should hide this failure. Standard stationary starts
mostlyexclude sidefringes; clipped behind-puck setup can be unable to touch it.

Offline scripted feasibility:504 proposals,174 restoreinterior afterwall bounce,
158 also avoid later goals during6s. Retained demo restores in1.02s,peakload.432658.
The earlier fastest.92s proposal later conceded when left unattended; it is not
the retained demonstration. No scripted motion is added to deployment/training
actions. Artifact edge-wall-feasibility-demo.json clearly labels the diagnostic.
New learned-only randomized edge benchmark66physicalcases×3requests,12s,seed20272926.
v3initial:109touches,2restored,49controls,3requestedfast; no goals/concedes.

Added optional side-fringe curriculum,12s followthrough, feasible-contact shaping,
once-per-possession restoration credit80 with no fakecontrol/earlytermination.
Filtered slow-fringe anchors that otherwise preserve failed behavior.228tests
and real2048transition training smoke pass. Run10216M fromexactv3,8Msnapshots,
limits12/60unchanged, all motion stays neural. Quiet exploration onlyinpractice.
Automatic percheckpoint edge/standard-skill/selfplay/sustained audits; no automatic
promotion. Packagedv3 docs now explicitly identify the shown failed replay.


102 finished16M at859848704.8Medge2/198restorations,final4/198; postrestorefast1→3,
exact replay andmirrorstillfail.8Mauditsfailedhot1.083903/23.22soverload andcold
1.035475/816.62s; not a replacement. Alternate load-context tests reject a simple
heat-distribution explanation: v3/102final replay-profile restorations4/2; both
hot.9zero. Final102loadchecks pending.

103 launched32M fromsamev3 andseed102, sameedgecurriculum/rewards, withmean-zero
random offsets held12decisions(.24s) only in quiet side-fringe stationarypractice.
No staging point/direction/shot sequence supplied. Original conditional likelihood
handling retained; deployments/fullgames have no offsets.229tests and real2048
transition bias smoke pass (initialKL~2e-6). Periodic8Msnapshots get edges, standard
skills, selfplay and sustained hot/cold audits. Backgroundwatcher writes combined
checkpoint-review.json after all finish, without promoting anything. The actual
learned ability is still unproven; never substitute the scripted feasibility demo
for a neural replay. Originalbad selectedreplay remainsavailable as evidence.


Edge work continued:102final passes hot/cold development loads(.984907/.986602)
but fails edge ability.103-8M similarly passes loads(.970877/.968130) but restores
only4/198.103-16M restores2/198. Neither addresses the reported failure.

104 self-imitation uses successful trajectories generated ONLY by103-8M neural
policy plus randomized training exploration. Four independent collection seeds;
whole physical cases held out. No scripted demonstration actions. Frozen512prefix,
ordinary and hot v3 behavior anchors. Final20k supervised updates restore16/198
(v3:2), control after restoration2, requested fast after restoration3. Ordinary
shots398/457 vs baseline399/453. Still fails the explicit recovery/followthrough
screen, so no production candidate or full qualification claim.

Found the initial edge sampler's3mm minimum contact margin excluded the reported
replay(~1.15mm with the motion guard). Curriculum now includes .5mm minimum and
oversamples .5–10mm margins; ten edge tests pass. Running103 imported the previous
sampler and is not retroactively changed. Experiment105 collects eight fresh
neural-only example seeds20273426..20273433, half narrow margins, and fits with
unfrozen actor and ordinary/hot anchor weights8/4. It keeps whole physical cases
held out and never uses the external benchmark fixtures as training examples.
No runtime controller, decoder change, physical limit change or hardware action.


105 completed: final18/198 recoveries,1 controlled after recovery,5 requested-fast
followthroughs,8 requested fast total; ordinary shooting349/427 vs baseline399/453.
It is rejected as a replacement and is NOT the next run's initialization.
104final instead preserves ordinary shots398/457 and restores16/198. Its original
replay fixture restores on1/3requests, mirror on2/3; still far below the full
screen. Six original/mirrored request replays can now be emitted directly by
edge evaluator (--record-prefix), without success selection.

106 launched from104-20k, SHA90171f7878ac92dc8c90cb634c7bb04048c27303b1dc49a814c2e3022e4ee3d4.
128M PPO steps,120minute ceiling,16M snapshots,lr2e-5 with backtracking/frozen512,
edgefraction.9/gamefraction.15, oversampled thin reach margins. Source is an
unqualified training initialization, not a new production choice.4096-transition
smoke passed; actualrun advancing with initialKL~3e-6. TrainPID1132288,
evaluatorPID1132289. Each checkpoint gets edge/standard shooting/selfplay screen,
six fixed edge replays and updating selfplay alias. After run, best eligible
checkpoint gets sustained audits, fresh edge seed20273026 and full qualification.
No automatic promotion; still need compare results againstv3 and fresh loads.
Commands/plans/results live in logs/neural-player/requests/possession106/.

UI alias http://localhost:8420/?replay=neural-edge-recovery-training.json initially
shows104final and clearly labels itself the unqualified106starting policy; later
checkpoint evaluator updates it. UI and recordingAPI verifiedHTTP200. Core229tests
passed before final sampler tweak; all10edge regression tests pass after tweak;
compile/whitespace checks clean. Physical hardware and deployment defaults untouched.


106 status reviewed after completion:128,057,344additional PPO steps,6261.49s,
8/8development evaluations complete. No checkpoint passed. Final971849728 SHA
da94d530cefca729af23ae89e34e386ab4a3442022206e5505b9d222cc3cf7ce:5/198restorations,
0controlled afterward,2requested fast follow-ups; ordinary393/452 and shortpeak
.962705. Full/sustained qualification was not started after failed screens.
No training/evaluator currently active. Review persisted in run/review.json.

Added /training and /api/training, visible links in main header and Replay panel.
Read-only discovery of current processes, progress within resumed run, heartbeat,
evaluation completeness, screening results, exact checkpoint selfplay and six
edge diagnostic links, and separate packaged-policy section. Future PPO statuses
include pid/time/initial_step/target_steps. Old statuses derive run-local progress
from rollout size and update count; stale running flags are not called active.
Seven dashboard/listing regression tests pass. Headless Chromium verified desktop
and390pxmobile, all8checkpoint rows, expansion retained on automatic5srefresh,
exact final replay navigation, no JS errors or mobile overflow. No hardware touched.


2026-09-27: User again reports slow outgoing possession loss and forward defense.
Found two training gaps: a brief capture excuses later unproductive loss in the
old entry metric; exact-release-point coverage assigns zero risk to centered
paddles at all depths for a centered stationary puck. Added opt-in slow_exit_penalty
once per possession for slow departure beyond front contact reach after >=.1s
reachable opportunity, excluding goals/useful fast returns but not prior captures.
Counter works independently of weight and mirrored for opponent; partial resets
isolate ledger. Added .25m release uncertainty only while preparing, not for known
fast incoming shots. Added actual lateral free-puck motion before hidden delayed
practice shot; re-aim from real release position, no teleport. All training-only;
actor inputs/actions, physical limits and production policy unchanged.

Baselinev3 same1024seeds: stationary saves881 vs laterally moving release700;
705/1024forward above.60m, mean depth.60848m. Failed106final:725saves,700forward,
mean.60873m. New benchmark computes validity against actual release position and
velocity. Old strict outgoing v3:161controls/101controlled requested fast/1536.
Filtered old preservation examples: remove62104opponent-half and62516slow outgoing;
keep107476other skill states, avoiding anchoring the behaviors being changed.

107 launched frompackagedv3,48M additional steps,8M checkpoints,90min ceiling,
lr8e-6/KLbacktracking/frozen512prefix. slow loss80,uncertainty.25m,lateral drift1.2m/s,
35%defense practice with80%moving delayed release; recovery fraction.8, speeds.08–.8,
get-ahead30/cushion10once-only shaping; coherent random recoveryexploration.4.
Caps12/60,gain1.3,shutdown1.0unchanged. TrainPID137925/evaluatorinitialPID137926.
237regressions plus4096transition smoke pass, actual run~18.7ksteps/s. Plans/source
hash/gates/commands in logs/neural-player/requests/possession107. Screen requires
moving saves>=740, static>=864, controls>=178/controlledfast>=112, >=98%ordinary
shooting, shortload<1, no edge regression. Full/fresh/sustained checks before any
replacement, no autopromotion. Runtime remainsoneNN; no hardwareactivated.

Dashboard now includes slow-drift controls/conversions and moving-release defense
metrics/forward count per checkpoint. Current107visible withlive progress/eval
worker; desktop/mobile browserverified,7dashboardregressionspass,compile/diffclean.
First actual checkpoint replay appears after8M save/evaluation, no placeholder
claim that an initialization replay is a trained checkpoint.
