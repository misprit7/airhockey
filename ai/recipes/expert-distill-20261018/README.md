# Physics teacher → one neural network (2026-10-18 overnight)

Simulation only. No hardware calls, firmware changes or deployment-default
changes. The robot still runs exactly one learned network at the table; the
teacher below exists only to train and to benchmark it.

## Objective

A much stronger player against people on the physical table, under the same
physical limits as production `rail30-100-20261001-v1`: 12 m/s, 100 m/s²,
3 ms ramp, rail30 workspace, 50 Hz, 15 ms command delay, motion guard,
thermal recipe `motor-load-20261001` at gain 1.3 (0.95 training shutdown,
≥1.0 counted as overload in evaluation).

## Why this direction (evidence, not taste)

- Production blocks **27%** of the fast-attack benchmark (left banks 2%,
  right banks 9%). The 56-line `deep-bank` heuristic in
  `defense_baselines.py`, which sees the same delayed observations and drives
  the same decoder/guard/thermal model, blocks **84%**. The gap is
  positional: production waits 0.63 m forward, the heuristic 0.18 m.
- On the table (Oct 2) 6 of 7 concessions were single side-rail banks at
  3.7–6.5 m/s, with the paddle 0.54–0.72 m forward.
- Three weeks of reward shaping (depth costs 200/800, stricter capture,
  widening) moved the stance 0.63 → 0.53 m at best and cost shooting or the
  ladder each time. PPO here takes ~6 actor steps per 131k-transition
  rollout (KL stop at 0.02, lr 3e-6–1e-5); strong imitation anchors held the
  old stance in place.
- In August a hand-written striker beat its own learned clone 0.80/0.06 vs
  0.15/0.90 goals for/against per game: the clone failed by distribution
  shift. DAgger (the teacher labels the student's own states) is the fix
  for that failure mode, and it already worked here once (modern26 took
  shot routes from 0/64 to ~89% each, Sep 23).

## Plan

1. **Teacher** `ai/airhockey/expert_player.py` (`ExpertPlayer`): same
   16-frame observation and 6-D arrival action as the network; forecasts
   the puck with drag and lossy rails; blocks on the line with the most
   time margin; strikes on the requested route (or away from the rival
   mallet); guards on the puck–goal line and backs off when hot.
   Evaluators accept `expert[:key=value,...]` anywhere they take a
   checkpoint.
2. **Human-proxy benchmark** `ai/bin/eval_human_proxy.py`: a far-side
   opponent with human reach (to the centre line), human-calibrated shot
   speeds (club: median ~5.5 m/s; elite 8–14), ~50% banks, reaction delay.
   Headline metric: goal share against it, with bootstrap CIs.
3. **DAgger** `ai/bin/distill_expert.py`: student = production's
   architecture initialised from production; roll-in mixes teacher and
   student; labels always from the teacher; opponents mix teacher,
   production and the student.
4. **RL fine-tune**: `train_neural_player.py` with a decaying teacher
   regulariser, the teacher as one of the opponents, and a less crippled
   optimiser (minimum actor steps per rollout, critic warm-up).
5. **Selection** on dev seeds, **confirmation** on reserved seeds
   (20261980–20261999, 20262000+ are not used for development).

## Status log

- 00:30 — mapped the stack (7 parallel readers; summaries in the session).
- 01:00 — teacher v0: fast-attack 82.2% (straight 97, left 71, right 78);
  16×120 s vs production 13–5; its shots weak (28% on target, ~5 m/s), one
  60 s overload holding deep near a side rail.
- 01:15 — launched in parallel: DAgger pipeline, trainer extensions,
  human-proxy benchmark; teacher defense and attack specialists.
- 01:30 — opt-in measured paddle contact (`paddle_restitution_range`,
  `paddle_tangential_range`; table: e 0.47-0.59, tangential 0.76-0.84).
  Defaults unchanged (production fast defense 220/768 before and after).
  Under the measured contact: teacher 83.1%, production 28.8% fast defense.
- 01:35 — DAgger smoke (`_distill-smoke-20261018`, 3 rounds x 2M labels,
  ~3 min, ~83k labels/s): student fast defense 67 -> 71 -> 74.5%; games vs
  production 15-84 (round 0, pure BC: distribution shift) -> 8-42 -> 11-12.

- 02:00 — defense specialist (branch worktree-wf_dda5e07e-1a6-1, a2fe081):
  minimum-time bang-bang arrival command (v0's fixed 40 ms request overshot
  75-82 mm and settled in 180-200 ms), block point shifted 3 cm upstream,
  banks forecast under three rail scenarios before the bounce, bounce-
  straddling reports repaired, guard depth 0.11/0.17/0.30 m by hottest load.
  Fast defense 97.3% / 97.8% (seeds 20261073 / 20261051; banks 96-98%),
  4-10 m/s tier 99.9%, centre-line releases 100%; hot 4x600 s no overload
  (peak 0.881); conceded to production 3 in 32x60 s (v0: 13). Its own
  scoring fell (45 -> 18 goals) because saves return the puck to the rival.

- 02:15 — STOPPED (usage limit). On main: teacher v0 + eval hooks, this
  record and suite.py, opt-in measured paddle contact, trainer fine-tune
  flags (cherry-picked be35050). Not yet on main, committed on branches:
  defense teacher `worktree-wf_dda5e07e-1a6-1` (a2fe081), attack teacher
  `worktree-wf_dda5e07e-1a6-2` (4ef588b, in progress), DAgger pipeline
  `worktree-wf_331ba6a6-2c6-1` (cefbade; conflicts with the trainer commit in
  ai/bin/train_neural_player.py and ai/airhockey/neural_vector.py: keep the
  trainer's new args inside distill's build_parser(), keep its teacher
  loading in main(), and use environment_options() with `or teacher is not
  None` added to shot_conditioned). Human-proxy benchmark was uncommitted in
  `.claude/worktrees/wf_331ba6a6-2c6-3`.
- Next: merge the above, merge defense+attack teacher, run distill_expert.py
  at scale (12 workers, ~83k labels/s), then the fine-tune command in the
  trainer report (lr 1e-6 adaptive, std floor 0.1, teacher weight 1 -> 0.2,
  teacher opponents 30%), with --paddle-restitution-range 0.40 0.65 and
  --paddle-tangential-range 0.76 0.84 plumbed through environment_options.

- 05:00 — resumed. Cherry-picked onto main: trainer review fixes (diagnostic
  sample covered only reference slots), the DAgger pipeline (conflicts
  resolved; paddle-contact flags added to train_neural_player and
  distill_expert), the defense and attack teachers, and the human-proxy
  benchmark (`eval_human_proxy.py`, committed from its worktree).
- 05:10 — merged teacher: fast defense 95.1% (seed 20261073), 16x60 s vs
  production 37-5 (56% on target, 6.6 m/s). Human proxy (64x60 s/tier):
  club 314-17 (94.9%, CI 92.4-97.1) vs production 172-47 (78.5%); elite
  368-27 (93.2%) vs 234-111 (67.8%); teacher 0 s overload (peak 0.943 club,
  0.881 elite), production 73 s / 64 s overload.
- 05:10 — DAgger arm A `_neural-teacher-dagger-20261018-a` (default
  physics) and arm B `_neural-teacher-dagger-20261018-b` (measured paddle
  contact, teacher paddle_restitution 0.52), 8 rounds x 6M labels each,
  from the smoke student.

- 05:15 — teacher heat gate (1330a39): above 0.86 observed load the attack
  leaves pucks within 12 cm of the side/back limits to the guard. Hot 4x600 s
  vs production: overload 110.7 s -> 0 (peak 1.026 -> 0.858), 19-1.
  Merged teacher dev suite before the gate: fast defense 95.1%, immediate
  47.3% (production 25.7%), requested fast shots stationary S/L/R 95/88/89
  of 128 (production 75/38/42) but receiving 51/38/37 (production 88/86/81),
  slow-control fast-aimed 38.5% (61.7%), corner recovery 63/128 (22) but side
  1/128 (17), games vs production 37-5 / 51-4. Under the measured contact the
  teacher still beats production 12-1 (16x60 s), shots 4.7 vs 6.6 m/s.
- 05:15 — arm A rounds (defense / games vs production 16x60 s): r0 94.4% /
  13-7, r1 89.7 / 12-2, r2 91.4 / 8-0, r3 87.0 / 10-5, r4 87.0 / 12-3. The
  student's release depth drifts 0.12 -> 0.19 m as roll-in becomes its own
  (noisier, hotter states; the teacher's guard depth is load-scheduled), and
  its shots stay ~30% on target at ~5.2 m/s vs the teacher's 56% at 6.6.
  Stopped after round 5. Arm B (measured contact) r3: 84.6% / 2-0, one
  student overload (peak 1.08; trained before the heat gate).
- 05:20 — continuation C `_neural-teacher-dagger-20261018-c`: from arm A's
  final, widened to 1536, heat-gated teacher, attack-weighted loss (modes
  3,5,1), student noise 0.05, betas 0.3/0.2/0.1/0.1/0/0, 8M labels/round.

- 05:30 — arm A final (round 5) dev suite: fast defense 88.7% (S 96.5 /
  L 83.2 / R 86.3, depth 0.19), immediate 42.1%, corner 94/128 (production
  22), side 3/128, games vs production 18-3 / 11-3, hot 4x600 s 4-1 with no
  overload (peak 0.99). But requested fast shots 0-4/128 in every cohort
  (production 75/38/42 stationary) and slow-control fast aimed 0.3% (61.7%):
  the teacher's aligned strike-through did not survive imitation. Human
  proxy: club 71-30 (70.3%) vs production 78.5%; elite 115-54 (68.0%) vs
  67.8% -- concedes far less (GA/min 0.47 / 0.84 vs 0.73 / 1.73) but scores
  far less (1.1 / 1.8 GF/min vs 2.7 / 3.7), shots 3.4-4.3 m/s; elite peak
  load 1.19 (20 s overload).
- 05:30 — arm D `_neural-teacher-dagger-20261018-d`: from PRODUCTION widened
  to 1536; teacher labels threats, fast incoming pucks and the guard;
  production's own action labels the states where the teacher would attack a
  slow (< 2.5 m/s) or outgoing puck (`--attack-labels production`). Arm B
  stopped at round 6 to free memory; C paused (SIGSTOP) to give D the CPU.

- 05:40 — the teacher's stationary strikes are excellent (first shots
  9.2-9.5 m/s, 93-111/128 on target) but students average the set-up/strike
  decision into weak pushes (arm A 2-3 m/s). Arm D round 1 recovers
  production-style striking (stationary requested >=6 m/s 50/31/23 of 128,
  round 0 16/1/21) and is the first clear win on the headline metric:
  human proxy club 109-17 (86.5%, CI 80.2-92.4) vs production 78.5%; elite
  168-27 (86.2%, CI 80.6-90.8) vs 67.8%; no overload (peak 0.85 / 0.97).
- 05:40 — C stopped after round 2; arm E `_neural-teacher-dagger-20261018-e`
  continues C with a teacher that commits to strikes earlier
  (strike_tolerance 6, strike_lateral_speed 0.4: stationary 82/76/69 of 128,
  9.1-9.4 m/s, defense 94.9%), velocity outputs weighted 3x.

- 05:45 — arm D rounds (fast defense / games vs production): r0 85.3% /
  21-6, r1 86.5 / 16-2, r2 83.2 / 18-0, r3 85.7 / 13-5, r4 84.5 / 16-1,
  r5 (final) 86.8 / 19-3. Arm E round 0: 89.3% / 23-4; its stationary first
  shots are 7.1-8.0 m/s (production 5.5-6.9, arm A 2-3) at 43/24/24 of 128
  requested >=6 m/s (production 75/38/42).
- 05:45 — fine-tune `_neural-teacher-ppo-20261018-d` from arm D's final:
  `finetune.py --teacher-mask defense --minutes 60` (default physics).

- 05:50 — arm D final dev suite: human proxy club 141-12 (92.2%, CI
  87.8-96.2) vs production 78.5%; elite 169-26 (86.7%, CI 81.7-91.1) vs
  67.8%; no overload. Fast defense 86.8% (S 93.8 / L 76.6 / R 90.2), immediate
  40.9%, requested fast shots stationary S/L/R 44/21/36 of 128 (production
  75/38/42), receiving 1-4 (81-88), slow-control fast aimed 45.6% (61.7%),
  corner 36/128 (22), side 15/128 (17), games vs production 19-3 / 14-3, hot
  4x600 s 14-2 with zero overload (peak 0.847; production 511 s / 1.057),
  reachable stall 474 s (production 1361 s). Arm E round 3 on the proxy:
  club 77.9%, elite 72.4% -- E stopped; D is the line to pursue.
- Replays (web UI replay tab, http://localhost:8420/?replay=<file>):
  `teacher-dagger-d-vs-production.json`,
  `teacher-dagger-d-vs-proxy_vs_human-proxy-{club,elite}-g{0,1}.json`.

- 06:00 — HELD-OUT evaluation (`suite.py --tier holdout`, reserved seeds,
  each candidate evaluated once), arm D final vs production:

  | | production v1 | arm D final |
  |---|---:|---:|
  | human proxy, club, 128x60 s | 71.0% (303-124), GA 0.97/min, 38.7 s overload | **92.4% (278-23)**, CI 89.5-95.1, GA 0.18/min, 0 s |
  | human proxy, elite, 128x60 s | 59.5% (379-258), GA 2.02/min, 63.3 s overload | **88.2% (388-52)**, CI 85.0-91.2, GA 0.41/min, 0 s |
  | fast attacks blocked, 3072 | 27.1% (S 70.1 / L 2.1 / R 9.1) | **86.9%** (S 95.4 / L 76.5 / R 88.9) |
  | immediate release, 1536 | 26.8% | 43.3% |
  | games vs production, 48x60 s each seat | 58-71 / 73-72 (itself) | **26-6 / 45-5** |
  | hot endurance 16x900 s (0.8, gain 1.3) vs production | 74-61; 2879 s overload, peak 1.057; stall 5204 s | **29-6; 0 s overload, peak 0.854; stall 0** |
  | requested fast shots stationary S/L/R (of 256) | 126 / 82 / 80 | 88 / 35 / 64 |
  | requested fast shots receiving S/L/R (of 256) | 172 / 158 / 179 | 12 / 6 / 6 |
  | slow control: fast aimed return (1536) | 60.9% | 47.1% |
  | corner / side recovery (of 256) | 37 / 32 | 57 / 39 |

  Packaged as `runs/teacher-dagger-20261018-v1` (sha256 01f7f482...,
  `expert_dagger_v1`, width 1536); loads through `NeuralPolicy` with
  production's limits and workspace. Not physically qualified.

- 06:05 — fine-tune of D at 40M steps (dev): fast defense 94.0% (D 86.8%),
  stationary requested shots 53/35/58 of 128 (D 44/21/36), games vs
  production 23-0 / 28-0, human proxy club 129-3 (97.7%), elite 146-17
  (89.6%). But it learned to HOLD: 494 s / 422 s of reachable stall against
  the proxy (D: 0 / 171 s). Diagnostic (32 club games): stalled pucks sit in
  the open interior (y 0.44-0.61, x 0.27-0.49) with the paddle 0.10-0.18 m
  away at load 0.60-0.74 -- passive possession, not the heat gate. D shows
  zero such samples. The possession-delay cost (6/s after 4 s) was too weak.
- 06:10 — stopped at 51M; restarted from that checkpoint as
  `_neural-teacher-ppo-20261018-e` with possession delay 30/s (45 min).

- 06:27 — anti-holding fine-tune at 40M steps (`_neural-teacher-ppo-20261018-e`
  agent_step_040108032): human proxy club 175-10 (94.6%, CI 90.8-97.8),
  GF/min 2.73 (production 2.69), GA/min 0.16; elite 259-33 (88.7%, CI
  85.2-92.2), GF/min 4.05 (production 3.66), GA/min 0.52; shots 5.8-6.3 m/s;
  no overload. Stall 26 s / 78 s; the stall diagnostic now finds only pucks
  pinned in the back corner (y 0.041, x 0.959), no held pucks.

## Queued next (not started)

1. Shots off moving pucks: the hybrid student is weak at one-timers
   (receiving requested fast shots 6-12/256 vs production 158-179). Label
   every non-threat state with production (the teacher's threat test false),
   not only slow/outgoing ones, so production's receiving shots are kept.
2. Longer anti-holding fine-tunes, selected on proxy goal share with a
   reachable-stall gate (< 100 s per 64 games) and hot endurance.
3. Transfer: repeat the hybrid DAgger and fine-tune under the measured paddle
   contact (`--paddle-restitution-range 0.40 0.65 --paddle-tangential-range
   0.76 0.84`); the plumbing is in place and arm B shows it trains.
4. Teacher gaps: side-rail recovery (1/128), receiving one-timers, puck
   control; sim gaps: goal posts, puck dropouts near the ends, end-rail
   coefficients (see the critic notes in this session).
5. Table: `bash ai/bin/play.sh --policy neural:teacher-dagger-20261018-v1
   --gentle` first, at the user's discretion. Watch bank defense, corner
   behaviour, heat (drive RMS) and whether it ever holds the puck.

## Runs

| Run | What | Status |
|---|---|---|
| `_distill-smoke-20261018` | 3-round DAgger smoke, teacher v0 | done: fast defense 74.5%, 11-12 vs production |
| `_neural-teacher-dagger-20261018-a` | DAgger, merged teacher, default physics | stopped after round 5 |
| `_neural-teacher-dagger-20261018-c` | continuation of A: width 1536, attack-weighted, heat-gated teacher | stopped after round 2 |
| `_neural-teacher-dagger-20261018-b` | DAgger, measured paddle contact (pre heat gate) | stopped after round 6 (memory) |
| `_neural-teacher-dagger-20261018-d` | production widened to 1536; teacher defense labels, production attack labels | done (6 rounds) |
| `_neural-teacher-ppo-20261018-d` | PPO fine-tune of D, teacher-regularised defense | stopped at 51M: better defense, learned to hold the puck |
| `_neural-teacher-ppo-20261018-e` | continuation from 51M with possession delay 30/s | running (45 min) |
| `_neural-teacher-dagger-20261018-e` | continuation of C with an earlier-striking teacher | stopped after round 4 |

## Results

Development suite (`suite.py --tier dev`, seeds in `TIERS['dev']`):

| | production v1 | teacher v0 |
|---|---:|---:|
| fast attacks blocked (768, 10-18 m/s, 2/3 banks) | 28.6% (S 72 / L 2.3 / R 11.3), depth 0.63 m | 82.2% (S 97 / L 71 / R 78), depth 0.17 m |
| immediate release (768) | 25.7% | |
| requested fast shots, stationary S/L/R (of 128) | 75 / 38 / 42 | |
| requested fast shots, receiving S/L/R (of 128) | 88 / 86 / 81 | |
| slow control: fast aimed return | 61.7% | |
| games vs production, 16x60 s, blue / red | 24-24 / 16-23 (self-play noise) | 8x120 s: 8-2 / 5-3 |
| hot 4x600 s (0.8, gain 1.3) vs production | 15-13, overload 511 s, peak 1.057, stall 1361 s | |
