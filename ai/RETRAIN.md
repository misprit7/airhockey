# Retrain checklist (from scratch, curriculum pipeline)

Started 2026-09-06. The plan is to stop patching `curriculum_selfplay_smooth6`
and retrain from scratch through `ai/bin/run_full_pipeline.sh` with the
changes below. Nothing here has been trained yet; the run starts only when
every box that gates it is ticked and the user says go.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done and tested,
`[?]` needs a decision from the user.

## 1. Physical parameters the rig can actually run

- [x] Accel pinned at **20 m/s²** (`AGENT_DR_ACCEL_M_S2 = (20, 20)`), was 60.
      The tick logs say the paddle follows 24 and not 60; the tracking test
      (`ai/airhockey/follow_test.py`) is the tool to firm this up before the
      run starts. Cap features stay in the observation as constants.
- [x] Speed stays **12 m/s** (user's call; not the binding constraint).
- [x] Control rate: **50 Hz** decisions instead of 100 (see item 7). One
      constant, `ACTION_HZ`, derived everywhere: env default, curriculum
      episode lengths (specified in seconds, not steps), trainers, eval,
      deploy `--cmd-hz`, stuck-puck timer.
- [x] Runner defaults (`run_policy.Caps`) follow the same constants.
- [x] Tracking test WITH the camera at 12 / 20 (2026-09-06 00:58): **CLOSE**.
      Camera-vs-controller gap 4 mm p50 / 13 mm p90 at speed, 0 mm at
      rest, arrival lag 2 ms, camera latency 5 ms. At 12 / 40: CLOSE,
      6 / 21 mm. At 12 / 60: CLOSE by a hair, 8 / 26 mm against the 30 mm
      line, max 68 mm on the flips. The first scoring called all three
      LAGGING for a bad reason: the camera lost the marker cluster on four
      moves near the robot-end rail (x -250, y +250, corner ++, corner +-)
      and "never arrived" was counted against the drives. Those moves are
      now reported as unseen, not late. Re-judge any run with
      `python ai/bin/follow_test_rescore.py logs/follow_test/<stamp>.csv`.
- [x] Encoders vs steps: gains 1.003-1.010 on all four (the 800 counts/rev
      step input is right), residual 1-5 mm at rest, 9-40 mm moving. The
      earlier reading of "motors 2 and 3 lag 100 mm" was an artefact: the
      ENC round trip takes ~40 ms and follows the STATUS read, so each
      encoder was compared with steps up to 20 ms younger -- 60-120 mm at
      3 m/s. The fit now searches the read lag per motor (0-20 ms found)
      and scores the residual after it. RETRACTED: nothing points at a
      drive tune.
- [x] Pipeline dry run: `train_tdmpc2.py --curriculum-stage contact` for
      8k steps then `train_selfplay.py` for 20k steps at 8 envs, scratch
      names `_smoke_*`: both trained, recorded games, logged per-opponent
      results (vs external, vs sniper) and the shot counters.

Thoughts: the thermal budget is a separate constraint from following. At
24 m/s² the drives tripped RMS overload within ~30 s of bang-bang play, and
20 is not far below that. If the retrained policy still accelerates flat
out all the time, the fix is a duty cost (parked with the smoothness work),
not a lower cap. The tracking test at 20 m/s² is worth running before the
training run so the number is measured rather than argued.

Items 2-6 below are implemented and unit-tested (`ai/tests/
test_retrain_changes.py`, 20 tests; full suite green on the system Python).
The `[ ]` boxes that remain are things only a training run can tell.

## 2. On-target shots pay much more than forward hits

- [x] Shared predictor `rewards.predict_shot(x, y, vx, vy, cfg)`: the puck's
      free path through the MEASURED lossy-wall model (normal 0.785,
      tangential 0.66, up to 4 rail bounces), returning where it crosses the
      far goal line, when, how many rails it touched and which rail first.
      Lifted from `ExchangeRewardShaper._predict_goal_crossing` so both
      shapers and the shot-type term share one definition of "on target".
- [x] `on_target_reward` (10) paid at the hit for the FIRST on-target shot
      of a possession, in every stage that rewards forward hits (contact,
      scoring, goalie, selfplay). Proximity stage unchanged.
- [x] Shot speed: `shot_speed_weight` (1.0 per m/s) paid ONLY on on-target
      shots, so speed matters when aimed and not otherwise. The old
      `directed_hit_weight * vy` (any forward hit) drops to 0.5.

Thoughts: "on target" ignores friction and the opponent by design (the
user's spec: would it go in with no blocker). A shot that is on target but
blocked still pays, which is the point of the term. Friction makes slow
on-target shots optimistic; the speed term and the goal reward cover it.

## 3. Control the puck before shooting; shot speed

- [x] Possession tracking per env in the shaper: begins when the puck enters
      the agent's half, ends when it leaves (a shot) or on a goal/reset.
- [x] `trap_reward` (2.0), once per possession, the first time the puck is
      brought under 0.3 m/s within reach of the paddle after arriving
      faster than 0.8 m/s. A puck that was already at rest earns nothing.
- [x] `controlled_shot_bonus`: the on-target reward is multiplied by 1.5
      when the possession included a trap, so stop-then-shoot beats the
      instant slap when both are on target.
- [x] Env: the stuck-puck relaunch currently fires after 1.2 s of a slow
      puck ANYWHERE, which yanks a controlled puck away from the paddle
      and fines the agent for it. Change to: 1.2 s if no paddle is within
      reach, 3 s if one is, so control is possible but not indefinite.

Objections/risks: any reward for stopping the puck can be farmed (bump,
stop, bump) or teach passivity. Once-per-possession, the arrival-speed
condition and the 3 s cap are the guards; the sniper opponent (item 5) is
the pressure that makes dawdling lose. Watch the possession length and
the trap rate in the eval, not just the score.

## 4. Shot types on demand (self-play)

- [x] Observation grows 17 -> 20: one-hot `[bank_left, bank_right,
      straight]`, all zero = no preference. Left/right are the robot's
      view facing the opponent (sim x = 0 rail is LEFT).
- [x] Per possession the env draws one of the four with probability 1/4
      each (self-play stage only; zeros in the pretrain stages). The far
      side draws its own when it is a copy of the robot.
- [x] `shot_type_reward` (10) paid when the first on-target shot of the
      possession matches: straight = no rail, bank left/right = the first
      rail touched is that side. No preference: nothing extra.
- [x] `policy_loader.load_checkpoint` pads any narrower checkpoint to the
      current width (was 15 -> 17 only). Scalar `AirHockeyEnv` (UI) grows
      to 20 with zeros. `deploy.ReportEncoder` takes a shot type;
      `run_policy --shot-type none|left|right|straight|mix`.

## 5. Opponent mix in self-play

- [x] Per episode the far side is drawn from a mix, default
      60% copy of self, 20% `sniper`, 20% `weak_goalie` (flag on
      `train_selfplay.py`). The env re-draws each env's opponent at reset.
- [x] `sniper`: scripted, on a FREE body (first-order lag, 5-8 m/s strike
      at 300 m/s², not the robot's accel), waits on its line and, when the
      puck is on its half and hittable, strikes through it at a random
      point in the mouth, a third of the time off a rail. Puck leaves at
      8-12 m/s; against an open net it scores about once every 8 s.
- [x] `weak_goalie`: scripted, slow body (2 m/s, 15 m/s²), tracks puck x
      near its line with a dead zone, never shoots.
- [x] Win rate logged per opponent kind; the recorded games stay vs self.

Thoughts: mixing opponents changes what `train/win_rate` means; the
per-kind rates are what to read. The sniper is also the only thing in
training that produces fast shots at the robot at 20 m/s², so it is what
the blocking is learned against.

## 6. Sensing fuzz (mild)

- [x] 20% of episodes get 1-2 windows (0.3-1.5 s) where the camera does
      not see the opponent's mallet: the observation shows the deploy
      encoder's fallback (parked at the far-side default, at rest), and
      the velocity is zeroed on both edges of the window as the encoder
      does. Applies to both views.
- [x] The same episodes get 1-3 puck dropouts of 50-150 ms, injected into
      `PuckPerception` as forced blindness, so the existing coast model
      handles them exactly as it handles the IR blind spot.
- [x] Parity fix: after the 150 ms coast expires the sim kept reporting
      the last velocity; the deploy encoder reports zero. Sim now zeroes.

## 7. Control rate vs planner depth vs real-time cost

- [x] Benchmark `_plan`/`_plan_batch` on the rig's GPU across iterations,
      samples, horizon, and with CUDA graphs / bf16, single-env (deploy)
      and 32-env (training). `ai/bin/bench_planner.py --compile`.
- [x] Decision: **50 Hz, 6 iterations, CUDA graphs at deploy.** Training
      and deploy share `ACTION_HZ` (dynamics.py) and `PLAN_ITERATIONS`
      (policy_loader.py). `deploy.TDMPC2Policy(compile_plan=True)` wraps
      the single-env planner in torch.compile(reduce-overhead).

Measured 2026-09-06 on the RTX 4090, one env, 512 samples, horizon 5,
milliseconds per decision (p50 / p95):

| config                  | p50  | p95  |
|-------------------------|------|------|
| prior only              | 0.21 | 0.25 |
| 1 iteration             | 4.1  | 5.6  |
| 3 iterations            | 7.1  | 7.6  |
| 6 iterations            | 12.6 | 14.6 |
| 3 it, 128 samples       | 6.8  | 7.5  |
| 3 it, horizon 3         | 5.1  | 5.6  |
| 3 it, horizon 8         | 9.4  | 11.0 |
| 3 it, bf16              | 8.1  | 9.2  |
| 3 it, CUDA graphs       | 3.4  | 3.7  |
| 6 it, CUDA graphs       | 6.3  | 6.8  |
| 32 envs, 3 it (train)   | 66   | 68   |
| 32 envs, 6 it (train)   | 132  | 139  |

Reading: the cost is kernel-launch overhead, not arithmetic. Samples are
free (128 = 512), bf16 is slower, and iterations and horizon are linear.
CUDA graphs halve everything. At 100 Hz even 3 iterations eager left no
room for the master's ~1 ms I/O; at 50 Hz, 6 iterations under CUDA graphs
use a third of the tick. Compile + warm-up costs ~20 s at start-up.
Training throughput in sim-seconds per wall-second is unchanged by the
move (half the decisions per second, each twice as expensive).

Why 50 Hz when 6 iterations take 6.3 ms: what matters for intercepting a
shot is lookahead in TIME, and the planner's cost is linear in horizon
steps. Horizon 5 at 50 Hz sees 100 ms for 6.3 ms; the same 100 ms at
100 Hz is horizon 10, 12.6 ms compiled, which does not fit a 10 ms tick
even before the master's ~1 ms I/O and the tracker. Within a 20 ms tick
the compiled planner leaves 13 ms of margin for GPU hiccups (one eager
row in the sweep hit 114 ms); within 10 ms it would leave two. The price
is 10 ms more reaction latency on average, 50 mm of puck at 5 m/s. It is
a judgement call, and `ACTION_HZ` is one constant if it goes the other
way -- but training and the table must agree, so it is decided before
the run, not after.

**Why the old policy oscillated on the table (2026-09-06, after the
tracking test cleared the drives).** It is the planner, and the sim does
it too. Same checkpoint, sim, puck parked on the far side, nothing
moving: under MPPI the target jumps >300 mm on 17-22% of ticks (3 or 6
iterations) and the paddle sweeps 120-140 mm per half second at 60 m/s²,
up to 460 mm; the prior alone never jumps. TD-MPC2 executes one SAMPLED
elite even in eval mode, and on a flat value landscape the elites are
random samples. Executing the elite mean (`plan_eval_mean`, now the
eval/deploy default) cuts static jumps 20% -> 1% at 3 iterations and,
in play vs the goalie, doubles goals (0.75 -> 1.56 per 20 s; 0.88 -> 1.62
at 6 iterations). What it does not fix: the paddle still wanders at
~1 m/s when idle, because the value landscape has no station in it --
the idle reward (0.005/step) is invisible to the two-hot value bins. For
the retrain that is the argument for a visible idle term or the parked
planner action-change cost; a deployment-only alternative is the prior
while the puck is away and the planner during an exchange.

Still open: in the 12/60 play session the camera put the paddle 200-440
mm from the controller's position, which the tracking test at the same
caps does not reproduce (26 mm p90). The runner's tracker (blobtrack,
200 Hz) and the test's (vision service, 50 Hz) are different pipelines;
the next live session should log both before trusting either.

## Gate before the run

- [x] Full test suite green (on the system Python; the venv's tensordict
      is stale, see memory).
- [x] Tracking test run on the rig at 12 m/s, 20 m/s² -> CLOSE.
- [x] `run_full_pipeline.sh` dry check with a tiny step budget (both trainers, scratch names).
- [x] User go-ahead (2026-09-06 ~01:55): "let's just try training at a
      lower speed" -- accel 40 m/s², idle pull raised to 0.05/step.

## The run

`PREFIX=retrain40 bash ai/bin/run_full_pipeline.sh`, started 2026-09-06
~01:55, log `logs/retrain40.log`, runs `runs/retrain40_{proximity,contact,
scoring,goalie,selfplay}`. Stage budgets 200k / 300k / 500k / 500k / 3M
steps at 50 Hz. Settings frozen at commit e4fd2c5: 12 m/s, 40 m/s²
pinned, 50 Hz, 6 MPPI iterations, elite mean at eval, obs 20, the reward
table in `rewards.CURRICULUM`, opponent mix 60/20/20, fuzz 20%.

**Self-play restarted at 09:55 as `retrain40_selfplay2`** from
`runs/retrain40_selfplay/agent_step_0400000.pt` (2.6M steps to go), with
the trainer speedups measured by `ai/bin/profile_selfplay.py`: at 32 envs
the two planner calls were 93% of an iteration (290 ms each, 6 iterations
x 512 samples, fp32 eager) and compute-bound; TF32 matmuls (1.5x), 256
collection samples (1.9x) and CUDA graphs on both batched planners (1.4x)
take a planner call to 75 ms and the iteration 3.2x. Eval and the table
keep 512 samples. The replay buffer restarted empty (updates resume once
two games are in, ~50k steps). Log `logs/retrain40_selfplay2.log`,
tensorboard `runs/retrain40_selfplay2/logs`. The pretrain stages and the
first 400k self-play steps ran at ~100 env-steps/s.

Observed at 340-400k (before the restart): shots ~40% on target, type
matched ~9%, the sniper beaten 30-6 but scoring 2.3 per game on the
robot, the weak goalie conceding only 1.3 per game -- and TRAPS = 0. The
policy never stops the puck. At 50 Hz with discount 0.99 a half-second
of control costs ~22% of a reward and the controlled shot only pays 3-5
more than an instant one, so stop-then-shoot loses the arithmetic. For a
next run: trap_reward and controlled_shot_bonus several times larger, or
a longer discount.

Things to read while it runs (tensorboard on `runs/retrain40_selfplay2/logs`):
`vs_external/win_rate`, `vs_sniper/goals_against_per_game` (blocking),
`vs_weak_goalie/goals_for_per_game` (shooting), `shots/on_target`,
`shots/traps`, `shots/type_matched`. First table run: `--plan 6` at the
default 50 Hz and 40 m/s², `--shot-type mix`, short sessions, watch
"shed" and the lag in the status line.


# Run 2 (prepared 2026-09-06 ~12:30, not started)

What run 1 (retrain40) showed on the table: 20 s at 12 / 40 tripped motor 2
on RMS overload; the paddle travelled 33 m, was moving 82% of the time and
sat its target on a box edge 48% of the time, 45% even with the puck away.
Traps were zero all run: at 50 Hz with discount 0.99, one second of
waiting cost 39% of a reward and the controlled shot paid only 3-5 more.

Changes, all implemented and tested (`ai/tests/test_run2_changes.py`,
14 tests; suite 340 green):

- [x] **Action = position + accel fraction** (`action_mode="profile_a"`,
      3 dims; speed stays 12 m/s). Slot [-1, 1] -> 5%..100% of the 40
      m/s² cap. Taxed `accel_cost_weight` per step: 0.01 in scoring and
      goalie, 0.02 in self-play (a whole episode at full accel costs 30
      against a goal's 100). The copy of the robot commands its own too.
      The previous action in the obs is three wide.
- [x] **Time on side** in the observation ([21], seconds since the puck
      last crossed the centre line / 5 s), in the env, the scalar env and
      the deploy encoder.
- [x] **Patience**: the hit rewards (on-target + speed, directed, shot
      type) are multiplied by 0.5 + 0.5 * min(1, t_side / 1.5 s). An
      instant slap pays half; a shot after 1.5 s of control pays whole.
      With the trap's 1.5x that is 3x an instant one. The user asked for
      "less reward for hits under 5 s"; 5 s is longer than any real
      possession and the ramp does the same job at 1.5 s -- change
      `patience_s` if that reads wrong.
- [x] **Discount 0.995** in both trainers (was 0.99): 1.5 s of waiting
      keeps 69% of a reward instead of 47%. Without this the patience
      term cannot win.
- [x] Attended stuck relaunch 5 s (was 3) so control has room.
- [x] Pretrain budgets halved: 100k / 150k / 250k / 250k, and UTD 0.5
      (`--updates-per-step 16`) -- the stages were winning their recorded
      games long before their budgets ran out.
- [x] Table: `CMD x y speed accel`; the master forwards ACCEL to the
      Teensy only when it changes (master rebuilt 12:11 -- restart it).
      The runner sends the policy's accel every tick. Older 2-dim
      checkpoints load and play unchanged.
- [x] Pretrain trainer: 256 collection samples and CUDA graphs on the
      batched planner, as in self-play.

Not changed: on-target 10 + 1/m/s, trap 2, controlled 1.5x, shot type 10,
idle pull 0.05, action-change tax 0.2, opponent mix, fuzz.

**Run 1 finished 12:24** (`runs/retrain40_selfplay2/agent_final.pt`;
`runs/retrain40_try/agent.pt` is the 1.95M snapshot used on the table).
Final self-play tallies over 1731 games: vs copy of self 129 W / 207 L /
688 D (0.22 for, 0.32 against per game); vs sniper 292 / 16 / 40 (4.02 /
1.22); vs weak goalie 284 / 3 / 72 (1.55 / 0.01). On target ~40%, type
matched ~10%, traps 0 throughout.

**Run 2 started 12:26**: `PREFIX=run2 bash ai/bin/run_full_pipeline.sh`,
log `logs/run2.log`, runs `runs/run2_*`. Budgets 100k / 150k / 250k /
250k / 3M at UTD 0.5 in the pretrain. What to read in the self-play
stage: `shots/traps` and `shots/patience_sum` (patience actually paid),
`shots/accel_frac_sum / shots/steps` (the mean accel fraction asked for:
1.0 means the tax is being ignored, ~0.3 means idling is cheap), and the
per-opponent lines. First table run: the master must be restarted for
the per-command ACCEL (rebuilt 12:11); then
`python ai/bin/run_policy.py --live --gentle --opponent --policy
tdmpc2:run2_selfplay --plan 3 --shot-type mix`.


**Run 2 finished 17:05** (`runs/run2_selfplay/agent_final.pt`, also
`agent.pt`; the pipeline took 4 h 40 min end to end against run 1's 11 h).
Final self-play tallies over 1990 games: vs copy of self 196 W / 241 L /
727 D (0.30 for, 0.35 against per game); vs sniper 305 / 59 / 49 (3.66 /
1.48); vs weak goalie 282 / 2 / 129 (1.22 / 0.04). Last 100k steps: shots
44% on target, 11% matching the request, 150 shots per 10k steps.

What the new terms did:
- The accel tax WORKED as a lever: the mean accel fraction asked for
  settled at 0.52-0.54 for the whole stage, never drifting to 1.0. On the
  table that should roughly halve the duty run 1 tripped a drive with;
  it is the first thing to check.
- The patience term did NOT: the multiplier on on-target shots went from
  0.76 at the start to 0.62 at the end (shooting sooner), and traps stayed
  at zero. A half-price instant shot still beat a full-price shot 1.5 s
  later. Next lever: a much lower floor (0.2) or a controlled-shot
  multiplier of 4-5x, so the trap route pays several times the slap.
- Blocking improved through the stage (sniper 2.55 -> 1.48 goals against
  per game) and finished a little better than run 1 (1.22 there).

Table: `python ai/bin/run_policy.py --live --gentle --opponent --policy
tdmpc2:run2_selfplay --plan 3 --shot-type mix` after restarting the master
(rebuilt 12:11 for the per-command ACCEL). Watch the status line for
`shed` and `badfix`, and the tick log's cmd_accel column for how the
accel command is being used.


# Run 3 (started 18:21, self-play only, from run 2's final checkpoint)

Two changes on the user's read of run 2 ("still doesn't control the puck;
the accel tax should cut more than half"):

- **Patience floor 0.2, and it now scales the GOAL a shot produces.** Run 2
  scaled only the ~10-point hit rewards and left the 100-point goal whole,
  so an instant slap that scored still paid full price and the policy
  learned to shoot sooner. The patience of the agent's last hit in a
  possession now multiplies the goal it leads to: 20 from an instant
  slap, 100 after 1.5 s of control; the discounted patient route pays
  4-6x the instant one at any plausible goal probability. Logged as
  `shots/goal_patience_sum` over `shots/goals`.
- **Accel cost 0.04 per step** (0.02 in the pretrain stages; run 2's 0.02
  settled the mean fraction at 0.52).

`train_selfplay.py --resume runs/run2_selfplay/agent.pt --steps 1500000
--run-name run3_selfplay`, log `logs/run3.log`, ~1.5 h. The value
function was trained under the old scale and re-adapts; the first 200k
steps are not representative. Watch `shots/patience_sum / shots/on_target`
climbing above 0.7, `shots/traps` above zero, and the mean accel fraction
below 0.4.

Restarted 18:39 (same command, same checkpoint) with one more change
(commit 6c34467): **self-play pays nothing for an off-target shot**.
Contact, directed hit, near-miss placement and the bank/straight mix are
gone from the stage, puck progress is a tenth, and the on-target reward
is 15 + 1/m/s. Runs 1-2 landed 35-45% of shots while a miss still
collected about 5. ETA ~20:05.


# Run 4 (started 19:28, self-play from run 3's 900k checkpoint)

Run 3 at 900k (stopped there): patience on goals worked as a lever (the
multiplier on scored goals 0.48 -> 0.66), on-target 46-47%, but traps
stayed at zero and the accel slot was NOT USED -- fraction 0.48 whether
the puck was near or far, i.e. the network's neutral output plus noise.
Games vs self went almost all to draws (157 of 190 at 500k).

Before changing anything I checked the physics: a paddle retreating at
0.47x the puck's speed stops a 3 m/s puck dead in the sim (0.30x leaves
1 m/s, standing still returns it at 2.6). The skill exists; the reward
never lit the path to it -- a trap is a rare endpoint reached through
touches that merely slow the puck, and those earned nothing.

Changes (commit below), all unit-tested (`tests/test_run2_changes.py`):

- [x] **Cushion reward**: any touch on the robot's side that takes speed
      off a fast puck pays 1.5 per m/s absorbed (1.0 in the pretrain).
- [x] **Hold income**: 0.03 per step while a trapped puck sits under the
      paddle (until the 5 s relaunch), trap itself 3.
- [x] **Control gate** replaces the time ramp: the shot and goal rewards
      pay 1.0x when the possession was trapped and 0.2x otherwise (a puck
      that arrived too slowly to be trapped counts as controlled after
      1.5 s on the side). Time on side stays in the observation.
- [x] **Accel slot: the neutral output is the cheap one.** The mapping is
      quadratic (slot -1 -> 5%, 0 -> 29%, +1 -> 100%), so not using the
      slot means low accel, and a save or a strike has to raise it --
      which the goal and shot rewards pay for. One definition
      (`BatchAirHockeyEnv.accel_fraction`) used by the env, the scalar
      env, the shaper and the deploy path.

`train_selfplay.py --resume runs/run3_selfplay/agent.pt --steps 1500000
--run-name run4_selfplay`, log `logs/run4.log`, ETA ~21:00. Read
`shots/cushion_sum` (rising = the path is being walked), `shots/traps`,
`shots/hold_steps`, and the accel fraction (should now differ between
puck near and puck far).


# Run 5 (self-play from run 4's ~1M checkpoint)

Run 4 at 1M: cushioning rose 127 -> 173 per 10k (touches that slow the
puck are learned), but traps and hold stayed at zero and 79% of shots
paid the floor. Watching 112 possessions of that policy in the sim: the
FIRST touch sends the puck away at 4.2 m/s median, 65% of possessions
are one touch, it waits 0.66 s then slaps -- and in 12% of possessions
the puck ended nearly stopped within reach and it did nothing with it,
because income needed a formal trap (< 0.3 m/s) first and the gate was
binary on that trap.

Changes (commit below):
- [x] **Control income is continuous**: 0.05 per step x (1 - speed/1 m/s)
      while a puck that arrived fast is within reach, no trap required.
      Partial slowing already pays.
- [x] **The outcome gate keys on the slowest the puck got within reach**
      this possession (< 0.6 m/s = controlled), not on the 0.3 m/s trap.
- [x] **Uncontrolled floor 0.05** (was 0.2): a slap is nearly worthless.

`--resume runs/run4_selfplay/agent.pt --steps 1500000 --run-name
run5_selfplay`, log `logs/run5.log`. Read `shots/hold_steps` (should
climb well above run 4's ~15/10k) and the controlled multiplier
(`shots/patience_sum / shots/on_target`, 0.05 = none controlled).


# Run 6 (started 21:06, self-play from run 5's 680k checkpoint, WITH demonstrations)

Run 5 at 680k: held-puck income collected on ~2.7% of steps and rising,
10-15% of on-target shots from controlled possessions (up from ~2%),
goal multiplier 0.5-0.66 -- movement, but slow, and play worse meanwhile
(sniper 26-29-14, all draws vs self).

The reward route was converging too slowly, so the chain is now put in
the buffer directly. `airhockey/cushion_bot.py` is a scripted controller
that waits on the puck's path with room behind it, retreats along the
puck's motion as it arrives (a 0.47x retreat stops it dead in the
physics), holds ~1.2 s, then strikes at the far mouth. Measured in the
sim: controls the puck in 57% of fast possessions vs the weak goalie and
27% vs the sniper, 600-2400 held steps per 16 x 30 s. `train_selfplay.py
--demo-envs 8 --demo-until 1000000` steps eight such envs alongside the
agent's 32 and stores their episodes, shaped rewards and all; the value
function then knows what a controlled possession is worth. Logged under
`demo/*` next to `shots/*`.

`--resume runs/run5_selfplay/agent.pt --steps 1500000 --demo-envs 8
--demo-until 1000000 --run-name run6_selfplay`, log `logs/run6.log`,
ETA ~22:35. Read `shots/hold_steps` and the controlled multiplier on the
AGENT's shots (`shots/patience_sum / shots/on_target`, 0.05 = none), not
the demo/* lines, which are the bot.
