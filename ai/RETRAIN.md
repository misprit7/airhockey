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
- [ ] User go-ahead.
