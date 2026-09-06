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
- [ ] Tracking test WITH the camera at 12 / 20. The six runs of 2026-09-06
      00:44-00:48 were all camera-less (the camera was not running when
      Run was pressed), so they score drives-vs-steps only. What they say:
      at 0.2 m/s every drive's encoder moves exactly 1.000 mm per mm of
      steps (signs +,-,+,-), so the 800 counts/rev step input is
      confirmed on all four. At speed the drives lag their steps unevenly:
      p90 of |encoder - steps| while moving, mm, per motor 0..3 --
      20 m/s²: 13 / 26 / 53 / 63; 40: 16-18 / 36-45 / 54-65 / 80;
      60: 21 / 41 / 69 / 102. Motors 2 and 3 (the near-side pair) lag
      three to five times motors 0 and 1, and are still 5-22 mm off in
      the rest windows after fast moves, while peaking at only 18-47% of
      torque (motor 3: 19%). A drive that lags 100 mm of cable at a fifth
      of its torque is not torque-limited; look at its ClearView tune /
      torque limit before blaming the physics.

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

## Gate before the run

- [x] Full test suite green (on the system Python; the venv's tensordict
      is stale, see memory).
- [ ] Tracking test run on the rig at 12 m/s, 20 m/s² -> CLOSE.
- [ ] `run_full_pipeline.sh` dry check with a tiny step budget.
- [ ] User go-ahead.
