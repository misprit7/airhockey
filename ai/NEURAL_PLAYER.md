# Neural air-hockey experiments

These experiments train one neural actor with PPO. They do not open a hardware
connection or change physical deployment defaults. Run commands from the
repository root. Experimental checkpoint directories are marked `deployment_ready: false`.

**Most recent packaged policy: `neural:possession-20260926-v3` — known edge-recovery failure.**
Exact weights are run97 step843792384, selected after five sustained hot/cool
load audits. [Pinned self-play replay](http://localhost:8420/?replay=neural-possession-candidate-20260926-v3.json).
It retains the reset fix and stronger defensive preparation. Slow outgoing
control improves from32/1536 for v2 to161/1536; ordinary requested fast shots
are preserved. Immediate outgoing fast returns decrease, and overall head-to-head
superiority over v2 is not established. Long-game quiet-puck stalls remain.
See `POSSESSION_RESULTS_20260926_V3.md` for the measured comparison and launch
command. The pinned replay itself stalls on a reachable left-edge puck from about20s to180s; this is a failed possession case, not an unreachable-puck reset bug. Targeted edge training is in progress. v2 remains available as the previous qualified option. The earlier
package without a version suffix remains withheld.

The actor is a three-layer ELU network, 256 units per layer by default. Its six outputs request an
arrival position, terminal velocity, arrival time, and acceleration allowance.
The existing arrival decoder and kinematic guard translate these into ordinary
firmware-profile commands. There is no tactical interception, shot-selection,
or cooldown controller attached to the actor.

During physical play, the runner brakes on a goal or prolonged puck-tracking
loss as before. After **one second without a puck detection**, it returns to
the center of the robot's workspace at up to 0.5m/s and 2m/s², once the brake
has settled and fresh controller velocity confirms it is nearly stopped.
It stays centered until the puck has a stable in-table track again, then
resumes the policy with its normal session limits. No new launch flag is needed.
At startup without a puck, the one-second timer begins with the tracking loop.

Inputs are 42 measured/estimated physical features, previous commands, and
cached motor-load estimates. `--history 4` supplies the latest four physical
frames to the same network. `--shot-conditioned` appends three current-possession
request inputs: left bank, right bank, straight. Requests are sampled uniformly
on entering a half and remain fixed throughout that possession. Training task
identity and reward bookkeeping are absent from actor inputs. The separate value network receives
training context but is not used to choose actions at evaluation time.

## Continuous possession experiments (September 26)

The current possession experiments add `--continuous-rallies` and
`--possession-followthrough`. These remove the old three-second possession
re-serve and let receiving exercises continue through control and a shot.
Reachable quiet pucks stay in play; only genuinely unreachable dead pucks are
re-served after eight seconds. Early trials used a 120-second game boundary;
the later runs and selected checkpoint use 900 seconds with value bootstrapping.
Replay evaluation does not reset at this boundary.
Replay goal/dead-puck re-serves have explicit event labels.

Outgoing recovery practice rewards getting ahead, measured control, then a
requested shot. Evaluation `--recovery` version 2 stops at the **first loss of
possession**; version 1 could count a later return from the opponent and must
not be compared with version 2. `eval_neural_preparation.py` separately tests
whether the policy chooses a useful defensive position before a delayed fast
shot. A training-only request-consistency loss discourages an expired shot
request from steering defense; the deployed inputs/actions are unchanged.

Optional recovery/quiet exploration floors, successful learned-trajectory
replay, failure-state resets, and geometric reward potentials are training
mechanisms only. They do not add a runtime planner, tactical controller, or
teacher. Hot-load qualification still applies to each exact checkpoint; an
improved skill score alone does not establish production suitability. See
`NEURAL_PLAYER_PROGRESS.md` and the per-run qualification reports for results.

## Training

`ai/bin/train_neural_player.py` uses CUDA when available, otherwise CPU.
`--device cpu|cuda|auto` selects the device. A fresh curriculum run can start
with the following command on this machine:

```sh
PYTHONPATH=ai /usr/bin/python3 ai/bin/train_neural_player.py \
  --run-name _neural-example --stage 0 --capture-first \
  --n-envs 1024 --workers 4 --steps 8000000 --minutes 60
```

Later stages broaden physical starts, incoming speeds, and defense, then mix
practice with full games against learned neural snapshots. `--resume` restores
this format's model and compatible optimizer state. Frozen learned references
can be supplied with `--opponent-checkpoint` and repeatable
`--opponent-reference`. The exact settings, parent checkpoint, and source hashes
are saved in each run's `run.json`; immutable source copies are in `source/`.

The experiments use 60 m/s² acceleration and 12 m/s speed ceilings inside their
simulation environment. Reward weights and training shutdown margins are run
options. `--thermal-gain 1.3` pins load to the high end of the provisional model;
it does not alter physical hardware configuration. Heat persists across full
game episode resets and goals. A training shutdown is penalized as a failure.

Adding physical history or shot conditioning zero-pads input weights to preserve the starting
policy's action function. It requires a fresh optimizer because parameter
shapes change. Within a fixed architecture, compatible optimizer moments are
retained. All current neural experiments descend from fresh neural initializations
made for this restart, not the rejected hybrid controller's encoder.

Hidden width is inherited on resume. `--width 512` can widen a smaller actor
without changing its starting action/value functions: new hidden features have
zero initial influence on the existing outputs. Widening resets optimizer
moments and remains a single network; it does not add inference-time teachers.

## Evaluation and replay

Historical September 24 development: the three-route candidate was `_neural-player-requests-modern26`
(step 443,498,496). Receiving/load refinement continues in
`_neural-player-requests-defense30`, following control29 and receive28. Earlier stage15 reward-only continuation
did not escape the right-bank preference. The 45-input actor has a uniformly sampled straight/left/right request per
possession. Accurate requested shots earn extra reward; wrong prepared routes
lose reward. An immediate accurate return can still earn shot credit when
control fails, while capture followed by a shot earns an additional bonus.
Unproductive completed visits with at least 100 ms of workspace opportunity
receive a modest penalty. These are reward changes, not action overrides.

`--productive-receive-drill` optionally mixes short receiving exercises into
the curriculum: measured control earns 60, an aimed return at least 4 m/s earns
35, and failure earns -30 plus distance/speed costs. These exercises remove
positive contact bonuses to keep control preferable to immediate volleys.
They end at success or after 1.2 seconds. Full games keep their usual rewards
and termination rules; the actor receives no exercise identity. Receive28
practices incoming speeds of 5–8 m/s alongside full games.

Control29 adds `--control-potential-weight 30`: a smooth, bounded reward
potential for slowing a recently touched puck nearby in reachable space. It
uses the usual discounted potential difference, with zero terminal potential,
instead of a repeatable holding bonus. It changes training feedback only.

`--target-kl` stops further actor updates within the current rollout if sampled
policy divergence exceeds the threshold. The independent value network keeps
training. This protects precise warm-started actions from large PPO changes;
it is a training constraint, not an inference controller.

Defense30 retains full self-play while increasing fast defensive practice.
`--practice-defense-fraction` now also applies to productive receiving curricula.
`--defense-min-speed` sets the direct-shot speed floor;
`--random-defense-start-fraction` varies initial paddle positions in training;
`--defense-clear-reward` ends isolated defense exercises on measured control
or a touched puck cleared into the opponent half. These settings do not choose
actions or alter explicit evaluation fixtures. Direct/bank attacks are sampled
independently of training slot indices.

`--request-suite` evaluates the same physical skill fixtures under each request.
The success denominator includes trials with no shot. Reports include requested
versus executed route counts and full-game outcomes split into incoming speeds
below 2, 2–5, 5–8, and at least 8 m/s. A route means the first contacted side
rail; straight means no side-rail contact before the goal. Only aimed shots count
as requested-shot successes. Multiple-bank shots are classified by first rail.

`ai/bin/watch_neural_requests.py` evaluates saved checkpoints, writes reports,
and refreshes `neural-requests-wip.json` with a complete self-play game and visible
request labels. It also schedules held-out and hot-load evaluations after the
run ends. A refreshed WIP replay is not a qualified or promoted checkpoint.
Modern26's replay is `neural-requests-modern-wip.json`; receiving refinement
uses `neural-requests-receive-wip.json`, and control29 uses
`neural-requests-control-wip.json`. The watcher defaults to randomized
practice opponents and separately evaluates 5–8 m/s receiving at each snapshot.
Defense30's latest-replay alias is `neural-requests-defense-wip.json`.
The watcher tests 8–12 m/s direct shots, ready-position banks and
varied-position banks at every checkpoint, with larger fresh-seed and hot-motor
versions at completion. Strong defense is required before selecting a final
candidate; the completed control29 policy has not met that requirement.

Training replays are now retained as `<run-name>_step_<step>.json`, so the UI
groups them under the actual run and shows checkpoint progression. The familiar
WIP link remains an alias to the most recently published checkpoint. A starting
policy replay is published before the first learned checkpoint, including when
the watcher starts before the trainer finishes writing `agent_initial.pt`.

Additional conditioning experiments use request-dependent setup rewards and
`--shot-setup-fraction` for easier stationary training starts. This option never
changes supplied evaluation fixtures. `ai/bin/distill_neural_requests.py` tests
transferring previously learned straight and bank skills into the same actor;
teachers and left/right data reflection are training-only. Its saved model is
an ordinary 45-input network, followed by PPO and unchanged physical/load
evaluation. Reflected load features are an approximate training prior because
the measured motor model is asymmetric. See the progress log for branch results.

Use `ai/bin/eval_neural_player.py CHECKPOINT --output REPORT.json`. The evaluator
loads the recorded architecture, including physical history. Both sides use the
same actor in self-play. `--opponent-checkpoint` selects a different learned
opponent; `--swap-sides` evaluates the candidate as red. `--stochastic` samples
the learned Gaussian policy; the default uses its mean.

`--record ai/recordings/NAME.json --seconds 180 --games 8` saves the complete
first game, not selected highlights. Open it directly at:

```text
http://localhost:8420/?replay=NAME.json
```

For long load tests, use `--games-only --seconds 3600 --initial-load 0.95
--thermal-gain 1.3`. Evaluation never terminates a game or clears accumulated
heat at an overload. `--audit-motion` independently reproduces actual simulator
integration intervals with the firmware motion implementation and reports
acceleration peaks, workspace margins, and reproduction error.

Skill options include randomized practice opponents, 8–12 m/s bank shots,
random defensive starts, and fast receiving cohorts. Use a new `--seed` for final
qualification after selecting a checkpoint. Historical metric versions before
v4 did not reliably seed underlying physics; do not treat them as paired tests.

## Interpreting results

- A controlled reception requires puck speed below 0.65 m/s, close reachable
  proximity to the paddle, recent contact, and 0.12 s of sustained control.
- On-target measures whether the resulting puck trajectory would enter the goal
  without an opponent blocking it. Actual goals are reported separately.
- Shot events distinguish strikes after measured control from other returns.
  Other returns include defensive contacts and quick strikes from slow pucks;
  they should not all be described as deliberate attacking shots.
- Full-game visit outcomes include contact, control, aimed shots, and untouched
  visits with at least 100 ms of geometric workspace opportunity. Geometric
  opportunity does not prove dynamic reachability from the current paddle state.
- Short replay quality, isolated drill accuracy, cross-play strength, motion
  limits, and sustained motor load are separate checks. Passing one does not
  establish the others.

The load model is a provisional fit from one recorded session plus explicit
conservative priors. Passing its tests is simulation evidence, not physical
validation. The physical runner now supports explicit `neural:` checkpoints as
described below. Physical testing requires separate, express authorization.

See `NEURAL_PLAYER_PROGRESS.md` for the experiment history and qualification
status. No checkpoint should be called final merely because it is newest.


## Running the neural player on the table

The production runner accepts `neural:<run-name>` or `neural:<checkpoint-path>`.
Use an explicit checkpoint to keep a session pinned. `tdmpc2:latest` and all
existing deployment defaults remain unchanged; neural experiments are not
promoted into that selection. The metadata's `deployment_ready: false` records
that these are experimental candidates, not a completed performance qualification.
Explicit `neural:` selection permits testing the current implementation.

From the repository root, this command **starts the master, enables the robot,
and plays against the human** using the latest defense30 checkpoint:

```bash
bash ai/bin/play.sh --policy neural:runs/_neural-player-requests-defense30/agent_step_515850240.pt
```

It uses 50 Hz decisions, 12 m/s speed and 60 m/s² acceleration ceilings from this
run's metadata. The network chooses its per-command acceleration below that
ceiling; the shared motion guard can raise braking authority. The default
`--shot-type mix` requests straight, left bank, and right bank uniformly at each
entry into the robot half, holding the request through the possession.
Use `--shot-type straight`, `left`, or `right` to fix it.
`--accel 40000` lowers the session ceiling to 40 m/s²; `--gentle` selects the
existing slow preset. These overrides change the dynamics seen during training.
No firmware changes are required. Ctrl-C uses the existing brake/stop cleanup.

Entirely offline checkpoint/decoder preflight (no camera, master, or robot):

```bash
bash ai/bin/play.sh --policy neural:runs/_neural-player-requests-defense30/agent_step_515850240.pt --check-policy
```

Add `--dry` instead for camera observation and inference without opening the
hardware master or sending commands. Motor load is modeled in this mode.
The launcher preflights neural models before starting the master in live mode.

Deployment shares the neural observation builder, actor, arrival decoder, and
kinematic guard with training. Fresh controller samples are projected forward
by their measured age through the command history; observation and decoder use
the same projected state. Camera positions and finite-difference velocity provide
the fallback when controller telemetry is unavailable. A copy of the firmware profile estimates unreported
acceleration and executes delayed feedback of the commands actually sent,
including watchdog holds. This acceleration estimate and the fixed 15 ms command
latency remain approximations to verify against the next physical recording.

The runner polls the master's cached LOAD response at 10 Hz on its existing
connection. Fast RMS for motors 0–3 comes first, followed by slow RMS 0–3;
percentage-of-shutdown values are divided by 100. Validity and per-field sample
age are checked. Missing/stale (>0.5 s) channels use the persistent thermal model
with gain 1.3; unknown initial levels start at 0.8. Heat is never cleared by a
goal or policy reset. This supplies learned load avoidance, not a guarantee
against a drive shutdown.

The existing goal/puck-loss watchdog still holds position and resets policy
history after stable reacquisition. Stale own-paddle data also pauses neural
inference and holds a fixed target. Session CSVs now contain pipe-delimited
`neural_obs`, `neural_action`, and `neural_load_fresh` columns, alongside requested
and sent commands; replay metadata pins the checkpoint when launched normally.
All integration validation is offline; no robot was activated during this work.


September 25 follow-up: recorded sessions exposed stale controller-state inputs
and a launcher Ctrl-C race. Both are corrected offline; see
`logs/analysis/neural-live-20260924/README.md` for evidence and remaining physical
validation. Near-contact replay events now retain raw marker evidence.
The defensepower31 continuation adds optional `--wide-defense` practice and
`--shot-power-exponent 2`; defaults preserve earlier training/evaluation behavior.


`--report-sensing` opts neural training/evaluation into the deployed encoder's
bounce-cut velocity fit and real-fix dropout handling. It changes the simulated
sensor model, not actions or network inputs. Evaluation follows checkpoint sensing metadata by default; use `--report-sensing`
or `--no-report-sensing` to explicitly select a common model for an ablation.
Compare candidates and references with the same mode; historical scores use
legacy sensing unless explicitly marked.
The mode is recorded in evaluation JSON and training options/source hashes.

## Possession and preparation training (September 26)

The possession experiments opt into `--continuous-rallies`. This disables the
legacy three-second possession clock and the short attended-puck timeout.
A reachable stationary puck remains in play. Goals still re-serve normally;
a puck that is stationary outside both paddles' actual contact workspaces is
re-served after eight seconds. New replay frames label these events. Replay
generation inherits this setting from checkpoint metadata.

`--possession-followthrough` keeps receiving exercises running after control,
until a shot, first possession loss, goal, or the six-second exercise deadline.
Full games have their own horizon, now 900 seconds in the later experiments.
Measured motor heat persists through goals and ordinary game resets.

The curriculum adds slow outgoing pucks, first with the paddle already ahead,
then with gradually wider starting angles and trailing starts. Optional rewards
measure a collision-avoiding approach to the puck's leading side, actual
cushioning, control, and conversion into a requested shot. The first-contact
cushion bonus cannot be earned repeatedly. These measurements supply training
rewards only; they never produce actions or actor inputs. The final policy is
still one network with 45 inputs and six arrival outputs.

Defense practice now includes a delay before an 8–12 m/s goal-directed launch,
so the policy must first choose a useful waiting position. Coverage rewards
measure whether it can reach the possible shot paths without prescribing a
home position. A training-only consistency loss discourages an expired shot
request from changing far-half defense. Additional opponent-style practice
keeps a neural opponent on one route for a whole game while the player's own
shot requests continue to vary.

Optional exploration floors apply to recovery/quiet training observations and
shrink near the motor-load limit. Later runs explore arrival time and effort as
well as position and velocity. Deployment remains deterministic. Cold short
drills teach bursts, while hot long games teach sustained load management.
Run-specific options do not change physical deployment defaults.

The strict recovery benchmark is version 2: it ends at the **first** loss of
the initial possession. Older results could include control after the opponent
returned the puck and must not be used to claim successful outgoing recovery.
Qualification also measures preparation saves, requested shots at least 6 m/s,
continuous self-play, cross-play in both colors, and long hot-load simulations.
Reachable-stall seconds are summed over the puck-owning side and divided by
game time, not twice game time. In cross-play, only the selected candidate's
motor loads count against its qualification; the old opponent is reported
separately. The checked report summarizer handles this attribution:

```bash
PYTHONPATH=ai python3 ai/bin/summarize_neural_qualification.py \
  logs/neural-player/requests/possession51/qualification-final
```

Exact experimental outcomes and rejected checkpoints are recorded in
`ai/NEURAL_PLAYER_PROGRESS.md` and the run-specific qualification directories.
Training progress alone is not production qualification.


## Current follow-up experiment

Run `_neural-player-requests-possession107` trains from packagedv3 on slow outgoing
possession and defensive preparation, with48M additional steps and8M evaluation
intervals. It penalizes losing a previously reachable slow puck even after brief
capture, and trains against lateral release uncertainty and moving preparation.
This is an unqualified experiment; production weights are unchanged.
[Live status and checkpoint replays](http://localhost:8420/training) show progress,
slow-drift control-to-shot results and moving-release saves. Use that page for
current activity rather than historical status statements below.

## Training status and edge recovery

Use [Training & evaluations](http://localhost:8420/training) for current activity,
progress, saved checkpoints, evaluation results and exact replay links. The page
refreshes every5seconds and is linked from the main UI header and Replay panel.
An old running flag without a matching process is shown as interrupted.

Run `_neural-player-requests-possession106` finished128,057,344additional steps in
1h44m. All8development checkpoint evaluations completed. None passed the screen;
final checkpoint971849728 recovered5/198edges, controlled0afterward and completed
2requested fast follow-ups. No replacement is selected. No training or evaluation
process was active when this status was reviewed. The latest replay alias now
shows106final, not its initialization, and remains an unqualified experiment.

The packagedv3 replay's wall puck is contactable; its stall remains a known policy
failure. Production packaging and experimental training are separate on the page.
Future runs are discovered from runs/*/status.json and run.json. Evaluations can
register their artifact directory with run.json's evaluation_dir; historical
possession runs are discovered automatically. Run-local review.json provides an
explicit selection decision/limitations without altering checkpoint weights.
