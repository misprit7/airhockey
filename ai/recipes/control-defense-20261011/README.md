# No-imitation control and defense experiment

Launch from the repository root: `PYTHONPATH=ai OMP_NUM_THREADS=2 python ai/recipes/control-defense-20261011/run.py`.
Simulation only. No hardware calls, firmware changes or automatic deployment.
Live report: http://localhost:8420/training/reports/20261011

Starts from `rail30-100-20261001-v1`, SHA256
`39aa7a501edc127bc37fd2e500578366c2b0a735a45eb0b628acd07c5bddf629`.
Both old-policy imitation sources are removed, with both loss weights zero.
Warm-start retains the actor, not an experience replay buffer. The critic and
optimizer start fresh; long continuations resume their complete PPO state.
Old policies remain opponents. Stationary failure fixtures are initial states,
not action targets. Runtime remains one neural network with its existing shot
request input and ordinary arrival/motion decoder.

Five matched 20M-transition arms: no new rewards, defensive depth weight 200,
stricter control alone, and control with depth weight 200 or 800. Every arm
receives the same expanded slow-puck curriculum and unchanged physical limits
(100 m/s², rail30, four identical 2331 motors). Actor LR 1e-5, KL budget .02;
learning rate is shared rather than confounded with reward comparisons.

Strict capture requires <0.15 m/s for 0.20 s, recent contact, <7 cm paddle-edge
gap, reachability and 4 cm inside forward reach. Capture credit is once per
possession. First-touch cushioning of 0.15–2 m/s pucks receives signed feedback;
stationary pucks and repeated taps earn no such bonus. Control potential uses
0.4 m/s instead of 2 m/s as its velocity scale. Fast aimed shot and requested
route rewards remain; receiving trials continue through a shot. No rule
selects a waiting, catching, or shooting action.

Periodically evaluate exact shared development cohorts: 768 fast on-goal
attacks, 384 slow-puck trials, plus self-play replays. Includes intermediate
checkpoints in selection. Three heuristics compare deep/forward bank-aware
interception and deep direct interception. They see the same delayed/noisy
physical observations and use the same arrival decoder and motion/load model;
they have no hidden shot/launch information or per-trial physics parameters.
They are evaluation-only and never supply training action targets.

After the reward sweep, compare width 1024 and 1536 from the same selected
parent with function-preserving initialization and 20M transitions each. Continue
the best two distinct branches for 96M transitions each. Estimated total is
332M transitions plus evaluation, roughly 4–6 hours on this machine; this is a
budget, not a claim that training will converge.

Development ranking = block-rate gain + 0.6*settle-rate gain + 0.3*fast-aimed
return gain, penalizing fast-return regression beyond 10 points and defense
regression beyond 3 points. Any observed thermal overload disqualifies that
checkpoint from ranking. Final validation uses unused seeds: 3072 attacks,
1536 immediate-release attacks and 1536 control trials, plus shooting skills
and matches against three past policies in both player roles.

Acceptance requires positive lower 95% paired normal-approximation confidence
bound on defense improvement, settling regression <=2 percentage points,
fast-aimed return regression <=10 points, immediate defense regression <=3
points, aggregate ladder goal-share regression <=5 points and zero observed
candidate overload in these tests. These tests do not establish physical
safety or universal skill. Retain the current production model if a gate fails.

State/results: `logs/neural-player/control-defense-20261011/experiment.json`.
Per-arm exact argv, logs, evaluations and SHA256 checks accompany each run.
The generated HTML is served directly and refreshed every minute. Failures
are reported in the page rather than silently claiming the experiment finished.

## Completed results and review

The sweep completed at 04:26 EDT on October 11. Nine actual training branches
used 332,529,664 transitions (about 3.43 hours of training plus evaluation).
No replacement was qualified or deployed. No training process remains active.
The initial handoff after launch was premature: the script completed its fixed
sweep, but interactive diagnosis and follow-up were not performed until the
user returned. Repository instructions now explicitly require ownership of
that full cycle and a durable handoff across context boundaries.

Selected development candidate:
`runs/_neural-control-defense-long1-20261011/agent_step_101318656.pt`.
This is the **1536-wide, depth-200 branch with original capture rewards**;
it is not evidence that the new stricter capture reward succeeded. The
stricter-reward short branches did not beat that branch. Larger capacity was
promising in this one matched-seed experiment, not conclusively established.

Fresh-seed qualification:

| Metric | Production baseline | Experimental candidate |
|---|---:|---:|
| Fast attacks blocked, 3072 trials | 27.2% | 33.4% |
| Left banks blocked | 2.1% | 9.4% |
| Right banks blocked | 9.2% | 16.5% |
| Near-stop control, 1536 trials | 3.5% | 17.4% |
| Control followed by aimed shot | 3.1% | 10.9% |
| Fast aimed returns in control tests | 60.4% | 51.0% |
| Goal share across past-policy ladder | 58.1% (125–90) | 40.6% (69–101) |
| Games with candidate-side modeled overload | 0/48 | 4/48 |

Paired 95% approximate CI for the block-rate improvement: +4.9 to +7.4
percentage points. The candidate still sits too far forward (mean depth
55.4 cm at release, versus 63.2 cm before) and is poor against both banks.
Most settling improvement is on incoming pucks; drifting-away control
remains below 1%. In standalone stationary shot drills, first requested
on-target shots at >=6 m/s declined from 80/192 to 64/192.

The overload events were predominantly low-speed edge/corner stalls, not
bursts at the 100 m/s² cap. Candidate first-overload positions include a
stationary paddle at (0.918, 0.081) m with requested acceleration about
6.6 m/s². Modeled overload time totaled 107.58 player-seconds across the
48 games. Reducing requested acceleration cannot eliminate the holding
load at such a position. These are simulation predictions, not hardware
measurements; no physical robot was activated by this experiment.

Interpretation: removing imitation and increasing capacity allowed some
new behavior, but also lost productive shooting and corner recovery.
The isolated-shot development ranking was too permissive about full-game
regression. Those regressions appeared only in final validation. A bigger
sweep with the same ranking would not be the appropriate next step.

Follow-up performed during review: `review_early_checkpoints.py` checks the
20M depth-200 model and the 20M wider model against production, both player
roles, on a new seed (12 games x 90 s per role). Results and replays are in
`logs/neural-player/control-defense-20261011/followup/`. This checks whether
an earlier checkpoint avoided the late regressions; it is a development
screen, not a fresh final qualification cohort.

Next experimental design: include sustained cross-play, stationary shot
completion, drifting-away control, and corner escape checks at intermediate
checkpoints, before choosing continuations. Preserve shooting through fresh
practice/rollouts rather than restoring the old action imitation. Add
starting states from the observed corner stalls so the network can learn
escape under holding load. Increase emphasis on fast bank defense while
requiring that full-game and thermal gates continue to pass. Stop or redirect
branches that improve an isolated metric by sacrificing those gates. Keep
production `rail30-100-20261001-v1` until a candidate passes all of them.

Follow-up results (new development seed, 24 games per candidate):

| Earlier checkpoint | Goals vs production | Modeled overload time | Peak modeled load |
|---|---:|---:|---:|
| depth200, 20M | 39–44 | 41.0 s | 114.8% |
| capacity1536, 20M | 39–40 | 0 s | 94.4% |

The early wider model is a promising restart point for a subsequent experiment,
but an approximately even match and one zero-overload screen do not establish
superiority or qualify it for production. This review and follow-up are now
complete; no additional training was launched during the review.
