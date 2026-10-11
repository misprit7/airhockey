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
