# Defense science — 2026-10-05

Run from the repository root:

```bash
PYTHONPATH=ai OMP_NUM_THREADS=2 python3 ai/recipes/defense-science-20261005/run.py
```

Simulation only. Never opens a camera, motor API, or serial device. Keeps the
production checkpoint and deployment defaults unchanged. The HTML report is
served at `/training/reports/20261005` and refreshes every minute. Outputs,
recipes and process status live in `logs/neural-player/defense-science-20261005`.
The script refuses to overwrite existing training run directories.

## Question and controlled comparison

Does an explicit preference for deeper preparation improve fast-shot blocking?
Start all first-pass actors from `rail30-100-20261001-v1/agent.pt`, preserving
the learned actor and observation/action representation. Start fresh critics
and optimizers equally in all arms because the source is an imitation export
without PPO optimizer state. Keep acceleration 100 m/s², speed 12 m/s, the
3 cm rail buffer, calibrated physics, delayed report observations and thermal
models. All six arms have equal 12-million-transition budgets and the same
training seed; independent longer continuations receive 48 million transitions.

The control includes the new shared drill distribution: 10–18 m/s attacks,
two thirds banks among delayed releases, hidden release times and targets,
varied starting positions, and success for blocking the first attack rather
than controlling or clearing it. Existing direct-only readiness costs are
zeroed in every arm. Thus baseline-versus-control tests this whole curriculum
change; control-versus-depth200/depth800 isolates the added depth reward.
Additional arms independently change the load penalty, actor learning rate,
or preferred depth. The depth penalty applies only while the puck is in the
opponent half and outgoing or slow. It stops during fast incoming attacks.
It never supplies a target, changes an action, or reaches the policy input.

All ordinary game/attack/recovery rewards and physical guardrails remain.
Full games train against self snapshots and fixed historical neural policies.
No scripted tactical controller is deployed.

## Measurement

`ai/bin/eval_fast_defense.py` gives each actor the same seeded fixtures and
randomized physics. Attack routes are balanced; speeds span 10–18 m/s;
initial motor loads are 25% and 70%. The actor prepares for a hidden
0.35–1.2-second delay, then reacts normally. An identical no-defender rollout
must score for a trial to be eligible. This avoids credit for inaccurate banks.

A block requires contact followed by reversal of goalward travel or stopping
the puck below 0.3 m/s, without a conceded goal. Goalward glancing contacts are
followed until resolved. A later return shot/possession earns no extra credit.
Unresolved attacks are failures. This measures termination of the first threat,
not eventual rally outcome. An immediate-release cohort separately measures
reaction from arbitrary positions without time to prepare.

Development cohorts choose continuations and freeze a final candidate before
opening a fresh-seed 3,072-trial cohort. Report Wilson confidence intervals,
route/speed/load splits and release depth. The ladder uses four historical
policies, both side assignments, eight two-minute games per matchup, identical
seeds and a common current environment; old workspace actions are translated.
Separate shot drills check attack quality. No automatic production promotion.

Periodic replays are deterministic self-play. Drill clips deliberately include
both a save and a miss where available, and are labelled as selected examples.
