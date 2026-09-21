# 3.12: acceleration and shot accuracy

Requested 2026-09-20. Resume `3.11-shot-ramp-selfplay/agent.pt` for
5,000,000 additional environment transitions (previous run: 1,000,000).
Keep horizon 8, 50 Hz decisions, realistic sensing, measured physics,
32 environments, and the existing 60% self / 20% sniper / 20% weak-goalie mix.
The replay buffer and optimizer start fresh, as in previous continuations;
the model weights resume from the parent.

Changes are local to this run:

- Both simulated robots have a 60 m/s² acceleration ceiling, including DR
  resets. Speed stays 12 m/s. Deployment now reads the selected checkpoint's
  limits from `run.json`: 60 m/s² for this run, 40 m/s² for legacy runs.
- Misses cost up to 15, scaled from zero at 1.5 m/s to full at 3 m/s.
  The existing outgoing-hit detector and lossy-wall predictor classify
  shots; slow setup touches and incoming pucks do not incur this penalty.
  A possession incurs at most one miss penalty, without a patience discount.
  Correcting a miss can still earn the on-target reward.
- On-target reward increases from 30 to 40. The existing held-puck gate
  still applies to shot shaping, but actual goals always earn 100.
- Action-change cost falls from 0.5 to 0.2. The parent's last logs showed
  this cost dominating shot rewards, with many scoreless self-play games.
- Acceleration-fraction cost increases from 0.04 to 0.06 to retain the
  same cost per unit of commanded acceleration at the higher ceiling.

The trainer writes the resolved rewards, actuator range, CLI arguments,
parent checkpoint path, and horizon into `runs/<run>/run.json`.
It saves checkpoints/opponent updates every 50k and replays every 250k.
The longer budget is not evidence of improvement: compare goals and shots.

`ai/bin/run_accuracy_training.sh` runs training and then a fixed evaluation
against sniper, weak goalie and goalie (8 games x 30 seconds per opponent,
seed 7, 6 planner iterations, 60 m/s²). A parent evaluation with identical
settings is in `logs/3.12-accel60-accuracy-baseline.{log,json}`; final results
go to `logs/3.12-accel60-accuracy-selfplay-eval.{log,json}`.
On-target fraction uses outgoing strikes above 1.5 m/s; legacy `shots`
also includes slower touches. Both are heuristic hit detection, not a
collision-event ground-truth counter. `pay_miss` is a component of `pay_shot`.

Parent baseline at 60 m/s² (8 games per row):

| Opponent | Goals for / against, total | On-target strikes |
| --- | --- | --- |
| sniper | 16 / 34 | 7 / 27 |
| weak goalie | 3 / 2 | 3 / 12 |
| goalie | 5 / 0 | 2 / 12 |

The combined strike accuracy is 12/51 (23.5%). This small evaluation is a
reference for the continuation, not a measurement of physical-table play.

Run under the user service `airhockey-train-3-12`:

```bash
tail -f logs/3.12-accel60-accuracy-selfplay.log
systemctl --user status airhockey-train-3-12
systemctl --user stop airhockey-train-3-12
```

This recipe does not authorize a hardware run. The physical runner uses the
checkpoint's limits by default and encodes the applied ceilings in policy
observations. Explicit `--speed` / `--accel` and `--gentle` still override them.
