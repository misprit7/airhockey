# Runs

Names follow `<major>.<minor>-<description>-<stage>` (`airhockey/run_names.py`):

- **major** bumps when a checkpoint cannot resume from the previous
  lineage: a new action space, observation layout, planning horizon or
  model.
- **minor** bumps for every recipe change (reward, opponents,
  demonstrations) that resumes within the lineage.
- **description**: the one thing that changed, kebab-case.
- **stage**: `proximity`, `contact`, `scoring`, `goalie` or `selfplay`;
  a pinned snapshot of a run adds `-<step>k`.

The trainers refuse any other name (names starting with `_` are scratch).
`python -m airhockey.run_names 3 shot-clock selfplay` prints the next free
minor under a major. The full pipeline takes `PREFIX=<major>.<minor>-<description>`.

The pre-scheme names (`retrain40_*`, `runN_*`) are symlinks under `runs/`
and their recordings were renamed. Older runs (`curriculum_*`, `sac_*`,
`ablate_*`, `_bench_*`) predate the retrain and are left as they are.
`ai/RETRAIN.md` has each run's numbers and reasoning.

| version | parent | what changed |
|---|---|---|
| `1.0-accel40-*` | `-` | curriculum from scratch; position-only action, 20-wide obs, accel pinned 40 m/s^2 (retrain40_*) |
| `1.1-resumed-selfplay` | `1.0-accel40-selfplay` | self-play resumed from 400k with the fast trainer (retrain40_selfplay2); -1950k is the table snapshot (retrain40_try) |
| `2.0-accel-action-*` | `-` | curriculum from scratch; accel cap in the action (3 wide), 22-wide obs with time on side, patience ramp (run2) |
| `2.1-patience-floor-selfplay` | `2.0-accel-action-selfplay` | patience floor 0.2 on goals too, accel tax 0.04 (run3) |
| `2.2-cushion-income-selfplay` | `2.1` | cushion and hold income, trap 3, control gate v1 (run4) |
| `2.3-control-gate-selfplay` | `2.2` | continuous hold income, controlled = slowed within reach (run5) -- the user's pick from the replays |
| `2.4-demos-selfplay` | `2.3` | CushionBot demonstrations in the buffer (run6) |
| `2.5-behaviour-cloning-selfplay` | `2.4` | cloning term on the demonstrated actions (run7) |
| `2.6-held-gate-no-defense-selfplay` | `2.5` | defense income 1.0 -> 0.05, controlled = held 0.3 s, goal multiplier follows any touch, control rewards ~ a goal (run8) |
| `2.7-hold-cap-selfplay` | `2.6` | hold income capped at 1 s per possession, no demos (run9) |
| `2.8-speed-ramp-shot-clock-selfplay` | `2.7` | on-target pay by shot speed, shot clock 0.1/step (run10) |
| `2.9-windup-demos-selfplay` | `2.8` | the bot winds up before striking; demos back on (run11) |
| `2.10-windup-income-selfplay` | `2.9` | the wind-up position is paid; the clock counts from the hold (run12) |
| `2.11-drive-income-selfplay` | `2.10` | the drive toward a held puck is paid, linear (run13) |
| `2.12-drive-squared-selfplay` | `2.11` | drive pay by speed squared; on-target floor 2 m/s (run14); -500k is the pinned snapshot |
| `3.0-horizon8-selfplay` | `2.12` | planning horizon 5 -> 8 (run15): first checkpoint that strikes from a hold |
| `3.1-no-demos-selfplay` | `3.0` | same reward, demonstrations off (run16) |
| `3.2-shot-clock-selfplay` | `3.1` | shot clock 1.0/step, wind-up income off (run17) |
| `3.3-turnover-selfplay` | `3.2` | a dead puck on our side is a turnover; the clock has no reach (run18); -300k pinned |

Lineages: **1.x** position-only action (20-wide obs). **2.x** accel in
the action, 22-wide obs, horizon 5. **3.x** horizon 8 (deploy plans at 8
for these, `policy_loader.HORIZON_8_RUNS`). The next from-scratch
curriculum is **4.0**.
