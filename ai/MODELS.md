# Neural-player models

One scheme for every neural player: **`v<major>.<study>.<model>`**.

- **major**: a large or architectural change (network, observation or action
  layout); a checkpoint that cannot simply continue the previous one.
- **study**: goes up each time a person or an AI session finishes a study
  and trains from its conclusions.
- **model**: the models trained within that study, in order.

`ai/models.json` is the registry and the only place a version is tied to a
run directory (`runs/` is not in git). `airhockey/models.py` reads it:

```bash
PYTHONPATH=ai python3 -m airhockey.models                  # every version, latest, difficulty modes
PYTHONPATH=ai python3 -m airhockey.models path latest      # its checkpoint file
PYTHONPATH=ai python3 -m airhockey.models next study       # next free version (study | model | major)
```

**The latest model is `v4.1.2`** (what was called a3). On the table:

```bash
bash ai/bin/play.sh --tension 1.5                      # latest, full strength
bash ai/bin/play.sh --difficulty easy --tension 1.5    # max | hard | medium | easy | beginner
bash ai/bin/play.sh --policy neural:v4.0.0 --tension 1.5
```

Evaluators take the path: `runs/v4.1.2/agent.pt` (each version is also a
symlink under `runs/`).

## Versions

| version | was called | parent | body accel | what it is |
|---|---|---|---|---|
| `v0.1.0` | `rail30-100-20261001-v1` ("production") | - | 100 | PPO network of the rail30 / 100 m/s² work. Training reference and opponent only. |
| `v1.0.0` | `teacher-dagger-20261018-v1` | v0.1.0 | 100 | Widened to 1536 and distilled from the physics teacher. |
| `v2.0.0` | `teacher-ppo-20261018-v2` | v1.0.0 | 100 | PPO with possession delay. Failed its table test (idle jitter). |
| `v3.0.0` | `teacher-ppo-measured-20261018-v3` | v2.0.0 | 100 | v2 under the measured paddle contact. Experimental. |
| `v4.0.0` | `teacher-calm-20261018-v4` ("v4") | v2.0.0 | 100 | Calm-accel cost. 13 table sessions, 2026-10-19 to 11-08. |
| `v4.1.0` | `_neural-table-faithful-20261110-a` ("v5a") | v4.0.0 | 100 | Table-faithful sim, the table people. |
| `v4.1.1` | `..-a2` | v4.1.0 | 100 | Dead-puck clock, guard and effort costs, corner drills. |
| **`v4.1.2`** | `..-a3` | v4.1.1 | 100 | Two imitation anchors on resting pucks. **Latest.** On the table 2026-11-10 and 11-17. |
| `v4.2.0` | - | v4.1.2 | 40 | v4.1.2 continued with its body limited to 40 m/s². Sim only. |
| `v4.2.1` | - | v4.1.2 | 25 | The same at 25 m/s². Sim only. |
| `v4.2.2` | - | v4.1.2 | 15 | The same at 15 m/s². Sim only. |
| `v4.3.0` | - | v4.1.2 | 100 | v4.1.2 continued with a compliant mallet in the sim. Sim only. |
| `v4.3.1` | - | v4.1.2 | 100 | v4.3.0 plus a cost on unproductive returns. Sim only. |
| `v4.3.2` | - | v4.1.2 | 100 | v4.3.0 with a softer mallet on every worker. Base of v4.3.4 onward. Sim only. |
| `v4.3.3` | - | v4.1.2 | 100 | v4.3.1 plus faster recovery drills. Sim only. |
| `v4.3.4` | - | v4.3.2 | 100 | Calibrated soft-mallet sim, slow exits cost 100. Sim only. |
| `v4.3.5` | - | v4.3.2 | 100 | "Control first" rewards. Lost defense; not for use. |
| `v4.3.6` | - | v4.3.2 | 100 | v4.3.4 plus defense drills that continue after the block. Sim only. |
| `v4.3.7` | - | v4.3.2 | 100 | Failed attempt at a roaming difficulty model; not for use. |
| `v4.3.8` | - | v4.3.6 | 100 | Chase drills on pucks leaving at 0.6-2.4 m/s (its 40M-step snapshot). Sim only. |
| `v4.3.9` | - | v4.3.8 | 100 | v4.3.8 with a larger policy step. Defense slipped; not for use. |
| `v4.3.10` | - | v4.3.8 | 100 | v4.3.8 pushed harder (its 40M-step snapshot). Level `new-chase`. Sim only. |
| `v4.3.11` | - | v4.3.8 | 100 | v4.3.8 tilted toward stopping the puck. Lost the gain; not for use. |
| `v4.3.12` | - | v4.3.8 | 100 | v4.3.8's recipe, three times the batch (its 20M-step snapshot). Level `new`. Sim only. |

Majors 1 to 4 keep the numbers the packaged models already had (v1 to v4);
only v1 was an architectural change. From here on a major means what the
scheme says. Arm a4 of the table-faithful recipe is a recipe only; it has not
been trained and has no version.

Older write-ups (`ai/RUNS.md`, `ai/recipes/*/README.md`) use the old names;
the "was called" column is the translation.

## Difficulty modes

A mode is a model plus an acceleration limit for its body. The limit clips
the acceleration cap the model commands and the motion guard's braking; the
model's inputs and its other outputs keep the units it trained in. The table
runner (`run_policy.py --accel`, in mm/s²), the trainer (`--body-accel`) and
the evaluators (a `run.json` carrying `body_accel`) all apply that one rule
(`neural_coordinates.limit_cap`, `tests/test_body_accel.py`).

Before 2026-11-18 the runner rescaled the model's units to the session cap
instead. In sim that cost v4.1.2 half its goal share at 60 m/s² (51%, against
98% clipped), so `--accel` below the trained cap was not usable as a
difficulty setting.

Modes are entries in `ai/models.json` (`difficulty`); change a mode by
editing its `model` and `accel` there (no `accel` = the limit the model
trained under).

| mode | model | limit, m/s² | goal share | robot goals / min | conceded / min | shot speed, m/s |
|---|---|---|---|---|---|---|
| `max` | v4.1.2 | 100 | 97.7% | 5.9 | 0.14 | 7.3 |
| `hard` | v4.1.2, clipped | 60 | 97.1% | 5.2 | 0.16 | 6.1 |
| `medium` | v4.2.0 | 40 | 97.2% | 4.2 | 0.12 | 5.3 |
| `easy` | v4.2.1 | 25 | 90.3% | 2.5 | 0.27 | 3.3 |
| `beginner` | v4.2.2 | 15 | 80.6% | 1.4 | 0.35 | 2.6 |

Sim only: `eval_human_proxy` against the `table` and `table_plus` people
under `--sim-profile table-20261029`, 64 one-minute games each; the three
trained rows pool two seeds (256 games). The sim flatters the robot (v4.0.0
took about 80% on the table against 95-99% here), so read the column as an
order, not a forecast. What falls with the limit is pace: shot speed and
goals per minute.

### v4.2.x: trained under the limit (2026-11-18)

Each is v4.1.2 continued for about 285M steps (290 minutes, three runs side
by side) with `--body-accel`, the teacher's accelerations scaled to the same
limit, and nothing else changed from arm a3
(`ai/recipes/table-faithful-20261110/finetune.py --arm accel --body-accel N
--run-name <version>`). Only the learner's seat is limited; observations and
units stay those of 100 m/s², so v4.1.2 itself is untouched.

Against v4.1.2 clipped to the same limit, same games:

| limit | goal share, clipped -> trained | stall map, stalled trials | worst corner cell touched | rest spells >= 4 s in 64 games |
|---|---|---|---|---|
| 40 | 93.8% -> 97.2% | 5.1% -> 1.8% | 48/48 -> 44/48 | 2 -> 0 |
| 25 | 71.5% -> 90.3% | 5.3% -> 2.9% | 47/48 -> 34/48 | 4 -> 1 |
| 15 | 49.1% -> 80.6% | 6.1% -> 3.7% | 41/48 -> 27/48 | 2 -> 1 |

- The trained models are closer together in strength than the clipped ones.
  For a weaker opponent at the same speed, run the clipped model
  (`--policy neural:v4.1.2 --accel 25000`), or clip a trained one further:
  v4.2.2 at 12 / 10 / 8 m/s² took 69% / 67% / 55% (one seed).
- Worse than clipped: a puck in the LEFT back corner with a right-bank or
  straight request is left alone more often (v4.2.1 34-35 of 48, v4.2.2
  27-28 of 48). No resting-puck gate passes for any of them, v4.1.2
  included.
- Playability gates at the limit (`eval_playability`, `table`): idle cap p50
  7-8 m/s², no tick above 40, forward lunge on incoming shots about half of
  clipped v4.1.2's. Modeled load peaks 0.95-1.00 in the worst game of each,
  as it does for v4.1.2 at full strength (0.99). v4.2.2 is the one to watch
  for heat: 13 s of 32 minutes at or above 0.95 in the playability replay,
  against none for clipped v4.1.2 and none for the other two.
- Every snapshot was scored on both seeds; none is separable from the final
  checkpoint (40: 96.4-98.3%, 25: 89.3-93.9%, 15: 70.2-80.6% with the final
  highest), so each version is its run's final `agent.pt`.
- None of the three has been on the table.

## v4.3.x: the soft mallet and the puck after a block (2026-11-19)

Why, what was changed in the sim and the trainer, every arm and the failed
attempts: "Round 5" in `ai/recipes/table-faithful-20261110/README.md`. In
short: on the table v4.1.2's possessions end in a shot half the time (83% in
the sim it trained in), and the commonest way it loses the puck is its own
block: the rebound leaves at 1-2 m/s with the mallet behind it. The sim's
mallet was rigid; the table's hangs on cables. With a compliant mallet
calibrated on the table's rebounds (`--sim-profile table-20261119`) the sim
reproduces those numbers, and the models below were trained there.

`latest` and `max` are still **v4.1.2**: nothing here has been on the table.
Two extra levels run the two best checkpoints at full strength:

| level | model | what it is |
|---|---|---|
| `new` | v4.3.12 | The balanced one: concedes least, scores as much, gives the puck away slowly less often. |
| `new-chase` | v4.3.10 | Goes after the rebound hardest; scores less in the rigid sim and leaves a resting rail puck slightly more often. |

Against the `table` and `table_plus` people, 64 one-minute games per tier
and seed, three seeds each; possession numbers from 32 of those minutes per
seed in the calibrated sim. "Table" is v4.1.2's own sessions of 2026-11-17
and 11-18 (37 min), for the scale.

| | table, v4.1.2 | v4.1.2 | v4.3.8 | v4.3.10 | v4.3.12 |
|---|---|---|---|---|---|
| rigid sim (`table-20261029`): goal share | | 97.5% | 97.7% | 97.9% | 98.7% |
| rigid sim: goals for / against per minute | | 6.07 / 0.16 | 5.32 / 0.13 | 4.76 / 0.10 | 5.70 / 0.08 |
| calibrated sim (`table-20261119`): goal share | | 94.6% | 95.9% | 96.9% | 97.5% |
| calibrated sim: goals for / against per minute | | 4.75 / 0.27 | 5.25 / 0.22 | 5.08 / 0.16 | 5.60 / 0.15 |
| possessions ending in a shot | 49.5% | 61.9% | 69.6% | 70.4% | 67.2% |
| puck leaves under 2 m/s, robot not playing it | 13.2% | 13.0% | 8.7% | 8.0% | 9.5% |
| puck leaves at 1-2 m/s | 19.6% | 22.4% | 13.9% | 11.6% | 14.9% |
| shots that are a single touch | 35.8% | 28.0% | 22.0% | 21.8% | 24.1% |
| after a block of a >= 4 m/s shot: puck recovered | 25.8% | 27.2% | 29.4% | 35.3% | 30.9% |
| after a block: rebound leaves slowly, untouched | 21.8% | 26.7% | 22.5% | 18.5% | 21.7% |
| lighter compliant sim (`table-20261118`), one seed: goal share | | 97.5% | 97.4% | 98.0% | 97.7% |

What the numbers do NOT show, and what is worse:

- Sim only. The calibrated sim was fitted to sessions played before the
  camera was recalibrated on 2026-11-18 (the puck was seen 13-18 mm off), so
  the table as it is now may be kinder than that sim, and v4.1.2 itself may
  play better than it did.
- None of them takes control of a puck that is moving away (0-1 of 128 in
  the recovery drill, like v4.1.2): they strike it from behind. What changed
  is how often they go for it and how hard the block rebounds.
- Robot against robot they are behind v4.1.2: over 64 one-minute games at
  both ends in the calibrated sim v4.3.12 lost 23-27 with 814 shots on
  target against 968, v4.3.10 19-23 with 904 against 1029, v4.3.8 15-21
  with 926 against 982. Fewer, more prepared shots is the trade; against
  people in the sim it pays, against v4.1.2's defense it does not.
- Gates beside v4.1.2 in the calibrated sim (`eval_playability.py`,
  `eval_resting_puck.py`): motion is about the same (load 0.84-0.85, idle
  cap 7-8 m/s², reversals 47-51 a minute against 46; the cap is above
  40 m/s² in 28-34% of far-half ticks against 23%). Resting pucks: v4.3.10
  stalls on 1.6% of the map against 1.0%, leaves a left-corner puck with a
  straight request 6 times in 48 and once left a rail puck for 8.5 s in 64
  games; v4.3.8 0.9%, 7 in 48, longest 3.6 s; v4.3.12 1.2%, 9 in 48,
  longest 0.8 s. If a puck sits in the robot's left back corner, nudge it.
- The gain is a property of these checkpoints, not of the recipe: the same
  runs 40M steps later had lost it (v4.3.8 at 80M: shots back to 61%). Only
  the large-batch run (v4.3.12) held it from snapshot to snapshot.

## Adding a version

1. Train into `runs/<version>` (or any directory) with the next free
   version from `python3 -m airhockey.models next study|model`.
2. Add the entry to `ai/models.json` (`run`, `file`, `date`, `accel`,
   `parent`, one-line `summary`) and a row to the table above.
3. Move `latest` only when the new model has replaced the old one at full
   strength, and say on what evidence.
