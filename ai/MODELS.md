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
| `v4.3.10` | - | v4.3.8 | 100 | v4.3.8 pushed harder (its 40M-step snapshot). Superseded by v4.3.12. Sim only. |
| `v4.3.11` | - | v4.3.8 | 100 | v4.3.8 tilted toward stopping the puck. Lost the gain; not for use. |
| **`v4.3.12`** | - | v4.3.8 | 100 | v4.3.8's recipe, three times the batch (its 60M-step snapshot). Best of the study; level `new`. Sim only. |
| `v4.3.13` | - | v4.3.12 | 100 | v4.3.12 continued with six times the batch. Equal to it on two seeds; gates not run. Sim only. |
| `v4.4.0` | - | v4.3.12 | 100 | First launch of the self-play arm (a game pays goals and physical costs only). Ends not equal; not for use. Sim only. |
| `v4.4.1` | - | v4.4.0 | 100 | The self-play arm with equal ends (its 101M-step snapshot). About v4.3.12, fewer pucks given away; later snapshots aim at the edge of a goal that had no posts. Sim only. |
| `v4.4.2` | - | v4.4.1 | 100 | The self-play arm continued in the sim with goal posts (its 101M-step snapshot). Same play as v4.4.1 within noise; the aim held to 150M steps and moved out again after. Sim only. |
| `v4.4.3` | - | v4.4.2 | 100 | The self-play arm plus one payment in games for a prepared, fast, on-target shot (100, a sixth of a goal), 111M steps (its 101M-step snapshot). Prepared shots 30% -> 33% of on-target shots in training; nothing measurable against the simulated people. Sim only. |
| `v4.4.4` | - | v4.4.3 | 100 | v4.4.3 continued with that payment at 300 (half a goal); its 101M-step snapshot. 40% of shots from a controlled puck (v4.4.2 30%), 25 prepared fast on-target shots per 100 possessions (17); concedes twice as often to the simulated people; loses to v4.1.2 head to head 8-17, as v4.4.2 does (9-16). Sim only. |
| `v4.4.5` | - | v4.4.4 | 100 | v4.4.4 continued with v4.1.2 and v4.3.12 as fixed opponents in a quarter of the games; its 101M-step snapshot. Keeps v4.4.4's prepared shots and is level with v4.1.2 head to head (20-18, was 8-17); still concedes 0.30 a minute to the simulated people, most of it through the mallet lift. Sim only. |
| `v4.4.6` | - | v4.4.5 | 100 | v4.4.5 continued with the learner's mallet lift on in every game (`--selfplay-lift`); its 202M-step snapshot. 37-43% of shots from a controlled puck, 72-73% on target, 28-31 prepared fast on-target shots per 100 possessions (v4.3.12: 12), concedes 0.22 a minute to the simulated people (v4.1.2 0.32), level with v4.1.2 head to head. NOT FOR THE TABLE: hovers behind a puck at the front edge of its reach (4-5% of resting pucks left; v4.3.12 0.4%), and 3 of 256 rigid-sim games reached the load shutdown level. Sim only. |
| `v4.4.7` | - | v4.4.6 | 100 | v4.4.6 continued with the front-edge resting-puck drill (`--reach-edge-drill-fraction 0.3`); its 202M-step snapshot. Best shot quality so far: 47% of shots from a controlled puck, 75% on target, 36 prepared fast on-target shots per 100 possessions (v4.3.12: 12), 0.22 conceded a minute, level with v4.1.2 head to head (18-19), load peak 0.92. Resting pucks left 1.3% (v4.4.6 4-5%, v4.1.2 1.0%), the front edge fixed. NOT FOR THE TABLE yet: in 2 of 64 games against the simulated people a puck lay against the right rail for 43 and 56 s. Sim only. |
| `v4.4.8` | - | v4.4.7 | 100 | v4.4.7 continued from its 202M snapshot with the rail-rest drill (`--rail-rest-drill-fraction 0.35`): a puck at rest on or just off a side rail with the mallet level with it, just short of contact, the state v4.4.7 creeps in for the rest of a game. In training; not evaluated. Sim only. |

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
One extra level, `new`, runs the best checkpoint, **v4.3.12**, at full
strength (Start AI in the web UI, or `bash ai/bin/play.sh --difficulty new
--tension 1.5`). If it plays better on the table, make it `latest` and `max`
in `ai/models.json`.

Against the `table` and `table_plus` people, 64 one-minute games per tier
and seed, three seeds each; possession numbers from 32 of those minutes per
seed in the calibrated sim. "Table" is v4.1.2's own sessions of 2026-11-17
and 11-18 (37 min), for the scale.

| | table, v4.1.2 | v4.1.2 | v4.3.8 | v4.3.10 | **v4.3.12** |
|---|---|---|---|---|---|
| rigid sim (`table-20261029`): goal share | | 97.5% | 97.7% | 97.9% | 98.6% |
| rigid sim: goals for / against per minute | | 6.07 / 0.16 | 5.32 / 0.13 | 4.76 / 0.10 | 5.03 / 0.07 |
| calibrated sim (`table-20261119`): goal share | | 94.6% | 95.9% | 96.9% | 96.9% |
| calibrated sim: goals for / against per minute | | 4.75 / 0.27 | 5.25 / 0.22 | 5.08 / 0.16 | 5.66 / 0.18 |
| possessions ending in a shot | 49.5% | 61.9% | 69.6% | 70.4% | 71.4% |
| puck leaves under 2 m/s, robot not playing it | 13.2% | 13.0% | 8.7% | 8.0% | 7.3% |
| puck leaves at 1-2 m/s | 19.6% | 22.4% | 13.9% | 11.6% | 12.6% |
| shots that are a single touch | 35.8% | 28.0% | 22.0% | 21.8% | 21.3% |
| puck speed before the last touch of a shot, 75th percentile | 1.37 m/s | 2.2 m/s | 1.1 | 1.2 | 1.2 |
| after a block of a >= 4 m/s shot: puck recovered | 25.8% | 27.2% | 29.4% | 35.3% | 37.5% |
| ... of rebounds at 1-1.5 / 1.5-2 m/s | 28% / 19% | 41% / 8% | | | 53% / 27% |
| after a block: rebound leaves slowly, untouched | 21.8% | 26.7% | 22.5% | 18.5% | 18.6% |
| robot against v4.1.2, calibrated sim, 64 games at both ends: goals, shots on target | | | 15-21, 926 v 982 | 19-23, 904 v 1029 | 19-18, 1022 v 1028 |
| lighter compliant sim (`table-20261118`), one seed: goal share | | 97.5% | 97.4% | 98.0% | 97.7% |

Gates for v4.3.12 beside v4.1.2 in the calibrated sim
(`eval_playability.py`, `eval_resting_puck.py`):

| | v4.3.12 | v4.1.2 |
|---|---|---|
| idle accel cap, p50 / p90 | 7.6 / 9.3 m/s² | 8.4 / 10.4 |
| time above 60 m/s², rms accel | 4.7%, 26.5 m/s² | 5.7%, 26.8 |
| reversals at 1.5 m/s or more, per minute | 49 | 46 |
| cap above 40 m/s² while the puck is on the far half | 28% of ticks | 23% |
| forward excursion on shots of 5 m/s or more | 131 mm | 150 mm |
| modeled motor load, peak | 0.82 | 0.84 |
| resting puck never struck in 10 s (1632 placements) | 0.4% | 1.0% |
| corner pucks touched, worst cell of 8 | 41 of 48 (right corner, left-bank request) | 47 of 48 |
| rest spells of 4 s or more in 64 games | 0 (longest 1.0 s) | 0 |

What the numbers do NOT show, and what is worse:

- Sim only. The calibrated sim was fitted to sessions played before the
  camera was recalibrated on 2026-11-18 (the puck was seen 13-18 mm off), so
  the table as it is now may be kinder than that sim, and v4.1.2 itself may
  play better than it did.
- It scores less per minute in the rigid sim (5.0 against 6.1 goals): fewer
  one-touch shots, more prepared ones. In the calibrated sim that trade
  pays (5.7 against 4.75).
- It does not take control of a puck that is moving away (0-1 of 128 in the
  recovery drill, like every model here): it strikes it from behind, on
  target 77-93 times of 128 against 66-70 for v4.1.2. What changed is how
  often it goes for a rebound and how hard its block rebounds (median 1.61
  against 1.88 m/s).
- A puck resting in the robot's right back corner under a left-bank request
  is left alone in 7 of 48 trials (v4.1.2: 1). If one sits there, nudge it.
- The gain was found by evaluating snapshots, and the small-batch runs lost
  it again 40M steps later (v4.3.8 at 80M: shots back to 61%). The
  large-batch run kept it at every snapshot (20M to 90M: 67-72% shots,
  29-37% of blocks recovered); the registry points at its 60M one
  (`"file"`), which has three seeds, the head-to-head and the gates.

## Adding a version

1. Train into `runs/<version>` (or any directory) with the next free
   version from `python3 -m airhockey.models next study|model`.
2. Add the entry to `ai/models.json` (`run`, `file`, `date`, `accel`,
   `parent`, one-line `summary`) and a row to the table above.
3. Move `latest` only when the new model has replaced the old one at full
   strength, and say on what evidence.
