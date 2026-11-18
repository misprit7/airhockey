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
editing its `model` and `accel` there.

## Adding a version

1. Train into `runs/<version>` (or any directory) with the next free
   version from `python3 -m airhockey.models next study|model`.
2. Add the entry to `ai/models.json` (`run`, `file`, `date`, `accel`,
   `parent`, one-line `summary`) and a row to the table above.
3. Move `latest` only when the new model has replaced the old one at full
   strength, and say on what evidence.
