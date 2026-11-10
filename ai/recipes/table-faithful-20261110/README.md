# Table-faithful fine-tune: v4 -> v5 (2026-11-10)

Simulation only. Nothing here touches the table, and the robot still runs one
network through the existing decoder; the people, the teacher and the costs
below exist only in training. **No v5 has been trained yet**: this directory
holds the trainer options, the launcher and the command to run. The only run
so far is the 3-minute smoke test at the end.

## Why

`runs/teacher-calm-20261018-v4` on the table (12 sessions, 2026-10-19 to
11-08) against what the sim said:

| | table | sim (human proxy) |
|---|---|---|
| goal share | ~80% (58-15) | 95-99% |
| goals for / against per minute | ~3 / ~0.8 | ~5.5 / 0.06-0.25 |
| mallet-puck restitution | ~0.45 (0.5 slow, 0.33 above 8 m/s), tangential 0.80 | 0.60-0.95, frictionless |
| cap > 80 m/s² | 13% of policy ticks (production 0.1%) | not charged |
| velocity reversals within 100 ms | 32 / min (production 13) | not charged |
| puck dropouts, hidden opponent mallet | ~20 / min >= 50 ms | never trained with |

Goals were conceded mostly on 4-9 m/s shots that passed under the mallet
while it braked hard from a 4-5 m/s lunge (about 7-12 verified events, so
thin). Three times the mallet ended 220-400 mm from its commanded position
after a strike-and-return at 92-100 m/s².

The sim side of this is `airhockey/table_fidelity.py` (profile
`table-20261029`) and the table human-proxy tiers; the judging side is
`ai/bin/eval_playability.py`. This recipe is the training side.

## What the trainer gained (all opt-in)

With none of these flags a command is the trainer as it was: same random
streams, same numbers (checked, see "Smoke test").

| flag (`ai/bin/train_neural_player.py`) | default | what it does |
|---|---|---|
| `--sim-profile SPEC` | none | Training envs use a table profile: `table-20261029`, or `table-20261029:contact,lift,sensing` for some parts. Contact, lift, dropout and latency parameters are drawn per env per episode by the env. |
| `--sim-option KEY=JSON` | none | Replace one keyword of the profile, e.g. `mallet_lift='{"off_probability": 0.1}'`; `null` removes it. Repeatable. |
| `--sim-profile-share F` | 1.0 | Share of `--workers` whose envs use the profile (evenly spread); the rest stay historical. Needs at least two workers when below 1. |
| `--human-proxy-share F` | 0 | Share of each worker's full-game slots played against a person (`airhockey/human_proxy.py` on the free far body). Taken from the snapshot pool's share; fixed references (id % 4 == 0) and the teacher keep theirs. Scores appear as `game_scores.human_proxy`. |
| `--human-proxy-tiers LIST` | `table` | Comma-separated tiers; one is drawn per game together with a new person (speed, reaction, aim, shot plan). Table tiers also serve after goals, as on the table. |
| `--defense-strike-fraction F` | 0 | Share of defense-drill slots where a person strikes a puck lying on the far half, instead of the scripted launch. The robot sees the mallet swing through the camera. Drill limit 6 s. |
| `--defense-strike-speed LO HI` | 4 9 | Planned release speeds of those strikes, m/s (clipped to the tier's own range). |
| `--defense-strike-moving-fraction F` | 0 | Share of those drills whose puck is still rolling in from the centre line at 1-3 m/s (people strike moving pucks 26-30% of the time). |
| `--left-bank-request-weight W` | 1 | Weight of left-bank requests under `--shot-request random` (right and straight stay 1). Both seats: a robot-bodied far side then banks more off the x = 0 rail too, the rail people use. |
| `--accel-duty-weight W`, `--accel-duty-threshold A` | 0, 60 | Cost per second while the realised tick-to-tick \|dv/dt\| is above A, scaled by \|dv/dt\| / `--accel`. |
| `--reversal-weight W`, `--reversal-speed V` | 0, 1.5 | Cost per velocity reversal within 100 ms with at least V m/s on both sides, events 150 ms apart (the playability gate's count). |
| `--far-cap-weight W`, `--far-cap-threshold A` | 0, 40 | Cost per second of commanded cap above A while the puck is on the far half, scaled by (cap - A) / (`--accel` - A). |
| `--stagger-games` | off | Each game slot's first game starts at a random point of the `--game-episode-seconds` clock, so the 900 s games no longer all end in one rollout at one thermal phase. |

The three motion rates are counted on every step whatever the weights and
logged per update in `status.json` / `metrics.jsonl` under `motion`:
`accel_duty_share`, `reversals_per_minute`, `far_half_share`, `far_cap_share`
(of far-half time) and the reward each cost took per env-second
(`*_cost_per_s`). With a profile, `lift_events_per_minute` (own / opponent)
is logged too. The env keeps cumulative `motion_totals` like `calm_seconds`.

`neural_setup.checkpoint_environment` does not carry the profile: evaluators
take `--sim-profile` explicitly, so a v5 checkpoint evaluated without it is
scored in the historical sim.

## Continuation: `--resume`, not `--initialize-from`

Every calm-accel run of October (g, h, i, j) started with `--initialize-from`:
fresh critic, 10 warm-up updates, and the teacher pulled back in at weight 1.
`--resume <checkpoint> --run-name <new name>` already continues a run under a
new name and keeps the critic, the optimiser moments, the adapted actor rate
(v4: 5.06e-6), the opponent pool, the step count (v4: 144,965,632) and the
teacher anneal's origin, so v4's teacher weight is its final 0.2 from the
first update. `--steps` counts from the resume point. The launcher uses it
and passes no `--value-warmup-updates`. The critic has not seen the new
costs or opponents; in the smoke run its explained variance was 0.71 on the
first update and 0.93 by the fourth.

## Pitfalls carried over from `expert-distill-20261018`

The launcher keeps each of these on; do not drop them.

- **Possession delay** `--possession-delay-weight 30`. At 6/s the first
  fine-tune learned to hold the puck (494 s of stall in 64 proxy games).
- **Teacher mask** `--teacher-mask defense`. The teacher's set-up/strike
  decision averages into weak pushes when imitated.
- **Free guard accel** `--teacher-free-guard-accel`. The teacher's
  distance-scheduled guard cap outweighed every calm cost tried.
- **Calm accel** `--calm-accel-weight 200 --calm-accel-power 1`. v2 idled at
  57 m/s² and shook the mallet off the table.

## Is a real save still worth it?

Recommended weights: accel duty 20/s, reversal 5/event, far cap 40/s. An
all-out save, costed at its worst:

| term | what the save does | cost |
|---|---|---|
| accel duty | 0.15 s at 100 m/s² out, 0.15 s braking: 0.30 s x 100/100 x 20 | 6 |
| reversal | out and back inside 100 ms at speed: one event | 5 |
| far cap | cap 100 for 0.2 s before the puck crosses: 0.2 x (100-40)/(100-40) x 40 | 8 |
| | | **19** |

Conceding costs 600 in a game (`--game-goal-weight`) and 1000 in a drill
(`--skill-concede-weight`), where a block also earns 300. With gamma 0.999
nothing is discounted over a save. So the full-cap lunge pays whenever it
lowers the chance of conceding by more than 19/600 = 3.2% in a game, or
19/1300 = 1.5% in a drill. A save that needs the cap keeps it; a lunge at a
shot going wide, or one a slower move would have covered, does not.

What the habit costs: in the smoke run v4's own play paid 1.2 + 2.4 + 2.5 =
6.1 per env-second (accel duty share 7.7%, 29 reversals a minute, cap > 40
on 27% of far-half time), about 370 a minute, 0.6 of a goal. That is a firm
pull but below the goal flow (4.5 goals a minute against the people). If
goals against in the proxy slots rise during a run, halve the three weights
before changing anything else.

## Recommended v5 command

From the repo root. The launcher prints the full trainer command with
`--print`; without it, it runs it and writes `logs/<run-name>.log`.

```bash
# What it will run (inspect first)
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --run-name _neural-table-faithful-20261110-a --minutes 120 --print

# Arm A: everything on (the defaults of the launcher)
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --run-name _neural-table-faithful-20261110-a --minutes 120
```

Arm A is v4's own command continued, plus:

```
--resume runs/teacher-calm-20261018-v4/agent.pt
--sim-profile table-20261029 --sim-profile-share 0.75
--human-proxy-share 0.3 --human-proxy-tiers table,table_plus,club,elite
--defense-strike-fraction 0.4 --defense-strike-speed 4 9 --defense-strike-moving-fraction 0.3
--left-bank-request-weight 2.0
--accel-duty-weight 20.0 --reversal-weight 5.0 --far-cap-weight 40.0
--stagger-games
```

Per worker (128 envs, 38 game slots) that is 10 reference, 11 people, about
11 teacher and 6 pool games, 15 of the 38 defense-drill slots struck by a
person, and 12 of the 16 workers in the table profile.

Two controls worth running beside it if the GPU allows, to tell the parts
apart:

```bash
# Arm B: the table sim and the people, no motion costs
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --run-name _neural-table-faithful-20261110-b --minutes 120 \
    --accel-duty 0 --reversal 0 --far-cap 0

# Arm C: the motion costs alone, in the historical sim
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --run-name _neural-table-faithful-20261110-c --minutes 120 \
    --sim-profile none --human-proxy-share 0 --defense-strike 0 --left-bank 1
```

Judge checkpoints in the table sim, with v4 and production beside them
(numbers for v4 itself come from the same commands):

```bash
python ai/bin/eval_playability.py runs/<run>/agent.pt --reference runs/teacher-calm-20261018-v4/agent.pt \
    --tier table --sim-profile table-20261029
python ai/bin/eval_playability.py runs/<run>/agent.pt --reference runs/teacher-calm-20261018-v4/agent.pt \
    --tier table_plus --sim-profile table-20261029
python ai/bin/eval_human_proxy.py runs/<run>/agent.pt --tier tables --sim-profile table-20261029
```

## Open choices, stated

- **Profile share 0.75.** A quarter of the workers stay historical so the
  policy is not fitted to one contact and lift model; the lift evidence is
  thin. 1.0 is the other defensible setting.
- **Teacher spec `expert`.** The teacher still plans with the historical
  contact. It only labels defense states and plays 30% of games; a
  `expert:paddle_restitution=...` teacher (`--teacher`) was not tried.
- **Weights 20 / 5 / 40.** Set from the arithmetic above, not from a run.
- **Four tiers.** `club` and `elite` bank about half their shots off either
  rail; the table tiers bank mostly off the robot's left. Dropping the two
  older tiers halves the people's per-step cost (see below).

## Smoke test (2026-11-10)

Full-size, 2048 envs on 16 workers, from v4 into a throwaway `_smoke-` run
(deleted afterwards), on a machine other jobs were loading at the time
(load average 25-30 of 32 cores), so read the ratio, not the level.

| 3 min, arm A | |
|---|---|
| updates | 13 (3.4M steps) |
| steps / s | 18.5k (rollout 10-14 s, update 1.3-2.4 s) |
| same launcher, every option off, 1.5 min | 21.3k (rollout ~10 s) |
| new metrics | all present and finite in all 13 updates |
| first update | `value_warmup` false, teacher weight 0.2, actor lr 5.06e-6, 256 actor steps |
| explained variance | 0.71, 0.88, 0.91, 0.92 ... 0.93 |
| lift events per env-minute | 0.27-0.46 own, 0.05-0.14 opponent |
| games vs people (exploring policy) | 442-47, 4.5 for / 0.48 against per minute |

Single-env cost of the parts, 128 envs, ms per step: historical 15.9,
profile 18.2, people in 30% of games with 1 / 2 / 4 tiers 18.7 / 20.3 / 21.8,
everything 25.4.

Flags off is the trainer from before: the same fixed-seed command (v4's
recipe, `--resume` v4, CPU, 2 updates) was run on the code before and after
the change, once with 256 envs on 4 workers and once with 128 envs in one
process. All 29 pre-change metrics of both updates and the SHA-256 of the
final weights are identical; the only difference is the added `motion` key.

## Not done

- No real training run and no held-out evaluation.
- The per-env profile share is per worker; a single-process trainer
  (`--workers 1`) takes the profile for all envs or none.
- A defense drill released by a person runs on the person's own time and
  aim, and the 4-9 m/s band is the PLANNED release. Checked once with a
  robot parked at its post (128 envs, 60 s, four tiers): the puck crossed
  the centre line at p10 3.0 / p50 6.3 / p90 8.4 m/s (the people's attacks:
  p50 6.4 / p90 9.0), drills lasted p50 0.9 s / p90 3.2 s and 5% ran out
  the 6 s limit. With 30% moving pucks the line speed is p10 1.5 / p50 5.6
  / p90 8.2, because soft returns of the rolling puck cross too.
- Not modelled, as in `table_fidelity.py`: mallet follow lag as a compliant
  body and the 220-400 mm strike-and-return position losses. The accel-duty
  and reversal costs are the only pressure against the moves that caused
  them.
