# Table-faithful fine-tune: v4 -> v5 (2026-11-10)

Simulation only. Nothing here touches the table, and the robot still runs one
network through the existing decoder; the people, the teacher and the costs
below exist only in training. Arm A below was run on 2026-11-10 (150 min,
145M -> 352M steps): `runs/_neural-table-faithful-20261110-a`, "v5a". Its
evaluation is in `logs/table-faithful-20261110/eval-a/`; what it found and
the second arm that answers it are under "Round 2" at the end. Arm a2 has
not been trained: its only run is a smoke test.

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
| `--human-proxy-share F` | 0 | Share of each worker's full-game slots played against a person (`airhockey/human_proxy.py` on the free far body). Taken from the snapshot pool's share; fixed references (id % 4 == 0) and the teacher keep theirs. Goals count as in any game; the far side's shaping income (shot requests, captures, returns), which a game against a copy takes from the learner, is not charged for a person. Scores appear as `game_scores.human_proxy`. |
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

Measured, v4 playing its own mean action in this recipe's mix (400 envs in
the table profile, 40 s; cost per finished defense drill at 20 / 5 / 40):

| defense drill | n | motion cost, mean / p90 / max | drill reward, mean |
|---|---|---|---|
| scripted launch, saved with the mallet under 0.5 m/s at contact | 459 | 0.7 / 1.2 / 16 | +217 |
| scripted launch, saved with the mallet moving | 3119 | 1.3 / 2.5 / 21 | +226 |
| scripted launch, conceded | 173 | 1.9 / 3.6 / 10 | -1043 |
| struck by a person, saved, mallet under 0.5 m/s | 153 | 5.3 / 11.8 / 21 | +190 |
| struck by a person, saved, mallet moving | 1457 | 6.4 / 12.8 / 35 | +224 |
| struck by a person, conceded | 37 | 8.8 / 11.8 / 15 | -1060 |

The dearest save of 5,200 cost 35; no save came within a factor of 17 of a
goal in a game. A block standing still at contact is the cheapest outcome
in both drills. The person's drills cost more because the far-cap term
runs while they walk to the puck (v4 holds a cap above 40 through 15% of
that time): the term is by the puck's half, not its speed, so a cap raised
the moment the person swings is charged until the puck crosses, at most
the 8 of the table above.

What the habit costs: the mean action pays 0.7 + 1.6 + 1.1 = 3.3 per
env-second over all slots (4.9 in games against the pool, 7.1 against
people, 2.4 in drills); with the exploration noise of a rollout it is the
smoke run's figure: in the smoke run v4's own play paid 1.2 + 2.4 + 2.5 =
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
| steps / s | 18.5k (rollout 10-14 s, update 1.3-2.4 s); 17-20k in the 2.5 min rerun after the review fixes |
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

## Review fixes (2026-11-10, after the first commit)

- Games against a person charged the learner the far side's shaping income
  as if a copy were playing: the person's strikes scored against a shot
  request nobody gave them, their traps as robot captures. Measured with
  v4: 16 per env-second either way, a net +7.9 a second to the learner
  (0.8 of a goal a minute it did nothing for), more than the motion costs
  in the same slots. People games now pay goals and the learner's own
  terms only.
- `--workers 1 --sim-profile` raised a duplicate-keyword error (the contact
  part replaces `paddle_tangential_range`); fixed and in the trainer test.
- `--human-proxy-tiers` is checked when the command is parsed.
- Flags-off identity rerun on exported trees of the commit before and
  after: one update, CPU, 256 envs on 4 workers and 128 in one process; the
  weights, the optimiser moments and the 29 earlier metrics are identical.

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

## Round 2: arm a2, from v5a (2026-11-10)

Simulation only, and still one network: everything below is a reset state, a
reward term, exploration noise or a sim parameter. Nothing is added to the
runtime.

### What v5a's evaluation found

Better than v4 against the table proxies (goals for +1.2-2.0 a game, goal
share +5 pt, lifts down 2-3x, idle cap p50 6.5 vs 8.1 m/s²), and worse in six
ways that replicated on fresh seeds. Their causes, from the three diagnoses
(scripts and traces in the session scratch, summarised here):

| finding | cause |
|---|---|
| Open-table stall: 3 spells of 21-59 s in 128 games (v4: none over 1.2 s) | A blind spot v4 has too. A puck at rest in the forward strip (y 0.65-0.85) with the bank off the OPPOSITE rail requested gets one constant action: arrival through the puck, time at its 0.25 s maximum, cap at the 5% floor. Re-issued every tick the decoder holds the mallet still 0.2 m behind the puck. Not sensing (puck seen within 1.5 mm), not the guard (0 interventions), not the rules. Placed pucks, right-forward with a left-bank request: v5a stalls 91%, v4 34%; the mirror cell 65% / 68%. |
| Why training left it | In the hover the time and cap outputs are tanh-saturated (pre-tanh +2 and -1.7 to -1.9) with a learned std of 0.02-0.07. Under `--std-floor 0.1` no sampled cap passed 20 m/s² there, so the drill's -600 at 6 s and the game's 30/s after 4 s arrived with nothing to tell actions apart. `--recovery-exploration` was 0, which makes `--quiet-exploration` a no-op. `by_kind.0.aimed` sat at 0.80 for the whole run. |
| Time at load >= 0.95 (26.7 s vs 5.5 s) | The stalls themselves: 23.4 s of it is the mallet parked at x >= 0.85 in three stalled games. Elsewhere v5a runs cooler than v4. No acceleration cost touches it. |
| Accel > 60 duty +13-35% | More and harder strikes and counter-hits, which the reward pays 15-30 times what the motion costs charge; accel duty is linear above a step and cannot make a strike gentler. About 40% of the extra duty is the guard braking (next row). |
| Guard interventions 1.4-4.7x, unresolved 201-275 vs 55-70 | The far-cap cost charges the cap from the moment the puck crosses, which is when the mallet has to brake. v5a asks for 42 m/s² (v4: 58) with 3.4 m/s still on and 76 mm of room, and the motion guard stops it at 70-100. 97% of the front-limit interventions are within 0.5 s of a set strike. |
| Idle target change twice v4's | The walk back after strikes taken further forward (97% within 2 s of its own contact); later the two policies are the same. Calm accel charges the cap, not the walking. |
| Corner pucks touched 84-95 of 128 (v4 124-128) | Drift from 185M steps on in states nothing anchors: the corner drill starts the mallet 0.12-0.22 m from the puck, games and the fringe test start it anywhere. Worst under the left-bank request, where the mallet stands off 0.25 m and never approaches. |
| Lift advantage gone when lift does not depend on braking | The braking term rests on one session (9 of 17 events; without it 5/38 braking vs 3/22 not). v5a meets 6-8 m/s pucks accelerating into them, a regime with five table contacts. 39 of its 42 goals against v4 in the profile went under v4's mallet; people never lift. |

`--left-bank-request-weight 2` is the only left/right-asymmetric option of
arm A and the stall got worse on one side only. That is an attribution, not
an ablation.

### What the trainer gained (all opt-in, defaults unchanged)

| flag | default | what it does |
|---|---|---|
| `--dead-puck-weight W`, `--dead-puck-grace S` | 0, 1.0 | Cost per second a puck has lain within the mallet's reach on the own half, slower than 0.3 m/s, for longer than S. Every slot kind. The clock is cleared by a release above 1 m/s, the puck leaving reach or a goal, not by a nudge. |
| `--dead-puck-exploration TIME EFFORT`, `-start S`, `-hold R` | 0 0, 0.5, 0 | Pre-tanh std floors of the arrival-time and accel-cap dimensions where that clock is past S, in the rollout and in PPO's likelihood alike (the mask is stored with the rollout). `-hold` makes that noise an AR(1) with unit variance: a strike is about ten ticks of a short time AND a raised cap, and at the hover's equilibrium a raised cap alone asks for no acceleration. |
| `--calm-reachable-scale X` | none | Calm-accel scale while a slow puck is within reach on the own half (otherwise `--calm-own-half-scale`, 0.25). 0 = a puck waiting to be struck is not a quiet table. |
| `--cross-bank-drill-fraction F` | 0 | Share of stationary drills in the stall's own state: puck at rest at y 0.60-0.88 and 0.05-0.35 m off the centre line, the opposite rail's bank requested in 70%, and in 60% the mallet in the hover pose with the hover as its previous action. |
| `--corner-far-start-fraction F` | 0 | Share of corner drills whose mallet starts anywhere in the workspace. |
| `--guard-weight W`, `--guard-unresolved-weight W` | 0, 0 | Cost per tick the motion guard replaced the learner's command, and per tick it predicted the workspace clamp even under a full brake. |
| `--far-cap-max-speed V` | none | Charge `--far-cap-weight` only on ticks begun with the mallet slower than V m/s. |
| `--accel-effort-weight W`, `--accel-effort-threshold A` | 0, 40 | Cost per second of (a² - A²) / `--accel`², a the realised tick-to-tick acceleration. Unlike accel duty it has a slope in how hard a strike is. |
| `--idle-speed-weight W` | 0 | Cost per metre the mallet moves in full games, times the calm factor of the calm-accel cost. Not in drills, which start the mallet anywhere and exist to teach the walk to a post. |
| `--sim-option mallet_lift='{...}'` | | `MalletLift` gained a per-episode mechanism: `flat_probability` (rate independent of the mallet's acceleration, `flat_rate` 0.08-0.40), `closing_probability` (rate by the mallet's speed at the puck, full at `closing_full` 3 m/s), the rest braking as before. In the two new ones a mallet under 0.5 m/s never lifts (0/30 on the table). `far_body: false` keeps a robot-bodied far side from lifting. |

Logged per update whatever the weights: `motion` gained
`guard_changed_per_minute`, `guard_unresolved_per_minute`, `dead_puck_share`
and each new cost per env-second; `motion_by_slot` gives the rates for
stationary / receiving / defense / games (robot far side) / people
separately, because the all-slot average stayed flat through arm A while its
games against people got rougher; `dead_puck_exploration_share` is the share
of steps drawn with the wide noise. Watch `by_kind.0.aimed` and
`motion_by_slot.stationary.dead_puck_share`.

No bug was found in the sim profile, the sensing, the guard or an evaluator;
nothing there was changed. `MalletLift` and `info["motion"]` only gained
fields (the six earlier motion columns keep their places).

### Arm a2

```bash
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --arm a2 --run-name _neural-table-faithful-20261110-a2 --minutes 150 --print   # inspect
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --arm a2 --run-name _neural-table-faithful-20261110-a2 --minutes 150
```

Arm A's command with these differences (`ARMS["a2"]` in the launcher; a typed
flag still wins):

```
--resume runs/_neural-table-faithful-20261110-a/agent.pt     (was v4)      --seed 20261112
--left-bank-request-weight 2 removed (1)
--dead-puck-weight 40 --dead-puck-grace 1.0
--dead-puck-exploration 2.0 2.0 --dead-puck-exploration-start 1.0 --dead-puck-exploration-hold 0.9
--calm-reachable-scale 0.0
--cross-bank-drill-fraction 0.25
--corner-drill-fraction 0.2 (0.1) --edge-drill-fraction 0.15 (0.1) --corner-far-start-fraction 0.5
--guard-weight 10 --guard-unresolved-weight 50 --far-cap-max-speed 1.0
--accel-effort-weight 100 --idle-speed-weight 20
--shot-speed-scale 7.0 (8.0)
--sim-option mallet_lift={"b0": [0.0, 0.30], "a0": [10, 90], "flat_probability": 0.25,
                          "closing_probability": 0.25, "far_body": false}
```

Why each, and what was left alone:

- **Resume `agent.pt` (352.06M), not `agent_step_345505792.pt`.** Every
  verified number of v5a is of `agent.pt`; the 345M save is 6.5M steps (four
  minutes) older, has no evaluation of its own and is the same on the one
  thing measured per checkpoint (corner cells under a left-bank request
  36/39 of 48 against 34/34). `agent_step_185073664.pt` is the only save
  with every corner cell at 48/48, but its strength, its stall map and its
  playability are unmeasured, and it would give up 167M steps of the gains.
  `--resume` keeps the critic, the optimiser, the pool, the adapted rate
  (5.06e-6) and the teacher weight 0.2. The critic has not seen the new
  terms: explained variance was 0.83, 0.95, 0.98 over the smoke run's three updates.
- **The stall is four changes that only work together.** The cost alone
  repeats arm A (a penalty with no action that avoids it in reach of the
  noise); the noise alone has nothing to climb. The cost starts at 1 s
  because a healthy strike on a resting puck is under way by then; the
  noise starts with it so strikes that already work are drawn as before.
  The drill puts the state in 25% of stationary resets instead of the ~4%
  of uniform placement (0.3% of envs). Calm accel at 0 there removes the one
  per-tick term that paid the cap floor. Left-bank weight back to 1 removes
  the one-sided pressure; the people still bank off the robot's left rail.
  `--stationary-failure-penalty 600` and `--possession-delay-weight 30`
  stay.
- **The replay file was not rebuilt.** The cross-bank drill generates the
  hover states (pose, previous action, request) instead of replaying
  recorded ones; the 2026-10-01 production failures keep their half of the
  stationary slots.
- **No anchor to v4 on resting pucks** (`--skill-reference`): v4 has the
  same hover in 22% of the forward strip, so the anchor would hold the stall
  in place.
- **Guard and far cap.** The stop after a strike needs 120-200 mm and has
  385; nothing is lost by braking in time, so the price can be firm: an
  intervention of one or two ticks costs 10-20 against a set shot's 160 of
  shaping plus its chance of a 600-point goal. With the far cap free above
  1 m/s of mallet speed, asking for the brake no longer costs anything.
- **Effort.** 84 a second at 100 m/s², 20 at 60, 0 below 40: a 0.13 s strike
  at the full cap costs 11, an all-out 0.3 s save 25 against 600 (game) or
  1300 (drill). Accel duty 20, reversal 5 and far cap 40 stay as they were.
  `--shot-speed-scale 7` stops paying for shot speed above v4's 7.4 m/s
  median; goals still pay whatever the speed. `--fallback-shot-scale` stays
  1.0: halving the pay of a counter-hit was recommended, but it changes
  what every fast return in every drill is worth and was not tried.
- **Load weights untouched.** They already charge 23-29/s, are 90% holding
  position, and the time above 0.95 is the stall.
- **Lift.** A quarter of lifting episodes flat, a quarter by closing speed,
  half by braking with the baseline up to 0.30 and the knee anywhere from 10
  to 90 m/s²; a quarter of all episodes still never lift. The one answer
  that is safe in all of them is a mallet that is set and slow at contact.
  The far robot no longer lifts, so a shot is worth against a copy what it
  is worth against a person. Profile share stays 0.75.

### Are saves and strikes still paid?

Worst cases with every cost on (arm A's 19 for a save included):

| | costs | pays |
|---|---|---|
| all-out save: 0.3 s at 100 m/s², one reversal, cap raised early, guard takes the stop for two ticks | 6 + 5 + 8 + 25 + 20 = 64 | 600 in a game, 1300 in a drill: worth it above an 11% / 5% better chance |
| set strike at the full cap, braked in time | 2.5 + 11 = 14 | up to 160 of shaping and the goal |
| the same, stop left to the guard (2 ticks, one unresolved) | 14 + 70 = 84 | the same: still positive, and the 70 is avoidable |
| a puck left lying 3 s in a game | 80 | nothing |

Measured in the smoke run below (the exploring policy, per env-second over
all slots): guard 5.2-6.2, effort 5.0, dead puck 1.5-4.0 (rising as the long
stalled drills fill the rollout), idle 0.2-0.3; 12-15 together. Arm A's three
cost it 1.4 + 2.8 + 1.2 = 5.4. In the games against people the new terms come
to about 15 a second (guard 7-9, effort 5.5, idle 1.1, dead puck 0) against a
goal flow of 53 (30 goals in 5.6 game-minutes at 600). The guard's share is
the avoidable one: v4's rates would pay 3. If goals for in the people slots
(`game_scores.human_proxy`) fall below arm A's 5.0 a minute, halve
`--guard` / `--guard-unresolved` and `--accel-effort` before anything else.

### Gates for a2 (all in sim, before any table time)

- Resting-puck grid (the diagnosis's `stall/grid.py`: pucks at rest over the
  own half, three requests, struck within 3 s / stalled at 10 s) for a2, v5a
  and v4. No evaluator in the repo places a puck at rest yet.
- Zero own-half rest spells over 4 s in 128 proxy games per table tier.
- Corner cells by request (`corner_diag.py`): at least 46/48 in each, in the
  historical sim and in the profile.
- `eval_playability.py` as for arm A: accel duty, guard interventions and
  time at load >= 0.95 at or below v4's, goals for not below v5a's by more
  than its seed spread.
- Lift: goals against under a braking-independent lift (`flat30`) not above
  v4's.

### Smoke test of arm a2 (2026-11-10)

`--arm a2 --n-envs 512 --workers 4 --device cpu --steps 196608`, from v5a into
a throwaway `_smoke-a2` run (deleted afterwards). Three updates, 196,608
steps; CPU only, so the speed says nothing about the real run.

| | update 1 | 2 | 3 |
|---|---|---|---|
| explained variance | 0.83 | 0.95 | 0.98 |
| final KL (target 0.02) | 0.016 | 0.008 | 0.009 |
| first update | `value_warmup` false, teacher weight 0.2, actor lr 5.06e-6 | | |
| guard changed / unresolved per env-minute | 23.8 / 2.0 | 22.6 / 3.0 | 19.2 / 2.4 |
| far-cap share of far-half time (arm A's smoke: 0.27 ungated) | 0.117 | 0.113 | 0.109 |
| cost per env-second: guard / effort / idle / dead puck | 5.6 / 5.0 / 0.2 / 1.5 | 6.2 / 5.0 / 0.3 / 3.1 | 5.2 / 5.1 / 0.3 / 4.0 |
| arm A's three: accel duty / reversal / far cap | 1.4 / 2.9 / 1.2 | 1.4 / 2.8 / 1.2 | 1.4 / 2.7 / 1.2 |
| stationary slots: share of time with a puck lying past 1 s | 0.20 | 0.46 | 0.60 |
| the same in games vs robots / vs people | 0.005 / 0 | 0.009 / 0 | 0.003 / 0 |
| steps drawn with the wide time/cap noise | 3.5% | 7.7% | 9.9% |
| lifts per env-minute, own / far robot | 1.2 / 0 | 1.8 / 0 | 1.7 / 0 |
| games vs people, cumulative | 9-0 | 17-1 | 30-3 |

Two things to read from it. The stationary slots are now mostly the stall:
with a quarter of them in the hover, a fifth in a corner and the 6 s / 12 s
limits, v5a leaves a puck lying for 60% of those slots' time, which is the
state the cost and the noise are for (the share climbs over the three
updates only because the long drills take that long to fill the rollout).
And own lifts are four times arm A's smoke (0.3-0.5 a minute): the wider
model lifts at about the table's pooled rate (17 of 71 central contacts at
6-8 m/s) in the episodes that lift at all.

Flags off is still the trainer from before: v4's recipe with every v5 option
off, and arm A's own command, each run for one update (CPU, 128 envs, one
process, `--resume` v4) on the commit before and on this tree. Weights,
optimiser moments and every earlier metric are identical in both pairs; the
only difference is the added `motion` keys and `motion_by_slot`.

### Not done in round 2

- No ablation arm. Left-bank weight, far cap and the stall are attributed,
  not separated; arms B and C above were never run either.
- The resting-puck gate is a scratch script, not an evaluator in `ai/bin`,
  and the idle-motion gate of `airhockey/playability.py` still counts the
  2 s after the robot's own contact (where v5a's extra idle motion is).
- The lift mechanisms are readings of the same 17 events, not new
  measurements. The table data that would decide between them: 6-8 m/s
  contacts with the mallet not braking (23 so far) or accelerating into the
  puck (5), and the standard paddle's centre of mass against the 32.7 mm
  cable plane.
- The hover pose at x >= 0.85 sits where the load model extrapolates
  (steady-state load above 1.0 there); the table's real holding current at
  that pose has not been measured.
- The historical sim's own findings (more own-deflection goals against, the
  head-to-head loss to v4) have no dedicated term; the wider lift model and
  the effort cost are the only pressure on meeting pucks under acceleration.
