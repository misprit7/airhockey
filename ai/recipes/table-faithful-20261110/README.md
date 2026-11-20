# Table-faithful fine-tune: v4 -> v5 (2026-11-10)

Simulation only. Nothing here touches the table, and the robot still runs one
network through the existing decoder; the people, the teacher and the costs
below exist only in training. Arm A below was run on 2026-11-10 (150 min,
145M -> 352M steps): `runs/_neural-table-faithful-20261110-a`, "v5a". Its
evaluation is in `logs/table-faithful-20261110/eval-a/`; what it found and
the second arm that answers it are under "Round 2". Arm a2 was started the
same day and was still training when "Round 3" at the end was written: arm
a3, a2 plus two imitation anchors on resting pucks. Arm a3 has since been run
and evaluated (`logs/table-faithful-20261110/eval-a3/`); "Round 4" is arm a4,
from a3, for the three things that still keep it off the table. Not run.

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
| `--dead-puck-weight W`, `--dead-puck-grace S` | 0, 1.0 | Cost per second since a puck came to lie (under 0.3 m/s) within the mallet's reach on the own half, past the first S. Every slot kind. Clock and cost stop at a release above 1 m/s, the puck leaving reach or a goal; a tap that leaves it drifting under 1 m/s neither clears nor pauses them (it paused the cost as first written, which made the tap the cheapest way out). |
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

- `ai/bin/eval_resting_puck.py` (below), in the historical sim and in the
  profile, with `--games 128`: no stall-map cell above 5%, every corner
  cell at least 46/48, zero own-half rest spells over 4 s.
- `eval_playability.py` as for arm A: accel duty, guard interventions and
  time at load >= 0.95 at or below v4's, goals for not below v5a's by more
  than its seed spread.
- Lift: goals against under a braking-independent lift (`flat30`) not above
  v4's.

### The resting-puck evaluator (2026-11-10)

`ai/bin/eval_resting_puck.py` (library `ai/airhockey/resting_puck.py`, tests
`ai/tests/test_resting_puck.py`) is the diagnosis's scratch scripts as one
evaluator. Simulation only; about 8 minutes per candidate on two CPU threads.

```bash
python ai/bin/eval_resting_puck.py runs/<run>/agent.pt --reference runs/teacher-calm-20261018-v4/agent.pt
python ai/bin/eval_resting_puck.py runs/<run>/agent.pt --reference runs/teacher-calm-20261018-v4/agent.pt \
    --sim-profile table-20261029 --games 128 --record ai/recordings/<name>
```

- `stall`: a puck at rest on x 0.10-0.90, y 0.10-0.85 in steps of 0.05, for
  the mallet at home and already beside the puck, under the left-bank,
  right-bank and straight requests (1632 trials, 10 s, a `table` person at
  the far end, load 0.45). Stalled = never above 1.5 after a touch and never
  over the centre line. A cell is a zone (forward y 0.65-0.85, mid, back,
  each left / centre / right; the rail columns; the corners) x a request;
  cells under `--min-cell-n` 20 trials are printed and not gated. The
  benchmark env's referee (a dead puck out of everyone's reach is relaunched
  from the centre line after 8 s) is off in these trials: with it on, every
  puck placed at y 0.90, 3 mm past the mallet's reach, read as "crossed" at
  8.02 s without the robot doing anything, and a puck nudged out of reach
  would have too. Such a puck is now stalled, `out_of_reach_share` counts
  them, and cells with a puck placed out of reach are not gated.
- `corner`: `corner_diag.py`, 8 cells x 48 mirror-paired corner pucks.
- `games`: `--games` 32 x 60 s against `table` and `table_plus`; a rest spell
  is the puck on the own half, slower than 0.05 and where the mallet's edge
  can touch it, so a puck wedged in a back corner counts. The games keep
  the referee; own-half pucks at rest out of reach for 1 s or more are
  reported per tier (`out_of_reach`; none for v5a or v4), not gated.

It reproduces the scratch numbers: with the diagnosis's grid and seed
(`--parts stall --seed 7 --ys 0.5,0.55,...,0.9`) and the referee still on,
every trial was identical (right-forward + left bank 64/70 v5a, 24/70 v4 in
the profile). With the referee off the y 0.90 row is 102/102 stalled instead
of 102 "crossed", and the relaunches' random draws no longer shift the other
trials after 8 s: the same cell reads 64/70 and 26/70. The corner cells are
identical at the default seed; the rest spells are the ones in the
diagnosis's 32-game traces. The table below (default grid, to y 0.85) had no
relaunch in it and is unchanged by the referee; a second seed (4242, profile)
gives the same picture: right-forward + left bank 64/70 v5a and 28/70 v4,
corner cells v5a 28-48, v4 47-48. At the default seed 20261207:

| | v5a historical | v4 historical | v5a profile | v4 profile |
|---|---|---|---|---|
| stalled, all 1632 trials | 7.8% | 5.0% | 9.2% | 7.5% |
| forward strip (450) | 22.9% | 11.8% | 25.8% | 19.3% |
| mid / back (540 / 426) | 0.4% / 0% | 0.2% / 0% | 0.9% / 0.7% | 0.7% / 0.2% |
| rail columns (168) | 13.7% | 16.1% | 14.3% | 17.9% |
| forward-right + left bank (70) | 61 = 87% [77-93] | 19 = 27% [18-39] | 64 = 91% [83-96] | 28 = 40% [29-52] |
| forward-left + right bank (70) | 19 = 27% | 18 = 26% | 22 = 31% | 21 = 30% |
| forward-right + straight (70) | 19 = 27% | 5 = 7% | 22 = 31% | 15 = 21% |
| forward-left + straight (70) | 2 = 3% | 10 = 14% | 4 = 6% | 16 = 23% |
| rail + straight (56) | 13 = 23% | 12 = 21% | 14 = 25% | 14 = 25% |
| gated cells over 5% (of 21) | 5 | 7 | 6 | 8 |
| corner cells, left none / L / R / S | 48 34 41 36 | 48 48 47 46 | 48 35 43 33 | 48 48 48 46 |
| corner cells, right none / L / R / S | 47 34 41 25 | 48 48 48 48 | 44 34 46 33 | 48 48 48 48 |
| rest spells >= 4 s, table / table_plus (32 games each) | 1 / 0 | 0 / 0 | 2 / 2 | 0 / 1 |
| longest rest spell | 34.5 s, corner | 0.4 s | 55.0 s, corner | 42.8 s, corner |
| gates stall / corner / games | FAIL / FAIL / FAIL | FAIL / PASS / PASS | FAIL / FAIL / FAIL | FAIL / PASS / FAIL |

v4 does not pass the stall map either: the hover is a blind spot of both
networks, worse in v5a on the right side under the left-bank request and
under the straight request, and both leave a puck that touches a side rail
in the forward strip. A stalled trial ends with the mallet 0.2-0.35 m behind
the puck, untouched in 97-100% of them, at an accel cap of 5.6 m/s² (v5a;
the floor is 5) or 8.5 (v4). Of the six rest spells over 4 s in these games,
four are a puck wedged in a back corner at (0.04, 0.04) or (0.96, 0.04),
v4's 42.8 s among them, although v4 touches 46-48 of 48 placed corner
pucks; the other two are v5a's in the profile, 53.6 s in the forward-right
strip under the left-bank request and 20.7 s at (0.82, 0.53).

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
- The idle-motion gate of `airhockey/playability.py` still counts the 2 s
  after the robot's own contact (where v5a's extra idle motion is).
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

## Round 3: arm a3, imitation anchors on resting pucks (2026-11-10)

Simulation only, and still one network. Round 2's answer to the resting puck
was a cost, wide noise and a drill; 40M steps into arm a2 the cross-bank stall
was down (91% -> 44%, v4 40%) but other forward cells were still at 17-33% and
the right corner had collapsed (right / straight 0 of 48 touched, right /
left bank 9; v4 46-48). Reward plus exploration was not finding the strike.
Round 3 adds two supervised terms that say what to do there. Both label
sources are training-only: the teacher (`expert_player.py`) and v4.

### What the teacher does in the hover (measured before anything was built)

`eval_resting_puck`'s stall map in the profile, arm a2's checkpoint at 432.3M
steps (a copy, taken while it trained), the teacher labelling every tick of
every trial the way the trainer does. 66 of 1632 trials stall, 51 of them in
the forward strip.

- **It attacks, with one steady label.** In the stalled trials the teacher is
  in attack mode on 100% of ticks. Its label hardly moves while the mallet
  hovers (per-dimension std over time 0.01-0.03) and it is a strike: the cap
  slot at 0.89 (90 m/s²) where the student holds -0.66 to -0.88 (6-8 m/s²),
  the arrival time at -0.4 to 0 (0.10-0.15 s) where the student holds +0.96
  (the 0.25 s maximum). It does not wind back. It first re-poses the mallet
  onto the shot line (about 0.1 m, at that cap), and once there its label is
  the strike itself: through the puck, time at its minimum, cap 1.0.
- **Tick by tick it is the student's own action.** Same 16-frame,
  request-conditioned observation (`teacher_shares_obs`), same 6-D arrival
  encoding through the same decoder, and the teacher is a function of the
  observation alone, so a label on a state the student reached is well
  defined. The squared error to the student in the hover is 0.69 a
  dimension, the largest of any state: target y 1.5, time 0.78, cap 1.4.
- **Following the labels only where the anchor applies strikes the puck.**
  The same trials with the teacher's action executed on the ticks where the
  puck has lain past 1 s, and the student's everywhere else: 63 of the 66
  stalled trials are struck (forward strip 51 -> 3, rail columns 14 -> 0, mid
  1 -> 1), after a median 40 ticks (0.8 s) of the teacher, at a median
  3.7 m/s (p10 2.3, p90 6.1). The teacher has 2.7% of all ticks. Handing
  over at 3 s instead leaves 3; v4 as the student leaves 5.
- **The hover is a fixed point, not ignorance.** From the tick after the
  mallet starts to move under the teacher's command, the student's own output
  is a strike too (cap +0.89). It only needs the state changed.
- **The teacher's own stall is in the labels.** The 3-5 trials left are all
  at x 0.15 under the right-bank request: the teacher, in attack mode, asks
  for a set-up pose past the workspace's left edge and waits there
  (its forward-left / right-bank cell is 5 of 70 in
  `expert-resting.json`, not 0; forward-right / left-bank is 0 of 70). The
  anchor cannot teach better than that there; the dead-puck cost and noise
  stay on.
- **Corner pucks are v4's, not the teacher's.** `corner_recovery` under arm
  a2's sim options (the profile with its three-mechanism lift): v4 touches 48
  of 48 in seven cells and 43 in left / straight at the default seed, 46-48
  in all eight at seed 4242; the teacher 34-45.

DAgger was already there: the trainer calls the teacher on every observation
of the learner's own rollout and stores label and mode
(`teacher_actions`, `teacher_modes`). The new mask selects among those.

### Where the loss has a gradient

The existing teacher regression is the squared error of tanh(actor mean), so
its gradient carries a factor 1 - tanh². In the hover that is 0.08 on the
arrival time (pre-tanh +2) and 0.2-0.6 on the cap (-0.7 to -1.4): not zero,
because the error is as large as it can be, but small where it matters. Over
the stalled trials the mean gradient magnitude on the pre-tanh mean is
0.20 (time) and 0.31 (cap) squashed against 0.42 and 0.58 taken before the
squash; on a typical saturated trial 0.04 against 0.8 on the time. So the
anchors can be taken in either space (`--anchor-space`), and arm a3 uses
`pre-squash`: the mean regressed toward atanh(label), labels clamped at
+-0.964 (atanh 2.0, where the hover itself sits), and a label at the clamp
read as "at least that far" so a clipped teacher target never pulls a
saturated output back.

### What the trainer gained (all opt-in, defaults unchanged)

| flag | default | what it does |
|---|---|---|
| `--teacher-resting-weight W`, `--teacher-resting-start S` | 0, 1.0 | Actor loss on the teacher's action where the dead-puck clock is past S, the puck is on the open table and the teacher is attacking. On top of `--teacher-weight` whatever `--teacher-mask`, and constant: the defense weight's anneal does not touch it. |
| `--teacher-resting-dims all\|time-cap\|cap` | all | Outputs that anchor copies from the teacher; the others keep no error term. Arm a3: `cap` (review below). |
| `--skill-reference-rail-weight W` | 0 | Actor loss on `--skill-reference`'s action in every step of a corner / edge drill slot and wherever a puck lies in the rail strip. Independent of `--skill-reference-weight`, which stays 0 (v4 has the hover; anchoring every resting puck to it would hold the stall). |
| `--anchor-space squashed\|pre-squash` | squashed | Where both errors are taken. |
| `--anchor-kl-exempt` | off | The KL that stops, backtracks and adapts the actor step is taken over the samples no anchor covers. Not in arm a3; see the tiny runs. |

The environment reports `info["anchor_zone"]` every step: 0, 1 (open table:
the dead-puck clock is running) or 2 (rail: the same with the puck's centre
within 15 mm of the mallet-centre box's side or back edge, i.e. where the
corner and edge drills put it and the mallet cannot get behind it; and every
step of a corner / edge drill slot). The stall map's rail columns (x 0.10,
0.90) are open table: the teacher stalls on 0 of 168 there, v4 on 18%. It is
bookkeeping from the true puck position that selects training examples, as
`controlled` already is for the skill reference; it enters no observation,
reward or random draw. The zone belongs to the state the action was chosen
in (it is reported with the step before, like the exploration mask), and a
slot's first step after a reset has none.

Each anchor's term is its weight times the mean error over its samples in the
minibatch, with at least 16 in the denominator. Like the teacher regression's
masked mean it does not shrink with how rare the state is, so a hover sample
weighs about a hundred times a PPO sample; the floor keeps two or three from
carrying the whole weight. Logged per update under `anchors`: share of steps,
samples, the loss over the minibatches, and on a fixed set of the anchored
samples the error before and after the update in both spaces, per dimension,
with the mean label and the mean action.

### Arm a3

```bash
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --arm a3 --run-name _neural-table-faithful-20261110-a3 --minutes 150 --print   # inspect
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --arm a3 --run-name _neural-table-faithful-20261110-a3 --minutes 150
# from another starting point: add --resume runs/_neural-table-faithful-20261110-a/agent.pt
#                              or  --resume runs/teacher-calm-20261018-v4/agent.pt
```

Arm a2's command (`ARMS["a3"]`; a typed flag still wins) with:

```
--resume runs/_neural-table-faithful-20261110-a2/agent.pt   (a launcher flag)     --seed 20261113
--teacher-resting-weight 1.0 --teacher-resting-start 1.0 --teacher-resting-dims cap   (the review below; the
                                                         smoke and tiny runs of this section were all six outputs)
--skill-reference runs/teacher-calm-20261018-v4/agent.pt --skill-reference-rail-weight 1.0
--anchor-space pre-squash
```

- **Everything of a2 stays**: the dead-puck cost and noise, the cross-bank
  drill, the corner drills from anywhere. The anchor applies from the same
  second the cost and the noise start, and the wide noise there is what makes
  moving the time and cap means cheap in KL (std 2.0 on those two
  dimensions); the target position and velocity move under the 0.1 floor.
- **The starting point is a flag.** From a2 both anchors have work to do.
  From v4 the rail anchor is zero at the first update (the student is the
  reference) and only holds the corners while the rest changes; v4's critic
  has seen neither round's terms (arm a2 from v5a: explained variance 0.83,
  0.95, 0.98 over three updates).
- **Where the teacher is weak the reward is alone** (task: its mid-left /
  right-bank cell stalls 10 of 84, mid-right / left-bank 8 of 84). Nothing
  was added for it. Arm a2's checkpoint stalls on 1 of 540 mid trials, strikes
  a mid puck at a median 0.68 s (p99 1.4 s), so the anchor's states (lain past
  1 s) are 1.2% of its mid ticks, and with the teacher acting on exactly
  those the mid count stayed 1. The cross-bank drill reaches down to y 0.60;
  below that the cover is the uniform stationary reset and the dead-puck cost.

### Smoke test of arm a3 (2026-11-10)

`--arm a3 --n-envs 512 --workers 4 --device cpu --steps 196608` from the copy
of a2 into a throwaway `_smoke-a3` run (deleted). Three updates:

| | update 1 | 2 | 3 |
|---|---|---|---|
| teacher anchor: share of steps (samples of 65,536) | 0.40% (261) | 0.32% (212) | 0.44% (288) |
| its error before -> after the update, pre-squash | 0.868 -> 0.838 | 1.158 -> 1.107 | 1.631 -> 1.510 |
| the same, squashed | 0.362 -> 0.350 | 0.438 -> 0.427 | 0.554 -> 0.530 |
| its loss over the minibatches (weight 1) | 0.37 | 0.46 | 0.84 |
| rail anchor: share of steps | 10.3% | 11.9% | 11.9% |
| its error before -> after, pre-squash | 0.091 -> 0.083 | 0.103 -> 0.093 | 0.106 -> 0.096 |
| the same, squashed | 0.044 -> 0.042 | 0.052 -> 0.049 | 0.052 -> 0.050 |
| final KL (target 0.02), actor steps of 64 | 0.021, 16 | 0.026, 16 | 0.063, 16 |
| actor lr | 7.6e-6 | 5.1e-6 | 3.4e-6 |
| explained variance | 0.91 | 0.95 | 0.96 |

Everything finite. The teacher anchor's states are rare: the stationary
slots' lying time is mostly corner and edge drills (the rail anchor's), and
the first second of every rest is left to the student. Its error is ten
times the rail anchor's. The rise of the "before" error over these three
updates is the rollout filling with the long drills, as in arm a2's smoke
(stationary dead-puck share 0.27, 0.51, 0.57), not the policy getting worse:
see the fixed states below.

### Does the hover's loss fall closed loop? (tiny runs)

The smoke run's three updates cannot answer that, so: arm a3 at 256
environments on two CPU workers, 16 updates (524,288 steps, about two updates
of the real run's data, under 400 actor steps), from the same copy of a2,
two seeds, each with and without `--anchor-kl-exempt`; the two without it
then continued for 16 more. Throwaway `_smoke-a3*` runs, deleted. Each end
checkpoint was then played closed loop, deterministic, in the profile: the
stall map's forward strip (450 trials, x 0.15-0.85, y 0.65-0.85) and the
corner cells. "Label error" is the pre-squash error to the teacher on the
ticks of that stall map where the puck had lain past 1 s (the anchor's own
states, so the population shrinks as the stall does).

| | updates | forward strip stalled / 450 | right + left bank, left + right bank / 70 | label error | corner cells, worst of 8 |
|---|---|---|---|---|---|
| a2 at 432.3M, the start | 0 | 55 | 24, 15 | 1.62 | 9 (right / left bank; right / straight 10) |
| seed A | 16 | 57 | 16, 13 | 1.00 | 48 |
| seed A | 32 | 7 | 1, 4 | 0.75 | 47 |
| seed B | 16 | 73 | 16, 16 | 1.09 | 48 |
| seed B | 32 | 46 | 8, 8 | 0.45 | 48 |
| seed A, KL exempt | 16 | 16 | 2, 6 | 0.78 | 48 |
| seed B, KL exempt | 16 | 33 | 11, 8 | 0.89 | 47 |

In training, the teacher anchor's error on each rollout's own samples (before
that update, sample-weighted) went from 1.06 over the first four updates to
0.53 over the last four of the first leg (seed A; squashed 0.40 -> 0.25) and
sat at 0.26-0.59 through seed B's second leg; the rail anchor's went 0.092 ->
0.036 -> 0.02-0.04. Every update lowered both on its own samples.

What this says, and what it does not:

- **The corners come back at once.** Every end checkpoint touches 47 or 48
  of 48 in all eight cells, from 9 and 10 in the two collapsed ones, after
  half a million steps. The rail anchor is 10-15% of all steps.
- **The hover's loss falls, in the rollouts and on the stall map's own
  states, in all six runs.** The stall itself follows later and noisily:
  nothing or worse at 16 updates without the exemption (57, 73), then 7 and
  46 at 32. Between the two there is a stretch where the labels are half
  learned; a stalled trial is one never struck above 1.5 m/s, which a weak
  push still is.
- **The exemption was faster twice out of two and is not the reason it
  works.** The KL with and without the anchored samples differs by under 10%
  in most updates (20% at most), the exempt runs took fewer actor steps, and
  each pair is the same run through its first four or five updates. Two
  seeds do not separate that from chance, the plain runs get there with more
  updates, and the real run has some three hundred. Arm a3 leaves it off; `--anchor-kl-exempt` on the
  launcher turns it on.
- **Scale.** These runs are 1/8 of the real batch with at most 32 actor
  steps an update at 3-8e-6; the adaptive rate fell under the anchors' KL in
  the first updates (7.6e-6 -> 3.4e-6) and came back to 5.1e-6. Nothing here
  says what 150 minutes do to strength, playability or the mid cells: gate
  a3 as a2 (`eval_resting_puck.py`, `eval_playability.py`, the lift arms).

Watch in the real run: `anchors.teacher_resting.pre_squash_initial` falling
toward the rail anchor's level and its `share` shrinking (fewer pucks left
lying), `anchors.reference_rail.pre_squash_initial` under 0.03,
`by_kind.0.aimed`, `motion_by_slot.stationary.dead_puck_share`, and
`game_scores.human_proxy` not below a2's. If the teacher anchor's error
stalls above about 0.5, raise `--teacher-resting` (2-4) before anything else;
if goals for against people fall, the anchors are the first thing to halve.

Flags off is still the trainer from before: arm a2's own command, one update
(CPU, 128 envs, one process, `--resume` the copy of a2) on the commit before
and on this tree gives identical weights, optimiser moments and metrics, with
no new keys. Arm a2's printed command is byte-identical.

### Round 3 review (2026-11-10): the teacher anchor copies the cap only

An independent check of the commit above, from v5a (`runs/_neural-table-faithful-20261110-a/agent.pt`,
not a2), found the anchors do what they are for and that the teacher anchor on
all six outputs costs game offence. Arm a3 now passes
`--teacher-resting-dims cap`; `all` is the first commit's behaviour.

**What held up.** Arm a2's command for one update on the commit before and on
this tree is identical (weights, optimiser moments, metrics, no new keys).
The rollout holds the learner's seat only, the zone and clock are those of
the state the action was chosen in, the teacher labels the observation the
actor saw, and nothing of the zone reaches an observation or a reward. At
v5a's own hover (4096 ticks of its stalled stall-map trials, profile): actor
mean pre-tanh +1.87 on the time and -1.43 on the cap; gradient per sample on
the time 0.043 squashed against 0.80 pre-squash, on the cap 0.14 against 0.98.
On a real minibatch (2048, 6-11 teacher and 190-270 rail samples) the gradient
norms on the actor are PPO 18-28, teacher anchor 9-18, rail anchor 3.5-5.4,
defense teacher 0.15-0.18: weight 1.0 is of PPO's order, not above it.

**Which outputs break the hover.** v5a on the forward strip (450 trials,
profile), with the teacher's label executed in place of the student's on the
listed outputs where the puck has lain past 1 s:

| teacher's outputs used | stalled / 450 | ticks under the teacher | strikes after 1 s, m/s p10 / p50 / p90 |
|---|---|---|---|
| none (v5a) | 111 | 0 | 3.6 / 6.3 / 7.9 |
| all six | 3 | 4.2% | 3.6 / 6.3 / 8.2 |
| arrival time and cap | 0 | 1.5% | 2.5 / 5.0 / 7.2 |
| cap | 0 | 1.7% | 3.5 / 5.8 / 7.6 |
| arrival time | 5 | 3.0% | 3.6 / 5.4 / 7.5 |

The hover is the cap held at its floor; with the cap raised the student's own
target, arrival velocity and time strike the puck. (A raised cap alone does
move the mallet there, contrary to the note under `--dead-puck-exploration-hold`.)

**Tiny runs from v5a** (512 envs, 4 CPU workers, 65,536 steps an update, seed
20261113, throwaway `_smoke-r3*`, deleted). Each end checkpoint played
deterministically in the profile: forward strip of the stall map, the corner
cells, and 64 x 60 s against each of `table` / `table_plus`
(`eval_human_proxy.py --tier tables`, same seeds for every candidate).

| | updates | forward strip stalled / 450 | right + left bank, left + right bank / 70 | corner cells under 46 (worst) | goals for per min | mean shot m/s | calm share above 20% cap |
|---|---|---|---|---|---|---|---|
| v5a, the start | 0 | 111 | 62, 21 | 6 (33) | 5.75 / 5.72 | 7.36 / 7.57 | 0.09 / 0.11 |
| arm a2, no anchors (control) | 10 | 93 | 55, 18 | 6 (32) | 6.05 / 5.84 | 7.37 / 7.54 | 0.10 / 0.10 |
| a3 as first committed (all six, weight 1) | 10 | 19 | 4, 8 | 1 (41) | 4.38 / 4.94 | 6.76 / 6.87 | 0.21 / 0.22 |
| the same | 20 | 13 | 3, 5 | 1 (39) | 4.72 / 4.94 | 6.28 / 6.42 | 0.24 / 0.23 |
| the same, `--lr-max 1e-6` | 10 | 34 | 13, 15 | 3 (34) | 5.94 / 5.42 | 7.35 / 7.46 | 0.14 / 0.15 |
| **cap only, weight 1 (arm a3 now)** | 10 | 12 | 3, 9 | 1 (42) | 6.11 / 6.20 | 7.40 / 7.52 | 0.16 / 0.18 |
| the same through the launcher's `--arm a3`, seed 20261114 | 10 | 8 | 2, 6 | 1 (42) | 6.20 / 5.61 | 7.41 / 7.56 | 0.15 / 0.22 |
| cap only, weight 3 | 10 | 7 | 1, 6 | 1 (41) | 5.77 / 5.48 | 7.11 / 7.21 | 0.20 / 0.22 |

For reference a2's final `agent.pt` (583.8M) in the same games: 5.52 / 5.20
goals a minute, shots 7.42 / 7.39, calm share 0.18 / 0.18. Goals against moved
within 0.14-0.42 a minute in all rows with no pattern. One seed per row; the
cap-only arm twice.

- **The stall moves, and it is the anchors.** 111 -> 8-19 of 450 in 0.66M
  steps, where a2's options alone leave it at 93; the corner cells go from six
  short to one (left / straight, 39-42, v5a 33).
- **All six outputs at weight 1 took a fifth of the goals in 0.66M steps** and
  1.1 m/s of shot speed by 1.3M, and held the actor rate at 1e-6 through twelve
  updates (it had been 5e-6). About 70% of the teacher anchor's samples are in receiving slots (a
  received puck that has lain past 1 s) and under 1% of stationary-slot steps
  are its: the cross-bank drills end fast under the wide noise. So most of what
  the six-output anchor copied was the teacher's aim and arrival velocity on
  possession pucks, from a teacher this network beats.
- **The cap alone fixes as much and takes nothing measurable from the games**,
  and the rate was back at 3.4e-6 by the tenth update. Weight 3 starts to cost
  shots; weight 1 stays.
- **A lower rate also protects the games** (`--lr-max 1e-6`: the controller
  sat at 2e-7 to 7e-7) at a third of the progress; not needed with the cap alone.
- **What is left is the teacher's own stall.** With the cap alone the only
  cell over 5% is forward-left under the right-bank request (6-9 of 70), where
  the teacher asks for a pose past the left edge; the stalled mallet ends
  0.4-0.55 m behind the puck at 20 m/s², no longer in the hover. The left /
  straight corner cell stays at 41-42 of 48 in every anchor run (v4 43-46).
- **The first two updates overshoot.** `--actor-steps-min 16` takes sixteen
  steps whatever the KL: 0.37 and 0.18 with the cap alone (0.61 / 0.33 with all
  six, 0.02 / 0.01 without anchors), 0.31-0.42 in the game slots, mostly the
  rail anchor undoing v5a's corner play. From the third update it is at the
  target. The cap-only row above includes that transient.
- **The calm share above 20% cap rises with every anchor run** (0.10 -> 0.16
  with the cap alone; a2's own final is 0.18). `eval_human_proxy` counts a
  slow puck within reach on the own half as a quarter-calm table, so striking
  pucks that used to lie there is part of it; how much is not separated. Check
  `eval_playability.py`'s idle cap before any table time.

Logged since this review, per update when an anchor is on:
`anchors.<name>.samples_by_slot` / `share_by_slot` / `initial_by_slot`
(stationary, receiving, defense, games, people), `anchors.<name>.final_kl`
(the update's policy change on its own samples), and
`anchors_final_kl_outside` / `_by_slot` (the change on the samples no anchor
covers: what the anchors cost the rest of the play). In the real run watch
`anchors_final_kl_outside_by_slot.games` and `.people` back at the target by
the third or fourth update, `game_scores.human_proxy` goals for not below
a2's, and the things listed above. If goals for fall, halve
`--reference-rail` first (it is 10-15% of all steps and most of the early KL).

### Not done in round 3

- No real run. Arm a3 has only the runs above.
- The teacher's own stalls are in its labels: x 0.15 under the right-bank
  request (forward-left 5 of 70), and its mid cross-bank cells (8-10 of 84)
  for any puck the student leaves lying there past a second.
- v4 is the rail anchor everywhere in the rail strip and in every corner /
  edge drill. It was verified on corner pucks only; the thin side-fringe
  strip of the edge drills was not measured for v4 or for the teacher.
- A puck's first second at rest, and anything the teacher would guard
  instead of attack, carries no label.
- The zone test uses the true puck position at 15 mm from the mallet-centre
  box; the margin was not tuned.

## Round 4: arm a4, from a3 (2026-11-10)

Simulation only, one network, nothing added to the runtime: two cost terms, a
narrower free pass on an existing one, two drills, and which training-only
label a lying puck gets. Arm a4 has not been trained; everything below is
measurement on a3, v4 and the teacher, unit tests and a three-update smoke run.

### What a3's evaluation still held against a table trial

a3 (`runs/_neural-table-faithful-20261110-a3/agent.pt`, 821M steps) is
stronger than v4 against the table people, gentler on blocks, heat and guard
interventions, and strikes most resting pucks. Three things are worse than or
equal to v4 (`logs/table-faithful-20261110/eval-a3/`):

1. **After its own shot the cap stays up.** 0.2-0.5 s after the puck leaves:
   cap > 40 m/s² on 55% of ticks (v4 16%), realised rms 39 against 23 m/s²;
   cap > 20 with the puck away 12.2 against 8.4 s a minute; reversals at
   1.5 m/s 49.5 against 38.1 a minute.
2. **Beside a puck it does not strike it dithers** at a cap p50 of 43 m/s²
   (v4 hovers at 8-14).
3. **Resting pucks still left**: the front edge of reach (58%), the left
   rail (18 of 30), and about one game in 64 a slow puck dribbled into its
   own back corner.

### Coverage, measured before anything was built

`resting_puck.stall_map` in the profile, seed 20261207, on a shifted grid: x
0.075-0.925 in steps of 0.05, y 0.125, 0.325, 0.525, 0.625-0.825 in steps of
0.05, 0.86, 0.875, 0.89; both starts, three requests, 1188 trials, the same
ones for a3, v4 and the teacher (`expert`), with the env's own
`info["anchor_zone"]` and dead-puck clock recorded per tick. Every puck was
placed within reach (reach ends at y 0.897; the mallet's forward limit is
0.806). Stalled trials:

| | n | a3 | v4 | teacher |
|---|---|---|---|---|
| front edge, y 0.86 | 108 | 34 | 57 | 31 |
| front edge, y 0.875 | 108 | 60 | 89 | 78 |
| front edge, y 0.89 | 108 | 97 | 108 | 108 |
| front edge, all three rows | 324 | 191 = 59% | 254 | 217 = 67% |
| left rail, x 0.075, y 0.625-0.825 | 30 | 19 | 16 | 0 |
| right rail, x 0.925, same y | 30 | 3 | 7 | 0 |
| x 0.125, same y | 30 | 3 | 16 | 7 (all under the right-bank request) |
| open forward table, x 0.175-0.825, same y | 420 | 0 | 67 | 0 |

**(a) The front edge.**
- The dead-puck clock's reach test covers it (it is the referee's own, to
  0.897), so the cost of round 2 and its noise run there.
- No drill does. The stationary drill and the cross-bank drill both stop at
  the forward limit + 91 mm - 20 mm = y 0.877, and the stationary drill's x
  is 0.17-0.83. y 0.877-0.897 was never a reset state; 0.86-0.877 is the top
  17 mm of a 0.7 m range.
- Its zone is ANCHOR_OPEN (82% of a3's stalled front ticks; the rest are the
  two rail columns), so it gets the teacher's cap. **The teacher does not
  strike these pucks**: it stalls on 67% of them, standing 0.15 m behind the
  puck in attack mode at 90 m/s² for the whole trial. Its label there is a
  raised cap in a state it has no strike from. a3 in its own stalled front
  trials: mallet 0.45-0.83 m from the puck (median y 0.24), never touching,
  cap p50 56 m/s², above 40 on 63% of ticks and above 80 on 43%. That is the
  dither of finding 2, and where it comes from. (An attribution from the zone
  and the label, not an ablation.)
- It is physically easy. 2048 placed pucks per row, random arrivals re-drawn
  every 0.24 s with the target on the forward limit within 0.15 m of the
  puck's x, 6 s: the puck crossed the centre line in 100% / 100% / 99.5% /
  87% of trials at y 0.86 / 0.875 / 0.89 / 0.895, and was struck above
  1.5 m/s in 64% / 45% / 11% / 0.2%. At y 0.89 the mallet has 7 mm of
  overlap to stop in, so a strike in the gate's sense is nearly out of
  reach and a push over the line (which the gate accepts) nearly always
  works.

**(b) The left rail.**
- Its zone is ANCHOR_RAIL on 100% of ticks: the puck's centre is within
  15 mm of the mallet-centre box's side edge (x < 0.095), so round 3 handed
  it to v4's anchor. v4 itself leaves 16 of 30 there; the teacher none.
- The edge drill's y range ends at the forward limit - 50 mm = 0.756, and an
  edge-drill slot is v4's for all its steps.
- a3 with the teacher's label executed where such a puck has lain past 1 s
  (side strip, y > 0.55; the grid's two rail columns, 60 trials): left rail
  stalled 19 -> **2 with the cap alone**, 1 with time and cap, 0 with all
  six; right rail 2 -> 2 / 1 / 0. Strike speed p50 5.6-5.9 m/s in all. The
  teacher attacks on every one of those ticks. So the same cap-only anchor
  that fixed the open table fixes this strip, if the strip is the teacher's.

**(c) The back corner.**
- A wedged puck is freeable in the sim, rarely by chance. 512 pucks touching
  both rails at (0.041, 0.041), random arrivals for 12 s (half of them aimed
  into the corner region): touched in 60%, moved more than 5 mm in 22%, got
  more than 200 mm out in 5.1%. Starting 20-50 mm off one rail: 12-33%.
- Training already charges it heavily: a reachable resting puck is never
  relaunched in a training game (`continuous_rallies` relaunches only pucks
  out of both mallets' reach), so the dead-puck cost, the possession delay
  and, at load 0.95, the 12,000 shutdown all apply. What is missing is not
  size. The penalty arrives seconds after the dribble that caused it, and no
  drill starts on the back rail between the corners (corner drills: within
  15 mm of the box's corner; edge drills: the side fringes).

### What the trainer gained (all opt-in, defaults unchanged)

| flag | default | what it does |
|---|---|---|
| `--far-cap-brake-seconds S` | none | With `--far-cap-max-speed`: a tick begun at or above that mallet speed is free of the far-cap cost only within S of the learner's last puck contact. None = round 2's rule (free at any time). |
| `--dead-puck-cap-weight W`, `--dead-puck-cap-grace S` | 0, 2.5 | Cost per second of the cap fraction above the 5% floor while the dead-puck clock is past S. The cap at its floor costs nothing. |
| `--teacher-resting-end S` | none | The teacher anchor applies only while the clock is between `--teacher-resting-start` and S. |
| `--anchor-side-rail-open-y Y` | none | `info["anchor_zone"]`: a puck lying in a side-rail strip forward of Y is ANCHOR_OPEN (the teacher's), not ANCHOR_RAIL (the reference's). Also in an edge-drill slot. |
| `--anchor-front-margin M` | none | A puck lying more than M past the mallet's forward limit is ANCHOR_NONE. |
| `--reach-edge-drill-fraction F` | 0 | Share of stationary drills with the puck at rest in the last 50 mm of forward reach across the whole width (60%), or against a side rail at y 0.58 to there (40%). Half the mallets start within 0.3 m of the puck, behind or beside it. An ordinary stationary slot: 6 s, the dead-puck clock, no reference anchor. |
| `--back-rail-drill-fraction F` | 0 | Share of stationary drills with a puck on the back rail between the corners, at rest or sliding along the rail at up to 0.8 m/s. |
| `--own-corner-weight W` | 0 | Cost, once per entry, when a puck the learner touched goes into a back-corner pocket (centre behind the mallet-centre box's back edge and outside a side edge) slower than 2 m/s and comes to lie there. Not in a corner / edge drill slot, whose puck starts there. |

Two motion columns were appended (`dead_puck_cap`, `own_corner`; the eleven
earlier ones keep their places). `status.json` gains
`motion.dead_puck_cap_cost_per_s`, `motion.own_corner_per_minute`,
`motion.own_corner_cost_per_s` and, per slot class in `motion_by_slot`, those
three plus `far_cap_cost_per_s` and `dead_puck_cost_per_s`, only when a round-4
option is on: an earlier arm's status has the keys it always had.

### Arm a4

```bash
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --arm a4 --run-name _neural-table-faithful-20261110-a4 --minutes 150 --print   # inspect
PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
    --arm a4 --run-name _neural-table-faithful-20261110-a4 --minutes 150
```

Arm a3's command (`ARMS["a4"]`; a typed flag still wins) with:

```
--resume runs/_neural-table-faithful-20261110-a3/agent.pt   (a launcher flag)     --seed 20261115
--far-cap-brake-seconds 0.25 --far-cap-threshold 20          (was: no window, 40)
--dead-puck-cap-weight 100 --dead-puck-cap-grace 2.5 --teacher-resting-end 2.5
--anchor-side-rail-open-y 0.55 --anchor-front-margin 0.06
--reach-edge-drill-fraction 0.2
--back-rail-drill-fraction 0.05 --own-corner-weight 300
```

Why each:

- **The walk back (finding 1).** Round 2 made every tick begun above 1 m/s
  free of the far cap so that asking for the brake after a strike costs
  nothing; a walk back to guard at 3 m/s is above 1 m/s too. The window keeps
  the stop free and nothing else: from 5 m/s the stop takes 0.08 s at
  60 m/s², the puck is over the line within 0.1 s of the contact, and a3's
  excess is at 0.2-1.0 s. A lunge at an incoming shot is before the contact
  and is charged as in arm A (at most the 8 of its table). The threshold
  goes from 40 to 20 because the gate that fails is cap > 20 with the puck
  away, and at 40 a walk back at 39 m/s² is free. Of the changes here this
  is the one with the least evidence behind its number.
- **The dither (finding 2).** `--calm-reachable-scale 0` stays: the run into
  a strike is free. A strike on a resting puck starts at a median 0.3 s and
  by about 1 s at p99, and the teacher's cap label begins at 1 s, so by 2.5 s
  on the clock a strike that was going to happen has. From there the cap is
  charged, and the label that says "90" stops, so the two do not pull against
  each other on the same sample. The network cannot see the clock (16 frames
  are 0.32 s): what it can learn is that a raised cap that does not turn
  into a strike is paid for a second and a half later. The wide time/cap
  noise of round 2 keeps running past 2.5 s, so in the rollout part of this
  cost is the noise's own (see the smoke run).
- **The front edge (3a).** The drill puts the state in the rollout; half its
  mallets start close enough that the push is a short move. The teacher's
  cap is dropped more than 60 mm past the forward limit (y > 0.866): at 0.86
  the teacher stalls on 29%, at 0.875 on 72%. Dead-puck cost, noise and the
  600 at 6 s are what is left there, as in round 2; this is the part of a4
  that rests on reward and exploration alone, which is what round 2 found
  wanting on the open table.
- **The left rail (3b).** Forward of y 0.55 the side strips become the
  teacher's (cap only, as on the open table); below it v4 strikes them (0
  stalls at y 0.525 and under on this grid) and keeps them, with the corners.
  The drill's side part covers y 0.58-0.847, the edge drill's own range ends
  at 0.756.
- **The back corner (3c).** 300 at the moment the puck comes to lie in the
  pocket, half a goal, for the dribble that put it there; a fast puck (it
  bounces out) or one the learner did not touch is free. 5% of stationary
  drills start on the back rail. Nothing was added for getting a wedged puck
  out: v4, whose corner play is the anchor, hovers over one for 42.8 s too.

Per 128-env worker the stationary slots (20 of 128) become about: corner 20%,
edge 12%, reach edge 13%, cross-bank 13%, back rail 3.4%, the rest uniform or
replayed.

### What the new terms cost, against the goal flow

Goals pay 600; against people the flow is about 50 a second (5 a minute).

| | rate | a case |
|---|---|---|
| far cap, threshold 20 | 40/s x (cap - 20) / 80: 10/s at 40, 20/s at 60, 40/s at 100 | 0.5 s of walk back at 60 m/s²: 10. The stop within 0.25 s of the contact: 0. A cap of 100 for 0.2 s before an incoming shot crosses: 8, as in arm A's table. A guard held at 30 through the puck's whole time away: 5/s. |
| dead-puck cap | 100/s x (cap fraction - 0.05) | a3's dither at 43 m/s²: 38/s, on top of the 40/s dead-puck cost. v4's hover at 8-14: 3-9/s. A strike begun after 2.5 s, 0.3 s at the full cap: 28, against up to 160 of shaping, the goal and the end of the 40/s. Before 2.5 s: 0. |
| own corner | 300 per puck | half a goal, once; a save deflected slowly into the pocket pays it too and is still 300 better than the goal in a game. |

A set strike costs what it did in round 2 (14): none of the three charges a
tick of it. The all-out save's 64 already counted the far cap at its worst
(8), which the lower threshold does not change.

### Smoke test of arm a4 (2026-11-10)

`--arm a4 --n-envs 512 --workers 4 --device cpu --steps 196608` from a3 into a
throwaway `_smoke-a4` run (run, log and evaluation directory deleted). Three
updates, everything finite.

| | update 1 | 2 | 3 |
|---|---|---|---|
| explained variance | 0.63 | 0.92 | 0.96 |
| final KL (target 0.02), actor steps | 0.014, 64 | 0.009, 64 | 0.020, 64 |
| actor lr | 5.06e-6 | 5.06e-6 | 7.59e-6 |
| KL outside the anchors, games / people | 0.015 / 0.019 | 0.010 / 0.010 | 0.020 / 0.019 |
| far-cap cost per env-second, all slots | 3.1 | 3.2 | 3.1 |
| the same: stationary / receiving / defense / games / people | 2.0 / 0.03 / 5.2 / 3.9 / 4.7 | 1.8 / 0.03 / 5.4 / 4.0 / 5.2 | 1.6 / 0.03 / 5.3 / 4.2 / 4.4 |
| far-half time with the cap above 20 and charged | 0.45 | 0.41 | 0.41 |
| dead-puck cap cost per env-second, all slots | 0.06 | 1.7 | 1.8 |
| the same: stationary / games / people | 0.4 / 0 / 0 | 10.7 / 0 / 0 | 11.5 / 0 / 0.15 |
| dead-puck cost: stationary / games / people | 10.3 / 0.07 / 0.28 | 15.6 / 0.12 / 0.43 | 18.7 / 0.07 / 1.1 |
| stationary slots: time with a puck lying past 1 s | 0.26 | 0.39 | 0.47 |
| own-corner events (in 1,310 env-seconds an update) | 1, a people game | 1, a defense drill | 0 |
| teacher anchor: samples, pre-squash error before -> after | 141, 0.057 -> 0.050 | 167, 0.060 -> 0.052 | 162, 0.098 -> 0.083 |
| rail anchor: samples, error | 6720, 0.006 | 7434, 0.009 | 7824, 0.008 |
| games vs people, cumulative | 7-1 | 20-2 | 34-2 (6.0 for, 0.36 against a minute) |

Read from it:

- **The far cap is now the largest of the round's terms in play**: 4-5 a
  second in games and against people, 5.3 in defense drills, a tenth of the
  goal flow, where arm a2's gated version took 1.2 over all slots. That is
  the intended pull (cap > 20 on 41-45% of the charged far-half time is the
  thing to bring down), and it is also the first weight to halve
  (`--far-cap 20`) if goals against people or the defense drills' block rate
  fall. Over all slots the exploring policy pays about 20 a second for every
  motion and dead-puck term together (arm a2's smoke: 17-20).
- **The dead-puck cap is paid in the stationary slots only**, 11 a second
  there once the 6 s and 12 s drills fill the rollout, next to 16-19 of the
  dead-puck cost itself. Part of it is the round-2 noise drawing caps at
  random where a puck has lain past 1 s, whatever the mean; the mean's share
  is what the policy can remove.
- **The anchors start near zero** (a3 was trained on them; the teacher's
  samples are now 0.2% of steps with the window and the front margin), and
  the policy change outside them is at the target from the first update.
- Two own-corner events in three updates are too few to say anything about a
  rate.

Flags off is still the trainer from before: arm a3's own command for one
update (CPU, 256 envs on two workers, `--resume` a3), run from an exported
tree of the commit before and from this tree. Weights, optimiser state, RNG
state and every metric but the wall-clock ones are identical; the saved
`args` gain the new option names. The printed commands of arms a, a2 and a3
are byte-identical.

### Gates for a4

As for a2 and a3 (`eval_resting_puck.py`, `eval_playability.py`,
`eval_human_proxy.py --tier tables`, all with `--sim-profile table-20261029`),
plus the shifted stall grid that shows the front edge and the rails:

```bash
python ai/bin/eval_resting_puck.py runs/_neural-table-faithful-20261110-a4/agent.pt \
    --reference runs/_neural-table-faithful-20261110-a3/agent.pt --sim-profile table-20261029 --parts stall \
    --xs 0.075,0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.525,0.575,0.625,0.675,0.725,0.775,0.825,0.875,0.925 \
    --ys 0.125,0.325,0.525,0.625,0.675,0.725,0.775,0.825,0.86,0.875,0.89
```

and from the a3 evaluation's own measures: cap > 40 at 0.2-0.5 s after its own
shot and cap > 20 with the puck away at or below v4's (16%, 8.4 s a minute),
the cap beside a stalled puck at v4's (p50 under 15), goals for not below
a3's.

### Not done in round 4, and what is not known

- No real run. Whether 150 minutes move any of the three findings is
  untested; the smoke run shows only that the terms are finite, sized as
  intended and leave the KL at its target.
- The task's premise that the teacher strikes front-edge pucks did not hold
  on this grid (67% stalled), so a4 has no label source there. If the front
  edge does not move, the next step is a label that does strike it (a
  scripted push from the forward limit as a training-only anchor), not a
  larger cost.
- The left-rail fix was shown by executing the teacher's cap in a3's own
  trials (19 -> 2 of 30), not by training.
- The far-cap threshold of 20 and the 0.25 s window are set from the brake's
  arithmetic and a3's timing, not from a run. Their effect on defense
  (a low cap held until the shot is on its way) is not measured.
- The cap charge past 2.5 s and the end of the teacher's label there could
  bring back a calm hover (v4's behaviour: quiet, but still a puck left
  lying) in states where the strike is not found. That is the lesser
  failure of the two on a table without a referee only in that it does not
  heat or shake the machine.
- The corner cost charges a slow block deflection that dies in the pocket
  (one of the smoke run's two events). Fast pucks and untouched ones are
  free; nothing separates a dribble from a slow deflection.
- A wedged corner puck is still not struck out by anything here; the 5% of
  random sequences that free one were not turned into a drill or a label.
- Small reversals (49.5 against 38.1 a minute) have no new term; the
  reversal cost of arm A is unchanged.

## Free shot requests: who picks the route (2026-11-10)

Until now the network executed a route and never chose one: every possession
carried an explicit request (left bank / right bank / straight), drawn
uniformly in training (`batch_env._draw_shot_types`) and on the table
(`--shot-type mix`). The all-zero input ("none") exists in the encoding and
the reward (`route_ok = (requested == 0) | matched`) but had probability 0,
apart from the ticks before a game's first entry with the puck on the far
half. Three pieces, all opt-in, the defaults unchanged (a fixed-seed rollout
of 48 slots x 600 steps, two option sets, hashes equal against the commit
before: observations of both seats, rewards, requests).

### Training: `--free-request-share P`

In that share of the LEARNER's possessions the request is none. Then any
aimed, on-target shot earns the request bonus a matched one earns
(`--shot-request-weight` x the speed term), whatever its route; nothing is a
wrong route (no quarter scaling, no `--wrong-shot-penalty`); the rest of the
reward is the same code. The other 1 - P keep explicit requests in the old
proportions (`--left-bank-request-weight` included), so every route stays
trained. The far seat is never left free: it may be an older pool snapshot
that has not seen "none" with the puck on its side. A `--shot-setup-fraction`
start under a free request is laid out for a drawn route and stays free.

`status["routes"]` (only with the option on) has, for `free` and `requested`
shots of the learner in that rollout: `shots`, `unresolved`, `share` by
route, `aimed_rate` and `goal_rate` per route (a goal goes to the learner's
last shot unless the far side touched the puck after it), and
`entropy_bits` of the route mix (1.585 = even thirds, 0 = one route).
All slot kinds are counted, drills too.

Arm a4 now carries `--free-request-share 0.5` (launcher `--free-request`;
`--free-request 0` gives the round-4 arm as it was written up above).

### The governor and the route it reads back (`airhockey/shot_selector.py`)

`ShotSelector("auto")` is asked once per possession (the puck entering the
robot's half) and answers none unless variety is at risk: over the last 30
completed robot shots, by REALISED route, a route under 15% is requested
explicitly (the most starved first), else a route over 60% gets the less
used of the other two requested, else none. Before 9 shots are in the window
it answers none. In every state a uniform explicit request is drawn 15% of
the time, so the requested skills stay in use. It controls the request, not
the shot: against a network that ignores requests it asks for the missing
route every possession and `stats()` reports the honoured share.

Simulated shooters, 3000 possessions each, three seeds
(`tests/test_shot_selector.py`):

| shooter when free (honours a request) | free possessions | realised L / R / S | worst 60-shot stretch |
|---|---|---|---|
| always straight | 55-56% | 20 / 20 / 61 | banks >= 15% each, straight <= 65% |
| never left (right / straight evenly) | 72-74% | 17 / 42 / 41 | left >= 13%, any route <= 62% |
| even thirds | 82-85% | 33 / 33 / 33 | every route 18-52% |
| always straight, ignores requests | 0.2-0.3% | 0 / 0 / 100 | nothing it can do; honoured share 0 |

`RouteClassifier` reads the realised route from what the actor itself is
given each tick (puck fix, 30 ms slope velocity, own mallet; robot-local
metres), so the same code runs in the evaluator, on the table and on a
session's ticks.csv. A shot starts when the puck's forward speed jumps above
1 m/s on the own half with the mallet at it and not ahead of it. The route
is the reward's definition: a side-rail bounce seen before the centre line
decides it, otherwise the state at the line is traced to the far goal line
through `rewards.predict_shot` (straight = no rail, else the first rail). The
person cannot reach over the line, so their touch never enters. Not counted:
the puck lost for more than 0.2 s in flight, stopped or turned back before
the line, or never arriving.

How reliable, measured:

- In sim against the env's own route of the same shot (a3 / v4, the runs
  below, both tiers, all three request modes): a3 96-98% of 1627-2051
  resolved shots per tier, v4 92-95% of 1371-1570; resolved shots are
  92-99% of the robot's attacking crossings of the line, and 1-25 per tier
  were dropped. Most disagreements are a bank by the env's launch-time
  trace read as straight at the line.
- On the table, `python ai/bin/shot_route_replay.py 20261110-203214
  20261029-182715` (a3, 6.3 min; v4, 5.9 min): 134 and 106 shots detected,
  133 and 103 resolved (131 fast upward crossings of the line in a3's
  ticks). Against the 200 Hz puck track of the replay file (which rail the
  puck actually reached before the far end), where the track gets that far:
  100 of 105 (95%) and 80 of 85 (94%) agree. The other 29 and 19 shots
  cannot be checked this way (the person took the puck first); the
  classifier called most of them straight. So the realised route is used as
  it is; the fallback (count the request where the route is undetermined,
  `count_requested_when_unclassified`) exists and is off.
- The same replay shows what the table does with requests: a3 made the
  requested route on 63% of its classified shots (left 37/48, right 20/40,
  straight 27/45) and v4 on 71% (32/39, 18/34, 23/30), against 93% and
  80-82% in the sim below. Blocks that send the puck back count as shots
  here as they do in the env.

### Zero-shot: a3 and v4 with no request (they were never trained on it)

`eval_human_proxy.py <ckpt> --tier tables --sim-profile table-20261029
--shot-request random|free|auto`, 64 x 60 s per tier, seed 20261018 (one
seed; a second pass of all twelve gave the same scores). Reports:
`logs/human-proxy/shot-request-20261110/`. Mix =
share of routed shots left / right / straight; on target and goals are per
shot of that route.

| | request | tier | goals | GF / GA per min | shots | on target | mix L/R/S | on target by route | goals per shot by route | stall s |
|---|---|---|---|---|---|---|---|---|---|---|
| a3 | random | table | 389-7 | 6.08 / 0.11 | 1565 | 89% | 35/32/33 | 88/89/90% | 25/33/15% | 44 |
| a3 | random | table_plus | 368-11 | 5.75 / 0.17 | 1652 | 88% | 35/32/33 | 88/87/91% | 19/28/20% | 53 |
| a3 | free | table | 59-1 | 0.92 / 0.02 | 2032 | 3.5% | 3/90/7 | 13/1/30% | 3/2/10% | 93 |
| a3 | free | table_plus | 60-3 | 0.94 / 0.05 | 1976 | 3.7% | 3/91/6 | 20/1/32% | 3/3/8% | 54 |
| a3 | auto | table | 208-5 | 3.25 / 0.08 | 1856 | 38% | 21/61/18 | 85/10/79% | 24/5/14% | 84 |
| a3 | auto | table_plus | 189-8 | 2.95 / 0.12 | 1886 | 38% | 20/60/19 | 84/9/81% | 20/4/16% | 43 |
| v4 | random | table | 292-25 | 4.56 / 0.39 | 1275 | 76% | 38/29/33 | 73/73/83% | 21/31/13% | 143 |
| v4 | random | table_plus | 282-31 | 4.41 / 0.48 | 1375 | 70% | 35/28/37 | 69/65/76% | 19/25/16% | 148 |
| v4 | free | table | 68-11 | 1.06 / 0.17 | 1309 | 7% | 10/81/9 | 22/3/29% | 12/3/11% | 209 |
| v4 | free | table_plus | 52-16 | 0.81 / 0.25 | 1403 | 6% | 9/83/8 | 23/2/30% | 8/2/12% | 99 |
| v4 | auto | table | 155-23 | 2.42 / 0.36 | 1322 | 32% | 22/60/18 | 68/9/69% | 24/6/14% | 155 |
| v4 | auto | table_plus | 127-32 | 1.98 / 0.50 | 1371 | 29% | 22/60/18 | 59/10/60% | 17/5/14% | 87 |

What it says:

- **Free requests are not usable on a3 or v4.** With no request both send
  80-90% of their shots toward the right rail and 1-3% of those are on
  target: a sixth of the goals. It is one failure, the same in both
  checkpoints, not a choice of route. Concessions do not rise (they fall
  with the goals: fewer restarts), reachable-puck stall time is mixed (a3
  table 44 -> 93 s of 64 min, table_plus unchanged; v4 143 -> 209 and
  148 -> 99), and v4 on `table` ran 17 s at the load limit where it has
  none under `random`.
- **`auto` on these checkpoints holds the mix inside its band and costs
  half the goals.** 61-71% of possessions were left free, and the free ones
  are the broken shot (a3: 91% right, 2% on target); the governor's explicit
  requests are as good as under `random` (on target 87-93%, honoured 93-95%;
  v4 66-87%, 86-87%). It keeps the right bank at its 60% ceiling, which is
  where the mix sits. So the governor does what it promises and that is not
  enough when the free shot itself is bad: **keep `--shot-type mix` for a3
  and v4.** `free` and `auto` are for a checkpoint trained with a free share,
  and both print as a DEVIATION otherwise.
- Under `random` the bank off the right rail scores most per shot in this
  sim (28-33% against 15-20% straight) at the same on-target rate. If that
  holds in training, the free half of a4 should drift right, and the
  entropy in `status["routes"]` is the number to watch; the governor's
  ceiling is what bounds it on the table.

### Table

```bash
bash ai/bin/play.sh --policy neural:<run> --shot-type auto     # free, with the variety governor
bash ai/bin/play.sh --policy neural:<run> --shot-type free     # no request at all
#   --shot-window 30 --shot-floor 0.15 --shot-ceiling 0.60 --shot-min-shots 9 --shot-explore 0.15
python ai/bin/run_policy.py --check-policy --policy neural:<run> --shot-type auto   # offline: prints the alignment line
python ai/bin/shot_route_replay.py <stamp> --governor          # a session's requests, routes and the window
```

Default stays `mix`. The runner prints a `sim/real alignment: shot request
...` line with what the checkpoint trained with (`free_request_share` in its
run.json) and marks `free` / `auto` a DEVIATION for a run without one, as a
fixed route always was. Each tick logs `shot_request` and, on the tick a
robot shot resolves, `shot_route` (or why it was dropped); the session ends
with a `[shots]` line of the selector's counts. Sessions before this have the
request in the last three values of `neural_obs`, which is what the replay
tool reads.

### Smoke test of arm a4 with the free share

`--arm a4 --n-envs 384 --workers 3 --device cpu --steps 98304` from a3 into a
throwaway `_smoke-free-a4` run (run, log and evaluation directory deleted).
Two updates, every status value finite, `free_request_share` 0.5 in its
run.json.

| | update 1 | 2 |
|---|---|---|
| explained variance | 0.70 | 0.78 |
| final KL (target 0.02), actor steps | 0.014, 48 | 0.012, 48 |
| free shots: n, unresolved | 339, 4 | 330, 11 |
| free mix L / R / S, entropy | 15 / 75 / 10%, 1.05 bits | 17 / 74 / 9%, 1.06 bits |
| free aimed rate L / R / S | 33 / 9 / 29% | 29 / 8 / 26% |
| free goals per shot L / R / S | 0 / 2.0 / 0% | 0 / 0.4 / 0% |
| requested shots: n, mix, entropy | 331, 35 / 32 / 33%, 1.58 bits | 351, 32 / 33 / 35%, 1.58 bits |
| requested aimed rate L / R / S | 67 / 71 / 77% | 69 / 64 / 82% |
| requested goals per shot L / R / S | 7 / 5 / 11% | 7 / 10 / 12% |

About half the learner's shots are free and they start where the zero-shot
evaluation found them: three quarters toward the right rail, a tenth of
those aimed. Two updates show the log, not a trend. (The rates are over all
slots with exploration noise, so they are below the evaluation's.)

### Not done, and what is not known

- No training run. Whether a network trained with a free share picks routes
  by scoring chance, or collapses onto one, is not known; the zero-shot
  numbers say only where a3 starts from (the free half of a4's possessions
  begins at 1-3% on target).
- Arm a4 now changes two things at once from a3: round 4's fixes and the
  free share. If its evaluation moves, the two are not separated; `--arm a4
  --free-request 0` is the arm without it.
- Under a free request the bonus pays an AIMED shot of any route, not a
  goal. The routes are about equally easy to aim (88-90% under requests),
  so the tilt between them comes from goals and conversion; nothing rewards
  variety in training. Variety is the governor's job at run time.
- `auto` has not been on the table. The classifier was checked on two
  logged sessions, not live; its numbers on mallet-pinned pucks at a rail
  are worse (a scripted striker in the tests agreed with the env on 75%).
- Only `eval_human_proxy.py` takes `--shot-request`; the other evaluators
  still draw explicit requests.
- The governor's numbers (30 / 15% / 60% / 9 / 15%) are chosen, not tuned
  against play.

## Round 5: the soft mallet and the puck after a block (2026-11-19, v4.3.x)

Asked after the 2026-11-17/18 table sessions of v4.1.2: defense is as wanted
(the person "basically just can't score"), offense is not. Two complaints:
it lets a puck drift slowly off its half with room to go round it and take
control, and it rushes shots before it is set up.

### What the table logs say (v4.1.2, 37 min, 10 sessions, 11-17 and 11-18)

Same code on the table's `replay.jsonl` and on sim recordings against the
`table` / `table_plus` people (possession = one stay of the puck in the
robot's half that ends with it leaving):

| | table | rigid sim (`table-20261029`) |
|---|---|---|
| possessions ending in a shot (leaves >= 2 m/s within 0.7 s of a touch) | 49.5% | 82.6% |
| leaves under 2 m/s, robot not touching it in the last 0.7 s | 13.2% | 5.0% |
| leaves at 1-2 m/s | 19.6% | 6.0% |
| shots: puck speed before the last touch, 75th percentile | 1.37 m/s | 0.87 m/s |
| shots: one touch only | 35.8% | 15.1% |

Ruled out: the puck does not move by itself. Free slow pucks (560 windows of
0.4 s, away from both mallets and the rails) accelerate at 0.008 m/s^2 on
average, i.e. not at all, so table tilt or air is not what carries them off.

What the slow untouched exits are (64 of them, 1.75 a minute): the puck came
in at 6 m/s (median), the robot BLOCKED it, and the rebound left at 1-1.7 m/s
about 0.9 s later with the mallet staying 300-500 mm behind it, moving at
0.8 m/s under a 40 m/s^2 cap. The puck was within the mallet's reach for
0.8 s (median) of that. This is the "drifts away with room to go round it".

After a block of a shot of 4 m/s or more:

| | table | rigid sim | `table-20261118` (gain 0.5-1.5) | gain 1.5-3.0 | gain 2.5-4.5 |
|---|---|---|---|---|---|
| rebound speed, quartiles, m/s | 1.06 / 1.74 / 2.91 | 0.55 / 0.90 / 2.00 | 0.77 / 1.36 / 2.44 | 1.14 / 1.96 / 3.14 | 1.36 / 2.15 / 3.33 |
| rebound / incoming speed, median | 0.27 | 0.14 | 0.20 | 0.29 | 0.33 |
| rebound under 2 m/s then recovered (touched again) | 34% | 70% | 56% | 47% | 39% |
| block followed by recovery and a shot | 19% | 53% | 37% | 26% | 17% |

In the rigid sim the policy cushions its blocks (the reward's off-target
penalty makes a hard unaimed return expensive) and collects the dead puck.
On the table the same blocks come off twice as fast and two thirds of the
slow ones are not collected.

### The sim's mallet was rigid

The policy's own-mallet input is the controller's step-count position, on
the table and in the sim. On the table the mallet hangs on cables and
springs: the camera sees it 5-9 mm rms from the controller's position, 40 mm
at the 90th percentile when moving fast. In the sim the mallet WAS the
profile. `BatchAirHockeyEnv.enable_mallet_compliance` (profile part
`compliance`) makes it a damped mass driven by the profile's acceleration,
per episode and axis: frequency, damping, gain. The puck meets the true
mallet; the policy still sees the profile.

- `table-20261118`: frequency 10-26 Hz, damping 0.03-0.15, gain 0.5-1.5 (a
  first guess from the ring test). Moves v4.1.2 about half way to the table.
- `table-20261119`: gain 1.0-3.5, frequency 8-26 Hz, damping 0.02-0.15,
  chosen so the block statistics above land on the table's. Under it v4.1.2
  reads: shots 62%, slow untouched exits 12%, exits at 1-2 m/s 22%, one-touch
  shots 28%, rebound median 1.90 m/s, slow rebounds recovered 45%.

CAVEAT. Every one of those table sessions predates the camera recalibration
of 2026-11-18 21:53. The table markers had drifted 13-18 mm since October, so
the puck was seen that far from where the step counts put the mallet. The
calibrated gain absorbs that error too. The 11-18 evening sessions (drive
gains at KI x0.35, 9 min) already differ from 11-17 (27 min): one-touch
shots 26% against 40%, block-then-shot 27% against 16%. The table as it is
now may sit between the two profiles.

### Why v4.1.2 never learned the recovery

- Defense drills are block-only (`--defense-block-only`): the episode ends,
  paid 300, the moment the touched puck moves away. Nothing in 45% of the
  practice slots ever shows what the block leaves behind.
- In games a slow puck leaving the half costs nothing by itself
  (`--slow-exit-penalty` 0, `--unproductive-return-penalty` 0), while
  chasing it costs acceleration, heat and position. The trainer counts
  these: 4.3-5.4 "slow possession losses" a minute in training games,
  against 1.9 goals a minute.
- The recovery drill (a puck moving away at 0.15-1.2 m/s, mallet behind it)
  pays a fast aimed strike from behind as a success. In the drill v4.1.2
  takes control of 1 puck in 128 and strikes 97 from behind on target.

### What the trainer gained (all opt-in, defaults unchanged)

- `--sim-profile table-20261118 | table-20261119`: the compliant mallet.
  `finetune.py --trainer-arg=<token>` appends raw trainer options (a later
  flag overrides the arm's).
- `--defense-followthrough-fraction F`: that share of block-only defense
  drills goes on after the block. The block is paid (300) once, the episode
  then runs until the puck leaves the half, a goal, or 4 s; a conversion
  adds 60, a fast aimed strike 35, holding the puck to the clock costs 60.
  The slow-exit and unproductive-return costs apply on the way out.
- Already there, off in every earlier arm: `--slow-exit-penalty` (a puck
  that was within reach leaves the reach zone forward under 2 m/s with no
  fast aimed shot this possession), `--unproductive-return-penalty`,
  `--recovery-min-speed / --recovery-max-speed / --recovery-fraction`,
  `--recovery-capture-bonus`, `--recovery-get-ahead-bonus`,
  `--recovery-persistent-bias` (random offsets held 8 ticks before the first
  contact of a recovery drill: correlated exploration).
- `--roam-weight` (difficulty models, see the end).

### Arms (all `finetune.py --arm accel`, 1280 envs, learner at 100 m/s^2)

| version | from | sim | what changed against v4.1.2's recipe | steps |
|---|---|---|---|---|
| v4.3.0 | v4.1.2 | `-20261118`, 75% of workers | nothing else | 127M |
| v4.3.1 | v4.1.2 | same | unproductive return costs 30 | 128M |
| v4.3.2 | v4.1.2 | gain 0.8-2.5, every worker | nothing else | 127M |
| v4.3.3 | v4.1.2 | `-20261118`, 75% | unproductive 30; recovery drill to 2.2 m/s, capture bonus 60, get-ahead 30 | 83M |
| v4.3.4 | v4.3.2 | `-20261119` with gain 0.3-3.5, every worker | v4.3.3's terms + slow exit costs 100 | 84M |
| v4.3.5 | v4.3.2 | same | "control first": slow exit 200, unproductive 60, shots at a moving puck worth half and fined 20, strike-from-behind credit 0.3 | 48M, stopped |
| v4.3.6 | v4.3.2 | same | v4.3.4 + follow-through defense drills 0.5 | 95M |
| v4.3.8 | v4.3.6 at 40M | same | v4.3.6 + recovery drill at 0.6-2.4 m/s in 70% of receiving drills, persistent exploration 1.0 | 98M |
| v4.3.9 | v4.3.8 at 40M | same | v4.3.8 with KL target 0.05 (0.02 elsewhere) | 48M, stopped |
| v4.3.10 | v4.3.8 at 40M | same | slow exit 200, unproductive 60, follow-through 0.7, recovery share 0.8, exploration 2.0, KL 0.03 | 91M |
| v4.3.11 | v4.3.8 at 40M | same | tilted to control: capture bonus 120, get-ahead 60, strike-from-behind credit 0.3, exploration 2.0, KL 0.03 | 50M, stopped |
| v4.3.12 | v4.3.8 at 40M | same | v4.3.8's recipe with 3840 envs on 15 workers: 655k samples an update against 164k | 90M |
| v4.3.13 | v4.3.12 at 90M | same | the same with 7680 envs on 24 workers: 983k samples an update (35k steps/s alone on the machine) | 96M |

The first four ran side by side from 23:07 and were stopped at 02:02 when
their 80M and 120M snapshots showed goal share up a little in the soft sims
and nothing else moving. v4.3.4-6 first ran 15 minutes from v4.3.3 under
`-20261118` and were restarted from v4.3.2 once the calibration was done.
v4.3.5 lost defense (30 and 42 goals conceded in the two sims against 18
and 35 for v4.1.2) without gaining control and was stopped.

### Difficulty models that roam: three attempts, none worked (v4.3.7)

Asked for at the same time: the easier levels (v4.2.x, a body limited to
40 / 25 / 15 m/s^2) cannot shoot and still defend as well as the full model
(0.13 goals conceded a minute at `medium` against 0.14 at `max`); make them
strategically worse at full speed instead, standing out from the goal.

`--roam-weight W` pays `W` per second for standing forward in the workspace
while the puck is on the far half and no fast shot is incoming. Three runs,
each evaluated by where the mallet stands while the puck is on the far half
(v4.1.2: median 127 mm in front of its goal line, 0.14 conceded a minute):

| attempt | reward | from | steps | standing depth, median | conceded / min |
|---|---|---|---|---|---|
| 1 | W 40 and 120, forward of 0.30 m only | v4.1.2 | 34M each | 134, 139 mm | 0.11, 0.08 |
| 2 | W 60, from the back edge of the workspace; teacher anchor off | v4.3.2 | 26M | 120 mm | 0.12 |
| 3 | W 100, half on the mallet's position and half on the destination of the arrival asked for; teacher off | v4.3.2 | 21M | 121 mm | 0.12 |

Attempt 1 had no slope where the mallet stands (a bug in the reward).
Attempts 2 and 3 have one and still moved nothing in 130-160 updates. Not
understood; candidates are the size of an update at this learning rate
(7.6e-6, KL 0.02) against a habit 900M steps old, and the 600-per-goal cost
of every step forward against people who shoot at 7 m/s. The run kept as
v4.3.7 is attempt 3; the others were deleted. It is NOT a difficulty model.

A different handle that does work in sim, measured but NOT built into the
runner: delaying what the camera shows the policy. v4.1.2 at full speed
against the table people (32 games per tier, `table-20261029`):

| extra camera latency | goal share (`table` / `table_plus`) | robot goals / min | conceded / min | shot speed, m/s |
|---|---|---|---|---|
| 0 | 97.7% | 5.9 | 0.14 | 7.3 |
| 60 ms | 86% / 74% | 3.5 / 3.1 | 0.6 / 1.1 | 5.2 |
| 120 ms | 57% / 58% | 2.1 / 2.6 | 1.6 / 1.9 | 4.2 |
| 200 ms | 49% / 45% | 2.0 / 2.2 | 2.1 / 2.7 | 3.7 |

That is slower reactions, not worse strategy, and zero-shot on a network
never trained with the delay (its strikes at moving pucks mistime too).

### Results (sim only; nothing here has been on the table)

Against the `table` / `table_plus` people, 64 one-minute games per tier.
Possession and block numbers from 32 recorded minutes per seed in the
calibrated sim. v4.1.2 and the bold rows are means of three seeds; the
others one seed unless marked, for which a difference of 3 points in the
possession columns is noise.

| model (snapshot) | rigid: share, GF / GA per min | calibrated: share, GF / GA | ends in a shot | slow untouched exit | after a block: recovered / left slowly |
|---|---|---|---|---|---|
| table, v4.1.2 | | | 49.5% | 13.2% | 25.8% / 21.8% |
| v4.1.2 | 97.5%, 6.07 / 0.16 | 94.6%, 4.75 / 0.27 | 61.9% | 13.0% | 27.2% / 26.7% |
| v4.3.0 | 98.4%, 5.20 / 0.09 | 96.4%, 4.87 / 0.18 | 62.7% | 10.9% | 29.7% / 24.3% |
| v4.3.1 | 98.7%, 4.77 / 0.06 | 96.7%, 4.85 / 0.16 | 60.3% | 14.0% | 27.5% / 28.5% |
| v4.3.2 | 98.5%, 5.19 / 0.08 | 97.1%, 5.20 / 0.16 | 59.7% | 12.1% | 23.9% / 25.8% |
| v4.3.3 | 98.7%, 5.53 / 0.07 | 95.9%, 5.10 / 0.22 | 58.2% | 12.6% | 24.5% / 26.6% |
| v4.3.4 (80M) | 99.3%, 5.52 / 0.04 | 96.4%, 5.20 / 0.20 | 60.4% | 12.5% | 24.4% / 24.6% |
| v4.3.5 (40M) | 95.5%, 5.02 / 0.23 | 93.5%, 4.73 / 0.33 | 62.7% | 11.8% | 23.5% / 25.0% |
| v4.3.6 (80M) | 98.6%, 5.40 / 0.08 | 95.9%, 5.26 / 0.23 | 65.2% | 10.2% | 29.2% / 24.1% |
| **v4.3.8 (40M)** | 97.7%, 5.32 / 0.13 | 95.9%, 5.25 / 0.22 | 69.6% | 8.7% | 29.4% / 22.5% |
| v4.3.8 (80M, two seeds) | 97.2%, 4.68 / 0.13 | 95.3%, 4.42 / 0.22 | 61.1% | 11.8% | 21.9% / 28.1% |
| v4.3.9 (40M) | 95.2%, 4.62 / 0.23 | 95.6%, 4.55 / 0.21 | 58.7% | 13.5% | 22.3% / 25.2% |
| **v4.3.10 (40M)** | 97.9%, 4.76 / 0.10 | 96.9%, 5.08 / 0.16 | 70.4% | 8.0% | 35.3% / 18.5% |
| v4.3.11 (40M) | 98.4%, 4.77 / 0.08 | 95.9%, 4.73 / 0.20 | 63.6% | 10.7% | 27.1% / 26.3% |
| v4.3.10 (80M, two seeds) | 97.5%, 4.99 / 0.13 | 95.2%, 5.61 / 0.28 | 66.6% | 6.7% | 31.9% / 17.6% |
| v4.3.12 (20M, three seeds) | 98.7%, 5.70 / 0.08 | 97.5%, 5.60 / 0.15 | 67.2% | 9.5% | 30.9% / 21.7% |
| v4.3.12 (40M, two seeds) | 97.9%, 5.13 / 0.11 | 95.9%, 5.13 / 0.22 | 67.8% | 9.5% | 28.8% / 21.6% |
| **v4.3.12 (60M)**, the one registered | 98.6%, 5.03 / 0.07 | 96.9%, 5.66 / 0.18 | 71.4% | 7.3% | 37.5% / 18.6% |
| v4.3.12 (80M, two seeds) | 97.7%, 5.36 / 0.12 | 96.7%, 5.71 / 0.19 | 69.8% | 7.7% | 31.1% / 19.9% |
| v4.3.12 (90M, the last file, two seeds) | 98.0%, 5.24 / 0.11 | 96.5%, 5.47 / 0.20 | 69.5% | 7.0% | 36.1% / 19.6% |
| v4.3.13 (21M, two seeds) | 98.3%, 4.88 / 0.09 | 96.8%, 5.76 / 0.19 | 71.5% | 6.8% | 34.8% / 19.5% |
| v4.3.13 (62M, two seeds) | 99.0%, 5.10 / 0.05 | 97.0%, 5.76 / 0.18 | 70.5% | 6.9% | 35.5% / 16.9% |
| v4.3.13 (96M, the last file, two seeds) | 98.3%, 5.35 / 0.09 | 97.7%, 5.84 / 0.14 | 69.8% | 6.5% | 33.3% / 17.7% |

Read it this way:

- Training with a compliant mallet (every v4.3 row) costs nothing in the
  rigid sim's goal share and takes 6-15 goals off the 35 conceded per 128
  minutes in the calibrated one (v4.3.5 excepted). It scores somewhat less per minute in the
  rigid sim (5.2-5.7 against 6.1).
- The possession columns moved only once the drills covered what the table
  shows: pucks leaving at 0.6-2.4 m/s (v4.3.8 on), with the block-and-play-on
  defense drill and the slow-exit cost. Costs alone (v4.3.4) and softer
  credit for quick shots (v4.3.5) did not.
- No model takes control of a puck moving away: 0-1 of 128 in the recovery
  drill for every row. They strike it from behind (73-93 of 128 on target in
  the calibrated sim for the bold rows, against 66-70 for v4.1.2). The
  rebound off a block is also softer (median 1.61-1.73 m/s against 1.88).
  v4.3.12 recovers 53% of rebounds at 1-1.5 m/s and 27% at 1.5-2 m/s;
  v4.1.2 41% and 8%; the table's v4.1.2 28% and 19%.
- Snapshots of a small-batch run differ more than arms do. v4.3.8 at 80M
  and its continuations v4.3.9 and v4.3.11 lost what the 40M snapshot had.
  The run with three times the batch (v4.3.12, 655k samples an update) kept
  it at every snapshot and improved on it. The registry points at snapshots
  (`"file"`): v4.3.8 at 40M, v4.3.10 at 40M, v4.3.12 at 60M.
- Robot against robot (64 games at both ends, calibrated sim) v4.3.8 and
  v4.3.10 are behind v4.1.2 (shots on target 926 against 982 and 904 against
  1029, goals 15-21 and 19-23); v4.3.12 at 60M is level (1022 against 1028,
  19-18). Its 20M snapshot was behind too (814 against 968).
- Gates beside v4.1.2 (calibrated sim), v4.3.8 / v4.3.10 / v4.3.12 at 60M:
  playability about equal (load 0.85 / 0.84 / 0.82 against 0.84; reversals
  47 / 51 / 49 a minute against 46; cap above 40 m/s^2 in 28 / 34 / 28% of
  far-half ticks against 23%; forward excursion on shots of 5 m/s or more
  160 / 140 / 131 mm against 150). Resting pucks: stall map 0.9% / 1.6% /
  0.4% against 1.0%; worst corner cell touched 41 / 42 / 41 of 48 against
  47 (left corner with a straight request for the first two, right corner
  with a left-bank request for v4.3.12); rest spells of 4 s or more in 64
  games: 0 / 1 (8.5 s, on a rail) / 0 against 0.

### Not done, and what is not known

- The table. Everything above is the sim, and the calibration is from
  sessions before the camera fix.
- Taking control of a puck moving away. It needs more than a bonus: the
  network has a reliable answer (strike from behind) that a compliant mallet
  makes less accurate, not useless. A teacher that goes round the puck and
  stops it, labelled the way the physics teacher labels defense, is the
  obvious next thing; so is giving the network the camera's view of its own
  mallet, which is where the error it cannot see comes from.
- The larger batch came late: 76 minutes at three times (v4.3.12), then 48
  at six times (v4.3.13, which held v4.3.12's level without adding to it).
  It is the one setting that gave steady snapshots; the next run should
  start there. v4.3.13 has two seeds and no gates or head-to-head.
- Self-play ideas not tried: v4.1.2 as a fixed reference opponent
  (`--opponent-reference`), a league of the v4.3 snapshots.
- Roaming difficulty models (above).

## Round 6: mostly self-play, a game pays goals and physical costs only (2026-11-19, v4.4.0)

Asked after round 5: push toward self-play that learns from first principles
rather than being guided, at least half self-play, the reward mostly goals
for and against plus the physical costs; some simulated people and drills
may stay. And a correction to how round 5 was judged: goals per minute is
NOT the measure. Wanted is a robot that takes the time to gain full control
of the puck and then makes a fast shot on target, every time, instead of
rushing every shot.

### What training had been (v4.1.2 to v4.3.13)

30% of the slots were full games, 70% drills. Game time by far-end player,
measured on v4.3.12's run: simulated people 30%, the scripted physics
teacher 30%, two fixed older networks 26%, the learner's own snapshots 14%.
So self-play was about 4% of all experience. In a game the learner was paid
600 a goal and, on top, for shots by speed and aim, the requested route,
captures, conversions and a setup potential; from round 5 also charged for
slow exits. The actor was pulled toward the teacher's action on defense
states (weight 0.2) and toward two imitation anchors on resting pucks.

Two of those terms push toward a quick shot: a rushed shot at a moving puck
earns the same shot income as a prepared one (`--fallback-shot-scale 1`),
and a puck that has lain within reach for 1 s costs 40 a second
(`--dead-puck-weight 40 --dead-puck-grace 1`).

### Shot quality, the measure from here on

From the recorded games against the `table` / `table_plus` people in the
calibrated sim (three seeds, 32 minutes each). A shot is the puck leaving
the half at 2 m/s or more within 0.7 s of a touch; "controlled" is the puck
under 0.5 m/s just before the striking touch; on target is
`shot_flight.first_goal_crossing` at the centre line.

| | v4.1.2 | v4.3.12 |
|---|---|---|
| shots struck from a controlled puck | 16% | 26% |
| first touch to shot, median | 0.61 s | 0.75 s |
| shots on target: all / from a controlled puck / at a moving puck | 51% / 53% / 51% | 67% / 78% / 63% |
| speed of on-target shots, median | 6.0 m/s | 6.2 m/s |
| per 100 possessions: controlled, on target and 4 m/s or more | 5.1 | 13.9 |
| per 100 possessions: any on-target shot of 4 m/s or more | 28 | 44 |

The complaint in numbers: three shots in four are struck at a moving puck
within three quarters of a second of the first touch, and under a soft
mallet those are the less accurate ones (63% against 78% for v4.3.12).

### Arm `selfplay` (`finetune.py --arm selfplay`)

- 80% of the slots are full games. No fixed older networks and no teacher
  at the far end: three quarters of the games are against the pool of the
  learner's own snapshots (6, one added every 50 updates), a quarter against
  the simulated people. Self-play is 60% of all experience, people 20%.
- `--game-shaping-scale 0` (new): in a full game the event income (shots,
  captures, conversions, requests, slow-exit and unproductive-return costs,
  both seats), the passive-return cost and the setup / control potentials
  are scaled out. A game pays +-600 a goal, the physical costs (motor load,
  calm accel, accel duty and effort, reversals, far cap, guard, idle speed,
  edge dwell, the 12000 shutdown) and the rules that stand in for a referee
  the table does not have: possession delay after 4 s, turnover, and the
  dead-puck cost, whose grace goes from 1 s to 3 s.
- No imitation: teacher weight 0, no resting-puck anchors.
- The other 20% of the slots are v4.3.12's drills with their own rewards
  (chase drills, defense that plays on after the block, stationary strikes).
- From v4.3.12 (its 60M snapshot), in the calibrated soft-mallet sim, 5120
  envs on 16 workers (655k samples an update), the critic alone for the
  first 15 updates because the reward under it changed.
- Not from first principles in the strict sense: it starts from a network
  that was guided for a billion steps, and the far seat's mallet is rigid.

### What the first two launches showed: the two ends were not equal

`v4.4.0` (63M steps, stopped) was the arm as first written. Its 50M snapshot
played like v4.3.12, as expected that early, but the trainer's own counters
said self-play was no contest: against its own snapshots the learner scored
0.06 goals a minute and conceded 0.73. Two things were one-sided, both left
from recipes in which the far side was mostly not a copy of the learner:

- The soft mallet was the learner's only (round 5). In the round-5
  head-to-heads the far seat put 40-50% more shots on target than the near
  seat whichever model sat there. `mallet_compliance far_body=true` now gives
  a robot-bodied far side the same mallet. Alone it changed nothing in the
  goal counts (0.07 / 0.70).
- The mallet lift (a 6-8 m/s puck slipping under a braking mallet) has been
  the learner's only since arm a2 (`mallet_lift far_body=false`, so the
  learner could not farm goals under the far robot's mallet). The counters:
  0.71 lifts a minute for the learner, none for its copy. That is where the
  0.7 goals a minute came from. `--no-selfplay-lift` (new) takes the
  learner's lift out of full games against a network; it keeps it against
  the simulated people and in the drills, where it models the table.

`v4.4.1` has both. Against its own snapshots, first 3,300 game-minutes:
0.074 goals a minute scored, 0.122 conceded. So the plain fact about
self-play with this robot: two copies score on each other about once in ten
minutes. Per update that is about 10 goals for and 16 against from the
self-play games, and about 220 scored against the simulated people, who
take a quarter of the games. Most of the goal signal still comes from the
people, although self-play is 60% of the experience.

### v4.4.1 over 270M steps: paid only for goals, it aimed at the sim's goal, which was too wide

Stopped at 271M of a planned 800M steps. Shot quality by snapshot (calibrated
sim, the measures above; v4.3.12 is three seeds, 101M and 252M two, the
others one):

| | v4.3.12 | 101M | 151M | 202M | 252M |
|---|---|---|---|---|---|
| shots struck from a controlled puck | 26% | 30% | 26% | 26% | 24% |
| first touch to shot, median | 0.75 s | 0.77 s | 0.78 s | 0.78 s | 0.77 s |
| shots on target | 67% | 63% | 61% | 54% | 52% |
| of them: from a controlled puck / at a moving puck | 78% / 63% | 78% / 57% | 75% / 56% | 60% / 52% | 66% / 47% |
| goals per 100 shots | 34 | 39 | 36 | 37 | 35 |
| per 100 possessions: controlled, on target and 4 m/s or more | 13.9 | 16.9 | 14.3 | 11.8 | 11.1 |
| puck leaves slowly with the robot not playing it | 7.3% | 4.5% | 5.5% | 5.2% | 4.2% |
| goals conceded per minute | 0.18 | 0.16 | 0.20 | 0.20 | 0.21 |

Against its own snapshots the learner's scoring went from 0.08 to about 0.10
goals a minute in the first 100M steps and stayed there (conceded 0.12-0.18);
against the simulated people 5.0 to 5.4 a minute.

So: at 101M a little more of what was wanted and fewer pucks given away, and
after that the aim got worse while the goals per shot did not. Where each
shot was heading when it crossed the centre line, as a distance from the
centre of the goal (mouth 379 mm, puck radius 40.7 mm):

| | fits between the posts (< 149 mm) | post band (149-189 mm) | wide | goes straight in: fits / post band / wide |
|---|---|---|---|---|
| v4.1.2 | 53% | 9% | 38% | 31% / 30% / 4% |
| v4.3.12 | 69% | 7% | 24% | 28% / 43% / 3% |
| v4.4.1 at 101M | 66% | 10% | 24% | 32% / 40% / 4% |
| at 151M | 63% | 10% | 26% | 28% / 41% / 5% |
| at 202M | 56% | 9% | 35% | 37% / 41% / 3% |
| at 252M | 55% | 12% | 34% | 34% / 38% / 3% |

The shots moved outward (mean distance from the goal's centre of the shots
inside the mouth 70 -> 87 mm), toward a band where the sim paid best and
where, on the table, the post is.

### The sim's goal was a puck diameter too wide (`table-20261119-posts`)

`batch_physics` counts a goal when the puck's CENTRE passes the end line
inside the mouth, and skips the end-rail bounce on the same test. A puck
whose centre is within a radius of the mouth's end therefore scores, though
it does not fit: the sim's goal is 379 mm wide, the slot a puck passes
through cleanly 298 mm. Rounds 1 to 5 paid the aim bonus on
`first_goal_crossing`, which uses the 298 mm, so those networks aimed inside
it. A game that pays only goals has no such term.

The table (34 sessions, 2026-09-20 to 11-18, 200 Hz puck track; every puck
entering the last 140 mm before an end rail at 1.5 m/s or more on a straight
line, at either end; "in" = not seen again for 0.7 s; rows with the
defender's mallet more than 150 mm from the arrival point):

| arrival, from the goal's centre | n | counted in |
|---|---|---|
| 0-80 mm | 146 | 71% |
| 80-149 mm (still fits) | 100 | 67% |
| 149-170 mm (inner half of the post band) | 29 | 41% |
| 170-190 mm (outer half) | 26 | 23% |
| 190-210 mm | 36 | 11% |
| beyond 210 mm | 349 | 7% |

The method misses about a quarter of clean goals and counts 7% of wide pucks
(deflections, tracker losses), so read the rows against each other: relative
to a clean arrival the inner half of the post band goes in 0.55 times as
often and the outer half 0.26 times. In the sim both were 1.

Model (`engine.goal_posts`, `BatchPhysicsEngine._collide_posts`, opt-in, off
by default): the four mouth ends are corners of the end rail. A puck whose
centre is inside the mouth but within a radius of a corner collides with it,
rail restitution along the corner-to-centre line and the end rail's
tangential retention across it, swept over the substep so a fast puck cannot
step over the contact. A straight shot is turned into the goal when it
arrives within 159 mm of the centre and is sent back beyond that; an angled
shot needs more room at the post it passes. No fitted parameter. Profile
`table-20261119-posts` = `table-20261119` plus the part `posts`; every other
profile and the default sim are unchanged. Tests in
`tests/test_table_fidelity.py`. Not in `physics.py` (the web UI's manual
sim).

The same models in the sim with posts (one seed):

| | v4.1.2 | v4.3.12 | v4.4.1 at 101M | at 252M |
|---|---|---|---|---|
| goals per 100 shots | 26 | 27 | 31 | 31 |
| shots on target | 50% | 64% | 64% | 58% |
| per 100 possessions: controlled, on target and 4 m/s or more | 6.2 | 12.6 | 19.7 | 11.8 |
| goal share against the simulated people | 93.3% | 97.4% | 98.4% | 96.8% |

Goal share barely moves, because the people's shots meet the same posts.
Every earlier goals-per-shot number in this file is about a fifth too high.

`v4.4.2` is arm `selfplay` again from v4.4.1's 101M snapshot under
`--sim-profile table-20261119-posts`, otherwise unchanged.

What this run does NOT show is the patient shot. First touch to shot stayed
at 0.77 s throughout, and the incentives explain it: the simulated people
concede to about a third of all shots however they are struck and score
0.2-0.3 a minute themselves, so a rushed shot that misses costs nothing; and
the learner's own copy lets in one shot in a few hundred. A reward of goals
alone, in this sim, pays for shooting often.

### Does a prepared shot pay? Not established (corrected 2026-11-19)

An earlier version of this section said a prepared shot scored 1.4 times as
often as a rushed one on the table and no more often in the sim, and blamed
the simulated people's defence. That rested on three November sessions and
82 simulated shots. With the October sessions and 128 recorded games it does
not hold.

Robot shots of 4 m/s or more at the centre line. "Controlled" = the puck was
under 0.5 m/s before the striking touch. A table goal = a puck seen going in
at the person's end within 2 s (a detector that misses some). "On target" =
the track at the line, traced through the rails, fits between the posts.

| | n controlled / moving | shot goes in | on-target shot goes in |
|---|---|---|---|
| table, October: v4 against the two people `table` / `table_plus` are modelled on (10-19, 10-29) | 50 / 95 | 20% / 24% | 23% / 27% |
| table, November: v4.1.2 (11-10, 11-17, 11-18) | 50 / 164 | 42% / 31% | 53% / 39% |
| sim: v4.1.2 against `table` / `table_plus`, posts, 128 games | 322 / 1499 | 24% / 24% | 33% / 32% |

An on-target fast shot went in 27% of the time on 10-19 (n 51), 24% on 10-29
(90), 57% on 11-10 (40), 35% on 11-17 (83) and 43% on 11-18 (23). So:

- The simulated people defend like the two October people they were built
  from (25% on the table, 32% in the sim). Their defence is not too strong.
- The November sessions were defended more loosely, and it is only there
  that a prepared shot did better than a rushed one. The logs do not say who
  played or how seriously.
- In the sim a prepared shot and a rushed one score equally often. The
  earlier "16% against 25%" was 82 shots.

Per possession the robot touched (puck arriving at 1.5 m/s or more;
"prepared" = the last touch was on a puck under 0.5 m/s after an earlier
touch):

| | prepared: n, leaves as a shot >= 4 m/s, scores | struck moving: n, shot, scores |
|---|---|---|
| table, October (v4) | 41, 88%, 27% | 172, 41%, 16% |
| table, November (v4.1.2) | 69, 51%, 25% | 313, 49%, 20% |
| sim (v4.1.2) | 100, 72%, 22% | 678, 48%, 21% |

Pooled, a prepared possession scored 26% of the time on the table against
19% (110 and 485 possessions; 1.4 standard errors), and 22% against 21% in
the sim. A difference between table and sim is not established here either.

Where the defender stands when the robot strikes (sideways from the goal's
centre, median of the absolute value; mean): 10-19 72 mm (-66), 10-29 34 mm
(+6), 11-10 76 mm (-66), 11-17 46 mm (+1), 11-18 48 mm (-46); sim 35 mm (0).
In three of the five groups the person stood 46-66 mm to one side.
`HumanTier.guard_scatter_m` (opt-in, no tier uses it; test in
`tests/test_human_proxy.py`) gives a simulated person a new standing place
each time the puck goes over to the robot. Trials, v4.1.2, 128 games each,
on-target fast shots that go in, controlled / moving:

| defence | controlled / moving |
|---|---|
| `table` / `table_plus` as they are | 33% / 32% |
| `guard_scatter_m` (0.05, 0.085) | 29% / 36% |
| the same, `anticipation` 0 (no read of the wind-up) | 35% / 40% |
| the same, `anticipation` (0, 0.3) | 44% / 38% |

The last is within the November sessions' 53% / 39% (n 36 / 110). It is a
description of looser defending, not a correction of the existing people.

What follows: goals alone do not teach this robot to prepare its shots, and
the table does not show that they should. If the patient style is wanted it
has to be paid for directly, or looser defenders added as a further kind of
person once it is known who the November sessions were.

### v4.4.2 to 252M steps: the posts delay the aim drift, they do not stop it

Posts sim, against the simulated people; two seeds at 101M and 252M, one at
151M and 202M (steps since the restart from v4.4.1's 101M snapshot):

| | v4.3.12 | v4.4.1 at 101M | v4.4.2 at 101M | 151M | 202M | 252M |
|---|---|---|---|---|---|---|
| shots that fit between the posts | 65% | 66% | 67% | 65% | 57% | 61% |
| wide misses | 28% | 26% | 27% | 28% | 33% | 31% |
| shots struck from a controlled puck | 23% | 31% | 30% | 29% | 28% | 27% |
| first touch to shot, median | 0.74 s | 0.78 s | 0.75 s | 0.76 s | 0.68 s | 0.72 s |
| per 100 possessions: controlled, on target and 4 m/s or more | 11.8 | 18.0 | 16.8 | 16.1 | 14.0 | 14.2 |
| goals per 100 shots | 30 | 31 | 31 | 29 | 29 | 32 |
| puck given away slowly | 7.8% | 4.2% | 4.0% | 4.5% | 4.7% | 2.9% |
| goals conceded per minute | 0.15 | 0.16 | 0.14 | 0.23 | 0.19 | 0.20 |

Goals per shot do not move while the aim does: against these defenders an
on-target shot 75-149 mm from the centre goes in 39% of the time and one
nearer the centre 28% (table, October: 28% and 24%), so a wider aim that
misses more often can cost nothing in goals.

The 303M snapshot (one seed) is the 252M one again: 62% between the posts,
14.5 per 100 possessions. `v4.4.2` is registered as its 101M snapshot and
the run was stopped there.

### v4.4.3 / v4.4.4: one payment in games for a prepared shot

`train_neural_player.py --game-prepared-shot-weight W` (opt-in; test in
`tests/test_table_training.py`): in full-game slots only, for the learner
only and outside `--game-shaping-scale`, W for a shot the learner strikes
from a CAPTURED puck (slower than 0.65 m/s at its mallet for 0.12 s and not
let go since) that is on target, times a speed ramp from 0 at 2 m/s to 1 at
6 m/s. A shot at a moving puck earns none of it. Everything else is the
`selfplay` arm in the posts sim:

    finetune.py --run-name v4.4.3 --arm selfplay --resume runs/v4.4.2/agent_step_1356070912.pt \
        --sim-profile table-20261119-posts --value-warmup 15 --n-envs 5120 --workers 16 \
        --trainer-arg=--save-every --trainer-arg=50000000 \
        --trainer-arg=--game-prepared-shot-weight --trainer-arg=100

Why a payment at all: see the two sections above. Goals alone leave the
share of prepared shots where it is, and that is what was asked to change.

`v4.4.3` (W = 100, a sixth of a goal; stopped at 111M steps). In its
training games prepared shots went from 31% to 33% of on-target shots
between 36M and 111M steps (4.4 -> 4.7 a minute, captures 6.6 -> 7.1; v4.4.2
stayed at 29-31% throughout). Against the simulated people, two
seeds, nothing measurable: shots from a controlled puck 30% (v4.4.2 30%),
first touch to shot 0.77 s (0.75), controlled + on target + 4 m/s per 100
possessions 17.1 (16.8). Shots between the posts 60% (67%), as v4.4.2
itself had 100M steps later. The size fits: against a person a shot is
worth about 0.3 x 600 in goals however it is struck, so 100 more for a
prepared one is small there, and it is only against its own copy (0.1 goals
a minute) that the payment is the larger income.

`v4.4.4` = the same from v4.4.3's 101M snapshot with W = 300 (half a goal,
more than a shot's expected goals against a person). At 101M steps, two
seeds, posts sim, against the simulated people:

| | v4.3.12 | v4.4.2 | v4.4.4 |
|---|---|---|---|
| shots struck from a controlled puck | 23% | 30% | 40% |
| first touch to shot, median | 0.74 s | 0.75 s | 0.85 s |
| shots between the posts | 65% | 67% | 66% |
| per 100 possessions: controlled, on target and 4 m/s or more | 11.8 | 16.8 | 25.3 |
| possessions ending in a shot | 71% | 75% | 83% |
| puck given away slowly | 7.8% | 4.0% | 2.3% |
| goals for / against per minute | 5.0 / 0.15 | 6.0 / 0.14 | 6.8 / 0.27 |
| goal share | 97.0% | 97.7% | 96.2% |
| head to head against v4.1.2, 64 one-minute games, both ends | 12-8 | 9-16 | 8-17 |

The payment does what it is for. The cost is goals conceded: 68 against 36
in 256 game-minutes, most of the difference straight shots (41 against 16)
taken with the mallet back on guard and no touch in the last second
(conceded on 3.3% of the people's shots of 3 m/s or more then, 0.3% for
v4.4.2); its mallet also waits further out (365 mm from its goal line when
the person's shot crosses the centre line, 304 mm for v4.4.2). No game
reached the load shutdown level (peak 0.935).

THE SELF-PLAY ARM LOSES TO v4.1.2 HEAD TO HEAD, payment or not (9-16 and
8-17 where v4.3.12 wins 12-8; small counts, the same seed). In this arm the
only robots the learner meets are its six latest snapshots; v4.3.12 had
fixed older networks in a quarter of its slots. `v4.4.5` = v4.4.4 from its
101M snapshot plus `--opponent-reference runs/v4.1.2/agent.pt
--opponent-reference runs/v4.3.12/agent_step_1091305472.pt`: a quarter of
the games against those two, a quarter against people, half against its
own snapshots.

`v4.4.5` at 101M steps, two seeds: the head to head against v4.1.2 is level
(20-18; it scores 20 where v4.4.4 scored 8 and concedes the same), the
prepared shots are kept (39% of shots from a controlled puck, 25.4 per 100
possessions, 70% of shots between the posts), and it still concedes 0.30 a
minute to the simulated people.

WHERE THE CONCEDED GOALS COME FROM: THE MALLET LIFT. Calibrated sim with
posts, against the simulated people, 256 game-minutes each (v4.1.2: 128):

| | lift events / min | conceded / min | through a lift | otherwise |
|---|---|---|---|---|
| v4.1.2 | 0.59 | 0.32 | 0.23 | 0.09 |
| v4.3.12 | 0.29 | 0.15 | 0.10 | 0.05 |
| v4.4.2 | 0.23 | 0.14 | 0.09 | 0.05 |
| v4.4.4 | 0.46 | 0.27 | 0.22 | 0.05 |
| v4.4.5 | 0.38 | 0.30 | 0.18 | 0.12 |

Taking control of a fast shot means braking under it, which is what lets a
6-8 m/s puck through. Every goal v4.4.4 conceded in its recorded games (24)
was a puck the mallet was on (19 mm to the side, median). Apart from the
lift v4.4.4 defends as v4.4.2 does. And the `selfplay` arm had switched the
learner's lift off in games against a network (`--no-selfplay-lift`, for
equal ends), three quarters of its games, so nothing there told it that
braking under a fast puck costs goals. `v4.4.6` = v4.4.5 from its 101M
snapshot with `--selfplay-lift`: the learner's mallet lifts in every game,
the far robot's still does not (people's mallets do not lift, so goals
through a far robot's lift are not worth learning).

`v4.4.6`, two seeds per snapshot, against the simulated people in the posts
sim:

| | v4.3.12 | v4.4.2 | v4.4.5 | v4.4.6 at 101M | at 202M |
|---|---|---|---|---|---|
| shots struck from a controlled puck | 23% | 30% | 39% | 43% | 37% |
| first touch to shot, median | 0.74 s | 0.75 s | 0.82 s | 0.92 s | 0.83 s |
| shots between the posts | 65% | 67% | 70% | 74% | 75% |
| shots from a controlled puck that are on target | 73% | 74% | 79% | 86% | 87% |
| per 100 possessions: controlled, on target and 4 m/s or more | 11.8 | 16.8 | 25.4 | 31.5 | 27.8 |
| per 100 possessions: any on-target shot of 4 m/s or more | 41 | 45 | 53 | 58 | 60 |
| goals for / against per minute | 5.0 / 0.15 | 6.0 / 0.14 | 6.4 / 0.30 | 6.3 / 0.22 | 6.2 / 0.23 |
| conceded through a lift / otherwise | 0.10 / 0.05 | 0.09 / 0.05 | 0.18 / 0.12 | 0.13 / 0.09 | 0.13 / 0.10 |
| head to head against v4.1.2 | 12-8 | 9-16 | 20-18 | 8-10 | 22-25 |

In training its lift events fell from 0.76 to 0.61 a minute and the goals
its own snapshots score on it from 1.07 to 0.91 a minute. Playability gates
at 101M beside v4.1.2 (soft-mallet sim): calmer on every line (idle cap p50
6.6 against 8.4 m/s^2, |accel| > 60 duty 3.2% against 5.7%, rms accel 23
against 27, reversals 25 against 46 a minute, load peak 0.82 against 0.84).

TWO THINGS KEEP IT OFF THE TABLE:

- Resting pucks. Stalled share of 1,632 placed pucks: v4.1.2 1.0%, v4.3.12
  0.4%, v4.4.2 and v4.4.4 0.4%, v4.4.5 1.0%, v4.4.6 5.3% at 101M and 4.0% at
  202M, nearly all in the row at the front edge of reach (y 0.85: a third to
  all of the trials at each x) and on the forward side rails. In the
  recorded trials the mallet lines up about 0.35 m behind the puck and
  dithers there for the ten seconds without going forward of y 0.55. It
  came in with the lift in every game; the arm has no drill for those
  states (`--reach-edge-drill-fraction` 0, an arm-a4 option never trained).
  At 101M one game had a puck lie 59 s.
- Load. 3 of 256 games in the RIGID sim reach the 0.95 shutdown level at
  each snapshot, for 14-18 s each (v4.1.2 1 of 128; v4.3.12, v4.4.2, v4.4.5
  none), though the typical game peaks lower than v4.1.2's (0.76-0.78
  against 0.80). None of 256 in the soft-mallet sim it trained in. It is
  the same thing: every one of those games ended 0-0 or 1-0 where the mean
  is 5.5 goals a game, i.e. a puck lay all minute and the mallet dithered
  behind it (the evaluator's `reachable_stall_seconds`: 58-126 s per 64
  games, v4.1.2 44-53 s). v4.1.2's one such game is the same.

`v4.4.7` = v4.4.6 from its 202M snapshot plus `--reach-edge-drill-fraction
0.3` (30% of the stationary drills: a puck at rest in the last 50 mm of
forward reach, or against a side rail forward of y 0.58).

### v4.4.7 at 101M steps: the drill works where it has got to

`runs/v4.4.7/agent_step_1961623552.pt`, two seeds, beside v4.4.6:

|  | v4.3.12 | v4.4.6 101M | v4.4.6 202M | v4.4.7 101M |
|---|---|---|---|---|
| shots from a controlled puck | 23% | 43% | 37% | 41% |
| first touch to shot, median | 0.74 s | 0.92 s | 0.83 s | 0.82 s |
| shots on target | 63% | 72% | 73% | 69% |
| controlled, on target, >= 4 m/s per 100 possessions | 11.8 | 31.5 | 27.8 | 30.4 |
| possessions ending in a shot | 71% | 84% | 87% | 90% |
| conceded per minute (soft-mallet / rigid) | 0.15 / 0.10 | 0.22 / 0.12 | 0.23 / 0.14 | 0.23 / 0.09 |
| head to head with v4.1.2 | 12-8 | 8-10 | 22-25 | 9-9 |
| resting pucks left (of 1,632) | 0.4% | 5.3% | 4.0% | 3.1% |
| ... in the forward zone | | 13.1% | 11.1% | 7.8% |

Shot quality is v4.4.6's. The stall map by cell (70 trials each; request in
brackets), v4.4.6 at 101M / 202M -> v4.4.7:

- forward RIGHT: 17 / 17% (left bank), 14 / 4% (right bank), 11 / 1%
  (straight) -> 3%, 0%, 0%. Fixed.
- forward LEFT: 20 / 17% (left bank), 3 / 26% (right bank), 13 / 1%
  (straight) -> 19%, 24%, 1%. Not moved.
- side rails: 11 / 9% (left bank), 23 / 13% (right bank), 9 / 7% (straight)
  -> 2%, 20%, 4%.

The drill places pucks on both sides and draws the request uniformly, so the
left side is a matter of steps, not of coverage. In games against the
simulated people the longest rest spell is 3.7 s (v4.4.6: one of 59 s).
3 of 256 rigid-sim games still touch the shutdown level (peak 0.96, no time
above it; v4.4.6: 1.01 and 6.6 s above). Playability beside v4.1.2 is calmer
on every line but one: the idle command moves more (p90 14 mm a tick against
4 mm). STILL NOT FOR THE TABLE; the run continues and the resting-puck gate
is run on every snapshot.

At 152M steps (`agent_step_2012086272.pt`, gates only): resting pucks left
2.9%. The forward zone went on down (7.8% -> 4.2%; forward-left with a bank
requested 19-24% -> 10-11%), the SIDE RAILS went up (8.3% -> 14.9%: left
bank 14%, right bank 20%, straight 11%), and two of 64 games against the
simulated people had a puck lie against the right rail at mid depth
(y 0.29-0.41) for 13 s and 26 s. Rail stalls have moved between 8% and 15%
from snapshot to snapshot on this line (v4.1.2: 5.9%) without a trend: the
front-edge drill does not reach them (its rail part is forward of y 0.58)
and the arm's edge drill is 15% of the stationary drills.

### v4.4.7 at 202M steps: the front edge is fixed; one rail puck is not

`runs/v4.4.7/agent_step_2062548992.pt` (the registered file), two seeds:

|  | v4.3.12 | v4.4.6 202M | v4.4.7 101M | v4.4.7 202M |
|---|---|---|---|---|
| shots from a controlled puck | 23% | 37% | 41% | 47% |
| first touch to shot, median | 0.74 s | 0.83 s | 0.82 s | 0.88 s |
| shots on target (controlled / moving) | 63% (73 / 60) | 73% (87 / 66) | 69% (83 / 59) | 75% (87 / 64) |
| shots that fit between the posts / wide | | | 70% / 23% | 76% / 18% |
| controlled, on target, >= 4 m/s per 100 possessions | 11.8 | 27.8 | 30.4 | 35.9 |
| possessions ending in a shot | 71% | 87% | 90% | 88% |
| conceded per minute (soft-mallet / rigid) | 0.15 / 0.10 | 0.23 / 0.14 | 0.23 / 0.09 | 0.22 / 0.07 |
| head to head with v4.1.2 | 12-8 | 22-25 | 9-9 | 18-19 |
| motor load peak, rigid sim (limit 0.95) | 0.93 | 1.01 | 0.96 | 0.92 |
| resting pucks left: all / forward zone / side rails | 0.4% | 4.0 / 11.1 / 9.5% | 3.1 / 7.8 / 8.3% | 1.3 / 0.4 / 7.7% |

Forward zone by snapshot: 13.1% (v4.4.6 101M), 11.1%, 7.8%, 4.2%, 0.4%: the
drill at 0.3 took 200M steps. v4.1.2's own stall map is 1.0 / 1.6 / 5.9%,
so the map is now level with it. Playability beside v4.1.2: calmer on every
line (idle cap p50 6.6 against 8.4 m/s^2, |accel| > 60 duty 3.2% against
5.7%, reversals 24 against 46 a minute, forward lunge on shots 91 against
150 mm) except that the idle command moves more (p90 16 mm a tick against
4 mm, at that low cap).

WHAT STILL FAILS: REST SPELLS IN GAMES. In 2 of 64 games against the
simulated people a puck came to rest against the RIGHT rail at mid depth
((0.96, 0.41) and (0.93, 0.42)) and lay there 55.5 s and 42.8 s, i.e. to the
end of the game; at 152M the same (13 s and 26 s at (0.92, 0.29) and
(0.95, 0.41)); at 101M none over 4 s. v4.1.2: none over 0.2 s, with a stall
map no better on the rails (rail cells 4-7% against 7-9%). So the placed
rail pucks of the stall map do not show it; it is a state that arises in
play.

What the spell is (recorded games, `eval_resting_puck.py --parts games
--record`): the robot's own dribble carries a slow puck to the right rail,
where it stops on the rail or 35 mm off it at y 0.41-0.42; the mallet then
stays level with it on the table's side, 10-25 mm short of touching
(mallet (0.85, 0.41) beside a puck at (0.957, 0.414)), creeping at
3-6 cm/s, for the remaining 43-56 s. It is the hover of the front-edge
pucks in another place. The edge drill has pucks there but starts the
mallet anywhere on the half.

`v4.4.8` = v4.4.7 from its 202M snapshot plus `--rail-rest-drill-fraction
0.35` (new, opt-in): 35% of the stationary drills have a puck at rest from
touching a side rail to 45 mm off it, anywhere up the rail below the front
edge, and the mallet half the time level with it 8-70 mm short of contact,
a quarter below it, a quarter anywhere. `--reach-edge-drill-fraction 0.3`
stays. v4.4.7 was stopped at 240M steps.

    PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py --run-name v4.4.8 --arm selfplay \
        --resume runs/v4.4.7/agent_step_2062548992.pt --sim-profile table-20261119-posts --value-warmup 8 \
        --n-envs 5120 --workers 16 --minutes 240 --steps 400000000 --seed 20261148 \
        --trainer-arg=--save-every --trainer-arg=50000000 \
        --trainer-arg=--game-prepared-shot-weight --trainer-arg=300 \
        --trainer-arg=--opponent-reference --trainer-arg=runs/v4.1.2/agent.pt \
        --trainer-arg=--opponent-reference --trainer-arg=runs/v4.3.12/agent_step_1091305472.pt \
        --trainer-arg=--selfplay-lift --trainer-arg=--reach-edge-drill-fraction --trainer-arg=0.3 \
        --trainer-arg=--rail-rest-drill-fraction --trainer-arg=0.35

### v4.4.8: the rail-rest drill made it worse (stopped at 130M steps)

| resting-puck gates | v4.4.7 202M | v4.4.8 50M | v4.4.8 101M |
|---|---|---|---|
| placed pucks left: all / forward zone / side rails | 1.3 / 0.4 / 7.7% | 2.6 / 4.7 / 12.5% | 5.1 / 10.9 / 19.1% |
| rest spells >= 4 s in 64 games | 2 (43 s, 56 s) | 0 (longest 3.8 s) | 2 (56 s, 59 s, right rail y 0.40) |

Shot quality unchanged (47% of shots from a controlled puck, 36 prepared
fast on-target shots per 100 possessions); head to head with v4.1.2 16-24;
conceded in the rigid sim 0.18 a minute (0.07). DO NOT USE
`--rail-rest-drill-fraction` AS BUILT. Why it failed is not established.
What the logs do show: in TRAINING the stationary drills end in an aimed
shot 91-95% of the time in v4.4.7 and v4.4.8 alike, and a puck lies dead
0.3% of game time, against 2.6% for the same network played without
exploration noise (the two spells). The noise (std 0.13, doubled on the
timing and effort outputs once a puck has lain 3 s) gets the training
policy out of the hover that the deployed mean stays in, so the drill's
states were mostly "solved" as far as the gradient could see, while the
drill took a third of the front-edge drill's share (20% -> 13% of the
stationary drills) and the forward zone came back. Next for the rail
spell: measure the deployed (noise-free) policy on each drill's own start
states before adding practice (`scratch drilldet.py`), not the training
counters.

### The table, 2026-11-19: the beginner defends better per shot

The user's observation after an evening against `max`, `new` and
`beginner`: the beginner "jitters a lot less and so doesn't ever jump /
randomly bounce the puck in"; wanted: hard acceleration discouraged unless
necessary, WITHOUT softer shots. The logs agree (straight shots only;
17-19 November, live sessions):

|  | v4.1.2 (40 min) | v4.3.12 (38 min) | v4.2.2 beginner, 15 m/s^2 (24 min) |
|---|---|---|---|
| straight shots reaching its end, per minute | 0.65 | 0.89 | 1.64 |
| of those, in | 19% (5) | 21% (7) | 8% (3) |
| goals that went in off or under its own mallet (mallet within touching range in the last 0.5 s) | 2 of 5 | 4 of 7 | 0 of 3 |
| conceded per minute | 0.13 | 0.18 | 0.13 |

Playability gates from the same evening's logs (`eval_playability.py
--from-table-log`), v4.3.12 / beginner: rms acceleration 24.8 / 7.3 m/s^2,
reversals 30 / 0 a minute, forward lunge on shots of 5 m/s or more 98 /
20 mm (p90 243 / 49), mallet speed at contact 1.36 / 0.78 m/s. v4.4.7 in
the sim is where v4.3.12 is on the table (lunge 91 mm, 1.54 m/s at
contact): the sim's mallet lift does not punish a moving mallet as hard as
the table does. (5 of v4.3.12's 7 goals fell in the minutes its loop was
late, see CLAUDE.md "REALTIME PRIORITY"; the split off-the-mallet / past-it
is the same in and out of them.)

`--defense-motion-scale S` (new, opt-in): the accel-duty, reversal and
accel-effort charges are multiplied by S on every tick that is not the
learner's own possession (a puck on its half under 2 m/s, or within 0.25 s
of its own contact). An incoming shot, the puck away and a volley at a
moving puck pay S times; the wind-up and strike at a puck it holds pay what
they did. `v4.4.9` = v4.4.7's arm from its 202M snapshot with S = 4 (a
lunge of 80 m/s^2 for 0.2 s costs about 60 against 15; a goal is 600).
Read with it: shot speed (must not fall), lunge and mallet speed at
contact, conceded and through a lift, the resting-puck gates.

    ... same command as v4.4.7 with --run-name v4.4.9 --resume runs/v4.4.7/agent_step_2062548992.pt --seed 20261149
        --trainer-arg=--defense-motion-scale --trainer-arg=4      (no --rail-rest-drill-fraction)
