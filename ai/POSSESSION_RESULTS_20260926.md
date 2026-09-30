# September 26 selected possession and defense candidate

Selected package: `runs/possession-20260926-v2/agent.pt`.
Exact source: `runs/_neural-player-requests-possession63/agent_step_827736064.pt`.
This is a pinned checkpoint, not the latest experimental weights.

**The reset bug is fixed and defensive preparation improved substantially.
Deliberate control of slowly outgoing pucks remains unreliable.** Long continuous
self-play can also stall on reachable quiet pucks. This is a stronger qualified
candidate, not a claim that all three requested behaviors are solved.

[Open the selected three-minute self-play replay](http://localhost:8420/?replay=neural-possession-candidate-20260926-v2.json).
Both sides use exactly the selected neural network. The recording uses the
standard qualification seed and conditions; it was not chosen after searching
for a favorable game. Goal/dead-puck re-serves have explicit labels.

From the repository root, when the user starts a physical session:

```bash
bash ai/bin/play.sh --policy neural:possession-20260926-v2
```

Offline check, which does not start the camera or hardware master:

```bash
bash ai/bin/play.sh --policy neural:possession-20260926-v2 --check-policy
```

The final packaged weights passed that check: 1.28 ms median for inference,
arrival decoding and motion guarding, against a 20 ms control interval.
Session ceilings are 12 m/s and 60 m/s². The network chooses its arrival action
and effort; fresh drive load telemetry is used with persistent modeled fallback
for missing channels. This work did not activate hardware or change global
physical deployment defaults.

## What changed

- Continuous games remove the legacy three-second possession re-serve and
  short attended-puck timeout. Reachable quiet pucks remain playable. Only
  genuinely unreachable dead pucks are re-served after eight seconds.
  Receiving practice continues through control, a shot, or first possession loss.
- Defense practice includes time to choose a position before an unpredictable
  fast goal-directed launch. Coverage rewards and a training-only request
  consistency loss discourage exposed waiting positions and stale-request bias.
  There is no prescribed defensive home position in production.
- Slow outgoing practice uses easier starting states, collision-avoiding reward
  shaping and once-only progress, cushioning and control rewards. Conflicting
  shot-preservation examples were investigated. These changes did not establish
  reliable recovery; the selected checkpoint's small gain is reported below.
- Long games preserve heat across goals and episode boundaries, include cool
  and warm initial states, and vary opponents' shot styles. Using 512 parallel
  environments and running 48M additional transitions allowed individual
  games to reach 900 seconds. Extra exploration stays in short practice drills;
  full games use the learned variance. The selected snapshot is 32M into this run.

The actor remains one three-layer, 512-unit ELU network: 45 inputs, 6 arrival outputs,
50 Hz and the existing three-way requested-shot input. Task labels reach only the
training value network. No tactical planner, teacher, or cooldown controller is
attached to the deployed actor.

## Matched simulation comparison

Reference: previously reviewed reportdefense33, step 532103168. Both actors use
report sensing and continuous rallies for this comparison. The three requested
shot routes repeat 512 physical fixtures; the summed counts below are descriptive,
not 1,536 independent scenarios.

| Measurement | Reviewed reference | Selected candidate |
|---|---:|---:|
| Preparation saves before an 8–12 m/s direct shot | 809/2048 | 1744/2048 |
| Same preparation starting at 90% modeled load | 727/2048 | 1564/2048 |
| Ready-position direct saves | 1009/1024 | 1013/1024 |
| Varied-position bank saves | 787/1024 | 785/1024 |
| Hot varied-position bank saves | 774/1024 | 792/1024 |
| First requested on-target shot ≥6 m/s, stationary starts | 774/1536 | 806/1536 |
| First requested on-target shot ≥6 m/s, incoming 3–8 m/s | 940/1536 | 965/1536 |
| Controlled stationary puck → requested fast shot | 637/1536 | 588/1536 |
| Controlled incoming puck → requested fast shot | 918/1536 | 898/1536 |
| Strict outgoing recovery before first possession loss | 24/1536 | 32/1536 |
| Outgoing recovery → controlled requested fast shot | 15/1536 | 18/1536 |
| Reachable-puck stall time, eight 180-second games | 503.6 s | 124.0 s |

Preparation saves increased from 39.5% to 85.2%; hot preparation from 35.5% to 76.4%.
Reaction defense was preserved, rather than traded away for preparation.
Requested fast-shot success improved, but controlled-then-requested-fast shots
decreased in the stationary and incoming suites. The control tradeoff is retained
in the table rather than hidden behind the faster-shot metric.

Self-play goals were 47 versus 12 over eight 180-second games. This is not a
head-to-head win rate. Against the old actor across both colors, the selected
actor scored 29 and conceded 24 with mixed requests, and 17 versus 14 with the old
actor restricted to straight requests. These small suites do not establish
superhuman play. Strict outgoing control remains only 32/1536 request trials;
getting around and cushioning a departing puck is unfinished.

## Sustained load qualification

Each row is eight 900-second continuous self-play games at thermal gain 1.3.
Shutdown in this provisional normalized model is 1.0. Exact selected weights:

| Seed | Initial load | Peak modeled load | Overload seconds |
|---|---:|---:|---:|
| 20263545 | 0.95 | 0.974930 | 0 |
| 20265926 | 0.95 | 0.961298 | 0 |
| 20266026 | 0.30 | 0.983651 | 0 |
| 20269926 | 0.95 | 0.958838 | 0 |
| 20269927 | 0.30 | 0.966109 | 0 |

All five sets reported zero firmware-profile acceleration-cap violations.
Candidate loads also stayed below 1 in cross-play in both colors. Opponent loads
are attributed separately and are not mistaken for candidate overloads.

These are simulation checks, not a guarantee against physical RMS shutdown.
Long-game activity remains a limitation: the primary hot suite accumulated
3917.3 reachable-puck stall seconds over 7200 game seconds; the primary cool-start
suite accumulated 5362.2/7200 (74.5%). Short-game activity gains must not be
presented as uniformly active 15-minute self-play.

## Selection and reproducibility

Run 63's final 48M weights improved the shot metrics further but increased
short-game stalls and lost the small strict-recovery gain. Run 64 removed the
remaining near-contact outgoing-shot preservation examples; its modest recovery
gain did not justify promotion because an additional sustained cool-start audit
exceeded the modeled load threshold. The earlier 49 provisional package without
`-v2` is explicitly withheld; it is not this selection.

`runs/possession-20260926-v2/qualification.json` records all report paths,
checkpoint identity, SHA256, additional confirmation audits and selection reasons.
The primary reports are under
`logs/neural-player/requests/possession63/qualification-middle32m/`.
The five thermal report directories and rejected experiment records remain local.

209 relevant regression tests pass, including reset behavior, reward accounting,
exploration likelihoods, observation/deployment parity, physical simulation,
launcher isolation and candidate/opponent load attribution. UI JavaScript syntax
and diff-whitespace checks pass. The pinned replay and its API both return HTTP 200,
with 9000 frames and metadata matching the packaged checkpoint hash.

## Additional frozen-seed check

A separately declared fixture set (seeds20268001–20268005) confirms the strong
preparation result:1748/2048 saves,1601/2048 with hot starts; direct defense
1006/1024, varied banks793/1024, hot varied banks799/1024. Outgoing control remains
poor at54/1536 request trials,31 resulting in controlled requested fast shots.
Stationary/incoming requested fast shots were792/938 out of1536 each.

Eight180-second self-play games on this seed set scored29 total goals and
accumulated418.9/1440 reachable-puck stall seconds (29.1%). This is worse activity
than the original short suite and reinforces the long-game limitation. These
reports are in `logs/neural-player/requests/possession72/fresh-baseline/`; they
are additional evidence for this exact pinned package, not a comparison against
a newer policy evaluated on different seeds.
