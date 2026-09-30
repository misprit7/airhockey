# September 26 possession candidate v3

Package: `runs/possession-20260926-v3/agent.pt`.

**Known failed replay: the puck settles near the left side at about20s and stays
there through180s. It is contactable under the existing guarded travel limits;
a wall bounce can recover it. Passing the load/defense checks did not establish
adequate edge play. This policy should not be described as a finished player.
Run102 adds explicit side-fringe practice and outcome-based recovery rewards;
no improved checkpoint has yet been selected.**
Exact source: run97, step843792384.
SHA256: `a3cd692c80c4ca42f6f200475aa5ccf967793df27853d9770c56be265886c2dc`.

This candidate retains the possession-reset fix and stronger defensive
preparation, and improves deliberate recovery of departing pucks. Recovery is
still unreliable, and the stronger preference for control sacrifices many
immediate fast returns. It has not established a better overall win rate than v2.

[Three-minute self-play replay](http://localhost:8420/?replay=neural-possession-candidate-20260926-v3.json).
Both sides use these exact weights, with mixed requested shot routes. The
recording uses the standard qualification seed, not a selected favorable game.

From the repository root, for a user-started physical session:

```bash
bash ai/bin/play.sh --policy neural:possession-20260926-v3
```

Previous qualified package:

```bash
bash ai/bin/play.sh --policy neural:possession-20260926-v2
```

The offline `--check-policy` command passed: 2.01 ms median for inference,
decoding and motion guarding within the 20 ms control interval. Session caps
remain 12 m/s and 60 m/s². No physical hardware was activated during this work,
and physical deployment defaults were not changed.

## Measured behavior

The reviewed reference is reportdefense33. v2 was the earlier qualified
possession candidate from this work. All columns below use matching fixtures,
continuous rallies and report-based sensing. Shot totals repeat 512 physical
fixtures across three requests; they are not 1,536 independent physical starts.

| Measurement | Reviewed reference | v2 | v3 |
|---|---:|---:|---:|
| Preparation saves before an 8–12 m/s direct shot | 809/2048 | 1744/2048 | 1744/2048 |
| Same preparation from hot starts | 727/2048 | 1564/2048 | 1567/2048 |
| Ready-position direct saves | 1009/1024 | 1013/1024 | 1009/1024 |
| Varied-position bank saves | 787/1024 | 785/1024 | 783/1024 |
| Hot varied-position bank saves | 774/1024 | 792/1024 | 790/1024 |
| Requested on-target shot ≥6 m/s, stationary starts | 774/1536 | 806/1536 | 821/1536 |
| Same, incoming puck at 3–8 m/s | 940/1536 | 965/1536 | 971/1536 |
| Strict outgoing control before first possession loss | 24/1536 | 32/1536 | 161/1536 |
| Outgoing control → requested fast shot | 15/1536 | 18/1536 | 101/1536 |
| Reachable-puck stall time, eight 180-second games | 503.6 s | 124.0 s | 174.3 s |

Relative to v2, first requested fast shots in the outgoing recovery suite fell
from 652 to 217. Catching more pucks does not by itself demonstrate more useful
possessions. Strict control is now 10.5%, and remains a weakness.

Expanded head-to-head against v2 used 32 games of 180 seconds per color and
opponent style, seed20272803. Aggregating both colors:

| Requests | v3 goals : opponent goals | Matching v2 baseline |
|---|---:|---:|
| Mixed versus mixed | 112 : 123 | 104 : 104 |
| Mixed versus straight-only | 81 : 63 | 51 : 56 |

v3 scores more against the straight-only opponent, but also concedes more
(63 versus 56). This does not establish improved defense over v2. The earlier
eight-game comparison was 27:29 mixed and 14:15 against straight-only, with
matching baselines 15:15 and 15:6. Those results are retained alongside the larger
cohort. No claim of superhuman defense is justified by these tests.

## Training and architecture

There is one learned actor: three 1,024-unit ELU layers and six arrival outputs.
Inputs comprise 16 physical frames of 42 features, plus three shot-request
inputs (675 total). Past frames are encoded as differences from the current
frame inside the network. The original 512-unit prefix was retained while
additional capacity learned recovery. The training value network receives
privileged episode bookkeeping; the deployed actor does not.

Training combined PPO, a curriculum of outgoing recovery and defensive
preparation, and imitation of successful simulated trajectories. Offline search
supplied some training examples; no search, scripted interception, tactical
switch, or cooldown controller runs in the deployed policy.

Later recovery examples varied the opponent's position and load state. An
earlier fixed opponent position had allowed a spurious shortcut. Receiving
practice follows through to a shot or first possession loss. Rewards prefer a
controlled requested fast shot, while allowing an accurate fast fallback when
control fails; capture alone is not enough to earn a successful episode.

The selected run removes persistent recovery action offsets used in earlier
exploration and retains ordinary PPO sampling during training. Long games retain
heat across goals and include both hot and cold starts. Short drills, full games,
and both colors are evaluated separately. Newer runs were rejected when better
recovery came with overloads or shooting regressions.

A subsequent full-network experiment exposed oversized optimizer updates after
unfreezing. Optional `--backtrack-kl` now shortens an actor update that exceeds
the sampled minibatch KL budget. It preserves the independent critic step and
does not add any runtime action restriction. This optional training change is
not part of the selected run97 weights.

## Sustained loads and remaining stalls

Each audit is eight continuous 900-second self-play games at thermal gain1.3.
Modeled shutdown is normalized load1.0. All five have zero modeled overload
seconds and zero firmware-profile acceleration-cap violations.

| Seed | Initial load | Peak modeled load |
|---|---:|---:|
| 20263545 | 0.95 | 0.965505 |
| 20265926 | 0.95 | 0.961299 |
| 20266026 | 0.30 | 0.975387 |
| 20272601, fresh selection check | 0.95 | 0.962049 |
| 20272602, fresh selection check | 0.30 | 0.973017 |

Long-game stalls remain substantial: 4,307/7,200 seconds in the primary hot
suite, and 5,102/7,200 in the development cold suite. The two fresh suites had
2,523 and 2,505 stall seconds. These finite checks use a provisional load model
and cannot guarantee that physical RMS shutdowns will never occur.

## Reproduction

`runs/possession-20260926-v3/qualification.json` records checkpoint identity,
all qualification paths, additional load checks, selection reasons and limitations.
Full reports are under `logs/neural-player/requests/possession97/qualification-final/`.
The two head-to-head cohorts are under that run's `cross-pinned/` and
`cross-pinned-expanded/` directories. The previous package remains available.

219 relevant regression tests pass, including reset behavior, reward accounting,
observation/deployment parity, physical simulation, launcher isolation, player
attribution and optimizer backtracking. A real PPO smoke run exercised the new
optional optimizer check. The selected packaged policy passed its offline
deployment check without opening hardware.
