# Policy improvement results

**Historical hybrid result — rejected in user review; not the selected playing agent.**
The checks below passed their declared thresholds, but those thresholds did not
establish good possession-to-shot behavior or meet the user's learned-policy
requirement. The replay showed passive play. A fresh single-network effort is
tracked in `NEURAL_PLAYER_PROGRESS.md`; this hybrid must not be presented as
completion of that objective.

**Original simulation qualification v3 results:**
Candidate: `runs/_foundation-dual-load-arrival-controller/agent.pt`.
It remains `deployment_ready: false`; physical defaults and hardware are unchanged.

## What changed

This is a hybrid of preserved learned motor skills and explicit arrival,
interception and load control. It gets behind a slow puck, commits to a contact
deadline, aims a 3 m/s direct shot and recovers early. Setup uses 10 m/s²,
arrival strokes 30 m/s², and urgent defense/recovery can use 60 m/s². The
interceptor chooses the cheapest forecast acceleration that can reach a block
and avoids chasing clear goal misses. Commands use the firmware's existing
motion law and a required external motion guard.

Fast and slow motor memories have independent cooldown states. Fast load
triggers at 0.88 and releases at 0.82; slow load triggers at 0.92 and releases
at 0.89. While cooling, attacks are suppressed but necessary defensive
acceleration remains available below separate reserve ceilings (0.91 fast,
0.94 slow). Above a ceiling, defense requests 8 m/s²; the motion guard can still
use stronger braking. Both cooldown states persist across point resets.
This avoids keeping attacks disabled for minutes after a fast-load event solely
because an untripped slow memory remains warm.

## Held-out results

Policy and thresholds were frozen before seeds 20261701–20261709 were opened.
Each skill row has 500 trials with randomized physics and realistic observations.
Shooting begins at arbitrary nonoverlapping paddle positions. Bank tests alternate
rails and aim near goal center; every launch was verified to score without a
defender under its actual trial physics. Those parameters never reach the policy.

| Task | Result |
| --- | ---: |
| Stationary goal within 2 s | 497/500 (99.4%) |
| Moving-puck goal within 2 s | 339/500 (67.8%) |
| Cushioning | 488/500 (97.6%) |
| Home defense, 2–8 m/s | 500/500 (100.0%) |
| Random-pose defense, 2–8 m/s | 478/500 (95.6%) |
| Fast direct defense, 8–12 m/s | 494/500 (98.8%) |
| Fast single-bank defense, 8–12 m/s launch | 497/500 (99.4%) |

| Against 3.12 using full MPC | Candidate score | Peak candidate load |
| --- | ---: | ---: |
| Ordinary, first side | 25–4 | 0.862 |
| Ordinary, opposite side | 30–7 | 0.842 |
| Hot start, first side | 3–1 | 0.904 |
| Hot start, opposite side | 7–1 | 0.900 |

Ordinary sets contain 8 × 180-second games each; hot-start sets contain 4 × 180
seconds each, with every candidate fast/slow motor memory initialized at 0.90.
Both bodies have the same instantaneous 12 m/s and 60 m/s² caps. The reference's
load is reported separately and is not itself qualified. Promotion requires a
positive score margin from each side, no overload, and peak candidate load below
0.95. The earlier candidates' failed margins and hot-start scores are preserved.

Four continuous one-hour self-play games completed with peak modeled load **0.860**, 0.00 seconds above the modeled load limit, and peak actual acceleration 60.0009 m/s² (checked against the 60 m/s² cap with numerical tolerance). Heat persists across points. Aggregate self-play score: 2–3. There were 4,796 forward returns and 650 referee turnovers across all games.

In ordinary reference games, actual acceleration exceeded 40 m/s² for
**2.79%** of candidate time versus **20.34%** for the reference.
A separate development stress test starting at 0.93 load won 10–3 and stayed
below that starting peak. It is additional evidence, not part of the held-out score.

## What remains weaker

Moving-puck setup remains the clearest weakness: 339/500 goal completions within
2 seconds, with any on-goal contact in 360/500 trials. Stationary trials had an
on-goal contact in 498/500 cases. These are trial outcomes, not the fraction of
all game contacts that are shots. Full-game return-direction and actual-goal
metrics are recorded separately.

The RMS model is provisional, fitted from one session with conservative priors.
Simulation results do not establish real drive safety or superhuman performance
against humans. Occasional unresolved guard forecasts are retained in the diagnostics; actual
acceleration limits are checked separately. The physical loader/42-feature adapter is not integrated or
qualified for this composite policy. No physical activation was performed.

## Training and rejected experiments

Behavior transfer retained the complete checkpoint. Longer imitation/PPO runs
and a fresh shot-intent selector were tested; the 4M-transition tactical
continuation did not beat fixed arrival control and is not selected. Compiling
the frozen actor mean improved that training run to roughly 3,300 transitions/s.
Checkpoints, optimizer/replay state, configurations and failed results remain.

An aimed defensive rebound controller improved isolated trials but regressed
bank defense. Restricting it to unbanked paths restored saves without a meaningful
full-game accuracy gain, so it was archived. Selection follows complete play and
load measurements rather than training loss or isolated favorable metrics.

## Review and reproduce

Replays are in the web UI at <http://localhost:8420>, named
`_foundation-dual-load-arrival-controller_heldout_*`. Self-play uses the same policy on both sides.
Replay clips show the first 30 seconds of the first game; qualification JSON
covers every complete game. Both policy identities are recorded in metadata.

Run this **simulation-only** command from the repository root with the system
Python that has CUDA Torch:

```sh
PYTHONPATH=ai:ai/bin python3 ai/bin/eval_foundation.py \
  runs/_foundation-dual-load-arrival-controller/agent.pt \
  --games 4 --seconds 180 --guard \
  --output logs/foundation/dual-load-selfplay-review.json \
  --record ai/recordings/dual-load-selfplay-review.json
```

Add `--rival runs/3.12-accel60-accuracy-selfplay/agent.pt --compile-mpc` for the
reference matchup. The loader automatically retains the saved motion-guard
requirement even if the flag is omitted.

Evidence: `runs/_foundation-dual-load-arrival-controller/qualification/` contains the manifest, source/configuration
snapshots, per-test metrics and final overview plot. The full experiment history
is in `ai/POLICY_IMPROVEMENT.md`. Twenty-one focused tests passed; the saved-policy
loader was separately smoke-tested to confirm automatic guard activation in sim.
