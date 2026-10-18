# Physics teacher → one neural network (2026-10-18 overnight)

Simulation only. No hardware calls, firmware changes or deployment-default
changes. The robot still runs exactly one learned network at the table; the
teacher below exists only to train and to benchmark it.

## Objective

A much stronger player against people on the physical table, under the same
physical limits as production `rail30-100-20261001-v1`: 12 m/s, 100 m/s²,
3 ms ramp, rail30 workspace, 50 Hz, 15 ms command delay, motion guard,
thermal recipe `motor-load-20261001` at gain 1.3 (0.95 training shutdown,
≥1.0 counted as overload in evaluation).

## Why this direction (evidence, not taste)

- Production blocks **27%** of the fast-attack benchmark (left banks 2%,
  right banks 9%). The 56-line `deep-bank` heuristic in
  `defense_baselines.py`, which sees the same delayed observations and drives
  the same decoder/guard/thermal model, blocks **84%**. The gap is
  positional: production waits 0.63 m forward, the heuristic 0.18 m.
- On the table (Oct 2) 6 of 7 concessions were single side-rail banks at
  3.7–6.5 m/s, with the paddle 0.54–0.72 m forward.
- Three weeks of reward shaping (depth costs 200/800, stricter capture,
  widening) moved the stance 0.63 → 0.53 m at best and cost shooting or the
  ladder each time. PPO here takes ~6 actor steps per 131k-transition
  rollout (KL stop at 0.02, lr 3e-6–1e-5); strong imitation anchors held the
  old stance in place.
- In August a hand-written striker beat its own learned clone 0.80/0.06 vs
  0.15/0.90 goals for/against per game: the clone failed by distribution
  shift. DAgger (the teacher labels the student's own states) is the fix
  for that failure mode, and it already worked here once (modern26 took
  shot routes from 0/64 to ~89% each, Sep 23).

## Plan

1. **Teacher** `ai/airhockey/expert_player.py` (`ExpertPlayer`): same
   16-frame observation and 6-D arrival action as the network; forecasts
   the puck with drag and lossy rails; blocks on the line with the most
   time margin; strikes on the requested route (or away from the rival
   mallet); guards on the puck–goal line and backs off when hot.
   Evaluators accept `expert[:key=value,...]` anywhere they take a
   checkpoint.
2. **Human-proxy benchmark** `ai/bin/eval_human_proxy.py`: a far-side
   opponent with human reach (to the centre line), human-calibrated shot
   speeds (club: median ~5.5 m/s; elite 8–14), ~50% banks, reaction delay.
   Headline metric: goal share against it, with bootstrap CIs.
3. **DAgger** `ai/bin/distill_expert.py`: student = production's
   architecture initialised from production; roll-in mixes teacher and
   student; labels always from the teacher; opponents mix teacher,
   production and the student.
4. **RL fine-tune**: `train_neural_player.py` with a decaying teacher
   regulariser, the teacher as one of the opponents, and a less crippled
   optimiser (minimum actor steps per rollout, critic warm-up).
5. **Selection** on dev seeds, **confirmation** on reserved seeds
   (20261980–20261999, 20262000+ are not used for development).

## Status log

- 00:30 — mapped the stack (7 parallel readers; summaries in the session).
- 01:00 — teacher v0: fast-attack 82.2% (straight 97, left 71, right 78);
  16×120 s vs production 13–5; its shots weak (28% on target, ~5 m/s), one
  60 s overload holding deep near a side rail.
- 01:15 — launched in parallel: DAgger pipeline, trainer extensions,
  human-proxy benchmark; teacher defense and attack specialists.

## Runs

| Run | What | Status |
|---|---|---|

## Results

Development suite (`suite.py --tier dev`, seeds in `TIERS['dev']`):

| | production v1 | teacher v0 |
|---|---:|---:|
| fast attacks blocked (768, 10-18 m/s, 2/3 banks) | 28.6% (S 72 / L 2.3 / R 11.3), depth 0.63 m | 82.2% (S 97 / L 71 / R 78), depth 0.17 m |
| immediate release (768) | 25.7% | |
| requested fast shots, stationary S/L/R (of 128) | 75 / 38 / 42 | |
| requested fast shots, receiving S/L/R (of 128) | 88 / 86 / 81 | |
| slow control: fast aimed return | 61.7% | |
| games vs production, 16x60 s, blue / red | 24-24 / 16-23 (self-play noise) | 8x120 s: 8-2 / 5-3 |
| hot 4x600 s (0.8, gain 1.3) vs production | 15-13, overload 511 s, peak 1.057, stall 1361 s | |
