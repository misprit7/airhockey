# Live review, October 2, 2026

Offline review of `20261002-114149`, checkpoint
`runs/rail30-100-20261001-v1/agent.pt`, SHA-256
`39aa7a501edc127bc37fd2e500578366c2b0a735a45eb0b628acd07c5bddf629`.
Motor log: `logs/motor_load/1790955708784819-2411119.jsonl`.
The recording continued while idle; comparisons use the 27,372 active decisions
through camera time 832.33 s, not the idle tail. Raw recordings remain untracked.

Run from the repository root:

```sh
PYTHONPATH=ai python ai/bin/audit_neural_sessions.py 20261002-114149 --output-dir logs/analysis/neural-live-20261002
python ai/recipes/live-review-20261002/review.py
python ai/recipes/live-review-20261002/motion_load.py
python ai/recipes/live-review-20261002/marker_geometry.py
python ai/recipes/live-review-20261002/selfplay.py
python ai/recipes/live-review-20261002/goal_rollouts.py
python ai/recipes/live-review-20261002/report.py
```

`selfplay.py` uses CUDA and performs eight 60-second matches. Nothing in this
recipe opens hardware or changes weights. The HTML report is served at
`http://localhost:8420/training/reports/20261002`.

Interpretation constraints:

- The first 42 history features are current; history is newest first.
- Logged raw_accel already includes the arrival guard. Actor acceleration is
  reconstructed from action component 5 using the actual decoder mapping.
- Seven near-goal watchdog events are likely concessions, not an official score
  count. The 760.74 s event has no observed midline crossing in its two-second
  window; its rollout starts 350 ms before the watchdog instead.
- Open-loop motion tests initialize once and replay commands. They do not feed
  later robot measurements into simulation. The full-cap sensitivity exercise
  ends at the last fresh puck observation, not the later watchdog event. Changed
  contact would change future observations/actions, so this cannot count saves.
- Side-rail fits require continuous observations, low line-fit residuals, and
  distance from the robot. First 400 seconds select coefficients; later bounces
  test them. The small fitted changes do not justify changing physics defaults.
- Near-contact raw blobs are a selected sample, not a full camera recording.
- Stationary current samples are concentrated at the home pose. They cannot
  identify a replacement load model across the expanded workspace.

Implemented preparation for a future training comparison:
`--defense-windup-bank-fraction 0.5` samples hidden delayed direct/left-bank/
right-bank releases, including lateral drift. Default zero preserves old runs.
No new policy has been trained by this review. Suggested comparison: current
energy weight 2 versus 0.5, keeping load/overload and edge terms intact, combined
with bank-aware preparation and explicit burst-action exploration. Qualify on
held-out human-like banks, direct-shot saves, shot speed/on-target rate, and
continuous hot self-play; do not select on self-play score alone.
