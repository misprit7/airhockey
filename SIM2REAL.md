# Sim-to-real: status and what's left

Living document. The point of it is the split between **what you have to do at
the table** (nobody else can) and **what is software** (I can).

---

## The one-paragraph version

The simulator was built before the table existed, so almost every physical
constant in it is a placeholder. None of the sim2real work is hard; it is
mostly *replacing guesses with measurements*, in an order where each step can
be checked. The blocking items are all measurements, and three of them need
you at the table for about an hour total.

---

## Done

| | what |
|---|---|
| ✅ | **Coordinate frames unified.** The sim used to map its whole half-table onto the *workspace* rectangle, so the scale changed whenever a limit was retuned and paddle speed was silently rescaled with it. Frame is now the table; the workspace enters only as a clamp. |
| ✅ | **Speed/accel caps: one definition.** Were in 7 places with 3 different values. Now `MAX_SPEED_M_S` / `MAX_ACCEL_M_S2` in `ai/airhockey/dynamics.py`, currently 15 m/s and 200 m/s². |
| ✅ | **Cable model moved to `shared/`.** `shared/check_geometry.py` used to reach into `ai/bin/` to import the model it validates. |
| ✅ | **Puck recorder + fitter** — `vision/bin/record_puck.py`, `vision/bin/fit_puck.py`. Validated against synthetic data: recovers μ=0.0042 from truth 0.0042, e=0.718 from truth 0.72. |
| ✅ | **Latency probe** — `FLASH` command in firmware (flashed), `vision/bin/measure_latency.py`. Moves nothing. |

---

## ⚠️ YOUR PART — three measurements, ~1 hour total

Nothing here energises a motor. Do them in any order.

### 1. Puck friction and restitution (~20 min at the table)

```bash
python vision/bin/record_puck.py
# push the puck around, then Ctrl-C
python vision/bin/fit_puck.py logs/puck_<timestamp>.jsonl
```

Drives can be off. What to push:

- **long straight glides**, no wall contact → friction
- **square-on hits into each of the four cushions** → restitution
- **glancing hits, 20–40°** → whether the puck picks up spin
- **vary the speed** — gentle to hard. Restitution is usually speed-dependent
  and one speed can't show that.

Five varied minutes beats twenty repetitive ones. The fitter prints exactly
what to put in the sim, and warns you if the data is too thin.

This replaces `puck_friction = 0.01`, `wall_restitution = 0.85`,
`paddle_restitution = 0.9` — all invented.

### 2. End-to-end latency (~10 min)

```bash
# cdpr_master must NOT be running (it holds the Teensy port)
python vision/bin/measure_latency.py --n 50
```

**Physical setup:** the Teensy's on-board LED must be visible to the camera —
taping it to the table pointing up is fine, it doesn't need to be in focus.
Take the **puck off the table** and leave the mallet still so nothing else
changes. Normal tracking lighting.

Splits the loop into the command half (host→USB→Teensy, via serial round
trip) and the sensing half (LED lit→camera→Python). You need both: a puck at
5 m/s moves 5 mm per millisecond, and the sim has to apply the same delay or
a policy trained there acts on information the robot won't have yet.

### 3. Spool radius (~5 min, no camera)

Still unverified through three pulley changes, and it is the **largest scale
factor in the machine** — every commanded millimetre converts through it.

Drives down. Rotate one spool **by hand through 10 whole turns** and measure
the cable paid out with a tape:

| if it measures | then r is |
|---|---|
| ~3016 mm | 48.0 mm — current value correct |
| ~2011 mm | 32 mm — the constant is 50% wrong |

A tape good to 5 mm pins r to 0.15%. This is also the leading suspect for the
unexplained edge overloads, since its error grows with distance from the
calibration point.

---

## Software remaining (mine)

In dependency order. Nothing here needs the robot.

### A. Real motion profile in the sim ← *next*
The sim's `DelayedDynamics` is a first-order filter with the same bang-bang
relay bug we removed from the firmware. It is not what your robot runs.

Plan: `motionProfileAdvance()` in `fw/include/motion_profile.h` (step +
integrate, so `cdpr.cpp` and the sim share one implementation), a batched C
wrapper with its own build target, ctypes binding, and a **tick-divergence
test** to choose the sim's timestep. The firmware ticks at 20 µs for step-rate
reasons the sim doesn't have; the sim's tick should come from the profile's
own timescales (3 ms jerk ramp, 6 ms velocity loop) — probably ~1 ms, 250×
less work, but measured rather than assumed.

### B. Collapse the duplicated physics
`physics.py`+`env.py` (scalar) and `batch_physics.py`+`batch_env.py` (batch)
are parallel implementations kept in sync by 1087 lines of parity tests. Those
tests pass, which is the proof one is redundant. Plan: batch becomes the only
engine, `AirHockeyEnv` becomes a thin Gymnasium adapter over `n_envs=1`,
parity tests deleted. Removes ~1800 lines and a whole class of bug.

### C. Feed the sim the real estimator
Right now the policy gets ground-truth puck velocity in sim and a 6-frame
least-squares slope in reality. Run the actual estimator over simulated noisy
positions instead. Also model the **IR blind spot** — ~92×103 mm at table
centre, up to 150 ms of coasting — as a structured dropout, because Gaussian
noise does not describe it and it sits in the highest-traffic part of the
table.

### D. Replay harness — the gap metric
Same recorded command sequence into sim and real from the same initial state;
measure divergence at 0.5/1/2 s. **This is the number that makes every other
claim falsifiable.** Without it, "the sim is good now" is an opinion.

### E. Then, and only then: domain randomisation and training
Ranges set by measurement uncertainty, not guessed. Tight where measured,
wide where genuinely variable.

---

## Open questions that need a decision

**Action space.** Currently target (x, y). Air hockey is "arrive *here*,
moving *this way*, at *this time*" and position encodes only the first — so
striking has to be induced indirectly. Likely landing place is `(x, y, vx, vy)`
with a terminal-velocity planner in firmware. A free intermediate exists: the
protocol already carries a speed argument, so `(x, y, speed)` costs no
firmware change. **Decide after A**, when the sim can actually tell the
options apart.

**Observation is not Markov.** With jerk limiting, the profile's acceleration
is state, and nothing in the observation exposes it. Worth fixing regardless
of the action-space decision.

---

## Parked

**Edge overloads.** RMSOverloadShutdown while merely *holding* near an edge.
Two modelling attempts of mine were wrong — a diverging `max(n)/min(n)` means
a cable going *slack*, not one overloading, and at the measured spring rate
(2.10 N/mm) the worst corner carries 4.4 N against a ~15 N limit. Cause
unknown. What is solid: only ~5 mm of retraction headroom exists before the
continuous limit, so something is adding a few mm near the edges — see
measurement 3. Workspace is currently trimmed empirically to
x 1350–1917.5, y 172.9–793.0.
