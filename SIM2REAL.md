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
| ✅ | **Sim runs the firmware's real motion profile.** `motionProfileAdvance()` in the shared header (law + integration + parking, previously split), built as a host library by `fw/host`, bound via `ai/airhockey/motion.py`. One implementation, no mirror. 4.3M env-steps/s at 4096 envs. |
| ✅ | **Sim tick chosen by measurement.** `ai/tests/test_motion_fidelity.py`. Reasoning said 1 ms; measurement said that diverges **1.32 mm on diagonals**, 3.5× the motor step. Settled on 0.2 ms (worst 0.252 mm). |
| ✅ | **Replay harness** — `ai/bin/replay_gap.py`, self-validated both ways (reproduces its own output to 0.0000 mm; detects a 20% accel error as 63.9 mm peak). |
| ✅ | **Hardware logger** — `ai/bin/log_hardware.py`. Commands nothing; reads `cdpr_master` STATUS and the camera. |
| ✅ | Dead code removed (`profile_gpu.py`, `ai/training/index.html`); an 8th copy of the speed/accel caps removed from `server.py`. |
| ✅ | **One physics implementation.** Scalar `PhysicsEngine` deleted; `scalar_engine.py` presents its interface over `BatchPhysicsEngine(n_envs=1)`. **−1123 lines**, including ~900 of parity tests. |
| ✅ | **Real profile wired into the env** as a `profile` dynamics type (`ideal` / `delayed` / `profile`). |
| ✅ | **Realistic perception** — `perception.py`: 6-frame slope estimator, back-projection noise, and the IR blind spot as a structured dropout with coasting. Off by default. |
| ✅ | **Mallet from the blob stream** — `vision/bin/mallet_stream.py`, back-projecting at the marker height (33 mm for the robot's arms, 67 mm for a dot on top). |
| ✅ | **Marker convention inverted, 2026-08-26** — four dots on the PUCK, one on the mallet, because a hand covers the mallet and nothing covers the puck. `vision/bin/puck_markers.py` solves the square; three corners still fix the centre exactly. Verified end to end through the real camera pose: ~1 mm regardless of how many corners survive. |
| ✅ | **Spin is now measured, not inferred** — four corners give orientation, so `record_puck.py` logs `th`/`w` per frame. |
| ✅ | **Latency LED moved to A9** (external, on the playing surface) per your note; firmware flashed. |

---

## Measured so far

**Loop latency — DONE 2026-08-23.** `command 0.11 ms one way`, `sensing 7.7 ms
mean over [5.1, 10.3]`. Wired into `perception.py` as
`CAMERA_DELAY_RANGE_S`.

Worth knowing how it nearly went wrong: the first reading was 9.8 ms with a
suspiciously tight 1.1 ms spread. That was the measuring loop flashing
immediately after a frame read, pinning the phase at its worst point.
Randomising the phase gave 7.7 ms with a 5.13 ms spread — one frame interval,
which is what quantisation should produce and the confirmation it's right.
The tool now randomises phase by default.

**Consequence for the sim: raise the control rate.** At 60 Hz one action step
is 16.7 ms, longer than the entire loop latency, so the delay can't be
represented at all. 100 Hz makes it exactly one step. 200 Hz matches the
camera and resolves the jitter too.

---

## ⚠️ YOUR PART — two measurements left, ~25 min

Nothing here energises a motor. Do them in any order.

### 1. Puck friction and restitution (~20 min at the table)

**First, 30 seconds, before recording anything.** The puck now carries four
dots and they have to come through as four separate blobs:

```bash
python vision/bin/puck_stream.py       # look at the "n/4 dots" column
```

Four for most frames is what you want. If it mostly says 2 or 3, the dots are
either too dim (raise `--gain`, or lower `--threshold`) or they are merging
into one blob (they project ~19 px apart at table centre, so anything under
about 15 mm across stays separate). A two-corner fix leans on the previous
frame to disambiguate, so its errors correlate rather than average out — which
is the one kind of error a friction fit cannot see through. `record_puck.py`
prints the same histogram at the end and warns below 60%.

```bash
python vision/bin/record_puck.py
# push the puck around, then Ctrl-C
python vision/bin/fit_puck.py logs/puck_<timestamp>.jsonl
```

Drives can be off. If you want paddle restitution too, one dot on the mallet
you are holding is now the right marking — pass `--mallet-z <height of that
dot above the surface>`. What to push:

- **long straight glides**, no wall contact → friction
- **square-on hits into each of the four cushions** → restitution
- **glancing hits, 20–40°** → whether the puck picks up spin. The last
  recording showed bounces returning 0.64 of their tangential velocity but
  could not say where that momentum went; with four corners the recording now
  carries orientation, so this run settles it
- **vary the speed** — gentle to hard. Restitution is usually speed-dependent
  and one speed can't show that.

Five varied minutes beats twenty repetitive ones. The fitter prints exactly
what to put in the sim, and warns you if the data is too thin.

This replaces `puck_friction = 0.01`, `wall_restitution = 0.85`,
`paddle_restitution = 0.9` — all invented.

### ~~2. End-to-end latency~~ — DONE

```bash
# cdpr_master must NOT be running (it holds the Teensy port)
python vision/bin/measure_latency.py --n 50
```

**Physical setup:** the external LED on **A9** must be in the camera's view.
The table does *not* need to be clear — the LED is located by differencing
frames, so permanent markers and IR glare are fine.

```bash
python vision/bin/locate_led.py                    # finds the pixel
python vision/bin/measure_latency.py --n 50 --led-px X,Y
```

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

### B. Collapse the duplicated physics
`physics.py`+`env.py` (scalar) and `batch_physics.py`+`batch_env.py` (batch)
are parallel implementations kept in sync by 1087 lines of parity tests. Those
tests pass, which is the proof one is redundant. Plan: batch becomes the only
engine, `AirHockeyEnv` becomes a thin Gymnasium adapter over `n_envs=1`,
parity tests deleted. Removes ~1800 lines and a whole class of bug.

**Not started, deliberately.** `server.py` reaches deep into the scalar env's
internals — `env.engine.state.paddle_agent.x`, `env._action_low`, hot-swapping
`env.agent_dynamics` to `HardwareDynamics` mid-session. Half-doing this would
leave the web UI broken while you're away from it, so it wants one
uninterrupted pass, not a partial one.

### C. Feed the sim the real estimator
Right now the policy gets ground-truth puck velocity in sim and a 6-frame
least-squares slope in reality. Run the actual estimator over simulated noisy
positions instead. Also model the **IR blind spot** — ~92×103 mm at table
centre, up to 150 ms of coasting — as a structured dropout, because Gaussian
noise does not describe it and it sits in the highest-traffic part of the
table.

### D. Replay harness — BUILT, blocked on A0
`ai/bin/replay_gap.py` is done and self-validated: it reproduces the
simulator's own output to 0.0000 mm, and catches a deliberately-detuned 20%
accel cap as a 63.9 mm peak divergence (so it is not blind). It reports
against two existing yardsticks — one motor step (0.377 mm) and mallet
tracking error (~4 mm) — and classifies the growth shape, since linear growth
implicates a scale error while quadratic implicates acceleration.

It needs camera-measured paddle position to score against, which is A0.

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
