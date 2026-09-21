# Motor RMS and torque recordings

`cdpr_master` now records motor load automatically at a target **10 Hz** in
`logs/motor_load/<unix-microseconds>-<pid>.jsonl`. Files are unique and are never
overwritten. Logging runs while the master is connected, including while the
motors are disabled, so a master left running after DISABLE can capture recovery.
Exiting the master ends the recording. This change does not enable motors,
change limits, add a load governor, or launch training.

The user's 2026-09-20 13:09 recording exercised all four drives: fast/slow RMS
and current channels were valid, with no logged drive fault. Offline results
and provisional model coefficients are in
`logs/analysis/20260920-130924/summary.json`; `analyze.py` reproduces the analysis
and `rms.png` plots it. No new hardware session was started for that analysis.

## Training integration

The `4.0-arrival-rms-selfplay` experiment now uses a provisional per-motor
fast/slow load state and reward cost (`thermal.py`). It combines the recorded
motion fit below with explicitly conservative acceleration/holding priors and
load uncertainty. Heat persists across points and episode resets. This is not
a deployed runtime limiter or an overload guarantee. See `ARRIVAL_TRAINING.md`.

## First physical recording

The enabled portion lasts 60.28 seconds at 8.89 snapshots/second. Fast RMS peaks
for nodes 0–3 are **70%, 30%, 90%, 30%**; slow peaks are **11%, 15%, 16%, 15%**.
Node 2 is the limiting motor in this recording. Its fast current limit is 4 A;
nodes 0/1/2/3 have fast limits 4/5.8/4/5.8 A, so one shared motor scale is wrong.

The SDK's fast setting of 2.985 seconds is **not** the exponential e-fold time.
`convertRMStc` in the installed `cpmAPI.cpp` returns `log(8/9)/b`, where `b` is
the log decay coefficient per second. Thus the e-fold time is
`setting / -log(8/9)`, about **25.34 seconds**. Filtering squared sampled current
with this time gives fast-RMS errors of 5.2/1.8/5.9/1.8 percentage points, versus
24.6/10.1/35.4/7.9 when incorrectly treating the setting as the e-fold time.
The slow conversion similarly implies about 2543/509/2543/509 seconds; this
one-minute run cannot independently validate those long memories or the full
drive protection algorithm.

A provisional mean-square-current model using actual simulated acceleration
and speed, replayed through the firmware motion law, was fitted on the first
40 seconds only. Continuous predictions on the remaining 20 seconds have
fast-RMS RMSE 2.14/0.67/2.53/1.74 percentage points. This is a within-session
holdout, not independent validation. Raw coefficients remain in the analysis
artifacts. The training integration above adds explicit conservative priors;
no model-based motor-load limiter is installed in physical deployment.

Position/holding load matters too: node 2 averaged 2.93 A while stationary at
(1858, 173) mm near the low-Y edge, versus 0.048 A during an interior hold at
(1527, 411) mm. A more flexible position model overfit this short recording and
worsened held-out errors for three motors. It is premature to treat either
regression as an overload guarantee. Preserve load across goals, expose both
per-motor load states to the policy, and validate longer recordings and changed
motion before relying on the model for burst allocation.

Telemetry acquisition took 107 ms median / 147 ms p95 for a full four-drive
snapshot. The policy uses cached data, but the SDK transport is still shared
with the watchdog; sampling current at this rate also misses short peaks. The
drive's internal RMS measurement remains the target for calibration.

## What gets recorded

For each of four motors:

- Fast RMS and, where supported, slow RMS, in percent of their respective
  shutdown thresholds. Unsupported/failed channels are `null`, with an error;
  they are not substituted with zero.
- Signed measured torque-producing current in amperes (not shaft torque in Nm).
- Measured encoder position in counts and velocity in counts/second.
- Actual drive enabled/alert-present bits and three raw alert words.
- Peak-current scale, RMS current limits, fast/slow time constants, encoder
  resolution, drive serial number, firmware code and torque limit. Configuration
  is sampled initially and every 60 seconds, with its original timestamps.
  The SDK reports the fast time constant in seconds and the slow one in minutes;
  the names retain those units. These values are evidence for fitting, not a
  claim that one exponential reproduces the complete drive protection algorithm.

Each field has `valid`, `value`, `start`, `end`, and `error`. Acquisition times
use Linux `CLOCK_MONOTONIC`, matching Python's `time.monotonic()` on this host.
The four drives and different channels are read sequentially, not simultaneously.
Use field times for fitting. Each snapshot also carries wall time, total
acquisition duration, watchdog check time/gaps, master enable/fault state,
configured startup pretension and spool radius, and
the latest timestamped Teensy position, velocity, step counts and active limits.
Those controller readings are estimates; replay camera measurements remain
separate ground truth.

Command-receipt events capture CMD, LIMITS, RAMP, ENABLE and DISABLE, as well as
watchdog shutdown notifications. They are **not execution acknowledgments**.
The bounded event queue reports dropped events rather than growing indefinitely.
The policy replay already contains command acknowledgments and camera alignment.
It now includes a `motor_load_source` record pointing to the matching master log.

## RMS scaling

The logger reads `Info.Ex.Parameter(CPM_P_DRV_RMS_LVL)` and
`CPM_P_DRV_RMS_SLOW_LVL`. In the installed SDK, these use `convertRMSlevel` /
`convertRMSlevelSlow`, which return 0–100 percent of shutdown (rounded to integer
percent). It deliberately avoids `Status.RMSlevel.Value()`: this SDK version
also applies torque-unit scaling to that convenience property.

Measured current likewise uses the underlying engineering-unit parameter,
without changing global SDK torque/velocity units. No writes to drive parameters,
fault clearing or enable requests are performed by the telemetry reader.

## Running and inspecting

The rebuilt master will log on its next normal launch. An already-running
master must be restarted during the next authorized hardware session to load
the new binary. No running process was restarted as part of implementation.
`play.sh` now checks the master build on every live launch so an existing old
binary does not silently skip source updates.

There are no extra flags needed for recording. The master and `play.sh` also
accept `--load-hz N` (0 disables it; maximum 50). This is a requested rate, not
an achieved-rate guarantee. Begin with the default and check the data before
raising it: bus capacity and fault-check latency matter more than a nominal Hz.

Inspect the newest recording without any robot connection:

```bash
python3 ai/bin/inspect_motor_load.py
```

Or inspect a specific recording:

```bash
python3 ai/bin/inspect_motor_load.py logs/motor_load/RECORDING.jsonl
```

The inspector reports sample rate, largest sample gap/acquisition duration,
per-motor RMS/current peaks, missing RMS samples, dropped events and latest
controller/watchdog context. It can read a recording still being written and
ignores an unfinished final line. It does not contact, enable or command motors.

Send back the motor-load JSONL together with the policy `.replay.jsonl` for the
session. Ordinary `play.sh` cleanup stops the master, so it cannot record a long
disabled recovery afterward. For a recovery segment, keep the separately run
master alive after disabling through the existing authorized session workflow.

## Cached API and scheduling

The existing master's TCP connection accepts:

- `LOADMETA`: recording path, clock origin, PID, requested rate and unit conventions.
- `LOAD`: source plus cached latest sample and `logging_ok`.

Python: `client.get_motor_load(metadata_only=True)` or `client.get_motor_load()`.
Use the existing client; opening a second connection competes with the master's
single-owner control interface. The file inspector needs no connection at all.

Cached requests never read the drives or disk. `run_policy.py` makes one metadata
request before enabling, with no telemetry requests in the policy tick. Sampling
and disk writes happen in a background thread. The worker releases the motor
mutex between SDK field calls and gives a waiting fault watchdog priority.
There is no catch-up burst after a slow sample. Individual SDK calls are still
synchronous and cannot be preempted; logged field durations and watchdog gaps
must be checked on hardware. Read failures back off instead of hammering the bus.

File-write failures appear on stderr and as `logging_ok:false` in the cached
API. Shutdown performs the existing de-energizing sequence before waiting for
the logging thread, so slow log writes do not move ahead of that sequence.
Existing drive protections remain in place.


Controller position freshness is now recorded independently of query timing:
`POS` appends the master's monotonic status-receive timestamp, tick logs contain
`ctl_age_ms`, and replay logs contain `controller` records with velocity and sample
age. The runner ignores controller poses older than 50 ms when a fresh camera
pose is available. This avoids treating a responsive cached POS endpoint as a
fresh physical measurement. The master shares one persistent serial framer across
ACK waits and background status reads so neither path discards the other's bytes.
