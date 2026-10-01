# Acceleration, workspace and current characterization

Run commands from the repository root. These experiments use no AI policy.
The program previews offline unless `--live` is supplied, and then requires
typing `RUN` before opening hardware. Do not run a policy, the UI hardware
mode, another master, or the UI camera concurrently. Remove the puck.

## Start with a holding-load and slow-travel survey

```sh
python ai/bin/characterize_robot.py --stage baseline --grid 3
python ai/bin/characterize_robot.py --stage baseline --grid 3 --live
```

The runner owns the master and camera. It measures a fresh initial pose,
uses fixed 1.5 mm pretension (override with `--tension`), moves at 0.2 m/s
and 0.4 m/s² between sites, and records eight seconds of holding at each.
Keep the hardware stop available. Ctrl-C stops the experiment, attempts to
brake, disables, and shuts down its master. No automatic recovery/re-enable.

## Acceleration ladder

For a rough cap near the previously used 60 m/s², use the quick screen:

```sh
python ai/bin/characterize_robot.py --quick
python ai/bin/characterize_robot.py --quick --tension 1.5 --live
```

This tests 40/60/80/100/120 m/s² at the center, eight directions once per
cap: 40 pulses instead of the full center sweep's 168. It uses a two-second
initial hold, no added rest, and skips redundant reposition commands when
already settled at the next start. Tracking, current, RMS, voltage and
freshness checks remain unchanged. Speed stays at 1.5 m/s. This screens a
local cap, not endurance or the whole workspace. Explicit `--accels`,
`--repeats`, `--hold`, `--rest` and `--grid` override the quick defaults.
The outcome lists caps that passed every planned direction; interrupted or
partially completed caps are not counted as passes.

After the slow survey passes, preview/test the center, then expand coverage:

```sh
python ai/bin/characterize_robot.py --stage sweep --grid 1
python ai/bin/characterize_robot.py --stage sweep --grid 1 --live
python ai/bin/characterize_robot.py --stage sweep --grid 3 --live
```

Defaults: 120 mm moves in eight directions, three repeats per direction,
1.5 m/s speed ceiling, 3 ms acceleration ramp, caps 2/5/10/20/30/40/60 m/s².
`--accels 2,5,10,20,40,60,80` extends the ladder explicitly, up to the
firmware ceiling of 120. `--speed`, `--stroke`, `--repeats`, and `--ramp-ms`
define the tested trajectory; results do not transfer automatically to other
speeds, ramps, positions, pretension, or starting thermal conditions.
Braking uses the same firmware acceleration cap being tested, so the test
keeps endpoints at least 40 mm inside bounds. These margins do not guarantee
recovery from arbitrary cable/motor faults. This is not a destructive stall test.

The same firmware profile is run offline to report predicted acceleration
actually exercised by each move, separating launch from braking. Low-speed,
very short moves or a longer ramp may not reach the cap: at a 1.5 m/s speed
ceiling, an 80 m/s² cap and 10 ms ramp produce about 62 m/s² launch acceleration
and 77 m/s² braking. A trajectory exercising <80% of its cap in either phase
is refused before opening hardware. The preview reports both peaks; a
braking peak alone does not qualify the launch.

The session tolerates rejected/missing camera frames for up to 50 ms from
the last accepted frame timestamp (including camera latency). Rejected
frames are logged with their reason and excluded from tracking and
acceleration measurements; the results report their count and flag gaps.
A new move requires a fresh accepted pose. Longer camera gaps, tracking
failures, lost/stale motor telemetry, current >=12 A on any drive,
fast/slow RMS >=70%, bus voltage
<60 V, gross camera/controller disagreement >40 mm, or monitor stall >150 ms.
Limits are explicit CLI arguments with bounded ranges. Startup requires RMS
<40%. These are host-side checks, not replacements for drive protections;
short faults between samples are not guaranteed detectable.

Qualification: camera tracking p95 <=20 mm with a fixed 15 ms pipeline
correction, settled camera error <=8 mm, and independent camera sample gaps
<=50 ms. Both unshifted and corrected errors are reported. No per-trial
latency fitting that could conceal motor lag. Controller reports are 50 Hz;
motor telemetry is approximately 10 Hz. Camera acceleration uses seven-frame
quadratic fits and reports the fitting window. An instantaneous acceleration
peak shorter than that window is not resolved. A passed cap is a lower bound
on capability for that specific trajectory, not an exact global maximum.

## RMS identification and endurance

```sh
python ai/bin/characterize_robot.py --stage endurance --grid 3 --accels 10 --repeats 10 --rest 0.2 --live
python ai/bin/characterize_robot.py --analyze logs/characterization/SESSION
```

Use a cap already qualified by the sweep for endurance. This version still
settles between moves; it does not qualify moving reversals or continuous
game duty. Current/RMS history is never reset between trials. More cooling
time is not assumed to help when stationary holding itself heats the motors.

Each run writes `plan.json`, `preview.html`, `samples.jsonl`, `results.json`,
`outcome.json`, and `master.log`; the plan links the full motor-load recording.
Motor readings carry acquisition times, validity, encoder resolution, drive
limits/time constants and voltage. Samples include continuous camera poses
and controller positions/velocities, rather than only the startup image.
The preview has a slider/play button and requires no web server.
Aborted runs also save `camera-diagnostic.png` and `camera-diagnostic.json`
when a processed frame is available. The frame is captured in memory at the
failure and written after shutdown; the error includes the detector's reason
and pose age instead of a generic tracking failure.
Each trial's results summarize sampled mean/peak absolute current, voltage,
and initial/final/peak fast and slow RMS separately for each motor, including
the RMS rise per second. Holding trials make these directly comparable by
position; short-term slopes alone do not establish sustainable holding.

Offline fitting uses a spatial 3×3 basis, speed squared and separate positive
and negative acceleration terms for each axis, independently per motor.
Whole trials are withheld for current-error evaluation. Fast/slow RMS curves
are compared using the actual recorded limits/time constants, initialized
from measured heat. Those RMS comparisons are in-session diagnostics, not an
independent validation. Sampling can miss short current peaks; collect longer
strokes and multiple speeds before trusting the acceleration terms.

Insufficient data produces only `fit-report.json`. Sufficient data additionally
produces `current-model-candidate.json`, explicitly unvalidated. It is a new
identification format, not a drop-in production thermal recipe. No production
limits, training defaults, drive settings, or thermal models are changed.

## Expanding the workspace

The explicit `teensy41_probe` firmware profile permits paddle-center
coordinates **x = 1200–1937.5 mm, y = 61.4–904.5 mm**. This leaves 30 mm
between the paddle rim and each of the three surrounding rails. The front
boundary, toward the human, stays at the historical wider-region value
of x = 1200 mm; it is a separate cable-geometry boundary, not a rail.
The normal firmware build and policy workspace remain unchanged.

Generate a preview without accessing hardware:

```sh
python ai/bin/characterize_robot.py --stage expand --ramp-ms 10
```

With motors disabled and existing hardware sessions stopped, upload the
probe image, then start the physical survey:

```sh
pio run -d fw -e teensy41_probe -t upload
python ai/bin/characterize_robot.py --stage expand --ramp-ms 10 --tension 1.5 --live
```

The runner verifies the actual firmware workspace before enabling. An old
or ordinary firmware image is rejected rather than silently clipping the
probe targets. Start with the paddle inside the existing policy region,
preferably near its center, remove the puck, and free the camera. The
runner still requires typed `RUN` and retains the electrical, tracking,
camera-gap and loop-timing cutoffs.

The initial plan has one center hold and 39 outward-and-return excursions
along eight rays. Each successive tip advances at most 40 mm. Every excursion
returns to a previously surveyed baseline location inside the old region.
Defaults are 0.3 m/s and 1 m/s²: this tests initial reachability, not the
high-speed dynamic envelope. There is no mandatory settle or hold at the
outward tip; the runner reverses near it and verifies camera approach within
8 mm. It does require settling back at the return point. The 30 mm clearance
is a commanded geometric margin; actual tracking error reduces it.

`workspace-results.json` records each tip as passed, failed, interrupted or
untested, plus tracking/current/RMS measurements. The first fault or failed
qualification stops the session, with no automatic re-enable. Brief reach
success does not claim sustained holding capability.

To compare finite holding cost, make a separate run with `--edge-hold 5`.
To compare lower pretension, repeat with `--tension 0.75`, for example:

```sh
python ai/bin/characterize_robot.py --stage expand --ramp-ms 10 --tension 0.75 --live
python ai/bin/characterize_robot.py --stage expand --ramp-ms 10 --tension 0.75 --edge-hold 5 --live
```

Pretension is fixed throughout each run and saved with its results. This
setup does not yet adjust tension while moving: firmware tension commands
are stopped-mode relative cable steps, not a live tension controller.
Record slack separately, since low current can indicate loss of cable
tension. A failed hold does not erase a successful brief reach, and a short
successful hold does not prove indefinite thermal sustainability. Use these
comparisons to select later tension scheduling and local acceleration tests.
The offline current-model fit uses the recorded workspace for its spatial
features, so expanded-region recordings retain the correct coordinate basis.

Restore the ordinary firmware before returning to normal policy operation:

```sh
pio run -d fw -e teensy41 -t upload
```
