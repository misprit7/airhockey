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
actually exercised by each move. Low-speed, very short moves may not reach
the cap. A trajectory exercising <80% of its cap cannot pass qualification.

The session stops on the first tracking failure, lost/stale camera or motor
telemetry, current >=12 A on any drive, fast/slow RMS >=70%, bus voltage
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

```sh
python ai/bin/characterize_robot.py --stage workspace --grid 5 --hold 15
python ai/bin/characterize_robot.py --stage workspace --grid 5 --hold 15 --live
```

First map the existing region (sites 40 mm inside current limits). Choose
candidate extensions by rail clearance, cable attachment geometry and
measured holding/tracking cost. Probe outward in small strips at low speed,
qualify both outward and return motion, then test acceleration locally.
Record slack separately: low current can mean a cable has lost tension,
not that the location is good. A few seconds without an RMS trip cannot
qualify an indefinitely sustainable holding position.

This implementation intentionally does not expand the firmware's hard
workspace or activate the archived WIDE bounds. Outside-boundary tests need
a separately reviewed, session-scoped firmware test envelope; changing only
the Python targets would silently clip motion and invalidate the experiment.
The final useful result should distinguish sustainable holding locations
from locations usable only for brief reaches, rather than blindly replacing
the current box with a larger rectangle.
