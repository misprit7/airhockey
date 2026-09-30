# Motor choreography

Three deterministic routines, independent of the AI. Run from the repository
root with the puck removed. Release the camera and hardware connection from
the web UI or policy first; startup uses the camera to measure the paddle pose.

```sh
python3 ai/bin/motor_demo.py slalom --live
python3 ai/bin/motor_demo.py rosette --live
python3 ai/bin/motor_demo.py starburst --live
```

- **Slalom:** two passes through linked figure-eights around three imaginary
  posts arranged across the table. Smooth winding turns and repeated crossings.
- **Rosette:** two traces of a five-petal flower, repeatedly crossing the same
  center from different directions.
- **Starburst:** eight rapid outward-and-back strokes in alternating directions,
  with short stops at the tips and center to demonstrate repeatability.

Run all three with `python3 ai/bin/motor_demo.py all --live`. Each command builds
and starts its own `cdpr_master`, measures/calibrates the starting pose, enables,
gently moves to the start, performs the routine, returns to center, and disables.
Ctrl-C brakes using the current acceleration cap and then disables. A second
Ctrl-C during cleanup does not interrupt disable. If using an already-running,
idle master, append `--existing-master`; the demo still disables when it exits.
It never kills another controller or takes over an active policy connection.

Add `--tension 1.5` to retract each cable by 1.5mm during startup, using the
same pretension routine as `play.sh`. This is cable take-up in millimetres,
not a force/torque setting. Default remains zero; shutdown releases it. With
`--existing-master`, configure tension on that master's own command line;
the demo rejects `--tension` rather than silently ignoring it.

The defaults are **4 m/s speed cap and 30 m/s² acceleration cap**. These apply
only to this session; no policy or firmware defaults are changed. For faster
routines, append `--accel 60`. The same three curves then run in about 5–7s
each, with predicted peaks around 2.3–2.8 m/s. `--repeat 2` repeats the selected
routine(s), with a gentle return to center and 2s rest between each. These are
finite bursts, not endless loops. Tight turns mean actual speeds are lower
than the cap, which is printed separately from predicted peaks.

For a faster slalom with pretension:

```sh
python3 ai/bin/motor_demo.py slalom --live --tension 1.5 --speed 12 --accel 60
```

This predicts a 2.90m/s peak, versus 2.05m/s with the default caps, and reduces
the slalom from 8.6s to 6.3s. The curves are acceleration-limited: a 12m/s cap
does not imply they can reach 12m/s in this workspace. These are model
predictions, not measured hardware maxima. Use `all` to run all three demos.

The runner reads cached fast/slow RMS telemetry at 10Hz and stops at **85% of
the drive's RMS shutdown threshold**. Fresh fast RMS from all four motors is
required; slow RMS is also checked wherever supported. A fault, stale position,
missing load telemetry, or command loop more than 40ms late aborts the routine.
After enabling or changing limits while stationary, it waits up to 2s for
newly acquired telemetry before sending movement commands. Enabling temporarily
locks out the master's load sampler; the old cache is not treated as a running
telemetry failure. Actual high RMS or faults still abort immediately, and the
0.5s freshness limit remains unchanged during motion.
This is a host-side early stop, not a guarantee against the drive tripping.
The drives retain their own protection. There is no automatic restart after
an abort. `--rms-stop` accepts 50–90; it cannot disable the guard.

## Offline preview and verification

Omit `--live` to make an animated HTML preview without opening the camera,
connecting to a master, or enabling hardware:

```sh
python3 ai/bin/motor_demo.py all --output logs/motor_demo/preview
xdg-open logs/motor_demo/preview/preview.html
```

The preview has a selector, pause, and scrubber. Gold is the geometric reference;
cyan is the motion predicted by the actual C firmware profile. Planning runs
before any hardware connection, rejects out-of-range caps, and retimes paths
until the model stays within 5mm of the reference and obeys workspace, speed,
and acceleration limits. References remain 65mm inside the active workspace.
Smooth phase ramps start and end at rest; starburst stops use quintic ramps.

The planner converts these curves to ordinary position commands at 100Hz,
accounting for a nominal 8ms command delay and the 3ms firmware jerk ramp.
It runs the shared C profile at 0.2ms resolution to check them. Commands are
precomputed; there is no learned actor or new firmware action space. The
preview models controller state, **not** elastic cables, tracking error,
step loss, camera measurements, or motor heat. Physical accuracy still needs
measurement on the machine.

Each invocation writes `preview.html` and `plan.json` under
`logs/motor_demo/<timestamp>/`. Live runs also write `master.log` and
`ticks.csv`, with reference, position command, cached controller position and
velocity, sample age, maximum reported RMS, deadline lateness, telemetry/command
round-trip durations, and accumulated scheduling shift. Small delays shift the
remaining schedule rather than causing a burst of overdue commands; delays
over 40ms still stop the routine. The master's periodic torque/alert diagnostics
run off the command thread so their SDK reads don't stall the 100Hz stream.
Controller position is the
Teensy's step-integrated estimate, not independently measured paddle position.
The master's usual `logs/motor_load/` recording contains the individual motor
channels. Command-line cap units are m/s and m/s²; CSV coordinates are mm.

Offline regression checks:

```sh
PYTHONPATH=ai python3 -m pytest ai/tests/test_motor_patterns.py -q
```
