# Comparing a physical session with simulation

Open **http://localhost:8422/**, or the same port on the machine's Tailscale
address. The main web UI also links to **Sim / Real recording viewer**.
The comparison server is separate from robot control and never opens the
camera, motor interface, policy checkpoint, or controller socket.

Select a hardware session in the left column. By default, the viewer runs
independent 10-second simulations from 0, 10, 20, … seconds and shows their
paths and mean errors in a grid. Click a tile to inspect it. The timeline,
mouse wheel over the table, and arrow keys scrub the recording; Space plays
or pauses. **Simulate from this time** starts a new 10-second rollout at the
selected timestamp. **Return to grid** restores the interval comparisons.

Cyan is camera measurement, amber is simulation, purple is the recorded
human paddle. Circles with a cross are paddles. Gaps in measured tracks
are hidden; the error readout is unavailable without both measurements.

## What is simulated

- Initial puck and robot positions come from measured tracks interpolated
  at the selected time. Velocities are fitted using only the preceding
  40 ms; with insufficient history they start at zero. Initial profile
  acceleration is unknown and starts at zero. Starts near impacts or
  tracking glitches therefore have additional initialization uncertainty.
- The robot follows the recorded target positions and actual speed/accel
  arguments through `BatchAirHockeyEnv._update_dynamics` and its compiled
  firmware motion profile, including workspace containment and jerk ramp.
- Puck motion, rail losses, drag and paddle impacts use the same
  `BatchPhysicsEngine` as the environment, at 2 ms physics steps, with the
  current nominal physical parameters and no domain randomization.
- Only the human paddle's measured trajectory is supplied after the start.
  Short observed intervals are linearly interpolated. Longer tracking
  holes stop the rollout; the final measurement may be held for at most
  150 ms. No future robot/puck measurements correct the simulated state.
- A simulated goal ends the rollout. No artificial serve, stuck-puck
  relaunch, shot-clock turnover, or policy inference is performed.
  Recorded end/stop events also end the rollout.

This is telemetry replay, not camera video. The measured trajectories are
tracker outputs and can contain identity mistakes; the viewer does not
treat the controller's step-count position as camera ground truth.

## Recording format

Normal `ai/bin/run_policy.py` sessions now also write
`logs/run_policy/<stamp>.replay.jsonl`. No new logging flag is needed;
`--no-log` disables it along with the existing logs. Human tracking must
be enabled (`play.sh` already supplies `--opponent`). This does not grant
permission to run the physical robot.

The JSONL stream contains:

- `meta`: version, units, live/dry status, policy, jerk ramp, nominal table
  parameters and the measured mean camera delay.
- `clock`: camera timestamp and monotonic host reception time for the
  newest frame of each received batch.
- `frame`: camera time and fresh puck/robot/human fixes in table mm, or
  null when an object wasn't seen. Every processed frame is logged,
  including policy holds and frames skipped for decision making.
- `puck_watchdog`: pause/resume transitions, with `goal` or `tracking_loss`
  as the pause reason. Holds also appear in the tick CSV (`blind=1`,
  `flags=puck_hold`); no new policy observation is fabricated while paused.
- `command`: successful command arguments, monotonic send/ack times and
  their midpoint. LIMITS changes to a previous target and blind holds
  are recorded too. Dry-run commands are labeled by the session metadata.
- `end`: end of tracking, before the existing shutdown/braking path.
- `tracking_rejection`: raw pixel marker blobs and cumulative rejection count
  when the puck's temporal association rejects an otherwise matching square.
  This diagnostic does not supply a puck measurement to the policy or replay.

Command times use the minimum observed host/camera offset minus the
measured 7.7 ms sensing delay. Send/ack midpoint approximates application
time; firmware does not report its exact command execution timestamp.
The timing control in **How the comparison works** shifts commands by
±100 ms to explore that uncertainty. Positive means later commands.

Old `.ticks.csv` recordings remain supported, with warnings: they sample
at decision rate, omit blind intervals and have approximate command
timing and camera-fix freshness. They are useful for inspection, but a
new JSONL session is better for measuring the simulation gap.

## Server and validation

```bash
PYTHONPATH=ai python3 -m airhockey.replay_server
# Background service started during implementation:
systemctl --user status airhockey-replay-ui
```

The tests in `ai/tests/test_hardware_replay.py` verify open-loop isolation,
command and human-input effects, independent grid windows, timestamp
alignment, missing-data handling and goal termination. Deployment runner
tests exercise logging with mocked hardware only.

## Motor-load sidecar

New hardware sessions also write `logs/motor_load/*.jsonl` from the master.
The replay's `motor_load_source` record identifies the matching file; both use
the host monotonic clock for alignment. This includes per-motor RMS/current,
encoder motion, status and configuration. See [motor-load logging](MOTOR_LOAD_LOGGING.md)
for the schema and the file-only inspection command. There is no extra drive
polling in the policy tick. Existing recordings do not acquire these fields.
