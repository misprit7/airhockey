"""Named simulator profiles that reproduce what the table did; no hardware I/O.

The sim the neural player trained in was kinder than the table in four
measured ways. Each is an OPT-IN model, off by default, and this module is
the one place that turns a profile name into NeuralTrainingEnv keyword
arguments, so a trainer and every evaluator ask for the same table:

    env = NeuralTrainingEnv(n, **options, **table_fidelity("table-20261029"))

"table-20261029" is the calibration from the 13 table sessions of
runs/teacher-calm-20261018-v4 (2026-10-19, 10-29 and 11-08, standard paddle):

  contact   speed_restitution + paddle_tangential_range. Robot-mallet
            restitution about 0.50 below 3 m/s of closing speed falling to
            0.33 above 9, tangential retention 0.70-0.90, per-contact
            scatter 0.10 (batch_physics.SpeedRestitution). Training used
            0.60-0.95, frictionless.
  lift      mallet_lift. A 6-8 m/s puck passes under a mallet that is
            braking hard, keeping 45-95% of its speed and its heading
            (batch_physics.MalletLift). Off in a quarter of episodes.
            Events are counted in info["lift_events"] /
            info["opponent_lift_events"] per step and cumulatively in
            engine.lift_events[body].
  sensing   table_sensing. The larger centre glare hole, the blind bands at
            both ends, flicker anywhere, and the opponent's mallet hidden
            and reported as the runner reports it (perception.TableSensing).
            The base env's fuzz_p windows are drawn over the episode length
            and never open in a NeuralTrainingEnv, whose base episode does
            not end; these are per-frame hazards instead.
  actuator  command_delay_range + late_command_probability. Commands take
            9-19 ms per episode instead of exactly the 15 ms the decoder
            plans with, and 0.1-2.8% of them land a tick late
            (BatchAirHockeyEnv.enable_command_jitter).

BOTH SEATS. A far side on the robot's body (self-play, the scripted kinds
that drive it) is a copy of the same machine: it shares the episode's contact
and lift parameters, lifts by the same rule, sees the puck through the same
dropouts, loses the agent's mallet in the same spells and gets the same
command latency with its own late ticks. A free body -- the human proxy, the
sniper, the weak goalie -- is not the machine: its mallet keeps
engine.paddle_restitution (and the shared tangential retention), never lifts,
and the human proxy keeps the view it was calibrated on (the legacy tracker
model, on ring columns of its own), because a person watches the table, not
the camera. HumanProxy.ASSUMED_RESTITUTION therefore stays as it is.

NOT modelled: mallet follow lag as a separate compliant body (6-20 mm of
deflection at 100 m/s^2), the 220-400 mm strike-and-return position losses,
and the runner's goal watchdog pausing the policy.
"""

from __future__ import annotations

from typing import Any

PROFILES = {
    "table-20261029": {
        "contact": {"speed_restitution": True, "paddle_tangential_range": (0.70, 0.90)},
        "lift": {"mallet_lift": True},
        "sensing": {"table_sensing": True},
        "actuator": {"command_delay_range": (0.009, 0.019),
                     "late_command_probability": (0.001, 0.028)},
    },
}
DEFAULT_PROFILE = "table-20261029"


def table_fidelity(profile: str = DEFAULT_PROFILE, *, contact: bool = True,
                   lift: bool = True, sensing: bool = True, actuator: bool = True,
                   **overrides: Any) -> dict[str, Any]:
    """NeuralTrainingEnv keyword arguments for a named table profile.

    contact/lift/sensing/actuator drop a part of the profile (for ablations
    and for telling the parts' effects apart). `overrides` replace individual
    keyword arguments, e.g. mallet_lift={"off_probability": 0.0} for a table
    whose mallet always can lift, or speed_restitution={"scatter": 0.0}; None
    removes one. sensing needs realistic=True (the env's default).
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown table profile {profile!r}; known: {sorted(PROFILES)}")
    parts = PROFILES[profile]
    options: dict[str, Any] = {}
    for name, wanted in (("contact", contact), ("lift", lift),
                         ("sensing", sensing), ("actuator", actuator)):
        if wanted:
            options.update(parts[name])
    known = {key for part in parts.values() for key in part}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"not table-fidelity arguments: {sorted(unknown)}")
    options.update(overrides)
    return {key: value for key, value in options.items() if value is not None}
