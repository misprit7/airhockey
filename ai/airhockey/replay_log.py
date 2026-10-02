"""Compact offline replay log. This module has no hardware interfaces."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from airhockey.physics import TableConfig
from airhockey.thermal import DEFAULT_MODEL


class ReplayLog:
    def __init__(self, path, args):
        self.file = open(path, "w")
        self.origin = None
        self.frames = 0
        self._rejected_jumps = 0
        config = TableConfig()
        motor_profile = json.loads(DEFAULT_MODEL.read_text())
        from airhockey.dynamics import _geom as geom
        workspace = [geom.WS_MIN_X, geom.WS_MAX_X, geom.WS_MIN_Y, geom.WS_MAX_Y]
        checkpoint = getattr(args, "resolved_checkpoint", None)
        if checkpoint and str(args.policy).startswith('neural:'):
            from airhockey.neural_setup import checkpoint_environment
            environment = checkpoint_environment(checkpoint)
            workspace = environment['workspace_bounds_mm']
            model = environment.get('thermal_path')
            if model is not None:
                motor_profile = model if isinstance(model, dict) else json.loads(Path(model).read_text())
            config.max_puck_speed = max(config.max_puck_speed, environment['defense_max_speed'])
        self._write(
            dict(
                type="meta",
                version=1,
                units="table_mm",
                policy=args.policy,
                live=args.live,
                ramp=args.ramp,
                camera_delay_s=0.0077,
                table_config=asdict(config),
                workspace_bounds_mm=workspace,
                motor_profile=motor_profile,
                speed_mm_s=getattr(args, "speed", None),
                accel_mm_s2=getattr(args, "accel", None),
                checkpoint=getattr(args, "resolved_checkpoint", None),
            )
        )

    def _write(self, record):
        self.file.write(
            json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        )

    def sync(self, camera_t, received):
        if self.origin is None:
            self.origin = received - camera_t - 0.0077
        # Offline alignment uses the minimum receive offset over all batches,
        # avoiding a queued first frame masquerading as transport latency.
        self._write(dict(type="clock", t=camera_t, monotonic=received))

    def frame(self, t, report):
        self._write(
            dict(
                type="frame",
                t=t,
                puck=list(report.puck[0][:2]) if report.t_puck == t else None,
                agent=report.mallet if report.t_mallet == t else None,
                human=report.opponent if report.t_opponent == t else None,
            )
        )
        self.frames += 1
        if self.frames % 100 == 0:
            self.file.flush()

    def tracking_diagnostics(self, t, tracker, blobs, report=None):
        """Preserve marker evidence when temporal association rejects a fix."""
        # Raw marker evidence is especially useful when apparent overlap could
        # be marker confusion, lift, or a real grazing collision. Tracks alone
        # cannot distinguish these; retain the underlying detections nearby.
        if (report is not None and report.puck and report.mallet is not None
                and report.t_puck == t and report.t_mallet == t
                and math.hypot(report.puck[0][0]-report.mallet[0],
                               report.puck[0][1]-report.mallet[1]) < 150):
            members = getattr(tracker, "frame_puck_members", None)
            self._write(dict(type="near_contact_tracking", t=t,
                             puck=list(report.puck[0][:2]), agent=list(report.mallet),
                             puck_markers=getattr(tracker, "n_markers", None),
                             puck_members=None if members is None else list(map(int, members)),
                             blobs_px=blobs.tolist()))
        if tracker.rejected_jumps != self._rejected_jumps:
            self._rejected_jumps = tracker.rejected_jumps
            self._write(dict(type="tracking_rejection", t=t,
                             rejected_jumps=self._rejected_jumps, blobs_px=blobs.tolist()))

    def motor_load_source(self, source):
        """Link the independently sampled master log using the same host clock.

        No per-tick queries are necessary. Source may be recorded before the
        camera clock is initialized; monotonic timestamps still align the files.
        """
        self._write(dict(type="motor_load_source", monotonic=time.monotonic(),
                         source=source))
        self.file.flush()

    def controller(self, t, x, y, vx, vy, age_s):
        """Keep the measured cache age and velocity, independently of camera.

        Age is from the same host monotonic clock as the master. Old masters
        have unknown age, represented as null rather than fresh zero.
        """
        queried = time.monotonic()
        known_age = math.isfinite(age_s)
        self._write(dict(type="controller", t=t, queried_monotonic=queried,
                         sample_monotonic=queried-age_s if known_age else None,
                         age_s=age_s if known_age else None,
                         x=x, y=y, vx=vx, vy=vy))

    def puck_watchdog(self, t, paused, reason):
        self._write(dict(type="puck_watchdog", t=t, paused=paused, reason=reason))

    def command(self, x, y, speed, accel, sent_at):
        if self.origin is not None:
            received = time.monotonic()
            mid = (sent_at + received) / 2
            self._write(
                dict(
                    type="command",
                    t=mid - self.origin,
                    monotonic=mid,
                    sent_monotonic=sent_at,
                    ack_monotonic=received,
                    x=x,
                    y=y,
                    speed=speed,
                    accel=accel,
                )
            )

    def end(self):
        if self.origin is not None:
            now = time.monotonic()
            self._write(dict(type="end", t=now - self.origin, monotonic=now))
        self.file.flush()

    def close(self):
        self.file.close()
