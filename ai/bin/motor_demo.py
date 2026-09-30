#!/usr/bin/env python3
"""Preview offline, or explicitly --live to run a finite motor choreography.

From the repository root:
    python3 ai/bin/motor_demo.py all
    python3 ai/bin/motor_demo.py slalom --live
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ai'))

import numpy as np
from airhockey.motor_patterns import (
    CENTER, DT, HIGH, LOW, NAMES, LoadTelemetryNotReady, check_load, plan_pattern, write_preview,
)


def fresh_position(client):
    sample = client.get_position_sample()
    if not all(math.isfinite(v) for v in sample) or not 0 <= sample[4] <= .15:
        raise RuntimeError('controller position is stale or invalid')
    if np.any(np.array(sample[:2]) < LOW-1) or np.any(np.array(sample[:2]) > HIGH+1):
        raise RuntimeError('controller position outside active workspace')
    return sample


def brake(client, clock=time.monotonic, sleep=time.sleep):
    """Keep current deceleration authority until the paddle has stopped."""
    x, y, _, _, _ = fresh_position(client)
    client.command_position(*np.clip([x, y], LOW, HIGH), 0)
    deadline, settled = clock()+3, 0
    while clock() < deadline:
        sample = fresh_position(client)
        settled = settled+1 if math.hypot(*sample[2:4]) < 10 else 0
        if settled >= 5:
            return
        sleep(.02)
    raise RuntimeError('controller did not settle while braking')


def shutdown(client, *, attempted_enable, clock=time.monotonic, sleep=time.sleep):
    disable_error = None
    if attempted_enable:
        try:
            brake(client, clock, sleep)
        except Exception as exc:
            print(f'Brake: {exc}', file=sys.stderr)
        finally:
            # Also disable if ENABLE partially failed, telemetry went stale, or
            # the master already latched a fault. Never re-enable on recovery.
            try:
                client.disable()
            except Exception as exc:
                print(f'Disable: {exc}', file=sys.stderr)
                disable_error = exc
    try:
        client.close()
    finally:
        if disable_error is not None:
            raise RuntimeError('DISABLE was not acknowledged; check motor state') from disable_error


def wait_for_load(client, stop_pct, *, not_before=None, timeout=2.):
    """Bounded setup barrier; sends no motion commands and only retries missing data.

    ENABLE holds the master's SDK mutex, so its cached LOAD can legitimately
    predate enabling by >0.5s. LIMITS/RAMP also perform blocking transactions.
    Only use this while stationary, before any trajectory command is issued.
    """
    deadline = time.monotonic()+timeout
    while True:
        try:
            return check_load(client.get_motor_load(), time.monotonic(), stop_pct,
                              not_before=not_before)
        except LoadTelemetryNotReady as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'timed out waiting for fresh motor telemetry: {exc}') from exc
            time.sleep(.02)


class Monitor:
    def __init__(self, client, stop_pct):
        self.client, self.stop_pct = client, stop_pct
        self.next_load, self.peak = 0., 0.

    def ready_after_setup(self):
        # Called at rest after a blocking configuration operation. Require
        # newly acquired values, not merely an old cache that is still <0.5s.
        self.peak = wait_for_load(self.client, self.stop_pct, not_before=time.monotonic())
        self.next_load = time.monotonic()+.1
        fresh_position(self.client)

    def poll(self):
        now = time.monotonic()
        if now >= self.next_load:
            snapshot = self.client.get_motor_load()
            self.peak = check_load(snapshot, time.monotonic(), self.stop_pct)
            self.next_load = now+.1
        return fresh_position(self.client)

    def rest(self, duration):
        end = time.monotonic()+duration
        while time.monotonic() < end:
            self.poll()
            time.sleep(.02)


def move_to(client, monitor, point):
    # Every caller arrives here at rest. Do not lower caps on a moving cart.
    sample = monitor.poll()
    if math.hypot(*sample[2:4]) >= 10:
        brake(client)
    client.set_limits(500, 3000)
    monitor.ready_after_setup()
    client.command_position(*point, 0)
    end, settled = time.monotonic()+8, 0
    while time.monotonic() < end:
        sample = monitor.poll()
        good = np.linalg.norm(np.array(sample[:2])-point) < 1.5 and math.hypot(*sample[2:4]) < 10
        settled = settled+1 if good else 0
        if settled >= 10:
            return
        time.sleep(.02)
    raise RuntimeError('paddle did not reach the demo start')


def perform(client, monitor, plan, writer):
    move_to(client, monitor, plan.reference[0])
    client.set_limits(plan.speed, plan.accel)
    monitor.ready_after_setup()
    monitor.rest(.3)
    start = time.monotonic()
    schedule_shift = 0.
    for i, target in enumerate(plan.commands):
        deadline = start+i*DT+schedule_shift
        time.sleep(max(0, deadline-time.monotonic()))
        poll_start = time.monotonic()
        sample = monitor.poll()
        now = time.monotonic()
        late = now-deadline
        if late > .04:
            raise RuntimeError(f'command loop fell {late*1000:.1f} ms behind '
                               f'(telemetry poll {(now-poll_start)*1000:.1f} ms); stopping')
        if late > DT:
            # Preserve command spacing after a small scheduling hiccup. Do
            # not send a burst of overdue waypoints to catch the old clock.
            schedule_shift += late
        client.command_position(*target, 0)
        sent = time.monotonic()
        writer.writerow([plan.name, i, now, now-start, *plan.reference[i], *target,
                         *sample, monitor.peak, late*1000, (now-poll_start)*1000,
                         (sent-now)*1000, schedule_shift])
    brake(client)


def listening():
    with socket.socket() as s:
        s.settimeout(.2)
        return s.connect_ex(('127.0.0.1', 8421)) == 0


def start_master(log, tension_mm=0.):
    if listening():
        raise RuntimeError('port 8421 is already in use. Stop the policy/UI hardware session, '
                           'or use --existing-master with an idle cdpr_master.')
    subprocess.run(['make', '-C', 'sw', 'build/cdpr_master'], cwd=ROOT, check=True)
    proc = subprocess.Popen([str(ROOT / 'sw/build/cdpr_master'), '--load-hz', '10',
                             '--tension', str(tension_mm)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic()+25
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('cdpr_master exited; see session master.log')
            if listening():
                return proc
            time.sleep(.1)
        raise RuntimeError('cdpr_master did not open port 8421 within 25 seconds')
    except BaseException:
        stop_master(proc)
        raise


def stop_master(proc):
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            # A second signal is the master's force-exit path and can skip
            # de-energizing. Do not turn slow shutdown into an unclean exit.
            print(f'cdpr_master {proc.pid} is still completing shutdown; see master.log', file=sys.stderr)


def live(args, plans, directory):
    from airhockey.hardware import CDPRClient
    client = CDPRClient()
    proc, attempted_enable = None, False
    old_handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, interrupt)
    try:
        with (directory/'master.log').open('w') as master_log, (directory/'ticks.csv').open('w') as ticks:
            if not args.existing_master:
                proc = start_master(master_log, args.tension or 0.)
            client.connect()
            # Fresh load before enabling; a warm motor is not a cold start.
            wait_for_load(client, args.rms_stop)
            sys.path.insert(0, str(ROOT/'vision/bin'))
            from track_mallet import measure
            print('Measuring starting paddle pose (camera must be free)...', flush=True)
            x, y, theta = measure()
            if (not all(math.isfinite(v) for v in (x, y, theta)) or
                    np.any(np.array([x, y]) < LOW) or np.any(np.array([x, y]) > HIGH)):
                raise RuntimeError('measured paddle pose is invalid/outside the active workspace')
            print(f'Enabling at measured ({x:.1f}, {y:.1f}) mm. Ctrl-C brakes and disables.', flush=True)
            attempted_enable = True
            client.enable(x, y, math.degrees(theta))
            client.set_ramp(3.)
            monitor = Monitor(client, args.rms_stop)
            print('Waiting for fresh post-enable motor telemetry...', flush=True)
            monitor.ready_after_setup()
            monitor.rest(.3)
            writer = csv.writer(ticks)
            writer.writerow(['pattern', 'tick', 'monotonic', 'elapsed_s', 'reference_x_mm',
                             'reference_y_mm', 'command_x_mm', 'command_y_mm',
                             'controller_x_mm', 'controller_y_mm', 'controller_vx_mm_s',
                             'controller_vy_mm_s', 'controller_age_s', 'max_rms_pct',
                             'deadline_lateness_ms', 'telemetry_poll_ms', 'command_roundtrip_ms',
                             'schedule_shift_s'])
            for repeat in range(args.repeat):
                for plan in plans:
                    print(f'{repeat+1}/{args.repeat}: {plan.name} ({len(plan.commands)*DT:.1f}s)', flush=True)
                    perform(client, monitor, plan, writer)
                    ticks.flush()
                    move_to(client, monitor, CENTER)
                    monitor.rest(2.)
    finally:
        # A second terminal signal must not interrupt DISABLE halfway through.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            shutdown(client, attempted_enable=attempted_enable)
        finally:
            stop_master(proc)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pattern', choices=(*NAMES, 'all'))
    parser.add_argument('--live', action='store_true', help='enable and move physical robot; default is offline preview')
    parser.add_argument('--existing-master', action='store_true', help='use an idle local master instead of starting one')
    parser.add_argument('--tension', type=float, help='startup cable retraction in mm per motor (default 0; owned master only)')
    parser.add_argument('--speed', type=float, default=4., help='session speed cap in m/s (default 4; max 12)')
    parser.add_argument('--accel', type=float, default=30., help='session acceleration cap in m/s² (default 30; max 60)')
    parser.add_argument('--repeat', type=int, default=1, help='finite repetitions, 1–10; 2s center rest between patterns')
    parser.add_argument('--rms-stop', type=float, default=85., help='stop at this percent of RMS shutdown, 50–90 (default 85)')
    parser.add_argument('--output', type=Path, help='directory for preview, plan metrics and live CSV')
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error(f'run from repository root: cd {ROOT}')
    if not 1 <= args.repeat <= 10 or not 50 <= args.rms_stop <= 90:
        parser.error('--repeat must be 1–10 and --rms-stop must be 50–90')
    if args.existing_master and not args.live:
        parser.error('--existing-master requires --live')
    if args.tension is not None:
        if not math.isfinite(args.tension) or args.tension < 0:
            parser.error('--tension must be a finite, nonnegative number of mm')
        if args.existing_master:
            parser.error('--tension configures a new master; for --existing-master, '
                         'set --tension when launching cdpr_master itself')
    try:
        plans = [plan_pattern(name, args.speed*1000, args.accel*1000)
                 for name in (NAMES if args.pattern == 'all' else (args.pattern,))]
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    directory = args.output or ROOT/'logs/motor_demo'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True, exist_ok=True)
    write_preview(plans, directory/'preview.html')
    (directory/'plan.json').write_text(json.dumps(dict(
        live=args.live, repeat=args.repeat, rms_stop_pct=args.rms_stop,
        startup_tension_mm=None if args.existing_master else (args.tension or 0.),
        sample_period_s=DT, patterns=[p.summary() for p in plans]), indent=2)+'\n')
    for plan in plans:
        m = plan.summary()
        print(f"{plan.name}: {m['duration_s']:.1f}s, predicted peak "
              f"{m['predicted_peak_speed_m_s']:.2f}m/s / {m['predicted_peak_accel_m_s2']:.1f}m/s²")
    print(f'Preview: {(directory / "preview.html").resolve().as_uri()}', flush=True)
    if args.live:
        try:
            live(args, plans, directory)
        except KeyboardInterrupt:
            print('Interrupted; shutdown completed.')
            return 130
        except Exception as exc:
            print(f'Demo stopped: {exc}\nSession: {directory}', file=sys.stderr)
            return 1
        print(f'Demo complete; drives disabled. Logs: {directory}')
    else:
        print('Offline preview only. Add --live to run the robot.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
