"""Neural arrival policy adapter. No camera, sockets, or hardware activation.

The runner owns I/O and feeds back commands actually sent. The actor, feature
layout, arrival decoder and motion guard are shared with simulation. Firmware
acceleration is unobserved: a command-driven profile observer estimates it;
fresh controller velocity (or camera finite differences) corrects the observer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from airhockey.arrival import ArrivalDecoder, copy_cart
from airhockey.deploy import ReportEncoder, mm_velocity_to_sim
from airhockey.dynamics import workspace_in_sim, table_mm_to_sim, sim_to_table_mm
from airhockey.heuristics import Command
from airhockey.motion import CartState
from airhockey.motion_guard import guard_command, predict
from airhockey.neural_observation import neural_features
from airhockey.thermal import MotorThermal

ROOT = Path(__file__).resolve().parents[2]


def resolve_neural_checkpoint(name):
    # Explicit selection only: experimental neural models never replace TD latest.
    if name == 'latest':
        raise ValueError('neural: requires an explicit run name or checkpoint path')
    path = Path(name).expanduser()
    if not path.exists():
        path = ROOT / 'runs' / name
    if path.is_dir():
        steps = list(path.glob('agent_step_*.pt'))
        if steps:
            path = max(steps, key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        else:
            path = path / 'agent.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def neural_limits(checkpoint):
    meta = json.loads((Path(checkpoint).parent / 'run.json').read_text())
    algorithms = ('neural_ppo_v1', 'successful_neural_trajectory_imitation_v1')
    if meta.get('algorithm') not in algorithms or meta.get('action_mode') != 'arrival':
        raise ValueError('not a neural arrival training run')
    # The live command clamp and this adapter still use the deployment
    # workspace. Silently decoding an expanded actor in that box changes its
    # physical targets; simulation metadata is not a hardware configuration.
    if meta.get('workspace_bounds_mm') is not None or meta.get('args', {}).get('workspace', 'legacy') != 'legacy':
        raise ValueError('expanded-workspace neural checkpoint is simulation-only; '
                         'live runner workspace integration and physical qualification are pending')
    caps = meta['physical_limits']
    values = (caps['speed_m_s'], caps['acceleration_m_s2'])
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid neural training limits')
    return tuple(float(v) for v in values)


class LiveMotorLoad:
    """Eight normalized RMS channels, with a persistent model for missing fields.

    Fresh telemetry replaces individual channels, never invalidates good peers.
    Unknown startup heat is initialized to 0.8, not silently assumed cold.
    Neither a goal nor policy.reset clears motor heat.
    """
    def __init__(self):
        self.model = MotorThermal(1, randomize=False)
        self.model.gain[:] = 1.3
        self.model.h[:] = 0.8**2
        self.model.observed[:] = 0.8
        self.values = np.full((2, 4), 0.8)
        self.sample_times = np.full((2, 4), -np.inf)
        self.fresh = np.zeros((2, 4), bool)

    def update(self, snapshot, now):
        # LOAD returns {source, sample, logging_ok}; recordings store the sample directly.
        snapshot = snapshot.get('sample', snapshot) or {}
        for motor in snapshot.get('motors', []):
            node = motor.get('node')
            if type(node) is not int or not 0 <= node < 4:
                continue
            for channel, field in enumerate(('rms_pct', 'rms_slow_pct')):
                item = motor.get(field, {})
                value, end = item.get('value'), item.get('end')
                if (item.get('valid') is not True or
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or
                    not isinstance(end, (int, float)) or not math.isfinite(end) or
                    not 0 <= now - end <= 0.5 or end <= self.sample_times[channel, node]):
                    continue
                self.values[channel, node] = value / 100
                self.sample_times[channel, node] = end
                self.model.h[0, channel, node] = (value / 100)**2
                self.model.observed[0, channel, node] = value / 100

    def features(self, now):
        self.fresh = (now - self.sample_times >= 0) & (now - self.sample_times <= 0.5)
        return np.where(self.fresh, self.values, self.model.observed[0]).reshape(1, 8).astype(np.float32)


class NeuralPolicy:
    def __init__(self, checkpoint, speed_mm_s, accel_mm_s2, *, shot_mode='mix', device=None):
        import torch
        from airhockey.neural_player import NeuralPlayer, PhysicalHistory

        self.ckpt = resolve_neural_checkpoint(str(checkpoint))
        trained_speed, trained_accel = neural_limits(self.ckpt)
        if not (0 < speed_mm_s <= trained_speed * 1000 and
                0 < accel_mm_s2 <= trained_accel * 1000):
            raise ValueError('neural session caps must be positive and no higher than training caps')
        if shot_mode not in ('mix', 'left', 'right', 'straight'):
            raise ValueError('neural shot request must be mix, left, right, or straight')
        torch.set_num_threads(1)
        state = torch.load(self.ckpt, map_location='cpu', weights_only=False)
        self.net = NeuralPlayer(state['width'], history=state.get('history', 1),
                                shot_conditioned=state.get('shot_conditioned', False))
        self.net.load_weights(state['model'])
        self.net.to(device or 'cpu').eval()
        self.history = PhysicalHistory(self.net.history)
        self.speed = speed_mm_s / 1000
        self.accel = accel_mm_s2 / 1000
        self.encoder = ReportEncoder(shot_mode='straight', speed_m_s=self.speed, accel_m_s2=self.accel)
        ws = workspace_in_sim(self.encoder.width, self.encoder.half_h)
        self.decoder = ArrivalDecoder([ws[k] for k in ('min_x', 'max_x', 'min_y', 'max_y')])
        self.cart = CartState(1)
        self.delay = 0.015
        self.motion_t = None
        self.target = None
        self.command_cap = np.array([self.accel])
        self.command_speed = self.speed
        self.pending = []
        self.motion_history = []
        self.command_history = []
        self.state_source = 'camera'
        self.state_age = 0.0
        self.loads = LiveMotorLoad()
        self.shot_mode = shot_mode
        self.rng = np.random.default_rng()
        self.last_action = np.zeros(6, np.float32)
        self.last_obs = None
        self.last_flags = []
        self.sensor_hold = None
        self.reset()

    def reset(self):
        # Episode inputs reset; executing commands and physical heat persist.
        self.encoder.reset()
        self.history.values = self.history.request = None
        self.last_action[:] = 0
        self.in_half = None
        self.shot = 2 if self.shot_mode == 'mix' else ('left', 'right', 'straight').index(self.shot_mode)
        self.last_obs = None

    def update_motor_load(self, snapshot, now):
        self.loads.update(snapshot, now)

    def _advance(self, t):
        if self.motion_t is None or t <= self.motion_t:
            return
        while self.motion_t < t - 1e-9:
            while self.pending and self.pending[0][0] <= self.motion_t + 1e-9:
                _, self.target, self.command_cap, self.command_speed = self.pending.pop(0)
            end = min(t, self.pending[0][0] if self.pending else t, self.motion_t + 0.0025)
            dt = end - self.motion_t
            old_v = np.column_stack((self.cart.vx, self.cart.vy)) / 1000
            predict(self.cart, self.target, self.command_cap, self.command_speed,
                    dt, self.decoder.bounds)
            velocity = np.column_stack((self.cart.vx, self.cart.vy)) / 1000
            position = np.column_stack((self.cart.x, self.cart.y)) / 1000
            self.loads.model.advance(position, velocity, (velocity - old_v) / dt, dt)
            self.motion_t = end
            self.motion_history.append((end, copy_cart(self.cart)))
        self.motion_history = [(ts, state) for ts, state in self.motion_history if ts >= t-.15]
        # At the command boundary the new command is now executing.
        while self.pending and self.pending[0][0] <= t + 1e-9:
            _, self.target, self.command_cap, self.command_speed = self.pending.pop(0)

    def on_command(self, t, command, *, delivery_delay=None):
        """Feedback of successful sends, with measured ACK-midpoint timing.

        Offline/synthetic callers can omit timing and use the training delay.
        The observer uses actual feedback timing; arrival planning retains the
        trained 15 ms delay approximation.
        """
        if self.motion_t is None:
            return
        self._advance(t)
        target = np.array([table_mm_to_sim(command.x_mm, command.y_mm,
                                          self.encoder.width, self.encoder.half_h)])
        delay = self.delay if delivery_delay is None else float(delivery_delay)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError('command delivery delay must be finite and nonnegative')
        entry = (t + delay, target, np.array([command.accel_mm_s2 / 1000]), command.speed_mm_s / 1000)
        self.pending.append(entry)
        self.command_history.append(entry)
        # Keep the command already active at the oldest supported sample time.
        while len(self.command_history) > 1 and self.command_history[1][0] < t-.2:
            self.command_history.pop(0)

    def project_controller(self, sample, t):
        """Propagate a timestamped controller state to the decision time.

        Never mix a delayed camera position with an unrelated controller velocity.
        Commands and jerk-state history are estimates; received position/velocity
        replace those components at the sample time, before forward propagation.
        """
        age = float(sample['age_s'])
        if not 0 <= age <= .1:
            raise ValueError('controller projection requires a fresh sample')
        when = t-age
        history = [state for ts, state in self.motion_history if ts <= when]
        state = copy_cart(history[-1] if history else self.cart)
        xy = table_mm_to_sim(*sample['position'], self.encoder.width, self.encoder.half_h)
        vel = mm_velocity_to_sim(*sample['velocity'], self.encoder.width, self.encoder.half_h)
        state.x[:], state.y[:] = xy[0]*1000, xy[1]*1000
        state.vx[:], state.vy[:] = vel[0]*1000, vel[1]*1000
        target, cap, speed = self.target, self.command_cap, self.command_speed
        earlier = [entry for entry in self.command_history if entry[0] <= when]
        if earlier:
            _, target, cap, speed = earlier[-1]
        future = [entry for entry in self.command_history if when < entry[0] <= t]
        for end, next_target, next_cap, next_speed in [*future, (t, target, cap, speed)]:
            if end > when:
                predict(state, target, cap, speed, end-when, self.decoder.bounds)
            when = end
            target, cap, speed = next_target, next_cap, next_speed
        return state

    def observe(self, report):
        t = float(report['t_s'])
        if self.encoder._t_prev is not None and not 0 < t - self.encoder._t_prev <= 0.1:
            self.reset()
        # Initialize the observer without consuming an encoder/history tick.
        if self.motion_t is None:
            own = (report.get('controller_sample') or {}).get('position', report['mallet'])
            xy = table_mm_to_sim(*own, self.encoder.width, self.encoder.half_h)
            self.cart.reset(xy[0]*1000, xy[1]*1000)
            self.motion_t = t
            self.target = np.array([xy], dtype=float)
        self._advance(t)
        sample = report.get('controller_sample')
        if sample is not None:
            self.cart = self.project_controller(sample, t)
            report = dict(report, mallet=sim_to_table_mm(float(self.cart.x[0])/1000,
                                                       float(self.cart.y[0])/1000,
                                                       self.encoder.width, self.encoder.half_h))
            self.state_source = 'controller_projected'
            self.state_age = float(sample['age_s'])
        else:
            self.state_source = 'camera_or_legacy'
            self.state_age = 0.0
        # Training exposes the 20 ms position difference as the actor's own
        # velocity. Preserve that convention using projected positions; the
        # arrival decoder separately uses the instantaneous projected velocity.
        base = self.encoder.encode(report)[None, :]
        self.encoder.fresh = False
        if sample is None:
            self.cart.x[:], self.cart.y[:] = base[:, 4]*1000, base[:, 5]*1000
            velocity = base[0, 6:8]
            if report.get('controller_velocity') is not None:
                velocity = mm_velocity_to_sim(*report['controller_velocity'], self.encoder.width, self.encoder.half_h)
            self.cart.vx[:], self.cart.vy[:] = velocity[0]*1000, velocity[1]*1000
        self.motion_history.append((t, copy_cart(self.cart)))
        in_half = base[0, 1] < self.encoder.half_h
        if self.shot_mode == 'mix' and in_half and not self.in_half:
            self.shot = int(self.rng.integers(3))
        self.in_half = in_half
        base[:, 18:21] = np.eye(3)[self.shot]
        obs = neural_features(base, self.last_action[None], self.loads.features(time.monotonic()),
                              np.column_stack((self.cart.ax, self.cart.ay)), self.target,
                              self.command_cap, self.accel, self.decoder.low, self.decoder.high,
                              shot_conditioned=self.net.shot_conditioned)
        self.history.append(obs)
        self.last_obs = self.history.for_policy(self.net)[0].copy()
        return self.last_obs

    def __call__(self, report):
        started = time.perf_counter()
        if report.get('own_fresh') is False:
            self._advance(float(report['t_s']))
            if self.sensor_hold is None:
                self.sensor_hold = (sim_to_table_mm(float(self.cart.x[0]) / 1000,
                                                   float(self.cart.y[0]) / 1000,
                                                   self.encoder.width, self.encoder.half_h)
                                    if self.motion_t is not None else tuple(report['mallet']))
            self.reset()
            self.last_flags = ['neural_own_state_stale']
            return Command(*self.sensor_hold, self.speed * 1000, self.accel * 1000)
        self.sensor_hold = None
        obs = self.observe(report)
        action = self.net.act(obs[None])[0]
        if not np.isfinite(obs).all() or not np.isfinite(action).all():
            raise ValueError('nonfinite neural observation/action')
        target, cap = self.decoder.decode(self.cart, action[None], 0.0, self.target,
                                          self.command_cap, max_speed=self.speed,
                                          max_accel=self.accel, delay_s=self.delay)
        target, cap, changed, unresolved = guard_command(
            self.cart, target, cap, self.target, self.command_cap,
            bounds=self.decoder.bounds, max_accel=self.accel, max_speed=self.speed,
            delay=self.delay, action_dt=0.02)
        self.last_flags = (['arrival_guard'] if changed[0] else []) + (['arrival_guard_unresolved'] if unresolved[0] else [])
        self.last_action = action.copy()
        x, y = sim_to_table_mm(*target[0], self.encoder.width, self.encoder.half_h)
        self.last_ms = (time.perf_counter() - started) * 1000
        return Command(float(x), float(y), self.speed * 1000, float(cap[0] * 1000))

    def warm_up(self):
        # Full decoder/guard startup on synthetic data; no I/O or activation.
        point = sim_to_table_mm(0.5, 0.35)
        times = []
        for i in range(20):
            t = i * 0.02
            command = self({'t_s': t, 'mallet': point, 'opponent': None,
                            'puck': [(*sim_to_table_mm(0.5, 0.6), t)]})
            self.on_command(t, command)
            times.append(self.last_ms)
        self.motion_t = None
        self.target = None
        self.command_cap = np.array([self.accel])
        self.command_speed = self.speed
        self.pending.clear()
        self.motion_history.clear()
        self.command_history.clear()
        self.sensor_hold = None
        self.loads = LiveMotorLoad()
        self.reset()
        return float(np.median(times))
