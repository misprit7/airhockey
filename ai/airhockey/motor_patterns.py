"""Deterministic motor demonstrations; planning is entirely offline.

All coordinates are table-frame millimetres. A reference curve is converted
to ordinary position commands using the inverse of the firmware braking law,
then checked using the actual firmware profile. This predicts controller
motion, not camera-measured tracking accuracy or cable/motor loads.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'shared'))
from cdpr_geometry import WS_MIN_X, WS_MAX_X, WS_MIN_Y, WS_MAX_Y
from airhockey.motion import CartState, advance

BOUNDS = np.array([WS_MIN_X, WS_MAX_X, WS_MIN_Y, WS_MAX_Y])
LOW = BOUNDS[[0, 2]]
HIGH = BOUNDS[[1, 3]]
CENTER = (LOW + HIGH) / 2
RADII = (HIGH - LOW) / 2 - 65  # extra clearance inside the active workspace
DT = .01
DELAY = .008
RAMP = .003
NAMES = ('slalom', 'rosette', 'starburst')


def smooth(u):
    return u**3 * (10 + u * (-15 + 6*u))


def reference(name, speed, accel, stretch=1.):
    """Smooth rest-to-rest references, sampled at the command frequency."""
    if name not in NAMES:
        raise ValueError(f'unknown pattern {name}')
    if name == 'starburst':
        points = [CENTER.copy()]
        for k in (0, 4, 2, 6, 1, 5, 3, 7):
            theta = k * np.pi / 4
            tip = CENTER + RADII * [np.cos(theta), np.sin(theta)]
            for dest, dwell in ((tip, .12), (CENTER, .08)):
                start = points[-1]
                distance = np.linalg.norm(dest-start)
                duration = max(1.875*distance/(.7*speed),
                               np.sqrt(5.774*distance/(.6*accel))) * stretch
                n = max(2, int(np.ceil(duration / DT)))
                points.extend(start + smooth(np.arange(1, n+1)/n)[:, None]*(dest-start))
                points.extend([dest.copy()] * int(np.ceil(dwell*stretch/DT)))
        return np.array(points)
    u = np.linspace(0, 1, 6001)

    def curve(u):
        theta = smooth(u) * (4*np.pi if name == 'slalom' else 2*np.pi)
        if name == 'slalom':
            # Out-and-back branches weave around three imaginary posts.
            xy = np.column_stack((np.sin(3*theta), np.cos(theta)))
        else:
            r = np.cos(5*theta)
            xy = np.column_stack((r*np.cos(theta), r*np.sin(theta)))
        return CENTER + xy * RADII

    p = curve(u)
    v = np.gradient(p, u, axis=0)
    a = np.gradient(v, u, axis=0)
    duration = max(np.linalg.norm(v, axis=1).max()/(.7*speed),
                   np.sqrt(np.linalg.norm(a, axis=1).max()/(.6*accel))) * stretch
    n = int(np.ceil(duration / DT))
    return curve(np.linspace(0, 1, n+1))


@dataclass
class Plan:
    name: str
    reference: np.ndarray
    commands: np.ndarray
    predicted: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    speed: float
    accel: float

    def summary(self):
        error = np.linalg.norm(self.predicted - self.reference, axis=1)
        return dict(name=self.name, duration_s=len(self.commands)*DT,
                    speed_cap_m_s=self.speed/1000, accel_cap_m_s2=self.accel/1000,
                    predicted_peak_speed_m_s=float(np.linalg.norm(self.velocity, axis=1).max()/1000),
                    predicted_peak_accel_m_s2=float(np.linalg.norm(self.acceleration, axis=1).max()/1000),
                    predicted_max_error_mm=float(error.max()),
                    predicted_rms_error_mm=float(np.sqrt(np.mean(error**2))))


def _step(cart, target, duration, speed, accel):
    n = round(duration/.0002)
    advance(cart, [target[0]], [target[1]], speed, accel, RAMP,
            duration/n, n, bounds=tuple(BOUNDS))


def _simulate(name, p, speed, accel):
    # Append a stationary tail for velocity-loop settling. Never teleport state.
    p = np.concatenate((p, np.repeat(p[-1:], 80, axis=0)))
    v = np.gradient(p, DT, axis=0)
    a = np.gradient(v, DT, axis=0)
    cart = CartState(1)
    cart.reset(*p[0])
    old = p[0].copy()
    commands, positions, velocities, accelerations = [], [p[0]], [[0, 0]], [[0, 0]]
    for i in range(len(p)-1):
        # Old target persists during the measured order-of-magnitude serial delay.
        _step(cart, old, DELAY, speed, accel)
        pos = np.array([cart.x[0], cart.y[0]], dtype=float)
        vel = np.array([cart.vx[0], cart.vy[0]], dtype=float)
        # This target is held for a FULL command period (the next command
        # also incurs the delay). Its midpoint is t + delay + period/2.
        j = min(i+2, len(p)-1)
        fraction = (DELAY + DT/2 - DT)/DT
        goal = p[i+1]*(1-fraction) + p[j]*fraction
        goal_v = v[i+1]*(1-fraction) + v[j]*fraction
        goal_a = a[i+1]*(1-fraction) + a[j]*fraction
        midpoint = pos + vel*DT/2 + goal_a*DT**2/8
        desired_v = goal_v + 2*RAMP*goal_a + 22*(goal-midpoint)
        norm = np.linalg.norm(desired_v)
        if norm > speed:
            desired_v *= speed/norm
        target = midpoint + desired_v*np.linalg.norm(desired_v)/(1.28*accel)
        if np.linalg.norm(goal_v) < .1 and np.linalg.norm(goal_a) < 1:
            # Let the firmware park exactly during deliberate stops. Inverse
            # velocity guidance near zero only amplifies float quantization.
            target = goal.copy()
        target = np.clip(target, LOW, HIGH)
        _step(cart, target, DT-DELAY, speed, accel)
        commands.append(target)
        positions.append([cart.x[0], cart.y[0]])
        velocities.append([cart.vx[0], cart.vy[0]])
        accelerations.append([cart.ax[0], cart.ay[0]])
        old = target
    return Plan(name, p, np.array(commands), np.array(positions),
                np.array(velocities), np.array(accelerations), speed, accel)


def plan_pattern(name, speed=4000., accel=30000.):
    if not (np.isfinite(speed) and 0 < speed <= 12000 and
            np.isfinite(accel) and 0 < accel <= 60000):
        raise ValueError('caps must be finite: speed (0, 12] m/s; accel (0, 60] m/s²')
    for stretch in (1., 1.2, 1.5, 2., 3.):
        plan = _simulate(name, reference(name, speed, accel, stretch), speed, accel)
        metrics = plan.summary()
        if (metrics['predicted_max_error_mm'] <= 5 and
                np.all(plan.predicted >= LOW) and np.all(plan.predicted <= HIGH) and
                np.linalg.norm(plan.velocity, axis=1).max() <= speed*1.001 and
                np.linalg.norm(plan.acceleration, axis=1).max() <= accel*1.001):
            return plan
    raise RuntimeError(f'{name}: could not validate trajectory at these caps')


class LoadTelemetryNotReady(RuntimeError):
    """May be awaited at a stationary setup boundary, never during motion."""


def check_load(snapshot, now, stop_pct=85., *, not_before=None):
    """Fail closed on stale/missing fast RMS; check slow RMS where supported."""
    sample = snapshot.get('sample') or {}
    seen, peak, pending = set(), 0., []
    if sample.get('context', {}).get('fault'):
        raise RuntimeError('motor fault reported')
    for motor in sample.get('motors', []):
        node = motor.get('node')
        if type(node) is not int or not 0 <= node < 4:
            continue
        for key in ('rms_pct', 'rms_slow_pct'):
            item = motor.get(key, {})
            value, end = item.get('value'), item.get('end')
            if item.get('valid') is not True:
                continue
            if (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or
                    not isinstance(end, (int, float)) or not np.isfinite(end)):
                raise RuntimeError(f'motor {node}: invalid {key}')
            # A reported overload/fault is never a reason to wait and retry,
            # even if some other channel is still awaiting its next sample.
            peak = max(peak, value)
            if not 0 <= now-end <= .5:
                pending.append(f'motor {node}: stale {key} (age {now-end:.3f}s; maximum 0.500s)')
                continue
            if not_before is not None and end < not_before:
                pending.append(f'motor {node}: awaiting post-setup {key}')
                continue
            if key == 'rms_pct':
                seen.add(node)
    if peak >= stop_pct:
        raise RuntimeError(f'RMS load {peak:.1f}% reached demo stop threshold {stop_pct:.1f}%')
    if pending:
        raise LoadTelemetryNotReady(pending[0])
    if seen != {0, 1, 2, 3}:
        raise LoadTelemetryNotReady('fresh RMS telemetry required for all four motors (master --load-hz 10)')
    return peak


def write_preview(plans, path):
    """Standalone animated HTML; no server, camera or hardware connection."""
    data = [dict(**p.summary(), reference=p.reference[::2].round(2).tolist(),
                 predicted=p.predicted[::2].round(2).tolist()) for p in plans]
    html = '''<!doctype html><meta charset="utf-8"><title>Motor demos · offline preview</title>
<style>body{background:#111927;color:#eee;font:17px system-ui;max-width:1000px;margin:35px auto;padding:16px}
canvas{background:#182436;width:100%;border-radius:18px}button,select,input{font:inherit;margin:8px;padding:8px}
#stats{white-space:pre-wrap}small{color:#aabbcf}</style>
<h1>Motor choreography</h1><p>Predetermined paths through the firmware motion profile.</p>
<select id="pick"></select><button id="play">Pause</button><input id="scrub" type="range" min="0" max="1" step=".001" value="0">
<canvas id="canvas" width="1000" height="670"></canvas><p id="stats"></p>
<small>Gold: reference · cyan: predicted firmware motion. This is an offline controller prediction,
not a measurement of physical tracking precision. No hardware connection.</small>
<script>const plans=DATA, bounds=BOUNDS;let idx=0,t=0,playing=true,last=null;
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),pick=document.querySelector('#pick'),scrub=document.querySelector('#scrub');
plans.forEach((p,i)=>pick.add(new Option(p.name,i)));pick.onchange=()=>{idx=+pick.value;t=0;last=null;draw();};
document.querySelector('#play').onclick=e=>{playing=!playing;last=null;e.target.textContent=playing?'Pause':'Play';draw();};
scrub.oninput=()=>{t=+scrub.value*plans[idx].duration_s;last=null;draw();};
// Rotate table frame so the three-post slalom runs across the screen.
function xy(p){return [70+(p[1]-bounds[2])/(bounds[3]-bounds[2])*860,600-(p[0]-bounds[0])/(bounds[1]-bounds[0])*530];}
function line(points,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{const [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
function draw(){const p=plans[idx];
ctx.clearRect(0,0,1000,670);ctx.strokeStyle='#53647a';ctx.strokeRect(70,70,860,530);
line(p.reference,'#b79643',2);const n=Math.max(0,Math.min(p.predicted.length-1,Math.floor(t/p.duration_s*(p.predicted.length-1))));
line(p.predicted.slice(Math.max(0,n-80),n+1),'#55dcff',5);const [x,y]=xy(p.predicted[n]);
ctx.fillStyle='#55dcff';ctx.beginPath();ctx.arc(x,y,13,0,2*Math.PI);ctx.fill();
if(p.name==='slalom'){ctx.fillStyle='#ff9078';[-.75,0,.75].forEach(k=>{const [x,y]=xy([(bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2+k*((bounds[3]-bounds[2])/2-65)]);ctx.beginPath();ctx.arc(x,y,7,0,2*Math.PI);ctx.fill();});}
scrub.value=t/p.duration_s;document.querySelector('#stats').textContent=`${p.name} · ${p.duration_s.toFixed(1)} s\\nPredicted peak: ${p.predicted_peak_speed_m_s.toFixed(2)} m/s · ${p.predicted_peak_accel_m_s2.toFixed(1)} m/s²\\nPredicted path error: ${p.predicted_rms_error_mm.toFixed(2)} mm RMS / ${p.predicted_max_error_mm.toFixed(2)} mm max`;
}
// Start from the first animation timestamp, not performance.now(): a frame's
// timestamp can precede script execution. Negative time used to index -1 and
// kill the loop. Clamp long background-tab gaps rather than skipping ahead.
function frame(now){if(playing && last!==null)t=(t+Math.max(0,Math.min(.1,(now-last)/1000)))%plans[idx].duration_s;
last=now;draw();requestAnimationFrame(frame);}
document.addEventListener('visibilitychange',()=>{last=null;});
draw();requestAnimationFrame(frame);</script>'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html.replace('DATA', json.dumps(data)).replace('BOUNDS', json.dumps(BOUNDS.tolist())))
