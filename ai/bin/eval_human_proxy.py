#!/usr/bin/env python3
"""Human-proxy benchmark: how a candidate does against people. Simulation only.

The candidate (a checkpoint path or an `expert...` spec, loaded through
eval_neural_player.load) always plays the robot's body and seat at the fixed
production limits; the far end is a panel of scripted PEOPLE
(airhockey/human_proxy.py) on a human-capable body that reaches the centre
line. Sensing and rules are the live-like ones: the report estimator,
continuous rallies (no shot clock or referee resets except for a dead puck out
of everyone's reach), initial motor load 0.4 and thermal gain 1.3.

Panel tiers (--tier): `club`, matched to the people who played the robot (a
deliberate strike reaches the centre line at about their >= 3 m/s attacks,
p50 ~5.9, p90 ~8.0 m/s; intended release lognormal median 6.6, p90 9.0, clipped
3-11), and `elite` (intended release 8-14 m/s, faster hands, the same human
reaction time). Each tier is N games x S seconds with seeds shared across
candidates: game i is the same person (body, reaction, noise) with the same
sequence of intended shots for every candidate, on the same randomised table.

The headline number per tier is the candidate's goal share GF/(GF+GA) with a
bootstrap 95% CI over games, with GF/min and GA/min. The report also gives the
panel's own measured shots against the human TARGETS below, so a reader can
see how human-like the opponent was in the very games being scored.

`table` and `table_plus` are the two people who played
runs/teacher-calm-20261018-v4 (2026-10-29 and 2026-10-19; TABLE_TARGETS): they
also swing at moving pucks, bank off the robot's left rail and serve after a
goal. `--tier tables` runs both.

`--sim-profile table-20261029` plays the same games in the table-faithful sim
(airhockey/table_fidelity.py: measured mallet contact, mallet lift, camera
dropouts, command latency); the report then records the profile and, under
`lift`, the pucks that went under the robot's mallet and the goals conceded
that way. Without the flag the simulator and the report are the historical
ones.

`--shot-request` says who picks the candidate's route: `random` (default, as
trained: a uniform explicit request each time the puck enters its half),
`free` (no request) or `auto` (airhockey/shot_selector.py's governor: free
unless a route leaves its share band). Every report has `candidate_routes`
(shots by request and route, on-target and goals per route, the mix's
entropy) and, for a shot-conditioned candidate, `shot_request` (the requests
given and how often the route read back from the candidate's own observation
agreed with the env's).

The proxy never drives the robot and nothing here is a runtime controller.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_neural_player import load, set_initial_load
from airhockey.human_proxy import HumanProxy, STRIKE, VOLLEY, command_env, serve_env
from airhockey.neural_coordinates import CoordinateReference
from airhockey.neural_player import PhysicalHistory
from airhockey.neural_setup import checkpoint_environment
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.recorder import FrameData, Recorder
from airhockey.shot_selector import DEFAULTS as SELECTOR_DEFAULTS, RouteClassifier, ShotSelector
from airhockey.table_fidelity import LiftTally, profile_options

BASE = Path('runs/rail30-100-20261001-v1/agent.pt')
DT = 0.02
INITIAL_LOAD = 0.4
THERMAL_GAIN = 1.3
SHUTDOWN_LEVEL = 0.95          # the training shutdown; reported, not enforced
PUCK_SPEED_CAP = 16.0          # the elite panel releases up to 14 m/s
TIER_SEED = {'club': 101, 'elite': 202, 'table': 303, 'table_plus': 404}
TIER_GROUPS = {'both': ('club', 'elite'), 'tables': ('table', 'table_plus')}
ROUTE_NAMES = ('none', 'left_bank', 'right_bank', 'straight')   # env shot-route codes 0-3
REQUEST_NAMES = ('free', 'left_bank', 'right_bank', 'straight')  # the request a shot was made under
REQUEST_MODES = ('random', 'free', 'auto')
CROSSING_MIN_SPEED = 0.3       # along-table speed, as in the table analysis
OUTCOME_WINDOW_S = 4.0
TOUCH_GAP_S = 0.10             # contacts closer together than this are one touch

# What people did against the production checkpoint on the real table, from
# the run_policy replay logs of the live rail30-100-20261001-v1 sessions.
# Ranges are per-session values over the six sessions with human play
# analysed; pooled values and rates are over all eight of its sessions
# (2026-10-02 .. 10-11, 44.3 active minutes, 668 attacks, 600 robot
# crossings). Classified by centre-line crossings: an attack is the puck
# crossing toward the robot faster than 0.3 m/s along the table, "banked" if it
# meets a side rail on the robot's half before the robot touches it, and
# "conceded" if a goal follows before the next crossing and within 4 s. The
# first entries are calibration targets for the panel's own play; the last is
# for setting the production row of this benchmark beside reality.
TARGETS = dict(
    attack_speed_centre_line_m_s=dict(p50=[4.5, 5.7], p90=[6.9, 8.2], max_about=10.0,
                                      pooled_p50=5.12, pooled_p90=7.78),
    attack_bank_fraction=dict(range=[0.40, 0.60], pooled=0.485),
    attack_angle_from_axis_deg=dict(median=[16.0, 30.0], pooled=26.3, bank_median=36.7, straight_median=14.4),
    # Crossings of at least 3 m/s, the closest the logs come to deliberate shots.
    deliberate_attack_reference=dict(speed_m_s=dict(p25=4.45, p50=5.91, p75=7.2, p90=7.96, max=10.79),
                                     bank_fraction=0.497, angle_median_deg=23.9,
                                     bank_angle_median_deg=34.1, straight_angle_median_deg=13.1),
    human_mallet_speed_m_s=dict(p99=[1.8, 1.9], p999=[4.0, 5.5]),
    conceded_goals=dict(single_bank_share_analysed='6 of 7', bank_share_all='28 of 44',
                        speed_m_s=[3.7, 6.5], midline_to_goal_s=[0.25, 0.35]),
    production_against_people=dict(
        goals_for_per_min=1.11, goals_against_per_min=[0.70, 0.99], goal_share=[0.53, 0.61],
        attacks_per_min=15.1, attack_conceded_rate=0.066,
        attack_conceded_rate_by_speed={'0-3': 0.013, '3-4.5': 0.028, '4.5-6': 0.049, '6-8': 0.125, '8+': 0.143},
        robot_crossing_scored_rate_by_speed={'0-3': 0.011, '3-4.5': 0.062, '4.5-6': 0.148, '6-7': 0.120,
                                             '7-8': 0.111},
        robot_crossing_speed_m_s=dict(p50=5.08, p90=6.92)),
)
# The two people who played runs/teacher-calm-20261018-v4, from its replay
# logs with strikes separated from end-rail rebounds (13.9 unpaused minutes on
# 2026-10-29; 3.1 on 2026-10-19 plus that evening's production session for the
# set shots). A "set shot" is a strike on a puck at <= 1.5 m/s, the proxy's
# STRIKE; an "active counter" is a swing (mallet > 2 m/s) at a faster puck,
# its VOLLEY; a "strike" is any crossing the person sent back, a puck off a
# block included. Each entry is [measured, low, high]: the range a panel of
# this tier is checked against (calibration.checks), wide where the count is
# small. The other entries are for reading beside calibration.table.
# set_share_by_arrival_speed is, of the crossings toward the person that they
# sent back (serves and end-rail rebounds apart), the share sent back with a set
# shot, by the speed the puck arrived at (ROBOT_SHOT_BINS; n = 71/44/51/40/22
# and, over all three 2026-10-19 sessions, 28/21/25/31/7): what the tiers'
# CUSHION is read against. The panel matches it up to 6 m/s and falls short
# above (0.05-0.12 at 6-8 m/s): such a shot is on the person within about
# their reaction time, and the panel gives with nothing it has not yet seen.
# Set shots per minute also come out short of the people's under
# --sim-profile because the sim's v4 sends over fewer slow pucks than the
# table's did (returned arrivals under 2 m/s: 5.1 a minute on 2026-10-29,
# 2-3.5 here) and those are the ones a person nearly always sets up.
ROBOT_SHOT_OUTCOMES = {      # both people pooled, n = 471, by the robot shot's line speed
    '0-2': dict(first_touch_return=0.12, controlled=0.87, goal=0.0, rebound=0.0),
    '2-4': dict(first_touch_return=0.56, controlled=0.31, goal=0.05, rebound=0.04),
    '4-6': dict(first_touch_return=0.49, controlled=0.21, goal=0.08, rebound=0.20),
    '6-8': dict(first_touch_return=0.45, controlled=0.18, goal=0.20, rebound=0.11),
    '8+': dict(first_touch_return=0.21, controlled=0.15, goal=0.29, rebound=0.14),
}
TABLE_TARGETS = dict(
    table=dict(
        sessions='2026-10-29 (20261029-18*)',
        set_shot_line_speed_p50_m_s=[7.00, 6.3, 7.7], set_shot_line_speed_p90_m_s=[9.16, 8.2, 10.1],
        set_shot_line_speed_max_m_s=[10.78, 9.0, 12.0],
        set_shot_bank_fraction=[0.46, 0.34, 0.58], bank_x0_rail_share=[0.87, 0.60, 0.97],
        straight_set_shot_on_target=[0.38, 0.26, 0.50],
        set_shot_hold_p50_s=[0.72, 0.45, 1.10],
        active_counter_share_of_strikes=[0.125, 0.06, 0.20], active_counter_release_p50_m_s=[6.8, 5.2, 8.4],
        moving_strike_share=[0.46, 0.34, 0.58],
        mallet_speed_p99_m_s=[4.61, 3.2, 6.0], mallet_speed_p999_m_s=[6.46, 5.0, 7.6],
        set_shots_per_min=11.1, person_strikes_per_min=20.6, active_counters_per_min=2.6,
        serve_share_of_set_shots=0.21, serve_to_strike_s=dict(p10=0.62, p50=1.77, p90=4.84),
        crossing_line_speed_m_s=dict(p10=1.99, p50=5.43, p90=8.66, max=10.78),
        robot_shot_outcomes_pooled=ROBOT_SHOT_OUTCOMES, active_counter_by_robot_shot_speed=(0.15, 0.30, 0.05, 0.0, 0.0),
        set_share_by_arrival_speed=(0.83, 0.34, 0.24, 0.17, 0.18),
        conceded='0 of 56 serves; 1/71 straight and 1/81 banked set shots, 3/73 moving strikes (>= 3 m/s)'),
    table_plus=dict(
        sessions='2026-10-19 (20261019-214911, -220855; set shots also -213539)',
        set_shot_line_speed_p50_m_s=[7.68, 6.6, 8.5], set_shot_line_speed_p90_m_s=[9.68, 8.4, 10.6],
        set_shot_line_speed_max_m_s=[11.13, 9.3, 12.0],
        set_shot_bank_fraction=[0.27, 0.12, 0.38], bank_x0_rail_share=[0.69, 0.45, 0.92],
        straight_set_shot_on_target=[0.51, 0.38, 0.64],
        set_shot_hold_p50_s=[0.78, 0.45, 1.10],
        active_counter_share_of_strikes=[0.21, 0.12, 0.30], active_counter_release_p50_m_s=[5.75, 4.4, 7.2],
        moving_strike_share=[0.44, 0.32, 0.56],
        mallet_speed_p99_m_s=[4.79, 3.3, 6.2], mallet_speed_p999_m_s=[6.54, 5.0, 8.1],
        set_shots_per_min=13.9, person_strikes_per_min=25.1, active_counters_per_min=5.3,
        serve_share_of_set_shots=0.35, serve_to_strike_s=dict(p10=0.84, p50=1.50, p90=2.41),
        crossing_line_speed_m_s=dict(p10=2.76, p50=6.23, p90=9.23, max=11.13),
        robot_shot_outcomes_pooled=ROBOT_SHOT_OUTCOMES, active_counter_by_robot_shot_speed=(0.15, 0.50, 0.30, 0.18, 0.05),
        set_share_by_arrival_speed=(1.0, 0.24, 0.24, 0.16, 0.14),
        conceded='5/34 straight set shots, 0/7 banked, 0/25 moving (>= 3 m/s); 5/26 at 6-8 m/s, 0/19 at 8+'),
)
SPEED_BINS = [(0, 3), (3, 4.5), (4.5, 6), (6, 8), (8, 100)]
ROBOT_SPEED_BINS = [(0, 3), (3, 4.5), (4.5, 6), (6, 7), (7, 8), (8, 100)]
ROBOT_SHOT_BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 100)]


def bootstrap(values_for, values_against, *, seed, samples=2000):
    """95% percentile CIs over games for goal share, GF and GA per game."""
    gf, ga = np.asarray(values_for, float), np.asarray(values_against, float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(gf), (samples, len(gf)))
    f, a = gf[pick].sum(1), ga[pick].sum(1)
    share = f / np.maximum(f + a, 1)
    return dict(goal_share=np.percentile(share, [2.5, 97.5]).tolist(),
                goals_for_per_game=np.percentile(f / len(gf), [2.5, 97.5]).tolist(),
                goals_against_per_game=np.percentile(a / len(gf), [2.5, 97.5]).tolist())


def percentiles(values, qs=(10, 50, 90, 100)):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return dict(zip([f'p{q}' if q < 100 else 'max' for q in qs], np.percentile(values, qs).round(3).tolist()))


def speed_bin_rates(events, outcome, bins):
    table = {}
    for lo, hi in bins:
        sub = [e for e in events if lo <= e['speed'] < hi]
        label = f'{lo:g}-{hi:g}' if hi < 100 else f'{lo:g}+'
        table[label] = dict(n=len(sub), rate=(sum(e['outcome'] == outcome for e in sub) / len(sub)) if sub else None)
    return table


def calibration_checks(c, tier):
    """Measured panel statistics against the people's, as [value, target, inside]."""
    def within(value, lo, hi):
        return [None if value is None else round(float(value), 4), [round(lo, 4), round(hi, 4)],
                None if value is None else bool(lo <= value <= hi)]
    strike = c['strike_attacks']['speed_centre_line_m_s'] or {}
    body = dict(
        reach_nearest_robot_goal_y_m=within(c['mallet_nearest_robot_goal_y_m'], c['reach_limit_y_m'] - 1e-6, 2.0),
        body_accel_over_drawn_cap=within(c['body_peak_accel_over_drawn_cap'], 0.0, 1.0 + 1e-6))
    if tier in TABLE_TARGETS:
        measured = c['table']
        return dict(body, **{name: within(measured.get(name), target[1], target[2])
                             for name, target in TABLE_TARGETS[tier].items() if isinstance(target, list)})
    if tier != 'club':
        intended = c['strike_intended_m_s'] or {}
        return dict(body, intended_release_p10_m_s=within(intended.get('p10'), 8.0, 14.0),
                    intended_release_p90_m_s=within(intended.get('p90'), 8.0, 14.0))
    t = TARGETS
    attack = c['attack_speed_centre_line_m_s'] or {}
    ref = t['deliberate_attack_reference']['speed_m_s']
    return dict(
        body,
        attack_speed_p50_m_s=within(attack.get('p50'), *t['attack_speed_centre_line_m_s']['p50']),
        attack_speed_p90_m_s=within(attack.get('p90'), *t['attack_speed_centre_line_m_s']['p90']),
        attack_speed_max_m_s=within(attack.get('max'), 8.0, 12.0),
        attack_bank_fraction=within(c['attack_bank_fraction_geometric'], *t['attack_bank_fraction']['range']),
        attack_angle_median_deg=within(c['attack_angle_median_deg'], *t['attack_angle_from_axis_deg']['median']),
        strike_speed_p50_m_s=within(strike.get('p50'), 0.9 * ref['p50'], 1.1 * ref['p50']),
        strike_speed_p90_m_s=within(strike.get('p90'), 0.9 * ref['p90'], 1.1 * ref['p90']),
        strike_angle_median_deg=within(c['strike_attacks']['angle_median_deg'],
                                       0.75 * t['deliberate_attack_reference']['angle_median_deg'],
                                       1.25 * t['deliberate_attack_reference']['angle_median_deg']),
        strike_bank_fraction=within(c['strike_attacks']['bank_fraction'], 0.40, 0.60),
        mallet_speed_p99_m_s=within(c['mallet_speed_m_s']['p99'], *t['human_mallet_speed_m_s']['p99']),
        mallet_speed_p999_m_s=within(c['mallet_speed_m_s']['p999'], *t['human_mallet_speed_m_s']['p999']),
        conceded_midline_to_goal_p50_s=within((c['conceded_midline_to_goal_s'] or {}).get('p50'),
                                              *t['conceded_goals']['midline_to_goal_s']))


class MatchTracker:
    """Evaluator-side bookkeeping from engine truth. Never feeds a player."""

    def __init__(self, env, proxy, record_games=0):
        self.env, self.proxy, self.e = env, proxy, env.engine
        n = env.n_envs
        self.n = n
        self.last_touch = np.full(n, -1)          # -1 none (serve/relaunch), 0 robot, 1 proxy
        self.last_route = np.zeros((2, n), int)   # env route of each side's last contact
        self.last_speed = np.zeros((2, n))
        self.last_mode = np.zeros(n, int)        # the person's mode at its last touch
        self.slow_since = np.full(n, np.nan)     # puck first slow on the person's half (hold time)
        self.served = np.zeros(n, bool)
        self.shot_events = []
        self.crossings = []                       # resolved centre-line crossings
        self.open = [None] * n                    # the crossing still awaiting its outcome
        self.goals = []
        self.prev_y = self.e.puck_y.copy()
        self.prev_score = (self.e.score_agent.copy(), self.e.score_opponent.copy())
        self.mallet_speed = []
        self.mallet_y_min = np.inf
        self.body_accel_over_cap = 0.0              # peak |dv/dt| of the commanded body / its drawn cap
        self.stall_seconds = 0.0
        self.shutdown_seconds = np.zeros(n)
        self.record_games = record_games
        self.events = [[] for _ in range(record_games)]
        self.frames = [Recorder() for _ in range(record_games)]
        free = env.base._opp_dyn_free
        self._prev_body_v = np.column_stack((free["vx"], free["vy"]))
        original = env.engine.contact_callback

        def contact(event):
            side = int(event['body'] != 'agent')
            ids = event['indices']
            before = env.shot_count[side, ids].copy()
            aimed_before = env.aimed_count[side, ids].copy()
            original(event)
            out = event['outgoing_before_speed_cap']
            speed = np.minimum(np.linalg.norm(out, axis=1), env.cfg.max_puck_speed)
            self.last_touch[ids] = side
            self.last_route[side, ids] = env.last_shot_route[side, ids]
            self.last_speed[side, ids] = speed
            if side:
                self.last_mode[ids] = proxy.mode[ids]
                # The person's touches of a robot shot, one per TOUCH_GAP_S of contact.
                for i in ids:
                    c = self.open[int(i)]
                    if c is not None and c['toward'] == 'proxy':
                        at = float(self.e.time[i])
                        c['person_touches'] += at - c['touched_at'] > TOUCH_GAP_S
                        c['touched_at'] = at
            for j in np.flatnonzero(env.shot_count[side, ids] > before):
                i = int(ids[j])
                local_y = env.cfg.height - self.e.puck_y[i] if side else self.e.puck_y[i]
                striking = bool(side and proxy.mode[i] == STRIKE)
                self.shot_events.append(dict(
                    side=side, game=i, time=float(self.e.time[i]), speed=float(speed[j]),
                    volley=bool(side and proxy.mode[i] == VOLLEY), serve=bool(side and self.served[i]),
                    hold_s=float(self.e.time[i] - self.slow_since[i]) if striking else None,
                    route=ROUTE_NAMES[int(env.last_shot_route[side, i])],
                    request=REQUEST_NAMES[int(env.last_shot_request[side, i])],
                    aimed=bool(env.aimed_count[side, i] > aimed_before[j]), local_y=float(local_y),
                    strike=striking,
                    intended_speed=float(proxy.plan_speed[i, proxy.plan_index[i] % proxy.PLANS])
                    if striking else None,
                    intended_route=int(proxy.strike_route[i]) if striking else None))
                if side and i < self.record_games:
                    self.events[i].append((float(self.e.time[i]),
                                           f"Person strikes: {ROUTE_NAMES[int(env.last_shot_route[1, i])]} {speed[j]:.1f} m/s"))
                if side:
                    self.served[i] = False

        env.engine.contact_callback = contact
        relaunch = env.base._relaunch

        def tracked_relaunch(mask, *args, **kwargs):
            self.last_touch[np.asarray(mask, bool)] = -1
            for i in np.flatnonzero(mask):
                if i < self.record_games:
                    self.events[i].append((float(self.e.time[i]), 'Dead puck relaunched'))
            return relaunch(mask, *args, **kwargs)

        env.base._relaunch = tracked_relaunch

    def _resolve(self, i, outcome, now):
        c = self.open[i]
        if c is None:
            return
        c['outcome'] = outcome
        c['resolved_after_s'] = now - c['time']
        self.crossings.append(c)
        self.open[i] = None

    def after_step(self, tick):
        env, e, n = self.env, self.e, self.n
        now = e.time.copy()
        scored = e.score_agent - self.prev_score[0]
        conceded = e.score_opponent - self.prev_score[1]
        for i in np.flatnonzero((scored > 0) | (conceded > 0)):
            robot_scored = bool(scored[i] > 0)
            c = self.open[i]
            toward_scorer_goal = c is not None and c['toward'] == ('proxy' if robot_scored else 'robot')
            goal = dict(game=int(i), time=float(now[i]), by='robot' if robot_scored else 'proxy',
                        last_touch=('robot', 'proxy')[int(self.last_touch[i])] if self.last_touch[i] >= 0 else 'none')
            if toward_scorer_goal:
                goal.update(midline_speed=c['speed'], midline_to_goal_s=float(now[i] - c['time']),
                            route=c['route'], banked=c['banked'], angle_deg=c['angle_deg'])
                self._resolve(i, 'goal', now[i])
            else:
                self._resolve(i, 'other', now[i])
            self.goals.append(goal)
            if i < self.record_games:
                text = 'Goal: robot' if robot_scored else f"Goal: person ({goal.get('route', 'unclear')})"
                self.events[i].append((float(now[i]), text))
            self.last_touch[i] = -1
        # Centre-line crossings (engine truth), with the real-table thresholds.
        y = e.puck_y
        down = (self.prev_y >= 1.0) & (y < 1.0) & (-e.puck_vy > CROSSING_MIN_SPEED)
        up = (self.prev_y <= 1.0) & (y > 1.0) & (e.puck_vy > CROSSING_MIN_SPEED)
        for i in np.flatnonzero(down | up):
            if self.open[i] is not None:
                # How a robot shot came back: off the person's mallet in this mode, or untouched.
                self.open[i]['returned_by'] = (self.proxy.mode_names[int(self.last_mode[i])]
                                               if self.last_touch[i] == 1 else None)
                self._resolve(i, 'returned', now[i])
            toward = 'robot' if down[i] else 'proxy'
            attacker = 1 if toward == 'robot' else 0
            speed = float(np.hypot(e.puck_vx[i], e.puck_vy[i]))
            self.open[i] = dict(
                game=int(i), time=float(now[i]), toward=toward, speed=speed,
                angle_deg=float(np.degrees(np.arctan2(abs(e.puck_vx[i]), abs(e.puck_vy[i])))),
                attack=bool(self.last_touch[i] == attacker),
                person_mode=self.proxy.mode_names[int(self.last_mode[i])] if toward == "robot" else None,
                route=ROUTE_NAMES[int(self.last_route[attacker, i])] if self.last_touch[i] == attacker else 'none',
                vx_sign=float(np.sign(e.puck_vx[i])), banked=False, defender_touched=False,
                person_touches=0, touched_at=-np.inf)
        # Rail bank on the defender's half before the defender touches it.
        for i in range(n):
            c = self.open[i]
            if c is None:
                continue
            on_defender_half = (y[i] < 1.0) if c['toward'] == 'robot' else (y[i] > 1.0)
            defender = 0 if c['toward'] == 'robot' else 1
            if self.last_touch[i] == defender:
                c['defender_touched'] = True
            sign = np.sign(e.puck_vx[i])
            if (on_defender_half and not c['defender_touched'] and sign != 0 and c['vx_sign'] != 0
                    and sign != c['vx_sign'] and abs(e.puck_vx[i]) > 0.2):
                c.setdefault('first_rail', 'x0' if e.puck_x[i] < env.cfg.width / 2 else 'x1')
                c['banked'] = True
            if sign != 0:
                c['vx_sign'] = sign
            if now[i] - c['time'] > OUTCOME_WINDOW_S:
                self._resolve(i, 'kept>4s', now[i])
        self.prev_y = y.copy()
        self.prev_score = (e.score_agent.copy(), e.score_opponent.copy())
        slow = (y > 1.0) & (np.hypot(e.puck_vx, e.puck_vy) < 1.5)
        self.slow_since = np.where(slow, np.where(np.isnan(self.slow_since), now, self.slow_since), np.nan)
        # The person's mallet at 50 Hz: speed and reach as the table saw them,
        # and the commanded body's acceleration against the cap it was drawn
        # (the centre-line clamp stops the mallet itself like a wall).
        self.mallet_speed.append(np.hypot(e.paddle_opp_vx, e.paddle_opp_vy))
        free = env.base._opp_dyn_free
        body_v = np.column_stack((free['vx'], free['vy']))
        ratio = np.linalg.norm(body_v - self._prev_body_v, axis=1) / DT / self.proxy.max_accel
        self.body_accel_over_cap = max(self.body_accel_over_cap, float(ratio.max()))
        self._prev_body_v = body_v
        self.mallet_y_min = min(self.mallet_y_min, float(e.paddle_opp_y.min()))
        # Candidate stall (same definition as eval_neural_player.games) and heat.
        if tick % 5 == 0:
            puck = np.column_stack((e.puck_x, e.puck_y))
            reachable = np.linalg.norm(puck - np.clip(puck, env.decoder.low, env.decoder.high), axis=1) <= (
                env.cfg.puck_radius + env.cfg.paddle_radius)
            stalled = (puck[:, 1] < 1.0) & reachable & (np.hypot(e.puck_vx, e.puck_vy) < .05) & (env.base._t_side > 4)
            self.stall_seconds += .1 * stalled.sum()
        self.shutdown_seconds += DT * (env.loads[0].levels.max(axis=(1, 2)) >= SHUTDOWN_LEVEL)
        for i in range(self.record_games):
            text = ''
            if self.events[i] and now[i] - self.events[i][-1][0] < .8:
                text = self.events[i][-1][1]
            self.frames[i].record(FrameData(
                time=float(now[i]), puck_x=float(e.puck_x[i]), puck_y=float(e.puck_y[i]),
                puck_vx=float(e.puck_vx[i]), puck_vy=float(e.puck_vy[i]),
                agent_x=float(e.paddle_agent_x[i]), agent_y=float(e.paddle_agent_y[i]),
                opponent_x=float(e.paddle_opp_x[i]), opponent_y=float(e.paddle_opp_y[i]),
                score_agent=int(e.score_agent[i]), score_opponent=int(e.score_opponent[i]),
                shot_type_agent=int(env.base._shot_type[i]), rally_event=text))
        return scored, conceded

    def serve(self, goal):
        """A serving tier's restart after the goals of this step (human_proxy.serve_env)."""
        serve_env(self.env, self.proxy, goal)
        self.served |= goal
        self.prev_y = self.e.puck_y.copy()
        self.slow_since[goal] = np.nan

    def finish(self):
        for i in range(self.n):
            self._resolve(i, 'unfinished', float(self.e.time[i]))


class RequestGovernor:
    """`--shot-request free | auto`: the candidate's request comes from a
    `shot_selector.ShotSelector` per game instead of the env's uniform draw,
    at the same moment (the puck entering the robot's half). The realised
    route is read back by `RouteClassifier` from the candidate's own
    observation, as on the table, never from engine truth; `report()` says
    how often that reading agreed with the env's route of the same shot."""

    def __init__(self, env, mode, seed, selector_options=None):
        self.env, self.mode = env, mode
        n = env.n_envs
        self.selectors = [ShotSelector(mode, seed=seed + 7919 * i, **(selector_options or {})) for i in range(n)]
        self.classifiers = [RouteClassifier(width=env.cfg.width, height=env.cfg.height,
                                            puck_radius=env.cfg.puck_radius, paddle_radius=env.cfg.paddle_radius)
                            for _ in range(n)]
        self.confusion = np.zeros((4, 4), int)      # [env route, classified route]
        self.how = {}
        self.tick = 0
        if mode != 'random':
            draw = env.base._draw_shot_types

            def governed(mask, agent):
                if not agent:
                    return draw(mask, agent)
                for i in np.flatnonzero(mask):
                    env.base._shot_type[i] = self.selectors[i].request()
            env.base._draw_shot_types = governed

    def after_step(self, obs):
        """`obs` is the candidate's newest frame: puck, puck velocity / 6, own mallet."""
        self.tick += 1
        t = self.tick * DT
        for i, classifier in enumerate(self.classifiers):
            row = obs[i]
            shot = classifier.update(t, row[0:2], row[2:4] * 6, row[4:6])
            if shot is None:
                continue
            self.how[shot['how']] = self.how.get(shot['how'], 0) + 1
            # The env's route of the robot's last contact: a second touch re-routes the shot for both.
            self.confusion[int(self.env.last_shot_route[0, i]), shot['route']] += 1
            if shot['how'] in ('bounce', 'traced'):
                self.selectors[i].record(shot['route'], int(self.env.base._shot_type[i]))

    def report(self, robot_crossings):
        decided = self.confusion[:, 1:]
        stats = [s.stats() for s in self.selectors]
        requests = np.sum([[s['requests'][k] for k in ('none', 'left', 'right', 'straight')] for s in stats], axis=0)
        reasons = {k: int(sum(s['reasons'][k] for s in stats)) for k in stats[0]['reasons']}
        shots = np.sum([s.shots for s in self.selectors], axis=0)
        explicit = shots[1:, 1:]
        return dict(
            mode=self.mode,
            selector=dict(window=self.selectors[0].window, floor=self.selectors[0].floor,
                          ceiling=self.selectors[0].ceiling, min_shots=self.selectors[0].min_shots,
                          explore=self.selectors[0].explore) if self.mode == 'auto' else None,
            possessions=int(requests.sum()),
            requests=dict(zip(REQUEST_NAMES, requests.tolist())),
            free_share=float(requests[0] / max(requests.sum(), 1)),
            request_reasons=reasons,
            honoured_share_classified=float(np.trace(explicit) / explicit.sum()) if explicit.sum() else None,
            classifier=dict(
                resolved=int(decided.sum()), dropped=int(self.confusion[:, 0].sum()), how=self.how,
                robot_crossings=robot_crossings,
                agrees_with_env_route=float(np.trace(decided[1:]) / decided[1:].sum()) if decided[1:].sum() else None,
                confusion_env_route_by_classified=self.confusion.tolist()))


def route_summary(shots, goals, crossings):
    """The candidate's shots by the request they were made under and the
    route the env gave them (the reward's definition, from the launch).

    Per route: shots, share of the routed shots, on-target rate, goals (a
    robot goal goes to the robot's last shot in the OUTCOME_WINDOW_S before it) and
    goals per shot; crossings of the centre line and goals per crossing.
    `entropy_bits` is of the route mix (1.585 = even thirds, 0 = one route).
    """
    scored = set()
    by_game = {}
    for k, s in enumerate(shots):
        by_game.setdefault(s['game'], []).append((s['time'], k))
    for g in goals:
        if g['by'] != 'robot':
            continue
        before = [k for t, k in by_game.get(g['game'], []) if g['time'] - OUTCOME_WINDOW_S <= t <= g['time']]
        if before:
            scored.add(before[-1])

    def block(rows):
        routed = [(k, s) for k, s in rows if s['route'] != 'none']
        table = {}
        for name in ROUTE_NAMES[1:]:
            mine = [(k, s) for k, s in routed if s['route'] == name]
            n = len(mine)
            table[name] = dict(
                shots=n, share=round(n / len(routed), 4) if routed else None,
                on_target_rate=round(sum(s['aimed'] for _, s in mine) / n, 4) if n else None,
                goals=sum(k in scored for k, _ in mine),
                goal_rate=round(sum(k in scored for k, _ in mine) / n, 4) if n else None,
                mean_speed_m_s=round(float(np.mean([s['speed'] for _, s in mine])), 3) if n else None)
        share = np.array([table[name]['shots'] for name in ROUTE_NAMES[1:]], float)
        share = share / share.sum() if share.sum() else share
        positive = share[share > 0]
        return dict(shots=len(rows), unresolved=len(rows) - len(routed), routes=table,
                    goals=sum(k in scored for k, _ in rows),
                    entropy_bits=round(float(-(positive * np.log2(positive)).sum()), 4) if len(positive) else None)
    rows = list(enumerate(shots))
    result = dict(all=block(rows), free=block([r for r in rows if r[1]['request'] == 'free']),
                  requested=block([r for r in rows if r[1]['request'] != 'free']))
    explicit = [s for _, s in rows if s['request'] != 'free' and s['route'] != 'none']
    result['requested']['honoured_share'] = (round(sum(s['route'] == s['request'] for s in explicit) / len(explicit), 4)
                                             if explicit else None)
    result['requested']['by_request'] = {
        name: dict(shots=sum(s['request'] == name for s in explicit),
                   honoured=sum(s['request'] == name and s['route'] == name for s in explicit))
        for name in ROUTE_NAMES[1:]}
    result['crossings'] = {
        name: dict(n=sum(c['route'] == name for c in crossings),
                   goal_rate=round(sum(c['route'] == name and c['outcome'] == 'goal' for c in crossings)
                                   / max(sum(c['route'] == name for c in crossings), 1), 4))
        for name in ROUTE_NAMES[1:]}
    return result


def play_tier(net, tier, *, games, seconds, seed, options, stochastic=False, record=None,
              record_games=1, candidate='', sim_profile=None, observe=None, shot_request='random',
              selector_options=None):
    """One tier's games. `observe(env)` may return a callable run after every step
    (evaluator-side instrumentation such as playability.SimRecorder).
    `shot_request`: 'random' (the env's uniform draw per possession, as
    trained), 'free' (never a request) or 'auto' (shot_selector's governor)."""
    if shot_request not in REQUEST_MODES:
        raise ValueError(f'shot_request must be one of {REQUEST_MODES}')
    if shot_request != 'random' and not net.shot_conditioned:
        raise ValueError('a shot-request mode needs a shot-conditioned candidate')
    torch.manual_seed(seed)
    fidelity = profile_options(sim_profile)
    env = NeuralTrainingEnv(games, stage=3, seed=seed, games=True, **{**options, **fidelity}, report_sensing=True,
                            continuous_rallies=True, shot_conditioned=net.shot_conditioned)
    env.cfg.max_puck_speed = max(env.cfg.max_puck_speed, PUCK_SPEED_CAP)
    governor = RequestGovernor(env, shot_request, seed, selector_options) if net.shot_conditioned else None
    obs = env.reset(seed=seed, opponent='human_proxy')
    for model in env.loads:
        model.gain[:] = THERMAL_GAIN
    obs = set_initial_load(env, INITIAL_LOAD)
    proxy = HumanProxy(games, tier, seed=seed + TIER_SEED[tier], config=env.cfg)
    command_env(env, proxy, reset=True)
    record_games = min(record_games, games) if record else 0
    tracker = MatchTracker(env, proxy, record_games)
    lifts = LiftTally(env)
    watch = observe(env) if observe is not None else None
    history = PhysicalHistory(net.history)
    start = time.monotonic()
    for tick in range(round(seconds / DT)):
        command_env(env, proxy)
        history.append(obs)
        obs, *_ = env.step(net.act(history.for_policy(net), stochastic=stochastic))
        scored, conceded = tracker.after_step(tick)
        if governor is not None:
            governor.after_step(obs)
        lifts.after_step(conceded, scored, DT)
        if watch is not None:
            watch()
        if proxy.tier.serve_x_m is not None:
            tracker.serve((scored > 0) | (conceded > 0))
    tracker.finish()
    wall = time.monotonic() - start
    result = summarize(env, proxy, tracker, games=games, seconds=seconds, seed=seed, tier=tier)
    result['wall_seconds'] = round(wall, 1)
    if governor is not None:
        result['shot_request'] = governor.report(result['calibration']['robot_crossings'])
    if sim_profile is not None:
        result['lift'] = dict(lifts.report(), goals_against=result['goals_against'])
    if record:
        result['recordings'] = save_recordings(tracker, proxy, record, tier, candidate=candidate,
                                               seed=seed, options=options, env=env, seconds=seconds)
    return result


def robot_shot_outcomes(crossings):
    """What became of each robot shot, by its line speed: the table analysis' four outcomes.

    A first-touch return comes back off one touch of a mallet that was not
    striking a set puck; controlled is kept, or touched again before it is
    sent back; a rebound comes back off the end rail untouched.
    """
    def outcome(c):
        if c['outcome'] in ('goal', 'kept>4s'):
            return 'goal' if c['outcome'] == 'goal' else 'controlled'
        if c['outcome'] != 'returned':
            return None
        if c['returned_by'] is None:
            return 'rebound'
        return 'first_touch_return' if c['person_touches'] == 1 and c['returned_by'] != 'strike' else 'controlled'
    table = {}
    for lo, hi in ROBOT_SHOT_BINS:
        shots = [c for c in crossings if lo <= c['speed'] < hi and outcome(c) is not None]
        found = [outcome(c) for c in shots]
        table[f'{lo:g}-{hi:g}' if hi < 100 else f'{lo:g}+'] = dict(
            n=len(found), **{name: round(found.count(name) / len(found), 3) if found else None
                             for name in ('first_touch_return', 'controlled', 'goal', 'rebound')},
            # Of all of them: sent back by a counter-swing (volley_probability's own terms).
            active_counter=round(sum(c.get('returned_by') == 'volley' for c in shots) / len(shots), 3)
            if shots else None)
    return table


def set_share_by_arrival(crossings, minutes):
    """Of the crossings toward the person that they sent back, the share sent back with a set shot, by arrival speed."""
    table = {}
    for lo, hi in ROBOT_SHOT_BINS:
        arrived = [c for c in crossings if lo <= c['speed'] < hi]
        sent = [c for c in arrived if c['outcome'] == 'returned' and c['returned_by'] is not None]
        table[f'{lo:g}-{hi:g}' if hi < 100 else f'{lo:g}+'] = dict(
            arrivals_per_min=round(len(arrived) / minutes, 3), returned_per_min=round(len(sent) / minutes, 3),
            set_share=round(sum(c['returned_by'] == 'strike' for c in sent) / len(sent), 3) if sent else None)
    return table


def table_calibration(person, attacks, robot_crossings, arrivals, mallet, minutes, proxy):
    """The panel's play in the terms of TABLE_TARGETS (the table tiers only)."""
    def share(flags):
        return float(np.mean(flags)) if len(flags) else None

    def q(values, percent):
        return float(np.percentile(values, percent)) if len(values) else None
    sets = [s for s in person if s['strike']]
    volleys = [s for s in person if s['volley']]
    struck = [c for c in attacks if c['person_mode'] == 'strike']
    line = [c['speed'] for c in struck]
    banks = [c for c in struck if c['banked']]
    played = [c for c in attacks if c['person_mode'] in ('strike', 'volley')]
    holds = [s['hold_s'] for s in sets if not s['serve'] and s['hold_s'] is not None and np.isfinite(s['hold_s'])]
    return dict(
        set_shots=len(sets), active_counters=len(volleys), serves=int(proxy.serves.sum()),
        volley_attempts=int(proxy.volley_attempts.sum()),
        # Tries to stop a moving puck, those timed from a read, and the ones that left it slow.
        **{name: int(getattr(proxy, name).sum()) if proxy.cushioning else 0
           for name in ('cushion_attempts', 'braces', 'cushions')},
        set_shots_per_min=len(sets) / minutes,
        # Every crossing the person sent back, as the table analysis counts
        # strikes: set shots, counter-swings and pucks that came off a block.
        person_strikes_per_min=len(attacks) / minutes,
        active_counters_per_min=sum(c['person_mode'] == 'volley' for c in attacks) / minutes,
        passive_returns_per_min=sum(c['person_mode'] not in ('strike', 'volley') for c in attacks) / minutes,
        moving_strike_share=share([c['person_mode'] != 'strike' for c in attacks]),
        serve_share_of_set_shots=share([s['serve'] for s in sets]),
        # The hold clock restarts when the puck is put down, so a serve's is the wait to its strike.
        serve_to_strike_s=percentiles([s['hold_s'] for s in sets if s['serve'] and s['hold_s'] is not None
                                       and np.isfinite(s['hold_s'])], (10, 50, 90)),
        crossing_line_speed_m_s=percentiles([c['speed'] for c in attacks]),
        set_shot_line_speed_p50_m_s=q(line, 50), set_shot_line_speed_p90_m_s=q(line, 90),
        set_shot_line_speed_max_m_s=q(line, 100),
        set_shot_release_m_s=percentiles([s['speed'] for s in sets]),
        set_shot_bank_fraction=share([c['banked'] for c in struck]),
        bank_x0_rail_share=share([c.get('first_rail') == 'x0' for c in banks]),
        straight_set_shot_on_target=share([s['aimed'] for s in sets if s['intended_route'] == 0]),
        set_shot_hold_p50_s=q(holds, 50), set_shot_hold_s=percentiles(holds, (10, 25, 50, 75, 90)),
        active_counter_share_of_strikes=share([c['person_mode'] == 'volley' for c in attacks]),
        active_counter_share_of_swings=share([s['volley'] for s in sets + volleys]),
        active_counter_release_p50_m_s=q([s['speed'] for s in volleys], 50),
        active_counter_release_m_s=percentiles([s['speed'] for s in volleys]),
        active_counter_on_target=share([s['aimed'] for s in volleys]),
        robot_shot_outcomes=robot_shot_outcomes(robot_crossings),
        set_share_by_arrival_speed=set_share_by_arrival(arrivals, minutes),
        volley_crossing_conceded_rate=share([c['outcome'] == 'goal' for c in played if c['person_mode'] == 'volley']),
        set_shot_conceded_rate=share([c['outcome'] == 'goal' for c in struck]),
        set_shot_conceded_rate_by_speed=speed_bin_rates(struck, 'goal', SPEED_BINS),
        mallet_speed_p99_m_s=q(mallet, 99), mallet_speed_p999_m_s=q(mallet, 99.9))


def summarize(env, proxy, tracker, *, games, seconds, seed, tier):
    e = env.engine
    gf, ga = e.score_agent.astype(int), e.score_opponent.astype(int)
    minutes = games * seconds / 60
    total_f, total_a = int(gf.sum()), int(ga.sum())
    shots = [s for s in tracker.shot_events if s['side'] == 0]
    person = [s for s in tracker.shot_events if s['side'] == 1]
    # Crossings after a touch by the attacking side; serves and relaunches excluded.
    attacks = [c for c in tracker.crossings if c['toward'] == 'robot' and c['attack']]
    robot_crossings = [c for c in tracker.crossings if c['toward'] == 'proxy' and c['attack']]
    conceded = [g for g in tracker.goals if g['by'] == 'proxy']
    by_route = {}
    for g in conceded:
        key = g.get('route', 'unattributed') if g['last_touch'] == 'proxy' else (
            'robot_deflection' if g['last_touch'] == 'robot' else 'unattributed')
        by_route[key] = by_route.get(key, 0) + 1
    attack_speed = np.array([c['speed'] for c in attacks])
    attack_angle = np.array([c['angle_deg'] for c in attacks])
    banked = np.array([c['banked'] or c['route'] in ('left_bank', 'right_bank') for c in attacks], bool)
    mallet = np.concatenate(tracker.mallet_speed) if tracker.mallet_speed else np.zeros(1)
    routed = [s for s in person if s['route'] != 'none']
    strikes = [s for s in person if s['strike']]
    ratio = [s['speed'] / s['intended_speed'] for s in strikes if s['intended_speed']]
    geometric = np.array([c['banked'] for c in attacks], bool)
    struck = [c for c in attacks if c['person_mode'] == 'strike']
    struck_bank = np.array([c['banked'] for c in struck], bool)
    struck_angle = np.array([c['angle_deg'] for c in struck])
    targets = TARGETS
    calibration = dict(
        attacks=len(attacks), attacks_per_min=len(attacks) / minutes,
        attack_speed_centre_line_m_s=percentiles(attack_speed),
        attack_speed_target=targets['attack_speed_centre_line_m_s'],
        attack_bank_fraction=float(banked.mean()) if len(banked) else None,
        attack_bank_fraction_geometric=float(np.mean([c['banked'] for c in attacks])) if attacks else None,
        attack_bank_target=targets['attack_bank_fraction'],
        attack_angle_median_deg=float(np.median(attack_angle)) if len(attack_angle) else None,
        attack_angle_median_banked_deg=float(np.median(attack_angle[geometric])) if geometric.any() else None,
        attack_angle_median_unbanked_deg=float(np.median(attack_angle[~geometric])) if (~geometric).any() else None,
        attack_angle_target=targets['attack_angle_from_axis_deg'],
        attacks_by_person_mode={m: dict(n=sum(c['person_mode'] == m for c in attacks),
                                        speed_m_s=percentiles([c['speed'] for c in attacks if c['person_mode'] == m]),
                                        angle_median_deg=float(np.median([c['angle_deg'] for c in attacks
                                                                          if c['person_mode'] == m]))
                                        if any(c['person_mode'] == m for c in attacks) else None)
                                for m in proxy.mode_names},
        strike_attacks=dict(
            n=len(struck), speed_centre_line_m_s=percentiles([c['speed'] for c in struck]),
            bank_fraction=float(struck_bank.mean()) if len(struck) else None,
            angle_median_deg=float(np.median(struck_angle)) if len(struck) else None,
            angle_median_banked_deg=float(np.median(struck_angle[struck_bank])) if struck_bank.any() else None,
            angle_median_unbanked_deg=float(np.median(struck_angle[~struck_bank])) if (~struck_bank).any() else None,
            target=targets['deliberate_attack_reference']),
        strike_routes_intended={name: sum(s['intended_route'] == code for s in strikes)
                                for code, name in ((0, 'straight'), (-1, 'left_bank'), (1, 'right_bank'))},
        strike_routes_executed={k: sum(s['route'] == k for s in strikes) for k in ROUTE_NAMES},
        strike_puck_distance_from_own_goal_m=percentiles([s['local_y'] for s in strikes]),
        strikes_released=len(strikes),
        strike_release_m_s=percentiles([s['speed'] for s in strikes]),
        strike_intended_m_s=percentiles([s['intended_speed'] for s in strikes]),
        strike_achieved_over_intended_median=float(np.median(ratio)) if ratio else None,
        person_shots=len(person), person_shots_per_min=len(person) / minutes,
        person_shot_release_m_s=percentiles([s['speed'] for s in person]),
        person_shot_routes={k: sum(s['route'] == k for s in person) for k in ROUTE_NAMES},
        person_bank_fraction_of_routed=(sum(s['route'] in ('left_bank', 'right_bank') for s in routed) / len(routed))
        if routed else None,
        person_shots_on_target=sum(s['aimed'] for s in person),
        person_strikes=int(proxy.strikes.sum()),
        person_soft_blocks=int(proxy.traps.sum()),
        mallet_speed_m_s=dict(zip(('p50', 'p90', 'p99', 'p999', 'max'),
                                  np.percentile(mallet, [50, 90, 99, 99.9, 100]).round(3).tolist())),
        mallet_speed_target=targets['human_mallet_speed_m_s'],
        body_peak_accel_over_drawn_cap=round(tracker.body_accel_over_cap, 4),
        mallet_nearest_robot_goal_y_m=round(tracker.mallet_y_min, 4),
        reach_limit_y_m=round(env.cfg.height / 2 + env.cfg.paddle_radius, 4),
        mode_fraction=dict(zip(proxy.mode_names, (proxy.mode_ticks.sum(0) / max(1, proxy.mode_ticks.sum())).round(4).tolist())),
        conceded_midline_speed_m_s=percentiles([g['midline_speed'] for g in conceded if 'midline_speed' in g]),
        conceded_midline_to_goal_s=percentiles([g['midline_to_goal_s'] for g in conceded if 'midline_to_goal_s' in g]),
        attack_conceded_rate=(sum(c['outcome'] == 'goal' for c in attacks) / len(attacks)) if attacks else None,
        attack_conceded_rate_by_speed=speed_bin_rates(attacks, 'goal', SPEED_BINS),
        robot_crossings=len(robot_crossings),
        robot_crossing_speed_m_s=percentiles([c['speed'] for c in robot_crossings]),
        robot_crossing_scored_rate_by_speed=speed_bin_rates(robot_crossings, 'goal', ROBOT_SPEED_BINS),
        robot_crossing_scored_rate_by_speed_banked=speed_bin_rates(
            [c for c in robot_crossings if c['banked']], 'goal', ROBOT_SPEED_BINS),
        robot_crossing_scored_rate_by_speed_unbanked=speed_bin_rates(
            [c for c in robot_crossings if not c['banked']], 'goal', ROBOT_SPEED_BINS),
        reference_production_against_people=targets['production_against_people'],
    )
    if tier in TABLE_TARGETS:
        arrivals = [c for c in tracker.crossings if c['toward'] == 'proxy']
        calibration['table'] = table_calibration(person, attacks, robot_crossings, arrivals, mallet, minutes, proxy)
        calibration['table_target'] = TABLE_TARGETS[tier]
    calibration['checks'] = calibration_checks(calibration, tier)
    share = total_f / max(1, total_f + total_a)
    intervals = bootstrap(gf, ga, seed=seed)
    return dict(
        tier=tier, games=games, seconds=seconds, seed=seed, proxy_seed=proxy.seed,
        goals_for=total_f, goals_against=total_a,
        goal_share=share, goal_share_ci95=intervals['goal_share'],
        goals_for_per_min=total_f / minutes, goals_against_per_min=total_a / minutes,
        ci95_per_game=intervals,
        candidate_shots=int(env.shot_count[0].sum()), candidate_shots_on_target=int(env.aimed_count[0].sum()),
        candidate_routes=route_summary(shots, tracker.goals, robot_crossings),
        candidate_on_target_fraction=float(env.aimed_count[0].sum() / max(1, env.shot_count[0].sum())),
        candidate_mean_shot_speed=float(env.shot_speed_sum[0].sum() / max(1, env.shot_count[0].sum())),
        candidate_shot_speed_m_s=percentiles([s['speed'] for s in shots]),
        conceded_by_route=by_route,
        reachable_stall_seconds=round(tracker.stall_seconds, 2),
        candidate_overload_seconds=float(env.overload_seconds[0].sum()),
        candidate_seconds_above_shutdown_level=float(tracker.shutdown_seconds.sum()),
        candidate_games_reaching_shutdown_level=int((env.peak_load[0] >= SHUTDOWN_LEVEL).sum()),
        candidate_peak_load=float(env.peak_load[0].max()),
        candidate_guard_unresolved=int(env.guard_unresolved[0].sum()),
        # Accel cap asked for while the puck and the person's mallet are slow
        # (calm-weighted mean fraction of the machine's, and the calm share
        # spent above 20%). Production on the table: ~0.05; v2: 0.57, which
        # shook the mallet and sagged the supply (2026-10-18).
        candidate_calm_cap_fraction=float(env.calm_cap_seconds.sum() / max(1e-9, env.calm_seconds.sum())),
        candidate_calm_high_cap_share=float(env.calm_high_cap_seconds.sum() / max(1e-9, env.calm_seconds.sum())),
        per_game=dict(goals_for=gf.tolist(), goals_against=ga.tolist(),
                      peak_load=env.peak_load[0].round(4).tolist(),
                      overload_seconds=env.overload_seconds[0].round(3).tolist()),
        calibration=calibration,
        person_parameters=proxy.parameters(),
        goals=tracker.goals,
    )


def save_recordings(tracker, proxy, prefix, tier, *, candidate, seed, options, env, seconds):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    params = proxy.parameters()
    paths = []
    for i, rec in enumerate(tracker.frames):
        dest = prefix.parent / f'{prefix.name}_vs_human-proxy-{tier}-g{i}.json'
        person = {k: (v[i] if isinstance(v, list) else v) for k, v in params.items()}
        rec.save(dest, metadata=dict(
            algo='Single neural network' if not str(candidate).startswith('expert') else 'Physics teacher (benchmark only)',
            run_name=prefix.name, opponent=f'human proxy ({tier})', match_type='human-proxy benchmark',
            policy_sides='candidate blue on the robot body; red is a scripted person (benchmark opponent)',
            checkpoint=str(candidate), simulation_only=True, fps=50, seed=seed, game=i, seconds=seconds,
            person=person, workspace_sim=env.base._ws, accel_cap_m_s2=float(env.base._agent_dyn['max_accel'].max()),
            thermal_gain=THERMAL_GAIN, initial_load=INITIAL_LOAD, report_sensing=True, continuous_rallies=True,
            shot_type_names=['none', 'left bank', 'right bank', 'straight']))
        paths.append(str(dest))
    return paths


def candidate_hash(spec):
    if str(spec).startswith('expert'):
        return hashlib.sha256(Path('ai/airhockey/expert_player.py').read_bytes()).hexdigest()
    return hashlib.sha256(Path(spec).read_bytes()).hexdigest()


def evaluate(candidate, *, tiers=('club', 'elite'), games=64, seconds=60.0, seed=20261018,
             environment_from=BASE, device='cpu', stochastic=False, record=None, record_games=1,
             sim_profile=None, shot_request='random', selector_options=None):
    torch.set_num_threads(4)
    options = checkpoint_environment(environment_from)
    net, state = load(candidate)
    if net.environment_options != options:
        net = CoordinateReference(net, net.environment_options, options)
    net.to(device).eval()
    result = dict(
        benchmark='human-proxy', version=1, candidate=str(candidate), candidate_sha256=candidate_hash(candidate),
        step=state.get('step'), environment_from=str(environment_from), simulation_environment=options,
        sensing='report estimator', continuous_rallies=True, initial_load=INITIAL_LOAD,
        thermal_gain=THERMAL_GAIN, puck_speed_cap_m_s=PUCK_SPEED_CAP, stochastic=stochastic,
        proxy_source_sha256=hashlib.sha256(Path('ai/airhockey/human_proxy.py').read_bytes()).hexdigest(),
        tiers={})
    if sim_profile is not None:
        result.update(sim_profile=sim_profile, sim_profile_options=profile_options(sim_profile))
    if shot_request != 'random':
        result.update(shot_request=shot_request)
    for tier in tiers:
        result['tiers'][tier] = play_tier(net, tier, games=games, seconds=seconds, seed=seed, options=options,
                                          stochastic=stochastic, record=record, record_games=record_games,
                                          candidate=candidate, sim_profile=sim_profile,
                                          shot_request=shot_request, selector_options=selector_options)
    return result


def headline(result):
    rows = []
    for tier, r in result['tiers'].items():
        c = r['calibration']
        rows.append(dict(
            tier=tier, goals=f"{r['goals_for']}-{r['goals_against']}",
            goal_share=round(r['goal_share'], 3), ci95=[round(x, 3) for x in r['goal_share_ci95']],
            gf_per_min=round(r['goals_for_per_min'], 3), ga_per_min=round(r['goals_against_per_min'], 3),
            on_target=f"{r['candidate_shots_on_target']}/{r['candidate_shots']}",
            mean_shot_speed=round(r['candidate_mean_shot_speed'], 2), conceded_by_route=r['conceded_by_route'],
            stall_s=r['reachable_stall_seconds'], overload_s=round(r['candidate_overload_seconds'], 2),
            peak_load=round(r['candidate_peak_load'], 3),
            calm_cap=round(r.get('candidate_calm_cap_fraction', float('nan')), 3),
            calm_high=round(r.get('candidate_calm_high_cap_share', float('nan')), 3),
            person_attack_speed=c['attack_speed_centre_line_m_s'], person_bank_fraction=c['attack_bank_fraction'],
            wall_s=r['wall_seconds']))
        routes = r['candidate_routes']['all']
        rows[-1].update(route_share={k: v['share'] for k, v in routes['routes'].items()},
                        route_goal_rate={k: v['goal_rate'] for k, v in routes['routes'].items()},
                        route_entropy_bits=routes['entropy_bits'])
        if 'shot_request' in result:
            rows[-1].update(shot_request=result['shot_request'], free_share=round(r['shot_request']['free_share'], 3),
                            classifier_agreement=r['shot_request']['classifier']['agrees_with_env_route'])
        if 'lift' in r:
            rows[-1].update(sim_profile=result['sim_profile'], lift_events=r['lift']['lift_events'],
                            conceded_through_lift=r['lift']['goals_conceded_through_lift'])
        failed = [k for k, v in c['checks'].items() if v[2] is False]
        if tier in TABLE_TARGETS:
            rows[-1].update(calibration_outside=failed)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('candidate', help="checkpoint path, or an 'expert' / 'expert:key=value,...' spec")
    p.add_argument('--tier', choices=sorted(TIER_SEED) + sorted(TIER_GROUPS), default='both',
                   help="'both' = club + elite (default); 'tables' = table + table_plus")
    p.add_argument('--games', type=int, default=64)
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--seed', type=int, default=20261018, help='shared by all candidates (paired panel)')
    p.add_argument('--environment-from', type=Path, default=BASE,
                   help='run whose physical limits/workspace/thermal model the robot body uses')
    p.add_argument('--output', type=Path, help='JSON report (default logs/human-proxy/<candidate>-<seed>.json)')
    p.add_argument('--record', type=Path,
                   help='write replays as <dir>/<name>_vs_human-proxy-<tier>-g<i>.json (e.g. ai/recordings/prod)')
    p.add_argument('--record-games', type=int, default=1, help='games per tier to record (default 1)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--stochastic', action='store_true', help='sample the candidate policy instead of its mean')
    p.add_argument('--sim-profile', help="table-fidelity profile, e.g. table-20261029 or table-20261029:contact,lift "
                                         '(default: none, the historical simulator)')
    p.add_argument('--shot-request', choices=REQUEST_MODES, default='random',
                   help="who picks the candidate's route: random = a uniform explicit request per possession (as "
                        "trained, the default); free = no request, the network's own choice; auto = free under "
                        'the variety governor (airhockey/shot_selector.py). The report gains candidate_routes '
                        '(always) and shot_request (the governor and the route classifier)')
    for flag, kind, key in (('--selector-window', int, 'window'), ('--selector-floor', float, 'floor'),
                            ('--selector-ceiling', float, 'ceiling'), ('--selector-min-shots', int, 'min_shots'),
                            ('--selector-explore', float, 'explore')):
        p.add_argument(flag, type=kind, default=SELECTOR_DEFAULTS[key],
                       help=f'--shot-request auto: governor {key} (default {SELECTOR_DEFAULTS[key]})')
    a = p.parse_args()
    if a.games < 2 or not np.isfinite(a.seconds) or a.seconds <= 0:
        p.error('--games must be at least 2 and --seconds positive')
    tiers = TIER_GROUPS.get(a.tier, (a.tier,))
    try:
        profile_options(a.sim_profile)
    except ValueError as error:
        p.error(str(error))
    name = 'expert' if str(a.candidate).startswith('expert') else Path(a.candidate).parent.name
    if a.sim_profile is not None:
        name += '-' + a.sim_profile.replace(':', '-').replace(',', '-')
    if a.shot_request != 'random':
        name += '-request-' + a.shot_request
    selector_options = ({key: getattr(a, 'selector_' + key) for key in SELECTOR_DEFAULTS}
                        if a.shot_request == 'auto' else None)
    output = a.output or Path('logs/human-proxy') / f'{name}-{a.seed}.json'
    result = evaluate(a.candidate, tiers=tiers, games=a.games, seconds=a.seconds, seed=a.seed,
                      environment_from=a.environment_from, device=a.device, stochastic=a.stochastic,
                      record=a.record, record_games=a.record_games, sim_profile=a.sim_profile,
                      shot_request=a.shot_request, selector_options=selector_options)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(candidate=a.candidate, output=str(output), headline=headline(result))), flush=True)


if __name__ == '__main__':
    main()
