"""Run `airhockey.heuristics` bots inside BatchAirHockeyEnv.

A separate file from heuristics.py on purpose. The bots have to be importable
on the table, where there is a camera and a Teensy and no environment; this is
the half that knows what an observation column means, and it is the half that
gets thrown away when the same bots are driven by `vision/bin/puck_stream.py`.

The translation is deliberately narrow: the bridge reads the env's OBSERVATION
and nothing else. It does not touch `engine.puck_x`, and it does not consult
the true camera latency. That is the point -- a bot evaluated against ground
truth is being evaluated on a table that does not exist. What it does read off
the env are the two things a real controller also knows about itself: its own
position, and its own speed and acceleration ceilings.

The env's action space is normalised over the FULL half, in sim metres. Bots
speak table millimetres and absolute caps. Both conversions are exact
inverses of what `BatchAirHockeyEnv.step` does with them, and
`ai/tests/test_heuristics.py` asserts the round trip rather than trusting the
arithmetic to have been copied correctly.
"""

from __future__ import annotations

import numpy as np

from airhockey.dynamics import (MAX_ACCEL_M_S2, MAX_SPEED_M_S,
                                sim_to_table_mm, table_mm_to_sim)
from airhockey.heuristics import Command, PuckSample, TrackerReport
from airhockey.perception import FRAME_INTERVAL_S

# The action's cap dims map [-1, 1] onto this fraction band of the MACHINE
# caps; see BatchAirHockeyEnv.step. Restated here as the inverse's constants,
# and pinned by a round-trip test rather than by agreement.
CAP_FRAC_MIN = 0.05
CAP_FRAC_SPAN = 0.95


class SimBridge:
    """Observation -> TrackerReport, and Command -> normalised action.

    One bridge per env object; it holds no per-episode state beyond the clock,
    so it is safe to reuse across resets.
    """

    def __init__(self, env):
        if env.obs_mode != "history":
            raise ValueError(
                "SimBridge needs obs_mode='history': the bots take a position "
                "HISTORY, and the kinematic observation hands out an estimated "
                "velocity the real tracker would have had to compute itself")
        if env.action_mode != "profile_v":
            raise ValueError(
                "SimBridge needs action_mode='profile_v': a bot commands its "
                "own speed and accel caps, and the 2-dim action cannot carry "
                "them")
        self.env = env
        self.n = env.n_envs
        cfg = env.table_config
        self.sim_width = cfg.width
        self.sim_half_height = cfg.height / 2.0

        # Observation layout, derived from the env rather than copied out of
        # it -- the lag sets are tuning knobs and have already moved once.
        self.puck_lags = tuple(env.HISTORY_PUCK_LAGS)
        self.opp_lags = tuple(env.HISTORY_OPP_LAGS)
        self._n_puck = 2 * len(self.puck_lags)
        self._own = self._n_puck + 2 * len(self.opp_lags)

        self._low = np.asarray(env._action_low, dtype=float)
        self._high = np.asarray(env._action_high, dtype=float)
        self.action_dt = env.action_dt
        self.step_index = 0

    # ── env -> bot ───────────────────────────────────────────────────────

    def reset(self) -> None:
        self.step_index = 0

    @property
    def t_s(self) -> float:
        return self.step_index * self.action_dt

    def reports(self, obs: np.ndarray) -> list[TrackerReport]:
        """One TrackerReport per env, in table millimetres.

        Sample times are the CAMERA's, not the action loop's: the history
        columns are frames at fixed lags behind whatever frame the env was
        entitled to see, so the spacing is 5 ms times the lag and a bot fitting
        a velocity to them has to use that.
        """
        obs = np.asarray(obs)
        mm_x, mm_y = self._mm(obs[:, 0:self._n_puck:2],
                              obs[:, 1:self._n_puck:2])
        opp_x, opp_y = self._mm(obs[:, self._n_puck:self._own:2],
                                obs[:, self._n_puck + 1:self._own:2])
        own_x, own_y = self._mm(obs[:, self._own], obs[:, self._own + 1])

        now = self.t_s
        times = [now - lag * FRAME_INTERVAL_S for lag in self.puck_lags]
        out = []
        for i in range(self.n):
            puck = tuple(PuckSample(float(mm_x[i, k]), float(mm_y[i, k]), t)
                         for k, t in enumerate(times))
            out.append(TrackerReport(
                puck=puck,
                mallet=(float(own_x[i]), float(own_y[i])),
                opponent=(float(opp_x[i, 0]), float(opp_y[i, 0])),
                t_s=now,
            ))
        return out

    def _mm(self, sim_x, sim_y):
        return sim_to_table_mm(np.asarray(sim_x), np.asarray(sim_y),
                               self.sim_width, self.sim_half_height)

    # ── bot -> env ───────────────────────────────────────────────────────

    def caps(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each env's machine caps in mm units, read from the observation.

        Domain randomisation resamples acceleration per episode, and the
        observation carries it as a ratio to nominal precisely so the
        controller can know what body it is driving. Taking it from
        `env._agent_dyn` instead would work and would be cheating.
        """
        obs = np.asarray(obs)
        return (obs[:, -2] * MAX_SPEED_M_S * 1000.0,
                obs[:, -1] * MAX_ACCEL_M_S2 * 1000.0)

    def actions(self, commands: list[Command], obs: np.ndarray) -> np.ndarray:
        """Commands in mm -> [N, 4] normalised action.

        Positions invert the env's own rescale exactly. The caps become
        FRACTIONS of this env's machine caps, so a bot asking for more than the
        machine has gets the machine's -- which is what the Teensy's LIMITS
        clamp does on the table, and the reason the fraction is the natural
        unit here at all.
        """
        cap_v, cap_a = self.caps(obs)
        mm = np.array([[c.x_mm, c.y_mm] for c in commands], dtype=float)
        sim_x, sim_y = table_mm_to_sim(mm[:, 0], mm[:, 1],
                                       self.sim_width, self.sim_half_height)
        act = np.zeros((len(commands), 4), dtype=np.float32)
        act[:, 0] = 2.0 * (sim_x - self._low[0]) / (self._high[0] - self._low[0]) - 1.0
        act[:, 1] = 2.0 * (sim_y - self._low[1]) / (self._high[1] - self._low[1]) - 1.0
        act[:, 2] = self._cap_action(
            np.array([c.speed_mm_s for c in commands]), cap_v)
        act[:, 3] = self._cap_action(
            np.array([c.accel_mm_s2 for c in commands]), cap_a)
        return np.clip(act, -1.0, 1.0)

    @staticmethod
    def _cap_action(want: np.ndarray, machine: np.ndarray) -> np.ndarray:
        frac = np.clip(want / np.maximum(machine, 1e-9),
                       CAP_FRAC_MIN, 1.0)
        return (frac - CAP_FRAC_MIN) / (0.5 * CAP_FRAC_SPAN) - 1.0
