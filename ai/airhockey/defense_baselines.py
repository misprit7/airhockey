"""Observation-only benchmark opponents. Never used by the neural controller."""
import numpy as np
from airhockey.arrival import ArrivalDecoder
from airhockey.dynamics import workspace_in_sim
from airhockey.physics import TableConfig


class DefensiveBaseline:
    history = 16
    shot_conditioned = True

    def __init__(self, options, style='deep-bank'):
        if style not in ('deep-bank', 'forward-bank', 'deep-direct'):
            raise ValueError('unknown defense baseline')
        self.style = style
        self.environment_options = options
        ws = workspace_in_sim(bounds_mm=options['workspace_bounds_mm'])
        self.decoder = ArrivalDecoder([ws['min_x'], ws['max_x'], ws['min_y'], ws['max_y']])
        self.cfg = TableConfig()

    def to(self, device):
        return self

    def eval(self):
        return self

    def act(self, observation):
        # Exactly the same delayed/noisy physical features as the actor.
        # No fixture, future shot, trial-specific physics or true state access.
        obs = np.asarray(observation)
        x, y = obs[:, 0].copy(), obs[:, 1].copy()
        vx, vy = obs[:, 2].copy()*6, obs[:, 3].copy()*6
        depth = .65 if self.style == 'forward-bank' else .18
        contact_y = depth + self.cfg.paddle_radius + self.cfg.puck_radius
        incoming = (vy < -.3) & (y > contact_y)
        if self.style != 'deep-direct':
            # Nominal restitution, recomputed from observations each 20 ms.
            lo, hi = self.cfg.puck_radius, self.cfg.width-self.cfg.puck_radius
            for _ in range(5):
                t_goal = (contact_y-y)/np.minimum(vy, -1e-6)
                t_wall = np.where(vx > 0, (hi-x)/np.maximum(vx, 1e-6),
                                  (lo-x)/np.minimum(vx, -1e-6))
                bounce = incoming & (t_wall >= 0) & (t_wall < t_goal)
                x[bounce] += vx[bounce]*t_wall[bounce]
                y[bounce] += vy[bounce]*t_wall[bounce]
                vx[bounce] *= -self.cfg.wall_restitution
                vy[bounce] *= self.cfg.wall_tangential
        time = np.maximum(0, (contact_y-y)/np.minimum(vy, -1e-6))
        target_x = np.where(incoming, x+vx*time, .5)
        target = np.column_stack((target_x, np.full(len(obs), depth)))
        a = np.zeros((len(obs), 6))
        a[:, :2] = 2*(target-self.decoder.low)/(self.decoder.high-self.decoder.low)-1
        # Rest at interception, earliest supported arrival, full available cap.
        a[:, 4] = -1
        a[:, 5] = 1
        return np.clip(a, -1, 1)
