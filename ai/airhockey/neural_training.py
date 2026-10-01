"""Simulation-only curriculum and outcome rewards for a single neural policy."""

import json
import numpy as np

from airhockey.arrival_env import ArrivalEnv
from airhockey.batch_env import _OPP_POLICY_MAP
from airhockey.motion import CartState
from airhockey.motion_guard import guard_command
from airhockey.shot_flight import PARAMETERS, first_goal_crossing
from airhockey.rewards import shot_matches_type
from airhockey.skill_benchmark import Fixtures
from airhockey.thermal import DEFAULT_MODEL


class NeuralTrainingEnv(ArrivalEnv):
    obs_dim = 42
    action_dim = 6
    critic_context_dim = 9

    def __init__(
        self,
        n_envs=128,
        *,
        stage=0,
        accel=60.,
        workspace_bounds_mm=None,
        thermal_path=DEFAULT_MODEL,
        edge_dwell_weight=0.,
        edge_dwell_band=.06,
        project_rail_contacts=False,
        seed=0,
        realistic=True,
        report_sensing=False,
        randomize=True,
        games=False,
        goals_only=False,
        game_fraction=None,
        load_weight=0.15,
        load_energy_weight=2.,
        load_holding_weight=0.,
        load_holding_horizon=15.,
        capture_weight=2.0,
        conversion_weight=4.0,
        capture_first=False,
        receive_drill=False,
        productive_receive_drill=False,
        shot_power_weight=2.0,
        shot_power_exponent=1.0,
        off_target_penalty=1.0,
        conversion_speed_weight=0.0,
        discount=0.995,
        warm_start_max=0.8,
        warm_game_start=False,
        terminate_overload=False,
        fixed_practice_roles=False,
        random_practice_opponent=False,
        fixed_opponent_style_fraction=0.0,
        shutdown_penalty=100.0,
        game_goal_weight=25.0,
        skill_goal_weight=10.0,
        skill_concede_weight=12.0,
        turnover_weight=25.0,
        practice_selfplay_fraction=0.0,
        reward_scale=1.0,
        shutdown_level=1.0,
        practice_defense_fraction=0.25,
        defense_min_speed=2.0,
        defense_max_speed=12.0,
        wide_defense=False,
        random_defense_start_fraction=0.0,
        defense_clear_reward=0.0,
        defense_windup_fraction=0.0,
        shot_speed_scale=6.0,
        conversion_speed_scale=5.0,
        receiving_min_speed=0.6,
        receiving_max_speed=None,
        rushed_shot_penalty=0.0,
        thermal_gain=None,
        shot_conditioned=False,
        shot_request="random",
        shot_request_weight=0.0,
        wrong_shot_penalty=0.0,
        fallback_shot_scale=0.0,
        unproductive_return_penalty=0.0,
        setup_weight=3.0,
        setup_avoid_puck=False,
        shot_setup_fraction=0.0,
        control_potential_weight=0.0,
        continuous_rallies=False,
        possession_followthrough=False,
        recovery_fraction=0.0,
        recovery_min_speed=0.15,
        recovery_max_speed=1.2,
        recovery_easy_fraction=0.0,
        recovery_easy_arc=0.0,
        recovery_capture_bonus=0.0,
        recovery_fallback_scale=0.0,
        recovery_cushion_bonus=0.0,
        recovery_cushion_signed=False,
        recovery_get_ahead_bonus=0.0,
        recovery_approach_weight=0.0,
        readiness_weight=0.0,
        readiness_cost_weight=0.0,
        possession_delay_weight=0.0,
        game_episode_seconds=30.0,
        stationary_failure_penalty=0.0,
        stationary_rest_fraction=0.0,
        stationary_replay=None,
        stationary_replay_fraction=0.0,
        correlated_warm_fraction=0.0,
        cold_practice_fraction=0.0,
        cold_game_start_fraction=0.0,
        edge_drill_fraction=0.0,
        corner_drill_fraction=0.0,
        edge_recovery_weight=0.0,
        edge_approach_weight=0.0,
        edge_clearance_weight=0.0,
        slow_exit_penalty=0.0,
        readiness_lateral_uncertainty=0.0,
        defense_windup_lateral_speed=0.0,
    ):
        if goals_only and (terminate_overload or receive_drill or productive_receive_drill or possession_followthrough):
            raise ValueError('goals-only training requires full games without auxiliary drill/overload terminations')
        self.goals_only = bool(goals_only)
        games = games or self.goals_only
        continuous_rallies = continuous_rallies or self.goals_only
        super().__init__(
            n_envs,
            seed=seed,
            realistic=realistic,
            randomize=randomize,
            load_soft_start=0.8,
            accel=accel,
            workspace_bounds_mm=workspace_bounds_mm,
            thermal_path=thermal_path,
        )
        if not np.isfinite([edge_dwell_weight,edge_dwell_band]).all() or edge_dwell_weight<0 or edge_dwell_band<=0:
            raise ValueError('invalid edge dwell penalty')
        self.edge_dwell_weight,self.edge_dwell_band=edge_dwell_weight,edge_dwell_band
        self.engine.project_rail_contacts = bool(project_rail_contacts)
        if report_sensing and self.base._perception is not None:
            self.base._perception.enable_report_estimator()
        if not 0 <= recovery_fraction <= 1:
            raise ValueError("recovery fraction must be in [0,1]")
        if any(not np.isfinite(x) or x < 0 for x in (readiness_weight, readiness_cost_weight, possession_delay_weight, recovery_capture_bonus, recovery_cushion_bonus, recovery_get_ahead_bonus, recovery_approach_weight)):
            raise ValueError("possession reward weights must be finite and nonnegative")
        if not np.isfinite(game_episode_seconds) or game_episode_seconds <= 0:
            raise ValueError("game episode duration must be finite and positive")
        if possession_followthrough and receive_drill:
            raise ValueError("followthrough cannot use capture-only termination")
        self.continuous_rallies = continuous_rallies
        if not np.isfinite([slow_exit_penalty, readiness_lateral_uncertainty, defense_windup_lateral_speed]).all() or min(slow_exit_penalty, readiness_lateral_uncertainty, defense_windup_lateral_speed) < 0:
            raise ValueError("possession loss and release variability settings must be finite and nonnegative")
        self.slow_exit_penalty = slow_exit_penalty
        self.readiness_lateral_uncertainty = readiness_lateral_uncertainty
        self.defense_windup_lateral_speed = defense_windup_lateral_speed
        self.slow_possession_losses = np.zeros((2, n_envs), int)
        self._slow_loss_paid = np.zeros((2, n_envs), bool)
        if not np.isfinite(edge_drill_fraction) or not 0 <= edge_drill_fraction <= 1:
            raise ValueError("edge drill fraction must be within [0,1]")
        if not np.isfinite([edge_recovery_weight, edge_approach_weight, edge_clearance_weight]).all() or min(edge_recovery_weight, edge_approach_weight, edge_clearance_weight) < 0:
            raise ValueError("edge reward weights must be finite and nonnegative")
        self.edge_drill_fraction = edge_drill_fraction
        if not np.isfinite(corner_drill_fraction) or not 0<=corner_drill_fraction<=1:
            raise ValueError('corner drill fraction must be in [0,1]')
        self.corner_drill_fraction=corner_drill_fraction
        self.edge_recovery_weight = edge_recovery_weight
        self.edge_approach_weight = edge_approach_weight
        self.edge_clearance_weight = edge_clearance_weight
        self.edge_drill = np.zeros(n_envs, bool)
        self._edge_contact = np.zeros((2, n_envs), bool)
        self._edge_paid = np.zeros((2, n_envs), bool)
        self.edge_recovery_count = np.zeros((2, n_envs), int)
        self.possession_followthrough = possession_followthrough
        self.recovery_fraction = recovery_fraction
        if not np.isfinite([recovery_min_speed, recovery_max_speed]).all() or not 0 < recovery_min_speed <= recovery_max_speed:
            raise ValueError("invalid outgoing recovery speed range")
        self.recovery_min_speed, self.recovery_max_speed = recovery_min_speed, recovery_max_speed
        if not 0 <= recovery_easy_fraction <= 1:
            raise ValueError("recovery easy fraction must be in [0,1]")
        self.recovery_easy_fraction = recovery_easy_fraction
        if not np.isfinite(recovery_easy_arc) or not 0 <= recovery_easy_arc <= np.pi:
            raise ValueError("recovery easy arc must be in [0,pi] radians")
        self.recovery_easy_arc = recovery_easy_arc
        self.recovery_capture_bonus = recovery_capture_bonus
        if not np.isfinite(recovery_fallback_scale) or not 0 <= recovery_fallback_scale <= 1:
            raise ValueError("recovery fallback scale must be in [0,1]")
        self.recovery_fallback_scale = recovery_fallback_scale
        self._recovery_fast_fallback = np.zeros(n_envs, bool)
        self.recovery_cushion_bonus = recovery_cushion_bonus
        self.recovery_cushion_signed = recovery_cushion_signed
        self.recovery_get_ahead_bonus = recovery_get_ahead_bonus
        self._recovery_started_behind = np.zeros(n_envs, bool)
        self._recovery_ahead_paid = np.zeros(n_envs, bool)
        if not 0 <= cold_game_start_fraction <= 1:
            raise ValueError("cold game start fraction must be in [0,1]")
        self.cold_game_start_fraction = cold_game_start_fraction
        self.recovery_approach_weight = recovery_approach_weight
        if not np.isfinite(stationary_failure_penalty) or stationary_failure_penalty < 0 or not 0 <= stationary_rest_fraction <= 1:
            raise ValueError("invalid stationary completion curriculum")
        self.stationary_failure_penalty = stationary_failure_penalty
        self.stationary_rest_fraction = stationary_rest_fraction
        if not 0 <= stationary_replay_fraction <= 1 or not 0 <= correlated_warm_fraction <= 1:
            raise ValueError("invalid replay/warm curriculum fraction")
        self.stationary_replay_fraction = stationary_replay_fraction
        self.correlated_warm_fraction = correlated_warm_fraction
        if not 0 <= cold_practice_fraction <= 1:
            raise ValueError("cold practice fraction must be within [0,1]")
        self.cold_practice_fraction = cold_practice_fraction
        self.stationary_replay = None
        if stationary_replay is not None:
            with np.load(stationary_replay) as replay:
                self.stationary_replay = {k: replay[k].copy() for k in ('puck', 'paddle', 'previous_action', 'request')}
                source=json.loads(str(replay['environment_options'])) if 'environment_options' in replay else dict(accel=60,workspace_bounds_mm=None)
                from airhockey.neural_coordinates import action_coordinates
                self.stationary_replay['previous_action']=action_coordinates(
                    self.stationary_replay['previous_action'],source,
                    dict(accel=accel,workspace_bounds_mm=workspace_bounds_mm))
                if 'initial_load' in replay:
                    self.stationary_replay['initial_load']=replay['initial_load'].copy()
            count = len(self.stationary_replay['puck'])
            for key, width in (('puck',4),('paddle',2),('previous_action',6),('request',3)):
                if self.stationary_replay[key].shape != (count,width) or not count or not np.isfinite(self.stationary_replay[key]).all():
                    raise ValueError("invalid stationary failure replay")
            if 'initial_load' in self.stationary_replay:
                levels=self.stationary_replay['initial_load']
                if levels.shape!=(count,8):raise ValueError('invalid replay load shape')
                unspecified=np.isnan(levels).all(axis=1)
                valid=np.isfinite(levels).all(axis=1)&(levels>=0).all(axis=1)&(levels<shutdown_level).all(axis=1)
                if not (unspecified|valid).all():raise ValueError('replay loads must be below shutdown, or entirely unspecified')
        elif stationary_replay_fraction:
            raise ValueError("stationary replay fraction needs a dataset")
        self.readiness_weight = readiness_weight
        self.readiness_cost_weight = readiness_cost_weight
        self._readiness_running_cost = np.zeros(n_envs)
        self.possession_delay_weight = possession_delay_weight
        self.game_episode_seconds = game_episode_seconds
        self.recovery_drill = np.zeros(n_envs, bool)
        if continuous_rallies:
            self.base.SHOT_CLOCK_S = 0
            self.base._stuck_unattended_steps = round(8 / self.base.action_dt)
            self.base._stuck_attended_steps = round(8 / self.base.action_dt)
            self.base.stuck_relaunch_filter = self._unreachable_dead_puck
        if thermal_gain is not None:
            if not np.isfinite(thermal_gain) or thermal_gain <= 0:
                raise ValueError("thermal gain must be finite and positive")
            for load_model in self.loads:
                load_model.gain[:] = thermal_gain
        self.stage = stage
        self.games = games
        if game_fraction is not None and not 0 <= game_fraction <= 1:
            raise ValueError("game fraction must be in [0,1]")
        self.game_fraction_override = game_fraction
        self.load_weight = load_weight
        if not np.isfinite(load_energy_weight) or load_energy_weight < 0:
            raise ValueError("load energy weight must be finite and nonnegative")
        if not np.isfinite([load_holding_weight,load_holding_horizon]).all() or load_holding_weight < 0 or load_holding_horizon <= 0:
            raise ValueError("invalid holding-load reward settings")
        for model in self.loads:
            if load_holding_weight and not model.spatial:
                raise ValueError("holding-load reward requires a spatial current model")
            model.energy_weight = float(load_energy_weight)
            model.holding_forecast_weight = float(load_holding_weight)
            model.holding_forecast_seconds = float(load_holding_horizon)
        self.capture_weight = capture_weight
        self.conversion_weight = conversion_weight
        self.capture_first = capture_first
        self.receive_drill = receive_drill
        self.productive_receive_drill = productive_receive_drill
        if receive_drill and productive_receive_drill:
            raise ValueError("choose capture-only or productive receiving drills, not both")
        self.shot_power_weight = shot_power_weight
        if not np.isfinite(shot_power_exponent) or not 1 <= shot_power_exponent <= 4:
            raise ValueError("shot power exponent must be in [1,4]")
        self.shot_power_exponent = shot_power_exponent
        self.wide_defense = wide_defense
        self.off_target_penalty = off_target_penalty
        self.conversion_speed_weight = conversion_speed_weight
        self.discount = discount
        self.warm_start_max = warm_start_max
        self.terminate_overload = terminate_overload
        self.fixed_practice_roles = fixed_practice_roles
        self.random_practice_opponent = random_practice_opponent
        if not 0 <= fixed_opponent_style_fraction <= 1:
            raise ValueError("fixed opponent style fraction must be in [0,1]")
        self.fixed_opponent_style_fraction = fixed_opponent_style_fraction
        if not np.isfinite(shutdown_penalty) or shutdown_penalty < 0 or (shutdown_penalty == 0 and not goals_only):
            raise ValueError("shutdown penalty must be positive and finite")
        self.shutdown_penalty = shutdown_penalty
        self.game_goal_weight = game_goal_weight
        self.skill_goal_weight = skill_goal_weight
        self.skill_concede_weight = skill_concede_weight
        self.turnover_weight = turnover_weight
        if not 0 <= practice_selfplay_fraction <= 1:
            raise ValueError("practice self-play fraction must be in [0,1]")
        if practice_selfplay_fraction and stage < 3:
            raise ValueError(
                "practice self-play requires neural opponent actions (stage >=3)"
            )
        self.practice_selfplay_fraction = practice_selfplay_fraction
        self.reward_scale = reward_scale
        if not 0 < shutdown_level <= 1 or warm_start_max > shutdown_level:
            raise ValueError("shutdown level must cover warm starts and be in (0,1]")
        if not 0 <= practice_defense_fraction <= 1:
            raise ValueError("practice defense fraction must be in [0,1]")
        self.shutdown_level = shutdown_level
        self.practice_defense_fraction = practice_defense_fraction
        if not np.isfinite(defense_max_speed) or not 8<=defense_max_speed<=20:
            raise ValueError('defense maximum must be in [8,20] m/s')
        self.defense_max_speed=defense_max_speed
        self.cfg.max_puck_speed=max(self.cfg.max_puck_speed,defense_max_speed)
        if not 0 < defense_min_speed <= (defense_max_speed if stage >= 4 else 4 + 2 * min(stage, 2)):
            raise ValueError("defense minimum exceeds the stage's speed range")
        if not 0 <= random_defense_start_fraction <= 1:
            raise ValueError("random defense start fraction must be in [0,1]")
        if not np.isfinite(defense_clear_reward) or defense_clear_reward < 0:
            raise ValueError("defense clear reward must be finite and nonnegative")
        self.defense_min_speed = defense_min_speed
        self.random_defense_start_fraction = random_defense_start_fraction
        self.defense_clear_reward = defense_clear_reward
        if not 0 <= defense_windup_fraction <= 1:
            raise ValueError("defense windup fraction must be in [0,1]")
        self.defense_windup_fraction = defense_windup_fraction
        self.defense_windup = np.zeros(n_envs, bool)
        self._windup_release = np.full(n_envs, np.inf)
        self._windup_velocity = np.zeros((n_envs, 2))
        self._windup_aim = np.full(n_envs, .5)
        if not 1 < shot_speed_scale <= 12 or not 1 < conversion_speed_scale <= 12:
            raise ValueError("shot reward speed scales must be in (1,12]")
        self.shot_speed_scale = shot_speed_scale
        self.conversion_speed_scale = conversion_speed_scale
        if receiving_max_speed is not None and not 1 <= receiving_max_speed <= 12:
            raise ValueError("receiving maximum speed must be in [1,12]")
        self.receiving_max_speed = receiving_max_speed
        receiving_upper = receiving_max_speed or 1.2 + 0.8 * min(stage, 2)
        if not 0 < receiving_min_speed <= receiving_upper:
            raise ValueError("receiving minimum must be positive and below its maximum")
        self.receiving_min_speed = receiving_min_speed
        if not np.isfinite(rushed_shot_penalty) or rushed_shot_penalty < 0:
            raise ValueError("rushed shot penalty must be finite and nonnegative")
        self.rushed_shot_penalty = rushed_shot_penalty
        self.shot_conditioned = bool(shot_conditioned)
        self.obs_dim = 42 + 3 * self.shot_conditioned
        if shot_request not in ("random", "left", "right", "straight"):
            raise ValueError("shot request must be random, left, right or straight")
        if not 0 <= fallback_shot_scale <= 1:
            raise ValueError("fallback shot scale must be in [0,1]")
        for weight in (shot_request_weight, wrong_shot_penalty, unproductive_return_penalty,
                       setup_weight, control_potential_weight):
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("shot and return reward weights must be nonnegative and finite")
        self.shot_request = shot_request
        self.setup_weight = setup_weight
        self.setup_avoid_puck = setup_avoid_puck
        self.control_potential_weight = control_potential_weight
        if not 0 <= shot_setup_fraction <= 1 or (shot_setup_fraction and not self.shot_conditioned):
            raise ValueError("shot setup curriculum requires conditioning and a fraction in [0,1]")
        self.shot_setup_fraction = shot_setup_fraction
        self.shot_request_weight = shot_request_weight
        self.wrong_shot_penalty = wrong_shot_penalty
        self.fallback_shot_scale = fallback_shot_scale
        self.unproductive_return_penalty = unproductive_return_penalty
        self._shutdown_reset = np.zeros(n_envs, bool)
        self.warm_game_start = warm_game_start
        self._initial_thermal_state = np.ones(n_envs, bool)
        self.base.symmetric_referee = True
        self.base.shot_types = self.shot_conditioned
        self.base._shot_type_p[:] = (
            [0, 1/3, 1/3, 1/3] if shot_request == "random"
            else np.eye(4)[("none", "left", "right", "straight").index(shot_request)]
        )
        if self.fixed_opponent_style_fraction:
            self.base._shot_type_p_opp = np.tile(self.base._shot_type_p, (n_envs, 1))
        self.base.max_score = 1000000
        self.base.max_episode_time = 1000000
        self.kind = np.zeros(n_envs, int)
        self.touch_count = np.zeros((2, n_envs), int)
        self.shot_count = np.zeros((2, n_envs), int)
        self.aimed_count = np.zeros((2, n_envs), int)
        self.fast_aimed_count = np.zeros((2, n_envs), int)
        self.unproductive_returns = np.zeros((2, n_envs), int)
        # Axes: body, environment, requested type, executed type.
        # Types: 0 none/unresolved, 1 left bank, 2 right bank, 3 straight.
        self.shot_route_counts = np.zeros((2, n_envs, 4, 4), int)
        self.aimed_route_counts = np.zeros_like(self.shot_route_counts)
        self.last_shot_route = np.zeros((2, n_envs), int)
        self.last_shot_request = np.zeros((2, n_envs), int)
        self.shot_speed_sum = np.zeros((2, n_envs))
        self.shot_speed_histogram = np.zeros((2, n_envs, 5), int)
        self.target_histogram = np.zeros((2, n_envs, 3), int)
        self.turnovers = np.zeros((2, n_envs), int)
        self.entry_outcomes = np.zeros((2, n_envs, 6), int)
        self.entry_speed = np.zeros((2, n_envs))
        self.entry_outcomes_by_speed = np.zeros((2, n_envs, 4, 6), int)
        self.productive_entries_by_speed = np.zeros((2, n_envs, 4), int)
        self.opportunity_entries_by_speed = np.zeros((2, n_envs, 4), int)
        self.productive_opportunities_by_speed = np.zeros((2, n_envs, 4), int)
        self.entry_flags = np.zeros((2, n_envs, 4), bool)
        self.entry_active = np.zeros((2, n_envs), bool)
        self.entry_reachable_time = np.zeros((2, n_envs))
        self.capture_count = np.zeros((2, n_envs), int)
        self.convert_count = np.zeros((2, n_envs), int)
        self.captured = np.zeros((2, n_envs), bool)
        self.capture_paid = np.zeros((2, n_envs), bool)
        self.control_time = np.zeros((2, n_envs))
        self.contact_clock = np.full((2, n_envs), -100.0)
        self.reward_events = np.zeros((2, n_envs))
        self.guard_changed = np.zeros((2, n_envs), int)
        self.guard_unresolved = np.zeros((2, n_envs), int)
        self.peak_load = np.zeros((2, n_envs))
        self.overload_seconds = np.zeros((2, n_envs))
        self.passive_returns = np.zeros(n_envs, int)
        self.own_touched = np.zeros(n_envs, bool)
        self._potential = np.zeros(n_envs)
        self._last_observation = None
        self.motion_accel_peak = np.zeros((2, n_envs))
        self.motion_speed_peak = np.zeros((2, n_envs))
        self.high_accel_seconds = np.zeros((2, n_envs))
        original_relaunch = self.base._relaunch

        def relaunch(mask, *args, **kwargs):
            ids = np.flatnonzero(mask)
            own = self.engine.puck_y[ids] < 1
            self.turnovers[0, ids[own]] += 1
            self.turnovers[1, ids[~own]] += 1
            return original_relaunch(mask, *args, **kwargs)

        self.base._relaunch = relaunch

    def _motion(self, dt, old_agent_v, old_opp_v):
        super()._motion(dt, old_agent_v, old_opp_v)
        for side, (dyn, old) in enumerate(
            ((self.base._agent_dyn, old_agent_v), (self.base._opp_dyn, old_opp_v))
        ):
            velocity = np.column_stack((dyn["vx"], dyn["vy"]))
            acceleration = np.linalg.norm((velocity - old) / dt, axis=1)
            self.motion_accel_peak[side] = np.maximum(
                self.motion_accel_peak[side], acceleration
            )
            self.motion_speed_peak[side] = np.maximum(
                self.motion_speed_peak[side], np.linalg.norm(velocity, axis=1)
            )
            self.high_accel_seconds[side] += dt * (acceleration > 40)

    def _features(self, base_obs, opponent=False):
        # Measured physical state, command history and measured-load proxy.
        # No drill identity, phase or timer. Optional shot requests are explicit
        # user-facing inputs, held fixed for a possession by the base env.
        from airhockey.neural_observation import neural_features
        dyn = self.base._opp_dyn if opponent else self.base._agent_dyn
        cart = self._cart(dyn, opponent)
        queued = np.column_stack((dyn["command_x"], dyn["command_y"]))
        if opponent:
            queued[:, 1] = self.cfg.height - queued[:, 1]
        return neural_features(
            base_obs, self.last_opp_action if opponent else self.last_action,
            self.loads[int(opponent)].features(),
            np.column_stack((cart.ax, cart.ay)), queued, dyn["command_accel"],
            dyn["max_accel"], self.decoder.low, self.decoder.high,
            shot_conditioned=self.shot_conditioned,
            shot_request=self.base._shot_onehot(
                self.base._shot_type_opp if opponent else self.base._shot_type))

    def critic_context(self):
        # A training-only baseline may observe reward/episode state. These
        # columns never enter the actor, its exploration head, or act().
        return np.column_stack(
            (
                np.eye(4)[self.kind],
                np.where(self.kind == 3, 0, np.minimum(self.elapsed / np.where(self.edge_drill, 12, 4), 1)),
                self.captured.T,
                self.capture_paid.T,
            )
        ).astype(np.float32)

    def _decode(self, action, opponent=False):
        target, cap = super()._decode(action, opponent)
        dyn = self.base._opp_dyn if opponent else self.base._agent_dyn
        ids = (
            np.flatnonzero(self.base._opp_policy_id == _OPP_POLICY_MAP["external"])
            if opponent
            else np.arange(self.n_envs)
        )
        if not len(ids):
            return target, cap
        previous = np.column_stack((dyn["command_x"], dyn["command_y"]))
        if opponent:
            previous[:, 1] = self.cfg.height - previous[:, 1]
        full = self._cart(dyn, opponent)
        cart = CartState(len(ids))
        for key in CartState.__slots__:
            getattr(cart, key)[:] = getattr(full, key)[ids]
        target[ids], cap[ids], changed, unresolved = guard_command(
            cart,
            target[ids],
            cap[ids],
            previous[ids],
            dyn["command_accel"][ids],
            bounds=self.decoder.bounds,
            max_accel=dyn["max_accel"][ids],
            max_speed=dyn["max_speed"][ids],
            delay=self.base.command_delay_s,
            action_dt=self.base.action_dt,
        )
        self.guard_changed[int(opponent), ids] += changed
        self.guard_unresolved[int(opponent), ids] += unresolved
        return target, cap

    def _contact(self, event):
        side = int(event["body"] != "agent")
        ids = event["indices"]
        self.touch_count[side, ids] += 1
        self.entry_flags[side, ids, 0] = True
        self.own_touched[ids] |= side == 0
        incoming, outgoing = (
            event["incoming"].copy(),
            event["outgoing_before_speed_cap"].copy(),
        )
        if side:
            incoming[:, 1] *= -1
            outgoing[:, 1] *= -1
        fresh = self.elapsed[ids] - self.contact_clock[side, ids] > 0.20
        self.contact_clock[side, ids] = self.elapsed[ids]
        speed = np.linalg.norm(outgoing, axis=1)
        outgoing *= np.minimum(1, self.cfg.max_puck_speed / np.maximum(speed, 1e-9))[
            :, None
        ]
        speed = np.minimum(speed, self.cfg.max_puck_speed)
        incoming_speed = np.linalg.norm(incoming, axis=1)
        if self.edge_recovery_weight:
            # Label a slow contact at a side fringe. A later reward requires
            # actual return to the interior; touching alone earns nothing.
            x = self.engine.puck_x[ids]
            fringe = (x < self.decoder.low[0] + .04) | (x > self.decoder.high[0] - .04)
            self._edge_contact[side, ids] |= fringe & (incoming_speed < 2)
        if side == 0 and (self.recovery_cushion_bonus or self.recovery_get_ahead_bonus):
            # Separate approach and cushioning signals for the difficult first
            # catch. Only the first outgoing recovery contact counts;
            # subsequent taps cannot farm it. Full control credit is separate.
            leading = (incoming[:, 1] > .05) & ((event["normal"] * incoming).sum(axis=1) < -.05)
            eligible = self.recovery_drill[ids] & (self.touch_count[side, ids] == 1) & leading
            if self.recovery_get_ahead_bonus:
                # A first leading-side collision also establishes progress
                # from a trailing start. The sampled geometric milestone is
                # narrower and can miss an oblique/fast approach before contact.
                # Share its once-only ledger: never pay twice or reward a
                # later front contact after first pushing from behind.
                progressed = (eligible & self._recovery_started_behind[ids]
                              & ~self._recovery_ahead_paid[ids])
                self.reward_events[side, ids] += self.recovery_get_ahead_bonus * progressed
                self._recovery_ahead_paid[ids] |= progressed
            slowed = np.clip(1 - speed / np.maximum(incoming_speed, 1e-6), 0, 1)
            if self.recovery_cushion_signed:
                # The positive-only reward is flat for every overpowered
                # contact. Give bounded, continuous feedback on that side too:
                # zero for unchanged speed, positive for cushioning, negative
                # for accelerating the puck. Still first leading contact only.
                ratio = speed / np.maximum(incoming_speed, 1e-6)
                slowed = np.exp(-np.minimum(ratio**2, 80)) - np.exp(-1)
            self.reward_events[side, ids] += self.recovery_cushion_bonus * eligible * slowed
        slow = (incoming_speed > 0.8) & fresh
        self.reward_events[side, ids] += (
            (4 if self.capture_first else 1)
            * slow
            * np.maximum(0, 1 - speed / np.maximum(incoming_speed, 1e-6))
        )
        # Reward one distinct return, never persistent overlap/contact chatter.
        # A controlled puck can be struck immediately after cushioning. A
        # blanket contact debounce would erase that very sequence's credit.
        launch = (
            fresh
            | self.captured[side, ids]
            | ((incoming[:, 1] < 1.0) & (outgoing[:, 1] - incoming[:, 1] > 0.5))
        )
        shot = (outgoing[:, 1] > 1.0) & launch
        y = self.engine.puck_y[ids]
        if side:
            y = self.cfg.height - y
        crossing, aimed, banks, rail = first_goal_crossing(
            np.column_stack((self.engine.puck_x[ids], y, outgoing)),
            {k: getattr(self.engine, k)[ids] for k in PARAMETERS},
            self.cfg,
            return_route=True,
        )
        requested = (self.base._shot_type_opp if side else self.base._shot_type)[ids]
        route = np.where(np.isfinite(crossing), np.where(banks == 0, 3, np.where(rail < 0, 1, 2)), 0)
        matched = shot_matches_type(requested, banks, rail)
        route_ok = (requested == 0) | matched
        self.last_shot_route[side, ids] = route
        self.last_shot_request[side, ids] = requested
        np.add.at(self.shot_route_counts[side], (ids[shot], requested[shot], route[shot]), 1)
        np.add.at(self.aimed_route_counts[side], (ids[shot & aimed], requested[shot & aimed], route[shot & aimed]), 1)
        # After initial contact training, every point within the goal mouth
        # receives equal accuracy shaping. No permanent center-shot preference.
        error = abs(crossing - 0.5)
        if self.stage:
            error = np.maximum(
                error - (self.cfg.goal_width / 2 - self.cfg.puck_radius), 0
            )
        precision = np.exp(-np.minimum((error / 0.15) ** 2, 80))
        self.shot_count[side, ids] += shot
        self.aimed_count[side, ids] += shot & aimed
        self.fast_aimed_count[side, ids] += shot & aimed & (speed >= 4)
        self.entry_flags[side, ids, 2] |= shot & aimed
        self.entry_flags[side, ids, 3] |= shot & aimed & (speed >= 4)
        self.shot_speed_sum[side, ids] += shot * speed
        speed_bin = np.digitize(speed, [2, 4, 6, 8])
        self.shot_speed_histogram[side, ids[shot], speed_bin[shot]] += 1
        goal_half = self.cfg.goal_width / 2 - self.cfg.puck_radius
        target_bin = np.digitize(crossing, [0.5 - goal_half / 3, 0.5 + goal_half / 3])
        self.target_histogram[side, ids[shot & aimed], target_bin[shot & aimed]] += 1
        reward = (
            1.0
            + 2.0 * precision
            + self.shot_power_weight
            * aimed
            * np.minimum(speed / self.shot_speed_scale, 1) ** self.shot_power_exponent
            - (self.off_target_penalty if self.stage else 0.0) * ~aimed
        )
        if self.capture_first:
            rushed = (incoming_speed > 0.8) & ~self.captured[side, ids]
            # Configurable fallback credit for accurate immediate returns.
            # Capturing then shooting earns the additional conversion reward.
            reward = np.where(
                rushed, np.minimum(reward, 0) + self.fallback_shot_scale * np.maximum(reward, 0), reward
            )
            reward -= self.rushed_shot_penalty * rushed
        route_scale = np.where(route_ok, 1.0, 0.25) if self.shot_conditioned else 1.0
        if self.shot_conditioned:
            reward = np.minimum(reward, 0) + route_scale * np.maximum(reward, 0)
            reward += self.shot_request_weight * aimed * matched * np.minimum(speed / self.shot_speed_scale, 1) ** self.shot_power_exponent
            # A defensive return can remain useful even if the requested route
            # is unavailable. Penalize mismatches only after control/slow setup.
            prepared = self.captured[side, ids] | (incoming_speed <= 0.8)
            reward -= self.wrong_shot_penalty * ~route_ok * prepared
        if side == 0:
            # Early recovery curricula can require control. Later curricula
            # may also credit a fast requested shot when a catch is unlikely;
            # separate capture/approach/conversion rewards still favor control.
            # This labels outcomes only and never chooses an actor action.
            recovery_without_control = self.recovery_drill[ids] & ~self.captured[side, ids]
            qualified_fallback = shot & aimed & route_ok & (speed >= 6)
            self._recovery_fast_fallback[ids] |= (
                recovery_without_control & qualified_fallback & (self.recovery_fallback_scale > 0)
            )
            reward = np.where(
                recovery_without_control,
                np.minimum(reward, 0) + self.recovery_fallback_scale * qualified_fallback * np.maximum(reward, 0),
                reward,
            )
        self.reward_events[side, ids] += shot * reward
        converted = shot & aimed & self.captured[side, ids]
        self.convert_count[side, ids] += converted
        quality = (
            1 - self.conversion_speed_weight
        ) + self.conversion_speed_weight * np.clip(
            (speed - 1) / (self.conversion_speed_scale - 1), 0, 1
        )
        self.reward_events[side, ids] += self.conversion_weight * converted * quality * route_scale
        self.captured[side, ids[shot]] = False
        self.control_time[side, ids[shot]] = 0
        self.captured[1 - side, ids] = False
        self.capture_paid[1 - side, ids] = False
        self.control_time[1 - side, ids] = 0
        self.reward_events[side, ids] += 0.25 * fresh

    def _goal(self, event):
        # Goals are accounted from score deltas, including both sides.
        pass

    def _recovery_approach_event(self):
        """Once-only progress reward for going around an outgoing puck.

        A trailing start must reach the leading side without first touching the
        puck. This intermediate training event does not terminate the exercise
        or choose an action; cushioning, control and shooting are still needed.
        """
        e = self.engine
        velocity = np.column_stack((e.puck_vx, e.puck_vy))
        speed = np.linalg.norm(velocity, axis=1)
        direction = velocity / np.maximum(speed[:, None], 1e-6)
        delta = np.column_stack((e.paddle_agent_x-e.puck_x, e.paddle_agent_y-e.puck_y))
        ahead = (delta * direction).sum(axis=1)
        lateral = np.abs(delta[:, 0]*direction[:, 1]-delta[:, 1]*direction[:, 0])
        radius = self.cfg.puck_radius + self.cfg.paddle_radius
        event = (self.recovery_drill & self._recovery_started_behind & ~self._recovery_ahead_paid
                 & (self.touch_count[0] == 0) & (e.puck_vy > .05) & (speed < 2)
                 & (ahead > radius+.01) & (lateral < .12)
                 & (np.linalg.norm(delta, axis=1) < .25) & (e.puck_y < self.cfg.height/2))
        self._recovery_ahead_paid |= event
        return event

    def _unreachable_dead_puck(self):
        """Only replace dead pucks outside both robot paddles' contact reach."""
        puck = np.column_stack((self.engine.puck_x, self.engine.puck_y))
        radius = self.cfg.puck_radius + self.cfg.paddle_radius
        # The motion guard retains about1mm inside the center workspace. A
        # generous outside tolerance would strand genuinely untouchable pucks
        # indefinitely, so require actual contact from a guarded paddle pose.
        low, high = self.decoder.low + .001, self.decoder.high - .001
        own = np.linalg.norm(puck - np.clip(puck, low, high), axis=1) < radius - 1e-5
        far = puck.copy()
        far[:, 1] = self.cfg.height - far[:, 1]
        opponent = np.linalg.norm(far - np.clip(far, low, high), axis=1) < radius - 1e-5
        return ~(own | opponent)

    def _edge_recovery_event(self, side, excluded=None):
        """Once per possession: a touched fringe puck returns at controllable speed."""
        e = self.engine
        y = self.cfg.height - e.puck_y if side else e.puck_y
        interior = ((e.puck_x >= self.decoder.low[0] + .04)
                    & (e.puck_x <= self.decoder.high[0] - .04)
                    & (y >= self.decoder.low[1] + .04)
                    & (y <= self.decoder.high[1])
                    & (np.hypot(e.puck_vx, e.puck_vy) < 2))
        event = self._edge_contact[side] & ~self._edge_paid[side] & interior
        if excluded is not None:
            event &= ~excluded
        self._edge_paid[side] |= event
        self.edge_recovery_count[side] += event
        return event

    def _slow_possession_loss(self, side, excluded):
        """Lost reachable slow possession, even after a brief earlier capture."""
        e = self.engine
        y = self.cfg.height-e.puck_y if side else e.puck_y
        vy = -e.puck_vy if side else e.puck_vy
        radius = self.cfg.puck_radius+self.cfg.paddle_radius
        lost = (self.entry_active[side] & ~self._slow_loss_paid[side]
                & (self.entry_reachable_time[side] >= .1)
                & (y > self.decoder.high[1]+radius) & (y < self.cfg.height/2)
                & (vy > .05) & (np.hypot(e.puck_vx, e.puck_vy) < 2)
                & ~self.entry_flags[side, :, 3] & ~excluded)
        self._slow_loss_paid[side] |= lost
        self.slow_possession_losses[side] += lost
        return lost

    def potential(self):
        e = self.engine
        puck = np.column_stack((e.puck_x, e.puck_y))
        pad = np.column_stack((e.paddle_agent_x, e.paddle_agent_y))
        speed = np.hypot(e.puck_vx, e.puck_vy)
        direction = np.column_stack((0.5 - puck[:, 0], self.cfg.height - puck[:, 1]))
        if self.shot_conditioned:
            request = self.base._shot_type
            bank = (request == 1) | (request == 2)
            rail_x = np.where(request == 1, self.cfg.puck_radius, self.cfg.width - self.cfg.puck_radius)
            # Unfold a single bank using the calibrated normal/tangent ratio.
            # This only shapes slow-puck setup reward, never an action or input.
            bank_dx = rail_x - puck[:, 0] + (
                e.wall_tangential / np.maximum(e.wall_restitution, 1e-6)
            ) * (rail_x - self.cfg.width / 2)
            direction[bank, 0] = bank_dx[bank]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        behind = np.clip(puck - 0.13 * direction, self.decoder.low, self.decoder.high)
        # Potential shapes approach/setup only; it never changes an action.
        setup_distance = np.linalg.norm(pad - behind, axis=1)
        if self.setup_avoid_puck:
            from airhockey.neural_possession import disc_avoiding_distance
            radius = self.cfg.puck_radius+self.cfg.paddle_radius+.005
            clear_target = np.linalg.norm(behind-puck,axis=1) > radius
            detour = disc_avoiding_distance(pad, behind, puck, radius)
            setup_distance = np.where(clear_target, detour, setup_distance)
        contact_distance = np.maximum(np.linalg.norm(pad - puck, axis=1) - 0.09, 0)
        slow = np.exp(-np.minimum((speed / 1.0) ** 4, 80))
        distance = slow * setup_distance + (1 - slow) * contact_distance
        own = np.clip((1.1 - puck[:, 1]) / 0.2, 0, 1)
        potential = -self.setup_weight * np.minimum(distance, 1) * own
        if self.recovery_approach_weight:
            # A departing puck must first be met on its leading side, whereas
            # shooting setup rewards approach from behind. Blend out that
            # conflicting setup potential until the puck has been controlled.
            # This is reward shaping only: no target reaches the actor/decoder.
            velocity = np.column_stack((e.puck_vx, e.puck_vy))
            leading = puck + velocity * (.08 + .13 / np.maximum(speed, .05))[:, None]
            leading = np.clip(leading, self.decoder.low, self.decoder.high)
            outgoing = np.clip(e.puck_vy / .2, 0, 1)
            recover = outgoing * np.clip((2 - speed) / .8, 0, 1) * ~self.captured[0]
            from airhockey.neural_possession import disc_avoiding_distance
            approach = np.minimum(disc_avoiding_distance(pad, leading, puck,
                self.cfg.puck_radius+self.cfg.paddle_radius+.005), 1)
            potential = (1 - recover) * potential - self.recovery_approach_weight * recover * approach * own
        if self.control_potential_weight:
            # Smooth training feedback for slowing a recently touched puck
            # nearby. This is a bounded potential, used through gamma*Phi'-Phi;
            # it pays no repeatable holding bonus and never chooses an action.
            gap = np.linalg.norm(pad - puck, axis=1) - (
                self.cfg.puck_radius + self.cfg.paddle_radius
            )
            reachable = np.linalg.norm(
                puck - np.clip(puck, self.decoder.low, self.decoder.high), axis=1
            ) <= self.cfg.puck_radius + self.cfg.paddle_radius
            recent = self.elapsed - self.contact_clock[0] < .6
            close = np.exp(-np.minimum((np.maximum(gap, 0) / .12) ** 2, 80))
            slow_quality = np.exp(-np.minimum((speed / 2) ** 2, 80))
            potential += self.control_potential_weight * close * slow_quality * (
                recent & reachable & (gap > -.02) & (puck[:, 1] < 1)
            )
        if self.readiness_weight or self.readiness_cost_weight:
            from airhockey.neural_possession import direct_goal_coverage_cost
            velocity = np.column_stack((e.paddle_agent_vx, e.paddle_agent_vy))
            preparing = (e.puck_vy >= 0) | (speed < 2)
            risk = direct_goal_coverage_cost(puck, pad, velocity, self.decoder.bounds, self.cfg,
                acceleration=self.base._agent_dyn['nominal_accel'],shot_speed=self.defense_max_speed,
                lateral_uncertainty=self.readiness_lateral_uncertainty * preparing)
            # While the opponent has the puck, learn positions from which
            # several direct attacks can be covered. No fixed home target,
            # forecast, task label, or risk score enters the actor's input.
            potential -= self.readiness_weight * risk * np.clip((puck[:, 1] - 1) / .15, 0, 1)
            # Charge time spent exposed while an opponent can prepare a shot,
            # including the flight of our outgoing shot. Once a fast return
            # is incoming, actual saving outcomes govern the interception.
            self._readiness_running_cost = risk * preparing * np.clip((puck[:, 1] - 1) / .15, 0, 1)
        if self.edge_approach_weight:
            # A clipped behind-the-puck shot setup can be unable to touch a
            # fringe puck. Shape progress toward a feasible contact instead;
            # the actor still chooses all actions and any recovery route.
            depth = np.maximum(self.decoder.low[0] + .04 - puck[:, 0],
                               puck[:, 0] - (self.decoder.high[0] - .04))
            blend = np.clip(depth / .04, 0, 1) * np.clip((2 - speed) / 1.5, 0, 1)
            blend *= (puck[:, 1] < self.decoder.high[1]) & (puck[:, 1] > self.cfg.puck_radius)
            contact = np.clip(puck, self.decoder.low + .002, self.decoder.high - .002)
            distance = np.linalg.norm(pad - contact, axis=1)
            potential = (1 - blend) * potential - blend * self.edge_approach_weight * distance
        if self.edge_clearance_weight:
            # A bounded state potential supplies incremental credit for
            # getting a fringe puck into playable space. Gamma*Phi'-Phi is
            # applied below, so repeated visits cannot farm a recovery bonus.
            # No preferred paddle position, route, or motion is prescribed.
            margins=np.column_stack((puck[:,0]-self.cfg.puck_radius,
                self.cfg.width-self.cfg.puck_radius-puck[:,0],
                puck[:,1]-self.cfg.puck_radius))
            clearance=np.array([self.decoder.low[0]+.04-self.cfg.puck_radius,
                self.cfg.width-self.decoder.high[0]+.04-self.cfg.puck_radius,
                self.decoder.low[1]+.04-self.cfg.puck_radius])
            debt=np.square(np.clip(1-margins/clearance,0,1)).sum(axis=1)
            potential-=self.edge_clearance_weight*debt*np.clip((1-puck[:,1])/.15,0,1)
        return potential

    def reset(self, *, seed=None, mask=None, fixtures=None, opponent=None):
        mask = np.ones(self.n_envs, bool) if mask is None else np.asarray(mask, bool)
        ids = np.flatnonzero(mask)
        n = len(ids)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        starting_requests = np.zeros(n, dtype=int)
        recovery = np.zeros(n, bool)
        windup = np.zeros(n, bool)
        release_velocity = np.zeros((n, 2))
        replay_actions = np.full((n, 6), np.nan)
        replay_loads = np.full((n, 8), np.nan)
        replay_fringe = np.zeros(n, bool)
        edge = np.zeros(n, bool)
        self.game_fraction = 1.0 if self.games else (self.game_fraction_override if self.game_fraction_override is not None else (0 if self.stage < 3 else 0.5))
        if fixtures is None:
            kind = rng.choice(
                3,
                n,
                p=(
                    [0.2, 0.7, 0.1]
                    if self.capture_first and self.stage < 2
                    else [0.3, 0.45, 0.25]
                    if self.capture_first
                    else [0.8, 0.2, 0]
                    if self.stage == 0
                    else [0.45, 0.4, 0.15]
                    if self.stage == 1
                    else [0.3, 0.4, 0.3]
                ),
            )
            self.kind[ids] = kind
            if self.fixed_practice_roles:
                stationary_end = round(20 * 0.4 * (1 - self.practice_defense_fraction))
                receiving_end = round(20 * (1 - self.practice_defense_fraction))
                kind = np.where(
                    ids % 20 < stationary_end,
                    0,
                    np.where(ids % 20 < receiving_end, 1, 2),
                )
                self.kind[ids] = kind
            if self.receive_drill:
                kind = np.where(ids % 10 < 2, 0, np.where(ids % 10 < 9, 1, 2))
                self.kind[ids] = kind
            elif self.productive_receive_drill:
                # Honor the configured defense allocation. The former 10-slot
                # override silently reduced it to 10% of practice slots.
                defense_start = round(20 * (1 - self.practice_defense_fraction))
                stationary_end = min(4, defense_start)
                kind = np.where(ids % 20 < stationary_end, 0,
                                np.where(ids % 20 < defense_start, 1, 2))
                self.kind[ids] = kind
            puck = np.zeros((n, 4))
            spread = min(self.stage, 2)
            puck[:, 0] = rng.uniform(0.35 - 0.09 * spread, 0.65 + 0.09 * spread, n)
            puck[:, 1] = rng.uniform(0.40, 0.60 + 0.025 * spread, n)
            if self.stage >= 2:
                stationary = kind == 0
                puck[stationary, 1] = rng.uniform(
                    self.decoder.low[1]
                    + self.cfg.puck_radius
                    + self.cfg.paddle_radius
                    + 0.03,
                    self.decoder.high[1]
                    + self.cfg.puck_radius
                    + self.cfg.paddle_radius
                    - 0.02,
                    stationary.sum(),
                )
            paddle = puck[:, :2] + rng.uniform(
                [-0.025 - 0.04 * spread, -0.20], [0.025 + 0.04 * spread, -0.12], (n, 2)
            )
            receiving = kind == 1
            k = receiving.sum()
            puck[receiving, 1] += 0.14
            puck[receiving, 2] = rng.uniform(-0.2, 0.2, k) * (1 + spread)
            incoming_speed = rng.uniform(
                self.receiving_min_speed, self.receiving_max_speed or 1.2 + 0.8 * spread, k
            )
            puck[receiving, 3] = -incoming_speed
            # Fast receiving exercises need room to observe and accelerate
            # before contact. This changes initial states only; no target or
            # intercept suggestion is exposed to the policy.
            puck[receiving, 1] += 0.08 * np.maximum(incoming_speed - 3, 0)
            recovery = receiving & (rng.random(n) < self.recovery_fraction) if self.recovery_fraction else np.zeros(n, bool)
            if recovery.any():
                k = recovery.sum()
                # Drifting away from the robot: it must get around the puck,
                # cushion it, and complete a shot rather than chase from behind.
                puck[recovery, 0] = rng.uniform(self.decoder.low[0] + .10, self.decoder.high[0] - .10, k)
                puck[recovery, 1] = rng.uniform(self.decoder.low[1] + .18, self.decoder.high[1] - .12, k)
                angle = rng.uniform(-.65, .65, k)
                speed = rng.uniform(self.recovery_min_speed, self.recovery_max_speed, k)
                puck[recovery, 2:] = speed[:, None] * np.column_stack((np.sin(angle), np.cos(angle)))
                paddle[recovery] = puck[recovery, :2] + rng.uniform([-.12, -.22], [.12, -.12], (k, 2))
                if self.recovery_easy_fraction:
                    easier = np.flatnonzero(recovery)[rng.random(k) < self.recovery_easy_fraction]
                    direction = puck[easier, 2:] / np.linalg.norm(puck[easier, 2:], axis=1, keepdims=True)
                    if self.recovery_easy_arc:
                        angle = rng.uniform(-self.recovery_easy_arc, self.recovery_easy_arc, len(easier))
                        dx, dy = direction.T.copy()
                        direction = np.column_stack((dx*np.cos(angle)-dy*np.sin(angle), dx*np.sin(angle)+dy*np.cos(angle)))
                    paddle[easier] = puck[easier, :2] + direction*rng.uniform(.13, .19, (len(easier), 1))
            defense = kind == 2
            k = defense.sum()
            puck[defense, 0] = rng.uniform(0.15, 0.85, k)
            puck[defense, 1] = rng.uniform(0.9, 1.6, k)
            mouth = self.cfg.goal_width / 2 - self.cfg.puck_radius - .005
            aim_half = mouth if self.wide_defense else .07
            velocity = np.column_stack(
                (rng.uniform(.5-aim_half, .5+aim_half, k) - puck[defense, 0], -puck[defense, 1])
            )
            max_incoming = self.defense_max_speed if self.stage >= 4 else 4 + 2 * spread
            velocity *= rng.uniform(self.defense_min_speed, max_incoming, k)[:, None] / np.maximum(
                np.linalg.norm(velocity, axis=1, keepdims=True), 1e-8
            )
            puck[defense, 2:] = velocity
            if self.stage >= 4:
                from airhockey.policy_benchmark import bank_defense_launches

                # Independent of slot index: otherwise an odd-only defense
                # role receives banks exclusively and never direct attacks.
                banks = defense & (rng.random(n) < .5)
                puck[banks] = bank_defense_launches(
                    int(rng.integers(2**31)), banks.sum(),
                    config=self.cfg,
                    **(dict(speed_range=(self.defense_min_speed, max_incoming),
                            goal_half_width=mouth) if self.wide_defense else {})
                )
                # The benchmark alternates wall sides within a batch. Training
                # often resets only one finished drill, which would otherwise
                # always select its first (right-wall) launch. Random reflection
                # keeps both directions represented even for singleton resets.
                reflect = banks & (rng.random(n) < 0.5)
                puck[reflect, 0] = self.cfg.width - puck[reflect, 0]
                puck[reflect, 2] *= -1
            if self.defense_windup_fraction:
                windup = defense & (rng.random(n) < self.defense_windup_fraction)
                k = windup.sum()
                # An isolated shooting-machine exercise: give the learner time
                # to prepare before a sudden fast direct launch. Existing
                # immediate-launch drills primarily teach reacting afterward.
                # Launch time/direction are hidden from the actor.
                puck[windup, 0] = rng.uniform(.15, .85, k)
                puck[windup, 1] = rng.uniform(1.08, 1.25, k)
                velocity = np.column_stack((rng.uniform(.5-mouth, .5+mouth, k) - puck[windup, 0], -puck[windup, 1]))
                velocity *= rng.uniform(8, self.defense_max_speed, k)[:, None] / np.linalg.norm(velocity, axis=1, keepdims=True)
                release_velocity[windup] = velocity
                puck[windup, 2:] = 0
                if self.defense_windup_lateral_speed:
                    # Actual free puck motion before an externally timed shot;
                    # use the normal physics (including friction/wall bounces).
                    puck[windup, 2] = rng.uniform(-self.defense_windup_lateral_speed,
                                                self.defense_windup_lateral_speed, k)
            paddle[defense] = [0.5, 0.25]
            if self.random_defense_start_fraction:
                from airhockey.policy_benchmark import random_paddle_starts

                varied = defense & (rng.random(n) < self.random_defense_start_fraction)
                paddle[varied] = random_paddle_starts(
                    rng, puck[varied], self.base._ws,
                    self.cfg.puck_radius + self.cfg.paddle_radius + .01,
                )
            if self.stage >= 2:
                random_start = (kind == 0) & (rng.random(n) < 0.5)
                paddle[random_start] = rng.uniform(
                    self.decoder.low + 0.02,
                    self.decoder.high - 0.02,
                    (random_start.sum(), 2),
                )
                bad = np.linalg.norm(paddle - puck[:, :2], axis=1) < 0.11
                paddle[bad] = puck[bad, :2] - [0, 0.16]
            paddle = np.clip(
                paddle, self.decoder.low + 0.005, self.decoder.high - 0.005
            )
            if self.shot_setup_fraction:
                # Easy striking starts expose successful examples of all routes.
                # This changes reset states only; the actor still chooses every
                # action, and explicit evaluation fixtures are never modified.
                requests = rng.choice(4, n, p=self.base._shot_type_p)
                rail_x = np.where(requests == 1, self.cfg.puck_radius, self.cfg.width - self.cfg.puck_radius)
                dx = np.where(requests == 3, self.cfg.width / 2 - puck[:, 0],
                              rail_x - puck[:, 0] + self.cfg.wall_tangential / self.cfg.wall_restitution * (rail_x - self.cfg.width / 2))
                direction = np.column_stack((dx, self.cfg.height - puck[:, 1]))
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                gap = self.cfg.puck_radius + self.cfg.paddle_radius + rng.uniform(.004, .025, n)
                ready = puck[:, :2] - gap[:, None] * direction
                eligible = (kind == 0) & (rng.random(n) < self.shot_setup_fraction)
                eligible &= ((ready > self.decoder.low + .002) & (ready < self.decoder.high - .002)).all(1)
                paddle[eligible] = ready[eligible]
                starting_requests[eligible] = requests[eligible]
            if self.stationary_replay is not None:
                selected = np.flatnonzero((kind == 0) & (rng.random(n) < self.stationary_replay_fraction))
                examples = rng.integers(len(self.stationary_replay['puck']), size=len(selected))
                puck[selected] = self.stationary_replay['puck'][examples]
                puck[selected, 2:] = 0
                puck[selected, :2] += rng.uniform(-.005, .005, (len(selected), 2))
                paddle[selected] = np.clip(self.stationary_replay['paddle'][examples], self.decoder.low+.002, self.decoder.high-.002)
                replay_actions[selected] = self.stationary_replay['previous_action'][examples]
                if 'initial_load' in self.stationary_replay:
                    replay_loads[selected]=self.stationary_replay['initial_load'][examples]
                    exact=np.isfinite(replay_loads[selected]).all(axis=1)
                    # Measured failure starts include tight rail clearances;
                    # preserve them rather than jittering through a wall.
                    puck[selected[exact]]=self.stationary_replay['puck'][examples[exact]]
                    fringe=((puck[selected,:2]<self.decoder.low+.04)
                            |(puck[selected,:2]>self.decoder.high-.04)).any(axis=1)
                    replay_fringe[selected]=exact&fringe
                starting_requests[selected] = 1 + self.stationary_replay['request'][examples].argmax(1)
            if self.edge_drill_fraction:
                edge = (kind == 0) & (rng.random(n) < self.edge_drill_fraction)
                k = int(edge.sum())
                radius = self.cfg.puck_radius + self.cfg.paddle_radius
                # Both side fringes, including the narrow contact strip outside
                # paddle-center travel. Leave a real guarded contact margin.
                left = rng.random(k) < .5
                margin = rng.uniform(.0005, radius, k)
                # Half the drills cover the thin, still-contactable strip where
                # a small alignment error prevents contact entirely.
                thin = rng.random(k) < .5
                margin[thin] = rng.uniform(.0005, .01, int(thin.sum()))
                puck[edge, 0] = np.where(left, self.decoder.low[0] + .002 - radius + margin,
                                         self.decoder.high[0] - .002 + radius - margin)
                puck[edge,0]=np.clip(puck[edge,0],self.cfg.puck_radius+.001,self.cfg.width-self.cfg.puck_radius-.001)
                puck[edge, 1] = rng.uniform(self.decoder.low[1] + .12, self.decoder.high[1] - .05, k)
                puck[edge, 2:] = 0
                paddle[edge] = rng.uniform(self.decoder.low + .005, self.decoder.high - .005, (k, 2))
                # The expanded workspace permits starts overlapping a rail
                # puck; those would grant a contact without an actual approach.
                overlap=np.linalg.norm(paddle[edge]-puck[edge,:2],axis=1)<radius+.005
                selected=np.flatnonzero(edge)[overlap]
                paddle[selected]=[(self.decoder.low[0]+self.decoder.high[0])/2,
                                  (self.decoder.low[1]+self.decoder.high[1])/2]
                replay_actions[edge] = np.nan
                starting_requests[edge] = 0
            if self.corner_drill_fraction:
                # Reset-only practice for pucks behind/beside the paddle.
                # Bouncing them out or leaving an expensive hold is learned.
                corner=(kind==0)&(rng.random(n)<self.corner_drill_fraction)
                k=int(corner.sum());left=rng.random(k)<.5
                inset=rng.uniform(self.cfg.puck_radius+.002,self.decoder.low[0]+.015,k)
                puck[corner,0]=np.where(left,inset,self.cfg.width-inset)
                puck[corner,1]=rng.uniform(self.cfg.puck_radius+.002,self.decoder.low[1]+.02,k)
                tight=rng.random(k)<.35
                inset=np.where(tight,self.cfg.puck_radius+.0002,inset)
                puck[corner,0]=np.where(left,inset,self.cfg.width-inset)
                puck[np.flatnonzero(corner)[tight],1]=self.cfg.puck_radius+.0002
                puck[corner,2:]=0
                paddle[corner,0]=np.where(left,self.decoder.low[0]+.015,self.decoder.high[0]-.015)
                paddle[corner,1]=puck[corner,1]+rng.uniform(.12,.22,k)
                # Practice approaching both along the side rail and along the
                # back rail. These are reset poses, never prescribed actions.
                side_start=rng.random(k)<.5
                ids_side=np.flatnonzero(corner)[side_start]
                distance=rng.uniform(.12,.22,len(ids_side))
                paddle[ids_side,0]=puck[ids_side,0]+np.where(left[side_start],distance,-distance)
                paddle[ids_side,1]=self.decoder.low[1]+.015
                replay_actions[corner]=np.nan;starting_requests[corner]=0
                edge|=corner
            edge |= replay_fringe & np.isfinite(replay_actions[:,0])
            fixtures = Fixtures(np.zeros(n, int), puck, paddle, np.full(n, 0.5))
        else:
            self.kind[ids] = fixtures.task
        # Preserve the caller's fixtures but use a uniform reward/task semantics.
        fi = Fixtures(np.zeros(n, int), fixtures.puck, fixtures.paddle, fixtures.aim)
        obs = super().reset(
            seed=seed, mask=mask, fixtures=fi, opponent=opponent or "idle"
        )
        game_ids = ids[ids < round(self.n_envs * self.game_fraction)]
        if len(game_ids):
            game_mask = np.zeros(self.n_envs, bool)
            game_mask[game_ids] = True
            # Base reset initializes real game serves and synchronizes histories.
            self.base._opp_policy_id[game_ids] = _OPP_POLICY_MAP[opponent or "external"]
            self.base.reset(mask=game_mask)
            self.task[game_ids] = 3
            self.kind[game_ids] = 3
        self.recovery_drill[ids] = recovery & (self.kind[ids] == 1)
        self._recovery_fast_fallback[ids] = False
        velocity = np.column_stack((self.engine.puck_vx[ids], self.engine.puck_vy[ids]))
        delta = np.column_stack((self.engine.paddle_agent_x[ids]-self.engine.puck_x[ids],
                                self.engine.paddle_agent_y[ids]-self.engine.puck_y[ids]))
        self._recovery_started_behind[ids] = (delta*velocity).sum(axis=1) < 0
        self._recovery_ahead_paid[ids] = False
        self.defense_windup[ids] = windup & (self.kind[ids] == 2)
        self.edge_drill[ids] = edge & (self.kind[ids] == 0)
        self._edge_contact[:, ids] = False
        self._edge_paid[:, ids] = False
        self.edge_recovery_count[:, ids] = 0
        self._windup_release[ids] = np.inf
        self._windup_aim[ids] = .5
        windup_ids = ids[self.defense_windup[ids]]
        if len(windup_ids):
            self._windup_release[windup_ids] = rng.uniform(.3, 1.2, len(windup_ids))
            self._windup_velocity[windup_ids] = release_velocity[self.defense_windup[ids]]
            v = self._windup_velocity[windup_ids]
            self._windup_aim[windup_ids] = (self.engine.puck_x[windup_ids]
                - self.engine.puck_y[windup_ids] * v[:, 0] / v[:, 1])
            self.base._opp_policy_id[windup_ids] = _OPP_POLICY_MAP["idle"]
        if self.random_practice_opponent:
            practice_ids = ids[(ids >= round(self.n_envs * self.game_fraction)) & ~self.defense_windup[ids]]
            if len(practice_ids):
                local = rng.uniform(
                    self.decoder.low, self.decoder.high, (len(practice_ids), 2)
                )
                local[:, 1] = self.cfg.height - local[:, 1]
                b, e = self.base, self.engine
                for col, axis in enumerate(("x", "y")):
                    value = local[:, col]
                    getattr(e, "paddle_opp_" + axis)[practice_ids] = value
                    for dyn in (b._opp_dyn, b._opp_dyn_free):
                        dyn[axis][practice_ids] = value
                        dyn["command_" + axis][practice_ids] = value
                    for prefix in ("opp", "own_opp"):
                        getattr(b, "_prev_" + prefix + "_" + axis)[practice_ids] = value
                    if b._cam_active:
                        b._cam_ring[:, practice_ids, 4 + col] = value
        if self.practice_selfplay_fraction:
            practice_ids = ids[(ids >= round(self.n_envs * self.game_fraction)) & ~self.defense_windup[ids]]
            active = practice_ids[
                rng.random(len(practice_ids)) < self.practice_selfplay_fraction
            ]
            self.base._opp_policy_id[active] = _OPP_POLICY_MAP["external"]
        for value in (
            self.touch_count,
            self.shot_count,
            self.aimed_count,
            self.fast_aimed_count,
            self.unproductive_returns,
            self.slow_possession_losses,
            self._slow_loss_paid,
            self.shot_route_counts,
            self.aimed_route_counts,
            self.last_shot_route,
            self.last_shot_request,
            self.shot_speed_sum,
            self.capture_count,
            self.convert_count,
            self.captured,
            self.capture_paid,
            self.control_time,
            self.reward_events,
            self.guard_changed,
            self.guard_unresolved,
            self.peak_load,
            self.overload_seconds,
            self.motion_accel_peak,
            self.motion_speed_peak,
            self.high_accel_seconds,
            self.shot_speed_histogram,
            self.target_histogram,
            self.turnovers,
            self.entry_outcomes,
            self.entry_speed,
            self.entry_outcomes_by_speed,
            self.productive_entries_by_speed,
            self.opportunity_entries_by_speed,
            self.productive_opportunities_by_speed,
            self.entry_flags,
            self.entry_active,
            self.entry_reachable_time,
        ):
            value[:, ids] = 0
        self.contact_clock[:, ids] = -100
        self.passive_returns[ids] = 0
        self.own_touched[ids] = False
        self.entry_active[0, ids] = self.engine.puck_y[ids] < 1
        self.entry_active[1, ids] = self.engine.puck_y[ids] > 1
        self.entry_speed[:, ids] = np.hypot(self.engine.puck_vx[ids], self.engine.puck_vy[ids])
        if self.shot_conditioned:
            # Fixtures may replace the original serve after base.reset().
            # Initialize requests against the actual new puck state and keep
            # base possession tracking synchronized, including partial resets.
            own = self.engine.puck_y < self.cfg.height / 2
            far = self.engine.puck_y > self.cfg.height / 2
            self.base._prev_in_half[ids] = own[ids]
            self.base._prev_in_far[ids] = far[ids]
            self.base._shot_type[ids] = 0
            self.base._shot_type_opp[ids] = 0
            if self.fixed_opponent_style_fraction:
                probabilities = self.base._shot_type_p_opp
                probabilities[ids] = self.base._shot_type_p
                fixed = ids[(self.kind[ids] == 3) & (rng.random(len(ids)) < self.fixed_opponent_style_fraction)]
                probabilities[fixed] = np.eye(4)[rng.integers(1, 4, len(fixed))]
            self.base._draw_shot_types(mask & own, agent=True)
            self.base._draw_shot_types(mask & far, agent=False)
            prepared = (starting_requests != 0) & (self.kind[ids] != 3)
            self.base._shot_type[ids[prepared]] = starting_requests[prepared]
            if len(windup_ids):
                self.base._shot_type[windup_ids] = rng.integers(1, 4, len(windup_ids))
        # Randomize initial heat in training drills; keep game heat continuous.
        skill_ids = ids[(self.kind[ids] != 3) | self._shutdown_reset[ids]
                        | (self.warm_game_start & self._initial_thermal_state[ids])]
        if len(skill_ids):
            for load in self.loads:
                level = rng.uniform(0.05, self.warm_start_max, (len(skill_ids), 2, 4))
                if self.correlated_warm_fraction:
                    hot = rng.random(len(skill_ids)) < self.correlated_warm_fraction
                    level[hot] = rng.uniform(min(.75, self.warm_start_max), self.warm_start_max, (hot.sum(), 1, 1))
                if self.cold_practice_fraction:
                    cold = (self.kind[skill_ids] != 3) & (rng.random(len(skill_ids)) < self.cold_practice_fraction)
                    level[cold] = rng.uniform(.05, min(.4,self.warm_start_max), (cold.sum(),2,4))
                if self.cold_game_start_fraction:
                    cold = ((self.kind[skill_ids] == 3) & self._initial_thermal_state[skill_ids]
                            & (rng.random(len(skill_ids)) < self.cold_game_start_fraction))
                    level[cold] = rng.uniform(.05, min(.4,self.warm_start_max), (cold.sum(),2,4))
                load.h[skill_ids] = level**2
                load.observed[skill_ids] = level
        self._shutdown_reset[ids] = False
        self._initial_thermal_state[ids] = False
        if self.stationary_rest_fraction:
            resting = ids[(self.kind[ids] == 0) & (rng.random(n) < self.stationary_rest_fraction)]
            if len(resting):
                # A physically consistent previous arrival action that held
                # the current paddle still. Quiet full-game states otherwise
                # differ from every drill's all-zero action-history reset.
                point = np.column_stack((self.engine.paddle_agent_x[resting], self.engine.paddle_agent_y[resting]))
                self.last_action[resting] = 0
                self.last_action[resting, :2] = 2 * (point-self.decoder.low) / (self.decoder.high-self.decoder.low) - 1
                self.last_action[resting, 4] = rng.uniform(-1, 1, len(resting))
                self.last_action[resting, 5] = 1
        replayed = np.isfinite(replay_actions[:, 0]) & (self.kind[ids] == 0)
        self.last_action[ids[replayed]] = replay_actions[replayed]
        thermal_replay=replayed&np.isfinite(replay_loads).all(axis=1)
        if thermal_replay.any():
            levels=replay_loads[thermal_replay].reshape(-1,2,4)
            self.loads[0].h[ids[thermal_replay]]=levels**2
            self.loads[0].observed[ids[thermal_replay]]=levels
        # Reset already computed observation may contain pre-randomization loads.
        # Reuse raw measured state; no extra _make_obs_direct velocity update.
        raw = self.base._make_obs_direct()
        obs = self._features(raw)
        if self._last_observation is not None:
            obs[~mask] = self._last_observation[~mask]
        self._last_observation = obs.copy()
        self._potential = self.potential()
        return obs

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (self.n_envs, 6) or not np.isfinite(action).all():
            raise ValueError("expected finite [N,6] arrival actions")
        action = np.clip(action, -1, 1)
        target, cap = self._decode(action)
        b, e = self.base, self.engine
        launch = self.elapsed >= self._windup_release
        if launch.any():
            if self.defense_windup_lateral_speed:
                # Retain the sampled hidden goal target and shot speed, but
                # launch from where the puck actually drifted to. No teleport.
                ids = np.flatnonzero(launch)
                v = self._windup_velocity[ids]
                aim = self._windup_aim[ids]
                direction = np.column_stack((aim-e.puck_x[ids], -e.puck_y[ids]))
                self._windup_velocity[ids] = direction * (
                    np.linalg.norm(v, axis=1) / np.maximum(np.linalg.norm(direction, axis=1), 1e-9))[:, None]
            e.puck_vx[launch], e.puck_vy[launch] = self._windup_velocity[launch].T
            self._windup_release[launch] = np.inf
        low = 2 * (target - b._action_low) / (b._action_high - b._action_low) - 1
        accel = (
            2 * np.sqrt(np.clip((cap / b._agent_dyn["max_accel"] - 0.05) / 0.95, 0, 1))
            - 1
        )
        previous_action = self.last_action.copy()
        self.last_action[:] = action
        self.reward_events[:] = 0
        self.load_cost[:] = 0
        self.peak_accel[:] = 0
        scores, conceded = e.score_agent.copy(), e.score_opponent.copy()
        previous_turnovers = self.turnovers[0].copy()
        old_own = e.puck_y < 1
        b.referee_active_mask = self.kind == 3
        raw, _, _, _, info = b.step(np.column_stack((low, accel)))
        self.elapsed += b.action_dt
        gf, ga = e.score_agent - scores, e.score_opponent - conceded
        if self.recovery_get_ahead_bonus:
            self.reward_events[0] += self.recovery_get_ahead_bonus * self._recovery_approach_event()
        speed = np.hypot(e.puck_vx, e.puck_vy)
        for side in (0, 1):
            prefix = "paddle_opp" if side else "paddle_agent"
            gap = (
                np.hypot(
                    e.puck_x - getattr(e, prefix + "_x"),
                    e.puck_y - getattr(e, prefix + "_y"),
                )
                - 0.0911
            )
            local_puck = np.column_stack(
                (e.puck_x, self.cfg.height - e.puck_y if side else e.puck_y)
            )
            closest = np.clip(local_puck, self.decoder.low, self.decoder.high)
            reachable = np.linalg.norm(local_puck - closest, axis=1) <= (
                self.cfg.puck_radius + self.cfg.paddle_radius
            )
            controlled = (
                (speed < 0.65)
                & (gap < 0.07)
                & (gap > -0.02)
                & (self.elapsed - self.contact_clock[side] < 0.6)
                & reachable
            )
            own_half = e.puck_y > 1 if side else e.puck_y < 1
            self.captured[side] &= (gap < 0.30) & own_half & reachable
            self.capture_paid[side, ~own_half] = False
            self.control_time[side] = np.where(
                controlled, self.control_time[side] + 0.02, 0
            )
            newly = (self.control_time[side] >= 0.12) & ~self.captured[side]
            self.capture_count[side] += newly
            self.entry_flags[side, newly, 1] = True
            self.captured[side] |= newly
            self.reward_events[side] += self.capture_weight * (
                newly & ~self.capture_paid[side]
            )
            if side == 0:
                self.reward_events[side] += self.recovery_capture_bonus * (
                    newly & ~self.capture_paid[side] & self.recovery_drill
                )
            self.capture_paid[side] |= newly
            if self.edge_recovery_weight:
                self.reward_events[side] += self.edge_recovery_weight * self._edge_recovery_event(side, excluded=(gf + ga) > 0)
            levels = self.loads[side].levels.max(axis=(1, 2))
            self.peak_load[side] = np.maximum(self.peak_load[side], levels)
            self.overload_seconds[side] += 0.02 * (levels >= 1)
            ended = self.entry_active[side] & ~own_half
            self.reward_events[side] -= self.slow_exit_penalty * self._slow_possession_loss(side, (gf+ga)>0)
            self._slow_loss_paid[side, ended | ((gf+ga)>0)] = False
            self._edge_contact[side, ended | ((gf + ga) > 0)] = False
            self._edge_paid[side, ended | ((gf + ga) > 0)] = False
            unproductive = (
                ended & (self.entry_reachable_time[side] >= 0.1)
                & ~self.entry_flags[side, :, 1] & ~self.entry_flags[side, :, 3]
                & ((gf + ga) == 0)
            )
            self.unproductive_returns[side] += unproductive
            self.reward_events[side] -= self.unproductive_return_penalty * unproductive
            self.entry_outcomes[side, ended, 0] += 1
            self.entry_outcomes[side, ended, 1:5] += self.entry_flags[side, ended]
            self.entry_outcomes[side, ended, 5] += (
                self.entry_reachable_time[side, ended] >= 0.1
            ) & ~self.entry_flags[side, ended, 0]
            ended_ids = np.flatnonzero(ended)
            speed_bins = np.digitize(self.entry_speed[side, ended], [2, 5, 8])
            outcomes = np.column_stack((
                np.ones(len(ended_ids), int), self.entry_flags[side, ended],
                (self.entry_reachable_time[side, ended] >= 0.1) & ~self.entry_flags[side, ended, 0],
            ))
            self.entry_outcomes_by_speed[side, ended_ids, speed_bins] += outcomes
            productive = self.entry_flags[side, ended, 1] | self.entry_flags[side, ended, 3]
            opportunity = self.entry_reachable_time[side, ended] >= .1
            self.productive_entries_by_speed[side, ended_ids, speed_bins] += productive
            self.opportunity_entries_by_speed[side, ended_ids, speed_bins] += opportunity
            self.productive_opportunities_by_speed[side, ended_ids, speed_bins] += productive & opportunity
            entered = ~self.entry_active[side] & own_half
            self.entry_speed[side, entered] = speed[entered]
            self.entry_flags[side, entered] = False
            self.entry_reachable_time[side, entered] = 0
            self.entry_reachable_time[side] += 0.02 * (own_half & reachable)
            self.entry_active[side] = own_half
        returned = old_own & (e.puck_y >= 1) & (gf + ga == 0)
        passive = returned & ~self.own_touched
        self.passive_returns += passive
        self.own_touched[~old_own & (e.puck_y < 1)] = False
        game = self.kind == 3
        # Self-play must not reward mutually farming gentle exchanges. Skill
        # outcomes are competitive in games; real goals remain the main prize.
        reward = np.where(game, self.game_goal_weight, self.skill_goal_weight) * gf
        reward -= np.where(game, self.game_goal_weight, self.skill_concede_weight) * ga
        reward = reward + self.reward_events[0] - game * self.reward_events[1] - 0.005
        reward -= 0.5 * passive
        reward += 0.5 * info["penalty"]
        reward += (25 - self.turnover_weight) * (self.turnovers[0] - previous_turnovers)
        reward -= self.load_weight * self.load_cost
        # Bounded per-second cost near the two side rails and back rail.
        # A brief interception costs little; there is no edge exclusion or
        # action override. Opponent-side dwell cannot earn positive reward.
        edge_cost = self.edge_dwell_cost()
        reward -= self.edge_dwell_weight * b.action_dt * edge_cost
        reward -= (game | self.defense_windup) * self.readiness_cost_weight * b.action_dt * self._readiness_running_cost
        # Small action-change cost, not a cap on learned rapid responses.
        reward -= 0.002 * ((action - previous_action) ** 2).sum(axis=1)
        skill = self.kind != 3
        skill_seconds = np.where(self.edge_drill, 12, 6 if self.possession_followthrough else 4)
        terminal = skill & (((gf + ga) > 0) | (self.elapsed >= skill_seconds))
        reward[terminal & (gf == 0)] -= 2
        # Expiring without attempting an aimed shot must not be cheaper than
        # playing. A conceded goal already has its own penalty; do not charge
        # both, which would restore an incentive to wait out the exercise.
        stationary_timeout = (self.elapsed >= skill_seconds) & ((gf + ga) == 0)
        reward -= self.stationary_failure_penalty * (stationary_timeout & (self.kind == 0) & (self.aimed_count[0] == 0))
        if self.defense_clear_reward:
            # Training-only isolated defense: a touched puck cleared into the
            # opponent half, or measured control, is a successful defense.
            # A miss that merely times out cannot earn this reward. Full games
            # keep their competitive scoring and ordinary termination rules.
            defended = (self.kind == 2) & (ga == 0) & (
                (self.capture_count[0] > 0)
                | ((self.touch_count[0] > 0) & (e.puck_y >= 1) & ((e.puck_vy > 0) | (gf > 0)))
            )
            terminal |= defended
            reward[defended] += self.defense_clear_reward
        if self.possession_followthrough:
            receive = self.kind == 1
            reward[receive] -= self.skill_goal_weight * gf[receive]
            converted = self.convert_count[0] > 0
            fallback = ((self.fast_aimed_count[0] > 0) & ~self.recovery_drill) | self._recovery_fast_fallback
            success = converted | fallback
            finish = receive & (success | terminal | (e.puck_y >= self.cfg.height / 2))
            terminal |= finish
            reward[finish & converted] += 60
            fallback_finish = finish & ~converted & fallback
            fallback_credit = 35 * np.where(self.recovery_drill, self.recovery_fallback_scale, 1)
            reward[fallback_finish] += fallback_credit[fallback_finish]
            reward[finish & ~success] -= 60
        elif self.receive_drill or self.productive_receive_drill:
            receive = self.kind == 1
            # Curriculum episodes end at a successful receive, not a volley.
            # The same network learns stationary shots in neighboring slots;
            # no task label is exposed and deployment has no such episode rule.
            reward[receive] -= self.skill_goal_weight * gf[receive]
            # Scoring an early volley also ends a skill episode. It must count
            # as a failed reception, rather than escape the uncaught-puck cost
            # simply by reaching the goal before the drill's time limit.
            captured = self.capture_count[0] > 0
            fallback = self.fast_aimed_count[0] > 0
            success = captured | (self.productive_receive_drill & fallback)
            finish = receive & (success | (self.elapsed >= 1.2) | terminal)
            terminal |= finish
            failed = finish & ~success
            gap = np.hypot(e.puck_x - e.paddle_agent_x, e.puck_y - e.paddle_agent_y)
            reward[failed] -= (2 * np.minimum(speed, 5) + 3 * np.minimum(gap, 1))[
                failed
            ]
            if self.productive_receive_drill:
                # A short curriculum exercise: control is preferred, but an
                # accurate >=4 m/s return is a useful alternative. Avoid paying
                # the full requested-shot bonus while ending immediately at
                # control, which would unfairly favor volleys in this exercise.
                # Full games and ordinary strike exercises keep their rewards.
                reward[receive] -= np.maximum(self.reward_events[0, receive], 0)
                reward[finish & captured] += 60
                reward[finish & ~captured & fallback] += 35
                reward[failed] -= 30
        if self.possession_delay_weight:
            waiting = np.clip(self.base._t_side - 4, 0, 10)
            reward -= game * self.possession_delay_weight * b.action_dt * waiting * np.where(e.puck_y < self.cfg.height / 2, 1, -1)
        truncate = ~skill & (self.elapsed >= self.game_episode_seconds)
        shutdown = np.zeros(self.n_envs, bool)
        if self.terminate_overload:
            shutdown = self.loads[0].levels.max(axis=(1, 2)) >= self.shutdown_level
            terminal |= shutdown
            reward[shutdown] -= self.shutdown_penalty
            self._shutdown_reset |= shutdown
        after = self.potential()
        reward += self.discount * np.where(terminal, 0, after) - self._potential
        self._potential = after
        if self.goals_only:
            # Deliberately bypass ALL shaping, including hard-coded time,
            # passive-return, effort and potential-difference terms. Keep
            # event counters for evaluation, but give PPO only real goals.
            reward = gf.astype(np.float32) - ga.astype(np.float32)
        self.captured[:, (gf + ga) > 0] = False
        self.capture_paid[:, (gf + ga) > 0] = False
        self.own_touched[(gf + ga) > 0] = False
        self.control_time[:, (gf + ga) > 0] = 0
        obs = self._features(raw)
        self._last_observation = obs.copy()
        info.update(
            shutdown=shutdown,
            critic_context=self.critic_context(),
            goals=gf,
            conceded=ga,
            kind=self.kind.copy(),
            contacts=self.touch_count[0].copy(),
            captures=self.capture_count[0].copy(),
            edge_recoveries=self.edge_recovery_count[0].copy(),
            conversions=self.convert_count[0].copy(),
            shots=self.shot_count[0].copy(),
            aimed=self.aimed_count[0].copy(),
            shot_speed_sum=self.shot_speed_sum[0].copy(),
            passive_returns=self.passive_returns.copy(),
            unproductive_returns=self.unproductive_returns[0].copy(),
            slow_possession_losses=self.slow_possession_losses[0].copy(),
            request_matches=np.trace(self.aimed_route_counts[0, :, 1:, 1:], axis1=1, axis2=2),
            load_peak=self.peak_load[0].copy(),
            overload_seconds=self.overload_seconds[0].copy(),
            peak_accel=self.peak_accel.copy(),
            edge_dwell_cost=edge_cost.copy(),
        )
        return (
            obs,
            (reward if self.goals_only else self.reward_scale * reward).astype(np.float32),
            terminal,
            truncate,
            info,
        )

    def edge_dwell_cost(self):
        e=self.engine
        margin=np.column_stack((e.paddle_agent_x-self.decoder.low[0],
            self.decoder.high[0]-e.paddle_agent_x,e.paddle_agent_y-self.decoder.low[1]))
        return np.square(np.clip(1-margin/self.edge_dwell_band,0,1)).sum(axis=1)
