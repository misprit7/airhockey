#!/usr/bin/env python3
"""PPO for a single neural player; simulation only, no tactical controller."""

import argparse
import copy
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import signal
from airhockey.dynamics import workspace_in_sim
import time

import numpy as np
import torch

from airhockey.neural_player import NeuralPlayer, PhysicalHistory, RecoveryExplorationBias, log_probability, defensive_request_consistency, recovery_exploration_scale, recovery_exploration_window, thermal_effort_exploration_scale, backtrack_actor_step
from airhockey.neural_training import NeuralTrainingEnv
from airhockey.on_policy import advantages
from airhockey.thermal import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RUN = None
ACTIVE_SAVE = None
ACTIVE_ENV = None


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def make_optimizer(net, actor_lr, value_lr, saved=None):
    groups = [
        {"params": net.actor_parameters(), "lr": actor_lr},
        {"params": net.value_parameters(), "lr": value_lr},
    ]
    optimizer = torch.optim.Adam(groups, eps=1e-5)
    if saved is not None:
        if len(saved["param_groups"]) == 1:
            # Older files used module parameter order in one group. Load that
            # order first, then retain moments keyed by the actual Parameters.
            previous = torch.optim.Adam(net.parameters(), lr=actor_lr, eps=1e-5)
            previous.load_state_dict(saved)
            optimizer.state = previous.state
        else:
            optimizer.load_state_dict(saved)
        optimizer.param_groups[0]["lr"] = actor_lr
        optimizer.param_groups[1]["lr"] = value_lr
    return optimizer


def main():
    global ACTIVE_RUN, ACTIVE_SAVE, ACTIVE_ENV
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--evaluation-dir", type=Path, help="Local evaluation artifacts for the training dashboard")
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float)
    p.add_argument("--std-scale", type=float, default=1)
    p.add_argument("--entropy", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.025,
                   help="Stop actor updates within a rollout once sampled policy KL exceeds this value")
    p.add_argument("--backtrack-kl", action="store_true",
                   help="Shorten each actor optimizer proposal if its post-step minibatch KL exceeds target-kl")
    p.add_argument("--stage", type=int, default=0)
    p.add_argument('--accel',type=float,default=60)
    p.add_argument('--workspace',choices=['legacy','rail30'],default='legacy')
    p.add_argument('--thermal-model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--edge-dwell-weight',type=float,default=0,help='Maximum cost per second per nearby rail')
    p.add_argument('--edge-dwell-band',type=float,default=.06,help='Soft penalty band inside workspace boundary, meters')
    p.add_argument('--project-rail-contacts',action='store_true',help='Resolve rail penetration introduced by paddle contact')
    p.add_argument('--edge-clearance-weight',type=float,default=0,help='Bounded potential for recovering a puck from rail corners')
    p.add_argument("--seed", type=int, default=20261801)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--width", type=int, help="Hidden width; inherit on resume, otherwise 256")
    p.add_argument("--history", type=int, help="Physical frames; inherit on resume, otherwise one")
    p.add_argument("--history-deltas", action="store_true",
                   help="Learn from past-minus-current physical frames; checkpoint migration preserves initial behavior")
    p.add_argument("--freeze-base-width", type=int,
                   help="Keep this existing actor prefix fixed after widening; inherit on resume, otherwise zero")
    p.add_argument("--shot-conditioned", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--shot-request", choices=["random", "left", "right", "straight"], default="random")
    p.add_argument("--shot-request-weight", type=float, default=0)
    p.add_argument("--wrong-shot-penalty", type=float, default=0)
    p.add_argument("--fallback-shot-scale", type=float, default=0)
    p.add_argument("--unproductive-return-penalty", type=float, default=0)
    p.add_argument("--setup-weight", type=float, default=3)
    p.add_argument("--setup-avoid-puck", action="store_true")
    p.add_argument("--control-potential-weight", type=float, default=0,
                   help="Bounded potential for slowing a recently touched, nearby puck")
    p.add_argument("--shot-setup-fraction", type=float, default=0,
                   help="Fraction of stationary training resets starting near a requested strike")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--resume", type=Path)
    p.add_argument("--initialize-from", type=Path,
                   help="Initialize actor from checkpoint; fresh optimizer, critic, opponent pool and run step count")
    p.add_argument("--goals-only", action="store_true",
                   help="Full-game neural self-play; reward exactly +1 scored/-1 conceded, no shaping")
    p.add_argument("--minutes", type=float, default=90)
    p.add_argument("--load-weight", type=float, default=0.15)
    p.add_argument("--load-energy-weight", type=float, default=2.,
        help="Ordinary I-squared energy cost inside the load penalty; near-overload cost remains separate")
    p.add_argument("--load-holding-weight", type=float, default=0.,
        help="Predictive penalty for continued holding at the current position")
    p.add_argument("--load-holding-horizon", type=float, default=15.)
    p.add_argument("--capture-weight", type=float, default=2)
    p.add_argument("--conversion-weight", type=float, default=4)
    p.add_argument("--capture-first", action="store_true")
    p.add_argument("--receive-drill", action="store_true")
    p.add_argument("--productive-receive-drill", action="store_true",
                   help="Short receiving curriculum: prefer control, accept fast aimed fallback shots")
    p.add_argument("--shot-power-weight", type=float, default=2)
    p.add_argument("--shot-power-exponent", type=float, default=1,
                   help="Speed credit exponent for aimed shots and requested routes")
    p.add_argument("--off-target-penalty", type=float, default=1)
    p.add_argument("--conversion-speed-weight", type=float, default=0)
    p.add_argument("--opponent-checkpoint", type=Path)
    p.add_argument("--opponent-reference", action="append", default=[])
    p.add_argument("--opponent-pool-size", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--warm-start-max", type=float, default=0.8)
    p.add_argument("--terminate-overload", action="store_true")
    p.add_argument("--fixed-practice-roles", action="store_true")
    p.add_argument("--random-practice-opponent", action="store_true")
    p.add_argument("--shutdown-penalty", type=float, default=100)
    p.add_argument("--game-goal-weight", type=float, default=25)
    p.add_argument("--skill-goal-weight", type=float, default=10)
    p.add_argument("--skill-concede-weight", type=float, default=12)
    p.add_argument("--turnover-weight", type=float, default=25)
    p.add_argument("--practice-selfplay-fraction", type=float, default=0)
    p.add_argument("--reward-scale", type=float, default=1)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--shutdown-level", type=float, default=1)
    p.add_argument("--practice-defense-fraction", type=float, default=0.25)
    p.add_argument("--defense-min-speed", type=float, default=2)
    p.add_argument('--defense-max-speed',type=float,default=12)
    p.add_argument("--wide-defense", action="store_true",
                   help="Practice direct and bank attacks across the scoring mouth and configured speeds")
    p.add_argument("--random-defense-start-fraction", type=float, default=0)
    p.add_argument("--defense-clear-reward", type=float, default=0)
    p.add_argument("--defense-windup-fraction", type=float, default=0, help="Fraction of defensive practice with a hidden delay before a fast direct launch")
    p.add_argument("--shot-speed-scale", type=float, default=6)
    p.add_argument("--conversion-speed-scale", type=float, default=5)
    p.add_argument("--stochastic-opponents", action="store_true")
    p.add_argument("--receiving-min-speed", type=float, default=0.6)
    p.add_argument("--receiving-max-speed", type=float)
    p.add_argument("--rushed-shot-penalty", type=float, default=0)
    p.add_argument("--thermal-gain", type=float)
    p.add_argument("--ideal-sensing", action="store_true")
    p.add_argument("--report-sensing", action="store_true", help="Match live bounce-aware velocity fitting and real-fix dropout handling")
    p.add_argument("--continuous-rallies", action="store_true", help="No shot-clock resets; only re-serve dead pucks outside both robots' reach")
    p.add_argument("--possession-followthrough", action="store_true", help="Receiving drills continue through control to a shot")
    p.add_argument("--recovery-fraction", type=float, default=0, help="Fraction of receiving starts with a slow outgoing puck")
    p.add_argument("--recovery-min-speed", type=float, default=.15)
    p.add_argument("--recovery-max-speed", type=float, default=1.2)
    p.add_argument("--recovery-easy-fraction", type=float, default=0, help="Recovery curriculum starts already ahead of the departing puck")
    p.add_argument("--recovery-easy-arc", type=float, default=0, help="Radians of angular variation around easy recovery starts, bridging leading and trailing positions")
    p.add_argument("--recovery-capture-bonus", type=float, default=0, help="Once-per-possession control bonus in slow outgoing training drills only")
    p.add_argument("--recovery-fallback-scale", type=float, default=0, help="Later recovery curriculum credit [0,1] for an accurate requested >=6m/s shot without prior control")
    p.add_argument("--recovery-cushion-bonus", type=float, default=0, help="First leading-side contact credit proportional to actual slowing, in recovery drills only")
    p.add_argument("--recovery-cushion-signed", action="store_true", help="Smooth first-contact cushioning feedback, including a bounded cost for overpowered contacts")
    p.add_argument("--recovery-get-ahead-bonus", type=float, default=0, help="Once-only progress credit for going from behind to ahead without touching the outgoing puck")
    p.add_argument("--fixed-opponent-style-fraction", type=float, default=0, help="Fraction of full games whose neural opponent keeps one shot route for the entire game")
    p.add_argument("--recovery-approach-weight", type=float, default=0, help="Potential for meeting a slow outgoing puck on its leading side before control")
    p.add_argument("--readiness-weight", type=float, default=0, help="Training-only potential for coverage of direct goal threats")
    p.add_argument("--readiness-cost-weight", type=float, default=0, help="Per-second cost of exposed defense while the opponent can prepare a shot")
    p.add_argument("--readiness-lateral-uncertainty", type=float, default=0, help="Opponent release-position uncertainty in meters for preparation rewards")
    p.add_argument("--defense-windup-lateral-speed", type=float, default=0, help="Maximum lateral puck drift before a hidden delayed practice shot")
    p.add_argument("--slow-exit-penalty", type=float, default=0, help="Penalty for slow loss of reachable possession without a useful shot, even after capture")
    p.add_argument("--possession-delay-weight", type=float, default=0, help="Per-second cost for prolonged own-half possession, without teleporting the puck")
    p.add_argument("--game-episode-seconds", type=float, default=30)
    p.add_argument("--stationary-failure-penalty", type=float, default=0)
    p.add_argument("--stationary-rest-fraction", type=float, default=0)
    p.add_argument("--stationary-replay", type=Path)
    p.add_argument("--stationary-replay-fraction", type=float, default=0)
    p.add_argument("--edge-drill-fraction", type=float, default=0,
                   help="Fraction of stationary practice starting in reachable side fringes")
    p.add_argument('--corner-drill-fraction',type=float,default=0)
    p.add_argument("--edge-recovery-weight", type=float, default=0,
                   help="Once-per-possession credit for returning a touched fringe puck to the interior")
    p.add_argument("--edge-approach-weight", type=float, default=0,
                   help="Potential for approaching feasible contact with a slow side-fringe puck")
    p.add_argument("--edge-persistent-bias", type=float, default=0,
                   help="Training-only coherent random offsets in quiet side-fringe practice")
    p.add_argument("--correlated-warm-fraction", type=float, default=0)
    p.add_argument("--cold-practice-fraction", type=float, default=0, help="Fraction of short drills initialized below 40 percent load; full games retain their hot-start curriculum")
    p.add_argument("--cold-game-start-fraction", type=float, default=0, help="Fraction of initial full games starting cool; later ordinary resets preserve heat")
    p.add_argument("--warm-game-start", action="store_true", help="Randomize initial game heat too; never clear heat at subsequent game resets")
    p.add_argument("--game-fraction", type=float, default=None, help="Override full-game allocation for focused curricula")
    p.add_argument("--recovery-exploration", type=float, default=0, help="Training-only action std floor for slow departing or distant stationary pucks")
    p.add_argument("--recovery-exploration-seconds", type=float, default=0, help="Limit extra recovery noise to the beginning of receiving drills; 0 retains the unrestricted window")
    p.add_argument("--recovery-exploration-load-aware", action="store_true")
    p.add_argument("--recovery-persistent-bias", type=float, default=0,
                   help="Training-only persistent recovery perturbation strength; 0 disables")
    p.add_argument("--recovery-bias-block-steps", type=int, default=8,
                   help="Hold each recovery perturbation for this many 20ms decisions before first contact")
    p.add_argument("--recovery-explore-timing", action="store_true", help="Also explore arrival time and acceleration allowance in recovery states")
    p.add_argument("--quiet-exploration", action="store_true", help="Also explore near stationary pucks at half the recovery std floor; training only")
    p.add_argument("--hot-effort-exploration", type=float, default=0, help="Training-only raw effort std floor at high motor load, including saturated actions")
    p.add_argument("--practice-only-exploration", action="store_true", help="Restrict extra exploration floors to short drills; full games use the learned Gaussian variance")
    p.add_argument("--defense-request-consistency", type=float, default=0,
                   help="Auxiliary actor loss ignoring expired shot requests while the puck is in the opponent half")
    p.add_argument("--skill-reference", type=Path, help="Training-only learned reference for established stationary/fast receiving skills")
    p.add_argument("--skill-reference-weight", type=float, default=0)
    p.add_argument("--skill-reference-max-load", type=float, default=None, help="Preserve learned reference only below this observed load; let hot behavior change")
    p.add_argument("--skill-reference-incoming-only", action="store_true", help="Preserve only fast incoming responses, leaving quiet-state failures free to change")
    p.add_argument("--skill-replay", type=Path, help="Previously successful neural observations/actions for training-only preservation")
    p.add_argument("--skill-replay-weight", type=float, default=0)
    args = p.parse_args()
    if not np.isfinite(args.accel) or not 0<args.accel<=120:p.error('accel must be within (0,120] m/s²')
    from airhockey.neural_setup import workspace_bounds
    run_bounds=workspace_bounds(args.workspace)
    args.thermal_model=args.thermal_model.resolve()
    from airhockey.neural_setup import checkpoint_environment
    from airhockey.neural_coordinates import CoordinateReference,action_coordinates,observation_coordinates
    coordinate_options=dict(accel=args.accel,workspace_bounds_mm=run_bounds)
    if args.resume and args.initialize_from:
        p.error('choose --resume or --initialize-from')
    if args.goals_only:
        if (args.receive_drill or args.productive_receive_drill or args.possession_followthrough or
                args.terminate_overload or args.skill_reference or args.skill_replay or
                args.skill_reference_weight or args.skill_replay_weight or args.defense_request_consistency or
                args.opponent_checkpoint or args.opponent_reference):
            p.error('--goals-only excludes auxiliary drills/losses, overload terminations and fixed external opponents')
        args.stage, args.game_fraction, args.continuous_rallies = 5, 1., True
    if not np.isfinite(args.recovery_persistent_bias) or not 0 <= args.recovery_persistent_bias <= 3:
        p.error("persistent recovery bias must be finite and in [0,3]")
    if not 1 <= args.recovery_bias_block_steps <= 50:
        p.error("recovery bias blocks must contain 1–50 decisions")
    if not np.isfinite(args.recovery_exploration) or not 0 <= args.recovery_exploration <= 1.5:
        p.error("recovery exploration must be finite and in [0,1.5]")
    if not np.isfinite(args.recovery_exploration_seconds) or not 0 <= args.recovery_exploration_seconds <= 4:
        p.error("recovery exploration window must be finite and in [0,4] seconds")
    if not np.isfinite(args.hot_effort_exploration) or not 0 <= args.hot_effort_exploration <= 3:
        p.error("hot effort exploration must be finite and in [0,3]")
    if args.report_sensing and args.ideal_sensing:
        p.error("--report-sensing and --ideal-sensing are mutually exclusive")
    if not np.isfinite(args.target_kl) or args.target_kl <= 0:
        p.error("--target-kl must be positive and finite")
    if not np.isfinite(args.defense_request_consistency) or args.defense_request_consistency < 0:
        p.error("--defense-request-consistency must be finite and nonnegative")
    if not np.isfinite(args.skill_reference_weight) or args.skill_reference_weight < 0 or (args.skill_reference_weight and args.skill_reference is None):
        p.error("skill reference needs a checkpoint and a finite nonnegative weight")
    if args.skill_reference_max_load is not None and (not np.isfinite(args.skill_reference_max_load) or not 0 < args.skill_reference_max_load <= 1):
        p.error("skill reference maximum load must be within (0,1]")
    if not np.isfinite(args.skill_replay_weight) or args.skill_replay_weight < 0 or (args.skill_replay_weight and args.skill_replay is None):
        p.error("skill replay needs a dataset and a finite nonnegative weight")
    if not np.isfinite(args.edge_persistent_bias) or args.edge_persistent_bias < 0:
        p.error("edge persistent bias must be finite and nonnegative")
    if args.edge_persistent_bias and args.recovery_persistent_bias:
        p.error("choose edge or outgoing-recovery persistent exploration for this run")
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN = run
    initialization = args.resume or args.initialize_from
    state = torch.load(initialization, map_location=device, weights_only=False) if initialization else None
    if args.width is None:
        args.width = state["width"] if state else 256
    if args.history is None:
        args.history = state.get("history", 1) if state else 1
    if args.freeze_base_width is None:
        args.freeze_base_width = state.get("args", {}).get("freeze_base_width", 0) if state and args.resume else 0
    if not 0 <= args.freeze_base_width < args.width:
        raise ValueError("frozen base width must be zero or smaller than model width")
    if args.shot_conditioned is None:
        args.shot_conditioned = state.get("shot_conditioned", False) if state else False
    net = NeuralPlayer(args.width, history=args.history, shot_conditioned=args.shot_conditioned,
                       history_deltas=args.history_deltas).to(device)
    if not np.isfinite(args.std_scale) or args.std_scale <= 0:
        raise ValueError("std-scale must be positive and finite")
    total = 0
    upgraded = False
    saved_optimizer = None
    if initialization:
        fresh_value = copy.deepcopy(net.value_trunk.state_dict()) if args.initialize_from else None
        fresh_head = copy.deepcopy(net.critic.state_dict()) if args.initialize_from else None
        upgraded = net.load_weights(state["model"])
        if args.initialize_from:
            net.value_trunk.load_state_dict(fresh_value)
            net.critic.load_state_dict(fresh_head)
        elif not upgraded:
            saved_optimizer = state["optimizer"]
        total = state["step"] if args.resume else 0
    args.history_deltas = net.history_deltas
    optimizer = make_optimizer(net, args.lr, args.value_lr or args.lr, saved_optimizer)
    from airhockey.neural_player import ActorPrefixFreeze
    skill_reference = None
    if args.skill_reference is not None:
        reference_state = torch.load(args.skill_reference, map_location=device, weights_only=False)
        skill_reference = NeuralPlayer(reference_state['width'], history=reference_state.get('history', 1), shot_conditioned=reference_state.get('shot_conditioned', False)).to(device).eval()
        if skill_reference.obs_dim != net.obs_dim:
            raise ValueError("skill reference must use the same observation layout")
        skill_reference.load_weights(reference_state['model'])
        skill_reference.requires_grad_(False)
        skill_reference=CoordinateReference(skill_reference,checkpoint_environment(args.skill_reference,reference_state),coordinate_options)
    skill_replay = None
    if args.skill_replay is not None:
        with np.load(args.skill_replay) as replay:
            rx, ry = replay['observation'], replay['action']
            replay_options=json.loads(str(replay['environment_options'])) if 'environment_options' in replay else dict(accel=60,workspace_bounds_mm=None)
            rx=observation_coordinates(rx,replay_options,coordinate_options,args.history)
            ry=action_coordinates(ry,replay_options,coordinate_options)
            if rx.ndim != 2 or rx.shape[1] != net.obs_dim or ry.shape != (len(rx), 6) or not len(rx) or not np.isfinite(rx).all() or not np.isfinite(ry).all():
                raise ValueError("invalid neural skill replay")
            skill_replay = (torch.as_tensor(rx, dtype=torch.float32, device=device), torch.as_tensor(ry, dtype=torch.float32, device=device))
    with torch.no_grad():
        net.log_std.add_(np.log(args.std_scale))
    actor_prefix = ActorPrefixFreeze(net, args.freeze_base_width) if args.freeze_base_width else None
    opponent = copy.deepcopy(net).eval()
    opponents = deque([opponent], maxlen=args.opponent_pool_size)
    if args.resume and state.get("opponent_pool"):
        opponents.clear()
        for weights in state["opponent_pool"][-args.opponent_pool_size :]:
            saved_rival = NeuralPlayer(args.width, history=args.history, shot_conditioned=args.shot_conditioned).to(device).eval()
            saved_rival.load_weights(weights)
            opponents.append(saved_rival)
        opponent = opponents[-1]
    reference_paths = (
        [str(args.opponent_checkpoint)] if args.opponent_checkpoint else []
    ) + args.opponent_reference
    references = []
    for path in reference_paths:
        reference_state = torch.load(path, map_location=device, weights_only=False)
        rival = NeuralPlayer(reference_state["width"], history=reference_state.get("history", 1), shot_conditioned=reference_state.get("shot_conditioned", False)).to(device).eval()
        rival.load_weights(reference_state["model"])
        rival=CoordinateReference(rival,checkpoint_environment(path,reference_state),coordinate_options)
        references.append(rival)
    reference = references[0] if references else None
    options = dict(
        accel=args.accel,workspace_bounds_mm=run_bounds,thermal_path=args.thermal_model,
        edge_dwell_weight=args.edge_dwell_weight,edge_dwell_band=args.edge_dwell_band,
        project_rail_contacts=args.project_rail_contacts,
        edge_clearance_weight=args.edge_clearance_weight,
        goals_only=args.goals_only,
        stage=args.stage,
        seed=args.seed,
        realistic=not args.ideal_sensing,
        report_sensing=args.report_sensing,
        load_weight=args.load_weight,
        load_energy_weight=args.load_energy_weight,
        load_holding_weight=args.load_holding_weight,
        load_holding_horizon=args.load_holding_horizon,
        capture_weight=args.capture_weight,
        conversion_weight=args.conversion_weight,
        capture_first=args.capture_first,
        receive_drill=args.receive_drill,
        productive_receive_drill=args.productive_receive_drill,
        shot_power_weight=args.shot_power_weight,
        shot_power_exponent=args.shot_power_exponent,
        off_target_penalty=args.off_target_penalty,
        conversion_speed_weight=args.conversion_speed_weight,
        discount=args.gamma,
        warm_start_max=args.warm_start_max,
        warm_game_start=args.warm_game_start,
        terminate_overload=args.terminate_overload,
        fixed_practice_roles=args.fixed_practice_roles,
        random_practice_opponent=args.random_practice_opponent,
        fixed_opponent_style_fraction=args.fixed_opponent_style_fraction,
        shutdown_penalty=args.shutdown_penalty,
        game_goal_weight=args.game_goal_weight,
        skill_goal_weight=args.skill_goal_weight,
        skill_concede_weight=args.skill_concede_weight,
        turnover_weight=args.turnover_weight,
        practice_selfplay_fraction=args.practice_selfplay_fraction,
        reward_scale=args.reward_scale,
        shutdown_level=args.shutdown_level,
        practice_defense_fraction=args.practice_defense_fraction,
        defense_min_speed=args.defense_min_speed,
        defense_max_speed=args.defense_max_speed,
        wide_defense=args.wide_defense,
        random_defense_start_fraction=args.random_defense_start_fraction,
        defense_clear_reward=args.defense_clear_reward,
        defense_windup_fraction=args.defense_windup_fraction,
        shot_speed_scale=args.shot_speed_scale,
        conversion_speed_scale=args.conversion_speed_scale,
        receiving_min_speed=args.receiving_min_speed,
        receiving_max_speed=args.receiving_max_speed,
        rushed_shot_penalty=args.rushed_shot_penalty,
        thermal_gain=args.thermal_gain,
        shot_conditioned=args.shot_conditioned or any(r.shot_conditioned for r in references),
        shot_request=args.shot_request,
        shot_request_weight=args.shot_request_weight,
        wrong_shot_penalty=args.wrong_shot_penalty,
        fallback_shot_scale=args.fallback_shot_scale,
        unproductive_return_penalty=args.unproductive_return_penalty,
        setup_weight=args.setup_weight,
        setup_avoid_puck=args.setup_avoid_puck,
        control_potential_weight=args.control_potential_weight,
        shot_setup_fraction=args.shot_setup_fraction,
        continuous_rallies=args.continuous_rallies,
        possession_followthrough=args.possession_followthrough,
        recovery_fraction=args.recovery_fraction,
        recovery_min_speed=args.recovery_min_speed,
        recovery_max_speed=args.recovery_max_speed,
        recovery_easy_fraction=args.recovery_easy_fraction,
        recovery_easy_arc=args.recovery_easy_arc,
        recovery_capture_bonus=args.recovery_capture_bonus,
        recovery_fallback_scale=args.recovery_fallback_scale,
        recovery_cushion_bonus=args.recovery_cushion_bonus,
        recovery_cushion_signed=args.recovery_cushion_signed,
        recovery_get_ahead_bonus=args.recovery_get_ahead_bonus,
        recovery_approach_weight=args.recovery_approach_weight,
        readiness_weight=args.readiness_weight,
        readiness_cost_weight=args.readiness_cost_weight,
        readiness_lateral_uncertainty=args.readiness_lateral_uncertainty,
        defense_windup_lateral_speed=args.defense_windup_lateral_speed,
        slow_exit_penalty=args.slow_exit_penalty,
        possession_delay_weight=args.possession_delay_weight,
        game_episode_seconds=args.game_episode_seconds,
        stationary_failure_penalty=args.stationary_failure_penalty,
        stationary_rest_fraction=args.stationary_rest_fraction,
        edge_drill_fraction=args.edge_drill_fraction,
        corner_drill_fraction=args.corner_drill_fraction,
        edge_recovery_weight=args.edge_recovery_weight,
        edge_approach_weight=args.edge_approach_weight,
        stationary_replay=args.stationary_replay,
        stationary_replay_fraction=args.stationary_replay_fraction,
        correlated_warm_fraction=args.correlated_warm_fraction,
        cold_practice_fraction=args.cold_practice_fraction,
        cold_game_start_fraction=args.cold_game_start_fraction,
        game_fraction=args.game_fraction,
    )
    if args.workers > 1:
        from airhockey.neural_vector import ParallelNeuralEnv

        env = ParallelNeuralEnv(args.n_envs, args.workers, **options)
    else:
        env = NeuralTrainingEnv(args.n_envs, **options)
    ACTIVE_ENV = env
    own_history = PhysicalHistory(args.history)
    opponent_history = PhysicalHistory(max([args.history, *[r.history for r in references]]))
    own_history.reset(env.reset(seed=args.seed))
    obs = own_history.for_policy(net)
    opponent_history.reset(env.opponent_obs())
    meta = dict(
        algorithm="neural_ppo_v1",
        action_mode="arrival",
        obs_dim=net.obs_dim,
        physical_history_frames=args.history,
        shot_conditioned=args.shot_conditioned,
        device=str(device),
        critic_context_dim=net.critic_context_dim,
        critic_inputs="physical observations plus training-only exercise, time and possession flags",
        action_dim=6,
        controller=None,
        deployment_ready=False,
        simulation_only=True,
        initialization="random" if not initialization else str(initialization),
        reward_mode="goals_only: +1 scored, -1 conceded; no other reward" if args.goals_only else "shaped",
        critic_initialization="fresh" if args.initialize_from else "checkpoint" if args.resume else "fresh",
        exploration="learned state-dependent variance with optional training-only recovery/quiet-state floors; deterministic inference",
        optimizer_transfer="fresh for new training heads"
        if upgraded
        else "restored"
        if args.resume
        else "fresh",
        physical_limits=dict(speed_m_s=12, acceleration_m_s2=args.accel),
        workspace_bounds_mm=run_bounds,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        training_opponents="pool of neural policy snapshots"
        if args.stage >= 3
        else "stationary practice paddles",
        fixed_reference_checkpoints=reference_paths,
        policy_inputs="physical state, previous commands, load estimates; no task/phase labels"
        + ("; current possession request [left bank, right bank, straight]" if args.shot_conditioned else "; no shot request"),
        source_hashes={},
        skill_replay_sha256=hashlib.sha256(args.skill_replay.read_bytes()).hexdigest() if args.skill_replay else None,
        stationary_replay_sha256=hashlib.sha256(args.stationary_replay.read_bytes()).hexdigest() if args.stationary_replay else None,
    )
    if args.evaluation_dir is not None:
        meta['evaluation_dir'] = str(args.evaluation_dir)
    meta["thermal_model"] = json.loads(args.thermal_model.read_text())
    for path in [
        Path(__file__),
        ROOT / "ai/airhockey/neural_player.py",
        ROOT / "ai/airhockey/neural_observation.py",
        ROOT / "ai/airhockey/report_sensing.py",
        ROOT / "ai/airhockey/deploy.py",
        ROOT / "ai/airhockey/heuristics.py",
        ROOT / "ai/airhockey/neural_training.py",
        ROOT / "ai/airhockey/neural_possession.py",
        ROOT / "ai/airhockey/arrival.py",
        ROOT / "ai/airhockey/arrival_env.py",
        ROOT / "ai/airhockey/motion_guard.py",
        ROOT / "ai/airhockey/batch_env.py",
        ROOT / "ai/airhockey/batch_physics.py",
        ROOT / "ai/airhockey/neural_vector.py",
        ROOT / "ai/airhockey/on_policy.py",
        ROOT / "ai/airhockey/policy_benchmark.py",
        ROOT / "ai/airhockey/shot_flight.py",
        ROOT / "ai/airhockey/rewards.py",
        ROOT / "ai/airhockey/thermal.py",
        ROOT / "ai/airhockey/dynamics.py",
        ROOT / "ai/airhockey/motion.py",
        ROOT / "ai/airhockey/physics.py",
        ROOT / "ai/airhockey/perception.py",
        args.thermal_model,
        ROOT / 'ai/airhockey/neural_setup.py',
        ROOT / 'ai/airhockey/neural_coordinates.py',
        ROOT / "shared/cdpr_geometry.py",
        ROOT / "fw/include/motion_profile.h",
        ROOT / "fw/host/motion_batch.cpp",
        ROOT / "fw/host/motion_limits.cpp",
        ROOT / "fw/host/Makefile",
    ]:
        dst = run / "source" / path.name
        dst.parent.mkdir(exist_ok=True)
        dst.write_bytes(path.read_bytes())
        meta["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    write_json(run / "run.json", meta)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    history = deque(maxlen=2000)
    kind_history = [deque(maxlen=500) for _ in range(4)]
    ep_reward = np.zeros(args.n_envs)
    ep_goals = np.zeros(args.n_envs)
    ep_conceded = np.zeros(args.n_envs)
    ep_seconds = np.zeros(args.n_envs)
    started = time.monotonic()
    initial = total
    next_save = total + args.save_every
    update = 0
    shape = (args.rollout, args.n_envs)
    observations = torch.empty((*shape, net.obs_dim), device=device)
    critic_contexts = torch.empty((*shape, net.critic_context_dim), device=device)
    raw_actions = torch.empty((*shape, 6), device=device)
    persistent_bias = (RecoveryExplorationBias(
        args.n_envs, args.edge_persistent_bias or args.recovery_persistent_bias, args.recovery_bias_block_steps,
        device=device, load_aware=args.recovery_exploration_load_aware,
        mode="edge" if args.edge_persistent_bias else "recovery",
        workspace=workspace_in_sim(bounds_mm=run_bounds))
        if (args.recovery_persistent_bias or args.edge_persistent_bias) else None)
    exploration_biases = torch.empty_like(raw_actions) if persistent_bias else None
    contacted = np.zeros(args.n_envs, dtype=bool)
    old_logp, values, rewards, next_values = [
        torch.empty(shape, device=device) for _ in range(4)
    ]
    terminals, dones = [
        torch.empty(shape, device=device, dtype=torch.bool) for _ in range(2)
    ]

    def save(name):
        state = dict(
            model=net.state_dict(),
            optimizer=optimizer.state_dict(),
            opponent=opponent.state_dict(),
            opponent_pool=[r.state_dict() for r in opponents],
            reference=reference.state_dict() if reference is not None else None,
            references=[r.state_dict() for r in references],
            step=total,
            width=args.width,
            history=args.history,
            shot_conditioned=args.shot_conditioned,
            stage=args.stage,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None,
            args=meta["args"],
        )
        temp = run / (name + ".tmp")
        torch.save(state, temp)
        temp.replace(run / name)

    ACTIVE_SAVE = save
    save("agent_initial.pt")
    while (
        total - initial < args.steps
        and not stopped
        and not (run / "STOP").exists()
        and time.monotonic() - started < args.minutes * 60
    ):
        rollout_start = time.monotonic()
        for t in range(args.rollout):
            if not np.isfinite(obs).all():
                np.savez_compressed(run / "nonfinite_observation.npz", observation=obs)
                raise FloatingPointError(
                    "Non-finite simulator observation; diagnostic saved"
                )
            x = torch.as_tensor(obs, device=device)
            context = torch.as_tensor(env.critic_context(), device=device)
            with torch.no_grad():
                mu, val, scale = net.distribution(x, context)
                practice = context[:, 3] < .5 if args.practice_only_exploration else None
                recovery_mask = recovery_exploration_window(context, args.recovery_exploration_seconds, practice)
                scale = recovery_exploration_scale(x, scale, args.recovery_exploration, args.recovery_exploration_load_aware, args.quiet_exploration, mean=mu, timing=args.recovery_explore_timing, practice_mask=recovery_mask)
                scale = thermal_effort_exploration_scale(x, scale, args.hot_effort_exploration, practice_mask=practice)
                sampling_mean = mu
                if persistent_bias is not None:
                    bias = persistent_bias.sample(x, context, torch.as_tensor(contacted, device=device))
                    exploration_biases[t] = bias
                    sampling_mean = mu + bias
                raw = sampling_mean + scale.exp() * torch.randn_like(mu)
                lp = log_probability(raw, sampling_mean, scale)
            if (
                not torch.isfinite(raw).all()
                or not torch.isfinite(val).all()
                or not torch.isfinite(lp).all()
            ):
                np.savez_compressed(
                    run / "nonfinite_policy.npz",
                    observation=obs,
                    mean=mu.cpu().numpy(),
                    log_std=scale.cpu().numpy(),
                )
                raise FloatingPointError(
                    "Non-finite policy/value output; diagnostic saved"
                )
            observations[t], raw_actions[t], values[t], old_logp[t] = x, raw, val, lp
            critic_contexts[t] = context
            if args.stage >= 3:
                opponent_obs = opponent_history.get()
                if not np.isfinite(opponent_obs).all():
                    np.savez_compressed(
                        run / "nonfinite_opponent_observation.npz",
                        own_observation=obs,
                        opponent_observation=opponent_obs,
                    )
                    raise FloatingPointError(
                        "Non-finite opponent observation; diagnostic saved"
                    )
                opponent_action = np.empty((args.n_envs, 6), np.float32)
                groups = np.arange(args.n_envs) % len(opponents)
                for j, rival in enumerate(opponents):
                    mask = groups == j
                    opponent_action[mask] = rival.act(
                        opponent_history.for_policy(rival)[mask], stochastic=args.stochastic_opponents
                    )
                for j, fixed_rival in enumerate(references):
                    ids = np.arange(args.n_envs)
                    mask = (ids % 4 == 0) & ((ids // 4) % len(references) == j)
                    if mask.any():
                        opponent_action[mask] = fixed_rival.act(opponent_history.for_policy(fixed_rival)[mask])
                if not np.isfinite(opponent_action).all():
                    np.savez_compressed(
                        run / "nonfinite_opponent_action.npz",
                        opponent_observation=opponent_obs,
                        opponent_action=opponent_action,
                    )
                    raise FloatingPointError(
                        "Non-finite opponent action; diagnostic saved"
                    )
                env.set_opponent_action(opponent_action)
            obs, rew, term, trunc, info = env.step(raw.tanh().cpu().numpy())
            if persistent_bias is not None:
                contacted = np.asarray(info["contacts"]) > 0
            own_history.append(obs)
            obs = own_history.for_policy(net)
            opponent_history.append(env.opponent_obs())
            done = term | trunc
            with torch.no_grad():
                nv = net.value(
                    torch.as_tensor(obs, device=device),
                    torch.as_tensor(info["critic_context"], device=device),
                )
            rewards[t] = torch.as_tensor(rew, device=device)
            next_values[t] = nv
            terminals[t] = torch.as_tensor(term, device=device)
            dones[t] = torch.as_tensor(done, device=device)
            ep_reward += rew
            ep_goals += info["goals"]
            ep_conceded += info["conceded"]
            ep_seconds += 0.02
            for i in np.flatnonzero(done):
                row = {
                    key: float(info[key][i])
                    for key in [
                        "kind",
                        "contacts",
                        "captures",
                        "edge_recoveries",
                        "conversions",
                        "shots",
                        "aimed",
                        "shot_speed_sum",
                        "passive_returns",
                        "unproductive_returns",
                        "slow_possession_losses",
                        "request_matches",
                        "load_peak",
                        "overload_seconds",
                        "shutdown",
                    ]
                }
                row.update(
                    reward=float(ep_reward[i]),
                    goals=float(ep_goals[i]),
                    conceded=float(ep_conceded[i]),
                    episode_seconds=float(ep_seconds[i]),
                )
                history.append(row)
                kind_history[int(row["kind"])].append(row)
            if done.any():
                if persistent_bias is not None:
                    persistent_bias.reset(done)
                    contacted[done] = False
                own_history.reset(env.reset(mask=done), mask=done)
                obs = own_history.for_policy(net)
                opponent_history.reset(env.opponent_obs(), mask=done)
                ep_reward[done] = ep_goals[done] = ep_conceded[done] = 0
                ep_seconds[done] = 0
            total += args.n_envs
        rollout_seconds = time.monotonic() - rollout_start
        adv, returns = advantages(
            rewards,
            values,
            next_values,
            terminals,
            dones,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )
        x = observations.flatten(0, 1)
        context = critic_contexts.flatten(0, 1)
        ra = raw_actions.flatten(0, 1)
        offsets = exploration_biases.flatten(0, 1) if exploration_biases is not None else None
        lp = old_logp.flatten()
        advantage = adv.flatten()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        target = returns.flatten()
        explained_variance = 1 - (
            target - values.flatten()
        ).var() / target.var().clamp_min(1e-8)
        kl = 0
        rollout_initial_kl = None
        actor_stopped = False
        actor_updates = 0
        actor_step_scales = []
        consistency_losses = []
        preservation_losses = []
        for epoch in range(args.epochs):
            for ids in torch.randperm(len(x), device=device).split(args.minibatch):
                mu, value, scale = net.distribution(x[ids], context[ids])
                practice = context[ids, 3] < .5 if args.practice_only_exploration else None
                recovery_mask = recovery_exploration_window(context[ids], args.recovery_exploration_seconds, practice)
                scale = recovery_exploration_scale(x[ids], scale, args.recovery_exploration, args.recovery_exploration_load_aware, args.quiet_exploration, mean=mu, timing=args.recovery_explore_timing, practice_mask=recovery_mask)
                scale = thermal_effort_exploration_scale(x[ids], scale, args.hot_effort_exploration, practice_mask=practice)
                # Condition both PPO likelihoods on the same realized,
                # parameter-independent exploration state. Do not resample it
                # here or incorrectly score biased actions under an unbiased mean.
                sampling_mean = mu if offsets is None else mu + offsets[ids]
                logp = log_probability(ra[ids], sampling_mean, scale)
                log_ratio = logp - lp[ids]
                ratio = log_ratio.clamp(-20, 20).exp()
                # Check each minibatch, before taking another actor step. With
                # large rollouts an epoch contains dozens of updates; waiting
                # until its end can erase a precise warm-started contact policy.
                kl = float(((ratio - 1) - log_ratio).mean().detach())
                if rollout_initial_kl is None:
                    rollout_initial_kl = kl
                actor_stopped |= kl > args.target_kl
                policy_loss = -torch.minimum(
                    ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids]
                ).mean()
                value_loss = 0.5 * (value - target[ids]).square().mean()
                entropy = scale.sum(-1).mean()
                loss = 0.5 * value_loss
                if not actor_stopped:
                    loss = loss + policy_loss - args.entropy * entropy
                    if args.defense_request_consistency:
                        consistency = defensive_request_consistency(net, x[ids], mu)
                        loss += args.defense_request_consistency * consistency
                        consistency_losses.append(float(consistency.detach()))
                    if args.skill_reference_weight:
                        from airhockey.neural_player import established_skill_preservation
                        preservation = established_skill_preservation(net, skill_reference, x[ids], mu, controlled=context[ids, 5] > .5, max_load=args.skill_reference_max_load, incoming_only=args.skill_reference_incoming_only)
                        loss += args.skill_reference_weight * preservation
                        preservation_losses.append(float(preservation.detach()))
                    if args.skill_replay_weight:
                        rx, ry = skill_replay
                        samples = torch.randint(len(rx), (512,), device=device)
                        prediction = net.actor(net.trunk(rx[samples])).tanh()
                        preservation = (prediction - ry[samples]).square().mean()
                        loss += args.skill_replay_weight * preservation
                        preservation_losses.append(float(preservation.detach()))
                    actor_updates += 1
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.actor_parameters(), 1, error_if_nonfinite=True
                )
                torch.nn.utils.clip_grad_norm_(
                    net.value_parameters(), 1, error_if_nonfinite=True
                )
                actor_parameters = net.actor_parameters()
                before = ([p.detach().clone() for p in actor_parameters]
                          if args.backtrack_kl and not actor_stopped else None)
                optimizer.step()
                if actor_prefix is not None:
                    actor_prefix.restore()
                if before is not None:
                    def measure_actor_kl():
                        features = net.trunk(x[ids])
                        new_mean = net.actor(features)
                        new_scale = (net.log_std + net.noise(features)).clamp(-4, 0.5)
                        new_scale = recovery_exploration_scale(x[ids], new_scale, args.recovery_exploration, args.recovery_exploration_load_aware, args.quiet_exploration, mean=new_mean, timing=args.recovery_explore_timing, practice_mask=recovery_mask)
                        new_scale = thermal_effort_exploration_scale(x[ids], new_scale, args.hot_effort_exploration, practice_mask=practice)
                        new_mean = new_mean if offsets is None else new_mean + offsets[ids]
                        difference = log_probability(ra[ids], new_mean, new_scale) - lp[ids]
                        return (difference.clamp(-20, 20).exp() - 1 - difference).mean()

                    step_scale, _ = backtrack_actor_step(
                        actor_parameters, before, measure_actor_kl, args.target_kl)
                    actor_step_scales.append(step_scale)
                # The value network is independent: continue fitting it after
                # actor early stopping without changing actor parameters.
        update += 1
        if update % 50 == 0:
            opponent = copy.deepcopy(net).eval()
            opponents.append(opponent)
        elapsed = time.monotonic() - started
        means = (
            {key: float(np.mean([row[key] for row in history])) for key in history[0]}
            if history
            else {}
        )
        by_kind = {}
        for kind in range(4):
            rows = kind_history[kind]
            if rows:
                by_kind[kind] = {
                    key: float(np.mean([r[key] for r in rows]))
                    for key in [
                        "goals",
                        "conceded",
                        "captures",
                        "edge_recoveries",
                        "conversions",
                        "aimed",
                        "slow_possession_losses",
                        "overload_seconds",
                        "load_peak",
                        "shutdown",
                    ]
                }
                by_kind[kind]["episodes"] = len(rows)
                minutes = sum(r["episode_seconds"] for r in rows) / 60
                by_kind[kind]["goals_per_minute"] = sum(r["goals"] for r in rows) / max(
                    minutes, 1e-9
                )
                by_kind[kind]["conceded_per_minute"] = sum(
                    r["conceded"] for r in rows
                ) / max(minutes, 1e-9)
                by_kind[kind]["episode_seconds"] = 60 * minutes / len(rows)
        status = dict(
            pid=os.getpid(),
            updated_at=time.time(),
            initial_step=initial,
            target_steps=args.steps,
            step=total,
            update=update,
            stage=args.stage,
            elapsed_s=elapsed,
            transitions_per_s=(total - initial) / elapsed,
            rollout_s=rollout_seconds,
            update_s=time.monotonic() - rollout_start - rollout_seconds,
            kl=kl,
            rollout_initial_kl=rollout_initial_kl,
            persistent_bias_fraction=(float((exploration_biases.abs().sum(-1) > 0).float().mean())
                                      if exploration_biases is not None else 0.0),
            actor_updates=actor_updates,
            actor_min_step_scale=min(actor_step_scales) if actor_step_scales else 1.0,
            actor_backtracked_steps=sum(s < 1 for s in actor_step_scales),
            defensive_request_consistency_loss=float(np.mean(consistency_losses)) if consistency_losses else 0.0,
            skill_preservation_loss=float(np.mean(preservation_losses)) if preservation_losses else 0.0,
            actor_early_stopped=actor_stopped,
            explained_variance=float(explained_variance),
            std=scale.detach().exp().mean(0).tolist(),
            means=means,
            by_kind=by_kind,
            running=True,
        )
        write_json(run / "status.json", status)
        with (run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(status) + "\n")
        if update % 5 == 0 or update == 1:
            print(json.dumps(status), flush=True)
        if total >= next_save:
            save(f"agent_step_{total:09d}.pt")
            save("agent.pt")
            next_save = total + args.save_every
    save(f"agent_step_{total:09d}.pt")
    save("agent.pt")
    status["running"] = False
    write_json(run / "status.json", status)
    if args.workers > 1:
        env.close()
    print(f"finished {run} at {total}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_RUN is not None:
            path = ACTIVE_RUN / "status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(
                running=False, failed=True, error=f"{type(error).__name__}: {error}"
            )
            write_json(path, status)
            if ACTIVE_SAVE is not None:
                ACTIVE_SAVE("agent_failure_debug.pt")
        if ACTIVE_ENV is not None and hasattr(ACTIVE_ENV, "close"):
            ACTIVE_ENV.close()
        raise
