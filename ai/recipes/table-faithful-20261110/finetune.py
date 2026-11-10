#!/usr/bin/env python3
"""Continue v4 into a table-faithful v5: PPO in the sim that did to v4 what the table did.

    PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
        --run-name _neural-table-faithful-20261110-a --minutes 120 [--print]

v4's own recipe (expert-distill-20261018/finetune.py as run for
_neural-teacher-calm-20261018-j: teacher-regularised defense with the guard's
accel slot free, possession delay 30/s, linear calm-accel cost 200) CONTINUED
with --resume, so the critic, the optimiser moments, the adapted actor rate,
the opponent pool and the teacher anneal (already at 0.2) carry over: no
critic warm-up, no teacher re-pull. On top of it, each one a flag of this
launcher (README.md says why and what the numbers mean):

  the table profile on a share of the workers, people in a share of the game
  slots and striking in a share of the defense drills, more left-bank
  requests, the three physical-motion costs, staggered game clocks.

ROUND 2 (`--arm a2`, after v5a was evaluated): continue v5a itself with the
changes its evaluation called for -- the dead-puck stall, the brake left to
the motion guard, strike effort, corner pucks from anywhere, a lift model
with more than one mechanism. Each is again a flag of this launcher whose
default is arm A; README.md "Round 2" has the reasons and the numbers.

    PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
        --arm a2 --run-name _neural-table-faithful-20261110-a2 --minutes 150

ROUND 3 (`--arm a3`): arm a2's options plus two IMITATION ANCHORS on resting
pucks, because reward and exploration did not find the strike: the physics
teacher's action where a puck has lain on the open table, v4's in corner and
edge drills and on pucks lying in the rail strip. Label sources in training
only. `--resume` chooses the starting point (default arm a2's `agent.pt`);
README.md "Round 3".

    PYTHONPATH=ai python3 ai/recipes/table-faithful-20261110/finetune.py \
        --arm a3 --run-name _neural-table-faithful-20261110-a3 --minutes 150

Simulation only; nothing here touches the table.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = "runs/rail30-100-20261001-v1/agent.pt"
V4 = "runs/teacher-calm-20261018-v4/agent.pt"
V5A = "runs/_neural-table-faithful-20261110-a/agent.pt"
A2 = "runs/_neural-table-faithful-20261110-a2/agent.pt"
# Arm a3's anchor weights (README "Round 3" has the smoke run they were read from).
TEACHER_RESTING, REFERENCE_RAIL = 1.0, 1.0
# Launcher defaults per arm; "a" is the parser's own.
ARMS = {
    "a": {},
    "a2": dict(
        resume=V5A, seed=20261112, left_bank=1.0, shot_speed_scale="7.0",
        dead_puck=40.0, dead_puck_exploration=[2.0, 2.0], calm_reachable=0.0, cross_bank_drill=0.25,
        corner_drill=".2", edge_drill=".15", corner_far_start=0.5,
        guard=10.0, guard_unresolved=50.0, far_cap_max_speed=1.0, accel_effort=100.0, idle_speed=20.0,
        sim_option=['mallet_lift={"b0": [0.0, 0.30], "a0": [10, 90], "flat_probability": 0.25, '
                    '"closing_probability": 0.25, "far_body": false}']),
}
# Round 3: a2's options and the two anchors; --resume picks a, a2 or v4.
ARMS["a3"] = dict(ARMS["a2"], resume=A2, seed=20261113, teacher_resting=TEACHER_RESTING, reference_rail=REFERENCE_RAIL,
                  anchor_space="pre-squash")


def argv(a):
    cmd = [sys.executable, "ai/bin/train_neural_player.py",
           "--run-name", a.run_name, "--evaluation-dir", f"logs/neural-player/table-faithful-20261110/{a.run_name}",
           "--resume", a.resume, "--opponent-checkpoint", PRODUCTION,
           "--opponent-reference", PRODUCTION, "--opponent-reference", V4,
           "--n-envs", str(a.n_envs), "--workers", str(a.workers), "--rollout", "128", "--epochs", "2", "--minibatch", "2048",
           "--steps", str(a.steps), "--minutes", str(a.minutes), "--save-every", "40000000", "--seed", str(a.seed),
           "--value-lr", "0.0005", "--std-scale", "1.0", "--entropy", "0.0002", "--target-kl", "0.02", "--backtrack-kl",
           "--stage", "5", "--accel", "100.0", "--workspace", "rail30",
           "--thermal-model", str(ROOT / "ai/recipes/motor-load-20261001.json"),
           "--edge-dwell-weight", "20", "--edge-dwell-band", ".12", "--history", "16", "--history-deltas",
           "--freeze-base-width", "0", "--shot-conditioned", "--shot-request", "random", "--shot-request-weight", "100.0",
           "--wrong-shot-penalty", "20.0", "--fallback-shot-scale", "1.0", "--unproductive-return-penalty", "12.0",
           "--setup-weight", "12.0", "--setup-avoid-puck", "--control-potential-weight", "30.0", "--shot-setup-fraction", "0.0",
           "--device", a.device, "--torch-threads", "2", "--capture-weight", "30.0", "--conversion-weight", "25.0",
           "--capture-first", "--productive-receive-drill", "--shot-power-weight", "60", "--shot-power-exponent", "2.0",
           "--off-target-penalty", "40", "--conversion-speed-weight", "0.75", "--opponent-pool-size", "6",
           "--gamma", "0.999", "--gae-lambda", "0.98", "--warm-start-max", "0.94", "--terminate-overload",
           "--fixed-practice-roles", "--random-practice-opponent", "--shutdown-penalty", "12000", "--shutdown-level", "0.95",
           "--game-goal-weight", "600.0", "--skill-goal-weight", "60.0", "--skill-concede-weight", "1000",
           "--turnover-weight", "100.0", "--reward-scale", "1.0", "--wide-defense", "--random-defense-start-fraction", "0.9",
           "--shot-speed-scale", a.shot_speed_scale, "--conversion-speed-scale", "9.0", "--stochastic-opponents",
           "--receiving-min-speed", "3.0", "--receiving-max-speed", "8.0", "--rushed-shot-penalty", "0.0",
           "--thermal-gain", "1.3", "--report-sensing", "--continuous-rallies", "--possession-followthrough",
           "--recovery-fraction", "0.35", "--recovery-min-speed", "0.15", "--recovery-max-speed", "1.2",
           "--recovery-easy-fraction", "0.5", "--recovery-easy-arc", "1.5707963267948966", "--recovery-capture-bonus", "0.0",
           "--recovery-fallback-scale", "1.0", "--recovery-cushion-bonus", "0.0", "--recovery-cushion-signed",
           "--recovery-get-ahead-bonus", "0.0", "--fixed-opponent-style-fraction", "0.5", "--recovery-approach-weight", "60.0",
           "--readiness-weight", "0", "--readiness-cost-weight", "0", "--readiness-lateral-uncertainty", "0.12",
           "--defense-windup-fraction", "0.85", "--defense-windup-bank-fraction", "0.5", "--defense-windup-lateral-speed", "0.5",
           "--defense-min-speed", "5", "--defense-max-speed", "16", "--defense-clear-reward", "300", "--defense-block-only",
           "--defensive-depth-weight", "0", "--defensive-depth-target", "0.3", "--slow-exit-penalty", "0.0",
           "--possession-delay-weight", str(a.possession_delay), "--game-episode-seconds", "900.0",
           "--stationary-failure-penalty", "600.0", "--stationary-rest-fraction", "0.5",
           "--stationary-replay", "logs/neural-player/expanded-100-consolidation-20261001/expanded-failures.npz",
           "--stationary-replay-fraction", "0.5", "--edge-drill-fraction", a.edge_drill, "--corner-drill-fraction", a.corner_drill,
           "--edge-recovery-weight", "30", "--edge-approach-weight", "0", "--edge-persistent-bias", "0.0",
           "--edge-clearance-weight", "20", "--project-rail-contacts", "--correlated-warm-fraction", ".5",
           "--cold-practice-fraction", ".35", "--cold-game-start-fraction", "0.5", "--warm-game-start",
           "--recovery-exploration", "0.0", "--recovery-exploration-seconds", "0.16", "--recovery-exploration-load-aware",
           "--recovery-persistent-bias", "0.0", "--recovery-bias-block-steps", "8", "--recovery-explore-timing",
           "--quiet-exploration", "--hot-effort-exploration", "3", "--defense-request-consistency", "1.0",
           "--load-weight", "32", "--load-energy-weight", "1", "--load-holding-weight", "3", "--load-holding-horizon", "15",
           "--skill-replay-weight", "0", "--skill-reference-weight", "0", "--practice-defense-fraction", "0.45",
           "--game-fraction", "0.3", "--slow-receiving-fraction", "0.65", "--practice-selfplay-fraction", "0",
           # The teacher as v4 had it. --resume keeps the anneal's origin, so
           # these give the final weight 0.2 from the first update.
           "--teacher", a.teacher, "--teacher-weight", "1", "--teacher-weight-final", "0.2",
           "--teacher-anneal-steps", "60000000", "--teacher-mask", "defense", "--teacher-free-guard-accel",
           "--teacher-opponent-fraction", str(a.teacher_opponents),
           "--calm-accel-weight", str(a.calm_accel), "--calm-accel-power", "1.0",
           # No --value-warmup-updates: the critic is the checkpoint's.
           "--actor-steps-min", "16", "--lr", "1e-6", "--lr-min", "2e-7", "--lr-max", "5e-5", "--std-floor", "0.1",
           "--adaptive-lr"]
    if a.sim_profile != "none":
        cmd += ["--sim-profile", a.sim_profile, "--sim-profile-share", str(a.sim_profile_share)]
        for option in a.sim_option:
            cmd += ["--sim-option", option]
    if a.human_proxy_share or a.defense_strike:
        cmd += ["--human-proxy-tiers", a.human_proxy_tiers]
    if a.human_proxy_share:
        cmd += ["--human-proxy-share", str(a.human_proxy_share)]
    if a.defense_strike:
        cmd += ["--defense-strike-fraction", str(a.defense_strike), "--defense-strike-speed", "4", "9",
                "--defense-strike-moving-fraction", str(a.defense_strike_moving)]
    if a.left_bank != 1:
        cmd += ["--left-bank-request-weight", str(a.left_bank)]
    for flag, weight in (("--accel-duty-weight", a.accel_duty), ("--reversal-weight", a.reversal),
                         ("--far-cap-weight", a.far_cap)):
        if weight:
            cmd += [flag, str(weight)]
    if not a.shared_clock:
        cmd += ["--stagger-games"]
    # Round 2: each absent at its arm-A default.
    if a.dead_puck:
        cmd += ["--dead-puck-weight", str(a.dead_puck), "--dead-puck-grace", str(a.dead_puck_grace)]
    if any(a.dead_puck_exploration):
        cmd += ["--dead-puck-exploration", *map(str, a.dead_puck_exploration),
                "--dead-puck-exploration-start", str(a.dead_puck_grace), "--dead-puck-exploration-hold", str(a.dead_puck_hold)]
    if a.calm_reachable is not None:
        cmd += ["--calm-reachable-scale", str(a.calm_reachable)]
    if a.far_cap_max_speed is not None and a.far_cap:
        cmd += ["--far-cap-max-speed", str(a.far_cap_max_speed)]
    for flag, value in (("--cross-bank-drill-fraction", a.cross_bank_drill),
                        ("--corner-far-start-fraction", a.corner_far_start),
                        ("--guard-weight", a.guard), ("--guard-unresolved-weight", a.guard_unresolved),
                        ("--accel-effort-weight", a.accel_effort), ("--idle-speed-weight", a.idle_speed)):
        if value:
            cmd += [flag, str(value)]
    # Round 3: each absent at its arm-A / a2 default.
    if a.teacher_resting:
        cmd += ["--teacher-resting-weight", str(a.teacher_resting), "--teacher-resting-start", str(a.teacher_resting_start)]
    if a.reference_rail:
        cmd += ["--skill-reference", a.skill_reference, "--skill-reference-rail-weight", str(a.reference_rail)]
    if a.teacher_resting or a.reference_rail:
        cmd += ["--anchor-space", a.anchor_space] + (["--anchor-kl-exempt"] if a.anchor_kl_exempt else [])
    return cmd


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", default=V4, help="Checkpoint to continue (critic, optimiser, pool and step count kept)")
    p.add_argument("--minutes", type=float, default=120)
    p.add_argument("--steps", type=int, default=400_000_000)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--teacher", default="expert")
    p.add_argument("--teacher-opponents", type=float, default=0.3, help="Share of game slots against the teacher")
    p.add_argument("--possession-delay", type=float, default=30.0)
    p.add_argument("--calm-accel", type=float, default=200.0)
    p.add_argument("--sim-profile", default="table-20261029", help="'none' for the historical sim")
    p.add_argument("--sim-profile-share", type=float, default=0.75, help="Share of the 16 workers in the profile")
    p.add_argument("--sim-option", action="append", default=[], metavar="KEY=JSON")
    p.add_argument("--human-proxy-share", type=float, default=0.3, help="Share of game slots against a person")
    p.add_argument("--human-proxy-tiers", default="table,table_plus,club,elite")
    p.add_argument("--defense-strike", type=float, default=0.4, help="Share of defense drills struck by a person at 4-9 m/s")
    p.add_argument("--defense-strike-moving", type=float, default=0.3)
    p.add_argument("--left-bank", type=float, default=2.0, help="Weight of left-bank requests (right and straight are 1)")
    p.add_argument("--accel-duty", type=float, default=20.0, help="Cost per second of realised accel above 60 m/s^2")
    p.add_argument("--reversal", type=float, default=5.0, help="Cost per velocity reversal within 100 ms at 1.5 m/s")
    p.add_argument("--far-cap", type=float, default=40.0, help="Cost per second of cap above 40 m/s^2, puck on the far half")
    p.add_argument("--shared-clock", action="store_true", help="Do not stagger the 900 s game clocks")
    # Round 2. Every default here is arm A; --arm a2 sets them (ARMS).
    p.add_argument("--arm", choices=sorted(ARMS), default="a", help="Preset of the defaults above and below")
    p.add_argument("--dead-puck", type=float, default=0.0,
                   help="Cost per second a slow puck has lain within reach on the own half past the grace")
    p.add_argument("--dead-puck-grace", type=float, default=1.0)
    p.add_argument("--dead-puck-exploration", type=float, nargs=2, default=[0.0, 0.0], metavar=("TIME", "EFFORT"),
                   help="Pre-tanh std floors of arrival time and accel cap where such a puck has lain past the grace")
    p.add_argument("--dead-puck-hold", type=float, default=0.9, help="Tick-to-tick correlation of that noise")
    p.add_argument("--calm-reachable", type=float, help="Calm-accel scale while a slow puck is within reach")
    p.add_argument("--cross-bank-drill", type=float, default=0.0, help="Share of stationary drills in the stall's own state")
    p.add_argument("--corner-drill", default=".1", help="Share of stationary drills in a back corner")
    p.add_argument("--edge-drill", default=".1", help="Share of stationary drills on a side fringe")
    p.add_argument("--corner-far-start", type=float, default=0.0, help="Share of corner drills started from anywhere")
    p.add_argument("--guard", type=float, default=0.0, help="Cost per tick the motion guard replaced the command")
    p.add_argument("--guard-unresolved", type=float, default=0.0, help="Cost per tick it could not resolve it")
    p.add_argument("--far-cap-max-speed", type=float, help="Charge the far cap only below this mallet speed, m/s")
    p.add_argument("--accel-effort", type=float, default=0.0, help="Weight of the quadratic accel cost above 40 m/s^2")
    p.add_argument("--idle-speed", type=float, default=0.0, help="Cost per metre moved while the table is calm")
    p.add_argument("--shot-speed-scale", default="8.0", help="Shot speed at which the shot shaping saturates, m/s")
    # Round 3. Defaults are arms A and a2; --arm a3 sets them.
    p.add_argument("--teacher-resting", type=float, default=0.0,
                   help="Weight of the teacher's action where a puck has lain on the open table")
    p.add_argument("--teacher-resting-start", type=float, default=1.0, help="s the puck has lain before it applies")
    p.add_argument("--reference-rail", type=float, default=0.0,
                   help="Weight of --skill-reference's action in corner / edge drills and on rail-strip pucks")
    p.add_argument("--skill-reference", default=V4, help="Checkpoint the rail anchor copies (training only)")
    p.add_argument("--anchor-space", choices=["squashed", "pre-squash"], default="squashed")
    p.add_argument("--anchor-kl-exempt", action=argparse.BooleanOptionalAction, default=False,
                   help="Keep anchored samples out of the KL that limits the actor step")
    p.add_argument("--n-envs", type=int, default=2048, help="128 per worker keeps the slot layout")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--print", action="store_true", help="print the command instead of running it")
    return p


def parse(argv=None):
    """Arguments with the chosen arm's defaults under whatever was typed."""
    p = build_parser()
    arm = p.parse_known_args(argv)[0].arm
    p.set_defaults(**ARMS[arm])
    return p.parse_args(argv)


def main():
    a = parse()
    cmd = argv(a)
    if a.print:
        print(shlex.join(cmd))
        return
    log = ROOT / "logs" / f"{a.run_name}.log"
    with open(log, "w") as fh:
        sys.exit(subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                 env={**__import__("os").environ, "PYTHONPATH": "ai:ai/bin", "OMP_NUM_THREADS": "2"}))


if __name__ == "__main__":
    main()
