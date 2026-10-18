#!/usr/bin/env python3
"""PPO fine-tune of a distilled student with a decaying teacher regulariser.

    PYTHONPATH=ai python3 ai/recipes/expert-distill-20261018/finetune.py \
        --init runs/_neural-teacher-dagger-20261018-a/agent.pt --run-name _neural-teacher-ppo-20261018-a \
        --minutes 75 [--measured-contact]

The control-defense no-imitation recipe (production-equivalent rewards and
limits) with: the teacher as regression target (--teacher-mask, weight 1 -> 0.2)
and as 30% of game opponents, production and the student itself as rivals,
human-speed-inclusive defense drills (5-16 m/s, half banked), the trainer
agent's optimiser regime (adaptive lr from 1e-6, std floor 0.1, 16 forced
actor steps, critic warm-up), 2048 envs on 16 workers. Simulation only.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = "runs/rail30-100-20261001-v1/agent.pt"


def argv(a):
    teacher = "expert:paddle_restitution=0.52" if a.measured_contact else "expert"
    cmd = [sys.executable, "ai/bin/train_neural_player.py",
           "--run-name", a.run_name, "--evaluation-dir", f"logs/neural-player/expert-20261018/{a.run_name}",
           "--initialize-from", a.init, "--opponent-checkpoint", PRODUCTION,
           "--opponent-reference", PRODUCTION, "--opponent-reference", a.init,
           "--n-envs", "2048", "--workers", "16", "--rollout", "128", "--epochs", "2", "--minibatch", "2048",
           "--steps", str(a.steps), "--minutes", str(a.minutes), "--save-every", "40000000", "--seed", str(a.seed),
           "--value-lr", "0.0005", "--std-scale", "1.0", "--entropy", "0.0002", "--target-kl", "0.02", "--backtrack-kl",
           "--stage", "5", "--accel", "100.0", "--workspace", "rail30",
           "--thermal-model", str(ROOT / "ai/recipes/motor-load-20261001.json"),
           "--edge-dwell-weight", "20", "--edge-dwell-band", ".12", "--history", "16", "--history-deltas",
           "--freeze-base-width", "0", "--shot-conditioned", "--shot-request", "random", "--shot-request-weight", "100.0",
           "--wrong-shot-penalty", "20.0", "--fallback-shot-scale", "1.0", "--unproductive-return-penalty", "12.0",
           "--setup-weight", "12.0", "--setup-avoid-puck", "--control-potential-weight", "30.0", "--shot-setup-fraction", "0.0",
           "--device", "cuda", "--torch-threads", "2", "--capture-weight", "30.0", "--conversion-weight", "25.0",
           "--capture-first", "--productive-receive-drill", "--shot-power-weight", "60", "--shot-power-exponent", "2.0",
           "--off-target-penalty", "40", "--conversion-speed-weight", "0.75", "--opponent-pool-size", "6",
           "--gamma", "0.999", "--gae-lambda", "0.98", "--warm-start-max", "0.94", "--terminate-overload",
           "--fixed-practice-roles", "--random-practice-opponent", "--shutdown-penalty", "12000", "--shutdown-level", "0.95",
           "--game-goal-weight", "600.0", "--skill-goal-weight", "60.0", "--skill-concede-weight", "1000",
           "--turnover-weight", "100.0", "--reward-scale", "1.0", "--wide-defense", "--random-defense-start-fraction", "0.9",
           "--shot-speed-scale", "8.0", "--conversion-speed-scale", "9.0", "--stochastic-opponents",
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
           "--possession-delay-weight", str(a.possession_delay), "--game-episode-seconds", "900.0", "--stationary-failure-penalty", "600.0",
           "--stationary-rest-fraction", "0.5",
           "--stationary-replay", "logs/neural-player/expanded-100-consolidation-20261001/expanded-failures.npz",
           "--stationary-replay-fraction", "0.5", "--edge-drill-fraction", ".1", "--corner-drill-fraction", ".1",
           "--edge-recovery-weight", "30", "--edge-approach-weight", "0", "--edge-persistent-bias", "0.0",
           "--edge-clearance-weight", "20", "--project-rail-contacts", "--correlated-warm-fraction", ".5",
           "--cold-practice-fraction", ".35", "--cold-game-start-fraction", "0.5", "--warm-game-start",
           "--recovery-exploration", "0.0", "--recovery-exploration-seconds", "0.16", "--recovery-exploration-load-aware",
           "--recovery-persistent-bias", "0.0", "--recovery-bias-block-steps", "8", "--recovery-explore-timing",
           "--quiet-exploration", "--hot-effort-exploration", "3", "--defense-request-consistency", "1.0",
           "--load-weight", "32", "--load-energy-weight", "1", "--load-holding-weight", "3", "--load-holding-horizon", "15",
           "--skill-replay-weight", "0", "--skill-reference-weight", "0", "--practice-defense-fraction", "0.45",
           "--game-fraction", "0.3", "--slow-receiving-fraction", "0.65", "--practice-selfplay-fraction", "0",
           "--teacher", teacher, "--teacher-weight", "1", "--teacher-weight-final", "0.2",
           "--teacher-anneal-steps", str(a.anneal), "--teacher-mask", a.teacher_mask, "--teacher-opponent-fraction", "0.3",
           "--value-warmup-updates", "10", "--actor-steps-min", "16", "--lr", "1e-6", "--lr-min", "2e-7",
           "--lr-max", "5e-5", "--std-floor", "0.1", "--adaptive-lr"]
    if a.calm_accel:
        cmd += ["--calm-accel-weight", str(a.calm_accel), "--calm-accel-power", str(a.calm_power)]
    if a.measured_contact:
        cmd += ["--paddle-restitution-range", "0.40", "0.65", "--paddle-tangential-range", "0.76", "0.84"]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--minutes", type=float, default=75)
    p.add_argument("--steps", type=int, default=400_000_000)
    p.add_argument("--anneal", type=int, default=200_000_000)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--measured-contact", action="store_true")
    p.add_argument("--possession-delay", type=float, default=6.0,
                   help="Cost per second (after 4 s, capped at 14 s) of keeping the puck on our half in games")
    p.add_argument("--calm-accel", type=float, default=0.0,
                   help="Per-second cost of the squared accel-cap fraction while the puck and the opponent are slow")
    p.add_argument("--calm-power", type=float, default=2.0, choices=[1.0, 2.0])
    p.add_argument("--teacher-mask", default="all", choices=["all", "defense", "threat", "attack"],
                   help="States regularised toward the teacher (defense = threat + guard)")
    p.add_argument("--print", action="store_true", help="print the command instead of running it")
    a = p.parse_args()
    cmd = argv(a)
    if a.print:
        print(" ".join(cmd))
        return
    log = ROOT / "logs" / f"{a.run_name}.log"
    with open(log, "w") as fh:
        sys.exit(subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                 env={**__import__("os").environ, "PYTHONPATH": "ai:ai/bin", "OMP_NUM_THREADS": "2"}))


if __name__ == "__main__":
    main()
