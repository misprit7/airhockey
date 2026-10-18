#!/usr/bin/env python3
"""DAgger: distill the observation-only teacher into ONE NeuralPlayer. Simulation only.

The teacher (airhockey.expert_player) labels every visited state; the saved
student is an ordinary NeuralPlayer that needs no teacher at play. Rollouts use
the curriculum of a recorded training command (default: production's run.json
args) through train_neural_player.environment_options, so drills, games,
sensing, thermal model and limits match the PPO training they came from.

Each round: collect --samples-per-round robot-seat transitions (the teacher
acts with probability beta per environment and block, else the student; the
far seat is the teacher / production / student mix), label both seats with the
teacher, aggregate into a capped buffer, fit the actor, save a snapshot and
evaluate it in the background (fast defense, games vs production).

Fitting defaults come from controlled comparisons on one fixed dataset
(2026-10-18): threat/attack/guard samples weighted 3/2/1 and a standardized
first layer took the 768-trial defense benchmark from 74.6% to 81.6% (teacher
82.2%); a fresh initialization, an L1 loss and widening to 2048 did not help.
--load-balance fixes the guard-depth bias at the teacher's 0.70 load bend but
lost cold-start blocks and overheated one round's student in games; off.

Writes runs/<name>/{agent.pt, agent_round_<k>.pt, run.json, status.json,
metrics.jsonl, evals/}. Example (from the repo root):

    python3 ai/bin/distill_expert.py --run-name _distill-smoke-20261018 \\
        --betas 1,0.5,0.25 --samples-per-round 2000000
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from airhockey.arrival import ArrivalDecoder
from airhockey.distillation import (DistillationEnv, Collector, LabelBuffer, RollIn, StandardizedInputs,
                                    actor_parameters, fit, heldout_report, parse_floats, parse_mix,
                                    physical_scales, widened)
from airhockey.dynamics import workspace_in_sim
from airhockey.expert_player import load_expert, parse_spec
from eval_neural_player import load
from train_neural_player import environment_options, stored_arguments

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = Path("runs/rail30-100-20261001-v1/agent.pt")
ALGORITHM = "expert_dagger_v1"
# Fields of a parent run.json that describe how THAT checkpoint was made.
PARENT_ONLY = ("training_frames", "validation_frames", "groups", "data", "anchor_weight",
               "hot_preservation_weight", "completion_definition", "holdout", "initialization_source_hashes",
               "replay_file", "report_day", "package_name", "physical_qualification", "initial_sha256",
               "selected_checkpoint", "selected_checkpoint_sha256", "skill_replay_sha256",
               "stationary_replay_sha256", "fixed_reference_checkpoints", "source_hashes", "evaluation_dir",
               "simulation_candidate", "reward_mode", "training_opponents", "exploration",
               "optimizer_transfer", "critic_initialization", "initialization", "device")
SOURCES = ["ai/bin/distill_expert.py", "ai/airhockey/distillation.py", "ai/airhockey/expert_player.py",
           "ai/airhockey/neural_player.py", "ai/airhockey/neural_training.py", "ai/airhockey/neural_vector.py",
           "ai/airhockey/neural_observation.py", "ai/airhockey/arrival.py", "ai/airhockey/arrival_env.py",
           "ai/airhockey/motion_guard.py", "ai/airhockey/batch_env.py", "ai/airhockey/thermal.py",
           "ai/airhockey/puck_prediction.py", "ai/bin/train_neural_player.py"]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinates(options):
    return dict(accel=float(options["accel"]), workspace_bounds_mm=[float(v) for v in options["workspace_bounds_mm"]])


def defense_summary(result):
    s = result["summary"]
    return dict(block_rate=s["block_rate"], ci95=s["ci95"], trials=s["trials"], saved=s["saved"],
                conceded=s["conceded"], unresolved=s["unresolved"], mean_release_depth=s["mean_release_depth"],
                peak_load=s["peak_load"], overload_seconds=s["overload_seconds"],
                groups={k: round(v["block_rate"], 4) for k, v in result["groups"].items()})


def games_summary(result):
    g = result["selfplay"]
    score, shots, aimed = (np.asarray(g[k]) for k in ("score", "shots", "on_target"))
    return dict(games=g["games"], seconds=g["seconds"], goals_for=int(score[0].sum()), goals_against=int(score[1].sum()),
                wins=int((score[0] > score[1]).sum()), losses=int((score[0] < score[1]).sum()),
                draws=int((score[0] == score[1]).sum()), shots=shots.sum(1).tolist(), on_target=aimed.sum(1).tolist(),
                on_target_rate=(aimed.sum(1) / np.maximum(shots.sum(1), 1)).round(4).tolist(),
                mean_shot_speed=[round(v, 3) for v in g["mean_shot_speed"]],
                captures=np.asarray(g["captures"]).sum(1).tolist(),
                overload_seconds=np.asarray(g["overload_seconds"]).sum(1).round(3).tolist(),
                peak_load=np.asarray(g["peak_load"]).max(1).round(4).tolist(),
                guard_interventions=np.asarray(g["guard_interventions"]).sum(1).tolist())


class Evaluations:
    """Background evaluation subprocesses, at most `parallel` at a time."""

    def __init__(self, directory, *, parallel, device, production, defense, games, on_result):
        self.directory, self.parallel, self.device = directory, parallel, device
        self.production, self.defense, self.games = production, defense, games
        self.on_result = on_result
        self.queue, self.running, self.results = [], [], {}
        directory.mkdir(parents=True, exist_ok=True)

    def submit(self, label, checkpoint):
        out = self.directory
        self.queue.append((label, "defense", out / f"{label}-defense.json", [
            sys.executable, "ai/bin/eval_fast_defense.py", str(checkpoint), "--trials", str(self.defense["trials"]),
            "--seed", str(self.defense["seed"]), "--device", self.device, "--output", str(out / f"{label}-defense.json")]))
        self.queue.append((label, "games", out / f"{label}-games.json", [
            sys.executable, "ai/bin/eval_neural_player.py", str(checkpoint), "--games-only",
            "--games", str(self.games["games"]), "--seconds", str(self.games["seconds"]),
            "--opponent-checkpoint", str(self.production), "--initial-load", str(self.games["initial_load"]),
            "--report-sensing", "--continuous-rallies", "--seed", str(self.games["seed"]),
            "--output", str(out / f"{label}-games.json")]))
        self.poll()

    def poll(self):
        for job in list(self.running):
            label, kind, output, argv, process, log, started = job
            if process.poll() is None:
                continue
            log.close()
            self.running.remove(job)
            entry = dict(label=label, kind=kind, output=str(output), seconds=time.monotonic() - started,
                         returncode=process.returncode)
            if process.returncode == 0 and output.exists():
                result = json.loads(output.read_text())
                entry["summary"] = defense_summary(result) if kind == "defense" else games_summary(result)
            else:
                entry["error"] = f"exit {process.returncode}; see {output.with_suffix('.log')}"
            self.results.setdefault(label, {})[kind] = entry
            self.on_result(entry)
        while self.queue and len(self.running) < self.parallel:
            label, kind, output, argv = self.queue.pop(0)
            log = output.with_suffix(".log").open("w")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "ai"), str(ROOT / "ai/bin")]),
                       OMP_NUM_THREADS="2")
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            self.running.append((label, kind, output, argv, process, log, time.monotonic()))

    def pending(self):
        return len(self.queue) + len(self.running)

    def wait(self, should_stop=lambda: False):
        while self.pending() and not should_stop():
            self.poll()
            time.sleep(2)

    def close(self):
        self.queue.clear()
        for _, _, _, _, process, log, _ in self.running:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            log.close()
        self.running.clear()


def distilled_run_metadata(parent_meta, options, **fields):
    """run.json for a distilled student.

    Starts from the initialization's run.json; what described how THAT
    checkpoint was made moves under parent_provenance. workspace_bounds_mm,
    thermal_model and physical_limits stay identical, so checkpoint_environment
    gives the student exactly its parent's simulation.
    """
    meta = {k: v for k, v in parent_meta.items() if k not in PARENT_ONLY}
    meta.update(fields, parent_provenance={k: parent_meta[k] for k in PARENT_ONLY if k in parent_meta})
    for key in ("workspace_bounds_mm", "thermal_model", "physical_limits"):
        if key in parent_meta:
            meta[key] = parent_meta[key]
    meta.setdefault("workspace_bounds_mm", options["workspace_bounds_mm"])
    meta.setdefault("physical_limits", dict(speed_m_s=12, acceleration_m_s2=options["accel"]))
    if "thermal_model" not in meta:
        thermal = options["thermal_path"]
        meta["thermal_model"] = dict(thermal) if isinstance(thermal, dict) else json.loads(Path(thermal).read_text())
    return meta


def student_checkpoint(student, *, init_state, base_args, **fields):
    """agent.pt contents in the production checkpoint's layout (no optimizer state)."""
    return dict(algorithm=ALGORITHM, model={k: v.detach().cpu() for k, v in student.state_dict().items()},
                step=init_state.get("step", 0), width=student.width, history=student.history,
                shot_conditioned=student.shot_conditioned, stage=init_state.get("stage", base_args.get("stage")),
                args=base_args, **fields)


def compact(report):
    """Held-out report for the round summary: totals and per-group MSE only."""
    out = dict(samples=report.get("samples", 0), mse=report.get("mse"), rmse=report.get("rmse"))
    for key, table in report.items():
        if key.startswith("by_"):
            out[key] = {k: v.get("mse") for k, v in table.items() if v.get("samples")}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="New directory runs/<name>; never overwritten")
    p.add_argument("--init", type=Path, default=PRODUCTION, help="Student initialization (a NeuralPlayer checkpoint)")
    p.add_argument("--width", type=int, help="Widen the student function-preservingly to this hidden width")
    p.add_argument("--teacher", default="expert", help="Teacher spec for expert_player.load_expert, e.g. expert:guard_depth=0.2")
    p.add_argument("--production", type=Path, default=PRODUCTION,
                   help="Far-seat network in the opponent mix; also the teacher's coordinate source")
    p.add_argument("--environment", type=Path, default=PRODUCTION.parent / "run.json",
                   help="Recorded training command whose curriculum the rollouts reproduce: a run.json (args) or recipe.json (argv)")
    p.add_argument("--betas", default="1,0.5,0.25,0.1,0", help="Per-round probability that the teacher acts; the last repeats")
    p.add_argument("--rounds", type=int, help="Number of rounds (default: one per beta)")
    p.add_argument("--samples-per-round", type=int, default=2_000_000, help="Robot-seat transitions collected per round")
    p.add_argument("--warmup-steps", type=int, default=100, help="Unrecorded decisions before round 0 to desynchronize episodes")
    p.add_argument("--block-steps", type=int, default=25, help="Decisions per teacher/student choice in one environment")
    p.add_argument("--opponent-block-steps", type=int, default=250, help="Decisions per far-seat choice in one environment")
    p.add_argument("--opponent-mix", default="0.4,0.3,0.3", help="Far-seat weights: teacher,production,student")
    p.add_argument("--label-opponents", action=argparse.BooleanOptionalAction, default=True,
                   help="Also label far-seat states played by the teacher or the student")
    p.add_argument("--student-noise", type=float, default=0.1, help="Pre-tanh Gaussian std when the student acts")
    p.add_argument("--opponent-noise", type=float, default=0.1, help="Pre-tanh Gaussian std for network far seats")
    p.add_argument("--teacher-noise", type=float, default=0.0, help="Action-space noise when the teacher acts (labels stay clean)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--envs-per-worker", type=int, default=1024)
    p.add_argument("--worker-threads", type=int, default=1, help="BLAS/OpenMP threads per simulation worker")
    p.add_argument("--dataset-cap", type=int, default=8_000_000,
                   help="Keep the most recent this many labelled states (1350 bytes each)")
    p.add_argument("--dataset-device", default="cpu", help="Where the float16 dataset lives (cpu or cuda)")
    p.add_argument("--heldout-fraction", type=float, default=0.1, help="Environment slots whose states are only evaluated")
    p.add_argument("--epochs", type=int, default=6, help="Passes over the aggregated training set per round")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final-fraction", type=float, default=0.1, help="Cosine decay within each round to lr times this")
    p.add_argument("--warmup-updates", type=int, default=200, help="Linear learning-rate warm-up in round 0")
    p.add_argument("--dim-weights", default="1,1,1,1,1,1", help="Loss weights for x,y,vx,vy,duration,accel")
    p.add_argument("--mode-weights", default="3,2,1",
                   help="Sample weights by teacher mode: threat,attack,guard (unknown mode weighs 1)")
    p.add_argument("--load-balance", type=float, default=0.0,
                   help="Weight samples to even out observed motor load, capped at this factor (0 disables)")
    p.add_argument("--standardize-inputs", action=argparse.BooleanOptionalAction, default=True,
                   help="Fit the first layer on standardized inputs, folded back into the saved weights")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--evals", action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate each round's snapshot in the background")
    p.add_argument("--teacher-evals", action=argparse.BooleanOptionalAction, default=True,
                   help="Also evaluate the teacher once on the same fixtures")
    p.add_argument("--eval-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-parallel", type=int, default=2)
    p.add_argument("--defense-trials", type=int, default=768)
    p.add_argument("--defense-seed", type=int, default=20261073)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--game-seconds", type=float, default=60)
    p.add_argument("--game-seed", type=int, default=20261300)
    p.add_argument("--game-initial-load", type=float, default=0.4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--paddle-restitution-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   help="Override the recorded curriculum's paddle restitution range (table: 0.47-0.59)")
    p.add_argument("--paddle-tangential-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   help="Override the recorded curriculum's paddle tangential retention range (table: 0.76-0.84)")
    args = p.parse_args()

    betas = parse_floats(args.betas)
    if any(not 0 <= b <= 1 for b in betas):
        p.error("--betas must lie in [0,1]")
    rounds = args.rounds or len(betas)
    mix = parse_mix(args.opponent_mix)
    weights = parse_floats(args.dim_weights, 6)
    if min(weights) < 0 or sum(weights) <= 0:
        p.error("--dim-weights must be nonnegative with a positive sum")
    mode_weights = parse_floats(args.mode_weights, 3)
    if min(mode_weights) <= 0:
        p.error("--mode-weights must be positive")
    if not 0 <= args.load_balance < 1000:
        p.error("--load-balance must be a cap in [0,1000)")
    if rounds < 1 or args.samples_per_round < 1 or args.workers < 1 or args.envs_per_worker < 1:
        p.error("rounds, samples, workers and environments must be positive")
    if not 0 < args.heldout_fraction < 1:
        p.error("--heldout-fraction must be in (0,1)")
    if min(args.student_noise, args.opponent_noise, args.teacher_noise) < 0:
        p.error("noise levels must be nonnegative")
    if str(args.init).startswith("expert"):
        p.error("--init must be a neural checkpoint; the teacher is never the deployed network")
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    run = ROOT / "runs" / args.run_name
    if run.exists():
        p.error(f"{run} exists; distillation never overwrites a run")

    # Student, far-seat network, teacher and the rollout curriculum.
    student, init_state = load(args.init)
    if args.width and args.width != student.width:
        student = widened(student, args.width)
    production, _ = load(args.production)
    teacher, _ = load_expert(args.teacher, args.production)
    record = json.loads(args.environment.read_text())
    train_args = stored_arguments(args.environment)
    if train_args.stationary_replay is not None and not Path(train_args.stationary_replay).exists():
        p.error(f"the recorded curriculum needs {train_args.stationary_replay} (paths are relative to the repo root)")
    for key in ("paddle_restitution_range", "paddle_tangential_range"):
        if getattr(args, key) is not None:
            setattr(train_args, key, list(getattr(args, key)))
    options = environment_options(train_args, shot_conditioned=True)
    options["seed"] = args.seed
    for name, net_options in (("student", student.environment_options), ("production", production.environment_options),
                              ("teacher", teacher.environment_options)):
        if coordinates(net_options) != coordinates(options):
            raise ValueError(f"{name} coordinates {coordinates(net_options)} differ from the rollout's {coordinates(options)}")
    if student.history != teacher.history or not student.shot_conditioned:
        raise ValueError("student needs the teacher's 16-frame, shot-conditioned observation")
    student.to(device).eval()
    production.to(device).eval()
    decoder = ArrivalDecoder([workspace_in_sim(bounds_mm=options["workspace_bounds_mm"])[k]
                              for k in ("min_x", "max_x", "min_y", "max_y")])
    scales = physical_scales(decoder)
    run.mkdir(parents=True, exist_ok=False)
    log_path = run / "metrics.jsonl"

    def emit(event, **values):
        row = dict(event=event, time=time.time(), **values)
        with log_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    base_args = dict(record["args"]) if "args" in record else {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(train_args).items()}
    base_args.update(width=student.width, history=student.history, history_deltas=student.history_deltas,
                     shot_conditioned=student.shot_conditioned)
    # Recorded so checkpoint_environment (every evaluator) replays the contact.
    for key in ("paddle_restitution_range", "paddle_tangential_range"):
        if getattr(args, key) is not None:
            base_args[key] = list(getattr(args, key))
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    settings.update(rounds=rounds, betas=betas, opponent_mix=dict(zip(("teacher", "production", "student"), mix.tolist())),
                    dim_weights=weights, mode_weights=dict(zip(("threat", "attack", "guard"), mode_weights)),
                    n_envs=args.workers * args.envs_per_worker)
    teacher_config = asdict(parse_spec(args.teacher))
    for path in SOURCES:
        destination = run / "source" / Path(path).name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes((ROOT / path).read_bytes())
    parent_meta_path = Path(args.init).parent / "run.json"
    parent_meta = json.loads(parent_meta_path.read_text()) if parent_meta_path.exists() else {}
    meta = distilled_run_metadata(
        parent_meta, options,
        algorithm=ALGORITHM, action_mode="arrival", obs_dim=student.obs_dim,
        physical_history_frames=student.history, shot_conditioned=student.shot_conditioned,
        controller=None, deployment_ready=False, simulation_only=True, simulation_candidate=False,
        args=base_args, reward_mode="none: supervised teacher labels (DAgger)",
        initialization=str(args.init), initialization_sha256=sha256(args.init),
        critic_initialization="checkpoint (distillation never updates the critic)",
        optimizer_transfer="fresh Adam over actor trunk and mean head; log_std, noise head and critic as initialized",
        exploration="unchanged learned variance; roll-in noise is collection-only",
        training_opponents="far seat drawn from teacher / production network / current student",
        teacher=args.teacher, teacher_config=teacher_config,
        teacher_sha256=sha256(ROOT / "ai/airhockey/expert_player.py"),
        teacher_role="training labels and sparring only; never saved with or run by the student",
        environment_source=str(args.environment), environment_source_sha256=sha256(args.environment),
        opponent_checkpoint=str(args.production), distillation=settings,
        parent_run_json=str(parent_meta_path) if parent_meta else None,
        parent_run_json_sha256=sha256(parent_meta_path) if parent_meta else None,
        evaluation_dir=str(run / "evals"), rounds=[],
        source_hashes={path: sha256(ROOT / path) for path in SOURCES})
    write_json(run / "run.json", meta)

    status = dict(pid=os.getpid(), running=True, phase="starting", round=None, rounds=rounds,
                  initial_step=0, step=0, target_steps=rounds * args.samples_per_round, started_at=time.time(),
                  run=args.run_name, teacher=args.teacher, evals={}, history=[])
    started = time.monotonic()

    def publish(**changes):
        status.update(changes, updated_at=time.time(), elapsed_s=time.monotonic() - started)
        write_json(run / "status.json", status)

    stopping = []

    def stop(*_):
        stopping.append(True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    should_stop = lambda: bool(stopping) or (run / "STOP").exists()

    def eval_result(entry):
        emit("eval", **entry)
        status["evals"].setdefault(entry["label"], {})[entry["kind"]] = entry.get("summary", entry.get("error"))
        print(json.dumps(dict(eval=entry["label"], kind=entry["kind"], result=entry.get("summary", entry.get("error")))), flush=True)
        publish()

    evaluations = Evaluations(run / "evals", parallel=args.eval_parallel, device=args.eval_device,
                              production=args.production, on_result=eval_result,
                              defense=dict(trials=args.defense_trials, seed=args.defense_seed),
                              games=dict(games=args.games, seconds=args.game_seconds, seed=args.game_seed,
                                         initial_load=args.game_initial_load)) if args.evals else None
    if evaluations and args.teacher_evals:
        evaluations.submit("teacher", args.teacher)
    publish()

    n = args.workers * args.envs_per_worker
    env = None
    try:
        print(f"starting {args.workers} workers x {args.envs_per_worker} environments; teacher {args.teacher}", flush=True)
        env = DistillationEnv(n, args.workers, teacher=args.teacher, environment_from=args.production,
                              history=student.history, worker_threads=args.worker_threads, **options)
        buffer = LabelBuffer(args.dataset_cap, student.obs_dim, device=args.dataset_device)
        heldout = rng.random(n) < args.heldout_fraction
        generator = torch.Generator(device=device).manual_seed(args.seed) if device.type == "cuda" else None
        collector = Collector(env, buffer, student, production, rng=rng, heldout=heldout, device=device,
                              rollin=RollIn(n, rng, args.block_steps, args.opponent_block_steps), mix=mix,
                              label_opponents=args.label_opponents, student_noise=args.student_noise,
                              opponent_noise=args.opponent_noise, teacher_noise=args.teacher_noise,
                              generator=generator)
        optimizer = standardizer = None  # created at the first fit, from its data
        cpu_generator = torch.Generator().manual_seed(args.seed)
        steps = math.ceil(args.samples_per_round / n)
        if args.warmup_steps:
            publish(phase="warm-up")
            collector.run(args.warmup_steps, beta=betas[0], round_index=-1, store=False, should_stop=should_stop)
        updates = robot_total = 0
        summaries = []
        for k in range(rounds):
            if should_stop():
                break
            beta = betas[min(k, len(betas) - 1)]
            publish(phase="collecting", round=k, beta=beta)
            def progress(row, k=k):
                print(json.dumps(dict(round=k, collecting=row)), flush=True)
                publish(phase="collecting", collect_progress=row)
                if evaluations:
                    evaluations.poll()

            collected = collector.run(steps, beta=beta, round_index=k, should_stop=should_stop, log=progress)
            robot_total += collected["robot_transitions"]
            emit("collect", round=k, beta=beta, **collected)
            print(json.dumps(dict(round=k, beta=beta, collected=collected["stored"],
                                  labelled_per_s=round(collected["labelled_per_s"]),
                                  robot_transitions_per_s=round(collected["transitions_per_s"]),
                                  buffer=buffer.size, games=collected["games"], shutdowns=collected["shutdowns"])), flush=True)
            before = heldout_report(student, buffer, device=device, scales=scales, rounds=[k])
            publish(phase="fitting", step=robot_total, labelled=buffer.added)
            if optimizer is None:
                if args.standardize_inputs:
                    standardizer = StandardizedInputs.from_buffer(student.trunk[0], buffer, generator=cpu_generator)
                optimizer = torch.optim.Adam(actor_parameters(student, standardizer), lr=args.lr, eps=1e-5)
            fitted = fit(student, optimizer, buffer, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                         final_fraction=args.lr_final_fraction, weights=weights, device=device,
                         warmup=args.warmup_updates if k == 0 else 0, clip=args.grad_clip, generator=cpu_generator,
                         log=lambda row: print(json.dumps(dict(round=k, **row)), flush=True),
                         mode_weights=mode_weights, standardizer=standardizer, load_balance=args.load_balance)
            updates += fitted["updates"]
            after_new = heldout_report(student, buffer, device=device, scales=scales, rounds=[k])
            after = heldout_report(student, buffer, device=device, scales=scales)
            emit("fit", round=k, **fitted, heldout_new_before=before, heldout_new_after=after_new, heldout_all=after)
            summary = dict(round=k, beta=beta, steps=collected["steps"], labelled=collected["stored"],
                           labelled_per_s=collected["labelled_per_s"], buffer=buffer.size, updates=fitted["updates"],
                           epoch_losses=fitted["epoch_losses"], fit_seconds=fitted["seconds"],
                           heldout_new_before=compact(before), heldout_new_after=compact(after_new),
                           heldout_all=compact(after), games=collected["games"], shutdowns=collected["shutdowns"],
                           teacher_modes=collected["teacher_modes"])
            summaries.append(summary)
            state = student_checkpoint(student, init_state=init_state, base_args=base_args, teacher=args.teacher,
                                       teacher_config=teacher_config, source_checkpoint=str(args.init),
                                       supervised_update=updates, distill_samples=buffer.added, round=k,
                                       rounds=summaries, distillation=settings)
            snapshot = run / f"agent_round_{k}.pt"
            for path in (snapshot, run / "agent.pt"):
                temp = path.with_suffix(".tmp")
                torch.save(state, temp)
                temp.replace(path)
            meta.update(rounds=summaries, selected_checkpoint=f"runs/{args.run_name}/agent.pt",
                        selected_checkpoint_sha256=sha256(run / "agent.pt"), selected_round=k)
            write_json(run / "run.json", meta)
            print(json.dumps(dict(round=k, heldout_mse_new_before=before.get("mse"), heldout_mse_new_after=after_new.get("mse"),
                                  heldout_mse_all=after.get("mse"), by_mode=compact(after_new).get("by_mode"),
                                  rmse=after_new.get("rmse"), epoch_losses=fitted["epoch_losses"])), flush=True)
            status["history"].append(dict(round=k, beta=beta, mse=after.get("mse"), mse_new_before=before.get("mse"),
                                          mse_new_after=after_new.get("mse"), by_mode=compact(after_new).get("by_mode"),
                                          labelled_per_s=collected["labelled_per_s"]))
            publish(phase="evaluating" if evaluations else "round complete", step=robot_total, labelled=buffer.added,
                    buffer=buffer.size,
                    heldout_mse=after.get("mse"), transitions_per_s=collected["labelled_per_s"], updates=updates)
            if evaluations:
                evaluations.submit(f"round_{k}", snapshot)
        if evaluations:
            publish(phase="waiting for evaluations")
            evaluations.wait(should_stop)
        status.update(running=False, phase="stopped" if should_stop() else "finished")
        publish()
        print(f"finished {run}", flush=True)
    except BaseException as error:
        status.update(running=False, failed=True, phase="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise
    finally:
        if evaluations:
            evaluations.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
