#!/usr/bin/env python3
"""Run a finite stability training experiment with matched before/after tests.

CPU evaluation -> preserved 186-to-198 warm start -> bounded GPU training ->
the same CPU evaluation. No infinite restarts or automatic success claims.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wheelleg_gym.optimization_plan import compare_reports, evaluation_cases


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True, help="New or empty experiment directory")
    parser.add_argument("--iterations", type=int, default=200, help="Finite additional PPO updates")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--eval-envs", type=int, default=1)
    parser.add_argument("--standing-seconds", type=float, default=60.0)
    parser.add_argument("--dynamic-segment-steps", type=int, default=100)
    parser.add_argument("--rapid-segment-steps", type=int, default=10)
    parser.add_argument("--rapid-repeats", type=int, default=10)
    parser.add_argument("--train-timeout", type=float, default=7200)
    parser.add_argument("--evaluation-timeout", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating files or launching children")
    args = parser.parse_args()
    for name in ("iterations", "num_envs", "save_interval", "eval_envs"):
        if getattr(args, name) < 1:
            parser.error("{} must be positive".format(name))
    for name in ("train_timeout", "evaluation_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("{} must be positive and finite".format(name))
    try:
        cases = evaluation_cases(args.standing_seconds, args.dynamic_segment_steps,
                                 args.rapid_segment_steps, args.rapid_repeats)
    except ValueError as error:
        parser.error(str(error))
    source = args.checkpoint.resolve()
    if not source.is_file():
        parser.error("Checkpoint does not exist: {}".format(source))
    # Gym is only imported inside independent child processes; parent uses CPU Torch.
    import torch
    state = torch.load(str(source), map_location="cpu")
    env = state.get("environment_config", {})
    observations = state["model_state_dict"]["actor.0.weight"].shape[1]
    if observations not in (186, 198):
        parser.error("Only 186- or 198-observation feed-forward checkpoints are supported")
    if bool(env.get("observe_leg_lengths", False)) != (observations == 198):
        parser.error("Checkpoint observation layout and environment metadata disagree")
    if not math.isclose(float(env.get("control_dt", .02)), .02, abs_tol=1e-9):
        parser.error("This evaluation suite requires the 0.02 s control period")
    start_iteration = int(state["iter"])
    log_dir = args.log_dir.resolve()
    train_dir = log_dir/"training"
    upgraded = log_dir/"input"/"model_{}_legs.pt".format(start_iteration) if observations == 186 else source
    final_checkpoint = train_dir/"model_{}.pt".format(start_iteration+args.iterations)
    physical = ["--physics-dt", str(env.get("physics_dt", .0005)),
                "--closure-stiffness", str(env.get("closure_stiffness", 50000.0)),
                "--closure-damping", str(env.get("closure_damping", 25.0)),
                "--seed", str(env.get("seed", 42))]
    common_evaluation = ["--sim-device", "cpu", "--num-envs", str(args.eval_envs),
                         "--command", "0", "0", "0", ".08", "--disable-time-limit",
                         "--linear-acceleration", ".8", "--yaw-acceleration", "2",
                         "--allow-approximate-loop"] + physical
    train_command = [sys.executable, str(ROOT/"train.py"), "--profile", "stability",
                     "--checkpoint", str(upgraded), "--headless", "--sim-device", "cuda:0",
                     "--num-envs", str(args.num_envs), "--max-iterations", str(args.iterations),
                     "--save-interval", str(args.save_interval), "--log-dir", str(train_dir),
                     "--allow-approximate-loop"] + physical
    plan = {"source_checkpoint": str(source), "source_iteration": start_iteration,
            "source_observations": observations, "training_checkpoint": str(upgraded),
            "final_checkpoint": str(final_checkpoint), "planned_new_iterations": args.iterations,
            "num_envs": args.num_envs, "evaluation_cases": cases,
            "common_evaluation_arguments": common_evaluation, "training_command": train_command,
            "train_timeout_s": args.train_timeout, "evaluation_timeout_s": args.evaluation_timeout,
            "quality_note": "Scripted quality checks are separate from successful program completion."}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if log_dir.exists() and any(log_dir.iterdir()):
        parser.error("Experiment directory must be new or empty: {}".format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(log_dir/"plan.json", plan)
    child = None
    phase = "starting"

    def status(state_name, **extra):
        value = dict(state=state_name, phase=phase,
                     updated_at_utc=datetime.now(timezone.utc).isoformat(),
                     pipeline_pid=os.getpid(), child_pid=child.pid if child is not None else None,
                     source_checkpoint=str(source), final_checkpoint=str(final_checkpoint),
                     log_dir=str(log_dir), **extra)
        atomic_json(log_dir/"optimization_status.json", value)

    def stop_child():
        if child is None or child.poll() is not None:
            return
        for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                return
            try:
                child.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Pipeline received signal {}".format(signum))

    def run(command, output, timeout):
        nonlocal child
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as log:
            child = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            status("running", command=command, console_log=str(output))
            print("phase={} child_pid={}".format(phase, child.pid), flush=True)
            code = child.wait(timeout=timeout)
        if code:
            raise RuntimeError("{} exited with code {}; see {}".format(phase, code, output))

    def evaluate(checkpoint, label):
        nonlocal phase
        reports = {}
        for case in cases:
            phase = label+"/"+case["name"]
            report = log_dir/"evaluation"/label/(case["name"]+".json")
            command = [sys.executable, str(ROOT/"evaluate.py"), "--checkpoint", str(checkpoint),
                       "--report", str(report)] + common_evaluation + case["arguments"]
            run(command, report.with_suffix(".log"), args.evaluation_timeout)
            data = json.loads(report.read_text())
            if data.get("completed_steps") != case["expected_steps"]:
                raise RuntimeError("Evaluation did not complete the requested steps: "+phase)
            reports[case["name"]] = data
        return reports

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        status("running")
        baseline = evaluate(source, "baseline")
        if observations == 186:
            phase = "observation_upgrade"
            status("running")
            from wheelleg_gym.observation_upgrade import save_upgraded_checkpoint
            save_upgraded_checkpoint(source, upgraded)
        phase = "training"
        run(train_command, train_dir/"console.log", args.train_timeout)
        if not final_checkpoint.is_file():
            raise RuntimeError("Training did not produce the final checkpoint")
        trained = torch.load(str(final_checkpoint), map_location="cpu")
        actual_env = trained["environment_config"]
        if not all(actual_env.get(name) for name in ("observe_leg_lengths", "stability_rewards", "tracking_penalties")):
            raise RuntimeError("Training checkpoint does not contain the requested stability profile")
        expected = {"physics_dt": float(env.get("physics_dt", .0005)),
                    "closure_stiffness": float(env.get("closure_stiffness", 50000.0)),
                    "closure_damping": float(env.get("closure_damping", 25.0)),
                    "linear_acceleration": .8, "yaw_acceleration": 2.,
                    "command_resample_seconds": 2.}
        for key, value in expected.items():
            if not math.isclose(float(actual_env.get(key, float("nan"))), value, rel_tol=1e-8, abs_tol=1e-9):
                raise RuntimeError("Training/evaluation configuration mismatch: "+key)
        atomic_json(log_dir/"actual_training_environment.json", actual_env)
        candidate = evaluate(final_checkpoint, "candidate")
        phase = "comparison"
        comparison = compare_reports(baseline, candidate)
        comparison.update(source_checkpoint=str(source), candidate_checkpoint=str(final_checkpoint),
                          matched_evaluation_arguments=common_evaluation,
                          initial_state_perturbations=False)
        atomic_json(log_dir/"comparison.json", comparison)
        status("completed", quality_status=comparison["quality_status"],
               scripted_checks_passed=comparison["scripted_checks_passed"],
               comparison_report=str(log_dir/"comparison.json"))
        print(json.dumps({"program_state": "completed", "quality_status": comparison["quality_status"],
                          "comparison_report": str(log_dir/"comparison.json")}), flush=True)
    except KeyboardInterrupt as error:
        stop_child()
        status("stopped", error=str(error))
        raise SystemExit(130)
    except BaseException as error:
        stop_child()
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
