#!/usr/bin/env python3
"""Compare actual PPO throughput while preserving a paused training process.

The running policy is never edited. Each candidate loads the same checkpoint,
uses identical PPO settings and writes its own logs. SIGCONT runs in finally.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-counts", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    if args.updates < 3 or any(n < 1 for n in args.env_counts):
        parser.error("Need at least three updates and positive environment counts")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error("Checkpoint not found")
    cmdline = Path("/proc/{}/cmdline".format(args.training_pid)).read_bytes()
    if b"train.py" not in cmdline or b"isaacgym" not in cmdline:
        parser.error("The requested PID is not an Isaac Gym training process")
    config = json.loads((checkpoint.parent / "config.json").read_text())
    env = config["environment"]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(checkpoint), "paused_training_pid": args.training_pid,
              "warmup_updates_discarded": 1, "ppo_settings": config["training"],
              "note": "Sample throughput comparison; not a convergence-speed guarantee.", "results": []}

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Benchmark interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    paused = False
    try:
        os.kill(args.training_pid, signal.SIGSTOP)
        paused = True
        time.sleep(0.3)  # Let already-issued GPU work finish.
        for n in args.env_counts:
            run_dir = args.output / "env_{}".format(n)
            run_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(ROOT / "train.py"), "--headless",
                "--num-envs", str(n), "--max-iterations", str(args.updates),
                "--checkpoint", str(checkpoint), "--log-dir", str(run_dir),
                "--allow-approximate-loop"]
            for option, key in (("--physics-dt", "physics_dt"),
                                ("--closure-stiffness", "closure_stiffness"),
                                ("--closure-damping", "closure_damping"),
                                ("--seed", "seed")):
                command += [option, str(env[key])]
            for option, key in (("--command", "command"),
                                ("--forward-speed-range", "forward_speed_range"),
                                ("--yaw-speed-range", "yaw_speed_range"),
                                ("--leg-length-range", "leg_length_range")):
                command += [option] + [str(x) for x in env[key]]
            if env.get("random_commands"):
                command += ["--random-commands"]
            if env.get("tracking_penalties"):
                command += ["--tracking-penalties"]
            print("Benchmarking {} environments...".format(n), flush=True)
            with (run_dir / "console.log").open("w") as log:
                subprocess.run(command, cwd=str(ROOT), stdout=log,
                               stderr=subprocess.STDOUT, timeout=90, check=True)
            summary = json.loads((run_dir / "training_summary.json").read_text())
            measured = summary["iterations"][1:]
            seconds = statistics.median(x["wall_seconds"] for x in measured)
            result = {"num_envs": n, "updates_measured": len(measured),
                "median_seconds_per_update": seconds,
                "samples_per_update": n * config["training"]["num_steps_per_env"],
                "environment_steps_per_second": n*config["training"]["num_steps_per_env"]/seconds,
                "max_closure_error_m": max(x["Loop/max_anchor_error_m"] for x in summary["iterations"]),
                "physical_failures": sum(x["Episode/physical_failures"] for x in summary["iterations"]),
                "pytorch_peak_bytes_excludes_physx": summary.get("pytorch_cuda_peak_allocated_bytes")}
            report["results"].append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(result), flush=True)
    finally:
        if paused:
            try:
                os.kill(args.training_pid, signal.SIGCONT)
                report["original_training_resumed"] = True
            except ProcessLookupError:
                report["original_training_resumed"] = False
        (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print("Original training resumed.", flush=True)


if __name__ == "__main__":
    main()
