#!/usr/bin/env python3
"""历史 186 维实验的有限训练/评估流程，状态和日志写入磁盘。

本工具仍只透传旧版环境参数，未完整支持新增观测、稳定奖励和指令调度。
新增实验使用 train.py / evaluate.py 明确指定参数；先阅读 tools/README.md。
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--end-iteration", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    state = torch.load(str(checkpoint), map_location="cpu")
    start = int(state["iter"])
    if args.end_iteration <= start:
        parser.error("End iteration must exceed the checkpoint iteration")
    env = state["environment_config"]
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    status_file = ROOT / "validation" / "final_evaluation_pipeline.json"
    result_file = ROOT / "validation" / "final_speed_scenarios.json"
    final_checkpoint = log_dir / "model_{}.pt".format(args.end_iteration)
    child = None

    def status(phase, **extra):
        value = dict(state=phase, pipeline_pid=os.getpid(),
            child_pid=child.pid if child is not None else None,
            checkpoint=str(checkpoint), start_iteration=start,
            planned_end_iteration=args.end_iteration, num_envs=args.num_envs,
            log_dir=str(log_dir), final_checkpoint=str(final_checkpoint),
            report=str(result_file), **extra)
        temp = status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2)+"\n")
        temp.replace(status_file)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate()
        raise KeyboardInterrupt("Pipeline stopped")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        command = [sys.executable, str(ROOT/"train.py"), "--headless",
            "--num-envs", str(args.num_envs), "--max-iterations", str(args.end_iteration-start),
            "--checkpoint", str(checkpoint), "--log-dir", str(log_dir),
            "--save-interval", "5", "--allow-approximate-loop"]
        for option, key in (("--physics-dt", "physics_dt"), ("--closure-stiffness", "closure_stiffness"),
                            ("--closure-damping", "closure_damping"), ("--seed", "seed")):
            command += [option, str(env[key])]
        for option, key in (("--command", "command"), ("--forward-speed-range", "forward_speed_range"),
                            ("--yaw-speed-range", "yaw_speed_range"), ("--leg-length-range", "leg_length_range")):
            command += [option]+[str(x) for x in env[key]]
        if env.get("random_commands"):
            command += ["--random-commands"]
        if env.get("tracking_penalties"):
            command += ["--tracking-penalties"]
        with (log_dir/"console.log").open("w") as log:
            child = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            status("training")
            code = child.wait()
        if code:
            raise RuntimeError("Training exited with code {}".format(code))
        if not final_checkpoint.is_file():
            raise RuntimeError("Final checkpoint was not produced")
        command = [sys.executable, str(ROOT/"evaluate.py"), "--checkpoint", str(final_checkpoint),
            "--num-envs", "3", "--num-steps", "800", "--command", "0", "0", "0", str(env["command"][3]),
            "--test-speeds", "-0.3", "0", "0.3", "--report", str(result_file), "--allow-approximate-loop"]
        with (log_dir/"evaluation.log").open("w") as log:
            child = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            status("evaluating")
            code = child.wait(timeout=500)
        if code:
            raise RuntimeError("Evaluation exited with code {}".format(code))
        status("completed")
    except KeyboardInterrupt as error:
        status("stopped", error=str(error))
        raise
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
