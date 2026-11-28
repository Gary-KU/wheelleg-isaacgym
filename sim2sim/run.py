"""MuJoCo viewer for original or fine-tuned checkpoints (no Isaac Gym import)."""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg
from legged_gym.learning.policy import make_policy
from sim2sim.keyboard import KeyboardCommands, KeyboardViewer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "exports/h723_small/checkpoint.pt",
    )
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--terrain", choices=("flat", "wave", "ramp"), default="flat")
    p.add_argument("--wave-height", type=float, default=0.005, help="Peak-to-trough m")
    p.add_argument("--wave-length", type=float, default=0.4)
    p.add_argument("--ramp-height", type=float, default=0.03)
    p.add_argument("--flight-mode", choices=("support", "soft"), default="support")
    p.add_argument("--flight-compliance", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument(
        "--no-keyboard", action="store_true", help="Viewer only; use fixed --speed"
    )
    p.add_argument("--speed", type=float, default=0)
    p.add_argument("--report", type=Path, default=ROOT / "logs/mujoco/report.json")
    args = p.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        p.error("seconds must be positive and finite")
    if not np.isfinite(args.speed) or abs(args.speed) > 1:
        p.error("speed must be within [-1,1] m/s")
    torch.set_num_threads(1)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env = MujocoWheelLeg(
        source["legged_gym_env_config"],
        1,
        episode_seconds=args.seconds + 1,
        terrain=args.terrain,
        wave_height=args.wave_height,
        wave_length=args.wave_length,
        flight_compliance=args.flight_compliance,
        ramp_height=args.ramp_height,
        flight_mode=args.flight_mode,
    )
    policy = make_policy(132, 6, "cpu", source["training_config"]["policy"])
    policy.load_state_dict(source["model_state_dict"])
    policy.eval()
    viewer = KeyboardViewer(env.model, env.data[0]) if args.viewer else None
    keyboard = (
        KeyboardCommands(initial_height=env.initial_length)
        if args.viewer and not args.no_keyboard
        else None
    )
    if keyboard:
        print(
            "按住方向键驾驶，松键减速；Q/E缩腿/伸腿；空格停车；R重置；离开窗口自动零速。",
            flush=True,
        )
    peak = 0.0
    failures = 0
    completed = 0
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for step in range(round(args.seconds / env.dt)):
                if viewer is not None and not viewer.is_running():
                    break
                if keyboard:
                    keys, focused = viewer.keys()
                    command, reset = keyboard.update(keys, env.dt, focused)
                    if reset:
                        env.reset()
                    env.targets[0] = torch.tensor(command)
                elif step * env.dt >= 3:
                    env.targets[:, 0] = args.speed
                _, _, _, done, info = env.step(policy.act_inference(env.obs))
                peak = max(peak, info["angles"].abs().max().item())
                failures += int(info["physical_failures"].sum())
                completed = step + 1
                if keyboard and bool(done.any()):
                    keyboard.height = env.initial_length
                if viewer is not None:
                    if not keyboard:
                        viewer.keys()
                    viewer.sync(env.targets[0].tolist())
                    time.sleep(
                        max(0, completed * env.dt - (time.perf_counter() - start))
                    )
        report = dict(
            seconds=completed * env.dt,
            failures=failures,
            tilt_peak_rad=peak,
            simulator="mujoco",
            checkpoint=str(args.checkpoint.resolve()),
            full_acceptance_passed=False,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
