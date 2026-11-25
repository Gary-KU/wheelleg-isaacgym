#!/usr/bin/env python3
"""Inspect the active CAD robot without loading a policy or claiming balance."""

from pathlib import Path
import argparse
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from legged_gym.envs.wheelleg.tasks.base import WheelLegged
from legged_gym.envs.wheelleg.config.base import WheelLeggedStandCfg
from legged_gym.envs.wheelleg.model.geometry import asset_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float, default=0.10)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--sim-device", default="cpu")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--free-base",
        action="store_true",
        help="Remove fixed-base inspection fixture; no trained balancing policy is used",
    )
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        p.error("Use positive steps/env count")
    torch.set_num_threads(1)
    cfg = WheelLeggedStandCfg()
    cfg.env.num_envs = args.num_envs
    cfg.env.initial_leg_length = args.length
    cfg.commands.command = (0.0, 0.0, 0.0, args.length)
    cfg.commands.leg_length_range = (args.length, args.length)
    cfg.env.fix_base = not args.free_base
    env = WheelLegged(cfg, sim_device=args.sim_device, headless=args.headless)
    try:
        peak = 0.0
        failures = 0
        completed = 0
        for _ in range(args.steps):
            if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
                break
            _, _, _, _, info, _ = env.step(
                torch.zeros(env.num_envs, 6, device=env.device)
            )
            peak = max(peak, float(info["loop_error_max"]))
            failures += int(info["physical_failures"].sum())
            completed += 1
        result = dict(
            asset_fingerprint=asset_fingerprint(),
            fixed_base_inspection=cfg.env.fix_base,
            completed_steps=completed,
            sim_device=env.device,
            num_envs=env.num_envs,
            physical_failures=failures,
            closure_peak_m=peak,
            initial_pose=env.initial_pose_diagnostics,
            gravity_z=env.gym.get_sim_params(env.sim).gravity.z,
            note="Model inspection only; fixed base is an explicit fixture, not a standing result. Free base has PD/support but no balancing policy.",
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if cfg.env.fix_base and (failures or peak > .005):
            raise RuntimeError("Fixed-base CAD inspection exceeded the physical failure/5mm closure check")
    finally:
        env.close()


if __name__ == "__main__":
    main()
