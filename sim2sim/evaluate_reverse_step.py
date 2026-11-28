"""Diagnose reversing into the vertical back of the ramp, without flight gates.

World coordinates initialize the test and record progress only. Actor inputs,
motor limits, sensor noise/delay and the 0.8 m/s² command slew are unchanged.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg
from legged_gym.learning.policy import make_policy


def run(checkpoint, seed, height, seconds):
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env = MujocoWheelLeg(source["legged_gym_env_config"], 1, seed,
                        episode_seconds=seconds + 1, terrain="ramp", ramp_height=height)
    # Move the stationary reset pose to the flat ground behind the lip. This
    # translation changes neither the IMU nor joint/encoder reset readings.
    env.data[0].qpos[0] += 1.5
    mujoco.mj_forward(env.model, env.data[0])
    policy = make_policy(132, 6, "cpu", source["training_config"]["policy"])
    policy.load_state_dict(source["model_state_dict"])
    policy.eval()
    peak = 0.0
    cross = 0.0
    torque = torch.zeros(6)
    trace = []
    failures = 0
    final_x = 1.5
    with torch.inference_mode():
        for step in range(round(seconds / env.dt)):
            if step * env.dt >= 2:
                env.targets[:, 0] = -1.0
            _, _, _, done, info = env.step(policy.act_inference(env.obs))
            failures += int(info["physical_failures"].sum())
            peak = max(peak, float(info["angles"].abs().max()))
            torque = torch.maximum(torque, env.last_torque_peak[0])
            # Automatic reset overwrites qpos on a failure; use the last
            # pre-failure position and explicitly mark that terminal sample.
            if not bool(done.any()):
                final_x = float(env.data[0].qpos[0])
                cross = max(cross, abs(float(env.data[0].qpos[1])))
            if step % 5 == 0 or bool(done.any()):
                trace.append(dict(t=step * env.dt, x=final_x,
                                  pitch=float(info["angles"][0, 0]),
                                  pitch_rate=float(info["velocity"][0, 1]),
                                  speed=float(info["velocity"][0, 3]),
                                  terminal=bool(done.any())))
            if bool(done.any()):
                break
    result = dict(seed=seed, seconds=(step + 1) * env.dt, failures=failures,
                  tilt_peak_rad=peak, cross_track_peak_m=cross, final_x_m=final_x,
                  completed_reverse_crossing=failures == 0 and final_x < .35,
                  torque_peak_nm=torque.tolist(), trace=trace)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ramp-height", type=float, default=.01)
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports = [run(args.checkpoint, seed, args.ramp_height, args.seconds) for seed in (17, 42)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
                                         ramp_height_m=args.ramp_height, reports=reports), indent=2))
    print(json.dumps([{k: v for k, v in report.items() if k != "trace"} for report in reports], indent=2))


if __name__ == "__main__":
    main()
