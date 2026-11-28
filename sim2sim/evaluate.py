"""Deterministic multi-seed MuJoCo playback, without reset duration inflation."""

import argparse, json, sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg
from legged_gym.learning.policy import make_policy


def evaluate(
    checkpoint,
    seconds=10.0,
    seed=42,
    num_envs=4,
    speed=0.0,
    terrain="flat",
    wave_height=0.005,
    wave_length=0.4,
):
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env = MujocoWheelLeg(
        source["legged_gym_env_config"],
        num_envs,
        seed,
        episode_seconds=seconds + 2,
        terrain=terrain,
        wave_height=wave_height,
        wave_length=wave_length,
    )
    policy = make_policy(132, 6, "cpu", source["training_config"]["policy"])
    policy.load_state_dict(source["model_state_dict"])
    policy.eval()
    peak = 0.0
    roll = 0.0
    failures = 0
    longest = torch.zeros(num_envs, dtype=torch.float64)
    consecutive = torch.zeros(num_envs, dtype=torch.float64)
    initial = torch.from_numpy(np.array([d.qpos[:3].copy() for d in env.data]))
    displacement = 0.0
    reward_sum = 0.0
    torques = torch.zeros(6)
    speed_sum = torch.zeros(num_envs, dtype=torch.float64)
    speed_error_square = torch.zeros(num_envs, dtype=torch.float64)
    speed_samples = 0
    with torch.inference_mode():
        for step in range(round(seconds / env.dt)):
            if step * env.dt >= 3:
                env.targets[:, 0] = speed
            obs, _, reward, done, info = env.step(policy.act_inference(env.obs))
            peak = max(peak, info["angles"][:, 0].abs().max().item())
            roll = max(roll, info["angles"][:, 1].abs().max().item())
            failures += int(info["physical_failures"].sum())
            consecutive += env.dt
            longest = torch.maximum(longest, consecutive)
            consecutive[done] = 0
            displacement = max(
                displacement,
                float(
                    (
                        torch.from_numpy(
                            np.array([d.qpos[:3].copy() for d in env.data])
                        )
                        - initial
                    )[:, :2]
                    .norm(dim=1)
                    .max()
                ),
            )
            torques = torch.maximum(torques, env.last_torque_peak.max(0).values)
            reward_sum += reward.mean().item()
            if step * env.dt >= 6:
                speed_sum += info["velocity"][:, 3]
                speed_error_square += (info["velocity"][:, 3] - speed).square()
                speed_samples += 1
    result = dict(
        checkpoint=str(checkpoint),
        terrain=terrain,
        wave_height_m=wave_height if terrain == "wave" else None,
        wave_length_m=wave_length if terrain == "wave" else None,
        seed=seed,
        num_envs=num_envs,
        seconds=seconds,
        failures=failures,
        pitch_peak_rad=peak,
        roll_peak_rad=roll,
        longest_continuous_s=longest.tolist(),
        peak_displacement_m=displacement,
        settled_forward_speed_m_s=(
            (speed_sum / speed_samples).tolist() if speed_samples else None
        ),
        settled_forward_rms_error_m_s=(
            (speed_error_square / speed_samples).sqrt().tolist()
            if speed_samples
            else None
        ),
        mean_reward=reward_sum / (seconds / env.dt),
        torque_peak_nm=torques.tolist(),
        standing_gate_passed=bool(
            speed == 0
            and failures == 0
            and max(peak, roll) <= 0.2
            and displacement <= 0.05
        ),
    )
    env.close()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--speed", type=float, default=0)
    p.add_argument("--terrain", choices=("flat", "wave"), default="flat")
    p.add_argument("--wave-height", type=float, default=0.005)
    p.add_argument("--wave-length", type=float, default=0.4)
    p.add_argument("--report", required=True, type=Path)
    a = p.parse_args()
    if a.seconds <= 0 or a.num_envs < 1:
        p.error("seconds and num-envs must be positive")
    torch.set_num_threads(1)
    result = evaluate(
        a.checkpoint,
        a.seconds,
        a.seed,
        a.num_envs,
        a.speed,
        a.terrain,
        a.wave_height,
        a.wave_length,
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
