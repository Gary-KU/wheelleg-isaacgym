#!/usr/bin/env python3
"""传感器策略的确定性站立检查；连续时长、漂移和力矩按重置前状态统计。"""
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from legged_gym.envs import register_tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wheelleg.diagnostics.motion_evaluation import MotionEvaluation
from legged_gym.envs.wheelleg.constants import ACTION_JOINTS
from legged_gym.utils.helpers import viewer_is_closed


def main():
    args = get_args("play")
    if args.task not in (
        "wheelleg_sensor",
        "wheelleg_sensor_tracking",
        "wheelleg_sensor_quiet",
        "wheelleg_sensor_drive",
    ):
        raise ValueError(
            "Select a sensor task: wheelleg_sensor, wheelleg_sensor_quiet, wheelleg_sensor_drive or wheelleg_sensor_tracking"
        )
    torch.set_num_threads(1)
    register_tasks()
    cfg, train = task_registry.prepare_cfgs(args.task, args)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        runner, _ = task_registry.make_alg_runner(
            env, args.task, args, train, log_root=None
        )
        policy = runner.get_inference_policy()
        env.max_episode_length = args.num_steps + 1  # 不把 16 秒超时重置当作站稳
        initial_root_positions = env.root_states[:, :3].tolist()
        motion = MotionEvaluation(env.root_states, env.dt)
        age = torch.zeros(env.num_envs, device=env.device)
        longest = age.clone()
        saturation = torch.zeros(6, device=env.device)
        torque_peak = saturation.clone()
        wheel_squares = torch.zeros(2, device=env.device)
        wheel_peak = wheel_squares.clone()
        legs = torch.zeros(2, device=env.device)
        leg_errors = legs.clone()
        action_abs = saturation.clone()
        rewards = {}
        failures = timeouts = completed = 0
        pitch_peak = roll_peak = loop_peak = 0.0
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(args.num_steps):
                if viewer_is_closed(env):
                    break
                obs, _ = env.get_observations()
                action = policy(obs)
                action_abs += action.abs().mean(0)
                obs, _, reward, done, info, _ = env.step(action)
                if not bool(torch.isfinite(obs).all() & torch.isfinite(reward).all()):
                    raise RuntimeError("Non-finite evaluation output")
                motion.update(
                    info["pre_reset_root_states"],
                    info["pre_reset_commands"],
                    done,
                    env.root_states,
                )
                age += env.dt
                longest = torch.maximum(longest, age)
                age[done] = 0
                failures += int(info["physical_failures"].sum())
                timeouts += int(info["time_outs"].sum())
                pitch_peak = max(pitch_peak, float(info["pre_reset_pitch"].abs().max()))
                roll_peak = max(roll_peak, float(info["pre_reset_roll"].abs().max()))
                loop_peak = max(loop_peak, float(info["loop_error_max"]))
                saturation += info["pre_reset_torque_saturation_fraction"].mean(0)
                torque_peak = torch.maximum(
                    torque_peak, info["pre_reset_peak_abs_torques"].max(0).values
                )
                lengths = info["pre_reset_leg_length"]
                legs += lengths.mean(0)
                leg_errors += (
                    (lengths - info["pre_reset_commands"][:, 3:4]).square().mean(0)
                )
                wheels = info["pre_reset_active_rates"][:, 4:]
                wheel_squares += wheels.square().mean(0)
                wheel_peak = torch.maximum(wheel_peak, wheels.abs().max(0).values)
                for key, value in info["reward_terms"].items():
                    rewards[key] = rewards.get(key, 0.0) + float(value.mean())
                completed = i + 1
                if completed % max(1, round(10.0 / env.dt)) == 0:
                    print(
                        "balance_eval simulated_s={:.1f} failures={} timeouts={}".format(
                            completed * env.dt, failures, timeouts
                        ),
                        flush=True,
                    )
        report = dict(
            checkpoint=str(args.checkpoint_path),
            task=args.task,
            seed=cfg.seed,
            simulation=dict(
                physics_dt_s=env.physics_dt,
                control_dt_s=env.dt,
                policy_hz=1.0 / env.dt,
                environment_spacing_m=cfg.env.spacing,
                gpu_buffer_size_multiplier=env.gpu_buffer_size_multiplier,
                closure_stiffness_n_m=cfg.closure.closure_stiffness,
                closure_damping_ns_m=cfg.closure.closure_damping,
                sensor_delay_seconds=cfg.sensors.delay_steps * env.dt,
            ),
            initial_root_positions_m=initial_root_positions,
            initialization_provenance=runner.train_cfg.get("initialization_provenance"),
            steps=completed,
            seconds_per_environment=completed * env.dt,
            num_envs=env.num_envs,
            wall_seconds=time.perf_counter() - start,
            physical_failures=failures,
            timeouts=timeouts,
            longest_continuous_seconds_by_env=longest.tolist(),
            pitch_peak_rad=pitch_peak,
            roll_peak_rad=roll_peak,
            closure_peak_m=loop_peak,
            motion=motion.report(),
            initial_com_balance=getattr(env, "initial_com_balance_diagnostics", None),
            joint_order=list(ACTION_JOINTS),
            saturation_fraction_by_joint=(saturation / max(completed, 1)).tolist(),
            mean_leg_length_m=(legs / max(completed, 1)).tolist(),
            rms_leg_length_error_m=(leg_errors / max(completed, 1)).sqrt().tolist(),
            mean_absolute_policy_action_by_joint=(
                action_abs / max(completed, 1)
            ).tolist(),
            torque_peak_nm_by_joint=torque_peak.tolist(),
            wheel_rms_rad_s=(wheel_squares / max(completed, 1)).sqrt().tolist(),
            wheel_peak_rad_s=wheel_peak.tolist(),
            mean_reward_terms={k: v / max(completed, 1) for k, v in rewards.items()},
            note="No policy exploration noise; configured sensor noise remains. Physical failures reset automatically and are counted; displacement excludes reset teleports. No quality pass is implied by completion.",
        )
        motion_report = report["motion"]
        checks = {
            "at_least_60_seconds": completed * env.dt >= 60.0,
            "no_physical_failures": failures == 0,
            "no_timeout_resets": timeouts == 0,
            "zero_motion_command": all(abs(v) < 1e-6 for v in cfg.commands.command[:3]),
            "forward_rms_below_002_m_s": motion_report["rms_forward_speed_error_m_s"]
            is not None
            and motion_report["rms_forward_speed_error_m_s"] <= 0.02,
            "lateral_rms_below_002_m_s": motion_report["rms_lateral_speed_m_s"]
            is not None
            and motion_report["rms_lateral_speed_m_s"] <= 0.02,
            "yaw_rate_rms_below_006_rad_s": motion_report["rms_yaw_rate_error_rad_s"]
            is not None
            and motion_report["rms_yaw_rate_error_rad_s"] <= 0.06,
            "forward_displacement_below_01_m": motion_report[
                "segment_peak_forward_displacement_m"
            ]
            <= 0.1,
            "lateral_displacement_below_01_m": motion_report[
                "segment_peak_cross_track_m"
            ]
            <= 0.1,
            "heading_change_below_01_rad": motion_report[
                "segment_peak_heading_change_rad"
            ]
            <= 0.1,
            "leg_rms_error_below_005_m": max(report["rms_leg_length_error_m"]) <= 0.005,
            "pitch_roll_peaks_below_02_rad": max(pitch_peak, roll_peak) <= 0.2,
        }
        report["quiet_stand_checks"] = checks
        report["quiet_stand_passed"] = all(checks.values())
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
