#!/usr/bin/env python3
"""四机器人模拟按键、松键和反向，检查实际运动；不向策略提供仿真真值。"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from legged_gym.envs import register_tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wheelleg.diagnostics.motion_evaluation import MotionEvaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sim-device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--combined",
        action="store_true",
        help="Test simultaneous forward/backward and turning keys",
    )
    p.add_argument("--speed", type=float, default=0.2)
    p.add_argument("--turn-speed", type=float, default=0.6)
    p.add_argument("--motion-seconds", type=float, default=6.0)
    p.add_argument("--stop-seconds", type=float, default=5.0)
    args = p.parse_args()
    if (
        not all(
            math.isfinite(x) and x > 0
            for x in (
                args.speed,
                args.turn_speed,
                args.motion_seconds,
                args.stop_seconds,
            )
        )
        or min(args.motion_seconds, args.stop_seconds) < 3
    ):
        p.error("Use positive finite speeds and phases of at least three seconds")
    register_tasks()
    torch.set_num_threads(1)
    execution = get_args(
        "play",
        [
            "--task",
            "wheelleg_sensor_drive",
            "--checkpoint",
            str(args.checkpoint),
            "--headless",
            "--sim_device",
            args.sim_device,
            "--num_envs",
            "4",
            "--seed",
            str(args.seed),
        ],
    )
    cfg, train = task_registry.prepare_cfgs(execution.task, execution)
    if args.speed > max(
        abs(x) for x in cfg.commands.forward_speed_range
    ) or args.turn_speed > max(abs(x) for x in cfg.commands.yaw_speed_range):
        p.error("Requested evaluation speeds exceed the checkpoint training ranges")
    env, _ = task_registry.make_env(execution.task, args=execution, env_cfg=cfg)
    phases = []
    started = time.perf_counter()
    try:
        runner, _ = task_registry.make_alg_runner(
            env, execution.task, execution, train, log_root=None
        )
        policy = runner.get_inference_policy()
        pattern = [
            ("initial_stop", args.stop_seconds, 0),
            ("keys", args.motion_seconds, 1),
            ("release", args.stop_seconds, 0),
            ("reverse_keys", args.motion_seconds, -1),
            ("final_release", args.stop_seconds, 0),
        ]
        env.max_episode_length = sum(round(s / env.dt) for _, s, _ in pattern) + 1
        commands = env.commands.new_tensor(
            [
                [args.speed, 0, 0, 0.08],
                [-args.speed, 0, 0, 0.08],
                [0, 0, args.turn_speed, 0.08],
                [0, 0, -args.turn_speed, 0.08],
            ]
        )
        if args.combined:
            commands[:, 0] = commands.new_tensor(
                [args.speed, args.speed, -args.speed, -args.speed]
            )
            commands[:, 2] = commands.new_tensor(
                [args.turn_speed, -args.turn_speed, args.turn_speed, -args.turn_speed]
            )
        age = torch.zeros(4, device=env.device)
        longest = age.clone()
        torques = env.actions.new_zeros(6)
        closure = pitch = roll = 0.0
        failures = timeouts = 0
        with torch.inference_mode():
            for name, seconds, direction in pattern:
                target = commands.clone()
                target[:, :3] *= direction
                env.set_commands(target)
                measurements = [
                    MotionEvaluation(env.root_states[i : i + 1], env.dt)
                    for i in range(4)
                ]
                phase_failures = phase_timeouts = 0
                rate_sum = torch.zeros_like(age)
                settled_measurements = None
                samples = 0
                for step in range(round(seconds / env.dt)):
                    obs, _ = env.get_observations()
                    obs, _, reward, done, info, _ = env.step(policy(obs))
                    if not bool(
                        torch.isfinite(obs).all() & torch.isfinite(reward).all()
                    ):
                        raise RuntimeError("Non-finite motion evaluation")
                    for i, measurement in enumerate(measurements):
                        measurement.update(
                            info["pre_reset_root_states"][i : i + 1],
                            info["pre_reset_commands"][i : i + 1],
                            done[i : i + 1],
                            env.root_states[i : i + 1],
                        )
                    if settled_measurements is not None:
                        for i, measurement in enumerate(settled_measurements):
                            measurement.update(
                                info["pre_reset_root_states"][i : i + 1],
                                info["pre_reset_commands"][i : i + 1],
                                done[i : i + 1],
                                env.root_states[i : i + 1],
                            )
                    if (step + 1) * env.dt >= 2:
                        if settled_measurements is None:
                            # 松键后的前两秒包含合法减速，原地漂移从减速后计量。
                            settled_measurements = [
                                MotionEvaluation(
                                    env.root_states[i : i + 1], env.dt, settle_seconds=0
                                )
                                for i in range(4)
                            ]
                        rate_sum += info["pre_reset_ang_vel"][:, 2]
                        samples += 1
                    age += env.dt
                    longest = torch.maximum(longest, age)
                    age[done] = 0
                    phase_failures += int(info["physical_failures"].sum())
                    phase_timeouts += int(info["time_outs"].sum())
                    closure = max(closure, float(info["loop_error_max"]))
                    pitch = max(pitch, float(info["pre_reset_pitch"].abs().max()))
                    roll = max(roll, float(info["pre_reset_roll"].abs().max()))
                    torques = torch.maximum(
                        torques, info["pre_reset_peak_abs_torques"].max(0).values
                    )
                failures += phase_failures
                timeouts += phase_timeouts
                rows = []
                for i, m in enumerate(measurements):
                    motion = m.report()
                    settled_motion = settled_measurements[i].report()
                    path = settled_motion if direction == 0 else motion
                    requested = target[i].tolist()
                    turning = abs(requested[2]) > 1e-6
                    speed_limit = 0.02 if direction == 0 else 0.05
                    yaw_limit = 0.06 if direction == 0 else 0.12
                    checks = dict(
                        command_reached=bool(
                            torch.allclose(
                                info["pre_reset_commands"][i], target[i], atol=1e-4
                            )
                        ),
                        forward_tracking=motion["rms_forward_speed_error_m_s"]
                        <= speed_limit,
                        lateral_speed=motion["rms_lateral_speed_m_s"] <= 0.02,
                        yaw_tracking=motion["rms_yaw_rate_error_rad_s"] <= yaw_limit,
                    )
                    if not turning:
                        checks.update(
                            heading=path["segment_peak_heading_change_rad"] <= 0.1,
                            cross_track=path["segment_peak_cross_track_m"] <= 0.05,
                        )
                    if direction == 0:
                        checks["stopped_displacement"] = (
                            path["segment_peak_forward_displacement_m"] <= 0.05
                        )
                    rows.append(
                        dict(
                            environment=i,
                            requested_command=requested,
                            mean_yaw_rate_rad_s=float(rate_sum[i] / max(samples, 1)),
                            motion=motion,
                            settled_motion=settled_motion,
                            checks=checks,
                            passed=all(checks.values()),
                        )
                    )
                phase = dict(
                    name=name,
                    seconds=seconds,
                    physical_failures=phase_failures,
                    timeouts=phase_timeouts,
                    environments=rows,
                )
                phases.append(phase)
                print(
                    "phase={} failures={} tracking_pass={}".format(
                        name, phase_failures, all(r["passed"] for r in rows)
                    ),
                    flush=True,
                )
        report = dict(
            checkpoint=str(args.checkpoint.resolve()),
            seed=args.seed,
            command_mode="combined" if args.combined else "cardinal",
            control_dt=env.dt,
            num_envs=4,
            seconds_per_environment=sum(s for _, s, _ in pattern),
            longest_continuous_seconds_by_env=longest.tolist(),
            physical_failures=failures,
            timeouts=timeouts,
            pitch_peak_rad=pitch,
            roll_peak_rad=roll,
            closure_peak_m=closure,
            torque_peak_nm_by_joint=torques.tolist(),
            phases=phases,
            wall_seconds=time.perf_counter() - started,
        )
        report["motion_passed"] = (
            failures == timeouts == 0
            and max(pitch, roll) <= 0.25
            and closure <= 0.001
            and all(r["passed"] for phase in phases for r in phase["environments"])
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("motion_passed={}".format(report["motion_passed"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
