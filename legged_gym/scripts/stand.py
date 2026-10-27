#!/usr/bin/env python3
"""Space: request a jump without readiness gates; X: stop; arrows: drive; R: reset."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-jumps", type=int, default=0)
    parser.add_argument(
        "--jump-speed",
        type=float,
        default=0.0,
        help="Headless validation: steady straight speed (m/s); stop for the final 8 s",
    )
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--sim-device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=float, default=0.10)
    parser.add_argument("--physics-solver", choices=("pgs", "tgs"))
    parser.add_argument("--wheel-friction", type=float)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "logs/standing_support_20261022_093823/input/standing_hold.pt",
    )
    parser.add_argument(
        "--wheel-collision", choices=("sphere_tread", "mesh"), default="sphere_tread"
    )
    parser.add_argument(
        "--no-parking",
        action="store_true",
        help="Comparison only: disable idle position feedback",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--record-trace", action="store_true")
    args = parser.parse_args()
    if (
        not 0 < args.seconds <= 3600
        or args.auto_jumps < 0
        or args.num_envs < 1
        or not 0 <= args.seed < 2**32
    ):
        parser.error("Use positive finite duration/counts and a valid seed")
    if not math.isfinite(args.jump_speed) or abs(args.jump_speed) > 0.6:
        parser.error("Jump validation speed must be within -0.6..0.6 m/s")
    if args.jump_speed and (not args.headless or args.seconds < 16):
        parser.error("--jump-speed requires --headless and at least 16 seconds")
    if not args.headless and args.num_envs != 1:
        parser.error("Keyboard demonstration requires one robot")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    if not 0.08 <= args.length <= 0.14:
        parser.error("Standing leg length must be within 8..14 cm")
    if args.wheel_friction is not None and (
        not math.isfinite(args.wheel_friction) or args.wheel_friction < 0
    ):
        parser.error("Wheel friction must be finite and nonnegative")

    import isaacgym  # binary extension before torch
    import torch
    from legged_gym.envs.wheelleg.jump_config import WheelLeggedJumpCfg
    from legged_gym.envs.wheelleg.wheelleg import environment_kwargs
    from legged_gym.envs.wheelleg.control.jump import JumpController, JumpKeyboard
    from legged_gym.envs.wheelleg.control.wheel_lift import WheelLiftEnv
    from legged_gym.learning.policy import make_policy, viewer_is_closed
    from legged_gym.utils.helpers import class_to_dict

    torch.set_num_threads(1)
    cfg = WheelLeggedJumpCfg()
    cfg.env.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.env.initial_leg_length = args.length
    cfg.commands.command = (0.0, 0.0, 0.0, args.length)
    if args.physics_solver is not None:
        cfg.sim.physics_solver = args.physics_solver
    if args.wheel_friction is not None:
        cfg.asset.wheel_friction = args.wheel_friction
    if args.no_parking:
        cfg.parking.position_kp = cfg.parking.velocity_kd = 0.0
    state = torch.load(str(args.checkpoint), map_location="cpu")
    if state["model_state_dict"]["actor.0.weight"].shape[1] != 198:
        parser.error("This controller requires the 198-input standing wheel policy")
    env = WheelLiftEnv(
        wheel_collision=args.wheel_collision,
        sim_device=args.sim_device,
        headless=args.headless,
        **environment_kwargs(cfg)
    )
    report = {}
    try:
        steps = round(args.seconds / env.dt)
        if steps < 1:
            parser.error("Duration must include at least one control step")
        env.max_episode_length = steps + 1
        policy = make_policy(
            env.num_obs, env.num_actions, env.device, state["training_config"]["policy"]
        )
        policy.load_state_dict(state["model_state_dict"], strict=True)
        policy.eval()
        env.reset()
        controller = JumpController(env, cfg)
        keyboard = None
        if not args.headless:
            keyboard = JumpKeyboard(
                speed=0.6,
                turn_speed=2.0,
                leg_length=args.length,
                leg_range=(0.08, 0.14),
            )
            keyboard.bind(env.gym, env.viewer, isaacgym.gymapi)
            print(
                "点击仿真窗口：空格=起跳，无需等稳；方向键=驾驶；X=停；Q/E=调腿长；R=重置。",
                flush=True,
            )
            print(
                "加速、转向也能起跳；着地后可再次按空格，不用等恢复。预蹲/蹬伸/腾空中不重启动作、不排队。旧站立权重+显式跳跃控制，没有新训练。",
                flush=True,
            )
        failures = timeouts = manual_resets = event_cursor = completed = 0
        closure_peak = torque_peak = 0.0
        nonwheel_contact_steps = 0
        trace, standing_samples = [], []
        finite = True
        started = time.perf_counter()
        last_finished = -100.0
        last_driving = -100.0
        with torch.inference_mode():
            for step in range(steps):
                if viewer_is_closed(env):
                    break
                speed = (
                    args.jump_speed
                    if 2.0 <= controller.time < args.seconds - 8.0
                    else 0.0
                )
                user = [speed, 0.0, 0.0, args.length]
                jump_key = False
                if keyboard is not None:
                    if keyboard.process_events(
                        env.gym.query_viewer_action_events(env.viewer)
                    ):
                        for i, state_ in enumerate(controller.states):
                            if state_["attempt"] is not None:
                                controller._finish(i, "manual_reset")
                        env.reset()
                        controller.reset()
                        keyboard.reset_leg_length(args.length)
                        keyboard.pressed.clear()
                        keyboard.jump_requested = False
                        manual_resets += 1
                    # Keep the selected standing height current even during
                    # a jump; the controller defers applying it until touchdown.
                    keyboard.advance_leg_length(env.dt)
                    user = keyboard.command()
                    jump_key = keyboard.consume_jump()
                obs = controller.before_step(user)
                finished_count = len(controller.attempts) // env.num_envs
                ready = all(
                    s["mode"] == "idle" and s["stable"] >= cfg.jump.stable_seconds
                    for s in controller.states
                )
                auto = (
                    finished_count < args.auto_jumps
                    and ready
                    and controller.time - last_finished >= 2.0
                    and (
                        not args.jump_speed or (controller.time >= 4.0 and speed != 0.0)
                    )
                )
                if jump_key or auto:
                    controller.request_jump()
                actions = controller.policy_actions(policy, obs)
                _, _, rewards, dones, info = env.step(actions)
                completed += 1
                failures += int(info["physical_failures"].sum())
                timeouts += int(info["time_outs"].sum())
                closure_peak = max(closure_peak, float(info["loop_error_max"]))
                torque_peak = max(
                    torque_peak, float(info["pre_reset_peak_abs_torques"].max())
                )
                nonwheel_contact_steps += int(info["lift_nonwheel_contact"].sum())
                controller.after_step(info, dones)
                values = (
                    env.obs_buf,
                    rewards,
                    info["pre_reset_root_states"],
                    info["lift_clearance_m"],
                    info["pre_reset_peak_abs_torques"],
                )
                if not all(bool(torch.isfinite(value).all()) for value in values):
                    finite = False
                    break
                for event in controller.events[event_cursor:]:
                    if keyboard is None:
                        print(json.dumps(event), flush=True)
                    else:
                        messages = {
                            "jump_started": "预蹲、起跳。",
                            "touchdown": "已接触地面，腿长与驾驶指令已恢复。",
                            "jump_completed": "跳跃动作完成。",
                            "jump_ignored": "正在预蹲、蹬伸或腾空；着地后可再次按空格。",
                            "flight_timeout": "未检测到完整落地，正在返回支撑姿态。",
                        }
                        if event["event"] == "jump_finished":
                            message = (
                                "上次落地恢复已由新起跳中断。"
                                if event["failure"] == "restarted_before_recovery"
                                else (
                                    "落地稳定性已验证。"
                                    if event["success"]
                                    else "本次未通过稳定性检查，仍可再次起跳；R 可复位。"
                                )
                            )
                        else:
                            message = messages.get(event["event"], event["event"])
                        print(message, flush=True)
                    if event["event"] == "jump_finished":
                        last_finished = controller.time
                event_cursor = len(controller.events)
                if any(abs(v) > 1e-4 for v in user[:3]):
                    last_driving = controller.time
                row = dict(
                    requested_commands=list(user),
                    applied_commands=env.commands.cpu().tolist(),
                    time_s=controller.time,
                    modes=[s["mode"] for s in controller.states],
                    root=info["pre_reset_root_states"][:, :3].cpu().tolist(),
                    pitch=info["pre_reset_pitch"].cpu().tolist(),
                    roll=info["pre_reset_roll"].cpu().tolist(),
                    yaw=env.yaw.cpu().tolist(),
                    velocity=info["pre_reset_root_states"][:, 7:10].cpu().tolist(),
                    wheel_speed=env.dof_state[:, env.action_indices[4:], 1]
                    .cpu()
                    .tolist(),
                    wheel_position=env.dof_state[:, env.action_indices[4:], 0]
                    .cpu()
                    .tolist(),
                    clearance=info["lift_clearance_m"].cpu().tolist(),
                    wheel_contact=info["lift_wheel_contact_norm_n"].cpu().tolist(),
                    peak_torques=info["pre_reset_peak_abs_torques"].cpu().tolist(),
                )
                if args.record_trace:
                    trace.append(row)
                if (
                    controller.time >= 2.0
                    and controller.time - last_finished >= 2.0
                    and controller.time - last_driving >= 2.0
                    and all(s["mode"] == "idle" for s in controller.states)
                    and not any(abs(v) > 1e-4 for v in user[:3])
                ):
                    standing_samples.append(row)
        metrics = {}
        if standing_samples:
            wheel = torch.tensor([s["wheel_speed"] for s in standing_samples])
            pitch = torch.tensor([s["pitch"] for s in standing_samples])
            roll = torch.tensor([s["roll"] for s in standing_samples])
            xy = torch.tensor([s["root"] for s in standing_samples])[..., :2]
            metrics = dict(
                samples=len(standing_samples),
                seconds=len(standing_samples) * env.dt,
                wheel_rms_rad_s=float(wheel.square().mean().sqrt()),
                wheel_peak_rad_s=float(wheel.abs().max()),
                pitch_rms_rad=float(pitch.square().mean().sqrt()),
                roll_rms_rad=float(roll.square().mean().sqrt()),
                xy_span_m=(xy.max(0).values - xy.min(0).values).tolist(),
            )
        successful = sum(a["success"] for a in controller.attempts)
        final_stable = all(
            s["mode"] == "idle" and s["stable"] >= 2.0 for s in controller.states
        )
        final_offset = env.root_states[:, :2] - controller.anchor_xy
        final_stable = (
            final_stable
            and bool((torch.linalg.vector_norm(final_offset, dim=-1) < 0.03).all())
            and bool(
                (
                    torch.linalg.vector_norm(env.root_states[:, 7:9], dim=-1) < 0.005
                ).all()
            )
        )
        passed = (
            finite
            and failures == 0
            and timeouts == 0
            and nonwheel_contact_steps == 0
            and final_stable
            and successful >= args.auto_jumps * env.num_envs
            and all(a["success"] for a in controller.attempts)
        )
        report = dict(
            status="passed" if passed else "incomplete_or_needs_improvement",
            scope="Standing and straight rolling flat-ground jumps with existing RL wheels and explicit leg control; no new RL training.",
            completed_steps=completed,
            environment_seconds=completed * env.dt,
            wall_seconds=time.perf_counter() - started,
            num_envs=env.num_envs,
            finite_outputs=finite,
            physical_failure_resets=failures,
            time_limit_resets=timeouts,
            closure_error_peak_m=closure_peak,
            active_joint_torque_peak_nm=torque_peak,
            nonwheel_contact_steps=nonwheel_contact_steps,
            manual_resets=manual_resets,
            automatic_jumps_per_env=args.auto_jumps,
            validation_speed_m_s=args.jump_speed,
            successful_jumps=successful,
            attempts=controller.attempts,
            unfinished_attempts=sum(
                s["attempt"] is not None for s in controller.states
            ),
            final_modes=[s["mode"] for s in controller.states],
            final_stable=final_stable,
            final_offset_xy_m=final_offset.cpu().tolist(),
            standing_metrics=metrics,
            collision_model=args.wheel_collision,
            fixed_base=False,
            external_body_support=False,
            root_state_writes_outside_resets=False,
            joint_torque_limit_nm=9.0,
            closure_model="external_force_compliant_approximation",
            config=class_to_dict(cfg),
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            keyboard_enabled=keyboard is not None,
        )
        if args.record_trace:
            report["trace"] = trace
        target = args.report or ROOT / "validation/stand_latest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("trace", "config")},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        print("Report: " + str(target), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
