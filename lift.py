#!/usr/bin/env python3
"""原地交替抬轮：先站稳，再短暂抬轮，落轮恢复后才换边；无前进指令。"""

from datetime import datetime
import json
import hashlib
import math
from pathlib import Path
import sys
import time

from wheelleg_gym.cli import common_parser, environment_kwargs, validate_common
from wheelleg_gym.runtime_config import (
    explicit_argument_destinations,
    resolve_runtime_args,
    checkpoint_observation_size,
)
from wheelleg_gym.support_metrics import duration_steps


def main():
    parser = common_parser(__doc__, 1)
    parser.set_defaults(
        sim_device="cpu",
        physics_threads=0,
        wheel_control="heading",
        heading_kp=6.0,
        heading_kd=0.6,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--wheel-collision", choices=("mesh", "sphere_tread"), default="mesh"
    )
    parser.add_argument(
        "--wheel-observations", choices=("original", "sagittal"), default="original"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--keyboard", dest="keyboard", action="store_true")
    mode.add_argument("--auto", dest="keyboard", action="store_false")
    parser.set_defaults(keyboard=None)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--length", type=float, default=0.10)
    parser.add_argument("--shortening", type=float, default=0.035)
    parser.add_argument("--retract", type=float, default=0.02)
    parser.add_argument("--hold", type=float, default=0.06)
    parser.add_argument("--lower", type=float, default=0.12)
    parser.add_argument("--lean", type=float, default=0.012)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--recovery", type=float, default=0.8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--clearance", type=float, default=0.003)
    parser.add_argument("--airborne-seconds", type=float, default=0.04)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--record-trace", action="store_true")
    args = parser.parse_args()
    if args.keyboard is None:
        args.keyboard = not args.headless
    if args.keyboard and (args.headless or args.num_envs != 1):
        parser.error("Keyboard wheel lifts require a viewer and --num-envs 1")
    if args.seconds is None:
        args.seconds = 300.0 if args.keyboard else 60.0
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    try:
        steps = duration_steps(args.seconds, 0.02)
    except ValueError as error:
        parser.error(str(error))
    if args.random_commands or any(abs(x) > 1e-12 for x in args.command[:3]):
        parser.error("This task accepts zero velocity commands only")
    import isaacgym  # binary extension before torch
    import torch
    from wheelleg_gym.wheel_lift import (
        AlternatingWheelLift,
        WheelLiftEnv,
        sagittal_wheel_observations,
        WheelLiftKeyboard,
    )
    from wheelleg_gym.training import make_policy, viewer_is_closed

    torch.set_num_threads(1)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    if checkpoint_observation_size(checkpoint) != 198:
        parser.error("Wheel lift requires the 198-observation standing checkpoint")
    if args.profile is not None:
        parser.error("Wheel lift has its own task controller; omit --profile")
    explicit = explicit_argument_destinations(parser)
    if "terrain" in explicit and args.terrain != "plane":
        parser.error("The first wheel-lift task supports flat ground only")
    if "leg_control" in explicit and args.leg_control != "hip_pd":
        parser.error("Wheel lift requires hip_pd leg control")
    if "command" in explicit and abs(args.command[3] - args.length) > 1e-9:
        parser.error("Use --length for both initial and nominal leg length")
    if "initial_leg_length" in explicit and args.initial_leg_length != args.length:
        parser.error("Use --length for both initial and nominal leg length")
    explicit.update(("wheel_control", "heading_kp", "heading_kd"))
    try:
        args = resolve_runtime_args(args, explicit, checkpoint, "play")
    except ValueError as error:
        parser.error(str(error))
    validate_common(parser, args)
    kwargs = environment_kwargs(args)
    kwargs.update(
        terrain="plane",
        observe_terrain=False,
        stair_curriculum=False,
        random_commands=False,
        command=(0.0, 0.0, 0.0, args.length),
        initial_leg_length=args.length,
        leg_control="hip_pd",
        command_resample_seconds=0.0,
    )
    root = Path(__file__).resolve().parent
    source_sha256 = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "lift.py",
            "lift.sh",
            "wheelleg_gym/wheel_lift.py",
            "wheelleg_gym/leg_control.py",
            "wheelleg_gym/env.py",
        )
    }
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    env = WheelLiftEnv(wheel_collision=args.wheel_collision, **kwargs)
    report_path = args.report or Path(__file__).resolve().parent / "validation" / (
        "wheel_lift_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    try:
        try:
            controller = AlternatingWheelLift(
                env,
                **{
                    key: getattr(args, key)
                    for key in (
                        "length",
                        "shortening",
                        "retract",
                        "hold",
                        "lower",
                        "settle",
                        "recovery",
                        "cycles",
                        "clearance",
                        "lean",
                    )
                },
                airborne_seconds=args.airborne_seconds,
                interactive=args.keyboard
            )
        except ValueError as error:
            parser.error(str(error))
        env.max_episode_length = steps + 1
        policy = make_policy(
            env.num_obs,
            env.num_actions,
            env.device,
            checkpoint.get("training_config", {}).get("policy"),
        )
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        observations, _ = env.reset()
        keyboard = WheelLiftKeyboard(controller) if args.keyboard else None
        if keyboard is not None:
            keyboard.bind(env.gym, env.viewer, isaacgym.gymapi)
            print(
                "键盘抬轮：先点击仿真窗口；N=下一侧（动作中不排队），空格=连续/落稳后暂停。方向键不用于本入口。",
                flush=True,
            )
        started = time.perf_counter()
        failures = timeouts = completed = event_cursor = 0
        trace = []
        finite = True
        pitch_peak = roll_peak = loop_peak = tread_tilt_peak = 0.0
        print(
            "Existing RL wheel balance + independent hip PD/support targets; no new training.",
            flush=True,
        )
        print(
            "Only measured wheel clearance AND loss of contact count as lift; no fixed base or body support.",
            flush=True,
        )
        with torch.inference_mode():
            for step in range(steps):
                if viewer_is_closed(env):
                    break
                if keyboard is not None:
                    keyboard.process_events(
                        env.gym.query_viewer_action_events(env.viewer)
                    )
                observations = controller.before_step()
                wheel_observations = (
                    sagittal_wheel_observations(observations)
                    if args.wheel_observations == "sagittal"
                    else observations
                )
                actions = policy.act_inference(wheel_observations).clone()
                actions[:, :4] = 0.0
                if not bool(torch.isfinite(actions).all()):
                    finite = False
                    break
                observations, _, _, dones, info = env.step(actions)
                completed = step + 1
                failures += int(info["physical_failures"].sum())
                timeouts += int(info["time_outs"].sum())
                if not all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        observations,
                        info["pre_reset_root_states"],
                        info["lift_clearance_m"],
                        info["lift_wheel_contact_norm_n"],
                        info["pre_reset_peak_abs_torques"],
                    )
                ):
                    finite = False
                    break
                controller.after_step(info, dones)
                pitch_peak = max(pitch_peak, float(info["pre_reset_pitch"].abs().max()))
                roll_peak = max(roll_peak, float(info["pre_reset_roll"].abs().max()))
                loop_peak = max(loop_peak, float(info["loop_error"].max()))
                tread_tilt_peak = max(
                    tread_tilt_peak, float(info["lift_tread_axis_tilt_rad"].max())
                )
                for event in controller.events[event_cursor:]:
                    print(json.dumps(event), flush=True)
                event_cursor = len(controller.events)
                if args.record_trace:
                    for i in range(env.num_envs):
                        trace.append(
                            dict(
                                step=completed,
                                time_s=completed * env.dt,
                                env=i,
                                stage=controller.states[i]["mode"],
                                side=controller.states[i]["side"],
                                target_lengths_m=env.lift_lengths[i].cpu().tolist(),
                                target_support_force_n=env.lift_forces[i]
                                .cpu()
                                .tolist(),
                                clearance_m=info["lift_clearance_m"][i].cpu().tolist(),
                                contact_norm_n=info["lift_wheel_contact_norm_n"][i]
                                .cpu()
                                .tolist(),
                                normal_force_n=info["lift_wheel_normal_force_n"][i]
                                .cpu()
                                .tolist(),
                                nonwheel_contact=bool(info["lift_nonwheel_contact"][i]),
                                pitch_rad=float(info["pre_reset_pitch"][i]),
                                roll_rad=float(info["pre_reset_roll"][i]),
                                yaw_rad=float(env.yaw[i]),
                                angular_velocity_body_rad_per_s=info[
                                    "pre_reset_ang_vel"
                                ][i]
                                .cpu()
                                .tolist(),
                                linear_velocity_world_m_per_s=info[
                                    "pre_reset_root_states"
                                ][i, 7:10]
                                .cpu()
                                .tolist(),
                                wheel_velocity_correction_m_per_s=float(
                                    controller.parking_velocity[i]
                                ),
                                stable_seconds=controller.states[i]["stable"],
                                root_xy_m=info["pre_reset_root_states"][i, :2]
                                .cpu()
                                .tolist(),
                                torque_peak_nm=info["pre_reset_peak_abs_torques"][i]
                                .cpu()
                                .tolist(),
                                physical_failure=bool(info["physical_failures"][i]),
                            )
                        )
                if completed % 250 == 0:
                    print(
                        "lift_seconds={:.1f} stages={} confirmed={}".format(
                            completed * env.dt,
                            [s["mode"] for s in controller.states],
                            controller.summary(failures)["confirmed_lifts"],
                        ),
                        flush=True,
                    )
        summary = controller.summary(failures)
        collision_scope_passed = (
            args.wheel_collision != "sphere_tread"
            or tread_tilt_peak <= math.asin(0.02 / 0.0625)
        )
        summary["engineering_screen_passed"] &= (
            finite and completed == steps and timeouts == 0 and collision_scope_passed
        )
        if not summary["engineering_screen_passed"]:
            summary["status"] = (
                "interactive_session"
                if args.keyboard and finite and failures == 0
                else "needs_improvement"
            )
        report = dict(
            summary=summary,
            resolved_environment=kwargs,
            protocol={
                key: getattr(args, key)
                for key in (
                    "seconds",
                    "length",
                    "shortening",
                    "retract",
                    "hold",
                    "lower",
                    "settle",
                    "recovery",
                    "cycles",
                    "clearance",
                    "lean",
                    "airborne_seconds",
                )
            },
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_sha256,
            source_sha256=source_sha256,
            requested_steps=steps,
            completed_steps=completed,
            finite_outputs=finite,
            time_limit_resets=timeouts,
            wall_seconds=time.perf_counter() - started,
            pitch_peak_deg=math.degrees(pitch_peak),
            roll_peak_deg=math.degrees(roll_peak),
            closure_error_peak_m=loop_peak,
            tread_axis_tilt_peak_deg=math.degrees(tread_tilt_peak),
            collision_scope_passed=collision_scope_passed,
            events=controller.events,
            new_training_updates=0,
            keyboard_enabled=args.keyboard,
            wheel_policy_observation_adapter=args.wheel_observations,
            wheel_collision=args.wheel_collision,
            user_velocity_command=[0.0, 0.0, 0.0],
            fixed_base=False,
            external_body_support=False,
            root_state_writes_outside_resets=False,
            contact_metric_semantics={
                "swing_force": "maximum contact-force norm across every physics substep of a control interval",
                "support_force": "mean vertical contact force across every physics substep",
                "airborne": "clearance >= threshold AND swing max-contact <1N AND support mean-force >5N; only active lift phase counts",
            },
        )
        if args.record_trace:
            report["trace"] = trace
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(report_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("Lift report: " + str(report_path.resolve()), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
