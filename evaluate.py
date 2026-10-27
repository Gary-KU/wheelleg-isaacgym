#!/usr/bin/env python3
"""Deterministic, headless policy evaluation with physical failures separated from timeouts."""

import json
import math
import time
import sys
from pathlib import Path

from legged_gym.compat.cli import common_parser, environment_kwargs, validate_common
from legged_gym.compat.runtime_config import explicit_argument_destinations, resolve_runtime_args


from legged_gym.envs.wheelleg.diagnostics.evaluation_metrics import (
    make_speed_scenarios,
    velocity_statistics,
    tracking_statistics,
    make_speed_sequence,
    dynamics_statistics,
    command_tracking_statistics,
    sequence_segment_at_step,
)


def main():
    parser = common_parser("Evaluate standing or commanded driving without a viewer", 8)
    parser.set_defaults(headless=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-steps", "--num_steps", type=int,
                        help="Default 800; in sequence mode must equal segment-steps times sequence length")
    parser.add_argument("--report", type=Path)
    speed_mode = parser.add_mutually_exclusive_group()
    speed_mode.add_argument("--test-speeds", nargs="+", type=float,
                           help="Evaluate groups of fixed forward speeds in parallel, e.g. -0.3 0 0.3")
    speed_mode.add_argument("--speed-sequence", nargs="+", type=float,
                           help="Sequential speeds applied to all environments without a requested reset, e.g. 0.3 0 -0.3 0")
    parser.add_argument("--segment-steps", type=int, default=200,
                        help="Control steps for each speed-sequence segment (default 200)")
    parser.add_argument("--yaw-sequence", nargs="+", type=float,
                        help="Yaw targets aligned with speed-sequence entries; otherwise --command yaw is retained")
    parser.add_argument("--sequence-repeats", type=int, default=1,
                        help="Repeat the speed/yaw pattern without resetting (default 1)")
    parser.add_argument("--disable-time-limit", action="store_true",
                        help="Keep physical terminations but disable automatic episode time-limit resets")
    parser.add_argument("--record-command-trajectory", action="store_true",
                        help="Include raw and filtered per-step command trajectories in JSON")
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    if args.random_commands:
        parser.error("Evaluation uses a fixed --command; omit --random-commands")
    try:
        command_rows, scenarios = make_speed_scenarios(args.num_envs, args.test_speeds, args.command)
        sequence = make_speed_sequence(args.speed_sequence, args.segment_steps, args.yaw_sequence,
                                       args.sequence_repeats, args.command[2])
    except ValueError as error:
        parser.error(str(error))
    if sequence:
        total_steps = sequence[-1]["end_step_exclusive"]
        if args.num_steps is not None and args.num_steps != total_steps:
            parser.error("--num-steps must equal sequence length times --segment-steps ({})".format(total_steps))
        args.num_steps = total_steps
    elif args.num_steps is None:
        args.num_steps = 800
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")

    import isaacgym  # noqa: F401; binary extension must precede Torch.
    import torch
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    try:
        args = resolve_runtime_args(args, explicit_argument_destinations(parser, argv), checkpoint, "evaluate")
    except ValueError as error:
        parser.error(str(error))
    validate_common(parser, args)
    # Leg target and numerical settings may have been restored from the policy.
    command_rows, scenarios = make_speed_scenarios(args.num_envs, args.test_speeds, args.command)

    # Isaac Gym must load its extension before PyTorch is imported.
    import isaacgym  # noqa: F401
    from isaacgym import gymtorch
    import torch
    from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot
    from legged_gym.envs.wheelleg.constants import TRAIN_CONFIG
    from legged_gym.learning.policy import make_policy, scalar
    from legged_gym.envs.wheelleg.commands import CommandGovernor

    kwargs = environment_kwargs(args)
    kwargs["headless"] = True
    env = WheelLegRobot(**kwargs)
    try:
        if args.disable_time_limit:
            env.max_episode_length = args.num_steps+1
        policy = make_policy(env.num_obs, env.num_actions, env.device,
                             checkpoint.get("training_config", {}).get("policy", TRAIN_CONFIG["policy"]))
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        observations, _ = env.reset()
        target_commands = torch.tensor(command_rows, dtype=torch.float32, device=env.device)
        governor = CommandGovernor(target_commands if scenarios else env.commands.clone(),
                                   linear_acceleration=args.linear_acceleration,
                                   yaw_acceleration=args.yaw_acceleration)
        target_vx = [row[0] for row in command_rows]
        velocity_history = []
        target_history = []
        raw_command_history, applied_command_history, yaw_history = [], [], []
        motion_history = {key: [] for key in ("roll", "pitch", "angular_xy", "torque", "torque_rate")}
        previous_torque = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        properties = env.gym.get_actor_dof_properties(env.envs[0], env.actors[0])
        torque_limits = [float(properties["effort"][j]) for j in env.action_indices.cpu().tolist()]
        contact_descriptor = env.gym.acquire_net_contact_force_tensor(env.sim)
        contact_forces = gymtorch.wrap_tensor(contact_descriptor).view(env.num_envs, env.num_bodies, 3)
        base_index = env.body_names.index("base_link")
        wheel_indices = [env.body_names.index(name) for name in ("L_lun", "R_lun")]
        diagnostic_history = {name: [] for name in (
            "height", "contact", "length", "theta", "valid", "wheel_height", "wheel_pose_valid")}
        robot_mass = sum(p.mass for p in env.gym.get_actor_rigid_body_properties(env.envs[0], env.actors[0]))
        gravity_z = abs(env.gym.get_sim_params(env.sim).gravity.z)
        terminal_diagnostics_available = True
        initial_base_xy = env.root_states[:, :2].clone()
        end_base_xy = initial_base_xy.clone()
        episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        episode_lengths = []
        failure_lengths = []
        timeout_lengths = []
        failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        timeouts = torch.zeros_like(failures)
        failure_history, timeout_history = [], []
        first_failure = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        peak_pitch = peak_tilt = peak_closure = 0.0
        finite_outputs = True
        completed_steps = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for step in range(args.num_steps):
                if sequence:
                    segment = sequence_segment_at_step(sequence, step)
                    target_commands[:, 0] = segment["target_vx_m_per_s"]
                    target_commands[:, 2] = segment["target_yaw_rate_rad_per_s"]
                # Fixed evaluation never uses the environment's random-training
                # governor. This is the same shared command path as keyboard play.
                applied_commands = governor.step(target_commands, env.dt)
                observations = env.set_commands(applied_commands)
                actions = policy.act_inference(observations)
                if not bool(torch.isfinite(actions).all()):
                    finite_outputs = False
                    break
                observations, _, rewards, dones, infos = env.step(actions)
                env.gym.refresh_net_contact_force_tensor(env.sim)
                contacts = contact_forces.clone()
                roots = infos.get("pre_reset_root_states")
                lengths = infos.get("pre_reset_leg_length")
                thetas = infos.get("pre_reset_leg_theta")
                valid_diagnostic = torch.ones_like(dones, dtype=torch.bool)
                if roots is None or lengths is None or thetas is None:
                    terminal_diagnostics_available = False
                    roots = env.root_states
                    lengths = torch.stack((env.length_l, env.length_r), dim=-1)
                    thetas = torch.stack((env.theta_l, env.theta_r), dim=-1)
                    # No terminal pose instrumentation: don't count a teleported
                    # reset pose as a real supported/unsupported body height.
                    valid_diagnostic = ~dones
                diagnostic_history["height"].append(roots[:, 2].clone())
                diagnostic_history["contact"].append(contacts)
                diagnostic_history["length"].append(lengths.clone())
                diagnostic_history["theta"].append(thetas.clone())
                diagnostic_history["valid"].append(valid_diagnostic.clone())
                diagnostic_history["wheel_height"].append(env.rb_states[:, wheel_indices, 2].clone())
                diagnostic_history["wheel_pose_valid"].append((~dones).clone())
                end_base_xy.copy_(roots[:, :2])
                episode_steps += 1
                completed_steps = step + 1
                # env.step auto-resets. Restore terminal observations for metrics so
                # falling does not appear to be an upright zero-velocity sample.
                measured = observations.clone()
                terminal_ids = infos.get("terminal_env_ids")
                if terminal_ids is not None:
                    measured[terminal_ids] = infos["terminal_observation"]
                timeout = infos.get("time_outs", torch.zeros_like(dones)).bool() & dones
                physical_failure = dones & ~timeout & ~infos.get("terrain_success", torch.zeros_like(dones))
                failure_history.append(physical_failure.clone())
                timeout_history.append(timeout.clone())
                failures += physical_failure.long()
                timeouts += timeout.long()
                new_first_failure = physical_failure & (first_failure < 0)
                first_failure[new_first_failure] = completed_steps
                for mask, output in ((dones, episode_lengths),
                                     (physical_failure, failure_lengths),
                                     (timeout, timeout_lengths)):
                    output.extend((episode_steps[mask].float()*env.dt).cpu().tolist())
                episode_steps[dones] = 0
                step_closure = scalar(infos["loop_error_max"])
                if not math.isfinite(step_closure):
                    finite_outputs = False
                    break
                peak_closure = max(peak_closure, step_closure)
                finite = bool(torch.isfinite(measured).all() & torch.isfinite(rewards).all())
                if not finite:
                    finite_outputs = False
                    break
                frame = measured[:, -getattr(env, "observation_frame_size", 31):]
                vx = frame[:, 0]/1.5
                velocity_history.append(vx.clone())
                target_history.append(target_commands[:, 0].clone())
                raw_command_history.append(target_commands.clone())
                applied_command_history.append(applied_commands.clone())
                yaw_history.append((frame[:, 5]/0.5).clone())
                pitch = torch.asin(frame[:, 6].clamp(-1, 1))
                roll = infos.get("pre_reset_roll", torch.atan2(-frame[:, 7], -frame[:, 8]))
                tilt = torch.acos((-frame[:, 8]).clamp(-1, 1))
                applied_torque = infos.get("pre_reset_applied_torques")
                if applied_torque is None:
                    applied_torque = infos.get("pre_reset_torques", env.torques)[:, env.action_indices]
                applied_torque = applied_torque.clone()
                prior_torque = infos.get("pre_reset_previous_torques", previous_torque)
                motion_history["roll"].append(roll.clone())
                motion_history["pitch"].append(pitch.clone())
                motion_history["angular_xy"].append((frame[:, 3:5]/0.5).clone())
                motion_history["torque"].append(applied_torque)
                motion_history["torque_rate"].append((applied_torque-prior_torque)/env.dt)
                previous_torque.copy_(applied_torque)
                previous_torque[dones] = 0.0
                if bool(dones.any()):
                    governor.reset(torch.where(dones.unsqueeze(-1), env.commands, governor.commands))
                peak_pitch = max(peak_pitch, float(pitch.abs().max()))
                peak_tilt = max(peak_tilt, float(tilt.max()))
                if completed_steps % 100 == 0:
                    recent_height = torch.stack(diagnostic_history["height"][-100:])
                    recent_contacts = torch.stack(diagnostic_history["contact"][-100:])
                    recent_base_contact = torch.linalg.vector_norm(recent_contacts[:, :, base_index], dim=-1) > 1.0
                    recent_length = torch.stack(diagnostic_history["length"][-100:])
                    print("evaluation_steps={} height_mean={:.4f}m height_min={:.4f}m base_contact={:.1f}% leg_length_mean={:.4f}m failures={} timeouts={}".format(
                        completed_steps, float(recent_height.mean()), float(recent_height.min()),
                        float(recent_base_contact.float().mean())*100, float(recent_length.mean()),
                        int(failures.sum()), int(timeouts.sum())), flush=True)

        def mean_or_none(values):
            return sum(values)/len(values) if values else None

        final_alive_seconds = (episode_steps.float()*env.dt).cpu().tolist()
        failure_steps = first_failure.cpu().tolist()
        velocity_rows = torch.stack(velocity_history).cpu().tolist() if velocity_history else []
        target_rows = torch.stack(target_history).cpu().tolist() if target_history else []
        raw_rows = torch.stack(raw_command_history).cpu().tolist() if raw_command_history else []
        applied_rows = torch.stack(applied_command_history).cpu().tolist() if applied_command_history else []
        yaw_rows = [[float(x) for x in row] for row in torch.stack(yaw_history).cpu().tolist()] if yaw_history else []
        velocity = tracking_statistics(velocity_rows, target_rows) if sequence else velocity_statistics(velocity_rows, target_vx)
        command_tracking = command_tracking_statistics(velocity_rows, yaw_rows, raw_rows, applied_rows)
        motion_tensors = {key: torch.stack(values) for key, values in motion_history.items()} if velocity_rows else {}

        def motion_summary(selection=None):
            if not motion_tensors:
                return None
            tensor = motion_tensors["roll"]
            mask = torch.ones_like(tensor, dtype=torch.bool) if selection is None else selection[:len(tensor)]
            return dynamics_statistics(
                **{key: value[mask].cpu().tolist() for key, value in motion_tensors.items()},
                torque_limits=torque_limits)

        def grouped_tracking(indices, start=0, end=None):
            def selected(rows):
                return [[row[index] for index in indices] for row in rows[start:end]]
            return command_tracking_statistics(selected(velocity_rows), selected(yaw_rows),
                                               selected(raw_rows), selected(applied_rows))

        environment_results = []
        for index in range(env.num_envs):
            mask = torch.zeros(len(velocity_rows), env.num_envs, dtype=torch.bool, device=env.device)
            mask[:, index] = True
            environment_results.append({"environment_index": index,
                "physical_failure_resets": int(failures[index]), "time_limit_resets": int(timeouts[index]),
                "command_tracking": grouped_tracking([index]), "motion_diagnostics": motion_summary(mask)})
        support = None
        scenario_results = []
        sequence_results = []
        if diagnostic_history["height"]:
            heights = torch.stack(diagnostic_history["height"])
            contacts = torch.stack(diagnostic_history["contact"])
            lengths = torch.stack(diagnostic_history["length"])
            thetas = torch.stack(diagnostic_history["theta"])
            valid = torch.stack(diagnostic_history["valid"])
            wheel_heights = torch.stack(diagnostic_history["wheel_height"])
            wheel_pose_valid = torch.stack(diagnostic_history["wheel_pose_valid"])
            normal_force = contacts[..., 2].clamp_min(0)
            force_norm = torch.linalg.vector_norm(contacts, dim=-1)
            base_contact = force_norm[:, :, base_index] > 1.0
            wheel_contact = force_norm[:, :, wheel_indices] > 1.0

            def supported_summary(selection):
                selected = valid & selection
                h = heights[selected]
                if not h.numel():
                    return None
                positive_vertical_total = float(normal_force.sum(-1)[selected].sum())
                base_vertical = float(normal_force[:, :, base_index][selected].sum())
                wheel_vertical = float(normal_force[:, :, wheel_indices].sum(-1)[selected].sum())
                nonwheel_indices = [i for i in range(env.num_bodies) if i not in wheel_indices]
                result = {
                    "samples": int(selected.sum()),
                    "base_height_min_m": float(h.min()),
                    "base_height_mean_m": float(h.mean()),
                    "base_height_p05_m": float(torch.quantile(h, 0.05)),
                    "base_height_max_m": float(h.max()),
                    "base_contact_fraction_over_1n": float(base_contact[selected].float().mean()),
                    "base_contact_force_peak_n": float(force_norm[:, :, base_index][selected].max()),
                    "base_positive_vertical_support_fraction": base_vertical/positive_vertical_total if positive_vertical_total > 1e-8 else None,
                    "wheel_positive_vertical_support_fraction": wheel_vertical/positive_vertical_total if positive_vertical_total > 1e-8 else None,
                    "wheel_contact_fraction_over_1n_left_right": wheel_contact[selected].float().mean(0).cpu().tolist(),
                    "both_wheels_contact_fraction_over_1n": float(wheel_contact[selected].all(-1).float().mean()),
                    "nonwheel_contact_fraction_any_over_1n": float((force_norm[:, :, nonwheel_indices][selected] > 1.0).any(-1).float().mean()),
                    "total_positive_vertical_contact_mean_n": positive_vertical_total/int(selected.sum()),
                    "wheel_positive_vertical_contact_mean_n": wheel_vertical/int(selected.sum()),
                    "robot_weight_n": robot_mass*gravity_z,
                    "wheel_mean_vertical_force_fraction_of_weight": (wheel_vertical/int(selected.sum()))/(robot_mass*gravity_z) if robot_mass*gravity_z > 1e-8 else None,
                    "geometric_leg_length_mean_m_left_right": lengths[selected].mean(0).cpu().tolist(),
                    "geometric_leg_length_min_m_left_right": lengths[selected].min(0).values.cpu().tolist(),
                    "geometric_leg_length_max_m_left_right": lengths[selected].max(0).values.cpu().tolist(),
                    "leg_theta_mean_rad_left_right": thetas[selected].mean(0).cpu().tolist(),
                    "leg_theta_abs_max_rad_left_right": thetas[selected].abs().max(0).values.cpu().tolist(),
                }
                wheel_selected = selected & wheel_pose_valid
                result["wheel_pose_samples_excluding_auto_resets"] = int(wheel_selected.sum())
                if wheel_selected.any():
                    result["wheel_center_world_z_mean_m_left_right"] = wheel_heights[wheel_selected].mean(0).cpu().tolist()
                    result["root_minus_wheel_vertical_mean_m_left_right"] = (heights.unsqueeze(-1)-wheel_heights)[wheel_selected].mean(0).cpu().tolist()
                return result

            all_selection = torch.ones_like(valid)
            tail_selection = torch.zeros_like(valid)
            tail_selection[max(0, len(heights)-100):] = True
            support = {
                "contact_force_source": "PhysX net_contact_force_tensor, refreshed after each control step; 1 N contact threshold",
                "contact_collection_mode": int(env.gym.get_sim_params(env.sim).physx.contact_collection),
                "terminal_pose_metrics_available": terminal_diagnostics_available,
                "excluded_terminal_pose_samples": int((~valid).sum()),
                "all_steps": supported_summary(all_selection),
                "last_100_control_steps": supported_summary(tail_selection),
                "initial_base_xy_m": initial_base_xy.cpu().tolist(),
                "end_base_xy_m": end_base_xy.cpu().tolist(),
            }
            for scenario in scenarios:
                indices = scenario["environment_indices"]
                group_selection = torch.zeros_like(valid)
                group_selection[:, indices] = True
                group_velocities = [[row[index] for index in indices] for row in velocity_rows]
                group_velocity = velocity_statistics(
                    group_velocities, [target_vx[index] for index in indices])
                scenario_results.append(dict(
                    scenario,
                    num_envs=len(indices),
                    command=[scenario["target_vx_m_per_s"], *args.command[1:]],
                    **group_velocity,
                    physical_failure_resets=int(failures[indices].sum()),
                    physical_failure_resets_per_env=failures[indices].cpu().tolist(),
                    time_limit_resets=int(timeouts[indices].sum()),
                    time_limit_resets_per_env=timeouts[indices].cpu().tolist(),
                    environments_without_physical_failure=int((failures[indices] == 0).sum()),
                    ground_support_all_steps=supported_summary(group_selection),
                    ground_support_last_100_control_steps=supported_summary(group_selection & tail_selection),
                    command_tracking=grouped_tracking(indices),
                    motion_diagnostics=motion_summary(group_selection),
                ))
            if sequence:
                failures_by_step = torch.stack(failure_history)
                timeouts_by_step = torch.stack(timeout_history)
                for segment in sequence:
                    start, end = segment["start_step"], segment["end_step_exclusive"]
                    segment_selection = torch.zeros_like(valid)
                    segment_selection[start:end] = True
                    segment_rows = velocity_rows[start:end]
                    segment_targets = target_rows[start:end]
                    segment_velocity = tracking_statistics(segment_rows, segment_targets)
                    tail_start = max(start, min(end, len(velocity_rows))-100)
                    tail_velocity = tracking_statistics(velocity_rows[tail_start:end], target_rows[tail_start:end])
                    segment_support_tail = torch.zeros_like(valid)
                    segment_support_tail[tail_start:end] = True
                    sequence_results.append(dict(
                        segment,
                        planned_control_steps=end-start,
                        completed_control_steps=max(0, min(end, completed_steps)-start),
                        planned_duration_s=(end-start)*env.dt,
                        command=[segment["target_vx_m_per_s"], args.command[1], segment["target_yaw_rate_rad_per_s"], args.command[3]],
                        **segment_velocity,
                        last_100_control_steps_velocity=tail_velocity,
                        physical_failure_resets=int(failures_by_step[start:end].sum()),
                        time_limit_resets=int(timeouts_by_step[start:end].sum()),
                        ground_support_all_segment=supported_summary(segment_selection),
                        ground_support_last_100_control_steps=supported_summary(segment_support_tail),
                        command_tracking=grouped_tracking(list(range(env.num_envs)), start, end),
                        motion_diagnostics=motion_summary(segment_selection),
                    ))
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_iteration": int(checkpoint.get("iter", 0)),
            "num_envs": env.num_envs,
            "requested_steps": args.num_steps,
            "completed_steps": completed_steps,
            "control_dt_s": env.dt,
            "simulated_seconds_per_env": completed_steps*env.dt,
            "wall_seconds": time.perf_counter()-started,
            "command": list(args.command) if not scenarios and not sequence else None,
            "deterministic_policy": True,
            "initial_state_perturbations": False,
            "physical_failure_resets": int(failures.sum()),
            "physical_failure_resets_per_env": failures.cpu().tolist(),
            "time_limit_resets": int(timeouts.sum()),
            "time_limit_resets_per_env": timeouts.cpu().tolist(),
            "completed_episode_mean_seconds": mean_or_none(episode_lengths),
            "failed_episode_mean_seconds": mean_or_none(failure_lengths),
            "time_limit_episode_mean_seconds": mean_or_none(timeout_lengths),
            "current_surviving_episode_seconds_per_env": final_alive_seconds,
            "first_failure_seconds_per_env": [x*env.dt if x >= 0 else None for x in failure_steps],
            "environments_without_physical_failure": int((failures == 0).sum()),
            "mean_vx_m_per_s": velocity["mean_vx_m_per_s"],
            "vx_tracking_rmse_m_per_s": velocity["vx_tracking_rmse_m_per_s"],
            "peak_abs_pitch_rad": peak_pitch,
            "peak_tilt_rad": peak_tilt,
            "peak_anchor_error_mm": peak_closure*1000 if math.isfinite(peak_closure) else None,
            "finite_outputs": finite_outputs,
            "ground_support_diagnostics": support,
            "command_filter": {"linear_acceleration_m_per_s2": args.linear_acceleration,
                               "yaw_acceleration_rad_per_s2": args.yaw_acceleration},
            "command_tracking": command_tracking,
            "environment_results": environment_results,
            "motion_diagnostics": {"torque_source": "Active DOF effort targets after clamping; not joint reaction forces",
                                   "torque_rate_dt_s": env.dt,
                                   "first_torque_rate_references_zero_initial_effort": True,
                                   "active_joint_names": [env.dof_names[i] for i in env.action_indices.cpu().tolist()],
                                   "effort_limits_nm": torque_limits,
                                   "overall": motion_summary()},
            "time_limit_disabled": args.disable_time_limit,
            "observation_frame_size": getattr(env, "observation_frame_size", 31),
            "approximate_loop": True,
            "interpretation": "Physical failures and time-limit resets are separate. This fixed-initial-state evaluation does not establish perturbation robustness or rigid-loop accuracy.",
        }
        if scenarios:
            result["commands_per_env"] = command_rows
            result["speed_scenarios"] = scenario_results
        if sequence:
            result["segment_steps"] = args.segment_steps
            result["speed_sequence"] = sequence_results
            result["requested_resets_at_speed_changes"] = False
        if args.record_command_trajectory:
            result["command_trajectory"] = {"control_dt_s": env.dt,
                                             "raw_targets": raw_rows, "applied_targets": applied_rows}
        content = json.dumps(result, indent=2, allow_nan=False)+"\n"
        print(content, flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(content, encoding="utf-8")
        if not finite_outputs:
            raise SystemExit("Non-finite policy evaluation output")
    finally:
        env.close()


if __name__ == "__main__":
    main()
