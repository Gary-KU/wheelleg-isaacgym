#!/usr/bin/env python3
"""Evaluate actual stair-course crossings, separate from falls and time limits."""

import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legged_gym.compat.cli import common_parser, environment_kwargs, validate_common
from legged_gym.envs.wheelleg.constants import DEFAULT_CONTROL_DT
from legged_gym.compat.runtime_config import explicit_argument_destinations, resolve_runtime_args
from legged_gym.envs.wheelleg.diagnostics.support_metrics import duration_steps


def summarize_stairs(samples, num_envs):
    """Summarize terminal-safe physical measurements; reward is not success."""
    if not samples:
        return None
    rms = lambda values: math.sqrt(math.fsum(value*value for value in values)/len(values))
    environments = []
    reason_names = sorted({key for row in samples for key in row.get("termination_reasons", {})})
    for index in range(num_envs):
        rows = [row for row in samples if row["environment_index"] == index]
        if not rows:
            continue
        successes = sum(row["course_success_verified"] for row in rows)
        reported_successes = sum(row["course_success_reported"] for row in rows)
        failures = sum(row["physical_failure"] for row in rows)
        timeouts = sum(row["timeout"] for row in rows)
        attempts = reported_successes+failures+timeouts
        environments.append({
            "environment_index": index,
            "samples": len(rows),
            "course_crossings": successes,
            "reported_successes": reported_successes,
            "unverified_successes": sum(row["course_success_reported"] and not row["course_success_verified"] for row in rows),
            "physical_failures": failures, "time_limit_resets": timeouts,
            "completed_attempts": attempts,
            "course_success_fraction_completed_attempts": successes/attempts if attempts else None,
            "max_progress_m": max(row["progress_m"] for row in rows),
            "max_best_progress_m": max(row["best_progress_m"] for row in rows),
            "highest_ground_height_m": max(row["ground_height_m"] for row in rows),
            "base_height_min_m": min(row["root_state"][2] for row in rows),
            "base_height_max_m": max(row["root_state"][2] for row in rows),
            "final_progress_m": rows[-1]["progress_m"],
            "final_sample_terminated": any(rows[-1][key] for key in ("course_success_reported", "physical_failure", "timeout")),
        })
    successes = sum(row["course_crossings"] for row in environments)
    failures = sum(row["physical_failures"] for row in environments)
    timeouts = sum(row["time_limit_resets"] for row in environments)
    complete = sum(row["completed_attempts"] for row in environments)
    return {
        "samples": len(samples), "num_envs_with_samples": len(environments),
        "course_crossings": successes,
        "reported_successes": sum(row["reported_successes"] for row in environments),
        "unverified_successes": sum(row["unverified_successes"] for row in environments),
        "physical_failures": failures, "time_limit_resets": timeouts,
        "completed_attempts": complete,
        "course_success_fraction_completed_attempts": successes/complete if complete else None,
        "environments_with_course_crossing": sum(row["course_crossings"] > 0 for row in environments),
        "max_progress_m": max(row["progress_m"] for row in samples),
        "highest_ground_height_m": max(row["ground_height_m"] for row in samples),
        "forward_velocity_body_mean_m_per_s": math.fsum(row["forward_velocity_body_m_per_s"] for row in samples)/len(samples),
        "forward_velocity_tracking_rmse_m_per_s": rms([row["forward_velocity_body_m_per_s"]-row["target_vx_m_per_s"] for row in samples]),
        "pitch_rms_rad": rms([row["pitch_rad"] for row in samples]),
        "roll_rms_rad": rms([row["roll_rad"] for row in samples]),
        "pitch_abs_peak_rad": max(abs(row["pitch_rad"]) for row in samples),
        "roll_abs_peak_rad": max(abs(row["roll_rad"]) for row in samples),
        "leg_length_mean_m_left_right": [math.fsum(row["leg_lengths_m"][j] for row in samples)/len(samples) for j in range(2)],
        "leg_length_tracking_rmse_m_left_right": [rms([row["leg_lengths_m"][j]-row["target_leg_length_m"] for row in samples]) for j in range(2)],
        "passive_limit_margin_min_rad": min(min(row["passive_limit_margin_rad"]) for row in samples),
        "passive_near_limit_fraction": sum(min(row["passive_limit_margin_rad"]) <= .01 for row in samples)/len(samples),
        "applied_torque_endpoint_rms_nm_per_joint": [rms([row["applied_torques_nm"][j] for row in samples]) for j in range(6)],
        "applied_torque_substep_peak_nm_per_joint": [max(row["peak_abs_torques_nm"][j] for row in samples) for j in range(6)],
        "applied_torque_substep_saturation_fraction_per_joint": [math.fsum(row["torque_saturation_fraction"][j] for row in samples)/len(samples) for j in range(6)],
        "closure_error_peak_m": max(row["closure_error_max_m"] for row in samples),
        "both_wheels_contact_fraction_over_1n": sum(all(row["wheel_contacts_over_1n"]) for row in samples)/len(samples),
        "nonwheel_contact_fraction_over_1n": sum(row["nonwheel_contact_over_1n"] for row in samples)/len(samples),
        "termination_reason_counts": {name: sum(row["physical_failure"] and row.get("termination_reasons", {}).get(name, False)
                                                 for row in samples) for name in reason_names},
        "per_environment": environments,
    }


def assess_stair_climbing(summary, num_envs, requested_steps, completed_steps, finite_outputs):
    metric = summary or {}
    checks = {
        "all_requested_steps": requested_steps == completed_steps,
        "all_environment_samples_recorded": metric.get("samples") == completed_steps*num_envs,
        "finite_outputs": finite_outputs is True,
        "all_environments_actually_crossed_course": metric.get("environments_with_course_crossing") == num_envs,
        "no_unverified_success_events": metric.get("unverified_successes") == 0,
        "no_physical_failures": metric.get("physical_failures") == 0,
        "no_time_limit_resets": metric.get("time_limit_resets") == 0,
    }
    return {
        "stair_climbing_screen_passed": all(checks.values()),
        "checks": checks,
        "quality_status": "fixed_level_course_crossings_observed" if all(checks.values()) else "needs_improvement",
        "interpretation": "A pass requires real course crossings in every tested environment. Survival, reward, or partial progress alone never count as stair-climbing success. Fixed-level finite evaluation does not establish general stair robustness.",
    }


def main():
    parser = common_parser(__doc__, 4)
    parser.set_defaults(sim_device="cpu", headless=True, terrain="stairs", stair_curriculum=False, stair_min_level=0,
                        command=[.25, 0., 0., .14])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=26., help="Simulated seconds per environment, in multiples of 0.02")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--record-trace", action="store_true")
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    try:
        steps = duration_steps(args.seconds, DEFAULT_CONTROL_DT)
    except ValueError as error:
        parser.error(str(error))

    import isaacgym  # noqa: F401; binary extension must precede Torch.
    import torch
    from legged_gym.envs.wheelleg.model.constraints import quat_rotate_inverse
    from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot
    from legged_gym.envs.wheelleg.constants import TRAIN_CONFIG
    from legged_gym.learning.policy import make_policy

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    explicit = explicit_argument_destinations(parser, argv)
    # This entry point always evaluates a fixed requested stair level and
    # forward task, never the source training curriculum or inference zero-vx.
    explicit.update(("terrain", "stair_curriculum", "stair_min_level", "stair_level", "command"))
    try:
        args = resolve_runtime_args(args, explicit, checkpoint, "evaluate")
    except ValueError as error:
        parser.error(str(error))
    args.terrain, args.stair_curriculum, args.headless = "stairs", False, True
    validate_common(parser, args)
    if args.random_commands or args.command[0] <= 0 or any(abs(value) > 1e-12 for value in args.command[1:3]):
        parser.error("Stair evaluation requires fixed forward --command VX 0 0 LENGTH with VX > 0")
    if not args.observe_terrain or not args.observe_leg_lengths:
        parser.error("Stair evaluation requires a terrain-aware 252-D checkpoint")
    kwargs = environment_kwargs(args)
    env = WheelLegRobot(**kwargs)
    started = time.perf_counter()
    try:
        # Preserve the configured training time limit; distinguish resets from
        # actual failures and successful course endings in all statistics.
        policy = make_policy(env.num_obs, env.num_actions, env.device,
                             checkpoint.get("training_config", {}).get("policy", TRAIN_CONFIG["policy"]))
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        env.reset()
        required = (
            "terrain_success", "physical_failures", "time_outs", "terrain_level", "terrain_progress",
            "terrain_best_progress", "terrain_ground_height", "pre_reset_root_states", "pre_reset_leg_length",
            "pre_reset_pitch", "pre_reset_roll", "pre_reset_wheel_states", "pre_reset_contact_forces",
            "pre_reset_applied_torques", "pre_reset_peak_abs_torques", "pre_reset_torque_saturation_fraction",
            "pre_reset_passive_limit_margin", "loop_error",
        )
        wheel_indices = env.wheel_indices.cpu().tolist()
        nonwheel_indices = [j for j in range(env.num_bodies) if j not in wheel_indices]
        samples, success_events = [], []
        event_counts = {key: [0]*env.num_envs for key in ("reported_successes", "physical_failures", "time_limit_resets")}
        termination_reason_counts = {}
        episode_peak_ground = [0.]*env.num_envs
        episode_steps = [0]*env.num_envs
        completed_steps, finite_outputs, nonfinite_step = 0, True, None
        with torch.inference_mode():
            for step in range(steps):
                observations = env.set_commands(args.command)
                actions = policy.act_inference(observations)
                if not bool(torch.isfinite(actions).all()):
                    finite_outputs, nonfinite_step = False, step+1
                    break
                observations, _, rewards, dones, info = env.step(actions)
                completed_steps = step+1
                missing = [key for key in required if key not in info]
                if missing:
                    raise RuntimeError("Environment lacks terminal-safe stair diagnostics: {}".format(", ".join(missing)))
                if "termination_reasons" not in info:
                    raise RuntimeError("Environment lacks terminal-safe termination reasons")
                reason_flags = {name: flags.bool().cpu().tolist() for name, flags in info["termination_reasons"].items()}
                # Boolean terminal events remain countable even when a failed
                # step contains non-finite physical state and cannot be pooled.
                events = torch.stack((info["terrain_success"].bool(), info["physical_failures"].bool(),
                                      info["time_outs"].bool()), dim=-1).cpu().tolist()
                for index, (success, failed, timeout) in enumerate(events):
                    event_counts["reported_successes"][index] += int(success)
                    event_counts["physical_failures"][index] += int(failed)
                    event_counts["time_limit_resets"][index] += int(timeout and not success and not failed)
                    for name, flags in reason_flags.items():
                        termination_reason_counts[name] = termination_reason_counts.get(name, 0)+int(failed and flags[index])
                if not all(bool(torch.isfinite(value).all()) for value in
                           [observations, rewards]+[info[key] for key in required]):
                    finite_outputs, nonfinite_step = False, step+1
                    break
                roots, wheels = info["pre_reset_root_states"], info["pre_reset_wheel_states"]
                levels = info["terrain_level"]
                origins = env.terrain_origins[env.environment_indices, levels]
                wheel_ground = env.terrain_model.heights_at(wheels[:, :, :2], env.environment_indices, levels)
                wheel_x = wheels[:, :, 0]-origins[:, None, 0]
                wheel_clearance = wheels[:, :, 2]-wheel_ground-.0625
                contacts = torch.linalg.vector_norm(info["pre_reset_contact_forces"], dim=-1)
                wheel_contact = contacts[:, wheel_indices] > 1.
                nonwheel_contact = (contacts[:, nonwheel_indices] > 1.).any(-1)
                endpoint_valid = ((wheel_x > env.terrain_model.finish_x).all(-1)
                                  & (wheel_clearance.abs() < .025).all(-1) & wheel_contact.all(-1)
                                  & ~nonwheel_contact
                                  & ((roots[:, 1]-origins[:, 1]).abs() < env.terrain_model.lane_half_width-.15))
                forward_velocity = quat_rotate_inverse(roots[:, 3:7], roots[:, 7:10])[:, 0].cpu().tolist()
                values = {key: info[key].cpu().tolist() for key in required}
                xs, clearances = wheel_x.cpu().tolist(), wheel_clearance.cpu().tolist()
                contacts_list, nonwheel_list = wheel_contact.cpu().tolist(), nonwheel_contact.cpu().tolist()
                endpoint = endpoint_valid.cpu().tolist()
                for index in range(env.num_envs):
                    level = int(values["terrain_level"][index])
                    ground = values["terrain_ground_height"][index]
                    episode_peak_ground[index] = max(episode_peak_ground[index], ground)
                    episode_steps[index] += 1
                    side_peaks = env.terrain_model.section_heights[index, level].max(axis=-1)
                    required_peak = max(.005, float(side_peaks.min())-.005)
                    success = bool(values["terrain_success"][index])
                    physical_failure = bool(values["physical_failures"][index])
                    timeout = bool(values["time_outs"][index]) and not success and not physical_failure
                    verified = (success and endpoint[index] and not physical_failure
                                and episode_peak_ground[index] >= required_peak)
                    row = {
                        "step": completed_steps, "time_s": completed_steps*env.dt, "environment_index": index,
                        "terrain_level": level, "progress_m": values["terrain_progress"][index],
                        "best_progress_m": values["terrain_best_progress"][index], "ground_height_m": ground,
                        "root_state": values["pre_reset_root_states"][index],
                        "wheel_x_relative_to_course_origin_m": xs[index],
                        "wheel_clearance_error_m": clearances[index],
                        "wheel_contacts_over_1n": contacts_list[index], "nonwheel_contact_over_1n": nonwheel_list[index],
                        "forward_velocity_body_m_per_s": forward_velocity[index], "target_vx_m_per_s": args.command[0],
                        "pitch_rad": values["pre_reset_pitch"][index], "roll_rad": values["pre_reset_roll"][index],
                        "leg_lengths_m": values["pre_reset_leg_length"][index], "target_leg_length_m": args.command[3],
                        "passive_limit_margin_rad": values["pre_reset_passive_limit_margin"][index],
                        "applied_torques_nm": values["pre_reset_applied_torques"][index],
                        "peak_abs_torques_nm": values["pre_reset_peak_abs_torques"][index],
                        "torque_saturation_fraction": values["pre_reset_torque_saturation_fraction"][index],
                        "closure_error_max_m": max(values["loop_error"][index]),
                        "course_success_reported": success, "course_success_verified": verified,
                        "physical_failure": physical_failure, "timeout": timeout,
                        "episode_step": episode_steps[index], "episode_time_s": episode_steps[index]*env.dt,
                        "termination_reasons": {name: flags[index] for name, flags in reason_flags.items()},
                    }
                    samples.append(row)
                    if success:
                        success_events.append({"step": completed_steps, "environment_index": index,
                                               "terrain_level": level, "episode_seconds": episode_steps[index]*env.dt,
                                               "independently_verified": verified,
                                               "episode_peak_ground_height_m": episode_peak_ground[index],
                                               "required_peak_ground_height_m": required_peak,
                                               "terminal_wheel_x_m": xs[index],
                                               "terminal_wheel_clearance_error_m": clearances[index]})
                    if bool(dones[index]):
                        episode_peak_ground[index], episode_steps[index] = 0., 0
                if completed_steps % 250 == 0:
                    print("stair_steps={} crossings={} failures={} timeouts={}".format(
                        completed_steps, sum(row["course_success_verified"] for row in samples),
                        sum(row["physical_failure"] for row in samples), sum(row["timeout"] for row in samples)), flush=True)
        summary = summarize_stairs(samples, env.num_envs)
        screening = assess_stair_climbing(summary, env.num_envs, steps, completed_steps, finite_outputs)
        iteration = int(checkpoint.get("iter", 0))
        report = {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_iteration": iteration,
            "checkpoint_has_new_rl_updates": iteration > 0,
            "checkpoint_initialization_provenance": checkpoint.get("training_config", {}).get("initialization_provenance"),
            "num_envs": env.num_envs, "requested_steps": steps, "completed_steps": completed_steps,
            "simulated_seconds_per_env": completed_steps*env.dt, "control_dt_s": env.dt,
            "wall_seconds": time.perf_counter()-started, "finite_outputs": finite_outputs,
            "physical_failure_resets": sum(event_counts["physical_failures"]),
            "time_limit_resets": sum(event_counts["time_limit_resets"]),
            "reported_course_crossings": sum(event_counts["reported_successes"]),
            "verified_course_crossings": (summary or {}).get("course_crossings", 0),
            "terminal_event_counts_per_environment": event_counts,
            "termination_reason_counts": termination_reason_counts,
            "nonfinite_step": nonfinite_step, "command": list(args.command), "resolved_environment": kwargs,
            "terrain_level": args.stair_level, "curriculum_enabled": False, "deterministic_policy": True,
            "initial_state_perturbations": False, "time_limit_disabled": False,
            "episode_time_limit_seconds": env.max_episode_length*env.dt,
            "course_finish_x_relative_to_origin_m": env.terrain_model.finish_x,
            "level_nominal_step_height_m": float(env.terrain_model.level_step_heights[args.stair_level]),
            "summary": summary, "screening": screening, "course_crossing_events": success_events,
            "diagnostic_sources": {
                "pose_and_contacts": "Pre-reset snapshots; automatic-reset pose never substitutes for terminal measurements",
                "summary_sampling": "Finite physical samples only; raw terminal event counts also include a non-finite failing step",
                "success": "Environment completion flag plus independent terminal wheel positions, clearance, wheel contact, no nonwheel support, lane containment and same-episode elevated terrain traversal",
                "torque_rms": "Control-period endpoint commands, not all physics-substep RMS",
                "torque_peak_and_saturation": "All physics substeps within each control period",
                "contacts": "Net contact vector norm >1N; contact direction is not a measured normal-force decomposition",
            },
        }
        if args.record_trace:
            report["trace"] = samples
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix+".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
            temporary.replace(args.report)
            print("Stair report: {}".format(args.report.resolve()))
        print(json.dumps({key: value for key, value in report.items() if key != "trace"},
                         ensure_ascii=False, allow_nan=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
