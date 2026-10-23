#!/usr/bin/env python3
"""Reproducible 28-second flat keyboard-drive screening, with terminal-safe metrics."""

import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wheelleg_gym.cli import common_parser, environment_kwargs, validate_common
from wheelleg_gym.config import DEFAULT_CONTROL_DT
from wheelleg_gym.runtime_config import explicit_argument_destinations, resolve_runtime_args
from wheelleg_gym.support_metrics import duration_steps


def make_drive_sequence(speed=.6, turn_speed=2., dt=DEFAULT_CONTROL_DT):
    """One continuous sequence; only actual physical failures request a reset."""
    if not all(math.isfinite(x) and x > 0 for x in (speed, turn_speed)):
        raise ValueError("Drive and turn speeds must be finite and positive")
    pattern = [
        ("idle_initial", 3., 0., 0.),
        ("forward", 3., speed, 0.),
        ("stop_after_forward", 2., 0., 0.),
        ("backward", 3., -speed, 0.),
        ("stop_after_backward", 2., 0., 0.),
        ("start_forward", 1., speed, 0.),
        ("start_stop_forward", 1., 0., 0.),
        ("start_backward", 1., -speed, 0.),
        ("start_stop_backward", 1., 0., 0.),
        ("turn_left", 2., 0., turn_speed),
        ("hold_after_left", 2., 0., 0.),
        ("turn_right", 2., 0., -turn_speed),
        ("hold_after_right", 2., 0., 0.),
        ("idle_final", 3., 0., 0.),
    ]
    sequence, step = [], 0
    for name, seconds, vx, yaw_rate in pattern:
        end = step+duration_steps(seconds, dt)
        sequence.append(dict(name=name, start_step=step, end_step_exclusive=end,
                             seconds=seconds, raw_vx_m_per_s=vx,
                             raw_yaw_rate_rad_per_s=yaw_rate))
        step = end
    return sequence


def wrapped_increment(after, before):
    return math.atan2(math.sin(after-before), math.cos(after-before))


def quaternion_yaw(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def summarize_drive(rows):
    """All finite physical rows, including the last sample before a reset."""
    if not rows:
        return None
    def mean(key):
        return math.fsum(row[key] for row in rows)/len(rows)
    def rms(key):
        return math.sqrt(math.fsum(row[key]**2 for row in rows)/len(rows))
    def std(key):
        center = mean(key)
        return math.sqrt(math.fsum((row[key]-center)**2 for row in rows)/len(rows))
    result = {"samples": len(rows), "duration_s": math.fsum(row["dt_s"] for row in rows),
              "physical_failures": sum(row["physical_failure"] for row in rows),
              "timeouts": sum(row["timeout"] for row in rows),
              "signed_body_forward_distance_m": math.fsum(row["delta_body_forward_m"] for row in rows),
              "signed_world_x_distance_m": math.fsum(row["delta_xy_world_m"][0] for row in rows),
              "signed_world_y_distance_m": math.fsum(row["delta_xy_world_m"][1] for row in rows),
              "planar_path_length_m": math.fsum(math.hypot(*row["delta_xy_world_m"]) for row in rows),
              "raw_requested_forward_distance_m": math.fsum(row["raw_vx_m_per_s"]*row["dt_s"] for row in rows),
              "applied_requested_forward_distance_m": math.fsum(row["applied_vx_m_per_s"]*row["dt_s"] for row in rows),
              "base_height_min_m": min(row["base_height_m"] for row in rows),
              "closure_error_peak_m": max(row["closure_error_m"] for row in rows),
              "final_yaw_error_rad": rows[-1]["yaw_error_rad"],
              "final_physical_yaw_rad_excluding_resets": rows[-1]["physical_yaw_rad"],
              "final_requested_yaw_rad_excluding_resets": rows[-1]["requested_yaw_rad"],
              "final_global_yaw_error_rad": rows[-1]["physical_yaw_rad"]-rows[-1]["requested_yaw_rad"]}
    for key in ("yaw_error_rad", "yaw_rate_rad_per_s", "pitch_rad", "roll_rad", "vx_body_m_per_s",
                "vx_error_raw_m_per_s", "vx_error_applied_m_per_s", "yaw_rate_error_raw_rad_per_s",
                "yaw_rate_error_applied_rad_per_s"):
        result[key+"_mean"] = mean(key)
        result[key+"_rms"] = rms(key)
        result[key+"_std"] = std(key)
        result[key+"_abs_peak"] = max(abs(row[key]) for row in rows)
    result["termination_reason_counts"] = {
        key: sum(row["physical_failure"] and row["termination_reasons"].get(key, False) for row in rows)
        for key in sorted({key for row in rows for key in row["termination_reasons"]})}
    result["active_torque_endpoint_rms_nm_per_joint"] = [
        math.sqrt(math.fsum(row["torques_nm"][j]**2 for row in rows)/len(rows)) for j in range(6)]
    result["active_torque_substep_peak_nm_per_joint"] = [max(row["torque_peaks_nm"][j] for row in rows) for j in range(6)]
    return result


def main():
    parser = common_parser(__doc__, 1)
    parser.set_defaults(sim_device="cpu", physics_threads=0, headless=True,
                        command=[0., 0., 0., .08], initial_leg_length=.05,
                        yaw_acceleration=4., terrain="plane", stair_curriculum=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--drive-speed", type=float, default=.6)
    parser.add_argument("--turn-speed", type=float, default=2.)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    try:
        sequence = make_drive_sequence(args.drive_speed, args.turn_speed)
    except ValueError as error:
        parser.error(str(error))

    import isaacgym  # noqa: F401; SDK binary extension must precede Torch.
    import torch
    from wheelleg_gym.commands import CommandGovernor
    from wheelleg_gym.constraints import quat_rotate_inverse
    from wheelleg_gym.env import WheelLegGymEnv
    from wheelleg_gym.training import make_policy

    torch.set_num_threads(1)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    explicit = explicit_argument_destinations(parser)
    # Match stand.sh's explicitly selected task and initial crouch. Other
    # controller/physics settings follow the checkpoint unless overridden.
    explicit.update(("command", "initial_leg_length", "yaw_acceleration", "terrain", "stair_curriculum"))
    try:
        args = resolve_runtime_args(args, explicit, checkpoint, "evaluate")
    except ValueError as error:
        parser.error(str(error))
    validate_common(parser, args)
    if args.num_envs != 1 or args.sim_device != "cpu" or args.physics_threads != 0:
        parser.error("This matched drive screening requires --num-envs 1 --sim-device cpu --physics-threads 0")
    if args.random_commands or any(abs(x) > 1e-12 for x in args.command[:3]):
        parser.error("Use a stationary initial --command 0 0 0 LENGTH; the fixed sequence supplies motion")
    kwargs = environment_kwargs(args)
    kwargs.update(headless=True, terrain="plane", stair_curriculum=False)
    env = WheelLegGymEnv(**kwargs)
    total_steps = sequence[-1]["end_step_exclusive"]
    try:
        # Keyboard play also removes periodic timeouts, retaining all physical
        # failure checks and automatic physical resets.
        env.max_episode_length = total_steps+1
        policy = make_policy(env.num_obs, env.num_actions, env.device,
                             checkpoint.get("training_config", {}).get("policy"))
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        env.reset()
        governor = CommandGovernor(env.commands.clone(), args.linear_acceleration, args.yaw_acceleration)
        samples, completed_steps = [], 0
        physical_yaw = requested_yaw = episode_yaw = episode_reference = 0.
        previous_yaw = quaternion_yaw(env.root_states[0, 3:7].cpu().tolist())
        failures = timeouts = 0
        finite_outputs, nonfinite_step = True, None
        required = ("pre_reset_root_states", "pre_reset_ang_vel", "pre_reset_pitch", "pre_reset_roll",
                    "pre_reset_commands", "pre_reset_applied_torques", "pre_reset_peak_abs_torques",
                    "physical_failures", "time_outs", "termination_reasons", "loop_error")
        started = time.perf_counter()
        with torch.inference_mode():
            for phase in sequence:
                for step in range(phase["start_step"], phase["end_step_exclusive"]):
                    raw = env.commands.new_tensor([[phase["raw_vx_m_per_s"], 0.,
                                                    phase["raw_yaw_rate_rad_per_s"], args.command[3]]])
                    applied = governor.step(raw, env.dt)
                    observations = env.set_commands(applied)
                    before_xy = env.root_states[0, :2].clone()
                    before_heading = quaternion_yaw(env.root_states[0, 3:7].cpu().tolist())
                    actions = policy.act_inference(observations)
                    if not bool(torch.isfinite(actions).all()):
                        finite_outputs, nonfinite_step = False, step+1
                        break
                    observations, _, rewards, dones, info = env.step(actions)
                    completed_steps = step+1
                    missing = [key for key in required if key not in info]
                    if missing:
                        raise RuntimeError("Terminal-safe drive diagnostics missing: {}".format(missing))
                    failure, timeout = bool(info["physical_failures"][0]), bool(info["time_outs"][0])
                    failures += int(failure)
                    timeouts += int(timeout)
                    tensors = [observations, rewards]+[info[key] for key in required if isinstance(info[key], torch.Tensor)]
                    if not all(bool(torch.isfinite(value).all()) for value in tensors):
                        finite_outputs, nonfinite_step = False, step+1
                        break
                    roots = info["pre_reset_root_states"]
                    root = roots[0].cpu().tolist()
                    vx = float(quat_rotate_inverse(roots[:, 3:7], roots[:, 7:10])[0, 0])
                    heading = quaternion_yaw(root[3:7])
                    increment = wrapped_increment(heading, previous_yaw)
                    physical_yaw += increment
                    episode_yaw += increment
                    previous_yaw = heading
                    command = info["pre_reset_commands"][0].cpu().tolist()
                    requested_yaw += command[2]*env.dt
                    episode_reference += command[2]*env.dt
                    delta_xy = (roots[0, :2]-before_xy).cpu().tolist()
                    yaw_rate = float(info["pre_reset_ang_vel"][0, 2])
                    samples.append({
                        "step": completed_steps, "time_s": completed_steps*env.dt, "dt_s": env.dt,
                        "phase": phase["name"], "raw_vx_m_per_s": float(raw[0, 0]),
                        "raw_yaw_rate_rad_per_s": float(raw[0, 2]),
                        "applied_vx_m_per_s": command[0], "applied_yaw_rate_rad_per_s": command[2],
                        "vx_body_m_per_s": vx, "vx_error_raw_m_per_s": vx-float(raw[0, 0]),
                        "vx_error_applied_m_per_s": vx-command[0],
                        "yaw_rate_rad_per_s": yaw_rate,
                        "yaw_rate_error_raw_rad_per_s": yaw_rate-float(raw[0, 2]),
                        "yaw_rate_error_applied_rad_per_s": yaw_rate-command[2],
                        "physical_yaw_rad": physical_yaw, "requested_yaw_rad": requested_yaw,
                        "episode_yaw_rad": episode_yaw, "episode_yaw_reference_rad": episode_reference,
                        "yaw_error_rad": episode_yaw-episode_reference,
                        "pitch_rad": float(info["pre_reset_pitch"][0]), "roll_rad": float(info["pre_reset_roll"][0]),
                        "base_height_m": root[2], "delta_xy_world_m": delta_xy,
                        "delta_body_forward_m": delta_xy[0]*math.cos(before_heading)+delta_xy[1]*math.sin(before_heading),
                        "torques_nm": info["pre_reset_applied_torques"][0].cpu().tolist(),
                        "torque_peaks_nm": info["pre_reset_peak_abs_torques"][0].cpu().tolist(),
                        "closure_error_m": float(info["loop_error"][0].max()),
                        "physical_failure": failure, "timeout": timeout,
                        "termination_reasons": {key: bool(value[0]) for key, value in info["termination_reasons"].items()},
                    })
                    if bool(dones[0]):
                        previous_yaw = quaternion_yaw(env.root_states[0, 3:7].cpu().tolist())
                        episode_yaw = episode_reference = 0.
                        governor.reset(env.commands.clone())
                print("drive_phase={} steps={} failures={}".format(phase["name"], completed_steps, failures), flush=True)
                if not finite_outputs:
                    break
        phase_reports = []
        for phase in sequence:
            rows = [row for row in samples if row["phase"] == phase["name"]]
            phase_reports.append(dict(phase, all_steps=summarize_drive(rows),
                                      last_second=summarize_drive(rows[-int(round(1./env.dt)):])))
        report = {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_iteration": int(checkpoint.get("iter", 0)),
            "seed": args.seed, "resolved_environment": kwargs, "num_envs": 1, "sim_device": "cpu",
            "physics_threads": 0, "requested_steps": total_steps, "completed_steps": completed_steps,
            "simulated_seconds": completed_steps*env.dt, "wall_seconds": time.perf_counter()-started,
            "finite_outputs": finite_outputs, "nonfinite_step": nonfinite_step,
            "physical_failure_resets": failures, "time_limit_resets": timeouts,
            "time_limit_disabled_as_in_keyboard_play": True, "deterministic_policy": True,
            "root_frozen": False, "forced_root_state_writes": False,
            "summary": summarize_drive(samples), "phases": phase_reports,
            "metric_semantics": {
                "yaw_error": "Physical unwrapped episode heading minus integrated applied yaw-rate command; target resets only after counted physical failures, matching the controller",
                "global_yaw": "Cumulative physical heading increments and applied command integral; reset orientation teleports excluded from physical rotation",
                "turn_return": "Equal positive and negative yaw-rate phases; after both command ramps settle the intended global heading returns to its original value",
                "distances": "Sum of pre-reset step displacement; reset translation teleports excluded; signed body-forward distance uses heading at each step start",
                "rms_and_std": "RMS includes mean offset; std isolates variation about the mean; terminal failure samples are included",
                "velocity": "Root linear velocity transformed into body frame; angular rate from pre-reset body-frame snapshot",
                "last_second": "Diagnostic subset only; all_steps and top-level failure counts always retain the whole run",
                "quality": "Descriptive screening only; program completion and absence of falls do not establish stable drive tracking",
            },
        }
        if args.record_trace:
            report["trace"] = samples
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix+".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
        temporary.replace(args.report)
        print("Drive report: {}".format(args.report.resolve()))
        print(json.dumps({key: report[key] for key in ("completed_steps", "physical_failure_resets", "time_limit_resets", "summary")}, ensure_ascii=False, allow_nan=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
