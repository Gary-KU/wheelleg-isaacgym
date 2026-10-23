#!/usr/bin/env python3
"""Evaluate active leg support without hiding physical failures behind resets."""

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
from wheelleg_gym.support_metrics import (
    assess_support, duration_steps, summarize_support, velocity_kick_schedule,
)


def apply_velocity_kick(env, delta_world_m_per_s):
    """Change base WORLD velocity once, keeping the closure cache consistent."""
    delta = env.root_states.new_tensor(delta_world_m_per_s)
    before = env.root_states[:, 7:10].clone()
    env.root_states[:, 7:10] += delta
    env.gym.set_actor_root_state_tensor(env.sim, env._root_tensor)
    # A root velocity change moves the whole articulation. The cached body
    # velocities must reflect the same translation before the next loop force.
    env.rb_states[:, :, 7:10] += delta
    env._kinematics()
    env._observations(advance_history=False)
    return {"base_linear_velocity_before_world_m_per_s": before.cpu().tolist(),
            "base_linear_velocity_after_world_m_per_s": env.root_states[:, 7:10].cpu().tolist()}


def quaternion_yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def main():
    parser = common_parser(__doc__, 1)
    parser.set_defaults(sim_device="cpu", headless=True, command=[0.0, 0.0, 0.0, 0.08])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=20.0, help="Total simulated seconds per environment (default 20)")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="Exclude only the first seconds of the whole run from settled metrics (default 2)")
    parser.add_argument("--report", type=Path, help="Write a JSON report; program completion alone does not imply a quality pass")
    parser.add_argument("--record-trace", action="store_true", help="Include per-step/per-environment records in the report")
    parser.add_argument("--kick-at-seconds", type=float,
                        help="Apply one additive base velocity kick at this exact control time")
    parser.add_argument("--kick-velocity", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                        help="Add WORLD-frame X Y Z base velocity in m/s; requires --kick-at-seconds")
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    try:
        steps = duration_steps(args.seconds, DEFAULT_CONTROL_DT)
        settle_steps = duration_steps(args.settle_seconds, DEFAULT_CONTROL_DT, allow_zero=True)
        if settle_steps >= steps:
            raise ValueError("--settle-seconds must be smaller than --seconds")
        kick = velocity_kick_schedule(args.kick_at_seconds, args.kick_velocity, DEFAULT_CONTROL_DT, steps)
    except ValueError as error:
        parser.error(str(error))

    import isaacgym  # noqa: F401; binary extension must precede Torch.
    import torch
    from isaacgym import gymtorch
    from wheelleg_gym.env import WheelLegGymEnv
    from wheelleg_gym.training import make_policy

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    explicit = explicit_argument_destinations(parser, argv)
    # The support task defaults to 8cm even when the checkpoint was trained at
    # another height. Physics/observation settings still follow its checkpoint.
    explicit.add("command")
    try:
        args = resolve_runtime_args(args, explicit, checkpoint, "evaluate")
    except ValueError as error:
        parser.error(str(error))
    validate_common(parser, args)
    if args.random_commands or any(abs(value) > 1e-12 for value in args.command[:3]):
        parser.error("Support evaluation requires a fixed --command 0 0 0 LENGTH")
    kwargs = environment_kwargs(args)
    kwargs["headless"] = True
    env = WheelLegGymEnv(**kwargs)
    started = time.perf_counter()
    try:
        # Never intentionally reset for a time limit. env.step may still reset
        # physical failures; every such event is counted, including startup.
        env.max_episode_length = steps+1
        policy = make_policy(env.num_obs, env.num_actions, env.device,
                             checkpoint.get("training_config", {}).get("policy"))
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        observations, _ = env.reset()
        initial_xy = env.root_states[:, :2].cpu().tolist()
        previous_yaw = [quaternion_yaw(row[3:7]) for row in env.root_states.cpu().tolist()]
        yaw_drift = [0.0]*env.num_envs
        kick_event = None
        properties = env.gym.get_actor_dof_properties(env.envs[0], env.actors[0])
        action_indices = env.action_indices.cpu().tolist()
        hip_effort_limits = [float(properties["effort"][index]) for index in action_indices[:4]]
        wheel_indices = [env.body_names.index(name) for name in ("L_lun", "R_lun")]
        nonwheel_indices = [index for index in range(env.num_bodies) if index not in wheel_indices]
        descriptor = env.gym.acquire_net_contact_force_tensor(env.sim)
        contact_forces = gymtorch.wrap_tensor(descriptor).view(env.num_envs, env.num_bodies, 3)
        samples = []
        failures = [0]*env.num_envs
        timeouts = [0]*env.num_envs
        completed_steps = 0
        finite_outputs = True
        nonfinite_step = None
        required = ("pre_reset_dof_positions", "pre_reset_passive_limit_margin",
                    "pre_reset_leg_length", "pre_reset_applied_torques", "pre_reset_pitch", "pre_reset_roll",
                    "pre_reset_root_states")
        with torch.inference_mode():
            for step in range(steps):
                if kick is not None and step == kick["before_step_index_zero_based"]:
                    kick_event = dict(kick, **apply_velocity_kick(env, kick["delta_linear_velocity_m_per_s"]))
                # Reassert the fixed task after any automatic physical reset.
                observations = env.set_commands(args.command)
                actions = policy.act_inference(observations)
                if not bool(torch.isfinite(actions).all()):
                    finite_outputs, nonfinite_step = False, step+1
                    break
                observations, _, rewards, dones, infos = env.step(actions)
                completed_steps = step+1
                timeout = infos.get("time_outs", torch.zeros_like(dones)).bool() & dones
                physical_failure = dones & ~timeout & ~infos.get("terrain_success", torch.zeros_like(dones))
                for index in range(env.num_envs):
                    failures[index] += int(physical_failure[index])
                    timeouts[index] += int(timeout[index])
                missing = [key for key in required if key not in infos]
                if missing:
                    raise RuntimeError("Environment lacks terminal-safe support diagnostics: {}".format(", ".join(missing)))
                env.gym.refresh_net_contact_force_tensor(env.sim)
                contacts = contact_forces.clone()
                optional_torques = ("pre_reset_peak_abs_torques", "pre_reset_torque_saturation_fraction")
                tensors = ([observations, rewards, contacts]+[infos[key] for key in required]
                           +[infos[key] for key in optional_torques if key in infos])
                if not all(bool(torch.isfinite(value).all()) for value in tensors):
                    finite_outputs, nonfinite_step = False, step+1
                    break
                force_norm = torch.linalg.vector_norm(contacts, dim=-1)
                values = {key: infos[key].cpu().tolist() for key in required}
                for key in optional_torques:
                    if key in infos:
                        values[key] = infos[key].cpu().tolist()
                wheel_contact = (force_norm[:, wheel_indices] > 1.0).cpu().tolist()
                nonwheel_contact = (force_norm[:, nonwheel_indices] > 1.0).any(-1).cpu().tolist()
                for index in range(env.num_envs):
                    root = values["pre_reset_root_states"][index]
                    heading = quaternion_yaw(root[3:7])
                    delta_yaw = heading-previous_yaw[index]
                    yaw_drift[index] += math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
                    previous_yaw[index] = heading
                    row = {
                        "step": completed_steps, "time_s": completed_steps*env.dt, "environment_index": index,
                        "target_leg_length_m": args.command[3],
                        "leg_lengths_m": values["pre_reset_leg_length"][index],
                        "passive_limit_margin_rad": values["pre_reset_passive_limit_margin"][index],
                        "dof_positions_rad": values["pre_reset_dof_positions"][index],
                        "hip_torques_nm": values["pre_reset_applied_torques"][index][:4],
                        "pitch_rad": values["pre_reset_pitch"][index],
                        "roll_rad": values["pre_reset_roll"][index],
                        "base_xy_displacement_world_m": [root[j]-initial_xy[index][j] for j in range(2)],
                        "base_linear_velocity_world_m_per_s": root[7:10],
                        "base_yaw_drift_rad": yaw_drift[index],
                        "wheel_contacts_over_1n": wheel_contact[index],
                        "nonwheel_contact_over_1n": nonwheel_contact[index],
                        "physical_failure": bool(physical_failure[index]), "timeout": bool(timeout[index]),
                    }
                    # Older environments only expose the endpoint. Leave the
                    # optional keys absent so summary metadata records fallback.
                    if "pre_reset_peak_abs_torques" in values:
                        row["hip_torque_peak_nm_per_joint"] = values["pre_reset_peak_abs_torques"][index][:4]
                    if "pre_reset_torque_saturation_fraction" in values:
                        row["hip_torque_substep_saturation_fraction"] = values["pre_reset_torque_saturation_fraction"][index][:4]
                    samples.append(row)
                    if bool(dones[index]):
                        # Exclude automatic-reset orientation teleports from
                        # accumulated physical yaw, retaining the terminal row.
                        previous_yaw[index] = quaternion_yaw(env.root_states[index, 3:7].cpu().tolist())
                if completed_steps % 250 == 0:
                    print("support_steps={} failures={} timeouts={}".format(completed_steps, sum(failures), sum(timeouts)), flush=True)
        all_steps = summarize_support(samples, hip_effort_limits)
        settled_samples = [row for row in samples if row["step"] > settle_steps]
        after_settle = summarize_support(settled_samples, hip_effort_limits)
        screening = assess_support(after_settle, requested_steps=steps, completed_steps=completed_steps,
                                   num_envs=env.num_envs, sample_count=len(samples), finite_outputs=finite_outputs,
                                   physical_failures=sum(failures), timeouts=sum(timeouts), require_motion=True)
        report = {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_iteration": int(checkpoint.get("iter", 0)),
            "num_envs": env.num_envs, "requested_steps": steps, "completed_steps": completed_steps,
            "control_dt_s": env.dt, "simulated_seconds_per_env": completed_steps*env.dt,
            "settle_seconds": args.settle_seconds, "settle_steps": settle_steps,
            "settling_scope": "First steps of the entire run only; not restarted after a failure",
            "wall_seconds": time.perf_counter()-started, "command": list(args.command),
            "deterministic_policy": True, "initial_state_perturbations": False, "time_limit_disabled": True,
            "velocity_kick_requested": kick, "velocity_kick_applied": kick_event,
            "motion_diagnostics": {
                "source": "pre_reset_root_states; linear velocity and XY displacement in WORLD frame",
                "xy_reference": "original run position, including terminal positions before resets",
                "yaw": "unwrapped physical heading increments; automatic-reset orientation teleports excluded",
                "stationary_screen": "0.05 m/s planar RMS, 0.05 m peak XY displacement, 0.1 rad peak yaw drift after initial settling",
                "kick_interpretation": "A single specified velocity kick; no arbitrary-disturbance robustness guarantee",
            },
            "finite_outputs": finite_outputs, "nonfinite_step": nonfinite_step,
            "physical_failure_resets": sum(failures), "physical_failure_resets_per_env": failures,
            "time_limit_resets": sum(timeouts), "time_limit_resets_per_env": timeouts,
            "resolved_environment": kwargs, "dof_names": list(env.dof_names),
            "hip_joint_names": [env.dof_names[index] for index in action_indices[:4]],
            "passive_joint_names": list(env.metadata["passive_joints"]),
            "torque_sampling": {
                "endpoint": "pre_reset_applied_torques: last applied physics-substep command of each control period",
                "rms": "hip_torque_rms_nm_per_joint uses control-step endpoints only; it is not all-substep RMS",
                "peak": "Maximum of within-control-period physics-substep absolute peaks when available; otherwise absolute endpoint",
                "saturation": "Mean within-control-period physics-substep saturation fraction when available; otherwise endpoint saturation indicator",
                "control_period_s": env.dt, "physics_period_s": env.physics_dt,
                "source_sample_counts": all_steps["torque_sampling"] if all_steps else None,
            },
            "diagnostic_sources": {
                "joint_positions_and_torques": "Pre-reset snapshots from env.step; effort commands after clipping, not joint reaction forces",
                "contacts": "Refreshed net-contact tensor from the last physics result; vector norm >1N",
                "sampling": "Per environment per control step, pooled equally; RMS includes mean tilt",
                "joint_stop_reaction_forces_measured": False,
            },
            "all_steps": all_steps, "after_settle": after_settle, "screening": screening,
        }
        if args.record_trace:
            report["trace"] = samples
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix+".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
            temporary.replace(args.report)
            print("Support report: {}".format(args.report.resolve()))
        print(json.dumps(report if not args.record_trace else {key: value for key, value in report.items() if key != "trace"},
                         ensure_ascii=False, allow_nan=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
