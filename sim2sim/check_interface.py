"""Integration checks for partial resets and the real MuJoCo torque boundary."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg


def tensor_rows(owner, row, prefix=""):
    """Snapshot every state tensor of an outer loop, including nested filters."""
    result = {}
    for name, value in vars(owner).items():
        path = prefix + name
        if isinstance(value, torch.Tensor):
            result[path] = value[row].clone()
        elif hasattr(value, "__dict__"):
            result.update(tensor_rows(value, row, path + "."))
    return result


def capture_step(env, actions):
    """Observe the actual solver inputs without replacing any dynamics."""
    original = mujoco.mj_step
    forces = [[] for _ in env.data]
    indices = {id(data): index for index, data in enumerate(env.data)}

    def record(model, data, *args, **kwargs):
        forces[indices[id(data)]].append(data.qfrc_applied.copy())
        return original(model, data, *args, **kwargs)

    mujoco.mj_step = record
    try:
        result = env.step(actions)
    finally:
        mujoco.mj_step = original
    return result, np.asarray(forces)


def check(config):
    # Noise is disabled only to compare identical physical transitions exactly.
    config = deepcopy(config)
    config["sensors"]["noise_std"] = [0.0] * 13
    config["sensors"]["bias_std"] = [0.0] * 13
    config["sensors"]["wheel_angle_noise_std"] = 0.0
    config["sensors"]["wheel_angle_bias_std"] = 0.0
    checks, measurements = {}, {}

    env = MujocoWheelLeg(config, 2)
    reference = MujocoWheelLeg(config, 2)
    action = torch.tensor([[0.0] * 6, [0.01, -0.01, -0.01, 0.01, -0.02, 0.02]])
    for _ in range(6):
        env.step(action)
        reference.step(action)
    # Trigger only environment zero's timeout during an ordinary vector step.
    env.episode_length_buf[0] = env.max_episode_length - 1
    observation, _, _, done, info = env.step(action)
    expected_observation, _, _, _, _ = reference.step(action)
    actual_hold, expected_hold = tensor_rows(env.hold, 1), tensor_rows(
        reference.hold, 1
    )
    hold_equal = all(
        torch.equal(value, expected_hold[name]) for name, value in actual_hold.items()
    )
    checks["partial_reset_preserves_survivor_observation"] = bool(
        done.tolist() == [True, False]
        and torch.equal(observation[1], expected_observation[1])
        and torch.equal(observation[1], info["terminal_observation"][1])
    )
    checks["partial_reset_preserves_survivor_history_and_sensors"] = bool(
        torch.equal(env.history[1], reference.history[1])
        and torch.equal(env.sensor.sample[1], reference.sensor.sample[1])
        and torch.equal(env.wheel_sensor.sample[1], reference.wheel_sensor.sample[1])
    )
    checks["partial_reset_preserves_survivor_outer_loop"] = hold_equal
    checks["partial_reset_clears_only_done_history_and_actions"] = bool(
        torch.count_nonzero(env.history[0]) == 0
        and torch.count_nonzero(env.actions[0]) == 0
        and torch.count_nonzero(observation[0, :110]) == 0
        and np.array_equal(env.data[1].qpos, reference.data[1].qpos)
    )
    measurements["survivor_outer_loop_tensors_compared"] = len(actual_hold)

    # A baseline plus six independent motor impulses verifies the backend
    # mapping, including the source LF,LH,RF,RH,Lwheel,Rwheel joint order.
    env = MujocoWheelLeg(config, 7)
    actions = torch.zeros(7, 6)
    actions[1:] = 0.25 * torch.eye(6)
    _, applied = capture_step(env, actions)
    first_active = applied[:, 0, :][:, env.vi]
    expected_delta = np.diag([0.17] * 4 + [0.025] * 2)
    actual_delta = first_active[1:] - first_active[0]
    scale_error = float(np.abs(actual_delta - expected_delta).max())
    checks["actual_motor_scaling_and_joint_order"] = scale_error < 1e-7
    passive = [env.model.joint(n).dofadr[0] for n in env.metadata["passive_joints"]]
    checks["base_and_passive_joints_receive_zero_actuation"] = bool(
        np.count_nonzero(applied[:, :, :6]) == 0
        and np.count_nonzero(applied[:, :, passive]) == 0
    )
    joint_velocity = np.array([data.qvel[env.vi] for data in env.data])
    velocity_delta = joint_velocity[1:] - joint_velocity[0]
    checks["positive_motor_impulses_produce_positive_encoder_response"] = bool(
        np.all(np.diag(velocity_delta) > 0)
    )
    truth = env.truth()
    checks["wheel_forward_sensor_mirror_sign"] = bool(
        truth[5, 11] - truth[0, 11] < 0 and truth[6, 12] - truth[0, 12] > 0
    )
    measurements.update(
        actual_motor_scale_error_max_nm=scale_error,
        positive_joint_velocity_response_rad_s=np.diag(velocity_delta).tolist(),
    )

    # Exercise the overload boundary itself: fresh peak, fully spent budget,
    # and a budget crossing the threshold on the first native physics step.
    env = MujocoWheelLeg(config, 3)
    env.budget[1] = 2.0
    env.budget[2] = 2.0 - 0.5 * env.physics_dt
    _, applied = capture_step(env, torch.full((3, 6), 100.0))
    active = applied[:, :, env.vi]
    checks["hardware_peak_torque_limits"] = bool(
        np.all(np.abs(active[:, :, :4]) <= 0.6800001)
        and np.all(np.abs(active[:, :, 4:]) <= 0.1000001)
        and np.allclose(active[0, :, :4], 0.68, atol=1e-7, rtol=0)
    )
    checks["continuous_hip_limit_after_budget"] = bool(
        np.allclose(active[1, :, :4], 0.32, atol=1e-7, rtol=0)
        and np.allclose(active[2, 0, :4], 0.68, atol=1e-7, rtol=0)
        and np.allclose(active[2, 1:, :4], 0.32, atol=1e-7, rtol=0)
    )
    checks["overload_budget_counts_only_peak_duration"] = bool(
        torch.allclose(env.budget[0], torch.full((4,), env.dt), atol=1e-7, rtol=0)
        and torch.equal(env.budget[1], torch.full((4,), 2.0))
        and torch.allclose(
            env.budget[2],
            torch.full((4,), 2.0 + 0.5 * env.physics_dt),
            atol=3e-7,
            rtol=0,
        )
    )
    checks["reported_torque_peaks_match_solver_input"] = bool(
        np.allclose(
            env.last_torque_peak.numpy(), np.abs(active).max(axis=1), atol=1e-7, rtol=0
        )
    )
    measurements.update(
        transition_hip_torque_first_two_substeps_nm=active[2, :2, :4].tolist(),
        final_overload_budget_s=env.budget.tolist(),
    )
    return dict(
        passed=bool(all(checks.values())), checks=checks, measurements=measurements
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "exports/h723_small/metadata.json"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "logs/mujoco/interface_check.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = check(json.loads(args.config.read_text())["environment"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
