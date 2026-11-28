"""Check the converted robot independently of the actor, before RL training.

This verifies source-asset parity and short physical responses. Passing does not
certify policy balance, locomotion, or sim-to-real transfer. No body is fixed and
no external support force is applied in the grounded check.
"""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from build_model import ASSET, ROOT, build

sys.path.insert(0, str(ROOT))
from legged_gym.envs.wheelleg.control.leg_control import (
    hip_pd_torques,
    hip_support_torques,
    hip_targets,
)
from legged_gym.envs.wheelleg.model.initial_pose import (
    _axis_rotation,
    _origin,
    make_initial_pose,
)


def reset(model, metadata, height=0.0):
    """Use the same closed-chain branch and COM-balanced pose as playback."""
    data = mujoco.MjData(model)
    pose = make_initial_pose(
        ASSET / metadata["urdf"], metadata, 0.08, balance_initial_com=True
    )
    data.qpos[:3] = pose["root_position"]
    data.qpos[2] += height
    x, y, z, w = pose["root_quaternion_xyzw"]
    data.qpos[3:7] = [w, x, y, z]
    for name, value in pose["joint_positions"].items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return data


def closure_error(model, data, metadata):
    return max(
        np.linalg.norm(
            data.site_xpos[model.site(c["name"] + "_0_" + str(n)).id]
            - data.site_xpos[model.site(c["name"] + "_1_" + str(n)).id]
        )
        for c in metadata["constraints"]
        for n in (0, 1)
    )


def audit():
    metadata = json.loads((ASSET / "loop_constraints.json").read_text())
    model_path = build(ASSET / "mujoco/robot.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    source = ET.parse(ASSET / metadata["urdf"]).getroot()
    data = reset(model, metadata)
    checks, measurements = {}, {}
    mass_error, inertia_error, com_error = [], [], []
    for link in source.findall("link"):
        body = model.body(link.get("name"))
        inertial = link.find("inertial")
        mass_error.append(abs(body.mass[0] - float(inertial.find("mass").get("value"))))
        com_error.append(
            np.linalg.norm(body.ipos - _origin(inertial.find("origin"))[:3, 3])
        )
        values = inertial.find("inertia").attrib
        xx, yy, zz, xy, xz, yz = [
            float(values[n]) for n in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ]
        rotation = _origin(inertial.find("origin"))[:3, :3]
        expected = (
            rotation @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rotation.T
        )
        actual_rotation = np.empty(9)
        mujoco.mju_quat2Mat(actual_rotation, body.iquat)
        actual_rotation = actual_rotation.reshape(3, 3)
        actual = actual_rotation @ np.diag(body.inertia) @ actual_rotation.T
        inertia_error.append(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )
    measurements.update(
        total_mass_kg=float(model.body_mass.sum()),
        mass_error_max_kg=max(mass_error),
        inertia_error_relative_max=max(inertia_error),
        com_error_max_m=max(com_error),
    )
    checks["source_mass_com_inertia"] = (
        max(mass_error) < 1e-12 and max(inertia_error) < 1e-8 and max(com_error) < 1e-12
    )

    # Evaluate tree transforms directly from the URDF, without MuJoCo FK.
    root_transform = np.eye(4)
    root_transform[:3, :3] = data.xmat[model.body("base_link").id].reshape(3, 3)
    root_transform[:3, 3] = data.qpos[:3]
    transforms = {"base_link": root_transform}
    remaining = list(source.findall("joint"))
    while remaining:
        for joint in remaining[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            motion = np.eye(4)
            if joint.get("type") != "fixed":
                axis = [float(v) for v in joint.find("axis").get("xyz").split()]
                angle = data.qpos[model.joint(joint.get("name")).qposadr[0]]
                motion[:3, :3] = _axis_rotation(axis, angle)
            transforms[joint.find("child").get("link")] = (
                transforms[parent] @ _origin(joint.find("origin")) @ motion
            )
            remaining.remove(joint)
    fk_error = max(
        np.linalg.norm(transform[:3, 3] - data.xpos[model.body(name).id])
        for name, transform in transforms.items()
    )
    rotation_error = max(
        np.linalg.norm(transform[:3, :3] - data.xmat[model.body(name).id].reshape(3, 3))
        for name, transform in transforms.items()
    )
    measurements.update(
        fk_position_error_max_m=fk_error,
        fk_rotation_error_max=rotation_error,
        initial_closure_error_m=closure_error(model, data, metadata),
    )
    checks["source_joint_frames"] = fk_error < 1e-10 and rotation_error < 1e-10
    checks["initial_closed_chain"] = measurements["initial_closure_error_m"] < 1e-7
    checks["free_base_and_passive_joints"] = (
        model.nq == 17 and model.nv == 16 and model.nu == 0
    )
    limits_match = True
    for joint in source.findall("joint"):
        actual = model.joint(joint.get("name"))
        limit = joint.find("limit")
        if joint.get("type") == "revolute":
            limits_match &= np.allclose(
                actual.range,
                [float(limit.get("lower")), float(limit.get("upper"))],
                atol=1e-12,
                rtol=0,
            )
        effort = float(limit.get("effort"))
        if effort:
            limits_match &= np.allclose(
                model.jnt_actfrcrange[actual.id], [-effort, effort], atol=1e-12, rtol=0
            )
    checks["source_joint_and_effort_limits"] = bool(limits_match)

    visuals = {g.get("name") for g in source.findall(".//visual")}
    colliders = {g.get("name") for g in source.findall(".//collision")}
    checks["visuals_noncolliding"] = all(
        model.geom(n).contype[0] == 0 and model.geom(n).conaffinity[0] == 0
        for n in visuals
    )
    checks["source_collision_meshes_and_self_filter"] = all(
        model.geom(n).contype[0] == 1 and model.geom(n).conaffinity[0] == 2
        for n in colliders
    )
    checks["initial_ground_clearance"] = data.ncon == 0

    # BODY local axes are the diagonalized inertia frame, which differs from
    # the URDF link/IMU frame. XBODY is the regular body frame required here.
    data.qvel[3:6] = [0, 0.7, 0.2]
    mujoco.mj_forward(model, data)
    local_velocity = np.empty(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        model.body("base_link").id,
        local_velocity,
        1,
    )
    matrix = data.xmat[model.body("base_link").id].reshape(3, 3)
    pitch_before = np.arcsin(-matrix[2, 0])
    mujoco.mj_integratePos(model, data.qpos, data.qvel, 1e-6)
    mujoco.mj_forward(model, data)
    matrix = data.xmat[model.body("base_link").id].reshape(3, 3)
    pitch_difference = float((np.arcsin(-matrix[2, 0]) - pitch_before) / 1e-6)
    measurements.update(
        imu_link_pitch_rate_rad_s=float(local_velocity[1]),
        imu_link_yaw_rate_rad_s=float(local_velocity[2]),
        imu_pitch_rate_finite_difference_rad_s=pitch_difference,
    )
    checks["imu_link_frame_and_pitch_rate_sign"] = bool(
        np.allclose(local_velocity[:3], [0, 0.7, 0.2], atol=1e-9, rtol=0)
        and abs(pitch_difference - 0.7) < 1e-6
    )

    # In flight, the COM must fall at g. A constraint can redistribute internal
    # forces but must not provide hidden support to the floating base.
    data = reset(model, metadata, height=0.5)
    start_com = data.subtree_com[model.body("base_link").id].copy()
    duration = 0.01
    for _ in range(round(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    drop = start_com[2] - data.subtree_com[model.body("base_link").id, 2]
    expected_drop = 0.5 * 9.81 * duration * (duration + model.opt.timestep)
    measurements.update(
        freefall_drop_m=float(drop), freefall_expected_drop_m=expected_drop
    )
    checks["gravity_without_support"] = (
        abs(drop - expected_drop) < 1e-7 and data.ncon == 0
    )

    # Equal forward axle torques must spin both wheels forward and react against
    # positive body pitch. Signs match the original encoder/IMU contract.
    model.opt.gravity[:] = 0
    data = reset(model, metadata, height=0.5)
    wheel_dofs = [model.joint(n).dofadr[0] for n in metadata["active_joints"][-2:]]
    data.qfrc_applied[wheel_dofs] = [-0.01, 0.01]
    for _ in range(round(0.02 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    forward_wheel_rates = data.qvel[wheel_dofs] * [-1, 1]
    measurements.update(
        airborne_forward_wheel_rates_rad_s=forward_wheel_rates.tolist(),
        airborne_pitch_reaction_rad_s=float(data.qvel[4]),
    )
    checks["wheel_torque_and_pitch_sign"] = bool(
        np.all(forward_wheel_rates > 0) and data.qvel[4] < 0
    )

    # Short support check uses only the existing hip PD/feedforward. The wheels
    # are unactuated: this is a physics check, not a balance-controller test.
    model.opt.gravity[:] = [0, 0, -9.81]
    data = reset(model, metadata)
    hip_q = [model.joint(n).qposadr[0] for n in metadata["active_joints"][:4]]
    hip_v = [model.joint(n).dofadr[0] for n in metadata["active_joints"][:4]]
    target = hip_targets(torch.tensor([0.08]))
    peak_closure, peak_pitch, contacts, normal_forces = 0.0, 0.0, set(), []
    for step in range(round(0.2 / model.opt.timestep)):
        q = torch.tensor(data.qpos[hip_q], dtype=torch.float32)[None]
        v = torch.tensor(data.qvel[hip_v], dtype=torch.float32)[None]
        torque = hip_pd_torques(q, v, target, kp=2.0, kd=0.08, differential_kd=0.08)
        torque += hip_support_torques(q, 6.482961259305951)
        data.qfrc_applied[hip_v] = torque.clamp(-0.68, 0.68).numpy()[0]
        mujoco.mj_step(model, data)
        peak_closure = max(peak_closure, closure_error(model, data, metadata))
        matrix = data.xmat[model.body("base_link").id].reshape(3, 3)
        peak_pitch = max(peak_pitch, abs(np.arcsin(np.clip(-matrix[2, 0], -1, 1))))
        total_normal = 0.0
        for index, contact in enumerate(data.contact):
            names = [model.geom(int(i)).name for i in (contact.geom1, contact.geom2)]
            contacts.update(n for n in names if n != "floor")
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, force)
            total_normal += abs((contact.frame.reshape(3, 3).T @ force[:3])[2])
        if step * model.opt.timestep > 0.15:
            normal_forces.append(total_normal)
    mean_support = float(np.mean(normal_forces))
    measurements.update(
        grounded_peak_closure_m=peak_closure,
        grounded_peak_pitch_rad=float(peak_pitch),
        grounded_final_root_height_m=float(data.qpos[2]),
        grounded_contact_geoms=sorted(contacts),
        grounded_mean_vertical_contact_force_n=mean_support,
        robot_weight_n=float(model.body_mass.sum() * 9.81),
    )
    checks["grounded_short_support"] = (
        peak_closure < 1e-5
        and peak_pitch < 0.1
        and contacts == {"L_lun_tread", "R_lun_tread"}
        and data.qpos[2] > 0.1
        and 0.7 < mean_support / measurements["robot_weight_n"] < 1.3
    )
    return {
        "passed": bool(all(checks.values())),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "checks": {name: bool(value) for name, value in checks.items()},
        "measurements": measurements,
        "scope": "Source parity and 0.2 s actor-free physics only; policy transfer is not certified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "logs/mujoco/model_audit.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
