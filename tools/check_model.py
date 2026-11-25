#!/usr/bin/env python3
"""Audit the active CAD URDF, mesh references, inertias and closed initial poses."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.model.geometry import (
    ASSET_ROOT,
    MIN_LENGTH,
    MAX_LENGTH,
    WHEEL_RADIUS,
    WHEEL_WIDTH,
    asset_fingerprint,
)
from legged_gym.envs.wheelleg.model.initial_pose import make_initial_pose


def check_model():
    info = json.loads((ASSET_ROOT / "model_info.json").read_text())
    metadata = json.loads((ASSET_ROOT / "loop_constraints.json").read_text())
    urdf = ASSET_ROOT / metadata["urdf"]
    tree = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in tree.findall("link")}
    joints = tree.findall("joint")
    assert len(links) == 11 and len(joints) == 10
    child_names = [j.find("child").get("link") for j in joints]
    assert len(set(child_names)) == len(child_names)
    assert set(links) - set(child_names) == {"base_link"}
    for joint in joints:
        assert joint.find("parent").get("link") in links
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        assert np.isfinite(axis).all() and abs(np.linalg.norm(axis) - 1) < 1e-5
    actuators = metadata["actuator_limits"]
    for joint in joints:
        name = joint.get("name")
        expected = (
            actuators["wheel_peak_nm"]
            if name in ("L_lun_Joint", "R_lun_Joint")
            else actuators["hip_peak_nm"] if name in metadata["active_joints"] else 0.0
        )
        assert abs(float(joint.find("limit").get("effort")) - expected) < 1e-9, name
    mass = 0.0
    for name, link in links.items():
        inertial = link.find("inertial")
        m = float(inertial.find("mass").get("value"))
        assert np.isfinite(m) and m > 0, name
        mass += m
        values = inertial.find("inertia").attrib
        tensor = np.array(
            [
                [
                    float(
                        values["i" + a + b]
                        if "i" + a + b in values
                        else values["i" + b + a]
                    )
                    for b in "xyz"
                ]
                for a in "xyz"
            ]
        )
        principal = np.linalg.eigvalsh(tensor)
        assert np.isfinite(principal).all() and principal.min() > 0, name
        assert principal[-1] <= principal[:2].sum() + 1e-10, name
        origin = inertial.find("origin")
        if origin is not None:
            assert np.isfinite(np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")).all()
    meshes = set()
    for mesh in tree.findall(".//mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        assert ASSET_ROOT in path.parents and path.is_file(), path
        meshes.add(path)
    for path in meshes:
        vertices = []
        faces = []
        for line in path.read_text().splitlines():
            words = line.split()
            if words and words[0] == "v":
                vertices.append([float(x) for x in words[1:4]])
            elif words and words[0] == "f":
                faces.extend(int(x.split("/")[0]) for x in words[1:])
        vertices = np.array(vertices)
        assert len(vertices) > 0 and np.isfinite(vertices).all(), path
        assert faces and min(faces) >= 1 and max(faces) <= len(vertices), path
        if path.name in ("L_lun_collision.obj", "R_lun_collision.obj"):
            # The CAD tire mesh uses local Z as its axle.
            extent = np.ptp(vertices, axis=0)
            assert np.allclose(
                extent, [2 * WHEEL_RADIUS, 2 * WHEEL_RADIUS, WHEEL_WIDTH], atol=1e-6
            ), extent
    for name, expected in info["files"].items():
        path = ASSET_ROOT / Path(name).relative_to("assets/wheelleg")
        assert (
            hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == expected
        ), name
    poses = {
        str(length): make_initial_pose(urdf, metadata, length)
        for length in (MIN_LENGTH, 0.10, 0.1175, MAX_LENGTH)
    }
    for pose in poses.values():
        assert max(pose["closure_error_m_by_constraint"].values()) < 1e-6
        assert max(pose["closure_axis_error_rad_by_constraint"].values()) < 1e-5
        assert min(pose["minimum_collision_height_m_by_body"].values()) >= 0.000999
        for joint in joints:
            q = pose["joint_positions"][joint.get("name")]
            limit = joint.find("limit")
            if joint.get("type") != "continuous":
                assert float(limit.get("lower")) <= q <= float(limit.get("upper"))
    return dict(
        model_id=info["model_id"],
        asset_fingerprint=asset_fingerprint(),
        links=len(links),
        tree_joints=len(joints),
        loop_constraints=len(metadata["constraints"]),
        referenced_meshes=len(meshes),
        total_mass_kg=mass,
        actuator_limits=actuators,
        wheel_radius_m=WHEEL_RADIUS,
        wheel_width_m=WHEEL_WIDTH,
        command_range_m=[MIN_LENGTH, MAX_LENGTH],
        initial_poses=poses,
        inertia_status=info["inertia_status"],
        status="passed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check_model()
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
