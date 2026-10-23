"""可选腿长课程的完整 URDF 姿态；不创建仿真或提供支撑力矩。"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import MAX_LEG_LENGTH


def symmetric_joint_positions(length):
    """Return all ten joint angles on the same assembly branch as kinematics.py."""
    if not math.isfinite(length) or not 0.05 <= length <= MAX_LEG_LENGTH:
        raise ValueError("Initial leg length must be in [0.05,{}] m".format(MAX_LEG_LENGTH))
    distance = math.hypot(0.044, length)
    phi1 = (math.asin(length/distance)
            + math.acos((distance*distance+0.105**2-0.16**2)/(2*distance*0.105)))
    phi2 = math.atan2(length-0.105*math.sin(phi1), 0.044-0.105*math.cos(phi1))
    initial_phi1 = 3*math.pi/4
    initial_x = 0.044-0.105*math.cos(initial_phi1)
    initial_phi2 = math.atan2(math.sqrt(0.16**2-initial_x**2), initial_x)
    hip = phi1-initial_phi1
    knee = (initial_phi2-initial_phi1)-(phi2-phi1)
    return {"L_F_Joint1": hip, "L_H_Joint1": -hip,
            "R_F_Joint1": -hip, "R_H_Joint1": hip,
            "L_F_Joint2": -knee, "L_H_Joint2": knee,
            "R_F_Joint2": knee, "R_H_Joint2": -knee,
            "L_lun_Joint": 0.0, "R_lun_Joint": 0.0}


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("Joint axis must be nonzero")
    x, y, z = axis/norm
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*cross.dot(cross)


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = [float(value) for value in element.get("xyz", "0 0 0").split()]
        r, p, y = [float(value) for value in element.get("rpy", "0 0 0").split()]
        transform[:3, :3] = (_axis_rotation([0, 0, 1], y)
                              .dot(_axis_rotation([0, 1, 0], p))
                              .dot(_axis_rotation([1, 0, 0], r)))
    return transform


def _quaternion_rotation(values):
    x, y, z, w = np.asarray(values, dtype=float)/np.linalg.norm(values)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def make_initial_pose(urdf_path, metadata, length, clearance=0.001):
    """Compute closed-chain joint angles and root height from collision vertices.

    Returned rigid transforms are relative to root translation and include its
    configured rotation. The SDK must still perform real FK before caching the
    environment's reset states; this function never fabricates that cache.
    """
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("Initial ground clearance must be positive")
    path = Path(urdf_path)
    robot = ET.parse(str(path)).getroot()
    joints = symmetric_joint_positions(length)
    root_pose = metadata.get("default_root_pose", {})
    quaternion = root_pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    if len(quaternion) != 4 or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-12:
        raise ValueError("Root quaternion must be finite and nonzero")
    root_transform = np.eye(4)
    root_transform[:3, :3] = _quaternion_rotation(quaternion)
    transforms = {metadata["root_body"]: root_transform}
    pending = list(robot.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            name, kind = joint.get("name"), joint.get("type")
            angle = joints.get(name, 0.0)
            limit = joint.find("limit")
            if kind == "revolute" and not float(limit.get("lower"))-1e-6 <= angle <= float(limit.get("upper"))+1e-6:
                raise ValueError("Initial pose violates the limit of {}".format(name))
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                axis = [float(value) for value in joint.find("axis").get("xyz").split()]
                motion[:3, :3] = _axis_rotation(axis, angle)
            elif kind != "fixed":
                raise ValueError("Unsupported initial-pose joint type: {}".format(kind))
            transforms[joint.find("child").get("link")] = transforms[parent].dot(_origin(joint.find("origin"))).dot(motion)
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF joint tree is disconnected")
    lowest = {}
    for link in robot.findall("link"):
        body = link.get("name")
        values = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Initial pose currently requires mesh collision geometry")
            mesh_path = path.parent/mesh.get("filename")
            vertices = []
            with mesh_path.open() as stream:
                for line in stream:
                    if line.startswith("v "):
                        vertices.append([float(value) for value in line.split()[1:4]])
            if not vertices:
                raise ValueError("Collision mesh has no vertices: {}".format(mesh_path))
            scale = np.array([float(value) for value in mesh.get("scale", "1 1 1").split()])
            transform = transforms[body].dot(_origin(collision.find("origin")))
            points = np.asarray(vertices)*scale
            values.append(float(np.min(points.dot(transform[:3, :3].T)[:, 2]+transform[2, 3])))
        if values:
            lowest[body] = min(values)
    wheel_lowest = min(lowest[name] for name in ("L_lun", "R_lun"))
    root_position = list(root_pose.get("pos", [0.0, 0.0, 0.25]))
    root_position[2] = clearance-wheel_lowest
    world_lowest = {name: value+root_position[2] for name, value in lowest.items()}
    if min(world_lowest.values()) < -1e-6:
        raise ValueError("Initial pose places a nonwheel collision mesh below the floor")
    gaps = {}
    for constraint in metadata["constraints"]:
        def endpoint(body_key, point_key):
            return transforms[constraint[body_key]].dot(np.r_[constraint[point_key], 1.0])[:3]
        gaps[constraint["name"]] = float(np.linalg.norm(endpoint("body0", "local_pos0")-endpoint("body1", "local_pos1")))
    if max(gaps.values()) > 0.0001:
        raise ValueError("Initial closed-chain anchors disagree by more than 0.1mm")
    return {"joint_positions": joints, "root_position": root_position,
            "root_quaternion_xyzw": list(quaternion), "collision_clearance_m": clearance,
            "minimum_collision_height_m_by_body": world_lowest,
            "closure_error_m_by_constraint": gaps}
