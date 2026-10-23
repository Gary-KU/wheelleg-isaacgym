#!/usr/bin/env python3
"""Export the repository's USD robot to an Isaac Gym URDF tree and loop metadata.

Run with a Python interpreter containing ``usd-core`` (pxr). This is an offline
asset tool; Isaac Gym's Python environment does not need USD or this dependency.
The source USD is never edited. Geometry comes from its instance proxies, not
from replacement primitives. Every non-root link frame is moved to its incoming
joint's child frame because URDF requires that frame to be the link origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[2]


def numbers(values):
    return " ".join(format(float(v), ".12g") for v in values)


def vector(value):
    return [float(v) for v in value]


def frame(position, quaternion):
    # Convert float quaternions directly to a normalized double matrix. Passing
    # through axis/angle Rotation can erase small principal-axis rotations when
    # a float quaternion's real component rounds to exactly one.
    result = Gf.Matrix4d().SetRotate(Gf.Quatd(quaternion).GetNormalized())
    result.SetTranslateOnly(Gf.Vec3d(position))
    return result


def quaternion_wxyz(matrix):
    q = matrix.ExtractRotationQuat().GetNormalized()
    return [float(q.GetReal()), *vector(q.GetImaginary())]


def rpy(matrix):
    # Gf matrices multiply row vectors; URDF's R = Rz(yaw) Ry(pitch) Rx(roll).
    pitch = math.asin(max(-1.0, min(1.0, -matrix[0][2])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(matrix[1][2], matrix[2][2])
        yaw = math.atan2(matrix[0][1], matrix[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[1][0], matrix[1][1])
    return [roll, pitch, yaw]


def from_rpy(xyz, angles):
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = Gf.Matrix4d(
        cy * cp, sy * cp, -sp, 0,
        cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr, 0,
        cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr, 0,
        *xyz, 1,
    )
    return result


def origin(parent, matrix):
    ET.SubElement(parent, "origin", xyz=numbers(matrix.ExtractTranslation()), rpy=numbers(rpy(matrix)))


def ancestor_body(prim, bodies):
    while prim and not prim.IsPseudoRoot():
        if str(prim.GetPath()) in bodies:
            return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def mesh_colour(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material:
        for shader_prim in Usd.PrimRange(material.GetPrim()):
            shader = UsdShade.Shader(shader_prim)
            if shader:
                for key in ("diffuse_color_constant", "diffuseColor"):
                    value = shader.GetInput(key).Get()
                    if value is not None:
                        return [*vector(value), 1.0]
    return [0.7, 0.72, 0.75, 1.0]


def export(source, destination):
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise RuntimeError(f"Cannot open USD: {source}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("This exporter expects the source robot's Z-up, metre stage.")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "urdf").mkdir(exist_ok=True)
    (destination / "meshes").mkdir(exist_ok=True)
    cache = UsdGeom.XformCache()
    bodies = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = [UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    tree, closures = [], []
    incoming = {}
    body_frames = {path: Gf.Matrix4d(1) for path in bodies}
    for joint in joints:
        if joint.GetPrim().GetAttribute("physics:excludeFromArticulation").Get():
            closures.append(joint)
        else:
            child = str(joint.GetBody1Rel().GetTargets()[0])
            if child in incoming:
                raise ValueError(f"More than one tree parent for {child}")
            incoming[child] = joint
            body_frames[child] = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            tree.append(joint)
    roots = set(bodies) - set(incoming)
    if len(roots) != 1 or len(tree) != len(bodies) - 1:
        raise ValueError("Expected exactly one connected articulation tree.")
    root = roots.pop()
    if not Gf.IsClose(cache.GetLocalToWorldTransform(bodies[root]), Gf.Matrix4d(1), 1e-7):
        raise ValueError("Root transform must be identity; convert the USD root explicitly first.")

    # local body_frames map exported link coordinates into original USD body coordinates.
    urdf_world = {p: body_frames[p] * cache.GetLocalToWorldTransform(b) for p, b in bodies.items()}
    robot = ET.Element("robot", name="WheelLegRobot")
    robot.append(ET.Comment(" Source: IsaacLab/Robot_USD/Robot_Model.usd; geometry and inertial data preserved. "))
    links, body_records = {}, {}
    for path, body in bodies.items():
        name = body.GetName()
        link = ET.SubElement(robot, "link", name=name)
        links[path] = link
        mass_api = UsdPhysics.MassAPI(body)
        mass = float(mass_api.GetMassAttr().Get())
        com = body_frames[path].GetInverse().Transform(Gf.Vec3d(mass_api.GetCenterOfMassAttr().Get()))
        principal = frame(Gf.Vec3d(0), mass_api.GetPrincipalAxesAttr().Get()) * body_frames[path].GetInverse()
        diagonal = vector(mass_api.GetDiagonalInertiaAttr().Get())
        inertia = [[sum(principal[k][i] * diagonal[k] * principal[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=format(mass, ".12g"))
        ET.SubElement(inertial, "inertia", ixx=format(inertia[0][0], ".12g"), ixy=format(inertia[0][1], ".12g"), ixz=format(inertia[0][2], ".12g"), iyy=format(inertia[1][1], ".12g"), iyz=format(inertia[1][2], ".12g"), izz=format(inertia[2][2], ".12g"))
        if min(diagonal) <= 0 or not math.isclose(sum(inertia[i][i] for i in range(3)), sum(diagonal), rel_tol=1e-10):
            raise ValueError(f"Invalid transformed inertia for {name}")
        body_records[name] = {"mass_kg": mass, "center_of_mass": vector(com), "inertia": inertia, "source_principal_inertia": diagonal,
                              "usd_body_from_urdf_body": {"pos": vector(body_frames[path].ExtractTranslation()), "quat_wxyz": quaternion_wxyz(body_frames[path])},
                              "source_world_pos": vector(urdf_world[path].ExtractTranslation())}

    geometry_hashes, mesh_records = {}, []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        body_path = ancestor_body(prim, bodies)
        if body_path is None:
            continue
        parts = str(prim.GetPath()).split("/")
        kind = "collision" if "collisions" in parts else "visual"
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim) * urdf_world[body_path].GetInverse()
        points = [transform.Transform(Gf.Vec3d(p)) for p in mesh.GetPointsAttr().Get()]
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        triangles, offset = [], 0
        left_handed = mesh.GetOrientationAttr().Get() == "leftHanded"
        for face_index, count in enumerate(counts):
            face = list(indices[offset:offset + count])
            offset += count
            if face_index in holes:
                continue
            if count < 3:
                raise ValueError(f"Degenerate source face in {prim.GetPath()}")
            for i in range(1, count - 1):
                tri = (face[0], face[i], face[i + 1])
                triangles.append(tuple(reversed(tri)) if left_handed else tri)
        content = "# Geometry exported from " + str(prim.GetPath()).replace("/collisions/", "/geometry/").replace("/visuals/", "/geometry/") + "\n"
        content += "".join("v " + numbers(p) + "\n" for p in points)
        content += "".join("f " + " ".join(str(i + 1) for i in tri) + "\n" for tri in triangles)
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest not in geometry_hashes:
            filename = f"{bodies[body_path].GetName()}_{kind}.obj"
            (destination / "meshes" / filename).write_text(content, encoding="utf-8")
            geometry_hashes[digest] = filename
        filename = geometry_hashes[digest]
        element = ET.SubElement(links[body_path], kind)
        ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "mesh", filename=f"../meshes/{filename}", scale="1 1 1")
        if kind == "visual":
            material = ET.SubElement(element, "material", name=f"{bodies[body_path].GetName()}_material")
            ET.SubElement(material, "color", rgba=numbers(mesh_colour(prim)))
        mesh_records.append({"source": str(prim.GetPath()), "file": f"meshes/{filename}", "kind": kind, "vertices": len(points), "triangles": len(triangles)})

    joint_records = []
    for joint in tree:
        p0 = str(joint.GetBody0Rel().GetTargets()[0])
        p1 = str(joint.GetBody1Rel().GetTargets()[0])
        local0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        joint_origin = local0 * body_frames[p0].GetInverse()
        axis_name = joint.GetPrim().GetAttribute("physics:axis").Get()
        axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[axis_name]
        revolute = UsdPhysics.RevoluteJoint(joint.GetPrim())
        lower, upper = float(revolute.GetLowerLimitAttr().Get()), float(revolute.GetUpperLimitAttr().Get())
        continuous = not math.isfinite(lower) and not math.isfinite(upper)
        element = ET.SubElement(robot, "joint", name=joint.GetPrim().GetName(), type="continuous" if continuous else "revolute")
        origin(element, joint_origin)
        ET.SubElement(element, "parent", link=bodies[p0].GetName())
        ET.SubElement(element, "child", link=bodies[p1].GetName())
        ET.SubElement(element, "axis", xyz=numbers(axis))
        # USD drives are deliberately not exported. The runtime drives six active
        # joints; all four knees are passive. Match the repository's actuator limits.
        limit = {"effort": "9", "velocity": "45" if "lun" in joint.GetPrim().GetName() else "30"}
        if not continuous:
            limit.update(lower=format(math.radians(lower), ".12g"), upper=format(math.radians(upper), ".12g"))
        ET.SubElement(element, "limit", **limit)
        ET.SubElement(element, "dynamics", damping="0", friction="0")
        joint_records.append({"name": joint.GetPrim().GetName(), "parent": bodies[p0].GetName(), "child": bodies[p1].GetName(), "origin": joint_origin})

    loop_records = []
    for joint in closures:
        record = {"name": joint.GetPrim().GetName(), "type": "revolute"}
        axis_name = joint.GetPrim().GetAttribute("physics:axis").Get()
        axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[axis_name]
        for i in (0, 1):
            path = str((joint.GetBody0Rel() if i == 0 else joint.GetBody1Rel()).GetTargets()[0])
            local = frame((joint.GetLocalPos0Attr() if i == 0 else joint.GetLocalPos1Attr()).Get(), (joint.GetLocalRot0Attr() if i == 0 else joint.GetLocalRot1Attr()).Get()) * body_frames[path].GetInverse()
            record[f"body{i}"] = bodies[path].GetName()
            record[f"local_pos{i}"] = vector(local.ExtractTranslation())
            record[f"local_axis{i}"] = vector(local.TransformDir(axis).GetNormalized())
            record[f"local_rot{i}_wxyz"] = quaternion_wxyz(local)
        loop_records.append(record)

    try:
        source_reference = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_reference = str(source)
    metadata = {"schema_version": 1, "source_usd": source_reference,
                "urdf": "urdf/Robot_Model.urdf", "units": "m, kg, s, rad", "root_body": bodies[root].GetName(),
                "coordinate_system": "X forward, Y left, Z up; all local constraints use exported URDF link frames",
                "default_root_pose": {"pos": [0, 0, 0.25], "quat_xyzw": [0, 0, 0, 1]},
                "default_joint_positions": {j.GetPrim().GetName(): 0.0 for j in tree},
                "active_joints": ["L_F_Joint1", "L_H_Joint1", "R_F_Joint1", "R_H_Joint1", "L_lun_Joint", "R_lun_Joint"],
                "passive_joints": ["L_F_Joint2", "L_H_Joint2", "R_F_Joint2", "R_H_Joint2"],
                "constraints": loop_records}
    ET.indent(robot, space="  ")
    output_urdf = destination / "urdf" / "Robot_Model.urdf"
    ET.ElementTree(robot).write(output_urdf, encoding="utf-8", xml_declaration=True)
    (destination / "loop_constraints.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Reparse the serialized URDF rather than validate only in-memory transforms.
    parsed = ET.parse(output_urdf).getroot()
    fk = {bodies[root].GetName(): Gf.Matrix4d(1)}
    unresolved = list(parsed.findall("joint"))
    while unresolved:
        progressed = False
        for element in unresolved[:]:
            parent = element.find("parent").get("link")
            if parent not in fk:
                continue
            child = element.find("child").get("link")
            o = element.find("origin")
            fk[child] = from_rpy([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]) * fk[parent]
            unresolved.remove(element)
            progressed = True
        if not progressed:
            raise ValueError("Serialized URDF contains a disconnected tree or cycle")
    checks = []
    for loop in loop_records:
        p0 = fk[loop["body0"]].Transform(Gf.Vec3d(loop["local_pos0"]))
        p1 = fk[loop["body1"]].Transform(Gf.Vec3d(loop["local_pos1"]))
        a0 = fk[loop["body0"]].TransformDir(Gf.Vec3d(loop["local_axis0"])).GetNormalized()
        a1 = fk[loop["body1"]].TransformDir(Gf.Vec3d(loop["local_axis1"])).GetNormalized()
        check = {"name": loop["name"], "anchor_error_m": (p0 - p1).GetLength(), "axis_dot": Gf.Dot(a0, a1), "world_anchor0": vector(p0), "world_anchor1": vector(p1)}
        if check["anchor_error_m"] > 1e-6 or check["axis_dot"] < 0.99999:
            raise ValueError(f"Exported loop does not close at default pose: {check}")
        checks.append(check)
    source_pose_error = max((fk[bodies[path].GetName()].ExtractTranslation() - urdf_world[path].ExtractTranslation()).GetLength() for path in bodies)
    unit_axes = [Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)]
    source_orientation_error = max((fk[bodies[path].GetName()].TransformDir(axis) - urdf_world[path].TransformDir(axis)).GetLength() for path in bodies for axis in unit_axes)
    if source_pose_error > 1e-6:
        raise ValueError(f"Exported body positions differ from USD: {source_pose_error}m")
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "num_bodies": len(bodies), "num_tree_joints": len(tree), "num_loop_joints": len(closures),
              "total_mass_kg": sum(r["mass_kg"] for r in body_records.values()), "max_source_body_position_error_m": source_pose_error,
              "max_source_body_axis_vector_error": source_orientation_error,
              "loop_checks": checks, "bodies": body_records, "meshes": mesh_records,
              "conversion_notes": ["URDF frames moved to incoming child-joint frames; geometry, COM, inertia and closure frames transformed together.", "Source angular limits converted from degrees to radians.", "No source position drives exported; runtime controls the six active joints and leaves four knee joints passive.", "Finite effort/velocity limits match source Python actuator config: 9 Nm, 30 rad/s legs, 45 rad/s wheels.", "Visual and collision geometry exported as triangles at original metre scale; identical meshes share files."]}
    (destination / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("num_bodies", "num_tree_joints", "num_loop_joints", "total_mass_kg", "max_source_body_position_error_m", "loop_checks")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "IsaacLab/Robot_USD/Robot_Model.usd")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "IsaacGym/assets/wheelleg")
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
