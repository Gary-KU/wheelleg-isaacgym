"""Convert the repository URDF to MJCF and restore its two closed hinges."""

from pathlib import Path
import json
import math
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/wheelleg"


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def _origin_pose(origin):
    """URDF fixed-axis roll/pitch/yaw, as MJCF position and wxyz quaternion."""
    if origin is None:
        return "0 0 0", np.array([1.0, 0.0, 0.0, 0.0])
    roll, pitch, yaw = [float(v) / 2 for v in origin.get("rpy", "0 0 0").split()]
    cr, cp, cy = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sr, sp, sy = math.sin(roll), math.sin(pitch), math.sin(yaw)
    return origin.get("xyz", "0 0 0"), np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _restore_source_precision(root, urdf):
    """Avoid the six-digit rounding of mj_saveLastXML for CAD frames/inertias.

    Body frames are unchanged because fusestatic is disabled. Reapply the source
    values rather than compensating small loop errors with altered constraints.
    """
    for link in urdf.findall("link"):
        body = root.find(".//body[@name='%s']" % link.get("name"))
        source = link.find("inertial")
        if source is None:
            continue
        inertial = body.find("inertial")
        position, quaternion = _origin_pose(source.find("origin"))
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        values = source.find("inertia").attrib
        xx, yy, zz, xy, xz, yz = [
            float(values[n]) for n in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ]
        tensor = (
            rotation @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rotation.T
        )
        inertial.attrib.clear()
        inertial.set("pos", position)
        inertial.set("mass", source.find("mass").get("value"))
        inertial.set(
            "fullinertia",
            _numbers(
                [
                    tensor[0, 0],
                    tensor[1, 1],
                    tensor[2, 2],
                    tensor[0, 1],
                    tensor[0, 2],
                    tensor[1, 2],
                ]
            ),
        )
    for source in urdf.findall("joint"):
        body = root.find(".//body[@name='%s']" % source.find("child").get("link"))
        position, quaternion = _origin_pose(source.find("origin"))
        body.set("pos", position)
        body.set("quat", _numbers(quaternion))
        joint = body.find("joint[@name='%s']" % source.get("name"))
        if joint is None:
            continue
        joint.set("axis", source.find("axis").get("xyz"))
        if source.get("type") == "revolute":
            limit = source.find("limit")
            joint.set("range", limit.get("lower") + " " + limit.get("upper"))


def build(
    output,
    terrain="flat",
    wave_height=0.005,
    wave_length=0.4,
    flight_imu=False,
    ramp_height=0.03,
):
    if terrain not in ("flat", "wave", "ramp"):
        raise ValueError("terrain must be flat, wave or ramp")
    if not all(math.isfinite(v) and v > 0 for v in (wave_height, wave_length)):
        raise ValueError("wave dimensions must be finite and positive")
    if not math.isfinite(ramp_height) or ramp_height <= 0:
        raise ValueError("ramp_height must be finite and positive")
    urdf = ET.parse(ASSET / "urdf/Robot_Model.urdf")
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str(ASSET / "meshes" / Path(mesh.get("filename")).name))
    ext = ET.SubElement(urdf.getroot(), "mujoco")
    ET.SubElement(ext, "compiler", discardvisual="false", fusestatic="false")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "robot.urdf"
        urdf.write(path)
        m = mujoco.MjModel.from_xml_path(str(path))
        mujoco.mj_saveLastXML(str(Path(tmp) / "robot.xml"), m)
        tree = ET.parse(Path(tmp) / "robot.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ASSET / "meshes"))
    _restore_source_precision(root, urdf.getroot())
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", ".0001")
    world = root.find("worldbody")
    base = world.find("body[@name='base_link']")
    ET.SubElement(base, "freejoint", name="root")
    for geom in world.findall(".//geom"):
        # URDF imports preserve visual-only meshes with both contact masks zero.
        # In particular their convex hulls may fill gaps which collision meshes
        # deliberately leave empty; enabling them changes the physical robot.
        visual_only = (
            geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0"
        )
        if not visual_only:
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
    ET.SubElement(
        world,
        "geom",
        contype="2",
        conaffinity="1",
        name="floor",
        type="plane",
        size="5 5 .1",
        friction="1 0.005 0.0001",
    )
    if terrain == "wave":
        # Height is peak-to-trough; keep 0.4 m of flat approach around reset.
        x = np.linspace(-5.0, 5.0, 2001)
        phase = np.maximum(np.abs(x) - 0.4, 0.0)
        profile = 0.5 * (1 - np.cos(2 * np.pi * phase / wave_length))
        elevation = np.tile(profile, (41, 1))
        ET.SubElement(
            root.find("asset"),
            "hfield",
            name="wave",
            nrow="41",
            ncol="2001",
            size=f"5 2 {wave_height} .05",
            elevation=_numbers(elevation.ravel()),
        )
        floor = world.find("geom[@name='floor']")
        floor.set("type", "hfield")
        floor.set("hfield", "wave")
        floor.attrib.pop("size")
    if terrain == "ramp":
        # 0.4m approach; 0.3m ascent to a 3cm lip, followed by a drop.
        ET.SubElement(
            root.find("asset"),
            "mesh",
            name="ramp_mesh",
            vertex=f".4 -1 0 .4 1 0 .7 -1 0 .7 1 0 .7 -1 {ramp_height} .7 1 {ramp_height}",
            face="0 1 5 0 5 4 0 4 2 1 3 5 2 4 5 2 5 3 0 2 3 0 3 1",
        )
        ET.SubElement(
            world,
            "geom",
            name="ramp",
            type="mesh",
            mesh="ramp_mesh",
            contype="2",
            conaffinity="1",
            friction="1 0.005 0.0001",
            rgba=".4 .6 .4 1",
        )
    if flight_imu:
        ET.SubElement(base, "site", name="flight_imu", pos="0 0 0", size=".001")
        ET.SubElement(
            ET.SubElement(root, "sensor"),
            "accelerometer",
            name="flight_accel",
            site="flight_imu",
        )
    ET.SubElement(world, "light", pos="0 -2 3")
    metadata = json.loads((ASSET / "loop_constraints.json").read_text())
    equality = ET.SubElement(root, "equality")
    # Two point constraints along the hinge axis leave exactly one rotational DOF.
    for c in metadata["constraints"]:
        for end in (0, 1):
            b = root.find(".//body[@name='%s']" % c["body%d" % end])
            for n in (0, 1):
                pos = [
                    p + n * 0.02 * a
                    for p, a in zip(c["local_pos%d" % end], c["local_axis%d" % end])
                ]
                ET.SubElement(
                    b,
                    "site",
                    name=c["name"] + "_%s_%s" % (end, n),
                    pos=" ".join(map(str, pos)),
                    size=".001",
                )
        for n in (0, 1):
            ET.SubElement(
                equality,
                "connect",
                solref=".002 1",
                solimp=".99 .99 .001",
                site1=c["name"] + "_0_%s" % n,
                site2=c["name"] + "_1_%s" % n,
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output)
    return output


if __name__ == "__main__":
    p = build(ROOT / "assets/wheelleg/mujoco/robot.xml")
    m = mujoco.MjModel.from_xml_path(str(p))
    print(p, "mass", sum(m.body_mass), "joints", m.njnt, "constraints", m.neq)
