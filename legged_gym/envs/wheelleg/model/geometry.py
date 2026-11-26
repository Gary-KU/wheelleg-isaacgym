"""CAD 尺寸的唯一入口（米）；几何、控制与传感器计算共用，禁止各自写死。"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from legged_gym.utils.checkpoints import validate_model_fingerprint

ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets" / "wheelleg"
_METADATA = json.loads((ASSET_ROOT / "loop_constraints.json").read_text())
CAD = _METADATA["cad_geometry"]
ACTUATORS = _METADATA["actuator_limits"]
SIDES = CAD["sides"]
# CAD 左右尺寸有微小导出差异；平面近似取均值，精确初始闭合另由 URDF 校验。
UPPER = sum(sum(s["upper_m"]) for s in SIDES.values()) / 4
LOWER = sum(sum(s["lower_m"]) for s in SIDES.values()) / 4
HIP_SPAN = sum(s["hip_span_m"] for s in SIDES.values()) / 2
HIP_DOWN = sum(s["hip4_down_m"] for s in SIDES.values()) / 2
WHEEL_RADIUS = CAD["tire_radius_m"]
WHEEL_WIDTH = CAD["tire_width_m"]
WHEEL_TRACK = CAD["wheel_track_m"]
MIN_LENGTH, MAX_LENGTH = CAD["recommended_validation_command_range_m"]

# 轮子重量直接由地面承担；髋部支撑前馈按其余刚体重量平分，随 URDF 自动更新。
_URDF = ET.parse(ASSET_ROOT / _METADATA["urdf"]).getroot()
_LINK_MASSES = {
    link.get("name"): float(link.find("inertial/mass").get("value"))
    for link in _URDF.findall("link")
}
TOTAL_MASS_KG = sum(_LINK_MASSES.values())
DEFAULT_SUPPORT_FORCE = (TOTAL_MASS_KG - _LINK_MASSES["L_lun"] - _LINK_MASSES["R_lun"]) * 9.81 / 2


def asset_fingerprint():
    """对全部活动资产求哈希；统一换行符，保证不同系统克隆后的标识一致。"""
    info = json.loads((ASSET_ROOT / "model_info.json").read_text())
    digest = hashlib.sha256()
    for name in sorted(info["files"]):
        path = ASSET_ROOT / Path(name).relative_to("assets/wheelleg")
        digest.update(name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def require_model_checkpoint(state):
    """轮腿专用入口：要求检查点与当前 URDF、网格和闭环元数据完全匹配。"""
    validate_model_fingerprint(state, asset_fingerprint())
