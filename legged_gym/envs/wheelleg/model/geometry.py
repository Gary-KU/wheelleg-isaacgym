"""CAD 尺寸的唯一入口（米）；几何、控制与传感器计算共用，禁止各自写死。"""

import hashlib
import json
from pathlib import Path

from legged_gym.utils.checkpoints import validate_model_fingerprint

ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets" / "wheelleg"
_METADATA = json.loads((ASSET_ROOT / "loop_constraints.json").read_text())
CAD = _METADATA["cad_geometry"]
SIDES = CAD["sides"]
# CAD 左右尺寸有微小导出差异；平面近似取均值，精确初始闭合另由 URDF 校验。
UPPER = sum(sum(s["upper_m"]) for s in SIDES.values()) / 4
LOWER = sum(sum(s["lower_m"]) for s in SIDES.values()) / 4
HIP_SPAN = sum(s["hip_span_m"] for s in SIDES.values()) / 2
HIP_DOWN = sum(s["hip4_down_m"] for s in SIDES.values()) / 2
WHEEL_RADIUS = CAD["tire_radius_m"]
WHEEL_WIDTH = CAD["tire_width_m"]
MIN_LENGTH, MAX_LENGTH = CAD["recommended_validation_command_range_m"]


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
