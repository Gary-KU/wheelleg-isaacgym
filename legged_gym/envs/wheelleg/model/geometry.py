"""Active CAD dimensions in metres, loaded once from the asset's metadata."""

import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets" / "wheelleg"
_METADATA = json.loads((ASSET_ROOT / "loop_constraints.json").read_text())
CAD = _METADATA["cad_geometry"]
SIDES = CAD["sides"]
UPPER = sum(sum(s["upper_m"]) for s in SIDES.values()) / 4
LOWER = sum(sum(s["lower_m"]) for s in SIDES.values()) / 4
HIP_SPAN = sum(s["hip_span_m"] for s in SIDES.values()) / 2
HIP_DOWN = sum(s["hip4_down_m"] for s in SIDES.values()) / 2
WHEEL_RADIUS = CAD["tire_radius_m"]
WHEEL_WIDTH = CAD["tire_width_m"]
MIN_LENGTH, MAX_LENGTH = CAD["recommended_validation_command_range_m"]


def asset_fingerprint():
    """Hash URDF, closure/geometry metadata and every referenced mesh."""
    info = json.loads((ASSET_ROOT / "model_info.json").read_text())
    digest = hashlib.sha256()
    for name in sorted(info["files"]):
        path = ASSET_ROOT / Path(name).relative_to("assets/wheelleg")
        digest.update(name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def require_model_checkpoint(state):
    if state.get("asset_fingerprint") != asset_fingerprint():
        raise ValueError(
            "Checkpoint belongs to a different/unidentified robot model. The CAD URDF was replaced; train a new policy. Use model preview to inspect the new robot."
        )
