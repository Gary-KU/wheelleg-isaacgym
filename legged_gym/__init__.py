"""Isaac Gym legged_gym task/config/registry interface for this five-bar robot."""

from pathlib import Path
import sys

LEGGED_GYM_ROOT_DIR = str(Path(__file__).resolve().parents[1])
_vendor = str(Path(LEGGED_GYM_ROOT_DIR) / "third_party")
if _vendor not in sys.path:
    sys.path.insert(0, _vendor)
