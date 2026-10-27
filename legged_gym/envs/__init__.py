"""One registration per robot task, following the legged_gym convention."""

from legged_gym.utils.task_registry import task_registry
from .wheelleg.wheelleg import WheelLegged
from .wheelleg.wheelleg_config import (
    WheelLeggedCfg,
    WheelLeggedCfgPPO,
    WheelLeggedStandCfg,
    WheelLeggedStandCfgPPO,
    WheelLeggedStairsCfg,
    WheelLeggedStairsCfgPPO,
)

task_registry.register(
    "wheelleg_flat", WheelLegged, WheelLeggedCfg(), WheelLeggedCfgPPO()
)
task_registry.register(
    "wheelleg_stand", WheelLegged, WheelLeggedStandCfg(), WheelLeggedStandCfgPPO()
)
task_registry.register(
    "wheelleg_stairs", WheelLegged, WheelLeggedStairsCfg(), WheelLeggedStairsCfgPPO()
)
