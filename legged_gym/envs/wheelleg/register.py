"""The only registration list for the built-in wheelleg tasks."""


def register(registry):
    from .wheelleg import WheelLegged
    from .wheelleg_config import (
        WheelLeggedCfg,
        WheelLeggedCfgPPO,
        WheelLeggedStandCfg,
        WheelLeggedStandCfgPPO,
        WheelLeggedStairsCfg,
        WheelLeggedStairsCfgPPO,
    )

    for name, env_cfg, train_cfg in (
        ("wheelleg_flat", WheelLeggedCfg, WheelLeggedCfgPPO),
        ("wheelleg_stand", WheelLeggedStandCfg, WheelLeggedStandCfgPPO),
        ("wheelleg_stairs", WheelLeggedStairsCfg, WheelLeggedStairsCfgPPO),
    ):
        if name not in registry.task_classes:
            registry.register(name, WheelLegged, env_cfg(), train_cfg())
