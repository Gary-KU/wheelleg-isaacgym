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

    from .sensor_task import SensorWheelLegged
    from .sensor_config import WheelLeggedSensorCfg, WheelLeggedSensorCfgPPO

    if "wheelleg_sensor" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor",
            SensorWheelLegged,
            WheelLeggedSensorCfg(),
            WheelLeggedSensorCfgPPO(),
        )
