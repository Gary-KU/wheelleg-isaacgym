"""The only registration list for the built-in wheelleg tasks."""


def register(registry):
    from legged_gym.envs.wheelleg.tasks.base import WheelLegged
    from legged_gym.envs.wheelleg.config.base import (
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

    from legged_gym.envs.wheelleg.tasks.sensors import SensorWheelLegged
    from legged_gym.envs.wheelleg.config.sensors import (
        WheelLeggedSensorCfg,
        WheelLeggedSensorCfgPPO,
    )

    if "wheelleg_sensor" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor",
            SensorWheelLegged,
            WheelLeggedSensorCfg(),
            WheelLeggedSensorCfgPPO(),
        )

    from legged_gym.envs.wheelleg.tasks.tracking import TrackingWheelLegged
    from legged_gym.envs.wheelleg.config.sensors import (
        WheelLeggedTrackingCfg,
        WheelLeggedTrackingCfgPPO,
    )

    if "wheelleg_sensor_tracking" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor_tracking",
            TrackingWheelLegged,
            WheelLeggedTrackingCfg(),
            WheelLeggedTrackingCfgPPO(),
        )

    # 候选奖励使用独立名称；原 wheelleg_sensor 不因试验而改变默认值。
    from legged_gym.envs.wheelleg.config.sensors import (
        WheelLeggedQuietStandCfg,
        WheelLeggedQuietStandCfgPPO,
    )

    if "wheelleg_sensor_quiet" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor_quiet",
            SensorWheelLegged,
            WheelLeggedQuietStandCfg(),
            WheelLeggedQuietStandCfgPPO(),
        )

    from legged_gym.envs.wheelleg.tasks.tracking import DrivingWheelLegged
    from legged_gym.envs.wheelleg.config.sensors import (
        WheelLeggedDriveCfg,
        WheelLeggedDriveCfgPPO,
        WheelLeggedTelescopicCfg,
        WheelLeggedTelescopicCfgPPO,
        WheelLeggedHeightStandCfg,
        WheelLeggedHeightStandCfgPPO,
    )

    if "wheelleg_sensor_drive" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor_drive",
            DrivingWheelLegged,
            WheelLeggedDriveCfg(),
            WheelLeggedDriveCfgPPO(),
        )

    if "wheelleg_sensor_telescopic" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor_telescopic",
            DrivingWheelLegged,
            WheelLeggedTelescopicCfg(),
            WheelLeggedTelescopicCfgPPO(),
        )

    if "wheelleg_sensor_height_stand" not in registry.task_classes:
        registry.register(
            "wheelleg_sensor_height_stand",
            DrivingWheelLegged,
            WheelLeggedHeightStandCfg(),
            WheelLeggedHeightStandCfgPPO(),
        )
