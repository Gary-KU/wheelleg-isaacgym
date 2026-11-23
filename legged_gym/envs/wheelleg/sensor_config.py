"""Opt-in MCU-obtainable observations; source dimensions/motor model are unchanged."""

from .wheelleg_config import WheelLeggedCfg, WheelLeggedCfgPPO
from .sensor_observations import (
    FRAME_FIELDS,
    HISTORY_FRAMES,
    OBSERVATION_SIZE,
    SENSOR_SCALES,
    SENSOR_SCHEMA,
)


class WheelLeggedSensorCfg(WheelLeggedCfg):
    class env(WheelLeggedCfg.env):
        num_observations = OBSERVATION_SIZE

    class sensors:
        schema = SENSOR_SCHEMA
        frame_fields = FRAME_FIELDS
        sensor_scales = SENSOR_SCALES
        history_frames = HISTORY_FRAMES
        # 20 ms at the default policy rate; an assumed training latency.
        delay_steps = 1
        # These are example training perturbations, NOT measured BMI088 statistics.
        # Order and units: sensor_observations.SENSOR_FIELDS; all values are SI.
        noise_std = (
            0.002,
            0.01,
            0.01,
            0.002,
            0.002,
            0.0005,
            0.0005,
            0.02,
            0.02,
            0.005,
            0.005,
            0.05,
            0.05,
        )
        bias_std = (
            0.002,
            0.005,
            0.005,
            0.002,
            0.002,
            0.0003,
            0.0003,
            0.0,
            0.0,
            0.0,
            0.0,
            0.02,
            0.02,
        )

    class control(WheelLeggedCfg.control):
        # Avoid the old heading PD's privileged absolute-yaw feedback. The
        # policy controls both wheels using observable pitch and yaw rate.
        wheel_control = "policy"
        heading_kp = 0.0
        heading_kd = 0.0


class WheelLeggedSensorCfgPPO(WheelLeggedCfgPPO):
    class runner(WheelLeggedCfgPPO.runner):
        experiment_name = "wheelleg_sensor"
        pretrained_checkpoint = None  # 198-input legacy policies are incompatible.
