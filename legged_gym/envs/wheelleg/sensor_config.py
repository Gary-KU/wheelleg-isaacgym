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

    class commands(WheelLeggedCfg.commands):
        # Stage one: balance before velocity, turning or height tracking.
        random_commands = False
        command = (0.0, 0.0, 0.0, 0.08)
        forward_speed_range = (0.0, 0.0)
        yaw_speed_range = (0.0, 0.0)
        leg_length_range = (0.08, 0.08)
        command_resample_seconds = 0.0

    class rewards(WheelLeggedCfg.rewards):
        upright_sigma = 0.10  # rad; exp(-(pitch^2 + roll^2)/sigma^2)
        linear_tracking_sigma = 0.20  # m/s
        yaw_tracking_sigma = 0.40  # rad/s

        class scales(WheelLeggedCfg.rewards.scales):
            # Keep early return magnitudes manageable for the untrained critic.
            termination = -10.0
            upright = 2.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            leg_tracking = 0.5
            hip_symmetry = 0.0
            leg_angle = 0.0
            heading = 0.0  # Absolute yaw is absent from this task's observation.

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
    class policy(WheelLeggedCfgPPO.policy):
        init_noise_std = [0.08, 0.08, 0.08, 0.08, 0.20, 0.20]
        actor_output_scale = 0.01

    class algorithm(WheelLeggedCfgPPO.algorithm):
        learning_rate = 0.0001
        max_learning_rate = 0.0001
        num_learning_epochs = 3
        num_mini_batches = 4
        entropy_coef = 0.001

    class runner(WheelLeggedCfgPPO.runner):
        experiment_name = "wheelleg_sensor_balance"
        pretrained_checkpoint = None  # 198-input legacy policies are incompatible.
