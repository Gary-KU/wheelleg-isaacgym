"""实机可获取状态的任务配置：观测/噪声 → 站立训练 → 速度课程。"""

from legged_gym.envs.wheelleg.config.base import WheelLeggedCfg, WheelLeggedCfgPPO
from legged_gym.envs.wheelleg.constants import MIN_LEG_LENGTH, MAX_LEG_LENGTH
from legged_gym.envs.wheelleg.observations.sensors import (
    FRAME_FIELDS,
    HISTORY_FRAMES,
    OBSERVATION_SIZE,
    SENSOR_SCALES,
    SENSOR_SCHEMA,
)


class WheelLeggedSensorCfg(WheelLeggedCfg):
    class env(WheelLeggedCfg.env):
        num_observations = OBSERVATION_SIZE
        balance_initial_com = False

    class commands(WheelLeggedCfg.commands):
        # 起步先学习零速支撑，再由 tracking 任务逐级增加速度。
        random_commands = False
        command = (0.0, 0.0, 0.0, 0.08)
        forward_speed_range = (0.0, 0.0)
        yaw_speed_range = (0.0, 0.0)
        leg_length_range = (0.08, 0.08)
        command_resample_seconds = 0.0

    class rewards(WheelLeggedCfg.rewards):
        upright_sigma = 0.10  # rad; exp(-(pitch^2 + roll^2)/sigma^2)
        upright_about_com = False  # True仅改训练打分，不给策略输入重心真值
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
            stopped_wheel_speed = 0.0  # 候选安静站立任务可启用；旧任务行为保持不变

    class sensors:
        schema = SENSOR_SCHEMA
        frame_fields = FRAME_FIELDS
        sensor_scales = SENSOR_SCALES
        history_frames = HISTORY_FRAMES
        wheel_angle_noise_std = .001  # rad；可选轮角里程的假设噪声，不是实测规格
        wheel_angle_bias_std = .01  # rad；每回合固定，角度差会消去固定零偏
        # 20 ms at the default policy rate; an assumed training latency.
        delay_steps = 1
        # These are example training perturbations, NOT measured BMI088 statistics.
        # Order and units: observations/sensors.py: SENSOR_FIELDS; all values are SI.
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
        # 1.454 kg 新模型的短步对照：降低旧增益，减少轻连杆的限幅振荡。
        leg_kp = 2.0
        leg_kd = 0.04
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
        save_interval = 25  # 短间隔保存，方便随时评估并续训
        pretrained_checkpoint = None  # 198-input legacy policies are incompatible.


class WheelLeggedTrackingCfg(WheelLeggedSensorCfg):
    """Refine stopping, then advance straight speed only after successful episodes."""

    class commands(WheelLeggedSensorCfg.commands):
        random_commands = True
        command_resample_seconds = 0.0  # One stop/straight target per episode.
        linear_acceleration = 0.4
        zero_command_fraction = 0.25

    class curriculum:
        enabled = True
        speeds = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        velocity_rms_limit = 0.08
        stationary_velocity_rms_limit = 0.02
        heading_error_limit = 0.10
        lateral_rms_limit = 0.02
        yaw_rate_rms_limit = 0.06
        settle_seconds = 2.0
        minimum_samples = 300
        assessment_interval = 100  # Separate deterministic episodes, not PPO samples.

    class rewards(WheelLeggedSensorCfg.rewards):
        linear_tracking_sigma = 0.10
        yaw_tracking_sigma = 0.08
        speed_error_scale = 0.20
        lateral_error_scale = 0.05
        yaw_error_scale = 0.10
        wheel_speed_scale = 2.0  # shaft rad/s; soft stop cost, never a wheel lock

        class scales(WheelLeggedSensorCfg.rewards.scales):
            speed_error = -2.0
            lateral_speed = -1.0
            yaw_rate_error = -0.5
            stopped_wheel_speed = -0.2


class WheelLeggedTrackingCfgPPO(WheelLeggedSensorCfgPPO):
    class algorithm(WheelLeggedSensorCfgPPO.algorithm):
        learning_rate = 0.00003
        max_learning_rate = 0.00003
        num_learning_epochs = 2

    class runner(WheelLeggedSensorCfgPPO.runner):
        experiment_name = "wheelleg_sensor_tracking_v2"


class WheelLeggedQuietStandCfg(WheelLeggedSensorCfg):
    """安静站立候选：独立实验，未通过验收前不替代基础任务默认值。"""

    class env(WheelLeggedSensorCfg.env):
        balance_initial_com = True
        num_envs = 32  # 同原点 GPU 广相容量已验证的并行数量
        spacing = 0.0  # 小尺寸高刚度闭环避免网格远原点的浮点精度损失

    class control(WheelLeggedSensorCfg.control):
        decimation = 10  # 2 kHz 物理，200 Hz 策略与传感器历史更新

    class rewards(WheelLeggedSensorCfg.rewards):
        linear_tracking_sigma = 0.08
        yaw_tracking_sigma = 0.12

        class scales(WheelLeggedSensorCfg.rewards.scales):
            termination = -25.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            leg_tracking = 2.0
            leg_error = -0.5
            tracking_error = 4.0
            yaw_error = -2.0
            stopped_wheel_speed = -0.02  # 软惩罚，不锁轮，非零运动指令下关闭


class WheelLeggedQuietStandCfgPPO(WheelLeggedSensorCfgPPO):
    class policy(WheelLeggedSensorCfgPPO.policy):
        # 从无振荡的物理模型重新学习，先减少探索对静态平衡的干扰。
        init_noise_std = [0.02, 0.02, 0.02, 0.02, 0.08, 0.08]

    class algorithm(WheelLeggedSensorCfgPPO.algorithm):
        entropy_coef = 0.0002

    class runner(WheelLeggedSensorCfgPPO.runner):
        experiment_name = "wheelleg_sensor_quiet"


class WheelLeggedDriveCfg(WheelLeggedQuietStandCfg):
    """已验证站立物理配置上的低速前后/转向任务。"""

    class commands(WheelLeggedQuietStandCfg.commands):
        random_commands = True
        forward_speed_range = (-0.3, 0.3)
        yaw_speed_range = (-0.8, 0.8)
        command_resample_seconds = 3.0
        linear_acceleration = 0.4
        yaw_acceleration = 1.8
        zero_command_fraction = 0.3
        leg_length_rate = 0.0  # m/s；0 保持旧任务的即时高度指令

    class control(WheelLeggedQuietStandCfg.control):
        decimation = 4  # 2 kHz 物理 / 500 Hz 策略，缩短转向反馈延迟
        leg_kd = 0.08  # Nm·s/rad，增加左右腿横滚模态阻尼；刚度仍为2
        standing_height_trim_ki = 0.0  # 伸腿静止修正默认关闭，旧权重行为不变
        standing_height_trim_limit = 0.08  # m/s，策略速度参考修正上限
        standing_height_trim_initial_coefficient = 0.0  # s^-1，可选高度偏置标定；旧权重为0
        encoder_hold = False  # 可选轮角里程外环；与轮速积分修正互斥
        encoder_hold_position_kp = .5  # s^-1，停车位置误差→速度参考
        encoder_hold_heading_kp = 1.0  # s^-1，直行相对航向误差→转速参考
        encoder_hold_turn_reference_scale = 1.0  # 转向参考标定，默认保持旧策略
        encoder_hold_gyro_heading = False  # 停车标定陀螺零偏，运动航向不依赖轮胎滚动
        sensor_pitch_rate_kd = 0.0  # Nm*s/rad，IMU俯仰角速度反馈到轮力矩，非物理阻尼
        sensor_pitch_guard_kp = 0.0  # Nm/rad，仅在俯仰超过软区间后增加轮子追赶力矩
        sensor_pitch_guard_angle = .16  # rad，软件反馈起点，不是姿态钳制或验收门槛
        encoder_hold_speed_limit = .08  # m/s，含高度标定的总参考修正限幅
        encoder_hold_yaw_limit = .2  # rad/s，转向指令非零时不叠加航向修正
        encoder_hold_height_slope = 0.0  # s^-1，离线标定的高度速度参考初值

    class rewards(WheelLeggedQuietStandCfg.rewards):
        standstill_position_sigma = 0.05  # m；仅训练打分，不向Actor提供位置
        standstill_heading_sigma = 0.10  # rad；相对本次停稳点
        standstill_settle_seconds = 2.0

        class scales(WheelLeggedQuietStandCfg.rewards.scales):
            standstill_position = 0.0
            standstill_heading = 0.0

    class keyboard:
        speed = 0.3
        turn_speed = 0.8
        adjust_leg_length = False  # 本轮固定 8 cm，只训练移动和停车
        leg_range = (MIN_LEG_LENGTH, MAX_LEG_LENGTH)
        leg_speed = 0.02  # m/s，按仿真时间连续调整；检查点可保存较慢的速度


class WheelLeggedDriveCfgPPO(WheelLeggedQuietStandCfgPPO):
    class policy(WheelLeggedQuietStandCfgPPO.policy):
        init_noise_std = [0.005] * 4 + [0.015, 0.015]

    class algorithm(WheelLeggedQuietStandCfgPPO.algorithm):
        # 200 → 500 Hz，保持折扣与 GAE 衰减对应的物理时长。
        gamma = 0.995 ** (200.0 / 500.0)
        lam = 0.95 ** (200.0 / 500.0)
        learning_rate = 0.000003
        max_learning_rate = 0.00001
        schedule = "fixed"
        num_learning_epochs = 2
        entropy_coef = 0.00001

    class runner(WheelLeggedQuietStandCfgPPO.runner):
        experiment_name = "wheelleg_sensor_drive"
        num_steps_per_env = 250  # 500 Hz 下每轮仍采样 0.5 s


class WheelLeggedTelescopicCfg(WheelLeggedDriveCfg):
    """8–10 cm 变高度运动候选；独立训练和验收，不覆盖已保存的播放配置。"""

    class commands(WheelLeggedDriveCfg.commands):
        forward_speed_range = (-0.7, 0.7)
        yaw_speed_range = (-1.0, 1.0)
        leg_length_range = (0.08, 0.10)
        leg_length_rate = 0.01
        command_resample_seconds = 4.0
        linear_acceleration = 0.8
        yaw_acceleration = 2.5
        zero_command_fraction = 0.5

    class control(WheelLeggedDriveCfg.control):
        leg_differential_kd = 0.08

    class keyboard(WheelLeggedDriveCfg.keyboard):
        speed = 0.7
        turn_speed = 1.0
        adjust_leg_length = True
        leg_range = (0.08, 0.10)
        leg_speed = 0.01

    class rewards(WheelLeggedDriveCfg.rewards):
        upright_sigma = 0.15  # 允许加减速时适量俯仰，避免奖励要求与平衡冲突
        upright_about_com = True
        speed_error_scale = 0.20
        lateral_error_scale = 0.05
        yaw_error_scale = 0.10

        class scales(WheelLeggedDriveCfg.rewards.scales):
            speed_error = -0.5
            lateral_speed = -0.5
            yaw_rate_error = -0.1
            stopped_wheel_speed = -0.002  # 允许必要的平衡修正，不鼓励只锁轮而机体滑动


class WheelLeggedTelescopicCfgPPO(WheelLeggedDriveCfgPPO):
    class algorithm(WheelLeggedDriveCfgPPO.algorithm):
        learning_rate = 0.00001
        max_learning_rate = 0.00001
        critic_learning_rate = 0.001
        use_clipped_value_loss = False  # fresh Critic 不受每轮0.2的旧值变化限制

    class runner(WheelLeggedDriveCfgPPO.runner):
        experiment_name = "wheelleg_sensor_telescopic"


class WheelLeggedHeightStandCfg(WheelLeggedTelescopicCfg):
    """变高度课程第一阶段：每回合保持一个高度，先学会长期停车。"""

    class env(WheelLeggedTelescopicCfg.env):
        episode_length_s = 80.0  # 原16秒回合不足以暴露长时间的缓慢漂移

    class commands(WheelLeggedTelescopicCfg.commands):
        forward_speed_range = (0., 0.)
        yaw_speed_range = (0., 0.)
        zero_command_fraction = 1.
        command_resample_seconds = 0.  # 整个80秒回合保持同一高度

    class rewards(WheelLeggedTelescopicCfg.rewards):
        speed_error_scale = .01
        lateral_error_scale = .01
        yaw_error_scale = .03

        class scales(WheelLeggedTelescopicCfg.rewards.scales):
            speed_error = -1.
            yaw_rate_error = -.2
            standstill_position = -.5
            standstill_heading = -.1


class WheelLeggedHeightStandCfgPPO(WheelLeggedTelescopicCfgPPO):
    class runner(WheelLeggedTelescopicCfgPPO.runner):
        experiment_name = "wheelleg_sensor_height_stand"
