"""Robot/task overrides: change settings here instead of editing the base robot."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class WheelLeggedCfg(LeggedRobotCfg):
    pass


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    pass


class WheelLeggedStandCfg(WheelLeggedCfg):
    class commands(WheelLeggedCfg.commands):
        random_commands = False
        forward_speed_range = (0.0, 0.0)
        yaw_speed_range = (0.0, 0.0)
        leg_length_range = (0.08, 0.08)
        command_resample_seconds = 0.0


class WheelLeggedStandCfgPPO(WheelLeggedCfgPPO):
    class runner(WheelLeggedCfgPPO.runner):
        experiment_name = "wheelleg_stand"


class WheelLeggedStairsCfg(WheelLeggedCfg):
    class env(WheelLeggedCfg.env):
        num_observations = 252
        initial_leg_length = 0.14

    class control(WheelLeggedCfg.control):
        leg_control = "hip_target"
        wheel_control = "policy"
        heading_kp = 6.0
        heading_kd = 0.3

    class commands(WheelLeggedCfg.commands):
        command = (0.25, 0.0, 0.0, 0.14)
        random_commands = False
        forward_speed_range = (0.25, 0.25)
        yaw_speed_range = (0.0, 0.0)
        leg_length_range = (0.14, 0.14)
        command_resample_seconds = 0.0
        linear_acceleration = 0.5

    class observations(WheelLeggedCfg.observations):
        observe_terrain = True

    class terrain(WheelLeggedCfg.terrain):
        terrain = "stairs"
        stair_curriculum = True

    class rewards(WheelLeggedCfg.rewards):
        class scales:
            progress = 1.0
            velocity = 1.0
            yaw = 1.0
            posture = 1.0
            angular_damping = 1.0
            stall = 1.0
            base_contact = 1.0
            passive_limit = 1.0
            effort = 1.0
            torque_smoothness = 1.0
            failure = 1.0
            success = 1.0
            heading = 1.0


class WheelLeggedStairsCfgPPO(WheelLeggedCfgPPO):
    class runner(WheelLeggedCfgPPO.runner):
        experiment_name = "wheelleg_stairs"
