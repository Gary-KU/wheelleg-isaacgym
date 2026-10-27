"""Robot defaults and task overrides: the one place to edit normal runs.

The small exploration noise here is intentional. Legacy command-line training
keeps its historical TRAIN_CONFIG profile for old experiment reproduction.
"""

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.wheelleg.constants import FLAT_REWARD_SCALES


class WheelLeggedCfg(BaseConfig):
    seed = 42

    class env:
        num_envs = 64
        num_observations = 198  # 33 values x 6 frames; do not stack a second time
        num_privileged_obs = None
        num_actions = 6
        obs_history_length = 1
        initial_leg_length = 0.08
        fix_base = False

    class sim:
        dt = 0.0005
        physics_solver = "tgs"
        physics_threads = 4

    class control:
        decimation = 40
        leg_control = "hip_pd"
        leg_kp = 80.0
        leg_kd = 4.0
        leg_feedforward_force = 22.4649
        hip_target_scale = 0.2
        wheel_control = "heading"
        heading_kp = 6.0
        heading_kd = 0.6
        wheel_pitch_damping = 0.0
        wheel_balance_scale = 1.0

    class commands:
        command = (0.0, 0.0, 0.0, 0.08)
        random_commands = True
        forward_speed_range = (-0.6, 0.6)
        yaw_speed_range = (-2.0, 2.0)
        leg_length_range = (0.08, 0.12)
        command_resample_seconds = 4.0
        linear_acceleration = 0.8
        yaw_acceleration = 2.0
        zero_command_fraction = 0.2

    class observations:
        observe_leg_lengths = True
        observe_terrain = False

    class asset:
        root = "{LEGGED_GYM_ROOT_DIR}/assets/wheelleg"
        file = "urdf/Robot_Model.urdf"
        wheel_friction = 1.0

    class closure:
        allow_approximate_loop = True
        closure_stiffness = 50000.0
        closure_damping = 25.0

    class terrain:
        terrain = "plane"
        stair_level = 0
        stair_min_level = 0
        stair_max_level = 4
        stair_curriculum = False
        stair_episode_seconds = 24.0

    class viewer:
        viewer_show_collisions = False
        viewer_width = 1280
        viewer_height = 720

    class rewards:
        # Preserve the existing per-control-step reward convention (no extra dt).
        tracking_penalties = True
        stability_rewards = True
        support_rewards = True
        only_positive_rewards = False

        class scales:
            termination = FLAT_REWARD_SCALES["termination"]
            tracking_lin_vel = FLAT_REWARD_SCALES["tracking_lin_vel"]
            direction = FLAT_REWARD_SCALES["direction"]
            tracking_ang_vel = FLAT_REWARD_SCALES["tracking_ang_vel"]
            lin_vel_z = FLAT_REWARD_SCALES["lin_vel_z"]
            effort = FLAT_REWARD_SCALES["effort"]
            action_rate = FLAT_REWARD_SCALES["action_rate"]
            leg_tracking = FLAT_REWARD_SCALES["leg_tracking"]
            leg_error = FLAT_REWARD_SCALES["leg_error"]
            leg_asymmetry = FLAT_REWARD_SCALES["leg_asymmetry"]
            passive_limit = FLAT_REWARD_SCALES["passive_limit"]
            hip_pose = FLAT_REWARD_SCALES["hip_pose"]
            upright = FLAT_REWARD_SCALES["upright"]
            angular_motion = FLAT_REWARD_SCALES["angular_motion"]
            hip_symmetry = FLAT_REWARD_SCALES["hip_symmetry"]
            leg_angle = FLAT_REWARD_SCALES["leg_angle"]
            heading = FLAT_REWARD_SCALES["heading"]
            tracking_error = FLAT_REWARD_SCALES["tracking_error"]
            stability = FLAT_REWARD_SCALES["stability"]
            yaw_error = FLAT_REWARD_SCALES["yaw_error"]


class WheelLeggedCfgPPO(BaseConfig):
    seed = 42
    initialization_provenance = None

    class policy:
        init_noise_std = [0.03, 0.03, 0.03, 0.03, 0.25, 0.25]
        actor_hidden_dims = [512, 256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 5
        learning_rate = 0.001
        max_learning_rate = 0.01
        schedule = "adaptive"
        gamma = 0.995
        lam = 0.95
        desired_kl = 0.008
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 25
        save_interval = 100
        max_iterations = 20000
        experiment_name = "wheelleg_flat"
        pretrained_checkpoint = (
            "logs/standing_support_20261022_093823/input/standing_hold.pt"
        )
        run_name = ""


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
        pretrained_checkpoint = (
            "logs/stair_training_20261022_103417/evaluation/verified_level0.pt"
        )
