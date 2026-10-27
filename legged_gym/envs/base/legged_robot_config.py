"""Nested legged_gym configuration; units and motor semantics stay unchanged."""

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
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
            termination = -100.0
            tracking_lin_vel = 3.0
            direction = 1.0
            tracking_ang_vel = 2.0
            lin_vel_z = -0.2
            effort = -0.0001
            action_rate = -0.003
            leg_tracking = 2.0
            leg_error = -0.25
            leg_asymmetry = -0.5
            passive_limit = -4.0
            hip_pose = -0.1
            upright = 6.0
            angular_motion = -0.02
            hip_symmetry = 0.7
            leg_angle = 0.5
            heading = -1.0
            tracking_error = 1.0
            stability = 1.0
            yaw_error = -0.5


class LeggedRobotCfgPPO(BaseConfig):
    seed = 42

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
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
