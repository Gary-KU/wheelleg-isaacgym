"""Six-value Wheel-Legged-Gym task API over the preserved five-bar physics."""

import math
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.wheelleg.model.geometry import require_model_checkpoint
from legged_gym.envs.wheelleg.robot import ASSET_ROOT, WheelLegRobot
from legged_gym.utils.helpers import class_to_dict
import torch
from legged_gym.envs.wheelleg.rewards.flat import flat_reward_terms
from legged_gym.envs.wheelleg.rewards.stairs import stair_task_reward_terms


# 嵌套任务配置到物理环境构造参数的唯一映射。
CONFIG_SECTIONS = {
    "env": ("num_envs", "spacing", "initial_leg_length", "fix_base"),
    "sim": ("physics_solver", "physics_threads"),
    "control": (
        "leg_control",
        "leg_kp",
        "leg_kd",
        "leg_feedforward_force",
        "hip_target_scale",
        "wheel_control",
        "heading_kp",
        "heading_kd",
        "wheel_pitch_damping",
        "wheel_balance_scale",
    ),
    "commands": (
        "command",
        "random_commands",
        "forward_speed_range",
        "yaw_speed_range",
        "leg_length_range",
        "command_resample_seconds",
        "linear_acceleration",
        "yaw_acceleration",
        "zero_command_fraction",
    ),
    "observations": ("observe_leg_lengths", "observe_terrain"),
    "asset": ("wheel_friction",),
    "closure": ("allow_approximate_loop", "closure_stiffness", "closure_damping"),
    "terrain": (
        "terrain",
        "stair_level",
        "stair_min_level",
        "stair_max_level",
        "stair_curriculum",
        "stair_episode_seconds",
    ),
    "viewer": ("viewer_show_collisions", "viewer_width", "viewer_height", "viewer_fps", "viewer_wall_clock"),
    "rewards": ("tracking_penalties", "stability_rewards", "support_rewards"),
}


def environment_kwargs(cfg):
    values = {
        name: getattr(getattr(cfg, section), name)
        for section, names in CONFIG_SECTIONS.items()
        for name in names
    }
    values.update(
        seed=cfg.seed,
        physics_dt=cfg.sim.dt,
        control_dt=cfg.sim.dt * cfg.control.decimation,
    )
    return values


class WheelLegged(WheelLegRobot):
    """Task owns config/reward registration; base owns real physics and resets."""

    # 模型专用校验由任务提供，共用注册器无需知道机器人资产路径。
    validate_checkpoint = staticmethod(require_model_checkpoint)

    @staticmethod
    def configure_execution(cfg, args, checkpoint_state):
        """Wheelleg-only playback/CLI rules stay outside the generic registry."""
        if checkpoint_state is not None:
            saved = checkpoint_state.get("legged_gym_env_config", {}).get("env", {})
            if "spacing" not in saved:
                cfg.env.spacing = 1.5  # 历史检查点保留原网格布局
        if args.viewer_fps is not None:
            cfg.viewer.viewer_fps = args.viewer_fps
        # 显式参数覆盖检查点里的速度斜坡；物理步长和电机限幅仍由原配置决定。
        for name in ("linear_acceleration", "yaw_acceleration"):
            value = getattr(args, name, None)
            if value is not None:
                setattr(cfg.commands, name, value)
        if args.command is not None:
            cfg.commands.command = args.command
        if args.mode == "play":
            cfg.commands.random_commands = False
            cfg.commands.command_resample_seconds = 0.0
            cfg.terrain.stair_curriculum = False
        if args.stair_level is not None:
            if cfg.terrain.terrain != "stairs":
                raise ValueError("--stair_level requires a stair task")
            cfg.terrain.stair_level = args.stair_level

    def __init__(
        self, cfg, sim_params=None, physics_engine=None, sim_device="cpu", headless=True
    ):
        from isaacgym import gymapi

        if physics_engine not in (None, gymapi.SIM_PHYSX):
            raise ValueError("Five-bar closure currently supports PhysX only")
        if sim_params is not None:
            raise ValueError(
                "Configure physics through cfg.sim; external SimParams are not supported"
            )
        root = cfg.asset.root.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        if (
            Path(root).resolve() != ASSET_ROOT
            or cfg.asset.file != "urdf/Robot_Model.urdf"
        ):
            raise ValueError(
                "This task requires the supplied five-bar asset and matching closure metadata"
            )
        self.cfg = cfg
        self.environment_config = environment_kwargs(cfg)
        self.environment_config.update(sim_device=sim_device, headless=headless)
        self._validate_observation_config(cfg)
        self._prepare_reward_function()
        super().__init__(**self.environment_config)
        self.obs_history_length = 1  # obs_buf already contains all six frames
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.extras = {}

    def _validate_observation_config(self, cfg):
        expected = (
            31
            + 2 * cfg.observations.observe_leg_lengths
            + 9 * cfg.observations.observe_terrain
        ) * 6
        if (
            cfg.env.num_observations != expected
            or cfg.env.num_actions != 6
            or cfg.env.num_privileged_obs is not None
            or cfg.env.obs_history_length != 1
        ):
            raise ValueError(
                "Config observation/action/history dimensions do not match the preserved layout"
            )

    def _prepare_reward_function(self):
        self.reward_scales = {}
        self.reward_functions = []
        for name, scale in class_to_dict(self.cfg.rewards.scales).items():
            if not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise ValueError("Reward scale must be finite: " + name)
            if scale:
                function = getattr(self, "_reward_" + name, None)
                if function is None:
                    raise ValueError("Missing _reward_" + name)
                self.reward_scales[name] = scale
                self.reward_functions.append((name, function))

    def _compute_reward_terms(self, terminated):
        return (
            stair_task_reward_terms(self, terminated)
            if self.terrain_model is not None
            else flat_reward_terms(self, terminated)
        )

    def _rewards(self, terminated):
        self._task_reward_terms = self._compute_reward_terms(terminated)
        self.reward_terms = {
            name: self.reward_scales[name] * function()
            for name, function in self.reward_functions
        }
        total = sum(
            self.reward_terms.values(), torch.zeros(self.num_envs, device=self.device)
        )
        return total.clamp(min=0) if self.cfg.rewards.only_positive_rewards else total

    def step(self, actions):
        obs, privileged, rewards, dones, info = super().step(actions)
        self.rew_buf = rewards
        self.extras = info
        return obs, privileged, rewards, dones, info, obs

    def get_observations(self):
        return self.obs_buf, self.obs_buf

    def reset_idx(self, env_ids):
        self._reset_idx(env_ids)
        self._kinematics()
        self._observations(advance_history=False)

    # Explicit methods keep reward registration searchable in editors. All
    # equations are evaluated once in rewards/flat.py / rewards/stairs.py above.

    def _reward_termination(self):
        return self._task_reward_terms["termination"]

    def _reward_tracking_lin_vel(self):
        return self._task_reward_terms["tracking_lin_vel"]

    def _reward_direction(self):
        return self._task_reward_terms["direction"]

    def _reward_tracking_ang_vel(self):
        return self._task_reward_terms["tracking_ang_vel"]

    def _reward_lin_vel_z(self):
        return self._task_reward_terms["lin_vel_z"]

    def _reward_effort(self):
        return self._task_reward_terms["effort"]

    def _reward_action_rate(self):
        return self._task_reward_terms["action_rate"]

    def _reward_leg_tracking(self):
        return self._task_reward_terms["leg_tracking"]

    def _reward_leg_error(self):
        return self._task_reward_terms["leg_error"]

    def _reward_leg_asymmetry(self):
        return self._task_reward_terms["leg_asymmetry"]

    def _reward_passive_limit(self):
        return self._task_reward_terms["passive_limit"]

    def _reward_hip_pose(self):
        return self._task_reward_terms["hip_pose"]

    def _reward_upright(self):
        return self._task_reward_terms["upright"]

    def _reward_angular_motion(self):
        return self._task_reward_terms["angular_motion"]

    def _reward_hip_symmetry(self):
        return self._task_reward_terms["hip_symmetry"]

    def _reward_leg_angle(self):
        return self._task_reward_terms["leg_angle"]

    def _reward_heading(self):
        return self._task_reward_terms["heading"]

    def _reward_tracking_error(self):
        return self._task_reward_terms["tracking_error"]

    def _reward_stability(self):
        return self._task_reward_terms["stability"]

    def _reward_yaw_error(self):
        return self._task_reward_terms["yaw_error"]

    def _reward_progress(self):
        return self._task_reward_terms["progress"]

    def _reward_velocity(self):
        return self._task_reward_terms["velocity"]

    def _reward_yaw(self):
        return self._task_reward_terms["yaw"]

    def _reward_posture(self):
        return self._task_reward_terms["posture"]

    def _reward_angular_damping(self):
        return self._task_reward_terms["angular_damping"]

    def _reward_stall(self):
        return self._task_reward_terms["stall"]

    def _reward_base_contact(self):
        return self._task_reward_terms["base_contact"]

    def _reward_torque_smoothness(self):
        return self._task_reward_terms["torque_smoothness"]

    def _reward_failure(self):
        return self._task_reward_terms["failure"]

    def _reward_success(self):
        return self._task_reward_terms["success"]
