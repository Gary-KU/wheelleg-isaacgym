"""沿用同一物理模型，策略只看 IMU/编码器可获取状态；奖励仍可使用仿真真值。"""

import torch

from legged_gym.envs.wheelleg.tasks.base import WheelLegged
from legged_gym.envs.wheelleg.model.kinematics import inverse_symmetric
from legged_gym.envs.wheelleg.model.constraints import quat_rotate_inverse
from legged_gym.envs.wheelleg.observations.sensors import (
    COMMAND_SLICE,
    COMMAND_SCALES,
    FRAME_SIZE,
    FRAME_FIELDS,
    HISTORY_FRAMES,
    OBSERVATION_SIZE,
    SENSOR_SCHEMA,
    SENSOR_SCALES,
    SensorSampler,
    build_sensor_frame,
    encoder_leg_state,
)


class SensorWheelLegged(WheelLegged):
    @staticmethod
    def configure_execution(cfg, args, checkpoint_state):
        WheelLegged.configure_execution(cfg, args, checkpoint_state)
        if checkpoint_state is not None:
            saved = checkpoint_state.get("legged_gym_env_config", {}).get("env", {})
            if "balance_initial_com" not in saved:
                cfg.env.balance_initial_com = False  # 旧检查点保持原始重置姿态

    def _load_robot(self, stiffness, damping):
        if self.cfg.env.balance_initial_com:
            from legged_gym.envs.wheelleg.model.initial_pose import make_initial_pose
            from legged_gym.envs.wheelleg.model.geometry import ASSET_ROOT

            pose = make_initial_pose(
                ASSET_ROOT / self.metadata["urdf"],
                self.metadata,
                self.initial_leg_length,
                balance_initial_com=True,
            )
            self.metadata["default_root_pose"]["quat_xyzw"] = pose[
                "root_quaternion_xyzw"
            ]
            self.initial_com_balance_diagnostics = pose["com_balance"]
        super()._load_robot(stiffness, damping)

    def _compute_reward_terms(self, terminated):
        # The legacy reward helper indexes its own 31-column frame for four
        # terms. Reconstruct those terms from physical state, not sensor columns.
        terms = super()._compute_reward_terms(terminated)
        l1, l4 = inverse_symmetric(self.length_l)
        r1, r4 = inverse_symmetric(self.length_r)
        pose_errors = torch.stack(
            (self.phi_l1 - l1, self.phi_l4 - l4, self.phi_r1 - r1, self.phi_r4 - r4), -1
        )
        terms.update(
            lin_vel_z=(self.lin_vel[:, 2] * 1.5).square(),
            hip_pose=pose_errors.abs().sum(-1),
            upright=torch.exp(
                -(self.pitch.square() + self.roll.square())
                / self.cfg.rewards.upright_sigma**2
            ),
            tracking_lin_vel=torch.exp(
                -(self.commands[:, :2] - self.lin_vel[:, :2]).square().sum(-1)
                / self.cfg.rewards.linear_tracking_sigma**2
            ),
            tracking_ang_vel=torch.exp(
                -(self.commands[:, 2] - self.ang_vel[:, 2]).square()
                / self.cfg.rewards.yaw_tracking_sigma**2
            ),
            angular_motion=(self.ang_vel[:, :2] * 0.5).square().sum(-1),
            stopped_wheel_speed=(self.dof_state[:, self.action_indices[4:], 1] / 3.0)
            .square()
            .mean(-1)
            .clamp(max=25)
            * (self.commands[:, :3].abs().max(-1).values < 0.02),
        )
        return terms

    def _reward_stopped_wheel_speed(self):
        return self._task_reward_terms["stopped_wheel_speed"]

    def _extra_step_info(self):
        info = super()._extra_step_info()
        # 重置前采样，避免把倒地后重置的零轮速计入评估。
        info["pre_reset_active_rates"] = self.dof_state[
            :, self.action_indices, 1
        ].clone()
        return info

    def training_body_linear_velocity(self, infos):
        # Logging may use terminal-safe simulation truth; actor inputs may not.
        roots = infos["pre_reset_root_states"]
        return quat_rotate_inverse(roots[:, 3:7], roots[:, 7:10])

    def _validate_observation_config(self, cfg):
        if not isinstance(cfg.env.balance_initial_com, bool):
            raise ValueError("balance_initial_com must be boolean")
        if cfg.env.balance_initial_com and cfg.env.initial_leg_length is None:
            raise ValueError(
                "COM-balanced reset requires an explicit initial leg length"
            )
        if (
            cfg.env.num_observations != OBSERVATION_SIZE
            or cfg.env.num_actions != 6
            or cfg.env.num_privileged_obs is not None
            or cfg.env.obs_history_length != 1
            or cfg.sensors.schema != SENSOR_SCHEMA
            or tuple(cfg.sensors.frame_fields) != FRAME_FIELDS
            or tuple(cfg.sensors.sensor_scales) != SENSOR_SCALES
            or cfg.sensors.history_frames != HISTORY_FRAMES
            or cfg.observations.observe_terrain
        ):
            raise ValueError(
                "Sensor task requires the 132-input mylegwheel_planar_v1 layout"
            )
        if cfg.commands.command[1] != 0:
            raise ValueError(
                "Planar sensor task does not support a lateral velocity command"
            )
        if (
            cfg.control.wheel_control != "policy"
            or cfg.control.heading_kp
            or cfg.control.heading_kd
        ):
            raise ValueError(
                "Sensor task cannot use privileged absolute-yaw heading feedback"
            )

    def _allocate_buffers(self):
        self.observation_frame_size, self.num_obs = FRAME_SIZE, OBSERVATION_SIZE
        super()._allocate_buffers()
        s = self.cfg.sensors
        self.sensor_sampler = SensorSampler(
            self.num_envs,
            self.device,
            s.delay_steps,
            s.noise_std,
            s.bias_std,
            self.cfg.seed,
        )
        self.sensor_reset_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        self.sensor_reset_pending[ids] = True

    def _sensor_truth(self):
        # Only IMU-equivalent orientation/rates and active encoder measurements.
        # No root position/linear velocity, contact forces, terrain scan or yaw angle.
        legs = encoder_leg_state(
            self.dof_state[:, self.action_indices[:4], 0],
            self.dof_state[:, self.action_indices[:4], 1],
        )
        wheel = self.dof_state[
            :, self.action_indices[4:], 1
        ] * self.commands.new_tensor([-1.0, 1.0])
        return torch.cat(
            (
                self.pitch[:, None],
                self.ang_vel[:, 1:2],
                self.ang_vel[:, 2:3],
                legs[:, :, 0],
                legs[:, :, 1],
                legs[:, :, 2],
                legs[:, :, 3],
                wheel,
            ),
            -1,
        )

    def _observations(self, advance_history=True):
        if bool(self.sensor_reset_pending.any()) or advance_history:
            truth = self._sensor_truth()
            ids = self.sensor_reset_pending.nonzero(as_tuple=False).flatten()
            if len(ids):
                self.sensor_sampler.reset(ids, truth)
                self.sensor_reset_pending[ids] = False
            if advance_history:
                self.sensor_sampler.advance(truth)
        self.slice_obs = build_sensor_frame(
            self.sensor_sampler.sample, self.commands, self.actions
        )
        self.obs_buf = torch.cat(
            (self.history.reshape(self.num_envs, -1), self.slice_obs), -1
        )
        if advance_history:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = self.slice_obs
        return self.obs_buf

    def _refresh_command_observations(self):
        # Commands are current; editing one never resamples sensors or shifts history.
        self.slice_obs[:, COMMAND_SLICE] = self.commands[
            :, [0, 2, 3]
        ] * self.commands.new_tensor(COMMAND_SCALES)
        self.obs_buf[:, -FRAME_SIZE:] = self.slice_obs
        self.history[:, -1] = self.slice_obs

    def set_commands(self, commands):
        value = torch.as_tensor(commands, device=self.device)
        if (
            value.ndim in (1, 2)
            and value.shape[-1] == 4
            and bool((value[..., 1] != 0).any())
        ):
            raise ValueError(
                "Planar sensor task does not support a lateral velocity command"
            )
        return super().set_commands(commands)
