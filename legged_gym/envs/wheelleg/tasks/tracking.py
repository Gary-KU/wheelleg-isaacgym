"""运动任务：低速前后/转向，以及独立的直行速度课程。"""

import math
import time

import torch

from legged_gym.envs.wheelleg.tasks.sensors import SensorWheelLegged
from legged_gym.envs.wheelleg.control.commands import slew_commands
from legged_gym.envs.wheelleg.diagnostics.training_metrics import (
    WheelLegTrainingMetrics,
)


class DrivingWheelLegged(SensorWheelLegged):
    """训练/键盘共用速度斜坡；可选伸腿站立修正，不覆盖网络力矩。"""

    @staticmethod
    def configure_execution(cfg, args, checkpoint_state):
        SensorWheelLegged.configure_execution(cfg, args, checkpoint_state)
        if checkpoint_state is not None:
            scales = checkpoint_state.get("legged_gym_env_config", {}).get("rewards", {}).get("scales", {})
            for name in ("standstill_position", "standstill_heading"):
                if name not in scales:
                    setattr(cfg.rewards.scales, name, 0.0)  # 旧检查点沿用原奖励

    def _validate_observation_config(self, cfg):
        super()._validate_observation_config(cfg)
        rate = cfg.commands.leg_length_rate
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("leg_length_rate must be finite and nonnegative")
        if not isinstance(cfg.control.encoder_hold, bool):
            raise ValueError("encoder_hold must be boolean")
        if cfg.control.encoder_hold and cfg.control.standing_height_trim_ki:
            raise ValueError("Encoder hold and wheel-rate integral trim are alternative controllers")

    def _allocate_buffers(self):
        super()._allocate_buffers()
        self.standstill_reward = None
        r = self.cfg.rewards
        if r.scales.standstill_position or r.scales.standstill_heading:
            from legged_gym.envs.wheelleg.rewards.standstill import StandstillReward
            self.standstill_reward = StandstillReward(
                self.num_envs, self.device, self.dt, r.standstill_settle_seconds,
                r.standstill_position_sigma, r.standstill_heading_sigma)
        self.height_standing_trim = None
        self.encoder_hold = self.wheel_position_sampler = None
        if self.cfg.control.encoder_hold:
            from legged_gym.envs.wheelleg.control.encoder_hold import EncoderHold
            from legged_gym.envs.wheelleg.observations.odometry import EncoderPositionSampler
            c, s = self.cfg.control, self.cfg.sensors
            self.encoder_hold = EncoderHold(
                self.num_envs, self.device, self.cfg.keyboard.leg_range[0],
                c.encoder_hold_position_kp, c.encoder_hold_heading_kp,
                c.encoder_hold_speed_limit, c.encoder_hold_yaw_limit, c.encoder_hold_height_slope,
                c.encoder_hold_turn_reference_scale)
            self.wheel_position_sampler = EncoderPositionSampler(
                self.num_envs, self.device, s.delay_steps,
                s.wheel_angle_noise_std, s.wheel_angle_bias_std, self.cfg.seed)
        if self.cfg.control.standing_height_trim_ki:
            from legged_gym.envs.wheelleg.control.standing_trim import HeightStandingTrim
            low, high = self.cfg.keyboard.leg_range
            self.height_standing_trim = HeightStandingTrim(
                self.num_envs, self.device, low, high-low,
                self.cfg.control.standing_height_trim_ki,
                self.cfg.control.standing_height_trim_limit,
                self.cfg.control.standing_height_trim_initial_coefficient)

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        self.commands[ids, :3] = 0.0
        if self.cfg.commands.leg_length_rate > 0:
            # 物理重置姿态是初始腿长；随机高度目标也要经过伸缩斜坡。
            self.commands[ids, 3] = self.initial_leg_length
        if self.height_standing_trim is not None:
            self.height_standing_trim.reset(ids)
        if self.standstill_reward is not None:
            self.standstill_reward.reset(ids)
        if self.encoder_hold is not None:
            self.encoder_hold.reset(ids)

    def _policy_commands(self):
        if self.encoder_hold is not None:
            reference = self.commands.clone()
            forward, yaw = self.encoder_hold.corrections(self.commands, self.command_targets)
            reference[:, 0] += forward
            reference[:, 2] += yaw
            return reference
        if self.height_standing_trim is None:
            return self.commands
        reference = self.commands.clone()
        reference[:, 0] += self.height_standing_trim.correction(self.commands)
        return reference

    def _observations(self, advance_history=True):
        resetting = (self.sensor_reset_pending.clone()
                     if self.height_standing_trim is not None or self.encoder_hold is not None else None)
        obs = super()._observations(advance_history)
        if self.encoder_hold is not None:
            angles = self.dof_state[:, self.action_indices[4:], 0]
            ids = resetting.nonzero(as_tuple=False).flatten()
            if len(ids):
                self.wheel_position_sampler.reset(ids, angles)
            if advance_history:
                self.wheel_position_sampler.advance(angles)
            observed = torch.ones_like(resetting) if advance_history else resetting
            self.encoder_hold.update(
                self.sensor_sampler.sample, self.wheel_position_sampler.sample,
                self.commands, self.command_targets, observed)
            self._refresh_command_observations()
        if self.height_standing_trim is not None and advance_history:
            self.height_standing_trim.update(
                self.sensor_sampler.sample, self.commands, ~resetting, self.dt)
            self._refresh_command_observations()
        return obs

    def _extra_step_info(self):
        info = super()._extra_step_info()
        if self.encoder_hold is not None:
            info['pre_reset_policy_commands'] = self._policy_commands().clone()
            info['pre_reset_encoder_odometry_m'] = self.encoder_hold.odometry.distance.clone()
            info['pre_reset_encoder_heading_rad'] = self.encoder_hold.odometry.yaw.clone()
        if self.standstill_reward is not None:
            info["pre_reset_standstill_drift_m"] = self.standstill_reward.displacement.clone()
            info["pre_reset_standstill_heading_rad"] = self.standstill_reward.heading_error.clone()
        if self.height_standing_trim is not None:
            info['pre_reset_policy_commands'] = self._policy_commands().clone()
            info['pre_reset_standing_height_trim_m_s'] = self.height_standing_trim.correction(self.commands)
        return info

    def _sample_command_targets(self, ids):
        # 对应方向键单按、组合按和松键；每个回合仍通过同一速度斜坡。
        speed = max(abs(x) for x in self.forward_speed_range)
        turn = max(abs(x) for x in self.yaw_speed_range)
        choices = self.commands.new_tensor(
            [
                [speed, 0],
                [-speed, 0],
                [0, turn],
                [0, -turn],
                [speed, turn],
                [speed, -turn],
                [-speed, turn],
                [-speed, -turn],
            ]
        )
        selection = choices[torch.randint(8, (len(ids),), device=self.device)]
        self.command_targets[ids] = self.default_command
        self.command_targets[ids, 0] = selection[:, 0]
        self.command_targets[ids, 2] = selection[:, 1]
        stopped = torch.rand(len(ids), device=self.device) < self.zero_command_fraction
        self.command_targets[ids[stopped], :3] = 0.0
        low, high = self.leg_length_range
        if high > low:
            # 包含中间高度及两个端点；旧的固定高度任务不多消耗随机数。
            heights = low + (high - low) * torch.rand(len(ids), device=self.device)
            endpoints = torch.randint(4, (len(ids),), device=self.device)
            heights = torch.where(endpoints == 0, low, heights)
            heights = torch.where(endpoints == 1, high, heights)
            self.command_targets[ids, 3] = heights

    def set_commands(self, commands):
        previous = self.commands.clone()
        super().set_commands(commands)  # 校验并更新请求目标
        self.commands.copy_(previous)  # 请求不是瞬时速度，留到控制步末做斜坡
        self._refresh_command_observations()
        return self.obs_buf

    def _advance_random_commands(self, dones):
        rate = self.cfg.commands.leg_length_rate
        height = self.commands[:, 3].clone() if rate > 0 else None
        if self.random_commands:
            resamples = super()._advance_random_commands(dones)
        else:
            self.commands.copy_(
                slew_commands(
                    self.commands,
                    self.command_targets,
                    self.dt,
                    self.linear_acceleration,
                    self.yaw_acceleration,
                )
            )
            resamples = 0
        if rate > 0:
            # 训练和播放使用同一腿长变化率，不能用瞬时改高度代替 Q/E。
            limited = height + (self.command_targets[:, 3] - height).clamp(
                -rate * self.dt, rate * self.dt)
            self.commands[:, 3] = torch.where(dones, height, limited)
        self._refresh_command_observations()
        return resamples

    def _compute_reward_terms(self, terminated):
        terms = super()._compute_reward_terms(terminated)
        if self.standstill_reward is not None:
            terms.update(self.standstill_reward.update(
                self.root_states[:, :3], self.yaw, self.commands, self.command_targets))
        r = self.cfg.rewards
        if hasattr(r, "lateral_error_scale"):
            # 仿真真值只用于打分，仍不进入 IMU/编码器策略观测。
            terms["lateral_speed"] = (
                self.lin_vel[:, 1] / r.lateral_error_scale
            ).square().clamp(max=25)
            terms["speed_error"] = (
                (self.lin_vel[:, 0] - self.commands[:, 0]) / r.speed_error_scale
            ).square().clamp(max=25)
            terms["yaw_rate_error"] = (
                (self.ang_vel[:, 2] - self.commands[:, 2]) / r.yaw_error_scale
            ).square().clamp(max=25)
        return terms

    def _reward_lateral_speed(self):
        return self._task_reward_terms["lateral_speed"]

    def _reward_speed_error(self):
        return self._task_reward_terms["speed_error"]

    def _reward_yaw_rate_error(self):
        return self._task_reward_terms["yaw_rate_error"]

    def _reward_standstill_position(self):
        return self._task_reward_terms["standstill_position"]

    def _reward_standstill_heading(self):
        return self._task_reward_terms["standstill_heading"]

    def runtime_control_contract(self):
        """报告额外控制状态的来源，避免把轮角外环误称为纯132维策略。"""
        c, s = self.cfg.control, self.cfg.sensors
        return dict(
            actor_input_size=self.num_obs,
            encoder_position_hold=bool(c.encoder_hold),
            wheel_rate_integral_trim_ki=c.standing_height_trim_ki,
            additional_measurements=(['left_wheel_angle_rad', 'right_wheel_angle_rad']
                                     if c.encoder_hold else []),
            wheel_angle_noise_std_rad=s.wheel_angle_noise_std if c.encoder_hold else None,
            wheel_angle_bias_std_rad=s.wheel_angle_bias_std if c.encoder_hold else None,
            measurement_delay_seconds=s.delay_steps * self.dt,
            position_kp_s_inv=c.encoder_hold_position_kp if c.encoder_hold else None,
            heading_kp_s_inv=c.encoder_hold_heading_kp if c.encoder_hold else None,
            turn_reference_scale=c.encoder_hold_turn_reference_scale if c.encoder_hold else None,
            physics_solver=self.physics_solver,
            height_speed_slope_s_inv=c.encoder_hold_height_slope if c.encoder_hold else None,
            odometry_assumption='Planar rolling; carrier rotation and leg translation compensated',
            world_pose_or_velocity_feedback=False,
        )


class TrackingWheelLegged(SensorWheelLegged):
    @staticmethod
    def configure_execution(cfg, args, checkpoint_state):
        if (
            args.mode == "train"
            and checkpoint_state is not None
            and "curriculum" not in checkpoint_state.get("legged_gym_env_config", {})
        ):
            raise ValueError(
                "Use --warm-start to enter tracking from a balance checkpoint; --resume restores the old task config"
            )
        SensorWheelLegged.configure_execution(cfg, args, checkpoint_state)
        if args.mode == "play":
            cfg.curriculum.enabled = False

    def _allocate_buffers(self):
        super()._allocate_buffers()
        self.speed_levels = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.tracking_sums = torch.zeros(self.num_envs, 3, device=self.device)
        self.tracking_counts = torch.zeros(self.num_envs, device=self.device)
        self.assessment_running = False
        self.assessment_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.assessment_rms = torch.zeros(self.num_envs, 3, device=self.device)
        self.assessment_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.assessment_failures = torch.zeros_like(self.assessment_success)
        self.last_assessment_iteration = 0
        self.tracking_failed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _compute_reward_terms(self, terminated):
        terms = super()._compute_reward_terms(terminated)
        c, r = self.commands, self.cfg.rewards
        errors = torch.stack(
            (
                self.lin_vel[:, 0] - c[:, 0],
                self.lin_vel[:, 1],
                self.ang_vel[:, 2] - c[:, 2],
            ),
            -1,
        )
        terms.update(
            speed_error=(errors[:, 0] / r.speed_error_scale).square().clamp(max=25),
            lateral_speed=(errors[:, 1] / r.lateral_error_scale).square().clamp(max=25),
            yaw_rate_error=(errors[:, 2] / r.yaw_error_scale).square().clamp(max=25),
            stopped_wheel_speed=(
                self.dof_state[:, self.action_indices[4:], 1] / r.wheel_speed_scale
            )
            .square()
            .mean(-1)
            .clamp(max=25)
            * (c[:, :3].abs().max(-1).values < 0.02),
        )
        # Only judge steady targets, after acceleration/initial settling.
        steady = ((c[:, 0] - self.command_targets[:, 0]).abs() < 0.01) & (
            self.episode_length_buf * self.dt >= self.cfg.curriculum.settle_seconds
        )
        self.tracking_failed |= terminated
        self.tracking_sums += errors.square() * steady[:, None]
        self.tracking_counts += steady
        return terms

    def _reset_idx(self, ids):
        if (
            self.cfg.curriculum.enabled
            and self.assessment_running
            and len(ids)
            and bool((self.episode_length_buf[ids] > 0).any())
        ):
            cc = self.cfg.curriculum
            rms = (
                self.tracking_sums[ids] / self.tracking_counts[ids, None].clamp_min(1)
            ).sqrt()
            moving_trial = (self.speed_levels[ids] == 0) | (
                self.command_targets[ids, 0] > 0.01
            )
            success = (
                (self.episode_length_buf[ids] >= self.max_episode_length)
                & (self.tracking_counts[ids] >= cc.minimum_samples)
                & moving_trial
            )
            velocity_limit = torch.where(
                self.speed_levels[ids] == 0,
                cc.stationary_velocity_rms_limit,
                cc.velocity_rms_limit,
            )
            success &= rms[:, 0] < velocity_limit
            success &= (rms[:, 1] < cc.lateral_rms_limit) & (
                rms[:, 2] < cc.yaw_rate_rms_limit
            )
            success &= self.total_yaw[ids].abs() < cc.heading_error_limit
            success &= ~self.tracking_failed[ids]
            success &= (
                (self.pitch[ids].abs() < 0.15)
                & (self.roll[ids].abs() < 0.10)
                & self.geometry_valid[ids]
            )
            # Demote early physical failures; explicit/manual resets don't count.
            graded = self.assessment_active[ids]
            failed = self.tracking_failed[ids] & graded
            success &= graded
            self.assessment_rms[ids] = torch.where(
                graded[:, None], rms, self.assessment_rms[ids]
            )
            self.assessment_success[ids] |= success
            self.assessment_failures[ids] |= failed
            self.assessment_active[ids] = False
            self.speed_levels[ids] = (
                self.speed_levels[ids] + success.long() - failed.long()
            ).clamp(0, len(cc.speeds) - 1)
        self.tracking_failed[ids] = False
        self.tracking_sums[ids] = 0
        self.tracking_counts[ids] = 0
        super()._reset_idx(ids)
        if not self.cfg.curriculum.enabled and self.linear_acceleration > 0:
            self.commands[ids, :2] = 0.0  # Playback starts through the trained ramp.

    def _advance_random_commands(self, dones):
        if not self.cfg.curriculum.enabled and not self.random_commands:
            self.commands.copy_(
                slew_commands(
                    self.commands,
                    self.command_targets,
                    self.dt,
                    self.linear_acceleration,
                    self.yaw_acceleration,
                )
            )
            self._refresh_command_observations()
            return 0
        return super()._advance_random_commands(dones)

    def _sample_command_targets(self, ids):
        speeds = self.commands.new_tensor(self.cfg.curriculum.speeds)[
            self.speed_levels[ids]
        ]
        stopped = (
            torch.rand(len(ids), device=self.device)
            < self.cfg.commands.zero_command_fraction
        )
        if self.assessment_running:
            stopped[:] = False  # Every evaluation tests the current speed stage.
        self.command_targets[ids] = self.default_command
        self.command_targets[ids, 0] = torch.where(
            stopped, torch.zeros_like(speeds), speeds
        )

    def assess_curriculum(self, policy, iteration):
        """Grade deterministic actions outside PPO storage, then reset for fresh data."""
        cc = self.cfg.curriculum
        if (
            not cc.enabled
            or self.last_assessment_iteration == iteration
            or (iteration != 1 and iteration % cc.assessment_interval)
        ):
            return None
        print(
            "Starting curriculum assessment with deterministic actions (up to one full episode).",
            flush=True,
        )
        started = time.perf_counter()
        previous_training = policy.training
        before = self.speed_levels.clone()
        self.reset()  # Discard the training rollout state before grading.
        self.assessment_running = True
        self.assessment_active[:] = True
        self.assessment_success[:] = False
        self.assessment_failures[:] = False
        self.assessment_rms[:] = 0
        self._sample_command_targets(torch.arange(self.num_envs, device=self.device))
        steps = 0
        try:
            policy.eval()
            with torch.inference_mode():
                for _ in range(self.max_episode_length):
                    obs, _ = self.get_observations()
                    self.step(policy.act_inference(obs))
                    steps += 1
                    if not bool(self.assessment_active.any()):
                        break
            self.last_assessment_iteration = iteration
            return {
                "Assessment/promotions": int((self.speed_levels > before).sum()),
                "Assessment/demotions": int((self.speed_levels < before).sum()),
                "Assessment/physical_failures": int(self.assessment_failures.sum()),
                "Assessment/mean_vx_error_m_s": float(self.assessment_rms[:, 0].mean()),
                "Assessment/mean_lateral_speed_m_s": float(
                    self.assessment_rms[:, 1].mean()
                ),
                "Assessment/mean_yaw_rate_error_rad_s": float(
                    self.assessment_rms[:, 2].mean()
                ),
                "Assessment/max_speed_m_s": float(
                    self.commands.new_tensor(cc.speeds)[self.speed_levels].max()
                ),
                "Curriculum/mean_speed_level": float(self.speed_levels.float().mean()),
                "Curriculum/max_speed_m_s": float(
                    self.commands.new_tensor(cc.speeds)[self.speed_levels].max()
                ),
                "Assessment/steps": steps,
                "Assessment/wall_seconds": time.perf_counter() - started,
            }
        finally:
            self.assessment_running = False
            self.assessment_active[:] = False
            policy.train(previous_training)
            self.reset()  # Next PPO action must see this new observation, not pre-test data.

    def _reward_speed_error(self):
        return self._task_reward_terms["speed_error"]

    def _reward_lateral_speed(self):
        return self._task_reward_terms["lateral_speed"]

    def _reward_yaw_rate_error(self):
        return self._task_reward_terms["yaw_rate_error"]

    def _reward_stopped_wheel_speed(self):
        return self._task_reward_terms["stopped_wheel_speed"]

    def create_training_metrics(self):
        return TrackingMetrics(self)


class TrackingMetrics(WheelLegTrainingMetrics):
    def reset(self):
        super().reset()
        self.velocity_sums = self.env.commands.new_zeros(2)

    def update(self, observations, rewards, dones, infos):
        super().update(observations, rewards, dones, infos)
        velocity = self.env.training_body_linear_velocity(infos)
        self.velocity_sums += torch.stack(
            (velocity[:, 0].mean(), velocity[:, 1].square().mean())
        )

    def compute(self, num_steps):
        metrics = super().compute(num_steps)
        metrics.update(
            {
                "Motion/mean_vx_m_s": float(self.velocity_sums[0] / num_steps),
                "Motion/rms_lateral_velocity_m_s": float(
                    (self.velocity_sums[1] / num_steps).sqrt()
                ),
                "Curriculum/mean_speed_level": float(
                    self.env.speed_levels.float().mean()
                ),
                "Curriculum/max_speed_m_s": float(
                    self.env.commands.new_tensor(self.env.cfg.curriculum.speeds)[
                        self.env.speed_levels
                    ].max()
                ),
            }
        )
        return metrics

    def checkpoint_metadata(self):
        return dict(
            super().checkpoint_metadata(),
            speed_curriculum_levels=self.env.speed_levels.cpu().tolist(),
            curriculum_assessment_iteration=self.env.last_assessment_iteration,
        )

    def restore_checkpoint(self, state):
        super().restore_checkpoint(state)
        self.env.last_assessment_iteration = state.get(
            "curriculum_assessment_iteration", 0
        )
        saved = state.get("speed_curriculum_levels")
        if saved is not None:
            levels = torch.tensor(saved, device=self.env.device, dtype=torch.long)
            if levels.numel():
                ids = (
                    torch.arange(self.env.num_envs, device=self.env.device)
                    % levels.numel()
                )
                self.env.speed_levels.copy_(
                    levels[ids].clamp(0, len(self.env.cfg.curriculum.speeds) - 1)
                )
                self.env.reset()
