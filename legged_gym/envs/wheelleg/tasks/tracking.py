"""Straight-speed curriculum; all extra truth is used for reward/evaluation only."""

import time

import torch

from legged_gym.envs.wheelleg.tasks.sensors import SensorWheelLegged
from legged_gym.envs.wheelleg.control.commands import slew_commands
from legged_gym.envs.wheelleg.diagnostics.training_metrics import WheelLegTrainingMetrics


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
