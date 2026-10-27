"""Wheel-leg diagnostics and checkpoint state, separate from the PPO engine."""

import torch

from legged_gym.envs.wheelleg.constants import OBSERVATION_BASE_FRAME_SIZE, TORQUE_LIMIT
from legged_gym.learning.policy import scalar


class WheelLegTrainingMetrics:
    """Collect terminal-safe robot diagnostics without changing PPO inputs."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        self.loop_peak = 0.0
        self.closure_failures = 0
        self.terrain_successes = 0
        self.physical_failures = 0
        self.terrain_progress_peak = 0.0
        self.terrain_level_sum = 0.0
        self.motion_sums = torch.zeros(8, device=self.env.device)
        self.support_sums = torch.zeros(4, device=self.env.device)
        self.reward_term_sums = {}

    def update(self, observations, rewards, dones, infos):
        command = infos["pre_reset_commands"]
        # Linear velocity in the final observation uses body coordinates.
        frame_size = getattr(
            self.env, "observation_frame_size", OBSERVATION_BASE_FRAME_SIZE
        )
        measured = observations[:, -frame_size:].clone()
        ids = infos.get("terminal_env_ids")
        if ids is not None:
            measured[ids] = infos["terminal_observation"][:, -frame_size:]
        torque = infos["pre_reset_applied_torques"]
        delta = torque - infos["pre_reset_previous_torques"]
        self.motion_sums += torch.stack(
            (
                infos["pre_reset_pitch"].square().mean(),
                infos["pre_reset_roll"].square().mean(),
                infos["pre_reset_ang_vel"][:, :2].square().sum(-1).mean(),
                delta.square().mean(),
                (torque.abs() >= TORQUE_LIMIT - 0.001).float().mean(),
                (measured[:, 0] / 1.5 - command[:, 0]).square().mean(),
                (infos["pre_reset_ang_vel"][:, 2] - command[:, 2]).square().mean(),
                (infos["pre_reset_leg_length"] - command[:, 3:4]).square().mean(),
            )
        )
        lengths = infos["pre_reset_leg_length"]
        self.support_sums += torch.stack(
            (
                lengths.mean(),
                ((lengths - command[:, 3:4]).abs() <= 0.005).all(-1).float().mean(),
                (infos["pre_reset_passive_limit_margin"] <= 0.01).float().mean(),
                (lengths[:, 0] - lengths[:, 1]).square().mean(),
            )
        )
        for name, value in infos.get("reward_terms", {}).items():
            self.reward_term_sums[name] = (
                self.reward_term_sums.get(name, 0.0) + value.mean()
            )
        successes = infos.get("terrain_success", torch.zeros_like(dones))
        self.terrain_successes += int(successes.sum())
        self.physical_failures += int(
            (
                dones & ~infos.get("time_outs", torch.zeros_like(dones)) & ~successes
            ).sum()
        )
        if "terrain_progress" in infos:
            self.terrain_progress_peak = max(
                self.terrain_progress_peak, scalar(infos["terrain_best_progress"])
            )
            self.terrain_level_sum += float(infos["terrain_level"].float().mean())
        self.closure_failures += int(infos.get("loop_failures", 0))
        if "loop_error_max" in infos:
            self.loop_peak = max(self.loop_peak, scalar(infos["loop_error_max"]))

    def compute(self, num_steps):
        metrics = {
            "Episode/physical_failures": self.physical_failures,
            "Loop/failures": self.closure_failures,
            "Loop/max_anchor_error_m": self.loop_peak,
        }
        if getattr(self.env, "terrain_model", None) is not None:
            metrics.update(
                {
                    "Terrain/successful_courses": self.terrain_successes,
                    "Terrain/max_forward_progress_m": self.terrain_progress_peak,
                    "Terrain/mean_level": self.terrain_level_sum / num_steps,
                    "Terrain/max_current_level": scalar(self.env.terrain_levels),
                }
            )
        motion = (self.motion_sums / num_steps).cpu().tolist()
        for index, name in enumerate(
            (
                "pitch_rad",
                "roll_rad",
                "xy_angular_velocity_rad_s",
                "torque_delta_nm",
                "torque_saturation_fraction",
                "vx_tracking_error_m_s",
                "yaw_tracking_error_rad_s",
                "leg_length_error_m",
            )
        ):
            metric_name = name if index == 4 else "rms_" + name
            metrics["Motion/" + metric_name] = (
                motion[index] if index == 4 else motion[index] ** 0.5
            )
        support = (self.support_sums / num_steps).cpu().tolist()
        for index, name in enumerate(
            (
                "mean_leg_length_m",
                "both_legs_within_5mm_fraction",
                "passive_near_limit_fraction",
                "rms_leg_length_asymmetry_m",
            )
        ):
            metrics["Support/" + name] = (
                support[index] ** 0.5 if index == 3 else support[index]
            )
        for name, value in self.reward_term_sums.items():
            metrics["Reward/" + name] = scalar(value / num_steps)
        return metrics

    def console_summary(self):
        return "loop_max={:.3f}mm ".format(self.loop_peak * 1000.0)

    def configuration_metadata(self):
        return {"loop_model": "external_force_compliant_approximation"}

    def checkpoint_metadata(self):
        return dict(
            self.configuration_metadata(),
            terrain_curriculum_state=(
                {
                    "levels": self.env.terrain_levels.cpu().tolist(),
                    "min_level": self.env.stair_min_level,
                    "max_level": self.env.stair_max_level,
                }
                if getattr(self.env, "terrain_model", None) is not None
                else None
            ),
        )

    def restore_checkpoint(self, state):
        terrain = state.get("terrain_curriculum_state")
        if terrain and getattr(self.env, "stair_curriculum", False):
            levels = torch.tensor(
                terrain["levels"], device=self.env.device, dtype=torch.long
            )
            if levels.numel():
                ids = (
                    torch.arange(self.env.num_envs, device=self.env.device)
                    % levels.numel()
                )
                self.env.terrain_levels.copy_(
                    levels[ids].clamp(
                        self.env.stair_min_level, self.env.stair_max_level
                    )
                )
                # Relocate actors and cached observations to the restored levels.
                reset = getattr(self.env, "reset", None)
                if reset is not None:
                    reset()
