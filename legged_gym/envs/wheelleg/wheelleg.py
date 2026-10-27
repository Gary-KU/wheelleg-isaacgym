"""Six-value Wheel-Legged-Gym task API over the preserved five-bar physics."""

import math
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import ASSET_ROOT, LeggedRobot
from legged_gym.utils.helpers import class_to_dict
import torch
from wheelleg_gym.rewards import stability_penalty, tracking_error_penalty
from wheelleg_gym.stair_rewards import stair_reward_terms


# Every saved legacy constructor setting has one readable nested config home.
CONFIG_SECTIONS = {
    "env": ("num_envs", "initial_leg_length", "fix_base"),
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
    "viewer": ("viewer_show_collisions", "viewer_width", "viewer_height"),
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


def restore_environment_config(cfg, values):
    known = {
        name: section for section, names in CONFIG_SECTIONS.items() for name in names
    }
    for name, value in values.items():
        if name in ("sim_device", "headless", "num_envs"):
            continue  # execution placement/count are chosen on this computer
        if name in known:
            setattr(getattr(cfg, known[name]), name, value)
        elif name == "seed":
            cfg.seed = value
        elif name == "physics_dt":
            cfg.sim.dt = value
        elif name == "control_dt":
            ratio = value / values.get("physics_dt", cfg.sim.dt)
            if not math.isclose(round(ratio), ratio):
                raise ValueError("Checkpoint has noninteger control decimation")
            cfg.control.decimation = round(ratio)
        else:
            raise ValueError("Unsupported checkpoint environment field: " + name)


class WheelLegged(LeggedRobot):
    """Task owns config/reward registration; base owns real physics and resets."""

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
        self._prepare_reward_function()
        super().__init__(**self.environment_config)
        self.obs_history_length = 1  # obs_buf already contains all six frames
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.extras = {}

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

    def _rewards(self, terminated):
        self._terminated = terminated
        self._stair_terms = None
        if self.terrain_model is not None:
            self._stair_terms = stair_reward_terms(
                self.stair_delta_x,
                self.stair_transport_velocity,
                self.commands,
                self.projected_gravity,
                self.ang_vel,
                self.torques[:, self.action_indices],
                self.last_applied_torques,
                self.passive_limit_margin,
                self.stair_base_contact,
                self.stair_stall_steps > 100,
                self.stair_success,
                terminated,
                dt=self.dt,
            )
            self._stair_terms["heading"] = (
                -4.0 * (self.total_yaw - self.target_yaw).square()
            )
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

    def _reward_termination(self):
        return self._terminated.float()

    def _reward_tracking_lin_vel(self):
        return torch.exp(
            -(self.commands[:, :2] - self.lin_vel[:, :2]).square().sum(-1) / 0.005
        )

    def _reward_direction(self):
        if self.support_rewards:
            return torch.zeros(self.num_envs, device=self.device)
        return (
            2
            * (
                torch.sign(self.commands[:, 0]) == torch.sign(self.lin_vel[:, 0])
            ).float()
            - 1
        )

    def _reward_tracking_ang_vel(self):
        return torch.exp(-(self.commands[:, 2] - self.ang_vel[:, 2]).square() / 0.001)

    def _reward_lin_vel_z(self):
        return self.slice_obs[:, 2].square()

    def _reward_effort(self):
        if self._stair_terms is not None:
            return self._stair_terms["effort"]
        values = (
            self.torques[:, self.action_indices]
            if self.support_rewards
            else self.actions
        )
        return values.square().sum(-1)

    def _reward_action_rate(self):
        if self.stability_rewards:
            return torch.zeros(self.num_envs, device=self.device)
        return (self.actions - self.last_actions).square().sum(-1)

    def _leg_errors(self):
        return (
            torch.stack((self.length_l, self.length_r), -1) - self.commands[:, 3:4]
        ) / 0.01

    def _reward_leg_tracking(self):
        if self.support_rewards:
            return torch.exp(-0.5 * self._leg_errors().square().sum(-1))
        return torch.exp(
            -(
                (self.commands[:, 3] - self.length_r).square()
                + (self.commands[:, 3] - self.length_l).square()
            )
            / 0.003
        )

    def _reward_leg_error(self):
        return (
            self._leg_errors().square().clamp(max=9).sum(-1)
            if self.support_rewards
            else torch.zeros(self.num_envs, device=self.device)
        )

    def _reward_leg_asymmetry(self):
        return (
            ((self.length_l - self.length_r) / 0.01).square().clamp(max=9)
            if self.support_rewards
            else torch.zeros(self.num_envs, device=self.device)
        )

    def _reward_passive_limit(self):
        if self._stair_terms is not None:
            return self._stair_terms["passive_limit"]
        return (
            (1 - self.passive_limit_margin / 0.1).clamp(0, 1).square().mean(-1)
            if self.support_rewards
            else torch.zeros(self.num_envs, device=self.device)
        )

    def _reward_hip_pose(self):
        return self.slice_obs[:, 13:17].abs().sum(-1)

    def _reward_upright(self):
        return torch.exp(-(1 + self.slice_obs[:, 8]).square() / 0.02)

    def _reward_angular_motion(self):
        return self.slice_obs[:, 3:5].square().sum(-1)

    def _reward_hip_symmetry(self):
        value = (math.pi - self.phi_l4 - self.phi_r1) + (
            math.pi - self.phi_r4 - self.phi_l1
        )
        return torch.exp(-value.square() / 0.01)

    def _reward_leg_angle(self):
        return torch.exp(-(self.theta_l.abs() + self.theta_r.abs()).square() / 0.1)

    def _reward_heading(self):
        if self._stair_terms is not None:
            return self._stair_terms["heading"]
        return (self.total_yaw - self.target_yaw).abs()

    def _reward_tracking_error(self):
        if not self.tracking_penalties:
            return torch.zeros(self.num_envs, device=self.device)
        if self.support_rewards:
            return -4 * (self.lin_vel[:, :2] - self.commands[:, :2]).square().sum(-1)
        return tracking_error_penalty(
            self.lin_vel, self.commands, self.length_l, self.length_r
        )

    def _reward_stability(self):
        if not self.stability_rewards:
            return torch.zeros(self.num_envs, device=self.device)
        return stability_penalty(
            self.projected_gravity,
            self.ang_vel,
            self.torques[:, self.action_indices],
            self.last_applied_torques,
        )

    def _reward_yaw_error(self):
        return (
            (self.ang_vel[:, 2] - self.commands[:, 2]).square()
            if self.stability_rewards
            else torch.zeros(self.num_envs, device=self.device)
        )


# Stair helper already defines named, weighted terms. Expose each as a regular
# _reward_<name> method so zeroing a cfg scale removes it from the task.
def _stair_reward(name):
    def reward(self):
        return self._stair_terms[name]

    reward.__name__ = "_reward_" + name
    return reward


for _name in (
    "progress",
    "velocity",
    "yaw",
    "posture",
    "angular_damping",
    "stall",
    "base_contact",
    "torque_smoothness",
    "failure",
    "success",
):
    setattr(WheelLegged, "_reward_" + _name, _stair_reward(_name))
