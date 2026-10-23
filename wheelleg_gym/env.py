# SPDX-License-Identifier: BSD-3-Clause
# Task equations adapted from the WheelLegRobot Isaac Lab environment.
"""GPU vector environment with explicitly approximate five-bar closure.

The URDF contains both branches, with the two closing hinges represented by
external spring/damper wrenches. This is not a rigid closed articulation.
"""

import json
import math
from pathlib import Path

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from .constraints import ClosureForces, quat_rotate_inverse
from .kinematics import fivebar, inverse_symmetric
from .commands import sample_command_targets, slew_commands, validate_command_limits
from .config import (
    ACTION_JOINTS, ACTION_SCALES, CLOSURE_MAX_FORCE, DEFAULT_CLOSURE_DAMPING,
    DEFAULT_CLOSURE_STIFFNESS, DEFAULT_COMMAND, DEFAULT_CONTROL_DT,
    DEFAULT_EPISODE_SECONDS, DEFAULT_FORWARD_SPEED_RANGE, DEFAULT_LEG_LENGTH_RANGE,
    DEFAULT_PHYSICS_DT, DEFAULT_YAW_SPEED_RANGE, MAX_LEG_LENGTH, MIN_LEG_LENGTH,
    NUM_ACTIONS, OBSERVATION_BASE_FRAME_SIZE, OBSERVATION_HISTORY_FRAMES,
    OBSERVATION_LEG_LENGTH_SIZE, TERMINATION_LOOP_ERROR, TERMINATION_TILT, TORQUE_LIMIT,
    OBSERVATION_TERRAIN_SIZE,
)
from .rewards import stability_penalty, tracking_error_penalty, support_reward_terms
from .leg_control import hip_targets, hip_pd_torques, hip_support_torques


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "wheelleg"


class WheelLegGymEnv:
    """轮腿环境：负责物理步进、观测、奖励、终止和局部重置。

    几何在 kinematics.py，闭环外力在 constraints.py，默认参数在 config.py。
    step() 执行一个控制周期；内部再按 physics_dt 推进多个物理步。
    """

    num_obs = (OBSERVATION_HISTORY_FRAMES + 1) * OBSERVATION_BASE_FRAME_SIZE
    num_privileged_obs = None
    num_actions = NUM_ACTIONS

    def __init__(self, num_envs=64, sim_device="cuda:0", headless=True,
                 seed=42, fix_base=False, physics_dt=DEFAULT_PHYSICS_DT,
                 control_dt=DEFAULT_CONTROL_DT,
                 closure_stiffness=DEFAULT_CLOSURE_STIFFNESS,
                 closure_damping=DEFAULT_CLOSURE_DAMPING,
                 random_commands=False, command=DEFAULT_COMMAND,
                 forward_speed_range=DEFAULT_FORWARD_SPEED_RANGE,
                 yaw_speed_range=DEFAULT_YAW_SPEED_RANGE,
                 leg_length_range=DEFAULT_LEG_LENGTH_RANGE, tracking_penalties=False,
                 stability_rewards=False, support_rewards=False, observe_leg_lengths=False, command_resample_seconds=0.0,
                 linear_acceleration=0.0, yaw_acceleration=0.0,
                 zero_command_fraction=0.2,
                 initial_leg_length=None,
                 leg_control="torque", leg_kp=80.0, leg_kd=4.0,
                 leg_feedforward_force=0.0,
                 heading_kp=0.0, heading_kd=0.0, wheel_control="policy", wheel_pitch_damping=0.,
                 wheel_balance_scale=1.,
                 physics_solver="tgs", wheel_friction=1.,
                 physics_threads=4, viewer_show_collisions=False,
                 viewer_width=None, viewer_height=None,
                 terrain="plane", observe_terrain=False, stair_level=0, stair_min_level=0,
                 stair_max_level=4, stair_curriculum=False, hip_target_scale=.2,
                 stair_episode_seconds=24.,
                 allow_approximate_loop=False):
        if not allow_approximate_loop:
            raise ValueError("This port uses approximate spring/damper loop closure. "
                             "Set allow_approximate_loop=True to acknowledge it.")
        if num_envs < 1 or physics_dt <= 0 or control_dt < physics_dt:
            raise ValueError("Require num_envs >= 1 and 0 < physics_dt <= control_dt")
        if not isinstance(physics_threads, int) or isinstance(physics_threads, bool) or physics_threads < 0:
            raise ValueError("physics_threads must be a nonnegative integer")
        for dimension in (viewer_width, viewer_height):
            if dimension is not None and (not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1):
                raise ValueError("Viewer dimensions must be positive integers or None")
        decimation = int(round(control_dt / physics_dt))
        if not math.isclose(decimation * physics_dt, control_dt, abs_tol=1e-9):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if len(command) != 4 or not all(math.isfinite(float(x)) for x in command):
            raise ValueError("command must contain four finite values")
        validate_command_limits(control_dt, linear_acceleration, yaw_acceleration)
        if not math.isfinite(command_resample_seconds) or command_resample_seconds < 0:
            raise ValueError("command_resample_seconds must be finite and nonnegative")
        if not math.isfinite(zero_command_fraction) or not 0 <= zero_command_fraction <= 1:
            raise ValueError("zero_command_fraction must be in [0,1]")
        if initial_leg_length is not None and (
                not math.isfinite(initial_leg_length) or not 0.05 <= initial_leg_length <= MAX_LEG_LENGTH):
            raise ValueError("initial_leg_length must be None or in [0.05,{}] m".format(MAX_LEG_LENGTH))
        if leg_control not in ("torque", "hip_pd", "hip_target"):
            raise ValueError("leg_control must be torque, hip_pd or hip_target")
        if (terrain not in ("plane", "stairs")
                or any(not isinstance(level, int) or isinstance(level, bool)
                       for level in (stair_min_level, stair_level, stair_max_level))
                or not 0 <= stair_min_level <= stair_level <= stair_max_level <= 4):
            raise ValueError("Require plane/stairs terrain and 0 <= stair_min_level <= stair_level <= stair_max_level <= 4")
        if not math.isfinite(hip_target_scale) or not 0 < hip_target_scale <= .5:
            raise ValueError("hip_target_scale must be in (0,.5] rad")
        if not math.isfinite(stair_episode_seconds) or stair_episode_seconds <= 0:
            raise ValueError("stair_episode_seconds must be positive and finite")
        if not math.isfinite(leg_kp) or leg_kp <= 0 or not math.isfinite(leg_kd) or leg_kd < 0:
            raise ValueError("Require finite leg_kp > 0 and leg_kd >= 0")
        if not math.isfinite(leg_feedforward_force) or leg_feedforward_force < 0:
            raise ValueError("Require finite leg_feedforward_force >= 0")
        if leg_feedforward_force and leg_control not in ("hip_pd", "hip_target"):
            raise ValueError("leg_feedforward_force requires a hip PD controller")
        if any(not math.isfinite(value) or value < 0 for value in (heading_kp, heading_kd)):
            raise ValueError("Require finite heading_kp >= 0 and heading_kd >= 0")
        if physics_solver not in ("tgs", "pgs"):
            raise ValueError("physics_solver must be tgs or pgs")
        if not math.isfinite(wheel_friction) or wheel_friction < 0:
            raise ValueError("wheel_friction must be finite and nonnegative")
        if wheel_control not in ("policy", "heading"):
            raise ValueError("wheel_control must be policy or heading")
        if not math.isfinite(wheel_pitch_damping) or wheel_pitch_damping < 0:
            raise ValueError("wheel_pitch_damping must be finite and nonnegative")
        if wheel_pitch_damping and wheel_control != "heading":
            raise ValueError("wheel_pitch_damping requires heading wheel control")
        if not math.isfinite(wheel_balance_scale) or wheel_balance_scale <= 0:
            raise ValueError("wheel_balance_scale must be finite and positive")
        if wheel_balance_scale != 1. and wheel_control != "heading":
            raise ValueError("wheel_balance_scale requires heading wheel control")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_envs = int(num_envs)
        self.device = str(torch.device(sim_device))
        self.headless = headless
        self.physics_threads = physics_threads
        self.viewer_show_collisions = bool(viewer_show_collisions)
        self.viewer_width, self.viewer_height = viewer_width, viewer_height
        self.fix_base = fix_base
        self.dt = control_dt
        self.physics_dt = physics_dt
        self.decimation = decimation
        self.max_episode_length = int(round(DEFAULT_EPISODE_SECONDS / self.dt))
        if terrain == "stairs":
            self.max_episode_length = int(round(stair_episode_seconds/self.dt))
        self.random_commands = random_commands
        self.forward_speed_range = validated_range(forward_speed_range, "forward speed")
        self.yaw_speed_range = validated_range(yaw_speed_range, "yaw speed")
        self.leg_length_range = validated_range(leg_length_range, "leg length", lower=MIN_LEG_LENGTH, upper=MAX_LEG_LENGTH)
        self.tracking_penalties = bool(tracking_penalties)
        self.stability_rewards = bool(stability_rewards)
        self.support_rewards = bool(support_rewards)
        self.observe_leg_lengths = bool(observe_leg_lengths)
        self.observe_terrain = bool(observe_terrain)
        self.terrain, self.stair_curriculum = terrain, bool(stair_curriculum)
        self.stair_initial_level, self.stair_max_level = int(stair_level), int(stair_max_level)
        self.stair_min_level = int(stair_min_level)
        self.hip_target_scale = float(hip_target_scale)
        self.terrain_model = None
        self.terrain_levels = torch.full((self.num_envs,), int(stair_level), dtype=torch.long, device=self.device)
        if terrain == "stairs":
            from .terrain import StairTerrain
            self.terrain_model = StairTerrain(self.num_envs, seed=seed)
            self.terrain_origins = torch.tensor(self.terrain_model.origins, dtype=torch.float32, device=self.device)
        self.observation_frame_size = (OBSERVATION_BASE_FRAME_SIZE
            + (OBSERVATION_LEG_LENGTH_SIZE if self.observe_leg_lengths else 0)
            + (OBSERVATION_TERRAIN_SIZE if self.observe_terrain else 0))
        self.num_obs = (OBSERVATION_HISTORY_FRAMES + 1) * self.observation_frame_size
        self.command_resample_seconds = float(command_resample_seconds)
        self.command_resample_steps = max(1, int(round(command_resample_seconds / self.dt))) if command_resample_seconds > 0 else 0
        self.linear_acceleration = float(linear_acceleration)
        self.yaw_acceleration = float(yaw_acceleration)
        self.zero_command_fraction = float(zero_command_fraction)
        self.initial_leg_length = None if initial_leg_length is None else float(initial_leg_length)
        self.leg_control, self.leg_kp, self.leg_kd = leg_control, float(leg_kp), float(leg_kd)
        self.leg_feedforward_force = float(leg_feedforward_force)
        self.heading_kp, self.heading_kd = float(heading_kp), float(heading_kd)
        self.wheel_control, self.wheel_pitch_damping = wheel_control, float(wheel_pitch_damping)
        self.wheel_balance_scale = float(wheel_balance_scale)
        self.physics_solver, self.wheel_friction = physics_solver, float(wheel_friction)
        self.default_command = torch.tensor(command, device=self.device, dtype=torch.float32)
        self.closed = False
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        with (ASSET_ROOT / "loop_constraints.json").open() as f:
            self.metadata = json.load(f)
        try:
            self._create_sim()
            self._load_robot(closure_stiffness, closure_damping)
            self._allocate_buffers()
            self.reset()
        except BaseException:
            self.close()
            raise

    def _create_sim(self):
        """创建 PhysX、地面和求解参数，选择 CPU 或 GPU 物理。"""
        p = gymapi.SimParams()
        p.dt = self.physics_dt
        p.substeps = 1  # Closure is recomputed at every physics step.
        p.up_axis = gymapi.UP_AXIS_Z
        p.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        p.use_gpu_pipeline = self.device.startswith("cuda")
        p.physx.use_gpu = p.use_gpu_pipeline
        p.physx.solver_type = 1 if self.physics_solver == "tgs" else 0
        p.physx.num_position_iterations = 32
        p.physx.num_velocity_iterations = 4
        p.physx.num_threads = self.physics_threads
        p.physx.contact_offset = 0.002
        p.physx.rest_offset = 0.0
        p.physx.max_depenetration_velocity = 1.0
        p.physx.max_gpu_contact_pairs = max(1 << 18, self.num_envs * 1024)
        dev_id = int(self.device.split(":")[-1]) if p.use_gpu_pipeline and ":" in self.device else 0
        self.sim = self.gym.create_sim(dev_id, -1 if self.headless else dev_id,
                                       gymapi.SIM_PHYSX, p)
        if self.sim is None:
            raise RuntimeError("Isaac Gym could not create a PhysX simulation")
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        ground.static_friction = ground.dynamic_friction = 1.0
        ground.restitution = 0.0
        self.gym.add_ground(self.sim, ground)
        if self.terrain_model is not None:
            self.terrain_model.add_to_sim(self.gym, self.sim)

    def _load_robot(self, stiffness, damping):
        """导入树形 URDF，设置力矩关节，并创建额外闭环外力模型。"""
        options = gymapi.AssetOptions()
        options.fix_base_link = self.fix_base
        options.collapse_fixed_joints = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.override_com = False
        options.override_inertia = False
        options.flip_visual_attachments = False
        options.disable_gravity = self.initial_leg_length is not None
        options.armature = 0.0
        options.linear_damping = 0.0
        options.angular_damping = 0.0
        options.max_linear_velocity = 100.0
        options.max_angular_velocity = 100.0
        self.asset = self.gym.load_asset(self.sim, str(ASSET_ROOT), self.metadata["urdf"], options)
        if self.asset is None:
            raise RuntimeError("Could not load the converted robot URDF")
        self.body_names = self.gym.get_asset_rigid_body_names(self.asset)
        self.dof_names = self.gym.get_asset_dof_names(self.asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)
        self.action_indices = torch.tensor([self.dof_names.index(n) for n in ACTION_JOINTS],
                                           device=self.device, dtype=torch.long)
        props = self.gym.get_asset_dof_properties(self.asset)
        hip_indices = self.action_indices[:4].cpu().numpy()
        self.hip_lower = torch.tensor(props["lower"][hip_indices].copy(), device=self.device)+.03
        self.hip_upper = torch.tensor(props["upper"][hip_indices].copy(), device=self.device)-.03
        # Use the imported URDF limits, in metadata order, for both rewards and
        # pre-reset diagnostics. Passive joints still receive zero actuation.
        self.passive_indices = torch.tensor(
            [self.dof_names.index(n) for n in self.metadata["passive_joints"]],
            device=self.device, dtype=torch.long)
        indices = self.passive_indices.cpu().numpy()
        self.passive_lower = torch.tensor(props["lower"][indices].copy(), device=self.device)
        self.passive_upper = torch.tensor(props["upper"][indices].copy(), device=self.device)
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        # Every passive joint remains passive. Do not inherit USD position drives.
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        props["armature"].fill(0.0)
        for n in ACTION_JOINTS:
            props["armature"][self.dof_names.index(n)] = 0.05
        root_pose = self.metadata.get("default_root_pose", {})
        initial_pose = None
        if self.initial_leg_length is not None:
            from .initial_pose import make_initial_pose
            initial_pose = make_initial_pose(
                ASSET_ROOT/self.metadata["urdf"], self.metadata, self.initial_leg_length)
            root_pose = {"pos": initial_pose["root_position"],
                         "quat_xyzw": initial_pose["root_quaternion_xyzw"]}
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*root_pose.get("pos", [0.0, 0.0, 0.25]))
        pose.r = gymapi.Quat(*root_pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]))
        if initial_pose is not None and self.device.startswith("cuda"):
            # GPU's first collision pass can still see the old zero-angle leg.
            # Spawn high enough for that geometry before committing its pose.
            pose.p.z = max(pose.p.z, self.metadata.get("default_root_pose", {}).get("pos", [0, 0, 0.25])[2])
        self.envs, self.actors = [], []
        side = int(math.ceil(math.sqrt(self.num_envs)))
        for i in range(self.num_envs):
            if self.terrain_model is None:
                env = self.gym.create_env(self.sim, gymapi.Vec3(-0.75, -0.75, 0.0),
                                          gymapi.Vec3(0.75, 0.75, 1.5), side)
            else:
                # Static meshes use global coordinates. Zero-origin environments
                # with explicit patch poses keep tensors and terrain consistent.
                env = self.gym.create_env(self.sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), side)
                origin = self.terrain_model.origins[i, self.stair_initial_level]
                pose.p.x, pose.p.y = float(origin[0]), float(origin[1])
            # Filter=1 disables all self-collision, matching the Lab asset config.
            actor = self.gym.create_actor(env, self.asset, pose, "wheelleg", i, 1)
            self.gym.set_actor_dof_properties(env, actor, props)
            if initial_pose is not None:
                states = np.zeros(self.num_dof, dtype=gymapi.DofState.dtype)
                states["pos"] = [initial_pose["joint_positions"].get(name, 0.0) for name in self.dof_names]
                if not self.gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL):
                    raise RuntimeError("Could not apply the complete initial leg pose")
            shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
            for shape in shapes:
                shape.friction = 1.0
                shape.restitution = 0.0
            if self.wheel_friction != 1.:
                # Shape counts need not equal body counts; only tire materials change.
                ranges = self.gym.get_actor_rigid_body_shape_indices(env, actor)
                for name in ("L_lun", "R_lun"):
                    span = ranges[self.body_names.index(name)]
                    for shape in shapes[span.start:span.start+span.count]:
                        shape.friction = self.wheel_friction
            self.gym.set_actor_rigid_shape_properties(env, actor, shapes)
            self.envs.append(env)
            self.actors.append(actor)
        self.gym.prepare_sim(self.sim)
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self._root_tensor).view(self.num_envs, 13)
        self.dof_state = gymtorch.wrap_tensor(self._dof_tensor).view(self.num_envs, self.num_dof, 2)
        self.rb_states = gymtorch.wrap_tensor(self._body_tensor).view(self.num_envs, self.num_bodies, 13)
        self._contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(self._contact_tensor).view(self.num_envs, self.num_bodies, 3)
        self._refresh()
        if initial_pose is not None:
            initial_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
            # Tensor FK is not guaranteed to update after setting joint states.
            # Commit full prepared state tensors. GPU updates FK above ground
            # to avoid contact skin impulses during this force-free step.
            # Restore each body's gravity flag before caching or running RL;
            # changing SimParams.gravity at runtime is not reliable in this SDK.
            try:
                heights = [pose.p.z if self.device.startswith("cuda")
                           else initial_pose["root_position"][2]]
                for height in heights:
                    self.dof_state[:, :, 0] = self.dof_state.new_tensor(
                        [initial_pose["joint_positions"].get(name, 0.0) for name in self.dof_names])
                    self.dof_state[:, :, 1] = 0.0
                    self.root_states[:, 2] = height
                    self.root_states[:, 7:] = 0.0
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim, self._root_tensor, gymtorch.unwrap_tensor(initial_ids), self.num_envs)
                    self.gym.set_dof_state_tensor_indexed(
                        self.sim, self._dof_tensor, gymtorch.unwrap_tensor(initial_ids), self.num_envs)
                    self.gym.simulate(self.sim)
                    self.gym.fetch_results(self.sim, True)
                    self._refresh()
            finally:
                for env, actor in zip(self.envs, self.actors):
                    properties = self.gym.get_actor_rigid_body_properties(env, actor)
                    for body in properties:
                        body.flags &= ~gymapi.RIGID_BODY_DISABLE_GRAVITY
                    self.gym.set_actor_rigid_body_properties(env, actor, properties, False)
            if self.device.startswith("cuda"):
                # Lower the SDK-validated whole-body pose by pure translation.
                # This is the same matching-cache operation as reset: no joint
                # angles or relative body transforms are invented or changed.
                shift = initial_pose["root_position"][2]-self.root_states[:, 2]
                self.root_states[:, 2] += shift
                self.rb_states[:, :, 2] += shift.unsqueeze(-1)
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, self._root_tensor, gymtorch.unwrap_tensor(initial_ids), self.num_envs)
        self.default_root_states = self.root_states.clone()
        self.default_rb_states = self.rb_states.clone()
        if initial_pose is None:
            self.default_dof_pos = torch.tensor(
                [self.metadata.get("default_joint_positions", {}).get(n, 0.0) for n in self.dof_names],
                device=self.device)
        else:
            self.default_dof_pos = self.dof_state[0, :, 0].clone()
        self.actor_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actors[0])
        com = [[p.com.x, p.com.y, p.com.z] for p in body_props]
        self.closure = ClosureForces(self.metadata, self.body_names, com, self.num_envs,
                                     self.device, stiffness=stiffness, damping=damping,
                                     max_force=CLOSURE_MAX_FORCE)
        if initial_pose is not None:
            self.initial_pose_diagnostics = initial_pose
            if float(self.closure.errors(self.rb_states).max()) > 0.0001:
                raise RuntimeError("Initialized loop gap exceeds 0.1mm after SDK forward kinematics")
        if not self.headless:
            camera = gymapi.CameraProperties()
            if self.viewer_width is not None:
                camera.width = self.viewer_width
            if self.viewer_height is not None:
                camera.height = self.viewer_height
            self.viewer = self.gym.create_viewer(self.sim, camera)
            if self.viewer is None:
                raise RuntimeError("Could not create Isaac Gym viewer")
            if self.terrain_model is not None:
                origin = self.terrain_model.origins[0, self.stair_initial_level]
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0],
                    gymapi.Vec3(float(origin[0])+1.8, float(origin[1])+4.5, 2.8),
                    gymapi.Vec3(float(origin[0])+1.8, float(origin[1]), .20))
            else:
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0],
                    gymapi.Vec3(1.0, 1.0, 0.65), gymapi.Vec3(0.0, 0.0, 0.2))

    def _allocate_buffers(self):
        """分配动作、指令、观测历史与各机器人独立的回合状态。"""
        n, d = self.num_envs, self.device
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=d)
        self.reset_buf = torch.zeros(n, dtype=torch.bool, device=d)
        self.actions = torch.zeros(n, self.num_actions, device=d)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_applied_torques = torch.zeros_like(self.actions)
        self.torques = torch.zeros(n, self.num_dof, device=d)
        self.commands = self.default_command.repeat(n, 1)
        self.command_targets = self.commands.clone()
        self.command_age_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.history = torch.zeros(n, OBSERVATION_HISTORY_FRAMES, self.observation_frame_size, device=d)
        self.obs_buf = torch.zeros(n, self.num_obs, device=d)
        self.slice_obs = torch.zeros(n, self.observation_frame_size, device=d)
        self.target_yaw = torch.zeros(n, device=d)
        self.total_yaw = torch.zeros(n, device=d)
        self.last_yaw = torch.zeros(n, device=d)
        self.episode_returns = torch.zeros(n, device=d)
        self.last_step_loop_error = torch.zeros(n, 2, device=d)
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=d).repeat(n, 1)
        self.environment_indices = torch.arange(n, device=d)
        self.wheel_indices = torch.tensor([self.body_names.index(name) for name in ("L_lun", "R_lun")], device=d)
        self.stair_stall_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.stair_best_progress = torch.zeros(n, device=d)
        self.stair_progress = torch.zeros(n, device=d)
        self.stair_success_count = torch.zeros(n, dtype=torch.long, device=d)
        self.stair_transport_velocity = torch.zeros(n, 3, device=d)
        # Mean of terrain heights underneath the two axle centers. This is
        # the scan reference, not the height below the leaning body origin.
        self.stair_ground_height = torch.zeros(n, device=d)
        self.terrain_scan_offsets = torch.tensor([[x, y] for x in (.1, .25, .45) for y in (-.12, 0., .12)], device=d)

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.terrain_model is not None:
            self.gym.refresh_net_contact_force_tensor(self.sim)

    def _kinematics(self):
        """由根姿态和关节状态计算机体速度、姿态、左右腿长与腿角。"""
        q = self.root_states[:, 3:7]
        self.lin_vel = quat_rotate_inverse(q, self.root_states[:, 7:10])
        self.ang_vel = quat_rotate_inverse(q, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(q, self.gravity)
        x, y, z, w = q.unbind(-1)
        self.pitch = torch.asin((2 * (w*y - z*x)).clamp(-1.0, 1.0))
        self.roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        self.yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        pos = self.dof_state[:, self.action_indices, 0]
        self.phi_l1 = 3 * math.pi / 4 + pos[:, 0]
        self.phi_l4 = math.pi / 4 + pos[:, 1]
        self.phi_r1 = 3 * math.pi / 4 + pos[:, 3]
        self.phi_r4 = math.pi / 4 + pos[:, 2]
        self.length_l, self.theta_l, valid_l = fivebar(self.phi_l1, self.phi_l4, self.pitch)
        self.length_r, self.theta_r, valid_r = fivebar(self.phi_r1, self.phi_r4, -self.pitch)
        self.geometry_valid = valid_l & valid_r
        positions = self.dof_state[:, self.passive_indices, 0]
        self.passive_limit_margin = torch.minimum(
            positions-self.passive_lower, self.passive_upper-positions)
        if self.terrain_model is not None:
            origins = self.terrain_origins[self.environment_indices, self.terrain_levels]
            wheel_xy = self.rb_states[:, self.wheel_indices, :2]
            self.stair_progress = wheel_xy[:, :, 0].mean(-1)-origins[:, 0]
            self.stair_ground_height = self.terrain_model.heights_at(
                wheel_xy, self.environment_indices, self.terrain_levels).mean(-1)

    def _observations(self, advance_history=True):
        """生成 31/33 维当前帧，并与五帧历史拼接；重置时可不推进历史。"""
        dl1, dl4 = inverse_symmetric(self.length_l)
        dr1, dr4 = inverse_symmetric(self.length_r)
        pos_errors = torch.stack((self.phi_l1-dl1, self.phi_l4-dl4,
                                   self.phi_r1-dr1, self.phi_r4-dr4), dim=-1)
        command_scales = self.commands.new_tensor([1.5, 0.5, 1.0, 5.0])
        self.slice_obs = torch.cat((self.lin_vel * 1.5, self.ang_vel * 0.5,
            self.projected_gravity, self.commands * command_scales, pos_errors,
            self.dof_state[:, self.action_indices, 1] * 0.05, self.actions,
            (self.lin_vel[:, 0]-self.commands[:, 0]).unsqueeze(-1) * 1.5,
            (self.total_yaw-self.target_yaw).unsqueeze(-1)), dim=-1)
        if self.observe_leg_lengths:
            self.slice_obs = torch.cat((self.slice_obs,
                5.0*torch.stack((self.length_l, self.length_r), dim=-1)), dim=-1)
        if self.observe_terrain:
            heights = torch.zeros(self.num_envs, OBSERVATION_TERRAIN_SIZE, device=self.device)
            if self.terrain_model is not None:
                dx, dy = self.terrain_scan_offsets.unbind(-1)
                cosine, sine = torch.cos(self.yaw)[:, None], torch.sin(self.yaw)[:, None]
                points = self.root_states[:, None, :2]+torch.stack(
                    (cosine*dx-sine*dy, sine*dx+cosine*dy), dim=-1)
                heights = self.terrain_model.heights_at(points, self.environment_indices, self.terrain_levels)
                heights = heights-self.stair_ground_height[:, None]
            self.slice_obs = torch.cat((self.slice_obs, 5*heights.clamp(-.5, .5)), dim=-1)
        if self.slice_obs.shape[1] != self.observation_frame_size:
            raise RuntimeError("Observation frame size does not match the configured layout")
        self.obs_buf = torch.cat((self.history.reshape(self.num_envs, -1), self.slice_obs), dim=-1)
        if advance_history:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = self.slice_obs
        return self.obs_buf

    def _rewards(self, terminated):
        """保留源任务基础奖励，并按开关加入跟踪/稳定项。"""
        if self.terrain_model is not None:
            from .stair_rewards import stair_reward_terms
            self.reward_terms = stair_reward_terms(
                self.stair_delta_x, self.stair_transport_velocity, self.commands, self.projected_gravity,
                self.ang_vel, self.torques[:, self.action_indices], self.last_applied_torques,
                self.passive_limit_margin, self.stair_base_contact,
                self.stair_stall_steps > 100, self.stair_success, terminated, dt=self.dt)
            self.reward_terms["heading"] = -4.0*(self.total_yaw-self.target_yaw).square()
            return sum(self.reward_terms.values())
        o, c, a = self.slice_obs, self.commands, self.actions
        r = -100.0 * terminated.float()
        r += 3.0 * torch.exp(-((c[:, :2]-self.lin_vel[:, :2])**2).sum(-1)/0.005)
        if not self.support_rewards:
            r += 2*(torch.sign(c[:, 0]) == torch.sign(self.lin_vel[:, 0])).float()-1
        r += 2.0 * torch.exp(-(c[:, 2]-self.ang_vel[:, 2])**2/0.001)
        r -= 0.2 * o[:, 2]**2
        effort = self.torques[:, self.action_indices] if self.support_rewards else a
        r -= 0.0001 * effort.square().sum(-1)
        if not self.stability_rewards:
            r -= 0.003 * ((a-self.last_actions)**2).sum(-1)
        self.reward_terms = {}
        if self.support_rewards:
            self.reward_terms = support_reward_terms(
                self.length_l, self.length_r, c[:, 3],
                self.dof_state[:, self.passive_indices, 0],
                self.passive_lower, self.passive_upper)
            for term in self.reward_terms.values():
                r += term
        else:
            r += 2.0 * torch.exp(-((c[:, 3]-self.length_r)**2+(c[:, 3]-self.length_l)**2)/0.003)
        r -= 0.1 * o[:, 13:17].abs().sum(-1)
        r += 6.0 * torch.exp(-(1+o[:, 8])**2/0.02)
        r -= 0.02 * (o[:, 3:5]**2).sum(-1)
        symmetry = (math.pi-self.phi_l4-self.phi_r1)+(math.pi-self.phi_r4-self.phi_l1)
        r += 0.7 * torch.exp(-symmetry**2/0.01)
        r += 0.5 * torch.exp(-(self.theta_l.abs()+self.theta_r.abs())**2/0.1)
        r -= (self.total_yaw-self.target_yaw).abs()
        if self.tracking_penalties:
            if self.support_rewards:
                # The support terms above already cover both leg errors.
                r -= 4.0*(self.lin_vel[:, :2]-c[:, :2]).square().sum(-1)
            else:
                r += tracking_error_penalty(self.lin_vel, self.commands, self.length_l, self.length_r)
        if self.stability_rewards:
            previous_torques = self.last_applied_torques
            r += stability_penalty(self.projected_gravity, self.ang_vel,
                                   self.torques[:, self.action_indices], previous_torques)
            r -= 0.5*(self.ang_vel[:, 2]-c[:, 2]).square()
        return r

    def _stair_diagnostics(self, before_wheel_xy):
        """Terminal-safe axle transport, lane containment and real contacts.

        Progress follows the mean of the two wheel centers along course/world
        X. Reward velocity uses their world-XY finite difference, with zero Z
        because the task rewards planar transport. Body pitching over wheels
        blocked at a riser must not earn forward-progress or speed rewards.
        """
        wheel_xy = self.rb_states[:, self.wheel_indices, :2].mean(1)
        delta_xy = wheel_xy-before_wheel_xy
        self.stair_delta_x = delta_xy[:, 0]
        self.stair_transport_velocity[:, :2] = delta_xy/self.dt
        self.stair_transport_velocity[:, 2] = 0.
        origins = self.terrain_origins[self.environment_indices, self.terrain_levels]
        advanced = self.stair_progress > self.stair_best_progress+.0002
        self.stair_stall_steps = torch.where(advanced, torch.zeros_like(self.stair_stall_steps),
                                              self.stair_stall_steps+1)
        self.stair_best_progress = torch.maximum(self.stair_best_progress, self.stair_progress)
        contact_norm = torch.linalg.vector_norm(self.contact_forces, dim=-1)
        self.stair_base_contact = contact_norm[:, self.body_names.index("base_link")] > 1.0
        wheel_states = self.rb_states[:, self.wheel_indices]
        wheel_ground = self.terrain_model.heights_at(wheel_states[:, :, :2],
                                                      self.environment_indices, self.terrain_levels)
        wheel_height_valid = ((wheel_states[:, :, 2]-wheel_ground-.0625).abs() < .025).all(-1)
        wheel_contact = (contact_norm[:, self.wheel_indices] > 1.0).all(-1)
        nonwheel = torch.ones(self.num_bodies, dtype=torch.bool, device=self.device)
        nonwheel[self.wheel_indices] = False
        no_body_support = (contact_norm[:, nonwheel] <= 1.0).all(-1)
        lateral = (self.root_states[:, 1]-origins[:, 1]).abs()
        self.stair_success = ((wheel_states[:, :, 0]-origins[:, None, 0]).min(-1).values
                              > self.terrain_model.finish_x)
        self.stair_success &= (lateral < self.terrain_model.lane_half_width-.15)
        self.stair_success &= wheel_height_valid & wheel_contact & no_body_support
        return {
            "lane": lateral > self.terrain_model.lane_half_width-.12,
            "stalled": self.stair_stall_steps > 300,
            "base_contact": contact_norm[:, self.body_names.index("base_link")] > 20.0,
        }

    def step(self, actions):
        """执行一个控制周期，保留终止前诊断，并自动重置终止机器人。"""
        if self.closed:
            raise RuntimeError("Environment is closed")
        if actions.shape != (self.num_envs, self.num_actions) or not bool(torch.isfinite(actions).all()):
            raise ValueError("Actions must have shape (num_envs, 6) and be finite")
        self.actions.copy_(actions.to(self.device).clamp(-100, 100))
        before_wheel_xy = (self.rb_states[:, self.wheel_indices, :2].mean(1).clone()
                           if self.terrain_model is not None else None)
        self.torques.zero_()
        effort = self.actions * self.actions.new_tensor(ACTION_SCALES)
        if self.leg_control == "hip_target":
            effort[:, :4] = 0.0
        if self.wheel_control == "heading":
            from .wheel_control import balance_heading_torques
            effort[:, 4:] = balance_heading_torques(
                self.wheel_balance_scale*effort[:, 4:], self.total_yaw-self.target_yaw,
                self.ang_vel[:, 2]-self.commands[:, 2], self.ang_vel[:, 1],
                self.heading_kp, self.heading_kd, self.wheel_pitch_damping, TORQUE_LIMIT)
        elif self.heading_kp or self.heading_kd:
            # Both wheel joint axes are mirrored in this asset. Equal positive
            # joint torques turn towards positive yaw (LF wheel axis is -Y,
            # RF wheel axis is +Y). The reference integrates
            # commanded yaw rate, so turning commands still advance the target.
            heading_bias = (self.heading_kp*(self.total_yaw-self.target_yaw)
                            +self.heading_kd*(self.ang_vel[:, 2]-self.commands[:, 2])).clamp(-1.0, 1.0)
            effort[:, 4:] -= heading_bias[:, None]
        self.torques[:, self.action_indices] = effort.clamp(-TORQUE_LIMIT, TORQUE_LIMIT)
        targets = hip_targets(self.commands[:, 3]) if self.leg_control in ("hip_pd", "hip_target") else None
        if self.leg_control == "hip_target":
            targets = (targets+self.hip_target_scale*self.actions[:, :4].clamp(-1, 1)).clamp(self.hip_lower, self.hip_upper)
        # A slowly varying load feedforward uses the measured five-bar Jacobian
        # once per 50Hz control period; the stabilizing PD still runs at 2kHz.
        support_bias = (hip_support_torques(self.dof_state[:, self.action_indices[:4], 0],
                                          self.leg_feedforward_force)
                        if self.leg_feedforward_force else None)
        torque_peak = (torch.zeros_like(self.actions) if targets is not None
                       else self.torques[:, self.action_indices].abs().clone())
        saturation_sum = torch.zeros_like(self.actions)
        peak_error = torch.zeros_like(self.last_step_loop_error)
        for _ in range(self.decimation):
            if targets is not None:
                # The learned residual is held at 50Hz; physical feedback is
                # recomputed at every physics step using freshly read DOF state.
                pd = hip_pd_torques(self.dof_state[:, self.action_indices[:4], 0],
                                    self.dof_state[:, self.action_indices[:4], 1],
                                    targets, self.leg_kp, self.leg_kd)
                total = effort.clone()
                total[:, :4] += pd
                if support_bias is not None:
                    # Radial support is a joint-torque bias, never an upward
                    # external body force. It shares the same 9Nm actuator cap.
                    total[:, :4] += support_bias
                active = total.clamp(-TORQUE_LIMIT, TORQUE_LIMIT)
                self.torques[:, self.action_indices] = active
                torque_peak = torch.maximum(torque_peak, active.abs())
                saturation_sum += (active.abs() >= TORQUE_LIMIT-1e-5).float()
            forces, moments = self.closure.compute(self.rb_states)
            self.gym.apply_rigid_body_force_tensors(self.sim,
                gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(moments), gymapi.ENV_SPACE)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            if targets is not None:
                self.gym.refresh_dof_state_tensor(self.sim)
            peak_error = torch.maximum(peak_error, self.closure.errors(self.rb_states))
        self._refresh()
        self.last_step_loop_error.copy_(peak_error)
        self.episode_length_buf += 1
        self._kinematics()
        yaw_delta = self.yaw-self.last_yaw
        self.total_yaw += torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
        self.last_yaw.copy_(self.yaw)
        if self.wheel_control == "heading":
            from .wheel_control import bounded_heading_target
            self.target_yaw.copy_(bounded_heading_target(
                self.total_yaw, self.target_yaw, self.commands[:, 2], self.dt))
        else:
            self.target_yaw += self.commands[:, 2] * self.dt
        self._observations()
        finite = torch.isfinite(self.obs_buf).all(-1) & torch.isfinite(self.rb_states).all(dim=-1).all(dim=-1)
        bad_loop = (peak_error > TERMINATION_LOOP_ERROR).any(-1)
        termination_reasons = {
            "pitch": self.pitch.abs() > (.65 if self.terrain_model is not None else TERMINATION_TILT),
            "roll": self.roll.abs() > TERMINATION_TILT,
            "nonfinite": ~finite,
            "geometry": ~self.geometry_valid,
            "loop": bad_loop,
        }
        if self.terrain_model is not None:
            termination_reasons.update(self._stair_diagnostics(before_wheel_xy))
        terminated = torch.stack(list(termination_reasons.values())).any(dim=0)
        terrain_success = torch.zeros_like(terminated)
        if self.terrain_model is not None:
            self.stair_success &= ~terminated
            terrain_success = self.stair_success.clone()
            self.stair_success_count += terrain_success.long()
        time_out = self.episode_length_buf >= self.max_episode_length
        rewards = self._rewards(terminated)
        # Do not let a failed simulation poison PPO; termination remains visible.
        rewards = torch.where(finite, rewards, torch.full_like(rewards, -100.0))
        self.episode_returns += rewards
        dones = terminated | time_out | terrain_success
        self.reset_buf.copy_(dones)
        info = {"time_outs": time_out & ~terminated & ~terrain_success, "loop_error_max": peak_error.max(),
                "terrain_success": terrain_success,
                "physical_failures": terminated.clone(),
                "loop_error": peak_error.clone(), "loop_failures": bad_loop.sum(),
                "pre_reset_root_states": self.root_states.clone(),
                "pre_reset_leg_length": torch.stack((self.length_l, self.length_r), dim=-1).clone(),
                "pre_reset_leg_theta": torch.stack((self.theta_l, self.theta_r), dim=-1).clone(),
                "pre_reset_dof_positions": self.dof_state[:, :, 0].clone(),
                "pre_reset_passive_limit_margin": self.passive_limit_margin.clone(),
                "pre_reset_pitch": self.pitch.clone(),
                "pre_reset_roll": self.roll.clone(),
                "pre_reset_commands": self.commands.clone(),
                "pre_reset_ang_vel": self.ang_vel.clone(),
                "pre_reset_applied_torques": self.torques[:, self.action_indices].clone(),
                "pre_reset_previous_torques": self.last_applied_torques.clone(),
                "pre_reset_peak_abs_torques": torque_peak,
                "pre_reset_torque_saturation_fraction": (saturation_sum/self.decimation
                    if targets is not None else (torque_peak >= TORQUE_LIMIT-1e-5).float()),
                "termination_reasons": termination_reasons,
                "reward_terms": {name: value.clone() for name, value in self.reward_terms.items()},
                "approximate_loop": True}
        if self.terrain_model is not None:
            info.update(terrain_level=self.terrain_levels.clone(),
                        terrain_progress=self.stair_progress.clone(),
                        terrain_ground_height=self.stair_ground_height.clone(),
                        terrain_transport_velocity_world_m_per_s=self.stair_transport_velocity.clone(),
                        terrain_progress_source="mean wheel-center X relative to selected course origin",
                        terrain_ground_height_source="mean terrain height underneath the two wheel centers",
                        terrain_best_progress=self.stair_best_progress.clone(),
                        terrain_success_count=self.stair_success_count.clone(),
                        pre_reset_wheel_states=self.rb_states[:, self.wheel_indices].clone(),
                        pre_reset_contact_forces=self.contact_forces.clone())
        ids = dones.nonzero(as_tuple=False).flatten()
        self.last_actions.copy_(self.actions)
        self.last_applied_torques.copy_(self.torques[:, self.action_indices])
        if len(ids):
            info["episode"] = {"reward": self.episode_returns[ids].mean(),
                "length": self.episode_length_buf[ids].float().mean(),
                "loop_error_max": peak_error[ids].max()}
            info["terminal_observation"] = self.obs_buf[ids].clone()
            info["terminal_env_ids"] = ids.clone()
            step_obs = self.obs_buf.clone()
            if self.terrain_model is not None and self.stair_curriculum:
                promote = terrain_success[ids].long()
                demote = (terminated[ids] & (self.stair_progress[ids] < self.terrain_model.finish_x*.5)).long()
                self.terrain_levels[ids] = (self.terrain_levels[ids]+promote-demote).clamp(self.stair_min_level, self.stair_max_level)
            self._reset_idx(ids)
            # Updating derived arrays is read-only and must not advance survivors' history.
            self._kinematics()
            reset_obs = self._observations(advance_history=False)
            step_obs[ids] = reset_obs[ids]
            self.obs_buf = step_obs
            self.history[ids, -1] = self.slice_obs[ids]
        # Reward and terminal diagnostics use the command that generated the
        # action. Only then prepare the command seen by the next policy call.
        info["command_resamples"] = self._advance_random_commands(dones)
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, self.viewer_show_collisions)
            self.gym.sync_frame_time(self.sim)
        return self.obs_buf, None, rewards, dones, info

    def _reset_idx(self, ids):
        """只恢复指定机器人，避免改动其他机器人的状态与历史。"""
        if self.terrain_model is not None:
            target_xy = self.terrain_origins[ids, self.terrain_levels[ids], :2]
            shift = target_xy-self.default_root_states[ids, :2]
            self.default_root_states[ids, :2] = target_xy
            self.default_rb_states[ids, :, :2] += shift[:, None, :]
        self.root_states[ids] = self.default_root_states[ids]
        self.dof_state[ids, :, 0] = self.default_dof_pos
        self.dof_state[ids, :, 1] = 0.0
        actor_ids = self.actor_ids[ids].contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self._root_tensor,
                                                    gymtorch.unwrap_tensor(actor_ids), len(ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, self._dof_tensor,
                                             gymtorch.unwrap_tensor(actor_ids), len(ids))
        # The rigid-body tensor is a cache; FK is updated by PhysX on the next simulate.
        # Restore its matching initial state so no stale terminal wrench is applied.
        self.rb_states[ids] = self.default_rb_states[ids]
        self.episode_length_buf[ids] = 0
        self.episode_returns[ids] = 0
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.last_applied_torques[ids] = 0
        self.history[ids] = 0
        self.target_yaw[ids] = 0
        self.total_yaw[ids] = 0
        self.last_yaw[ids] = 0
        self.stair_stall_steps[ids] = 0
        self.stair_best_progress[ids] = 0
        self.stair_progress[ids] = 0
        self.stair_transport_velocity[ids] = 0
        self.commands[ids] = self.default_command
        self.command_targets[ids] = self.default_command
        self.command_age_steps[ids] = 0
        if self.random_commands:
            self._sample_command_targets(ids)
            self.commands[ids] = self.command_targets[ids]
            if self.linear_acceleration > 0:
                self.commands[ids, :2] = 0.0
            if self.yaw_acceleration > 0:
                self.commands[ids, 2] = 0.0

    def _sample_command_targets(self, ids):
        fraction = self.zero_command_fraction if self.command_resample_steps else 0.0
        self.command_targets[ids] = sample_command_targets(
            len(ids), self.device, self.forward_speed_range, self.yaw_speed_range,
            self.leg_length_range, fraction)

    def _advance_random_commands(self, dones):
        if not self.random_commands or not (
                self.command_resample_steps or self.linear_acceleration or self.yaw_acceleration):
            return 0
        survivors = ~dones
        resamples = 0
        if self.command_resample_steps:
            self.command_age_steps[survivors] += 1
            due = survivors & (self.command_age_steps >= self.command_resample_steps)
            if bool(due.any()):
                ids = due.nonzero(as_tuple=False).flatten()
                self._sample_command_targets(ids)
                self.command_age_steps[ids] = 0
                resamples = len(ids)
        updated = slew_commands(self.commands, self.command_targets, self.dt,
                                self.linear_acceleration, self.yaw_acceleration)
        self.commands[survivors] = updated[survivors]
        self._refresh_command_observations()
        return resamples

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.reset_buf.zero_()
        self._kinematics()
        self._observations(advance_history=False)
        self.history[:, -1] = self.slice_obs
        return self.obs_buf, None

    def get_observations(self):
        return self.obs_buf

    def set_commands(self, commands):
        """Update the current command without advancing physics or history.

        Interactive inputs arrive between control steps. Only the most recent
        frame is edited; the five earlier frames keep their original commands.
        """
        value = torch.as_tensor(commands, device=self.device, dtype=torch.float32)
        if value.shape == (4,):
            value = value.unsqueeze(0).expand(self.num_envs, -1)
        if value.shape != (self.num_envs, 4) or not bool(torch.isfinite(value).all()):
            raise ValueError("Commands must be finite with shape [4] or [num_envs,4]")
        if bool(((value[:, 3] < MIN_LEG_LENGTH) | (value[:, 3] > MAX_LEG_LENGTH)).any()):
            raise ValueError("Commanded leg length must be in [{},{}] m".format(MIN_LEG_LENGTH, MAX_LEG_LENGTH))
        self.commands.copy_(value)
        self.command_targets.copy_(value)
        self._refresh_command_observations()
        return self.obs_buf

    def _refresh_command_observations(self):
        """Refresh the newest command fields without advancing frame history."""
        self.slice_obs[:, 9:13] = self.commands * self.commands.new_tensor([1.5, 0.5, 1.0, 5.0])
        self.slice_obs[:, 29] = (self.lin_vel[:, 0]-self.commands[:, 0])*1.5
        self.obs_buf[:, -self.observation_frame_size:] = self.slice_obs
        self.history[:, -1] = self.slice_obs

    def get_privileged_observations(self):
        return None

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        self.closed = True


def validated_range(values, name, lower=None, upper=None):
    if len(values) != 2 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("{} range must contain two finite values".format(name))
    low, high = map(float, values)
    if low > high or (lower is not None and low < lower) or (upper is not None and high > upper):
        raise ValueError("Invalid {} range".format(name))
    return low, high


