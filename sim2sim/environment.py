"""CPU MuJoCo environment for sensor-only fine tuning.

Physics truth is used for reward/termination/reporting only. Policy observations
and outer-loop feedback use the same noisy, delayed IMU/encoder contract as Gym.
"""

from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import mujoco
import torch
from sim2sim.build_model import build, ROOT
from legged_gym.envs.wheelleg.model.initial_pose import make_initial_pose
from legged_gym.envs.wheelleg.observations.sensors import (
    SensorSampler,
    encoder_leg_state,
    build_sensor_frame,
)
from legged_gym.envs.wheelleg.observations.odometry import EncoderPositionSampler
from legged_gym.envs.wheelleg.control.encoder_hold import EncoderHold
from legged_gym.envs.wheelleg.control.leg_control import (
    hip_targets,
    hip_pd_torques,
    hip_support_torques,
)
from legged_gym.envs.wheelleg.control.commands import slew_commands
from legged_gym.envs.wheelleg.constants import ACTION_SCALES


class MujocoWheelLeg:
    num_obs, num_actions, device, viewer = 132, 6, "cpu", None

    def __init__(
        self,
        config,
        num_envs=4,
        seed=42,
        episode_seconds=16.0,
        training=False,
        terrain="flat",
        wave_height=0.005,
        wave_length=0.4,
        flight_compliance=False,
        ramp_height=0.03,
        flight_mode="support",
    ):
        self.flight_mode = flight_mode
        self.config = deepcopy(config)
        self.environment_config = deepcopy(config)
        self.num_envs = num_envs
        self.seed = seed
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(
            str(
                build(
                    ROOT / f"assets/wheelleg/mujoco/{terrain}.xml",
                    terrain,
                    wave_height,
                    wave_length,
                    flight_imu=flight_compliance,
                    ramp_height=ramp_height,
                )
            )
        )
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.dt = 0.002
        self.physics_dt = self.model.opt.timestep
        self.substeps = round(self.dt / self.physics_dt)
        assert abs(self.substeps * self.physics_dt - self.dt) < 1e-10
        self.max_episode_length = round(episode_seconds / self.dt)
        self.metadata = json.loads(
            (ROOT / "assets/wheelleg/loop_constraints.json").read_text()
        )
        names = self.metadata["active_joints"]
        self.qi = np.array([self.model.joint(n).qposadr[0] for n in names])
        self.vi = np.array([self.model.joint(n).dofadr[0] for n in names])
        self.base = self.model.body("base_link").id
        self.initial_length = float(config["env"]["initial_leg_length"])
        self.initial_pose = make_initial_pose(
            ROOT / "assets/wheelleg/urdf/Robot_Model.urdf",
            self.metadata,
            self.initial_length,
            balance_initial_com=True,
        )
        self.c = config["control"]
        s = config["sensors"]
        c = self.c
        self.sensor = SensorSampler(
            num_envs, "cpu", s["delay_steps"], s["noise_std"], s["bias_std"], seed
        )
        self.wheel_sensor = EncoderPositionSampler(
            num_envs,
            "cpu",
            s["delay_steps"],
            s["wheel_angle_noise_std"],
            s["wheel_angle_bias_std"],
            seed,
        )
        self.hold = EncoderHold(
            num_envs,
            "cpu",
            0.08,
            c["encoder_hold_position_kp"],
            c["encoder_hold_heading_kp"],
            c["encoder_hold_speed_limit"],
            c["encoder_hold_yaw_limit"],
            c["encoder_hold_height_slope"],
            c["encoder_hold_turn_reference_scale"],
            c["encoder_hold_gyro_heading"],
            self.dt,
        )
        self.actions = torch.zeros(num_envs, 6)
        self.history = torch.zeros(num_envs, 5, 22)
        self.commands = torch.zeros(num_envs, 4)
        self.commands[:, 3] = self.initial_length
        self.targets = self.commands.clone()
        self.budget = torch.zeros(num_envs, 4)
        self.ramp_landing_steps = torch.zeros(num_envs, dtype=torch.long)
        self.ramp_start_s = torch.full((num_envs,), 2.0)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.returns = torch.zeros(num_envs)
        self.obs = torch.zeros(num_envs, 132)
        self.flight = None
        if flight_compliance:
            from sim2sim.flight import FlightCompliance

            self.flight = FlightCompliance(num_envs, self.dt, seed)
        self.last_regen_j = torch.zeros(num_envs, 6)
        self.last_speed_peak = torch.zeros(num_envs, 6)
        self.reset()

    @staticmethod
    def tensor(value):
        return torch.tensor(np.asarray(value).copy(), dtype=torch.float32)

    def joints(self):
        return (
            self.tensor([d.qpos[self.qi] for d in self.data]),
            self.tensor([d.qvel[self.vi] for d in self.data]),
        )

    def physical(self):
        angles = []
        velocities = []
        for d in self.data:
            R = d.xmat[self.base].reshape(3, 3)
            angles.append(
                [
                    np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])),
                    np.arctan2(R[2, 1], R[2, 2]),
                ]
            )
            v = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self.base, v, 1
            )
            velocities.append(v)
        return self.tensor(angles), self.tensor(velocities)

    def truth(self):
        q, v = self.joints()
        angles, velocity = self.physical()
        legs = encoder_leg_state(q[:, :4], v[:, :4])
        return torch.cat(
            (
                angles[:, :1],
                velocity[:, 1:3],
                legs[:, :, 0],
                legs[:, :, 1],
                legs[:, :, 2],
                legs[:, :, 3],
                v[:, 4:] * torch.tensor([-1.0, 1.0]),
            ),
            1,
        )

    def acceleration(self):
        return np.array([d.sensor("flight_accel").data.copy() for d in self.data])

    def reset_ids(self, ids):
        for i in ids.tolist():
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            pose = self.initial_pose
            d.qpos[:3] = pose["root_position"]
            q = pose["root_quaternion_xyzw"]
            d.qpos[3:7] = [q[3], *q[:3]]
            for name, v in pose["joint_positions"].items():
                d.qpos[self.model.joint(name).qposadr[0]] = v
            mujoco.mj_forward(self.model, d)
        if self.flight is not None:
            self.flight.reset(ids.numpy(), self.acceleration())
        self.actions[ids] = 0
        self.history[ids] = 0
        self.budget[ids] = 0
        self.commands[ids] = torch.tensor([0.0, 0.0, 0.0, self.initial_length])
        self.targets[ids] = self.commands[ids]
        self.ramp_landing_steps[ids] = 0
        if self.training == "ramp":
            # Vary arrival timing without changing commanded speed or acceleration.
            self.ramp_start_s[ids] = self.tensor(self.rng.uniform(1.5, 2.3, len(ids)))
        self.episode_length_buf[ids] = 0
        self.returns[ids] = 0
        self.hold.reset(ids)
        self.sensor.reset(ids, self.truth())
        q, _ = self.joints()
        self.wheel_sensor.reset(ids, q[:, 4:])

    def observe(self, advance=True, ids=None):
        previous_obs = self.obs.clone()
        observed = torch.ones(self.num_envs, dtype=torch.bool)
        if ids is not None:
            observed.zero_()
            observed[ids] = True
        q, _ = self.joints()
        if advance:
            self.sensor.advance(self.truth())
            self.wheel_sensor.advance(q[:, 4:])
        self.hold.update(
            self.sensor.sample,
            self.wheel_sensor.sample,
            self.commands,
            self.targets,
            observed & ~self.flight.air if self.flight is not None else observed,
        )
        ref = self.commands.clone()
        f, y = self.hold.corrections(self.commands, self.targets)
        ref[:, 0] += f
        ref[:, 2] += y
        frame = build_sensor_frame(self.sensor.sample, ref, self.actions)
        self.obs = torch.cat((self.history.reshape(self.num_envs, -1), frame), 1)
        if advance:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = frame
        if ids is not None:
            self.obs[~observed] = previous_obs[~observed]
        return self.obs

    def reset(self):
        self.reset_ids(torch.arange(self.num_envs))
        return self.observe(False), None

    def get_observations(self):
        return self.obs, None

    def step(self, actions):
        self.actions = actions.detach().clamp(-100, 100).clone()
        c = self.c
        q, _ = self.joints()
        was_air = self.flight.air.clone() if self.flight is not None else None
        blend = (
            self.flight.update(self.acceleration())
            if self.flight is not None
            else torch.ones(self.num_envs)
        )
        if self.flight is not None:
            changed = (self.flight.air != was_air).nonzero().flatten()
            if len(changed):
                self.hold.reset(changed)
        support = hip_support_torques(q[:, :4], c["leg_feedforward_force"])
        effort = self.actions * torch.tensor(ACTION_SCALES)
        pitch, rate = self.sensor.sample[:, 0], self.sensor.sample[:, 1]
        drive = c["sensor_pitch_rate_kd"] * rate + c[
            "sensor_pitch_guard_kp"
        ] * pitch.sign() * (pitch.abs() - c["sensor_pitch_guard_angle"]).clamp(min=0)
        effort[:, 4] -= drive
        effort[:, 5] += drive
        targets = hip_targets(self.commands[:, 3])
        self.last_regen_j.zero_()
        self.last_speed_peak.zero_()
        self.last_torque_peak = torch.zeros_like(effort)
        for _ in range(self.substeps):
            q, v = self.joints()
            total = effort.clone()
            if self.flight is not None and self.flight_mode == "support":
                total[:, :4] += (
                    hip_pd_torques(
                        q[:, :4],
                        v[:, :4],
                        targets,
                        c["leg_kp"],
                        c["leg_kd"],
                        c["leg_differential_kd"],
                    )
                    + support * blend[:, None]
                )
            else:
                # Preserve exact grounded controller; soften hips only in flight.
                total[:, :4] *= 0.2 + 0.8 * blend[:, None]
                total[:, :4] += (
                    hip_pd_torques(
                        q[:, :4],
                        v[:, :4],
                        targets,
                        c["leg_kp"],
                        c["leg_kd"],
                        c["leg_differential_kd"],
                    )
                    * blend[:, None]
                    + support * blend[:, None]
                )
                total[:, :4] += (0.4 * (targets - q[:, :4]) - 0.08 * v[:, :4]) * (
                    1 - blend[:, None]
                )
            limits = torch.full_like(total, 0.1)
            limits[:, :4] = torch.where(self.budget >= 2, 0.32, 0.68)
            torque = total.clamp(-limits, limits)
            self.last_regen_j += (-torque * v).clamp(min=0) * self.physics_dt
            self.last_speed_peak = torch.maximum(self.last_speed_peak, v.abs())
            self.budget += (torque[:, :4].abs() > 0.32) * self.physics_dt
            self.last_torque_peak = torch.maximum(self.last_torque_peak, torque.abs())
            for i, d in enumerate(self.data):
                d.qfrc_applied[:] = 0
                d.qfrc_applied[self.vi] = torque[i].numpy()
                mujoco.mj_step(self.model, d)
        for d in self.data:
            mujoco.mj_forward(self.model, d)
        angles, velocity = self.physical()
        sensor_truth = self.truth()
        # Simulator state appears only in learning targets and diagnostics.
        error = velocity[:, 3] - self.commands[:, 0]
        yaw_error = velocity[:, 2] - self.commands[:, 2]
        legs_error = (sensor_truth[:, 5:7] - self.commands[:, 3:4]).square().mean(1)
        reward = (
            2 * torch.exp(-angles.square().sum(1) / 0.1**2)
            + 2 * torch.exp(-error.square() / 0.08**2)
            + torch.exp(-yaw_error.square() / 0.12**2)
            - 0.5 * velocity[:, 4].square() / 0.05**2
            - 2e3 * legs_error
        )
        finite = torch.tensor(
            [np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() for d in self.data]
        )
        failure = (angles.abs().max(1).values > 0.7) | ~finite
        self.episode_length_buf += 1
        ramp_success = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.training == "ramp":
            position = self.tensor([d.qpos[:3] for d in self.data])
            # Train toward an interior target, leaving room for sensor variation
            # below the unchanged 0.2-rad playback acceptance threshold.
            reward -= 3 * ((angles[:, 0].abs() - 0.16).clamp(min=0) / 0.05).square()
            # Dampen the onset of a pitch swing, before the angle is already large.
            reward -= 0.15 * velocity[:, 1].square()
            # World position is a learning target only, never actor feedback.
            # Penalize accumulated drift as well as instantaneous lateral speed.
            reward -= 2 * (position[:, 1] / 0.05).square().clamp(max=25)
            safe = (
                (position[:, 0] > 1.1)
                & (position[:, 1].abs() < 0.05)
                & (angles.abs().max(1).values < 0.2)
                & ((velocity[:, 3] - 1.0).abs() < 0.1)
            )
            self.ramp_landing_steps = torch.where(safe, self.ramp_landing_steps + 1, 0)
            ramp_success = self.ramp_landing_steps == 250
            # Reward simulator truth only; no terrain/position enters the actor.
            reward -= 0.02 * (sensor_truth[:, 11:13].abs() / 50).square().sum(1)
            reward += ramp_success * 25
            self.targets[:, 0] = torch.where(
                self.episode_length_buf * self.dt >= self.ramp_start_s, 1.0, 0.0
            )
        timeout = self.episode_length_buf >= self.max_episode_length
        # Keep training through the full evaluation horizon after landing.
        done = failure | timeout
        reward -= failure * 25.0
        if self.training == "ramp":
            # PPO normalizes advantages, but its shared gradient clipping also
            # includes the Critic. Keep return magnitudes reasonable at 500 Hz.
            reward *= 0.01
        self.returns += reward
        if self.training == "drive":
            for i in range(self.num_envs):
                if self.episode_length_buf[i] % 1500 == 0:
                    self.targets[i, 0] = float(self.rng.choice([0.0, 0.0, -1.0, 1.0]))
        before_height = self.commands[:, 3].clone()
        self.commands = slew_commands(self.commands, self.targets, self.dt, 0.8, 2.5)
        self.commands[:, 3] = before_height + (
            self.targets[:, 3] - before_height
        ).clamp(-0.01 * self.dt, 0.01 * self.dt)
        obs = self.observe()
        info = dict(
            time_outs=timeout,
            ramp_success=ramp_success,
            physical_failures=failure,
            terminal_observation=obs.clone(),
            angles=angles,
            velocity=velocity,
            leg_error=legs_error,
        )
        if done.any():
            ids = done.nonzero().flatten()
            info["episode"] = dict(
                length=self.episode_length_buf[ids].float().mean(),
                reward=self.returns[ids].mean(),
            )
            self.reset_ids(ids)
            obs = self.observe(False, ids=ids)
        return obs, None, reward, done, info

    def close(self):
        pass
