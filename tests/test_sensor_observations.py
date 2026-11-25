"""Hardware-state availability, kinematic signs, latency and real task lifecycle."""

from copy import deepcopy
import math
import unittest

import isaacgym  # Binary extension must precede torch.
import torch

from legged_gym.envs import register_tasks
from legged_gym.envs.wheelleg.model.kinematics import fivebar
from legged_gym.envs.wheelleg.model.constraints import quat_rotate_inverse
from legged_gym.envs.wheelleg.sensor_config import (
    WheelLeggedSensorCfg,
    WheelLeggedSensorCfgPPO,
)
from legged_gym.learning.policy import make_policy
from legged_gym.envs.wheelleg.sensor_observations import (
    FRAME_FIELDS,
    FRAME_SIZE,
    OBSERVATION_SIZE,
    SensorSampler,
    build_sensor_frame,
    encoder_leg_state,
    from_mylegwheel,
)
from legged_gym.envs.wheelleg.sensor_task import SensorWheelLegged
from legged_gym.utils import get_args, task_registry


class SensorContractTests(unittest.TestCase):
    def test_small_actor_initialization_is_opt_in_and_checkpoint_load_is_exact(self):
        args = dict(
            actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8],
            init_noise_std=[0.08] * 4 + [0.2] * 2,
        )
        torch.manual_seed(42)
        original = make_policy(132, 6, "cpu", args)
        torch.manual_seed(42)
        balanced = make_policy(132, 6, "cpu", dict(args, actor_output_scale=0.01))
        torch.testing.assert_close(
            balanced.actor[-1].weight, original.actor[-1].weight * 0.01
        )
        torch.testing.assert_close(balanced.actor[-1].bias, torch.zeros(6))
        torch.testing.assert_close(balanced.critic[0].weight, original.critic[0].weight)
        balanced.load_state_dict(original.state_dict())
        torch.testing.assert_close(balanced.actor[-1].weight, original.actor[-1].weight)
        cfg = WheelLeggedSensorCfgPPO()
        self.assertLessEqual(cfg.algorithm.max_learning_rate, 0.0001)

    def test_frame_layout_and_units(self):
        sensors = torch.arange(13, dtype=torch.float32)[None]
        commands = torch.tensor([[0.2, 0, -0.3, 0.1]])
        actions = torch.arange(6, dtype=torch.float32)[None] / 10
        frame = build_sensor_frame(sensors, commands, actions)
        self.assertEqual((FRAME_SIZE, OBSERVATION_SIZE), (22, 132))
        self.assertEqual(len(set(FRAME_FIELDS)), FRAME_SIZE)
        torch.testing.assert_close(frame[:, 13:16], torch.tensor([[0.3, -0.3, 0.5]]))
        torch.testing.assert_close(frame[:, 16:], actions)
        self.assertAlmostEqual(float(frame[0, 11]), 0.55, places=6)

    def snapshot(self):
        return dict(
            body_pitch_rad=torch.tensor([0.1]),
            body_pitch_rate_rad_s=torch.tensor([0.2]),
            gyro_rad_s=torch.tensor([[10.0, 20.0, 0.3]]),
            leg_angle_rad=torch.tensor([[math.pi / 2 - 0.1, math.pi / 2 + 0.2]]),
            leg_length_m=torch.tensor([[0.1, 0.11]]),
            angle_speed_rad_s=torch.tensor([[0.4, -0.5]]),
            length_speed_m_s=torch.tensor([[0.01, -0.02]]),
            wheel_speed_rad_s=torch.tensor([[-3.0, 4.0]]),
            imu_valid=torch.ones(1),
            leg_valid=torch.ones(1, 2),
            wheel_valid=torch.ones(1, 2),
        )

    def test_firmware_snapshot_uses_model_angles_and_measured_rates_not_control_commands(
        self,
    ):
        result = from_mylegwheel(self.snapshot())
        torch.testing.assert_close(
            result,
            torch.tensor(
                [
                    [
                        0.1,
                        0.2,
                        0.3,
                        0.1,
                        -0.2,
                        0.1,
                        0.11,
                        -0.4,
                        0.5,
                        0.01,
                        -0.02,
                        3.0,
                        4.0,
                    ]
                ]
            ),
            atol=1e-6,
            rtol=0,
        )

    def test_missing_invalid_or_nonfinite_feedback_never_becomes_zero(self):
        bad = self.snapshot()
        del bad["wheel_speed_rad_s"]
        with self.assertRaises(KeyError):
            from_mylegwheel(bad)
        for key in ("leg_length_m", "gyro_rad_s"):
            bad = self.snapshot()
            bad[key].flatten()[0] = float("nan")
            with self.assertRaises(ValueError):
                from_mylegwheel(bad)
        bad = self.snapshot()
        bad["leg_valid"][0, 1] = 0
        with self.assertRaises(ValueError):
            from_mylegwheel(bad)

    def test_encoder_lengths_match_existing_model_and_rates_match_finite_differences(
        self,
    ):
        q = torch.tensor(
            [[0.42, -0.39, -0.41, 0.40], [0.3, -0.35, -0.32, 0.33]], dtype=torch.float64
        )
        v = torch.tensor(
            [[0.2, -0.4, 0.1, -0.3], [-0.1, 0.3, -0.2, 0.4]], dtype=torch.float64
        )
        state = encoder_leg_state(q, v)
        for side, (i, j) in enumerate(((0, 1), (3, 2))):
            length, _, _ = fivebar(
                3 * math.pi / 4 + q[:, i], math.pi / 4 + q[:, j], 0.0
            )
            torch.testing.assert_close(state[:, side, 1], length, atol=1e-10, rtol=0)
        eps = 1e-6
        derivative = (
            encoder_leg_state(q + eps * v, v) - encoder_leg_state(q - eps * v, v)
        ) / (2 * eps)
        torch.testing.assert_close(
            derivative[:, :, :2], state[:, :, 2:], atol=1e-7, rtol=0
        )
        # Mirrored URDF hip axes must yield equal physical leg states.
        mirror_q = torch.tensor([[0.4, -0.3, -0.3, 0.4]], dtype=torch.float64)
        mirror_v = torch.tensor([[0.2, -0.1, -0.1, 0.2]], dtype=torch.float64)
        both = encoder_leg_state(mirror_q, mirror_v)
        torch.testing.assert_close(both[:, 0, 1:], both[:, 1, 1:]*torch.tensor([1., -1., 1.]), atol=1e-10, rtol=0)
        torch.testing.assert_close(both[:, 0, 0], -both[:, 1, 0], atol=1e-10, rtol=0)

    def test_fixed_latency_and_partial_reset(self):
        sampler = SensorSampler(2, "cpu", 1, [0.0] * 13, [0.0] * 13, 42)
        truth = torch.zeros(2, 13)
        sampler.reset(torch.arange(2), truth)
        sampler.advance(truth + 1)
        torch.testing.assert_close(sampler.sample, truth)
        sampler.advance(truth + 2)
        torch.testing.assert_close(sampler.sample, truth + 1)
        queue = sampler.queue[1].clone()
        sampler.reset(torch.tensor([0]), truth + 5)
        torch.testing.assert_close(sampler.queue[1], queue, rtol=0, atol=0)
        torch.testing.assert_close(sampler.sample[0], truth[0] + 5)

    def test_noise_is_reproducible_and_bad_configuration_is_rejected(self):
        a = SensorSampler(2, "cpu", 0, [0.01] * 13, [0.01] * 13, 42)
        b = SensorSampler(2, "cpu", 0, [0.01] * 13, [0.01] * 13, 42)
        truth = torch.zeros(2, 13)
        a.reset(torch.arange(2), truth)
        b.reset(torch.arange(2), truth)
        torch.testing.assert_close(a.sample, b.sample, rtol=0, atol=0)
        self.assertGreater(float(a.sample.abs().sum()), 0.0)
        for delay in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                SensorSampler(2, "cpu", delay, [0.0] * 13, [0.0] * 13, 42)
        with self.assertRaises(ValueError):
            SensorSampler(2, "cpu", 0, [-1.0] * 13, [0.0] * 13, 42)


class SensorTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        register_tasks()
        cfg = WheelLeggedSensorCfg()
        cfg.env.num_envs = 2
        cfg.env.fix_base = True  # Isolate encoder-sign tests from free-body balance.
        cfg.sim.physics_threads = 0
        cfg.commands.random_commands = False
        cfg.sensors.delay_steps = 0
        cfg.sensors.noise_std = cfg.sensors.bias_std = (0.0,) * 13
        cls.env = SensorWheelLegged(cfg, sim_device="cpu", headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()
        self.env.max_episode_length = 800

    def test_registered_configuration_is_separate_and_legacy_checkpoint_is_rejected(
        self,
    ):
        sensor, _ = task_registry.get_cfgs("wheelleg_sensor")
        legacy, _ = task_registry.get_cfgs("wheelleg_stand")
        self.assertEqual(
            (sensor.env.num_observations, legacy.env.num_observations), (132, 198)
        )
        self.assertEqual(sensor.control.heading_kp, 0.0)
        args = get_args(
            "play",
            [
                "--task",
                "wheelleg_sensor",
                "--headless",
                "--checkpoint",
                "logs/standing_support_20261022_093823/input/standing_hold.pt",
            ],
        )
        with self.assertRaisesRegex(ValueError, "observation layout"):
            task_registry.prepare_cfgs(args.task, args)

    def test_reward_terms_use_physical_state_independently_of_sensor_columns(self):
        env = self.env
        env.lin_vel[:, 2] = 2.0
        env.ang_vel[:, :2] = torch.tensor([2.0, 4.0])
        env.projected_gravity[:, 2] = -1.0
        env.pitch[:] = env.roll[:] = 0.0
        done = torch.zeros(2, dtype=torch.bool)
        before = env._compute_reward_terms(done)
        env.slice_obs[:] = 999.0
        after = env._compute_reward_terms(done)
        for name in ("lin_vel_z", "hip_pose", "upright", "angular_motion"):
            torch.testing.assert_close(after[name], before[name])
        torch.testing.assert_close(after["lin_vel_z"], torch.full((2,), 9.0))
        torch.testing.assert_close(after["angular_motion"], torch.full((2,), 5.0))
        torch.testing.assert_close(after["upright"], torch.ones(2))

    def test_balance_profile_penalizes_tilt_and_starts_with_stationary_commands(self):
        env = self.env
        self.assertFalse(env.cfg.commands.random_commands)
        torch.testing.assert_close(env.commands[:, :3], torch.zeros(2, 3))
        env.pitch[:] = torch.tensor([0.0, 0.15])
        env.roll[:] = 0.0
        terms = env._compute_reward_terms(torch.zeros(2, dtype=torch.bool))
        self.assertAlmostEqual(float(terms["upright"][0]), 1.0)
        self.assertLess(float(terms["upright"][1]), 0.12)
        env.pitch[:] = 0.0
        env.roll[:] = 0.15
        rolled = env._compute_reward_terms(torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(rolled["upright"], terms["upright"][1].expand(2))

    def test_training_velocity_metric_uses_pre_reset_body_velocity(self):
        env = self.env
        obs, _, rewards, done, info, _ = env.step(torch.zeros(2, 6))
        roots = info["pre_reset_root_states"].clone()
        roots[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        roots[:, 7:10] = torch.tensor([2.0, 0.0, 0.0])
        info["pre_reset_root_states"] = roots
        info["pre_reset_commands"][:, 0] = 0.5
        obs[:] = 999.0
        done[:] = True
        metrics = env.create_training_metrics()
        metrics.reset()
        metrics.update(obs, rewards, done, info)
        self.assertAlmostEqual(
            metrics.compute(1)["Motion/rms_vx_tracking_error_m_s"], 1.5
        )

    def test_same_size_reordered_sensor_schema_is_rejected(self):
        cfg = deepcopy(self.env.cfg)
        fields = list(cfg.sensors.frame_fields)
        fields[0], fields[1] = fields[1], fields[0]
        cfg.sensors.frame_fields = fields
        with self.assertRaisesRegex(ValueError, "132-input"):
            self.env._validate_observation_config(cfg)

    def test_actor_cannot_read_privileged_linear_velocity_position_yaw_or_terrain(self):
        env = self.env
        before = env._sensor_truth().clone()
        env.root_states[:, :3] += 100.0
        env.root_states[:, 7:10] += 20.0
        env.lin_vel[:] = 30.0
        env.yaw[:] = 1.5
        env.total_yaw[:] = 10.0
        env.projected_gravity[:] = 5.0
        torch.testing.assert_close(env._sensor_truth(), before, rtol=0, atol=0)

    def test_command_edits_do_not_resample_noise_or_advance_history(self):
        env = self.env
        env.step(torch.zeros(2, 6))
        sample, queue, history = (
            env.sensor_sampler.sample.clone(),
            env.sensor_sampler.queue.clone(),
            env.history.clone(),
        )
        env.set_commands([0.2, 0, 0.3, 0.1])
        torch.testing.assert_close(env.sensor_sampler.sample, sample, rtol=0, atol=0)
        torch.testing.assert_close(env.sensor_sampler.queue, queue, rtol=0, atol=0)
        torch.testing.assert_close(env.history[:, :-1], history[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(
            env.obs_buf[:, -22 + 13 : -22 + 16],
            torch.tensor([[0.3, 0.3, 0.5]]).repeat(2, 1),
        )
        with self.assertRaises(ValueError):
            env.set_commands([0, 0.1, 0, 0.08])

    def test_auto_reset_does_not_double_advance_surviving_sensor_history(self):
        env = self.env
        env.step(torch.zeros(2, 6))
        history = env.history.clone()
        env.max_episode_length = 10
        env.episode_length_buf[0] = 9
        obs, _, _, done, info, _ = env.step(torch.zeros(2, 6))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(obs.shape, (2, 132))
        torch.testing.assert_close(obs[1, :-22], history[1].reshape(-1), rtol=0, atol=0)
        self.assertEqual(tuple(info["terminal_observation"].shape), (1, 132))
        self.assertEqual(float(env.history[0, :-1].abs().max()), 0.0)

    def test_sensor_task_does_not_reenable_absolute_yaw_controller(self):
        cfg = deepcopy(self.env.cfg)
        cfg.control.wheel_control = "heading"
        with self.assertRaisesRegex(ValueError, "absolute-yaw"):
            self.env._validate_observation_config(cfg)

    def test_encoder_forward_angle_sign_agrees_with_real_urdf_wheel_motion(self):
        env = self.env

        def angles():
            displacement = (
                env.rb_states[:, env.wheel_indices, :3] - env.root_states[:, None, :3]
            )
            body = quat_rotate_inverse(
                env.root_states[:, None, 3:7].expand(-1, 2, -1), displacement
            )
            return env._sensor_truth()[:, 3:5], torch.atan2(
                body[:, :, 0], -body[:, :, 2]
            )

        before_fk, before_physical = angles()
        actions = torch.tensor(
            [[2.0, 2.0, -2.0, -2.0, 0.0, 0.0], [-2.0, -2.0, 2.0, 2.0, 0.0, 0.0]]
        )
        for _ in range(3):
            env.step(actions)
        after_fk, after_physical = angles()
        self.assertTrue(bool(((after_fk - before_fk).abs() > 0.0001).all()))
        self.assertTrue(bool(((after_physical - before_physical).abs() > 0.0001).all()))
        torch.testing.assert_close(
            (after_fk - before_fk).sign(),
            (after_physical - before_physical).sign(),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
