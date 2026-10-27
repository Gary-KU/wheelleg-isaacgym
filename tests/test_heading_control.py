"""Fast real-contact checks for optional heading feedback and torque summation."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import isaacgym  # noqa: F401; the extension must load before Torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legged_gym.envs.wheelleg.wheelleg_robot as environment_module
from legged_gym.envs.wheelleg.constants import ACTION_SCALES, TORQUE_LIMIT
from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot


class HeadingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WheelLegRobot(
            num_envs=4, sim_device="cpu", headless=True, seed=19,
            allow_approximate_loop=True, initial_leg_length=.08,
            leg_control="hip_pd", leg_feedforward_force=22.465,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.heading_kp = self.env.heading_kd = 0.
        self.env.reset()
        self.actions = torch.zeros(self.env.num_envs, 6)

    def test_default_heading_is_disabled_even_with_heading_and_rate_errors(self):
        env = self.env
        self.assertEqual((env.heading_kp, env.heading_kd), (0., 0.))
        env.target_yaw[:] = torch.tensor([.2, -.2, 2., -2.])
        env.ang_vel[:, 2] = 1.
        _, _, _, _, info = env.step(self.actions)
        torch.testing.assert_close(info["pre_reset_applied_torques"][:, 4:], torch.zeros(4, 2))

    def test_ground_traction_turns_towards_both_positive_and_negative_heading_targets(self):
        # At this asset's wheel-axis convention, equal positive wheel torques
        # generate positive yaw on the ground. A suspended robot cannot test it.
        env = self.env
        env.heading_kp, env.heading_kd = .5, 0.
        env.target_yaw[:] = torch.tensor([.2, -.2, 0., 0.])
        for _ in range(15):  # 0.3 s avoids requiring a learned pitch policy.
            _, _, _, dones, _ = env.step(self.actions)
            self.assertFalse(bool(dones.any()))
        self.assertGreater(float(env.total_yaw[0]), .003)
        self.assertLess(float(env.total_yaw[1]), -.003)
        self.assertLess(abs(float(env.total_yaw[0])-.2), .2)
        self.assertLess(abs(float(env.total_yaw[1])+.2), .2)

    def test_heading_bias_is_capped_then_added_before_final_wheel_torque_clipping(self):
        env = self.env
        env.heading_kp = 1.
        # Heading correction is [-1,+1,+1,-1] Nm, independently of the policy.
        env.target_yaw[:] = torch.tensor([-5., 5., 5., -5.])
        self.actions[:, 4:] = torch.tensor([[.95], [-.95], [.85], [-.85]])
        _, _, _, _, info = env.step(self.actions)
        expected = torch.tensor([[8.5, 8.5], [-8.5, -8.5], [9., 9.], [-9., -9.]])
        torch.testing.assert_close(info["pre_reset_applied_torques"][:, 4:], expected)
        torch.testing.assert_close(info["pre_reset_torque_saturation_fraction"][:, 4:],
                                   torch.tensor([[0., 0.], [0., 0.], [1., 1.], [1., 1.]]))

    def test_rate_damping_opposes_measured_yaw_rate_relative_to_command(self):
        env = self.env
        env.heading_kd = .5
        env.ang_vel[:, 2] = torch.tensor([.3, -.3, .3, -.3])
        env.commands[:, 2] = torch.tensor([0., 0., .1, -.1])
        _, _, _, _, info = env.step(self.actions)
        expected = torch.tensor([[-.15, -.15], [.15, .15], [-.1, -.1], [.1, .1]])
        torch.testing.assert_close(info["pre_reset_applied_torques"][:, 4:], expected)

    def test_radial_load_is_held_once_while_real_physics_pd_is_recomputed_each_substep(self):
        env = self.env
        self.actions[0, :4] = torch.tensor([30., -30., -30., 30.])
        pd_outputs, load_outputs = [], []
        original_pd = environment_module.hip_pd_torques
        original_load = environment_module.hip_support_torques

        def record_pd(*args, **kwargs):
            result = original_pd(*args, **kwargs)
            pd_outputs.append(result.clone())
            return result

        def record_load(*args, **kwargs):
            result = original_load(*args, **kwargs)
            load_outputs.append(result.clone())
            return result

        with mock.patch.object(environment_module, "hip_pd_torques", side_effect=record_pd), \
                mock.patch.object(environment_module, "hip_support_torques", side_effect=record_load):
            _, _, _, _, info = env.step(self.actions)
        self.assertEqual(len(load_outputs), 1)
        self.assertEqual(len(pd_outputs), env.decimation)
        residual = self.actions*self.actions.new_tensor(ACTION_SCALES)
        residual[:, :4] += load_outputs[0]+pd_outputs[-1]
        torch.testing.assert_close(info["pre_reset_applied_torques"],
                                   residual.clamp(-TORQUE_LIMIT, TORQUE_LIMIT))
        self.assertGreater(float(info["pre_reset_peak_abs_torques"][0, :4].max()), 8.99)
        # Real state feedback changes during the 40 actual PhysX substeps.
        self.assertGreater(float((pd_outputs[-1]-pd_outputs[0]).abs().max()), .01)


if __name__ == "__main__":
    unittest.main()
