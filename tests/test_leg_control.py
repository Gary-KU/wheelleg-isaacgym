"""PD support has explicit mirror signs, damping and torque residual semantics."""

import math
import unittest

import isaacgym  # noqa: F401
import torch

from wheelleg_gym.leg_control import hip_targets, hip_pd_torques, hip_support_torques
from wheelleg_gym.kinematics import fivebar


class LegControlTests(unittest.TestCase):
    def test_targets_follow_the_real_four_hip_mirror_order(self):
        targets = hip_targets(torch.tensor([.08]))
        torch.testing.assert_close(targets, torch.tensor([[.613989, -.613989, -.613989, .613989]]), atol=1e-5, rtol=0)

    def test_zero_error_and_velocity_produce_zero_support_not_hidden_feedforward(self):
        targets = hip_targets(torch.tensor([.08, .1]))
        torch.testing.assert_close(hip_pd_torques(targets, torch.zeros_like(targets), targets), torch.zeros_like(targets))

    def test_folded_pose_pushes_all_four_hips_towards_extension(self):
        current = hip_targets(torch.tensor([.05]))
        target = hip_targets(torch.tensor([.08]))
        torque = hip_pd_torques(current, torch.zeros_like(current), target)
        self.assertTrue(bool((torque*(target-current) > 0).all()))
        self.assertGreater(float(torque.abs().max()), 9)  # Clip only after adding residual.

    def test_damping_opposes_joint_velocity_and_does_not_mix_environments(self):
        target = hip_targets(torch.tensor([.08, .08]))
        velocity = torch.tensor([[1., -1., 2., -2.], [0., 0., 0., 0.]])
        torch.testing.assert_close(hip_pd_torques(target, velocity, target), -4*velocity)


class RadialSupportTests(unittest.TestCase):
    @staticmethod
    def lengths(positions):
        left = fivebar(3*math.pi/4+positions[:, 0], math.pi/4+positions[:, 1], 0.)[0]
        right = fivebar(3*math.pi/4+positions[:, 3], math.pi/4+positions[:, 2], 0.)[0]
        return torch.stack((left, right), dim=-1)

    def test_matches_independent_autograd_on_asymmetric_poses(self):
        base = hip_targets(torch.tensor([.05, .08, .12, .15], dtype=torch.float64))
        offsets = torch.tensor([[.03, .02, -.01, .04], [-.04, .01, .03, -.02],
                                [.06, -.03, .02, .04], [-.02, .04, -.05, .01]], dtype=torch.float64)
        positions = (base+offsets).requires_grad_(True)
        expected = torch.autograd.grad(self.lengths(positions).sum(), positions)[0]
        actual = hip_support_torques(positions.detach(), 1.)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

    def test_virtual_work_matches_finite_difference_length_change(self):
        positions = hip_targets(torch.tensor([.08, .11], dtype=torch.float64))
        positions += torch.tensor([[.04, -.01, -.02, .03], [-.03, .02, .04, -.01]], dtype=torch.float64)
        velocity = torch.tensor([[.7, -.4, .2, -.5], [-.3, -.2, .8, .1]], dtype=torch.float64)
        epsilon, force = 1e-6, 22.465
        length_rate = ((self.lengths(positions+epsilon*velocity)
                        -self.lengths(positions-epsilon*velocity))/(2*epsilon)).sum(-1)
        joint_power = (hip_support_torques(positions, force)*velocity).sum(-1)
        torch.testing.assert_close(joint_power, force*length_rate, atol=1e-8, rtol=1e-7)

    def test_nominal_mirror_signs_extend_against_the_supported_load(self):
        positions = hip_targets(torch.tensor([.08], dtype=torch.float64))
        torque = hip_support_torques(positions, 22.465)
        magnitude = 22.465*.07299831929324221
        expected = torch.tensor([[-magnitude, magnitude, magnitude, -magnitude]], dtype=torch.float64)
        torch.testing.assert_close(torque, expected, atol=1e-10, rtol=1e-9)
        extension_direction = hip_targets(torch.tensor([.081], dtype=torch.float64))-positions
        self.assertTrue(bool((torque*extension_direction > 0).all()))

    def test_zero_force_and_invalid_geometry_are_finite_without_hidden_torque(self):
        positions = hip_targets(torch.tensor([.08], dtype=torch.float64))
        torch.testing.assert_close(hip_support_torques(positions, 0.), torch.zeros_like(positions))
        alpha = math.acos(.044/.105)
        # Both upper-link tips coincide: a closed chain Jacobian is undefined.
        coincident = torch.tensor([[alpha-3*math.pi/4, math.pi-alpha-math.pi/4,
                                    math.pi-alpha-math.pi/4, alpha-3*math.pi/4]], dtype=torch.float64)
        torch.testing.assert_close(hip_support_torques(coincident, 22.465), torch.zeros_like(coincident))
        invalid = torch.tensor([[float('nan'), 0., float('inf'), 0.]], dtype=torch.float64)
        torch.testing.assert_close(hip_support_torques(invalid, 22.465), torch.zeros_like(invalid))

    def test_near_singular_derivatives_are_bounded_and_validation_is_explicit(self):
        alpha = math.acos(.044/.105)
        positions = torch.tensor([[alpha-3*math.pi/4, math.pi-alpha-math.pi/4,
                                   math.pi-alpha-math.pi/4, alpha-3*math.pi/4]], dtype=torch.float64)
        for offset in (1e-8, 1e-6, 1e-4):
            torque = hip_support_torques(positions+offset, 22.465)
            self.assertTrue(bool(torch.isfinite(torque).all()))
            self.assertLessEqual(float(torque.abs().max()), .5*22.465)
        for force in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                hip_support_torques(positions, force)
        with self.assertRaises(ValueError):
            hip_support_torques(torch.zeros(2, 3), 22.465)
