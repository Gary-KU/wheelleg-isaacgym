"""CPU regression checks for physically scaled stability rewards."""

from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; must precede torch when combined with environment tests.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.rewards import stability_penalty


class StabilityPenaltyTests(unittest.TestCase):
    def setUp(self):
        self.gravity = torch.tensor([[0.0, 0.0, -1.0]])
        self.angular_velocity = torch.zeros(1, 3)
        self.torques = torch.zeros(1, 6)

    def test_upright_steady_turn_and_holding_torque_have_no_penalty(self):
        # Turning and supporting the robot are allowed when roll/pitch stay calm.
        velocity = torch.tensor([[0.0, 0.0, 0.8]])
        holding = torch.tensor([[1.0, -1.0, 2.0, -2.0, 0.3, -0.3]])
        actual = stability_penalty(self.gravity, velocity, holding, holding)
        torch.testing.assert_close(actual, torch.zeros(1), rtol=0, atol=0)

    def test_equal_physical_changes_cost_the_same_for_hips_and_wheels(self):
        # Raw policy actions scale by 0.5 on hips and 10 on wheels. Equal raw
        # action jumps would be physically unequal, so convert before rewarding.
        action_scales = torch.tensor([0.5, 0.5, 0.5, 0.5, 10.0, 10.0])
        hip_actions = self.torques.clone()
        wheel_actions = self.torques.clone()
        hip_actions[0, 0] = 4.0
        wheel_actions[0, 4] = 0.2
        hip = stability_penalty(self.gravity, self.angular_velocity,
                                (hip_actions*action_scales).clamp(-9, 9), self.torques)
        wheel = stability_penalty(self.gravity, self.angular_velocity,
                                  (wheel_actions*action_scales).clamp(-9, 9), self.torques)
        torch.testing.assert_close(hip, wheel, rtol=0, atol=0)
        self.assertLess(hip.item(), 0.0)

    def test_full_torque_reversal_receives_meaningful_joint_independent_cost(self):
        previous = self.torques.clone()
        current = self.torques.clone()
        previous[0, 4] = 9.0
        current[0, 4] = -9.0
        actual = stability_penalty(self.gravity, self.angular_velocity, current, previous)
        torch.testing.assert_close(actual, torch.tensor([-0.8]))

    def test_tilt_and_roll_pitch_motion_are_penalized_without_quartic_dead_zone(self):
        small_angle = 0.1
        gravity = torch.tensor([[torch.sin(torch.tensor(small_angle)), 0.0,
                                 -torch.cos(torch.tensor(small_angle))]])
        rate = torch.tensor([[1.0, -2.0, 7.0]])
        actual = stability_penalty(gravity, rate, self.torques, self.torques)
        expected = -4.0*torch.sin(torch.tensor(small_angle)).square()-0.1*5
        torch.testing.assert_close(actual, expected[None])
        tilt_only = stability_penalty(gravity, self.angular_velocity, self.torques, self.torques)
        self.assertLess(tilt_only.item(), -0.03)

    def test_batch_costs_do_not_mix_environments(self):
        gravity = self.gravity.repeat(3, 1)
        angular_velocity = self.angular_velocity.repeat(3, 1)
        previous = self.torques.repeat(3, 1)
        current = previous.clone()
        current[1, 4] = 1.0
        current[2, 4] = 2.0
        actual = stability_penalty(gravity, angular_velocity, current, previous)
        self.assertEqual(actual.shape, (3,))
        self.assertEqual(actual[0].item(), 0.0)
        torch.testing.assert_close(actual[2], 4.0*actual[1])

    def test_bad_shapes_and_torque_scales_are_rejected(self):
        for gravity, velocity, torques, previous in (
            (torch.zeros(1, 2), self.angular_velocity, self.torques, self.torques),
            (self.gravity, torch.zeros(2, 3), self.torques, self.torques),
            (self.gravity, self.angular_velocity, torch.zeros(6), self.torques),
            (self.gravity, self.angular_velocity, torch.zeros(2, 6), self.torques),
            (self.gravity, self.angular_velocity, torch.zeros(1, 0), torch.zeros(1, 0)),
            (self.gravity, self.angular_velocity, self.torques, torch.zeros(1, 5)),
        ):
            with self.assertRaises(ValueError):
                stability_penalty(gravity, velocity, torques, previous)
        for scale in (0, -9, float("inf"), float("nan")):
            with self.subTest(torque_scale=scale):
                with self.assertRaises(ValueError):
                    stability_penalty(self.gravity, self.angular_velocity,
                                      self.torques, self.torques, torque_scale=scale)


if __name__ == "__main__":
    unittest.main()
