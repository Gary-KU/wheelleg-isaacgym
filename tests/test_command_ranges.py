"""CPU checks for bounded commands and the optional driving reward term."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import isaacgym  # noqa: F401; extension must precede PyTorch in combined tests.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.rewards import tracking_error_penalty
from wheelleg_gym.env import WheelLegGymEnv, validated_range


class TrackingPenaltyTests(unittest.TestCase):
    def test_zero_error_has_zero_penalty_and_ignores_unrelated_axes(self):
        velocity = torch.tensor([[0.3, 0.1, 5.0], [-0.2, -0.1, -9.0]])
        command = torch.tensor([[0.3, 0.1, 2.0, 0.08], [-0.2, -0.1, -3.0, 0.12]])
        length = command[:, 3].clone()
        actual = tracking_error_penalty(velocity, command, length, length)
        torch.testing.assert_close(actual, torch.zeros(2), rtol=0, atol=0)

    def test_velocity_and_leg_penalty_weights_are_explicit(self):
        velocity = torch.zeros(1, 3)
        command = torch.tensor([[0.3, 0.0, 0.0, 0.08]])
        actual = tracking_error_penalty(velocity, command, torch.tensor([0.13]), torch.tensor([0.13]))
        # 4*(0.3 m/s)^2 + 0.25*(1^2+1^2), with each leg off by 0.05 m.
        torch.testing.assert_close(actual, torch.tensor([-0.86]))

    def test_larger_velocity_or_length_error_is_strictly_worse(self):
        command = torch.tensor([[0.0, 0.0, 0.0, 0.08]])
        baseline = tracking_error_penalty(torch.zeros(1, 3), command,
                                          torch.tensor([0.08]), torch.tensor([0.08]))
        small = tracking_error_penalty(torch.tensor([[0.1, 0.0, 0.0]]), command,
                                       torch.tensor([0.09]), torch.tensor([0.09]))
        large = tracking_error_penalty(torch.tensor([[0.2, 0.0, 0.0]]), command,
                                       torch.tensor([0.10]), torch.tensor([0.10]))
        self.assertLess(small.item(), baseline.item())
        self.assertLess(large.item(), small.item())

    def test_column_lengths_do_not_broadcast_across_environments(self):
        command = torch.tensor([[0.0, 0.0, 0.0, 0.08], [0.1, 0.0, 0.0, 0.12],
                                [-0.3, 0.1, 0.0, 0.10]])
        velocity = torch.zeros(3, 3)
        lengths = torch.tensor([0.09, 0.15, 0.07])
        vector = tracking_error_penalty(velocity, command, lengths, lengths)
        column = tracking_error_penalty(velocity, command, lengths[:, None], lengths[:, None])
        self.assertEqual(column.shape, (3,))
        self.assertTrue(torch.isfinite(column).all().item())
        self.assertTrue((column <= 0).all().item())
        torch.testing.assert_close(vector, column)

    def test_bad_batch_shape_is_rejected(self):
        velocity = torch.zeros(2, 3)
        command = torch.zeros(2, 4)
        lengths = torch.zeros(2)
        for v, c, left, right in ((torch.zeros(2, 2), command, lengths, lengths),
                                  (velocity, torch.zeros(3, 4), lengths, lengths),
                                  (velocity, command, torch.zeros(2, 2), lengths)):
            with self.assertRaises(ValueError):
                tracking_error_penalty(v, c, left, right)


class CommandRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only CPU PhysX is used here; the main training process owns the GPU.
        cls.env = WheelLegGymEnv(
            num_envs=8, sim_device="cpu", headless=True, seed=31,
            random_commands=True, forward_speed_range=(-0.3, 0.3),
            yaw_speed_range=(0.0, 0.0), leg_length_range=(0.08, 0.08),
            allow_approximate_loop=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.random_commands = True
        self.env.tracking_penalties = False
        self.env.reset()

    def test_random_commands_stay_inside_training_ranges(self):
        samples = []
        for _ in range(8):
            self.env.reset()
            samples.append(self.env.commands.clone())
        commands = torch.cat(samples)
        self.assertTrue((commands[:, 0] >= -0.3).all().item())
        self.assertTrue((commands[:, 0] <= 0.3).all().item())
        self.assertTrue((commands[:, 0] < 0).any().item())
        self.assertTrue((commands[:, 0] > 0).any().item())
        torch.testing.assert_close(commands[:, 1:3], torch.zeros(64, 2))
        torch.testing.assert_close(commands[:, 3], torch.full((64,), 0.08))

    def test_partial_reset_only_resamples_selected_environments(self):
        env = self.env
        before = env.commands.clone()
        env._reset_idx(torch.tensor([0, 2]))
        survivors = torch.tensor([1, 3, 4, 5, 6, 7])
        torch.testing.assert_close(env.commands[survivors], before[survivors], rtol=0, atol=0)
        self.assertTrue((env.commands[[0, 2], 0] != before[[0, 2], 0]).all().item())

    def test_fixed_commands_ignore_sampling_ranges(self):
        env = self.env
        env.random_commands = False
        env.reset()
        torch.testing.assert_close(env.commands, env.default_command.repeat(8, 1))

    def test_tracking_penalty_is_opt_in(self):
        env = self.env
        terminated = torch.zeros(8, dtype=torch.bool)
        extra = torch.full((8,), -2.5)
        with patch("wheelleg_gym.env.tracking_error_penalty", return_value=extra) as helper:
            baseline = env._rewards(terminated)
            helper.assert_not_called()
            env.tracking_penalties = True
            with_extra = env._rewards(terminated)
            helper.assert_called_once()
        torch.testing.assert_close(with_extra, baseline+extra)

    def test_invalid_ranges_are_rejected_and_constant_ranges_are_allowed(self):
        self.assertEqual(validated_range((0.0, 0.0), "yaw"), (0.0, 0.0))
        for invalid in ((1, -1), (0, float("inf")), (float("nan"), 1), (0,)):
            with self.subTest(interval=invalid):
                with self.assertRaises(ValueError):
                    validated_range(invalid, "test")
        for invalid in ((0.05, 0.08), (0.08, 0.2)):
            with self.assertRaises(ValueError):
                validated_range(invalid, "legs", lower=0.06, upper=0.15)


if __name__ == "__main__":
    unittest.main()
