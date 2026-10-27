"""CPU checks that support rewards prefer the target over joint-stop support."""

from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; extension must precede torch in combined discovery.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.rewards import support_reward_terms


class SupportRewardTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([0.08])
        self.lower = torch.full((4,), -1.233774)
        self.upper = -self.lower
        self.support_angles = torch.tensor([[-0.95462, 0.95462, 0.95462, -0.95462]])
        self.stop_angles = torch.tensor([[-1.233774, 1.233774, 1.233774, -1.233774]])

    def terms(self, left, right=None, angles=None, **kwargs):
        left = torch.tensor([left]) if not isinstance(left, torch.Tensor) else left
        right = left if right is None else right
        right = torch.tensor([right]) if not isinstance(right, torch.Tensor) else right
        return support_reward_terms(left, right, self.target,
                                    self.support_angles if angles is None else angles,
                                    self.lower, self.upper, **kwargs)

    def test_target_support_scores_about_ten_points_above_folded_stop(self):
        correct = self.terms(0.08)
        stopped = self.terms(0.05, angles=self.stop_angles)
        self.assertEqual(set(correct), {"leg_tracking", "leg_error", "leg_asymmetry", "passive_limit"})
        for value in correct.values():
            self.assertEqual(value.shape, (1,))
        self.assertAlmostEqual(sum(value.item() for value in correct.values()), 2.0, places=5)
        advantage = sum(value.item() for value in correct.values())-sum(value.item() for value in stopped.values())
        self.assertAlmostEqual(advantage, 10.5, delta=0.001)

    def test_equal_average_height_does_not_hide_left_right_error(self):
        symmetric = self.terms(0.08)
        asymmetric = self.terms(0.07, 0.09)
        self.assertAlmostEqual(asymmetric["leg_asymmetry"].item(), -2.0, places=5)
        self.assertLess(asymmetric["leg_error"].item(), symmetric["leg_error"].item())
        self.assertLess(asymmetric["leg_tracking"].item(), symmetric["leg_tracking"].item())
        torch.testing.assert_close(asymmetric["leg_asymmetry"], self.terms(0.09, 0.07)["leg_asymmetry"])

    def test_correct_target_pose_has_no_passive_limit_cost(self):
        self.assertEqual(self.terms(0.08)["passive_limit"].item(), 0.0)

    def test_touching_or_exceeding_either_boundary_receives_bounded_maximum(self):
        touching = self.terms(0.08, angles=self.stop_angles)["passive_limit"]
        outside = self.terms(0.08, angles=2*self.stop_angles)["passive_limit"]
        torch.testing.assert_close(touching, torch.tensor([-4.0]))
        torch.testing.assert_close(outside, touching)
        half_margin = torch.tensor([[self.lower[0]+0.05, self.upper[1]-0.05,
                                     self.upper[2]-0.05, self.lower[3]+0.05]])
        torch.testing.assert_close(self.terms(0.08, angles=half_margin)["passive_limit"],
                                   torch.tensor([-1.0]), rtol=1e-5, atol=1e-5)

    def test_far_initial_pose_costs_are_bounded_and_finite(self):
        # Extreme but finite inputs also test overflow in intermediate squares.
        terms = self.terms(1e30, -1e30, angles=1e30*self.stop_angles)
        for value in terms.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(terms["leg_tracking"].item(), 0.0)
        self.assertEqual(terms["leg_error"].item(), -4.5)
        self.assertEqual(terms["leg_asymmetry"].item(), -4.5)
        self.assertEqual(terms["passive_limit"].item(), -4.0)

    def test_length_gradients_pull_toward_target_on_both_sides(self):
        for initial, expected_sign in ((0.075, 1), (0.085, -1)):
            left = torch.tensor([initial], requires_grad=True)
            terms = self.terms(left)
            (terms["leg_tracking"]+terms["leg_error"]).sum().backward()
            self.assertTrue(torch.isfinite(left.grad).all())
            self.assertGreater(expected_sign*left.grad.item(), 0.0)

    def test_limit_gradient_points_away_from_both_soft_boundaries(self):
        angles = torch.tensor([[self.lower[0]+0.05, self.upper[1]-0.05,
                                0.0, 0.0]], requires_grad=True)
        self.terms(0.08, angles=angles)["passive_limit"].sum().backward()
        self.assertGreater(angles.grad[0, 0].item(), 0.0)
        self.assertLess(angles.grad[0, 1].item(), 0.0)
        torch.testing.assert_close(angles.grad[0, 2:], torch.zeros(2))

    def test_batch_targets_and_bounds_do_not_mix_environments(self):
        targets = torch.tensor([0.07, 0.08, 0.09])
        angles = self.support_angles.repeat(3, 1)
        terms = support_reward_terms(targets, targets, targets, angles,
                                     self.lower.repeat(3, 1), self.upper.repeat(3, 1))
        torch.testing.assert_close(terms["leg_tracking"], torch.full((3,), 2.0))
        torch.testing.assert_close(terms["leg_error"], torch.zeros(3))
        # Column-shaped lengths are accepted without accidentally broadcasting N×N.
        column = support_reward_terms(targets[:, None], targets[:, None], targets[:, None],
                                      angles, self.lower, self.upper)
        for name in terms:
            torch.testing.assert_close(terms[name], column[name])

    def test_invalid_shapes_bounds_and_scales_fail_clearly(self):
        valid = [self.target, self.target, self.target, self.support_angles, self.lower, self.upper]
        for index, bad in ((0, torch.zeros(1, 2)), (1, torch.zeros(2)),
                           (2, torch.zeros(2)), (3, torch.zeros(4)),
                           (3, torch.zeros(1, 0)), (4, torch.zeros(3)),
                           (5, self.lower)):
            args = list(valid)
            args[index] = bad
            with self.subTest(index=index, shape=tuple(bad.shape)):
                with self.assertRaises(ValueError):
                    support_reward_terms(*args)
        for name in ("length_scale", "soft_limit_margin"):
            for value in (0.0, -0.1, float("inf"), float("nan")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        support_reward_terms(*valid, **{name: value})


if __name__ == "__main__":
    unittest.main()
