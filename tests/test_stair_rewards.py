"""Stair rewards must prefer real net progress over standing or reversal loops."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; combined tests may import the binary extension.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.stair_rewards import stair_reward_terms


def inputs(size=1):
    return dict(
        delta_x=torch.full((size,), .005),
        lin_vel=torch.tensor([[.25, 0., 0.]]).repeat(size, 1),
        commands=torch.tensor([[.25, 0., 0., .14]]).repeat(size, 1),
        projected_gravity=torch.tensor([[0., 0., -1.]]).repeat(size, 1),
        ang_vel=torch.zeros(size, 3),
        applied_torques=torch.ones(size, 6),
        previous_torques=torch.ones(size, 6),
        passive_margin=torch.full((size, 4), .2),
        base_contact=torch.zeros(size, dtype=torch.bool),
        stalled=torch.zeros(size, dtype=torch.bool),
        success=torch.zeros(size, dtype=torch.bool),
        terminated=torch.zeros(size, dtype=torch.bool),
    )


class StairRewardTests(unittest.TestCase):
    def test_forward_motion_scores_above_a_stalled_upright_robot(self):
        moving = inputs()
        standing = deepcopy(moving)
        standing['delta_x'].zero_()
        standing['lin_vel'].zero_()
        standing['stalled'].fill_(True)
        forward = stair_reward_terms(**moving)
        stopped = stair_reward_terms(**standing)
        advantage = sum(forward.values())-sum(stopped.values())
        self.assertGreater(float(advantage[0]), 7.0)
        self.assertAlmostEqual(float(stopped['progress'][0]), 0.)
        self.assertFalse(any('leg' in key or 'symmetry' in key for key in forward))

    def test_back_and_forth_cannot_earn_net_progress_even_at_different_speeds(self):
        # Four slow forward steps, followed by one >1m/s reverse step, travel
        # exactly the same distance. A clipped rate reward would be exploitable.
        data = inputs(5)
        data['delta_x'] = torch.tensor([.01, .01, .01, .01, -.04])
        terms = stair_reward_terms(**data)
        self.assertAlmostEqual(float(terms['progress'].sum()), 0., places=6)
        self.assertLess(float(terms['progress'][-1]), 0.)

    def test_contact_stall_failure_and_success_flags_are_isolated_per_environment(self):
        data = inputs(4)
        data['base_contact'][0] = True
        data['stalled'][1] = True
        data['terminated'][2] = True
        data['success'][3] = True
        terms = stair_reward_terms(**data)
        expected = {
            'base_contact': [-5., 0., 0., 0.],
            'stall': [0., -2., 0., 0.],
            'failure': [0., 0., -100., 0.],
            'success': [0., 0., 0., 25.],
        }
        for key, values in expected.items():
            torch.testing.assert_close(terms[key], torch.tensor(values))
        for value in terms.values():
            self.assertEqual(value.shape, (4,))
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_passive_limit_cost_is_bounded_and_points_away_from_joint_stops(self):
        data = inputs(4)
        data['passive_margin'] = torch.tensor([[.2]*4, [.05]*4, [0.]*4, [-.03]*4], requires_grad=True)
        cost = stair_reward_terms(**data)['passive_limit']
        torch.testing.assert_close(cost, torch.tensor([0., -1., -4., -4.]))
        cost.sum().backward()
        self.assertTrue(bool((data['passive_margin'].grad[1] > 0).all()))

    def test_effort_and_smoothness_use_physical_torques_and_do_not_modify_inputs(self):
        data = inputs(2)
        data['applied_torques'][0].fill_(9.)
        data['previous_torques'][0].fill_(-9.)
        before = deepcopy(data)
        terms = stair_reward_terms(**data)
        self.assertAlmostEqual(float(terms['effort'][0]), -.0486, places=6)
        self.assertAlmostEqual(float(terms['torque_smoothness'][0]), -1.2, places=6)
        self.assertEqual(float(terms['torque_smoothness'][1]), 0.)
        for key in data:
            torch.testing.assert_close(data[key], before[key])

    def test_rate_units_are_consistent_with_control_period(self):
        first, second = inputs(), inputs()
        second['delta_x'] *= 2
        left = stair_reward_terms(**first, dt=.02)['progress']
        right = stair_reward_terms(**second, dt=.04)['progress']
        torch.testing.assert_close(left, right)

    def test_invalid_shapes_dtypes_and_control_period_fail_clearly(self):
        wrong = (
            ('delta_x', torch.zeros(1, 1)), ('lin_vel', torch.zeros(1, 2)),
            ('commands', torch.zeros(2, 4)), ('projected_gravity', torch.zeros(3)),
            ('ang_vel', torch.zeros(1, 2)), ('applied_torques', torch.zeros(1, 0)),
            ('previous_torques', torch.zeros(1, 5)), ('passive_margin', torch.zeros(4)),
            ('base_contact', torch.zeros(1)), ('stalled', torch.zeros(1, 1, dtype=torch.bool)),
            ('success', torch.zeros(2, dtype=torch.bool)), ('terminated', [False]),
        )
        for name, value in wrong:
            data = inputs()
            data[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                stair_reward_terms(**data)
        for dt in (0., -.02, float('nan'), float('inf')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                stair_reward_terms(**inputs(), dt=dt)


if __name__ == '__main__':
    unittest.main()
