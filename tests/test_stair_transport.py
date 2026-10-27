"""A body leaning over blocked wheels must not masquerade as stair progress."""

import math
from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; load the SDK before Torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot


class StairTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = WheelLegRobot(
            num_envs=2, sim_device="cpu", headless=True, physics_threads=0,
            terrain="stairs", stair_level=1, observe_terrain=True,
            observe_leg_lengths=True, leg_control="hip_target", hip_target_scale=.2,
            initial_leg_length=.14, leg_feedforward_force=22.4649,
            command=(.25, 0., 0., .14), allow_approximate_loop=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.terrain_levels.fill_(1)
        self.env.reset()

    def tearDown(self):
        # Constructed diagnostic tests intentionally edit cached states without
        # simulating them. Restore both the SDK and its cache before reuse.
        self.env.reset()

    def test_leaning_falling_body_with_stationary_wheels_earns_no_transport_reward(self):
        env = self.env
        before = env.rb_states[:, env.wheel_indices, :2].mean(1).clone()
        progress = env.stair_progress.clone()
        env.root_states[:, 0] += .08
        env.root_states[:, 2] -= .04
        angle = .6
        env.root_states[:, 3:7] = torch.tensor([0., math.sin(angle/2), 0., math.cos(angle/2)])
        env.root_states[:, 7:10] = torch.tensor([.2, 0., -.3])
        env._kinematics()
        self.assertTrue(bool((env.lin_vel[:, 0] > .3).all()))
        env._stair_diagnostics(before)
        terms_total = env._rewards(torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(env.stair_progress, progress, atol=0., rtol=0.)
        torch.testing.assert_close(env.stair_delta_x, torch.zeros(2), atol=0., rtol=0.)
        torch.testing.assert_close(env.stair_transport_velocity, torch.zeros(2, 3), atol=0., rtol=0.)
        torch.testing.assert_close(env.reward_terms['progress'], torch.zeros(2), atol=0., rtol=0.)
        expected_tracking = 4*math.exp(-.25**2/.04)
        torch.testing.assert_close(env.reward_terms['velocity'], torch.full((2,), expected_tracking))
        self.assertTrue(bool(torch.isfinite(terms_total).all()))

    def test_body_crossing_first_riser_does_not_erase_height_signal_before_wheels_cross(self):
        env = self.env
        origins = env.terrain_origins[env.environment_indices, env.terrain_levels]
        env.rb_states[:, env.wheel_indices, 0] = origins[:, None, 0]+.404
        env.root_states[:, 0] = origins[:, 0]+.449
        env._kinematics()
        env._observations(advance_history=False)
        before_scan = env.slice_obs[:, -9:].clone()
        env.root_states[:, 0] = origins[:, 0]+.451
        ground_below_body = env.terrain_model.heights_at(
            env.root_states[:, :2], env.environment_indices, env.terrain_levels)
        self.assertTrue(bool((ground_below_body > .015).all()))
        env._kinematics()
        env._observations(advance_history=False)
        torch.testing.assert_close(env.stair_ground_height, torch.zeros(2), atol=0., rtol=0.)
        torch.testing.assert_close(env.slice_obs[:, -9:], before_scan, atol=1e-6, rtol=0.)
        self.assertTrue(bool((env.slice_obs[:, -9:-6] > .075).all()))
        self.assertTrue(bool((env.stair_progress < env.terrain_model.start_x).all()))

    def test_ground_reference_averages_left_and_right_support_on_split_height_steps(self):
        env = self.env
        env.terrain_levels.fill_(3)
        env.reset()
        origins = env.terrain_origins[env.environment_indices, env.terrain_levels]
        env.rb_states[:, env.wheel_indices, 0] = origins[:, None, 0]+2.
        env.rb_states[:, env.wheel_indices[0], 1] = origins[:, 1]-.108
        env.rb_states[:, env.wheel_indices[1], 1] = origins[:, 1]+.108
        wheel_ground = env.terrain_model.heights_at(
            env.rb_states[:, env.wheel_indices, :2], env.environment_indices, env.terrain_levels)
        self.assertTrue(bool((wheel_ground[:, 0]-wheel_ground[:, 1]).abs().gt(1e-6).all()))
        env._kinematics()
        torch.testing.assert_close(env.stair_ground_height, wheel_ground.mean(-1), atol=0., rtol=0.)

    def test_real_physics_step_reports_wheel_finite_difference_and_uses_it_for_reward(self):
        env = self.env
        before = env.rb_states[:, env.wheel_indices, :2].mean(1).clone()
        actions = torch.zeros(2, 6)
        actions[:, 4:] = torch.tensor([-.02, .02])
        _, _, _, dones, info = env.step(actions)
        self.assertFalse(bool(dones.any()))
        after = info['pre_reset_wheel_states'][:, :, :2].mean(1)
        measured = info['terrain_transport_velocity_world_m_per_s']
        torch.testing.assert_close(measured[:, :2], (after-before)/env.dt, atol=0., rtol=0.)
        torch.testing.assert_close(measured[:, 2], torch.zeros(2), atol=0., rtol=0.)
        origins = env.terrain_origins[env.environment_indices, info['terrain_level']]
        torch.testing.assert_close(info['terrain_progress'], after[:, 0]-origins[:, 0], atol=0., rtol=0.)
        torch.testing.assert_close(info['reward_terms']['progress'], 9*measured[:, 0], atol=1e-7, rtol=1e-6)
        wheel_ground = env.terrain_model.heights_at(
            info['pre_reset_wheel_states'][:, :, :2], env.environment_indices, info['terrain_level'])
        torch.testing.assert_close(info['terrain_ground_height'], wheel_ground.mean(-1), atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
