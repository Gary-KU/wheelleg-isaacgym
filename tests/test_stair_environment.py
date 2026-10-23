"""Real PhysX contact, goal bookkeeping and curriculum-reset stair checks."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import isaacgym  # noqa: F401; the binary extension must precede Torch.
from isaacgym import gymtorch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wheelleg_gym.env as environment_module
from wheelleg_gym.env import WheelLegGymEnv
from wheelleg_gym.leg_control import hip_targets, hip_support_torques


class StairEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = WheelLegGymEnv(
            num_envs=2, sim_device="cpu", headless=True, physics_threads=0,
            terrain="stairs", observe_terrain=True, observe_leg_lengths=True,
            leg_control="hip_target", hip_target_scale=.2,
            initial_leg_length=.14, leg_kp=80., leg_kd=4.,
            leg_feedforward_force=22.4649, stair_curriculum=True,
            command=(0., 0., 0., .14), allow_approximate_loop=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        env = self.env
        env.stair_min_level = 0
        env.terrain_levels.zero_()
        env.stair_success_count.zero_()
        env.max_episode_length = 1000
        env.stair_curriculum = True
        env.reset()
        self.actions = torch.zeros(env.num_envs, 6)

    def translate_robot(self, index, position):
        """Move the real articulation and its matching pre-step closure cache."""
        env = self.env
        delta = env.root_states.new_tensor(position)-env.root_states[index, :3]
        env.root_states[index, :3] += delta
        env.root_states[index, 7:] = 0.
        env.rb_states[index, :, :3] += delta
        env.rb_states[index, :, 7:] = 0.
        ids = env.actor_ids[index:index+1].contiguous()
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, env._root_tensor, gymtorch.unwrap_tensor(ids), len(ids))
        env._kinematics()
        env._observations(advance_history=False)

    def settle_on_ground(self, steps=8):
        for _ in range(steps):
            _, _, _, dones, _ = self.env.step(self.actions)
            self.assertFalse(bool(dones.any()))

    def test_stair_mesh_preserves_actor_tensor_layout_and_exposes_terrain_observations(self):
        env = self.env
        self.assertEqual(env.num_obs, 252)
        self.assertEqual(env.observation_frame_size, 42)
        self.assertEqual(tuple(env.get_observations().shape), (2, 252))
        self.assertEqual(tuple(env.root_states.shape), (2, 13))
        self.assertEqual(tuple(env.rb_states.shape), (2, 11, 13))
        self.assertEqual(tuple(env.dof_state.shape), (2, 10, 2))
        self.assertEqual(env.actor_ids.tolist(), [0, 1])
        for handle in env.envs:
            self.assertEqual(env.gym.get_actor_count(handle), 1)
        selected = env.terrain_origins[env.environment_indices, env.terrain_levels]
        torch.testing.assert_close(env.root_states[:, :2], selected[:, :2], atol=1e-6, rtol=0)
        self.assertGreater(float(selected[1, 0]), 5.)
        self.assertLess(float(env.closure.errors(env.rb_states).max()), .0001)
        # The farthest sample sees the physical first vertical riser.
        self.assertGreater(float(env.slice_obs[:, -9:].max()), .049)

    def test_imported_top_step_supports_robot_at_its_real_height_in_second_patch(self):
        env = self.env
        index = 1
        origin = env.terrain_origins[index, 0]
        top_xy = origin[:2]+torch.tensor([2.0, 0.0])
        height = float(env.terrain_model.heights_at(top_xy[None], torch.tensor([index]), torch.tensor([0]))[0])
        self.assertAlmostEqual(height, .04, places=5)
        self.translate_robot(index, [float(top_xy[0]), float(top_xy[1]),
                                     float(env.default_root_states[index, 2])+height])
        last = None
        for _ in range(10):
            _, _, _, dones, last = env.step(self.actions)
            self.assertFalse(bool(dones.any()))
        self.assertAlmostEqual(float(last['terrain_ground_height'][index]), height, places=5)
        contacts = torch.linalg.vector_norm(env.contact_forces[index, env.wheel_indices], dim=-1)
        self.assertTrue(bool((contacts > 1.).all()))
        wheels = env.rb_states[index, env.wheel_indices, 2]
        torch.testing.assert_close(wheels-height, torch.full((2,), .0625), atol=.003, rtol=0)
        # Both robots use identical geometry: the raised step must physically
        # lift the second one, rather than only changing a queried height.
        root_heights = last['pre_reset_root_states'][:, 2]
        self.assertAlmostEqual(float(root_heights[1]-root_heights[0]), height, delta=.003)

    def test_hip_target_action_is_bounded_angle_offset_not_extra_hip_torque(self):
        env = self.env
        env.set_commands([0., 0., 0., .20])
        self.actions[:, :4] = torch.tensor([[-100., 100., 100., -100.],
                                            [100., -100., -100., 100.]])
        self.actions[:, 4:] = .4
        expected_targets = (hip_targets(env.commands[:, 3])
                            +.2*self.actions[:, :4].clamp(-1, 1)).clamp(env.hip_lower, env.hip_upper)
        load = hip_support_torques(env.dof_state[:, env.action_indices[:4], 0], env.leg_feedforward_force)
        targets_seen, last_pd = [], []
        original = environment_module.hip_pd_torques

        def read_pd(positions, velocities, targets, *gains):
            targets_seen.append(targets.clone())
            value = original(positions, velocities, targets, *gains)
            last_pd.append(value.clone())
            return value

        with mock.patch.object(environment_module, 'hip_pd_torques', side_effect=read_pd):
            _, _, rewards, _, info = env.step(self.actions)
        self.assertEqual(len(targets_seen), env.decimation)
        torch.testing.assert_close(targets_seen[0], expected_targets)
        self.assertTrue(bool((targets_seen[0] >= env.hip_lower).all()))
        self.assertTrue(bool((targets_seen[0] <= env.hip_upper).all()))
        torch.testing.assert_close(info['pre_reset_applied_torques'][:, :4],
                                   (last_pd[-1]+load).clamp(-9., 9.))
        torch.testing.assert_close(info['pre_reset_applied_torques'][:, 4:], torch.full((2, 2), 4.))
        self.assertLessEqual(float(info['pre_reset_peak_abs_torques'].max()), 9.)
        self.assertTrue(bool(torch.isfinite(rewards).all()))

    def test_success_at_time_limit_is_not_timeout_or_failure_and_promotes_lane(self):
        env = self.env
        self.settle_on_ground()
        origin = env.terrain_origins[0, 0]
        self.translate_robot(0, [float(origin[0])+env.terrain_model.finish_x+.15,
                                 float(origin[1]), float(env.root_states[0, 2])])
        env.episode_length_buf[:] = env.max_episode_length-1
        _, _, _, dones, info = env.step(self.actions)
        self.assertEqual(dones.tolist(), [True, True])
        self.assertEqual(info['terrain_success'].tolist(), [True, False])
        self.assertEqual(info['time_outs'].tolist(), [False, True])
        self.assertEqual(info['physical_failures'].tolist(), [False, False])
        self.assertEqual(info['terrain_level'].tolist(), [0, 0])
        self.assertGreater(float(info['terrain_progress'][0]), env.terrain_model.finish_x)
        self.assertEqual(info['reward_terms']['success'].tolist(), [25., 0.])
        self.assertEqual(info['reward_terms']['failure'].tolist(), [0., 0.])
        self.assertEqual(env.terrain_levels.tolist(), [1, 0])
        self.assertEqual(env.stair_success_count.tolist(), [1, 0])
        torch.testing.assert_close(env.root_states[0, :2], env.terrain_origins[0, 1, :2], atol=1e-6, rtol=0)
        self.assertLess(float(env.closure.errors(env.rb_states).max()), .0001)
        _, _, _, next_dones, next_info = env.step(self.actions)
        self.assertFalse(bool(next_dones.any()))
        self.assertLess(float(next_info['loop_error_max']), .005)

    def test_early_lane_failure_demotes_and_keeps_terminal_progress_and_pose(self):
        env = self.env
        env.terrain_levels[0] = 2
        env.reset()
        origin = env.terrain_origins[0, 2]
        displaced_y = float(origin[1])+env.terrain_model.lane_half_width-.02
        self.translate_robot(0, [float(origin[0]), displaced_y, float(env.root_states[0, 2])])
        _, _, _, dones, info = env.step(self.actions)
        self.assertEqual(dones.tolist(), [True, False])
        self.assertEqual(info['terrain_success'].tolist(), [False, False])
        self.assertEqual(info['physical_failures'].tolist(), [True, False])
        self.assertEqual(info['time_outs'].tolist(), [False, False])
        self.assertTrue(bool(info['termination_reasons']['lane'][0]))
        self.assertEqual(info['terrain_level'].tolist(), [2, 0])
        self.assertAlmostEqual(float(info['pre_reset_root_states'][0, 1]), displaced_y, delta=.003)
        self.assertEqual(env.terrain_levels.tolist(), [1, 0])
        torch.testing.assert_close(env.root_states[0, :2], env.terrain_origins[0, 1, :2], atol=1e-6, rtol=0)
        # Only env0 was reset; env1 retains its physically compliant state.
        self.assertLess(float(env.closure.errors(env.rb_states)[0].max()), .0001)

    def test_failure_at_curriculum_lower_bound_stays_on_requested_lane(self):
        env = self.env
        env.stair_min_level = 1
        env.terrain_levels.fill_(1)
        env.reset()
        origin = env.terrain_origins[0, 1]
        self.translate_robot(0, [float(origin[0]), float(origin[1])+env.terrain_model.lane_half_width-.02,
                                 float(env.root_states[0, 2])])
        _, _, _, dones, info = env.step(self.actions)
        self.assertEqual(dones.tolist(), [True, False])
        self.assertEqual(info['physical_failures'].tolist(), [True, False])
        self.assertEqual(env.terrain_levels.tolist(), [1, 1])
        torch.testing.assert_close(env.root_states[0, :2], env.terrain_origins[0, 1, :2], atol=1e-6, rtol=0)
        self.assertLess(float(env.closure.errors(env.rb_states)[0].max()), .0001)


if __name__ == '__main__':
    unittest.main()
