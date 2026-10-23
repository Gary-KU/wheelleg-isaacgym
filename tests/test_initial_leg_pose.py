"""Opt-in curriculum poses must be closed, grounded and free to fall under gravity."""

import json
import os
from pathlib import Path
import sys
import unittest

import isaacgym  # noqa: F401; extension must load before Torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.env import ASSET_ROOT, WheelLegGymEnv
from wheelleg_gym.initial_pose import make_initial_pose, symmetric_joint_positions


class InitialPoseGeometryTests(unittest.TestCase):
    def test_full_eight_cm_pose_uses_real_collision_clearance_and_closes_both_loops(self):
        metadata = json.loads((ASSET_ROOT/"loop_constraints.json").read_text())
        pose = make_initial_pose(ASSET_ROOT/metadata["urdf"], metadata, 0.08)
        self.assertEqual(len(pose["joint_positions"]), 10)
        self.assertAlmostEqual(pose["joint_positions"]["L_F_Joint1"], 0.6139891090, places=8)
        self.assertAlmostEqual(pose["joint_positions"]["R_F_Joint2"], 0.9546142124, places=8)
        self.assertGreater(pose["root_position"][2], 0.14)
        self.assertLess(pose["root_position"][2], 0.15)
        self.assertAlmostEqual(min(pose["minimum_collision_height_m_by_body"].values()), 0.001, places=8)
        self.assertLess(max(pose["closure_error_m_by_constraint"].values()), 0.0001)

    def test_requested_length_is_checked_and_not_silently_clamped(self):
        for length in (0.049, 0.201, float("nan")):
            with self.subTest(length=length), self.assertRaises(ValueError):
                symmetric_joint_positions(length)
        for length in (0.05, 0.15, 0.20):
            self.assertEqual(len(symmetric_joint_positions(length)), 10)


class InitialPosePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = WheelLegGymEnv(num_envs=2, sim_device="cpu", headless=True,
                                 initial_leg_length=0.08, allow_approximate_loop=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_initial_sdk_fk_matches_leg_target_and_gravity_is_restored(self):
        env = self.env
        self.assertAlmostEqual(env.gym.get_sim_params(env.sim).gravity.z, -9.81, places=5)
        torch.testing.assert_close(env.length_l, torch.full((2,), 0.08), atol=0.0001, rtol=0)
        torch.testing.assert_close(env.length_r, torch.full((2,), 0.08), atol=0.0001, rtol=0)
        self.assertLess(float(env.closure.errors(env.rb_states).max()), 0.0001)
        self.assertGreater(float(env.passive_limit_margin.min()), 0.25)
        self.assertLess(float(env.root_states[:, 7:].abs().max()), 1e-6)
        self.assertLess(float(env.rb_states[:, :, 7:].abs().max()), 1e-6)

    def test_reset_restores_matching_sdk_pose_cache_and_full_joint_state(self):
        env = self.env
        for _ in range(3):
            env.step(torch.zeros(2, 6))
        env.reset()
        torch.testing.assert_close(env.root_states, env.default_root_states, atol=0, rtol=0)
        torch.testing.assert_close(env.rb_states, env.default_rb_states, atol=0, rtol=0)
        torch.testing.assert_close(env.dof_state[:, :, 0], env.default_dof_pos.repeat(2, 1), atol=0, rtol=0)
        self.assertEqual(float(env.dof_state[:, :, 1].abs().max()), 0)
        _, _, _, _, infos = env.step(torch.zeros(2, 6))
        self.assertLess(float(infos["loop_error_max"]), 0.005)

    def test_zero_actions_do_not_artificially_hold_the_curriculum_height(self):
        env = self.env
        minimum = 1.0
        for _ in range(25):
            _, _, _, _, infos = env.step(torch.zeros(2, 6))
            minimum = min(minimum, float(infos["pre_reset_leg_length"].min()))
            self.assertEqual(float(infos["pre_reset_applied_torques"].abs().max()), 0)
        self.assertLess(minimum, 0.075)


class LegacyInitialPoseTests(unittest.TestCase):
    def test_none_preserves_old_initial_joints_and_root_height(self):
        env = WheelLegGymEnv(num_envs=1, sim_device="cpu", headless=True,
                             initial_leg_length=None, allow_approximate_loop=True)
        try:
            self.assertEqual(float(env.dof_state[:, :, 0].abs().max()), 0)
            self.assertAlmostEqual(float(env.root_states[0, 2]), 0.25, delta=1e-5)
            self.assertGreater(float(env.length_l[0]), 0.18)
            self.assertFalse(hasattr(env, "initial_pose_diagnostics"))
        finally:
            env.close()


@unittest.skipUnless(os.environ.get("WHEELLEG_TEST_DEVICE", "cpu").startswith("cuda"),
                     "Set WHEELLEG_TEST_DEVICE=cuda:0 for the finite GPU initialization check")
class InitialPoseGpuTests(unittest.TestCase):
    def test_gpu_fk_translation_reset_and_real_gravity(self):
        env = WheelLegGymEnv(num_envs=8, sim_device=os.environ["WHEELLEG_TEST_DEVICE"], headless=True,
                             initial_leg_length=0.08, allow_approximate_loop=True)
        try:
            self.assertLess(float(env.closure.errors(env.rb_states).max()), 0.0001)
            self.assertLess(float(env.rb_states[:, :, 7:].abs().max()), 1e-6)
            torch.testing.assert_close(env.length_l, torch.full_like(env.length_l, 0.08), atol=0.0001, rtol=0)
            minimum = 1.0
            for _ in range(25):
                _, _, _, _, infos = env.step(torch.zeros(8, 6, device=env.device))
                minimum = min(minimum, float(infos["pre_reset_leg_length"].min()))
                self.assertEqual(float(infos["pre_reset_applied_torques"].abs().max()), 0)
            self.assertLess(minimum, 0.075)
            env.reset()
            self.assertLess(float(env.closure.errors(env.rb_states).max()), 0.0001)
            torch.testing.assert_close(env.length_r, torch.full_like(env.length_r, 0.08), atol=0.0001, rtol=0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
