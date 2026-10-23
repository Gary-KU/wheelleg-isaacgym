"""Small real-PhysX regressions for the migrated model and episode contract.

Run through run.sh. WHEELLEG_TEST_DEVICE=cuda:0 selects the GPU pipeline;
the default CPU pipeline exercises the same two-environment state transitions.
"""

import math
import os
from pathlib import Path
import sys
import unittest

from isaacgym import gymapi, gymtorch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.env import ACTION_JOINTS, WheelLegGymEnv


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WheelLegGymEnv(
            num_envs=2, sim_device=os.environ.get("WHEELLEG_TEST_DEVICE", "cpu"),
            headless=True, seed=17, allow_approximate_loop=True,
        )
        # Isolate episode bookkeeping from falling/contact dynamics without
        # replacing any simulation calls: suspend the real robot above ground.
        params = cls.env.gym.get_sim_params(cls.env.sim)
        params.gravity = gymapi.Vec3(0, 0, 0)
        cls.env.gym.set_sim_params(cls.env.sim, params)
        shift = 1.0 - cls.env.default_root_states[:, 2]
        cls.env.default_root_states[:, 2] += shift
        cls.env.default_rb_states[:, :, 2] += shift.unsqueeze(1)
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.initial_observations, _ = self.env.reset()
        self.initial_observations = self.initial_observations.clone()
        self.zero_actions = torch.zeros(2, 6, device=self.env.device)

    def test_imported_model_contract_and_initial_closure(self):
        env = self.env
        self.assertEqual(self.initial_observations.shape, (2, 186))
        self.assertEqual(env.num_actions, 6)
        self.assertEqual(len(env.body_names), 11)
        self.assertEqual(len(env.dof_names), 10)
        self.assertEqual(set(ACTION_JOINTS), set(env.metadata["active_joints"]))
        self.assertLess(env.closure.errors(env.rb_states).max().item(), 0.0001)
        self.assertLess(env.closure.axis_errors(env.rb_states).max().item(), 0.0001)
        props = env.gym.get_actor_dof_properties(env.envs[0], env.actors[0])
        for name in list(ACTION_JOINTS) + env.metadata["passive_joints"]:
            index = env.dof_names.index(name)
            self.assertEqual(props["stiffness"][index], 0)
            self.assertEqual(props["damping"][index], 0)
            self.assertEqual(props["driveMode"][index], gymapi.DOF_MODE_EFFORT)
        active = set(ACTION_JOINTS)
        self.assertTrue(active.isdisjoint(env.metadata["passive_joints"]))

    def test_reset_initial_frame_is_available_on_next_step(self):
        observations, _, _, dones, _ = self.env.step(self.zero_actions)
        self.assertFalse(dones.any().item())
        # A six-frame observation must shift by precisely one frame.
        torch.testing.assert_close(
            observations[:, :-31], self.initial_observations[:, 31:],
            rtol=0, atol=0,
        )
        self.assertTrue(torch.isfinite(observations).all().item())

    def test_partial_reset_preserves_other_environments_history(self):
        env = self.env
        previous = self.initial_observations
        # Distinct visible commands make repeated/dropped history frames visible.
        for velocity in (0.1, 0.2):
            env.commands[1, 0] = velocity
            observations, _, _, dones, _ = env.step(self.zero_actions)
            self.assertFalse(dones.any().item())
            previous = observations.clone()
        env.episode_length_buf[0] = env.max_episode_length - 1
        env.commands[1, 0] = 0.3
        observations, _, _, dones, infos = env.step(self.zero_actions)
        self.assertEqual(dones.tolist(), [True, False])
        self.assertEqual(infos["time_outs"].tolist(), [True, False])
        torch.testing.assert_close(observations[1, :-31], previous[1, 31:], rtol=0, atol=0)
        torch.testing.assert_close(observations[0, :-31], torch.zeros_like(observations[0, :-31]))
        torch.testing.assert_close(observations[0, -31:], self.initial_observations[0, -31:])
        self.assertEqual(infos["terminal_env_ids"].tolist(), [0])

    def test_physical_failure_at_time_limit_is_not_timeout_bootstrap(self):
        env = self.env
        # This actual root-pose change remains tilted throughout the zero-gravity
        # free-flight step. Env 1 reaches the same time limit upright.
        angle = 0.8
        env.root_states[0, 3:7] = torch.tensor(
            [math.sin(angle / 2), 0, 0, math.cos(angle / 2)], device=env.device)
        ids = env.actor_ids[:1].contiguous()
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, env._root_tensor, gymtorch.unwrap_tensor(ids), 1)
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, _, dones, infos = env.step(self.zero_actions)
        self.assertEqual(dones.tolist(), [True, True])
        self.assertEqual(infos["time_outs"].tolist(), [False, True])
        self.assertEqual(infos["terminal_env_ids"].tolist(), [0, 1])
        self.assertEqual(infos["termination_reasons"]["roll"].tolist(), [True, False])
        self.assertGreater(abs(infos["pre_reset_roll"][0].item()), 0.5)
        self.assertLess(abs(env.roll[0].item()), 1e-6)
        torch.testing.assert_close(infos["pre_reset_commands"], env.default_command.repeat(2, 1))
        # Confirm the physical tilt reached the observation before autoreset.
        self.assertGreater(abs(infos["terminal_observation"][0, -31 + 7].item()), 0.5)
        self.assertEqual(infos["pre_reset_dof_positions"].shape, (2, 10))
        self.assertEqual(infos["pre_reset_passive_limit_margin"].shape, (2, 4))

    def test_interactive_command_updates_latest_frame_without_advancing_history(self):
        env = self.env
        previous_obs = env.obs_buf.clone()
        previous_history = env.history.clone()
        previous_lengths = env.episode_length_buf.clone()
        current = env.set_commands([0.3, 0.0, -0.8, 0.1])
        self.assertEqual(current.shape, (2, 186))
        torch.testing.assert_close(current[:, :-31], previous_obs[:, :-31], rtol=0, atol=0)
        torch.testing.assert_close(env.history[:, :-1], previous_history[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(env.episode_length_buf, previous_lengths, rtol=0, atol=0)
        expected = torch.tensor([0.45, 0.0, -0.8, 0.5], device=env.device).repeat(2, 1)
        torch.testing.assert_close(current[:, -31+9:-31+13], expected)
        torch.testing.assert_close(env.history[:, -1], current[:, -31:])
        torch.testing.assert_close(current[:, -2], (env.lin_vel[:, 0]-0.3)*1.5)
        observations, _, _, dones, _ = env.step(self.zero_actions)
        self.assertFalse(dones.any().item())
        torch.testing.assert_close(observations[:, -62+9:-62+13], expected)

    def test_invalid_interactive_command_does_not_change_state(self):
        env = self.env
        before = env.commands.clone()
        for command in ([float("nan"), 0, 0, 0.08], [0, 0, 0, 0.5], [0, 0]):
            with self.assertRaises(ValueError):
                env.set_commands(command)
            torch.testing.assert_close(env.commands, before, rtol=0, atol=0)

    def test_support_reward_uses_imported_limits_and_preserves_pre_reset_measurements(self):
        env = self.env
        previous = env.support_rewards
        try:
            env.support_rewards = True
            _, _, rewards, dones, info = env.step(self.zero_actions)
            self.assertFalse(dones.any().item())
            self.assertTrue(torch.isfinite(rewards).all().item())
            self.assertEqual(set(info["reward_terms"]), {
                "leg_tracking", "leg_error", "leg_asymmetry", "passive_limit"})
            passive = info["pre_reset_dof_positions"][:, env.passive_indices]
            expected = torch.minimum(passive-env.passive_lower, env.passive_upper-passive)
            torch.testing.assert_close(info["pre_reset_passive_limit_margin"], expected)
            for term in info["reward_terms"].values():
                self.assertEqual(term.shape, (2,))
        finally:
            env.support_rewards = previous


if __name__ == "__main__":
    unittest.main()
