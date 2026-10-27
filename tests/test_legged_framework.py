"""Framework behavior: config isolation, reward parity, six-value API and resume."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import isaacgym
import torch
from legged_gym.envs import register_tasks

register_tasks()
from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot
from legged_gym.envs.wheelleg.wheelleg import WheelLegged, environment_kwargs
from legged_gym.envs.wheelleg.wheelleg_config import WheelLeggedCfg
from legged_gym.utils import task_registry, class_to_dict, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym.learning.on_policy_runner import OnPolicyRunner
from tools.prepare_support_checkpoint import prepare_support_checkpoint


class FrameworkConfigTests(unittest.TestCase):
    def test_registered_tasks_have_independent_nested_configs(self):
        a, ta = task_registry.get_cfgs("wheelleg_flat")
        b, tb = task_registry.get_cfgs("wheelleg_stand")
        a.commands.command = [0.1, 0, 0, 0.09]
        ta.policy.actor_hidden_dims.append(7)
        self.assertNotEqual(a.commands.command, b.commands.command)
        self.assertNotIn(7, tb.policy.actor_hidden_dims)
        self.assertNotIn(
            7, task_registry.get_cfgs("wheelleg_flat")[1].policy.actor_hidden_dims
        )
        self.assertEqual(class_to_dict(b)["control"]["decimation"], 40)
        self.assertIsNot(
            WheelLeggedCfg().rewards.scales, WheelLeggedCfg().rewards.scales
        )

    def test_checkpoint_selection_uses_numeric_iterations_and_real_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run1"
            run.mkdir()
            for n in (9, 10, 100):
                (run / ("model_{}.pt".format(n))).touch()
            (Path(tmp) / "empty_new_run").mkdir()
            self.assertEqual(Path(get_load_path(tmp)).name, "model_100.pt")
            self.assertEqual(Path(get_load_path(tmp, "run1", 10)).name, "model_10.pt")
            with self.assertRaises(ValueError):
                get_load_path(tmp, "run1", 11)

    def test_zero_rewards_are_removed_and_unknown_rewards_fail_early(self):
        env = object.__new__(WheelLegged)
        env.cfg, _ = task_registry.get_cfgs("wheelleg_flat")
        env.cfg.rewards.scales.effort = 0.0
        env._prepare_reward_function()
        self.assertNotIn("effort", env.reward_scales)
        env.cfg.rewards.scales.typo = 1.0
        with self.assertRaisesRegex(ValueError, "Missing _reward_typo"):
            env._prepare_reward_function()

    def test_prepare_restores_legacy_dimensions_and_applies_execution_overrides(self):
        args = get_args(
            "play", ["--task", "wheelleg_stand", "--num_envs", "2", "--headless"]
        )
        cfg, train = task_registry.prepare_cfgs(args.task, args)
        self.assertEqual(cfg.env.num_envs, 2)
        self.assertEqual(cfg.env.num_observations, 198)
        self.assertEqual(cfg.control.heading_kp, 6.0)
        self.assertFalse(cfg.commands.random_commands)
        self.assertEqual(train.policy.actor_hidden_dims, [512, 256, 128, 64])
        args = get_args(
            "play", ["--task", "wheelleg_stairs", "--headless", "--stair_level", "1"]
        )
        cfg, _ = task_registry.prepare_cfgs(args.task, args)
        self.assertEqual(cfg.env.num_observations, 252)
        self.assertEqual(cfg.terrain.stair_level, 1)
        self.assertFalse(cfg.terrain.stair_curriculum)

    def test_wrong_checkpoint_observation_layout_is_rejected(self):
        args = get_args(
            "play",
            [
                "--task",
                "wheelleg_stairs",
                "--headless",
                "--checkpoint",
                "logs/standing_support_20261022_093823/input/standing_hold.pt",
            ],
        )
        with self.assertRaisesRegex(ValueError, "observation layout"):
            task_registry.prepare_cfgs(args.task, args)

    def test_earlier_nested_checkpoint_accepts_removed_cli_placeholders(self):
        env_cfg, train_cfg = task_registry.get_cfgs("wheelleg_stand")
        env_cfg.control.leg_kp = 75.0
        snapshot = class_to_dict(train_cfg)
        snapshot["runner"].update(resume=True, load_run="old", checkpoint=-1)
        snapshot["initialization_provenance"] = {"source_iteration": 42}
        state = {
            "model_state_dict": {"actor.0.weight": torch.zeros(1, 198)},
            "legged_gym_env_config": class_to_dict(env_cfg),
            "legged_gym_train_config": snapshot,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_framework.pt"
            torch.save(state, str(path))
            args = get_args(
                "train",
                [
                    "--task",
                    "wheelleg_stand",
                    "--resume",
                    "--checkpoint",
                    str(path),
                    "--headless",
                ],
            )
            restored_env, restored_train = task_registry.prepare_cfgs(args.task, args)
        self.assertEqual(restored_env.control.leg_kp, 75.0)
        self.assertEqual(
            restored_train.initialization_provenance, {"source_iteration": 42}
        )
        self.assertNotIn("resume", class_to_dict(restored_train.runner))

    def test_converted_checkpoint_restores_new_objective_and_provenance(self):
        # A registered-task source carries nested snapshots. Conversion must
        # restore the new support objective rather than the source control mode.
        source = torch.load(
            "logs/standing_support_20261022_093823/input/standing_hold.pt",
            map_location="cpu",
        )
        env_cfg, train_cfg = task_registry.get_cfgs("wheelleg_stand")
        env_cfg.control.leg_control = "hip_target"
        source["legged_gym_env_config"] = class_to_dict(env_cfg)
        source["legged_gym_train_config"] = class_to_dict(train_cfg)
        converted = prepare_support_checkpoint(source, source_path="source.pt")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "support.pt"
            torch.save(converted, str(path))
            args = get_args(
                "train",
                [
                    "--task",
                    "wheelleg_stand",
                    "--resume",
                    "--checkpoint",
                    str(path),
                    "--headless",
                ],
            )
            restored_env, restored_train = task_registry.prepare_cfgs(args.task, args)
        self.assertEqual(restored_env.control.leg_control, "torque")
        self.assertEqual(
            restored_train.initialization_provenance, converted["support_bootstrap"]
        )


class FrameworkPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg, _ = task_registry.get_cfgs("wheelleg_stand")
        cls.cfg.env.num_envs = 8
        cls.env = WheelLegged(cls.cfg, sim_device="cpu", headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_six_value_step_and_single_history_stack_keep_legacy_rewards(self):
        e = self.env
        for _ in range(5):
            result = e.step(torch.randn(8, 6) * 0.01)
            self.assertEqual(len(result), 6)
            self.assertEqual(result[0].shape, (8, 198))
            self.assertIs(result[0], result[5])
            self.assertIs(e.get_observations()[0], e.obs_buf)
            actual = e._rewards(torch.zeros(8, dtype=torch.bool)).clone()
            expected = WheelLegRobot._rewards(e, torch.zeros(8, dtype=torch.bool))
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
        self.assertEqual(
            e.environment_config,
            dict(environment_kwargs(e.cfg), sim_device="cpu", headless=True),
        )

    def test_runner_saves_completed_iterations_and_resumes_without_counter_reset(self):
        e = self.env
        _, cfg = task_registry.get_cfgs("wheelleg_stand")
        cfg.policy.actor_hidden_dims = [16, 8]
        cfg.policy.critic_hidden_dims = [16, 8]
        cfg.algorithm.num_learning_epochs = 1
        cfg.algorithm.num_mini_batches = 2
        cfg.runner.num_steps_per_env = 4
        cfg.runner.save_interval = 1
        with tempfile.TemporaryDirectory() as tmp:
            runner = OnPolicyRunner(e, class_to_dict(cfg), tmp, device="cpu")
            runner.learn(2)
            state = torch.load(str(Path(tmp) / "model_2.pt"), map_location="cpu")
            self.assertEqual(state["iter"], 2)
            self.assertEqual(state["total_steps"], 64)
            self.assertIn("legged_gym_env_config", state)
            resumed = OnPolicyRunner(e, class_to_dict(cfg), tmp, device="cpu")
            resumed.load(Path(tmp) / "model_2.pt")
            resumed.learn(1)
            state = torch.load(str(Path(tmp) / "model_3.pt"), map_location="cpu")
            self.assertEqual(state["iter"], 3)
            self.assertEqual(state["total_steps"], 96)
            self.assertEqual(resumed.current_learning_iteration, 3)

    def test_timeouts_keep_terminal_states_before_automatic_reset(self):
        e = self.env
        old_limit = e.max_episode_length
        try:
            e.max_episode_length = 1
            obs, _, _, done, info, history = e.step(torch.zeros(8, 6))
            self.assertTrue(done.all())
            self.assertTrue(info["time_outs"].all())
            self.assertEqual(info["terminal_observation"].shape, (8, 198))
            self.assertTrue(torch.isfinite(obs).all())
            self.assertTrue((e.episode_length_buf == 0).all())
        finally:
            e.max_episode_length = old_limit


class CurriculumResumeTests(unittest.TestCase):
    def test_loaded_levels_relocate_actors_and_observations_before_next_step(self):
        cfg, train = task_registry.get_cfgs("wheelleg_stairs")
        cfg.env.num_envs = 2
        train.policy.actor_hidden_dims = [16, 8]
        train.policy.critic_hidden_dims = [16, 8]
        env = WheelLegged(cfg, sim_device="cpu", headless=True)
        try:
            runner = OnPolicyRunner(env, class_to_dict(train), None, device="cpu")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model_0.pt"
                runner.save(path)
                state = torch.load(str(path), map_location="cpu")
                state["terrain_curriculum_state"]["levels"] = [1, 2]
                torch.save(state, str(path))
                runner.load(path, load_optimizer=False)
            torch.testing.assert_close(env.terrain_levels, torch.tensor([1, 2]))
            expected_xy = env.terrain_origins[
                env.environment_indices, env.terrain_levels, :2
            ]
            torch.testing.assert_close(env.root_states[:, :2], expected_xy)
            obs, _ = env.get_observations()
            self.assertTrue(torch.isfinite(obs).all())
            _, _, _, _, info, _ = env.step(torch.zeros(2, 6))
            self.assertFalse(info["physical_failures"].any())
            torch.testing.assert_close(env.terrain_levels, torch.tensor([1, 2]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
