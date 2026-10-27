"""Both public runner APIs must execute the same updates and checkpoint rules."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import isaacgym  # noqa: F401; the SDK must load before Torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.learning.ppo_runner import PPORunner
from legged_gym.envs.wheelleg.diagnostics.training_metrics import (
    WheelLegTrainingMetrics,
)
from legged_gym.learning.on_policy_runner import OnPolicyRunner


def training_config():
    return {
        "policy": {
            "actor_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "init_noise_std": 0.1,
            "activation": "elu",
        },
        "algorithm": {
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 0.001,
            "max_learning_rate": 0.01,
            "schedule": "fixed",
        },
        "num_steps_per_env": 4,
        "save_interval": 1,
    }


def nested_config(config):
    return {
        "seed": 42,
        "policy": deepcopy(config["policy"]),
        "algorithm": deepcopy(config["algorithm"]),
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": config["num_steps_per_env"],
            "save_interval": config["save_interval"],
            "max_iterations": 2,
        },
    }


class DiagnosticEnv:
    """Deterministic batched task, including timeout snapshots and rich metrics."""

    device = "cpu"
    num_envs = 4
    num_obs = 198
    num_actions = 6
    num_privileged_obs = None
    obs_history_length = 1
    observation_frame_size = 33
    max_episode_length = 3
    dt = 0.02
    viewer = None
    terrain_model = None
    stair_curriculum = False

    def __init__(self):
        self.environment_config = {
            "num_envs": self.num_envs,
            "observe_leg_lengths": True,
        }
        self.cfg = SimpleNamespace(env=SimpleNamespace(num_envs=self.num_envs))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_count = 0
        self.reset()

    def reset(self):
        self.reset_count += 1
        self.obs_buf = torch.linspace(-0.1, 0.1, self.num_envs * self.num_obs).reshape(
            self.num_envs, self.num_obs
        )
        self.episode_length_buf.zero_()
        return self.obs_buf, None

    def get_observations(self):
        return self.obs_buf, self.obs_buf

    def create_training_metrics(self):
        return WheelLegTrainingMetrics(self)

    def step(self, actions):
        self.obs_buf = 0.95 * self.obs_buf + 0.01 * actions.mean(-1, keepdim=True)
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        rewards = 1.0 - actions.square().mean(-1)
        ids = dones.nonzero(as_tuple=False).flatten()
        commands = torch.zeros(self.num_envs, 4)
        commands[:, 3] = 0.08
        info = {
            "pre_reset_commands": commands,
            "pre_reset_applied_torques": actions.clone(),
            "pre_reset_previous_torques": torch.zeros_like(actions),
            "pre_reset_pitch": self.obs_buf[:, 0].clone(),
            "pre_reset_roll": self.obs_buf[:, 1].clone(),
            "pre_reset_ang_vel": self.obs_buf[:, :3].clone(),
            "pre_reset_leg_length": torch.full((self.num_envs, 2), 0.08),
            "pre_reset_passive_limit_margin": torch.ones(self.num_envs, 4),
            "terminal_env_ids": ids,
            "terminal_observation": self.obs_buf[ids].clone(),
            "time_outs": dones.clone(),
            "reward_terms": {"tracking": rewards.clone()},
            "loop_error_max": 0.001,
        }
        if ids.numel():
            info["episode"] = {
                "length": float(self.max_episode_length),
                "reward": rewards[ids].mean(),
            }
            self.obs_buf[ids] = 0.0
            self.episode_length_buf[ids] = 0
        return self.obs_buf, None, rewards, dones, info


class SixValueEnv(DiagnosticEnv):
    def step(self, actions):
        result = super().step(actions)
        return result + (result[0],)


class MinimalEnv:
    """A different task with no robot fields or diagnostic infos."""

    device = "cpu"
    num_envs = 4
    num_obs = 3
    num_actions = 2

    def reset(self):
        self.obs = torch.zeros(self.num_envs, self.num_obs)
        return self.obs, None

    def step(self, actions):
        self.obs = 0.9 * self.obs + 0.01 * actions.mean(-1, keepdim=True)
        rewards = 1.0 - actions.square().mean(-1)
        return self.obs, None, rewards, torch.zeros(self.num_envs, dtype=torch.bool), {}


class RunnerCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_both_apis_produce_identical_updates_counters_and_legacy_reports(self):
        config = training_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(17)
            legacy = PPORunner(DiagnosticEnv(), root / "legacy", config=config)
            torch.manual_seed(17)
            modern = OnPolicyRunner(
                SixValueEnv(), nested_config(config), root / "modern"
            )
            torch.manual_seed(31)
            legacy.learn(2)
            torch.manual_seed(31)
            modern.learn(2)
            self.assert_nested_equal(
                legacy.policy.state_dict(), modern.alg.actor_critic.state_dict()
            )
            self.assert_nested_equal(
                legacy.algorithm.optimizer.state_dict(),
                modern.alg.optimizer.state_dict(),
            )
            self.assertEqual(
                (modern.current_learning_iteration, modern.tot_timesteps), (2, 32)
            )
            for run in ("legacy", "modern"):
                summary = json.loads((root / run / "training_summary.json").read_text())
                self.assertEqual(summary["completed_iterations"], 2)
                self.assertEqual(summary["total_environment_steps"], 32)
                self.assertEqual(len(summary["iterations"]), 2)
                self.assertIn("Motion/rms_pitch_rad", summary["iterations"][-1])
                self.assertIn("Loop/max_anchor_error_m", summary["iterations"][-1])
            checkpoint = torch.load(root / "modern" / "model_2.pt", map_location="cpu")
            self.assertIn("legged_gym_env_config", checkpoint)
            self.assertEqual(
                checkpoint["legged_gym_train_config"]["policy"], config["policy"]
            )
            self.assertIsNone(modern.writer)

    def test_resume_preserves_adam_history_provenance_and_clamps_learning_rate(self):
        config = training_config()
        config["initialization_provenance"] = {"source_checkpoint": "warm-start.pt"}
        with tempfile.TemporaryDirectory() as directory:
            source = PPORunner(DiagnosticEnv(), None, config=config)
            source.learn(1)
            checkpoint = Path(directory) / "resume.pt"
            source.save(checkpoint, infos={"source": "test"})
            target_config = training_config()
            target_config["algorithm"]["max_learning_rate"] = 0.0003
            resumed = OnPolicyRunner(
                SixValueEnv(), nested_config(target_config), directory
            )
            self.assertEqual(resumed.load(checkpoint), {"source": "test"})
            self.assert_nested_equal(
                source.algorithm.optimizer.state_dict()["state"],
                resumed.alg.optimizer.state_dict()["state"],
            )
            self.assertEqual(resumed.alg.learning_rate, 0.0003)
            self.assertTrue(
                all(
                    group["lr"] == 0.0003
                    for group in resumed.alg.optimizer.param_groups
                )
            )
            self.assertEqual(
                (resumed.current_learning_iteration, resumed.tot_timesteps), (1, 16)
            )
            resumed.learn(1)
            state = torch.load(Path(directory) / "model_2.pt", map_location="cpu")
            self.assertEqual((state["iter"], state["total_steps"]), (2, 32))
            self.assertEqual(
                state["training_config"]["initialization_provenance"],
                config["initialization_provenance"],
            )
            self.assertEqual(
                state["legged_gym_train_config"]["initialization_provenance"],
                config["initialization_provenance"],
            )

    def test_missing_optimizer_is_a_supported_warm_start(self):
        config = training_config()
        with tempfile.TemporaryDirectory() as directory:
            source = PPORunner(DiagnosticEnv(), None, config=config)
            checkpoint = Path(directory) / "warm-start.pt"
            source.save(checkpoint)
            state = torch.load(checkpoint, map_location="cpu")
            state.pop("optimizer_state_dict")
            torch.save(state, checkpoint)
            target = OnPolicyRunner(SixValueEnv(), nested_config(config), None)
            target.load(checkpoint)
            self.assertEqual(target.alg.optimizer.state_dict()["state"], {})
            observations = target.env.obs_buf
            torch.testing.assert_close(
                target.get_inference_policy()(observations),
                source.policy.act_inference(observations),
                rtol=0,
                atol=0,
            )
            target.learn(1)
            self.assertEqual(
                (target.current_learning_iteration, target.tot_timesteps), (1, 16)
            )

    def test_no_log_directory_creates_no_files_or_tensorboard_writer(self):
        with patch("torch.utils.tensorboard.SummaryWriter") as writer:
            with patch("legged_gym.learning.ppo_runner.Path.mkdir") as mkdir:
                with patch("legged_gym.learning.ppo_runner.Path.write_text") as write:
                    runner = OnPolicyRunner(
                        SixValueEnv(), nested_config(training_config()), None
                    )
                    runner.learn(1, init_at_random_ep_len=True)
            writer.assert_not_called()
            mkdir.assert_not_called()
            write.assert_not_called()
            self.assertIsNone(runner.log_dir)
            self.assertIsNone(runner.writer)

    def test_registered_api_continues_environment_between_learn_calls(self):
        env = SixValueEnv()
        runner = OnPolicyRunner(env, nested_config(training_config()), None)
        resets = env.reset_count
        runner.learn(1)
        runner.learn(1)
        self.assertEqual(env.reset_count, resets)
        self.assertEqual(
            (runner.current_learning_iteration, runner.tot_timesteps), (2, 32)
        )

    def test_split_devices_are_rejected_before_policy_creation(self):
        with self.assertRaisesRegex(ValueError, "same device"):
            OnPolicyRunner(
                SixValueEnv(), nested_config(training_config()), None, device="cuda:0"
            )

    def test_generic_task_trains_with_empty_infos_and_a_different_observation_layout(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runner = PPORunner(MinimalEnv(), directory, config=training_config())
            runner.learn(1)
            self.assertEqual((runner.iteration, runner.total_steps), (1, 16))
            report = json.loads((Path(directory) / "progress.json").read_text())
            self.assertTrue(
                all(
                    not key.startswith(("Loop/", "Motion/", "Support/", "Terrain/"))
                    for key in report
                )
            )
            self.assertIn("Loss/value", report)
            checkpoint = torch.load(Path(directory) / "model_1.pt", map_location="cpu")
            self.assertNotIn("loop_model", checkpoint)
            self.assertNotIn("terrain_curriculum_state", checkpoint)
            self.assertEqual(
                checkpoint["model_state_dict"]["actor.0.weight"].shape[1], 3
            )

    def test_custom_collector_needs_only_rollout_hooks(self):
        class ActionMetrics:
            def reset(self):
                self.steps = 0

            def update(self, observations, rewards, dones, infos):
                self.steps += 1

            def compute(self, num_steps):
                return {"Custom/steps": self.steps}

        with tempfile.TemporaryDirectory() as directory:
            collector = ActionMetrics()
            runner = PPORunner(
                MinimalEnv(),
                directory,
                config=training_config(),
                metrics_collector=collector,
            )
            runner.learn(1)
            report = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(report["Custom/steps"], 4)

    def test_robot_collector_keeps_terminal_velocity_torque_and_support_metrics(self):
        env = DiagnosticEnv()
        observations, _, rewards, _, infos = env.step(torch.full((4, 6), 9.0))
        observations.zero_()
        observations[:, -33] = 1.5
        terminal = observations[1:2].clone()
        terminal[:, -33] = 3.0
        infos.update(
            terminal_env_ids=torch.tensor([1]),
            terminal_observation=terminal,
            pre_reset_pitch=torch.tensor([1.0, 2.0, 0.0, 0.0]),
            pre_reset_roll=torch.full((4,), 0.5),
            pre_reset_ang_vel=torch.tensor([[1.0, 2.0, 3.0]]).repeat(4, 1),
            pre_reset_leg_length=torch.tensor([[0.08, 0.09]]).repeat(4, 1),
            pre_reset_passive_limit_margin=torch.full((4, 4), 0.005),
            loop_failures=1,
            loop_error_max=0.002,
            reward_terms={"dense": torch.full((4,), 2.0)},
        )
        infos["pre_reset_commands"][:, 2] = 1.0
        dones = torch.tensor([False, True, False, False])
        collector = WheelLegTrainingMetrics(env)
        collector.reset()
        collector.update(observations, rewards, dones, infos)
        metrics = collector.compute(1)
        self.assertAlmostEqual(metrics["Motion/rms_pitch_rad"], 1.25**0.5)
        self.assertAlmostEqual(metrics["Motion/rms_roll_rad"], 0.5)
        self.assertAlmostEqual(metrics["Motion/rms_torque_delta_nm"], 9.0)
        self.assertEqual(metrics["Motion/torque_saturation_fraction"], 1.0)
        self.assertAlmostEqual(metrics["Motion/rms_vx_tracking_error_m_s"], 1.75**0.5)
        self.assertAlmostEqual(metrics["Motion/rms_yaw_tracking_error_rad_s"], 2.0)
        self.assertAlmostEqual(metrics["Support/mean_leg_length_m"], 0.085)
        self.assertEqual(metrics["Support/both_legs_within_5mm_fraction"], 0.0)
        self.assertEqual(metrics["Support/passive_near_limit_fraction"], 1.0)
        self.assertAlmostEqual(metrics["Support/rms_leg_length_asymmetry_m"], 0.01)
        self.assertEqual(metrics["Reward/dense"], 2.0)
        self.assertEqual(metrics["Episode/physical_failures"], 1)
        self.assertEqual(metrics["Loop/failures"], 1)
        self.assertEqual(metrics["Loop/max_anchor_error_m"], 0.002)


if __name__ == "__main__":
    unittest.main()
