"""Stair warm starts preserve wheel initialization without hiding a new task."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import isaacgym  # noqa: F401; extension must precede Torch in discovery.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_stair_checkpoint import (
    expand_stair_input_columns, prepare_stair_checkpoint, save_stair_checkpoint,
)
from legged_gym.compat.runtime_config import STAIRS_ENVIRONMENT
from legged_gym.learning.policy import make_policy


class StairCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.config = {"policy": {"actor_hidden_dims": [16, 8], "critic_hidden_dims": [16, 8],
                                  "init_noise_std": [1.]*4+[.1]*2},
                       "algorithm": {"learning_rate": .001}, "save_interval": 100,
                       "initialization_provenance": {"source_iteration": 994}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.source_policy = make_policy(198, 6, "cpu", self.config["policy"])
        with torch.no_grad():
            self.source_policy.actor[-1].weight[:4].zero_()
            self.source_policy.actor[-1].bias[:4].zero_()
        self.checkpoint = {
            "model_state_dict": self.source_policy.state_dict(),
            "optimizer_state_dict": {"state": {0: {"step": 10}}},
            "training_config": self.config,
            "environment_config": {"physics_dt": .0005, "leg_control": "hip_pd", "heading_kp": 2., "heading_kd": .1,
                                   "wheel_control": "heading", "wheel_pitch_damping": 0.8,
                                   "wheel_balance_scale": 1.25},
            "iter": 0, "total_steps": 0,
            "legged_gym_env_config": {"terrain": {"terrain": "plane"}},
            "legged_gym_train_config": {"runner": {"experiment_name": "standing_source"}},
            "terrain_curriculum_state": {"levels": [3]},
        }

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_stair_checkpoint(self.checkpoint, source_path="standing_hold.pt", **kwargs)

    def test_every_old_input_column_stays_in_its_frame_and_terrain_starts_zero(self):
        old = torch.arange(2*198).reshape(2, 198)
        expanded = expand_stair_input_columns(old).reshape(2, 6, 42)
        torch.testing.assert_close(expanded[:, :, :33], old.reshape(2, 6, 33), rtol=0, atol=0)
        torch.testing.assert_close(expanded[:, :, 33:], torch.zeros(2, 6, 9, dtype=old.dtype), rtol=0, atol=0)

    def test_actor_hidden_and_wheel_predictions_preserved_with_nonzero_terrain(self):
        prepared = self.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            restored = make_policy(252, 6, "cpu", prepared["training_config"]["policy"])
        restored.load_state_dict(prepared["model_state_dict"], strict=True)
        old_observation = torch.randn(37, 198)
        new_observation = torch.randn(37, 6, 42)+5
        new_observation[:, :, :33] = old_observation.reshape(37, 6, 33)
        expected = self.source_policy.act_inference(old_observation)
        actual = restored.act_inference(new_observation.reshape(37, 252))
        # Wider BLAS matrices can change floating accumulation order slightly.
        torch.testing.assert_close(actual[:, 4:], expected[:, 4:], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual[:, :4], torch.zeros_like(actual[:, :4]), rtol=0, atol=0)
        for name, value in self.checkpoint["model_state_dict"].items():
            if name.startswith("actor.") and name != "actor.0.weight":
                torch.testing.assert_close(prepared["model_state_dict"][name], value, rtol=0, atol=0)

    def test_critic_noise_optimizer_counters_and_task_are_fresh(self):
        prepared = self.prepare()
        model = prepared["model_state_dict"]
        self.assertEqual(model["critic.0.weight"].shape, (16, 252))
        self.assertFalse(torch.equal(model["critic.0.bias"], self.checkpoint["model_state_dict"]["critic.0.bias"]))
        torch.testing.assert_close(model["std"], torch.tensor([.1]*4+[.02]*2), rtol=0, atol=0)
        self.assertNotIn("optimizer_state_dict", prepared)
        for key in ("legged_gym_env_config", "legged_gym_train_config", "terrain_curriculum_state"):
            self.assertNotIn(key, prepared)
            self.assertIn(key, self.checkpoint)
        self.assertEqual((prepared["iter"], prepared["total_steps"]), (0, 0))
        config = prepared["training_config"]
        self.assertEqual((config["algorithm"]["learning_rate"], config["algorithm"]["max_learning_rate"]), (.0003, .0003))
        self.assertEqual(config["save_interval"], 10)
        for key, value in STAIRS_ENVIRONMENT.items():
            self.assertEqual(prepared["environment_config"][key], value)
        env = prepared["environment_config"]
        self.assertEqual((env["terrain"], env["observe_terrain"], env["leg_control"]), ("stairs", True, "hip_target"))
        self.assertEqual(env["hip_target_scale"], .2)
        self.assertEqual(env["wheel_control"], "policy")
        self.assertEqual((env["wheel_pitch_damping"], env["wheel_balance_scale"]), (0.0, 1.0))
        self.assertEqual(env["command"], [.25, 0., 0., .14])
        self.assertEqual((env["stair_level"], env["stair_max_level"], env["stair_curriculum"]), (0, 4, True))
        self.assertEqual(env["physics_dt"], .0005)
        self.assertEqual(config["initialization_provenance"], prepared["stair_bootstrap"])
        self.assertEqual(prepared["stair_bootstrap"]["source_initialization_provenance"], {"source_iteration": 994})
        self.assertEqual(prepared["stair_bootstrap"]["new_rl_training_updates"], 0)
        self.assertIn("untrained", prepared["stair_bootstrap"]["description"])

    def test_source_nested_config_and_tensors_are_not_mutated_or_aliased(self):
        snapshot = deepcopy(self.checkpoint)
        prepared = self.prepare()
        for key, value in snapshot["model_state_dict"].items():
            torch.testing.assert_close(self.checkpoint["model_state_dict"][key], value, rtol=0, atol=0)
        for key in ("training_config", "environment_config", "optimizer_state_dict", "iter", "total_steps"):
            self.assertEqual(self.checkpoint[key], snapshot[key])
        prepared["model_state_dict"]["actor.0.weight"].zero_()
        prepared["environment_config"]["command"][3] = .20
        self.assertEqual(self.checkpoint["environment_config"], snapshot["environment_config"])
        torch.testing.assert_close(self.checkpoint["model_state_dict"]["actor.0.weight"], snapshot["model_state_dict"]["actor.0.weight"], rtol=0, atol=0)

    def test_seed_is_reproducible_and_wrong_input_width_or_invalid_seed_is_rejected(self):
        a, b = self.prepare(seed=7), self.prepare(seed=7)
        for key in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
        for name in ("actor.0.weight", "critic.0.weight"):
            bad = deepcopy(self.checkpoint)
            bad["model_state_dict"][name] = torch.zeros(16, 186)
            with self.assertRaisesRegex(ValueError, "198-D"):
                prepare_stair_checkpoint(bad)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.prepare(seed=seed)

    def test_atomic_save_preserves_source_and_refuses_existing_or_alias_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"source.pt", Path(folder)/"stairs.pt"
            torch.save(self.checkpoint, str(source))
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                save_stair_checkpoint(source, output)
            self.assertEqual(source.read_bytes(), original)
            saved = torch.load(str(output), map_location="cpu")
            self.assertEqual(saved["model_state_dict"]["actor.0.weight"].shape[1], 252)
            self.assertEqual(saved["stair_bootstrap"]["source_checkpoint"], str(source))
            old_output = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_stair_checkpoint(source, output)
            with self.assertRaisesRegex(ValueError, "different files"):
                save_stair_checkpoint(source, source)
            dangling = Path(folder)/"dangling.pt"
            dangling.symlink_to(Path(folder)/"absent.pt")
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_stair_checkpoint(source, dangling)
            result = subprocess.run([sys.executable, str(ROOT/"tools/prepare_stair_checkpoint.py"),
                                     "--checkpoint", str(source), "--output", str(output)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output already exists", result.stderr)
            self.assertEqual(output.read_bytes(), old_output)

    def test_import_does_not_load_torch_or_isaac_gym(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import tools.prepare_stair_checkpoint; "
                                 "assert 'torch' not in sys.modules; assert 'isaacgym' not in sys.modules"],
                                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
