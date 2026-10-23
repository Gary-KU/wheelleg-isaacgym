"""CPU checks for an explicit new-objective support bootstrap."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import isaacgym  # noqa: F401; extension must precede Torch.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_support_checkpoint import prepare_support_checkpoint, save_support_checkpoint
from wheelleg_gym.runtime_config import SUPPORT_ENVIRONMENT
from wheelleg_gym.training import make_policy


class SupportCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.config = {"policy": {"actor_hidden_dims": [16, 8], "critic_hidden_dims": [16, 8],
                                  "init_noise_std": [1.0]*4+[0.1]*2},
                       "algorithm": {"learning_rate": 0.001}, "save_interval": 100}
        with contextlib.redirect_stdout(io.StringIO()):
            self.source_policy = make_policy(198, 6, "cpu", self.config["policy"])
        self.checkpoint = {"model_state_dict": self.source_policy.state_dict(),
                           "optimizer_state_dict": {"state": {0: {"step": 10}}},
                           "training_config": self.config,
                           "environment_config": {"physics_dt": 0.0005, "random_commands": True},
                           "iter": 994, "total_steps": 1000000}

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_support_checkpoint(self.checkpoint, source_path="source_994.pt", **kwargs)

    def test_hidden_actor_and_wheel_predictions_stay_exact_but_hips_are_zero(self):
        prepared = self.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            restored = make_policy(198, 6, "cpu", prepared["training_config"]["policy"])
        restored.load_state_dict(prepared["model_state_dict"], strict=True)
        observations = torch.randn(37, 198)
        old, new = self.source_policy.act_inference(observations), restored.act_inference(observations)
        torch.testing.assert_close(new[:, 4:], old[:, 4:], rtol=0, atol=0)
        torch.testing.assert_close(new[:, :4], torch.zeros_like(new[:, :4]), rtol=0, atol=0)
        for name, tensor in self.source_policy.actor.state_dict().items():
            if name not in ("4.weight", "4.bias"):
                torch.testing.assert_close(restored.actor.state_dict()[name], tensor, rtol=0, atol=0)

    def test_new_critic_noise_and_training_objective_have_fresh_state(self):
        prepared = self.prepare()
        self.assertTrue(any(not torch.equal(prepared["model_state_dict"][key], value)
                            for key, value in self.checkpoint["model_state_dict"].items()
                            if key.startswith("critic.")))
        torch.testing.assert_close(prepared["model_state_dict"]["std"], torch.tensor([3.0]*4+[0.1]*2))
        self.assertNotIn("optimizer_state_dict", prepared)
        self.assertEqual((prepared["iter"], prepared["total_steps"]), (0, 0))
        config = prepared["training_config"]
        self.assertEqual(config["algorithm"]["learning_rate"], 0.0003)
        self.assertEqual(config["algorithm"]["max_learning_rate"], 0.0003)
        self.assertEqual(config["save_interval"], 10)
        for key, value in SUPPORT_ENVIRONMENT.items():
            self.assertEqual(prepared["environment_config"][key], value)
        self.assertEqual(prepared["environment_config"]["physics_dt"], 0.0005)
        self.assertEqual(prepared["support_bootstrap"]["source_iteration"], 994)
        self.assertEqual(prepared["support_bootstrap"]["source_checkpoint"], str(Path("source_994.pt").resolve()))
        self.assertIn("No feedforward", prepared["support_bootstrap"]["description"])
        self.assertEqual(config["initialization_provenance"], prepared["support_bootstrap"])

    def test_input_tensors_and_nested_configuration_are_not_mutated(self):
        snapshot = deepcopy(self.checkpoint)
        prepared = self.prepare()
        for key, value in snapshot["model_state_dict"].items():
            torch.testing.assert_close(self.checkpoint["model_state_dict"][key], value, rtol=0, atol=0)
        for key in ("optimizer_state_dict", "training_config", "environment_config", "iter", "total_steps"):
            self.assertEqual(self.checkpoint[key], snapshot[key])
        prepared["model_state_dict"]["actor.0.weight"].zero_()
        self.assertTrue(torch.equal(self.checkpoint["model_state_dict"]["actor.0.weight"], snapshot["model_state_dict"]["actor.0.weight"]))

    def test_seed_is_reproducible_and_custom_hip_std_is_used(self):
        a, b = self.prepare(seed=7, hip_std=2.0), self.prepare(seed=7, hip_std=2.0)
        for key in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
        torch.testing.assert_close(a["model_state_dict"]["std"], torch.tensor([2.0]*4+[0.1]*2))

    def test_wrong_observation_width_and_invalid_parameters_are_rejected(self):
        for key in ("actor.0.weight", "critic.0.weight"):
            bad = deepcopy(self.checkpoint)
            bad["model_state_dict"][key] = torch.zeros(16, 186)
            with self.assertRaisesRegex(ValueError, "198-D"):
                prepare_support_checkpoint(bad)
        for kwargs in ({"seed": -1}, {"seed": 1.5}, {"hip_std": 0},
                       {"hip_std": float("nan")}, {"hip_std": float("inf")},
                       {"hip_std": 1e308}, {"hip_std": 1e-50}):
            with self.assertRaises(ValueError):
                self.prepare(**kwargs)

    def test_safe_save_preserves_source_and_refuses_existing_output_and_cli_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"old.pt", Path(folder)/"new.pt"
            torch.save(self.checkpoint, str(source))
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                save_support_checkpoint(source, output)
            self.assertEqual(source.read_bytes(), original)
            saved = torch.load(str(output), map_location="cpu")
            self.assertEqual(saved["iter"], 0)
            self.assertEqual(saved["support_bootstrap"]["source_checkpoint"], str(source))
            old_output = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_support_checkpoint(source, output)
            with self.assertRaisesRegex(ValueError, "different files"):
                save_support_checkpoint(source, source)
            cli = subprocess.run([sys.executable, str(ROOT/"tools/prepare_support_checkpoint.py"),
                                  "--checkpoint", str(source), "--output", str(output)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(cli.returncode, 0)
            self.assertIn("Output already exists", cli.stderr)
            self.assertEqual(output.read_bytes(), old_output)
            dangling = Path(folder)/"dangling.pt"
            dangling.symlink_to(Path(folder)/"absent.pt")
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_support_checkpoint(source, dangling)
            self.assertFalse((Path(folder)/"absent.pt").exists())

    def test_module_import_does_not_load_gym_or_torch(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; import tools.prepare_support_checkpoint; "
                                 "assert 'torch' not in sys.modules; assert 'isaacgym' not in sys.modules"],
                                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
