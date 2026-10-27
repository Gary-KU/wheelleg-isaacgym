"""New tasks must not require wheelleg state, CLI rules or eager robot imports."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry


class ExtensionBoundaryTests(unittest.TestCase):
    def test_custom_task_prepares_without_wheelleg_config_sections(self):
        class CustomTask:
            def __init__(self, cfg, sim_device, headless):
                self.cfg = cfg
                self.sim_device = sim_device

        registry = TaskRegistry()
        registry.register(
            "custom",
            CustomTask,
            SimpleNamespace(
                seed=0, env=SimpleNamespace(num_envs=1, num_observations=5)
            ),
            SimpleNamespace(seed=7, runner=SimpleNamespace(experiment_name="custom")),
        )
        args = get_args(
            "train",
            [
                "--task",
                "custom",
                "--sim_device",
                "cpu",
                "--headless",
                "--num_envs",
                "3",
            ],
        )
        cfg, train_cfg = registry.prepare_cfgs("custom", args)
        env, _ = registry.make_env("custom", args=args, env_cfg=cfg)
        self.assertEqual(env.cfg.env.num_envs, 3)
        self.assertEqual(env.cfg.seed, train_cfg.seed)
        self.assertEqual(env.sim_device, "cpu")

    def test_generic_imports_do_not_load_robot_or_isaac_gym(self):
        code = (
            "import sys; import legged_gym.learning.ppo_runner; "
            "import legged_gym.utils.task_registry; "
            "assert 'isaacgym' not in sys.modules; "
            "assert not any(k.startswith('legged_gym.envs.wheelleg') for k in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_numeric_training_checkpoint_is_not_silently_ignored(self):
        import torch

        registry = TaskRegistry()
        registry.register(
            "custom",
            object,
            SimpleNamespace(
                seed=0, env=SimpleNamespace(num_envs=1, num_observations=5)
            ),
            SimpleNamespace(seed=7, runner=SimpleNamespace(experiment_name="custom")),
        )
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            path = run / "model_100.pt"
            torch.save(
                {"model_state_dict": {"actor.0.weight": torch.zeros(2, 5)}}, path
            )
            args = get_args(
                "train",
                [
                    "--task",
                    "custom",
                    "--checkpoint",
                    "100",
                    "--log_root",
                    tmp,
                    "--headless",
                ],
            )
            registry.prepare_cfgs("custom", args)
            self.assertEqual(args.checkpoint_path, path.resolve())


if __name__ == "__main__":
    unittest.main()
