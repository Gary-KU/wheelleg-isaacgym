"""A resumed curriculum obeys requested bounds, including legacy saved levels."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import isaacgym  # noqa: F401; must precede Torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.training import PPORunner


class StairCurriculumCheckpointTests(unittest.TestCase):
    def test_load_clamps_repeated_saved_levels_to_new_lower_and_upper_bounds(self):
        runner = object.__new__(PPORunner)
        runner.device = "cpu"
        runner.policy = torch.nn.Linear(1, 1)
        runner.algorithm = SimpleNamespace(optimizer=torch.optim.Adam(runner.policy.parameters()))
        runner.env = SimpleNamespace(num_envs=4, stair_curriculum=True, stair_min_level=1,
            stair_max_level=3, terrain_levels=torch.zeros(4, dtype=torch.long), terrain_model=object())
        runner.config = {}
        runner.environment_config = {"terrain": "stairs", "stair_min_level": 1, "stair_level": 1, "stair_max_level": 3}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"source.pt", Path(directory)/"output.pt"
            torch.save({"model_state_dict": runner.policy.state_dict(), "iter": 7,
                "terrain_curriculum_state": {"levels": [0, 4]}}, source)
            runner.load(source)
            self.assertEqual(runner.env.terrain_levels.tolist(), [1, 3, 1, 3])
            runner.save(output)
            saved = torch.load(output)
            self.assertEqual(saved["environment_config"]["stair_min_level"], 1)
            self.assertEqual(saved["terrain_curriculum_state"], {"levels": [1, 3, 1, 3], "min_level": 1, "max_level": 3})


if __name__ == "__main__":
    unittest.main()
