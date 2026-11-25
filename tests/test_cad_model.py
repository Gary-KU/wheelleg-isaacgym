"""The replacement asset and checkpoint provenance must agree before a rollout."""

import unittest
from legged_gym.envs.wheelleg.model.geometry import (
    asset_fingerprint,
    require_model_checkpoint,
)
from tools.check_model import check_model


class CadModelTests(unittest.TestCase):
    def test_referenced_meshes_inertias_joint_limits_and_closed_poses(self):
        report = check_model()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["referenced_meshes"], 26)
        self.assertEqual(report["command_range_m"], [0.08, 0.13])
        self.assertAlmostEqual(report["wheel_radius_m"], 0.0325)

    def test_old_missing_and_different_model_fingerprints_are_rejected(self):
        for state in (
            {},
            {"asset_fingerprint": None},
            {"asset_fingerprint": "old-robot"},
        ):
            with self.assertRaisesRegex(
                ValueError, "different/unidentified robot model"
            ):
                require_model_checkpoint(state)
        require_model_checkpoint({"asset_fingerprint": asset_fingerprint()})
