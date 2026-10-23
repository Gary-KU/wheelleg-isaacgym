"""Evaluation groups must track their own speed targets rather than a global zero."""

import math
from pathlib import Path
import sys
import unittest
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import (make_speed_scenarios, velocity_statistics, tracking_statistics,
                      make_speed_sequence, sequence_segment_at_step)


class EvaluationSpeedTests(unittest.TestCase):
    def test_group_commands_keep_lateral_yaw_and_leg_targets(self):
        commands, groups = make_speed_scenarios(6, [-0.3, 0, 0.3], [9, 0.1, -0.2, 0.15])
        self.assertEqual([row[0] for row in commands], [-0.3, -0.3, 0, 0, 0.3, 0.3])
        self.assertEqual([group["environment_indices"] for group in groups], [[0, 1], [2, 3], [4, 5]])
        self.assertTrue(all(row[1:] == [0.1, -0.2, 0.15] for row in commands))

    def test_legacy_fixed_command_is_unchanged(self):
        original = [0.2, 0, 0, 0.12]
        commands, groups = make_speed_scenarios(2, None, original)
        self.assertEqual(commands, [original, original])
        self.assertEqual(groups, [])
        commands[0][0] = 99
        self.assertEqual(commands[1][0], 0.2)
        self.assertEqual(original[0], 0.2)

    def test_invalid_group_counts_and_nonfinite_speeds_are_rejected(self):
        for num_envs, speeds in ((8, [-0.3, 0, 0.3]), (2, []), (2, [float("nan")]),
                                 (2, [float("inf")]), (0, [0])):
            with self.assertRaises(ValueError):
                make_speed_scenarios(num_envs, speeds, [0, 0, 0, 0.15])

    def test_perfect_opposite_driving_has_zero_rmse_despite_zero_global_mean(self):
        result = velocity_statistics([[-0.3, 0, 0.3], [-0.3, 0, 0.3]], [-0.3, 0, 0.3])
        self.assertEqual(result["samples"], 6)
        self.assertEqual(result["mean_vx_m_per_s"], 0.0)
        self.assertEqual(result["vx_tracking_rmse_m_per_s"], 0.0)

    def test_stationary_robot_is_not_perfect_tracking_for_nonzero_targets(self):
        result = velocity_statistics([[0, 0, 0], [0, 0, 0]], [-0.3, 0, 0.3])
        self.assertAlmostEqual(result["vx_tracking_rmse_m_per_s"], math.sqrt(0.06))
        backward = velocity_statistics([[-0.2], [-0.4]], [-0.3])
        self.assertAlmostEqual(backward["mean_vx_m_per_s"], -0.3)
        self.assertAlmostEqual(backward["vx_tracking_rmse_m_per_s"], 0.1)

    def test_missing_velocity_data_is_not_reported_as_zero_error(self):
        result = velocity_statistics([], [0])
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["vx_tracking_rmse_m_per_s"])
        with self.assertRaises(ValueError):
            velocity_statistics([[0, 0]], [0])


class EvaluationSequenceTests(unittest.TestCase):
    def test_cli_rejects_combined_modes_and_truncated_sequence_before_runtime_import(self):
        evaluator = Path(__file__).resolve().parents[1] / "evaluate.py"
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "dummy.pt"
            checkpoint.touch()
            common = [sys.executable, str(evaluator), "--checkpoint", str(checkpoint),
                      "--allow-approximate-loop"]
            for options, expected in ((
                    ["--test-speeds", "0", "--speed-sequence", "0.3", "0"], "not allowed"), (
                    ["--speed-sequence", "0.3", "0", "-0.3", "0", "--segment-steps", "2",
                     "--num-steps", "7"], "must equal sequence length")):
                process = subprocess.run(common+options, capture_output=True, text=True)
                self.assertEqual(process.returncode, 2)
                self.assertIn(expected, process.stderr)
                self.assertNotIn("Importing module", process.stdout)

    def test_forward_stop_backward_stop_exact_control_step_boundaries(self):
        segments = make_speed_sequence([0.3, 0, -0.3, 0], 2)
        self.assertEqual([(x["start_step"], x["end_step_exclusive"]) for x in segments],
                         [(0, 2), (2, 4), (4, 6), (6, 8)])
        self.assertEqual([sequence_segment_at_step(segments, step)["target_vx_m_per_s"]
                          for step in range(8)], [0.3, 0.3, 0, 0, -0.3, -0.3, 0, 0])
        with self.assertRaises(ValueError):
            sequence_segment_at_step(segments, -1)
        with self.assertRaises(ValueError):
            sequence_segment_at_step(segments, 8)

    def test_sequence_rejects_invalid_duration_or_speeds(self):
        for speeds, steps in (([0.3, 0], 0), ([0], -1), ([0], 0.5),
                              ([], 200), ([float("nan")], 200)):
            with self.assertRaises(ValueError):
                make_speed_sequence(speeds, steps)
        self.assertEqual(make_speed_sequence(None, 200), [])

    def test_sequence_rmse_follows_time_varying_targets(self):
        target_rows = [[0.3, 0.3], [0, 0], [-0.3, -0.3], [0, 0]]
        result = tracking_statistics(target_rows, target_rows)
        self.assertEqual(result["samples"], 8)
        self.assertEqual(result["mean_vx_m_per_s"], 0)
        self.assertEqual(result["vx_tracking_rmse_m_per_s"], 0)
        # Rolling forward through stop/backward commands must not appear to track.
        wrong = tracking_statistics([[0.3, 0.3]]*4, target_rows)
        self.assertAlmostEqual(wrong["vx_tracking_rmse_m_per_s"], math.sqrt(0.135))

    def test_segment_statistics_do_not_mix_transition_targets(self):
        observations = [[0.3], [0.3], [0.2], [0.0], [-0.1], [-0.3]]
        targets = [[0.3], [0.3], [0], [0], [-0.3], [-0.3]]
        self.assertEqual(tracking_statistics(observations[:2], targets[:2])["vx_tracking_rmse_m_per_s"], 0)
        stop = tracking_statistics(observations[2:4], targets[2:4])
        backward = tracking_statistics(observations[4:6], targets[4:6])
        self.assertAlmostEqual(stop["mean_vx_m_per_s"], 0.1)
        self.assertAlmostEqual(stop["vx_tracking_rmse_m_per_s"], math.sqrt(0.02))
        self.assertAlmostEqual(backward["mean_vx_m_per_s"], -0.2)
        self.assertAlmostEqual(backward["vx_tracking_rmse_m_per_s"], math.sqrt(0.02))
        with self.assertRaises(ValueError):
            tracking_statistics(observations, targets[:-1])


if __name__ == "__main__":
    unittest.main()
