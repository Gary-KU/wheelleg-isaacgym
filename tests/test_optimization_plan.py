"""Finite suites and quality reporting must not silently claim stability."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.optimization_plan import compare_reports, evaluation_cases, summarize_report


def report():
    return {"requested_steps": 100, "completed_steps": 100, "finite_outputs": True,
            "physical_failure_resets": 0, "time_limit_resets": 0,
            "motion_diagnostics": {"overall": {"roll_rms_rad": .02, "pitch_rms_rad": .03,
                "xy_angular_velocity_rms_rad_per_s": .1,
                "active_torque_rate_rms_nm_per_s": 20., "active_torque_saturation_fraction": .01}},
            "command_tracking": {"vx_rmse_against_applied_target_m_per_s": .03,
                "vx_rmse_against_raw_target_m_per_s": .08,
                "yaw_rmse_against_applied_target_rad_per_s": .1,
                "yaw_rmse_against_raw_target_rad_per_s": .2},
            "ground_support_diagnostics": {"last_100_control_steps": {
                "nonwheel_contact_fraction_any_over_1n": 0.0,
                "geometric_leg_length_mean_m_left_right": [.08, .08]}}}


class OptimizationPlanTests(unittest.TestCase):
    def test_default_suite_is_longer_than_old_short_test(self):
        cases = evaluation_cases()
        self.assertEqual([case["expected_steps"] for case in cases], [3000, 800, 600])
        dynamic = cases[1]["arguments"]
        self.assertIn("--yaw-sequence", dynamic)
        self.assertIn("-0.8", dynamic)

    def test_small_smoke_suite_is_explicit_and_bounded(self):
        self.assertEqual([x["expected_steps"] for x in evaluation_cases(.2, 2, 1, 1)], [10, 16, 6])
        for args in ((.001, 2, 1, 1), (1, 0, 1, 1), (1, 2, 0, 1), (1, 2, 1, 0)):
            with self.assertRaises(ValueError):
                evaluation_cases(*args)

    def test_comparison_keeps_raw_tracking_and_measured_changes(self):
        baseline = report(); candidate = deepcopy(baseline)
        candidate["motion_diagnostics"]["overall"]["pitch_rms_rad"] = .01
        compared = compare_reports({"standing": baseline}, {"standing": candidate})
        case = compared["cases"]["standing"]
        self.assertAlmostEqual(case["candidate_minus_baseline"]["pitch_rms_rad"], -.02)
        self.assertEqual(case["candidate"]["vx_rmse_against_raw_target_m_per_s"], .08)
        self.assertTrue(compared["scripted_checks_passed"])
        self.assertIn("requires_manual_validation", compared["quality_status"])

    def test_successful_process_with_failure_does_not_pass_quality(self):
        baseline = report(); candidate = report(); candidate["physical_failure_resets"] = 1
        compared = compare_reports({"standing": baseline}, {"standing": candidate})
        self.assertFalse(compared["scripted_checks_passed"])
        self.assertEqual(compared["quality_status"], "needs_improvement")

    def test_missing_motion_or_incomplete_evaluation_fails_quality(self):
        candidate = report(); candidate["motion_diagnostics"] = {}; candidate["completed_steps"] = 99
        compared = compare_reports({"standing": report()}, {"standing": candidate})
        self.assertFalse(compared["scripted_checks_passed"])
        self.assertIsNone(summarize_report(candidate)["pitch_rms_rad"])


if __name__ == "__main__":
    unittest.main()
