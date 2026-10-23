"""A stair evaluator cannot label survival or a reset pose as course success."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_stairs import assess_stair_climbing, summarize_stairs


def sample(index=0, success=False, verified=False, failure=False, timeout=False):
    return {"environment_index": index, "course_success_reported": success, "course_success_verified": verified,
            "physical_failure": failure, "timeout": timeout, "progress_m": 4. if success else .1,
            "best_progress_m": 4. if success else .1, "ground_height_m": .04,
            "root_state": [0., 0., .2]+[0.]*10, "forward_velocity_body_m_per_s": .2,
            "target_vx_m_per_s": .25, "pitch_rad": .1, "roll_rad": .02,
            "leg_lengths_m": [.14, .14], "target_leg_length_m": .14,
            "passive_limit_margin_rad": [.1]*4, "applied_torques_nm": [1.]*6,
            "peak_abs_torques_nm": [2.]*6, "torque_saturation_fraction": [0.]*6,
            "closure_error_max_m": .002, "wheel_contacts_over_1n": [True, True],
            "nonwheel_contact_over_1n": False}


class StairEvaluationTests(unittest.TestCase):
    def test_survival_and_partial_progress_are_never_a_pass(self):
        rows = [sample(0), sample(1)]
        result = summarize_stairs(rows, 2)
        self.assertEqual(result["course_crossings"], 0)
        self.assertEqual(result["physical_failures"], 0)
        self.assertFalse(assess_stair_climbing(result, 2, 1, 1, True)["stair_climbing_screen_passed"])

    def test_each_environment_must_have_verified_crossing_and_failures_stay_visible(self):
        rows = [sample(0, True, True), sample(1, True, True)]
        result = summarize_stairs(rows, 2)
        self.assertTrue(assess_stair_climbing(result, 2, 1, 1, True)["stair_climbing_screen_passed"])
        rows.append(sample(1, failure=True))
        result = summarize_stairs(rows, 2)
        self.assertEqual(result["physical_failures"], 1)
        self.assertFalse(assess_stair_climbing(result, 2, 2, 2, True)["stair_climbing_screen_passed"])

    def test_reported_but_unverified_completion_does_not_count_as_crossing(self):
        rows = [sample(0, True, False)]
        result = summarize_stairs(rows, 1)
        self.assertEqual((result["course_crossings"], result["reported_successes"], result["unverified_successes"]), (0, 1, 1))
        self.assertEqual(result["completed_attempts"], 1)
        self.assertFalse(assess_stair_climbing(result, 1, 1, 1, True)["stair_climbing_screen_passed"])

    def test_timeouts_are_separate_from_failures_and_terminal_statistics_survive(self):
        rows = [sample(0, timeout=True), sample(0, failure=True), sample(0, True, True)]
        rows[1]["root_state"][2] = .03
        rows[1]["pitch_rad"] = .7
        rows[1]["termination_reasons"] = {"pitch": True, "lane": False}
        original = deepcopy(rows)
        result = summarize_stairs(rows, 1)
        self.assertEqual((result["course_crossings"], result["physical_failures"], result["time_limit_resets"]), (1, 1, 1))
        self.assertEqual(result["per_environment"][0]["base_height_min_m"], .03)
        self.assertEqual(result["pitch_abs_peak_rad"], .7)
        self.assertAlmostEqual(result["course_success_fraction_completed_attempts"], 1/3)
        self.assertEqual(result["termination_reason_counts"], {"lane": 0, "pitch": 1})
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
