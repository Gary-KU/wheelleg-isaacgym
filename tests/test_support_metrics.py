"""Support screening must reject joint-stop crouching and hidden early failures."""

from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelleg_gym.support_metrics import (
    assess_support, duration_steps, summarize_support, velocity_kick_schedule,
)


def sample(**updates):
    row = {"step": 1, "leg_lengths_m": [0.08, 0.08], "target_leg_length_m": 0.08,
           "passive_limit_margin_rad": [0.25]*4, "hip_torques_nm": [1.0, -1.0, -1.0, 1.0],
           "dof_positions_rad": [0.6, -0.95, -0.6, 0.95], "pitch_rad": 0.02, "roll_rad": -0.01,
           "wheel_contacts_over_1n": [True, True], "nonwheel_contact_over_1n": False,
           "physical_failure": False, "timeout": False}
    row.update(updates)
    return row


def screen(summary, **updates):
    arguments = dict(requested_steps=100, completed_steps=100, num_envs=1, sample_count=100,
                     finite_outputs=True, physical_failures=0, timeouts=0)
    arguments.update(updates)
    return assess_support(summary, **arguments)


class SupportStatisticsTests(unittest.TestCase):
    def test_joint_stop_crouch_fails_even_when_upright_and_torque_has_headroom(self):
        result = summarize_support([sample(leg_lengths_m=[0.05, 0.05], passive_limit_margin_rad=[0.0]*4)])
        self.assertAlmostEqual(result["leg_length_error_rmse_m_left_right"][0], 0.03)
        self.assertEqual(result["hip_torque_saturation_fraction"], 0)
        checks = screen(result)["checks"]
        self.assertFalse(checks["both_legs_within_5mm"])
        self.assertFalse(checks["no_passive_limit_dwelling"])

    def test_opposite_leg_errors_do_not_cancel_into_a_good_average(self):
        result = summarize_support([sample(leg_lengths_m=[0.09, 0.07])])
        self.assertAlmostEqual(sum(result["leg_length_mean_m_left_right"])/2, 0.08)
        self.assertEqual(result["both_legs_within_tolerance_fraction"], 0)
        self.assertAlmostEqual(result["left_right_length_difference_rmse_m"], 0.02)

    def test_weighted_raw_samples_keep_rmse_peaks_and_saturation_distinct(self):
        rows = [sample(leg_lengths_m=[0.09, 0.08], hip_torques_nm=[9.0, -1., -1., 1.], pitch_rad=0.2),
                sample(leg_lengths_m=[0.08, 0.08], hip_torques_nm=[3.0, -1., -1., 1.], pitch_rad=0.0)]
        result = summarize_support(rows)
        self.assertAlmostEqual(result["leg_length_error_rmse_m_left_right"][0], 0.01/math.sqrt(2))
        self.assertAlmostEqual(result["hip_torque_rms_nm_per_joint"][0], math.sqrt(45))
        self.assertEqual(result["hip_torque_abs_peak_nm_per_joint"][0], 9.0)
        self.assertEqual(result["hip_torque_saturation_fraction_per_joint"], [0.5, 0, 0, 0])
        self.assertAlmostEqual(result["pitch_rms_rad"], 0.2/math.sqrt(2))
        self.assertEqual(result["pitch_abs_peak_rad"], 0.2)

    def test_substep_peak_and_saturation_are_not_hidden_by_low_endpoint(self):
        rows = [sample(hip_torques_nm=[1., -1., -1., 1.],
                       hip_torque_peak_nm_per_joint=[9., 2., 1., 1.],
                       hip_torque_substep_saturation_fraction=[.25, 0., 0., 0.]),
                sample(hip_torques_nm=[3., -1., -1., 1.],
                       hip_torque_peak_nm_per_joint=[4., 1., 1., 1.],
                       hip_torque_substep_saturation_fraction=[0., 0., 0., 0.])]
        result = summarize_support(rows)
        self.assertAlmostEqual(result["hip_torque_rms_nm_per_joint"][0], math.sqrt(5))
        self.assertEqual(result["hip_torque_abs_peak_nm_per_joint"], [9, 2, 1, 1])
        self.assertEqual(result["hip_torque_saturation_fraction_per_joint"], [.125, 0, 0, 0])
        self.assertEqual(result["hip_torque_saturation_fraction"], .03125)
        self.assertEqual(result["torque_sampling"]["peak_substep_samples"], 2)
        self.assertEqual(result["torque_sampling"]["saturation_endpoint_fallback_samples"], 0)
        self.assertIn("endpoint", result["torque_sampling"]["rms"])

    def test_mixed_new_and_legacy_samples_keep_endpoint_fallback_compatible(self):
        result = summarize_support([
            sample(hip_torque_peak_nm_per_joint=[2., 1., 1., 1.],
                   hip_torque_substep_saturation_fraction=[.5, 0., 0., 0.]),
            sample(hip_torques_nm=[9., -1., -1., 1.]),
        ])
        self.assertEqual(result["hip_torque_abs_peak_nm_per_joint"][0], 9)
        self.assertEqual(result["hip_torque_saturation_fraction_per_joint"][0], .75)
        self.assertEqual(result["torque_sampling"]["peak_endpoint_fallback_samples"], 1)
        self.assertEqual(result["torque_sampling"]["saturation_endpoint_fallback_samples"], 1)

    def test_early_failure_cannot_be_erased_by_settled_good_samples(self):
        settled = summarize_support([sample()])
        result = screen(settled, physical_failures=1)
        self.assertFalse(result["engineering_screen_passed"])
        self.assertFalse(result["checks"]["no_physical_failures_entire_run"])
        self.assertTrue(result["checks"]["both_legs_within_5mm"])

    def test_contact_nonfinite_truncation_and_missing_samples_fail_independently(self):
        summary = summarize_support([sample(nonwheel_contact_over_1n=True, wheel_contacts_over_1n=[True, False])])
        self.assertEqual(summary["both_wheels_contact_fraction_over_1n"], 0)
        self.assertEqual(summary["wheel_contact_fraction_over_1n_left_right"], [1, 0])
        self.assertFalse(screen(summary)["checks"]["no_nonwheel_contact"])
        self.assertFalse(screen(summary, finite_outputs=False)["checks"]["finite_outputs"])
        self.assertFalse(screen(summary, completed_steps=99, sample_count=99)["checks"]["all_requested_steps"])
        self.assertFalse(screen(summary, sample_count=99)["checks"]["all_completed_samples_recorded"])
        self.assertFalse(screen(summary, timeouts=1)["checks"]["no_time_limit_resets_entire_run"])

    def test_threshold_boundary_and_empty_settled_window(self):
        summary = summarize_support([sample(leg_lengths_m=[0.085, 0.075])])
        self.assertEqual(summary["both_legs_within_tolerance_fraction"], 1)
        self.assertTrue(screen(summary)["engineering_screen_passed"])
        self.assertIsNone(summarize_support([]))
        self.assertFalse(screen(None)["engineering_screen_passed"])
        loose = summarize_support([sample()], leg_tolerance_m=0.03)
        self.assertFalse(screen(loose)["checks"]["standard_summary_tolerances"])

    def test_nonfinite_and_malformed_terminal_diagnostics_are_rejected(self):
        rows = [sample()]
        snapshot = deepcopy(rows)
        summarize_support(rows)
        self.assertEqual(rows, snapshot)
        for update in ({"leg_lengths_m": [float("nan"), 0.08]},
                       {"passive_limit_margin_rad": [0.2]*3},
                       {"dof_positions_rad": []},
                       {"wheel_contacts_over_1n": [1, 1]},
                       {"hip_torque_peak_nm_per_joint": [float("nan")]*4},
                       {"hip_torque_peak_nm_per_joint": [0.0]*4},
                       {"hip_torque_substep_saturation_fraction": [1.1]*4},
                       {"hip_torque_substep_saturation_fraction": [0.0]*3}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                summarize_support([sample(**update)])

    def test_durations_are_exact_control_periods(self):
        self.assertEqual(duration_steps(20.0, 0.02), 1000)
        self.assertEqual(duration_steps(0, 0.02, allow_zero=True), 0)
        for value in (-1, 0, float("nan"), 0.021):
            with self.subTest(value=value), self.assertRaises(ValueError):
                duration_steps(value, 0.02)

    def test_motion_uses_vector_speed_and_displacement_and_preserves_terminal_peak(self):
        result = summarize_support([
            sample(environment_index=0, base_xy_displacement_world_m=[0.03, 0.04],
                   base_linear_velocity_world_m_per_s=[0.03, 0.04, 0.2], base_yaw_drift_rad=0.03),
            sample(environment_index=0, base_xy_displacement_world_m=[0.0, 0.0],
                   base_linear_velocity_world_m_per_s=[0.0, 0.0, 0.0], base_yaw_drift_rad=-0.04),
        ])
        motion = result["base_motion"]
        self.assertAlmostEqual(motion["planar_speed_rms_m_per_s"], 0.05/math.sqrt(2))
        self.assertAlmostEqual(motion["xy_displacement_peak_m"], 0.05)
        self.assertEqual(motion["yaw_drift_abs_peak_rad"], 0.04)
        self.assertEqual(motion["end_per_environment"][0]["xy_displacement_world_m"], [0, 0])
        self.assertTrue(screen(result)["engineering_screen_passed"])

    def test_upright_supported_robot_fails_motion_screen_when_driving_or_turning(self):
        for update, name in (
            ({"base_xy_displacement_world_m": [0.06, 0.]}, "stationary_xy_displacement"),
            ({"base_linear_velocity_world_m_per_s": [0., 0.06, 0.]}, "stationary_planar_speed"),
            ({"base_yaw_drift_rad": 0.11}, "stationary_yaw_drift"),
        ):
            base = dict(base_xy_displacement_world_m=[0., 0.],
                        base_linear_velocity_world_m_per_s=[0., 0., 0.], base_yaw_drift_rad=0.)
            base.update(update)
            with self.subTest(name=name):
                result = screen(summarize_support([sample(**base)]))
                self.assertFalse(result["engineering_screen_passed"])
                self.assertFalse(result["checks"][name])
        legacy = summarize_support([sample()])
        self.assertTrue(screen(legacy)["engineering_screen_passed"])
        self.assertFalse(screen(legacy, require_motion=True)["engineering_screen_passed"])

    def test_partial_or_nonfinite_motion_fields_are_rejected(self):
        for rows in (
            [sample(base_yaw_drift_rad=0.)],
            [sample(base_xy_displacement_world_m=[0., 0.], base_linear_velocity_world_m_per_s=[0., 0., 0.],
                    base_yaw_drift_rad=0.), sample()],
            [sample(base_xy_displacement_world_m=[float("nan"), 0.],
                    base_linear_velocity_world_m_per_s=[0., 0., 0.], base_yaw_drift_rad=0.)],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_support(rows)

    def test_single_kick_has_explicit_world_frame_and_exact_execution_period(self):
        self.assertIsNone(velocity_kick_schedule(None, None, 0.02, 1000))
        schedule = velocity_kick_schedule(5., [0.1, -0.1, 0.], 0.02, 1000)
        self.assertEqual(schedule["before_step_index_zero_based"], 250)
        self.assertEqual(schedule["frame"], "world")
        self.assertEqual(schedule["delta_linear_velocity_m_per_s"], [0.1, -0.1, 0.])
        self.assertEqual(velocity_kick_schedule(0, [0.1, 0, 0], 0.02, 10)["before_step_index_zero_based"], 0)
        for at, delta in ((None, [0.1, 0, 0]), (5., None), (20., [0.1, 0, 0]),
                          (-1., [0.1, 0, 0]), (0.021, [0.1, 0, 0]),
                          (5., [0., 0., 0.]), (5., [0.1, float("nan"), 0.]), (5., [0.1, 0.])):
            with self.subTest(at=at, delta=delta), self.assertRaises(ValueError):
                velocity_kick_schedule(at, delta, 0.02, 1000)


if __name__ == "__main__":
    unittest.main()
