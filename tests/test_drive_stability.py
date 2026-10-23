"""Drive screening retains failures, motion distance and signed heading errors."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_drive_stability import make_drive_sequence, summarize_drive, wrapped_increment


def row(**updates):
    value = dict(dt_s=.02, physical_failure=False, timeout=False, delta_body_forward_m=.01,
                 delta_xy_world_m=[.01, 0.], raw_vx_m_per_s=.6, applied_vx_m_per_s=.5,
                 base_height_m=.14, closure_error_m=.001, yaw_error_rad=.1,
                 physical_yaw_rad=.4, requested_yaw_rad=.3, yaw_rate_rad_per_s=.2,
                 pitch_rad=.02, roll_rad=-.01, vx_body_m_per_s=.4,
                 vx_error_raw_m_per_s=-.2, vx_error_applied_m_per_s=-.1,
                 yaw_rate_error_raw_rad_per_s=.2, yaw_rate_error_applied_rad_per_s=.2,
                 termination_reasons={"pitch": False}, torques_nm=[1.]*6, torque_peaks_nm=[2.]*6)
    value.update(updates)
    return value


class DriveStabilityTests(unittest.TestCase):
    def test_sequence_is_continuous_28_seconds_and_requests_return_to_original_heading(self):
        phases = make_drive_sequence()
        self.assertEqual(phases[0]['start_step'], 0)
        self.assertEqual(phases[-1]['end_step_exclusive'], 1400)
        self.assertTrue(all(a['end_step_exclusive'] == b['start_step'] for a, b in zip(phases, phases[1:])))
        self.assertAlmostEqual(sum(p['raw_yaw_rate_rad_per_s']*p['seconds'] for p in phases), 0.)
        self.assertIn('backward', [p['name'] for p in phases])

    def test_terminal_failure_cannot_be_hidden_by_good_final_samples(self):
        result = summarize_drive([row(physical_failure=True, termination_reasons={"pitch": True},
                                      pitch_rad=.4, delta_body_forward_m=.03), row()])
        self.assertEqual(result['physical_failures'], 1)
        self.assertEqual(result['termination_reason_counts']['pitch'], 1)
        self.assertEqual(result['pitch_rad_abs_peak'], .4)
        self.assertAlmostEqual(result['signed_body_forward_distance_m'], .04)

    def test_yaw_error_does_not_cancel_in_rms_and_torque_peak_is_not_endpoint(self):
        result = summarize_drive([row(yaw_error_rad=.2), row(yaw_error_rad=-.2)])
        self.assertEqual(result['yaw_error_rad_mean'], 0.)
        self.assertAlmostEqual(result['yaw_error_rad_rms'], .2)
        self.assertEqual(result['active_torque_substep_peak_nm_per_joint'], [2.]*6)
        self.assertEqual(result['active_torque_endpoint_rms_nm_per_joint'], [1.]*6)

    def test_unwrap_does_not_treat_pi_boundary_as_full_rotation(self):
        self.assertAlmostEqual(wrapped_increment(-math.pi+.01, math.pi-.01), .02)


if __name__ == '__main__':
    unittest.main()
