"""有限优化实验的评估计划与前后比较；不导入 Gym 或 PyTorch。"""

import math


def evaluation_cases(standing_seconds=60.0, dynamic_segment_steps=100,
                     rapid_segment_steps=10, rapid_repeats=10, control_dt=0.02):
    if not math.isfinite(standing_seconds) or standing_seconds <= 0:
        raise ValueError("standing_seconds must be positive and finite")
    stand_steps = int(round(standing_seconds/control_dt))
    if stand_steps < 1 or not math.isclose(stand_steps*control_dt, standing_seconds, abs_tol=1e-8):
        raise ValueError("standing_seconds must be a positive multiple of the control period")
    if any(not isinstance(value, int) or value < 1
           for value in (dynamic_segment_steps, rapid_segment_steps, rapid_repeats)):
        raise ValueError("Segment lengths and repeat count must be positive integers")
    return [
        {"name": "standing", "expected_steps": stand_steps,
         "arguments": ["--num-steps", str(stand_steps)]},
        {"name": "driving_turning", "expected_steps": dynamic_segment_steps*8,
         "arguments": ["--speed-sequence", "0", "0.3", "0", "-0.3", "0", "0", "0", "0",
                       "--yaw-sequence", "0", "0", "0", "0", "0", "0.8", "-0.8", "0",
                       "--segment-steps", str(dynamic_segment_steps)]},
        {"name": "rapid_switching", "expected_steps": rapid_segment_steps*6*rapid_repeats,
         "arguments": ["--speed-sequence", "0.3", "-0.3", "0.3", "0", "-0.3", "0",
                       "--yaw-sequence", "0", "0.8", "-0.8", "0", "0.8", "0",
                       "--segment-steps", str(rapid_segment_steps),
                       "--sequence-repeats", str(rapid_repeats)]},
    ]


def summarize_report(report):
    motion = (report.get("motion_diagnostics") or {}).get("overall") or {}
    tracking = report.get("command_tracking") or {}
    support = ((report.get("ground_support_diagnostics") or {}).get("last_100_control_steps") or {})
    result = {key: report.get(key) for key in (
        "checkpoint_iteration", "requested_steps", "completed_steps", "simulated_seconds_per_env",
        "finite_outputs", "physical_failure_resets", "time_limit_resets", "peak_anchor_error_mm")}
    for key in ("roll_rms_rad", "pitch_rms_rad", "xy_angular_velocity_rms_rad_per_s",
                "active_torque_rate_rms_nm_per_s", "active_torque_saturation_fraction"):
        result[key] = motion.get(key)
    for key in ("vx_rmse_against_raw_target_m_per_s", "vx_rmse_against_applied_target_m_per_s",
                "yaw_rmse_against_raw_target_rad_per_s", "yaw_rmse_against_applied_target_rad_per_s"):
        result[key] = tracking.get(key)
    result["tail_leg_lengths_m"] = support.get("geometric_leg_length_mean_m_left_right")
    result["nonwheel_contact_fraction"] = support.get("nonwheel_contact_fraction_any_over_1n")
    return result


def compare_reports(baseline, candidate):
    """Report changes without treating process completion as control quality."""
    comparisons = {}
    scripted_checks = []
    limits = {"roll_rms_rad": 0.10, "pitch_rms_rad": 0.10,
              "xy_angular_velocity_rms_rad_per_s": 1.0,
              "vx_rmse_against_applied_target_m_per_s": 0.10,
              "yaw_rmse_against_applied_target_rad_per_s": 0.25,
              "nonwheel_contact_fraction": 0.01}
    for name, report in candidate.items():
        before = summarize_report(baseline[name])
        after = summarize_report(report)
        deltas = {}
        for key, value in after.items():
            old = before.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(old, (int, float)):
                deltas[key] = value-old
        comparisons[name] = {"baseline": before, "candidate": after, "candidate_minus_baseline": deltas}
        checks = {"all_requested_steps": after["completed_steps"] == after["requested_steps"],
                  "finite_outputs": after["finite_outputs"] is True,
                  "no_physical_failures": after["physical_failure_resets"] == 0,
                  "no_time_limit_resets": after["time_limit_resets"] == 0}
        for key, limit in limits.items():
            value = after.get(key)
            checks[key] = isinstance(value, (int, float)) and math.isfinite(value) and value <= limit
        comparisons[name]["scripted_checks"] = checks
        scripted_checks.extend(checks.values())
    passed = bool(scripted_checks) and all(scripted_checks)
    return {"cases": comparisons, "scripted_limits": limits, "scripted_checks_passed": passed,
            "quality_status": "scripted_checks_passed_requires_manual_validation" if passed else "needs_improvement",
            "interpretation": "Completed programs are not proof of stable control. These deterministic fixed-initial-state tests do not establish perturbation robustness or rigid-loop accuracy."}
