"""主动伸腿支撑的纯 Python 统计与工程初筛，不导入仿真或张量库。"""

import math


LEG_TOLERANCE_M = 0.005
PASSIVE_LIMIT_MARGIN_RAD = 0.01


def duration_steps(seconds, control_dt, allow_zero=False):
    """Require an exact number of control periods instead of silently rounding."""
    if (not math.isfinite(seconds) or not math.isfinite(control_dt)
            or control_dt <= 0 or seconds < 0 or (seconds == 0 and not allow_zero)):
        raise ValueError("Duration and control period must be finite and positive")
    steps = int(round(seconds/control_dt))
    if not math.isclose(steps*control_dt, seconds, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Duration must be a multiple of the control period")
    return steps


def velocity_kick_schedule(at_seconds, delta_velocity, control_dt, total_steps):
    """Validate one additive WORLD-frame base velocity kick, before a step.

    Returning None keeps historical evaluation behavior. A kick at t=5 s is
    applied after 250 completed 20 ms periods, before period 251.
    """
    if at_seconds is None and delta_velocity is None:
        return None
    if at_seconds is None or delta_velocity is None:
        raise ValueError("--kick-at-seconds and --kick-velocity must be supplied together")
    step = duration_steps(at_seconds, control_dt, allow_zero=True)
    if step >= total_steps:
        raise ValueError("The velocity kick must occur before the end of evaluation")
    if not isinstance(delta_velocity, (list, tuple)) or len(delta_velocity) != 3:
        raise ValueError("Kick velocity must contain WORLD-frame X Y Z values")
    delta = [_number(value, "kick velocity") for value in delta_velocity]
    if not any(delta):
        raise ValueError("Kick velocity must have at least one nonzero component")
    return {"before_step_index_zero_based": step, "time_s": float(at_seconds),
            "frame": "world", "delta_linear_velocity_m_per_s": delta}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("{} must contain finite numbers".format(name))
    return float(value)


def _vector(row, key, width):
    values = row.get(key)
    if not isinstance(values, (tuple, list)) or len(values) != width:
        raise ValueError("{} must contain {} values".format(key, width))
    return [_number(value, key) for value in values]


def _rms(values):
    values = list(values)
    return math.sqrt(math.fsum(value*value for value in values)/len(values))


def summarize_base_motion(samples):
    """Keep terminal motion visible; never infer stillness from upright pose.

    Displacements use the original run position. Unwrapped yaw is supplied by
    the evaluator, excluding automatic-reset orientation teleports.
    """
    keys = ("base_xy_displacement_world_m", "base_linear_velocity_world_m_per_s",
            "base_yaw_drift_rad")
    availability = [all(key in row for key in keys) for row in samples]
    if not any(availability):
        if any(any(key in row for key in keys) for row in samples):
            raise ValueError("Base motion fields must be supplied together")
        return None
    if not all(availability):
        raise ValueError("Base motion is missing from some support samples")
    displacement = [_vector(row, keys[0], 2) for row in samples]
    velocity = [_vector(row, keys[1], 3) for row in samples]
    yaw = [_number(row[keys[2]], keys[2]) for row in samples]
    final = {}
    for row, xy, heading in zip(samples, displacement, yaw):
        index = row.get("environment_index", 0)
        final[index] = {"environment_index": index,
                        "xy_displacement_world_m": xy, "yaw_drift_rad": heading}
    return {
        "samples": len(samples), "frame": "world",
        "planar_speed_rms_m_per_s": _rms(math.hypot(*row[:2]) for row in velocity),
        "planar_speed_peak_m_per_s": max(math.hypot(*row[:2]) for row in velocity),
        "linear_velocity_rms_m_per_s_xyz": [_rms(row[j] for row in velocity) for j in range(3)],
        "linear_velocity_mean_m_per_s_xyz": [math.fsum(row[j] for row in velocity)/len(velocity) for j in range(3)],
        "xy_displacement_abs_peak_m_xy": [max(abs(row[j]) for row in displacement) for j in range(2)],
        "xy_displacement_peak_m": max(math.hypot(*row) for row in displacement),
        "yaw_drift_abs_peak_rad": max(abs(value) for value in yaw),
        "end_per_environment": [final[key] for key in sorted(final)],
    }


def summarize_support(samples, hip_effort_limits_nm=(9.0, 9.0, 9.0, 9.0),
                      leg_tolerance_m=LEG_TOLERANCE_M,
                      passive_limit_margin_rad=PASSIVE_LIMIT_MARGIN_RAD):
    """Summarize one finite sample per environment and control step.

    Callers must supply terminal/pre-reset positions and endpoint torques.
    Optional within-control-period peaks and saturation fractions preserve
    events hidden by the last physics substep. RMS still uses endpoints only.
    Samples are pooled equally; RMS is never averaged across episodes.
    """
    limits = [_number(value, "hip_effort_limits_nm") for value in hip_effort_limits_nm]
    if len(limits) != 4 or any(value <= 0 for value in limits):
        raise ValueError("Four positive hip effort limits are required")
    if (not math.isfinite(leg_tolerance_m) or leg_tolerance_m <= 0
            or not math.isfinite(passive_limit_margin_rad) or passive_limit_margin_rad < 0):
        raise ValueError("Support tolerances must be finite and valid")
    if not samples:
        return None
    lengths, errors, margins, torques, positions = [], [], [], [], []
    torque_peaks, saturation_fractions = [], []
    substep_peak_samples = substep_saturation_samples = 0
    pitch, roll, wheels, nonwheel = [], [], [], []
    physical_failures = timeouts = 0
    position_width = len(samples[0].get("dof_positions_rad", []))
    if not position_width:
        raise ValueError("dof_positions_rad must not be empty")
    for row in samples:
        length = _vector(row, "leg_lengths_m", 2)
        target = _number(row.get("target_leg_length_m"), "target_leg_length_m")
        lengths.append(length)
        errors.append([value-target for value in length])
        margins.append(_vector(row, "passive_limit_margin_rad", 4))
        torque = _vector(row, "hip_torques_nm", 4)
        torques.append(torque)
        if "hip_torque_peak_nm_per_joint" in row:
            peaks = _vector(row, "hip_torque_peak_nm_per_joint", 4)
            if any(peak < 0 or peak < abs(value)-1e-5 for peak, value in zip(peaks, torque)):
                raise ValueError("Hip torque peaks must be nonnegative and cover the endpoint")
            substep_peak_samples += 1
        else:
            peaks = [abs(value) for value in torque]
        torque_peaks.append(peaks)
        if "hip_torque_substep_saturation_fraction" in row:
            fractions = _vector(row, "hip_torque_substep_saturation_fraction", 4)
            if any(value < 0 or value > 1 for value in fractions):
                raise ValueError("Hip torque saturation fractions must be in [0,1]")
            substep_saturation_samples += 1
        else:
            fractions = [float(abs(value) >= limit-1e-5) for value, limit in zip(torque, limits)]
        saturation_fractions.append(fractions)
        positions.append(_vector(row, "dof_positions_rad", position_width))
        pitch.append(_number(row.get("pitch_rad"), "pitch_rad"))
        roll.append(_number(row.get("roll_rad"), "roll_rad"))
        contact = row.get("wheel_contacts_over_1n")
        if (not isinstance(contact, (tuple, list)) or len(contact) != 2
                or any(not isinstance(value, bool) for value in contact)):
            raise ValueError("wheel_contacts_over_1n must contain two booleans")
        wheels.append(contact)
        for key in ("nonwheel_contact_over_1n", "physical_failure", "timeout"):
            if not isinstance(row.get(key), bool):
                raise ValueError("{} must be a boolean".format(key))
        nonwheel.append(row["nonwheel_contact_over_1n"])
        physical_failures += int(row["physical_failure"])
        timeouts += int(row["timeout"])
    count = len(samples)
    near_limit = [[value <= passive_limit_margin_rad for value in row] for row in margins]
    return {
        "samples": count,
        "base_motion": summarize_base_motion(samples),
        "leg_length_mean_m_left_right": [math.fsum(row[j] for row in lengths)/count for j in range(2)],
        "leg_length_min_m_left_right": [min(row[j] for row in lengths) for j in range(2)],
        "leg_length_max_m_left_right": [max(row[j] for row in lengths) for j in range(2)],
        "leg_length_error_rmse_m_left_right": [_rms(row[j] for row in errors) for j in range(2)],
        "leg_length_error_abs_peak_m_left_right": [max(abs(row[j]) for row in errors) for j in range(2)],
        "left_right_length_difference_rmse_m": _rms(row[0]-row[1] for row in lengths),
        "both_legs_within_tolerance_fraction": sum(
            all(abs(value) <= leg_tolerance_m+1e-12 for value in row) for row in errors)/count,
        "leg_tolerance_m": leg_tolerance_m,
        "passive_min_limit_margin_rad_per_joint": [min(row[j] for row in margins) for j in range(4)],
        "passive_any_near_limit_fraction": sum(any(row) for row in near_limit)/count,
        "passive_near_limit_fraction_per_joint": [sum(row[j] for row in near_limit)/count for j in range(4)],
        "passive_near_limit_threshold_rad": passive_limit_margin_rad,
        "hip_torque_rms_nm_per_joint": [_rms(row[j] for row in torques) for j in range(4)],
        "hip_torque_abs_peak_nm_per_joint": [max(row[j] for row in torque_peaks) for j in range(4)],
        "hip_torque_saturation_fraction_per_joint": [math.fsum(row[j] for row in saturation_fractions)/count for j in range(4)],
        "hip_torque_saturation_fraction": math.fsum(math.fsum(row) for row in saturation_fractions)/(4*count),
        "torque_sampling": {
            "rms": "control-step endpoint commands only, not all physics substeps",
            "peak_substep_samples": substep_peak_samples,
            "peak_endpoint_fallback_samples": count-substep_peak_samples,
            "saturation_substep_samples": substep_saturation_samples,
            "saturation_endpoint_fallback_samples": count-substep_saturation_samples,
        },
        "hip_effort_limits_nm": limits,
        "pitch_rms_rad": _rms(pitch),
        "pitch_abs_peak_rad": max(abs(value) for value in pitch),
        "roll_rms_rad": _rms(roll),
        "roll_abs_peak_rad": max(abs(value) for value in roll),
        "wheel_contact_fraction_over_1n_left_right": [sum(row[j] for row in wheels)/count for j in range(2)],
        "both_wheels_contact_fraction_over_1n": sum(all(row) for row in wheels)/count,
        "either_wheel_contact_fraction_over_1n": sum(any(row) for row in wheels)/count,
        "nonwheel_contact_fraction_any_over_1n": sum(nonwheel)/count,
        "dof_position_mean_rad": [math.fsum(row[j] for row in positions)/count for j in range(position_width)],
        "dof_position_min_rad": [min(row[j] for row in positions) for j in range(position_width)],
        "dof_position_max_rad": [max(row[j] for row in positions) for j in range(position_width)],
        "physical_failure_resets": physical_failures,
        "time_limit_resets": timeouts,
    }


def assess_support(after_settle, *, requested_steps, completed_steps, num_envs,
                   sample_count, finite_outputs, physical_failures, timeouts,
                   require_motion=False):
    """Explicit engineering screen, including failures before the settling cutoff."""
    if requested_steps < 1 or num_envs < 1 or completed_steps < 0 or sample_count < 0:
        raise ValueError("Step and environment counts must be valid")
    metric = after_settle or {}
    available = after_settle is not None and after_settle.get("samples", 0) > 0
    checks = {
        "all_requested_steps": completed_steps == requested_steps,
        "all_completed_samples_recorded": sample_count == completed_steps*num_envs,
        "finite_outputs": finite_outputs is True,
        "has_after_settle_samples": available,
        "no_physical_failures_entire_run": physical_failures == 0,
        "no_time_limit_resets_entire_run": timeouts == 0,
    }
    thresholds = {
        "both_legs_within_5mm_fraction_min": 0.95,
        "passive_any_within_001rad_fraction_max": 0.0,
        "nonwheel_contact_fraction_over_1n_max": 0.0,
        "pitch_rms_rad_max": 0.1,
        "roll_rms_rad_max": 0.1,
    }
    for name, key, limit, minimum in (
        ("both_legs_within_5mm", "both_legs_within_tolerance_fraction", 0.95, True),
        ("no_passive_limit_dwelling", "passive_any_near_limit_fraction", 0.0, False),
        ("no_nonwheel_contact", "nonwheel_contact_fraction_any_over_1n", 0.0, False),
        ("pitch_rms", "pitch_rms_rad", 0.1, False),
        ("roll_rms", "roll_rms_rad", 0.1, False),
    ):
        value = metric.get(key)
        valid = (available and isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value))
        checks[name] = bool(valid and (value >= limit if minimum else value <= limit))
    # A caller must not use wider summary tolerances while labelling the check 5mm/0.01rad.
    checks["standard_summary_tolerances"] = bool(
        available and metric.get("leg_tolerance_m") == LEG_TOLERANCE_M
        and metric.get("passive_near_limit_threshold_rad") == PASSIVE_LIMIT_MARGIN_RAD)
    # Legacy reports lack root motion; current standing evaluation requires it.
    # These are explicit stationary engineering tolerances, not a proof of
    # robust recovery from arbitrary disturbances.
    motion = metric.get("base_motion")
    if motion is not None or require_motion:
        for name, key, limit in (
            ("stationary_planar_speed", "planar_speed_rms_m_per_s", 0.05),
            ("stationary_xy_displacement", "xy_displacement_peak_m", 0.05),
            ("stationary_yaw_drift", "yaw_drift_abs_peak_rad", 0.1),
        ):
            thresholds[key+"_max"] = limit
            value = (motion or {}).get(key)
            checks[name] = bool(isinstance(value, (int, float)) and not isinstance(value, bool)
                                and math.isfinite(value) and value <= limit)
    passed = all(checks.values())
    return {"engineering_screen_passed": passed,
            "quality_status": "engineering_screen_passed_requires_further_validation" if passed else "needs_improvement",
            "checks": checks, "thresholds": thresholds,
            "interpretation": "Fixed-initial-state support screening only. This does not establish arbitrary keyboard control, disturbance robustness, or measured joint-stop reaction forces."}
