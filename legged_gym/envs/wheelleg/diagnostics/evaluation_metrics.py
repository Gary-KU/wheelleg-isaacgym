"""评估的任务调度与数值统计，不依赖 Isaac Gym 或 PyTorch。

任务调度负责生成并行速度分组、连续速度/转向序列和控制步边界。
数值统计负责计算跟踪误差、姿态振荡和力矩变化；这些结果不代表策略已通过验收。
本模块只处理传入的数据，仿真与报告写入仍由 evaluate.py 负责。
"""

import math


# 任务调度：决定各环境、各控制步的目标，不主动重置机器人。
def make_speed_scenarios(num_envs, test_speeds, command):
    """Return fixed per-environment commands and contiguous speed groups."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if len(command) != 4 or not all(math.isfinite(float(x)) for x in command):
        raise ValueError("command must contain four finite values")
    if test_speeds is None:
        return [list(command) for _ in range(num_envs)], []
    speeds = [float(speed) for speed in test_speeds]
    if not speeds or not all(math.isfinite(speed) for speed in speeds):
        raise ValueError("--test-speeds must contain finite speeds")
    if num_envs % len(speeds):
        raise ValueError("--num-envs must be an integer multiple of the number of --test-speeds")
    per_group = num_envs // len(speeds)
    commands, scenarios = [], []
    for group, speed in enumerate(speeds):
        indices = list(range(group*per_group, (group+1)*per_group))
        scenarios.append({"target_vx_m_per_s": speed, "environment_indices": indices})
        for _ in indices:
            row = list(command)
            row[0] = speed
            commands.append(row)
    return commands, scenarios


def make_speed_sequence(speeds, segment_steps, yaw_rates=None, repeats=1, default_yaw=0.0):
    """Build zero-based, half-open segment bounds without requesting resets."""
    if speeds is None:
        if yaw_rates is not None:
            raise ValueError("--yaw-sequence requires --speed-sequence")
        return []
    if not isinstance(segment_steps, int) or segment_steps < 1:
        raise ValueError("--segment-steps must be a positive integer")
    speeds = [float(speed) for speed in speeds]
    if not speeds or not all(math.isfinite(speed) for speed in speeds):
        raise ValueError("--speed-sequence must contain finite speeds")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("--sequence-repeats must be a positive integer")
    yaw_rates = [default_yaw]*len(speeds) if yaw_rates is None else [float(x) for x in yaw_rates]
    if len(yaw_rates) != len(speeds) or not all(math.isfinite(x) for x in yaw_rates):
        raise ValueError("--yaw-sequence must match speed-sequence length and contain finite values")
    pattern = list(zip(speeds, yaw_rates))*repeats
    return [{"segment_index": index, "target_vx_m_per_s": speed,
             "target_yaw_rate_rad_per_s": yaw,
             "start_step": index*segment_steps,
             "end_step_exclusive": (index+1)*segment_steps}
            for index, (speed, yaw) in enumerate(pattern)]


def sequence_segment_at_step(segments, step):
    if step < 0:
        raise ValueError("Sequence step must be nonnegative")
    for segment in segments:
        if segment["start_step"] <= step < segment["end_step_exclusive"]:
            return segment
    raise ValueError("Step is outside the speed sequence")


# 数值统计：分别保留原始目标与限速后目标的误差。
def velocity_statistics(samples, target_vx):
    """Summarize step-by-environment speeds against each environment's target."""
    if not target_vx:
        raise ValueError("Velocity targets must not be empty")
    return tracking_statistics(samples, [target_vx]*len(samples))


def tracking_statistics(samples, target_rows):
    """Summarize against targets that can change at every control step."""
    if len(samples) != len(target_rows):
        raise ValueError("Velocity samples and target rows must have equal step counts")
    if any(not row or len(row) != len(target) for row, target in zip(samples, target_rows)):
        raise ValueError("Each velocity sample must match the number of targets")
    count = sum(len(row) for row in samples)
    if not count:
        return {"samples": 0, "mean_vx_m_per_s": None, "vx_tracking_rmse_m_per_s": None}
    speed_sum = math.fsum(float(value) for row in samples for value in row)
    squared_error = math.fsum((float(value)-float(target))**2
                             for row, targets in zip(samples, target_rows)
                             for value, target in zip(row, targets))
    return {"samples": count, "mean_vx_m_per_s": speed_sum/count,
            "vx_tracking_rmse_m_per_s": math.sqrt(squared_error/count)}


def dynamics_statistics(roll, pitch, angular_xy, torque, torque_rate, torque_limits):
    """Finite control-step diagnostics, independent of Gym for CPU verification."""
    if not roll:
        return None
    def rms(values):
        values = list(values)
        return math.sqrt(math.fsum(float(x)**2 for x in values)/len(values))
    xy_norm = [math.hypot(*xy) for xy in angular_xy]
    flat_rates = [x for row in torque_rate for x in row]
    saturation = [[abs(value) >= limit-1e-5 for value, limit in zip(row, torque_limits)] for row in torque]
    return {
        "samples": len(roll),
        "roll_rms_rad": rms(roll), "roll_abs_peak_rad": max(abs(x) for x in roll),
        "pitch_rms_rad": rms(pitch), "pitch_abs_peak_rad": max(abs(x) for x in pitch),
        "xy_angular_velocity_rms_rad_per_s": rms(xy_norm),
        "xy_angular_velocity_norm_peak_rad_per_s": max(xy_norm),
        "active_torque_rate_rms_nm_per_s": rms(flat_rates),
        "active_torque_rate_abs_peak_nm_per_s": max(abs(x) for x in flat_rates),
        "active_torque_saturation_fraction": sum(sum(row) for row in saturation)/(len(torque)*len(torque_limits)),
        "active_torque_rate_rms_nm_per_s_per_joint": [rms(row[j] for row in torque_rate) for j in range(len(torque_limits))],
        "active_torque_saturation_fraction_per_joint": [sum(row[j] for row in saturation)/len(saturation) for j in range(len(torque_limits))],
    }


def command_tracking_statistics(vx_rows, yaw_rows, raw_commands, applied_commands):
    """Keep requested-target errors visible when acceleration limits smooth commands."""
    if not vx_rows:
        return {"samples": 0}
    def column(rows, index):
        return [[command[index] for command in row] for row in rows]
    raw_vx = tracking_statistics(vx_rows, column(raw_commands, 0))
    applied_vx = tracking_statistics(vx_rows, column(applied_commands, 0))
    raw_yaw = tracking_statistics(yaw_rows, column(raw_commands, 2))
    applied_yaw = tracking_statistics(yaw_rows, column(applied_commands, 2))
    return {
        "samples": raw_vx["samples"],
        "mean_vx_m_per_s": raw_vx["mean_vx_m_per_s"],
        "mean_yaw_rate_rad_per_s": raw_yaw["mean_vx_m_per_s"],
        "vx_rmse_against_raw_target_m_per_s": raw_vx["vx_tracking_rmse_m_per_s"],
        "vx_rmse_against_applied_target_m_per_s": applied_vx["vx_tracking_rmse_m_per_s"],
        "yaw_rmse_against_raw_target_rad_per_s": raw_yaw["vx_tracking_rmse_m_per_s"],
        "yaw_rmse_against_applied_target_rad_per_s": applied_yaw["vx_tracking_rmse_m_per_s"],
        "raw_target_start": raw_commands[0], "raw_target_end": raw_commands[-1],
        "applied_target_start": applied_commands[0], "applied_target_end": applied_commands[-1],
    }
