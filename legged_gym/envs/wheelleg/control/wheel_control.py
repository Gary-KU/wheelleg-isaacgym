"""Optional explicit steering while retaining the learned wheel balance effort.

This asset has mirrored wheel axes: [-drive,+drive] rolls forward, while
[steer,steer] turns towards positive yaw. Torques are joint-side Nm.
"""

import math
import torch
from legged_gym.envs.wheelleg.constants import WHEEL_TORQUE_LIMIT


def balance_heading_torques(
    policy_torques,
    heading_error,
    yaw_rate_error,
    pitch_rate,
    kp,
    kd,
    pitch_damping=0.0,
    limit=WHEEL_TORQUE_LIMIT,
):
    """Keep RL differential effort and replace its common steering effort.

    Heading error is measured minus target. Steering uses at most 1 Nm per
    wheel and only the remaining actuator headroom after balance effort.
    Positive pitch rate needs positive drive effort: its reaction opposes
    positive body pitch. No state, root pose or contact force is imposed.
    """
    if policy_torques.ndim != 2 or policy_torques.shape[1] != 2:
        raise ValueError("Wheel policy torques must have shape [N,2]")
    if any(
        value.shape != policy_torques.shape[:1]
        for value in (heading_error, yaw_rate_error, pitch_rate)
    ):
        raise ValueError("Heading and rate vectors must have shape [N]")
    if any(not math.isfinite(value) or value < 0 for value in (kp, kd, pitch_damping)):
        raise ValueError("Wheel feedback gains must be finite and nonnegative")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Torque limit must be finite and positive")
    drive = (
        0.5 * (policy_torques[:, 1] - policy_torques[:, 0]) + pitch_damping * pitch_rate
    )
    drive = drive.clamp(-limit, limit)
    steer = (-kp * heading_error - kd * yaw_rate_error).clamp(-1.0, 1.0)
    headroom = limit - drive.abs()
    steer = torch.maximum(-headroom, torch.minimum(headroom, steer))
    return torch.stack((steer - drive, steer + drive), dim=-1)


def bounded_heading_target(measured, target, command, dt, max_error=0.35):
    """Integrate requested turning without storing an unachievable yaw backlog."""
    return measured + (target + command * dt - measured).clamp(-max_error, max_error)


def sensor_stand_actions(
    frame,
    pitch_target,
    kp=0.5,
    kd=0.06,
    yaw_kd=0.005,
    speed_kp=0.3,
    encoder_yaw_kd=0.03,
):
    """站立参考反馈，输入是归一化的最新传感器帧；输出六维策略动作。

    仅使用俯仰、IMU 角速度和编码器轮速。髋动作零，由原 PD/前馈支撑；
    这用于诊断和网络初始化，不代表已经完成强化学习训练。
    """
    from legged_gym.envs.wheelleg.observations.sensors import (
        FRAME_SIZE,
        FRAME_FIELDS,
        SENSOR_FIELDS,
        SENSOR_SCALES,
    )
    from legged_gym.envs.wheelleg.model.geometry import WHEEL_RADIUS, WHEEL_TRACK

    if frame.ndim != 2 or frame.shape[1] != FRAME_SIZE:
        raise ValueError("Expected one normalized sensor frame per environment")
    if not all(
        math.isfinite(v) and v >= 0 for v in (kp, kd, yaw_kd, speed_kp, encoder_yaw_kd)
    ) or not math.isfinite(pitch_target):
        raise ValueError("Feedback gains must be finite and nonnegative")

    def state(name):
        return (
            frame[:, FRAME_FIELDS.index(name)]
            / SENSOR_SCALES[SENSOR_FIELDS.index(name)]
        )

    rate = state("pitch_rate")
    velocity = WHEEL_RADIUS * (
        0.5 * (state("wheel_forward_rate_left") + state("wheel_forward_rate_right"))
        + rate
    )
    drive = kp * (state("pitch") - pitch_target) + kd * rate + speed_kp * velocity
    encoder_yaw_rate = (
        WHEEL_RADIUS
        / WHEEL_TRACK
        * (state("wheel_forward_rate_right") - state("wheel_forward_rate_left"))
    )
    steer = -yaw_kd * state("yaw_rate") - encoder_yaw_kd * encoder_yaw_rate
    actions = frame.new_zeros((len(frame), 6))
    actions[:, 4] = (-drive + steer) / WHEEL_TORQUE_LIMIT
    actions[:, 5] = (drive + steer) / WHEEL_TORQUE_LIMIT
    return actions.clamp(-1, 1)
