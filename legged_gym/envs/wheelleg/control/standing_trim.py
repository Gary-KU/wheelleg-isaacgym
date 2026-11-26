"""伸腿后静止漂移的慢速积分修正；只读 IMU/轮编码器，不读取仿真位置或速度。"""
import math

import torch

from legged_gym.envs.wheelleg.model.geometry import WHEEL_RADIUS
from legged_gym.envs.wheelleg.observations.sensors import SENSOR_FIELDS


class HeightStandingTrim:
    """学习每米伸腿量对应的速度参考偏置；中立腿长时输出严格为零。

    只在指令静止、腿长指令不变且已接近停稳时更新。行驶/伸缩时冻结系数，
    补偿随伸腿量变化，缩回中立长度自然归零，不更改原网络或电机限幅。
    initial_coefficient是离线标定的启动系数，不是传感器零偏或重力力矩；
    旧检查点默认0，重置时恢复同一初值，在线更新仅修正剩余速度误差。
    """

    def __init__(self, num_envs, device, baseline, span, ki, limit, initial_coefficient=0.0):
        if (not all(math.isfinite(x) for x in (baseline, span, ki, limit, initial_coefficient))
                or min(span, ki, limit) <= 0):
            raise ValueError("Standing trim requires finite positive span, ki and limit")
        self.baseline, self.ki, self.limit = baseline, ki, limit
        self.coefficient_limit = limit / span
        if abs(initial_coefficient) * span > limit:
            raise ValueError("Initial standing trim exceeds the reference correction limit")
        # 可选的离线高度标定初值，单位s^-1；8cm时仍严格为零。
        # 用它减少升高后从零积分的等待，在线修正仍只读原传感器。
        self.initial_coefficient = initial_coefficient
        self.coefficient = torch.full((num_envs,), float(initial_coefficient), device=device)
        self.settled_seconds = torch.zeros_like(self.coefficient)
        self.previous_length = torch.full_like(self.coefficient, baseline)
        self.velocity = torch.zeros_like(self.coefficient)
        self.wheel_columns = [
            SENSOR_FIELDS.index("wheel_forward_rate_" + side)
            for side in ("left", "right")
        ]
        self.pitch_rate_column = SENSOR_FIELDS.index("pitch_rate")

    def reset(self, ids):
        self.coefficient[ids] = self.initial_coefficient
        self.settled_seconds[ids] = 0.0
        self.previous_length[ids] = self.baseline
        self.velocity[ids] = 0.0

    def update(self, sensors, commands, valid, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Standing trim dt must be finite and positive")
        length = commands[:, 3]
        stopped = commands[:, :3].abs().max(-1).values < 0.02
        unchanged = (length - self.previous_length).abs() <= 1e-7
        self.settled_seconds = torch.where(
            stopped & unchanged & valid,
            (self.settled_seconds + dt).clamp_max(1.0), torch.zeros_like(length))
        self.previous_length.copy_(length)
        # 与现有平衡策略相同的低速编码器估计；腿长稳定后才使用。
        velocity = WHEEL_RADIUS * (sensors[:, self.wheel_columns].mean(-1)
                                  + sensors[:, self.pitch_rate_column])
        finite = torch.isfinite(sensors).all(-1)
        # 接触振动会使瞬时轮速越过门槛。先低通再判定，避免只积分某一符号
        # 的样本而产生偏置；该滤波仅属于慢速修正，不延迟策略原观测。
        alpha = -math.expm1(-dt / 0.1)
        self.velocity += alpha * (torch.where(finite, velocity, self.velocity) - self.velocity)
        extension = (length - self.baseline).clamp_min(0.0)
        active = (valid & finite & (extension > 0.001)
                  & (self.settled_seconds >= 0.5) & (self.velocity.abs() < 0.04))
        change = -self.ki * (self.velocity - commands[:, 0]) * dt / extension.clamp_min(0.001)
        self.coefficient += torch.where(active, change, torch.zeros_like(change))
        self.coefficient.clamp_(-self.coefficient_limit, self.coefficient_limit)

    def correction(self, commands):
        extension = (commands[:, 3] - self.baseline).clamp_min(0.0)
        return (self.coefficient * extension).clamp(-self.limit, self.limit)
