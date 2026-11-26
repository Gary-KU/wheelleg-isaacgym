"""停车后的累计漂移打分；仿真位置/航向仅用于奖励，不进入控制或观测。"""

import math

import torch


class StandstillReward:
    """伸腿和减速完成后留出稳定时间，再固定每台机器人的停车参考点。

    驾驶或伸缩时立即取消参考点，下一次停稳重新捕获。没有任何位置反馈
    力矩；该奖励只能鼓励减少漂移，不能让无位置观测的策略精确返回原点。
    """

    def __init__(self, n, device, dt, settle_seconds, position_sigma, heading_sigma):
        values = (dt, settle_seconds, position_sigma, heading_sigma)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Standstill reward parameters must be finite and positive")
        self.settle_steps = max(1, math.ceil(settle_seconds / dt))
        self.position_sigma, self.heading_sigma = position_sigma, heading_sigma
        self.age = torch.zeros(n, dtype=torch.long, device=device)
        self.origin = torch.zeros(n, 2, device=device)
        self.heading = torch.zeros(n, device=device)
        self.displacement = torch.zeros(n, device=device)
        self.heading_error = torch.zeros(n, device=device)

    def reset(self, ids):
        self.age[ids] = 0
        self.displacement[ids] = 0
        self.heading_error[ids] = 0

    def update(self, position, yaw, commands, targets):
        idle = (commands[:, :3].abs().amax(-1) < 1e-5)
        idle &= targets[:, :3].abs().amax(-1) < 1e-5
        idle &= (commands[:, 3] - targets[:, 3]).abs() < 1e-5
        self.age = torch.where(idle, (self.age + 1).clamp_max(self.settle_steps), 0)
        capturing = self.age < self.settle_steps
        self.origin[capturing] = position[capturing, :2]
        self.heading[capturing] = yaw[capturing]
        active = ~capturing
        distance = (position[:, :2] - self.origin).square().sum(-1).sqrt()
        angle = yaw - self.heading
        angle = torch.atan2(angle.sin(), angle.cos())
        self.displacement.copy_(torch.where(active, distance, 0))
        self.heading_error.copy_(torch.where(active, angle, 0))
        return {
            "standstill_position": (self.displacement / self.position_sigma).square().clamp_max(25),
            "standstill_heading": (self.heading_error / self.heading_sigma).square().clamp_max(25),
        }
