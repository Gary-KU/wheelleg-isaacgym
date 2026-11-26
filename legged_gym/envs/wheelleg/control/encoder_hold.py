"""轮角里程的停车/直行航向外环；仅修正策略速度参考，电机限幅不变。"""

import math

import torch

from legged_gym.envs.wheelleg.observations.odometry import PlanarWheelOdometry


class EncoderHold:
    def __init__(self, n, device, baseline, position_kp, heading_kp,
                 speed_limit, yaw_limit, height_slope):
        if (not all(math.isfinite(v) for v in
                    (baseline, position_kp, heading_kp, speed_limit, yaw_limit, height_slope))
                or min(position_kp, heading_kp, speed_limit, yaw_limit) <= 0):
            raise ValueError("Encoder hold requires finite parameters and positive gains/limits")
        self.odometry = PlanarWheelOdometry(n, device)
        self.baseline, self.height_slope = baseline, height_slope
        self.position_kp, self.heading_kp = position_kp, heading_kp
        self.speed_limit, self.yaw_limit = speed_limit, yaw_limit
        self.position_reference = torch.zeros(n, device=device)
        self.heading_reference = torch.zeros_like(self.position_reference)
        self.previous_forward_target = torch.zeros_like(self.position_reference)
        self.available = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.odometry.reset(ids)
        self.position_reference[ids] = self.heading_reference[ids] = 0.
        self.previous_forward_target[ids] = 0.
        self.available[ids] = False

    @staticmethod
    def _holding(commands, targets):
        # 用户开始驾驶/转向时即释放对应外环；指令斜坡结束后重新捕获参考点。
        stopped = (commands[:, 0].abs() < .02) & (targets[:, 0].abs() < 1e-5)
        straight = (commands[:, 2].abs() < .02) & (targets[:, 2].abs() < 1e-5)
        return stopped, straight

    def update(self, sensors, wheel_angles, commands, targets, observed):
        ready = self.odometry.update(sensors, wheel_angles, observed)
        stopped, straight = self._holding(commands, targets)
        position_capture = observed & (~ready | ~stopped)
        heading_capture = observed & (~ready | ~straight)
        # 起步/反向以用户此刻的车头方向为直行参考，不返回停车时的旧航向。
        new_direction = ((targets[:, 0].abs() > 1e-5)
                         & (targets[:, 0] * self.previous_forward_target <= 0.))
        heading_capture |= observed & straight & new_direction
        self.position_reference[position_capture] = self.odometry.distance[position_capture]
        self.heading_reference[heading_capture] = self.odometry.yaw[heading_capture]
        self.available[observed] = ready[observed]
        self.previous_forward_target[observed] = targets[observed, 0]

    def corrections(self, commands, targets):
        stopped, straight = self._holding(commands, targets)
        position_error = self.odometry.distance - self.position_reference
        heading_error = self.odometry.yaw - self.heading_reference
        forward = self.height_slope * (commands[:, 3] - self.baseline).clamp_min(0.)
        forward -= self.position_kp * position_error * stopped
        yaw = -self.heading_kp * heading_error * straight
        return (torch.where(self.available, forward.clamp(-self.speed_limit, self.speed_limit), 0.),
                torch.where(self.available, yaw.clamp(-self.yaw_limit, self.yaw_limit), 0.))
