"""由轮编码器角度和腿部几何估计相对里程；不读取仿真位姿或速度。"""

import math

import torch

from legged_gym.envs.wheelleg.model.geometry import (
    UPPER, LOWER, HIP_SPAN, HIP_DOWN, WHEEL_RADIUS, WHEEL_TRACK,
)


class PlanarWheelOdometry:
    """平地滚动近似：补偿轮子安装连杆的旋转及伸腿引起的轮心平移。

    输入为原13维SI传感器和两个原始轮轴角度（左/右、rad）。角度零偏
    在差分中抵消，不积分电机速度通道的恒定零偏。输出是相对路程/航向，
    不是绝对定位；轮胎打滑、腾空、明显横滚时不保证准确。
    """

    def __init__(self, n, device):
        self.distance = torch.zeros(n, device=device)
        self.yaw = torch.zeros_like(self.distance)
        self.previous_angles = torch.zeros(n, 2, device=device)
        self.previous_projection = torch.zeros_like(self.previous_angles)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)

    @staticmethod
    def _wheel_geometry(sensors, wheel_angles):
        pitch = sensors[:, 0:1]
        angle, length = sensors[:, 3:5], sensors[:, 5:7]
        # 当前CAD轮子分别安装在L_F_Link2/R_H_Link2，均对应各侧phi1分支。
        # FK平面坐标左右相反；由腿角/腿长反解安装杆角度，换模型须核对分支。
        side = sensors.new_tensor([-1., 1.])
        cx = HIP_SPAN / 2 + side * length * angle.sin()
        cy = HIP_DOWN / 2 + length * angle.cos()
        radius = (cx.square() + cy.square()).clamp_min(1e-12).sqrt()
        cosine = (radius.square() + UPPER**2 - LOWER**2) / (2 * UPPER * radius)
        upper_angle = torch.atan2(cy, cx) + cosine.clamp(-1., 1.).acos()
        carrier = torch.atan2(cy - UPPER * upper_angle.sin(),
                              cx - UPPER * upper_angle.cos())
        absolute_wheel_angle = side * wheel_angles + pitch + side * carrier
        projection = length * (angle - pitch).sin()
        valid = (torch.isfinite(sensors).all(-1)
                 & torch.isfinite(wheel_angles).all(-1)
                 & (length > 1e-5).all(-1) & (cosine.abs() <= 1.0001).all(-1))
        return absolute_wheel_angle, projection, valid

    def reset(self, ids):
        self.distance[ids] = self.yaw[ids] = 0.
        self.initialized[ids] = False

    def update(self, sensors, wheel_angles, observed=None):
        angles, projection, valid = self._wheel_geometry(sensors, wheel_angles)
        if observed is None:
            observed = torch.ones_like(valid)
        valid &= observed
        delta = angles - self.previous_angles
        delta = torch.atan2(delta.sin(), delta.cos())  # 编码器跨±pi时仍连续
        travel = WHEEL_RADIUS * delta - (projection - self.previous_projection)
        ready = valid & self.initialized
        travel = torch.where(ready[:, None], travel, 0.)
        self.distance += travel.mean(-1)
        self.yaw += (travel[:, 1] - travel[:, 0]) / WHEEL_TRACK
        self.previous_angles[valid] = angles[valid]
        self.previous_projection[valid] = projection[valid]
        self.initialized[observed] = valid[observed]
        return ready


class StationaryCalibratedGyroHeading:
    """停车时用轮角差估计陀螺零偏，运动时积分陀螺，避免轮胎滚动误差。

    不改变网络收到的原始传感器。未取得至少1秒安静样本时沿用轮角增量，
    不阻止用户起步；标定仅在指令、轮速和腿部运动均很小时更新。
    """

    def __init__(self, n, device, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Gyro heading requires a positive finite sample period")
        self.dt = dt
        self.heading = torch.zeros(n, device=device)
        self.bias = torch.zeros_like(self.heading)
        self.previous_wheel_yaw = torch.zeros_like(self.heading)
        self.quiet_time = torch.zeros_like(self.heading)
        self.gyro_integral = torch.zeros_like(self.heading)
        self.wheel_integral = torch.zeros_like(self.heading)
        self.calibrated = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        for value in (self.heading, self.bias, self.previous_wheel_yaw,
                      self.quiet_time, self.gyro_integral, self.wheel_integral):
            value[ids] = 0.
        self.calibrated[ids] = False

    def update(self, sensors, wheel_yaw, commands, targets, ready, observed):
        delta = wheel_yaw - self.previous_wheel_yaw
        quiet = (ready & (commands[:, :3].abs().amax(-1) < .01)
                 & (targets[:, :3].abs().amax(-1) < 1e-5)
                 & (sensors[:, 11:13].abs().amax(-1) < .5)
                 & (sensors[:, 1:3].abs().amax(-1) < .08)
                 & (sensors[:, 7:9].abs().amax(-1) < .15)
                 & (sensors[:, 9:11].abs().amax(-1) < .02))
        broken = observed & ~quiet
        self.quiet_time[broken] = 0.
        self.gyro_integral[broken] = self.wheel_integral[broken] = 0.
        self.quiet_time[quiet] += self.dt
        self.gyro_integral[quiet] += sensors[quiet, 2] * self.dt
        self.wheel_integral[quiet] += delta[quiet]
        learn = quiet & (self.quiet_time >= 1.)
        self.bias[learn] = ((self.gyro_integral[learn] - self.wheel_integral[learn])
                            / self.quiet_time[learn])
        self.calibrated[learn] = True
        increment = torch.where(quiet | ~self.calibrated, delta,
                                (sensors[:, 2] - self.bias) * self.dt)
        self.heading += torch.where(ready, increment, 0.)
        # 无效/丢失帧不积分；恢复时重捕获上一轮角，不补算未知区间。
        self.previous_wheel_yaw[observed] = wheel_yaw[observed]


class EncoderPositionSampler:
    """与13维传感器对齐的轮角采样；噪声为假设值，需按实机标定。

    每回合固定角度零偏、逐帧白噪声及同样的整数拍延迟。独立随机源不
    改变旧13维传感器的噪声序列；原始角度只供可选里程控制器使用。
    """

    def __init__(self, n, device, delay_steps, noise_std, bias_std, seed):
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or not 0 <= delay_steps <= 10:
            raise ValueError("Wheel angle delay must be an integer in0..10")
        if not all(math.isfinite(v) and v >= 0 for v in (noise_std, bias_std)):
            raise ValueError("Wheel angle noise/bias must be finite and nonnegative")
        self.noise_std, self.bias_std = noise_std, bias_std
        self.bias = torch.zeros(n, 2, device=device)
        self.queue = torch.zeros(n, delay_steps + 1, 2, device=device)
        self.sample = torch.zeros_like(self.bias)
        self.generator = torch.Generator(device=device).manual_seed(seed + 1429)

    def _noise(self, shape):
        return torch.randn(shape, device=self.sample.device, generator=self.generator)

    def reset(self, ids, angles):
        self.bias[ids] = self.bias_std * self._noise((len(ids), 2))
        value = angles[ids] + self.bias[ids] + self.noise_std * self._noise((len(ids), 2))
        self.queue[ids] = value[:, None]
        self.sample[ids] = value

    def advance(self, angles):
        self.queue[:, :-1] = self.queue[:, 1:].clone()
        self.queue[:, -1] = angles + self.bias + self.noise_std * self._noise(angles.shape)
        self.sample.copy_(self.queue[:, 0])
