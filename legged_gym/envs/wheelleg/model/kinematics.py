"""五连杆平面几何：正解、腿角/腿长，以及对称姿态逆解。

长度为米、角度为弧度，尺寸统一读取 geometry.py。
B/D 是上杆端点，C 是两根等长下杆的交点；始终使用同一个装配分支。
"""

import math

import torch
from .geometry import UPPER, LOWER, HIP_SPAN, HIP_DOWN


def fivebar(phi1, phi4, pitch):
    """输入两根上杆的平面角，返回腿长、相对竖直方向的腿角和几何有效标志。"""
    xb, yb = UPPER * torch.cos(phi1), UPPER * torch.sin(phi1)
    xd, yd = HIP_SPAN + UPPER * torch.cos(phi4), HIP_DOWN + UPPER * torch.sin(phi4)
    dx, dy = xd - xb, yd - yb
    distance2 = dx.square() + dy.square()
    distance = distance2.clamp_min(1e-12).sqrt()
    height2 = LOWER**2 - 0.25 * distance2
    height = height2.clamp_min(0).sqrt()
    # 两圆交点：杆端中点 + 垂直于 BD 的偏移，选取机器人实际装配分支。
    xc = 0.5 * (xb + xd) - height * dy / distance
    yc = 0.5 * (yb + yd) + height * dx / distance
    x, y = xc - HIP_SPAN / 2, yc - HIP_DOWN / 2
    length = torch.sqrt(x.square() + y.square())
    theta = torch.atan2(y, x) - math.pi / 2 + pitch
    return length, theta, (height2 >= -1e-8) & (distance2 > 1e-10)


def inverse_symmetric(length):
    """给定竖直腿长求两髋角；保留 CAD 中两髋的微小高度差。"""
    x = torch.full_like(length, HIP_SPAN / 2)
    y1, y4 = length + HIP_DOWN / 2, length - HIP_DOWN / 2
    d1, d4 = torch.sqrt(x * x + y1 * y1), torch.sqrt(x * x + y4 * y4)
    phi1 = torch.atan2(y1, x) + torch.acos(
        ((d1 * d1 + UPPER**2 - LOWER**2) / (2 * d1 * UPPER)).clamp(-1, 1)
    )
    phi4 = torch.atan2(y4, -x) - torch.acos(
        ((d4 * d4 + UPPER**2 - LOWER**2) / (2 * d4 * UPPER)).clamp(-1, 1)
    )
    return phi1, phi4
