"""五连杆平面几何：正解、腿角/腿长，以及对称姿态逆解。

长度单位为米，角度为弧度。这里保留源任务的装配分支与几何常量；
修改杆长时还需同步核对 URDF 和闭合点，不能只改本文件。
"""

import math

import torch
from .geometry import UPPER, LOWER, HIP_SPAN, HIP_DOWN


def fivebar(phi1, phi4, pitch):
    """Same assembly branch as the Lab task; report unreachable geometry."""
    xb, yb = UPPER * torch.cos(phi1), UPPER * torch.sin(phi1)
    xd, yd = HIP_SPAN + UPPER * torch.cos(phi4), HIP_DOWN + UPPER * torch.sin(phi4)
    dx, dy = xd - xb, yd - yb
    distance2 = dx.square() + dy.square()
    distance = distance2.clamp_min(1e-12).sqrt()
    height2 = LOWER**2 - 0.25 * distance2
    height = height2.clamp_min(0).sqrt()
    xc = 0.5 * (xb + xd) - height * dy / distance
    yc = 0.5 * (yb + yd) + height * dx / distance
    x, y = xc - HIP_SPAN / 2, yc - HIP_DOWN / 2
    length = torch.sqrt(x.square() + y.square())
    theta = torch.atan2(y, x) - math.pi / 2 + pitch
    return length, theta, (height2 >= -1e-8) & (distance2 > 1e-10)


def inverse_symmetric(length):
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
