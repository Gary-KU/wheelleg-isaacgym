"""五连杆平面几何：正解、腿角/腿长，以及对称姿态逆解。

长度单位为米，角度为弧度。这里保留源任务的装配分支与几何常量；
修改杆长时还需同步核对 URDF 和闭合点，不能只改本文件。
"""

import math

import torch


def fivebar(phi1, phi4, pitch):
    """Same assembly branch as the Lab task; report unreachable geometry."""
    xb, yb = 0.105*torch.cos(phi1), 0.105*torch.sin(phi1)
    xd, yd = 0.088+0.105*torch.cos(phi4), 0.105*torch.sin(phi4)
    dx, dy = xd-xb, yd-yb
    aa, bb = 0.32*dx, 0.32*dy
    cc = dx*dx+dy*dy
    disc = aa*aa+bb*bb-cc*cc
    phi2 = 2*torch.atan2(bb+torch.sqrt(disc.clamp_min(0)), aa+cc)
    xc, yc = xb+0.16*torch.cos(phi2), yb+0.16*torch.sin(phi2)
    length = torch.sqrt((xc-0.044)**2+yc*yc)
    theta = torch.atan2(yc, xc-0.044)-math.pi/2+pitch
    return length, theta, (disc >= -1e-8) & (cc > 1e-10)


def inverse_symmetric(length):
    a = torch.sqrt(0.044**2+length*length)
    alpha0 = torch.asin((length/a).clamp(-1, 1))
    alpha1 = torch.acos(((a*a+0.105**2-0.16**2)/(2*a*0.105)).clamp(-1, 1))
    return alpha1+alpha0, math.pi-alpha1-alpha0
