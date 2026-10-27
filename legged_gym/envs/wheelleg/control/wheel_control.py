"""Optional explicit steering while retaining the learned wheel balance effort.

This asset has mirrored wheel axes: [-drive,+drive] rolls forward, while
[steer,steer] turns towards positive yaw. Torques are joint-side Nm.
"""

import math
import torch


def balance_heading_torques(policy_torques, heading_error, yaw_rate_error,
                            pitch_rate, kp, kd, pitch_damping=0., limit=9.):
    """Keep RL differential effort and replace its common steering effort.

    Heading error is measured minus target. Steering uses at most 1 Nm per
    wheel and only the remaining actuator headroom after balance effort.
    Positive pitch rate needs positive drive effort: its reaction opposes
    positive body pitch. No state, root pose or contact force is imposed.
    """
    if policy_torques.ndim != 2 or policy_torques.shape[1] != 2:
        raise ValueError("Wheel policy torques must have shape [N,2]")
    if any(value.shape != policy_torques.shape[:1]
           for value in (heading_error, yaw_rate_error, pitch_rate)):
        raise ValueError("Heading and rate vectors must have shape [N]")
    if any(not math.isfinite(value) or value < 0 for value in (kp, kd, pitch_damping)):
        raise ValueError("Wheel feedback gains must be finite and nonnegative")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Torque limit must be finite and positive")
    drive = .5*(policy_torques[:, 1]-policy_torques[:, 0])+pitch_damping*pitch_rate
    drive = drive.clamp(-limit, limit)
    steer = (-kp*heading_error-kd*yaw_rate_error).clamp(-1., 1.)
    headroom = limit-drive.abs()
    steer = torch.maximum(-headroom, torch.minimum(headroom, steer))
    return torch.stack((steer-drive, steer+drive), dim=-1)


def bounded_heading_target(measured, target, command, dt, max_error=.35):
    """Integrate requested turning without storing an unachievable yaw backlog."""
    return measured+(target+command*dt-measured).clamp(-max_error, max_error)
