"""Playback motion measurements; simulation truth stays outside policy inputs."""

import torch

from ..model.constraints import quat_rotate_inverse


class MotionEvaluation:
    def __init__(self, roots, dt, settle_seconds=2.0):
        self.dt = dt
        self.settle_seconds = settle_seconds
        self.origin = roots[:, :3].clone()
        self.heading = self._yaw(roots)
        self.age = torch.zeros(len(roots), device=roots.device)
        self.sums = torch.zeros(5, device=roots.device)
        self.count = 0
        self.lateral_peak = self.forward_peak = self.heading_peak = 0.0
        self.stationary_speed_peak = 0.0
        self.stationary_speed_sum = 0.0
        self.stationary_count = 0

    @staticmethod
    def _yaw(roots):
        x, y, z, w = roots[:, 3:7].unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def update(self, roots, commands, dones, next_roots):
        velocity = quat_rotate_inverse(roots[:, 3:7], roots[:, 7:10])
        angular = quat_rotate_inverse(roots[:, 3:7], roots[:, 10:13])
        self.age += self.dt
        settled = self.age >= self.settle_seconds
        if bool(settled.any()):
            v, c = velocity[settled], commands[settled]
            self.sums += torch.stack(
                (
                    v[:, 0].sum(),
                    c[:, 0].sum(),
                    (v[:, 0] - c[:, 0]).square().sum(),
                    v[:, 1].square().sum(),
                    (angular[settled, 2] - c[:, 2]).square().sum(),
                )
            )
            self.count += int(settled.sum())
            stopped = settled & (commands[:, :3].abs().max(-1).values < 1e-5)
            if bool(stopped.any()):
                speed2 = velocity[stopped, :2].square().sum(-1)
                self.stationary_speed_sum += float(speed2.sum())
                self.stationary_count += len(speed2)
                self.stationary_speed_peak = max(
                    self.stationary_speed_peak, float(speed2.sqrt().max())
                )
        delta = roots[:, :3] - self.origin
        cosine, sine = self.heading.cos(), self.heading.sin()
        forward = cosine * delta[:, 0] + sine * delta[:, 1]
        lateral = -sine * delta[:, 0] + cosine * delta[:, 1]
        heading_error = self._yaw(roots) - self.heading
        heading_error = torch.atan2(heading_error.sin(), heading_error.cos())
        self.forward_peak = max(self.forward_peak, float(forward.abs().max()))
        self.lateral_peak = max(self.lateral_peak, float(lateral.abs().max()))
        self.heading_peak = max(self.heading_peak, float(heading_error.abs().max()))
        # New episode starts at the reset pose, never count a reset teleport as drift.
        self.origin[dones] = next_roots[dones, :3]
        self.heading[dones] = self._yaw(next_roots[dones])
        self.age[dones] = 0

    def report(self):
        values = (self.sums / self.count).tolist() if self.count else [None] * 5
        return dict(
            settle_seconds=self.settle_seconds,
            settled_samples=self.count,
            mean_forward_speed_m_s=values[0],
            mean_command_forward_speed_m_s=values[1],
            rms_forward_speed_error_m_s=None if not self.count else values[2] ** 0.5,
            rms_lateral_speed_m_s=None if not self.count else values[3] ** 0.5,
            rms_yaw_rate_error_rad_s=None if not self.count else values[4] ** 0.5,
            stationary_rms_speed_m_s=(
                None
                if not self.stationary_count
                else (self.stationary_speed_sum / self.stationary_count) ** 0.5
            ),
            stationary_peak_speed_m_s=self.stationary_speed_peak,
            segment_peak_forward_displacement_m=self.forward_peak,
            segment_peak_cross_track_m=self.lateral_peak,
            segment_peak_heading_change_rad=self.heading_peak,
            note="Displacements are relative to each reset pose/heading; cross-track is meaningful for straight commands. Settled speed statistics exclude the configured settle_seconds after each reset.",
        )
