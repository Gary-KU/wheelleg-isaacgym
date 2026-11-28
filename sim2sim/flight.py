"""Experimental accelerometer-only flight compliance; disabled by default."""

import numpy as np
import torch


class FlightCompliance:
    def __init__(self, n, dt, seed):
        self.dt = dt
        self.rng = np.random.default_rng(seed + 411)
        self.air = torch.zeros(n, dtype=torch.bool)
        self.low = torch.zeros(n)
        self.high = torch.zeros(n)
        self.blend = torch.ones(n)
        self.queue = np.zeros((n, 2, 3))
        self.bias = np.zeros((n, 3))
        self.samples = np.zeros((n, 3))

    def reset(self, ids, acc):
        ids = np.asarray(ids)
        self.air[ids] = False
        self.low[ids] = 0
        self.high[ids] = 0
        self.blend[ids] = 1
        self.bias[ids] = self.rng.normal(0, 0.05, (len(ids), 3))
        value = acc[ids] + self.bias[ids] + self.rng.normal(0, 0.05, (len(ids), 3))
        self.queue[ids] = value[:, None]
        self.samples[ids] = value

    def update(self, acc):
        self.queue[:, :-1] = self.queue[:, 1:]
        self.queue[:, -1] = acc + self.bias + self.rng.normal(0, 0.05, acc.shape)
        self.samples = self.queue[:, 0].copy()
        norm = torch.tensor(np.linalg.norm(self.samples, axis=1), dtype=torch.float32)
        self.low = torch.where(norm < 3, self.low + self.dt, 0)
        self.high = torch.where(norm > 6, self.high + self.dt, 0)
        self.air = torch.where(
            self.low >= 0.01, True, torch.where(self.high >= 0.004, False, self.air)
        )
        self.blend = torch.where(
            self.air, 0.0, (self.blend + self.dt / 0.02).clamp(max=1)
        )
        return self.blend
