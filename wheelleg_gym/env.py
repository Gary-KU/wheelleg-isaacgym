"""Compatibility import; physics now lives in legged_gym.envs.base.legged_robot.

Alias the module too, so existing diagnostic mocks patch the real backend.
"""
import sys
from legged_gym.envs.base import legged_robot as _backend

_backend.WheelLegGymEnv = _backend.LeggedRobot
sys.modules[__name__] = _backend
