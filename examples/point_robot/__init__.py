from .env import PointRobot
from .config import Config

def register(registry):
    registry.register('point_robot', PointRobot, Config(), Config.Train())
