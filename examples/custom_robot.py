"""相似五连杆机器人的注册示例；异构机器人请实现自己的环境。"""
from legged_gym.envs.wheelleg.config.sensors import WheelLeggedDriveCfg, WheelLeggedDriveCfgPPO
from legged_gym.envs.wheelleg.tasks.tracking import DrivingWheelLegged


class MyCfg(WheelLeggedDriveCfg):
    # 此示例沿用 WheelLeg 几何与关节；不能只替换 URDF 适配任意机器人。
    pass


class MyPPO(WheelLeggedDriveCfgPPO):
    class runner(WheelLeggedDriveCfgPPO.runner):
        experiment_name = 'my_robot'


def register(registry):
    registry.register('my_robot', DrivingWheelLegged, MyCfg(), MyPPO())
