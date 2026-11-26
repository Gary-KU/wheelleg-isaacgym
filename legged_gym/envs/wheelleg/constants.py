"""项目默认参数：关节顺序、单位、观测布局和 PPO 设置。

本模块只使用 Python 标准库，可以在未加载 Isaac Gym / PyTorch 时阅读或导入。
命令行可以覆盖运行参数；已有权重内保存的配置仍是对应实验的依据。
"""

import math
from .model.geometry import MIN_LENGTH, MAX_LENGTH, ACTUATORS


# 动作顺序不可随意改动：旧权重的六个输出按这个顺序解释。
ACTION_JOINTS = (
    "L_F_Joint1", "L_H_Joint1", "R_F_Joint1", "R_H_Joint1",
    "L_lun_Joint", "R_lun_Joint",
)
NUM_ACTIONS = len(ACTION_JOINTS)
# 输出轴力矩；±1 动作对应各自峰值，避免轮子动作大部分落在饱和区。
HIP_TORQUE_LIMIT = ACTUATORS["hip_peak_nm"]
HIP_CONTINUOUS_TORQUE = ACTUATORS["hip_continuous_nm"]
HIP_PEAK_DURATION = ACTUATORS["hip_peak_duration_s"]
WHEEL_TORQUE_LIMIT = ACTUATORS["wheel_peak_nm"]
ACTION_TORQUE_LIMITS = (HIP_TORQUE_LIMIT,)*4 + (WHEEL_TORQUE_LIMIT,)*2
ACTION_SCALES = ACTION_TORQUE_LIMITS

# 新 CAD 直接使用连杆惯量，旧模型的大人工转子惯量已移除。
DEFAULT_LEG_KD = 0.2  # 髋阻尼，N·m·s/rad；惯量实测后需重新核对。

# 物理步与策略控制步分开：默认每次动作执行 40 个物理步。
DEFAULT_PHYSICS_DT = 0.0005  # s
DEFAULT_CONTROL_DT = 0.02  # s
DEFAULT_EPISODE_SECONDS = 16.0
DEFAULT_CLOSURE_STIFFNESS = 50000.0  # N/m，柔性闭合点弹簧
DEFAULT_CLOSURE_DAMPING = 2.0  # N·s/m；轻量 CAD 模型在 0.5 ms 下避免显式阻尼振荡
CLOSURE_MAX_FORCE = 500.0  # N
TERMINATION_LOOP_ERROR = 0.03  # m
TERMINATION_TILT = 0.35  # rad，分别检查 pitch 和 roll

# 每帧旧观测 31 维；可选追加左右实际腿长 2 维；堆叠当前帧和 5 帧历史。
OBSERVATION_BASE_FRAME_SIZE = 31
OBSERVATION_LEG_LENGTH_SIZE = 2
OBSERVATION_TERRAIN_SIZE = 9  # 三个前向距离×三个横向位置的地面相对高度。
OBSERVATION_HISTORY_FRAMES = 5

# 命令顺序：[vx (m/s), vy (m/s), yaw_rate (rad/s), leg_length (m)]。
DEFAULT_COMMAND = (0.0, 0.0, 0.0, 0.08)
DEFAULT_FORWARD_SPEED_RANGE = (-2.0, 2.0)
DEFAULT_YAW_SPEED_RANGE = (-2 * math.pi, 2 * math.pi)
MIN_LEG_LENGTH = MIN_LENGTH
MAX_LEG_LENGTH = MAX_LENGTH
DEFAULT_LEG_LENGTH_RANGE = (MIN_LEG_LENGTH, MAX_LEG_LENGTH)


# 抬轮/评估工具加载权重时的网络结构后备值；训练参数统一在 config/base.py。
DEFAULT_POLICY_CONFIG = {
    "actor_hidden_dims": [512, 256, 128, 64],
    "critic_hidden_dims": [256, 128, 64],
    "activation": "elu",
    "init_noise_std": [0.03]*4 + [0.25, 0.25],
}

# Shared per-control-step flat reward scales. The task config can override each
# value; the legacy environment uses this same mapping without a second formula.
FLAT_REWARD_SCALES = {
    "termination": -100.0,
    "tracking_lin_vel": 3.0,
    "direction": 1.0,
    "tracking_ang_vel": 2.0,
    "lin_vel_z": -0.2,
    "effort": -0.0001,
    "action_rate": -0.003,
    "leg_tracking": 2.0,
    "leg_error": -0.25,
    "leg_asymmetry": -0.5,
    "passive_limit": -4.0,
    "hip_pose": -0.1,
    "upright": 6.0,
    "angular_motion": -0.02,
    "hip_symmetry": 0.7,
    "leg_angle": 0.5,
    "heading": -1.0,
    "tracking_error": 1.0,
    "stability": 1.0,
    "yaw_error": -0.5,
}
