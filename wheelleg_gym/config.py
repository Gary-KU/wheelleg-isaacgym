"""项目默认参数：关节顺序、单位、观测布局和 PPO 设置。

本模块只使用 Python 标准库，可以在未加载 Isaac Gym / PyTorch 时阅读或导入。
命令行可以覆盖运行参数；已有权重内保存的配置仍是对应实验的依据。
"""

import math


# 动作顺序不可随意改动：旧权重的六个输出按这个顺序解释。
ACTION_JOINTS = (
    "L_F_Joint1", "L_H_Joint1", "R_F_Joint1", "R_H_Joint1",
    "L_lun_Joint", "R_lun_Joint",
)
NUM_ACTIONS = len(ACTION_JOINTS)
ACTION_SCALES = (0.5, 0.5, 0.5, 0.5, 10.0, 10.0)  # 原始动作 → N·m
TORQUE_LIMIT = 9.0  # 每个主动关节的力矩限幅，N·m

# 物理步与策略控制步分开：默认每次动作执行 40 个物理步。
DEFAULT_PHYSICS_DT = 0.0005  # s
DEFAULT_CONTROL_DT = 0.02  # s
DEFAULT_EPISODE_SECONDS = 16.0
DEFAULT_CLOSURE_STIFFNESS = 50000.0  # N/m，柔性闭合点弹簧
DEFAULT_CLOSURE_DAMPING = 25.0  # N·s/m
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
MIN_LEG_LENGTH = 0.06
MAX_LEG_LENGTH = 0.20  # 软件目标上限，20cm对称腿姿仍有约0.16rad真实关节余量。
# 保留原训练默认范围；扩大手动可调范围不改变旧训练课程。
DEFAULT_LEG_LENGTH_RANGE = (MIN_LEG_LENGTH, 0.15)


# PPO 默认值保留原任务设置。训练入口会 deepcopy，再应用命令行或权重配置。
TRAIN_CONFIG = {
    "policy": {
        "actor_hidden_dims": [512, 256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
        "init_noise_std": 4.0,
    },
    "algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 5,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.995,
        "lam": 0.95,
        "desired_kl": 0.008,
        "max_grad_norm": 1.0,
    },
    "num_steps_per_env": 25,
    "save_interval": 100,
}
