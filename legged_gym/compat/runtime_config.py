"""运行参数合并：默认值 → 检查点 → 可选预设 → 显式命令行。

这里只处理 Python 数据，不导入 Gym 或 Torch。训练和推理共用恢复规则，
避免网络维度、物理步长和指令加速度在续训/播放时悄悄变化。
"""

from copy import deepcopy
import sys

from legged_gym.envs.wheelleg.constants import (
    DEFAULT_CONTROL_DT, OBSERVATION_BASE_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES, OBSERVATION_LEG_LENGTH_SIZE,
    OBSERVATION_TERRAIN_SIZE,
)


TRAIN_ENVIRONMENT_FIELDS = (
    "seed", "physics_dt", "closure_stiffness", "closure_damping", "command",
    "random_commands", "forward_speed_range", "yaw_speed_range", "leg_length_range",
    "tracking_penalties", "stability_rewards", "support_rewards", "observe_leg_lengths",
    "command_resample_seconds", "linear_acceleration", "yaw_acceleration",
    "zero_command_fraction", "initial_leg_length", "leg_control", "leg_kp", "leg_kd", "leg_feedforward_force",
    "heading_kp", "heading_kd", "wheel_control", "wheel_pitch_damping", "wheel_balance_scale", "physics_solver", "wheel_friction",
    "terrain", "observe_terrain", "stair_level", "stair_min_level", "stair_max_level", "stair_curriculum", "hip_target_scale",
    "stair_episode_seconds",
)
INFERENCE_ENVIRONMENT_FIELDS = (
    "seed", "physics_dt", "closure_stiffness", "closure_damping",
    "observe_leg_lengths", "linear_acceleration", "yaw_acceleration", "initial_leg_length",
    "leg_control", "leg_kp", "leg_kd", "leg_feedforward_force", "heading_kp", "heading_kd",
    "wheel_control", "wheel_pitch_damping", "wheel_balance_scale", "physics_solver", "wheel_friction",
    "terrain", "observe_terrain", "stair_level", "stair_min_level", "stair_max_level", "hip_target_scale",
    "stair_episode_seconds",
)
STABILITY_ENVIRONMENT = {
    "terrain": "plane", "stair_curriculum": False,
    "random_commands": True,
    "forward_speed_range": [-0.3, 0.3],
    "yaw_speed_range": [-0.8, 0.8],
    "leg_length_range": [0.08, 0.08],
    "command": [0.0, 0.0, 0.0, 0.08],
    "observe_leg_lengths": True,
    "tracking_penalties": True,
    "stability_rewards": True,
    "command_resample_seconds": 2.0,
    "zero_command_fraction": 0.2,
    "linear_acceleration": 0.8,
    "yaw_acceleration": 2.0,
}
SUPPORT_ENVIRONMENT = dict(STABILITY_ENVIRONMENT, **{
    "random_commands": False,
    "forward_speed_range": [0.0, 0.0],
    "yaw_speed_range": [0.0, 0.0],
    "command_resample_seconds": 0.0,
    "support_rewards": True,
    "leg_control": "torque",
    "leg_feedforward_force": 0.0,
    "heading_kp": 0.0,
    "heading_kd": 0.0,
})
HYBRID_ENVIRONMENT = dict(SUPPORT_ENVIRONMENT, **{
    "leg_control": "hip_pd",
    "leg_kp": 80.0,
    "leg_kd": 4.0,
    "initial_leg_length": 0.08,
})
STAIRS_ENVIRONMENT = dict(HYBRID_ENVIRONMENT, **{
    "terrain": "stairs", "observe_terrain": True,
    "stair_level": 0, "stair_min_level": 0, "stair_max_level": 4, "stair_curriculum": True,
    "stair_episode_seconds": 24.,
    "leg_control": "hip_target", "hip_target_scale": .2,
    "wheel_control": "policy",
    "wheel_pitch_damping": 0.0, "wheel_balance_scale": 1.0,
    "leg_feedforward_force": 22.4649, "heading_kp": 6., "heading_kd": .3,
    "initial_leg_length": .14, "command": [.25, 0., 0., .14],
    "forward_speed_range": [.25, .25], "yaw_speed_range": [0., 0.],
    "leg_length_range": [.14, .14], "random_commands": False,
    "linear_acceleration": .5, "yaw_acceleration": 2.,
})


def explicit_argument_destinations(parser, argv=None):
    """Find explicitly supplied argparse fields, including aliases and --x=value."""
    tokens = sys.argv[1:] if argv is None else argv
    explicit = set()
    for token in tokens:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            explicit.add(action.dest)
    return explicit


def checkpoint_observation_size(checkpoint):
    """Read the policy input width without importing its tensor library."""
    state = checkpoint.get("model_state_dict", {})
    weight = state.get("actor.0.weight")
    if weight is not None:
        if len(weight.shape) != 2:
            raise ValueError("Checkpoint actor input weight must be a matrix")
        return int(weight.shape[1])
    frame = OBSERVATION_BASE_FRAME_SIZE
    if checkpoint.get("environment_config", {}).get("observe_leg_lengths", False):
        frame += OBSERVATION_LEG_LENGTH_SIZE
    if checkpoint.get("environment_config", {}).get("observe_terrain", False):
        frame += OBSERVATION_TERRAIN_SIZE
    return frame * (OBSERVATION_HISTORY_FRAMES + 1)


def resolve_runtime_args(args, explicit, checkpoint=None, mode="train"):
    """Return resolved arguments while preserving device, window and batch choices.

    Inference restores the trained leg-length target but starts with zero motion
    unless --command was explicitly supplied. It never inherits random training
    mode, parallel environment count, device, or headless from a checkpoint.
    """
    if mode not in ("train", "play", "evaluate"):
        raise ValueError("Unknown runtime mode: {}".format(mode))
    resolved = deepcopy(args)
    original = deepcopy(vars(args))
    saved = (checkpoint or {}).get("environment_config", {})
    if saved.get("control_dt", DEFAULT_CONTROL_DT) != DEFAULT_CONTROL_DT:
        raise ValueError("This entry point requires checkpoint control_dt={}".format(DEFAULT_CONTROL_DT))
    fields = TRAIN_ENVIRONMENT_FIELDS if mode == "train" else INFERENCE_ENVIRONMENT_FIELDS
    for name in fields:
        if name in saved:
            setattr(resolved, name, deepcopy(saved[name]))
    if mode != "train" and "command" in saved:
        command = saved["command"]
        if len(command) != 4:
            raise ValueError("Checkpoint command must contain four values")
        resolved.command = [0.0, 0.0, 0.0, command[3]]

    profile = getattr(args, "profile", None)
    if profile in ("stability", "support", "hybrid", "stairs"):
        if mode == "train":
            preset = {"support": SUPPORT_ENVIRONMENT, "hybrid": HYBRID_ENVIRONMENT, "stairs": STAIRS_ENVIRONMENT,
                      "stability": STABILITY_ENVIRONMENT}[profile]
            for name, value in preset.items():
                if profile == "hybrid" and name in ("leg_feedforward_force", "heading_kp", "heading_kd"):
                    # Hybrid starts with zero feedforward, but resuming this
                    # controller family must retain saved support/heading gains.
                    continue
                setattr(resolved, name, deepcopy(value))
            resolved.num_envs = 1024
            resolved.max_learning_rate = 0.0003
            resolved.save_interval = 25
            if checkpoint is None:
                hip_noise = 3.0 if profile == "support" else 1.0
                resolved.init_noise_std = ([.1]*4+[.02, .02] if profile == "stairs"
                                           else [hip_noise]*4 + [0.1, 0.1])
                resolved.learning_rate = 0.0003
        else:
            # Network input dimensions always follow its saved observation mode.
            resolved.linear_acceleration = STABILITY_ENVIRONMENT["linear_acceleration"]
            resolved.yaw_acceleration = STABILITY_ENVIRONMENT["yaw_acceleration"]
            if profile != "stairs":
                # A selected flat-task profile also switches saved stair
                # checkpoints to a plane. Keep terrain input columns so its
                # 252-D network remains usable (the nine scans become zero).
                resolved.terrain = "plane"
                resolved.stair_curriculum = False
            if checkpoint is None:
                resolved.command = deepcopy(STABILITY_ENVIRONMENT["command"])

    for name in explicit:
        if name in original:
            setattr(resolved, name, original[name])

    if checkpoint is not None:
        frame = OBSERVATION_BASE_FRAME_SIZE + (
            OBSERVATION_LEG_LENGTH_SIZE if resolved.observe_leg_lengths else 0)
        if resolved.observe_terrain:
            frame += OBSERVATION_TERRAIN_SIZE
        wanted = frame * (OBSERVATION_HISTORY_FRAMES + 1)
        actual = checkpoint_observation_size(checkpoint)
        if wanted != actual:
            hint = ("Use tools/upgrade_observations.py to create a separate 198-D checkpoint first."
                    if actual == 186 and wanted == 198 else
                    "Keep observation flags consistent with the saved policy; use tools/prepare_stair_checkpoint.py for stairs.")
            raise ValueError("Checkpoint has {} observations, but runtime requests {}. {}".format(actual, wanted, hint))
    return resolved
