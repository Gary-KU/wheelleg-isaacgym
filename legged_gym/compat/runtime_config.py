"""运行参数合并：工具默认值 → 检查点 → 显式命令行。

这里只处理 Python 数据，不导入 Gym 或 Torch。仅供抬轮和专用评估工具恢复参数，
避免网络维度、物理步长和指令加速度在续训/播放时悄悄变化。
"""

from copy import deepcopy
import sys

from legged_gym.envs.wheelleg.constants import (
    DEFAULT_CONTROL_DT,
    OBSERVATION_BASE_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES,
    OBSERVATION_LEG_LENGTH_SIZE,
    OBSERVATION_TERRAIN_SIZE,
)

INFERENCE_ENVIRONMENT_FIELDS = (
    "seed",
    "physics_dt",
    "closure_stiffness",
    "closure_damping",
    "observe_leg_lengths",
    "linear_acceleration",
    "yaw_acceleration",
    "initial_leg_length",
    "leg_control",
    "leg_kp",
    "leg_kd",
    "leg_feedforward_force",
    "heading_kp",
    "heading_kd",
    "wheel_control",
    "wheel_pitch_damping",
    "wheel_balance_scale",
    "physics_solver",
    "wheel_friction",
    "terrain",
    "observe_terrain",
    "stair_level",
    "stair_min_level",
    "stair_max_level",
    "hip_target_scale",
    "stair_episode_seconds",
)


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


def resolve_runtime_args(args, explicit, checkpoint=None, mode="play"):
    """Return resolved arguments while preserving device, window and batch choices.

    Inference restores the trained leg-length target but starts with zero motion
    unless --command was explicitly supplied. It never inherits random training
    mode, parallel environment count, device, or headless from a checkpoint.
    """
    if mode not in ("play", "evaluate"):
        raise ValueError("Unknown runtime mode: {}".format(mode))
    resolved = deepcopy(args)
    original = deepcopy(vars(args))
    saved = (checkpoint or {}).get("environment_config", {})
    if saved.get("control_dt", DEFAULT_CONTROL_DT) != DEFAULT_CONTROL_DT:
        raise ValueError(
            "This entry point requires checkpoint control_dt={}".format(
                DEFAULT_CONTROL_DT
            )
        )
    for name in INFERENCE_ENVIRONMENT_FIELDS:
        if name in saved:
            setattr(resolved, name, deepcopy(saved[name]))
    if "command" in saved:
        command = saved["command"]
        if len(command) != 4:
            raise ValueError("Checkpoint command must contain four values")
        resolved.command = [0.0, 0.0, 0.0, command[3]]

    for name in explicit:
        if name in original:
            setattr(resolved, name, original[name])

    if checkpoint is not None:
        frame = OBSERVATION_BASE_FRAME_SIZE + (
            OBSERVATION_LEG_LENGTH_SIZE if resolved.observe_leg_lengths else 0
        )
        if resolved.observe_terrain:
            frame += OBSERVATION_TERRAIN_SIZE
        wanted = frame * (OBSERVATION_HISTORY_FRAMES + 1)
        actual = checkpoint_observation_size(checkpoint)
        if wanted != actual:
            raise ValueError(
                "Checkpoint has {} observations, but runtime requests {}. Select the matching task and observation layout; retrain when changing layouts.".format(
                    actual, wanted
                )
            )
    return resolved
