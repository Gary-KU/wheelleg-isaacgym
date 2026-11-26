#!/usr/bin/env python3
"""给已验证的站立网络接入速度指令，再使用统一 PPO 入口训练。"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from legged_gym.envs import register_tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wheelleg.observations.sensors import (
    FRAME_SIZE,
    FRAME_FIELDS,
    SENSOR_FIELDS,
    SENSOR_SCALES,
    COMMAND_SCALES,
)


def condition_commands(actor, gains, yaw_command_gain, yaw_rate_feedback_gain):
    """在网络权重中调整转向阻尼和指令参考，运行时仍只有原网络。"""
    first = actor[0]
    if not isinstance(first, torch.nn.Linear) or first.in_features % FRAME_SIZE:
        raise ValueError("Expected stacked sensor frames and a Linear actor input")
    pitch_scale = SENSOR_SCALES[SENSOR_FIELDS.index("pitch")]
    yaw_scale = SENSOR_SCALES[SENSOR_FIELDS.index("yaw_rate")]
    # v_error = v - requested_v；两个转向反馈都跟踪 requested_yaw。
    pitch_reference = -gains["speed_kp"] / gains["kp"] * pitch_scale / COMMAND_SCALES[0]
    yaw_reference = (
        -(gains["yaw_kd"] * yaw_rate_feedback_gain + gains["encoder_yaw_kd"])
        / gains["yaw_kd"]
        * yaw_scale
        / COMMAND_SCALES[1]
        * yaw_command_gain
    )
    with torch.no_grad():
        original = first.weight.clone()
        for offset in range(0, first.in_features, FRAME_SIZE):
            first.weight[
                :, offset + FRAME_FIELDS.index("yaw_rate")
            ] *= yaw_rate_feedback_gain
            for command, state, factor in (
                ("command_vx", "pitch", pitch_reference),
                ("command_yaw_rate", "yaw_rate", yaw_reference),
            ):
                first.weight[:, offset + FRAME_FIELDS.index(command)] += (
                    factor * original[:, offset + FRAME_FIELDS.index(state)]
                )
    return pitch_reference, yaw_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standing-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--yaw-command-gain", type=float, default=1.22)
    p.add_argument("--yaw-rate-feedback-gain", type=float, default=20.0)
    args = p.parse_args()
    if not all(
        math.isfinite(x) and x > 0
        for x in (args.yaw_command_gain, args.yaw_rate_feedback_gain)
    ):
        p.error("Yaw reference/feedback gains must be positive and finite")
    torch.set_num_threads(1)
    source = torch.load(str(args.standing_checkpoint), map_location="cpu")
    gains = source["legged_gym_train_config"]["initialization_provenance"]
    if (
        gains.get("method") != "analytic_sensor_feedback_initialization"
        or gains["kp"] <= 0
        or gains["yaw_kd"] <= 0
    ):
        raise ValueError(
            "This initializer requires the verified feedback-initialized standing actor"
        )
    register_tasks()
    execution = get_args(
        "train",
        [
            "--task",
            "wheelleg_sensor_drive",
            "--num_envs",
            "1",
            "--sim_device",
            "cpu",
            "--headless",
            "--warm-start",
            str(args.standing_checkpoint),
        ],
    )
    cfg, train = task_registry.prepare_cfgs(execution.task, execution)
    train.initialization_provenance.update(
        method="standing_actor_command_conditioning",
        source_rl_iterations=source["iter"],
        yaw_command_gain=args.yaw_command_gain,
        yaw_rate_feedback_gain=args.yaw_rate_feedback_gain,
        source_sha256=hashlib.sha256(args.standing_checkpoint.read_bytes()).hexdigest(),
        note="Reference conditioning in actor weights; no runtime teacher; requires motion training and evaluation",
    )
    env, _ = task_registry.make_env(execution.task, args=execution, env_cfg=cfg)
    try:
        runner, _ = task_registry.make_alg_runner(
            env, execution.task, execution, train, log_root=None
        )
        actor = runner.alg.actor_critic.actor
        previous = copy.deepcopy(actor)
        pitch_factor, yaw_factor = condition_commands(
            actor, gains, args.yaw_command_gain, args.yaw_rate_feedback_gain
        )
        samples = torch.rand(256, env.num_obs) * 0.1 - 0.05
        virtual = samples.clone()
        frames = virtual.view(-1, env.num_obs // FRAME_SIZE, FRAME_SIZE)
        frames[..., FRAME_FIELDS.index("pitch")] += (
            pitch_factor * frames[..., FRAME_FIELDS.index("command_vx")]
        )
        frames[..., FRAME_FIELDS.index("yaw_rate")] = (
            args.yaw_rate_feedback_gain * frames[..., FRAME_FIELDS.index("yaw_rate")]
            + yaw_factor * frames[..., FRAME_FIELDS.index("command_yaw_rate")]
        )
        with torch.no_grad():
            error = float((actor(samples) - previous(virtual)).abs().max())
        if error > 2e-5:
            raise RuntimeError(
                "Actor/reference transformation mismatch: {}".format(error)
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        runner.save(args.output)
        report = dict(
            checkpoint=str(args.output.resolve()),
            source=str(args.standing_checkpoint.resolve()),
            reference_transform_equivalence_error=error,
            pitch_reference_factor=pitch_factor,
            yaw_reference_factor=yaw_factor,
            rl_motion_updates=0,
        )
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
