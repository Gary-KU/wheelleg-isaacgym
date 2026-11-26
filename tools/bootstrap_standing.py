#!/usr/bin/env python3
"""把已验证的传感器反馈写成网络初值；随后仍用统一 PPO 入口微调。"""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from legged_gym.envs import register_tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wheelleg.constants import WHEEL_TORQUE_LIMIT
from legged_gym.envs.wheelleg.model.geometry import WHEEL_RADIUS, WHEEL_TRACK
from legged_gym.envs.wheelleg.observations.sensors import (
    FRAME_SIZE,
    FRAME_FIELDS,
    SENSOR_FIELDS,
    SENSOR_SCALES,
)
from legged_gym.envs.wheelleg.control.wheel_control import sensor_stand_actions


def initialize_actor(actor, num_obs, target, kp, kd, yaw_kd, speed_kp, encoder_yaw_kd):
    """ELU 在正半轴为恒等映射；用正偏置通道精确传递五个传感器量。"""
    layers = [m for m in actor if isinstance(m, torch.nn.Linear)]
    if any(not isinstance(m, (torch.nn.Linear, torch.nn.ELU)) for m in actor):
        raise ValueError("This initializer supports the current Linear/ELU actor only")
    if (
        len(layers) < 2
        or min(m.out_features for m in layers[:-1]) < 5
        or layers[-1].out_features != 6
    ):
        raise ValueError("Actor needs five hidden channels and six outputs")
    names = [
        "pitch",
        "pitch_rate",
        "yaw_rate",
        "wheel_forward_rate_left",
        "wheel_forward_rate_right",
    ]
    scales = [SENSOR_SCALES[SENSOR_FIELDS.index(n)] for n in names]
    offsets = layers[0].weight.new_tensor([1.0, 10.0, 10.0, 10.0, 10.0])
    # drive = kp*pitch + (kd+speed_kp*R)*pitch_rate + speed_kp*R*mean(wheels)
    drive = offsets.new_tensor(
        [
            kp / scales[0],
            (kd + speed_kp * WHEEL_RADIUS) / scales[1],
            0.0,
            0.5 * speed_kp * WHEEL_RADIUS / scales[3],
            0.5 * speed_kp * WHEEL_RADIUS / scales[4],
        ]
    )
    steer = torch.zeros_like(drive)
    steer[2] = -yaw_kd / scales[2]
    steer[3] = encoder_yaw_kd * WHEEL_RADIUS / WHEEL_TRACK / scales[3]
    steer[4] = -encoder_yaw_kd * WHEEL_RADIUS / WHEEL_TRACK / scales[4]
    with torch.no_grad():
        for layer in layers:
            layer.weight.normal_(0.0, 0.01)
            layer.bias.zero_()
        # 其他通道保留微弱随机支路，使 PPO 后续仍能学习非线性修正。
        layers[0].weight[:5].zero_()
        layers[-1].weight.mul_(1e-6)
        layers[-1].weight[:, :5].zero_()
        for i, name in enumerate(names):
            layers[0].weight[i, num_obs - FRAME_SIZE + FRAME_FIELDS.index(name)] = 1
        layers[0].bias[:5] = offsets
        for layer in layers[1:-1]:
            layer.weight[:5].zero_()
            layer.weight[:5, :5] = torch.eye(5, device=layer.weight.device)
        for row, sign in ((4, -1.0), (5, 1.0)):
            weights = (sign * drive + steer) / WHEEL_TORQUE_LIMIT
            layers[-1].weight[row, :5] = weights
            layers[-1].bias[row] = (
                -sign * kp * target / WHEEL_TORQUE_LIMIT - weights.dot(offsets)
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sim-device", default="cpu")
    p.add_argument(
        "--control-dt",
        type=float,
        help="Optional policy interval in seconds; an integer multiple of physics dt",
    )
    p.add_argument("--kp", type=float, default=0.5)
    p.add_argument("--kd", type=float, default=0.06)
    p.add_argument("--yaw-kd", type=float, default=0.005)
    p.add_argument("--speed-kp", type=float, default=0.3)
    p.add_argument("--encoder-yaw-kd", type=float, default=0.03)
    args = p.parse_args()
    torch.set_num_threads(1)
    register_tasks()
    execution = get_args(
        "train",
        [
            "--task",
            "wheelleg_sensor_quiet",
            "--headless",
            "--sim_device",
            args.sim_device,
            "--num_envs",
            "1",
        ],
    )
    cfg, train = task_registry.prepare_cfgs(execution.task, execution)
    if args.control_dt is not None:
        ratio = args.control_dt / cfg.sim.dt
        if not math.isfinite(ratio) or ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                "control-dt must be a positive integer multiple of physics dt"
            )
        cfg.control.decimation = round(ratio)
    train.policy.init_noise_std = [0.005] * 4 + [0.015, 0.015]
    train.algorithm.learning_rate = 0.000003
    train.algorithm.max_learning_rate = 0.00001
    train.algorithm.schedule = "fixed"
    train.algorithm.num_learning_epochs = 2
    train.algorithm.entropy_coef = 0.00001
    env, _ = task_registry.make_env(execution.task, args=execution, env_cfg=cfg)
    try:
        target = env.initial_com_balance_diagnostics["pitch_correction_rad"]
        gains = dict(
            kp=args.kp,
            kd=args.kd,
            yaw_kd=args.yaw_kd,
            speed_kp=args.speed_kp,
            encoder_yaw_kd=args.encoder_yaw_kd,
        )
        provenance = dict(
            method="analytic_sensor_feedback_initialization",
            rl_updates_at_initialization=0,
            control_dt_s=env.dt,
            pitch_target_rad=target,
            **gains,
            note="Network initialization, not an RL-trained standing result; no runtime teacher override"
        )
        train.initialization_provenance = provenance
        runner, _ = task_registry.make_alg_runner(
            env, execution.task, execution, train, log_root=None
        )
        actor = runner.alg.actor_critic.actor
        initialize_actor(actor, env.num_obs, target, **gains)
        # 在站立状态范围内检查网络与参考规律等价；旧历史帧随机化，检查只读最新帧。
        samples = torch.rand(512, env.num_obs, device=env.device) * 2 - 1
        samples[256:, -FRAME_SIZE:] *= 0.05
        with torch.no_grad():
            actual = actor(samples).clamp(-1, 1)
            expected = sensor_stand_actions(samples[:, -FRAME_SIZE:], target, **gains)
        error = float((actual - expected).abs().max())
        if error > 2e-5:
            raise RuntimeError("Actor/feedback equivalence error: {}".format(error))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        runner.save(args.output)
        report = dict(
            provenance,
            checkpoint=str(args.output.resolve()),
            max_action_equivalence_error=error,
        )
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
