#!/usr/bin/env python3
"""复制驾驶检查点并配置速度/伸缩腿；不训练、不修改网络或覆盖原权重。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from legged_gym.envs.wheelleg.constants import MIN_LEG_LENGTH, MAX_LEG_LENGTH
from legged_gym.envs.wheelleg.model.geometry import require_model_checkpoint
from legged_gym.envs.wheelleg.observations.sensors import SENSOR_SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speed", type=float, required=True, help="Forward/backward cap, m/s")
    p.add_argument("--leg-range", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    p.add_argument("--leg-speed", type=float, default=0.01, help="Q/E rate, m/s")
    p.add_argument("--standing-trim-ki", type=float, default=0.4, help="Stationary integral gain; 0 disables it")
    p.add_argument("--standing-trim-slope", "--height-speed-slope", type=float, default=0.0,
                   help="Calibrated initial speed correction per metre of extension (s^-1)")
    p.add_argument("--encoder-hold", action="store_true",
                   help="Use wheel-angle odometry for stop/straight heading feedback instead of rate integration")
    p.add_argument("--hold-position-kp", type=float, default=.5)
    p.add_argument("--hold-heading-kp", type=float, default=1.)
    p.add_argument("--linear-acceleration", type=float,
                   help="Optional command ramp acceleration m/s^2; retains the speed cap")
    args = p.parse_args()
    if args.output.suffix != ".pt" or args.output.resolve() == args.checkpoint.resolve():
        p.error("Save to a different .pt path; keep the original checkpoint")
    values = (args.speed, args.leg_speed, args.standing_trim_ki, args.standing_trim_slope,
              args.hold_position_kp, args.hold_heading_kp, *args.leg_range)
    if (not all(math.isfinite(x) for x in values)
            or min(args.speed, args.leg_speed, args.hold_position_kp, args.hold_heading_kp) <= 0
            or args.standing_trim_ki < 0
            or not MIN_LEG_LENGTH <= args.leg_range[0] < args.leg_range[1] <= MAX_LEG_LENGTH):
        p.error("Use positive finite speeds, nonnegative gain and a leg range within the CAD limits")
    if abs(args.standing_trim_slope) * (args.leg_range[1]-args.leg_range[0]) > .08:
        p.error("standing-trim-slope exceeds the0.08m/s correction limit")
    if args.standing_trim_slope and not args.standing_trim_ki and not args.encoder_hold:
        p.error("A calibrated trim slope requires standing-trim-ki >0")
    if args.linear_acceleration is not None and (
            not math.isfinite(args.linear_acceleration) or args.linear_acceleration <= 0):
        p.error("linear-acceleration must be finite and positive")
    state = torch.load(str(args.checkpoint), map_location="cpu")
    require_model_checkpoint(state)
    cfg = state["legged_gym_env_config"]
    if cfg.get("sensors", {}).get("schema") != SENSOR_SCHEMA or "keyboard" not in cfg:
        p.error("Use a wheelleg_sensor_drive checkpoint")
    # 修正的零点是原来的站立高度；修改下限会改变控制契约，因此必须匹配。
    initial_length = cfg["commands"]["command"][3]
    if abs(args.leg_range[0] - initial_length) > 1e-7:
        p.error("The lower length must match the checkpoint's initial standing length")
    cfg["keyboard"].update(
        speed=args.speed, adjust_leg_length=True,
        leg_range=args.leg_range, leg_speed=args.leg_speed)
    cfg["commands"].update(
        forward_speed_range=[-args.speed, args.speed], leg_length_range=args.leg_range,
        # 键盘、API目标和训练/验收共用同一伸腿斜坡；不能只改键盘显示档位。
        leg_length_rate=args.leg_speed)
    cfg["control"].update(
        standing_height_trim_ki=0.0 if args.encoder_hold else args.standing_trim_ki,
        standing_height_trim_limit=0.08,
        standing_height_trim_initial_coefficient=args.standing_trim_slope,
        encoder_hold=args.encoder_hold,
        encoder_hold_position_kp=args.hold_position_kp,
        encoder_hold_heading_kp=args.hold_heading_kp,
        encoder_hold_speed_limit=.08, encoder_hold_yaw_limit=.2,
        encoder_hold_height_slope=args.standing_trim_slope)
    if args.encoder_hold:
        # 显式保存假设的角度测量误差，避免以后类默认值改变旧候选的行为。
        cfg['sensors'].setdefault('wheel_angle_noise_std', .001)
        cfg['sensors'].setdefault('wheel_angle_bias_std', .01)
    # environment_config 仅存底层构造参数；修正器属于任务，不传入物理构造器。
    state["environment_config"].update(
        forward_speed_range=[-args.speed, args.speed], leg_length_range=args.leg_range)
    if args.linear_acceleration is not None:
        cfg['commands']['linear_acceleration'] = args.linear_acceleration
        state['environment_config']['linear_acceleration'] = args.linear_acceleration
    provenance = dict(
        source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        speed_m_s=args.speed, leg_range_m=args.leg_range,
        leg_speed_m_s=args.leg_speed,
        standing_trim_ki=cfg['control']['standing_height_trim_ki'],
        standing_trim_initial_coefficient_s_inv=args.standing_trim_slope,
        encoder_position_hold=args.encoder_hold,
        encoder_hold_position_kp=args.hold_position_kp,
        encoder_hold_heading_kp=args.hold_heading_kp,
        height_speed_slope_s_inv=args.standing_trim_slope,
        linear_acceleration_m_s2=cfg['commands']['linear_acceleration'],
        network_unchanged=True, extra_training=False,
        status="candidate_requires_evaluation",
        note="MCU must port the command ramp, hip control/limits and enabled standing trim as well as the network.",
    )
    infos = state.get("infos") or {}
    infos["driving_profile"] = provenance
    state["infos"] = infos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
