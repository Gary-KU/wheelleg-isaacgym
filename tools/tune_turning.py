#!/usr/bin/env python3
"""生成待验收的转向候选：校准网络 yaw 反馈，并配置左右腿差动阻尼。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from legged_gym.envs.wheelleg.observations.sensors import (
    SENSOR_SCHEMA, SENSOR_FIELDS, SENSOR_SCALES, COMMAND_SCALES,
    FRAME_FIELDS, FRAME_SIZE, HISTORY_FRAMES,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feedback-gain', type=float, default=2.0)
    p.add_argument('--turn-speed', type=float, default=1.0)
    p.add_argument('--differential-kd', type=float, default=0.08)
    args = p.parse_args()
    if args.output.suffix != '.pt':
        p.error('Use a .pt output checkpoint; .json is reserved for provenance')
    if (not all(math.isfinite(x) for x in
                (args.feedback_gain, args.turn_speed, args.differential_kd))
            or min(args.feedback_gain, args.turn_speed) <= 0 or args.differential_kd < 0):
        p.error('Require finite positive feedback gain/turn speed and nonnegative damping')
    if args.output.resolve() == args.checkpoint.resolve():
        p.error('Keep the source checkpoint; use a different output path')
    state = torch.load(str(args.checkpoint), map_location='cpu')
    cfg = state['legged_gym_env_config']
    sensors = cfg['sensors']
    if (sensors['schema'] != SENSOR_SCHEMA
            or sensors['history_frames'] != HISTORY_FRAMES
            or sensors['frame_fields'] != list(FRAME_FIELDS)):
        p.error('Checkpoint sensor schema does not match this project')
    infos = state.get('infos') or {}
    if 'turning_tuning' in infos:
        p.error('Already tuned; start from the original checkpoint')
    weights = state['model_state_dict']['actor.0.weight']
    if weights.shape[1] != FRAME_SIZE * HISTORY_FRAMES:
        p.error('Unexpected actor input size')
    original = weights.clone()
    yaw = FRAME_FIELDS.index('yaw_rate')
    command = FRAME_FIELDS.index('command_yaw_rate')
    ratio = SENSOR_SCALES[SENSOR_FIELDS.index('yaw_rate')] / COMMAND_SCALES[1]
    # y'=g*y+(1-g)*reference：调整 yaw 误差反馈，保留零误差参考。
    # 折叠进第一层；运行时仍使用原始传感器历史，不额外添加滤波/控制器。
    for offset in range(0, weights.shape[1], FRAME_SIZE):
        column = original[:, offset + yaw]
        weights[:, offset + yaw] *= args.feedback_gain
        weights[:, offset + command] += (1-args.feedback_gain) * ratio * column
    # 用 float64 验证代数变换，避免无关大权重的 float32 求和误差掩盖结果。
    samples = torch.randn(256, weights.shape[1], dtype=torch.float64,
                          generator=torch.Generator().manual_seed(42))
    virtual = samples.clone()
    for offset in range(0, weights.shape[1], FRAME_SIZE):
        virtual[:, offset + yaw] = (args.feedback_gain * samples[:, offset + yaw]
                                   + (1-args.feedback_gain) * ratio * samples[:, offset + command])
    error = float((samples @ weights.double().T - virtual @ original.double().T).abs().max())
    if error > 2e-5:
        raise RuntimeError('Yaw input remapping equivalence failed: {}'.format(error))
    cfg['commands']['yaw_speed_range'] = [-args.turn_speed, args.turn_speed]
    cfg['keyboard']['turn_speed'] = args.turn_speed
    cfg['control']['leg_differential_kd'] = args.differential_kd
    state['environment_config'].update(
        yaw_speed_range=[-args.turn_speed, args.turn_speed],
        leg_differential_kd=args.differential_kd)
    provenance = dict(
        method='yaw_error_input_remapping_and_differential_hip_damping',
        source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        feedback_gain=args.feedback_gain, turn_speed_rad_s=args.turn_speed,
        leg_differential_kd=args.differential_kd, equivalence_error=error,
        extra_training=False,
        note='Candidate requires driving/standing validation. MCU must also apply differential damping before motor limits. Further learning: actor warm-start, not optimizer resume.',
    )
    infos['turning_tuning'] = provenance
    state['infos'] = infos
    if state.get('optimizer_state_dict'):
        state['optimizer_state_dict']['state'] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
