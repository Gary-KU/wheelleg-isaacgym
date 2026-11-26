#!/usr/bin/env python3
"""把最新传感器帧的均值滤波折叠进 actor 权重，不改变观测接口或运行时。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from legged_gym.envs.wheelleg.observations.sensors import (
    FRAME_SIZE, HISTORY_FRAMES, SENSOR_FIELDS, SENSOR_SCHEMA,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--yaw-command-scale', type=float, default=1.0)
    parser.add_argument('--filter-yaw-feedback', dest='keep_yaw_feedback',
                        action='store_false', default=True,
                        help='Experimental: also delay yaw feedback; not recommended for the current policy')
    args = parser.parse_args()
    if not math.isfinite(args.yaw_command_scale) or args.yaw_command_scale <= 0:
        parser.error('yaw-command-scale must be positive and finite')
    if not 2 <= args.frames <= HISTORY_FRAMES:
        parser.error('frames must be between 2 and {}'.format(HISTORY_FRAMES))
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error('Keep the source checkpoint; use a different output path')
    state = torch.load(str(args.checkpoint), map_location='cpu')
    sensors = state['legged_gym_env_config']['sensors']
    if (sensors['schema'] != SENSOR_SCHEMA
            or sensors['history_frames'] != HISTORY_FRAMES
            or sensors['frame_fields'][:len(SENSOR_FIELDS)] != list(SENSOR_FIELDS)):
        parser.error('Checkpoint sensor schema does not match this project')
    if state.get('infos') and state['infos'].get('sensor_history_filter'):
        parser.error('Checkpoint already has this filter; start from the original')
    weights = state['model_state_dict']['actor.0.weight']
    if weights.shape[1] != FRAME_SIZE * HISTORY_FRAMES:
        parser.error('Unexpected actor input size')
    original = weights.clone()
    # 只平滑最新帧的传感器列，指令、上一动作和已有历史通道保持原定义。
    # 因此 MCU 仍输入原始历史帧，不能在部署端再重复施加本滤波。
    width = len(SENSOR_FIELDS)
    latest = (HISTORY_FRAMES - 1) * FRAME_SIZE
    contribution = original[:, latest:latest + width].clone()
    if args.keep_yaw_feedback:
        contribution[:, SENSOR_FIELDS.index('yaw_rate')] = 0
        # 两轮的共同分量用于前后平衡；差动分量用于转向，保留其原有延迟。
        left = SENSOR_FIELDS.index('wheel_forward_rate_left')
        right = SENSOR_FIELDS.index('wheel_forward_rate_right')
        common = (contribution[:, left] + contribution[:, right]) / 2
        contribution[:, left] = common
        contribution[:, right] = common
    weights[:, latest:latest + width] -= contribution
    for age in range(args.frames):
        offset = latest - age * FRAME_SIZE
        weights[:, offset:offset + width] += contribution / args.frames
    # 检查权重变换等价于在网络入口替换最新帧，防止历史索引/叠加出错。
    generator = torch.Generator().manual_seed(42)
    samples = torch.randn(128, weights.shape[1], generator=generator)
    samples[:32] = samples[:32, :FRAME_SIZE].repeat(1, HISTORY_FRAMES)
    frames = samples.view(-1, HISTORY_FRAMES, FRAME_SIZE)
    virtual = samples.clone().view_as(frames)
    virtual[:, -1, :width] = frames[:, -args.frames:, :width].mean(1)
    if args.keep_yaw_feedback:
        yaw = SENSOR_FIELDS.index('yaw_rate')
        virtual[:, -1, yaw] = frames[:, -1, yaw]
        delta = (frames[:, -1, left] + frames[:, -1, right] -
                 virtual[:, -1, left] - virtual[:, -1, right]) / 2
        # 只改变共同速度分量；左右差速直接透传。
        virtual[:, -1, left] = frames[:, -1, left] - delta
        virtual[:, -1, right] = frames[:, -1, right] - delta
    error = float((samples @ weights.T - virtual.reshape_as(samples) @ original.T).abs().max())
    if error > 2e-5:
        raise RuntimeError('History filter equivalence check failed: {}'.format(error))
    # 可选的小幅转向参考校准；零指令时严格不改变输出。
    command_yaw = sensors['frame_fields'].index('command_yaw_rate')
    for offset in range(0, weights.shape[1], FRAME_SIZE):
        weights[:, offset + command_yaw] *= args.yaw_command_scale
    cfg = state['legged_gym_env_config']
    dt = cfg['sim']['dt'] * cfg['control']['decimation']
    provenance = dict(
        method='latest_sensor_frame_mean_folded_into_actor',
        frames=args.frames,
        keep_yaw_feedback=args.keep_yaw_feedback,
        yaw_command_scale=args.yaw_command_scale,
        added_group_delay_seconds=(args.frames - 1) * dt / 2,
        source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        equivalence_error=error,
        note='Candidate requires standing and driving validation. Raw sensor history input unchanged. For further training use actor warm-start, not optimizer resume.',
    )
    if state.get('infos') is None:
        state['infos'] = {}
    state['infos']['sensor_history_filter'] = provenance
    # 旧策略的 Adam 动量不再对应变换后的权重，避免后续误续训沿用。
    if state.get('optimizer_state_dict'):
        state['optimizer_state_dict']['state'] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
