#!/usr/bin/env python3
"""生成变腿长候选：消除绝对高度偏移，并补偿 CAD 重心平衡姿态的变化。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from legged_gym.envs.wheelleg.model.geometry import (
    ASSET_ROOT, MIN_LENGTH, MAX_LENGTH, require_model_checkpoint,
)
from legged_gym.envs.wheelleg.model.initial_pose import make_initial_pose
from legged_gym.envs.wheelleg.observations.sensors import (
    FRAME_FIELDS, FRAME_SIZE, HISTORY_FRAMES, SENSOR_SCHEMA,
)


def remap_inputs(samples, reference, pitch_slope):
    """仅用于校验代数等价；部署仍输入原始132维历史，无额外运行时处理。"""
    frames = samples.reshape(-1, HISTORY_FRAMES, FRAME_SIZE).clone()
    # 最新帧的指令始终有效；用它作为共同参照，保留8cm启动时的空历史。
    delta = frames[:, -1, 15] - 5 * reference
    frames[:, :, 5:7] -= delta[:, None, None]
    frames[:, :, 15] -= delta[:, None]
    frames[:, :, 0] += pitch_slope / 5 * delta[:, None]
    return frames.reshape_as(samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--leg-range', nargs=2, type=float, default=(.08, .10))
    p.add_argument('--leg-speed', type=float, default=.01)
    p.add_argument('--speed', type=float, default=.7)
    args = p.parse_args()
    low, high = args.leg_range
    if (not all(math.isfinite(v) for v in
                (low, high, args.leg_speed, args.speed))
            or not MIN_LENGTH <= low < high <= MAX_LENGTH
            or min(args.leg_speed, args.speed) <= 0):
        p.error('Use valid CAD height bounds and positive finite speeds/gain')
    if args.output.suffix != '.pt' or args.output.exists():
        p.error('Use a new .pt output path; retain existing candidates')
    state = torch.load(str(args.checkpoint), map_location='cpu')
    require_model_checkpoint(state)
    cfg = state['legged_gym_env_config']
    if (cfg['sensors']['schema'] != SENSOR_SCHEMA
            or cfg['sensors']['frame_fields'] != list(FRAME_FIELDS)):
        p.error('Requires the original132-input sensor layout')
    infos = state.get('infos') or {}
    actor_metadata = state.get('actor_metadata') or {}
    if 'height_normalization' in infos or 'height_normalization' in actor_metadata:
        p.error('Already normalized; start from the source checkpoint')
    if abs(cfg['env']['initial_leg_length'] - low) > 1e-7:
        p.error('The reference height must match the source reset height')
    metadata = json.loads((ASSET_ROOT / 'loop_constraints.json').read_text())
    pitches = [make_initial_pose(
        ASSET_ROOT / metadata['urdf'], metadata, height, balance_initial_com=True
    )['com_balance']['pitch_correction_rad'] for height in (low, high)]
    slope = (pitches[0] - pitches[1]) / (high - low)
    weights = state['model_state_dict']['actor.0.weight']
    bias = state['model_state_dict']['actor.0.bias']
    original, original_bias = weights.clone(), bias.clone()
    if weights.shape[1] != FRAME_SIZE * HISTORY_FRAMES:
        p.error('Unexpected actor input dimension')
    height_column = (HISTORY_FRAMES - 1) * FRAME_SIZE + 15
    context = torch.zeros_like(bias)
    for offset in range(0, weights.shape[1], FRAME_SIZE):
        context += (slope / 5 * original[:, offset]
                    - original[:, offset + 5] - original[:, offset + 6]
                    - original[:, offset + 15])
    weights[:, height_column] += context
    bias -= 5 * low * context
    samples = torch.randn(256, weights.shape[1], dtype=torch.float64,
                          generator=torch.Generator().manual_seed(42))
    samples[:, height_column] = torch.linspace(5 * low, 5 * high, len(samples))
    mapped = remap_inputs(samples, low, slope)
    error = float((samples @ weights.double().T + bias.double()
                   - mapped @ original.double().T - original_bias.double()).abs().max())
    if error > 2e-5:
        raise RuntimeError('Input-remapping equivalence failed: {}'.format(error))
    cfg['commands'].update(forward_speed_range=[-args.speed, args.speed],
                           leg_length_range=[low, high], leg_length_rate=args.leg_speed)
    cfg['keyboard'].update(speed=args.speed, adjust_leg_length=True,
                           leg_range=[low, high], leg_speed=args.leg_speed)
    cfg['control']['standing_height_trim_ki'] = 0.
    state['environment_config'].update(
        forward_speed_range=[-args.speed, args.speed], leg_length_range=[low, high])
    provenance = dict(method='relative_height_and_cad_balance_input_remapping',
        source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        reference_height_m=low, height_range_m=[low, high], com_pitches_rad=pitches,
        pitch_slope_rad_per_m=slope,
        equivalence_error=error, extra_rl_training=False,
        note='Candidate, not an accepted policy. Remapping is folded into actor weights; MCU still supplies original sensor history. For further RL use actor warm-start.')
    infos['height_normalization'] = provenance
    state['infos'] = infos
    state['actor_metadata'] = dict(actor_metadata, height_normalization=provenance)
    if state.get('optimizer_state_dict'):
        state['optimizer_state_dict']['state'] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
