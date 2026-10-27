#!/usr/bin/env python3
"""Registered task playback; arrows drive and Q/E adjust leg length."""
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils import get_args, task_registry
from wheelleg_gym.keyboard import KeyboardDrive


def main():
    args = get_args("play")
    torch.set_num_threads(1)
    env_cfg, train_cfg = task_registry.prepare_cfgs(args.task, args)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(
            env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        policy = runner.get_inference_policy(device=args.rl_device)
        if env.terrain_model is None:
            env.max_episode_length = args.num_steps + 1
        keyboard = None
        if args.keyboard:
            keyboard = KeyboardDrive(
                speed=0.6,
                turn_speed=2.0,
                leg_length=env.commands[0, 3].item(),
                leg_range=(0.06, 0.20),
            )
            keyboard.bind(env.gym, env.viewer, isaacgym.gymapi)
            print(
                "点击仿真窗口：方向键前后/转向，Q/E调腿长，空格停止，R重置。",
                flush=True,
            )
        pitch = roll = loop = 0.0
        physical_failures = timeouts = successes = completed = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for step in range(args.num_steps):
                if env.viewer is not None and env.gym.query_viewer_has_closed(
                    env.viewer
                ):
                    break
                if keyboard is not None:
                    if keyboard.process_events(
                        env.gym.query_viewer_action_events(env.viewer)
                    ):
                        env.reset()
                        keyboard.reset_leg_length(env.cfg.commands.command[3])
                    keyboard.advance_leg_length(env.dt)
                    env.set_commands(keyboard.command())
                obs, _ = env.get_observations()
                action = policy(obs.to(args.rl_device)).to(env.device)
                obs, _, reward, done, info, _ = env.step(action)
                if not bool(torch.isfinite(obs).all() and torch.isfinite(reward).all()):
                    raise RuntimeError("Non-finite playback")
                completed = step + 1
                physical_failures += int(info["physical_failures"].sum())
                timeouts += int(info["time_outs"].sum())
                successes += int(
                    info.get("terrain_success", torch.zeros_like(done)).sum()
                )
                pitch = max(pitch, float(info["pre_reset_pitch"].abs().max()))
                roll = max(roll, float(info["pre_reset_roll"].abs().max()))
                loop = max(loop, float(info["loop_error_max"]))
        report = dict(
            task=args.task,
            checkpoint=str(args.checkpoint_path),
            completed_steps=completed,
            environment_seconds=completed * env.dt,
            wall_seconds=time.perf_counter() - started,
            num_envs=env.num_envs,
            physical_failures=physical_failures,
            timeouts=timeouts,
            terrain_successes=successes,
            pitch_peak_rad=pitch,
            roll_peak_rad=roll,
            closure_peak_m=loop,
            observation_size=env.num_obs,
            keyboard_enabled=args.keyboard,
            loop_model="external_force_compliant_approximation",
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
