#!/usr/bin/env python3
"""Registered task playback; arrows drive and Q/E adjust leg length."""
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from legged_gym.envs import register_tasks

register_tasks()
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wheelleg.control.keyboard import KeyboardDrive
from legged_gym.envs.wheelleg.constants import MIN_LEG_LENGTH, MAX_LEG_LENGTH
from legged_gym.envs.wheelleg.diagnostics.motion_evaluation import MotionEvaluation


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
        if getattr(env, "terrain_model", None) is None and hasattr(env, "set_commands"):
            env.max_episode_length = args.num_steps + 1
        if not args.task.startswith("wheelleg"):
            with torch.inference_mode():
                for _ in range(args.num_steps):
                    obs, _ = env.get_observations()
                    env.step(policy(obs))
            print("Playback completed", flush=True)
            return
        keyboard = None
        last_printed_length = None
        keyboard_cfg = getattr(env_cfg, "keyboard", None)
        adjust_leg_length = getattr(keyboard_cfg, "adjust_leg_length", True)
        if args.keyboard:
            keyboard = KeyboardDrive(
                speed=getattr(keyboard_cfg, "speed", 0.6),
                turn_speed=getattr(keyboard_cfg, "turn_speed", 2.0),
                leg_length=env.commands[0, 3].item(),
                leg_range=getattr(keyboard_cfg, "leg_range", (MIN_LEG_LENGTH, MAX_LEG_LENGTH)),
                leg_speed=getattr(keyboard_cfg, "leg_speed", 0.02),
            )
            keyboard.bind(env.gym, env.viewer, isaacgym.gymapi)
            last_printed_length = keyboard.leg_length
            print(
                "点击仿真窗口：方向键前后/转向，空格停止，R重置。"
                + ("Q缩腿/E伸腿（{:.0f}–{:.0f} cm）。".format(
                    keyboard.leg_range[0]*100, keyboard.leg_range[1]*100)
                   if adjust_leg_length else "本权重固定8 cm腿长。"),
                flush=True,
            )
        pitch = roll = loop = 0.0
        physical_failures = timeouts = successes = completed = 0
        motion = MotionEvaluation(env.root_states, env.dt)
        initial_render_count = env.render_count
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
                    if adjust_leg_length:
                        keyboard.advance_leg_length(env.dt)
                    env.set_commands(keyboard.command())
                obs, _ = env.get_observations()
                action = policy(obs.to(args.rl_device)).to(env.device)
                obs, _, reward, done, info, _ = env.step(action)
                if (keyboard is not None and adjust_leg_length
                        and abs(keyboard.leg_length - last_printed_length) >= 0.002 - 1e-7):
                    left, right = info["pre_reset_leg_length"][0].tolist()
                    print("腿长目标 {:.1f} cm，实际左/右 {:.1f}/{:.1f} cm".format(
                        keyboard.leg_length * 100, left * 100, right * 100), flush=True)
                    last_printed_length = keyboard.leg_length
                if not bool(torch.isfinite(obs).all() and torch.isfinite(reward).all()):
                    raise RuntimeError("Non-finite playback")
                motion.update(
                    info["pre_reset_root_states"],
                    info["pre_reset_commands"],
                    done,
                    env.root_states,
                )
                completed = step + 1
                physical_failures += int(info["physical_failures"].sum())
                timeouts += int(info["time_outs"].sum())
                successes += int(
                    info.get("terrain_success", torch.zeros_like(done)).sum()
                )
                pitch = max(pitch, float(info["pre_reset_pitch"].abs().max()))
                roll = max(roll, float(info["pre_reset_roll"].abs().max()))
                loop = max(loop, float(info["loop_error_max"]))
        wall_seconds = time.perf_counter() - started
        rendered_frames = env.render_count - initial_render_count
        report = dict(
            motion=motion.report(),
            task=args.task,
            checkpoint=str(args.checkpoint_path),
            completed_steps=completed,
            environment_seconds=completed * env.dt,
            wall_seconds=wall_seconds,
            rendered_frames=rendered_frames,
            measured_viewer_fps=(rendered_frames / max(wall_seconds, 1e-9)
                                 if env.viewer is not None else None),
            simulation_speed_ratio=completed * env.dt / max(wall_seconds, 1e-9),
            viewer_wall_clock=env.viewer_wall_clock,
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
