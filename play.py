#!/usr/bin/env python3
"""Play a checkpoint or explicitly run a zero-action mechanism check."""

from pathlib import Path
import math
import sys
import time

from legged_gym.compat.cli import common_parser, environment_kwargs, validate_common
from legged_gym.compat.runtime_config import explicit_argument_destinations, resolve_runtime_args
from legged_gym.envs.wheelleg.constants import MIN_LEG_LENGTH, MAX_LEG_LENGTH


def main():
    parser = common_parser("WheelLeg Isaac Gym playback (compliant loop approximation)", 1)
    # Tiny single-robot CPU workloads avoid GPU launch overhead at 2 kHz.
    parser.set_defaults(sim_device="cpu")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path, help="Path to model_N.pt")
    policy.add_argument("--zero-actions", action="store_true", help="Mechanism check without a trained policy")
    parser.add_argument("--num-steps", "--num_steps", type=int, default=1000)
    parser.add_argument("--fix-base", action="store_true", help="Hold the chassis fixed for mechanism inspection")
    parser.add_argument("--keyboard", action="store_true", help="Hold arrow keys to command the policy; space stops, R resets")
    parser.add_argument("--keyboard-speed", type=float, default=0.3, help="Forward/backward target speed in m/s")
    parser.add_argument("--keyboard-turn-speed", type=float, default=0.8, help="Left/right target yaw speed in rad/s")
    parser.add_argument("--keyboard-leg-speed", type=float, default=0.02,
                        help="Hold Q/E to shorten/extend the leg target at this many m/s")
    parser.add_argument("--keyboard-leg-range", nargs=2, type=float,
                        default=[MIN_LEG_LENGTH, MAX_LEG_LENGTH], metavar=("MIN", "MAX"),
                        help="Clamp keyboard leg targets in metres; supported range 0.06 to 0.20")
    parser.add_argument("--show-collisions", action="store_true", help="Draw collision meshes for inspection instead of only visual meshes")
    parser.add_argument("--viewer-width", type=int, help="Viewer width in pixels; stand.sh uses 1280")
    parser.add_argument("--viewer-height", type=int, help="Viewer height in pixels; stand.sh uses 720")
    parser.add_argument("--event-log", type=Path,
                        help="Append command changes, physical failures and timeouts as JSONL events")
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist: {}".format(args.checkpoint))
    state = None
    if args.checkpoint:
        import isaacgym  # noqa: F401; binary extension must precede Torch.
        import torch
        state = torch.load(str(args.checkpoint), map_location="cpu")
    try:
        args = resolve_runtime_args(args, explicit_argument_destinations(parser, argv), state, "play")
    except ValueError as error:
        parser.error(str(error))
    validate_common(parser, args)
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    if any(value is not None and value < 1 for value in (args.viewer_width, args.viewer_height)):
        parser.error("Viewer width and height must be positive")
    if args.keyboard:
        if args.headless or args.fix_base or args.zero_actions or args.random_commands or args.num_envs != 1:
            parser.error("--keyboard requires a visible window, a checkpoint, one environment, and a free base; omit --headless, --fix-base, --zero-actions and --random-commands")
        if not all(math.isfinite(x) and x > 0 for x in (args.keyboard_speed, args.keyboard_turn_speed)):
            parser.error("Keyboard speeds must be positive and finite")
        low, high = args.keyboard_leg_range
        if (not all(math.isfinite(x) for x in (low, high))
                or not MIN_LEG_LENGTH <= low < high <= MAX_LEG_LENGTH
                or not low <= args.command[3] <= high):
            parser.error("Keyboard leg range must contain the target leg length and stay within [{},{}] m".format(MIN_LEG_LENGTH, MAX_LEG_LENGTH))
        if not math.isfinite(args.keyboard_leg_speed) or args.keyboard_leg_speed <= 0:
            parser.error("--keyboard-leg-speed must be positive and finite")

    import isaacgym  # noqa: F401; must precede torch
    import torch
    from legged_gym.envs.wheelleg.wheelleg_robot import WheelLegRobot
    from legged_gym.envs.wheelleg.constants import TRAIN_CONFIG
    from legged_gym.learning.policy import make_policy, scalar, viewer_is_closed
    from legged_gym.envs.wheelleg.keyboard import KeyboardDrive
    from legged_gym.envs.wheelleg.commands import CommandGovernor
    from legged_gym.envs.wheelleg.diagnostics.playback_log import PlaybackEventLog, reset_diagnostics
    from isaacgym import gymapi

    kwargs = environment_kwargs(args)
    kwargs["fix_base"] = args.fix_base
    kwargs["viewer_show_collisions"] = args.show_collisions
    kwargs["viewer_width"], kwargs["viewer_height"] = args.viewer_width, args.viewer_height
    if args.keyboard:
        kwargs["command"] = (0.0, 0.0, 0.0, args.command[3])
    print("Loop closure: compliant external forces; monitor anchor errors in millimetres.")
    env = WheelLegRobot(**kwargs)
    event_log = None
    try:
        event_log = PlaybackEventLog(args.event_log)
        if args.keyboard:
            # Keep manual driving continuous; physical failures and R still reset.
            env.max_episode_length = args.num_steps + 1
        controller = None
        if args.checkpoint:
            config = state.get("training_config", {}).get("policy", TRAIN_CONFIG["policy"])
            controller = make_policy(env.num_obs, env.num_actions, env.device, config)
            controller.load_state_dict(state["model_state_dict"], strict=True)
            controller.eval()
            print("Loaded policy: {}".format(args.checkpoint.resolve()))
        else:
            print("Zero actions: mechanism inspection only; this is not a trained balancing policy.")
        observations, _ = env.reset()
        motion_origin_xy = env.root_states[0, :2].clone()
        governor = CommandGovernor(env.commands.clone(),
                                   linear_acceleration=args.linear_acceleration,
                                   yaw_acceleration=args.yaw_acceleration)
        keyboard = None
        if args.keyboard:
            keyboard = KeyboardDrive(args.keyboard_speed, args.keyboard_turn_speed,
                                     leg_length=args.command[3], leg_speed=args.keyboard_leg_speed,
                                     leg_range=args.keyboard_leg_range)
            keyboard.bind(env.gym, env.viewer, gymapi)
            print("Keyboard: hold UP/DOWN to drive, LEFT/RIGHT to turn, Q/E to shorten/extend; SPACE stops, R resets.")
            print("Motion targets: forward/backward +/-{:.2f}m/s, yaw +/-{:.2f}rad/s.".format(
                args.keyboard_speed, args.keyboard_turn_speed))
            print("Leg target: {:.1f}cm, range {:.1f}..{:.1f}cm, rate {:.1f}cm/s; release Q/E to hold target.".format(
                100*args.command[3], 100*args.keyboard_leg_range[0], 100*args.keyboard_leg_range[1], 100*args.keyboard_leg_speed))
            print("Keys send velocity targets to the loaded policy. Successful driving depends on policy training.")
        max_error = 0.0
        resets = 0
        physical_failures = 0
        timeouts = 0
        manual_resets = 0
        completed_steps = 0
        previous_command = None
        active_command = None
        event_log.emit("playback_started", step=0, sim_time_s=0.0,
                       options=vars(args), control_dt_s=env.dt,
                       working_directory=Path.cwd(),
                       max_episode_length_steps=env.max_episode_length)
        started = interval_started = time.perf_counter()
        interval_step = 0
        with torch.inference_mode():
            for step in range(args.num_steps):
                if viewer_is_closed(env):
                    print("Viewer closed; ending playback.")
                    break
                if keyboard is not None:
                    env.gym.poll_viewer_events(env.viewer)
                    if viewer_is_closed(env):
                        break
                    if keyboard.process_events(env.gym.query_viewer_action_events(env.viewer)):
                        observations, _ = env.reset()
                        motion_origin_xy.copy_(env.root_states[0, :2])
                        governor.reset(env.commands.clone())
                        keyboard.reset_leg_length(args.command[3])
                        manual_resets += 1
                        event_log.emit("manual_reset", step=step, sim_time_s=step*env.dt,
                                       active_command=keyboard.command())
                    keyboard.advance_leg_length(env.dt)
                    raw_command = keyboard.command()
                    applied_command = governor.step([raw_command], env.dt)
                    observations = env.set_commands(applied_command)
                    active_command = applied_command[0].cpu().tolist()
                    if raw_command != previous_command:
                        event_log.emit("command_change", step=step, sim_time_s=step*env.dt,
                                       active_command=active_command,
                                       raw_command=raw_command,
                                       applied_command=active_command,
                                       pressed_actions=sorted(keyboard.pressed))
                        previous_command = list(raw_command)
                actions = (
                    controller.act_inference(observations) if controller is not None
                    else torch.zeros(env.num_envs, env.num_actions, device=env.device)
                )
                observations, _, rewards, dones, infos = env.step(actions)
                completed_steps = step + 1
                reset_indices = dones.nonzero(as_tuple=False).flatten().cpu().tolist()
                resets += len(reset_indices)
                if 0 in reset_indices:
                    motion_origin_xy.copy_(env.root_states[0, :2])
                for index in reset_indices:
                    timeout_flags = infos.get("time_outs")
                    is_timeout = timeout_flags is not None and bool(timeout_flags[index].item())
                    is_success = bool(infos.get("terrain_success", torch.zeros_like(dones))[index])
                    physical_failures += int(not is_timeout and not is_success)
                    timeouts += int(is_timeout)
                    event_log.emit("terrain_success" if is_success else "time_limit" if is_timeout else "physical_failure",
                                   step=completed_steps, sim_time_s=completed_steps*env.dt,
                                   **reset_diagnostics(infos, index, active_command))
                if keyboard is not None and reset_indices:
                    governor.reset(env.commands.clone())
                    keyboard.reset_leg_length(args.command[3])
                if not torch.isfinite(observations).all() or not torch.isfinite(rewards).all():
                    raise RuntimeError("Non-finite observation/reward during playback")
                if "loop_error_max" in infos:
                    max_error = max(max_error, scalar(infos["loop_error_max"]))
                if (step + 1) % 100 == 0 or step + 1 == args.num_steps:
                    now = time.perf_counter()
                    steps_per_second = (completed_steps-interval_step)/max(now-interval_started, 1e-9)
                    print("steps={} loop_max={:.3f}mm resets={} physical_failures={} timeouts={} manual_resets={}".format(
                        step + 1, max_error * 1000.0, resets,
                        physical_failures, timeouts, manual_resets), flush=True)
                    print("{}={:.1f} simulation_speed={:.2f}x real time".format(
                        "viewer_fps" if env.viewer is not None else "steps_per_second",
                        steps_per_second, steps_per_second*env.dt), flush=True)
                    displacement_xy_cm = 100*(env.root_states[0, :2]-motion_origin_xy)
                    print("env0 displacement since reset world X/Y={:+.2f}/{:+.2f}cm; "
                          "body vx/vy={:+.4f}/{:+.4f}m/s".format(
                              *displacement_xy_cm.cpu().tolist(),
                              *env.lin_vel[0, :2].cpu().tolist()), flush=True)
                    if args.event_log is not None:
                        event_log.emit("motion_sample", step=completed_steps, sim_time_s=completed_steps*env.dt,
                                       environment_index=0, position_origin_world_xy_m=motion_origin_xy,
                                       displacement_world_xy_m=env.root_states[0, :2]-motion_origin_xy,
                                       base_linear_velocity_body_m_per_s=env.lin_vel[0],
                                       yaw_rad=env.yaw[0], pitch_rad=env.pitch[0], roll_rad=env.roll[0],
                                       command=env.commands[0], physical_failure_resets=physical_failures)
                    interval_started, interval_step = now, completed_steps
                    if keyboard is not None:
                        print("Leg target={:.1f}cm measured L/R={:.1f}/{:.1f}cm".format(
                            100*keyboard.leg_length, 100*env.length_l[0].item(), 100*env.length_r[0].item()), flush=True)
        print("Playback completed: {} steps".format(completed_steps))
        event_log.emit("playback_finished", step=completed_steps,
                       sim_time_s=completed_steps*env.dt, physical_failures=physical_failures,
                       timeouts=timeouts, manual_resets=manual_resets,
                       wall_seconds=time.perf_counter()-started)
    finally:
        env.close()
        if event_log is not None:
            event_log.close()


if __name__ == "__main__":
    main()
