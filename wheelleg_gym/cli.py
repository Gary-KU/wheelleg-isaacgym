"""Argument parsing stays independent of Isaac Gym and PyTorch import order."""

import argparse
import math

from .config import (
    DEFAULT_CLOSURE_DAMPING, DEFAULT_CLOSURE_STIFFNESS, DEFAULT_COMMAND,
    DEFAULT_CONTROL_DT, DEFAULT_FORWARD_SPEED_RANGE, DEFAULT_LEG_LENGTH_RANGE,
    DEFAULT_PHYSICS_DT, DEFAULT_YAW_SPEED_RANGE, MAX_LEG_LENGTH, MIN_LEG_LENGTH,
)


def common_parser(description, num_envs):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--profile", choices=("stability", "support", "hybrid", "stairs"),
                        help="stability: motion; support: pure-RL stance; hybrid: hip PD plus RL residual")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=num_envs)
    parser.add_argument("--sim-device", "--sim_device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leg-control", choices=("torque", "hip_pd", "hip_target"), default="torque",
                        help="Hip action meaning: total torque, or residual torque added to leg-position PD")
    parser.add_argument("--leg-kp", type=float, default=80.0, help="Hip PD proportional gain in Nm/rad")
    parser.add_argument("--leg-kd", type=float, default=4.0, help="Hip PD damping gain in Nm s/rad")
    parser.add_argument("--hip-target-scale", type=float, default=0.2,
                        help="In hip_target mode, each clipped hip action offsets the target by this many radians")
    parser.add_argument("--terrain", choices=("plane", "stairs"), default="plane")
    parser.add_argument("--observe-terrain", action="store_true", help="Append nine ahead-of-robot terrain height samples per observation frame")
    parser.add_argument("--no-observe-terrain", dest="observe_terrain", action="store_false")
    parser.add_argument("--stair-level", type=int, default=0, help="Initial/fixed stair difficulty, 0..4")
    parser.add_argument("--stair-min-level", type=int, default=0, help="Lowest automatic curriculum level, 0..4; failures never demote below it")
    parser.add_argument("--stair-max-level", type=int, default=4, help="Highest automatic curriculum level, 0..4")
    parser.add_argument("--stair-episode-seconds", type=float, default=24., help="Stair task episode duration; flat task remains 16s")
    parser.add_argument("--stair-curriculum", action="store_true", help="Promote only on successful course completion; demote on early physical failure")
    parser.add_argument("--no-stair-curriculum", dest="stair_curriculum", action="store_false")
    parser.add_argument("--leg-feedforward-force", type=float, default=0.0,
                        help="Approximate radial support force per leg in N, mapped through the five-bar Jacobian; requires hip_pd")
    parser.add_argument("--heading-kp", type=float, default=0.0,
                        help="Optional wheel heading feedback in Nm/rad; zero disables proportional feedback")
    parser.add_argument("--heading-kd", type=float, default=0.0,
                        help="Optional wheel yaw-rate damping in Nm s/rad; feedback is limited to 1Nm per wheel")
    parser.add_argument("--wheel-control", choices=("policy", "heading"), default="policy",
                        help="heading: retain learned forward balance effort and use explicit steering feedback")
    parser.add_argument("--wheel-pitch-damping", type=float, default=0.,
                        help="Additional pitch-rate damping in Nm s/rad in heading wheel mode")
    parser.add_argument("--wheel-balance-scale", type=float, default=1.,
                        help="Scale learned wheel balance differential in heading wheel mode; steering is independent")
    parser.add_argument("--physics-solver", choices=("tgs", "pgs"), default="tgs",
                        help="PhysX solver; tgs retains the original training physics")
    parser.add_argument("--wheel-friction", type=float, default=1.,
                        help="Tire contact friction coefficient only; original value 1.0")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--initial-leg-length", type=float,
                         help="Optional consistent initial leg pose in metres; changes initialization, not the controller")
    initial.add_argument("--default-leg-pose", dest="initial_leg_length", action="store_const", const=None,
                         help="Use the original model zero-joint initial pose, overriding a checkpoint")
    parser.add_argument(
        "--allow-approximate-loop", action="store_true",
        help="Acknowledge compliant external-force closure, not exact rigid loop joints",
    )
    parser.add_argument("--closure-stiffness", type=float, default=DEFAULT_CLOSURE_STIFFNESS, help="N/m")
    parser.add_argument("--closure-damping", type=float, default=DEFAULT_CLOSURE_DAMPING, help="N s/m")
    parser.add_argument("--physics-dt", type=float, default=DEFAULT_PHYSICS_DT, help="Physics timestep in seconds; control period is 0.02 s")
    parser.add_argument("--physics-threads", type=int, default=4,
                        help="PhysX CPU worker threads; 0 uses the calling thread and is faster for one robot")
    parser.add_argument(
        "--command", nargs=4, type=float, metavar=("VX", "VY", "WZ", "LENGTH"),
        default=list(DEFAULT_COMMAND),
        help="Fixed command: forward/lateral m/s, yaw rad/s, leg length m",
    )
    parser.add_argument(
        "--random-commands", action="store_true",
        help="Resample commands independently at episode resets instead of using a fixed command",
    )
    parser.add_argument("--no-random-commands", dest="random_commands", action="store_false",
                        help="Use fixed commands, including when overriding a checkpoint or profile")
    parser.add_argument("--forward-speed-range", nargs=2, type=float, default=list(DEFAULT_FORWARD_SPEED_RANGE), metavar=("MIN", "MAX"))
    parser.add_argument("--yaw-speed-range", nargs=2, type=float, default=list(DEFAULT_YAW_SPEED_RANGE), metavar=("MIN", "MAX"))
    parser.add_argument("--leg-length-range", nargs=2, type=float, default=list(DEFAULT_LEG_LENGTH_RANGE), metavar=("MIN", "MAX"))
    parser.add_argument("--tracking-penalties", action="store_true", help="Add dense linear-velocity and leg-length error penalties for command tracking")
    parser.add_argument("--stability-rewards", action="store_true", help="Add posture, angular damping, and applied-torque smoothness terms")
    parser.add_argument("--support-rewards", action="store_true", help="Replace permissive leg rewards with target-length, left/right and passive-limit terms")
    parser.add_argument("--observe-leg-lengths", action="store_true", help="Append measured left/right leg lengths to every observation frame")
    parser.add_argument("--no-tracking-penalties", dest="tracking_penalties", action="store_false")
    parser.add_argument("--no-stability-rewards", dest="stability_rewards", action="store_false")
    parser.add_argument("--no-support-rewards", dest="support_rewards", action="store_false")
    parser.add_argument("--no-observe-leg-lengths", dest="observe_leg_lengths", action="store_false")
    parser.add_argument("--command-resample-seconds", type=float, default=0.0,
                        help="Random-command interval in simulation seconds; zero keeps reset-only sampling")
    parser.add_argument("--linear-acceleration", type=float, default=0.0,
                        help="Planar command acceleration limit in m/s^2; zero disables limiting")
    parser.add_argument("--yaw-acceleration", type=float, default=0.0,
                        help="Yaw command acceleration limit in rad/s^2; zero disables limiting")
    parser.add_argument("--zero-command-fraction", type=float, default=0.2,
                        help="Stationary target fraction in timed random-command training")
    return parser


def validate_common(parser, args):
    if not args.allow_approximate_loop:
        parser.error(
            "需要 --allow-approximate-loop：旧 Isaac Gym 版使用外力实现柔性闭环近似，"
            "不等同于 Isaac Sim 原模型的刚性闭环约束。请先阅读 README.md。"
        )
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.physics_threads < 0:
        parser.error("--physics-threads must be nonnegative")
    if args.leg_control not in ("torque", "hip_pd", "hip_target"):
        parser.error("--leg-control must be torque, hip_pd or hip_target")
    if not math.isfinite(args.hip_target_scale) or not 0 < args.hip_target_scale <= .5:
        parser.error("--hip-target-scale must be finite and in (0,0.5] rad")
    if not 0 <= args.stair_min_level <= args.stair_level <= args.stair_max_level <= 4:
        parser.error("Require 0 <= --stair-min-level <= --stair-level <= --stair-max-level <= 4")
    if not math.isfinite(args.stair_episode_seconds) or args.stair_episode_seconds <= 0:
        parser.error("--stair-episode-seconds must be positive and finite")
    if not math.isfinite(args.leg_kp) or args.leg_kp <= 0:
        parser.error("--leg-kp must be positive and finite")
    if not math.isfinite(args.leg_kd) or args.leg_kd < 0:
        parser.error("--leg-kd must be nonnegative and finite")
    if not math.isfinite(args.leg_feedforward_force) or args.leg_feedforward_force < 0:
        parser.error("--leg-feedforward-force must be nonnegative and finite")
    if args.leg_feedforward_force and args.leg_control not in ("hip_pd", "hip_target"):
        parser.error("--leg-feedforward-force requires a hip PD controller")
    if args.physics_solver not in ("tgs", "pgs"):
        parser.error("--physics-solver must be tgs or pgs")
    if not math.isfinite(args.wheel_friction) or args.wheel_friction < 0:
        parser.error("--wheel-friction must be finite and nonnegative")
    for name in ("heading_kp", "heading_kd", "wheel_pitch_damping"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error("--{} must be nonnegative and finite".format(name.replace("_", "-")))
    if args.wheel_pitch_damping and args.wheel_control != "heading":
        parser.error("--wheel-pitch-damping requires --wheel-control heading")
    if not math.isfinite(args.wheel_balance_scale) or args.wheel_balance_scale <= 0:
        parser.error("--wheel-balance-scale must be finite and positive")
    if args.wheel_balance_scale != 1. and args.wheel_control != "heading":
        parser.error("--wheel-balance-scale requires --wheel-control heading")
    if args.initial_leg_length is not None and (
            not math.isfinite(args.initial_leg_length) or not 0.05 <= args.initial_leg_length <= MAX_LEG_LENGTH):
        parser.error("--initial-leg-length must be finite and in [0.05,{}] m".format(MAX_LEG_LENGTH))
    if (not math.isfinite(args.closure_stiffness) or not math.isfinite(args.closure_damping)
            or args.closure_stiffness <= 0 or args.closure_damping < 0):
        parser.error("closure stiffness must be positive and damping nonnegative")
    if not math.isfinite(args.physics_dt) or not 0 < args.physics_dt <= DEFAULT_CONTROL_DT:
        parser.error("--physics-dt must be finite and in (0, 0.02]")
    decimation = round(DEFAULT_CONTROL_DT / args.physics_dt)
    if not math.isclose(decimation * args.physics_dt, DEFAULT_CONTROL_DT, abs_tol=1.0e-9):
        parser.error("--physics-dt must divide the 0.02 s control period")
    if not all(math.isfinite(value) for value in args.command):
        parser.error("--command values must be finite")
    if not MIN_LEG_LENGTH <= args.command[3] <= MAX_LEG_LENGTH:
        parser.error("--command leg length must be in [{},{}] m".format(MIN_LEG_LENGTH, MAX_LEG_LENGTH))
    for name in ("forward_speed_range", "yaw_speed_range", "leg_length_range"):
        low, high = getattr(args, name)
        if not all(math.isfinite(x) for x in (low, high)) or low > high:
            parser.error("--{} must be a finite ordered range".format(name.replace("_", "-")))
    if not MIN_LEG_LENGTH <= args.leg_length_range[0] <= args.leg_length_range[1] <= MAX_LEG_LENGTH:
        parser.error("--leg-length-range must stay within [{},{}] m".format(MIN_LEG_LENGTH, MAX_LEG_LENGTH))
    for name in ("command_resample_seconds", "linear_acceleration", "yaw_acceleration"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error("--{} must be finite and nonnegative".format(name.replace("_", "-")))
    if not math.isfinite(args.zero_command_fraction) or not 0 <= args.zero_command_fraction <= 1:
        parser.error("--zero-command-fraction must be in [0,1]")


def environment_kwargs(args):
    return dict(
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        headless=args.headless,
        seed=args.seed,
        initial_leg_length=args.initial_leg_length,
        leg_control=args.leg_control,
        leg_kp=args.leg_kp,
        leg_kd=args.leg_kd,
        hip_target_scale=args.hip_target_scale,
        terrain=args.terrain,
        observe_terrain=args.observe_terrain,
        stair_level=args.stair_level,
        stair_min_level=args.stair_min_level,
        stair_max_level=args.stair_max_level,
        stair_episode_seconds=args.stair_episode_seconds,
        stair_curriculum=args.stair_curriculum,
        leg_feedforward_force=args.leg_feedforward_force,
        heading_kp=args.heading_kp,
        heading_kd=args.heading_kd,
        wheel_control=args.wheel_control,
        wheel_pitch_damping=args.wheel_pitch_damping,
        wheel_balance_scale=args.wheel_balance_scale,
        physics_solver=args.physics_solver,
        wheel_friction=args.wheel_friction,
        closure_stiffness=args.closure_stiffness,
        closure_damping=args.closure_damping,
        physics_dt=args.physics_dt,
        physics_threads=args.physics_threads,
        command=tuple(args.command),
        random_commands=args.random_commands,
        forward_speed_range=tuple(args.forward_speed_range),
        yaw_speed_range=tuple(args.yaw_speed_range),
        leg_length_range=tuple(args.leg_length_range),
        tracking_penalties=args.tracking_penalties,
        stability_rewards=args.stability_rewards,
        support_rewards=args.support_rewards,
        observe_leg_lengths=args.observe_leg_lengths,
        command_resample_seconds=args.command_resample_seconds,
        linear_acceleration=args.linear_acceleration,
        yaw_acceleration=args.yaw_acceleration,
        zero_command_fraction=args.zero_command_fraction,
        allow_approximate_loop=args.allow_approximate_loop,
    )
