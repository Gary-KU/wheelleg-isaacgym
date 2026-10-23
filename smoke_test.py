#!/usr/bin/env python3
"""Bounded GPU integration check; never implies a trained balancing policy."""

import json
import time
from pathlib import Path

from wheelleg_gym.cli import common_parser, environment_kwargs, validate_common


def main():
    parser = common_parser("Measure compliant closure under bounded test actions", 4)
    parser.add_argument("--num-steps", "--num_steps", type=int, default=250)
    parser.add_argument("--fix-base", action="store_true")
    parser.add_argument("--action-amplitude", type=float, default=1.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-error-mm", type=float, default=10.0)
    args = parser.parse_args()
    validate_common(parser, args)
    if args.num_steps < 1 or args.action_amplitude < 0:
        parser.error("Require positive steps and nonnegative action amplitude")
    import isaacgym  # noqa: F401
    import torch
    from wheelleg_gym.env import WheelLegGymEnv

    kwargs = environment_kwargs(args)
    kwargs.update(fix_base=args.fix_base, physics_dt=args.physics_dt)
    env = WheelLegGymEnv(**kwargs)
    try:
        initial_error = float(env.closure.errors(env.rb_states).max())
        initial_axis_error = float(env.closure.axis_errors(env.rb_states).max())
        errors, resets, failures = [], 0, 0
        initial_time = time.perf_counter()
        with torch.inference_mode():
            for i in range(args.num_steps):
                phase = torch.arange(env.num_envs*6, device=env.device).reshape(env.num_envs, 6)
                actions = args.action_amplitude * torch.sin(phase+i*0.13)
                obs, _, rewards, dones, infos = env.step(actions)
                if not bool(torch.isfinite(obs).all() & torch.isfinite(rewards).all()):
                    raise RuntimeError("Non-finite simulation output")
                errors.append(infos["loop_error"].flatten())
                resets += int(dones.sum())
                failures += int(infos["loop_failures"])
        elapsed = time.perf_counter()-initial_time
        error = torch.cat(errors)
        result = dict(num_envs=args.num_envs, steps=args.num_steps,
            fix_base=args.fix_base, action_amplitude=args.action_amplitude,
            physics_dt=args.physics_dt, closure_stiffness=args.closure_stiffness,
            closure_damping=args.closure_damping, initial_error_mm=initial_error*1000,
            initial_axis_error_rad=initial_axis_error,
            loop_error_max_mm=float(error.max())*1000,
            loop_error_p95_mm=float(torch.quantile(error, 0.95))*1000,
            resets=resets, closure_failures=failures, wall_seconds=elapsed,
            approximate_loop=True,
            passed=bool(error.max() <= args.max_error_mm/1000 and failures == 0))
        print(json.dumps(result, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        if not result["passed"]:
            raise SystemExit("Closure test exceeded the requested error tolerance")
    finally:
        env.close()


if __name__ == "__main__":
    main()
