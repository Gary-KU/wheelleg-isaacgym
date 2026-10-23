"""Small PPO runner for the port's five-value vector environment interface."""

from copy import deepcopy
from pathlib import Path
import json
import sys
import time

import torch

from .config import OBSERVATION_BASE_FRAME_SIZE, TORQUE_LIMIT, TRAIN_CONFIG

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def scalar(value):
    """Reduce an optional scalar/vector diagnostic for terminal/log output."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().max().item()
    return float(value)


def viewer_is_closed(env):
    viewer = getattr(env, "viewer", None)
    return viewer is not None and env.gym.query_viewer_has_closed(viewer)


def make_policy(num_obs, num_actions, device, policy_config=None):
    config = deepcopy(policy_config or TRAIN_CONFIG["policy"])
    return ActorCritic(num_obs, num_obs, num_actions, **config).to(device)


class PPORunner:
    def __init__(self, env, log_dir, config=None, environment_config=None):
        self.env = env
        self.device = env.device
        self.config = deepcopy(config or TRAIN_CONFIG)
        self.environment_config = environment_config or {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.total_steps = 0
        self.policy = make_policy(
            env.num_obs, env.num_actions, self.device, self.config["policy"]
        )
        self.algorithm = PPO(self.policy, device=self.device, **self.config["algorithm"])
        # The six observation frames are already stacked (186 or 198 values).
        # This feed-forward policy does not need the upstream sequence encoder.
        self.algorithm.init_storage(
            env.num_envs,
            self.config["num_steps_per_env"],
            [env.num_obs],
            [env.num_obs],
            [env.num_obs],
            [env.num_actions],
        )
        self.writer = None

    def load(self, checkpoint):
        state = torch.load(str(checkpoint), map_location=self.device)
        self.policy.load_state_dict(state["model_state_dict"], strict=True)
        if "optimizer_state_dict" in state:
            self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.algorithm.learning_rate = min(self.algorithm.optimizer.param_groups[0]["lr"], self.algorithm.max_learning_rate)
            for group in self.algorithm.optimizer.param_groups:
                group["lr"] = self.algorithm.learning_rate
        self.iteration = int(state.get("iter", 0))
        self.total_steps = int(state.get("total_steps", 0))
        terrain_state = state.get("terrain_curriculum_state")
        if terrain_state and getattr(self.env, "stair_curriculum", False):
            levels = torch.tensor(terrain_state["levels"], device=self.device, dtype=torch.long)
            if levels.numel():
                self.env.terrain_levels.copy_(levels[torch.arange(self.env.num_envs, device=self.device)%len(levels)].clamp(self.env.stair_min_level, self.env.stair_max_level))
        print("Resumed checkpoint at completed iteration {}".format(self.iteration))

    def save(self, checkpoint):
        checkpoint = Path(checkpoint)
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                "iter": self.iteration,
                "total_steps": self.total_steps,
                "training_config": self.config,
                "environment_config": self.environment_config,
                "loop_model": "external_force_compliant_approximation",
                "infos": None,
                "terrain_curriculum_state": ({"levels": self.env.terrain_levels.cpu().tolist(),
                                              "min_level": self.env.stair_min_level,
                                              "max_level": self.env.stair_max_level}
                    if getattr(self.env, "terrain_model", None) is not None else None),
            },
            str(temporary),
        )
        temporary.replace(checkpoint)

    def learn(self, num_iterations):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(self.log_dir), flush_secs=10)
        (self.log_dir / "config.json").write_text(
            json.dumps(
                {
                    "training": self.config,
                    "environment": self.environment_config,
                    "loop_model": "external_force_compliant_approximation",
                },
                indent=2,
            )
            + "\n"
        )
        observations, _ = self.env.reset()
        self.policy.train()
        iteration_metrics = []
        uses_cuda = str(self.device).startswith("cuda")
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for _ in range(num_iterations):
                started = time.perf_counter()
                reward_sum = 0.0
                max_loop_error = 0.0
                resets = 0
                completed_episodes = 0
                completed_duration = 0.0
                completed_return = 0.0
                timeouts = 0
                closure_failures = 0
                terrain_successes = 0
                physical_failures = 0
                terrain_progress_peak = 0.0
                terrain_level_sum = 0.0
                motion_sums = torch.zeros(8, device=self.device)
                support_sums = torch.zeros(4, device=self.device)
                reward_term_sums = {}
                with torch.inference_mode():
                    for _ in range(self.config["num_steps_per_env"]):
                        if viewer_is_closed(self.env):
                            self.save(self.log_dir / "model_{}.pt".format(self.iteration))
                            print("Viewer closed; saved the latest completed policy update.")
                            return
                        # PPO clones observations before env.step mutates its buffers.
                        actions = self.algorithm.act(observations, observations, observations)
                        observations, _, rewards, dones, infos = self.env.step(actions)
                        command = infos["pre_reset_commands"]
                        # Linear velocity in the final observation uses body coordinates.
                        frame_size = getattr(self.env, "observation_frame_size", OBSERVATION_BASE_FRAME_SIZE)
                        measured = observations[:, -frame_size:].clone()
                        ids = infos.get("terminal_env_ids")
                        if ids is not None:
                            measured[ids] = infos["terminal_observation"][:, -frame_size:]
                        torque = infos["pre_reset_applied_torques"]
                        delta = torque-infos["pre_reset_previous_torques"]
                        motion_sums += torch.stack((
                            infos["pre_reset_pitch"].square().mean(),
                            infos["pre_reset_roll"].square().mean(),
                            infos["pre_reset_ang_vel"][:, :2].square().sum(-1).mean(),
                            delta.square().mean(),
                            (torque.abs() >= TORQUE_LIMIT-0.001).float().mean(),
                            (measured[:, 0]/1.5-command[:, 0]).square().mean(),
                            (infos["pre_reset_ang_vel"][:, 2]-command[:, 2]).square().mean(),
                            (infos["pre_reset_leg_length"]-command[:, 3:4]).square().mean(),
                        ))
                        lengths = infos["pre_reset_leg_length"]
                        support_sums += torch.stack((
                            lengths.mean(),
                            ((lengths-command[:, 3:4]).abs() <= 0.005).all(-1).float().mean(),
                            (infos["pre_reset_passive_limit_margin"] <= 0.01).float().mean(),
                            (lengths[:, 0]-lengths[:, 1]).square().mean(),
                        ))
                        for name, value in infos.get("reward_terms", {}).items():
                            reward_term_sums[name] = reward_term_sums.get(name, 0.0)+value.mean()
                        if not torch.isfinite(observations).all() or not torch.isfinite(rewards).all():
                            raise RuntimeError("Non-finite observation/reward during PPO rollout")
                        self.algorithm.process_env_step(rewards, dones, infos, observations)
                        reward_sum += rewards.mean().item()
                        resets += dones.count_nonzero().item()
                        successes = infos.get("terrain_success", torch.zeros_like(dones))
                        terrain_successes += int(successes.sum())
                        physical_failures += int((dones & ~infos.get("time_outs", torch.zeros_like(dones)) & ~successes).sum())
                        if "terrain_progress" in infos:
                            terrain_progress_peak = max(terrain_progress_peak, scalar(infos["terrain_best_progress"]))
                            terrain_level_sum += float(infos["terrain_level"].float().mean())
                        if "episode" in infos:
                            count = int(dones.count_nonzero().item())
                            completed_episodes += count
                            completed_duration += scalar(infos["episode"]["length"]) * self.env.dt * count
                            completed_return += scalar(infos["episode"]["reward"]) * count
                        timeouts += int(infos.get("time_outs", torch.zeros(1)).sum().item())
                        closure_failures += int(infos.get("loop_failures", 0))
                        if "loop_error_max" in infos:
                            max_loop_error = max(max_loop_error, scalar(infos["loop_error_max"]))
                    self.algorithm.compute_returns(observations)
                value_loss, policy_loss, kl, _ = self.algorithm.update()
                if not all(torch.isfinite(parameter).all() for parameter in self.policy.parameters()):
                    raise RuntimeError("Non-finite policy parameter after PPO update")
                self.iteration += 1
                self.total_steps += self.env.num_envs * self.config["num_steps_per_env"]
                elapsed = time.perf_counter() - started
                mean_reward = reward_sum / self.config["num_steps_per_env"]
                metrics = {
                    "Train/mean_step_reward": mean_reward,
                    "Train/resets_per_rollout": resets,
                    "Episode/timeouts": timeouts,
                    "Episode/physical_failures": physical_failures,
                    "Loop/failures": closure_failures,
                    "Loss/value": value_loss,
                    "Loss/surrogate": policy_loss,
                    "Policy/kl": kl,
                    "Policy/learning_rate": self.algorithm.learning_rate,
                    "Loop/max_anchor_error_m": max_loop_error,
                    "Perf/environment_steps_per_second": self.env.num_envs
                    * self.config["num_steps_per_env"]
                    / max(elapsed, 1.0e-9),
                }
                if getattr(self.env, "terrain_model", None) is not None:
                    metrics.update({
                        "Terrain/successful_courses": terrain_successes,
                        "Terrain/max_forward_progress_m": terrain_progress_peak,
                        "Terrain/mean_level": terrain_level_sum/self.config["num_steps_per_env"],
                        "Terrain/max_current_level": scalar(self.env.terrain_levels),
                    })
                motion_means = (motion_sums/self.config["num_steps_per_env"]).cpu().tolist()
                for index, name in enumerate(("pitch_rad", "roll_rad", "xy_angular_velocity_rad_s",
                                              "torque_delta_nm", "torque_saturation_fraction",
                                              "vx_tracking_error_m_s", "yaw_tracking_error_rad_s",
                                              "leg_length_error_m")):
                    metrics["Motion/" + (name if index == 4 else "rms_"+name)] = motion_means[index] if index == 4 else motion_means[index]**0.5
                support_means = (support_sums/self.config["num_steps_per_env"]).cpu().tolist()
                for index, name in enumerate(("mean_leg_length_m", "both_legs_within_5mm_fraction",
                                              "passive_near_limit_fraction", "rms_leg_length_asymmetry_m")):
                    metrics["Support/"+name] = support_means[index]**0.5 if index == 3 else support_means[index]
                for name, value in reward_term_sums.items():
                    metrics["Reward/"+name] = scalar(value/self.config["num_steps_per_env"])
                if completed_episodes:
                    metrics["Episode/mean_completed_duration_s"] = completed_duration / completed_episodes
                    metrics["Episode/mean_completed_return"] = completed_return / completed_episodes
                metrics["Episode/mean_current_duration_s"] = self.env.episode_length_buf.float().mean().item() * self.env.dt
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, self.iteration)
                iteration_metrics.append(dict(iteration=self.iteration, wall_seconds=elapsed, **metrics))
                (self.log_dir / "progress.json").write_text(json.dumps(iteration_metrics[-1], indent=2)+"\n")
                print(
                    "iteration={} reward={:.4f} value_loss={:.4f} "
                    "policy_loss={:.4f} loop_max={:.3f}mm resets={} episode_s={:.3f} time={:.2f}s".format(
                        self.iteration, mean_reward, value_loss, policy_loss,
                        max_loop_error * 1000.0, resets, completed_duration / max(completed_episodes, 1), elapsed,
                    ),
                    flush=True,
                )
                if self.iteration % self.config["save_interval"] == 0:
                    self.save(self.log_dir / "model_{}.pt".format(self.iteration))
            self.save(self.log_dir / "model_{}.pt".format(self.iteration))
        finally:
            summary = {
                "completed_iterations": self.iteration,
                "total_environment_steps": self.total_steps,
                "iterations": iteration_metrics,
            }
            if uses_cuda:
                peak_bytes = torch.cuda.max_memory_allocated(self.device)
                summary["pytorch_cuda_peak_allocated_bytes"] = peak_bytes
                summary["memory_scope"] = "PyTorch allocator only; excludes PhysX, rendering and other GPU allocations"
                print("PyTorch CUDA allocation peak: {:.2f} MiB (excludes PhysX/rendering)".format(
                    peak_bytes / 1024**2), flush=True)
            (self.log_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.writer.close()
