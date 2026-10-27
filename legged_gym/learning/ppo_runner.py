"""Shared PPO training engine for the legacy and registered task interfaces."""

from copy import deepcopy
from pathlib import Path
import json
import time

import torch

from .policy import make_policy, scalar, viewer_is_closed

from rsl_rl.algorithms import PPO


class PPORunner:
    def __init__(
        self,
        env,
        log_dir,
        config,
        environment_config=None,
        checkpoint_metadata=None,
        metrics_collector=None,
    ):
        """Train a five-value vector environment with caller-owned configuration.

        An optional metrics collector implements reset(), update(obs, rewards,
        dones, infos), and compute(num_steps). Environments can supply it through
        create_training_metrics(); checkpoint/console hooks are optional.
        """
        self.env = env
        self.device = env.device
        self.config = deepcopy(config)
        self.environment_config = environment_config or {}
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_metadata = checkpoint_metadata
        factory = getattr(env, "create_training_metrics", None)
        self.metrics_collector = metrics_collector
        if self.metrics_collector is None and factory is not None:
            self.metrics_collector = factory()
        self.iteration = 0
        self.total_steps = 0
        self.total_time = 0.0
        self.policy = make_policy(
            env.num_obs, env.num_actions, self.device, self.config["policy"]
        )
        self.algorithm = PPO(
            self.policy, device=self.device, **self.config["algorithm"]
        )
        # The environment supplies the complete feed-forward observation vector.
        self.algorithm.init_storage(
            env.num_envs,
            self.config["num_steps_per_env"],
            [env.num_obs],
            [env.num_obs],
            [env.num_obs],
            [env.num_actions],
        )
        self.writer = None

    def load(self, checkpoint, load_optimizer=True):
        state = torch.load(str(checkpoint), map_location=self.device)
        self.policy.load_state_dict(state["model_state_dict"], strict=True)
        if load_optimizer and "optimizer_state_dict" in state:
            self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.algorithm.learning_rate = min(
                self.algorithm.optimizer.param_groups[0]["lr"],
                self.algorithm.max_learning_rate,
            )
            for group in self.algorithm.optimizer.param_groups:
                group["lr"] = self.algorithm.learning_rate
        self.iteration = int(state.get("iter", 0))
        self.total_steps = int(state.get("total_steps", 0))
        # Warm-start tools record provenance beside the four training settings.
        # Keep those extra fields even when the caller uses nested task configs.
        for name, value in state.get("training_config", {}).items():
            if name not in self.config:
                self.config[name] = deepcopy(value)
        restore = getattr(
            getattr(self, "metrics_collector", None), "restore_checkpoint", None
        )
        if restore is not None:
            restore(state)
        print("Resumed checkpoint at completed iteration {}".format(self.iteration))
        return state.get("infos")

    def save(self, checkpoint, infos=None):
        checkpoint = Path(checkpoint)
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        state = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "iter": self.iteration,
            "total_steps": self.total_steps,
            "training_config": self.config,
            "environment_config": self.environment_config,
            "infos": infos,
        }
        task_metadata = getattr(
            getattr(self, "metrics_collector", None), "checkpoint_metadata", None
        )
        if task_metadata is not None:
            state.update(task_metadata())
        metadata = getattr(self, "checkpoint_metadata", None)
        if metadata is not None:
            state.update(metadata())
        torch.save(state, str(temporary))
        temporary.replace(checkpoint)

    def learn(self, num_iterations, init_at_random_ep_len=False, reset_env=True):
        if self.log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.log_dir), flush_secs=10)
            config = {
                "training": self.config,
                "environment": self.environment_config,
            }
            task_metadata = getattr(
                self.metrics_collector, "configuration_metadata", None
            )
            if task_metadata is not None:
                config.update(task_metadata())
            if self.checkpoint_metadata is not None:
                config.update(self.checkpoint_metadata())
            (self.log_dir / "config.json").write_text(
                json.dumps(config, indent=2) + "\n"
            )
        observations, _ = self.env.reset() if reset_env else self.env.get_observations()
        if init_at_random_ep_len:
            self.env.episode_length_buf.copy_(
                torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
            )
        self.policy.train()
        iteration_metrics = []
        uses_cuda = str(self.device).startswith("cuda")
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for _ in range(num_iterations):
                started = time.perf_counter()
                reward_sum = 0.0
                resets = 0
                completed_episodes = 0
                completed_duration = 0.0
                completed_return = 0.0
                timeouts = 0
                if self.metrics_collector is not None:
                    self.metrics_collector.reset()
                with torch.inference_mode():
                    for _ in range(self.config["num_steps_per_env"]):
                        if viewer_is_closed(self.env):
                            if self.log_dir is not None:
                                self.save(
                                    self.log_dir / "model_{}.pt".format(self.iteration)
                                )
                                print(
                                    "Viewer closed; saved the latest completed policy update."
                                )
                            return
                        # PPO clones observations before env.step mutates its buffers.
                        actions = self.algorithm.act(
                            observations, observations, observations
                        )
                        observations, _, rewards, dones, infos = self.env.step(actions)
                        if self.metrics_collector is not None:
                            self.metrics_collector.update(
                                observations, rewards, dones, infos
                            )
                        if (
                            not torch.isfinite(observations).all()
                            or not torch.isfinite(rewards).all()
                        ):
                            raise RuntimeError(
                                "Non-finite observation/reward during PPO rollout"
                            )
                        self.algorithm.process_env_step(
                            rewards, dones, infos, observations
                        )
                        reward_sum += rewards.mean().item()
                        resets += dones.count_nonzero().item()
                        episode = infos.get("episode", {})
                        if "length" in episode and "reward" in episode:
                            count = int(dones.count_nonzero().item())
                            completed_episodes += count
                            completed_duration += (
                                scalar(episode["length"])
                                * getattr(self.env, "dt", 1.0)
                                * count
                            )
                            completed_return += scalar(episode["reward"]) * count
                        timeouts += int(
                            infos.get("time_outs", torch.zeros(1)).sum().item()
                        )
                    self.algorithm.compute_returns(observations)
                value_loss, policy_loss, kl, _ = self.algorithm.update()
                if not all(
                    torch.isfinite(parameter).all()
                    for parameter in self.policy.parameters()
                ):
                    raise RuntimeError("Non-finite policy parameter after PPO update")
                self.iteration += 1
                self.total_steps += self.env.num_envs * self.config["num_steps_per_env"]
                elapsed = time.perf_counter() - started
                self.total_time += elapsed
                mean_reward = reward_sum / self.config["num_steps_per_env"]
                metrics = {
                    "Train/mean_step_reward": mean_reward,
                    "Train/resets_per_rollout": resets,
                    "Episode/timeouts": timeouts,
                    "Loss/value": value_loss,
                    "Loss/surrogate": policy_loss,
                    "Policy/kl": kl,
                    "Policy/learning_rate": self.algorithm.learning_rate,
                    "Perf/environment_steps_per_second": self.env.num_envs
                    * self.config["num_steps_per_env"]
                    / max(elapsed, 1.0e-9),
                }
                if self.metrics_collector is not None:
                    metrics.update(
                        self.metrics_collector.compute(self.config["num_steps_per_env"])
                    )
                if completed_episodes:
                    metrics["Episode/mean_completed_duration_s"] = (
                        completed_duration / completed_episodes
                    )
                    metrics["Episode/mean_completed_return"] = (
                        completed_return / completed_episodes
                    )
                lengths = getattr(self.env, "episode_length_buf", None)
                if lengths is not None and hasattr(self.env, "dt"):
                    metrics["Episode/mean_current_duration_s"] = (
                        lengths.float().mean().item() * self.env.dt
                    )
                if self.writer is not None:
                    for name, value in metrics.items():
                        self.writer.add_scalar(name, value, self.iteration)
                iteration_metrics.append(
                    dict(iteration=self.iteration, wall_seconds=elapsed, **metrics)
                )
                if self.log_dir is not None:
                    (self.log_dir / "progress.json").write_text(
                        json.dumps(iteration_metrics[-1], indent=2) + "\n"
                    )
                console_summary = getattr(
                    self.metrics_collector, "console_summary", None
                )
                task_summary = console_summary() if console_summary is not None else ""
                print(
                    "iteration={} reward={:.4f} value_loss={:.4f} "
                    "policy_loss={:.4f} {}resets={} episode_s={:.3f} time={:.2f}s".format(
                        self.iteration,
                        mean_reward,
                        value_loss,
                        policy_loss,
                        task_summary,
                        resets,
                        completed_duration / max(completed_episodes, 1),
                        elapsed,
                    ),
                    flush=True,
                )
                if (
                    self.log_dir is not None
                    and self.iteration % self.config["save_interval"] == 0
                ):
                    self.save(self.log_dir / "model_{}.pt".format(self.iteration))
            if self.log_dir is not None:
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
                summary["memory_scope"] = (
                    "PyTorch allocator only; excludes PhysX, rendering and other GPU allocations"
                )
                print(
                    "PyTorch CUDA allocation peak: {:.2f} MiB (excludes PhysX/rendering)".format(
                        peak_bytes / 1024**2
                    ),
                    flush=True,
                )
            if self.log_dir is not None:
                (self.log_dir / "training_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
            if self.writer is not None:
                self.writer.close()
                self.writer = None
