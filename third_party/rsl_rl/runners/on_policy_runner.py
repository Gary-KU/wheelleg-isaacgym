# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import json
from pathlib import Path
from copy import deepcopy
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.train_cfg = deepcopy(train_cfg)
        self.cfg = self.train_cfg["runner"]
        self.alg_cfg = self.train_cfg["algorithm"]
        self.policy_cfg = self.train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if (
            self.cfg["policy_class_name"] != "ActorCritic"
            or self.cfg["algorithm_class_name"] != "PPO"
        ):
            raise ValueError(
                "This migration supports the checkpoint-compatible feed-forward ActorCritic/PPO only"
            )
        actor_critic_class = ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [self.env.obs_history_length * self.env.num_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = str(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, obs_history = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, obs_history, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            critic_obs.to(self.device),
        )
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            loop_peak = 0.0
            physical_failures = 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, obs_history = (
                        self.env.step(actions)
                    )
                    if not bool(
                        torch.isfinite(obs).all() and torch.isfinite(rewards).all()
                    ):
                        raise RuntimeError(
                            "Non-finite observation/reward during rollout"
                        )
                    loop_peak = max(loop_peak, float(infos.get("loop_error_max", 0.0)))
                    physical_failures += int(
                        infos.get("physical_failures", dones).sum()
                    )
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, obs_history, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss = (
                self.alg.update()
            )
            stop = time.time()
            learn_time = stop - start
            if not all(
                bool(torch.isfinite(p).all())
                for p in self.alg.actor_critic.parameters()
            ):
                raise RuntimeError("Non-finite policy after PPO update")
            self.current_learning_iteration = it + 1
            if self.log_dir is not None:
                self.log(locals())
                progress = dict(
                    completed_iterations=self.current_learning_iteration,
                    total_environment_steps=self.tot_timesteps,
                    value_loss=mean_value_loss,
                    policy_loss=mean_surrogate_loss,
                    kl=mean_kl,
                    loop_peak_m=loop_peak,
                    physical_failures=physical_failures,
                )
                Path(self.log_dir, "progress.json").write_text(
                    json.dumps(progress, indent=2) + "\n"
                )
                self.writer.add_scalar("Loop/max_anchor_error_m", loop_peak, it + 1)
                self.writer.add_scalar(
                    "Episode/physical_failures", physical_failures, it + 1
                )
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            "model_{}.pt".format(self.current_learning_iteration),
                        )
                    )
            else:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            ep_infos.clear()
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
                )
            )
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"] + 1)
                ep_string += f"""{f'Mean {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"] + 1
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"] + 1)
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"] + 1
        )
        self.writer.add_scalar(
            "Loss/learning_rate", self.alg.learning_rate, locs["it"] + 1
        )
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"] + 1)
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"] + 1)
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"] + 1)
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"] + 1
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"] + 1)
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"] + 1
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"] + 1,
            )

        str = f" \033[1m Learning iteration {locs['it']+1}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it']-locs['start_iter']+1) * (
                               locs['tot_iter'] - locs['it'] - 1):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        from legged_gym.utils.helpers import class_to_dict

        state = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "total_steps": self.tot_timesteps,
            "infos": infos,
            "training_config": dict(
                policy=self.policy_cfg,
                algorithm=self.alg_cfg,
                num_steps_per_env=self.num_steps_per_env,
                save_interval=self.save_interval,
            ),
            "environment_config": self.env.environment_config,
            "legged_gym_env_config": class_to_dict(self.env.cfg),
            "legged_gym_train_config": self.train_cfg,
            "loop_model": "external_force_compliant_approximation",
            "terrain_curriculum_state": (
                {"levels": self.env.terrain_levels.cpu().tolist()}
                if self.env.terrain_model is not None
                else None
            ),
        }
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, str(temporary))
        temporary.replace(path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(str(path), map_location=self.device)
        self.alg.actor_critic.load_state_dict(
            loaded_dict["model_state_dict"], strict=True
        )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.learning_rate = min(
                self.alg.optimizer.param_groups[0]["lr"], self.alg.max_learning_rate
            )
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate
        self.current_learning_iteration = int(loaded_dict["iter"])
        self.tot_timesteps = int(loaded_dict.get("total_steps", 0))
        terrain = loaded_dict.get("terrain_curriculum_state")
        if terrain and self.env.stair_curriculum:
            levels = torch.as_tensor(
                terrain["levels"], device=self.env.device, dtype=torch.long
            )
            if levels.numel():
                ids = (
                    torch.arange(self.env.num_envs, device=self.env.device)
                    % levels.numel()
                )
                self.env.terrain_levels.copy_(
                    levels[ids].clamp(
                        self.env.stair_min_level, self.env.stair_max_level
                    )
                )
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
