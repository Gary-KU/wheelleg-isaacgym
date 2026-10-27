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

"""legged_gym API adapter over the project's single PPO training engine."""

from copy import deepcopy

import torch

from legged_gym.learning.ppo_runner import PPORunner


class _FiveValueEnv:
    """Expose existing six-value tasks to the shared five-value rollout loop."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        observations, privileged, rewards, dones, infos, _ = self.env.step(actions)
        return observations, privileged, rewards, dones, infos


class OnPolicyRunner:
    """Keep registered-task configuration and inference APIs without a second loop."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if torch.device(device) != torch.device(env.device):
            raise ValueError("PPO and simulation must use the same device")
        if env.num_privileged_obs is not None or env.obs_history_length != 1:
            raise ValueError(
                "This runner requires the already-stacked feed-forward observations"
            )
        self.env = env
        self.train_cfg = deepcopy(train_cfg)
        self.cfg = self.train_cfg["runner"]
        if (
            self.cfg["policy_class_name"] != "ActorCritic"
            or self.cfg["algorithm_class_name"] != "PPO"
        ):
            raise ValueError(
                "This runner supports the checkpoint-compatible ActorCritic/PPO only"
            )
        config = {
            name: deepcopy(value)
            for name, value in self.train_cfg.items()
            if name not in ("runner", "seed") and value is not None
        }
        config.update(
            num_steps_per_env=self.cfg["num_steps_per_env"],
            save_interval=self.cfg["save_interval"],
        )
        self._runner = PPORunner(
            _FiveValueEnv(env),
            log_dir,
            config=config,
            environment_config=env.environment_config,
            checkpoint_metadata=self._checkpoint_metadata,
        )
        self.device = self._runner.device
        self.alg = self._runner.algorithm
        self.alg_cfg = self._runner.config["algorithm"]
        self.policy_cfg = self._runner.config["policy"]
        self.env.reset()

    def _checkpoint_metadata(self):
        from legged_gym.utils.helpers import class_to_dict

        train = deepcopy(self.train_cfg)
        train["policy"] = deepcopy(self.policy_cfg)
        train["algorithm"] = deepcopy(self.alg_cfg)
        train["runner"].update(
            num_steps_per_env=self.num_steps_per_env, save_interval=self.save_interval
        )
        if "initialization_provenance" in self._runner.config:
            train["initialization_provenance"] = deepcopy(
                self._runner.config["initialization_provenance"]
            )
        return {
            "legged_gym_env_config": class_to_dict(self.env.cfg),
            "legged_gym_train_config": train,
        }

    @property
    def current_learning_iteration(self):
        return self._runner.iteration

    @current_learning_iteration.setter
    def current_learning_iteration(self, value):
        self._runner.iteration = value

    @property
    def tot_timesteps(self):
        return self._runner.total_steps

    @tot_timesteps.setter
    def tot_timesteps(self, value):
        self._runner.total_steps = value

    @property
    def tot_time(self):
        return self._runner.total_time

    @property
    def num_steps_per_env(self):
        return self._runner.config["num_steps_per_env"]

    @property
    def save_interval(self):
        return self._runner.config["save_interval"]

    @property
    def log_dir(self):
        value = self._runner.log_dir
        return str(value) if value is not None else None

    @property
    def writer(self):
        return self._runner.writer

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # The registered API resets on construction/load and continues between
        # learn() calls; the legacy entrypoint resets when learn() starts.
        return self._runner.learn(
            num_learning_iterations, init_at_random_ep_len, reset_env=False
        )

    def save(self, path, infos=None):
        self._runner.save(path, infos=infos)

    def load(self, path, load_optimizer=True):
        return self._runner.load(path, load_optimizer=load_optimizer)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
