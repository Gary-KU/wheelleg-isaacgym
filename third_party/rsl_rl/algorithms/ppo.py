#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        extra_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        kl_decay=0,
        device="cpu",
        max_learning_rate=1e-2,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.kl_decay = max(kl_decay, 0)
        self.schedule = schedule
        if not math.isfinite(max_learning_rate) or max_learning_rate < 1e-5:
            raise ValueError("max_learning_rate must be finite and at least 1e-5")
        self.max_learning_rate = float(max_learning_rate)
        self.learning_rate = min(learning_rate, self.max_learning_rate)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.actor.parameters()},
                {"params": self.actor_critic.critic.parameters()},
                {"params": self.actor_critic.std},
            ],
            lr=self.learning_rate,
        )
        self.extra_optimizer = None
        if self.actor_critic.is_sequence:
            self.extra_optimizer = optim.Adam(
                [
                    {"params": self.actor_critic.encoder.parameters()},
                ],
                lr=extra_learning_rate,
            )
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        if self.actor_critic.is_sequence:
            self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
            latent = self.actor_critic.get_latent()
            critic_obs = torch.cat((critic_obs, latent), dim=-1)
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.clone()
        self.transition.observation_history = obs_history.clone()
        self.transition.critic_observations = critic_obs.clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # A time limit truncates a continuing task. Bootstrap the terminal
        # state, before the vector environment replaces it with a reset state.
        if "time_outs" in infos:
            timeout_mask = infos["time_outs"].to(self.device).bool().reshape(-1)
            timeout_mask = timeout_mask & dones.to(self.device).bool().reshape(-1)
            if bool(timeout_mask.any()):
                values = self._timeout_values(infos, timeout_mask)
                self.transition.rewards += self.gamma * values * timeout_mask

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def _timeout_values(self, infos, timeout_mask):
        """Evaluate pre-reset terminal observations for normal time limits.

        Sparse observations use ``terminal_env_ids`` to identify their rows;
        without IDs, a full batch in environment order is required. Old
        environments without terminal observations retain the previous
        V(s_t) approximation with a warning. Autoreset ``next_obs`` must never
        be used as the terminal state.
        """
        values = self.transition.values.detach().reshape(-1).clone()
        terminal = infos.get("terminal_observation")
        if terminal is None:
            warnings.warn(
                "Timeout terminal observations missing; falling back to V(s_t) "
                "for legacy compatibility. Provide pre-reset terminal_observation "
                "and terminal_env_ids for correct timeout bootstrapping.",
                RuntimeWarning,
                stacklevel=2,
            )
            return values
        terminal = terminal.to(self.device)
        ids = infos.get("terminal_env_ids")
        if ids is None:
            if terminal.shape[0] != values.numel():
                raise ValueError("Sparse terminal observations require terminal_env_ids")
            ids = torch.arange(values.numel(), device=self.device)
        else:
            ids = torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(-1)
            if terminal.shape[0] != ids.numel():
                raise ValueError("terminal_env_ids must match terminal observation rows")
            if bool(((ids < 0) | (ids >= values.numel())).any()) or ids.unique().numel() != ids.numel():
                raise ValueError("terminal_env_ids must be unique valid environment indices")
        selected = timeout_mask[ids]
        if int(selected.sum()) != int(timeout_mask.sum()):
            raise ValueError("Terminal observations must cover every timed-out environment")
        timeout_ids = ids[selected]
        with torch.no_grad():
            terminal_values = self.actor_critic.evaluate(terminal[selected]).reshape(-1)
        if terminal_values.numel() != timeout_ids.numel():
            raise ValueError("Critic must return one value per terminal observation")
        values[timeout_ids] = terminal_values
        return values

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        if self.kl_decay != 0:
            self.desired_kl = max(self.desired_kl - self.kl_decay, 0.001)
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        for (
            obs_batch,
            obs_history_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            if self.actor_critic.is_sequence:
                self.actor_critic.act(
                    obs_batch,
                    obs_history_batch,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )
            else:
                self.actor_critic.act(
                    obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            num_updates += 1

        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for next_obs_batch, critic_obs_batch, obs_history_batch in generator:
                if self.actor_critic.is_sequence:
                    latent_batch = self.actor_critic.encode(obs_history_batch)
                    vel_est_loss = (
                        (latent_batch[:, :3] - critic_obs_batch[:, :3]).pow(2).mean()
                    )
                    if self.actor_critic.latent_dim > 3:
                        obs_denoise_loss = (
                            (
                                latent_batch[:, 3 : self.actor_critic.latent_dim]
                                - critic_obs_batch[:, 3 : self.actor_critic.latent_dim]
                            )
                            .pow(2)
                            .mean()
                        )
                        extra_loss = vel_est_loss + obs_denoise_loss
                    else:
                        extra_loss = vel_est_loss

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.1)
                self.extra_optimizer.step()

                mean_extra_loss += extra_loss.item()
                num_updates_extra += 1

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        self.storage.clear()

        return (mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss)
