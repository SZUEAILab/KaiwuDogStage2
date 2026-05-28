#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
AlgorithmHierNav — hierarchical navigation training.

Freezes a pretrained locomotion policy (ActorCritic, 12-DOF output) and
trains a navigation policy (NavActorCritic, 3D velocity-cmd output) on top.

Flow:
  nav_obs (proprio + scan + goal) → nav model → velocity_cmd [vx,vy,wz]
  loco_obs (proprio + scan, cmd replaced) → frozen loco → joints [12]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
import time
import os

from agent_ppo.feature.definition import RolloutStorage


class AlgorithmHierNav:
    """
    Hierarchical navigation algorithm.

    Holds a frozen locomotion model and a trainable navigation model.
    Only the nav model's parameters are updated during training.
    """

    def __init__(
        self,
        nav_model: nn.Module,
        loco_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        logger: Any = None,
        monitor: Any = None,
        # PPO hyperparameters for nav training
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.90,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.02,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        normalize_value_loss: bool = True,
        num_mini_batches: int = 8,
        num_learning_epochs: int = 3,
        desired_kl: float = 0.01,
        schedule: str = "adaptive",
        # Entropy decay
        entropy_coef_end: float = 0.001,
        entropy_decay_steps: int = 10000,
        # Command injection: indices of velocity cmd [vx,vy,wz] in proprio obs
        cmd_indices: tuple[int, int] = (9, 12),
        # Proprio dimensions for nav_obs extraction
        num_proprio_obs: int = 45,
        num_nav_proprio: int = 12,
        num_nav_critic_body: int = 9,
        **kwargs,
    ):
        self.device = device
        self.nav_model = nav_model
        self.loco_model = loco_model
        self.optimizer = optimizer
        self.logger = logger
        self.monitor = monitor

        # Freeze locomotion model
        # 冻结运控模型
        for p in self.loco_model.parameters():
            p.requires_grad = False
        self.loco_model.eval()

        # PPO hyperparameters
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_value_loss = normalize_value_loss
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.desired_kl = desired_kl
        self.schedule = schedule

        # Entropy decay
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay_steps = entropy_decay_steps

        # Command injection indices
        self.cmd_start, self.cmd_end = cmd_indices

        # Proprio layout for nav obs extraction
        self.num_proprio_obs = num_proprio_obs
        self.num_nav_proprio = num_nav_proprio
        self.num_nav_critic_body = num_nav_critic_body

        # Nav model min std
        from agent_ppo.conf.conf import Config

        self.min_std = torch.tensor(Config.CURRENT.min_normalized_std[:3], device=device)

        # Training state
        self.train_step = 0
        self.last_report_monitor_time = 0
        self.storage = None

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        device: torch.device = None,
    ):
        device = device or self.device
        self.storage = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            device=device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> tuple:
        """
        Hierarchical act: nav → velocity_cmd → loco → joint_actions.

        Args:
            obs: [B, num_nav_obs] observation with goal info (305 dim for track)
            critic_obs: [B, num_critic_obs] critic observation

        Returns:
            (joint_actions, nav_values, nav_log_probs, nav_mu, nav_sigma)
        """
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            # 1. Nav model: slim obs → velocity_cmd [vx, vy, wz]
            nav_obs = self._build_nav_obs(obs)
            nav_critic_obs = self._build_nav_critic_obs(critic_obs)
            nav_actions = self.nav_model.act(nav_obs)
            nav_values = self.nav_model.evaluate(nav_critic_obs)
            nav_log_probs = self.nav_model.get_actions_log_prob(nav_actions)
            nav_mu = self.nav_model.action_mean.detach()
            nav_sigma = self.nav_model.action_std.detach()

            # 2. Build loco observation: replace velocity command in proprio
            loco_obs = self._build_loco_obs(obs, nav_actions)

            # 3. Frozen loco: loco_obs → joint_actions (deterministic)
            joint_actions = self.loco_model.act_inference(loco_obs)

        return joint_actions, nav_values, nav_log_probs, nav_mu, nav_sigma

    def _build_loco_obs(self, obs: torch.Tensor, velocity_cmd: torch.Tensor) -> torch.Tensor:
        """
        Build locomotion observation by replacing velocity command in proprio.

        obs layout: [proprio(45) | height_scan(256) | goal(4) | ...] = 305+
        loco_obs layout: [proprio(45, cmd replaced) | height_scan(256)] = 301

        The velocity cmd [vx, vy, wz] sits at proprio[self.cmd_start:self.cmd_end].
        """
        loco_obs = obs[:, :self.num_proprio_obs + 256].clone()
        loco_obs[:, self.cmd_start : self.cmd_end] = velocity_cmd
        return loco_obs

    def _build_nav_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract nav actor obs: base_body(6) + scan + goal + ...
        obs[:, :6] = base_ang_vel(3) + projected_gravity(3), no cmd, no joints.
        导航 actor 观测：基础机身信息 + 扫描 + goal，不含速度指令和关节。
        """
        return torch.cat([obs[:, : self.num_nav_proprio], obs[:, self.num_proprio_obs:]], dim=-1)

    def _build_nav_critic_obs(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """
        Extract nav critic obs: privileged body + scan + goal + ...
        critic_obs: [critic_proprio(60) | scan(256) | goal(4) | ...]
        Body: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) = 9
        导航 critic 观测：特权机身信息 + 扫描 + goal，去掉关节和运控特权信息。
        """
        return torch.cat(
            [critic_obs[:, : self.num_nav_critic_body], critic_obs[:, 60:]],
            dim=-1,
        )

    def compute_gae_returns(self, last_critic_obs: torch.Tensor):
        with torch.no_grad():
            nav_critic_obs = self._build_nav_critic_obs(last_critic_obs)
            last_values = self.nav_model.evaluate(nav_critic_obs)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self) -> tuple:
        """
        Train nav policy using PPO. Loco model is frozen and not updated.
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_grad_norm = 0
        kl_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for sample_idx, sample in enumerate(generator):
            (
                obs_batch,
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
            ) = sample

            # Forward pass through NAV model (slim obs)
            nav_obs_batch = self._build_nav_obs(obs_batch)
            nav_critic_obs_batch = self._build_nav_critic_obs(critic_obs_batch)
            self.nav_model.update_distribution(nav_obs_batch)
            actions_log_prob_batch = self.nav_model.get_actions_log_prob(actions_batch)
            entropy_batch = self.nav_model.entropy
            value_batch = self.nav_model.evaluate(nav_critic_obs_batch)
            mu_batch = self.nav_model.action_mean
            sigma_batch = self.nav_model.action_std

            # Adaptive learning rate (captures KL for reporting)
            kl_val = self._update_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)
            if kl_val is not None:
                mean_kl += kl_val
                kl_updates += 1

            # Entropy decay
            self._update_entropy_coef()

            # Compute losses
            surrogate_loss = self._compute_surrogate_loss(
                actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch
            )
            value_loss = self._compute_value_loss(value_batch, returns_batch, target_values_batch)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # NaN/Inf guard
            if not torch.isfinite(loss):
                if self.logger:
                    self.logger.warning(
                        f"[HierNav] NaN/Inf loss at step {self.train_step}, "
                        f"mini-batch {sample_idx}. Skipping."
                    )
                continue

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient NaN guard
            grad_finite = True
            for p in self.nav_model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break

            if not grad_finite:
                if self.logger:
                    self.logger.warning(
                        f"[HierNav] NaN/Inf gradient at step {self.train_step}, "
                        f"mini-batch {sample_idx}. Skipping optimizer.step()."
                    )
                self.optimizer.zero_grad()
                continue

            grad_norm = nn.utils.clip_grad_norm_(self.nav_model.parameters(), self.max_grad_norm)
            if grad_norm is not None and torch.isfinite(grad_norm):
                mean_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            self.optimizer.step()

            # Clamp nav std
            if hasattr(self.nav_model, "std") and self.min_std is not None:
                max_std_t = torch.full_like(self.nav_model.std.data, 1.0e6)
                safe_std = torch.nan_to_num(self.nav_model.std.data, nan=0.5, posinf=1.0e6, neginf=0.0)
                self.nav_model.std.data.copy_(torch.clamp(safe_std, min=self.min_std, max=max_std_t))

            sl = surrogate_loss.item()
            vl = value_loss.item()
            el = entropy_batch.mean().item()
            mean_surrogate_loss += sl if not (sl != sl) else 0.0
            mean_value_loss += vl if not (vl != vl) else 0.0
            mean_entropy_loss += el if not (el != el) else 0.0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl = (mean_kl / kl_updates) if kl_updates > 0 else 0.0
        mean_grad_norm /= num_updates

        self._report_training_metrics(mean_surrogate_loss, mean_value_loss, mean_entropy_loss, mean_kl, mean_grad_norm)

        self.train_step += 1
        return mean_surrogate_loss, mean_value_loss, mean_entropy_loss

    # ------------------------------------------------------------------
    # Helpers (shared with AlgorithmPPO)
    # ------------------------------------------------------------------

    def _update_learning_rate(self, mu_batch, sigma_batch, old_mu_batch, old_sigma_batch):
        if self.desired_kl is None or self.schedule != "adaptive":
            return None

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

            return kl_mean.item()

    def _update_entropy_coef(self):
        if self.train_step >= self.entropy_decay_steps:
            self.entropy_coef = self.entropy_coef_end
            return

        progress = self.train_step / self.entropy_decay_steps
        self.entropy_coef = self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end)

    def _compute_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(self, value_batch, returns_batch, target_values_batch):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            raw_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            raw_loss = (returns_batch - value_batch).pow(2).mean()

        if self.normalize_value_loss:
            returns_var = returns_batch.detach().var() + 1e-8
            return raw_loss / returns_var

        return raw_loss

    def _report_training_metrics(self, mean_surrogate_loss, mean_value_loss, mean_entropy_loss, mean_kl=0.0, mean_grad_norm=0.0):
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "policy_loss": mean_surrogate_loss,
                "value_loss": mean_value_loss,
                "entropy_loss": mean_entropy_loss,
                "total_loss": mean_surrogate_loss + mean_value_loss + mean_entropy_loss,
                "learning_rate": self.learning_rate,
                "kl_divergence": mean_kl,
                "grad_norm": mean_grad_norm,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now
