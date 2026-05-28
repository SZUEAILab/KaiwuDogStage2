#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
NavActorCritic — Actor-Critic for navigation policy.

Outputs 3D velocity commands [vx, vy, wz] instead of 12-DOF joint actions.
Used in hierarchical nav: nav policy → velocity cmd → frozen loco policy → joints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any

from agent_ppo.model.actor_critic import resolve_nn_activation


class NavActorCritic(nn.Module):
    """
    Actor-Critic network for navigation policy.
    导航策略的 Actor-Critic 网络。

    Actor outputs 3D velocity commands [vx, vy, wz].
    Critic outputs scalar state value.
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int = 3,
        actor_hidden_dims: tuple[int] | list[int] = (256, 128, 64),
        critic_hidden_dims: tuple[int] | list[int] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        cmd_upper: list[float] | None = None,
        cmd_lower: list[float] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        activation_fn = resolve_nn_activation(activation)

        # cmd_lower / cmd_upper: per-dim output bounds.
        # Network output: raw ∈ [-1,1] (tanh), then affine → [cmd_lower, cmd_upper].
        # 网络输出: raw ∈ [-1,1] (tanh)，仿射映射到 [cmd_lower, cmd_upper]。
        if cmd_upper is None:
            cmd_upper = [2.0, 1.5, 1.5]
        cmd_upper_t = torch.tensor(cmd_upper, dtype=torch.float32)
        if cmd_lower is None:
            cmd_lower_t = -cmd_upper_t
        else:
            cmd_lower_t = torch.tensor(cmd_lower, dtype=torch.float32)
        self.register_buffer("cmd_lower", cmd_lower_t)
        self.register_buffer("cmd_upper", cmd_upper_t)
        half_range = (cmd_upper_t - cmd_lower_t) / 2.0
        mid = (cmd_upper_t + cmd_lower_t) / 2.0
        self.register_buffer("cmd_half_range", half_range)
        self.register_buffer("cmd_mid", mid)

        # Actor MLP: obs → velocity_cmd [vx, vy, wz]
        # 策略网络：obs → 速度指令 [vx, vy, wz]
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
                actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(nn.LayerNorm(actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
        # 价值网络
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        # Orthogonal weight initialization
        # 正交权重初始化
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Action noise
        # 动作噪声
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs: torch.Tensor):
        raw = self.actor(obs)  # tanh → [-1, 1]
        mean = raw * self.cmd_half_range + self.cmd_mid
        if self.noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs)
        action = self.distribution.sample()
        return torch.clamp(action, self.cmd_lower, self.cmd_upper)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor(obs)
        return raw * self.cmd_half_range + self.cmd_mid

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def forward(self):
        raise NotImplementedError
