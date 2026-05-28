#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.feature.definition import ActData
from agent_ppo.conf.conf import Config
from agent_ppo.model.actor_critic import ActorCritic
from agent_ppo.model.nav_network import NavActorCritic
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.algorithm.algorithm_hier_nav import AlgorithmHierNav
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "ActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

        self.stage = stage
        env_conf = usr_conf["env"]
        self.num_envs = env_conf["num_envs"]

        # Model architecture dims come from StageConfig (architecture constants,
        # not user-tunable business params). Do NOT read them from TOML.
        # 模型架构维度来自 StageConfig（架构常量，非业务可调参数），不从 TOML 读。
        self.num_actions = stage.num_actions

        num_proprio = stage.num_proprio_obs
        num_scan = stage.num_scan

        self.num_goal_obs = getattr(stage, "num_goal_obs", 0)
        self.num_nav_proprio = getattr(stage, "num_nav_proprio", 12)
        # nav critic body: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) = 9
        self.num_nav_critic_body = self.num_nav_proprio + 3

        # policy obs = proprio + scan + optional goal
        # 策略观测 = 本体感知 + 扫描 + 可选 goal
        self.num_obs = num_proprio + num_scan + self.num_goal_obs
        self.num_critic_obs = stage.num_critic_observations + self.num_goal_obs

        # nav obs = base_body (no cmd, no joints) + scan + optional goal
        # 导航 actor 观测 = 基础机身信息（无 cmd，无关节）+ 扫描 + 可选 goal
        self.nav_obs_dim = self.num_nav_proprio + num_scan + self.num_goal_obs
        # nav critic obs = same scan/goal + privileged body state (9 vs 6)
        self.nav_critic_obs_dim = self.num_nav_critic_body + num_scan + self.num_goal_obs

        if stage.model_class == "NavActorCritic":
            self._init_hier(num_proprio, num_scan, stage, usr_conf)
        else:
            self._init_flat(num_proprio, num_scan, stage)

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        # Initialize storage
        # 初始化存储
        self.algorithm.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            actor_obs_shape=(self.num_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.num_actions,),
            device=self.device,
        )

        super().__init__(agent_type, device, logger, monitor)

    def _init_flat(self, num_proprio, num_scan, stage):
        """
        Initialize single-model (flat) architecture.
        初始化单模型（扁平）架构。
        """
        self.model = ActorCritic(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
        ).to(self.device)

        self.logger.info(f"Actor MLP: {self.model.actor}")
        self.logger.info(f"Critic MLP: {self.model.critic}")

        params = [{"params": self.model.parameters(), "name": "actor_critic"}]
        self.optimizer = optim.Adam(params, lr=stage.lr)

        self.algorithm = AlgorithmPPO(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            learning_rate=stage.lr,
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
        )

    def _init_hier(self, num_proprio, num_scan, stage, usr_conf):
        """
        Initialize hierarchical architecture: frozen loco + trainable nav.
        初始化层级架构：冻结运控 + 可训练导航。
        """
        # Nav model: uses slim obs (base proprio only, no joint info).
        # 导航模型：精简观测（仅基础机身信息，无关节信息）。
        self.nav_model = NavActorCritic(
            num_obs=self.nav_obs_dim,
            num_critic_obs=self.nav_critic_obs_dim,
            num_actions=stage.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            cmd_scale=getattr(stage, "cmd_scale", None),
        ).to(self.device)

        self.logger.info(f"Nav Actor: {self.nav_model.actor}")
        self.logger.info(f"Nav Critic: {self.nav_model.critic}")

        # Frozen locomotion model (standard ActorCritic, 12-DOF output).
        # loco receives proprio(45) + scan(256) = 301 dim — goal info is stripped
        # by _build_loco_obs since loco doesn't need navigation targets.
        # 冻结运控模型（标准 ActorCritic，12-DOF 输出）。
        # loco 输入为 proprio(45) + scan(256) = 301 维 — _build_loco_obs
        # 会去掉 goal，因为运控不需要导航目标。
        self.loco_model = ActorCritic(
            num_obs=num_proprio + num_scan,
            num_critic_obs=316,  # base critic dim (no goal)
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation=stage.activation,
        ).to(self.device)

        self.logger.info(f"Loco Actor (frozen): {self.loco_model.actor}")

        # Optimizer: only nav model parameters
        # 优化器：仅导航模型参数
        params = [{"params": self.nav_model.parameters(), "name": "nav_model"}]
        self.optimizer = optim.Adam(params, lr=stage.lr)

        # Hierarchical algorithm
        # 层级算法
        self.algorithm = AlgorithmHierNav(
            nav_model=self.nav_model,
            loco_model=self.loco_model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            learning_rate=stage.lr,
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
            cmd_indices=getattr(stage, "cmd_indices", (9, 12)),
            num_proprio_obs=num_proprio,
            num_nav_proprio=self.num_nav_proprio,
            num_nav_critic_body=self.num_nav_critic_body,
        )

        self.is_hierarchical = True

        # Nav decimation: cache velocity command between nav updates
        # Nav 降频：在两次 nav 更新之间缓存速度指令
        self._nav_decimation = usr_conf.get("policy", {}).get("nav_decimation", 1)
        self._nav_step_counter = 0
        self._cached_nav_result = None

    def reset_nav_decimation(self):
        """Reset nav decimation counter (call at episode start).
        重置 nav 降频计数器（episode 开始时调用）。
        """
        self._nav_step_counter = 0
        self._cached_nav_result = None

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        (obs) = list_obs_data
        with torch.no_grad():
            if getattr(self, "is_hierarchical", False):
                # Hierarchical: nav → velocity_cmd → loco → joints (deterministic)
                nav_obs = self.algorithm._build_nav_obs(obs)
                nav_actions = self.nav_model.act_inference(nav_obs)
                loco_obs = self.algorithm._build_loco_obs(obs, nav_actions)
                actions = self.loco_model.act_inference(loco_obs)
            else:
                actions = self.algorithm.actor_critic.act_inference(obs)
            return [ActData(action=actions)]

    def learn(self, list_sample_data=None):
        """
        Trigger learning process using sample data.
        使用样本数据触发学习过程。

        Note: AlgorithmPPO.learn() doesn't take batch_data as argument anymore.
        It reads from its internal storage that was filled by workflow's run_episodes_.
        注：AlgorithmPPO.learn() 不再接受 batch_data 参数，
        而是直接读取 workflow 的 run_episodes_ 填充的内部存储。
        """
        return self.algorithm.learn()

    def predict(self, list_obs_data):
        """
        Generate predictions with actor-critic network.
        使用 actor-critic 网络生成预测。

        Hierarchical mode: nav(obs) → velocity_cmd → loco(loco_obs) → joints.
        Returns joint_actions for env.step(), but values/log_probs from nav for training.
        层级模式：nav(obs) → velocity_cmd → loco(loco_obs) → joints。
        返回 joint_actions 用于 env.step()，但 values/log_probs 来自 nav（用于训练）。
        """
        (obs, critic_obs) = list_obs_data

        if getattr(self, "is_hierarchical", False):
            return self._predict_hier(obs, critic_obs)
        else:
            return self._predict_flat(obs, critic_obs)

    def _predict_flat(self, obs, critic_obs):
        """Standard single-model predict.
        Returns (actions, values, log_probs, mu, sigma, obs, critic_obs, None).
        The last element (nav_actions) is None in flat mode.
        """
        with torch.no_grad():
            actions = self.algorithm.actor_critic.act(obs)
            values = self.algorithm.actor_critic.evaluate(critic_obs)
            log_probs = self.algorithm.actor_critic.get_actions_log_prob(actions)
            action_mean = self.algorithm.actor_critic.action_mean.detach()
            action_std = self.algorithm.actor_critic.action_std.detach()

            return (
                actions,
                values,
                log_probs,
                action_mean,
                action_std,
                obs.detach(),
                critic_obs.detach(),
                None,  # nav_actions — None in flat mode
            )

    def _predict_hier(self, obs, critic_obs):
        """Hierarchical predict: nav → cmd → loco → joints.
        Returns (joint_actions, nav_values, nav_log_probs, nav_mu, nav_sigma, obs, critic_obs, nav_actions).
        joint_actions (12-DOF) → env.step(); nav_actions (3D velocity cmd) → storage.

        Supports nav_decimation: when decimation > 1, nav model only runs every N steps,
        holding the velocity command constant between updates for smoother control.
        支持 nav 降频：降频 > 1 时 nav 每 N 步推理一次，持速指令不变，控制更平滑。
        """
        with torch.no_grad():
            decimation = getattr(self, "_nav_decimation", 1)
            counter = self._nav_step_counter

            nav_obs = self.algorithm._build_nav_obs(obs)

            if counter % decimation == 0 or self._cached_nav_result is None:
                # 1. Nav model forward pass (slim obs: base proprio + scan + goal)
                self.nav_model.update_distribution(nav_obs)
                nav_actions = self.nav_model.distribution.sample()
                nav_values = self.nav_model.evaluate(
                    self.algorithm._build_nav_critic_obs(critic_obs)
                )
                nav_log_probs = self.nav_model.get_actions_log_prob(nav_actions)
                nav_mu = self.nav_model.action_mean.detach()
                nav_sigma = self.nav_model.action_std.detach()
                self._cached_nav_result = (nav_actions, nav_values, nav_log_probs, nav_mu, nav_sigma)
            else:
                nav_actions, nav_values, nav_log_probs, nav_mu, nav_sigma = self._cached_nav_result

            self._nav_step_counter = counter + 1

            # 2. Build loco observation with nav's velocity command
            loco_obs = self.algorithm._build_loco_obs(obs, nav_actions)

            # 3. Frozen loco: deterministic inference → joint actions
            joint_actions = self.loco_model.act_inference(loco_obs)

            return (
                joint_actions,
                nav_values,
                nav_log_probs,
                nav_mu,
                nav_sigma,
                obs.detach(),
                critic_obs.detach(),
                nav_actions.detach(),
            )

    def save_model(self, path=None, id="1"):
        """
        Save model checkpoint.
        保存模型 checkpoint。

        Hierarchical mode: saves nav model (trainable).
        Flat mode: saves the single ActorCritic.
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if getattr(self, "is_hierarchical", False):
            state = {
                "nav_model": self.nav_model.state_dict(),
                "loco_model": self.loco_model.state_dict(),
            }
        else:
            state = self.model.state_dict()
        torch.save(state, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        Load model checkpoint.
        加载模型 checkpoint。

        Hierarchical mode: loads loco_model + optional nav_model from hierarchical
        checkpoint, or falls back to loading loco from a flat locomotion checkpoint.
        If nav_model weights are missing in the checkpoint, logs the absence.
        层级模式：从层级 checkpoint 加载 loco_model + 可选 nav_model，
        或从扁平运控 checkpoint 加载 loco。缺失 nav 权重时记录日志。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)

        if getattr(self, "is_hierarchical", False):
            self._load_hier_weights(pretrained, model_file_path)
        else:
            self._load_flat_weights(pretrained, model_file_path)

        self.cur_model_name = model_file_path

    def _load_hier_weights(self, pretrained, model_file_path):
        """
        Load weights for hierarchical mode.

        Supports two checkpoint formats:
        1. Hierarchical dict {"loco_model": ..., "nav_model": ...} — loads both.
           Logs which parts are present/missing.
        2. Flat ActorCritic state_dict — treated as loco weights (legacy).
           Logs that nav will train from scratch.
        """
        is_hier_ckpt = isinstance(pretrained, dict) and "loco_model" in pretrained

        if is_hier_ckpt:
            # Hierarchical checkpoint
            loco_state = pretrained["loco_model"]
            self.logger.warning(
                f"[HierLoad] Hierarchical checkpoint detected at {model_file_path}"
            )

            if "nav_model" in pretrained and pretrained["nav_model"]:
                self._load_nav_weights(pretrained["nav_model"])
            else:
                self.logger.warning(
                    "[HierLoad] No nav_model in checkpoint — nav will train from scratch"
                )
        else:
            # Legacy: flat checkpoint → use as loco weights
            loco_state = pretrained
            self.logger.warning(
                f"[HierLoad] Flat checkpoint at {model_file_path} → loading as loco only. "
                "Nav will train from scratch."
            )

        # Load loco weights
        self._load_component_weights(self.loco_model, loco_state, model_file_path, component="loco")

    def _load_nav_weights(self, nav_state):
        """Load nav model weights and log the result."""
        self.logger.warning("[HierLoad] nav_model weights found — loading nav from checkpoint")
        self._load_component_weights(self.nav_model, nav_state, "checkpoint", component="nav")

    def _load_component_weights(self, model, pretrained, model_file_path, component="model"):
        """Load weights into a model component with mismatch detection and logging."""
        current_state = model.state_dict()

        has_mismatch = False
        for key in pretrained:
            if key in current_state and pretrained[key].shape != current_state[key].shape:
                has_mismatch = True
                break

        if not has_mismatch:
            model.load_state_dict(pretrained)
            self.logger.warning(f"[HierLoad] {component} weights loaded (exact match) from {model_file_path}")
        else:
            self._load_model_partial(model, pretrained, model_file_path)
            self.logger.warning(f"[HierLoad] {component} weights loaded (partial) from {model_file_path}")

    def _load_loco_weights(self, pretrained, model_file_path):
        """
        DEPRECATED: use _load_hier_weights instead.
        Kept for backward compatibility.
        """
        if isinstance(pretrained, dict) and "loco_model" in pretrained:
            pretrained = pretrained["loco_model"]

        self._load_component_weights(self.loco_model, pretrained, model_file_path, component="loco")

    def _load_flat_weights(self, pretrained, model_file_path):
        """
        Load weights for flat (single-model) mode.

        Supports:
        1. Flat checkpoint → load directly
        2. Hierarchical checkpoint → extract loco_model, ignore nav
        """
        is_hier_ckpt = isinstance(pretrained, dict) and "loco_model" in pretrained

        if is_hier_ckpt:
            self.logger.warning(
                f"[FlatLoad] Hierarchical checkpoint at {model_file_path} "
                "→ extracting loco_model, nav_model ignored"
            )
            pretrained = pretrained["loco_model"]

        self._load_component_weights(self.model, pretrained, model_file_path, component="model")

    def _load_model_partial(self, model, pretrained, model_file_path):
        """
        Partial checkpoint loading for cross-stage transfer.
        部分加载 checkpoint，用于跨阶段迁移。
        """
        current_state = model.state_dict()
        loaded_keys = []
        partial_keys = []
        skipped_keys = []

        for key in current_state:
            if key not in pretrained:
                skipped_keys.append(key)
                continue

            old_param = pretrained[key]
            new_param = current_state[key]

            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                loaded_keys.append(key)
            else:
                with torch.no_grad():
                    new_param.zero_()
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_param.shape, new_param.shape))
                    new_param[slices] = old_param[slices]
                partial_keys.append(f"{key} {list(old_param.shape)}→{list(new_param.shape)}")

        model.load_state_dict(current_state)

        self.logger.info(
            f"Partial load model {model_file_path}: "
            f"{len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped"
        )
        for info in partial_keys:
            self.logger.info(f"  Partial: {info}")
