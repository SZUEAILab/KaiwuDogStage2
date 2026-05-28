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
from agent_ppo.model.loco_actor_critic import LocoActorCritic
from agent_ppo.model.nav_network import NavActorCritic
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.algorithm.algorithm_hier_nav import AlgorithmHierNav
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "LocoActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self._cached_nav_state = None

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
        self.model = LocoActorCritic(
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
            cmd_upper=getattr(stage, "cmd_upper", None),
            cmd_lower=getattr(stage, "cmd_lower", None),
        ).to(self.device)

        self.logger.info(f"Nav Actor: {self.nav_model.actor}")
        self.logger.info(f"Nav Critic: {self.nav_model.critic}")

        # Frozen locomotion model (LocoActorCritic, 12-DOF output).
        # loco receives proprio(45) + scan(256) = 301 dim — goal info is stripped
        # by _build_loco_obs since loco doesn't need navigation targets.
        # 冻结运控模型（LocoActorCritic，12-DOF 输出）。
        # loco 输入为 proprio(45) + scan(256) = 301 维 — _build_loco_obs
        # 会去掉 goal，因为运控不需要导航目标。
        self.loco_model = LocoActorCritic(
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
            cmd_indices=getattr(stage, "cmd_indices", (6, 9)),
            num_proprio_obs=num_proprio,
            num_scan=num_scan,
            num_goal_obs=self.num_goal_obs,
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
                nav_actions = self.nav_model.act(nav_obs)
                self._cached_nav_result = nav_actions.detach()
            else:
                nav_actions = self._cached_nav_result

            self._nav_step_counter = counter + 1

            # Recompute training statistics against the current observation even
            # when holding a decimated command from a previous nav step.
            self.nav_model.update_distribution(nav_obs)
            nav_values = self.nav_model.evaluate(
                self.algorithm._build_nav_critic_obs(critic_obs)
            )
            nav_log_probs = self.nav_model.get_actions_log_prob(nav_actions)
            nav_mu = self.nav_model.action_mean.detach()
            nav_sigma = self.nav_model.action_std.detach()

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
        Save model checkpoint — always in hierarchical format.
        保存模型 checkpoint —— 始终使用分层格式。

        Flat mode upgrades to hierarchical: self.model → loco_model,
        cached nav (if any) → nav_model.
        扁平模式自动升级为分层格式：self.model → loco_model，
        缓存的 nav（如有）→ nav_model。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if getattr(self, "is_hierarchical", False):
            state = {
                "nav_model": self.nav_model.state_dict(),
                "loco_model": self.loco_model.state_dict(),
            }
        else:
            state = {
                "loco_model": self.model.state_dict(),
                "nav_model": self._cached_nav_state if self._cached_nav_state is not None else {},
            }
            self.logger.info("[FlatSave] Upgrading to hierarchical format on save")
        torch.save(state, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        Load model checkpoint — unified path, always treats result as hierarchical.
        加载模型 checkpoint —— 统一路径，始终按分层结构处理。

        Detects format automatically:
        - Hierarchical {"loco_model": ..., "nav_model": ...} → use as-is
        - Legacy flat → upgrade: wrap as loco_model, nav empty
        自动识别格式：分层直接用，扁平旧格式升级为分层。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        self._load_weights(pretrained, model_file_path)
        self.cur_model_name = model_file_path

    def _load_weights(self, pretrained, model_file_path):
        """
        Unified weight loading — auto-detect format, flat is upgraded to hierarchical.
        统一权重加载 —— 自动检测格式，扁平升级为分层。
        """
        is_hier_ckpt = isinstance(pretrained, dict) and "loco_model" in pretrained

        if is_hier_ckpt:
            loco_state = pretrained["loco_model"]
            nav_state = pretrained.get("nav_model", None)
            self.logger.info(f"[Load] Hierarchical checkpoint at {model_file_path}")
        else:
            loco_state = pretrained
            nav_state = None
            self.logger.warning(
                f"[Load] Legacy flat checkpoint at {model_file_path} → upgrading to hierarchical"
            )

        # Load loco into the right target
        loco_target = self.loco_model if getattr(self, "is_hierarchical", False) else self.model
        self._load_component_weights(loco_target, loco_state, model_file_path, component="loco")

        # Load or cache nav
        if getattr(self, "is_hierarchical", False):
            if nav_state:
                self._load_component_weights(self.nav_model, nav_state, model_file_path, component="nav")
            else:
                self.logger.warning("[Load] No nav_model in checkpoint — nav will train from scratch")
        else:
            self._cached_nav_state = nav_state
            if nav_state:
                self.logger.info("[Load] nav_model cached — will be preserved on save")

    def _load_component_weights(self, model, pretrained, model_file_path, component="model"):
        """Load weights into a model component with mismatch detection and logging."""
        current_state = model.state_dict()

        has_mismatch = set(pretrained.keys()) != set(current_state.keys())
        for key in pretrained:
            if key in current_state and pretrained[key].shape != current_state[key].shape:
                has_mismatch = True
                break

        if not has_mismatch:
            model.load_state_dict(pretrained)
            self.logger.info(f"[Load] {component} weights loaded (exact match) from {model_file_path}")
        else:
            self._load_model_partial(model, pretrained, model_file_path)
            self.logger.info(f"[Load] {component} weights loaded (partial) from {model_file_path}")

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
