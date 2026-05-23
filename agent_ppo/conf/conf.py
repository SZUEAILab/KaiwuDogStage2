#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os

import toml


# Valid task types (Isaac Lab native config format)
# 有效任务类型（Isaac Lab 原生配置格式）
_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """
    Base class for training stage configuration.
    训练阶段配置基类。

    Subclass this and override fields to define a new training stage.
    继承此类并覆盖字段来定义新的训练阶段。
    """

    # --- Stage identity
    # 阶段标识 ---
    name = ""
    task_type = "standard"

    # --- Model architecture dimensions (Isaac Lab Unitree-Go2-Velocity constants)
    # These are fixed by the Isaac Lab task definition and the network structure;
    # users are not expected to change them. Do NOT move them into user TOML.
    # 模型架构维度（Isaac Lab Unitree-Go2-Velocity 常量）
    # 由 Isaac Lab 任务定义与网络结构决定，用户不应修改；也不应放进用户 TOML。
    num_actions = 12  # Go2 joint action dim / Go2 关节动作维度
    num_proprio_obs = 45  # proprioceptive obs dim / 本体感知观测维度
    num_scan = 256  # 16x16 height-scan dim / 16x16 高度扫描维度
    num_critic_observations = 316  # proprio(45) + scan(256) + privileged(15)
    num_goal_obs = 0  # standard locomotion has no goal; track stages override with 4
    num_nav_scan = 0  # nav_scanner rays (track mode, wider range); set >0 to append to obs

    # --- Model architecture
    # 模型架构 ---
    model_class = "ActorCritic"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # --- Training hyperparameters
    # 训练超参数 ---
    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    # --- Saving
    # 保存 ---
    model_save_interval = 500


class CustomConfig(StageConfig):
    # TODO: you can refer to LocomotionConfig to design your own track-terrain
    # navigation training stage. The following items need to be specified:
    # 1. stage name;
    # 2. task_type;
    # 3. whether to use hierarchical training;
    # 4. semantics and dimension of the policy action;
    # 5. obs dimension (whether to concatenate goal information);
    # 6. training hyperparameters.
    #
    # After adding a new training stage, a corresponding training config file
    # must be created in the same directory.
    # Filename convention: train_env_conf_<task_type>_<stage.name>.toml
    # Refer to train_env_conf_standard_locomotion.toml as an example.
    #
    # TODO：可参考 LocomotionConfig 自行设计 track 地形导航训练阶段。
    # 需要明确：
    # 1. stage 名称；
    # 2. task_type；
    # 3. 是否采用分层训练；
    # 4. policy action 的语义和维度；
    # 5. obs 维度（是否拼接 goal 信息）；
    # 6. 训练超参。
    #
    # 新增训练阶段后，需在同目录创建对应训练配置文件。
    # 文件命名规则：train_env_conf_<task_type>_<stage.name>.toml
    # 可参考 train_env_conf_standard_locomotion.toml。
    pass


class LocomotionConfig(StageConfig):
    """
    Stage: locomotion — learn stable walking on mixed terrain.
    阶段：locomotion —— 在混合地形上学习稳定行走。
    """

    name = "locomotion"
    task_type = "standard"


class StairsDownConfig(StageConfig):
    """
    Stage: stairs_down — focused training on descending stairs (pyramid_stairs_inv).
    阶段：stairs_down —— 专注训练下台阶（pyramid_stairs_inv）。
    """

    name = "stairs_down"
    task_type = "standard"

    # Lower learning rate for stable convergence
    lr = 1e-4

    # More steps per env for better data collection
    num_steps_per_env = 64

    # Fewer epochs to prevent overfitting
    num_learning_epochs = 3

    # Larger batch for stable updates
    num_mini_batches = 8

    # Higher min std to prevent policy collapse
    min_normalized_std = [0.1, 0.05, 0.1] * 4


class AllTerrainConfig(StageConfig):
    """
    Stage: all_terrain — comprehensive locomotion training on all sub-terrains.
    阶段：all_terrain —— 在所有子地形上综合训练基本运动策略。

    Uses all 5 sub-terrain types and comprehensive rewards to learn
    robust locomotion (forward, lateral, turning, stair ascent/descent).
    使用全部 5 种子地形和综合奖励，学习鲁棒的运动策略。
    """

    name = "all_terrain"
    task_type = "standard"

    # Moderate learning rate for stable all-terrain training
    lr = 3e-4

    # Standard rollout length
    num_steps_per_env = 48

    # Standard epoch/mini-batch split
    num_learning_epochs = 5
    num_mini_batches = 4

    # Standard noise
    min_normalized_std = [0.05, 0.02, 0.05] * 4


class MazeConfig(StageConfig):
    """
    Stage: maze — focused training on maze terrain only.
    阶段：maze —— 专注训练迷宫地形。

    Maze is flat terrain with wall obstacles — the focus is on learning
    to turn and navigate around walls, not on slopes or stairs.
    迷宫为带墙壁障碍的平坦地形，重点学习绕墙转向而非坡道台阶。
    """

    name = "maze"
    task_type = "standard"


class TrackNavConfig(StageConfig):
    """
    Stage: nav — track-terrain navigation training (end-to-end).
    阶段：nav —— track 地形导航训练（端到端）。

    Joint locomotion + navigation training on track terrain with
    navigation-focused rewards. Locomotion rewards are down-weighted
    so the navigation signal (approach goal, avoid obstacles) dominates.
    在 track 地形上联合训练运控 + 导航，导航奖励权重更高，
    导航信号（接近目标、避障）主导训练。

    For true hierarchical nav (frozen locomotion + trainable nav policy),
    use the hierarchical config variant after standard locomotion pretraining.
    真正的层级导航（冻结运控 + 可训练导航策略）需在 standard 预训练后使用。
    """

    name = "nav"
    task_type = "track"

    # Navigation-relevant observation: adds 4D goal (robot-frame xyz + distance)
    # 导航相关观测：增加 4 维 goal（机器人坐标系 xyz + 距离）
    num_goal_obs = 4
    num_critic_observations = 320  # critic_base(316) + goal(4)

    # Lower learning rate for stable policy adaptation
    lr = 1e-4

    # Longer rollout for track terrain (longer episodes, more data)
    num_steps_per_env = 64

    # Fewer epochs to prevent overfitting on nav-specific data
    num_learning_epochs = 3

    # Larger mini-batches for stable updates
    num_mini_batches = 8

    # Preserve exploration for navigation
    min_normalized_std = [0.08, 0.03, 0.08] * 4


class HierTrackNavConfig(StageConfig):
    """
    Stage: hier_nav — hierarchical nav (frozen locomotion + trainable nav).
    阶段：hier_nav —— 层级式导航（冻结运控 + 可训练导航策略）。

    Nav policy outputs 3D velocity commands [vx, vy, wz]; frozen locomotion
    policy converts them to 12-DOF joint actions.
    导航策略输出 3D 速度指令 [vx, vy, wz]；冻结的运控策略将其转为 12-DOF
    关节动作。仅训练导航策略，运控参数不更新。

    Requires a pretrained locomotion checkpoint from a prior standard stage.
    需要从 prior standard 阶段获取预训练运控 checkpoint。
    """

    name = "hier_nav"
    task_type = "track"
    model_class = "NavActorCritic"

    # Nav model: 3D velocity command output
    # 导航模型：输出 3D 速度指令
    num_actions = 3
    num_goal_obs = 4
    num_critic_observations = 320  # critic_base(316) + goal(4)

    # Nav model — smaller than loco (high-level decision)
    # 导航模型 — 比运控小（高层决策）
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]

    # Velocity command indices in proprio observation [start, end)
    # 速度指令在 proprio 观测中的索引 [start, end)
    cmd_indices = (9, 12)

    # Lower learning rate for stable nav training
    lr = 1e-4

    # Longer rollout for track terrain
    num_steps_per_env = 64

    # Fewer epochs to prevent overfitting
    num_learning_epochs = 3

    # Larger mini-batches for stable updates
    num_mini_batches = 8

    # Preserve exploration for navigation
    min_normalized_std = [0.15, 0.08, 0.25]  # [vx, vy, wz]


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。
    """

    # Switch stage by changing CURRENT
    # 通过修改 CURRENT 切换阶段
    CURRENT = HierTrackNavConfig

    @staticmethod
    def load_conf(logger):
        """
        Load user configuration file based on current stage.
        根据当前阶段加载用户配置文件。

        Args:
            logger: logger instance | 日志实例

        Returns:
            tuple: (usr_conf, usr_conf_file, is_eval, stage)
        """
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'. " f"Only {_VALID_TASKS} are supported."
            )

        # Determine if it's evaluation mode
        # 判断是否为评估模式
        is_eval = False
        if hasattr(CONFIG, "run_mode"):
            is_eval = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval:
            usr_conf_file = f"tools/eval/conf/eval_env_conf.toml"
        else:
            usr_conf_file = f"agent_ppo/conf/train_env_conf_{task_type}_{stage.name}.toml"

        usr_conf = _load_conf(usr_conf_file, logger)

        if usr_conf is None:
            error_msg = f"usr_conf is None, please check {usr_conf_file}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, model: {stage.model_class}")

        return usr_conf, usr_conf_file, is_eval, stage


def _deep_merge(base, override):
    """
    Recursively merge override dict into base dict.
    递归将 override 字典合并到 base 字典中（override 优先）。

    Args:
        base: Base config dictionary | 基础配置字典
        override: Override config dictionary | 覆盖配置字典

    Returns:
        dict: Merged config dictionary
    """
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_conf(conf_file, logger):
    """
    Load config: first load base TOML, then deep-merge user TOML on top.
    加载配置：先加载 base TOML，再用用户 TOML 覆盖合并。

    Base files provide model architecture dimensions (num_actions, num_proprio_obs, etc.)
    so user configs only need business-tunable parameters.
    Base 文件提供模型架构维度参数，用户配置只需保留业务可调参数。

    Args:
        conf_file: Path to the user TOML config file | 用户配置文件路径
        logger: Logger instance | 日志实例

    Returns:
        dict: Merged config dictionary, or None on failure
    """
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    # Determine base file by mode (eval or train)
    # 根据模式选择 base 文件（eval 或 train）
    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    # Load base config (optional — missing base is not fatal)
    # 加载 base 配置（可选 — base 缺失不致命）
    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    # Load user config
    # 加载用户配置
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    # Deep merge: base ← user (user wins)
    # 深度合并：base ← user（用户配置优先）
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config
