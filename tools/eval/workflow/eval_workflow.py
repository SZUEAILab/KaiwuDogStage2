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
import torch


# Eval config file path | 评估配置文件路径
_EVAL_CONF_FILE = "tools/eval/conf/eval_env_conf.toml"
# Eval base config (model architecture dimensions) | 评估 base 配置（模型架构维度）
_EVAL_BASE_FILE = "tools/conf/base/eval_env_base.toml"


def _deep_merge(base, override):
    """Recursively merge override into base (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]

    # 评估开始
    logger.info(".......... Evaluation Start ..........")

    run_episodes(env, agents, logger, monitor)

    # 评估结束
    logger.info(".......... Evaluation End ..........")


def _load_eval_conf(logger):
    """Load evaluation configuration: base TOML + user TOML (deep merge).

    先加载 eval_env_base.toml（模型架构维度），再用 eval_env_conf.toml 覆盖合并。

    Args:
        logger: Logger instance | 日志实例

    Returns:
        dict: Merged config dictionary, or None on failure
    """
    if not os.path.exists(_EVAL_CONF_FILE):
        logger.error(f"Eval config file not found: {_EVAL_CONF_FILE}")
        return None

    # Load base config (provides num_actions, num_proprio_obs, etc.)
    # 加载 base 配置（提供 num_actions、num_proprio_obs 等）
    base_config = {}
    if os.path.exists(_EVAL_BASE_FILE):
        try:
            with open(_EVAL_BASE_FILE, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded eval base config: {_EVAL_BASE_FILE}")
        except Exception as e:
            logger.warning(f"Cannot load eval base config: {_EVAL_BASE_FILE}. Error: {e}")

    # Load user config
    try:
        with open(_EVAL_CONF_FILE, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded eval user config: {_EVAL_CONF_FILE}")
    except Exception as e:
        logger.error(f"Cannot load eval config: {_EVAL_CONF_FILE}. Error: {e}")
        return None

    # Deep merge: base ← user (user wins)
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config


def run_episodes(env, agents, logger, monitor):

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = _load_eval_conf(logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check eval config file")
        return

    # Validate configuration before proceeding | 在继续之前校验配置
    from tools.train_env_conf_validate import check_usr_conf

    valid, message = check_usr_conf(usr_conf, is_eval=True, logger=logger)
    if not valid:
        logger.error(message)
        raise Exception(message)

    # Print evaluation configuration for debugging
    # 打印评估配置用于调试
    logger.info("========== Evaluation Configuration ==========")
    for section, values in usr_conf.items():
        if isinstance(values, dict):
            print(f"[{section}]")
            logger.info(f"[{section}]")
            for key, val in values.items():
                print(f"  {key} = {val}")
                logger.info(f"  {key} = {val}")
        else:
            print(f"{section} = {values}")
            logger.info(f"{section} = {values}")
    logger.info("===============================================")

    agent = agents[0]

    # Determine task type for branching logic
    # 确定任务类型以决定分支逻辑
    task_type = usr_conf.get("env", {}).get("task", "standard")

    # Number of episodes to test per terrain
    # 每个 terrain 测试的 episode 数目
    episode_nums_per_terrain = usr_conf.get("custom_parameters", {}).get("episode_nums_per_terrain", 1)

    episode_cnt = 0
    try:
        # Mark as evaluation mode for base_env
        # 标记为评估模式供 base_env 使用
        usr_conf["is_eval"] = True

        data = env.reset(usr_conf)
        if data is None:
            error_message = f"episode {episode_cnt}, reset failed, please check"
            logger.error(error_message)
            raise Exception(error_message)

        (obs, critic_obs) = data

        # Record the cumulative rewards of the agent in the environment
        # 记录对局中智能体的累积回报，用于上报监控
        logger.info(f"episode {episode_cnt} start, usr_conf is {usr_conf}")

        # Calculate max episode length from config (episode_length_s / step_dt)
        # 从配置计算最大 episode 步数 (episode_length_s / 每步时间)
        # Default: episode_length_s=20.0, dt=0.005, decimation=4 -> step_dt=0.02 -> 1000 steps
        episode_length_s = usr_conf.get("env", {}).get("episode_length_s", 20.0)
        step_dt = 0.02  # dt(0.005) * decimation(4) for Isaac Lab
        max_episode_length = int(episode_length_s / step_dt)

        # Run enough steps to complete episode_nums_per_terrain full episodes
        # 运行足够的步数以完成 episode_nums_per_terrain 个完整 episode
        total_steps = episode_nums_per_terrain * int(max_episode_length) + 1
        for episode_step in range(total_steps):
            episode_cnt += 1
            with torch.inference_mode():
                # Only provide obs to prevent privileged information leakage
                # 只提供 obs 防止特权信息被使用
                act_data = agent.exploit((obs))
                actions = act_data[0].action

                step_data = env.step(actions)
                if step_data is None:
                    error_message = f"episode {episode_cnt}, failed, please check"
                    logger.error(error_message)
                    raise Exception(error_message)
                frame_no, obs, rewards, terminated, truncated, state = step_data

                # Exit when all environments are done
                # 如果环境里面所有的都完成后即结束
                infos, privileged_obs = state
                if "all_done" in infos and infos["all_done"]:
                    logger.info("all done is OK, so return")
                    break

    except Exception as e:
        logger.exception(e)
        raise
    finally:
        # Generate result json before close (scorer is cleared in close)
        env.close()
