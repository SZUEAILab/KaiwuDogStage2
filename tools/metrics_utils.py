#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.metrics_utils import collect_training_metrics
from tools.utils import load_env_keys_from_monitor_config


def get_training_metrics():
    """Fetch training metrics for legged_robot_competition_26 project."""
    # 从 monitor_default.yaml 的 env_global 组和 terrain_ 组动态加载全量指标
    env_keys = load_env_keys_from_monitor_config()
    env_metrics = {}
    for key in env_keys:
        is_count = (
            "completed_count" in key
            or "abnormal_count" in key
            or "timeout_count" in key
            or key in ("completed_task_count", "failed_task_count", "timeout_task_count")
        )
        env_metrics[key] = "sum" if is_count else "avg"

    metrics_dict = {
        "basic": {
            "train_global_step": "sum",
            "actor_predict_succ_cnt": "sum",
            "sample_production_and_consumption_ratio": "avg",
            "episode_cnt": "sum",
            "actor_load_last_model_succ_cnt": "sum",
            "sample_receive_cnt": "sum",
        },
        "algorithm": {
            "reward": "avg",
            "reward_mean": "avg",
            "reward_std": "avg",
            "total_loss": "avg",
            "policy_loss": "avg",
            "value_loss": "avg",
            "entropy_loss": "avg",
        },
        "reward": {
            "episode_reward": "avg",
            "rew_tracking_lin_vel": "avg",
            "rew_feet_air_time": "avg",
            "rew_base_height": "avg",
            "rew_orientation": "avg",
            "rew_lin_vel_z": "avg",
            "rew_ang_vel_xy": "avg",
            "rew_torques": "avg",
            "rew_action_rate": "avg",
            "rew_dof_acc": "avg",
            "rew_collision": "avg",
        },
        "env": env_metrics,
    }

    return collect_training_metrics(metrics_dict)
