#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import shutil
import cv2
import math
from collections import defaultdict
import yaml
import re


def load_env_keys_from_monitor_config(monitor_config_path="/root/tools/conf/monitor_default.yaml"):
    """
    从monitor配置文件中读取env_keys
    从env_global组和所有terrain_开头的地形组中提取所有metrics_name，去掉kaiwu_前缀和{}后缀

    Args:
        monitor_config_path (str): monitor配置文件路径，默认为 "/root/tools/conf/monitor_default.yaml"

    Returns:
        list: envkeys列表，例如 ["completed_task_count", "total_score_avg", "completed_count_pyramid_slope_level0", ...]

    Example:
        >>> env_keys = load_env_keys_from_monitor_config()
        >>> print(env_keys[:3])
        ['completed_task_count', 'failed_task_count', 'timeout_task_count']
    """
    env_keys = []
    try:
        with open(monitor_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config and "groups" in config:
            for group in config["groups"]:
                group_name_en = group.get("group_name_en", "")
                # 匹配 env_global 组和所有 terrain_ 开头的地形组
                if group_name_en == "env_global" or group_name_en.startswith("terrain_"):
                    panels = group.get("panels", [])
                    for panel in panels:
                        metrics = panel.get("metrics", [])
                        for metric in metrics:
                            metrics_name = metric.get("metrics_name", "")
                            if metrics_name:
                                # 从expr中提取metrics名称（去掉kaiwu_前缀和{}）
                                expr = metric.get("expr", "")
                                # 使用正则提取kaiwu_xxx{}中的xxx部分
                                match = re.search(r"kaiwu_(\w+)\{", expr)
                                if match:
                                    key = match.group(1)
                                    if key not in env_keys:
                                        env_keys.append(key)
    except Exception as e:
        print(f"Failed to load env_keys from {monitor_config_path}: {e}")

    return env_keys


def load_reward_keys_from_monitor_config(monitor_config_path="/root/tools/conf/monitor_default.yaml"):
    """
    从monitor配置文件中读取reward_keys
    支持两种格式：
    - standard 版：从 group_name_en == "reward" 的组中提取
    - track 版：从 group_name_en 以 "reward_" 开头的所有组中提取（reward_locomotion / reward_navigation）

    Args:
        monitor_config_path (str): monitor配置文件路径，默认为 "/root/tools/conf/monitor_default.yaml"

    Returns:
        list: reward_keys列表，例如 ["reward_track_lin_vel_xy", "reward_heuristic_navigation", ...]
    """
    reward_keys = []
    # 不上报的通用汇总指标（非单项 reward）
    _SKIP_METRICS = {"episode_reward", "reward_mean", "reward_std"}
    try:
        with open(monitor_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config and "groups" in config:
            for group in config["groups"]:
                group_en = group.get("group_name_en", "")
                # standard 版：group_name_en == "reward"
                # track 版：group_name_en 以 "reward_" 开头（reward_locomotion / reward_navigation）
                if group_en == "reward" or group_en.startswith("reward_"):
                    panels = group.get("panels") or []
                    for panel in panels:
                        metrics = panel.get("metrics", [])
                        for metric in metrics:
                            metrics_name = metric.get("metrics_name", "")
                            if metrics_name and metrics_name not in _SKIP_METRICS:
                                # 从expr中提取metrics名称（去掉kaiwu_前缀和{}）
                                expr = metric.get("expr", "")
                                match = re.search(r"kaiwu_(\w+)\{", expr)
                                if match:
                                    key = match.group(1)
                                    if key not in reward_keys:
                                        reward_keys.append(key)
    except Exception as e:
        print(f"Failed to load reward_keys from {monitor_config_path}: {e}")

    return reward_keys


def img2mp4(input_folder, output_file, fps=30):
    """保存视频文件
    Args:
        video_path: 视频路径
        fps: 帧率, 默认30
    """

    # 参数配置区（按需修改）
    input_folder = input_folder  # 图片存放目录（需确保目录存在）
    output_file = output_file  # 输出视频文件名
    fps = fps  # 帧率（每秒帧数）
    output_size = (1920, 1080)  # 输出视频分辨率（宽, 高）

    # 获取排序后的图片文件列表（确保文件名按顺序排列）
    img_files = sorted([f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg"))])
    if len(img_files) == 0:
        return False

    # 按照fps = M / target_duration来计算
    total_image_files_count = len(img_files)

    # 根据图片数量来调整视频播放的时长
    if total_image_files_count <= 50:
        video_duration = 5
    elif total_image_files_count <= 200:
        video_duration = 10
    else:
        video_duration = 20

    fps = total_image_files_count / video_duration

    """
    通用兼容性写法, 优先尝试H.264, 如果没有则尝试mp4v, 由于需要重新编译OpenCV, 故本次暂时按照mp4v来进行
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, output_size)

    # 逐帧处理图片
    for img_name in img_files:
        img_path = os.path.join(input_folder, img_name)

        # 读取图片并调整尺寸
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告：无法读取图片 {img_path}")
            continue

        # 调整图片尺寸匹配输出分辨率
        resized_frame = cv2.resize(frame, output_size)

        # 写入视频帧
        video_writer.write(resized_frame)

    # 释放资源
    video_writer.release()

    return True


def ensure_empty_directory(path):
    """
    创建目录
    """

    # 检查路径是否存在
    if os.path.exists(path):
        # 确认是否为目录
        if not os.path.isdir(path):
            raise NotADirectoryError(f"路径 {path} 是文件而非目录")
        # 清空目录内容
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.remove(entry_path)  # 删除文件或符号链接
            else:
                shutil.rmtree(entry_path)  # 删除子目录
    else:
        # 创建目录（包括父目录）
        os.makedirs(path, exist_ok=True)


# 常量定义
SUCCESS_WEIGHT = 500
SPEED_WEIGHT = 300
STABILITY_WEIGHT = 200

# 地形ID和类型的映射
terrain_name_map = {
    0: "ripples",
    1: "slope",
    2: "stairs_up",
    3: "stairs_down",
    4: "obstacles",
    5: "composite1",
    6: "composite2",
    7: "flat",
}


def sigmoid(x):
    return 2 / (1 + math.exp(-x)) - 1


def calculate_scores(success, velocity, stability, count):
    """
    计算分数
    """

    # 增加 NaN 和 Inf 的安全处理
    if (
        count == 0
        or any(math.isnan(x) for x in [success, velocity, stability, count])
        or any(math.isinf(x) for x in [success, velocity, stability, count])
    ):
        return (0.0, 0.0, 0.0, 0.0)

    # 总得分 = 终点 * 500 + sigmoid(速度) * 300 + exp(稳定性*1000) * 200
    success_score = (success / count) * SUCCESS_WEIGHT

    # 注意特别处理下0的情况
    avg_velocity = velocity / count
    speed_score = sigmoid(avg_velocity) * SPEED_WEIGHT
    avg_stability = stability / count
    if avg_stability == 0:
        stability_score = 0
    else:
        stability_score = math.exp(avg_stability * 1000) * STABILITY_WEIGHT

    return (
        round(success_score + speed_score + stability_score, 2),
        round(success_score, 2),
        round(speed_score, 2),
        round(stability_score, 2),
    )


def safe_divide(numerator, denominator):
    if denominator == 0 or math.isinf(numerator) or math.isinf(denominator):
        return 0.0
    if math.isnan(numerator) or math.isnan(denominator):
        return 0.0
    return numerator / denominator


def get_terrain_index(cumulative_proportions, choice):
    """
    Find the first index where cumulative proportion exceeds the choice value
    Args:
        cumulative_proportions (list): Sorted list of cumulative values
        choice (float): Random value between 0 and 1
    Returns:
        int: Index of terrain type
    Example:
        Input: [0.1,0.2,0.4,0.6,0.8,0.9,1,1], 0.05
        Output: 0

    返回第一个 cumulative_proportions[i] > choice 的下标 i

    参数:
        累积比例列表
        0-1之间的随机值

    返回:
        地形类型索引

    示例:
        输入: [0.1,0.2,0.4,0.6,0.8,0.9,1,1], 0.05
        输出: 0
    """
    for i, p in enumerate(cumulative_proportions):
        if choice < p:
            return i

    # Fallback to last index if choice >= all values
    # 如果choice大于等于所有区间，返回最后一个下标
    return len(cumulative_proportions) - 1


def get_cumulative_proportions(terrain_proportions):
    """
    Calculate cumulative proportions with ceiling at 1.0 and ensure last element is 1.0
    Args:
        terrain_proportions (list): List of terrain proportions
    Returns:
        list: Cumulative proportions
    Example:
        Input: [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.0]
        Output: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1, 1]

    计算累积地形比例(最大值1.0), 并确保最后一个元素为1.0
    参数:
        地形比例列表
    返回:
        累积比例列表
    示例:
        输入: [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.0]
        输出: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1, 1]
    """
    cumulative_proportions = []
    total = 0

    for p in terrain_proportions:
        total += p
        if total > 1:
            total = 1
        cumulative_proportions.append(total)

    if cumulative_proportions and cumulative_proportions[-1] != 1.0:
        cumulative_proportions[-1] = 1.0

    return cumulative_proportions


def calculate_terrain_stats(
    episode_stability,
    episode_success,
    episode_velocity,
    terrain_level,
    terrain_types,
    usr_conf,
):
    """
    Multi-metric Terrain Statistics Function (Enhanced Version)
    New features:

    Difficulty-level terrain score statistics
    Global total score statistics
    Global average metrics statistics

    多指标地形统计函数（增强版）
    新增功能：
    1. 各难度等级地形得分统计
    2. 全局总得分统计
    3. 全局平均指标统计
    """
    num_cols = usr_conf["terrain"]["num_cols"]
    terrain_proportions = usr_conf["terrain"]["terrain_proportions"]
    hidden_terrain_proportions = usr_conf["terrain"].get("hidden_terrain_proportions", [0.0, 0.0])
    terrain_proportions = terrain_proportions[:5] + hidden_terrain_proportions + terrain_proportions[5:]

    stability_np = episode_stability.cpu().numpy()
    success_np = episode_success.cpu().numpy()
    velocity_np = episode_velocity.cpu().numpy()
    level_np = terrain_level.cpu().numpy()
    type_np = terrain_types.cpu().numpy()

    stats = defaultdict(lambda: defaultdict(lambda: {"stability": [0.0, 0], "success": [0.0, 0], "velocity": [0.0, 0]}))

    # 总的环境计算变量
    total_stability = 0.0
    total_success = 0.0
    total_velocity = 0.0
    total_count = 0

    # 单个环境计算变量
    single_stability = 0.0
    single_success = 0.0
    single_velocity = 0.0
    single_count = 0

    for s_stab, s_succ, s_vel, lv, tp in zip(stability_np, success_np, velocity_np, level_np, type_np):
        t_code = int(tp)
        level = int(lv)

        # Total Score = Termination * 500 + sigmoid(Velocity) * 300 + exp(Stability * 1000) * 200
        # 总得分 = 终点 * 500 + sigmoid(速度) * 300 + exp(稳定性*1000) * 200
        count = 1
        single_score, single_success, single_velocity, single_stability = calculate_scores(s_succ, s_vel, s_stab, count)

        stats[t_code][level]["stability"][0] += single_stability
        stats[t_code][level]["stability"][1] += 1

        stats[t_code][level]["success"][0] += single_success
        stats[t_code][level]["success"][1] += 1

        stats[t_code][level]["velocity"][0] += single_velocity
        stats[t_code][level]["velocity"][1] += 1

        total_success += single_success
        total_velocity += single_velocity
        total_stability += single_stability

        total_count += 1

    result = {}
    cumulative_proportions = get_cumulative_proportions(terrain_proportions)
    for t_code, levels in stats.items():
        choice = t_code / num_cols + 0.001
        terrain_name = terrain_name_map.get(get_terrain_index(cumulative_proportions, choice))

        for level in range(10):
            level_data = levels.get(level, {"stability": [0.0, 0], "success": [0.0, 0], "velocity": [0.0, 0]})

            stab_avg = safe_divide(level_data["stability"][0], level_data["stability"][1])
            succ_avg = safe_divide(level_data["success"][0], level_data["success"][1])
            vel_avg = safe_divide(level_data["velocity"][0], level_data["velocity"][1])

            # 三个参数相加
            single_score = stab_avg + succ_avg + vel_avg
            result[f"{terrain_name}_score_level{level}"] = single_score
            result[f"{terrain_name}_avg_episode_stability_level{level}"] = stab_avg
            result[f"{terrain_name}_avg_episode_success_level{level}"] = succ_avg
            result[f"{terrain_name}_avg_episode_velocity_level{level}"] = vel_avg

    if total_count > 0:
        result["avg_episode_success"] = round(total_success / total_count, 2)
        result["avg_episode_velocity"] = round(total_velocity / total_count, 2)
        result["avg_episode_stability"] = round(total_stability / total_count, 2)

        total_score = result["avg_episode_stability"] + result["avg_episode_success"] + result["avg_episode_velocity"]

        result["total_score"] = round(total_score, 2)
    else:
        result["total_score"] = 0.0
        result["avg_episode_success"] = 0.0
        result["avg_episode_velocity"] = 0.0
        result["avg_episode_stability"] = 0.0

    return result


def sort_by_terrain_map_order(data_items, is_aggregated=False):
    """
    按 terrain_name_map 的顺序排序
    :param data_items: 可迭代对象，元素格式为 (key, stats)
    :param is_aggregated: 是否为聚合数据（key 是地形类型编号）
    :return: 排序后的列表
    """
    # 构建地形类型优先级字典 {0:0, 1:1, 2:2, 3:3, 7:4}
    terrain_order = {t: idx for idx, t in enumerate(terrain_name_map.keys())}

    def get_sort_key(item):
        # 解析地形类型
        if is_aggregated:
            terrain_type = item[0]  # 聚合数据的 key 是地形类型编号
        else:
            terrain_type = item[0][0]  # 详细数据的 key 是 (terrain_type, level)

        # 获取地形顺序，不在映射表中的排最后（返回无穷大）
        order = terrain_order.get(terrain_type, float("inf"))

        # 详细数据需要添加难度级别排序
        if not is_aggregated:
            level = item[0][1]
            return (order, level)  # 先按地形顺序，再按难度升序
        return (order,)

    return sorted(data_items, key=get_sort_key)
