#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration validation utilities for new-generation tasks (standard/track).
新一代任务（standard/track）配置校验工具。

Design: Each _validate_xxx() returns a list[str] of errors (empty = no errors).
check_usr_conf() collects all errors and reports them at once.
设计：每个 _validate_xxx() 返回 list[str] 错误列表（空 = 无错误）。
check_usr_conf() 收集所有错误后一次性报出。
"""


import os

import toml


def update_config(target, source):
    """
    Recursively merge source dict into target config object.
    递归将 source 字典合并到 target 配置对象中。

    Args:
        target: Target config object to update
        source: Source dictionary with new values
    """
    for key, value in source.items():
        if isinstance(value, dict):
            if not hasattr(target, key):
                setattr(target, key, type("DynamicConfig", (), {})())
            sub_target = getattr(target, key)
            update_config(sub_target, value)
        else:
            setattr(target, key, value)


def update_sim_config(target, source):
    """
    Update simulation config with new values (safe — only existing attributes).
    更新模拟器配置（安全 — 仅更新已存在的属性）。

    Args:
        target: Target config object to update
        source: Source dictionary with new values
    """
    for key, value in source.items():
        if hasattr(target, key):
            setattr(target, key, value)


def read_usr_conf(usr_conf_file, logger, eval=False):
    """
    Load new-generation (standard/track) config by directly reading TOML file.
    No base/global merging — the TOML file is self-contained.

    直接读取新一代配置文件（standard/track），TOML 文件自包含所有配置。

    Args:
        usr_conf_file: User config file path
        logger: Logger object
        eval: Whether in eval mode (True=eval, False=train)

    Returns:
        dict: Parsed config dictionary, or None on failure
    """
    if not usr_conf_file or not os.path.exists(usr_conf_file):
        logger.error(f"Config file not found: {usr_conf_file}")
        return None

    try:
        with open(usr_conf_file, "r", encoding="utf-8") as f:
            config = toml.load(f)
        logger.info(f"[new-gen] Loaded config directly: {usr_conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {usr_conf_file}. Error: {e}")
        return None

    return config


# ---------------------------------------------------------------------------
#  Validator helpers
# ---------------------------------------------------------------------------


def _is_numeric(v):
    """Check if value is numeric (int or float, but not bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_range_pair(value, section, field):
    """
    Validate a [min, max] pair: must be a 2-element numeric list with min <= max.
    Returns a list of error strings (empty = valid).
    """
    errors = []
    if not isinstance(value, list) or len(value) != 2:
        errors.append(f"[{section}] {field} must be an array of length 2, got: {value}")
        return errors

    if not _is_numeric(value[0]) or not _is_numeric(value[1]):
        errors.append(f"[{section}] {field} elements must be numeric, got: {value}")
        return errors

    if value[0] > value[1]:
        errors.append(f"[{section}] {field} min ({value[0]}) must be <= max ({value[1]})")

    return errors


def _validate_range_pair_within(value, section, field, lo, hi):
    """
    Validate a [min, max] pair AND check both elements are within [lo, hi].
    校验 [min, max] 的同时，检查两个元素都在 [lo, hi] 区间内。
    """
    errors = _validate_range_pair(value, section, field)
    if errors:
        return errors
    if value[0] < lo or value[0] > hi:
        errors.append(f"[{section}] {field}[0]={value[0]} out of valid range [{lo}, {hi}]")
    if value[1] < lo or value[1] > hi:
        errors.append(f"[{section}] {field}[1]={value[1]} out of valid range [{lo}, {hi}]")
    return errors


# ---------------------------------------------------------------------------
#  Section validators — each returns list[str] (errors)
# ---------------------------------------------------------------------------


def _validate_env_params(usr_conf, is_eval):
    """
    Validate [env] section parameters.
    校验 [env] 配置块参数。

    Note: num_actions / num_proprio_obs / num_scan / num_critic_observations
    are architecture constants defined in StageConfig (agent_ppo/conf/conf.py);
    they are intentionally NOT validated here.
    注：num_actions / num_proprio_obs / num_scan / num_critic_observations
    是架构常量，定义在 StageConfig (agent_ppo/conf/conf.py) 中，
    故意不在此校验。
    """
    errors = []
    env_conf = usr_conf.get("env", {})

    # episode_length_s: required, positive number
    if "episode_length_s" not in env_conf:
        errors.append("[env] Missing required parameter: episode_length_s")
    else:
        episode_length_s = env_conf["episode_length_s"]
        if not _is_numeric(episode_length_s):
            errors.append(f"[env] episode_length_s must be numeric type, got: {episode_length_s}")
        elif episode_length_s <= 0:
            errors.append(f"[env] episode_length_s must be > 0, got: {episode_length_s}")

    # num_envs: required for train, optional for eval | 训练必填，评估可选
    if "num_envs" in env_conf:
        num_envs = env_conf["num_envs"]
        if not isinstance(num_envs, int) or isinstance(num_envs, bool):
            errors.append("[env] num_envs must be integer type")
        elif not (1 <= num_envs <= 4096):
            errors.append(f"[env] num_envs value {num_envs} out of range [1, 4096]")
    elif not is_eval:
        errors.append("[env] Missing required parameter: num_envs")

    return errors


def _validate_init_state_params(usr_conf):
    """
    Validate [init_state] section parameters.
    校验 [init_state] 配置块参数。
    """
    errors = []
    init_state_conf = usr_conf.get("init_state", {})
    if "pos" not in init_state_conf:
        return errors

    pos = init_state_conf["pos"]
    if not isinstance(pos, list) or len(pos) != 3:
        errors.append("[init_state] pos must be an array of length 3 (x, y, z)")
        return errors

    for i, v in enumerate(pos):
        if not _is_numeric(v):
            errors.append(f"[init_state] pos[{i}] must be numeric type")

    if len(errors) == 0:
        init_height_z = pos[2]
        if not (0.30 <= init_height_z <= 0.60):
            errors.append(
                f"[init_state] pos[2] value {init_height_z}m out of valid range [0.30, 0.60]m, " f"recommended: 0.35m"
            )

    return errors


def _validate_terrain_params(usr_conf, is_eval):
    """
    Validate [terrain] section parameters including proportions and curriculum.
    校验 [terrain] 配置块参数，包括比例和课程学习。
    """
    errors = []
    terrain_conf = usr_conf.get("terrain", {})
    if not terrain_conf:
        return errors

    # mode
    mode = terrain_conf.get("mode", "standard")
    valid_modes = {"standard", "track"}
    if mode not in valid_modes:
        errors.append(f"[terrain] Invalid mode: '{mode}'. Allowed: {valid_modes}")

    # curriculum must be bool if present
    if "curriculum" in terrain_conf:
        if not isinstance(terrain_conf["curriculum"], bool):
            errors.append("[terrain] curriculum must be boolean type (true/false)")

    # num_rows: positive integer in [1, 10]
    # num_rows 表示地形课程学习的难度等级数（X 轴方向子地形数量），推荐 10
    if "num_rows" in terrain_conf:
        num_rows = terrain_conf["num_rows"]
        if not isinstance(num_rows, int) or isinstance(num_rows, bool):
            errors.append(f"[terrain] num_rows must be a positive integer, got: {num_rows}")
        elif not (1 <= num_rows <= 10):
            errors.append(f"[terrain] num_rows value {num_rows} out of valid range [1, 10], recommended: 10")

    # num_cols: positive integer in [1, 40]
    # num_cols 表示子地形类型分桶数（Y 轴方向子地形数量），推荐 20
    if "num_cols" in terrain_conf:
        num_cols = terrain_conf["num_cols"]
        if not isinstance(num_cols, int) or isinstance(num_cols, bool):
            errors.append(f"[terrain] num_cols must be a positive integer, got: {num_cols}")
        elif not (1 <= num_cols <= 40):
            errors.append(f"[terrain] num_cols value {num_cols} out of valid range [1, 40], recommended: 20")

    # difficulty_range: [min, max], elements must be in [0.0, 1.0]
    if "difficulty_range" in terrain_conf:
        dr = terrain_conf["difficulty_range"]
        range_errors = _validate_range_pair(dr, "terrain", "difficulty_range")
        errors.extend(range_errors)
        if not range_errors:
            if dr[0] < 0.0 or dr[0] > 1.0:
                errors.append(f"[terrain] difficulty_range[0] must be in [0.0, 1.0], got: {dr[0]}")
            if dr[1] < 0.0 or dr[1] > 1.0:
                errors.append(f"[terrain] difficulty_range[1] must be in [0.0, 1.0], got: {dr[1]}")

    # max_init_terrain_level: must be in [0, 9]
    if "max_init_terrain_level" in terrain_conf:
        level = terrain_conf["max_init_terrain_level"]
        if not isinstance(level, int) or isinstance(level, bool):
            errors.append(f"[terrain] max_init_terrain_level must be an integer, got: {level}")
        elif not (0 <= level <= 9):
            errors.append(f"[terrain] max_init_terrain_level value {level} out of valid range [0, 9]")
        else:
            # Cross check: max_init_terrain_level must be < num_rows
            # (difficulty tier index must fall within available tiers 0..num_rows-1)
            # 交叉校验：max_init_terrain_level 必须 < num_rows
            # （初始档索引不能超出可用难度档 0..num_rows-1 的范围）
            if "num_rows" in terrain_conf:
                num_rows_val = terrain_conf["num_rows"]
                if isinstance(num_rows_val, int) and not isinstance(num_rows_val, bool) and num_rows_val >= 1:
                    if level >= num_rows_val:
                        errors.append(
                            f"[terrain] max_init_terrain_level ({level}) must be < num_rows ({num_rows_val}); "
                            f"valid difficulty tier index range is [0, num_rows-1]"
                        )

    # level: eval-only deterministic difficulty level list (e.g. [0, 1, 3, 4])
    # level：仅评估模式生效，确定性指定要评估的难度档列表（如 [0, 1, 3, 4]）
    # When set:
    # - Standard mode: num_rows = max(level)+1 is auto-computed (do not set num_rows)
    #   num_cols  = len([terrain.standard].sub_terrains) is auto-computed (do not set num_cols)
    # - Track  mode: base default num_parallel_tracks=10; it is auto-extended
    #   only when max(level)+1 is larger.
    # 配置该字段时：
    # - Standard 模式：num_rows = max(level)+1 自动推导（不允许再配 num_rows）
    #   num_cols  = len([terrain.standard].sub_terrains) 自动推导（不允许再配 num_cols）
    # - Track  模式：base 默认 num_parallel_tracks=10，仅当 max(level)+1 更大时自动扩展
    if "level" in terrain_conf:
        level_list = terrain_conf["level"]
        if not is_eval:
            errors.append("[terrain] 'level' is eval-only and must not appear in train config; please remove it")
        if not isinstance(level_list, list) or len(level_list) == 0:
            errors.append(f"[terrain] level must be a non-empty list, got: {level_list}")
        else:
            # Type/range check only. Duplicates and order are auto-fixed later
            # (dedup + sort) in base_env.apply_usr_conf_to_env_cfg; here we just
            # warn so the user is aware of what was written.
            # 只做类型/范围检查。重复与顺序会在 base_env.apply_usr_conf_to_env_cfg 中
            # 自动去重 + 排序，这里只给出 warning 提示用户感知。
            for i, lv in enumerate(level_list):
                if not isinstance(lv, int) or isinstance(lv, bool):
                    errors.append(f"[terrain] level[{i}]={lv} must be an integer in [0, 9]")
                elif not (0 <= lv <= 9):
                    errors.append(f"[terrain] level[{i}]={lv} out of valid range [0, 9]")
            # Duplicates are treated as user typo and rejected; order is
            # auto-sorted ascending by the framework (no warning needed since
            # placement semantics are independent of the order).
            # 重复元素视为用户笔误，直接报错；顺序由框架自动升序（不影响评估语义，无需提示）。
            if len(set(level_list)) != len(level_list):
                dup = sorted({v for v in level_list if level_list.count(v) > 1})
                errors.append(
                    f"[terrain] level contains duplicated elements {dup} in {level_list}; " f"please remove duplicates"
                )
            # level cannot coexist with num_rows (avoid ambiguous source of truth)
            # level 与 num_rows 互斥（避免来源歧义）
            if "num_rows" in terrain_conf:
                errors.append("[terrain] 'level' and 'num_rows' cannot both be set; " "level auto-determines num_rows")

    # Standard mode: sub-terrain proportions must sum to 1.0 (train mode)
    # OR sub_terrains list must be valid (eval mode with level list)
    # Standard 模式：训练模式下子地形 proportion 之和必须为 1.0；
    # 或评估模式下使用 sub_terrains 列表形式（与 level 列表配合使用）
    if mode == "standard":
        standard_conf = terrain_conf.get("standard", {})
        if standard_conf:
            valid_terrain_types = {
                "pyramid_slope",
                "pyramid_slope_inv",
                "pyramid_stairs",
                "pyramid_stairs_inv",
                "maze",
            }

            # Reject track-only sub-terrains in standard mode
            # 拒绝仅允许在 track 模式下使用的子地形
            track_only_terrains = {"nav_maze", "open_entry_maze"}
            for bad_name in track_only_terrains:
                if bad_name in standard_conf and isinstance(standard_conf[bad_name], dict):
                    errors.append(
                        f"[terrain.standard.{bad_name}] '{bad_name}' is not allowed in standard mode, "
                        f"only available in track mode ([terrain.track].sub_terrains)"
                    )

            # Eval-only: sub_terrains list form (deterministic placement, no proportion)
            # 评估专用：sub_terrains 列表形式（确定性放置，无 proportion）
            if "sub_terrains" in standard_conf:
                sub_list = standard_conf["sub_terrains"]
                if not is_eval:
                    errors.append(
                        "[terrain.standard] 'sub_terrains' list form is eval-only; "
                        "use dict + proportion in train config"
                    )
                if not isinstance(sub_list, list) or len(sub_list) == 0:
                    errors.append(f"[terrain.standard] sub_terrains must be a non-empty list, got: {sub_list}")
                else:
                    for i, t in enumerate(sub_list):
                        if t not in valid_terrain_types:
                            errors.append(
                                f"[terrain.standard] sub_terrains[{i}]='{t}' invalid, "
                                f"allowed: {sorted(valid_terrain_types)}"
                            )
                    if len(set(sub_list)) != len(sub_list):
                        errors.append(f"[terrain.standard] sub_terrains contains duplicated elements: {sub_list}")
                    # sub_terrains list and num_cols are mutually exclusive
                    # sub_terrains 列表与 num_cols 互斥
                    if "num_cols" in terrain_conf:
                        errors.append(
                            "[terrain.standard] 'sub_terrains' (list) and "
                            "[terrain] 'num_cols' cannot both be set; "
                            "len(sub_terrains) auto-determines num_cols"
                        )

            total = 0.0
            has_proportions = False
            # Known non-terrain-type meta keys in [terrain.standard]
            # [terrain.standard] 中已知的非地形类型 meta 字段
            known_meta_keys = {"sub_terrains", "goal_distance"}
            for terrain_type, terrain_value in standard_conf.items():
                if terrain_type in known_meta_keys:
                    continue
                if terrain_type not in valid_terrain_types:
                    # Unknown dict-form sub-terrain → reject.
                    # Scalar values are tolerated (treated as meta-config).
                    # 未知的 dict 型子地形 → 拒绝；标量值视作 meta 配置，放行。
                    if isinstance(terrain_value, dict):
                        errors.append(
                            f"[terrain.standard.{terrain_type}] unknown sub-terrain type '{terrain_type}', "
                            f"allowed: {sorted(valid_terrain_types)}"
                        )
                    continue
                if isinstance(terrain_value, dict) and "proportion" in terrain_value:
                    has_proportions = True
                    p = terrain_value["proportion"]
                    if not _is_numeric(p):
                        errors.append(f"[terrain.standard.{terrain_type}] proportion must be numeric, got: {p}")
                    elif p < 0:
                        errors.append(f"[terrain.standard.{terrain_type}] proportion must be >= 0, got: {p}")
                    else:
                        total += p

            if has_proportions and not errors:
                # Allow small floating point tolerance
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"[terrain.standard] sub-terrain proportions must sum to 1.0, " f"got: {total:.6f}")

    # Track mode: validate sub_terrains list against track_length
    # track 模式：校验 sub_terrains 列表与 track_length 的一致性
    if mode == "track":
        track_conf = terrain_conf.get("track", {})
        if track_conf:
            valid_track_terrain_types = {
                "pyramid_slope",
                "pyramid_slope_inv",
                "pyramid_stairs",
                "pyramid_stairs_inv",
                "open_entry_maze",
            }

            track_length = track_conf.get("track_length")
            sub_terrains = track_conf.get("sub_terrains")

            # track_length and sub_terrains must appear together (or both absent).
            # One without the other is an ambiguous configuration.
            # track_length 与 sub_terrains 必须同时出现或同时缺失，单独出现语义不明确。
            if (track_length is None) != (sub_terrains is None):
                if track_length is None:
                    errors.append(
                        "[terrain.track] sub_terrains is set but track_length is missing; "
                        "both must be configured together"
                    )
                else:
                    errors.append(
                        "[terrain.track] track_length is set but sub_terrains is missing; "
                        "both must be configured together"
                    )

            # track_length: must be an integer in [1, 5] if present
            if track_length is not None:
                if not isinstance(track_length, int) or isinstance(track_length, bool):
                    errors.append(f"[terrain.track] track_length must be an integer, got: {track_length}")
                elif not (1 <= track_length <= 5):
                    errors.append(f"[terrain.track] track_length value {track_length} out of valid range [1, 5]")

            # sub_terrains_random: not supported, must not be configured
            # sub_terrains_random：不支持该配置，一旦出现即报错
            if "sub_terrains_random" in track_conf:
                errors.append(
                    "[terrain.track] sub_terrains_random is not supported and must not be configured; "
                    "please remove this field from the TOML"
                )

            # sub_terrains validation
            if sub_terrains is not None:
                if not isinstance(sub_terrains, list):
                    errors.append(f"[terrain.track] sub_terrains must be an array, got: {type(sub_terrains).__name__}")
                else:
                    # Rule 1: length must equal track_length
                    # 规则 1：sub_terrains 数组长度必须等于 track_length
                    if isinstance(track_length, int) and not isinstance(track_length, bool) and track_length > 0:
                        if len(sub_terrains) != track_length:
                            errors.append(
                                f"[terrain.track] sub_terrains length ({len(sub_terrains)}) "
                                f"must equal track_length ({track_length})"
                            )

                    # Rule 2: elements must not be duplicated
                    # 规则 2：sub_terrains 里的元素不能重复
                    seen = set()
                    duplicates = set()
                    for t in sub_terrains:
                        if t in seen:
                            duplicates.add(t)
                        else:
                            seen.add(t)
                    if duplicates:
                        errors.append(
                            f"[terrain.track] sub_terrains contains duplicated elements: "
                            f"{sorted(duplicates)}, all elements must be unique"
                        )

                    # Rule 3: last element must be open_entry_maze (track finish must be maze)
                    # 规则 3：sub_terrains 最后一个元素必须为 open_entry_maze（赛道终点地形必须是迷宫）
                    if sub_terrains and sub_terrains[-1] != "open_entry_maze":
                        errors.append(
                            f"[terrain.track] sub_terrains last element must be 'open_entry_maze', "
                            f"got: '{sub_terrains[-1]}'"
                        )

                    # Bonus: elements must be valid terrain types
                    # 附加校验：元素必须是合法的地形类型
                    for i, t in enumerate(sub_terrains):
                        if t not in valid_track_terrain_types:
                            errors.append(
                                f"[terrain.track] sub_terrains[{i}] invalid terrain type: '{t}'. "
                                f"Allowed: {sorted(valid_track_terrain_types)}"
                            )

    return errors


def _validate_domain_rand_params(usr_conf, is_eval):
    """
    Validate [domain_rand] section parameters including ranges.
    校验 [domain_rand] 配置块参数，包括范围。
    """
    errors = []
    domain_rand_conf = usr_conf.get("domain_rand", {})
    if not domain_rand_conf:
        return errors

    # enable_domain_rand: bool
    if "enable_domain_rand" in domain_rand_conf:
        if not isinstance(domain_rand_conf["enable_domain_rand"], bool):
            errors.append("[domain_rand] enable_domain_rand must be boolean type (true/false)")

    # randomize_friction: bool
    if "randomize_friction" in domain_rand_conf:
        if not isinstance(domain_rand_conf["randomize_friction"], bool):
            errors.append("[domain_rand] randomize_friction must be boolean type (true/false)")

    # friction_range: [min, max], both within [0, 10]
    # friction_range: [min, max]，两端都必须在 [0, 10] 内（物理常识软上限）
    if "friction_range" in domain_rand_conf:
        fr = domain_rand_conf["friction_range"]
        errors.extend(_validate_range_pair_within(fr, "domain_rand", "friction_range", 0.0, 10.0))

    # push_robots: bool
    if "push_robots" in domain_rand_conf:
        if not isinstance(domain_rand_conf["push_robots"], bool):
            errors.append("[domain_rand] push_robots must be boolean type (true/false)")

    # push_interval_s: positive number
    if "push_interval_s" in domain_rand_conf:
        pis = domain_rand_conf["push_interval_s"]
        if not _is_numeric(pis):
            errors.append(f"[domain_rand] push_interval_s must be numeric, got: {pis}")
        elif pis <= 0:
            errors.append(f"[domain_rand] push_interval_s must be > 0, got: {pis}")

    # max_push_vel_xy: non-negative number
    if "max_push_vel_xy" in domain_rand_conf:
        mpv = domain_rand_conf["max_push_vel_xy"]
        if not _is_numeric(mpv):
            errors.append(f"[domain_rand] max_push_vel_xy must be numeric, got: {mpv}")
        elif mpv < 0:
            errors.append(f"[domain_rand] max_push_vel_xy must be >= 0, got: {mpv}")

    return errors


def _validate_noise_params(usr_conf):
    """
    Validate [noise] section parameters.
    校验 [noise] 配置块参数。
    """
    errors = []
    noise_conf = usr_conf.get("noise", {})
    if not noise_conf:
        return errors

    if "add_noise" in noise_conf and not isinstance(noise_conf["add_noise"], bool):
        errors.append("[noise] add_noise must be boolean type (true/false)")

    return errors


def _validate_commands_params(usr_conf):
    """
    Validate [commands] section parameters.
    校验 [commands] 配置块参数。
    """
    errors = []
    commands_conf = usr_conf.get("commands", {})
    if not commands_conf:
        return errors

    # resampling_time: [min, max] pair, both within [0, 300] seconds
    # resampling_time：[min, max]，两端都在 [0, 300] 秒内
    if "resampling_time" in commands_conf:
        errors.extend(
            _validate_range_pair_within(commands_conf["resampling_time"], "commands", "resampling_time", 0.0, 300.0)
        )

    # Validate command ranges and limits.
    # Each velocity range element must be within [-10, 10] m/s (rad/s for angular).
    # 每个速度项的区间端点都必须落在 [-10, 10] 之内
    for section_key in ["ranges", "limit"]:
        ranges_conf = commands_conf.get(section_key, {})
        velocity_keys = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "ang_vel_yaw"]

        for param_name in velocity_keys:
            if param_name not in ranges_conf:
                continue
            errors.extend(
                _validate_range_pair_within(ranges_conf[param_name], f"commands.{section_key}", param_name, -10.0, 10.0)
            )

    return errors


def _validate_rewards_params(usr_conf):
    """
    Validate [rewards] section structure (train only).
    校验 [rewards] 配置块结构合法性（仅训练时校验）。

    Each reward term must have a numeric 'weight' field.
    Optional 'params' sub-dict is allowed for parameterized rewards.
    """
    errors = []
    rewards_conf = usr_conf.get("rewards", {})
    if not rewards_conf:
        return errors

    for reward_name, reward_value in rewards_conf.items():
        if not isinstance(reward_value, dict):
            errors.append(f"[rewards.{reward_name}] must be a table/dict, got: {type(reward_value).__name__}")
            continue

        # weight is required
        if "weight" not in reward_value:
            errors.append(f"[rewards.{reward_name}] missing required field: weight")
            continue

        weight = reward_value["weight"]
        if not _is_numeric(weight):
            errors.append(f"[rewards.{reward_name}] weight must be numeric, got: {weight}")

        # Only 'weight' and 'params' are expected keys
        for key in reward_value:
            if key not in ("weight", "params"):
                errors.append(
                    f"[rewards.{reward_name}] unexpected field: '{key}' " f"(only 'weight' and 'params' are allowed)"
                )

        # If params exists, it must be a dict
        if "params" in reward_value:
            if not isinstance(reward_value["params"], dict):
                errors.append(
                    f"[rewards.{reward_name}.params] must be a table/dict, "
                    f"got: {type(reward_value['params']).__name__}"
                )
            else:
                # Numeric-typed params: must be numeric (not bool, not string).
                # 数值型参数必须传入数字；禁止布尔值（True/False）或字符串，
                # 因为 bool 在 Python 中是 int 的子类，若不显式拦截会被当成 0/1 使用，
                # 静默污染奖励计算。
                numeric_param_keys = {
                    "std",
                    "threshold",
                    "obstacle_threshold",
                    "trapped_threshold",
                    "tanh_mult",
                    "target_height",
                    "velocity_threshold",
                    "stand_still_scale",
                    "goal_reached_threshold",
                }
                for p_key, p_val in reward_value["params"].items():
                    if p_key in numeric_param_keys and not _is_numeric(p_val):
                        errors.append(
                            f"[rewards.{reward_name}.params] '{p_key}' must be numeric, "
                            f"got: {p_val} (type={type(p_val).__name__})"
                        )

    return errors


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def check_usr_conf(usr_conf, is_eval, logger):
    """
    Unified entry point for configuration validation (new-generation tasks only).
    Collects ALL errors and reports them at once.

    配置验证统一入口（仅新一代任务）。收集所有错误后一次性报出。

    Args:
        usr_conf: User config dictionary
        is_eval: Whether in eval mode (True=eval, False=train)
        logger: Logger object

    Returns:
        (valid: bool, error_message: str)
    """
    if not usr_conf:
        return False, "usr_conf is None, please check"

    all_errors = []

    # Always validate these sections
    all_errors.extend(_validate_env_params(usr_conf, is_eval))
    all_errors.extend(_validate_init_state_params(usr_conf))
    all_errors.extend(_validate_terrain_params(usr_conf, is_eval))
    all_errors.extend(_validate_domain_rand_params(usr_conf, is_eval))
    all_errors.extend(_validate_noise_params(usr_conf))
    all_errors.extend(_validate_commands_params(usr_conf))

    # Rewards validation only for training (eval has no rewards section)
    if not is_eval:
        all_errors.extend(_validate_rewards_params(usr_conf))

    if all_errors:
        error_summary = f"Configuration validation failed with {len(all_errors)} error(s):\n" + "\n".join(
            f"  {i+1}. {e}" for i, e in enumerate(all_errors)
        )
        return False, error_summary

    return True, "OK"
