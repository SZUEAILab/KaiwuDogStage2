from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 需要使用"出口晋级"逻辑的子地形名称关键字（包含即匹配）
_EXIT_BASED_TERRAIN_KEYWORDS = ("maze", "corridor")


def _build_col_to_sub_index(terrain_gen_cfg) -> np.ndarray | None:
    """复现 Isaac Lab curriculum 模式下 列→子地形索引 的分配逻辑。

    Returns:
        np.ndarray of shape (num_cols,) mapping col index to sub-terrain index,
        or None if unable to compute.
    """
    sub_terrains = getattr(terrain_gen_cfg, "sub_terrains", None)
    if not sub_terrains:
        return None
    proportions = np.array([cfg.proportion for cfg in sub_terrains.values()])
    if proportions.sum() < 1e-9:
        return None
    proportions = proportions / proportions.sum()
    num_cols = terrain_gen_cfg.num_cols
    sub_indices = np.array(
        [int(np.min(np.where(col / num_cols + 0.001 < np.cumsum(proportions))[0])) for col in range(num_cols)],
        dtype=np.int32,
    )
    return sub_indices


def _build_exit_based_col_mask(terrain_gen_cfg, device: str) -> torch.Tensor | None:
    """构建一个 bool tensor，标识哪些列属于"需要出口晋级"的子地形。

    Returns:
        torch.Tensor of shape (num_cols,) dtype=bool, or None if unable to compute.
    """
    sub_terrains = getattr(terrain_gen_cfg, "sub_terrains", None)
    if not sub_terrains:
        return None
    sub_names = list(sub_terrains.keys())

    # track 模式：行方向是序列，用 sub_terrains_order
    track_length = getattr(terrain_gen_cfg, "track_length", None)
    if track_length is not None and track_length > 0:
        # Track 模式下所有列都是同一序列，整体用出口逻辑
        return torch.ones(terrain_gen_cfg.num_cols, dtype=torch.bool, device=device)

    # standard (grid) 模式：列→子地形按 proportion 分配
    sub_indices = _build_col_to_sub_index(terrain_gen_cfg)
    if sub_indices is None:
        return None

    mask = np.array(
        [any(kw in sub_names[idx] for kw in _EXIT_BASED_TERRAIN_KEYWORDS) for idx in sub_indices], dtype=bool
    )
    return torch.tensor(mask, dtype=torch.bool, device=device)


def _is_track_terrain(terrain_gen_cfg) -> bool:
    """Return whether the terrain generator is in track mode.

    判断 terrain generator 是否为 track 模式。
    """
    track_length = getattr(terrain_gen_cfg, "track_length", None)
    return track_length is not None and int(track_length) > 0


def _terrain_levels_vel_track(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    terrain: TerrainImporter,
    terrain_gen_cfg,
) -> torch.Tensor:
    """Track-specific curriculum: promote/demote col (= difficulty), not row.

    Track 专用课程逻辑：升降 col（难度档），不再升降 row（赛道段位置）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    exit_positions = getattr(terrain_gen_cfg, "terrain_exit_positions", None)
    if exit_positions is None or terrain.terrain_origins is None:
        return torch.mean(terrain.terrain_types.float())

    cached_exit_positions = getattr(terrain_gen_cfg, "_exit_positions_tensor", None)
    if cached_exit_positions is None:
        exit_positions = torch.as_tensor(exit_positions, dtype=torch.float32, device=env.device)
        terrain_gen_cfg._exit_positions_tensor = exit_positions
    else:
        exit_positions = cached_exit_positions.to(device=env.device, dtype=torch.float32)

    num_rows = int(exit_positions.shape[0])
    num_cols = int(exit_positions.shape[1])
    if num_rows <= 0 or num_cols <= 0:
        return torch.mean(terrain.terrain_types.float())

    # Track success means reaching the final exit of the whole track.
    # Track 晋级条件：到达整条赛道最后一段的出口，而不是当前 row 的局部出口。
    track_length = int(getattr(terrain_gen_cfg, "track_length", num_rows))
    final_row = max(0, min(track_length - 1, num_rows - 1))

    current_cols = terrain.terrain_types[env_ids].long().clamp(0, num_cols - 1)
    robot_xy = asset.data.root_pos_w[env_ids, :2]
    final_exit_xy = exit_positions[final_row, current_cols, :2]
    dist_to_final_exit = torch.norm(robot_xy - final_exit_xy, dim=1)

    # Keep this threshold aligned with `_goal_reached_termination` and
    # `rewards.reach_goal.params.threshold` in nav/hier_nav configs.
    # 该阈值必须与 `_goal_reached_termination` 和 nav/hier_nav 配置中的
    # `rewards.reach_goal.params.threshold` 保持一致，避免“完成但不晋级”的死区。
    exit_threshold = 0.6
    move_up = dist_to_final_exit < exit_threshold

    # Demotion keeps the original distance-vs-command heuristic.
    # 降级沿用原有“实际位移不足期望位移一半”的启发式。
    origin_xy = env.scene.env_origins[env_ids, :2]
    distance = torch.norm(robot_xy - origin_xy, dim=1)
    expected_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = distance < expected_distance
    move_down = move_down & ~move_up

    new_cols = current_cols + move_up.long() - move_down.long()
    new_cols = new_cols.clamp(0, num_cols - 1)
    terrain.terrain_types[env_ids] = new_cols.to(terrain.terrain_types.dtype)

    # Keep row unchanged here. The reset event owns track start-row policy.
    # 这里不修改 row；track 的起点 row 策略由 reset event 统一负责。
    if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
        rows = terrain.terrain_levels[env_ids].long().clamp(0, terrain.terrain_origins.shape[0] - 1)
    else:
        rows = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
    terrain.env_origins[env_ids] = terrain.terrain_origins[rows, new_cols]

    return torch.mean(terrain.terrain_types.float())


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum that uses mode-specific promotion logic.

    课程逻辑按地形模式区分：
      - Standard: row (`terrain_levels`) is difficulty, so promote/demote row.
      - Track: col (`terrain_types`) is difficulty, so promote/demote col.

    Standard 模式：row 是难度，因此升降 row。
    Track 模式：col 是难度，因此升降 col。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    terrain_gen_cfg = terrain.cfg.terrain_generator

    if _is_track_terrain(terrain_gen_cfg):
        return _terrain_levels_vel_track(env, env_ids, asset_cfg, terrain, terrain_gen_cfg)

    # --- 构建并缓存：哪些列需要出口晋级 ---
    if not hasattr(terrain_gen_cfg, "_exit_col_mask"):
        terrain_gen_cfg._exit_col_mask = _build_exit_based_col_mask(terrain_gen_cfg, env.device)

    exit_col_mask = terrain_gen_cfg._exit_col_mask  # (num_cols,) bool or None

    # --- 所有 env 的公共数据 ---
    robot_xy = asset.data.root_pos_w[env_ids, :2]
    origin_xy = env.scene.env_origins[env_ids, :2]
    distance = torch.norm(robot_xy - origin_xy, dim=1)

    # === 默认晋级/降级（距离逻辑）===
    half_size = terrain_gen_cfg.size[0] / 2
    move_up = distance > half_size
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = move_down & ~move_up

    # === 对出口类子地形覆盖晋级条件 ===
    exit_positions = getattr(terrain_gen_cfg, "terrain_exit_positions", None)
    if exit_col_mask is not None and exit_positions is not None:
        # 转换 exit_positions 为 tensor（并缓存）
        if not isinstance(exit_positions, torch.Tensor):
            exit_positions = torch.tensor(exit_positions, dtype=torch.float32, device=env.device)
            terrain_gen_cfg._exit_positions_tensor = exit_positions
        else:
            exit_positions = getattr(terrain_gen_cfg, "_exit_positions_tensor", exit_positions)

        types = terrain.terrain_types[env_ids].long()
        levels = terrain.terrain_levels[env_ids].long()

        # 找出哪些 env 在出口类子地形上
        is_exit_terrain = exit_col_mask[types]  # (N,) bool

        if is_exit_terrain.any():
            levels_clamped = levels.clamp(0, exit_positions.shape[0] - 1)
            types_clamped = types.clamp(0, exit_positions.shape[1] - 1)
            goal_xy = exit_positions[levels_clamped, types_clamped, :2]
            dist_to_exit = torch.norm(robot_xy - goal_xy, dim=1)

            exit_threshold = 0.5
            exit_move_up = dist_to_exit < exit_threshold

            # 仅覆盖出口类子地形的 env
            move_up = torch.where(is_exit_terrain, exit_move_up, move_up)
            # 降级条件不变，但需要重新排除已晋级的
            move_down = move_down & ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
