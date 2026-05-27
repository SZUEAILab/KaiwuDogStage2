"""Terrain exit position manager.

Manages exit/goal positions for all terrain types, enabling goal-based navigation tasks.
This module provides utilities to compute and store exit positions for terrain blocks.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainImporter


def compute_default_exit_for_block(
    terrain_size: tuple[float, float],
    block_origin_world: tuple[float, float, float],
) -> dict:
    """Compute default exit position for a terrain block.

    计算地形块的默认出口位置（X 轴正向边缘中心）。

    Args:
        terrain_size: (size_x, size_y) 地形块尺寸（米）
        block_origin_world: (x, y, z) 地形块左下角在世界坐标系中的位置

    Returns:
        exit_info: dict {"position": np.ndarray(3,), "yaw": float}
    """
    # Exit at X-positive edge center
    exit_x = block_origin_world[0] + terrain_size[0] - 0.5  # 0.5m from edge
    exit_y = block_origin_world[1] + terrain_size[1] / 2.0  # Y center
    exit_z = block_origin_world[2]  # ground level

    return {
        "position": np.array([exit_x, exit_y, exit_z], dtype=np.float32),
        "yaw": 0.0,  # facing +X direction
    }


def get_terrain_generator_runtime_state(terrain) -> object | None:
    """Resolve generator runtime state from a terrain importer-like object.

    Isaac Lab 的 ``TerrainImporter`` 不会保留真实 generator 实例，
    但会保留 ``terrain.cfg.terrain_generator`` 这份配置对象。
    我们把运行期回写的数据也挂在这份 cfg 上，因此这里统一解析两种来源。
    """
    if terrain is None:
        return None

    generator_state = getattr(terrain, "terrain_generator", None)
    if generator_state is not None:
        return generator_state

    terrain_cfg = getattr(terrain, "cfg", None)
    if terrain_cfg is None:
        return None
    return getattr(terrain_cfg, "terrain_generator", None)


def _get_terrain_generator_cfg(terrain_generator) -> object | None:
    """Return the config-like object behind a generator or cfg-like runtime state."""
    if terrain_generator is None:
        return None
    return getattr(terrain_generator, "cfg", terrain_generator)


def _resolve_fixed_track_sequence_names(terrain_generator) -> list[str] | None:
    """Resolve the effective fixed sub-terrain sequence for a track generator.

    Returns None when the generator is not in a deterministic fixed-order mode.
    """
    cfg = _get_terrain_generator_cfg(terrain_generator)
    if cfg is None or getattr(cfg, "sub_terrains_random", False):
        return None

    track_length = getattr(cfg, "track_length", None)
    if track_length is None or track_length <= 0:
        return None

    sequence = getattr(cfg, "sub_terrains_order", None)
    if sequence:
        sequence = list(sequence)
    else:
        sub_terrains = getattr(cfg, "sub_terrains", None)
        if not sub_terrains:
            return None
        sequence = list(sub_terrains.keys())

    if not sequence:
        return None

    while len(sequence) < track_length:
        sequence.extend(sequence)
    return sequence[:track_length]


def extract_named_track_exit_positions(terrain_generator, terrain_name: str) -> np.ndarray | None:
    """Extract exit positions for a named sub-terrain from a fixed-order track.

    In the track layout, the sub-terrain sequence runs along the **row** dimension
    (X-axis). Each column is a parallel track with the same sequence.

    Args:
        terrain_generator: TrackTerrainGenerator-like instance with cfg and terrain_exit_positions.
        terrain_name: Sub-terrain name to look for, e.g. "maze".

    Returns:
        np.ndarray of shape (K, 3) in world coordinates, or None if unavailable.
    """
    exit_positions = getattr(terrain_generator, "terrain_exit_positions", None)
    if exit_positions is None:
        return None

    sequence = _resolve_fixed_track_sequence_names(terrain_generator)
    if sequence is None:
        return None

    # Track sequence runs along rows (X-axis)
    matched_rows = [index for index, name in enumerate(sequence) if name == terrain_name]
    if not matched_rows:
        return None

    positions = np.asarray(exit_positions, dtype=np.float32)
    if positions.ndim != 3:
        return None

    # positions shape: (num_rows, num_cols, 3)
    # Select matched rows across all parallel tracks (cols)
    selected = positions[matched_rows, :, :]  # (num_matched, num_cols, 3)
    return selected.reshape(-1, 3)


class TerrainExitManager:
    """Manages terrain exit positions for goal-based navigation.

    管理地形出口位置，用于基于目标点的导航任务。

    This class:
    1. Stores precomputed exit positions for all terrain blocks
    2. Provides methods to query exit position based on robot's terrain block
    3. Supports both standard (grid) and track (linear) terrain layouts
    4. In track mode, supports progress-based goal assignment (by robot X position)
    """

    def __init__(self, device: str = "cuda:0"):
        """Initialize the terrain exit manager.

        Args:
            device: PyTorch device for tensors
        """
        self.device = device

        # Exit position storage (initialized when terrain is loaded)
        self._exit_positions: torch.Tensor | None = None  # (num_rows, num_cols, 3)
        self._exit_yaws: torch.Tensor | None = None  # (num_rows, num_cols)

        # Terrain info
        self._num_rows: int = 0
        self._num_cols: int = 0
        self._terrain_size: tuple[float, float] = (8.0, 8.0)

        # Track mode info
        self._is_track_mode: bool = False
        self._track_length: int = 0
        # X boundaries of each sub-terrain segment (after centering offset)
        # shape: (track_length + 1,) — boundaries[i] is start-X of segment i,
        # boundaries[track_length] is end-X of the last segment.
        self._track_x_boundaries: torch.Tensor | None = None

    @property
    def is_initialized(self) -> bool:
        """Check if exit positions have been initialized."""
        return self._exit_positions is not None

    @property
    def is_track_mode(self) -> bool:
        """Whether this manager is in track mode (progress-based goal)."""
        return self._is_track_mode

    @property
    def exit_positions(self) -> torch.Tensor | None:
        """Get all exit positions tensor. Shape: (num_rows, num_cols, 3)"""
        return self._exit_positions

    @property
    def exit_yaws(self) -> torch.Tensor | None:
        """Get all exit yaw angles tensor. Shape: (num_rows, num_cols)"""
        return self._exit_yaws

    def initialize_from_terrain_generator(
        self,
        terrain_generator,
        terrain_size: tuple[float, float],
    ):
        """Initialize exit positions from a terrain generator that supports exit info.

        从支持出口信息的地形生成器初始化出口位置。

        Args:
            terrain_generator: TerrainGenerator instance (TrackTerrainGenerator or similar)
            terrain_size: (size_x, size_y) terrain block size
        """
        if (
            hasattr(terrain_generator, "terrain_exit_positions")
            and terrain_generator.terrain_exit_positions is not None
        ):
            # Use precomputed exit positions from generator
            self._exit_positions = torch.tensor(
                terrain_generator.terrain_exit_positions,
                dtype=torch.float32,
                device=self.device,
            )
            self._exit_yaws = torch.tensor(
                terrain_generator.terrain_exit_yaws,
                dtype=torch.float32,
                device=self.device,
            )
            self._num_rows, self._num_cols = self._exit_positions.shape[:2]
            self._terrain_size = terrain_size
        else:
            # Compute default exits for standard terrain generator
            self._compute_default_exits(terrain_generator, terrain_size)

        # Detect track mode and build X boundaries
        cfg = _get_terrain_generator_cfg(terrain_generator)
        track_length = getattr(cfg, "track_length", None) if cfg is not None else None
        if track_length is not None and track_length > 0:
            self._is_track_mode = True
            self._track_length = track_length
            self._build_track_x_boundaries(terrain_size)

    def _build_track_x_boundaries(self, terrain_size: tuple[float, float]):
        """Build X-axis boundaries for each sub-terrain segment in track mode.

        构建 track 模式下每个子地形段的 X 轴边界。

        After the centering offset applied by TerrainGenerator, the grid origin
        is shifted by ``-size_x * num_rows * 0.5``.  Therefore segment *i*
        spans:  ``[offset + i * size_x,  offset + (i+1) * size_x)``.

        boundaries tensor has shape ``(track_length + 1,)``::

            boundaries[0]            = offset  (start of segment 0)
            boundaries[i]            = offset + i * size_x
            boundaries[track_length] = offset + track_length * size_x  (end)
        """
        size_x = terrain_size[0]
        offset = -size_x * self._track_length * 0.5
        boundaries = torch.tensor(
            [offset + i * size_x for i in range(self._track_length + 1)],
            dtype=torch.float32,
            device=self.device,
        )
        self._track_x_boundaries = boundaries

    def _compute_default_exits(
        self,
        terrain_generator_cfg,
        terrain_size: tuple[float, float],
    ):
        """Compute default exit positions for a standard terrain generator.

        为标准地形生成器计算默认出口位置。

        Args:
            terrain_generator_cfg: Standard terrain generator cfg or generator-like state
            terrain_size: (size_x, size_y) terrain block size
        """
        if terrain_generator_cfg is None:
            raise RuntimeError("terrain_generator_cfg is required to compute default exits")

        num_rows = terrain_generator_cfg.num_rows
        num_cols = terrain_generator_cfg.num_cols

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._terrain_size = terrain_size

        exit_positions = np.zeros((num_rows, num_cols, 3), dtype=np.float32)
        exit_yaws = np.zeros((num_rows, num_cols), dtype=np.float32)

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate block world origin
                block_origin = (
                    col * terrain_size[0],
                    row * terrain_size[1],
                    0.0,
                )
                exit_info = compute_default_exit_for_block(terrain_size, block_origin)
                exit_positions[row, col] = exit_info["position"]
                exit_yaws[row, col] = exit_info["yaw"]

        self._exit_positions = torch.tensor(exit_positions, dtype=torch.float32, device=self.device)
        self._exit_yaws = torch.tensor(exit_yaws, dtype=torch.float32, device=self.device)

    def initialize_for_plane(self, num_envs: int, default_distance: float = 10.0):
        """Initialize exit positions for plane terrain (no generator).

        为平面地形初始化出口位置（无地形生成器）。

        Args:
            num_envs: Number of environments
            default_distance: Default goal distance in front of robot
        """
        # For plane terrain, we'll update goals dynamically based on robot position
        # Initialize with zeros; actual positions set in update_goal_positions
        self._num_rows = 1
        self._num_cols = 1
        self._terrain_size = (default_distance * 2, default_distance * 2)
        self._is_track_mode = False

        self._exit_positions = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.device)
        self._exit_yaws = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

    def get_goal_positions(
        self,
        terrain_levels: torch.Tensor,
        terrain_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get goal positions and yaws for given terrain indices (standard grid mode).

        根据地形块索引获取目标位置和朝向。
        此方法用于标准 grid 模式；track 模式请使用 ``get_track_goal_positions``。

        Args:
            terrain_levels: (num_envs,) row indices
            terrain_types: (num_envs,) column indices

        Returns:
            Tuple of (goal_positions, goal_yaws)
            - goal_positions: (num_envs, 3) world coordinates
            - goal_yaws: (num_envs,) yaw angles in radians
        """
        if not self.is_initialized:
            raise RuntimeError("TerrainExitManager not initialized. Call initialize_* first.")

        # Clamp indices to valid range
        levels_clamped = terrain_levels.clamp(0, self._num_rows - 1).long()
        types_clamped = terrain_types.clamp(0, self._num_cols - 1).long()

        goal_positions = self._exit_positions[levels_clamped, types_clamped]
        goal_yaws = self._exit_yaws[levels_clamped, types_clamped]

        return goal_positions, goal_yaws

    def get_track_goal_positions(
        self,
        robot_pos_x: torch.Tensor,
        terrain_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get goal positions for track mode — always point to the track finish (maze exit).

        Track 模式下，所有机器人的目标始终固定指向赛道终点（最后一段 maze 的出口，
        即 row = track_length - 1）。不再根据机器人 X 坐标逐段递进。

        Args:
            robot_pos_x: (num_envs,) robot X coordinates in world frame (unused, kept for API compat)
            terrain_types: (num_envs,) column indices (parallel track index)

        Returns:
            Tuple of (goal_positions, goal_yaws)
            - goal_positions: (num_envs, 3) world coordinates
            - goal_yaws: (num_envs,) yaw angles in radians
        """
        if not self.is_initialized:
            raise RuntimeError("TerrainExitManager not initialized. Call initialize_* first.")
        if not self._is_track_mode:
            raise RuntimeError("get_track_goal_positions called but not in track mode.")

        # 固定目标：赛道终点 = 最后一段的出口 (row = track_length - 1)
        finish_row = self._track_length - 1
        types_clamped = terrain_types.clamp(0, self._num_cols - 1).long()

        goal_positions = self._exit_positions[finish_row, types_clamped]
        goal_yaws = self._exit_yaws[finish_row, types_clamped]

        return goal_positions, goal_yaws

    def get_track_segment_index(self, robot_pos_x: torch.Tensor) -> torch.Tensor:
        """Get track segment index for each robot based on X position.

        根据机器人 X 坐标返回当前所在 track 段的索引（0-indexed）。

        Args:
            robot_pos_x: (num_envs,) robot X coordinates in world frame

        Returns:
            torch.Tensor: (num_envs,) segment indices in [0, track_length-1]
        """
        if not self._is_track_mode or self._track_x_boundaries is None:
            raise RuntimeError("Not in track mode or boundaries not initialized.")

        boundaries = self._track_x_boundaries
        segment_idx = (robot_pos_x.unsqueeze(-1) >= boundaries.unsqueeze(0)).sum(dim=-1) - 1
        return segment_idx.clamp(0, self._track_length - 1).long()
