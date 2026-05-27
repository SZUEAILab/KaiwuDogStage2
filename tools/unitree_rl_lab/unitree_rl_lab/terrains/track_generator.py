"""Track terrain generator.

Extends Isaac Lab's TerrainGeneratorCfg to support track mode where sub-terrains
are concatenated along the X-axis to form a linear track/course.

Also provides utilities to compute terrain exit positions for goal-based navigation.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass
from isaaclab.utils.timer import Timer


def compute_default_exit_info(
    terrain_size: tuple[float, float],
    terrain_origin_world: tuple[float, float, float],
) -> dict:
    """Compute default exit position for non-custom terrains.

    计算非自定义地形的默认出口位置（地形块 X 轴正向边缘中心）。

    Args:
        terrain_size: (size_x, size_y) 地形块尺寸（米）
        terrain_origin_world: (x, y, z) 地形块在世界坐标系中的原点

    Returns:
        exit_info: dict {"position": (x, y, z), "yaw": float}
    """
    # Exit at X-positive edge center
    exit_x = terrain_origin_world[0] + terrain_size[0] - 0.5  # 0.5m from edge
    exit_y = terrain_origin_world[1] + terrain_size[1] / 2.0  # Y center
    exit_z = terrain_origin_world[2]  # ground level

    return {
        "position": (exit_x, exit_y, exit_z),
        "yaw": 0.0,  # facing +X direction
    }


class TrackTerrainGenerator(TerrainGenerator):
    """Terrain generator that arranges sub-terrains in a linear track layout.

    In track mode:
    - Sub-terrains are concatenated along the **X-axis** (row direction) within each track.
    - Each column (Y-axis) is an independent parallel track with its own difficulty level.
    - num_rows = track_length (number of sub-terrains per track, along X).
    - num_cols = num_parallel_tracks (number of parallel tracks, along Y).

    Isaac Lab coordinate convention (from parent ``_add_sub_terrain``):
        row → X-axis position,  col → Y-axis position.

    Additionally, this generator records exit positions for each terrain block,
    enabling goal-based navigation tasks.
    """

    def __init__(self, cfg: TrackTerrainGeneratorCfg, device: str = "cpu"):
        """Initialize the track terrain generator.

        Args:
            cfg: Configuration for the track terrain generator.
            device: The device to use for the flat patches tensor.
        """
        # Map track semantics onto Isaac Lab grid dimensions:
        #   track_length      → num_rows (X-axis, forward direction)
        #   num_parallel_tracks → num_cols (Y-axis, lateral direction)
        cfg.num_rows = cfg.track_length
        cfg.num_cols = cfg.num_parallel_tracks

        # Initialize exit position storage
        # Will be populated during terrain generation
        self.terrain_exit_positions: np.ndarray | None = None  # (num_rows, num_cols, 3)
        self.terrain_exit_yaws: np.ndarray | None = None  # (num_rows, num_cols)
        # Per-block spawn positions for maze random spawn
        self.terrain_spawn_positions: dict[tuple[int, int], list[tuple[float, float, float]]] | None = None

        super().__init__(cfg, device)
        # Apply same centering offset as TerrainGenerator.__init__:
        #   X offset = -size[0] * num_rows * 0.5
        #   Y offset = -size[1] * num_cols * 0.5
        if self.terrain_exit_positions is not None:
            self.terrain_exit_positions[:, :, 0] -= cfg.size[0] * cfg.num_rows * 0.5
            self.terrain_exit_positions[:, :, 1] -= cfg.size[1] * cfg.num_cols * 0.5
        cfg.terrain_exit_positions = self.terrain_exit_positions
        cfg.terrain_exit_yaws = self.terrain_exit_yaws
        # Apply centering offset to spawn positions
        if self.terrain_spawn_positions is not None:
            offset_x = -cfg.size[0] * cfg.num_rows * 0.5
            offset_y = -cfg.size[1] * cfg.num_cols * 0.5
            for key, positions in self.terrain_spawn_positions.items():
                self.terrain_spawn_positions[key] = [(p[0] + offset_x, p[1] + offset_y, p[2]) for p in positions]
        cfg.terrain_spawn_positions = self.terrain_spawn_positions

    def _post_init_exit_storage(self):
        """Initialize exit position storage arrays after knowing grid dimensions."""
        if self.terrain_exit_positions is None:
            self.terrain_exit_positions = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype=np.float32)
            self.terrain_exit_yaws = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.float32)

    def _store_exit_info(self, sub_row: int, sub_col: int, exit_info: dict, terrain_origin_world: tuple):
        """Store exit info for a terrain block.

        Args:
            sub_row: Row index of the terrain block (X-axis / track sequence position)
            sub_col: Column index of the terrain block (Y-axis / parallel track index)
            exit_info: Dict with "position" (local coords) and "yaw"
            terrain_origin_world: World origin of this terrain block
        """
        self._post_init_exit_storage()

        # Convert local exit position to world coordinates
        local_pos = exit_info["position"]
        world_pos = (
            terrain_origin_world[0] + local_pos[0],
            terrain_origin_world[1] + local_pos[1],
            terrain_origin_world[2] + local_pos[2],
        )

        self.terrain_exit_positions[sub_row, sub_col] = world_pos
        self.terrain_exit_yaws[sub_row, sub_col] = exit_info["yaw"]

        # Store spawn positions if provided (maze terrains)
        spawn_positions = exit_info.get("spawn_positions")
        if spawn_positions is not None:
            if self.terrain_spawn_positions is None:
                self.terrain_spawn_positions = {}
            # Convert local spawn positions to world coordinates
            world_spawns = [
                (terrain_origin_world[0] + sp[0], terrain_origin_world[1] + sp[1], terrain_origin_world[2] + sp[2])
                for sp in spawn_positions
            ]
            self.terrain_spawn_positions[(sub_row, sub_col)] = world_spawns

    def _generate_curriculum_terrains(self):
        """Generate track terrains with difficulty fixed per col (deterministic).

        Design decision: ``difficulty = col / num_cols`` (no random jitter).
        In track mode, each row along X is a distinct sub-terrain type in the
        track sequence (slope → stairs → maze ...), so "col varies difficulty"
        and "row varies terrain type" are fully decoupled — we don't need the
        ``rand`` jitter that Isaac Lab's parent class uses to diversify
        identical terrains across rows. This gives the caller a rock-solid
        guarantee: **col=k always corresponds to difficulty = k / num_cols**.

        - Outer loop: col (Y-axis) = each parallel track, difficulty FIXED by col.
        - Inner loop: row (X-axis) = track sequence position (slope→pyramid→...→maze).

        设计决策：``difficulty = col / num_cols``（无随机抖动）。
        Track 模式下同一 col 的不同 row 本来就是不同的子地形（slope → stairs → maze），
        "col 决定难度、row 决定类型" 已完全解耦，不需要父类那个用来给相同地形增加多样性的
        ``rand`` 抖动。由此给调用方一个硬保证：**col=k 就是 difficulty = k / num_cols**。
        """
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrain_names = list(self.cfg.sub_terrains.keys())

        for sub_col in range(self.cfg.num_cols):
            # Compute difficulty for this parallel track — fixed, no random jitter.
            # 计算当前并行赛道的难度 —— 固定值，无随机抖动
            lower, upper = self.cfg.difficulty_range
            difficulty = sub_col / self.cfg.num_cols
            difficulty = lower + (upper - lower) * difficulty

            # Generate sub-terrain sequence for this track
            # 生成当前赛道的子地形序列
            sequence = self._get_track_sequence(sub_terrains_cfgs, sub_terrain_names)

            for sub_row, sub_cfg in enumerate(sequence):
                mesh, origin, exit_info = self._get_terrain_mesh_with_exit(difficulty, sub_cfg)
                terrain_origin_world = self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)
                self._store_exit_info(sub_row, sub_col, exit_info, terrain_origin_world)

    def _generate_random_terrains(self):
        """Generate track terrains with difficulty fixed per col (same as curriculum path).

        In track mode we keep ``difficulty = col / num_cols`` even when
        ``curriculum=False``. Semantics:

        - ``curriculum=True``  → reset_root_state_track_start initializes col
          within ``[0, max_init_terrain_level]``, letting users gate early-stage
          difficulty; ``terrain_levels_vel`` then promotes/demotes col by
          training performance.
        - ``curriculum=False`` → reset uniformly covers the full col range
          (data-augmentation style), but col=k is still mapped to the same
          fixed difficulty as the curriculum path. Only the col **initialization**
          distribution differs between the two modes; the col→difficulty
          mapping is identical.

        This matches the user mental model: "col=0 is difficulty 0".

        Track 模式下 ``curriculum=False`` 仍保持 ``difficulty = col / num_cols``。语义：

        - ``curriculum=True``  → reset 在 ``[0, max_init_terrain_level]`` 内初始化 col，
          实现早期训练难度控制，后续由 ``terrain_levels_vel`` 根据表现升降 col。
        - ``curriculum=False`` → reset 在全 col 范围均匀采样（数据增强风格），
          但 col=k 依然映射到与 curriculum 路径完全相同的固定难度。两者的差别只在
          col 的**初始化分布**，col→难度 的映射完全一致。

        满足用户心智模型："col=0 就是 0 难度"。
        """
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrain_names = list(self.cfg.sub_terrains.keys())

        for sub_col in range(self.cfg.num_cols):
            # Same formula as curriculum path — deterministic col→difficulty.
            # 与 curriculum 路径相同的公式 —— 确定性的 col→难度 映射
            lower, upper = self.cfg.difficulty_range
            difficulty = sub_col / self.cfg.num_cols
            difficulty = lower + (upper - lower) * difficulty

            # Generate sub-terrain sequence for this track
            # 生成当前赛道的子地形序列
            sequence = self._get_track_sequence(sub_terrains_cfgs, sub_terrain_names)

            for sub_row, sub_cfg in enumerate(sequence):
                mesh, origin, exit_info = self._get_terrain_mesh_with_exit(difficulty, sub_cfg)
                terrain_origin_world = self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)
                self._store_exit_info(sub_row, sub_col, exit_info, terrain_origin_world)

    def _get_terrain_mesh_with_exit(
        self, difficulty: float, sub_cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
        """Get terrain mesh along with exit information.

        Calls the terrain function, merges the mesh list into a single
        ``trimesh.Trimesh``, and centers it — matching the behaviour of the
        parent ``_get_terrain_mesh`` method so the result can be passed
        directly to ``_add_sub_terrain``.

        Args:
            difficulty: Difficulty level for the terrain
            sub_cfg: Sub-terrain configuration

        Returns:
            Tuple of (mesh, origin, exit_info)
        """
        # Copy config and set difficulty / seed (same as parent _get_terrain_mesh)
        sub_cfg = sub_cfg.copy()
        sub_cfg.difficulty = float(difficulty)
        sub_cfg.seed = self.cfg.seed

        # Call the terrain function
        result = sub_cfg.function(difficulty, sub_cfg)

        # Handle different return formats
        if len(result) == 3:
            # New format: (meshes, origin, exit_info)
            meshes, origin, exit_info = result
        elif len(result) == 2:
            # Old format: (meshes, origin) - prefer cfg-attached exit, fallback to default
            meshes, origin = result
            exit_info = getattr(sub_cfg, "exit_info", None) or compute_default_exit_info(sub_cfg.size, (0, 0, 0))
        else:
            raise ValueError(f"Unexpected terrain function return format: {len(result)} elements")

        # Merge mesh list and center (same as parent _get_terrain_mesh)
        mesh = trimesh.util.concatenate(meshes)
        transform = np.eye(4)
        transform[0:2, -1] = -sub_cfg.size[0] * 0.5, -sub_cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]

        return mesh, origin, exit_info

    def _add_sub_terrain(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, sub_row: int, sub_col: int, sub_cfg: SubTerrainBaseCfg
    ) -> tuple[float, float, float]:
        """Add a sub-terrain to the terrain mesh and return its world origin.

        Override parent method to return terrain world origin for exit position calculation.

        Returns:
            Tuple of (terrain_origin_x, terrain_origin_y, terrain_origin_z) in world coordinates
        """
        # Calculate terrain world position (Isaac Lab: row→X, col→Y)
        terrain_origin_x = sub_row * sub_cfg.size[0]
        terrain_origin_y = sub_col * sub_cfg.size[1]
        terrain_origin_z = 0.0

        # Call parent implementation
        super()._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)

        return (terrain_origin_x, terrain_origin_y, terrain_origin_z)

    def _add_terrain_border(self):
        """Add the standard border plus side walls separating parallel tracks.

        First calls the parent to generate the surrounding border, then — if
        ``track_wall_enabled`` is True — places ``num_cols + 1`` vertical walls
        along the Y-axis boundaries between parallel tracks. Each wall spans
        the full track length along X.

        Wall coordinates are in the pre-centering frame (Isaac Lab applies the
        global centering transform after this method). X spans from 0 to
        ``num_rows × size_x``, Y boundaries sit at ``col × size_y`` for
        ``col ∈ [0, num_cols]``.
        """
        # Generate the standard outer border first
        super()._add_terrain_border()

        cfg: TrackTerrainGeneratorCfg = self.cfg
        if not getattr(cfg, "track_wall_enabled", True):
            return

        wall_h = float(getattr(cfg, "track_wall_height", 1.5))
        wall_t = float(getattr(cfg, "track_wall_thickness", 0.15))

        total_x = cfg.num_rows * cfg.size[0]
        total_y = cfg.num_cols * cfg.size[1]

        # Wall vertical span: from 1m below ground to wall_h above
        wall_bottom_z = -1.0
        wall_top_z = wall_h
        wall_center_z = (wall_top_z + wall_bottom_z) / 2.0
        wall_full_h = wall_top_z - wall_bottom_z

        wall_dims = (total_x, wall_t, wall_full_h)

        # Place ``num_cols + 1`` walls at Y = 0, size_y, 2*size_y, ..., total_y
        for col in range(cfg.num_cols + 1):
            wall_y = col * cfg.size[1]
            wall_center = (total_x / 2.0, wall_y, wall_center_z)
            wall = trimesh.creation.box(wall_dims, trimesh.transformations.translation_matrix(wall_center))
            self.terrain_meshes.append(wall)

    def _get_track_sequence(
        self,
        sub_terrains_cfgs: list[SubTerrainBaseCfg],
        sub_terrain_names: list[str],
    ) -> list[SubTerrainBaseCfg]:
        """Get the sub-terrain sequence for a single track.

        Args:
            sub_terrains_cfgs: List of available sub-terrain configurations.
            sub_terrain_names: List of sub-terrain names.

        Returns:
            List of sub-terrain configurations forming the track.
        """
        cfg: TrackTerrainGeneratorCfg = self.cfg

        if cfg.sub_terrains_random:
            # Random sampling from available sub-terrains
            proportions = np.array([c.proportion for c in sub_terrains_cfgs])
            proportions /= proportions.sum()
            indices = self.np_rng.choice(len(sub_terrains_cfgs), size=cfg.track_length, p=proportions)
            return [sub_terrains_cfgs[i] for i in indices]
        else:
            # Fixed sequence from sub_terrains_order
            if cfg.sub_terrains_order is not None:
                sequence = []
                for name in cfg.sub_terrains_order:
                    if name in cfg.sub_terrains:
                        sequence.append(cfg.sub_terrains[name])
                    else:
                        raise ValueError(
                            f"Sub-terrain '{name}' in sub_terrains_order not found in sub_terrains. "
                            f"Available: {list(cfg.sub_terrains.keys())}"
                        )
                # Repeat or truncate to match track_length
                while len(sequence) < cfg.track_length:
                    sequence.extend(sequence)
                return sequence[: cfg.track_length]
            else:
                # Use sub_terrains dict order, cycling as needed
                sequence = list(sub_terrains_cfgs)
                while len(sequence) < cfg.track_length:
                    sequence.extend(sub_terrains_cfgs)
                return sequence[: cfg.track_length]


@configclass
class TrackTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for the track terrain generator.

    In track mode, sub-terrains are concatenated along the **X-axis** (row direction)
    to form a linear track. The robot runs along +X through the sub-terrain sequence.

    Multiple parallel tracks are arranged along the **Y-axis** (col direction),
    each with its own difficulty level (curriculum) or random difficulty.

    Coordinate mapping:
        track_length       → num_rows (X-axis, sub-terrain sequence)
        num_parallel_tracks → num_cols (Y-axis, parallel tracks)
    """

    class_type: type = TrackTerrainGenerator

    track_length: int = 4
    """Number of sub-terrains per track (along X-axis). Defaults to 4."""

    num_parallel_tracks: int = 10
    """Number of parallel tracks (along Y-axis). Defaults to 10.

    Each parallel track is an independent course with the same sub-terrain sequence
    but potentially different difficulty levels (when using curriculum mode).
    """

    sub_terrains_random: bool = False
    """Whether to randomly sample sub-terrains for each track. Defaults to False.

    If True, sub-terrains are randomly sampled based on their proportion weights.
    If False, sub_terrains_order (or sub_terrains dict order) is used.
    """

    sub_terrains_order: list[str] | None = None
    """Ordered list of sub-terrain names to use for fixed track sequences. Defaults to None.

    If None and sub_terrains_random is False, the sub_terrains dict order is used.
    The list is cycled/truncated to match track_length.
    """

    # -- Track-level side walls -------------------------------------------

    track_wall_enabled: bool = True
    """Whether to generate side walls separating parallel tracks. Defaults to True.

    When enabled, ``num_parallel_tracks + 1`` walls are placed along the Y-axis
    at each column boundary, each wall spanning the full track length along X.
    This isolates each parallel track from its neighbours, preventing robots
    from falling into adjacent tracks on elevated terrains (stairs, slopes).
    """

    track_wall_height: float = 1.0
    """Height of the track dividing walls above ground (m). Defaults to 1.5.

    Should be tall enough to block the robot even at the highest point of
    the stairs/slope sub-terrains used in the track.
    """

    track_wall_thickness: float = 0.15
    """Thickness of the track dividing walls (m). Defaults to 0.15."""


class StandardTerrainGenerator(TerrainGenerator):
    """Standard grid terrain generator with exit position support.

    Inherits the original ``TerrainGenerator`` behaviour (proportion-based
    column assignment, curriculum / random difficulty) while additionally:
    - Tolerating terrain functions that return 3 values ``(meshes, origin, exit_info)``
    - Recording per-block exit positions for goal-based navigation
    """

    def __init__(self, cfg: "StandardTerrainGeneratorCfg", device: str = "cpu"):
        self.terrain_exit_positions: np.ndarray | None = None
        self.terrain_exit_yaws: np.ndarray | None = None
        self.terrain_spawn_positions: dict[tuple[int, int], list[tuple[float, float, float]]] | None = None
        super().__init__(cfg, device)
        if self.terrain_exit_positions is not None:
            self.terrain_exit_positions[:, :, 0] -= cfg.size[0] * cfg.num_rows * 0.5
            self.terrain_exit_positions[:, :, 1] -= cfg.size[1] * cfg.num_cols * 0.5
        cfg.terrain_exit_positions = self.terrain_exit_positions
        cfg.terrain_exit_yaws = self.terrain_exit_yaws
        if self.terrain_spawn_positions is not None:
            offset_x = -cfg.size[0] * cfg.num_rows * 0.5
            offset_y = -cfg.size[1] * cfg.num_cols * 0.5
            for key, positions in self.terrain_spawn_positions.items():
                self.terrain_spawn_positions[key] = [(p[0] + offset_x, p[1] + offset_y, p[2]) for p in positions]
        cfg.terrain_spawn_positions = self.terrain_spawn_positions

    # -- exit storage (same as TrackTerrainGenerator) ----------------------

    def _post_init_exit_storage(self):
        if self.terrain_exit_positions is None:
            self.terrain_exit_positions = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype=np.float32)
            self.terrain_exit_yaws = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.float32)

    def _store_exit_info(self, sub_row: int, sub_col: int, exit_info: dict, terrain_origin_world: tuple):
        self._post_init_exit_storage()
        local_pos = exit_info["position"]
        world_pos = (
            terrain_origin_world[0] + local_pos[0],
            terrain_origin_world[1] + local_pos[1],
            terrain_origin_world[2] + local_pos[2],
        )
        self.terrain_exit_positions[sub_row, sub_col] = world_pos
        self.terrain_exit_yaws[sub_row, sub_col] = exit_info["yaw"]

        # Store spawn positions if provided (maze terrains)
        spawn_positions = exit_info.get("spawn_positions")
        if spawn_positions is not None:
            if self.terrain_spawn_positions is None:
                self.terrain_spawn_positions = {}
            world_spawns = [
                (terrain_origin_world[0] + sp[0], terrain_origin_world[1] + sp[1], terrain_origin_world[2] + sp[2])
                for sp in spawn_positions
            ]
            self.terrain_spawn_positions[(sub_row, sub_col)] = world_spawns

    # -- mesh generation (compat with 2 or 3 return values) ----------------

    def _get_terrain_mesh_with_exit(
        self, difficulty: float, sub_cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
        import os

        from isaaclab.utils.dict import dict_to_md5_hash
        from isaaclab.utils.io import dump_yaml

        sub_cfg = sub_cfg.copy()
        sub_cfg.difficulty = float(difficulty)
        sub_cfg.seed = self.cfg.seed

        sub_terrain_hash = dict_to_md5_hash(sub_cfg.to_dict())
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            exit_info = compute_default_exit_info(sub_cfg.size, (0, 0, 0))
            return mesh, origin, exit_info

        result = sub_cfg.function(difficulty, sub_cfg)
        if len(result) == 3:
            meshes, origin, exit_info = result
        else:
            meshes, origin = result
            exit_info = getattr(sub_cfg, "exit_info", None) or compute_default_exit_info(sub_cfg.size, (0, 0, 0))

        mesh = trimesh.util.concatenate(meshes)
        transform = np.eye(4)
        transform[0:2, -1] = -sub_cfg.size[0] * 0.5, -sub_cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]

        if self.cfg.use_cache:
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, sub_cfg)

        return mesh, origin, exit_info

    # -- _add_sub_terrain: return world origin for exit calculation ---------

    def _add_sub_terrain(
        self, mesh, origin: np.ndarray, sub_row: int, sub_col: int, sub_cfg: SubTerrainBaseCfg
    ) -> tuple[float, float, float]:
        terrain_origin_x = sub_row * sub_cfg.size[0]
        terrain_origin_y = sub_col * sub_cfg.size[1]
        terrain_origin_z = 0.0
        super()._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)
        return (terrain_origin_x, terrain_origin_y, terrain_origin_z)

    # -- Curriculum: original proportion-based logic + exit collection ------

    def _generate_curriculum_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty

                mesh, origin, exit_info = self._get_terrain_mesh_with_exit(
                    difficulty, sub_terrains_cfgs[sub_indices[sub_col]]
                )
                terrain_origin_world = self._add_sub_terrain(
                    mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]]
                )
                self._store_exit_info(sub_row, sub_col, exit_info, terrain_origin_world)

    # -- Random: original proportion-based logic + exit collection ----------

    def _generate_random_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)

            mesh, origin, exit_info = self._get_terrain_mesh_with_exit(difficulty, sub_terrains_cfgs[sub_index])
            terrain_origin_world = self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])
            self._store_exit_info(sub_row, sub_col, exit_info, terrain_origin_world)


@configclass
class StandardTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for the standard terrain generator with 3-value compat.

    Behaves identically to Isaac Lab's ``TerrainGeneratorCfg`` (proportion-based
    grid layout, curriculum difficulty) but uses ``StandardTerrainGenerator``
    which can handle custom terrain functions returning 3 values.
    """

    class_type: type = StandardTerrainGenerator
