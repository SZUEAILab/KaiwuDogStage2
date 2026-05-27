"""Custom event functions for locomotion tasks."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform


def _is_track_mode(terrain: TerrainImporter) -> bool:
    """Check if the terrain generator is in Track mode.

    Track mode uses TrackTerrainGeneratorCfg where sub-terrains form a linear
    course — robots must spawn at the fixed entry point, not randomly.
    """
    from unitree_rl_lab.terrains.track_generator import TrackTerrainGeneratorCfg

    terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    return isinstance(terrain_gen_cfg, TrackTerrainGeneratorCfg)


def _randomize_standard_terrain_levels_on_reset(
    terrain: TerrainImporter,
    env_ids: torch.Tensor,
    *,
    device: torch.device,
    scene=None,
) -> bool:
    """Uniformly resample standard terrain levels while preserving terrain types.

    This is used only for standard mode with ``curriculum=false``: terrain meshes
    keep the deterministic row→difficulty grid, but each reset chooses a fresh
    row for the resetting envs. Track mode must not use this helper because its
    difficulty axis is terrain_types/columns, not terrain_levels/rows.
    """
    cfg = getattr(terrain, "cfg", None)
    if not bool(getattr(cfg, "randomize_terrain_levels_on_reset", False)):
        return False

    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    env_origins = getattr(terrain, "env_origins", None)
    scene_env_origins = getattr(scene, "env_origins", None) if scene is not None else None
    if env_origins is None:
        env_origins = scene_env_origins

    if terrain_levels is None or terrain_types is None or terrain_origins is None or env_origins is None:
        return False

    num_rows = int(terrain_origins.shape[0])
    num_cols = int(terrain_origins.shape[1]) if terrain_origins.ndim >= 2 else 1
    if num_rows <= 0 or num_cols <= 0:
        return False

    new_levels = torch.randint(0, num_rows, (len(env_ids),), device=device)
    terrain_levels[env_ids] = new_levels.to(dtype=terrain_levels.dtype)

    selected_levels = terrain_levels[env_ids].long().clamp(0, num_rows - 1)
    selected_types = terrain_types[env_ids].long().clamp(0, num_cols - 1)
    new_origins = terrain_origins[selected_levels, selected_types]
    env_origins[env_ids] = new_origins
    if scene_env_origins is not None and scene_env_origins is not env_origins:
        scene_env_origins[env_ids] = new_origins
    return True


def reset_root_state_maze_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot to a random cell center within the assigned terrain block.

    **Standard mode**: picks a random cell center from the maze's
    pre-computed ``spawn_positions`` list. Random yaw, no position offset.

    **Track mode**: falls back to the standard ``env_origins`` behaviour
    (fixed entry point), because Track terrains form a linear course where
    the robot must start from the beginning.

    If no ``spawn_positions`` are available for a given terrain block (e.g.
    non-maze terrain), it also falls back to ``env_origins``.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        velocity_range: Velocity randomization ranges (same as ``reset_root_state_uniform``).
        asset_cfg: The asset to reset. Defaults to ``"robot"``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    root_states = asset.data.default_root_state[env_ids].clone()

    # In Track mode, always use env_origins (fixed entry point)
    use_random_spawn = not _is_track_mode(terrain)

    spawn_positions_map = None
    if use_random_spawn:
        terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
        spawn_positions_map = getattr(terrain_gen_cfg, "terrain_spawn_positions", None) if terrain_gen_cfg else None

    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if use_random_spawn:
        _randomize_standard_terrain_levels_on_reset(terrain, env_ids, device=asset.device, scene=env.scene)

    # Start from default root state (standing height is relative to terrain origin)
    # Must add env_origins to get correct world-frame position, matching Isaac Lab's
    # reset_root_state_uniform: positions = default_root_state[:, 0:3] + env_origins
    positions = root_states[:, 0:3].clone()

    if (
        use_random_spawn
        and spawn_positions_map is not None
        and terrain_levels is not None
        and terrain_types is not None
    ):
        for i, env_id in enumerate(env_ids):
            level = terrain_levels[env_id].item()
            ttype = terrain_types[env_id].item()
            key = (int(level), int(ttype))

            if key in spawn_positions_map:
                sp_list = spawn_positions_map[key]
                idx = torch.randint(0, len(sp_list), (1,)).item()
                sp = sp_list[idx]
                # spawn_positions are in world coords; z from env_origins + default standing height
                positions[i, 0] = sp[0]
                positions[i, 1] = sp[1]
                positions[i, 2] += env.scene.env_origins[env_id, 2]
            else:
                # Non-maze block: use env_origins (xyz including z offset)
                positions[i, :3] += env.scene.env_origins[env_id, :3]
    else:
        # Track mode or no spawn positions: use env_origins (xyz including z offset)
        positions[:, :3] += env.scene.env_origins[env_ids, :3]

    # Random yaw
    yaw = sample_uniform(-3.14159, 3.14159, (len(env_ids),), device=asset.device)
    zero = torch.zeros_like(yaw)
    orientations_delta = quat_from_euler_xyz(zero, zero, yaw)
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # Velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    # Write to simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_track_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    pose_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot to the track with eval/train-aware spawning strategy.

    **Evaluation mode** (``env._is_eval == True``):
        All robots spawn at the track **start** (row=0, first sub-terrain),
        matching deterministic evaluation protocol. Col (= parallel track
        difficulty) is NOT modified: Isaac Lab's default spread across all
        cols is preserved so eval faithfully covers the full difficulty range.

    **Training mode** (``env._is_eval == False``, default):
        - **Row**: when curriculum is enabled, every robot starts from row=0
          (track start) so success means completing the whole track. When
          curriculum is disabled, each robot randomly picks a **non-maze**
          sub-terrain segment as data augmentation.
        - **Col (difficulty curriculum)**: if ``terrain.cfg.max_init_terrain_level``
          is set and ``terrain.cfg.terrain_generator.curriculum`` is enabled, col
          is initialized once per env into ``[0, max_init_terrain_level]``. Later
          promotion/demotion is handled by ``terrain_levels_vel`` track branch,
          which updates col (= difficulty) rather than row (= segment index). If
          ``max_init_terrain_level`` is None or ``>= num_cols-1`` the default
          full-range behaviour is preserved.

    If ``sub_terrains_random=True`` or ``sub_terrains_order`` is empty/all-maze,
    falls back to row=0 spawn (col handling is unchanged).

    **Non-track mode**: delegates to ``reset_root_state_maze_random``.

    Args:
        env: The environment instance. Reads ``env._is_eval`` to select strategy.
        env_ids: Indices of environments to reset.
        velocity_range: Velocity randomization ranges.
        pose_range: Optional XY/yaw offset ranges. Defaults to ±0.3 m XY, ±0.5 rad yaw.
        asset_cfg: The asset to reset. Defaults to ``"robot"``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # Non-track mode: fall back to maze_random spawn
    if not _is_track_mode(terrain):
        return reset_root_state_maze_random(env, env_ids, velocity_range, asset_cfg)

    # --- Resolve eval/train mode ---
    is_eval = bool(getattr(env, "_is_eval", False))
    terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    curriculum_on = bool(getattr(terrain_gen_cfg, "curriculum", False))

    # --- Choose target row per env ---
    num_rows = terrain.terrain_origins.shape[0] if terrain.terrain_origins is not None else 1
    device = asset.device

    if is_eval or curriculum_on:
        # Eval and track-curriculum training both start from row=0.
        # 评估与 track 课程训练都从 row=0（整条赛道起点）开始。
        #
        # In track mode, row is the segment index, not difficulty. Runtime
        # curriculum promotes/demotes col, so row should not be randomly
        # overwritten during curriculum training.
        # Track 模式下 row 是赛道段位置，不是难度。运行时课程升降 col，
        # 因此课程训练阶段不能再随机覆盖 row。
        target_rows = torch.zeros(len(env_ids), dtype=torch.long, device=device)
    else:
        # Non-curriculum training: randomly pick a non-maze row for each env.
        # 非课程训练：随机挑非 maze 段作为起点，作为数据增强。
        non_maze_rows = _resolve_non_maze_rows(terrain, num_rows)
        if len(non_maze_rows) == 0:
            # No non-maze rows available — fall back to row=0
            target_rows = torch.zeros(len(env_ids), dtype=torch.long, device=device)
        else:
            non_maze_tensor = torch.tensor(non_maze_rows, dtype=torch.long, device=device)
            # Random pick with replacement
            pick_idx = torch.randint(0, len(non_maze_rows), (len(env_ids),), device=device)
            target_rows = non_maze_tensor[pick_idx]

    # --- Track curriculum: initialize col (= difficulty) once in training mode ---
    # Track 课程：训练模式下只在首次 reset 初始化 col（= 难度档）。
    #
    # Track-mode col semantics: col indexes the parallel track whose difficulty
    # is `col / num_cols` (see TrackTerrainGenerator). Isaac Lab's default
    # initial allocation spreads envs uniformly across all cols, so early-stage
    # training would immediately see full difficulty. We use `max_init_terrain_level`
    # as the initial col cap, but only initialize once; later promotion/demotion is
    # owned by `terrain_levels_vel` track branch.
    #
    # Track 模式下 col = 并行赛道编号 = 难度档（见 TrackTerrainGenerator）。Isaac Lab
    # 初始会把 env 均匀分布到所有 col，导致训练一开始就覆盖全难度。这里用
    # `max_init_terrain_level` 作为初始 col 上限，但只初始化一次；后续升降档由
    # `terrain_levels_vel` 的 track 分支维护，避免每次 reset 覆盖课程结果。
    if (
        not is_eval
        and curriculum_on
        and terrain.terrain_origins is not None
        and hasattr(terrain, "terrain_types")
        and terrain.terrain_types is not None
    ):
        num_cols_total = terrain.terrain_origins.shape[1]
        max_init_level_cfg = getattr(terrain.cfg, "max_init_terrain_level", None)
        if max_init_level_cfg is not None:
            max_col_idx = min(int(max_init_level_cfg), num_cols_total - 1)
            if max_col_idx < num_cols_total - 1:
                initialized = getattr(terrain, "_track_curriculum_col_initialized", None)
                if initialized is None or initialized.numel() != terrain.terrain_types.numel():
                    initialized = torch.zeros_like(terrain.terrain_types, dtype=torch.bool, device=device)
                    terrain._track_curriculum_col_initialized = initialized

                uninit_mask = ~initialized[env_ids]
                if uninit_mask.any():
                    init_env_ids = env_ids[uninit_mask]
                    new_cols = torch.randint(0, max_col_idx + 1, (len(init_env_ids),), device=device)
                    terrain.terrain_types[init_env_ids] = new_cols.to(terrain.terrain_types.dtype)
                    initialized[init_env_ids] = True

    # --- Update terrain_levels and env_origins for the chosen rows ---
    if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
        terrain.terrain_levels[env_ids] = target_rows
        if terrain.terrain_origins is not None:
            terrain_types = (
                terrain.terrain_types[env_ids]
                if hasattr(terrain, "terrain_types") and terrain.terrain_types is not None
                else torch.zeros_like(env_ids)
            )
            terrain.env_origins[env_ids] = terrain.terrain_origins[target_rows, terrain_types]

    # --- Compute spawn positions ---
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3].clone()

    # Add env_origins (now pointing to the chosen row)
    positions[:, :3] += env.scene.env_origins[env_ids, :3]

    # Random XY offset + yaw randomization
    if pose_range is None:
        pose_range = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.5, 0.5)}
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    positions[:, 0] += sample_uniform(x_range[0], x_range[1], (len(env_ids),), device=device)
    positions[:, 1] += sample_uniform(y_range[0], y_range[1], (len(env_ids),), device=device)

    yaw_range = pose_range.get("yaw", (-0.5, 0.5))
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids),), device=device)
    zero = torch.zeros_like(yaw)
    orientations_delta = quat_from_euler_xyz(zero, zero, yaw)
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # Velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=device)
    velocities = root_states[:, 7:13] + rand_samples

    # Write to simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _resolve_non_maze_rows(terrain: TerrainImporter, num_rows: int) -> list[int]:
    """Resolve the list of row indices (X-axis sub-terrain positions) that are NOT maze.

    Reads the TrackTerrainGeneratorCfg's ``sub_terrains_order`` and returns
    all row indices whose sub-terrain name does not contain ``"maze"``.

    Args:
        terrain: The terrain importer instance.
        num_rows: Total number of rows (track_length).

    Returns:
        List of row indices in [0, num_rows) where the sub-terrain is non-maze.
        Returns an empty list if ``sub_terrains_random=True`` behaviour cannot
        determine fixed assignments, or if no sub_terrains_order is set.
        Returns all rows if the generator is in random mode (no fixed mapping,
        treat as non-maze to allow any row).
    """
    terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    if terrain_gen_cfg is None:
        return list(range(num_rows))

    # Random mode: no deterministic row-to-name mapping; treat all rows as usable
    if getattr(terrain_gen_cfg, "sub_terrains_random", False):
        return list(range(num_rows))

    sub_terrains_order = getattr(terrain_gen_cfg, "sub_terrains_order", None)
    if not sub_terrains_order:
        # No explicit order: fall back to sub_terrains dict order
        sub_terrains = getattr(terrain_gen_cfg, "sub_terrains", None)
        if not sub_terrains:
            return list(range(num_rows))
        sub_terrains_order = list(sub_terrains.keys())

    # Cycle/truncate to match num_rows (mirrors TrackTerrainGenerator._get_track_sequence)
    sequence = list(sub_terrains_order)
    while len(sequence) < num_rows:
        sequence.extend(sub_terrains_order)
    sequence = sequence[:num_rows]

    # Non-maze rows: name does not contain "maze" (case-insensitive)
    non_maze_rows = [i for i, name in enumerate(sequence) if "maze" not in name.lower()]
    return non_maze_rows


def reset_root_state_eval_level_aware(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    pose_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Eval-only deterministic reset: assign (row, col) per env from level list.

    Reads ``env.cfg._eval_level_list`` (and ``env.cfg._eval_sub_terrains`` for
    standard mode) which were stashed by ``apply_usr_conf_to_env_cfg`` when the
    eval config contains a ``[terrain] level = [...]`` field. Each env is
    deterministically assigned to a (row, col) cell so the evaluator can
    faithfully cover every (level, sub_terrain) combination.

    评估专用确定性 reset：根据 level 列表为每个 env 分配 (row, col)。
    读取 ``env.cfg._eval_level_list``（以及 standard 模式下的
    ``env.cfg._eval_sub_terrains``），这些字段由 ``apply_usr_conf_to_env_cfg``
    在评估配置含 ``[terrain] level = [...]`` 时挂载。每个 env 都被确定性地分配
    到一个 (row, col) 格子，确保评估覆盖所有 (level, sub_terrain) 组合。

    Standard mode placement order (Cartesian product, level outer, sub inner):
    Standard 模式放置顺序（笛卡尔积，level 外层 + sub 内层）：

        env_i → row = level_list[(i // num_sub) % num_levels]
                col = i % num_sub

    Track mode placement order (col cycles through level list):
    Track 模式放置顺序（col 按 level 列表循环分配）：

        env_i → row = 0 (always start from track origin)
                col = level_list[i % num_levels]

    Falls back to the original reset behaviour if the eval level list is not
    set on ``env.cfg`` (so this function is safe to wire as the default reset).
    若 ``env.cfg`` 上没有挂 eval level 列表，则回退到原有 reset 行为
    （因此该函数可以安全作为默认 reset 函数使用）。

    Args:
        env: The environment instance. Reads ``env.cfg._eval_level_list`` and
             ``env.cfg._eval_sub_terrains`` for placement; falls back when absent.
        env_ids: Indices of environments to reset.
        velocity_range: Velocity randomization ranges (same as other reset funcs).
        pose_range: Optional XY/yaw offset ranges. Defaults to ±0.3 m XY, ±0.5 rad yaw.
        asset_cfg: The asset to reset. Defaults to ``"robot"``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # --- Read eval-level config from env.cfg ---
    # 从 env.cfg 读取 eval-level 配置
    level_list = getattr(env.cfg, "_eval_level_list", None)
    sub_list = getattr(env.cfg, "_eval_sub_terrains", None)

    # If eval level list is not configured, fall back to the original reset
    # 未配置 eval level 列表时回退到原有逻辑
    if not level_list:
        if _is_track_mode(terrain):
            return reset_root_state_track_start(env, env_ids, velocity_range, pose_range, asset_cfg)
        return reset_root_state_maze_random(env, env_ids, velocity_range, asset_cfg)

    device = asset.device
    num_levels = len(level_list)
    levels_tensor = torch.tensor(level_list, dtype=torch.long, device=device)

    # `env_ids` is a tensor of global env indices; we use those indices directly
    # so that the (row, col) assignment is stable across resets (env i always
    # falls into the same cell, ensuring reproducible eval).
    # `env_ids` 是全局 env 索引张量，直接拿来计算 (row, col)，保证每次 reset
    # 同一个 env 始终落到同一格，评估结果可复现。
    env_idx_global = env_ids.to(dtype=torch.long, device=device)

    if _is_track_mode(terrain):
        # Track: row=0 (track start), col cycles through level list
        # Track：row=0（赛道起点），col 按 level 列表循环
        row_indices = torch.zeros(len(env_ids), dtype=torch.long, device=device)
        col_indices = levels_tensor[env_idx_global % num_levels]
    else:
        # Standard: cartesian product (level outer, sub inner)
        # Standard：笛卡尔积（level 外层、sub 内层）
        if sub_list:
            num_sub = len(sub_list)
        elif terrain.terrain_origins is not None:
            num_sub = terrain.terrain_origins.shape[1]
        else:
            num_sub = 1
        if num_sub <= 0:
            num_sub = 1

        sub_idx = env_idx_global % num_sub
        level_idx = (env_idx_global // num_sub) % num_levels
        row_indices = levels_tensor[level_idx]
        col_indices = sub_idx

    # --- Clamp indices to terrain grid bounds (defensive) ---
    # 防御性裁剪：把 (row, col) 限制在 terrain grid 范围内
    if terrain.terrain_origins is not None:
        max_row = terrain.terrain_origins.shape[0] - 1
        max_col = terrain.terrain_origins.shape[1] - 1
        row_indices = row_indices.clamp(min=0, max=max_row)
        col_indices = col_indices.clamp(min=0, max=max_col)

    # --- Update terrain_levels / terrain_types / env_origins ---
    # 写入 terrain_levels / terrain_types / env_origins
    if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
        terrain.terrain_levels[env_ids] = row_indices.to(terrain.terrain_levels.dtype)
    if hasattr(terrain, "terrain_types") and terrain.terrain_types is not None:
        terrain.terrain_types[env_ids] = col_indices.to(terrain.terrain_types.dtype)
    if terrain.terrain_origins is not None:
        terrain.env_origins[env_ids] = terrain.terrain_origins[row_indices, col_indices]

    # --- Compute spawn positions (with optional XY/yaw randomization) ---
    # 计算 spawn 位置（含可选 XY/yaw 随机扰动）
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3].clone()
    positions[:, :3] += env.scene.env_origins[env_ids, :3]

    if pose_range is None:
        pose_range = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.5, 0.5)}
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    positions[:, 0] += sample_uniform(x_range[0], x_range[1], (len(env_ids),), device=device)
    positions[:, 1] += sample_uniform(y_range[0], y_range[1], (len(env_ids),), device=device)

    yaw_range = pose_range.get("yaw", (-0.5, 0.5))
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids),), device=device)
    zero = torch.zeros_like(yaw)
    orientations_delta = quat_from_euler_xyz(zero, zero, yaw)
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # Velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=device)
    velocities = root_states[:, 7:13] + rand_samples

    # Write to simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
