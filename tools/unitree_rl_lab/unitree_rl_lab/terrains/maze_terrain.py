# Copyright (c) 2024-2026, Tencent Kaiwu Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Maze terrain generator for Isaac Lab (trimesh version).

Generates a random maze with configurable-height walls, guaranteeing at least one
traversable path from the entry edge to the opposite edge.

Uses randomized DFS (Recursive Backtracker) to carve passages, then converts
the maze grid into trimesh box primitives.

Curriculum:
    passage_width = passage_width_max - (max - min) * difficulty
    (difficulty=0 -> wide passages, difficulty=1 -> narrow passages)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Maze generation helpers
# ---------------------------------------------------------------------------


def _generate_maze_dfs(n_cells_x: int, n_cells_y: int, rng: np.random.Generator):
    """Generate a maze using iterative randomized DFS (Recursive Backtracker).

    Returns:
        h_walls: (n_cells_y-1, n_cells_x) bool array. True = wall between row cy and cy+1.
        v_walls: (n_cells_y, n_cells_x-1) bool array. True = wall between col cx and cx+1.
    """
    visited = np.zeros((n_cells_y, n_cells_x), dtype=bool)
    h_walls = np.ones((n_cells_y - 1, n_cells_x), dtype=bool)
    v_walls = np.ones((n_cells_y, n_cells_x - 1), dtype=bool)

    start_x, start_y = 0, 0
    visited[start_y, start_x] = True
    stack = [(start_x, start_y)]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < n_cells_x and 0 <= ny < n_cells_y and not visited[ny, nx]:
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            idx = rng.integers(len(neighbors))
            nx, ny, dx, dy = neighbors[idx]
            if dx == 1:
                v_walls[cy, cx] = False
            elif dx == -1:
                v_walls[ny, nx] = False
            elif dy == 1:
                h_walls[cy, cx] = False
            elif dy == -1:
                h_walls[ny, nx] = False
            visited[ny, nx] = True
            stack.append((nx, ny))
        else:
            stack.pop()

    return h_walls, v_walls


def _verify_connectivity_grid(
    n_cells_x: int,
    n_cells_y: int,
    h_walls: np.ndarray,
    v_walls: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """BFS verify that *start* cell can reach *goal* cell in the maze grid."""
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            return True
        # right
        if cx + 1 < n_cells_x and not v_walls[cy, cx] and (cx + 1, cy) not in visited:
            visited.add((cx + 1, cy))
            queue.append((cx + 1, cy))
        # left
        if cx - 1 >= 0 and not v_walls[cy, cx - 1] and (cx - 1, cy) not in visited:
            visited.add((cx - 1, cy))
            queue.append((cx - 1, cy))
        # down (cy+1)
        if cy + 1 < n_cells_y and not h_walls[cy, cx] and (cx, cy + 1) not in visited:
            visited.add((cx, cy + 1))
            queue.append((cx, cy + 1))
        # up (cy-1)
        if cy - 1 >= 0 and not h_walls[cy - 1, cx] and (cx, cy - 1) not in visited:
            visited.add((cx, cy - 1))
            queue.append((cx, cy - 1))
    return False


# ---------------------------------------------------------------------------
# Main terrain function
# ---------------------------------------------------------------------------


def maze_terrain(difficulty: float, cfg: "NavMazeTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a maze terrain as trimesh boxes.

    The maze is built on a grid of cells using randomized DFS. Entry is on the
    left edge (X=0) and exit on the right edge (X=size[0]). Connectivity is
    verified via BFS. The robot spawns at the entry opening.

    Args:
        difficulty: Terrain difficulty in [0, 1]. Controls passage width.
        cfg: Maze terrain configuration.

    Returns:
        A tuple containing:
            - meshes_list: list of trimesh objects
            - origin: terrain origin (robot spawn position at entry)
            - exit_info: dict with exit position and yaw
    """
    terrain_x, terrain_y = cfg.size  # (size_x, size_y)

    # --- Passage width with curriculum ---
    passage_width = cfg.passage_width_max - (cfg.passage_width_max - cfg.passage_width_min) * difficulty

    wall_thickness = cfg.wall_thickness
    wall_height = cfg.wall_height
    cell_size = passage_width + wall_thickness

    # Number of cells that fit in the terrain
    n_cells_x = max(2, int((terrain_x - wall_thickness) / cell_size))
    n_cells_y = max(2, int((terrain_y - wall_thickness) / cell_size))

    # Actual cell dimensions in world coordinates
    # total_x = n_cells_x * passage + (n_cells_x + 1) * wall  — re-derive cell_w/cell_h
    # to fill the terrain evenly:
    cell_w_x = (terrain_x - wall_thickness) / n_cells_x  # passage + wall per cell in X
    cell_w_y = (terrain_y - wall_thickness) / n_cells_y
    pw_x = cell_w_x - wall_thickness  # effective passage width in X (may differ slightly)
    pw_y = cell_w_y - wall_thickness

    rng = np.random.default_rng(
        cfg.seed + int(difficulty * 1000) if hasattr(cfg, "seed") and cfg.seed is not None else None
    )

    # --- Generate maze with guaranteed connectivity ---
    entry_cx, entry_cy = 0, 0  # entry: left-bottom cell
    exit_cx, exit_cy = n_cells_x - 1, n_cells_y - 1  # exit: right-top cell

    max_attempts = 50
    h_walls = v_walls = None
    for _ in range(max_attempts):
        h_walls, v_walls = _generate_maze_dfs(n_cells_x, n_cells_y, rng)
        if _verify_connectivity_grid(n_cells_x, n_cells_y, h_walls, v_walls, (entry_cx, entry_cy), (exit_cx, exit_cy)):
            break

    # --- Clear walls around entry cell to create safe spawn area ---
    # The spawn area needs to accommodate ±spawn_clearance random offset.
    # We remove internal walls between the entry cell and its neighbors so
    # that the combined open area is large enough.
    spawn_clearance = getattr(cfg, "spawn_clearance", 0.5)
    # Determine how many cells to clear in each direction
    # safe radius needed: spawn_clearance + robot half-width (~0.15m margin)
    safe_radius = spawn_clearance + 0.15
    cells_to_clear_x = max(0, int(np.ceil(safe_radius / pw_x)))
    cells_to_clear_y = max(0, int(np.ceil(safe_radius / pw_y)))

    for dy in range(-cells_to_clear_y, cells_to_clear_y + 1):
        for dx in range(-cells_to_clear_x, cells_to_clear_x + 1):
            nx, ny = entry_cx + dx, entry_cy + dy
            if not (0 <= nx < n_cells_x and 0 <= ny < n_cells_y):
                continue
            # Remove wall between (nx, ny) and (nx+1, ny) — vertical wall to the right
            if nx + 1 <= entry_cx + cells_to_clear_x and nx < n_cells_x - 1:
                v_walls[ny, nx] = False
            # Remove wall between (nx, ny) and (nx, ny+1) — horizontal wall below
            if ny + 1 <= entry_cy + cells_to_clear_y and ny < n_cells_y - 1:
                h_walls[ny, nx] = False

    # --- Helper: cell (cx, cy) -> world center coordinates ---
    def cell_center(cx: int, cy: int) -> tuple[float, float]:
        """Return (world_x, world_y) center of cell (cx, cy)."""
        x = wall_thickness + cx * cell_w_x + pw_x / 2.0
        y = wall_thickness + cy * cell_w_y + pw_y / 2.0
        return x, y

    # --- Build trimesh walls ---
    meshes_list: list[trimesh.Trimesh] = []

    def _add_box(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float):
        """Create a box at (cx,cy,cz) with dimensions (sx,sy,sz)."""
        box = trimesh.creation.box(
            (sx, sy, sz),
            trimesh.transformations.translation_matrix((cx, cy, cz)),
        )
        meshes_list.append(box)

    half_h = wall_height / 2.0

    # Outer boundary walls (4 sides), with openings for entry and exit
    # --- Left boundary (x = 0) ---
    # Entry opening at cell (0, entry_cy)
    entry_center_y = wall_thickness + entry_cy * cell_w_y + pw_y / 2.0
    entry_y_lo = entry_center_y - pw_y / 2.0
    entry_y_hi = entry_center_y + pw_y / 2.0
    # wall below entry
    if entry_y_lo > 0:
        _add_box(wall_thickness / 2.0, entry_y_lo / 2.0, half_h, wall_thickness, entry_y_lo, wall_height)
    # wall above entry
    if entry_y_hi < terrain_y:
        mid_y = (entry_y_hi + terrain_y) / 2.0
        _add_box(wall_thickness / 2.0, mid_y, half_h, wall_thickness, terrain_y - entry_y_hi, wall_height)

    # --- Right boundary (x = terrain_x) ---
    exit_center_y = wall_thickness + exit_cy * cell_w_y + pw_y / 2.0
    exit_y_lo = exit_center_y - pw_y / 2.0
    exit_y_hi = exit_center_y + pw_y / 2.0
    bx = terrain_x - wall_thickness / 2.0
    if exit_y_lo > 0:
        _add_box(bx, exit_y_lo / 2.0, half_h, wall_thickness, exit_y_lo, wall_height)
    if exit_y_hi < terrain_y:
        mid_y = (exit_y_hi + terrain_y) / 2.0
        _add_box(bx, mid_y, half_h, wall_thickness, terrain_y - exit_y_hi, wall_height)

    # --- Bottom boundary (y = 0) ---
    _add_box(terrain_x / 2.0, wall_thickness / 2.0, half_h, terrain_x, wall_thickness, wall_height)
    # --- Top boundary (y = terrain_y) ---
    _add_box(terrain_x / 2.0, terrain_y - wall_thickness / 2.0, half_h, terrain_x, wall_thickness, wall_height)

    # Internal vertical walls (between col cx and cx+1)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x - 1):
            if v_walls[cy, cx]:
                wx = wall_thickness + (cx + 1) * cell_w_x - wall_thickness / 2.0
                wy = wall_thickness + cy * cell_w_y + pw_y / 2.0
                _add_box(wx, wy, half_h, wall_thickness, pw_y, wall_height)

    # Internal horizontal walls (between row cy and cy+1)
    for cy in range(n_cells_y - 1):
        for cx in range(n_cells_x):
            if h_walls[cy, cx]:
                wx = wall_thickness + cx * cell_w_x + pw_x / 2.0
                wy = wall_thickness + (cy + 1) * cell_w_y - wall_thickness / 2.0
                _add_box(wx, wy, half_h, pw_x, wall_thickness, wall_height)

    # Wall posts at grid intersections (fill gaps between wall segments)
    # Each intersection at grid index (ix, iy) sits at the top-right corner of cell (ix-1, iy-1).
    # Posts inside the spawn clearance zone must be removed to keep the area open.
    for iy in range(n_cells_y + 1):
        for ix in range(n_cells_x + 1):
            px = ix * cell_w_x + wall_thickness / 2.0
            py = iy * cell_w_y + wall_thickness / 2.0
            # Skip posts on entry/exit openings at the boundary
            is_entry_post = (ix == 0 and iy == entry_cy) or (ix == 0 and iy == entry_cy + 1)
            is_exit_post = (ix == n_cells_x and iy == exit_cy) or (ix == n_cells_x and iy == exit_cy + 1)
            if is_entry_post or is_exit_post:
                continue
            # Skip interior posts inside the spawn clearance zone.
            # A post at (ix, iy) is an interior post when it has cells on all 4 sides.
            # It should be removed if ALL 4 surrounding cells are within the cleared zone.
            if 1 <= ix <= min(entry_cx + cells_to_clear_x, n_cells_x - 1) and 1 <= iy <= min(
                entry_cy + cells_to_clear_y, n_cells_y - 1
            ):
                continue
            _add_box(px, py, half_h, wall_thickness, wall_thickness, wall_height)

    # Ground plane
    ground = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground)

    # --- Origin: at center of the cleared spawn area ---
    # Place the origin at the geometric center of the cleared cell block,
    # so that the distance to the nearest wall is maximized in all directions.
    # Cleared cells span from (entry_cx, entry_cy) to
    # (entry_cx + cells_to_clear_x, entry_cy + cells_to_clear_y).
    # The passable area X-range: [wall_thickness, wall_thickness + (cells_to_clear_x+1)*cell_w_x - wall_thickness]
    clear_x_min = wall_thickness
    clear_x_max = wall_thickness + (min(cells_to_clear_x, n_cells_x - 1) + 1) * cell_w_x - wall_thickness
    clear_y_min = wall_thickness
    clear_y_max = wall_thickness + (min(cells_to_clear_y, n_cells_y - 1) + 1) * cell_w_y - wall_thickness
    spawn_cx = (clear_x_min + clear_x_max) / 2.0
    spawn_cy = (clear_y_min + clear_y_max) / 2.0
    origin = np.array([spawn_cx, spawn_cy, 0.0])

    # --- Compute all cell centers as valid spawn positions ---
    spawn_positions = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            sx = wall_thickness + cx * cell_w_x + pw_x / 2.0
            sy = wall_thickness + cy * cell_w_y + pw_y / 2.0
            spawn_positions.append((sx, sy, 0.0))

    # --- Exit info: at exit opening (right edge) ---
    exit_world_x = terrain_x - wall_thickness / 2.0
    exit_world_y = exit_center_y
    exit_info = {
        "position": (exit_world_x, exit_world_y, 0.0),
        "yaw": 0.0,  # facing +X direction (out of the maze)
        "spawn_positions": spawn_positions,
    }

    return meshes_list, origin, exit_info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class NavMazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a maze terrain (trimesh version).

    The maze consists of walls of fixed height with passages carved using
    randomized DFS. At least one path from entry to exit is guaranteed and
    verified via BFS.

    Curriculum controls passage width:
        passage_width = passage_width_max - (max - min) * difficulty
        difficulty=0 -> passage_width_max (easy, wide passages)
        difficulty=1 -> passage_width_min (hard, narrow passages)
    """

    function = maze_terrain

    # Wall height in meters
    wall_height: float = 0.5

    # Wall thickness in meters
    wall_thickness: float = 0.1

    # Passage width range (curriculum-controlled)
    passage_width_max: float = 1.0
    passage_width_min: float = 0.6

    # Safe spawn clearance radius (meters).
    # Internal walls within this radius of the entry cell center are removed
    # to prevent robots from spawning inside walls. Should be >= the random
    # pose offset used in reset_root_state_uniform (default ±0.5m).
    spawn_clearance: float = 0.5

    # Seed for reproducibility (set by terrain generator)
    seed: int | None = None
