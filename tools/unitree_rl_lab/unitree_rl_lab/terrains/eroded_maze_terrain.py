# Copyright (c) 2024-2026, Tencent Kaiwu Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Eroded-corner maze terrain generator for Isaac Lab (trimesh version).

Generates a random maze identical to ``maze_terrain`` but with an additional
**corner-erosion** post-processing step:

For every wall-post (corner pillar) — including both **interior** posts and
**boundary** posts on the outer walls — where walls from two different
directions meet (L-turn, T-junction, or cross), the post-processor:
  1. Removes the corner post itself.
  2. Removes the near-half of each **internal** wall segment connected to
     that post.  Outer boundary walls are never shortened, so their
     structural integrity is fully preserved.

The far-half of each internal wall segment is kept, so walls become shorter
stubs that no longer touch at corners.  This turns every sharp 90-degree
corner — even those formed where an internal wall meets the outer boundary —
into a wide opening, making the maze much friendlier for legged-robot
navigation while preserving the overall maze topology.

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
# Maze generation helpers (same as maze_terrain.py)
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
        if cx + 1 < n_cells_x and not v_walls[cy, cx] and (cx + 1, cy) not in visited:
            visited.add((cx + 1, cy))
            queue.append((cx + 1, cy))
        if cx - 1 >= 0 and not v_walls[cy, cx - 1] and (cx - 1, cy) not in visited:
            visited.add((cx - 1, cy))
            queue.append((cx - 1, cy))
        if cy + 1 < n_cells_y and not h_walls[cy, cx] and (cx, cy + 1) not in visited:
            visited.add((cx, cy + 1))
            queue.append((cx, cy + 1))
        if cy - 1 >= 0 and not h_walls[cy - 1, cx] and (cx, cy - 1) not in visited:
            visited.add((cx, cy - 1))
            queue.append((cx, cy - 1))
    return False


# ---------------------------------------------------------------------------
# Corner erosion logic
# ---------------------------------------------------------------------------


def _compute_corner_erosion(
    n_cells_x: int,
    n_cells_y: int,
    h_walls: np.ndarray,
    v_walls: np.ndarray,
):
    """Detect corners and compute erosion masks.

    A **corner** is a wall-post where walls from two or more perpendicular
    directions meet. This includes both interior posts and **boundary posts**
    where an internal wall segment meets the outer boundary wall.

    At each such post we:
      - Mark the post for removal.
      - Mark the near-half of every **internal** wall segment touching the
        post for removal.  Outer boundary walls are never eroded — their
        integrity is preserved.
      - If the corner is a **boundary corner** (involves an outer boundary
        wall), additionally mark the connected internal wall segments for
        extra erosion so that the gap between stub and boundary is wider.

    Wall-post grid indices ``(ix, iy)`` range from ``(0, 0)`` to
    ``(n_cells_x, n_cells_y)``.  Post ``(ix, iy)`` sits at the junction of:

        - h_wall to the LEFT  = h_walls[iy-1, ix-1]
        - h_wall to the RIGHT = h_walls[iy-1, ix]
        - v_wall ABOVE        = v_walls[iy-1, ix-1]
        - v_wall BELOW        = v_walls[iy, ix-1]

    Boundary posts additionally "see" the outer boundary wall running along
    the edge they lie on (horizontal boundary for iy==0 / iy==n_cells_y,
    vertical boundary for ix==0 / ix==n_cells_x).

    Returns:
        post_eroded: (n_cells_y+1, n_cells_x+1) bool — True = remove this post.
        h_erode_left:  (n_cells_y-1, n_cells_x) bool — True = erase left-half of h_wall[cy,cx].
        h_erode_right: (n_cells_y-1, n_cells_x) bool — True = erase right-half.
        v_erode_top:   (n_cells_y, n_cells_x-1) bool — True = erase top-half of v_wall[cy,cx].
        v_erode_bottom:(n_cells_y, n_cells_x-1) bool — True = erase bottom-half.
        h_boundary_left:  (n_cells_y-1, n_cells_x) bool — True = left-end touches boundary corner.
        h_boundary_right: (n_cells_y-1, n_cells_x) bool — True = right-end touches boundary corner.
        v_boundary_top:   (n_cells_y, n_cells_x-1) bool — True = top-end touches boundary corner.
        v_boundary_bottom:(n_cells_y, n_cells_x-1) bool — True = bottom-end touches boundary corner.
    """
    post_eroded = np.zeros((n_cells_y + 1, n_cells_x + 1), dtype=bool)

    h_erode_left = np.zeros_like(h_walls, dtype=bool)
    h_erode_right = np.zeros_like(h_walls, dtype=bool)
    v_erode_top = np.zeros_like(v_walls, dtype=bool)
    v_erode_bottom = np.zeros_like(v_walls, dtype=bool)

    h_boundary_left = np.zeros_like(h_walls, dtype=bool)
    h_boundary_right = np.zeros_like(h_walls, dtype=bool)
    v_boundary_top = np.zeros_like(v_walls, dtype=bool)
    v_boundary_bottom = np.zeros_like(v_walls, dtype=bool)

    for iy in range(n_cells_y + 1):
        for ix in range(n_cells_x + 1):
            # --- Internal wall segments connected to this post ---
            has_h_left = 0 <= iy - 1 < h_walls.shape[0] and 0 <= ix - 1 < h_walls.shape[1] and h_walls[iy - 1, ix - 1]
            has_h_right = 0 <= iy - 1 < h_walls.shape[0] and 0 <= ix < h_walls.shape[1] and h_walls[iy - 1, ix]
            has_v_above = 0 <= iy - 1 < v_walls.shape[0] and 0 <= ix - 1 < v_walls.shape[1] and v_walls[iy - 1, ix - 1]
            has_v_below = 0 <= iy < v_walls.shape[0] and 0 <= ix - 1 < v_walls.shape[1] and v_walls[iy, ix - 1]

            # --- Outer boundary walls touching this post ---
            # A post on the left/right boundary "sees" a vertical boundary wall.
            # A post on the top/bottom boundary "sees" a horizontal boundary wall.
            on_left_boundary = ix == 0
            on_right_boundary = ix == n_cells_x
            on_bottom_boundary = iy == 0
            on_top_boundary = iy == n_cells_y

            has_boundary_h = on_bottom_boundary or on_top_boundary
            has_boundary_v = on_left_boundary or on_right_boundary

            # Total horizontal / vertical presence (internal + boundary)
            has_any_h = has_h_left or has_h_right or has_boundary_h
            has_any_v = has_v_above or has_v_below or has_boundary_v

            # A corner exists when walls from BOTH directions meet
            if has_any_h and has_any_v:
                post_eroded[iy, ix] = True

                # Is this a boundary corner?
                is_boundary_corner = has_boundary_h or has_boundary_v

                # Erode near-half of each connected *internal* wall segment.
                # Boundary walls are drawn separately and are NOT eroded.
                if has_h_left:
                    h_erode_right[iy - 1, ix - 1] = True
                    if is_boundary_corner:
                        h_boundary_right[iy - 1, ix - 1] = True
                if has_h_right:
                    h_erode_left[iy - 1, ix] = True
                    if is_boundary_corner:
                        h_boundary_left[iy - 1, ix] = True
                if has_v_above:
                    v_erode_bottom[iy - 1, ix - 1] = True
                    if is_boundary_corner:
                        v_boundary_bottom[iy - 1, ix - 1] = True
                if has_v_below:
                    v_erode_top[iy, ix - 1] = True
                    if is_boundary_corner:
                        v_boundary_top[iy, ix - 1] = True

    return (
        post_eroded,
        h_erode_left,
        h_erode_right,
        v_erode_top,
        v_erode_bottom,
        h_boundary_left,
        h_boundary_right,
        v_boundary_top,
        v_boundary_bottom,
    )


# ---------------------------------------------------------------------------
# Main terrain function
# ---------------------------------------------------------------------------


def eroded_maze_terrain(
    difficulty: float, cfg: "ErodedMazeTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a corner-eroded maze terrain as trimesh boxes.

    Identical to ``maze_terrain`` but with a post-processing step that removes
    corner posts and the near-half of wall segments at every L/T/cross junction.

    Args:
        difficulty: Terrain difficulty in [0, 1]. Controls passage width.
        cfg: Eroded maze terrain configuration.

    Returns:
        A tuple containing:
            - meshes_list: list of trimesh objects
            - origin: terrain origin (robot spawn position at entry)
            - exit_info: dict with exit position and yaw
    """
    terrain_x, terrain_y = cfg.size

    # --- Passage width with curriculum ---
    passage_width = cfg.passage_width_max - (cfg.passage_width_max - cfg.passage_width_min) * difficulty

    wall_thickness = cfg.wall_thickness
    wall_height = cfg.wall_height
    cell_size = passage_width + wall_thickness

    n_cells_x = max(2, int((terrain_x - wall_thickness) / cell_size))
    n_cells_y = max(2, int((terrain_y - wall_thickness) / cell_size))

    cell_w_x = (terrain_x - wall_thickness) / n_cells_x
    cell_w_y = (terrain_y - wall_thickness) / n_cells_y
    pw_x = cell_w_x - wall_thickness
    pw_y = cell_w_y - wall_thickness

    rng = np.random.default_rng(
        cfg.seed + int(difficulty * 1000) if hasattr(cfg, "seed") and cfg.seed is not None else None
    )

    # --- Generate maze with guaranteed connectivity ---
    entry_cx, entry_cy = 0, 0
    exit_cx, exit_cy = n_cells_x - 1, n_cells_y - 1

    max_attempts = 50
    h_walls = v_walls = None
    for _ in range(max_attempts):
        h_walls, v_walls = _generate_maze_dfs(n_cells_x, n_cells_y, rng)
        if _verify_connectivity_grid(n_cells_x, n_cells_y, h_walls, v_walls, (entry_cx, entry_cy), (exit_cx, exit_cy)):
            break

    # --- Clear walls around entry cell (spawn area) ---
    spawn_clearance = getattr(cfg, "spawn_clearance", 0.5)
    safe_radius = spawn_clearance + 0.15
    cells_to_clear_x = max(0, int(np.ceil(safe_radius / pw_x)))
    cells_to_clear_y = max(0, int(np.ceil(safe_radius / pw_y)))

    for dy in range(-cells_to_clear_y, cells_to_clear_y + 1):
        for dx in range(-cells_to_clear_x, cells_to_clear_x + 1):
            nx, ny = entry_cx + dx, entry_cy + dy
            if not (0 <= nx < n_cells_x and 0 <= ny < n_cells_y):
                continue
            if nx + 1 <= entry_cx + cells_to_clear_x and nx < n_cells_x - 1:
                v_walls[ny, nx] = False
            if ny + 1 <= entry_cy + cells_to_clear_y and ny < n_cells_y - 1:
                h_walls[ny, nx] = False

    # ===================================================================
    # Corner erosion post-processing
    # ===================================================================
    (
        post_eroded,
        h_erode_left,
        h_erode_right,
        v_erode_top,
        v_erode_bottom,
        h_boundary_left,
        h_boundary_right,
        v_boundary_top,
        v_boundary_bottom,
    ) = _compute_corner_erosion(n_cells_x, n_cells_y, h_walls, v_walls)

    # Boundary erosion fraction: at boundary corners, erode more of the
    # internal wall stub so the gap to the outer wall is wider.
    # Normal erosion keeps pw * 0.5; boundary erosion keeps pw * (1 - bef).
    bef = getattr(cfg, "boundary_erosion_fraction", 0.8)

    # --- Build trimesh walls ---
    meshes_list: list[trimesh.Trimesh] = []

    def _add_box(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float):
        if sx < 1e-6 or sy < 1e-6 or sz < 1e-6:
            return
        box = trimesh.creation.box(
            (sx, sy, sz),
            trimesh.transformations.translation_matrix((cx, cy, cz)),
        )
        meshes_list.append(box)

    half_h = wall_height / 2.0

    # === Outer boundary walls (unchanged — no erosion on boundaries) ===

    # Left boundary (x = 0) with entry opening
    entry_center_y = wall_thickness + entry_cy * cell_w_y + pw_y / 2.0
    entry_y_lo = entry_center_y - pw_y / 2.0
    entry_y_hi = entry_center_y + pw_y / 2.0
    if entry_y_lo > 0:
        _add_box(wall_thickness / 2.0, entry_y_lo / 2.0, half_h, wall_thickness, entry_y_lo, wall_height)
    if entry_y_hi < terrain_y:
        mid_y = (entry_y_hi + terrain_y) / 2.0
        _add_box(wall_thickness / 2.0, mid_y, half_h, wall_thickness, terrain_y - entry_y_hi, wall_height)

    # Right boundary (x = terrain_x) with exit opening
    exit_center_y = wall_thickness + exit_cy * cell_w_y + pw_y / 2.0
    exit_y_lo = exit_center_y - pw_y / 2.0
    exit_y_hi = exit_center_y + pw_y / 2.0
    bx = terrain_x - wall_thickness / 2.0
    if exit_y_lo > 0:
        _add_box(bx, exit_y_lo / 2.0, half_h, wall_thickness, exit_y_lo, wall_height)
    if exit_y_hi < terrain_y:
        mid_y = (exit_y_hi + terrain_y) / 2.0
        _add_box(bx, mid_y, half_h, wall_thickness, terrain_y - exit_y_hi, wall_height)

    # Bottom boundary (y = 0)
    _add_box(terrain_x / 2.0, wall_thickness / 2.0, half_h, terrain_x, wall_thickness, wall_height)
    # Top boundary (y = terrain_y)
    _add_box(terrain_x / 2.0, terrain_y - wall_thickness / 2.0, half_h, terrain_x, wall_thickness, wall_height)

    # === Internal horizontal walls (between row cy and cy+1) ===
    # h_walls[cy, cx] is a horizontal wall segment spanning the width of cell cx.
    # Full segment: center_x = wall_thickness + cx * cell_w_x + pw_x/2
    #               center_y = wall_thickness + (cy+1) * cell_w_y - wall_thickness/2
    #               size = (pw_x, wall_thickness)
    #
    # Erosion:
    #   erode_left  → remove the left half  (near-post side = left end)
    #   erode_right → remove the right half (near-post side = right end)
    #   both eroded → wall disappears entirely
    #
    # Boundary erosion:
    #   If the eroded end touches a boundary corner, use a larger erosion
    #   fraction (bef) instead of the default 0.5, so the remaining stub
    #   is shorter and leaves a wider gap to the outer wall.
    for cy in range(n_cells_y - 1):
        for cx in range(n_cells_x):
            if not h_walls[cy, cx]:
                continue

            el = h_erode_left[cy, cx]
            er = h_erode_right[cy, cx]

            # Full wall anchor (before erosion)
            full_x = wall_thickness + cx * cell_w_x + pw_x / 2.0
            wy = wall_thickness + (cy + 1) * cell_w_y - wall_thickness / 2.0

            if el and er:
                # Both halves eroded → wall gone
                continue
            elif el and not er:
                # Left half removed → keep right portion
                frac_l = bef if h_boundary_left[cy, cx] else 0.5
                seg_len = pw_x * (1.0 - frac_l)
                seg_cx = full_x + pw_x / 2.0 - seg_len / 2.0
                _add_box(seg_cx, wy, half_h, seg_len, wall_thickness, wall_height)
            elif er and not el:
                # Right half removed → keep left portion
                frac_r = bef if h_boundary_right[cy, cx] else 0.5
                seg_len = pw_x * (1.0 - frac_r)
                seg_cx = full_x - pw_x / 2.0 + seg_len / 2.0
                _add_box(seg_cx, wy, half_h, seg_len, wall_thickness, wall_height)
            else:
                # No erosion → full wall
                _add_box(full_x, wy, half_h, pw_x, wall_thickness, wall_height)

    # === Internal vertical walls (between col cx and cx+1) ===
    # v_walls[cy, cx] is a vertical wall segment spanning the height of cell cy.
    # Full segment: center_x = wall_thickness + (cx+1) * cell_w_x - wall_thickness/2
    #               center_y = wall_thickness + cy * cell_w_y + pw_y/2
    #               size = (wall_thickness, pw_y)
    #
    # Erosion:
    #   erode_top    → remove the top half    (near-post side = top end)
    #   erode_bottom → remove the bottom half (near-post side = bottom end)
    #
    # Boundary erosion: same logic as horizontal walls.
    for cy in range(n_cells_y):
        for cx in range(n_cells_x - 1):
            if not v_walls[cy, cx]:
                continue

            et = v_erode_top[cy, cx]
            eb = v_erode_bottom[cy, cx]

            wx = wall_thickness + (cx + 1) * cell_w_x - wall_thickness / 2.0
            full_y = wall_thickness + cy * cell_w_y + pw_y / 2.0

            if et and eb:
                # Both halves eroded → wall gone
                continue
            elif et and not eb:
                # Top half removed → keep bottom portion
                frac_t = bef if v_boundary_top[cy, cx] else 0.5
                seg_len = pw_y * (1.0 - frac_t)
                seg_cy = full_y + pw_y / 2.0 - seg_len / 2.0
                _add_box(wx, seg_cy, half_h, wall_thickness, seg_len, wall_height)
            elif eb and not et:
                # Bottom half removed → keep top portion
                frac_b = bef if v_boundary_bottom[cy, cx] else 0.5
                seg_len = pw_y * (1.0 - frac_b)
                seg_cy = full_y - pw_y / 2.0 + seg_len / 2.0
                _add_box(wx, seg_cy, half_h, wall_thickness, seg_len, wall_height)
            else:
                # No erosion → full wall
                _add_box(wx, full_y, half_h, wall_thickness, pw_y, wall_height)

    # === Wall posts ===
    for iy in range(n_cells_y + 1):
        for ix in range(n_cells_x + 1):
            # Skip eroded corner posts
            if post_eroded[iy, ix]:
                continue

            px = ix * cell_w_x + wall_thickness / 2.0
            py = iy * cell_w_y + wall_thickness / 2.0

            # Skip entry/exit boundary posts
            is_entry_post = (ix == 0 and iy == entry_cy) or (ix == 0 and iy == entry_cy + 1)
            is_exit_post = (ix == n_cells_x and iy == exit_cy) or (ix == n_cells_x and iy == exit_cy + 1)
            if is_entry_post or is_exit_post:
                continue

            # Skip spawn clearance zone posts
            if 1 <= ix <= min(entry_cx + cells_to_clear_x, n_cells_x - 1) and 1 <= iy <= min(
                entry_cy + cells_to_clear_y, n_cells_y - 1
            ):
                continue

            _add_box(px, py, half_h, wall_thickness, wall_thickness, wall_height)

    # Ground plane
    ground = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground)

    # --- Origin (spawn position) ---
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

    # --- Exit info ---
    exit_world_x = terrain_x - wall_thickness / 2.0
    exit_world_y = exit_center_y
    exit_info = {
        "position": (exit_world_x, exit_world_y, 0.0),
        "yaw": 0.0,
        "spawn_positions": spawn_positions,
    }

    return meshes_list, origin, exit_info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class ErodedMazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a corner-eroded maze terrain (trimesh version).

    Identical to ``NavMazeTerrainCfg`` but with corner-erosion post-processing:
    at every L/T/cross wall junction, the corner post is removed and each
    connected wall segment is shortened by half (the half nearest the post is
    deleted). This eliminates sharp 90-degree corners while preserving the
    maze topology.

    For corners where an internal wall meets the **outer boundary wall**, an
    additional ``boundary_erosion_fraction`` controls how much of the internal
    wall stub is removed. The default (0.75) removes 75% of the wall segment
    at boundary corners (vs. 50% at interior corners), leaving a wider gap so
    legged robots can turn without getting stuck.

    Curriculum controls passage width:
        passage_width = passage_width_max - (max - min) * difficulty
        difficulty=0 -> passage_width_max (easy, wide passages)
        difficulty=1 -> passage_width_min (hard, narrow passages)
    """

    function = eroded_maze_terrain

    # Wall height in meters
    wall_height: float = 0.5

    # Wall thickness in meters
    wall_thickness: float = 0.1

    # Passage width range (curriculum-controlled)
    passage_width_max: float = 1.0
    passage_width_min: float = 0.6

    # Erosion fraction at boundary corners (where internal wall meets outer
    # boundary). 0.5 = same as interior corners (half removed), 1.0 = entire
    # wall segment removed. Default 0.75 removes 75%, keeping only 25% stub.
    boundary_erosion_fraction: float = 0.80

    # Safe spawn clearance radius (meters)
    spawn_clearance: float = 0.5

    # Seed for reproducibility
    seed: int | None = None
