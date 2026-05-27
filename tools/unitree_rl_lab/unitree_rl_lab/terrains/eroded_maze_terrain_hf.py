# Copyright (c) 2024-2026, Tencent Kaiwu Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Eroded-corner maze terrain generator for Isaac Lab (height-field version).

Generates a random maze identical to ``maze_terrain_unitree`` but with an
additional **corner-erosion** post-processing step on the height-field:

For every interior wall-post (corner pillar) where walls from two different
directions meet (L-turn, T-junction, or cross), the post-processor:
  1. Clears the corner post pixels to ground height (0).
  2. Clears the near-half pixels of each wall segment connected to that post.

This eliminates sharp 90-degree corners while preserving the overall maze
topology, making the maze friendlier for legged-robot navigation.

Curriculum:
    passage_width = passage_width_max - (max - min) * difficulty
    (difficulty=0 -> wide passages, difficulty=1 -> narrow passages)
"""

from __future__ import annotations

import numpy as np

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Corner erosion on height-field
# ---------------------------------------------------------------------------


def _erode_corners_hf(
    hf: np.ndarray,
    n_cells_x: int,
    n_cells_y: int,
    h_walls: np.ndarray,
    v_walls: np.ndarray,
    wall_thickness_px: int,
    passage_width_px: int,
    cell_plus_wall: int,
    width_pixels: int,
    length_pixels: int,
):
    """Erode corners in-place on the height-field array.

    For every wall-post where perpendicular walls meet:
      1. Clear the post pixels (set to 0).
      2. Clear the near-half of each connected wall segment.

    Wall-post ``(ix, iy)`` pixel region:
        x: [ix * cell_plus_wall, ix * cell_plus_wall + wall_thickness_px)
        y: [iy * cell_plus_wall, iy * cell_plus_wall + wall_thickness_px)

    h_wall ``(cy, cx)`` pixel region (wall between row cy and cy+1):
        x: [wall_thickness_px + cx * cell_plus_wall,
            wall_thickness_px + cx * cell_plus_wall + passage_width_px)
        y: [wall_thickness_px + cy * cell_plus_wall + passage_width_px,
            wall_thickness_px + cy * cell_plus_wall + passage_width_px + wall_thickness_px)

    v_wall ``(cy, cx)`` pixel region (wall between col cx and cx+1):
        x: [wall_thickness_px + cx * cell_plus_wall + passage_width_px,
            wall_thickness_px + cx * cell_plus_wall + passage_width_px + wall_thickness_px)
        y: [wall_thickness_px + cy * cell_plus_wall,
            wall_thickness_px + cy * cell_plus_wall + passage_width_px)
    """

    def clear_rect(x_start, y_start, x_end, y_end):
        xs = max(0, min(x_start, width_pixels))
        xe = max(0, min(x_end, width_pixels))
        ys = max(0, min(y_start, length_pixels))
        ye = max(0, min(y_end, length_pixels))
        hf[xs:xe, ys:ye] = 0

    # Iterate over all wall-post positions (ix, iy)
    # Post (ix, iy) sits at the top-right corner of cell (ix-1, iy-1).
    # Its pixel region starts at (ix * cell_plus_wall, iy * cell_plus_wall).
    for iy in range(n_cells_y + 1):
        for ix in range(n_cells_x + 1):
            # --- Check which wall segments connect to this post ---
            #
            # h_wall to the LEFT of post (ix, iy):
            #   This is h_walls[iy-1, ix-1] — the horizontal wall between
            #   row (iy-1) and row iy, at column (ix-1).
            #   It sits to the LEFT of the post along X.
            has_h_left = 0 <= iy - 1 < n_cells_y - 1 and 0 <= ix - 1 < n_cells_x and h_walls[iy - 1, ix - 1]

            # h_wall to the RIGHT of post (ix, iy):
            #   This is h_walls[iy-1, ix] — same wall row, next column.
            has_h_right = 0 <= iy - 1 < n_cells_y - 1 and 0 <= ix < n_cells_x and h_walls[iy - 1, ix]

            # v_wall ABOVE the post (ix, iy):
            #   This is v_walls[iy-1, ix-1] — the vertical wall between
            #   col (ix-1) and col ix, at row (iy-1).
            has_v_above = 0 <= iy - 1 < n_cells_y and 0 <= ix - 1 < n_cells_x - 1 and v_walls[iy - 1, ix - 1]

            # v_wall BELOW the post (ix, iy):
            #   This is v_walls[iy, ix-1] — same wall column, next row.
            has_v_below = 0 <= iy < n_cells_y and 0 <= ix - 1 < n_cells_x - 1 and v_walls[iy, ix - 1]

            has_any_h = has_h_left or has_h_right
            has_any_v = has_v_above or has_v_below

            if not (has_any_h and has_any_v):
                # No corner — skip
                continue

            # --- 1. Clear the corner post pixels ---
            post_px = ix * cell_plus_wall
            post_py = iy * cell_plus_wall
            clear_rect(post_px, post_py, post_px + wall_thickness_px, post_py + wall_thickness_px)

            # --- 2. Clear near-half of each connected wall segment ---
            half_pw_x = passage_width_px // 2
            half_pw_y = passage_width_px // 2

            if has_h_left:
                # h_walls[iy-1, ix-1]: horizontal wall to the LEFT.
                # Full x-range: [wt + (ix-1)*cpw, wt + (ix-1)*cpw + pw)
                # Full y-range: [wt + (iy-1)*cpw + pw, wt + (iy-1)*cpw + pw + wt)
                # Near-half = RIGHT half (closest to post along X).
                hw_x_start = wall_thickness_px + (ix - 1) * cell_plus_wall
                hw_y_start = wall_thickness_px + (iy - 1) * cell_plus_wall + passage_width_px
                # Right half: from midpoint to end
                mid_x = hw_x_start + half_pw_x
                clear_rect(mid_x, hw_y_start, hw_x_start + passage_width_px, hw_y_start + wall_thickness_px)

            if has_h_right:
                # h_walls[iy-1, ix]: horizontal wall to the RIGHT.
                # Full x-range: [wt + ix*cpw, wt + ix*cpw + pw)
                # Near-half = LEFT half (closest to post along X).
                hw_x_start = wall_thickness_px + ix * cell_plus_wall
                hw_y_start = wall_thickness_px + (iy - 1) * cell_plus_wall + passage_width_px
                # Left half: from start to midpoint
                mid_x = hw_x_start + half_pw_x
                clear_rect(hw_x_start, hw_y_start, mid_x, hw_y_start + wall_thickness_px)

            if has_v_above:
                # v_walls[iy-1, ix-1]: vertical wall ABOVE.
                # Full x-range: [wt + (ix-1)*cpw + pw, wt + (ix-1)*cpw + pw + wt)
                # Full y-range: [wt + (iy-1)*cpw, wt + (iy-1)*cpw + pw)
                # Near-half = BOTTOM half (closest to post along Y).
                vw_x_start = wall_thickness_px + (ix - 1) * cell_plus_wall + passage_width_px
                vw_y_start = wall_thickness_px + (iy - 1) * cell_plus_wall
                # Bottom half: from midpoint to end
                mid_y = vw_y_start + half_pw_y
                clear_rect(vw_x_start, mid_y, vw_x_start + wall_thickness_px, vw_y_start + passage_width_px)

            if has_v_below:
                # v_walls[iy, ix-1]: vertical wall BELOW.
                # Full x-range: [wt + (ix-1)*cpw + pw, wt + (ix-1)*cpw + pw + wt)
                # Full y-range: [wt + iy*cpw, wt + iy*cpw + pw)
                # Near-half = TOP half (closest to post along Y).
                vw_x_start = wall_thickness_px + (ix - 1) * cell_plus_wall + passage_width_px
                vw_y_start = wall_thickness_px + iy * cell_plus_wall
                # Top half: from start to midpoint
                mid_y = vw_y_start + half_pw_y
                clear_rect(vw_x_start, vw_y_start, vw_x_start + wall_thickness_px, mid_y)


# ---------------------------------------------------------------------------
# Main terrain function
# ---------------------------------------------------------------------------


@height_field_to_mesh
def eroded_maze_terrain(difficulty: float, cfg: "HfErodedMazeTerrainCfg") -> np.ndarray:
    """Generate a corner-eroded maze terrain as a height-field.

    Identical to ``maze_terrain`` (height-field version) but with a
    post-processing step that clears corner posts and the near-half of
    wall segments at every L/T/cross junction.

    Args:
        difficulty: Terrain difficulty in [0, 1]. Controls passage width.
        cfg: Eroded maze terrain configuration.

    Returns:
        A 2D numpy array (int16) representing the height-field.
    """
    # --- Resolve dimensions in pixel units ---
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    wall_height = int(cfg.wall_height / cfg.vertical_scale)

    passage_width_m = cfg.passage_width_max - (cfg.passage_width_max - cfg.passage_width_min) * difficulty
    passage_width_px = max(int(passage_width_m / cfg.horizontal_scale), 2)

    wall_thickness_px = max(int(cfg.wall_thickness / cfg.horizontal_scale), 1)

    cell_plus_wall = passage_width_px + wall_thickness_px
    n_cells_x = max(int((width_pixels - wall_thickness_px) / cell_plus_wall), 2)
    n_cells_y = max(int((length_pixels - wall_thickness_px) / cell_plus_wall), 2)

    # --- Generate maze using Randomized DFS (iterative) ---
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") and cfg.seed is not None else None)

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

    # --- Render maze to height-field ---
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    def fill_rect(x_start, y_start, x_end, y_end, value):
        x_s = max(0, min(x_start, width_pixels))
        x_e = max(0, min(x_end, width_pixels))
        y_s = max(0, min(y_start, length_pixels))
        y_e = max(0, min(y_end, length_pixels))
        hf_raw[x_s:x_e, y_s:y_e] = value

    # Fill entire field with walls first, then carve passages
    hf_raw[:, :] = wall_height

    # Carve each cell as a passage (height = 0)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            x_start = wall_thickness_px + cx * cell_plus_wall
            y_start = wall_thickness_px + cy * cell_plus_wall
            x_end = x_start + passage_width_px
            y_end = y_start + passage_width_px
            fill_rect(x_start, y_start, x_end, y_end, 0)

    # Carve horizontal passages (remove h_walls)
    for cy in range(n_cells_y - 1):
        for cx in range(n_cells_x):
            if not h_walls[cy, cx]:
                x_start = wall_thickness_px + cx * cell_plus_wall
                y_start = wall_thickness_px + cy * cell_plus_wall + passage_width_px
                x_end = x_start + passage_width_px
                y_end = y_start + wall_thickness_px
                fill_rect(x_start, y_start, x_end, y_end, 0)

    # Carve vertical passages (remove v_walls)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x - 1):
            if not v_walls[cy, cx]:
                x_start = wall_thickness_px + cx * cell_plus_wall + passage_width_px
                y_start = wall_thickness_px + cy * cell_plus_wall
                x_end = x_start + wall_thickness_px
                y_end = y_start + passage_width_px
                fill_rect(x_start, y_start, x_end, y_end, 0)

    # ===================================================================
    # Corner erosion post-processing
    # ===================================================================
    _erode_corners_hf(
        hf_raw,
        n_cells_x,
        n_cells_y,
        h_walls,
        v_walls,
        wall_thickness_px,
        passage_width_px,
        cell_plus_wall,
        width_pixels,
        length_pixels,
    )

    # --- Open entry and exit edges ---
    entry_y = wall_thickness_px + 0 * cell_plus_wall
    fill_rect(0, entry_y, wall_thickness_px, entry_y + passage_width_px, 0)

    exit_cx = n_cells_x - 1
    exit_y = wall_thickness_px + (n_cells_y - 1) * cell_plus_wall
    exit_x_start = wall_thickness_px + exit_cx * cell_plus_wall + passage_width_px
    fill_rect(exit_x_start, exit_y, width_pixels, exit_y + passage_width_px, 0)

    # --- Central platform (safe spawn area) ---
    platform_half = int(cfg.platform_width / (2.0 * cfg.horizontal_scale))
    cx_center = width_pixels // 2
    cy_center = length_pixels // 2
    x1 = max(0, cx_center - platform_half)
    x2 = min(width_pixels, cx_center + platform_half)
    y1 = max(0, cy_center - platform_half)
    y2 = min(length_pixels, cy_center + platform_half)
    hf_raw[x1:x2, y1:y2] = 0

    # --- Compute all cell centers as valid spawn positions (in meters) ---
    spawn_positions = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            sx = (wall_thickness_px + cx * cell_plus_wall + passage_width_px * 0.5) * cfg.horizontal_scale
            sy = (wall_thickness_px + cy * cell_plus_wall + passage_width_px * 0.5) * cfg.horizontal_scale
            spawn_positions.append((sx, sy, 0.0))

    # --- Compute exit position ---
    exit_local_x = cfg.size[0]
    exit_local_y = (
        wall_thickness_px + (n_cells_y - 1) * cell_plus_wall + passage_width_px * 0.5
    ) * cfg.horizontal_scale
    cfg.exit_info = {
        "position": (exit_local_x, exit_local_y, 0.0),
        "yaw": 0.0,
        "spawn_positions": spawn_positions,
    }

    return hf_raw


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class HfErodedMazeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a corner-eroded maze height-field terrain.

    Identical to ``HfMazeTerrainCfg`` but with corner-erosion post-processing:
    at every L/T/cross wall junction, the corner post pixels are cleared and
    each connected wall segment is shortened by half (the half nearest the
    post is set to ground height). This eliminates sharp 90-degree corners
    while preserving the maze topology.

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

    # Central platform width for safe spawning
    platform_width: float = 1.5
