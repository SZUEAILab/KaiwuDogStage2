# Copyright (c) 2024-2026, Tencent Kaiwu Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Maze terrain generator for Isaac Lab.

Generates a random maze with 0.5m-high walls, guaranteeing at least one
traversable path from the entry edge to the opposite edge.

Uses randomized DFS (Recursive Backtracker) to carve passages, then converts
the maze grid into a height-field array compatible with Isaac Lab's terrain
pipeline.

Curriculum:
    passage_width = 1.0 - 0.4 * difficulty
    (difficulty=0 -> 1.0m wide passages, difficulty=1 -> 0.6m wide passages)
"""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def maze_terrain(difficulty: float, cfg: HfMazeTerrainCfg) -> np.ndarray:
    """Generate a maze terrain as a height-field.

    The maze is built on a grid of cells. Walls are placed between cells, and a
    randomized DFS carves passages to ensure connectivity. The entry edge and
    exit edge (opposite side) are always open.

    Args:
        difficulty: Terrain difficulty in [0, 1]. Controls passage width.
        cfg: Maze terrain configuration.

    Returns:
        A 2D numpy array (int16) representing the height-field.
    """
    # --- Resolve dimensions in pixel units ---
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # Wall height in discrete units
    wall_height = int(cfg.wall_height / cfg.vertical_scale)

    # Passage width with curriculum
    passage_width_m = cfg.passage_width_max - (cfg.passage_width_max - cfg.passage_width_min) * difficulty
    passage_width_px = max(int(passage_width_m / cfg.horizontal_scale), 2)

    # Wall thickness in pixels
    wall_thickness_px = max(int(cfg.wall_thickness / cfg.horizontal_scale), 1)

    # --- Compute maze grid dimensions ---
    # Each cell = passage_width_px, each wall = wall_thickness_px
    # Total = n_cells * passage_width + (n_cells + 1) * wall_thickness
    # Solve for n_cells: n_cells = (total - wall_thickness) / (passage + wall)
    cell_plus_wall = passage_width_px + wall_thickness_px
    n_cells_x = max(int((width_pixels - wall_thickness_px) / cell_plus_wall), 2)
    n_cells_y = max(int((length_pixels - wall_thickness_px) / cell_plus_wall), 2)

    # --- Generate maze using Randomized DFS (iterative) ---
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") and cfg.seed is not None else None)

    # visited[cy][cx] = True if cell was visited
    visited = np.zeros((n_cells_y, n_cells_x), dtype=bool)
    # walls: horizontal walls (between cell(y) and cell(y+1)) and vertical walls
    # h_walls[y][x]: wall below cell (y, x), i.e., between row y and y+1
    # v_walls[y][x]: wall to the right of cell (y, x)
    h_walls = np.ones((n_cells_y - 1, n_cells_x), dtype=bool)  # True = wall present
    v_walls = np.ones((n_cells_y, n_cells_x - 1), dtype=bool)

    # Iterative DFS with explicit stack
    start_x, start_y = 0, 0
    visited[start_y, start_x] = True
    stack = [(start_x, start_y)]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (dx, dy)

    while stack:
        cx, cy = stack[-1]
        # Find unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < n_cells_x and 0 <= ny < n_cells_y and not visited[ny, nx]:
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            # Pick random neighbor
            idx = rng.integers(len(neighbors))
            nx, ny, dx, dy = neighbors[idx]
            # Remove wall between current and neighbor
            if dx == 1:  # moving right
                v_walls[cy, cx] = False
            elif dx == -1:  # moving left
                v_walls[ny, nx] = False
            elif dy == 1:  # moving down
                h_walls[cy, cx] = False
            elif dy == -1:  # moving up
                h_walls[ny, nx] = False
            visited[ny, nx] = True
            stack.append((nx, ny))
        else:
            stack.pop()

    # --- Render maze to height-field ---
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    def fill_rect(x_start, y_start, x_end, y_end, value):
        """Fill a rectangular region in the height-field."""
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

    # Carve horizontal passages (remove h_walls: between row cy and cy+1)
    for cy in range(n_cells_y - 1):
        for cx in range(n_cells_x):
            if not h_walls[cy, cx]:
                # Remove wall between cell(cy,cx) and cell(cy+1,cx)
                x_start = wall_thickness_px + cx * cell_plus_wall
                y_start = wall_thickness_px + cy * cell_plus_wall + passage_width_px
                x_end = x_start + passage_width_px
                y_end = y_start + wall_thickness_px
                fill_rect(x_start, y_start, x_end, y_end, 0)

    # Carve vertical passages (remove v_walls: between col cx and cx+1)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x - 1):
            if not v_walls[cy, cx]:
                # Remove wall between cell(cy,cx) and cell(cy,cx+1)
                x_start = wall_thickness_px + cx * cell_plus_wall + passage_width_px
                y_start = wall_thickness_px + cy * cell_plus_wall
                x_end = x_start + wall_thickness_px
                y_end = y_start + passage_width_px
                fill_rect(x_start, y_start, x_end, y_end, 0)

    # --- Open entry and exit edges ---
    # Entry: left edge (x=0), carve opening at first row cells
    entry_y = wall_thickness_px + 0 * cell_plus_wall
    fill_rect(0, entry_y, wall_thickness_px, entry_y + passage_width_px, 0)

    # Exit: right edge (x=width_pixels), carve opening at last row cell
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

    # --- Compute precise exit position and attach to cfg ---
    # Exit opening: right edge (x = size_x), Y = center of last-row cell opening
    # exit_y pixel = wall_thickness_px + (n_cells_y - 1) * cell_plus_wall + passage_width_px / 2
    exit_local_x = cfg.size[0]  # right edge of terrain block
    exit_local_y = (
        wall_thickness_px + (n_cells_y - 1) * cell_plus_wall + passage_width_px * 0.5
    ) * cfg.horizontal_scale
    cfg.exit_info = {
        "position": (exit_local_x, exit_local_y, 0.0),
        "yaw": 0.0,  # facing +X direction
        "spawn_positions": spawn_positions,
    }

    return hf_raw


@configclass
class HfMazeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a maze height-field terrain.

    The maze consists of walls of fixed height (0.5m by default) with passages
    carved using randomized DFS. At least one path from entry to exit is guaranteed.

    Curriculum controls passage width:
        passage_width = passage_width_max - (max - min) * difficulty
        difficulty=0 -> passage_width_max (easy, wide passages)
        difficulty=1 -> passage_width_min (hard, narrow passages)
    """

    function = maze_terrain

    # Wall height in meters (fixed at 0.5m per competition spec)
    wall_height: float = 0.5

    # Wall thickness in meters
    wall_thickness: float = 0.1

    # Passage width range (curriculum-controlled)
    # passage_width = max - (max - min) * difficulty
    passage_width_max: float = 1.0
    passage_width_min: float = 0.6

    # Central platform width for safe spawning
    platform_width: float = 1.5
