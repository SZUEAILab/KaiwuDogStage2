# Copyright (c) 2024-2026, Tencent Kaiwu Team.
# SPDX-License-Identifier: BSD-3-Clause

"""Open-entry eroded-corner maze terrain generator for Isaac Lab (trimesh version).

Identical to ``eroded_maze_terrain`` but with the **entire left boundary
(X=0 side)** removed — no wall segments, no corner posts on that edge.
This creates a fully open entry face so that the robot can seamlessly
transition from the previous track segment at any Y position.

The right boundary (exit side) retains its original behaviour: a wall with
a single exit opening at the top-right cell.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from .eroded_maze_terrain import (
    _compute_corner_erosion,
    _generate_maze_dfs,
    _verify_connectivity_grid,
)


# ---------------------------------------------------------------------------
# Main terrain function
# ---------------------------------------------------------------------------


def open_entry_eroded_maze_terrain(
    difficulty: float, cfg: "OpenEntryErodedMazeTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a corner-eroded maze terrain with the entire left boundary open.

    Identical to ``eroded_maze_terrain`` except:
    - The entire left boundary wall (X=0 side) is omitted.
    - No corner posts are placed along the left edge (ix == 0).
    - The entry opening logic is skipped since the whole side is open.

    Args:
        difficulty: Terrain difficulty in [0, 1]. Controls passage width.
        cfg: Open-entry eroded maze terrain configuration.

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
    # Exit is placed at the middle of the right boundary wall (Y-centered),
    # instead of the top-right corner, so the opening visually sits at the
    # mid-point of the wall and aligns better with the next track segment.
    exit_cx, exit_cy = n_cells_x - 1, n_cells_y // 2

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

    bef = getattr(cfg, "boundary_erosion_fraction", 0.8)

    # ===================================================================
    # Pre-screen clearance: remove ONLY the interior maze walls and posts
    # whose geometry overlaps with the screen wall's AABB.  We compute
    # the screen's world-coordinate bounding box, then check each wall
    # segment and post individually for overlap before removing it.
    # ===================================================================
    if getattr(cfg, "exit_screen_enabled", True):
        _scr_offset = float(getattr(cfg, "exit_screen_offset_cells", 1.0))
        _scr_wf = float(getattr(cfg, "exit_screen_width_factor", 2.5))
        _exit_cy_world = wall_thickness + exit_cy * cell_w_y + pw_y / 2.0
        # Screen AABB (world coords)
        _scr_cx = terrain_x - wall_thickness - _scr_offset * cell_w_x
        _scr_len_y = min(pw_y * _scr_wf, terrain_y - 2.0 * wall_thickness - 0.4)
        _scr_x_lo = _scr_cx - wall_thickness / 2.0
        _scr_x_hi = _scr_cx + wall_thickness / 2.0
        _scr_y_lo = _exit_cy_world - _scr_len_y / 2.0
        _scr_y_hi = _exit_cy_world + _scr_len_y / 2.0

        def _aabb_overlaps(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
            return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0

        # Check each h_wall segment: h_walls[cy, cx] sits at
        #   x: [wt + cx*cw_x, wt + cx*cw_x + pw_x]  y: [wt + (cy+1)*cw_y - wt, wt + (cy+1)*cw_y]
        for cy in range(h_walls.shape[0]):
            for cx in range(h_walls.shape[1]):
                if not h_walls[cy, cx]:
                    continue
                wx0 = wall_thickness + cx * cell_w_x
                wx1 = wx0 + pw_x
                wy0 = wall_thickness + (cy + 1) * cell_w_y - wall_thickness
                wy1 = wy0 + wall_thickness
                if _aabb_overlaps(wx0, wy0, wx1, wy1, _scr_x_lo, _scr_y_lo, _scr_x_hi, _scr_y_hi):
                    h_walls[cy, cx] = False
                    h_erode_left[cy, cx] = False
                    h_erode_right[cy, cx] = False

        # Check each v_wall segment: v_walls[cy, cx] sits at
        #   x: [wt + (cx+1)*cw_x - wt, wt + (cx+1)*cw_x]  y: [wt + cy*cw_y, wt + cy*cw_y + pw_y]
        for cy in range(v_walls.shape[0]):
            for cx in range(v_walls.shape[1]):
                if not v_walls[cy, cx]:
                    continue
                wx0 = wall_thickness + (cx + 1) * cell_w_x - wall_thickness
                wx1 = wx0 + wall_thickness
                wy0 = wall_thickness + cy * cell_w_y
                wy1 = wy0 + pw_y
                if _aabb_overlaps(wx0, wy0, wx1, wy1, _scr_x_lo, _scr_y_lo, _scr_x_hi, _scr_y_hi):
                    v_walls[cy, cx] = False
                    v_erode_top[cy, cx] = False
                    v_erode_bottom[cy, cx] = False

        # Check each post: post at (ix, iy) sits at
        #   x: [ix*cw_x, ix*cw_x + wt]  y: [iy*cw_y, iy*cw_y + wt]
        for iy in range(post_eroded.shape[0]):
            for ix in range(post_eroded.shape[1]):
                if post_eroded[iy, ix]:
                    continue  # already skipped
                px0 = ix * cell_w_x
                px1 = px0 + wall_thickness
                py0 = iy * cell_w_y
                py1 = py0 + wall_thickness
                if _aabb_overlaps(px0, py0, px1, py1, _scr_x_lo, _scr_y_lo, _scr_x_hi, _scr_y_hi):
                    post_eroded[iy, ix] = True

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

    # === Outer boundary walls ===
    # LEFT BOUNDARY (X=0): ENTIRELY REMOVED — this is the key difference
    # No left boundary wall segments are drawn at all.

    # Right boundary (x = terrain_x) with exit opening — same as original
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
    for cy in range(n_cells_y - 1):
        for cx in range(n_cells_x):
            if not h_walls[cy, cx]:
                continue

            el = h_erode_left[cy, cx]
            er = h_erode_right[cy, cx]

            full_x = wall_thickness + cx * cell_w_x + pw_x / 2.0
            wy = wall_thickness + (cy + 1) * cell_w_y - wall_thickness / 2.0

            if el and er:
                continue
            elif el and not er:
                frac_l = bef if h_boundary_left[cy, cx] else 0.5
                seg_len = pw_x * (1.0 - frac_l)
                seg_cx = full_x + pw_x / 2.0 - seg_len / 2.0
                _add_box(seg_cx, wy, half_h, seg_len, wall_thickness, wall_height)
            elif er and not el:
                frac_r = bef if h_boundary_right[cy, cx] else 0.5
                seg_len = pw_x * (1.0 - frac_r)
                seg_cx = full_x - pw_x / 2.0 + seg_len / 2.0
                _add_box(seg_cx, wy, half_h, seg_len, wall_thickness, wall_height)
            else:
                _add_box(full_x, wy, half_h, pw_x, wall_thickness, wall_height)

    # === Internal vertical walls (between col cx and cx+1) ===
    for cy in range(n_cells_y):
        for cx in range(n_cells_x - 1):
            if not v_walls[cy, cx]:
                continue

            et = v_erode_top[cy, cx]
            eb = v_erode_bottom[cy, cx]

            wx = wall_thickness + (cx + 1) * cell_w_x - wall_thickness / 2.0
            full_y = wall_thickness + cy * cell_w_y + pw_y / 2.0

            if et and eb:
                continue
            elif et and not eb:
                frac_t = bef if v_boundary_top[cy, cx] else 0.5
                seg_len = pw_y * (1.0 - frac_t)
                seg_cy = full_y + pw_y / 2.0 - seg_len / 2.0
                _add_box(wx, seg_cy, half_h, wall_thickness, seg_len, wall_height)
            elif eb and not et:
                frac_b = bef if v_boundary_bottom[cy, cx] else 0.5
                seg_len = pw_y * (1.0 - frac_b)
                seg_cy = full_y - pw_y / 2.0 + seg_len / 2.0
                _add_box(wx, seg_cy, half_h, wall_thickness, seg_len, wall_height)
            else:
                _add_box(wx, full_y, half_h, wall_thickness, pw_y, wall_height)

    # === Wall posts ===
    for iy in range(n_cells_y + 1):
        for ix in range(n_cells_x + 1):
            # Skip eroded corner posts
            if post_eroded[iy, ix]:
                continue

            # Skip ALL posts on the left boundary (ix == 0) — open entry
            if ix == 0:
                continue

            px = ix * cell_w_x + wall_thickness / 2.0
            py = iy * cell_w_y + wall_thickness / 2.0

            # Skip exit boundary posts (right side)
            is_exit_post = (ix == n_cells_x and iy == exit_cy) or (ix == n_cells_x and iy == exit_cy + 1)
            if is_exit_post:
                continue

            # Skip spawn clearance zone posts
            if 1 <= ix <= min(entry_cx + cells_to_clear_x, n_cells_x - 1) and 1 <= iy <= min(
                entry_cy + cells_to_clear_y, n_cells_y - 1
            ):
                continue

            _add_box(px, py, half_h, wall_thickness, wall_thickness, wall_height)

    # ===================================================================
    # Screen wall ("屏风"): a short wall placed directly in front of the
    # exit opening, centred on the exit Y and offset ~1 cell inward.
    # This guarantees NO straight-line path into the exit — the robot
    # MUST go around the screen to reach the opening.
    #
    # Layout (top view, X increases rightward):
    #
    #                  screen wall
    #                  ┌──────┐
    #   ...plaza...    │      │     exit opening
    #                  └──────┘     ← right boundary wall
    #
    # The screen is wider than the exit opening so a head-on approach
    # is fully blocked. Two gaps (one on each Y side) let the robot
    # pass around.
    # ===================================================================
    if getattr(cfg, "exit_screen_enabled", True):
        screen_offset_x = float(getattr(cfg, "exit_screen_offset_cells", 1.0)) * cell_w_x
        screen_width_factor = float(getattr(cfg, "exit_screen_width_factor", 2.5))
        # Screen position: same X as the exit minus offset, centred on exit Y
        screen_x = terrain_x - wall_thickness - screen_offset_x
        screen_y = exit_center_y
        # Width (Y direction): wider than exit opening so it blocks the straight shot
        screen_length_y = min(
            pw_y * screen_width_factor,
            terrain_y - 2.0 * wall_thickness - 0.4,  # leave gaps at top/bottom
        )
        _add_box(screen_x, screen_y, half_h, wall_thickness, screen_length_y, wall_height)

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
class OpenEntryErodedMazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a corner-eroded maze with fully open left boundary.

    Identical to ``ErodedMazeTerrainCfg`` but the entire left boundary wall
    (X=0 side) is removed. This allows seamless transition from the previous
    track segment — the robot can enter the maze at any Y position without
    hitting a wall.

    The right boundary retains its single exit opening at the top-right cell,
    same as the original eroded maze.

    Curriculum controls passage width:
        passage_width = passage_width_max - (max - min) * difficulty
        difficulty=0 -> passage_width_max (easy, wide passages)
        difficulty=1 -> passage_width_min (hard, narrow passages)
    """

    function = open_entry_eroded_maze_terrain

    # Wall height in meters
    wall_height: float = 0.5

    # Wall thickness in meters
    wall_thickness: float = 0.1

    # Passage width range (curriculum-controlled)
    passage_width_max: float = 1.0
    passage_width_min: float = 0.6

    # Erosion fraction at boundary corners
    boundary_erosion_fraction: float = 0.80

    # Safe spawn clearance radius (meters)
    spawn_clearance: float = 0.5

    # Seed for reproducibility
    seed: int | None = None

    # ------------------------------------------------------------------
    # Exit screen wall (屏风): a short wall directly in front of the exit
    # opening that blocks any straight-line approach. The robot must go
    # around the screen on either side.
    # ------------------------------------------------------------------
    exit_screen_enabled: bool = True
    # How many cells inward from the exit wall to place the screen.
    exit_screen_offset_cells: float = 1.0
    # Screen Y-length = exit_opening * this factor (must be > 1 to block).
    exit_screen_width_factor: float = 2.5
