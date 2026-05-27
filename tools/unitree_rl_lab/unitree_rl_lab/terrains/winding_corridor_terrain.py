"""Winding corridor terrain generator — obstacle corridor variant.

Generates a wide corridor bounded by tall side-walls with randomly placed
obstacle blocks inside.  The robot must navigate forward (+X) through the
corridor while avoiding obstacles.

Key guarantee:
- Every obstacle keeps at least ``robot_passable_width`` clearance from
  the side walls AND from every other obstacle, so the robot can always
  squeeze through.

Difficulty controls:
- Corridor width narrows (easier → wider, harder → narrower).
- Obstacle density increases with difficulty.
- Obstacle sizes grow with difficulty.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_box(dims: tuple[float, float, float], pos: tuple[float, float, float]) -> trimesh.Trimesh:
    """Create a trimesh box at a given position.

    Args:
        dims: (size_x, size_y, size_z) box dimensions.
        pos: (x, y, z) center position.

    Returns:
        A ``trimesh.Trimesh`` box.
    """
    return trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))


def _rect_gap(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """Return the minimum gap between two axis-aligned rectangles (XY plane).

    If the rectangles overlap, returns a negative value.

    Args:
        ax, ay: center of rect A.
        aw, ah: full size-x and size-y of rect A.
        bx, by: center of rect B.
        bw, bh: full size-x and size-y of rect B.

    Returns:
        Minimum clearance distance (negative if overlapping).
    """
    gap_x = abs(ax - bx) - (aw + bw) / 2.0
    gap_y = abs(ay - by) - (ah + bh) / 2.0

    if gap_x >= 0 and gap_y >= 0:
        # corner-to-corner distance (diagonal gap)
        return (gap_x**2 + gap_y**2) ** 0.5
    else:
        # overlap on at least one axis → return the larger (less negative) gap
        return max(gap_x, gap_y)


def _wall_clearance_y(
    oy: float,
    ow: float,
    corridor_y_min: float,
    corridor_y_max: float,
) -> float:
    """Return the smaller of the two gaps between an obstacle and the side walls.

    Args:
        oy: obstacle center Y.
        ow: obstacle full width (Y-extent).
        corridor_y_min: inner Y of the low-side wall.
        corridor_y_max: inner Y of the high-side wall.

    Returns:
        The minimum wall-to-obstacle gap (m).
    """
    gap_low = (oy - ow / 2.0) - corridor_y_min
    gap_high = corridor_y_max - (oy + ow / 2.0)
    return min(gap_low, gap_high)


# ---------------------------------------------------------------------------
# Main terrain function
# ---------------------------------------------------------------------------


def winding_corridor_terrain(
    difficulty: float, cfg: "WindingCorridorTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate an obstacle corridor terrain.

    A wide corridor runs along the X-axis with tall walls on both sides.
    Random obstacle blocks are scattered inside the corridor.  The robot
    spawns at one end and must reach the other.

    **Passability guarantee**: every obstacle is placed so that there is at
    least ``robot_passable_width`` clearance between it and the side walls
    as well as between it and every other obstacle.

    Args:
        difficulty: Terrain difficulty (0.0 – 1.0).
        cfg: Configuration dataclass.

    Returns:
        ``(meshes_list, origin, exit_info)``
    """
    terrain_x, terrain_y = cfg.size  # terrain dimensions
    passable = cfg.robot_passable_width  # guaranteed passable gap

    # --- corridor width from difficulty -----------------------------------
    corridor_width = cfg.corridor_width_range[1] + difficulty * (
        cfg.corridor_width_range[0] - cfg.corridor_width_range[1]
    )
    # difficulty 0 → max width (easy), difficulty 1 → min width (hard)

    # The corridor is centred in the terrain along Y
    corridor_y_center = terrain_y / 2.0
    corridor_y_min = corridor_y_center - corridor_width / 2.0
    corridor_y_max = corridor_y_center + corridor_width / 2.0

    meshes: list[trimesh.Trimesh] = []

    # --- side walls -------------------------------------------------------
    wall_h = cfg.wall_height
    wall_t = cfg.wall_thickness

    # Left wall (low-Y side)
    left_wall_y = corridor_y_min - wall_t / 2.0
    meshes.append(
        _create_box(
            (terrain_x, wall_t, wall_h),
            (terrain_x / 2.0, left_wall_y, wall_h / 2.0),
        )
    )

    # Right wall (high-Y side)
    right_wall_y = corridor_y_max + wall_t / 2.0
    meshes.append(
        _create_box(
            (terrain_x, wall_t, wall_h),
            (terrain_x / 2.0, right_wall_y, wall_h / 2.0),
        )
    )

    # --- obstacles --------------------------------------------------------
    # Number of obstacles scales with difficulty
    num_obstacles_min = cfg.num_obstacles_range[0]
    num_obstacles_max = cfg.num_obstacles_range[1]
    num_obstacles = int(num_obstacles_min + difficulty * (num_obstacles_max - num_obstacles_min))

    # Obstacle size scales with difficulty
    obs_w_min = cfg.obstacle_width_range[0]
    obs_w_max = cfg.obstacle_width_range[0] + difficulty * (cfg.obstacle_width_range[1] - cfg.obstacle_width_range[0])
    obs_d_min = cfg.obstacle_depth_range[0]
    obs_d_max = cfg.obstacle_depth_range[0] + difficulty * (cfg.obstacle_depth_range[1] - cfg.obstacle_depth_range[0])
    obs_h = cfg.obstacle_height

    # Clamp max obstacle width (Y) so it can never block the corridor
    # The obstacle must leave at least ``passable`` on BOTH sides to the wall,
    # but it's OK if it only leaves ``passable`` on ONE side (robot goes around).
    max_allowed_obs_w = corridor_width - 2 * passable
    if max_allowed_obs_w < obs_w_min:
        # Corridor is too narrow for any obstacle at this difficulty;
        # skip obstacle placement entirely.
        max_allowed_obs_w = 0.0
    obs_w_max = min(obs_w_max, max_allowed_obs_w) if max_allowed_obs_w > 0 else 0.0

    # RNG
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        rng = np.random.default_rng(seed + int(difficulty * 1000))
    else:
        rng = np.random.default_rng()

    # Keep-out zones: spawn area and exit area (no obstacles there)
    spawn_clearance = cfg.spawn_clearance  # metres free at start
    exit_clearance = cfg.exit_clearance  # metres free at end

    # (cx, cy, size_x, size_y) of already placed obstacles
    placed: list[tuple[float, float, float, float]] = []

    attempts = 0
    max_attempts = num_obstacles * 40  # more attempts since constraints are tighter

    while len(placed) < num_obstacles and attempts < max_attempts and obs_w_max > 0:
        attempts += 1

        # Random obstacle dimensions
        ow = rng.uniform(obs_w_min, max(obs_w_min, obs_w_max))  # Y-extent
        od = rng.uniform(obs_d_min, max(obs_d_min, obs_d_max))  # X-extent

        # Random position inside the corridor
        # X: must avoid spawn / exit zones
        ox = rng.uniform(
            spawn_clearance + od / 2.0,
            terrain_x - exit_clearance - od / 2.0,
        )
        # Y: must keep at least ``passable`` gap to each side wall
        y_lo = corridor_y_min + passable + ow / 2.0
        y_hi = corridor_y_max - passable - ow / 2.0
        if y_lo > y_hi:
            # This particular obstacle is too wide; try a smaller one next time
            continue
        oy = rng.uniform(y_lo, y_hi)

        # ------------------------------------------------------------------
        # Check 1: gap to side walls ≥ passable
        # ------------------------------------------------------------------
        wc = _wall_clearance_y(oy, ow, corridor_y_min, corridor_y_max)
        if wc < passable - 1e-6:
            continue

        # ------------------------------------------------------------------
        # Check 2: gap to every existing obstacle ≥ passable
        # ------------------------------------------------------------------
        too_close = False
        for px, py, pd, pw in placed:
            gap = _rect_gap(ox, oy, od, ow, px, py, pd, pw)
            if gap < passable - 1e-6:
                too_close = True
                break

        if too_close:
            continue

        # Place the obstacle
        meshes.append(
            _create_box(
                (od, ow, obs_h),
                (ox, oy, obs_h / 2.0),
            )
        )
        placed.append((ox, oy, od, ow))

    # --- ground plane -----------------------------------------------------
    ground = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes.append(ground)

    # --- origin (spawn) ---------------------------------------------------
    origin_x = spawn_clearance / 2.0
    origin_y = corridor_y_center
    origin = np.array([origin_x, origin_y, 0.0])

    # --- exit info --------------------------------------------------------
    exit_x = terrain_x - exit_clearance / 2.0
    exit_y = corridor_y_center
    exit_info = {
        "position": (exit_x, exit_y, 0.0),
        "yaw": 0.0,  # facing +X
    }

    return meshes, origin, exit_info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class WindingCorridorTerrainCfg(SubTerrainBaseCfg):
    """Configuration for an obstacle corridor terrain.

    A wide corridor with tall side-walls and randomly placed obstacle blocks.
    The robot navigates forward (+X) while avoiding obstacles.

    **Passability**: every obstacle keeps at least ``robot_passable_width``
    clearance from walls and from other obstacles, guaranteeing the robot
    can always find a path through.
    """

    function = winding_corridor_terrain

    # -- robot passability -------------------------------------------------

    robot_passable_width: float = 0.6
    """Minimum gap (m) that must exist between any obstacle and the walls,
    and between any two obstacles.  Defaults to 0.6 m (sufficient for Go2
    at ~0.35 m body width with turning margin).

    Set this to at least your robot's body width + some turning clearance.
    """

    # -- corridor geometry -------------------------------------------------

    wall_height: float = 2.0
    """Height of the side walls (m). Defaults to 2.0."""

    wall_thickness: float = 0.15
    """Thickness of the side walls (m). Defaults to 0.15."""

    corridor_width_range: tuple[float, float] = (2.0, 4.0)
    """Min and max corridor width (m).

    ``width = max - difficulty × (max - min)``

    difficulty 0 → 4.0 m (easy).  difficulty 1 → 2.0 m (hard).
    """

    # -- obstacles ---------------------------------------------------------

    num_obstacles_range: tuple[int, int] = (3, 12)
    """Min and max number of obstacle blocks.

    ``n = min + difficulty × (max - min)``

    difficulty 0 → 3 obstacles.  difficulty 1 → 12 obstacles.
    """

    obstacle_width_range: tuple[float, float] = (0.3, 1.2)
    """Min and max obstacle width along Y (m).

    The upper bound grows with difficulty.  It is automatically clamped so
    that the obstacle never blocks the entire corridor width.
    """

    obstacle_depth_range: tuple[float, float] = (0.3, 1.0)
    """Min and max obstacle depth along X (m).

    The upper bound grows with difficulty.
    """

    obstacle_height: float = 1.0
    """Height of obstacle blocks (m). Defaults to 1.0.

    Should be tall enough to block the robot but shorter than the side walls.
    """

    # -- spawn / exit zones ------------------------------------------------

    spawn_clearance: float = 2.0
    """Obstacle-free zone at the start of the corridor (m). Defaults to 2.0."""

    exit_clearance: float = 2.0
    """Obstacle-free zone at the end of the corridor (m). Defaults to 2.0."""
