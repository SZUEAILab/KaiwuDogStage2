"""Custom terrain generators for unitree_rl_lab.

This module provides custom trimesh terrain generators that integrate with Isaac Lab's
TerrainGeneratorCfg sub_terrains system.
"""


from .maze_terrain import NavMazeTerrainCfg
from .track_generator import TrackTerrainGeneratorCfg, compute_default_exit_info, StandardTerrainGeneratorCfg
from .terrain_exit_manager import TerrainExitManager, compute_default_exit_for_block
from .winding_corridor_terrain import WindingCorridorTerrainCfg
from .maze_terrain_unitree import HfMazeTerrainCfg
from .eroded_maze_terrain import ErodedMazeTerrainCfg
from .eroded_maze_terrain_hf import HfErodedMazeTerrainCfg
from .open_entry_eroded_maze_terrain import OpenEntryErodedMazeTerrainCfg
