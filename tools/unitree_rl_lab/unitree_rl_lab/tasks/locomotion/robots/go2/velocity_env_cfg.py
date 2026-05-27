import math

import torch
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.terrains import (
    StandardTerrainGeneratorCfg,
    TrackTerrainGeneratorCfg,
    WindingCorridorTerrainCfg,
    HfMazeTerrainCfg,
    NavMazeTerrainCfg,
    ErodedMazeTerrainCfg,
    OpenEntryErodedMazeTerrainCfg,
)

##
# Terrain mode: "standard" or "track"
# Change this to switch between standard grid layout and track (linear course) layout.
##
TERRAIN_MODE = "standard"

# -- Standard mode: grid layout with proportioned sub-terrains
STANDARD_TERRAIN_CFG = StandardTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.35, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "winding_corridor": WindingCorridorTerrainCfg(
            proportion=0.0,
            size=(8.0, 8.0),
            corridor_width_range=(2.0, 4.0),  # 走廊宽度 2~4m
            wall_height=2.0,  # 2m 高墙
            num_obstacles_range=(3, 12),  # 障碍物数量 3~12
            obstacle_height=1.0,  # 障碍物高度 1m
            robot_passable_width=0.6,  # 所有间隙至少 0.6m，保证狗子能通过
        ),
        "maze": ErodedMazeTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            wall_thickness=0.25,
            wall_height=0.5,
            passage_width_max=1.0,
            passage_width_min=0.6,
        ),
    },
)

# -- Track mode: linear course with concatenated sub-terrains along X-axis
TRACK_TERRAIN_CFG = TrackTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,  # will be overridden by track_length
    num_cols=10,  # will be overridden by num_parallel_tracks
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    track_length=4,  # 4 sub-terrains per track (along X-axis)
    num_parallel_tracks=10,  # 10 parallel tracks (along Y-axis)
    sub_terrains_random=False,
    sub_terrains_order=["pyramid_slope", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"],
    sub_terrains={
        # border_width: 地形块外围一圈 z=0 的平地/边框，用于和邻块无缝衔接。
        # - 坡道类 (slope): border=0 也不会有高差（坡面从边缘平滑过渡），故设 0 消除侧面平地；
        # - 阶梯类 (stairs): border=0 会导致最外圈台阶顶面偏离 z=0，与邻块（如 maze）
        #   产生 step_height 高差裂缝，因此保留 0.2m 最小 border——
        #   0.2m < Go2 身宽 (~0.3m)，机器狗无法从这条窄侧平地取巧绕过阶梯。
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.0
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=0.2,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=0.2,
            holes=False,
        ),
        # "maze": ErodedMazeTerrainCfg(
        #     proportion=0.1,
        #     size=(8.0, 8.0),
        #     wall_thickness=0.25,
        #     wall_height=0.5,
        #     passage_width_max=1.0,
        #     passage_width_min=0.6,
        # ),
        "open_entry_maze": OpenEntryErodedMazeTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            wall_thickness=0.25,
            wall_height=0.5,
            passage_width_max=1.0,
            passage_width_min=0.6,
        ),
    },
)

# Select active terrain config based on mode
if TERRAIN_MODE == "track":
    ACTIVE_TERRAIN_CFG = TRACK_TERRAIN_CFG
elif TERRAIN_MODE == "standard":
    ACTIVE_TERRAIN_CFG = STANDARD_TERRAIN_CFG
else:
    raise ValueError(f"Invalid terrain_mode: '{TERRAIN_MODE}'. Must be 'standard' or 'track'.")


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=ACTIVE_TERRAIN_CFG,  # None, STANDARD_TERRAIN_CFG, TRACK_TERRAIN_CFG
        max_init_terrain_level=5,  # taichu: 5
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="/workspace/local_assets/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # Forward height scanner: 16x16 = 256 rays
    # Covers 0~1.5m forward × ±0.75m lateral (resolution 0.1m > wall_thickness 0.25m → no miss)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.75, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Navigation scanner: 13×11 = 143 rays, 2.5m(前方)×2m(横向) 矩形
    # 前瞻转向用：中心前移1.25m → 覆盖前方 [0, 2.5m]，横向 ±1m
    # 2.5m ≈ 1.5~2 个迷宫 cell，能看到大多数 L 型拐角
    # 0.2m 分辨率 < 墙厚 0.25m，避免夹缝漏检
    # ray_alignment="base"：射线跟随完整姿态（含 pitch），上下坡/楼梯时
    # 前方地面在机器人视角下仍是"平面"，不会被误判成墙壁
    nav_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.25, 0.0, 20.0)),  # 中心前移 1.25m
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.5, 2.0]),  # 2.5m×2m, 0.2m 分辨率
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file="/workspace/local_assets/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events.

    Defaults aligned with taichu Stage1EventCfg.
    """

    # startup: friction [0.3, 1.5], restitution [0.0, 0.15]
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    # startup: base mass [-1.0, 1.0] kg added (taichu: [-1, 1])
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # startup: link mass [0.9, 1.1] scale (taichu)
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # startup: COM offset [-0.03, 0.03] m (taichu)
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_track_start if TERRAIN_MODE == "track" else mdp.reset_root_state_maze_random,
        mode="reset",
        params={
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval: push every 4s (taichu: 4s), velocity ±0.4 (taichu: 0.4)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),  # taichu: 5s
        rel_standing_envs=0.1,
        debug_vis=False,
        # taichu initial range
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
        # taichu curriculum target range
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-1.5, 1.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            scale=2.5,
            clip=(-5.0, 5.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            scale=2.5,
            clip=(-5.0, 5.0),
        )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


def _terrain_bounds_termination(env) -> torch.Tensor:
    """Terminate episode when robot walks out of its assigned terrain block (X+Y).

    当机器人走出当前地形块的 X 或 Y 边界时终止 episode。
    仅在 env._enable_terrain_bounds_termination = True 时生效（评估模式），
    训练时默认关闭，避免正常速度跟踪被频繁截断。

    配合 TerminationsCfg 中 time_out=True 使用，确保：
    - 不触发 termination penalty 奖励
    - scorer 将其归为 timeout（正常截断）
    """
    # 开关检查：默认关闭
    if not getattr(env, "_enable_terrain_bounds_termination", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain = env.scene.terrain
    terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    if terrain_gen_cfg is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    half_x = terrain_gen_cfg.size[0] / 2.0
    half_y = terrain_gen_cfg.size[1] / 2.0

    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)
    env_origins = env.scene.env_origins[:, :2]  # (num_envs, 2)
    offset = robot_pos - env_origins

    out_of_bounds = (torch.abs(offset[:, 0]) > half_x) | (torch.abs(offset[:, 1]) > half_y)
    return out_of_bounds


def _goal_reached_termination(env, threshold: float = 0.6) -> torch.Tensor:
    """Terminate episode when robot reaches the navigation goal.

    到达导航目标时终止 episode（距离 < threshold）。
    仅在 env.goal_positions 存在时生效（Stage3 nav），否则返回全 False。

    IMPORTANT: this threshold MUST match `rewards.reach_goal.params.threshold`
    in the nav/hier_nav TOMLs. Termination 与 reward 在同一帧计算，若两者
    阈值不一致，会出现"死区"（terminated 但 reward_reach_goal == 0），
    导致 scorer 统计 completed 时奖励丢失。
    重要：该阈值必须与 nav / hier_nav TOML 里 `rewards.reach_goal.params.threshold`
    保持一致（当前统一为 0.6m），否则会产生"终止-奖励死区"。
    """
    if not hasattr(env, "goal_positions") or env.goal_positions is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # goal_positions 全零 = 尚未被 observation_process 初始化
    if env.goal_positions.abs().max().item() < 1e-6:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    dist = torch.norm(goal_pos - robot_pos, dim=1)
    reached = dist < threshold

    # 调试日志：每 500 步输出一次最近距离（避免刷屏）
    if not hasattr(env, "_goal_term_log_counter"):
        env._goal_term_log_counter = 0
    env._goal_term_log_counter += 1
    if env._goal_term_log_counter % 500 == 1:
        min_dist = dist.min().item()
        n_reached = reached.sum().item()
        print(
            f"[goal_term] min_dist={min_dist:.2f}m, reached={n_reached}/{env.num_envs}, "
            f"goal[0]=({goal_pos[0,0]:.1f},{goal_pos[0,1]:.1f}), "
            f"robot[0]=({robot_pos[0,0]:.1f},{robot_pos[0,1]:.1f})"
        )

    return reached


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    # NOTE: threshold MUST match rewards.reach_goal.threshold in nav/hier_nav
    # TOMLs (currently unified at 0.6m). See _goal_reached_termination docstring.
    # 注意：该 threshold 必须与 nav/hier_nav TOML 中 rewards.reach_goal.threshold
    # 保持一致（统一为 0.6m），避免"终止-奖励死区"。
    goal_reached = DoneTerm(func=_goal_reached_termination, params={"threshold": 0.6})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 25.0  # taichu: 25s
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, "nav_scanner"):
            self.scene.nav_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
