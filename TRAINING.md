# 四足机器人（Unitree Go2）训练流程

## 两种训练路线

| | 指令跟踪路线 | 端到端路线（推荐） |
|---|---|---|
| 核心思想 | 策略跟踪速度指令，运控/导航分离 | 策略直接输出关节，无中间指令 |
| 训练速度 | 较慢（需要学会跟踪随机指令） | 较快（目标单一：往前走） |
| 泛化性 | 强（各种方向都能走） | 专注前向，track 模式补导航 |

---

# 端到端路线（E2E Pipeline）

四阶段课程设计：先学会走 → 学会避障 → 学会导航 → 赛道通关。

```
Stage 1                Stage 2              Stage 3              Stage 4
forward_e2e           maze_e2e             open_maze_e2e        full_e2e
(standard)            (standard)           (track)              (track)
─────────────────    ─────────────────    ─────────────────    ─────────────────
5 种地形前向行走      迷宫墙体感知          短赛道导航            完整赛道通关
固定前向指令          固定前向指令          固定前向指令          固定前向指令
forward_velocity_world      forward_velocity_world     forward_velocity_world     forward_velocity_world
+ 稳定性约束          + wall_proximity     + heuristic_nav      + heuristic_nav
                                         + approach_goal       + approach_goal
                                         + reach_goal          + reach_goal
```

## Stage 1：前向运动（forward_e2e）

**目标**：在所有地形上学会稳定前向行走。

- **配置文件**：`train_env_conf_standard_forward_e2e.toml`
- **Config 类**：`ForwardE2EConfig`
- **地形**：5 种（斜坡、逆斜坡、楼梯、逆楼梯、迷宫），均衡比例
- **模型**：ActorCritic（512-256-128 MLP），输出 12-DOF 关节角度
- **观测**：`[proprio(45) | height_scan(256)]` = 301 维
- **指令**：固定前向 `lin_vel_x=[0.3, 0.8]`，不跟踪
- **核心奖励**：`forward_velocity_world`（世界坐标系 X 轴速度，权重 1.5）
- **约束奖励**：能耗、步态对称、基座稳定、防倾倒、防绊脚
- **输出**：能在地形上稳定向前行走的 checkpoint

切换到该阶段：

```python
# agent_ppo/conf/conf.py
Config.CURRENT = ForwardE2EConfig
```

## Stage 2：迷宫感知（maze_e2e）

**目标**：在平坦迷宫中学会前向行走 + 反应式墙体感知。

- **配置文件**：`train_env_conf_standard_maze_e2e.toml`
- **Config 类**：`MazeE2EConfig`
- **地形**：100% maze（平坦地面 + 0.5m 随机墙体）
- **前置**：完成 Stage 1
- **核心奖励**：`forward_velocity_world`（1.5）+ `wall_proximity_brake`（-0.5，近墙减速）
- **约束奖励**：能耗、步态对称、基座高度、防倾倒
- **设计逻辑**：运控只负责"看着墙减速"，不负责"选方向绕墙"（那是导航的事）
- **输出**：能在迷宫中安全前行的 checkpoint

```python
Config.CURRENT = MazeE2EConfig
```

## Stage 3：开放迷宫导航（open_maze_e2e）

**目标**：在短 track 赛道上导航到目标点。

- **配置文件**：`train_env_conf_track_open_maze_e2e.toml`
- **Config 类**：`OpenMazeE2EConfig`
- **地形**：track 模式，2 段（斜坡 → 开放迷宫），10 难度级别
- **前置**：完成 Stage 1 + Stage 2
- **模型**：ActorCritic，观测增加 4 维 goal：`[proprio(45) | height_scan(256) | goal(4)]` = 305 维
- **Episode**：60s
- **核心奖励**：
  | 奖励 | 权重 | 作用 |
  |---|---|---|
  | `forward_velocity_world` | 1.0 | 世界前向速度，驱动前进 |
  | `heuristic_navigation` | 4.0 | 通畅时前进，堵塞时转向 clearance 侧 |
  | `approach_goal` | 2.0 | 距离目标越近越奖励 |
  | `reach_goal` | 10.0 | 到达目标一次性大奖 |
  | `navigation_time` | -0.005 | 每步固定扣分，鼓励快速到达 |
- **避障**：`obstacle_evasion`（-0.5）、`wall_proximity_brake`（-1.0）、`deadend_escape`（2.0）
- **输出**：能在短赛道上导航到目标的 checkpoint

```python
Config.CURRENT = OpenMazeE2EConfig
```

## Stage 4：完整赛道通关（full_e2e）

**目标**：完整 5 段赛道端到端导航。

- **配置文件**：`train_env_conf_track_full_e2e.toml`
- **Config 类**：`FullE2EConfig`
- **地形**：track 模式，5 段（斜坡 → 逆斜坡 → 楼梯 → 逆楼梯 → 开放迷宫）
- **前置**：完成 Stage 3
- **Episode**：120s
- **奖励结构**：与 Stage 3 相同，但导航奖励权重更高（`heuristic_navigation` 6.0、`deadend_escape` 3.0），终止惩罚更重（-10.0）
- **输出**：能通关完整赛道的最终策略

```python
Config.CURRENT = FullE2EConfig
```

---

## E2E 设计原则

1. **无指令跟踪**：所有阶段都不使用 `track_lin_vel_xy`、`track_ang_vel_z` 等指令跟踪奖励。指令固定前向，仅作为观测上下文。
2. **世界坐标系前向**：`forward_velocity_world` 使用 `root_lin_vel_w[:, 0]`（世界 X 轴），与评测的"前进距离分"对齐。
3. **逐级叠加**：每阶段在上一阶段基础上增加新的能力，而非重新学习。
4. **运控/导航分工**：Stage 1-2 只学运控（走 + 感知），Stage 3-4 才学导航（往哪走）。

---

# 指令跟踪路线（Command-Following）

原有的两阶段训练路线，策略学习跟踪速度指令。

## Stage 1：运控训练（Standard 地形）

| 子阶段 | 配置文件 | Config 类 | 地形 | 特点 |
|---|---|---|---|---|
| all_terrain | `train_env_conf_standard_all_terrain.toml` | `AllTerrainConfig` | 5 种地形均衡 | 推荐起步，全面 |
| locomotion | `train_env_conf_standard_locomotion.toml` | `LocomotionConfig` | 5 种地形混合 | 下楼梯权重 30% |
| upstairs | `train_env_conf_standard_upstairs.toml` | `UpstairsConfig` | 100% 上楼梯 | 专注训练 |
| upstairs_e2e | `train_env_conf_standard_upstairs_e2e.toml` | `UpstairsE2EConfig` | 100% 上楼梯 | 端到端前向 |
| maze | `train_env_conf_standard_maze.toml` | `MazeConfig` | 100% 迷宫 | 运控 + 反应避障 |

- **模型**：ActorCritic（512-256-128 MLP），输出 12-DOF 关节角度
- **观测**：`[proprio(45) | height_scan(256)]` = 301 维
- **奖励**：速度跟踪、基座稳定、关节平滑、碰撞避免、脚部接触质量等
- **地形**：10 难度级别 × 20 并行地块，课程学习自动晋级/降级

## Stage 2：导航训练（Track 地形）

### Path A：端到端导航

- **配置文件**：`train_env_conf_track_nav.toml`
- **Config 类**：`TrackNavConfig`
- 沿用 ActorCritic 直接输出 12-DOF 关节，导航+运控一体
- 观测增加 4 维 goal：`[proprio(45) | height_scan(256) | goal(4)]` = 305 维
- Episode 延长至 150s，导航奖励占主导

### Path B：层级导航

- **配置文件**：`train_env_conf_track_hier_nav.toml`
- **Config 类**：`HierTrackNavConfig`
- Nav 策略输出 3D 速度指令，冻结运控策略执行
- 仅训练 NavActorCritic（256-128-64 MLP）

---

## 切换阶段

修改 `agent_ppo/conf/conf.py` 中的 `Config.CURRENT`：

```python
# E2E 路线
Config.CURRENT = ForwardE2EConfig     # Stage 1
Config.CURRENT = MazeE2EConfig        # Stage 2
Config.CURRENT = OpenMazeE2EConfig    # Stage 3
Config.CURRENT = FullE2EConfig        # Stage 4

# 指令跟踪路线
Config.CURRENT = AllTerrainConfig     # 全地形
Config.CURRENT = UpstairsConfig       # 上楼梯
Config.CURRENT = MazeConfig           # 迷宫
Config.CURRENT = TrackNavConfig       # 端到端导航
Config.CURRENT = HierTrackNavConfig   # 层级导航
```
