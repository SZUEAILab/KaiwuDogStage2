# 四足机器人（Unitree Go2）训练流程

# 分层训练流程（Command-Following）

策略学习跟踪速度指令，运控/导航分离：第一阶段训练运控策略学会在各种地形跟踪速度指令，第二阶段训练导航策略在赛道上到达目标。

## Stage 1：运控训练（Standard 地形）

| 子阶段 | 配置文件 | Config 类 | 地形 | 特点 |
|---|---|---|---|---|
| locomotion_pro | `train_env_conf_standard_locomotion_pro.toml` | `LocomotionProConfig` | 5 种地形均衡 | 推荐起步，全面 |
| locomotion | `train_env_conf_standard_locomotion.toml` | `LocomotionConfig` | 5 种地形混合 | 下楼梯权重 30% |
| upstairs | `train_env_conf_standard_upstairs.toml` | `UpstairsConfig` | 100% 上楼梯 | 专注训练 |
| maze | `train_env_conf_standard_maze.toml` | `MazeConfig` | 100% 迷宫 | 运控 + 反应避障 |

- **模型**：ActorCritic（512-256-128 MLP），输出 12-DOF 关节角度
- **观测**：`[proprio(45) | height_scan(256)]` = 301 维
- **奖励**：速度跟踪、基座稳定、关节平滑、碰撞避免、脚部接触质量等
- **地形**：10 难度级别 × 20 并行地块，课程学习自动晋级/降级

## Stage 2：导航训练（Track 地形）

先 maze 后全地形，循序渐进。

### Step 1：迷宫导航

- **配置文件**：`train_env_conf_track_hier_nav_maze.toml`
- **Config 类**：`TrackHierNavMazeConfig`
- **地形**：100% maze
- Nav 策略输出 3D 速度指令，冻结运控策略执行
- 仅训练 NavActorCritic（256-128-64 MLP）

### Step 2：全地形导航

- **配置文件**：`train_env_conf_track_hier_nav.toml`
- **Config 类**：`TrackHierNavConfig`
- **地形**：5 种地形混合（斜坡、逆斜坡、楼梯、逆楼梯、迷宫）
- 前置：完成 Step 1

---

## 切换阶段

修改 `agent_ppo/conf/conf.py` 中的 `Config.CURRENT`：

```python
# 运控训练
Config.CURRENT = LocomotionProConfig     # 全地形
Config.CURRENT = UpstairsConfig       # 上楼梯
Config.CURRENT = MazeConfig           # 迷宫

# 导航训练
Config.CURRENT = TrackHierNavMazeConfig   # 迷宫导航（先）
Config.CURRENT = TrackHierNavConfig       # 全地形导航（后）
```
