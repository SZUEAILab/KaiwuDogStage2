# 两阶段训练

四足机器人（Unitree Go2）强化学习训练，采用两阶段课程设计。

## Stage 1：运控训练（Standard 地形）

**目标**：在混合地形上学稳走路。

| 子阶段 | 地形 | 特点 |
|---|---|---|
| all_terrain | 5 种地形各 20% | 推荐起步，均衡全面 |
| locomotion | 5 种地形混合 | 下楼梯权重 30% |
| stairs_down | 100% 下楼梯 | 专注训练，慢速强稳定 |

- **模型**：ActorCritic（512-256-128 MLP），输出 12-DOF 关节角度
- **观测**：`[proprio(45) | height_scan(256)]` = 301 维
- **奖励**：速度跟踪、基座稳定、关节平滑、碰撞避免、脚部接触质量等
- **地形**：10 难度级别 × 20 并行地块，课程学习自动晋级/降级
- **输出**：一个能在所有标准地形上行走的 locomotion checkpoint

## Stage 2：导航训练（Track 地形）

**目标**：在赛道上导航穿越障碍到达终点。两种模式可选：

### Path A：端到端导航（flat 模式）

- 沿用 ActorCritic 直接输出 12-DOF 关节，导航+运控一体训练
- 观测增加 4 维 goal：`[proprio(45) | height_scan(256) | goal(4)]` = 305 维
- Episode 延长至 150s，导航奖励占主导（接近目标、到达奖励、避障减速等）

### Path B：层级导航（hierarchical 模式）

```
obs → NavActorCritic → [vx, vy, wz] → 冻结 ActorCritic → 12-DOF joints
       ↑ 可训练                           ↑ 从 Stage 1 加载，冻结
```

- **High-level（nav）**：NavActorCritic（256-128-64 MLP），输出 3D 速度指令
- **Low-level（loco）**：冻结的 Stage 1 ActorCritic，将速度指令转为 12-DOF 关节动作
- 仅训练 nav 模型，移除关节级奖励（nav 无法直接控制），熵系数从 0.01 衰减至 0.001
- Checkpoint 加载支持精确匹配和跨阶段部分迁移
