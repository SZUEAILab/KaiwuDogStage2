# agent_ppo

## 模型架构

### 总览

```mermaid
flowchart LR
    subgraph flat["扁平模式 (flat)"]
        direction LR
        po1["policy_obs<br/>(301/305D)"] --> ac1["ActorCritic<br/>[512,256,128]"]
        co1["critic_obs<br/>(316/320D)"] --> ac1
        ac1 --> out1["12-DOF joint_actions"]
    end

    subgraph hier["层级模式 (hier_nav)"]
        direction LR
        po2["policy_obs<br/>(305D)"] --> nav["NavActorCritic<br/>[256,128,64]<br/>可训练"]
        co2["critic_obs<br/>(320D)"] --> nav
        nav --> cmd["velocity_cmd<br/>[vx, vy, wz]"]
        cmd --> loco["Frozen ActorCritic<br/>[512,256,128]<br/>冻结"]
        loco --> out2["12-DOF joint_actions"]
    end
```

### 模型对比

| | ActorCritic (运控) | NavActorCritic (导航) |
|---|---|---|
| **用途** | 关节级运动控制 | 高层速度指令决策 |
| **输出** | 12-DOF 关节位置 | 3D 速度指令 `[vx, vy, wz]` |
| **Actor MLP** | [512 → 256 → 128] → 12 | [256 → 128 → 64] → 3 |
| **Critic MLP** | [512 → 256 → 128] → 1 | [256 → 128 → 64] → 1 |
| **隐藏层激活** | ELU | ELU |
| **Actor 输出层** | Linear (无界 raw) | Linear → Tanh → 仿射映射 |
| **Actor 输出范围** | (-∞, +∞) | [cmd_lower, cmd_upper]，如 vx∈[0, 0.8] |
| **Actor LayerNorm** | 无 | 每个隐藏层后有 |
| **Critic LayerNorm** | 隐藏层后有 | 隐藏层后有 |
| **Actor 权重初始化** | 默认 PyTorch init | 正交初始化 (gain=0.01) |
| **Critic 权重初始化** | 默认 PyTorch init | 正交初始化 (gain=1.0) |
| **init_noise_std** | 1.0 | 0.5 |
| **训练状态** | 层级模式时冻结 | 始终可训练 |

### 原始观测 → 处理后观测布局

环境原始观测:

```
policy_obs (305D):
  [0:3]    base_ang_vel(3)    } 
  [3:6]    projected_gravity(3)} proprio(45)
  [6:9]    velocity_cmd(3)    }
  [9:21]   joint_pos(12)      }
  [21:33]  joint_vel(12)      } proprio(45)
  [33:45]  last_action(12)    }
  [45:301] height_scan(256)   — scan
  [301:305] goal(4)           — task info

critic_obs (320D):
  [0:3]    base_lin_vel(3)    ← 特权, policy 没有
  [3:6]    base_ang_vel(3)    }
  [6:9]    projected_gravity(3)} critic_proprio(60)
  [9:12]   velocity_cmd(3)    }
  [12:24]  joint_pos(12)      }
  [24:36]  joint_vel(12)      }
  [36:48]  joint_effort(12)   ← 特权, policy 没有
  [48:60]  last_action(12)    }
  [60:316] height_scan(256)   — scan
  [316:320] goal(4)           — task info
```

处理后观测路由:

| 模型 | obs 来源 | 维度 | 内容 |
|------|---------|:---:|------|
| **Loco** | `_build_loco_obs(obs, nav_cmd)` | 301 | proprio(45, cmd 替换为 nav 输出) + scan(256) |
| **Nav Actor** | `_build_nav_obs(obs)` | 266 | base_ang_vel(3) + projected_gravity(3) + scan(256) + goal(4) |
| **Nav Critic** | `_build_nav_critic_obs(critic_obs)` | 269 | base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + scan(256) + goal(4) |

各模型用到的原始字段:

| 原始字段 | 维度 | Loco | Nav Actor | Nav Critic |
|----------|:---:|:---:|:---:|:---:|
| base_lin_vel | 3 | — | — | ✓ (特权) |
| base_ang_vel | 3 | ✓ | ✓ | ✓ |
| projected_gravity | 3 | ✓ | ✓ | ✓ |
| velocity_cmd | 3 | ✓ (由 nav 替换) | — | — |
| joint_pos/vel/last_action | 33 | ✓ | — | — |
| joint_effort | 12 | — | — | — |
| height_scan | 256 | ✓ | ✓ | ✓ |
| goal | 4 | — | ✓ | ✓ |

### nav_obs 构建逻辑

```python
# Loco: proprio + scan, velocity_cmd 被 nav 输出替换
_build_loco_obs(obs, nav_actions):
    loco_obs = obs[:, :301]           # proprio(45) + scan(256)
    loco_obs[:, 6:9] = nav_actions  # 替换随机 velocity_cmd
    return loco_obs                    # 301 dim

# Nav Actor: 基础机身状态 + scan + goal (无 cmd, 无关节)
_build_nav_obs(obs):
    obs[:, :6]   = base_ang_vel(3) + projected_gravity(3)   # body state
    obs[:, 45:]  = height_scan(256) + goal(4)               # perception + task
    return cat → 266 dim

# Nav Critic: 特权机身状态 + scan + goal
_build_nav_critic_obs(critic_obs):
    critic_obs[:, :9]  = base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    critic_obs[:, 60:] = height_scan(256) + goal(4)
    return cat → 269 dim
```

### 关键设计决策

- **Nav 输出不对称有界**: Tanh 压缩输出到 [-1, 1]，再仿射映射到 [cmd_lower, cmd_upper]。`vx ∈ [0, 0.8]` 禁止后退，`vy ∈ [-0.3, 0.3]`、`wz ∈ [-1.5, 1.5]` 保留对称范围
- **Nav Actor 没有 base_lin_vel**: 线速度只在 critic_obs 中，是特权信息。nav actor 只用 `base_ang_vel + projected_gravity`，训推一致
- **Nav Actor 没有 velocity_cmd**: policy_obs[6:9] 的 velocity_cmd 来自环境 command_manager 的随机采样，跟 nav 输出无关，是噪声
- **Nav Critic 有 base_lin_vel**: 不对称 Actor-Critic，critic 用特权信息把 value 估得更准
- **Loco 的 velocity_cmd 被 nav 替换**: `_build_loco_obs` 把随机 velocity_cmd 换成 nav 输出的真实指令，loco 跟踪的是 nav 的指令而非随机指令

---

## 模型加载与 checkpoint 兼容性

层级训练（hier_nav）和扁平训练（standard/track e2e）的 checkpoint 可以**互相加载**，自动识别格式。

### checkpoint 格式

| 格式 | 内容 | 产生方式 |
|------|------|---------|
| 扁平 | `ActorCritic.state_dict()` — 参数名直接映射 | standard / track e2e 阶段 `save_model` |
| 层级 | `{"loco_model": ..., "nav_model": ...}` — 两个 dict | hier_nav 阶段 `save_model` |

### 加载行为

```mermaid
flowchart TD
    load["load_model(path)"] --> is_hier{"当前模式?"}

    is_hier -->|"hier_nav"| hier_type{"checkpoint 格式?"}
    is_hier -->|"flat"| flat_type{"checkpoint 格式?"}

    hier_type -->|"层级<br/>{loco_model, nav_model}"| has_nav{"有 nav_model?"}
    hier_type -->|"扁平<br/>ActorCritic state_dict"| hier_flat["加载为 loco<br/>nav 从头训"]

    has_nav -->|"有"| hier_full["加载 loco + nav<br/>续训"]
    has_nav -->|"无"| hier_loco["加载 loco<br/>nav 从头训"]

    flat_type -->|"层级"| flat_extract["提取 loco_model<br/>忽略 nav_model<br/>→ 回退运控训练"]
    flat_type -->|"扁平"| flat_direct["直接加载"]
```

### 日志关键字

| 日志前缀 | 含义 |
|---------|------|
| `[HierLoad] Hierarchical checkpoint detected` | 加载层级 ckpt |
| `[HierLoad] nav_model weights found` | 成功加载 nav |
| `[HierLoad] No nav_model in checkpoint` | ckpt 缺 nav，从头训 |
| `[HierLoad] Flat checkpoint` | 从扁平 ckpt 提取 loco |
| `[FlatLoad] Hierarchical checkpoint` | flat 模式加载层级 ckpt，提 loco 丢 nav |
| `exact match` | 权重形状完全匹配 |
| `partial` | 权重形状不匹配，按最小维度裁剪加载 |

### 典型工作流

```python
# 1. standard 训运控
Config.CURRENT = AllTerrainConfig

# 2. 切换 hier_nav，自动从 flat ckpt 加载 loco
Config.CURRENT = TrackHierNavMazeConfig
# 日志: [HierLoad] Flat checkpoint → loading as loco only. Nav will train from scratch.

# 3. 续训 hier_nav，加载层级 ckpt（含 nav）
# 日志: [HierLoad] nav_model weights found — loading nav from checkpoint

# 4. 回退 standard 继续训运控，从层级 ckpt 提取 loco
Config.CURRENT = AllTerrainConfig
# 日志: [FlatLoad] Hierarchical checkpoint → extracting loco_model, nav_model ignored
```
