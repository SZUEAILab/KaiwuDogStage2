"""BaseScorer —— 环境指标分数统计模块。

按地形类型动态分组统计以下指标：
- episode 计数: completed / abnormal / timeout
- 得分: total_score / step_score / forward_score / pose_score / energy_score
- 步数: step
- 前进距离: forward_distance (X 方向逐步累加正向位移，单位: 米)
- reward 分项: Isaac Lab reward_manager 的所有 episode reward sums
  (来自 env.extras["log"] 中的 "Episode_Reward/{term_name}" 条目)

task_type 区分:
  - "standard":
        总分 = 前进距离分数 × 0.4 + 时间分数 × 0.2 + 能耗分数 × 0.2 + 姿态分数 × 0.2
        前进距离分数 = min(d_forward / L_forward_score, 1.0) × 100
            d_forward = max over episode of ||current_xy - episode_start_xy||_2
                        （到出生点的 2D 欧氏距离，episode 内最远值）
            L_forward_score = 地形块 X 方向长度的一半（= 半块长度）
        时间分数 = max(0, 1 - traverse_step / max_steps) × 100（仅在首次走穿后生效）
                 走穿条件：d_forward >= L_traversal，L_traversal = L_terrain/2 - 0.1
                 未走穿时 time_score = 0
        episode 计数口径（BUGFIX: EVOL-20260424-001）:
            standard 模式严格按「走穿 = 成功，其他都算失败」的业务语义，
            timeout 作为 abnormal 的子类单独可观测（不互斥于 abnormal）：
              completed_count → self._traversed=True（走穿整块地形）
              abnormal_count  → self._traversed=False（其他一切，含 timeout/摔倒/早退/越界）
              timeout_count   → abnormal 的子类，仅在 abnormal 发生且 is_timeout=True 时 +1
            换言之: abnormal_count ⊇ timeout_count，completed + abnormal = 总结束 episode 数。
  - "track" (默认):
        总分 = 完成系数 × (时间分数 × 0.4 + 姿态分数 × 0.4 + 能耗分数 × 0.2)
        完成系数: 任务完成 → 1；任务失败或超时 → 0
        episode 计数口径（三分类互斥）:
            timeout   → is_timeout=True
            completed → goal_reached=True
            abnormal  → 其他

共用子分数:
- 时间分数: max(0, 1 - actual_steps / max_steps) × 100
- 前进距离分数（standard）: min(d_forward / L_forward_score, 1.0) × 100
                         d_forward = ||pos - start||_2（2D 欧氏距离，episode 内最大值）
- 姿态分数: 100 × exp(-5 × 平均偏移量)，偏移量 = |roll| + |pitch|
- 能耗分数: 100 × exp(-0.01 × 平均能耗)，能耗 = sum(|joint_vel| × |applied_torque|)

难度级别（level）分组：
  额外输出 {metric}_{terrain_type}_l{level} 形式的指标，用于绘制难度曲线。
  level 值来自 terrain.terrain_levels，需调用方在 step() 前拍快照传入（见 on_step 参数说明）。

<!-- @changelog -->
| 版本   | 变更说明                                                      | 关联                                            |
| v1.2.0 | standard 模式 episode 计数改为二分类：completed=走穿，          | BUGFIX: EVOL-20260424-001                       |
|        | abnormal=其他；timeout 作为 abnormal 子类（不互斥）便于细分观测 | 关联记忆: INIT-010（项目强约束清单）             |
|        | 彻底修复监控面板「环境失败数量」在 standard 模式下恒为 0 的缺陷   | 上报链路: EnvMonitor + monitor_default.yaml     |
| v1.1.0 | 首次引入 standard 三分类（已被 v1.2.0 覆写为二分类 + 子类）      | BUGFIX: EVOL-20260424-001（同一 EVOL 内迭代）    |
| v1.0.x | standard 模式 forward_score 归一化长度修复（半块基准）          | EVOL-20260423-001                               |
<!-- /@changelog -->

@author unknown
"""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import TYPE_CHECKING, Callable


class BaseScorer:
    """环境指标统计器 —— 按地形类型和难度级别分组统计 episode 级别的指标。

    使用方式:
        1. 在环境创建后调用 ``__init__`` 传入 env_unwrapped
        2. 可选：传入 ``on_flush`` 回调，flush() 触发时将 metrics 推送给调用方
        3. 每步调用 ``on_step(env_unwrapped, dones)``
        4. 需要上报时调用 ``flush()``（触发回调并清空缓冲区）
        5. 也可独立调用 ``get_metrics()`` 和 ``reset_metrics()``
    """

    # track 模式总分权重
    TRACK_TIME_WEIGHT = 0.4
    TRACK_POSTURE_WEIGHT = 0.4
    TRACK_ENERGY_WEIGHT = 0.2

    # standard 模式总分权重
    STD_FORWARD_WEIGHT = 0.4
    STD_TIME_WEIGHT = 0.2
    STD_ENERGY_WEIGHT = 0.2
    STD_POSTURE_WEIGHT = 0.2

    # Isaac Lab reward_manager extras 前缀
    _EPISODE_REWARD_PREFIX = "Episode_Reward/"

    def __init__(
        self,
        env_unwrapped,
        max_episode_length: int,
        min_episode_length: int = 50,
        on_flush: Callable[[dict], None] | None = None,
        task_type: str = "track",
        is_eval: bool = False,
        logger=None,
    ):
        """初始化评分器。

        Args:
            env_unwrapped: Isaac Lab 底层环境实例（ManagerBasedRLEnv）。
            max_episode_length: 最大 episode 步数，用于判断超时和计算完成系数。
            min_episode_length: 最小 episode 步数阈值，低于此步数视为异常终止。
            on_flush: 可选的 flush 回调函数。当 flush() 被调用时，
                将 get_metrics() 的返回值作为参数传入。默认为 None（不回调）。
            task_type: 任务类型，"standard" 或 "track"（默认）。
                "standard" 模式下完成系数 = 走穿地形且非超时→1，否则→0，
                总分 = 完成系数 × (前进距离分数×0.4 + 速度分数×0.2 + 能耗分数×0.2 + 姿态分数×0.2)；
                "track" 模式下总分乘以完成系数（完成→1，失败/超时→0），
                总分 = 完成系数 × (时间分数×0.4 + 姿态分数×0.4 + 能耗分数×0.2)。
            is_eval: 是否为评估模式。评估模式下启用全生命周期累积器，
                避免定时 flush 清空数据导致评估结果为空。
                训练模式下不启用，避免内存无限增长。
            logger: 可选的 logger 实例（通常由 base_env 传入）。用于记录内部
                静默 except 的上下文。为 None 时回退到 print。
        """
        self._max_episode_length = max_episode_length
        self._min_episode_length = min_episode_length
        self._on_flush = on_flush
        self._task_type = task_type.lower() if task_type else "track"
        self._is_eval = is_eval
        self._logger = logger
        # 记录已经打印过 warning 的 key，避免 on_step 路径上每步刷屏
        self._warned_keys: set[str] = set()

        # ------------------------------------------------------------------
        # 从环境中动态获取地形类型信息
        # ------------------------------------------------------------------
        terrain = env_unwrapped.scene.terrain
        terrain_cfg = terrain.cfg
        terrain_gen_cfg = terrain_cfg.terrain_generator

        if terrain_gen_cfg is None:
            # 非 generator 地形（如 plane），只有一个 "flat" 类型，难度为 1 级
            self.terrain_names: list[str] = ["flat"]
            self.num_levels: int = 1
            num_envs = env_unwrapped.num_envs
            self._env_terrain_name_idx = torch.zeros(num_envs, dtype=torch.long, device=terrain.device)
        else:
            # 获取 sub_terrains 名称列表（字典顺序 = 配置顺序）
            self.terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
            # num_rows 对应难度级别数量
            self.num_levels = int(terrain_gen_cfg.num_rows)

            # 复现 _generate_curriculum_terrains 中的 sub_indices 逻辑
            # 将列索引（terrain_types）映射到 sub_terrains 名称索引
            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_gen_cfg.sub_terrains.values()])
            proportions = proportions / proportions.sum()
            num_cols = terrain_gen_cfg.num_cols

            # col_to_sub_idx: 每个列索引 -> sub_terrains 名称索引
            col_to_sub_idx = []
            cumsum = np.cumsum(proportions)
            for col_i in range(num_cols):
                sub_idx = int(np.min(np.where(col_i / num_cols + 0.001 < cumsum)[0]))
                col_to_sub_idx.append(sub_idx)
            col_to_sub_idx_tensor = torch.tensor(col_to_sub_idx, dtype=torch.long, device=terrain.device)

            # terrain.terrain_types: (num_envs,) 列索引 -> 映射到 sub_terrain 名称索引
            self._env_terrain_name_idx = col_to_sub_idx_tensor[terrain.terrain_types.long()]

        # ------------------------------------------------------------------
        # 为每种地形类型初始化累积缓冲区（地形汇总，向后兼容）
        # ------------------------------------------------------------------
        # 动态获取 reward term 名称列表
        self.reward_term_names: list[str] = self._discover_reward_terms(env_unwrapped)

        # 定时上报用缓冲区（会被 flush/reset_metrics 清空）
        self._per_terrain: dict[str, dict] = {}
        for name in self.terrain_names:
            self._per_terrain[name] = self._new_terrain_buffer()

        # 全生命周期累积器（仅评估模式启用，永不清零，供 get_lifetime_metrics 使用）
        # 训练模式下为 None，避免内存无限增长
        if self._is_eval:
            self._lifetime_per_terrain: dict[str, dict] | None = {}
            for name in self.terrain_names:
                self._lifetime_per_terrain[name] = self._new_terrain_buffer()
        else:
            self._lifetime_per_terrain = None

        # ------------------------------------------------------------------
        # 按 (terrain_type, level) 二维分组的累积缓冲区（用于难度曲线）
        # 仅统计 8 个核心指标（不含 reward 分项，避免指标爆炸）
        # ------------------------------------------------------------------
        # 定时上报用缓冲区（会被 flush/reset_metrics 清空）
        self._per_terrain_level: dict[str, dict[int, dict]] = {}
        for name in self.terrain_names:
            self._per_terrain_level[name] = {}
            for lv in range(self.num_levels):
                self._per_terrain_level[name][lv] = self._new_level_buffer()

        # 全生命周期累积器（仅评估模式启用，永不清零）
        if self._is_eval:
            self._lifetime_per_terrain_level: dict[str, dict[int, dict]] | None = {}
            for name in self.terrain_names:
                self._lifetime_per_terrain_level[name] = {}
                for lv in range(self.num_levels):
                    self._lifetime_per_terrain_level[name][lv] = self._new_level_buffer()
        else:
            self._lifetime_per_terrain_level = None

        # ------------------------------------------------------------------
        # 获取地形块尺寸，用于 standard 模式的前进距离分数和边界检测
        # ------------------------------------------------------------------
        self._terrain_length_x: float = 8.0  # 默认值
        self._terrain_length_y: float = 8.0  # 默认值
        self._has_terrain_bounds: bool = False  # 是否启用地形块边界检测

        # Track 模式特有：整条赛道 X 方向跨 track_length 个子地形块
        # Track mode: the full track spans track_length sub-terrain blocks along X
        self._is_track_mode: bool = False
        self._track_length: int = 1
        self._track_total_x: float = 8.0  # = size_x × track_length

        try:
            if terrain_gen_cfg is not None:
                self._terrain_length_x = float(terrain_gen_cfg.size[0])
                self._terrain_length_y = float(terrain_gen_cfg.size[1])
                self._has_terrain_bounds = True

                # 检测 track 模式：TrackTerrainGeneratorCfg 有 track_length 属性
                track_length = getattr(terrain_gen_cfg, "track_length", None)
                if track_length is not None and track_length > 0:
                    self._is_track_mode = True
                    self._track_length = int(track_length)
                    self._track_total_x = self._terrain_length_x * self._track_length
        except Exception:
            # 地形尺寸读取失败时回退到默认值 (8.0m)，记录 warning 便于排查
            self._log(
                "warning",
                "读取 terrain_generator 尺寸/track_length 失败，使用默认 size=(8,8)、非 track 模式。",
                exc=True,
            )

        # ------------------------------------------------------------------
        # Track 模式：按列索引（= 难度档）分组的累积缓冲区
        # 在 TrackTerrainGenerator 中，列索引 col ∈ [0, num_cols) 对应难度从低到高；
        # 一列内所有 row 共用同一个难度，因此 col 本身就是难度分组维度。
        # 难度中心值（用于展示）:
        #   difficulty_range[0] + (difficulty_range[1] - difficulty_range[0]) * (col + 0.5) / num_cols
        # 非 track 模式下这些字段保持为空，不影响既有逻辑。
        # ------------------------------------------------------------------
        self._num_cols: int = 0
        # index = col, value = 该列难度中心值（track 模式专用）
        self.track_col_difficulties: list[float] = []
        # 定时上报用按列缓冲区（会被 reset_metrics 清空）
        self._per_col: dict[int, dict] = {}
        # 全生命周期按列缓冲区（仅评估模式启用）
        self._lifetime_per_col: dict[int, dict] | None = None

        if self._is_track_mode and terrain_gen_cfg is not None:
            self._num_cols = int(terrain_gen_cfg.num_cols)
            try:
                diff_lo, diff_hi = terrain_gen_cfg.difficulty_range
                diff_lo = float(diff_lo)
                diff_hi = float(diff_hi)
            except Exception:
                diff_lo, diff_hi = 0.0, 1.0

            if self._num_cols > 0:
                self.track_col_difficulties = [
                    diff_lo + (diff_hi - diff_lo) * (c + 0.5) / self._num_cols for c in range(self._num_cols)
                ]
            for c in range(self._num_cols):
                self._per_col[c] = self._new_level_buffer()
            if self._is_eval:
                self._lifetime_per_col = {}
                for c in range(self._num_cols):
                    self._lifetime_per_col[c] = self._new_level_buffer()

        # ------------------------------------------------------------------
        # Per-env 运行时缓冲区 —— 每步累积姿态、能耗和前进距离数据
        # ------------------------------------------------------------------
        num_envs = env_unwrapped.num_envs
        device = self._env_terrain_name_idx.device

        # 姿态偏移累积：每步 |roll| + |pitch| 的累积和
        self._pose_accum = torch.zeros(num_envs, dtype=torch.float32, device=device)
        # 能耗累积：每步 sum(|joint_vel| * |applied_torque|) 的累积和
        self._energy_accum = torch.zeros(num_envs, dtype=torch.float32, device=device)
        # 每个 env 当前 episode 已累积的步数计数器
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        # 前进距离：episode 内到出生点的 2D 欧氏距离（||pos - start||_2）最大值
        # forward_accum: running max of ||current_xy - episode_start_xy||_2 within the episode.
        # 与走穿判定（self._traversed）共用此度量，统一语义。
        self._forward_accum = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # Track 模式专有：最大达到的赛道段索引追踪
        # 用于记录机器人在整个 episode 中到达的最大段索引（0-based）
        # Track mode: track the max segment index reached during episode
        # ------------------------------------------------------------------
        # 最大段索引（-1 表示未计算或非 track 模式，0-based 索引）
        self._max_segment_reached = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # 地形块边界检测缓冲区
        # 当机器人走出当前地形块的 X 或 Y 边界后，停止累积姿态/能耗/前进距离，
        # 避免高难度相邻地形块或不同类型地形块"污染"当前块的评分。
        # step_count 仍然继续增长，保证完成系数正确反映存活时间。
        # ------------------------------------------------------------------
        # 每个 env 当前 episode 的地形块原点 XY（episode 开始时从 env_origins 获取）
        self._env_origin_xy = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        # 标记 env_origin_xy 是否已初始化
        self._env_origin_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 标记是否已越界（一旦越界，本 episode 不再累积质量指标）
        self._out_of_bounds = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # Standard 模式「走穿」判定（方案 A）专用缓冲区
        # 在 episode 开始时快照机器人位置 (x0, y0) 和朝向单位向量 (cos(yaw0), sin(yaw0))；
        # episode 结束时用「上一帧位置快照」作为真实结束位置（规避 Isaac Lab auto-reset
        # 对 root_pos_w 的覆写），计算 (end - start) 在起点朝向上的投影作为 net_displacement。
        # ------------------------------------------------------------------
        # episode 起点 XY
        self._episode_start_xy = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        # episode 起点朝向单位向量 (cos(yaw0), sin(yaw0))
        self._episode_start_heading_xy = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        # 标记 episode 起点是否已初始化（新 episode 的第一帧自动初始化）
        self._episode_start_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 上一帧（pre-reset）的位置 XY 快照：用作 done 时的真实结束位置
        # Isaac Lab auto-reset 会在 step() 内部覆写 root_pos_w，因此 on_step 里读到的
        # 是重置后的新起点；为拿到结束位置，需要保留上一帧的快照。
        self._last_pos_xy = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        # 标记 last_pos_xy 是否有效（第一帧为 False，第二帧起为 True）
        self._last_pos_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # Standard 模式 time_score「首次走穿」追踪（方案 B）
        # 记录每个 env 首次满足走穿条件时的物理步数，后续 time_score 公式
        # 用此步数而非 episode 最终步数，避免「走穿后仍需跑满 20s → time_score=0」的困境。
        # ------------------------------------------------------------------
        # 每个 env 当前 episode 是否已首次走穿
        self._traversed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 每个 env 首次走穿时的物理步数（1-based，未走穿时为 0）
        self._traverse_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        # 纯粹的本 episode 物理步数计数器（每步无条件 +1，区别于受 in_bounds 影响的 _step_count）
        self._physics_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # Track 模式硬校验：必须存在 `goal_reached` termination term
        # ------------------------------------------------------------------
        # Track 模式下「任务完成」严格依赖 termination_manager 中的 goal_reached
        # term（见 velocity_env_cfg._goal_reached_termination）。缺失会导致
        # 完成系数无法正确判定，此时直接抛错，强制配置修正。
        if self._task_type == "track":
            try:
                term_mgr = env_unwrapped.termination_manager
                active_terms = getattr(term_mgr, "active_terms", [])
            except Exception as exc:
                raise RuntimeError(
                    "[BaseScorer] track 模式要求 env_unwrapped.termination_manager 可用，" "但访问失败，请检查 Isaac Lab 环境初始化。"
                ) from exc
            if "goal_reached" not in active_terms:
                raise RuntimeError(
                    "[BaseScorer] track 模式要求 termination_manager 注册 `goal_reached` term，"
                    "请检查 TerminationsCfg（参考 unitree_rl_lab .../velocity_env_cfg.py 中的 "
                    "`goal_reached = DoneTerm(func=_goal_reached_termination, ...)`）。"
                )

        # Track 运行期检查标志：scorer 构造时 env.goal_positions 通常尚未由
        # observation_process 懒初始化，因此不在此处强校验；改为首次遇到 done 时
        # 在 on_step 内检查一次（缺失或全零即 raise，防止得分恒 0 静默发生）。
        self._goal_positions_runtime_checked = False

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def on_step(
        self,
        env_unwrapped,
        dones: torch.Tensor,
        *,
        pre_step_episode_lengths: torch.Tensor | None = None,
        pre_step_terrain_levels: torch.Tensor | None = None,
    ) -> None:
        """每步调用，累积姿态/能耗数据，并在 episode 结束时统计分数。

        Args:
            env_unwrapped: Isaac Lab 底层环境实例。
            dones: 形状为 (num_envs,) 的终止标记 Tensor。
            pre_step_episode_lengths: 可选，env.step() 调用前的
                episode_length_buf 快照。Isaac Lab 的 auto-reset 会在
                step() 内部清零 episode_length_buf，因此在 step() 返回后
                直接读取 episode_length_buf 得到的是 0。如果提供了此参数，
                将使用 step 前的快照值 +1 作为真实 episode 步数。
            pre_step_terrain_levels: 可选，env.step() 调用前的
                terrain.terrain_levels 快照。Isaac Lab 在 step() 内部的
                curriculum_manager.compute() 中会更新 terrain_levels，
                因此 step() 返回后读取的是已更新为新 level 的值。
                需在 step() 前拍快照以获得 done 时的真实 level。
        """
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.bool, device=self._env_terrain_name_idx.device)
        else:
            dones = dones.bool()

        # ------------------------------------------------------------------
        # 每步累积：姿态偏移和能耗（对所有 env，无论是否 done）
        # ------------------------------------------------------------------
        self._accumulate_per_step(env_unwrapped)

        if not dones.any():
            return

        # 获取结束的 env id
        done_env_ids = torch.where(dones)[0]

        # ---------- 从 extras["log"] 中提取 episode reward sums ----------
        reward_sums: dict[str, float] = {}
        try:
            log_extras = getattr(env_unwrapped, "extras", {}).get("log", {})
            prefix = self._EPISODE_REWARD_PREFIX
            for key, val in log_extras.items():
                if key.startswith(prefix):
                    term_name = key[len(prefix) :]
                    reward_sums[term_name] = float(val) if not isinstance(val, (int, float)) else val
        except Exception:
            # reward sums 读取失败不影响主评分，只让 reward_{term}_{terrain} 指标缺失。
            self._log(
                "warning",
                "从 env.extras['log'] 提取 Episode_Reward/* 失败，reward 分项指标将缺失。",
                once_key="reward_sums_extract",
                exc=True,
            )

        # episode_length_buf 记录了每个 env 当前 episode 已运行的步数。
        # ⚠️ Isaac Lab 的 ManagerBasedRLEnv.step() 内部会先执行
        #    episode_length_buf += 1，再 auto-reset 将 done 的 env 清零为 0。
        #    因此在 step() 返回后直接读取 episode_length_buf 得到的是 0。
        # 如果调用方提供了 step 前的 episode_length_buf 快照，
        # 则 real_steps = pre_step_value + 1（加上 step 中 += 1 的那一步）。
        if pre_step_episode_lengths is not None:
            episode_lengths = pre_step_episode_lengths[done_env_ids] + 1
        else:
            # 回退：直接读取（可能为 0，仅在无法获取快照时使用）
            episode_lengths = env_unwrapped.episode_length_buf[done_env_ids]

        # 获取 termination 信息：区分 time_out 和 abnormal
        time_outs = self._get_time_outs(env_unwrapped, done_env_ids)

        # ---------- Track 模式运行期强校验：goal_positions 必须已被正确维护 ----------
        # 正常链路下 base_env.step() 会在 env.step() 之后主动调用
        # `ensure_goal_positions_ready` + `update_goal_positions`（见
        # observation_process 模块级函数），保证到这里 env.goal_positions 已就绪。
        # 若仍缺失/全零，说明 TerrainExitManager 初始化失败或 base_env.step 被
        # 第三方入口绕过，此时 goal_reached 将恒为 False、completion_coeff 恒为 0、
        # total_score 恒为 0，属于配置性/环境性错误，直接 raise 让用户立刻修复，
        # 不做静默降级（降级到 standard 会用错 _terrain_length_x / 越界判定，
        # 产出错数据反而掩盖问题）。
        # ---------- Track-mode runtime hard check ----------
        if self._task_type == "track" and not self._goal_positions_runtime_checked:
            goal_positions = getattr(env_unwrapped, "goal_positions", None)
            if goal_positions is None:
                raise RuntimeError(
                    "[BaseScorer] track 模式首个 done 帧检测到 env.goal_positions 缺失/None。"
                    "正常链路下应由 base_env.step 主动维护（见 observation_process 的"
                    " `ensure_goal_positions_ready` / `update_goal_positions`）；"
                    "请检查 TerrainExitManager 是否可正常初始化，或自定义入口是否绕过了 base_env.step。"
                )
            try:
                max_abs = goal_positions.abs().max().item()
            except Exception:
                self._log(
                    "warning",
                    "读取 env.goal_positions.abs().max() 失败，保守视为全零触发配置校验。",
                    once_key="goal_positions_abs_max_fail",
                    exc=True,
                )
                max_abs = 0.0
            if max_abs < 1e-6:
                raise RuntimeError(
                    "[BaseScorer] track 模式首个 done 帧检测到 env.goal_positions 全零，"
                    "表明 TerrainExitManager 未成功写入目标点。请检查 base_env.step 主动维护链路"
                    "（`ensure_goal_positions_ready` + `update_goal_positions`）及地形类型配置。"
                )
            self._goal_positions_runtime_checked = True

        # 获取 goal_reached 标记（track 模式判定「任务完成」的唯一依据）
        # Track 模式在 __init__ 中已校验 term 必定存在；其他模式若无此 term 则为全 False。
        goal_reached_flags = self._get_goal_reached(env_unwrapped, done_env_ids)

        # ---------- 计算前进距离（X 方向位移，单位: 米）----------
        forward_distances = self._get_forward_distances(env_unwrapped, done_env_ids)

        # ---------- 读取 done env 的累积数据 ----------
        done_pose_accum = self._pose_accum[done_env_ids]
        done_energy_accum = self._energy_accum[done_env_ids]
        done_step_count = self._step_count[done_env_ids]
        done_forward_accum = self._forward_accum[done_env_ids]

        # 获取这些 env 对应的地形名称索引
        terrain_name_indices = self._env_terrain_name_idx[done_env_ids]

        # 获取 done env 的难度级别（优先使用 step 前快照）
        # ⚠️ Isaac Lab step() 内部 curriculum_manager.compute() 会更新 terrain_levels，
        #    step() 返回后读到的是新 level，需使用 step 前快照
        if pre_step_terrain_levels is not None:
            done_levels = pre_step_terrain_levels[done_env_ids]
        else:
            # 回退：直接读取（课程学习开启时可能不准确）
            try:
                done_levels = env_unwrapped.scene.terrain.terrain_levels[done_env_ids]
            except Exception:
                # 未能读到 terrain_levels，按 terrain 维度分组的难度曲线指标将缺失
                self._log(
                    "warning",
                    "读取 env.scene.terrain.terrain_levels 失败，(terrain, level) 二维难度曲线指标将缺失。",
                    once_key="read_terrain_levels_fail",
                    exc=True,
                )
                done_levels = None

        # 获取 done env 的列索引 = Track 模式下的难度档索引
        # terrain_types 由地形生成阶段固定分配，curriculum 不会在 step 内修改，
        # 因此无需 step 前快照（与 terrain_levels 不同）。
        done_cols = None
        if self._is_track_mode and self._num_cols > 0:
            try:
                done_cols = env_unwrapped.scene.terrain.terrain_types[done_env_ids]
            except Exception:
                self._log(
                    "warning",
                    "读取 env.scene.terrain.terrain_types 失败，Track 按难度分组指标将缺失。",
                    once_key="read_terrain_types_fail",
                    exc=True,
                )
                done_cols = None

        # 逐个 env 处理
        for i in range(len(done_env_ids)):
            terrain_idx = terrain_name_indices[i].item()
            terrain_name = self.terrain_names[terrain_idx]
            buf = self._per_terrain[terrain_name]

            steps = episode_lengths[i].item()
            is_timeout = time_outs[i].item() if isinstance(time_outs, torch.Tensor) else time_outs[i]
            is_goal_reached = bool(goal_reached_flags[i].item())
            fwd_dist = forward_distances[i].item() if forward_distances is not None else 0.0

            # ---------- 完成系数 ----------
            # 方案 B：time_score 由 _accumulate_per_step 逐步判定首次走穿并记录步数，
            # 此处仅需 env_idx 索引到 _traversed / _traverse_step 即可，
            # 不再计算 episode 结束瞬间的 net_displacement。
            env_idx = int(done_env_ids[i].item())
            if self._task_type == "standard":
                # standard: 不给完成系数
                pass
            else:
                # track: 到达 goal（由 termination_manager 的 goal_reached term 触发）→1，
                # 失败/超时 → 0
                completion_coeff = 1.0 if is_goal_reached else 0.0

            # ---------- 计算各子分数 ----------
            accum_steps = max(done_step_count[i].item(), 1)
            mean_pose_deviation = done_pose_accum[i].item() / accum_steps
            pose_score = self._compute_posture_score(mean_pose_deviation)
            mean_energy = done_energy_accum[i].item() / accum_steps
            energy_score = self._compute_energy_score(mean_energy)

            # 前进距离分数（使用逐步累加的正向位移）
            done_forward = done_forward_accum[i].item()
            forward_score = self._compute_forward_score(done_forward, self._forward_score_length())

            # 时间分数（方案 B）：
            # - standard: 用首次走穿那一刻的物理步数 _traverse_step 计算，
            #   公式 = max(0, 1 - traverse_step / max_steps) × 100；
            #   若 episode 未曾走穿，time_score = 0。
            # - track: 使用 episode 结束时的步数（由 completion_coeff 在总分中控制）。
            if self._task_type == "standard":
                if bool(self._traversed[env_idx].item()):
                    traverse_steps = int(self._traverse_step[env_idx].item())
                    time_score = self._compute_time_score(traverse_steps, self._max_episode_length)
                else:
                    time_score = 0.0
            else:
                time_score = self._compute_time_score(steps, self._max_episode_length)

            # ---------- 总分 ----------
            if self._task_type == "standard":
                # standard: 前进距离分数×0.4 + 时间分数×0.2 + 能耗分数×0.2 + 姿态分数×0.2
                total_score = (
                    forward_score * self.STD_FORWARD_WEIGHT
                    + time_score * self.STD_TIME_WEIGHT
                    + energy_score * self.STD_ENERGY_WEIGHT
                    + pose_score * self.STD_POSTURE_WEIGHT
                )
            else:
                # track: 完成系数 × (时间分数×0.4 + 姿态分数×0.4 + 能耗分数×0.2)
                total_score = completion_coeff * (
                    time_score * self.TRACK_TIME_WEIGHT
                    + pose_score * self.TRACK_POSTURE_WEIGHT
                    + energy_score * self.TRACK_ENERGY_WEIGHT
                )

            # 兜底：确保所有分数为有限值，NaN/Inf 视为 0.0
            if not math.isfinite(total_score):
                total_score = 0.0
            if not math.isfinite(time_score):
                time_score = 0.0
            if not math.isfinite(forward_score):
                forward_score = 0.0
            if not math.isfinite(pose_score):
                pose_score = 0.0
            if not math.isfinite(energy_score):
                energy_score = 0.0

            # ---------- 写入地形汇总缓冲区（定时上报 + 全生命周期）----------
            # 构建写入目标列表：定时上报缓冲区 + 评估模式下的 lifetime 缓冲区
            terrain_targets = [buf]
            if self._lifetime_per_terrain is not None:
                terrain_targets.append(self._lifetime_per_terrain[terrain_name])

            for target_buf in terrain_targets:
                target_buf["episode_total_scores"].append(total_score)
                target_buf["episode_step_scores"].append(time_score)
                target_buf["episode_forward_scores"].append(forward_score)
                target_buf["episode_pose_scores"].append(pose_score)
                target_buf["episode_energy_scores"].append(energy_score)
                target_buf["episode_steps"].append(steps)
                target_buf["episode_forward_distances"].append(fwd_dist)
                # 添加 track 模式的最大段索引指标
                if self._is_track_mode:
                    max_seg = self._max_segment_reached[done_env_ids[i]].item()
                    target_buf["episode_max_segments_reached"].append(max(0, max_seg))  # 确保非负
                else:
                    target_buf["episode_max_segments_reached"].append(-1)  # 非 track 模式用 -1

                if self._task_type == "track":
                    # track: 仅 goal_reached=True 才算 completed，其他非 timeout 都归 abnormal
                    if is_timeout:
                        target_buf["timeout_count"] += 1
                    elif is_goal_reached:
                        target_buf["completed_count"] += 1
                    else:
                        target_buf["abnormal_count"] += 1
                else:
                    # standard: 二分类 + timeout 子类（BUGFIX: EVOL-20260424-001）
                    #   completed: 走穿地形（self._traversed=True）
                    #   abnormal : 其他一切（未走穿，含 timeout、摔倒、早退、越界等）
                    #   timeout  : abnormal 的子类，是否是跑满最大步数导致的失败，
                    #              不互斥于 abnormal，便于单独观测超时占比
                    # 修复前：只有 timeout / completed 二分，abnormal_count 恒为 0，
                    # 且走穿与否没有体现，导致监控面板上「环境失败数量」永远显示 0。
                    # 走穿信号复用类内 self._traversed（与 time_score 判据一致，
                    # 见 _accumulate_per_step 中「方案 B：首次走穿检测」）。
                    if bool(self._traversed[env_idx].item()):
                        target_buf["completed_count"] += 1
                    else:
                        target_buf["abnormal_count"] += 1
                        if is_timeout:
                            target_buf["timeout_count"] += 1

            # reward 分项仅写入定时上报缓冲区（lifetime 不需要 reward 分项）
            for term_name, val in reward_sums.items():
                reward_key = f"episode_reward_{term_name}"
                if reward_key not in buf:
                    buf[reward_key] = []
                buf[reward_key].append(val)

            # ---------- 写入 (terrain, level) 二维缓冲区（定时上报 + 全生命周期）----------
            if done_levels is not None:
                lv = int(done_levels[i].item())
                # 容错：level 超出预分配范围时动态扩展
                if lv not in self._per_terrain_level[terrain_name]:
                    self._per_terrain_level[terrain_name][lv] = self._new_level_buffer()

                level_targets = [self._per_terrain_level[terrain_name][lv]]
                if self._lifetime_per_terrain_level is not None:
                    if lv not in self._lifetime_per_terrain_level[terrain_name]:
                        self._lifetime_per_terrain_level[terrain_name][lv] = self._new_level_buffer()
                    level_targets.append(self._lifetime_per_terrain_level[terrain_name][lv])

                for lbuf in level_targets:
                    if self._task_type == "track":
                        if is_timeout:
                            lbuf["timeout_count"] += 1
                        elif is_goal_reached:
                            lbuf["completed_count"] += 1
                        else:
                            lbuf["abnormal_count"] += 1
                    else:
                        # standard: 二分类 + timeout 子类（BUGFIX: EVOL-20260424-001）
                        # 与地形汇总缓冲区保持同口径，complete = 走穿，abnormal = 其他，
                        # timeout 作为 abnormal 的子类可被单独观测（不互斥）。
                        if bool(self._traversed[env_idx].item()):
                            lbuf["completed_count"] += 1
                        else:
                            lbuf["abnormal_count"] += 1
                            if is_timeout:
                                lbuf["timeout_count"] += 1

                    lbuf["episode_total_scores"].append(total_score)
                    lbuf["episode_step_scores"].append(time_score)
                    lbuf["episode_forward_scores"].append(forward_score)
                    lbuf["episode_pose_scores"].append(pose_score)
                    lbuf["episode_energy_scores"].append(energy_score)
                    lbuf["episode_steps"].append(steps)

            # ---------- 写入 Track 按列（= 按难度档）缓冲区（定时上报 + 全生命周期）----------
            if done_cols is not None:
                col = int(done_cols[i].item())
                # 容错：col 超出预分配范围时动态扩展（理论上不应发生）
                if col not in self._per_col:
                    self._per_col[col] = self._new_level_buffer()

                col_targets = [self._per_col[col]]
                if self._lifetime_per_col is not None:
                    if col not in self._lifetime_per_col:
                        self._lifetime_per_col[col] = self._new_level_buffer()
                    col_targets.append(self._lifetime_per_col[col])

                for cbuf in col_targets:
                    # track 模式下本分支才会启用，口径与 track 分支一致
                    if is_timeout:
                        cbuf["timeout_count"] += 1
                    elif is_goal_reached:
                        cbuf["completed_count"] += 1
                    else:
                        cbuf["abnormal_count"] += 1

                    cbuf["episode_total_scores"].append(total_score)
                    cbuf["episode_step_scores"].append(time_score)
                    cbuf["episode_forward_scores"].append(forward_score)
                    cbuf["episode_pose_scores"].append(pose_score)
                    cbuf["episode_energy_scores"].append(energy_score)
                    cbuf["episode_steps"].append(steps)

        # ---------- 清零已结束 env 的累积缓冲区 ----------
        self._pose_accum[done_env_ids] = 0.0
        self._energy_accum[done_env_ids] = 0.0
        self._step_count[done_env_ids] = 0
        self._forward_accum[done_env_ids] = 0.0
        # 重置边界检测状态（新 episode 重新初始化 origin 和越界标记）
        self._out_of_bounds[done_env_ids] = False
        self._env_origin_valid[done_env_ids] = False
        # 重置最大段索引为 -1（下个 episode 重新计算）
        self._max_segment_reached[done_env_ids] = -1
        # 重置方案 A「走穿」判定的起点快照
        # 下一帧进入 _accumulate_per_step 时，auto-reset 后的 root_pos_w 会被
        # 作为新 episode 的起点重新拍摄（_last_pos_xy 已在本帧 _accumulate_per_step
        # 末尾被更新为 auto-reset 后的新起点，正好作为下个 episode 的「上一帧位置」起始）。
        self._episode_start_valid[done_env_ids] = False
        # 重置方案 B「首次走穿」追踪
        self._traversed[done_env_ids] = False
        self._traverse_step[done_env_ids] = 0
        self._physics_step_count[done_env_ids] = 0

    def get_metrics(self) -> dict[str, float]:
        """获取当前累积期间的所有监控指标。

        返回的指标包括:
        - 全局计数: completed_count, abnormal_count, timeout_count
        - 全局得分: total_score, step_score, forward_score, pose_score, energy_score
        - 全局步数: step_avg
        - 前进距离: forward_distance_avg (米)
        - 按地形分组: xxx_[terrain_type] 形式
        - 按难度分组: xxx_[terrain_type]_l[level] 形式（难度曲线）

        Returns:
            指标字典，key 为指标名，value 为指标值。
        """

        def _safe_mean(vals: list[float], default: float = 0.0) -> float:
            """安全求均值：过滤 NaN/Inf 后求均值，空列表或全 NaN 返回 default。"""
            if not vals:
                return default
            result = float(np.nanmean(vals))
            return result if math.isfinite(result) else default

        metrics: dict[str, float] = {}

        # ---- 全局聚合 ----
        total_completed = 0
        total_abnormal = 0
        total_timeout = 0
        all_total_scores: list[float] = []
        all_step_scores: list[float] = []
        all_forward_scores: list[float] = []
        all_pose_scores: list[float] = []
        all_energy_scores: list[float] = []
        all_steps: list[float] = []
        all_forward_distances: list[float] = []
        all_max_segments: list[float] = []

        # reward 分项全局聚合
        all_rewards: dict[str, list[float]] = {name: [] for name in self.reward_term_names}

        for name in self.terrain_names:
            buf = self._per_terrain[name]

            # 全局累加
            total_completed += buf["completed_count"]
            total_abnormal += buf["abnormal_count"]
            total_timeout += buf["timeout_count"]
            all_total_scores.extend(buf["episode_total_scores"])
            all_step_scores.extend(buf["episode_step_scores"])
            all_forward_scores.extend(buf["episode_forward_scores"])
            all_pose_scores.extend(buf["episode_pose_scores"])
            all_energy_scores.extend(buf["episode_energy_scores"])
            all_steps.extend(buf["episode_steps"])
            all_forward_distances.extend(buf["episode_forward_distances"])
            # 添加 track 模式的最大段索引（非负值）
            max_segs = [max(0, int(s)) for s in buf.get("episode_max_segments_reached", []) if s >= -1]
            all_max_segments.extend(max_segs)

            # ---- 按地形分组的指标 ----
            metrics[f"completed_count_{name}"] = buf["completed_count"]
            metrics[f"abnormal_count_{name}"] = buf["abnormal_count"]
            metrics[f"timeout_count_{name}"] = buf["timeout_count"]

            metrics[f"total_score_{name}"] = _safe_mean(buf["episode_total_scores"])
            metrics[f"step_score_{name}"] = _safe_mean(buf["episode_step_scores"])
            metrics[f"forward_score_{name}"] = _safe_mean(buf["episode_forward_scores"])
            metrics[f"pose_score_{name}"] = _safe_mean(buf["episode_pose_scores"])
            metrics[f"energy_score_{name}"] = _safe_mean(buf["episode_energy_scores"])
            metrics[f"step_{name}"] = round(_safe_mean(buf["episode_steps"]))
            metrics[f"forward_distance_{name}"] = _safe_mean(buf["episode_forward_distances"])
            # Track 模式专有：最大段索引（取均值）
            if self._is_track_mode:
                max_segs_vals = [max(0, int(s)) for s in buf.get("episode_max_segments_reached", []) if s >= -1]
                metrics[f"max_segment_reached_{name}"] = round(_safe_mean(max_segs_vals)) if max_segs_vals else -1

            # ---- 按地形分组的 reward 指标 ----
            for term_name in self.reward_term_names:
                reward_key = f"episode_reward_{term_name}"
                vals = buf.get(reward_key, [])
                metrics[f"reward_{term_name}_{name}"] = _safe_mean(vals)
                all_rewards[term_name].extend(vals)

            # ---- 按 (terrain, level) 二维分组的指标（难度曲线）----
            for lv, lbuf in self._per_terrain_level[name].items():
                sfx = f"{name}_l{lv}"
                metrics[f"completed_count_{sfx}"] = lbuf["completed_count"]
                metrics[f"abnormal_count_{sfx}"] = lbuf["abnormal_count"]
                metrics[f"timeout_count_{sfx}"] = lbuf["timeout_count"]
                metrics[f"total_score_{sfx}"] = _safe_mean(lbuf["episode_total_scores"])
                metrics[f"step_score_{sfx}"] = _safe_mean(lbuf["episode_step_scores"])
                metrics[f"forward_score_{sfx}"] = _safe_mean(lbuf["episode_forward_scores"])
                metrics[f"pose_score_{sfx}"] = _safe_mean(lbuf["episode_pose_scores"])
                metrics[f"energy_score_{sfx}"] = _safe_mean(lbuf["episode_energy_scores"])
                metrics[f"step_{sfx}"] = round(_safe_mean(lbuf["episode_steps"]))

        # ---- 全局指标 ----
        metrics["completed_count"] = total_completed
        metrics["abnormal_count"] = total_abnormal
        metrics["timeout_count"] = total_timeout

        metrics["total_score"] = _safe_mean(all_total_scores)
        metrics["step_score"] = _safe_mean(all_step_scores)
        metrics["forward_score"] = _safe_mean(all_forward_scores)
        metrics["pose_score"] = _safe_mean(all_pose_scores)
        metrics["energy_score"] = _safe_mean(all_energy_scores)
        metrics["step_avg"] = round(_safe_mean(all_steps))
        metrics["forward_distance_avg"] = _safe_mean(all_forward_distances)
        # 全局 track 模式最大段索引（平均值）
        if self._is_track_mode and all_max_segments:
            metrics["max_segment_reached_avg"] = round(_safe_mean(all_max_segments))
        else:
            metrics["max_segment_reached_avg"] = -1

        # ---- Track 按列（= 按难度档）分组指标 ----
        # key 格式：{metric}_track_l{col}，col ∈ [0, num_cols)
        for col, cbuf in self._per_col.items():
            sfx = f"track_l{col}"
            metrics[f"completed_count_{sfx}"] = cbuf["completed_count"]
            metrics[f"abnormal_count_{sfx}"] = cbuf["abnormal_count"]
            metrics[f"timeout_count_{sfx}"] = cbuf["timeout_count"]
            metrics[f"total_score_{sfx}"] = _safe_mean(cbuf["episode_total_scores"])
            metrics[f"step_score_{sfx}"] = _safe_mean(cbuf["episode_step_scores"])
            metrics[f"forward_score_{sfx}"] = _safe_mean(cbuf["episode_forward_scores"])
            metrics[f"pose_score_{sfx}"] = _safe_mean(cbuf["episode_pose_scores"])
            metrics[f"energy_score_{sfx}"] = _safe_mean(cbuf["episode_energy_scores"])
            metrics[f"step_{sfx}"] = round(_safe_mean(cbuf["episode_steps"]))

        # ---- 全局 reward 指标 ----
        for term_name in self.reward_term_names:
            vals = all_rewards[term_name]
            metrics[f"reward_{term_name}"] = _safe_mean(vals)

        return metrics

    def get_lifetime_metrics(self) -> dict[str, float]:
        """获取全生命周期累积的指标（永不被 flush 清空）。

        与 get_metrics() 格式完全一致，但数据来自 _lifetime_per_terrain
        和 _lifetime_per_terrain_level 缓冲区，不受定时 flush/reset 影响。
        专门用于评估结束时的 _build_end_info。

        训练模式下 lifetime 缓冲区未启用，回退到 get_metrics()。
        """
        if self._lifetime_per_terrain is None:
            return self.get_metrics()

        def _safe_mean(vals: list[float], default: float = 0.0) -> float:
            if not vals:
                return default
            result = float(np.nanmean(vals))
            return result if math.isfinite(result) else default

        metrics: dict[str, float] = {}

        total_completed = 0
        total_abnormal = 0
        total_timeout = 0
        all_total_scores: list[float] = []
        all_step_scores: list[float] = []
        all_forward_scores: list[float] = []
        all_pose_scores: list[float] = []
        all_energy_scores: list[float] = []
        all_steps: list[float] = []
        all_forward_distances: list[float] = []
        all_max_segments: list[float] = []

        for name in self.terrain_names:
            buf = self._lifetime_per_terrain[name]

            total_completed += buf["completed_count"]
            total_abnormal += buf["abnormal_count"]
            total_timeout += buf["timeout_count"]
            all_total_scores.extend(buf["episode_total_scores"])
            all_step_scores.extend(buf["episode_step_scores"])
            all_forward_scores.extend(buf["episode_forward_scores"])
            all_pose_scores.extend(buf["episode_pose_scores"])
            all_energy_scores.extend(buf["episode_energy_scores"])
            all_steps.extend(buf["episode_steps"])
            all_forward_distances.extend(buf["episode_forward_distances"])
            # 添加 track 模式的最大段索引
            max_segs = [max(0, int(s)) for s in buf.get("episode_max_segments_reached", []) if s >= -1]
            all_max_segments.extend(max_segs)

            metrics[f"completed_count_{name}"] = buf["completed_count"]
            metrics[f"abnormal_count_{name}"] = buf["abnormal_count"]
            metrics[f"timeout_count_{name}"] = buf["timeout_count"]

            metrics[f"total_score_{name}"] = _safe_mean(buf["episode_total_scores"])
            metrics[f"step_score_{name}"] = _safe_mean(buf["episode_step_scores"])
            metrics[f"forward_score_{name}"] = _safe_mean(buf["episode_forward_scores"])
            metrics[f"pose_score_{name}"] = _safe_mean(buf["episode_pose_scores"])
            metrics[f"energy_score_{name}"] = _safe_mean(buf["episode_energy_scores"])
            metrics[f"step_{name}"] = round(_safe_mean(buf["episode_steps"]))
            metrics[f"forward_distance_{name}"] = _safe_mean(buf["episode_forward_distances"])
            # Track 模式专有最大段索引
            if self._is_track_mode:
                max_segs_vals = [max(0, int(s)) for s in buf.get("episode_max_segments_reached", []) if s >= -1]
                metrics[f"max_segment_reached_{name}"] = round(_safe_mean(max_segs_vals)) if max_segs_vals else -1

            for lv, lbuf in self._lifetime_per_terrain_level[name].items():
                sfx = f"{name}_l{lv}"
                metrics[f"completed_count_{sfx}"] = lbuf["completed_count"]
                metrics[f"abnormal_count_{sfx}"] = lbuf["abnormal_count"]
                metrics[f"timeout_count_{sfx}"] = lbuf["timeout_count"]
                metrics[f"total_score_{sfx}"] = _safe_mean(lbuf["episode_total_scores"])
                metrics[f"step_score_{sfx}"] = _safe_mean(lbuf["episode_step_scores"])
                metrics[f"forward_score_{sfx}"] = _safe_mean(lbuf["episode_forward_scores"])
                metrics[f"pose_score_{sfx}"] = _safe_mean(lbuf["episode_pose_scores"])
                metrics[f"energy_score_{sfx}"] = _safe_mean(lbuf["episode_energy_scores"])
                metrics[f"step_{sfx}"] = round(_safe_mean(lbuf["episode_steps"]))

        metrics["completed_count"] = total_completed
        metrics["abnormal_count"] = total_abnormal
        metrics["timeout_count"] = total_timeout

        metrics["total_score"] = _safe_mean(all_total_scores)
        metrics["step_score"] = _safe_mean(all_step_scores)
        metrics["forward_score"] = _safe_mean(all_forward_scores)
        metrics["pose_score"] = _safe_mean(all_pose_scores)
        metrics["energy_score"] = _safe_mean(all_energy_scores)
        metrics["step_avg"] = round(_safe_mean(all_steps))
        metrics["forward_distance_avg"] = _safe_mean(all_forward_distances)
        # 全局 track 模式最大段索引
        if self._is_track_mode and all_max_segments:
            metrics["max_segment_reached_avg"] = round(_safe_mean(all_max_segments))
        else:
            metrics["max_segment_reached_avg"] = -1

        # ---- Track 按列（= 按难度档）分组指标（全生命周期）----
        # key 格式：{metric}_track_l{col}，col ∈ [0, num_cols)
        if self._lifetime_per_col is not None:
            for col, cbuf in self._lifetime_per_col.items():
                sfx = f"track_l{col}"
                metrics[f"completed_count_{sfx}"] = cbuf["completed_count"]
                metrics[f"abnormal_count_{sfx}"] = cbuf["abnormal_count"]
                metrics[f"timeout_count_{sfx}"] = cbuf["timeout_count"]
                metrics[f"total_score_{sfx}"] = _safe_mean(cbuf["episode_total_scores"])
                metrics[f"step_score_{sfx}"] = _safe_mean(cbuf["episode_step_scores"])
                metrics[f"forward_score_{sfx}"] = _safe_mean(cbuf["episode_forward_scores"])
                metrics[f"pose_score_{sfx}"] = _safe_mean(cbuf["episode_pose_scores"])
                metrics[f"energy_score_{sfx}"] = _safe_mean(cbuf["episode_energy_scores"])
                metrics[f"step_{sfx}"] = round(_safe_mean(cbuf["episode_steps"]))

        return metrics

    def reset_metrics(self) -> None:
        """清空所有累积缓冲区（上报后调用）。"""
        for name in self.terrain_names:
            self._per_terrain[name] = self._new_terrain_buffer()
            for lv in list(self._per_terrain_level[name].keys()):
                self._per_terrain_level[name][lv] = self._new_level_buffer()
        # 清空 Track 按列缓冲区（定时上报用）
        for col in list(self._per_col.keys()):
            self._per_col[col] = self._new_level_buffer()

    def flush(self) -> None:
        """获取指标，触发 on_flush 回调，然后清空缓冲区。"""
        metrics = self.get_metrics()
        if self._on_flush is not None:
            self._on_flush(metrics)
        self.reset_metrics()

    # ------------------------------------------------------------------
    # 日志辅助
    # ------------------------------------------------------------------

    def _log(self, level: str, msg: str, once_key: str | None = None, exc: bool = False) -> None:
        """内部日志辅助：优先使用构造时传入的 logger，回退 print。

        Args:
            level: "debug" / "info" / "warning" / "error" 之一。
            msg: 日志正文。
            once_key: 若提供，则该 key 只输出一次（防止每步循环里刷屏）。
            exc: 若 True，附带当前异常堆栈（仅在 except 分支调用时有意义）。
        """
        if once_key is not None:
            if once_key in self._warned_keys:
                return
            self._warned_keys.add(once_key)

        if exc:
            import traceback

            msg = msg + "\n" + traceback.format_exc()

        if self._logger is not None:
            try:
                log_fn = getattr(self._logger, level, None)
                if callable(log_fn):
                    log_fn(msg)
                    return
            except Exception:
                pass
        # Fallback
        prefix = f"[BaseScorer][{level.upper()}] "
        print(prefix + msg)

    # ------------------------------------------------------------------
    # 子分数计算
    # ------------------------------------------------------------------

    def _traversal_threshold(self) -> float:
        """Standard mode 「走穿」threshold (2D Euclidean distance from spawn).

        Standard 模式「走穿」阈值（相对出生点的 2D 欧氏距离）。

        Unified with `_forward_score_length`: 走穿 ≈ forward_score 接近满分时成立。
        与 `_forward_score_length` 统一语义：走穿 ≈ forward_score 接近满分时成立。

        Default: L_terrain / 2 - 0.1 (slightly below the half-block boundary so that
        走穿判定略早于 forward_score 达满分，保留兜底空间).
        默认：L_terrain / 2 - 0.1（略小于半块边界，让走穿判定略早于 forward_score 满分）。
        """
        return self._terrain_length_x / 2.0 - 0.1

    def _forward_score_length(self) -> float:
        """返回前进距离分数的归一化长度。

        Standard 模式下默认出生点位于子地形中心，前向可用距离是半个地形块；
        因此前进距离分数应按半块长度归一化，而不是整块长度。
        """
        if self._task_type == "standard":
            return self._terrain_length_x / 2.0
        return self._terrain_length_x

    @staticmethod
    def _compute_time_score(actual_steps: int, max_steps: int) -> float:
        """计算时间分数。

        公式: max(0, 1 - actual_steps / max_steps) × 100
        """
        if max_steps <= 0:
            return 0.0
        return max(0.0, 1.0 - actual_steps / max_steps) * 100.0

    @staticmethod
    def _compute_posture_score(mean_deviation: float) -> float:
        """计算姿态分数。公式: 100 × exp(-5 × mean_deviation)"""
        if not math.isfinite(mean_deviation):
            return 0.0
        return 100.0 * math.exp(-5.0 * mean_deviation)
        """计算姿态分数。公式: 100 × exp(-5 × mean_deviation)"""
        if not math.isfinite(mean_deviation):
            return 0.0
        return 100.0 * math.exp(-5.0 * mean_deviation)

    @staticmethod
    def _compute_energy_score(mean_energy: float) -> float:
        """计算能耗分数。公式: 100 × exp(-0.01 × mean_energy)"""
        if not math.isfinite(mean_energy):
            return 0.0
        return 100.0 * math.exp(-0.01 * mean_energy)

    @staticmethod
    def _compute_forward_score(d_forward: float, l_terrain: float) -> float:
        """计算前进距离分数（仅 standard 模式使用）。

        公式: min(d_forward / L_terrain, 1.0) × 100
        """
        if l_terrain <= 0:
            return 0.0
        if not math.isfinite(d_forward):
            return 0.0
        return min(d_forward / l_terrain, 1.0) * 100.0

    # ------------------------------------------------------------------
    # 每步累积辅助
    # ------------------------------------------------------------------

    def _accumulate_per_step(self, env_unwrapped) -> None:
        """每步调用：累积所有 env 的姿态偏移、能耗和前进距离到 per-env 缓冲区。

        当启用地形块边界检测时，机器人走出当前地形块 X 或 Y 边界后，
        停止累积姿态/能耗/前进距离（避免相邻地形块污染评分），
        但 step_count 仍继续增长以保证完成系数正确。
        """
        try:
            robot = env_unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w  # (num_envs, 3)
            root_quat = robot.data.root_quat_w  # (num_envs, 4) WXYZ

            # --- 初始化 episode 起点 XY 和起点朝向（每个 episode 首步）---
            # 方案 A「走穿」判定所需：快照 (x0, y0) 与朝向单位向量 (cos(yaw0), sin(yaw0))。
            # auto-reset 后本帧的 root_pos_w 已是新起点，正好可以在此处拍摄。
            start_uninit = ~self._episode_start_valid
            if start_uninit.any():
                self._episode_start_xy[start_uninit] = root_pos[start_uninit, :2]
                yaw0 = self._quat_to_yaw(root_quat[start_uninit])
                self._episode_start_heading_xy[start_uninit, 0] = torch.cos(yaw0)
                self._episode_start_heading_xy[start_uninit, 1] = torch.sin(yaw0)
                self._episode_start_valid[start_uninit] = True

            # --- 初始化 env_origin（每个 episode 首步）---
            if self._has_terrain_bounds:
                uninit_mask = ~self._env_origin_valid
                if uninit_mask.any():
                    env_origins = env_unwrapped.scene.env_origins  # (num_envs, 3)
                    if self._is_track_mode:
                        # Track 模式：X 锚点 = 整条赛道中心 (= 0, 居中后)
                        #           Y 锚点 = env_origins[:, 1]（当前并行赛道中心）
                        # Track mode: X anchored at full-track center (0);
                        #             Y anchored at the per-column sub-terrain center
                        self._env_origin_xy[uninit_mask, 0] = 0.0
                        self._env_origin_xy[uninit_mask, 1] = env_origins[uninit_mask, 1]
                    else:
                        # Standard mode: anchor at robot's actual spawn pose,
                        # not the terrain block geometric center (env_origins).
                        # Standard 模式：锚点 = 机器人实际 spawn 位置，而非地形块几何中心。
                        #
                        # Rationale (改动背景):
                        # 1) reset_base event 会在 env_origins 基础上加随机 xy/yaw 扰动；
                        #    inverted pyramid / maze 等非对称地形的几何中心也未必等于
                        #    env_origins。两者叠加可能让机器人在 episode 首帧就被判
                        #    |offset| > L_terrain/2 → _out_of_bounds=True，继而：
                        #      - video_writer 全程 skip_mask=True → 生成 0 帧空壳 mp4
                        #      - forward_accum 被 in_bounds_f 门控 → 永远为 0
                        #      - _traversed 无法触发 → time_score 恒为 0
                        # 2) 改为 robot spawn pose 后，out_of_bounds 语义变成 “机器人
                        #    从自己实际起点走出半块长度” ，与 forward_distance =
                        #    ||pos - start||_2 / _traversed 的语义完全一致（三者共享
                        #    同一个起点参考系）。
                        self._env_origin_xy[uninit_mask] = root_pos[uninit_mask, :2]
                    self._env_origin_valid[uninit_mask] = True

                # --- X+Y 越界检测 ---
                # 一旦标记越界，本 episode 内不再翻转回 in-bounds
                if not self._out_of_bounds.all():
                    offset_xy = root_pos[:, :2] - self._env_origin_xy  # (num_envs, 2)
                    # Track 模式下 X 边界用整条赛道长度，Y 边界用单块宽度
                    # Track mode: X uses full-track span, Y uses single-block span
                    if self._is_track_mode:
                        half_x = self._track_total_x / 2.0
                    else:
                        half_x = self._terrain_length_x / 2.0
                    half_y = self._terrain_length_y / 2.0
                    newly_oob = (torch.abs(offset_xy[:, 0]) > half_x) | (torch.abs(offset_xy[:, 1]) > half_y)
                    self._out_of_bounds |= newly_oob

            # 确定哪些 env 在边界内（应该累积质量指标）
            in_bounds = ~self._out_of_bounds  # (num_envs,) bool
            in_bounds_f = in_bounds.float()

            # Track 模式：姿态/能耗/前进距离以及步数分母都应按 actual_steps 口径，
            # 不再受 in_bounds 阻断；仅 standard 模式保留 in-bounds 累积逻辑。
            # （_out_of_bounds 仍然维护，用于非评分场景如录屏 skip 等。）
            is_track_task = self._task_type == "track"
            if is_track_task:
                in_bounds_f = torch.ones_like(in_bounds_f)

            # --- 姿态偏移（仅对 in-bounds env 累积）---
            quat = robot.data.root_quat_w  # (num_envs, 4)
            roll, pitch = self._quat_to_roll_pitch(quat)
            pose_deviation = torch.abs(roll) + torch.abs(pitch)
            pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
            self._pose_accum += pose_deviation * in_bounds_f

            # --- 能耗（仅对 in-bounds env 累积）---
            joint_vel = robot.data.joint_vel
            applied_torque = robot.data.applied_torque
            energy_per_env = torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)
            energy_per_env = torch.nan_to_num(energy_per_env, nan=0.0, posinf=0.0, neginf=0.0)
            self._energy_accum += energy_per_env * in_bounds_f

            # --- Forward distance: 2D Euclidean distance from spawn ---
            # --- 前进距离：到出生点的 2D 欧氏距离 ---
            #
            # Unified semantics (统一语义):
            #   forward_distance = ||current_xy - episode_start_xy||_2
            # - Direction-agnostic: any direction of movement counts equally.
            #   不分方向：往任何方向走都等同计入距离。
            # - Endpoint-based (not accumulated): 瞬时端点差，取 max 保留 episode 内到达过的最远距离。
            #   基于端点差（非每步累加）：取 episode 内曾达到的最远点，避免机器人前进后回退导致分数缩水。
            # - Shared with `_traversed` check below: one metric, two usages.
            #   与下面的走穿判定共用：一个度量，两处使用。
            #
            # Only updated for `_episode_start_valid` envs; keeps in_bounds gating
            # consistent with other accumulators (out-of-bounds frames do not grow distance).
            # 仅对起点已初始化的 env 更新；与其他累加量一样受 in_bounds 门控。
            if self._episode_start_valid.any():
                delta_xy = root_pos[:, :2] - self._episode_start_xy  # (num_envs, 2)
                dist_from_start = torch.norm(delta_xy, dim=-1)  # (num_envs,)
                dist_from_start = torch.nan_to_num(dist_from_start, nan=0.0, posinf=0.0, neginf=0.0)
                # Gate by start-valid + in-bounds; keep running max across the episode.
                # 由起点有效 + 在界内双重门控，取 episode 内最大值作为最终 forward_accum。
                gate = self._episode_start_valid.float() * in_bounds_f
                gated_dist = dist_from_start * gate
                self._forward_accum = torch.maximum(self._forward_accum, gated_dist)

            # --- 步数计数 ---
            # Track 模式：无条件 +1（分母 = actual_steps，符合评分规范）
            # Standard 模式：仅对 in-bounds env 增长（用于计算块内平均值）
            # --- 更新 track 模式下的最大段索引 ---
            if self._is_track_mode:
                # 计算每个 env 的当前段索引（基于 X 位置）
                segment_indices = self._compute_segment_indices(root_pos[:, 0])
                # 更新最大段索引
                self._max_segment_reached = torch.maximum(self._max_segment_reached, segment_indices)

            if is_track_task:
                self._step_count += 1
            else:
                self._step_count += in_bounds.long()

            # --- 物理步数计数（所有 env 无条件 +1，不受 in_bounds 影响）---
            # 用作 standard 模式下「首次走穿步数」的时间基准，对应 episode_length_buf。
            self._physics_step_count += 1

            # --- Standard mode: detect first traversal (shares forward_accum) ---
            # --- Standard 模式：检测首次走穿（复用上面的 forward_accum）---
            #
            # Unified with `forward_distance`: traversed ⇔ 2D distance from spawn ≥ threshold.
            # 与 forward_distance 统一：走穿 ⇔ 到出生点的 2D 距离 ≥ 阈值。
            # 避免原来「投影越过阈值但世界 X 未前进」导致的 forward=0 time_score>0 矛盾。
            if self._task_type == "standard":
                not_yet = (~self._traversed) & self._episode_start_valid
                if not_yet.any():
                    threshold = self._traversal_threshold()
                    just_traversed = not_yet & (self._forward_accum >= threshold)
                    if just_traversed.any():
                        self._traverse_step[just_traversed] = self._physics_step_count[just_traversed]
                        self._traversed[just_traversed] = True

            # --- 更新「本帧位置」到 _last_pos_xy ---
            # 成为「下次 on_step 时的上一帧位置」。done 循环需要的是「上一帧」位置，
            # 已在 on_step 开头通过 prev_last_pos_xy 保存过副本，此处放心刷新。
            self._last_pos_xy[:] = root_pos[:, :2]
            self._last_pos_valid[:] = True
        except Exception:
            # 累积异常会掩盖真问题（如四元数解算 bug），首次失败通过 logger 输出完整堆栈一次，
            # 后续静默但仍保底递增 step_count，避免评估流程崩溃。
            self._log(
                "error",
                "_accumulate_per_step 抛出异常（同一进程仅报一次）；" "pose/energy/forward 累积本步将跳过，step_count 兜底 +1。",
                once_key="accumulate_per_step_exc",
                exc=True,
            )
            self._step_count += 1

    def _compute_segment_indices(self, pos_x: torch.Tensor) -> torch.Tensor:
        """计算每个 env 的当前赛道段索引（0-based）。

        在 track 模式下，根据机器人的 X 坐标计算其所在段的索引。
        段的 X 范围由 _terrain_length_x 定义，赛道从 -track_total_x/2 到 +track_total_x/2。

        Args:
            pos_x: (num_envs,) 机器人的 X 坐标

        Returns:
            (num_envs,) 段索引张量，范围 [0, track_length-1]
        """
        if not self._is_track_mode or self._track_length <= 1:
            return torch.full_like(pos_x, -1, dtype=torch.long)

        # 计算相对于赛道起点的相对位置
        # 赛道起点 X = -track_total_x / 2
        track_start_x = -self._track_total_x / 2.0
        relative_x = pos_x - track_start_x

        # 计算段索引
        segment_idx = (relative_x / self._terrain_length_x).long()
        # 限制在有效范围内 [0, track_length-1]
        segment_idx = torch.clamp(segment_idx, 0, self._track_length - 1)

        return segment_idx

    @staticmethod
    def _quat_to_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """从 WXYZ 格式四元数中提取 roll 和 pitch 角。"""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        return roll, pitch

    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
        """从 WXYZ 格式四元数中提取 yaw 角（世界系偏航角，单位 rad）。

        用于方案 A「走穿」判定：在 episode 起点快照 yaw0，
        后续位移向量在 (cos(yaw0), sin(yaw0)) 方向上的投影即为「沿起点前方」推进的距离。
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _discover_reward_terms(self, env_unwrapped) -> list[str]:
        """从 Isaac Lab 环境的 reward_manager 动态获取 reward term 名称列表。"""
        try:
            reward_mgr = env_unwrapped.reward_manager
            if hasattr(reward_mgr, "_episode_sums"):
                return list(reward_mgr._episode_sums.keys())
        except Exception:
            # 没有 reward_manager 或结构变更：reward 分项指标全部缺失
            self._log(
                "warning",
                "env_unwrapped.reward_manager._episode_sums 不可用，reward 分项指标将为空。",
                once_key="discover_reward_terms_fail",
                exc=True,
            )
            return []
        # reward_manager 存在但无 _episode_sums 属性
        self._log(
            "warning",
            "reward_manager 缺少 _episode_sums 属性，reward 分项指标将为空。",
            once_key="discover_reward_terms_no_episode_sums",
        )
        return []

    @staticmethod
    def _new_terrain_buffer() -> dict:
        """创建一个空的地形汇总统计缓冲区。"""
        return {
            "completed_count": 0,
            "abnormal_count": 0,
            "timeout_count": 0,
            "episode_total_scores": [],
            "episode_step_scores": [],
            "episode_forward_scores": [],
            "episode_pose_scores": [],
            "episode_energy_scores": [],
            "episode_steps": [],
            "episode_forward_distances": [],
            "episode_max_segments_reached": [],
        }

    @staticmethod
    def _new_level_buffer() -> dict:
        """创建一个空的 (terrain, level) 二维统计缓冲区。"""
        return {
            "completed_count": 0,
            "abnormal_count": 0,
            "timeout_count": 0,
            "episode_total_scores": [],
            "episode_step_scores": [],
            "episode_forward_scores": [],
            "episode_pose_scores": [],
            "episode_energy_scores": [],
            "episode_steps": [],
            "episode_max_segments_reached": [],
        }

    def _get_forward_distances(self, env_unwrapped, done_env_ids: torch.Tensor) -> torch.Tensor | None:
        """计算已结束 episode 中机器人在 X 方向的前进距离（米）。"""
        try:
            robot = env_unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w[done_env_ids]
            env_origins = env_unwrapped.scene.env_origins[done_env_ids]
            return root_pos[:, 0] - env_origins[:, 0]
        except Exception:
            self._log(
                "warning",
                "读取 robot.data.root_pos_w / scene.env_origins 失败，forward_distance 指标本次为 None。",
                once_key="get_forward_distances_fail",
                exc=True,
            )
            return None

    def _get_goal_reached(self, env_unwrapped, done_env_ids: torch.Tensor) -> torch.Tensor:
        """获取 done_env_ids 对应 env 的 `goal_reached` 终止标记。

        Track 模式用此判定「任务完成」。若 env 未注册 goal_reached term（例如
        standard/plane 模式或 env 配置缺失），返回与 done_env_ids 同形的全 False。
        """
        device = done_env_ids.device if isinstance(done_env_ids, torch.Tensor) else None
        try:
            term_mgr = env_unwrapped.termination_manager
            if "goal_reached" in getattr(term_mgr, "active_terms", []):
                term_flags = term_mgr.get_term("goal_reached")
                return term_flags[done_env_ids].bool()
        except Exception:
            # Track 模式下这意味着完成判定失效 → 影响总分；warn 一次便于排查
            self._log(
                "warning",
                "termination_manager.get_term('goal_reached') 读取失败，完成判定本帧退化为全 False。",
                once_key="get_goal_reached_fail",
                exc=True,
            )
        return torch.zeros(len(done_env_ids), dtype=torch.bool, device=device)

    @staticmethod
    def _get_time_outs(env_unwrapped, done_env_ids: torch.Tensor) -> torch.Tensor:
        """从环境中获取 time_out 标记。"""
        if hasattr(env_unwrapped, "termination_manager"):
            term_mgr = env_unwrapped.termination_manager
            if hasattr(term_mgr, "time_outs"):
                return term_mgr.time_outs[done_env_ids]

        episode_lengths = env_unwrapped.episode_length_buf[done_env_ids]
        return episode_lengths >= env_unwrapped.max_episode_length
