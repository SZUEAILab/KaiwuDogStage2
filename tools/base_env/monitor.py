"""EnvMonitor —— 环境监控上报模块。

负责定时通过 Kaiwu 平台的 MonitorProxy 将指标上报到 Prometheus。
在构造时向 BaseScorer 注册 _handle_metrics 回调；BaseScorer.flush() 被触发时，
EnvMonitor 接收 metrics 并执行上报。

task_type 区分:
  - "standard": 上报日志中标注 task=standard，不包含完成系数相关描述
  - "track"   : 上报日志中标注 task=track（默认）

使用方式:
    scorer = BaseScorer(env_unwrapped, max_episode_length, task_type="standard")
    monitor = EnvMonitor(scorer, logger, task_type="standard")   # 自动注册回调
    # 每步调用（直接调用 scorer，不再通过 monitor）
    scorer.on_step(env_unwrapped, dones)
    # 关闭时调用
    monitor.close()
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from base_scorer import BaseScorer


class EnvMonitor:
    """环境监控上报器 —— 定时通过 MonitorProxy 将指标上报到 Prometheus。

    内部持有 BaseScorer 实例，将 _handle_metrics 注册为 scorer 的 flush 回调。
    外部调用方需自行调用 scorer.on_step()，monitor 只负责定时触发 scorer.flush()。
    """

    def __init__(
        self,
        scorer: BaseScorer,
        logger,
        flush_interval_sec: int = 60,
        task_type: str = "track",
    ):
        """初始化监控上报器。

        Args:
            scorer: BaseScorer 实例，负责统计指标。
            logger: 日志实例（KaiwuLogger 或 _SimpleLogger）。
            flush_interval_sec: 上报间隔（秒），默认 60 秒。
            task_type: 任务类型，"standard" 或 "track"（默认）。
                影响上报日志中的描述信息，standard 模式不显示完成系数相关字段。
        """
        self._scorer = scorer
        self._logger = logger
        self._task_type = task_type.lower() if task_type else "track"
        # 注册 _handle_metrics 为 scorer 的 flush 回调。
        # EnvMonitor 在 scorer 已构造后注入，无法通过构造函数传参，
        # 故直接设置 scorer._on_flush。如需更严格的解耦，
        # 可在 BaseScorer 上提供 register_flush_callback() 公共方法。
        scorer._on_flush = self._handle_metrics
        self._flush_interval_sec = flush_interval_sec
        self._last_flush_time = time.time()
        self._current_pid = str(os.getpid())

        # 尝试获取 Kaiwu MonitorProxy，不可用时优雅降级
        self._enabled = False
        self._monitor_proxy = None

        # 检查 use_prometheus 配置（eval 模式下应设置为 false）
        use_prometheus = True
        try:
            use_prometheus_env = os.environ.get("use_prometheus", "").lower()
            if use_prometheus_env in ("false", "0", "no"):
                use_prometheus = False
        except Exception:
            pass

        if not use_prometheus:
            self._logger.info("EnvMonitor: use_prometheus=false，监控上报已禁用（静默模式）")
            return

        try:
            from common_python.monitor.monitor_manager import get_monitor_proxy

            config_file = "kaiwudrl/conf/kaiwudrl/aisrv.toml"
            self._monitor_proxy = get_monitor_proxy(file_path=config_file, section="aisrv")
            self._enabled = True
            self._logger.info("EnvMonitor: MonitorProxy 已连接，监控上报已启用")
        except ImportError:
            self._logger.info("EnvMonitor: 未检测到 Kaiwu 平台环境，监控上报已禁用（静默模式）")
        except Exception as e:
            self._logger.warning(f"EnvMonitor: 获取 MonitorProxy 失败: {e}，监控上报已禁用")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def on_step(self, env_unwrapped, dones, *, pre_step_episode_lengths=None, pre_step_terrain_levels=None) -> None:
        """代理方法：将 on_step 转发给内部 scorer，并检查是否需要 flush。

        Args:
            env_unwrapped: 解包后的环境实例。
            dones: 各环境的 done 标志。
            pre_step_episode_lengths: 可选，env.step() 调用前的
                episode_length_buf 快照。Isaac Lab 的 auto-reset 会在
                step() 内部清零 episode_length_buf，需要在 step() 之前保存。
            pre_step_terrain_levels: 可选，env.step() 调用前的
                terrain.terrain_levels 快照。Isaac Lab 在 step() 内部的
                curriculum_manager 中会更新 terrain_levels，需在 step() 前保存。
        """
        self._scorer.on_step(
            env_unwrapped,
            dones,
            pre_step_episode_lengths=pre_step_episode_lengths,
            pre_step_terrain_levels=pre_step_terrain_levels,
        )
        self._try_flush()

    def close(self) -> None:
        """关闭监控器，执行最终一次 flush。"""
        self._logger.info("EnvMonitor: 正在关闭，执行最终上报...")
        self._scorer.flush()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _handle_metrics(self, metrics: dict) -> None:
        """回调：接收 scorer.flush() 推送的 metrics，执行上报。"""
        if not metrics:
            return

        def _sf(key: str, fmt: str = ".4f", default: float = 0.0) -> str:
            """安全格式化 metrics 值，NaN/Inf 替换为 default。"""
            v = metrics.get(key, default)
            if not math.isfinite(v):
                v = default
            return format(v, fmt)

        if self._enabled and self._monitor_proxy is not None:
            # 上报前清理 metrics 中的 NaN/Inf
            clean_metrics = {
                k: (v if math.isfinite(v) else 0.0) if isinstance(v, float) else v for k, v in metrics.items()
            }
            monitor_data = {self._current_pid: clean_metrics}
            try:
                success = self._monitor_proxy.put_data(monitor_data)
                if success:
                    reward_count = sum(1 for k in metrics if k.startswith("reward_"))
                    if self._task_type == "standard":
                        self._logger.info(
                            f"EnvMonitor [task=standard]: 已上报 {len(metrics)} 个指标 "
                            f"(completed={metrics.get('completed_count', 0)}, "
                            f"abnormal={metrics.get('abnormal_count', 0)}, "
                            f"timeout={metrics.get('timeout_count', 0)}, "
                            f"reward_metrics={reward_count})"
                        )
                    else:
                        self._logger.info(
                            f"EnvMonitor [task=track]: 已上报 {len(metrics)} 个指标 "
                            f"(completed={metrics.get('completed_count', 0)}, "
                            f"abnormal={metrics.get('abnormal_count', 0)}, "
                            f"timeout={metrics.get('timeout_count', 0)}, "
                            f"reward_metrics={reward_count})"
                        )
                else:
                    self._logger.warning("EnvMonitor: put_data 返回 False（队列已满或数据异常）")
            except Exception as e:
                self._logger.error(f"EnvMonitor: 上报失败: {e}")
        else:
            if self._task_type == "standard":
                self._logger.info(
                    f"EnvMonitor [静默][task=standard]: "
                    f"completed={metrics.get('completed_count', 0)}, "
                    f"abnormal={metrics.get('abnormal_count', 0)}, "
                    f"timeout={metrics.get('timeout_count', 0)}, "
                    f"total_score={_sf('total_score')}, "
                    f"step_avg={_sf('step_avg', '.1f')}"
                )
            else:
                self._logger.info(
                    f"EnvMonitor [静默][task=track]: "
                    f"completed={metrics.get('completed_count', 0)}, "
                    f"abnormal={metrics.get('abnormal_count', 0)}, "
                    f"timeout={metrics.get('timeout_count', 0)}, "
                    f"total_score={_sf('total_score')}, "
                    f"step_avg={_sf('step_avg', '.1f')}"
                )

    def _try_flush(self) -> None:
        """检查是否到达上报时间间隔，满足则调用 scorer.flush() 触发回调上报。"""
        now = time.time()
        if now - self._last_flush_time >= self._flush_interval_sec:
            self._scorer.flush()
            self._last_flush_time = now
