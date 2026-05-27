#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
ObservationBridge: Bridge between ObservationProcess and Isaac Lab's ObservationManager.
ObservationBridge: ObservationProcess 与 Isaac Lab ObservationManager 之间的桥接层。

默认策略是“覆盖指定 observation group”：
- 保留原 group 配置的深拷贝，作为默认观测计算来源
- 用单个自定义 ObservationTermCfg 替换目标 group
- `ObservationProcess.process()` 内可以继续调用 `self._default_compute_observations()`
  获取原始 group 的默认观测，然后做二次加工
"""

from __future__ import annotations

import copy


class ObservationBridge:
    """将 `ObservationProcess.process()` 适配为 Isaac Lab observation group 的桥接器。"""

    def __init__(
        self,
        observation_process,
        target_group: str = "policy",
        term_name: str = "custom_obs",
    ):
        self._observation_process = observation_process
        self._target_group = target_group
        self._term_name = term_name
        self._source_group_cfgs: dict[str, object] = {}
        self._source_managers: dict[str, tuple[int, object]] = {}

    def _compute_source_group(self, group_name: str, env):
        """计算被覆盖前的原始 observation group。"""
        from isaaclab.managers import ObservationManager

        cached = self._source_managers.get(group_name)
        env_id = id(env)
        if cached is None or cached[0] != env_id:
            if group_name not in self._source_group_cfgs:
                raise RuntimeError(f"Observation source group '{group_name}' has not been captured yet.")
            source_group_cfg = copy.deepcopy(self._source_group_cfgs[group_name])
            manager = ObservationManager({group_name: source_group_cfg}, env)
            self._source_managers[group_name] = (env_id, manager)
        else:
            manager = cached[1]

        return manager.compute_group(group_name)

    def _make_default_compute_callback(self, group_name: str, env):
        """延迟创建默认观测回调，只有在用户真的调用时才计算原 group。"""

        def default_compute():
            return self._compute_source_group(group_name, env)

        return default_compute

    def _make_wrapper(self, group_name: str):
        """生成 Isaac Lab 兼容的 observation term wrapper。"""
        bound_method = self._observation_process.process

        def wrapper(env):
            self._observation_process.env = env
            self._observation_process._default_compute_observations = self._make_default_compute_callback(
                group_name, env
            )
            return bound_method()

        wrapper.__name__ = f"observation_{group_name}"
        wrapper.__qualname__ = f"ObservationBridge.observation_{group_name}"
        wrapper.__doc__ = bound_method.__doc__
        return wrapper

    def build_group_cfg(self, group_name: str | None = None):
        """构建一个只包含单个自定义 term 的 ObservationGroupCfg 实例。"""
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
        from isaaclab.utils import configclass

        group_name = group_name or self._target_group
        wrapper_func = self._make_wrapper(group_name)
        term_name = self._term_name

        namespace = {
            term_name: ObservationTermCfg(func=wrapper_func),
            "__annotations__": {term_name: ObservationTermCfg},
        }
        group_cfg_cls = type(f"{group_name.title()}ObservationBridgeGroupCfg", (ObservationGroupCfg,), namespace)
        group_cfg_cls = configclass(group_cfg_cls)
        group_cfg = group_cfg_cls()

        # 目标是让 ObservationProcess 完全控制输出维度，因此这里关闭 group 级别的额外拼接/历史/噪声干预。
        group_cfg.concatenate_terms = True
        group_cfg.enable_corruption = False
        group_cfg.history_length = None
        group_cfg.flatten_history_dim = True

        return group_cfg

    def override_group_in_env_cfg(self, env_cfg):
        """用桥接后的单 term group 覆盖 env_cfg 中指定的 observation group。"""
        if not hasattr(env_cfg, "observations"):
            raise ValueError("env_cfg does not have an 'observations' attribute.")
        if not hasattr(env_cfg.observations, self._target_group):
            raise ValueError(f"Observation group '{self._target_group}' not found in env_cfg.observations.")

        self._source_group_cfgs[self._target_group] = copy.deepcopy(getattr(env_cfg.observations, self._target_group))
        self._source_managers.pop(self._target_group, None)
        setattr(env_cfg.observations, self._target_group, self.build_group_cfg(self._target_group))

    register_to_env_cfg = override_group_in_env_cfg

    def get_group_names(self) -> list[str]:
        """返回当前 bridge 管理的 group 名称。"""
        return [self._target_group]

    def __repr__(self) -> str:
        return "ObservationBridge(" f"target_group={self._target_group!r}, " f"term_name={self._term_name!r}" ")"
