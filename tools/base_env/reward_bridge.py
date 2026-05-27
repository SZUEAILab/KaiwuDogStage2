#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

RewardBridge: Bridge between RewardProcess and Isaac Lab's RewardManager.
RewardBridge: RewardProcess 与 Isaac Lab RewardManager 之间的桥接层。

This module provides:
    `RewardBridge` class - build Isaac Lab RewardsCfg from TOML configs + registry.

Two construction paths / 两种构建路径:
    1. Legacy: `RewardBridge(reward_process, reward_weights=...)` — scan `_reward_*` methods
    2. Config-driven: `RewardBridge.from_configs(reward_process, reward_configs)` — full TOML control

Custom method parameter forwarding / 自定义方法参数透传:
    Custom `_reward_*` methods can optionally accept TOML params via their
    method signature. The bridge inspects the signature and forwards params
    accordingly:
    自定义 `_reward_*` 方法可以通过方法签名选择性地接收 TOML 参数。
    桥接器会检查签名并相应透传参数：

    - `_reward_xxx(self)`:            No params forwarded (backward compatible)
    - `_reward_xxx(self, std=0.5)`:   TOML params forwarded as keyword args
    - `_reward_xxx(self, **kwargs)`:  All TOML params forwarded

    Required params (no default) are validated at build time. If a required
    param is missing from TOML, a clear error is raised immediately.
    必需参数（无默认值）在构建时校验。如果 TOML 中缺少必需参数，
    会立即抛出明确的错误。

Timing / 时序说明:
    Isaac Lab 要求 env_cfg.rewards 必须在 gym.make() 之前配好，但 RewardProcess
    中的方法需要访问 env 的属性（如 contact_forces）。解决方案：延迟绑定。
    RewardProcess 可以先不传 env 构造，wrapper 闭包会在每次被 RewardManager 调用时
    自动将 Isaac Lab 传入的 env 同步给 RewardProcess.env。

Usage / 使用方式:
    # Config-driven (recommended):
    reward_configs = parse_reward_configs(usr_conf)
    reward_process = RewardProcess()
    bridge = RewardBridge.from_configs(reward_process, reward_configs)
    bridge.register_to_env_cfg(env_cfg)

    # Legacy:
    bridge = RewardBridge(reward_process, reward_weights=reward_scales)
    bridge.merge_to_env_cfg(env_cfg)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# RewardBridge / 奖励桥接器
# ============================================================================


class RewardBridge:
    """
    Bridge that adapts reward terms into Isaac Lab's RewardManager system.
    将奖励项适配到 Isaac Lab RewardManager 系统的桥接器。

    Supports two modes / 支持两种模式:

    1. **Legacy scan mode**: Discover `_reward_*` methods on RewardProcess
       and wrap them into RewardTermCfg.
       扫描模式：发现 RewardProcess 上的 `_reward_*` 方法并包装。

    2. **Config-driven mode** (via `from_configs`): Use TOML reward configs
       to build all terms from `_reward_*` methods on the RewardProcess.
       配置驱动模式（通过 `from_configs`）：使用 TOML 配置从 RewardProcess
       上的 `_reward_*` 方法构建所有 term。

    Weight priority (high → low) / 权重优先级（高 → 低）:
        TOML config weight > reward_weights dict > default (1.0)

    Lazy binding / 延迟绑定:
        RewardProcess 可以在没有 env 的情况下创建（env=None）。
        wrapper 闭包在每次被 RewardManager.compute() 调用时，会自动将
        Isaac Lab 传入的真实 env 同步给 RewardProcess.env。
    """

    # Prefix used to auto-discover reward methods / 自动发现奖励方法的前缀
    REWARD_PREFIX = "_reward_"

    # Default weight when no other source specifies it / 默认权重
    DEFAULT_WEIGHT = 1.0

    def __init__(
        self,
        reward_process,
        reward_weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
    ):
        """
        Initialize the bridge (legacy scan mode).
        初始化桥接器（遗留扫描模式）。

        Args:
            reward_process: A RewardProcess instance whose `_reward_*` methods
                            will be adapted for Isaac Lab.
            reward_weights: Centralized weight configuration dict.
            default_weight: Fallback weight when not specified elsewhere.
        """
        self._reward_process = reward_process
        self._reward_weights = reward_weights or {}
        self._default_weight = default_weight

        # Discovered reward terms:
        # For legacy: {term_name: {"method": bound_method, "weight": float, "params": dict}}
        # For config-driven: {term_name: {"func": callable, "weight": float, "params": dict}}
        self._terms: dict[str, dict[str, Any]] = {}

        # Track mode for correct wrapper generation
        self._mode = "legacy"

        # Auto-scan on init (legacy mode)
        self._scan_reward_methods()

    @classmethod
    def from_configs(cls, reward_process, reward_configs: dict) -> RewardBridge:
        """Create a config-driven RewardBridge from TOML-parsed reward configs.

        从 TOML 解析的 reward configs 创建配置驱动的 RewardBridge。

        For each term_name in reward_configs, look up `_reward_{term_name}`
        method on reward_process (including inherited base class methods).
        对于 reward_configs 中的每个 term_name，在 reward_process 上查找
        `_reward_{term_name}` 方法（包括继承的基类方法）。

        Args:
            reward_process: A RewardProcess instance.
            reward_configs: Dict from TOML `[rewards.*]` sections.
                Format: {term_name: {"weight": float, "params": dict}}

        Returns:
            A RewardBridge instance with all terms registered.
        """
        # Create instance without auto-scanning
        bridge = cls.__new__(cls)
        bridge._reward_process = reward_process
        bridge._reward_weights = {}
        bridge._default_weight = 1.0
        bridge._terms = {}
        bridge._mode = "config"

        for term_name, term_cfg in reward_configs.items():
            weight = term_cfg.get("weight", 1.0)
            toml_params = dict(term_cfg.get("params", {}))

            # Look up _reward_* method on reward_process (base or subclass)
            custom_method_name = f"{cls.REWARD_PREFIX}{term_name}"
            custom_method = getattr(reward_process, custom_method_name, None)

            if custom_method is not None and callable(custom_method):
                logger.info(f"Reward term '{term_name}': using _reward_{term_name}() (weight={weight})")
                bridge._terms[term_name] = {
                    "source": "custom",
                    "method": custom_method,
                    "weight": float(weight),
                    "params": toml_params,
                }
                continue

            # Not found
            available_methods = [
                attr[len(cls.REWARD_PREFIX) :]
                for attr in dir(reward_process)
                if attr.startswith(cls.REWARD_PREFIX) and callable(getattr(reward_process, attr))
            ]
            raise ValueError(
                f"Reward term '{term_name}' not found!\n"
                f"  - No _reward_{term_name}() method on {type(reward_process).__name__}\n"
                f"  Available methods: {available_methods}"
            )

        logger.info(f"RewardBridge.from_configs: {len(bridge._terms)} terms registered")

        return bridge

    # ------------------------------------------------------------------
    # Scanning (legacy mode) / 扫描（遗留模式）
    # ------------------------------------------------------------------

    def _scan_reward_methods(self):
        """
        Scan the RewardProcess instance for reward methods (legacy mode).
        扫描 RewardProcess 实例中的奖励方法（遗留模式）。
        """
        self._terms.clear()

        for attr_name in dir(self._reward_process):
            if attr_name.startswith("__"):
                continue

            method = getattr(self._reward_process, attr_name, None)
            if not callable(method):
                continue

            if not attr_name.startswith(self.REWARD_PREFIX):
                continue

            term_name = attr_name[len(self.REWARD_PREFIX) :]
            weight = float(self._reward_weights.get(term_name, self._default_weight))

            self._terms[term_name] = {
                "source": "custom",
                "method": method,
                "weight": weight,
                "params": {},
            }

    # ------------------------------------------------------------------
    # Wrapping / 包装
    # ------------------------------------------------------------------

    @staticmethod
    def _inspect_custom_method(method) -> tuple[list[str], list[str], bool]:
        """Inspect a custom _reward_* method's signature (excluding 'self').

        检查自定义 _reward_* 方法的签名（排除 'self'）。

        Returns:
            (all_param_names, required_param_names, has_var_keyword)
            - all_param_names: All parameter names the method accepts.
            - required_param_names: Parameters without default values.
            - has_var_keyword: Whether the method accepts **kwargs.
        """
        sig = inspect.signature(method)
        all_params: list[str] = []
        required_params: list[str] = []
        has_var_keyword = False

        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            all_params.append(name)
            if param.default is inspect.Parameter.empty:
                required_params.append(name)

        return all_params, required_params, has_var_keyword

    @staticmethod
    def _validate_custom_params(
        term_name: str,
        method,
        toml_params: dict,
        required_params: list[str],
    ):
        """Validate that TOML params cover all required method parameters.

        校验 TOML params 是否覆盖了方法的所有必需参数。
        在环境构建阶段提前报错，而非等到训练运行时才崩溃。

        Raises:
            ValueError: If required params are missing from TOML config.
        """
        missing = [p for p in required_params if p not in toml_params]
        if missing:
            method_name = f"_reward_{term_name}"
            provided = list(toml_params.keys()) if toml_params else []
            hint_lines = "\n".join(f"  {p} = <value>" for p in missing)
            raise ValueError(
                f"Custom reward method '{method_name}' requires params {required_params}, "
                f"but TOML [rewards.{term_name}.params] only provides {provided}.\n"
                f"  Missing: {missing}\n"
                f"  Hint: Add to TOML:\n"
                f"    [rewards.{term_name}.params]\n"
                f"{hint_lines}\n"
                f"  Or give the parameter a default value in the method signature."
            )

    def _make_wrapper(self, term_name: str):
        """
        Create an Isaac Lab-compatible wrapper function for a reward term.
        为奖励项创建 Isaac Lab 兼容的包装函数。

        Handles both custom methods (need env binding) and registry functions
        (already Isaac Lab compatible, just pass through).

        For custom methods, the wrapper behavior depends on the method signature:
        对于自定义方法，wrapper 行为取决于方法签名：

        - `_reward_xxx(self)`:         wrapper(env) — no params forwarded (backward compatible)
        - `_reward_xxx(self, std)`:    wrapper(env, std) — params forwarded from TOML
        - `_reward_xxx(self, **kw)`:   wrapper(env, **kw) — all TOML params forwarded

        Args:
            term_name: The reward term name.

        Returns:
            A function compatible with RewardTermCfg.func.
        """
        term_info = self._terms[term_name]

        # Custom _reward_* method — need wrapper with env binding
        bound_method = term_info["method"]
        toml_params = term_info["params"]
        param_keys = list(toml_params.keys())

        # Inspect the custom method's signature to decide forwarding strategy
        # 检查自定义方法签名以决定参数透传策略
        all_method_params, required_method_params, has_var_keyword = self._inspect_custom_method(bound_method)
        method_accepts_params = bool(all_method_params) or has_var_keyword

        if not method_accepts_params:
            # Method takes no params: _reward_xxx(self) — backward compatible
            # 方法无参数：_reward_xxx(self) — 向后兼容

            if not param_keys:

                def wrapper(env):
                    self._reward_process.env = env
                    return bound_method()

            else:
                # TOML has params but method doesn't accept them — generate
                # a wrapper that accepts (and discards) the params for Isaac Lab
                # TOML 有参数但方法不接受 — 生成接收（并丢弃）参数的 wrapper
                param_str = ", ".join(param_keys)
                func_code = (
                    f"def wrapper(env, {param_str}):\n"
                    f"    _self._reward_process.env = env\n"
                    f"    return _bound_method()\n"
                )
                local_ns: dict[str, Any] = {}
                exec(func_code, {"_self": self, "_bound_method": bound_method}, local_ns)  # noqa: S102
                wrapper = local_ns["wrapper"]

        else:
            # Method accepts params — validate and forward
            # 方法接受参数 — 校验并透传

            # Validate: all required params must be present in TOML or have defaults
            # 校验：所有必需参数必须在 TOML 中存在或有默认值
            self._validate_custom_params(term_name, bound_method, toml_params, required_method_params)

            if has_var_keyword:
                # Method has **kwargs — forward all TOML params as keyword args
                # 方法有 **kwargs — 将所有 TOML 参数作为关键字参数透传

                if not param_keys:

                    def wrapper(env):
                        self._reward_process.env = env
                        return bound_method()

                else:
                    param_str = ", ".join(param_keys)
                    forward_str = ", ".join(f"{k}={k}" for k in param_keys)
                    func_code = (
                        f"def wrapper(env, {param_str}):\n"
                        f"    _self._reward_process.env = env\n"
                        f"    return _bound_method({forward_str})\n"
                    )
                    local_ns: dict[str, Any] = {}
                    exec(func_code, {"_self": self, "_bound_method": bound_method}, local_ns)  # noqa: S102
                    wrapper = local_ns["wrapper"]

            else:
                # Method has explicit params — forward only matching ones
                # 方法有显式参数 — 只透传匹配的参数

                # Determine which TOML params to actually forward (intersection)
                # 确定实际要透传的 TOML 参数（取交集）
                forward_keys = [k for k in param_keys if k in all_method_params]
                # Also include method params with defaults that are NOT in TOML
                # (they will use their default values via Isaac Lab's param passing)
                extra_method_keys = [k for k in all_method_params if k not in param_keys]

                all_wrapper_keys = forward_keys + extra_method_keys
                if not all_wrapper_keys:

                    def wrapper(env):
                        self._reward_process.env = env
                        return bound_method()

                else:
                    # Build wrapper signature with all TOML param_keys (for Isaac Lab)
                    # and forward the matching ones to bound_method
                    # 构建包含所有 TOML param_keys 的 wrapper 签名（给 Isaac Lab）
                    # 并将匹配的参数透传给 bound_method
                    wrapper_param_str = ", ".join(param_keys) if param_keys else ""
                    forward_str = ", ".join(f"{k}={k}" for k in forward_keys)
                    func_code = (
                        f"def wrapper(env, {wrapper_param_str}):\n"
                        f"    _self._reward_process.env = env\n"
                        f"    return _bound_method({forward_str})\n"
                    )
                    local_ns: dict[str, Any] = {}
                    exec(func_code, {"_self": self, "_bound_method": bound_method}, local_ns)  # noqa: S102
                    wrapper = local_ns["wrapper"]

        wrapper.__name__ = f"reward_{term_name}"
        wrapper.__qualname__ = f"RewardBridge.reward_{term_name}"
        wrapper.__doc__ = getattr(bound_method, "__doc__", None)

        return wrapper

    # ------------------------------------------------------------------
    # Building RewardsCfg / 构建 RewardsCfg
    # ------------------------------------------------------------------

    def build_rewards_cfg_dict(self) -> dict:
        """
        Build a dict of {term_name: RewardTermCfg}.
        构建 {term_name: RewardTermCfg} 字典。
        """
        from isaaclab.managers import RewardTermCfg

        cfg_dict = {}
        for term_name, term_info in self._terms.items():
            wrapper_func = self._make_wrapper(term_name)
            cfg_dict[term_name] = RewardTermCfg(
                func=wrapper_func,
                weight=term_info["weight"],
                params=term_info["params"],
            )
        return cfg_dict

    def build_rewards_cfg(self):
        """
        Build a RewardsCfg class dynamically, compatible with Isaac Lab's
        configclass-based RewardManager._prepare_terms().
        动态构建 RewardsCfg 类。
        """
        from isaaclab.managers import RewardTermCfg
        from isaaclab.utils import configclass

        namespace = {}
        annotations = {}

        for term_name, term_info in self._terms.items():
            wrapper_func = self._make_wrapper(term_name)
            term_cfg = RewardTermCfg(
                func=wrapper_func,
                weight=term_info["weight"],
                params=term_info["params"],
            )
            namespace[term_name] = term_cfg
            annotations[term_name] = RewardTermCfg

        namespace["__annotations__"] = annotations

        RewardsCfg = type("RewardsCfg", (), namespace)
        RewardsCfg = configclass(RewardsCfg)

        return RewardsCfg

    # ------------------------------------------------------------------
    # Registration helpers / 注册辅助方法
    # ------------------------------------------------------------------

    def register_to_env_cfg(self, env_cfg):
        """
        Build RewardsCfg and replace env_cfg.rewards entirely (full override).
        构建 RewardsCfg 并完全替换 env_cfg.rewards（全量覆盖）。

        This is the recommended method for config-driven mode.
        这是配置驱动模式的推荐方法。

        Args:
            env_cfg: A ManagerBasedRLEnvCfg instance.
        """
        RewardsCfg = self.build_rewards_cfg()
        env_cfg.rewards = RewardsCfg()

    def merge_to_env_cfg(self, env_cfg):
        """
        Merge reward terms into existing env_cfg.rewards (additive).
        将奖励项合并到现有 env_cfg.rewards（追加模式）。

        Args:
            env_cfg: A ManagerBasedRLEnvCfg instance with existing rewards config.
        """
        from isaaclab.managers import RewardTermCfg

        existing_rewards = env_cfg.rewards
        for term_name, term_info in self._terms.items():
            wrapper_func = self._make_wrapper(term_name)
            term_cfg = RewardTermCfg(
                func=wrapper_func,
                weight=term_info["weight"],
                params=term_info["params"],
            )
            setattr(existing_rewards, term_name, term_cfg)

    def override_env_cfg(self, env_cfg):
        """
        Clear all existing reward terms, then set only bridged terms.
        清除所有现有奖励项，然后仅设置桥接层的奖励项。

        Args:
            env_cfg: A ManagerBasedRLEnvCfg instance.
        """
        from isaaclab.managers import RewardTermCfg

        existing_rewards = env_cfg.rewards

        for attr_name in list(vars(existing_rewards).keys()):
            if isinstance(getattr(existing_rewards, attr_name, None), RewardTermCfg):
                delattr(existing_rewards, attr_name)

        for term_name, term_info in self._terms.items():
            wrapper_func = self._make_wrapper(term_name)
            term_cfg = RewardTermCfg(
                func=wrapper_func,
                weight=term_info["weight"],
                params=term_info["params"],
            )
            setattr(existing_rewards, term_name, term_cfg)

    def bind_env(self, env):
        """
        Explicitly bind a real env to the underlying RewardProcess.
        显式地将真实 env 绑定到底层的 RewardProcess。

        Note: Usually NOT required — wrapper closures auto-sync env.
        """
        self._reward_process.env = env

    # ------------------------------------------------------------------
    # Introspection / 内省
    # ------------------------------------------------------------------

    def get_term_names(self) -> list[str]:
        """Get names of all discovered reward terms."""
        return list(self._terms.keys())

    def get_term_weights(self) -> dict[str, float]:
        """Get a dict of {term_name: weight} for all terms."""
        return {name: info["weight"] for name, info in self._terms.items()}

    def __repr__(self) -> str:
        lines = [f"RewardBridge ({len(self._terms)} terms, mode={self._mode}):"]
        for name, info in self._terms.items():
            source = info.get("source", "unknown")
            lines.append(f"  {name:30s}  weight={info['weight']:+.4f}  (source: {source})")
        return "\n".join(lines)
