# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
tools.base_env package — lazy-imports Robot to avoid circular dependency.
tools.base_env 包 — 延迟导入 Robot 以避免循环依赖。

Framework uses `importlib.import_module('tools.base_env')` + `getattr(module, 'Robot')`,
so we provide `__getattr__` for deferred lookup.
框架通过 `importlib.import_module('tools.base_env')` + `getattr(module, 'Robot')` 获取，
因此使用 `__getattr__` 实现延迟查找。
"""


def __getattr__(name):
    if name == "Robot":
        from tools.base_env.base_env import Robot

        return Robot
    raise AttributeError(f"module 'tools.base_env' has no attribute {name!r}")
