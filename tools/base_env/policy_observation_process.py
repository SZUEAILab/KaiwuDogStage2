#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    """与 Isaac Lab `PolicyCfg` 对齐的观测处理器。"""

    target_group = "policy"

    def process(self):
        """示例：组合默认 policy 观测和 `base_lin_vel` term。"""
        return self.concatenate_terms(
            # self.default_observation(),
            self.base_lin_vel(),
        )
