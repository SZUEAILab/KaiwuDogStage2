# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

Inherits from `tools.base_env.observation_process.ObservationProcess` and
targets the `critic` observation group in Isaac Lab.
继承自 `tools.base_env.observation_process.ObservationProcess`，
目标为 Isaac Lab 中的 `critic` observation group。
"""

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor aligned with Isaac Lab `CriticCfg`.

    与 Isaac Lab `CriticCfg` 对齐的 critic 观测处理器。
    """

    target_group = "critic"

    def process(self):
        """Compute custom critic observation.

        计算自定义 critic 观测。

        Returns the original critic observation concatenated with goal position.
        返回原始 critic 观测与目标位置的拼接。

        Output: default_critic_obs + goal_position(4) = critic_dim + 4
        输出：default_critic_obs + goal_position(4) = critic_dim + 4
        """
        return self.concatenate_terms(
            self.default_observation(),
            self.goal_position_in_robot_frame(),
        )
