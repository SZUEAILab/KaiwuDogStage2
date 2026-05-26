# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256)] → 316 dim
critic 观测布局：[critic_proprio(60) | height_scan(256)] → 316 维

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
critic obs layout (standard): [critic_proprio(60) | height_scan(256)] → 316 dim
critic obs layout (track):    [critic_proprio(60) | height_scan(256) | goal(4)] → 320 dim
"""

import torch

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor with optional goal obs.

    与 policy 观测保持同步的 critic 观测处理器，可选拼接 goal obs。
    """

    target_group = "critic"

    def goal_position_in_robot_frame(self):
        """Compute goal position and heading in robot body frame.

        计算机器人坐标系下的目标位置和朝向。

        Returns:
            torch.Tensor: shape (num_envs, 4) — [dx, dy, dz, dyaw]
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, 4, device=self.env.device)

        robot = self.env.scene["robot"]
        robot_pos = robot.data.root_pos_w
        robot_quat = robot.data.root_quat_w

        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        goal_pos = self.env.goal_positions
        goal_yaw = self.env.goal_yaw

        rel_pos = goal_pos - robot_pos

        cos = torch.cos(robot_yaw)
        sin = torch.sin(robot_yaw)

        dx = cos * rel_pos[:, 0] + sin * rel_pos[:, 1]
        dy = -sin * rel_pos[:, 0] + cos * rel_pos[:, 1]
        dz = rel_pos[:, 2]

        dyaw = goal_yaw - robot_yaw
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

        return torch.stack([dx, dy, dz, dyaw], dim=-1)

    def _get_num_yaw_obs(self):
        """Read yaw obs dimension from stage config.
        从 stage config 读取 yaw obs 维度。
        """
        from agent_ppo.conf.conf import Config
        return getattr(Config.CURRENT, "num_yaw_obs", 0)

    def world_yaw_observation(self):
        """Compute sin and cos of robot world-frame yaw angle.
        计算机器人世界坐标系偏航角的 sin 和 cos。

        Returns:
            torch.Tensor: shape (num_envs, 2) — [sin(yaw), cos(yaw)]
        """
        robot = self.env.scene["robot"]
        quat = robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)

    def _get_num_nav_scan(self):
        """Read nav_scanner ray count from stage config.
        从 stage config 读取 nav_scanner 射线数量。
        """
        from agent_ppo.conf.conf import Config
        return getattr(Config.CURRENT, "num_nav_scan", 0)

    def nav_scanner_observation(self):
        """Read nav_scanner data for longer-range obstacle awareness.
        Returns None if disabled/unavailable.

        读取 nav_scanner 远距离障碍感知数据。若未启用/不可用则返回 None。
        """
        num_rays = self._get_num_nav_scan()
        if num_rays <= 0 or "nav_scanner" not in self.env.scene.sensors:
            return None
        sensor = self.env.scene.sensors["nav_scanner"]
        origins_z = sensor.data.pos_w[:, 2:3]
        hits_z = sensor.data.ray_hits_w[..., 2]
        scan = origins_z - hits_z
        return scan.view(self.env.num_envs, -1)[:, :num_rays]

    def process(self):
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        if self._get_num_yaw_obs() > 0:
            yaw_obs = self.world_yaw_observation()
            obs = self.concatenate_terms(obs, yaw_obs)

        nav_scan = self.nav_scanner_observation()
        if nav_scan is not None:
            obs = self.concatenate_terms(obs, nav_scan)

        return obs
