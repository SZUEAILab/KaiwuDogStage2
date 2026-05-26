# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256)] → 301 dim
观测布局：[proprio(45) | height_scan(256)] → 301 维

Extending to track terrain (optional):
    In track terrain the environment additionally provides the following
    read-only attributes (not available in standard terrain):
      - env.goal_positions  (num_envs, 3)  — exit position in world frame
      - env.goal_yaw        (num_envs,)    — exit heading in world frame
    The environment always exposes these scene sensors (available in both
    standard and track terrains, accessed via env.scene.sensors["<name>"]):
      - "height_scanner"  — default forward ground-clearance scan
      - "nav_scanner"     — forward-looking occlusion scan (wider range,
                             suited for obstacle avoidance / turning)
    Players can construct their own obs from these inputs. After appending,
    update the Stage config (observation dim) and model input dim accordingly.

扩展到 track 地形时（可选）：
    track 地形下，环境会额外提供以下只读属性（standard 地形没有）：
      - env.goal_positions  (num_envs, 3)  — 出口在世界坐标系下的 3D 位置
      - env.goal_yaw        (num_envs,)    — 出口在世界坐标系下的朝向
    环境在两种地形下都会通过 env.scene.sensors["<name>"] 提供以下传感器：
      - "height_scanner"  — 默认前方地面高度扫描
      - "nav_scanner"     — 前瞻遮挡扫描（范围更大，适合避障 / 转向判断）
    选手可从这些属性和传感器自行构造 obs。
    拼接后需同步修改 Stage 的观测维度和 model 输入维度。
obs layout (standard): [proprio(45) | height_scan(256)] → 301 dim
obs layout (track):    [proprio(45) | height_scan(256) | goal(4)] → 305 dim
"""

import torch

from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan and optional goal obs.

    带 height_scan 和可选 goal obs 的 policy 观测处理器。
    """

    target_group = "policy"

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

        Returns flattened nav_scanner rays, or None if disabled/unavailable.
        Enable by setting num_nav_scan in the stage config to the nav_scanner ray count
        (check sensor.data.ray_hits_w.shape[-2] at runtime after env creation).

        读取 nav_scanner 远距离障碍感知数据。若未启用/不可用则返回 None。
        启用方式：将 stage config 的 num_nav_scan 设为 nav_scanner 射线数量
        （env 创建后查看 sensor.data.ray_hits_w.shape[-2]）。
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
