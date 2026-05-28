# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
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
    """Policy observation processor with height_scan and optional goal obs."""

    target_group = "policy"

    def goal_position_in_robot_frame(self):
        num_goal_obs = self._get_num_goal_obs()
        zeros = torch.zeros(self.env.num_envs, num_goal_obs, device=self.env.device)

        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return zeros

        try:
            robot = self.env.scene["robot"]
            robot_pos = robot.data.root_pos_w
            robot_quat = robot.data.root_quat_w
            rel_pos_w = self.env.goal_positions[:, :3] - robot_pos[:, :3]
            rel_pos_b = self._rotate_vector_inverse(robot_quat, rel_pos_w)
            distance = torch.norm(rel_pos_w[:, :2], dim=-1, keepdim=True)
            goal_obs = torch.cat([rel_pos_b, distance], dim=-1)
            return self._fit_goal_obs_dim(goal_obs, zeros)
        except Exception:
            return zeros

    def _rotate_vector_inverse(self, quat, vector):
        quat_vec = quat[:, 1:4]
        quat_w = quat[:, 0:1]
        dot = torch.sum(quat_vec * vector, dim=-1, keepdim=True)
        cross = torch.cross(quat_vec, vector, dim=-1)
        quat_vec_norm = torch.sum(quat_vec * quat_vec, dim=-1, keepdim=True)
        return 2.0 * dot * quat_vec + (quat_w * quat_w - quat_vec_norm) * vector - 2.0 * quat_w * cross

    def _fit_goal_obs_dim(self, goal_obs, zeros):
        num_goal_obs = zeros.shape[1]
        if goal_obs.shape[1] >= num_goal_obs:
            return goal_obs[:, :num_goal_obs]
        zeros[:, : goal_obs.shape[1]] = goal_obs
        return zeros

    def process(self):
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        return obs
