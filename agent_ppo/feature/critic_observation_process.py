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
    """Critic observation processor with optional goal obs."""

    target_group = "critic"

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
