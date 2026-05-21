# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess for PPO - Enhanced with DIY rewards for stair training.
RewardProcess for PPO - 集成 DIY 奖励用于台阶训练。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    """
    Custom reward processor with user-defined reward terms for stairs.
    自定义奖励处理器，包含用于台阶训练的用户定义奖励项。
    """

    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in the robot body frame.
        前向速度奖励：机器人本体坐标系下 x 方向速度。
        """
        asset = self._get_robot_asset()
        return asset.data.root_lin_vel_b[:, 0]

    def _reward_flat_orientation(self):
        """Penalize non-flat base orientation (deviation from upright).
        惩罚非平坦的基座朝向（偏离直立）。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_joint_vel(self):
        """Penalize large joint velocities.
        惩罚大的关节速度。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.3):
        """Reward long steps (feet air time above threshold when moving).
        奖励长步幅（移动时脚部滞空时间超过阈值）。

        Args:
            command_name: Command term name.
            threshold: Minimum air time threshold.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_feet_air_time_variance(self):
        """Penalize variance in foot air/contact time (gait symmetry).
        惩罚脚部滞空/接触时间的方差（步态对称性）。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).
        惩罚脚部在地面上的滑动（接触时的速度）。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        return reward

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).
        惩罚脚撞到垂直面（台阶边缘、墙壁）。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)

        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_joint_position_penalty(self, stand_still_scale: float = 5.0, velocity_threshold: float = 0.1):
        """Penalize joint position error from default pose.
        惩罚关节位置偏离默认姿态。
        """
        asset = self._get_robot_asset()
        cmd = torch.linalg.norm(self.env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
            reward,
            stand_still_scale * reward,
        )

    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out).
        惩罚真正的失败（被终止且非超时截断）。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        return failure.float()

    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward reaching the goal position (active only when track terrain with goal_positions maintained).
        奖励到达目标位置（仅在 track 地形且 env.goal_positions 被维护时生效）。

        Args:
            threshold: Distance threshold for goal completion (meters).
        """
        # Stub: returns zero. Replace with actual goal-distance reward for track terrain.
        return torch.zeros(self._get_robot_asset().data.root_lin_vel_b.shape[0], device=self._get_robot_asset().device)

    def _reward_energy(self):
        """Penalize energy consumption (torque × joint velocity).
        惩罚能耗（扭矩 × 关节速度）。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.abs(asset.data.joint_torques * asset.data.joint_vel), dim=1)

    def _reward_correct_base_height(self, target_height: float = 0.38):
        """Penalize deviation from target base height.
        惩罚基座离地高度偏离目标值。

        Args:
            target_height: Target base height in meters.
        """
        asset = self._get_robot_asset()
        return torch.square(asset.data.root_pos_w[:, 2] - target_height)

    def _reward_hip_to_default(self):
        """Penalize hip joints (first joint of each leg) deviating from default.
        惩罚髋关节偏离默认角度。
        """
        asset = self._get_robot_asset()
        # Hips are at indices 0, 3, 6, 9 (first joint of each leg)
        hip_indices = [0, 3, 6, 9]
        hip_pos = asset.data.joint_pos[:, hip_indices]
        hip_default = asset.data.default_joint_pos[:, hip_indices]
        return torch.sum(torch.square(hip_pos - hip_default), dim=1)

