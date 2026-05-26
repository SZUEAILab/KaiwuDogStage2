# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess for PPO.
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    """
    Custom reward processor for PPO.
    """

    # =========================================================================
    # Synced from agent_diy/feature/reward_process.py
    # 以下同步自 agent_diy/feature/reward_process.py
    # =========================================================================

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

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """Reward long steps (feet air time above threshold when moving).
        奖励长步幅（移动时脚部滞空时间超过阈值）。
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

    def _reward_air_time_variance_penalty(self):
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

    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        turn_std: float = 0.5,
    ):
        """Penalize forward-blocked path when robot is not actively turning.
        惩罚前方被障碍阻挡时未主动转向。
        """
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)
        window = grid[:, body_y_start:body_y_end, :near_x_end]
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        blocked = col_blocked.mean(dim=-1)
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()
        return blocked * not_evading * has_fwd_cmd

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).
        惩罚脚撞到垂直面。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_approach_goal(self):
        """Reward approaching the maze exit: -(current_dist - previous_dist).
        接近目标奖励：距离减少→正奖励，距离增加→负奖励。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        current_dist = torch.norm(goal_pos - robot_pos, dim=1)
        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)
        delta_dist = current_dist - self.env._previous_goal_dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0
        self.env._previous_goal_dist = current_dist.clone()
        return -delta_dist

    def _reward_reach_goal(self, threshold: float = 0.5):
        """Reward reaching the goal (distance < threshold).
        到达目标奖励（距离 < 阈值时返回 1.0）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_navigation_time(self):
        """Per-step penalty to encourage fast navigation.
        每步固定惩罚，鼓励快速到达。
        """
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out AND NOT goal-reached).
        惩罚真正的失败。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        if "goal_reached" in term_mgr.active_terms:
            goal_done = term_mgr.get_term("goal_reached")
            failure = failure & ~goal_done
        return failure.float()

    # =========================================================================
    # PPO-only rewards (locomotion / e2e)
    # PPO 独有奖励（运控 / e2e）
    # =========================================================================

    def _reward_forward_velocity_world(self):
        """Forward velocity reward: x-direction velocity in world frame.
        前向速度奖励：世界坐标系下 x 方向速度。
        """
        asset = self._get_robot_asset()
        return asset.data.root_lin_vel_w[:, 0]

    def _reward_forward_velocity_relative(self):
        """Forward velocity reward: x-direction velocity in robot body frame.
        前向速度奖励：机器人本体坐标系下 x 方向速度。
        """
        asset = self._get_robot_asset()
        return asset.data.root_lin_vel_b[:, 0]

    def _reward_energy(self):
        """Penalize energy consumption (torque × joint velocity).
        惩罚能耗（扭矩 × 关节速度）。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)

    def _reward_energy_exp(self, scale: float = 1.0):
        """Energy efficiency reward: exponential decay of mean joint mechanical power.
        能耗效率奖励：平均关节机械功率的指数衰减。
        """
        asset = self._get_robot_asset()
        power = torch.abs(asset.data.applied_torque * asset.data.joint_vel)
        mean_power = torch.mean(power, dim=1)
        return torch.exp(-scale * mean_power)

    def _reward_correct_base_height(self, target_height: float = 0.38, margin: float = 0.05):
        """Penalize base height deviation outside [target ± margin] relative to ground.
        基座离地高度超出 [target ± margin] 区间时给予惩罚。
        """
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]
        ground_height = sensor.data.ray_hits_w[..., 2].median(dim=1).values
        error = asset.data.root_pos_w[:, 2] - ground_height - target_height
        return torch.square(torch.clamp(torch.abs(error) - margin, min=0.0))

    def _reward_hip_to_default(self):
        """Penalize hip joints deviating from default.
        惩罚髋关节偏离默认角度。
        """
        asset = self._get_robot_asset()
        hip_indices = [0, 3, 6, 9]
        hip_pos = asset.data.joint_pos[:, hip_indices]
        hip_default = asset.data.default_joint_pos[:, hip_indices]
        return torch.sum(torch.square(hip_pos - hip_default), dim=1)

    def _reward_feet_height_body(self, command_name: str = "base_velocity", target_height: float = -0.30, tanh_mult: float = 2.0):
        """Penalize foot height deviation from target in body frame.
        惩罚脚部机体坐标系高度偏差。
        """
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]
        foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        base_pos = self._get_robot_asset().data.root_pos_w[:, 2:3]
        error = torch.square(foot_pos - base_pos - target_height)
        reward = torch.tanh(tanh_mult * error)
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return torch.mean(reward, dim=1) * is_moving.float()

    def _reward_action_smoothness(self):
        """Penalize 2nd-order action rate.
        惩罚二阶动作平滑度。
        """
        am = self.env.action_manager
        delta = am.action - am.prev_action
        if not hasattr(self.env, "_prev_action_delta") or self.env._prev_action_delta is None:
            self.env._prev_action_delta = delta.clone()
            return torch.zeros(self.env.num_envs, device=delta.device)
        smoothness = torch.sum(torch.square(delta - self.env._prev_action_delta), dim=1)
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta[reset_mask] = 0.0
        self.env._prev_action_delta = delta.clone()
        return smoothness

    def _reward_foot_force_symmetry(self):
        """Penalize left-right asymmetry in foot contact forces.
        惩罚左右脚接触力的不对称。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
        fl = torch.norm(forces[:, 0], dim=-1)
        fr = torch.norm(forces[:, 1], dim=-1)
        rl = torch.norm(forces[:, 2], dim=-1)
        rr = torch.norm(forces[:, 3], dim=-1)
        front_asym = torch.abs(fl - fr)
        rear_asym = torch.abs(rl - rr)
        return (front_asym + rear_asym) / (torch.clamp(fl + fr + rl + rr, min=1.0))

    def _reward_foot_height_symmetry(self):
        """Penalize left-right asymmetry in foot heights (body frame).
        惩罚左右脚高度的不对称（机体坐标系）。
        """
        asset_cfg = self._get_foot_asset_cfg()
        asset = self.env.scene[asset_cfg.name]
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        base_z = self._get_robot_asset().data.root_pos_w[:, 2:3]
        rel_z = foot_z - base_z
        fl = rel_z[:, 0]
        fr = rel_z[:, 1]
        rl = rel_z[:, 2]
        rr = rel_z[:, 3]
        front_asym = torch.abs(fl - fr)
        rear_asym = torch.abs(rl - rr)
        return front_asym + rear_asym

    # =========================================================================
    # PPO-only: navigation rewards (track terrain)
    # PPO 独有：导航奖励（track 地形）
    # =========================================================================

    def _reward_wall_proximity_brake(self, obstacle_threshold: float = -0.3):
        """Penalize high forward speed when walls are close ahead.
        前方有墙时惩罚高速前进，鼓励近墙减速。
        """
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)
        center_window = grid[:, 6:10, :8]
        near_wall = (center_window < obstacle_threshold).any(dim=-1).any(dim=-1).float()
        fwd_vel = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)
        return near_wall * fwd_vel

    def _get_scan_grid(self):
        """Get height_scanner data as 16x16 grid.
        获取 height_scanner 16x16 网格数据。
        """
        sensor = self.env.scene.sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        return scan.view(self.env.num_envs, 16, 16)

    def _get_far_field_blocked(self, obstacle_threshold: float = -0.4):
        """Use nav_scanner for longer-range forward obstacle detection (~2.5m).

        Uses a more stringent threshold than near-field because slopes accumulate
        height over distance. Walls are 0.5m — -0.4 threshold + fraction gating
        avoids slope false positives.
        远场用更严格的阈值：斜坡在远距离会积累高差。墙高 0.5m，-0.4 阈值 + 比例
        门控可避免斜坡误判。
        """
        if "nav_scanner" not in self.env.scene.sensors:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sensor = self.env.scene.sensors["nav_scanner"]
        origins_z = sensor.data.pos_w[:, 2:3]
        hits_z = sensor.data.ray_hits_w[..., 2]
        scan = origins_z - hits_z
        blocked_frac = (scan < obstacle_threshold).float().mean(dim=-1)
        return (blocked_frac > 0.3).float()

    def _reward_heuristic_navigation(self, obstacle_threshold: float = -0.3):
        """Wall-aware navigation: forward velocity when clear, clearance-guided turn when blocked.
        墙体感知导航奖励：通畅时奖励前进，堵塞时奖励向 clearance 更大侧转向。
        """
        asset = self._get_robot_asset()
        grid = self._get_scan_grid()
        far_blocked = self._get_far_field_blocked()
        near_x = 10
        left_region = grid[:, 0:8, :near_x]
        right_region = grid[:, 8:16, :near_x]
        front_center = grid[:, 5:11, :near_x]
        left_clear = (left_region > obstacle_threshold).float().mean(dim=(1, 2))
        right_clear = (right_region > obstacle_threshold).float().mean(dim=(1, 2))
        near_blocked_frac = (front_center < obstacle_threshold).float().mean(dim=(1, 2))
        far_amplification = far_blocked * 0.7 * (near_blocked_frac > 0.05).float()
        front_blocked_frac = torch.max(near_blocked_frac, far_amplification)
        side_preference = left_clear - right_clear
        fwd_vel = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)
        forward_ok = (1.0 - front_blocked_frac).clamp(min=0.0)
        fwd_reward = fwd_vel * forward_ok
        yaw_rate = asset.data.root_ang_vel_b[:, 2]
        turn_reward = torch.tanh(yaw_rate * side_preference * 2.0)
        return (1.0 - front_blocked_frac) * fwd_reward + front_blocked_frac * turn_reward

    def _reward_deadend_escape(self, obstacle_threshold: float = -0.3, trapped_threshold: float = 0.3):
        """Reward high angular velocity when trapped on left, front, and right.
        死胡同逃脱：三面被堵时奖励高角速度转向。
        """
        asset = self._get_robot_asset()
        grid = self._get_scan_grid()
        far_blocked = self._get_far_field_blocked()
        left_blocked = (grid[:, 0:5, 0:10] < obstacle_threshold).float().mean(dim=(1, 2)) > trapped_threshold
        front_blocked = (grid[:, 5:11, 0:8] < obstacle_threshold).float().mean(dim=(1, 2)) > trapped_threshold
        right_blocked = (grid[:, 11:16, 0:10] < obstacle_threshold).float().mean(dim=(1, 2)) > trapped_threshold
        is_trapped = (left_blocked & front_blocked & right_blocked & (far_blocked > 0.5)).float()
        yaw_rate_mag = torch.abs(asset.data.root_ang_vel_b[:, 2])
        return is_trapped * torch.clamp(yaw_rate_mag, max=1.5)
