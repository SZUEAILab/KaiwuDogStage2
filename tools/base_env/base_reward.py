#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcessBase — 奖励处理器基类，包含内置奖励方法。

所有内置奖励项实现为 `_reward_*` 方法，由 RewardBridge 自动发现。
TOML 参数通过方法签名透传（桥接器自动检测并转发）。
用户可在子类中覆盖任何 `_reward_{term_name}` 方法来自定义逻辑。

依赖：仅 isaaclab（不依赖 isaaclab_tasks 或 unitree_rl_lab）。
"""

import torch


class RewardProcessBase:
    """奖励处理器基类，包含内置奖励方法。

    所有 `_reward_*` 方法会被 `RewardBridge` 自动发现，子类可覆盖。

    方法签名约定:
        - `_reward_xxx(self)`:            无需 TOML 参数
        - `_reward_xxx(self, std=0.5)`:   TOML 参数透传，带默认值
        - `_reward_xxx(self, std)`:       TOML 参数必填（构建时校验）
        - `_reward_xxx(self, **kwargs)`:  所有 TOML 参数透传
    """

    def __init__(self, env=None):
        """初始化奖励处理器。

        Args:
            env: ManagerBasedRLEnv 实例，或 None（延迟绑定）。
                 RewardBridge 会在运行时自动注入真实 env。
        """
        self.env = env

    def process(self):
        """主奖励计算入口（可选），不自定义则使用默认流程。"""
        self.env._default_compute_reward()

    def create_bridge(self, reward_weights: dict[str, float] | None = None):
        """创建 RewardBridge，将所有 `_reward_*` 方法适配为 RewardTermCfg。

        Args:
            reward_weights: 可选的外部权重映射。

        Returns:
            RewardBridge 实例。
        """
        from tools.base_env.reward_bridge import RewardBridge

        return RewardBridge(self, reward_weights=reward_weights)

    def create_bridge_from_configs(self, reward_configs: dict):
        """从 TOML 解析的 reward configs 创建 RewardBridge（全量注册）。

        Args:
            reward_configs: TOML `[rewards.*]` 解析后的 dict。
                格式: {term_name: {"weight": float, "params": dict}}

        Returns:
            已注册所有 term 的 RewardBridge 实例。
        """
        from tools.base_env.reward_bridge import RewardBridge

        return RewardBridge.from_configs(self, reward_configs)

    # ------------------------------------------------------------------
    # Helper: SceneEntityCfg 缓存
    # ------------------------------------------------------------------

    _foot_sensor_cfg = None
    _undesired_sensor_cfg = None
    _robot_all_joints_cfg = None
    _foot_asset_cfg = None

    def _get_contact_sensor(self):
        """获取接触力传感器。"""
        return self.env.scene.sensors["contact_forces"]

    def _get_foot_sensor_cfg(self):
        """获取（并缓存）脚部 body 的 SceneEntityCfg。"""
        if self._foot_sensor_cfg is None:
            from isaaclab.managers import SceneEntityCfg

            self._foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
            self._foot_sensor_cfg.resolve(self.env.scene)
        return self._foot_sensor_cfg

    def _get_undesired_sensor_cfg(self):
        """Get (and cache) the SceneEntityCfg for undesired contact bodies.

        获取（并缓存）非期望接触 body 的 SceneEntityCfg。
        """
        if self._undesired_sensor_cfg is None:
            from isaaclab.managers import SceneEntityCfg

            self._undesired_sensor_cfg = SceneEntityCfg(
                "contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
            )
            self._undesired_sensor_cfg.resolve(self.env.scene)
        return self._undesired_sensor_cfg

    def _get_robot_all_joints_cfg(self):
        """获取（并缓存）所有机器人关节的 SceneEntityCfg。"""
        if self._robot_all_joints_cfg is None:
            from isaaclab.managers import SceneEntityCfg

            self._robot_all_joints_cfg = SceneEntityCfg("robot", joint_names=".*")
            self._robot_all_joints_cfg.resolve(self.env.scene)
        return self._robot_all_joints_cfg

    def _get_foot_asset_cfg(self):
        """获取（并缓存）机器人资产上脚部 body 的 SceneEntityCfg。"""
        if self._foot_asset_cfg is None:
            from isaaclab.managers import SceneEntityCfg

            self._foot_asset_cfg = SceneEntityCfg("robot", body_names=".*_foot")
            self._foot_asset_cfg.resolve(self.env.scene)
        return self._foot_asset_cfg

    def _get_robot_asset(self):
        """获取机器人关节资产。"""
        return self.env.scene["robot"]

    def _get_foot_contact(self, threshold=1.0):
        """根据力阈值获取脚部接触状态。

        Returns:
            Boolean tensor (num_envs, num_feet)。
        """
        contact_sensor = self._get_contact_sensor()
        foot_cfg = self._get_foot_sensor_cfg()
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, foot_cfg.body_ids, 2])
        return forces_z > threshold

    # ==================================================================
    # 内置奖励方法（可在子类中覆盖）
    # ==================================================================

    # --- 任务奖励 ---

    def _reward_track_lin_vel_xy(self, std: float = 0.25, command_name: str = "base_velocity"):
        """exp 核线速度跟踪奖励 (xy)。"""
        asset = self._get_robot_asset()
        lin_vel_error = torch.sum(
            torch.square(self.env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / std**2)

    def _reward_track_ang_vel_z(self, std: float = 0.25, command_name: str = "base_velocity"):
        """exp 核角速度跟踪奖励 (yaw)。"""
        asset = self._get_robot_asset()
        ang_vel_error = torch.square(
            self.env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
        )
        return torch.exp(-ang_vel_error / std**2)

    # --- 运动质量 ---

    def _reward_lin_vel_z(self):
        """惩罚 z 轴线速度（垂直弹跳）。"""
        asset = self._get_robot_asset()
        return torch.square(asset.data.root_lin_vel_b[:, 2])

    def _reward_ang_vel_xy(self):
        """惩罚 xy 角速度（横滚/俯仰振荡）。"""
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

    # --- 能量消耗 ---

    def _reward_joint_acc(self):
        """惩罚大的关节加速度。"""
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_acc), dim=1)

    def _reward_joint_torques(self):
        """惩罚大的关节力矩。"""
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.applied_torque), dim=1)

    def _reward_action_rate(self):
        """惩罚步间动作变化。"""
        return torch.sum(torch.square(self.env.action_manager.action - self.env.action_manager.prev_action), dim=1)

    # --- 安全性 ---

    def _reward_dof_pos_limits(self):
        """惩罚接近或超过限位的关节位置。"""
        asset = self._get_robot_asset()
        out_of_limits = -(asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 0]).clip(max=0.0) + (
            asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 1]
        ).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_undesired_contacts(self, threshold: float = 1.0):
        """惩罚非期望部位接触（大腿、小腿）。"""
        sensor_cfg = self._get_undesired_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        net_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1).float()

    # ==================================================================
    # IsaacLab 原生 reward（从 agent_ppo 下沉以统一 base 仅含原生）
    # IsaacLab-native rewards (sunk from agent_ppo to unify base with natives only)
    # ==================================================================

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

        Args:
            command_name: Command term name. / 命令项名称。
            threshold: Minimum air time threshold. / 最小滞空时间阈值。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        # Compute reward
        # 计算奖励
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        # No reward for zero commands
        # 当命令为零时不给奖励
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).

        惩罚脚部在地面上的滑动（接触时的速度）。
        对齐 Isaac Lab feet_slide：用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        # Check which feet are in contact (3D force norm, max over history frames)
        # 检查哪些脚在接触地面（3D 力的范数，历史帧取最大值）
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        # Get foot velocities (xy only)
        # 获取脚部速度（仅 xy 分量）
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        # Penalize velocity when in contact
        # 接触时惩罚速度
        reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        return reward
