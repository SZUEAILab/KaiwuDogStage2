#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    构建监控面板配置，为每个指标提供说明: 作用+如何判断训练好坏。
    description 限制 0~200 字符，仅允许中英文/数字/空格及指定特殊符号。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人训练监控")
        # ==============================================================
        # Group 1: 训练概览
        # ==============================================================
        .add_group(group_name="训练概览", group_name_en="overview")
        .add_panel(
            name="学习率",
            name_en="learning_rate",
            type="line",
            description="策略学习率，自适应调度下根据KL散度自动调整。升高:策略变化过大需更保守；降低:策略变化不足可更激进。若降至最低值(1e-5):策略可能已收敛或陷入局部最优。",
        )
        .add_metric(metrics_name="learning_rate", expr="avg(learning_rate{})")
        .end_panel()
        .add_panel(
            name="训练回合数",
            name_en="episode_cnt",
            type="line",
            description="已完成的总训练回合数，单调递增反映训练进度。配合其他指标可判断训练效率(如每N回合奖励提升多少)。",
        )
        .add_metric(metrics_name="episode_cnt", expr="episode_cnt{}")
        .end_panel()
        .add_panel(
            name="回合平均步数",
            name_en="episode_len_mean",
            type="line",
            description="每个episode的平均步数。正常训练:初期较短(频繁摔倒终止)，后期逐渐增长至上限。始终很短:机器人频繁摔倒；长期保持上限:可考虑增大episode_length_s。",
        )
        .add_metric(metrics_name="episode_len_mean", expr="avg(episode_len_mean{})")
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 2: 算法损失
        # ==============================================================
        .add_group(group_name="算法损失", group_name_en="algorithm_loss")
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
            description="策略损失+价值损失+熵损失的总和。正常:整体下降并趋于稳定。持续震荡不降:学习率可能过高；快速降至接近0:可能过拟合；突然飙升:训练崩溃(检查NaN/Inf防护日志)。",
        )
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
            description="PPO surrogate loss，反映策略改进幅度和clip效果。正常:缓慢下降或小幅波动。剧烈震荡:学习率过高或clip_param不当；持续为正:策略未在改进；恒为负:advantage计算可能有误。",
        )
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
            description="Critic对状态价值的估计误差。正常:持续下降。居高不下:Critic难以预测回报，检查网络结构或value_loss_coef；远大于策略损失:需调整value_loss_coef权重。",
        )
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
            description="策略动作分布的熵，度量探索程度。正常:缓慢下降(探索到利用)。快速趋近0:过早收敛陷入局部最优，需增大entropy_coef；一直很高:策略未学习有效行为，检查奖励设计。",
        )
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 3: 训练诊断
        # ==============================================================
        .add_group(group_name="训练诊断", group_name_en="diagnostics")
        .add_panel(
            name="KL散度",
            name_en="kl_divergence",
            type="line",
            description="新旧策略分布的KL散度，衡量每次参数更新幅度。正常:在desired_kl(默认0.01)附近小幅波动。持续远大于desired_kl:学习率过高更新激进；持续远小于:更新保守收敛慢。自适应调度据此自动调整学习率。",
        )
        .add_metric(metrics_name="kl_divergence", expr="avg(kl_divergence{})")
        .end_panel()
        .add_panel(
            name="梯度范数",
            name_en="grad_norm",
            type="line",
            description="参数梯度的L2范数(裁剪前)，反映更新信号强度。正常:在max_grad_norm(默认1.0)附近或以内。频繁远大于max_grad_norm:梯度爆炸风险，检查reward尺度；一直很小:梯度消失，训练可能停滞。",
        )
        .add_metric(metrics_name="grad_norm", expr="avg(grad_norm{})")
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 4: 回合奖励
        # ==============================================================
        .add_group(group_name="回合奖励", group_name_en="episode_rewards")
        .add_panel(
            name="回合总奖励",
            name_en="episode_reward",
            type="line",
            description="单回合内所有奖励项的加权总和，衡量训练效果的最核心指标。正常:整体上升趋势。长期不升或下降:检查奖励权重/环境配置；剧烈波动:环境数太少或随机性过大；不再上升:训练接近收敛。",
        )
        .add_metric(metrics_name="episode_reward", expr="avg(episode_reward{})")
        .end_panel()
        .add_panel(
            name="奖励均值",
            name_en="reward_mean",
            type="line",
            description="当前batch每步奖励的均值，反映策略平均表现。正常:与回合总奖励趋势一致整体上升。均值上升但总奖励不升:回合长度在缩短；均值突降:策略可能出现退化。",
        )
        .add_metric(metrics_name="reward_mean", expr="avg(reward_mean{})")
        .end_panel()
        .add_panel(
            name="奖励标准差",
            name_en="reward_std",
            type="line",
            description="batch内每步奖励的标准差，反映不同环境/状态下奖励差异。过大:curriculum难度不一致或策略在不同场景表现两极分化；过小:所有环境同质化(curriculum范围可能太窄)。正常:在合理范围波动。",
        )
        .add_metric(metrics_name="reward_std", expr="avg(reward_std{})")
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 5: 奖励-运控
        # ==============================================================
        .add_group(group_name="奖励-运控", group_name_en="reward_locomotion")
        .add_panel(
            name="线速度跟踪",
            name_en="reward_track_lin_vel_xy",
            type="line",
            description="XY平面线速度跟踪奖励，运控能力最核心指标。越高:速度跟踪越准。正常:逐渐趋近1.0。始终很低:未学习速度跟踪，检查command范围/权重；突降:策略遗忘或curriculum升级过快。",
        )
        .add_metric(metrics_name="reward_track_lin_vel_xy", expr="avg(reward_track_lin_vel_xy{})")
        .end_panel()
        .add_panel(
            name="角速度跟踪",
            name_en="reward_track_ang_vel_z",
            type="line",
            description="Z轴角速度跟踪奖励。越高:转向指令执行越准。在maze/track地形中决定机器人能否及时转弯避障。正常:与线速度跟踪协同上升。",
        )
        .add_metric(metrics_name="reward_track_ang_vel_z", expr="avg(reward_track_ang_vel_z{})")
        .end_panel()
        .add_panel(
            name="姿态稳定",
            name_en="reward_flat_orientation",
            type="line",
            description="基座偏离水平的惩罚(负值，趋近0:好)。绝对值大:机器人摇晃/倾倒。正常:逐渐趋近0。持续为较大负值:平衡不足，检查模型输出或增大该惩罚权重。",
        )
        .add_metric(metrics_name="reward_flat_orientation", expr="avg(reward_flat_orientation{})")
        .end_panel()
        .add_panel(
            name="失败惩罚",
            name_en="reward_termination",
            type="line",
            description="机器人终止(摔倒/超时)的惩罚(负值，趋近0:好)。绝对值大:大量机器人在本回合提前终止。正常:逐渐趋近0。持续为较大负值:频繁摔倒，环境过难或策略未收敛。配合奖励均值/回合总奖励诊断。",
        )
        .add_metric(metrics_name="reward_termination", expr="avg(reward_termination{})")
        .end_panel()
        .add_panel(
            name="碰撞惩罚",
            name_en="reward_undesired_contacts",
            type="line",
            description="身体(非脚)接触地面或障碍的惩罚(负值，趋近0:好)。绝对值大:频繁摔倒/趴下。在maze地形中也可能因撞墙触发。正常:逐渐趋近0。",
        )
        .add_metric(metrics_name="reward_undesired_contacts", expr="avg(reward_undesired_contacts{})")
        .end_panel()
        .add_panel(
            name="脚部绊倒",
            name_en="reward_feet_stumble",
            type="line",
            description="脚撞台阶边缘/垂直面的惩罚(负值，趋近0:好)。在楼梯地形中尤其关键。绝对值大:步态不适应台阶。正常:逐渐趋近0。",
        )
        .add_metric(metrics_name="reward_feet_stumble", expr="avg(reward_feet_stumble{})")
        .end_panel()
        .add_panel(
            name="动作平滑",
            name_en="reward_action_rate",
            type="line",
            description="相邻动作变化(一阶平滑)的惩罚(负值，趋近0:好)。绝对值大:关节指令剧烈抖动，步态不流畅。正常:逐渐趋近0。一直很大:策略输出震荡，需增大该惩罚权重。",
        )
        .add_metric(metrics_name="reward_action_rate", expr="avg(reward_action_rate{})")
        .end_panel()
        .add_panel(
            name="关节力矩",
            name_en="reward_joint_torques",
            type="line",
            description="关节力矩的惩罚(负值，趋近0:好)。绝对值大:策略使用过大扭矩。正常:缓慢减小。可以反映能效:更低的力矩则更节能的步态。",
        )
        .add_metric(metrics_name="reward_joint_torques", expr="avg(reward_joint_torques{})")
        .end_panel()
        .add_panel(
            name="能耗",
            name_en="reward_energy",
            type="line",
            description="能耗惩罚(torque*joint_vel，负值，趋近0:好)。绝对值大:步态耗能高/关节发力猛。正常:逐渐趋近0，策略学会节能步态。始终很大:步态低效，考虑增大该权重；趋近0后性能不降:已学到高效步态。",
        )
        .add_metric(metrics_name="reward_energy", expr="avg(reward_energy{})")
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 6: 奖励-导航
        # ==============================================================
        .add_group(group_name="奖励-导航", group_name_en="reward_navigation")
        .add_panel(
            name="到达目标",
            name_en="reward_reach_goal",
            type="line",
            description="到达导航目标的奖励(track/maze地形)，导航成功率的直接指标。正常:逐渐升高。始终为0:导航策略未生效，检查goal_positions和approach_goal；达到较高值后稳定:多数机器人能成功到达。",
        )
        .add_metric(metrics_name="reward_reach_goal", expr="avg(reward_reach_goal{})")
        .end_panel()
        .add_panel(
            name="接近目标",
            name_en="reward_approach_goal",
            type="line",
            description="接近目标的密集奖励(距离减少则正奖励)。正常:逐渐升高，机器人更高效接近目标。保持为负:机器人平均在远离目标，可能是导航策略或避障信号冲突。",
        )
        .add_metric(metrics_name="reward_approach_goal", expr="avg(reward_approach_goal{})")
        .end_panel()
        .add_panel(
            name="避障",
            name_en="reward_obstacle_evasion",
            type="line",
            description="前方有障碍时未转向的惩罚(负值，趋近0:好)。绝对值大:面对障碍不转向。在maze/track中决定能否走到目标还是被墙拦住。正常:绝对值逐渐减小。",
        )
        .add_metric(metrics_name="reward_obstacle_evasion", expr="avg(reward_obstacle_evasion{})")
        .end_panel()
        .add_panel(
            name="近墙减速",
            name_en="reward_wall_proximity_brake",
            type="line",
            description="前方有墙时高速前进的惩罚(负值，趋近0:好)。绝对值大:撞墙前不减速。正常:绝对值逐渐减小，机器人学会在墙前减速。配合避障指标协同观察。",
        )
        .add_metric(metrics_name="reward_wall_proximity_brake", expr="avg(reward_wall_proximity_brake{})")
        .end_panel()
        .add_panel(
            name="导航耗时",
            name_en="reward_navigation_time",
            type="line",
            description="每步固定时间惩罚(负值，趋近0:好)。绝对值越小的机器人越快速到达目标。正常:逐渐增大(更少步数到达则总惩罚更少)。注意:该值受episode长度影响，需配合reach_goal判断。",
        )
        .add_metric(metrics_name="reward_navigation_time", expr="avg(reward_navigation_time{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
