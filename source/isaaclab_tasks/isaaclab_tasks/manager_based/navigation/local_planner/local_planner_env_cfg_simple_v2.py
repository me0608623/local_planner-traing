# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
【Simple v2】基於DEBUG版本的優化配置

基於DEBUG版本的成功經驗（18.75%成功率），逐步增加難度：
1. 保持極簡獎勵設計（只3個獎勵項）
2. 逐步增加目標距離和環境數量
3. 逐步收緊成功閾值
"""

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .local_planner_env_cfg import (
    LocalPlannerEnvCfg,
    LocalPlannerSceneCfg,
    ObservationsCfg,
    CommandsCfg,
    ActionsCfg,
    TerminationsCfg,
    EventCfg,
)
import isaaclab_tasks.manager_based.navigation.local_planner.mdp as mdp


##
# 極簡獎勵（基於DEBUG成功經驗）
##


@configclass
class SimpleV2RewardsCfg:
    """【Simple v2】極簡獎勵 - 只保留最有效的"""
    
    # ✅ 主要驅動：接近目標
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=50.0,  # 極高權重（DEBUG版本證明有效）
        params={"command_name": "goal_command"},
    )
    
    # ✅ 成功大獎
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=500.0,  # 極高獎勵（DEBUG版本證明有效）
        params={"command_name": "goal_command", "threshold": 0.8},  # 稍微收緊
    )
    
    # ❌ 防止靜止
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=-0.1,
    )


##
# 課程學習：三個階段
##


@configclass
class SimpleV2CommandsCfg_STAGE1(CommandsCfg):
    """階段1：極近目標（和DEBUG一樣）"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(15.0, 15.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 1.0),
            pos_y=(-0.5, 0.5),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SimpleV2CommandsCfg_STAGE1_5(CommandsCfg):
    """階段1.5：過渡階段（溫和增加難度）"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(16.0, 16.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 1.3),  # 稍微比Stage1遠（0.3-1.0→0.3-1.3）
            pos_y=(-0.6, 0.6),  # 稍微比Stage1寬（±0.5→±0.6）
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SimpleV2CommandsCfg_STAGE2(CommandsCfg):
    """階段2：近目標（調整：縮小與Stage1的差距）"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(18.0, 18.0),  # 🔧 從20秒改為18秒
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 1.5),  # 🔧🔧 從0.5-2.0改為0.4-1.5（平均0.95米，更溫和）
            pos_y=(-0.8, 0.8),  # 🔧🔧 從±1.0改為±0.8（更窄）
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SimpleV2CommandsCfg_STAGE3(CommandsCfg):
    """階段3：中等目標"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(25.0, 25.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(1.0, 4.0),
            pos_y=(-2.0, 2.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


##
# 終止條件
##


@configclass
class SimpleV2TerminationsCfg(TerminationsCfg):
    """Simple v2 終止條件"""
    
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 0.8},
    )


##
# 三個階段的環境配置
##


@configclass
class LocalPlannerEnvCfg_SIMPLE_V2_STAGE1(LocalPlannerEnvCfg):
    """【Simple v2 - 階段1】和DEBUG一樣，驗證可重現
    
    目標：驗證DEBUG的成功可以重現
    預期：300 iterations 後成功率 > 15%
    """
    
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SimpleV2RewardsCfg = SimpleV2RewardsCfg()
    commands: SimpleV2CommandsCfg_STAGE1 = SimpleV2CommandsCfg_STAGE1()
    terminations: SimpleV2TerminationsCfg = SimpleV2TerminationsCfg()
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=16, env_spacing=10.0)
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        print("🎯 [Simple v2 - Stage 1] 極近目標（驗證DEBUG成功）")


@configclass
class LocalPlannerEnvCfg_SIMPLE_V2_STAGE1_5(LocalPlannerEnvCfg):
    """【Simple v2 - 階段1.5】過渡階段（NEW！）
    
    目的：在Stage 1和Stage 2之間建立平滑過渡
    
    進階條件：Stage 1 成功率 > 15%
    預期：500 iterations 後成功率 > 12%
    
    變化：
    - 目標距離：0.3-1.0米 → 0.3-1.3米（+30%）
    - 環境數量：16 → 32（+100%）
    - Episode時間：15秒 → 16秒（+7%）
    """
    
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SimpleV2RewardsCfg = SimpleV2RewardsCfg()
    commands: SimpleV2CommandsCfg_STAGE1_5 = SimpleV2CommandsCfg_STAGE1_5()
    terminations: SimpleV2TerminationsCfg = SimpleV2TerminationsCfg()
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=32, env_spacing=10.0)
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0
        print("🎯 [Simple v2 - Stage 1.5] 過渡階段（0.3-1.3米）")


@configclass
class LocalPlannerEnvCfg_SIMPLE_V2_STAGE2(LocalPlannerEnvCfg):
    """【Simple v2 - 階段2】稍微增加難度（已再次優化）
    
    進階條件：Stage 1.5 成功率 > 12%
    預期：500 iterations 後成功率 > 8%
    
    優化（v2）：
    - 環境數量從64降到48（減少學習噪音）
    - 目標距離保持0.4-1.5米
    - Episode時間保持18秒
    """
    
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SimpleV2RewardsCfg = SimpleV2RewardsCfg()
    commands: SimpleV2CommandsCfg_STAGE2 = SimpleV2CommandsCfg_STAGE2()
    terminations: SimpleV2TerminationsCfg = SimpleV2TerminationsCfg()
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=48, env_spacing=12.0)  # 🔧🔧 從64降到48
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 18.0
        print("🎯 [Simple v2 - Stage 2] 近目標（0.4-1.5米，環境數48）")


@configclass
class LocalPlannerEnvCfg_SIMPLE_V2_STAGE3(LocalPlannerEnvCfg):
    """【Simple v2 - 階段3】中等難度
    
    進階條件：Stage 2 成功率 > 20%
    預期：1000 iterations 後成功率 > 30%
    """
    
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SimpleV2RewardsCfg = SimpleV2RewardsCfg()
    commands: SimpleV2CommandsCfg_STAGE3 = SimpleV2CommandsCfg_STAGE3()
    terminations: SimpleV2TerminationsCfg = SimpleV2TerminationsCfg()
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=256, env_spacing=12.0)
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 25.0
        print("🎯 [Simple v2 - Stage 3] 中等目標（1.0-4.0米）")

