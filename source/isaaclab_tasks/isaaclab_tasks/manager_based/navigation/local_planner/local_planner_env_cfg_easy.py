# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
簡化版 Nova Carter 本地路徑規劃環境 - 易於訓練

基於訓練診斷工具的建議，此配置專門設計為：
1. 減少環境難度，提高初始成功率
2. 調整獎勵權重，提供更好的學習引導
3. 增加 episode 時間，給予更多學習機會
4. 簡化觀測和動作空間

適用場景：
- 初次訓練Nova Carter導航任務
- 快速驗證環境設置是否正確
- Curriculum Learning 的第一階段
"""

from __future__ import annotations

from isaaclab.utils import configclass

# 導入基礎配置
from .local_planner_env_cfg import (
    LocalPlannerEnvCfg,
    LocalPlannerSceneCfg,
    RewardsCfg,
    CommandsCfg,
)

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab_tasks.manager_based.navigation.local_planner import mdp


@configclass
class EasyRewardsCfg(RewardsCfg):
    """簡化的獎勵函數配置 - 強調引導性"""
    
    # 🔧 增加接近目標的獎勵權重（從 10.0 增加到 20.0）
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=20.0,  # 更強的引導
        params={"command_name": "goal_command"},
    )
    
    # 🔧 保持到達目標的大獎勵
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=200.0,  # 增加到 200.0
        params={"command_name": "goal_command", "threshold": 0.8},  # 放寬閾值
    )
    
    # 🔧 減少障礙物接近懲罰（從 -5.0 減少到 -2.0）
    obstacle_proximity_penalty = RewTerm(
        func=mdp.obstacle_proximity_penalty,
        weight=-2.0,  # 減少懲罰，鼓勵探索
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_distance": 0.8},
    )
    
    # 🔧 減少碰撞懲罰（從 -50.0 減少到 -20.0）
    collision_penalty = RewTerm(
        func=mdp.collision_penalty,
        weight=-20.0,  # 減少懲罰
        params={"sensor_cfg": SceneEntityCfg("lidar"), "collision_threshold": 0.3},
    )
    
    # 🔧 減少角速度懲罰
    ang_vel_penalty = RewTerm(
        func=mdp.base_angular_velocity_penalty,
        weight=-0.01,  # 非常小的懲罰
        params={"asset_cfg": SceneEntityCfg("nova_carter")},
    )
    
    # 🔧 減少靜止懲罰
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=-0.5,  # 減少懲罰
        params={"asset_cfg": SceneEntityCfg("nova_carter"), "threshold": 0.1},
    )


@configclass
class EasyCommandsCfg(CommandsCfg):
    """簡化的命令配置 - 更近的目標距離"""
    
    goal_command = CommandsCfg.goal_command
    
    def __post_init__(self):
        super().__post_init__()
        # 🔧 縮短目標距離範圍（從原本可能的 5-10m 縮短到 2-5m）
        self.goal_command.ranges.pos_x = (2.0, 5.0)
        self.goal_command.ranges.pos_y = (-3.0, 3.0)


@configclass
class EasySceneCfg(LocalPlannerSceneCfg):
    """簡化的場景配置 - 更少障礙物"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 🔧 減少靜態障礙物數量
        if hasattr(self, 'static_obstacles'):
            self.static_obstacles.num_obstacles = 3  # 從更多減少到 3 個
            
        # 🔧 移除或減少動態障礙物
        if hasattr(self, 'dynamic_obstacles'):
            self.dynamic_obstacles.num_obstacles = 0  # 完全移除動態障礙物


@configclass
class LocalPlannerEnvCfg_EASY(LocalPlannerEnvCfg):
    """
    簡化版 Nova Carter 本地路徑規劃環境配置
    
    特點：
    1. 更近的目標距離（2-5米）
    2. 更少的障礙物（3個靜態，0個動態）
    3. 更大的獎勵引導權重
    4. 更小的懲罰值
    5. 更長的 episode 時間（40秒）
    6. 更寬鬆的成功判定（0.8米）
    
    訓練建議：
    - 並行環境數：4-8
    - 訓練迭代數：500-1000
    - 期望成功率：> 30% after 500 iterations
    """
    
    # 使用簡化的配置
    rewards: EasyRewardsCfg = EasyRewardsCfg()
    commands: EasyCommandsCfg = EasyCommandsCfg()
    scene: EasySceneCfg = EasySceneCfg(num_envs=512, env_spacing=12.0)
    
    def __post_init__(self):
        """後處理：設定簡化的模擬參數"""
        super().__post_init__()
        
        # 🔧 增加 episode 時間（從 30秒 增加到 40秒）
        self.episode_length_s = 40.0
        
        print("🎓 [Easy Mode] 使用簡化訓練配置：")
        print("   ✅ 目標距離: 2-5m (較近)")
        print("   ✅ 障礙物: 3個靜態 (較少)")
        print("   ✅ Episode時間: 40秒 (較長)")
        print("   ✅ 獎勵引導: 增強 (20.0x)")
        print("   ✅ 懲罰: 減少 (更寬容)")
        print("   💡 建議: 先用此配置驗證環境，再逐步增加難度")


@configclass
class LocalPlannerEnvCfg_CURRICULUM_STAGE1(LocalPlannerEnvCfg_EASY):
    """Curriculum Learning - 階段 1：最簡單"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 進一步簡化
        self.episode_length_s = 50.0  # 更長時間
        self.scene.num_envs = 256  # 較少環境，更快迭代
        
        # 調整命令範圍 - 非常近的目標
        self.commands.goal_command.ranges.pos_x = (1.5, 3.0)
        self.commands.goal_command.ranges.pos_y = (-2.0, 2.0)
        
        print("📚 [Curriculum Stage 1] 最簡單階段：")
        print("   🎯 目標: 1.5-3m (非常近)")
        print("   ⏱️ 時間: 50秒 (充足)")
        print("   🎮 環境數: 256")


@configclass  
class LocalPlannerEnvCfg_CURRICULUM_STAGE2(LocalPlannerEnvCfg_EASY):
    """Curriculum Learning - 階段 2：中等難度"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 中等難度
        self.episode_length_s = 35.0
        self.scene.num_envs = 512
        
        # 增加目標距離
        self.commands.goal_command.ranges.pos_x = (3.0, 6.0)
        self.commands.goal_command.ranges.pos_y = (-4.0, 4.0)
        
        # 增加障礙物
        if hasattr(self.scene, 'static_obstacles'):
            self.scene.static_obstacles.num_obstacles = 5
        
        print("📚 [Curriculum Stage 2] 中等難度：")
        print("   🎯 目標: 3-6m (中等)")
        print("   🚧 障礙物: 5個")
        print("   ⏱️ 時間: 35秒")


@configclass
class LocalPlannerEnvCfg_CURRICULUM_STAGE3(LocalPlannerEnvCfg):
    """Curriculum Learning - 階段 3：完整難度"""
    
    def __post_init__(self):
        super().__post_init__()
        
        print("📚 [Curriculum Stage 3] 完整難度：")
        print("   🎯 使用原始配置")
        print("   💡 應該在Stage 1和2成功後使用")
