# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
【PCCBF 簡化版】只使用 CBF 獎勵，不使用預測觀測

適合初次測試，等穩定後再加入預測功能
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 導入原始配置作為基礎
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
# 🎁 獎勵函數配置（只用 CBF，不用預測）
##


@configclass
class PCCBFSimpleRewardsCfg:
    """【PCCBF 簡化版】獎勵函數配置
    
    只使用 CBF 安全獎勵，不使用預測（更穩定）
    """
    
    # ✅ 正向獎勵：接近目標（主要驅動力）
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=30.0,  # 🔧 從15.0增加到30.0，加強學習信號
        params={"command_name": "goal_command"},
    )
    
    # ✅ 正向獎勵：到達目標
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=200.0,  # 🔧🔧 從100增加到200，大幅獎勵成功
        params={"command_name": "goal_command", "threshold": 0.8},  # 🔧🔧🔧 放寬到0.8米
    )
    
    # 🔥 新增：近距離塑形獎勵（解決"最後一公里"問題）
    near_goal_shaping = RewTerm(
        func=mdp.near_goal_shaping,
        weight=15.0,  # 中等權重，引導最後逼近
        params={"command_name": "goal_command", "radius": 2.0},  # 2米內開始塑形
    )
    
    # 🔥 CBF 安全獎勵（PCCBF 核心，但不用預測版本）
    cbf_safety = RewTerm(
        func=mdp.cbf_safety_reward,
        weight=8.0,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "safe_distance": 1.5,
            "critical_distance": 0.5,
        },
    )
    
    # ❌ 懲罰：碰撞
    collision_penalty = RewTerm(
        func=mdp.collision_penalty,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "collision_threshold": 0.3},
    )
    
    # ❌ 懲罰：過大的角速度
    ang_vel_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # ❌ 懲罰：靜止不動（權重降低）
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=-0.05,
    )


##
# 📝 簡化的目標配置
##


@configclass
class PCCBFSimpleTerminationsCfg(TerminationsCfg):
    """【簡化版】終止條件配置（放寬成功閾值）"""
    
    # 到達目標（放寬到0.8米，與獎勵一致）
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 0.8},
    )


@configclass
class PCCBFSimpleCommandsCfg(CommandsCfg):
    """【簡化版】更近的目標距離"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 2.0),  # 🔧🔧 再次縮短：從1-3米改為0.5-2米（非常近！）
            pos_y=(-1.5, 1.5),  # 🔧🔧 再次縮小：從±2米改為±1.5米
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


##
# 🎯 PCCBF 簡化版環境
##


@configclass
class LocalPlannerEnvCfg_PCCBF_SIMPLE(LocalPlannerEnvCfg):
    """【PCCBF 簡化版】只使用 CBF 獎勵，不使用預測觀測
    
    這個版本更穩定，適合初次測試：
    - ✅ 使用修正後的 progress_to_goal（已修復負值問題）
    - ✅ 使用 CBF 安全獎勵（數學保證的安全約束）
    - ❌ 不使用預測觀測（避免複雜度）
    
    等這個版本穩定運行後，再嘗試帶預測的完整版本。
    
    訓練指令：
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Navigation-LocalPlanner-PCCBF-Simple-v0 \
        --num_envs 256 \
        --max_iterations 1000 \
        --headless
    
    預期結果：
    - Mean reward: 從負值逐漸上升到 +20 ~ +50
    - 成功率: 500 iterations 後應達到 20-30%
    - CBF safety: 應為正值，表示學會安全
    """
    
    # 使用原始觀測空間（不加預測）
    observations: ObservationsCfg = ObservationsCfg()
    
    # 使用 PCCBF 的 CBF 獎勵
    rewards: PCCBFSimpleRewardsCfg = PCCBFSimpleRewardsCfg()
    
    # 使用簡化的目標配置
    commands: PCCBFSimpleCommandsCfg = PCCBFSimpleCommandsCfg()
    
    # 減少環境數量（初次測試）
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=256, env_spacing=12.0)
    
    # 保留原始配置
    actions: ActionsCfg = ActionsCfg()
    terminations: PCCBFSimpleTerminationsCfg = PCCBFSimpleTerminationsCfg()  # 🔧 使用放寬版本
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        """後處理：設定模擬參數"""
        super().__post_init__()
        
        # 🔧🔧🔧 Episode時間改為35秒（給Agent足夠時間學習）
        self.episode_length_s = 35.0
        
        print("=" * 80)
        print("🚀 [PCCBF 簡化版] 訓練配置已載入")
        print("=" * 80)
        print("📚 特性：")
        print("   ✅ 修正後的 progress_to_goal 獎勵")
        print("   ✅ CBF 安全約束獎勵")
        print("   ❌ 暫不使用預測觀測（為了穩定性）")
        print("")
        print("🎯 環境設定：")
        print("   - 目標距離: 2-5 米")
        print("   - 並行環境: 256")
        print("   - Episode 時間: 40 秒")
        print("")
        print("📊 預期結果：")
        print("   - 500 iterations 後成功率 > 20%")
        print("   - Mean reward 逐步從負轉正")
        print("   - CBF safety 應為正值")
        print("=" * 80)

