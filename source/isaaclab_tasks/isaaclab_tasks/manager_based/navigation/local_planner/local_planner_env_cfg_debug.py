# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
【DEBUG 版本】極簡配置 - 用於診斷問題

策略：
1. 移除所有複雜獎勵，只保留最基本的
2. 超近的目標（0.3-1.0米）
3. 超寬鬆的成功閾值（1.5米）
4. 極少環境（16個）
5. 短 episode（15秒）

目的：
- 如果這個都學不會，說明環境本身有問題
- 如果能學會，說明是獎勵設計問題
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


@configclass
class DebugRewardsCfg:
    """【DEBUG】極簡獎勵 - 只用最基本的"""
    
    # 主要獎勵：接近目標的進度（已修正的版本）
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal_reward,
        weight=50.0,  # 超大權重，最強引導
        params={"command_name": "goal_command"},
    )
    
    # 到達獎勵（超大，超寬鬆）
    reached_goal = RewTerm(
        func=mdp.reached_goal_reward,
        weight=500.0,  # 超大獎勵
        params={"command_name": "goal_command", "threshold": 1.5},  # 1.5米就算成功
    )
    
    # 小懲罰：防止靜止
    standstill_penalty = RewTerm(
        func=mdp.standstill_penalty,
        weight=-0.1,
    )


@configclass
class DebugCommandsCfg(CommandsCfg):
    """【DEBUG】超近的目標"""
    
    goal_command = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="chassis_link",
        resampling_time_range=(15.0, 15.0),  # 15秒重新生成（和episode時間一致）
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 1.0),  # 極度近！
            pos_y=(-0.5, 0.5),  # 極度窄！
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class DebugTerminationsCfg(TerminationsCfg):
    """【DEBUG】超寬鬆的終止條件"""
    
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 1.5},  # 超寬鬆！
    )


@configclass
class LocalPlannerEnvCfg_DEBUG(LocalPlannerEnvCfg):
    """【DEBUG 版本】極簡配置 - 診斷用
    
    如果這個版本能達到 50%+ 成功率：
    → 說明環境正常，是獎勵設計問題
    
    如果這個版本成功率還是 < 10%：
    → 說明環境本身有問題（LiDAR、動作、物理等）
    
    訓練指令：
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
        --num_envs 16 \
        --max_iterations 500 \
        --headless
    
    預期結果：
    - 如果正常：100 iterations 後成功率應 > 30%
    - 如果異常：100 iterations 後成功率仍 < 5%
    """
    
    # 極簡配置
    observations: ObservationsCfg = ObservationsCfg()
    rewards: DebugRewardsCfg = DebugRewardsCfg()
    commands: DebugCommandsCfg = DebugCommandsCfg()
    terminations: DebugTerminationsCfg = DebugTerminationsCfg()
    
    # 極少環境
    scene: LocalPlannerSceneCfg = LocalPlannerSceneCfg(num_envs=16, env_spacing=10.0)
    
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # 短 episode
        self.episode_length_s = 15.0
        
        print("=" * 80)
        print("🔬 [DEBUG 模式] 極簡診斷配置")
        print("=" * 80)
        print("⚠️  這是診斷版本，用於測試環境是否正常工作")
        print("")
        print("配置：")
        print("   - 目標距離: 0.3-1.0 米（極度近！）")
        print("   - 成功閾值: 1.5 米（超寬鬆！）")
        print("   - 並行環境: 16（極少）")
        print("   - Episode 時間: 15 秒（極短）")
        print("   - 獎勵: 只用距離負值（最簡單）")
        print("")
        print("目標：")
        print("   - 100 iterations 後成功率應 > 30%")
        print("   - 如果達到 → 環境正常，是獎勵設計問題")
        print("   - 如果沒達到 → 環境本身有問題")
        print("=" * 80)

