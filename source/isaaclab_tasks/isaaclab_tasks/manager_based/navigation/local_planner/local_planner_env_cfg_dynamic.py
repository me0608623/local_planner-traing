# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
【動態障礙物版本】基於DEBUG成功配置，加入動態障礙物

策略：
1. 保持DEBUG的極簡獎勵設計（已驗證37.5%成功率）
2. 加入少量動態障礙物（1-2個移動球體）
3. 目標距離保持0.3-1.0米（已證明有效）
4. 16環境（已證明穩定）
"""

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from .local_planner_env_cfg import (
    LocalPlannerEnvCfg,
    LocalPlannerSceneCfg,
    ObservationsCfg,
    CommandsCfg,
    ActionsCfg,
    TerminationsCfg,
    EventCfg,
)
from .local_planner_env_cfg_simple_v2 import (
    SimpleV2RewardsCfg,
    SimpleV2CommandsCfg_STAGE1,
    SimpleV2TerminationsCfg,
)
import isaaclab_tasks.manager_based.navigation.local_planner.mdp as mdp


##
# 場景配置：加入動態障礙物
##


@configclass
class DynamicSceneCfg(LocalPlannerSceneCfg):
    """場景配置：加入少量動態障礙物"""
    
    # 動態障礙物：移動的球體
    dynamic_obstacles = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DynamicObstacle",
        spawn=sim_utils.SphereCfg(
            radius=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),  # 橘色
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.4),
            lin_vel=(0.3, 0.3, 0.0),  # 初始速度
        ),
    )


##
# 事件配置：動態障礙物運動
##


@configclass
class DynamicEventCfg(EventCfg):
    """事件配置：動態障礙物的隨機運動"""
    
    # 繼承原有的reset事件
    # ...原有事件保持不變
    
    # 新增：定期推動動態障礙物
    push_dynamic_obstacles = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),  # 每3-5秒改變一次速度
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("dynamic_obstacles"),
        },
    )


##
# 動態障礙物環境配置
##


@configclass
class LocalPlannerEnvCfg_DYNAMIC(LocalPlannerEnvCfg):
    """【動態障礙物版本】基於DEBUG成功配置
    
    基於DEBUG配置（37.5%成功率）的擴展版本：
    - 保持極簡獎勵設計（3項）
    - 保持0.3-1.0米目標距離
    - 保持16環境
    - 新增：1個動態障礙物（移動球體）
    
    預期：
    - 初期成功率可能下降到20-30%（因為多了移動障礙物）
    - 訓練5000 iterations後應穩定在25-35%
    
    訓練指令：
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Navigation-LocalPlanner-Dynamic-v0 \
        --num_envs 16 \
        --max_iterations 5000 \
        --headless
    
    或從靜態版本模型繼續：
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Navigation-LocalPlanner-Dynamic-v0 \
        --num_envs 16 \
        --max_iterations 3000 \
        --load_run logs/rsl_rl/local_planner_carter/[DEBUG-5000iter目錄] \
        --checkpoint model_4999.pt \
        --headless
    """
    
    # 使用極簡獎勵（和DEBUG一樣）
    observations: ObservationsCfg = ObservationsCfg()
    rewards: SimpleV2RewardsCfg = SimpleV2RewardsCfg()
    commands: SimpleV2CommandsCfg_STAGE1 = SimpleV2CommandsCfg_STAGE1()
    terminations: SimpleV2TerminationsCfg = SimpleV2TerminationsCfg()
    
    # 加入動態障礙物的場景
    scene: DynamicSceneCfg = DynamicSceneCfg(num_envs=16, env_spacing=12.0)
    
    actions: ActionsCfg = ActionsCfg()
    events: DynamicEventCfg = DynamicEventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        
        print("=" * 80)
        print("🔥 [動態障礙物版本] 訓練配置已載入")
        print("=" * 80)
        print("📚 基於DEBUG成功配置（37.5%成功率）")
        print("")
        print("新增特性：")
        print("   ✅ 1個動態障礙物（移動球體）")
        print("   ✅ 每3-5秒改變移動方向")
        print("   ✅ 保持極簡獎勵設計（3項）")
        print("   ✅ 保持0.3-1.0米目標距離")
        print("")
        print("預期：")
        print("   - 初期成功率：20-30%（比靜態版本略低）")
        print("   - 5000 iterations後：25-35%")
        print("=" * 80)

