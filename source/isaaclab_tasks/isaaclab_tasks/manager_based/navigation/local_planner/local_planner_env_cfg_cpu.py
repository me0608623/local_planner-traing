# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Nova Carter 本地規劃器 CPU 模式配置
修復 GPU/CPU 張量設備不匹配問題
"""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .local_planner_env_cfg import LocalPlannerEnvCfg, LocalPlannerEnvCfg_SIMPLE

##
# CPU 模式環境配置（修復設備不匹配）
##

@configclass 
class LocalPlannerEnvCfg_CPU(LocalPlannerEnvCfg):
    """Nova Carter 本地規劃器 - CPU 模式
    
    專門用於修復 PhysX GPU/CPU 張量設備不匹配問題：
    - 錯誤：expected device 0, received device -1
    - 解決：強制所有模擬組件使用 CPU
    """
    
    def __post_init__(self):
        """後處理：設定 CPU 模式模擬參數"""
        super().__post_init__()
        
        # 🔧 核心修復：強制 CPU 模式
        self.sim.device = "cpu"
        
        # 🔧 確保 PhysX 設定與 CPU 模式兼容
        self.sim.physx.use_gpu = False
        
        # 🔧 降低並行環境數量以補償 CPU 性能
        self.scene.num_envs = 16  # 從默認 1024 降低到 16
        
        # 📝 調整其他參數以優化 CPU 性能
        self.decimation = 8  # 增加 decimation 減少計算負擔
        self.sim.dt = 0.02   # 略微增加時間步長
        
        print("🔧 [修復] GPU/CPU 張量不匹配問題")
        print(f"   - 設備模式: {self.sim.device}")
        print(f"   - PhysX GPU: {self.sim.physx.use_gpu}")
        print(f"   - 環境數量: {self.scene.num_envs}")


@configclass
class LocalPlannerEnvCfg_CPU_SIMPLE(LocalPlannerEnvCfg_SIMPLE):
    """Nova Carter 本地規劃器 - CPU 模式簡化版
    
    最小化配置用於測試和調試
    """
    
    def __post_init__(self):
        """後處理：CPU 模式 + 最小化設定"""
        super().__post_init__()
        
        # 🔧 核心修復：強制 CPU 模式
        self.sim.device = "cpu"
        self.sim.physx.use_gpu = False
        
        # 🔧 測試友好的設定
        self.scene.num_envs = 4   # 僅 4 個環境便於觀察
        self.episode_length_s = 10.0  # 短回合用於快速測試
        
        print("🔧 [修復] GPU/CPU 張量不匹配問題 (簡化版)")
        print(f"   - 設備模式: {self.sim.device}")
        print(f"   - 環境數量: {self.scene.num_envs}")
        print(f"   - 回合長度: {self.episode_length_s}s")


##
# 漸進式 GPU 修復嘗試（實驗性）
##

@configclass
class LocalPlannerEnvCfg_GPU_FIXED(LocalPlannerEnvCfg):
    """Nova Carter 本地規劃器 - GPU 修復嘗試版
    
    實驗性：嘗試在 GPU 模式下修復張量不匹配
    如果此配置仍然失敗，請使用 CPU 版本
    """
    
    def __post_init__(self):
        """後處理：GPU 修復嘗試"""
        super().__post_init__()
        
        # 🔧 保持 GPU 但調整緩衝區大小
        self.sim.device = "cuda:0"
        
        # 🔧 增加 GPU 緩衝區大小（可能修復不匹配）
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 256 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        
        # 🔧 減少環境數量降低 GPU 負擔
        self.scene.num_envs = 64
        
        print("🔧 [實驗] GPU 張量匹配修復嘗試")
        print(f"   - 設備模式: {self.sim.device}")
        print(f"   - 環境數量: {self.scene.num_envs}")
        print("   - 如果仍有錯誤，請使用 CPU 版本")


