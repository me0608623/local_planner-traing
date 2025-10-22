# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GUI 模式專用 Nova Carter 本地路徑規劃環境配置

基於重要發現：PhysX tensor device 錯誤只在 GUI 模式出現，Headless 模式完全正常。
此配置專門針對 GUI 模式的特殊需求進行優化。

關鍵差異:
- GUI 模式自動啟用 GPU 物理管線
- 視覺渲染觸發更複雜的張量設備管理
- 需要強制設備一致性和增強的 GPU 緩衝區
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

# 基於Isaac Sim 5.0兼容性的動態導入
try:
    from isaacsim.core.api.utils.torch import set_cuda_device
    print("✅ [GUI Mode] 使用新版模組 isaacsim.core.api.utils.torch")
except ImportError:
    try:
        from omni.isaac.core.utils.torch import set_cuda_device
        print("⚠️ [GUI Mode] 回退到舊版模組 omni.isaac.core.utils.torch")
    except ImportError:
        print("❌ [GUI Mode] 無法導入 Isaac Sim 相關模組，使用純 PyTorch 替代方案")
        def set_cuda_device(device: str):
            if torch.cuda.is_available():
                if isinstance(device, str) and device.startswith("cuda:"):
                    device_id = int(device.split(":")[1])
                    torch.cuda.set_device(device_id)
                else:
                    torch.cuda.set_device(0)
            else:
                print("CUDA 不可用，無法設定設備。")

# 導入基礎配置
from .local_planner_env_cfg import LocalPlannerEnvCfg


@configclass
class LocalPlannerEnvCfg_GUI_FIXED(LocalPlannerEnvCfg):
    """GUI 模式專用的 Nova Carter 本地路徑規劃環境配置
    
    專門解決 GUI 模式下的 PhysX tensor device 不匹配問題：
    
    問題原因:
    - GUI 模式自動啟用 GPU 物理管線
    - 視覺渲染需求觸發 GPU 張量計算
    - 但某些張量（如root velocity）仍在CPU建立
    - 導致getRootVelocities函數設備不匹配
    
    解決策略:
    - 強制所有模擬組件使用統一GPU設備
    - 大幅增加 GPU 緩衝區容量
    - 啟用 GPU 動力學計算
    - 優化視覺渲染相關設定
    """
    
    def __post_init__(self):
        """GUI 模式專用的後處理配置"""
        # 調用父類的後處理
        super().__post_init__()
        
        print("🎮 [GUI Mode] 正在配置 GUI 模式專用的 PhysX 設定...")
        
        # 🔧 核心修復 1: 強制 GPU 設備一致性
        self.sim.device = "cuda:0"
        print(f"✅ [GUI Mode] 模擬設備設定為: {self.sim.device}")
        
        # 🔧 核心修復 2: 明確啟用 GPU 物理計算
        self.sim.physx.use_gpu = True
        print("✅ [GUI Mode] 啟用 GPU 物理計算")
        
        # 🔧 核心修復 3: GUI 模式專用的大容量 GPU 緩衝區
        # GUI 模式的視覺渲染需求較高，需要更大的緩衝區
        self.sim.physx.gpu_max_rigid_contact_count = 4096 * 1024  # 4M (比標準配置大4倍)
        self.sim.physx.gpu_max_rigid_patch_count = 2048 * 1024   # 2M (比標準配置大4倍)
        self.sim.physx.gpu_found_lost_pairs_capacity = 2048 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 512 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2048 * 1024
        print("✅ [GUI Mode] 設定大容量 GPU 緩衝區 (4M rigid contacts)")
        
        # 🔧 GUI 模式專用設定
        if hasattr(self.sim.physx, 'enable_gpu_dynamics'):
            self.sim.physx.enable_gpu_dynamics = True
            print("✅ [GUI Mode] 啟用 GPU 動力學計算")
            
        # 優化 GUI 模式的性能設定
        if hasattr(self.sim.physx, 'enable_enhanced_determinism'):
            self.sim.physx.enable_enhanced_determinism = False  # GUI 模式優先性能
            print("✅ [GUI Mode] 優化性能設定（降低確定性以換取流暢度）")
            
        # 🔧 張量設備一致性檢查
        try:
            # 設定 CUDA 設備
            set_cuda_device("cuda:0")
            print("✅ [GUI Mode] CUDA 設備設定完成")
        except Exception as e:
            print(f"⚠️ [GUI Mode] CUDA 設備設定警告: {e}")
            
        # 🔧 環境數量調整 (GUI 模式建議較少環境)
        # GUI 模式渲染開銷較大，建議減少並行環境數量
        if not hasattr(self, '_gui_env_count_adjusted'):
            original_num_envs = getattr(self.scene, 'num_envs', 4)
            suggested_num_envs = min(original_num_envs, 2)  # GUI 模式最多2個環境
            if hasattr(self.scene, 'num_envs'):
                self.scene.num_envs = suggested_num_envs
                print(f"✅ [GUI Mode] 調整環境數量: {original_num_envs} → {suggested_num_envs}")
            self._gui_env_count_adjusted = True
            
        print("🎮 [GUI Mode] GUI 模式專用配置完成！")
        print("💡 [GUI Mode] 如果仍遇到問題，建議使用 Headless 模式進行訓練")


@configclass  
class LocalPlannerEnvCfg_GUI_SIMPLE(LocalPlannerEnvCfg_GUI_FIXED):
    """GUI 模式的簡化版本（單環境，最大兼容性）"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 強制單環境配置（最高穩定性）
        self.scene.num_envs = 1
        print("🎮 [GUI Simple] 強制單環境模式 (最高穩定性)")
        
        # 進一步減少複雜度
        if hasattr(self.scene.terrain, 'terrain_generator'):
            # 簡化地形（如果適用）
            pass
            
        print("🎮 [GUI Simple] GUI 簡化模式配置完成")


# 用於與Headless模式對比的診斷配置
@configclass
class LocalPlannerEnvCfg_DIAGNOSTIC(LocalPlannerEnvCfg):
    """診斷專用配置 - 用於對比GUI vs Headless差異"""
    
    def __post_init__(self):
        super().__post_init__()
        
        print("🔍 [Diagnostic] 診斷模式配置")
        print(f"🔍 [Diagnostic] 模擬設備: {self.sim.device}")
        print(f"🔍 [Diagnostic] GPU 物理: {self.sim.physx.use_gpu}")
        
        # 添加診斷回調
        self._enable_diagnostic_mode = True
