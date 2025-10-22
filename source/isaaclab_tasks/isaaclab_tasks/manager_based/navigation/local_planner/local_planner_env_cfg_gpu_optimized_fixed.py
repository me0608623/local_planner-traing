# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Nova Carter 本地規劃器 GPU 深度優化配置 (修復版)
實施路線A：全程GPU，但不依賴 omni.isaac.core.utils.torch
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .local_planner_env_cfg import LocalPlannerEnvCfg, LocalPlannerEnvCfg_SIMPLE

##
# GPU 全程優化環境配置 (修復版)
##

@configclass 
class LocalPlannerEnvCfg_GPU_OPTIMIZED_FIXED(LocalPlannerEnvCfg):
    """Nova Carter 本地規劃器 - GPU 深度優化版本 (修復版)
    
    路線A：全程GPU（建議）
    核心策略：確保所有與PhysX tensors API交互的數據都是CUDA tensor
    
    修復版本：
    - 不依賴 omni.isaac.core.utils.torch（解決導入問題）
    - 直接使用 PyTorch 的設備管理
    - 保持GPU優化的核心理念
    """
    
    def __post_init__(self):
        """後處理：GPU 深度優化設定 (修復版)"""
        # 🔧 第一步：使用 PyTorch 直接設定設備（不依賴 omni.isaac.core）
        device_id = 0  # 使用 GPU 0
        
        # 設定 PyTorch 的預設裝置
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
            # 設定預設張量類型為 CUDA
            torch.set_default_dtype(torch.float32)
            # 將預設設備設為 CUDA（如果支援）
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device(device)
            print(f"🔧 [GPU優化-修復版] PyTorch 設備設定: {device}")
        else:
            print("⚠️ [警告] CUDA 不可用，使用 CPU")
            device_id = None
        
        # 調用父類設定
        super().__post_init__()
        
        # 🔧 第二步：強化 GPU 設定（僅在 CUDA 可用時）
        if torch.cuda.is_available():
            self.sim.device = f"cuda:{device_id}"
            
            # 確保 PhysX 完全使用 GPU
            self.sim.physx.use_gpu = True
            if hasattr(self.sim.physx, 'gpu_device_id'):
                self.sim.physx.gpu_device_id = device_id
            
            # 🔧 第三步：大幅增加 GPU 緩衝區容量
            self.sim.physx.gpu_max_rigid_contact_count = 2048 * 1024    # 2M contacts
            self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024     # 1M patches
            self.sim.physx.gpu_found_lost_pairs_capacity = 2048 * 1024 # 2M pairs
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 512 * 1024  # 512K
            self.sim.physx.gpu_total_aggregate_pairs_capacity = 2048 * 1024      # 2M
            
            # 🔧 第四步：啟用進階 GPU 優化
            self.sim.physx.enable_ccd = True  # 連續碰撞檢測
            self.sim.physx.enable_stabilization = True  # 物理穩定化
            if hasattr(self.sim.physx, 'use_gpu_pipeline'):
                self.sim.physx.use_gpu_pipeline = True  # 強制使用 GPU pipeline
            
            # 🔧 第五步：優化環境數量
            original_num_envs = self.scene.num_envs
            self.scene.num_envs = min(original_num_envs, 512)  # 最多512個環境
            
            print("🔧 [GPU深度優化-修復版] 配置完成")
            print(f"   - 設備模式: {self.sim.device}")
            print(f"   - PhysX GPU: {self.sim.physx.use_gpu}")
            print(f"   - 環境數量: {self.scene.num_envs} (原始: {original_num_envs})")
            print(f"   - GPU 緩衝區: {self.sim.physx.gpu_max_rigid_contact_count // 1024}K contacts")
            print(f"   - PyTorch CUDA 可用: {torch.cuda.is_available()}")
        else:
            print("🔧 [GPU深度優化-修復版] CUDA 不可用，使用標準配置")


@configclass
class LocalPlannerEnvCfg_GPU_OPTIMIZED_SIMPLE_FIXED(LocalPlannerEnvCfg_SIMPLE):
    """GPU 深度優化簡化版本 (修復版) - 用於測試和驗證"""
    
    def __post_init__(self):
        """後處理：GPU 優化 + 簡化設定 (修復版)"""
        # 使用 PyTorch 直接設定設備
        device_id = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            torch.set_default_dtype(torch.float32)
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device(f"cuda:{device_id}")
        
        # 調用父類設定
        super().__post_init__()
        
        # GPU 優化設定
        if torch.cuda.is_available():
            self.sim.device = f"cuda:{device_id}"
            self.sim.physx.use_gpu = True
            if hasattr(self.sim.physx, 'gpu_device_id'):
                self.sim.physx.gpu_device_id = device_id
            
            # 適中的緩衝區設定（測試用）
            self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024    
            self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024     
            self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 
            
            # 測試友好的環境數量
            self.scene.num_envs = 32   # 適中數量便於測試
            self.episode_length_s = 20.0  # 較短回合便於快速測試
            
            print("🔧 [GPU深度優化-簡化-修復版] 配置完成")
            print(f"   - 設備模式: {self.sim.device}")
            print(f"   - 環境數量: {self.scene.num_envs}")
            print(f"   - 回合長度: {self.episode_length_s}s")


##
# 張量設備工具函數 (不依賴 omni.isaac.core)
##

def ensure_cuda_tensor_fixed(data, device_id: int = 0):
    """確保數據是 CUDA tensor (修復版)
    
    不依賴 omni.isaac.core，直接使用 PyTorch
    
    Args:
        data: 輸入數據 (numpy array, CPU tensor, 或已經是 CUDA tensor)
        device_id: CUDA 設備 ID
    
    Returns:
        CUDA tensor (如果 CUDA 可用) 或 CPU tensor
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=torch.float32)
    else:
        # numpy array 或其他格式
        return torch.tensor(data, dtype=torch.float32, device=device)


def convert_positions_to_cuda_fixed(coords, orientations=None, device_id: int = 0):
    """將位置和方向數據轉換為 CUDA tensor (修復版)
    
    專門用於 set_world_poses() 等 PhysX API 調用
    不依賴 omni.isaac.core
    
    Args:
        coords: 位置座標 (N, 3) 或 (3,)
        orientations: 四元數方向 (N, 4) 或 (4,)，如果為 None 則使用預設方向
        device_id: CUDA 設備 ID
    
    Returns:
        tuple: (coords_tensor, orientations_tensor)
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    # 轉換位置
    if isinstance(coords, (list, tuple)):
        coords = torch.tensor(coords, dtype=torch.float32, device=device)
    elif isinstance(coords, torch.Tensor):
        coords = coords.to(device, dtype=torch.float32)
    else:
        # numpy array
        coords = torch.tensor(coords, dtype=torch.float32, device=device)
    
    # 確保是 2D tensor (N, 3)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    
    # 轉換方向
    if orientations is None:
        # 預設方向 (無旋轉): [0, 0, 0, 1]
        batch_size = coords.shape[0]
        orientations = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(batch_size, 1)
    else:
        if isinstance(orientations, (list, tuple)):
            orientations = torch.tensor(orientations, dtype=torch.float32, device=device)
        elif isinstance(orientations, torch.Tensor):
            orientations = orientations.to(device, dtype=torch.float32)
        else:
            # numpy array
            orientations = torch.tensor(orientations, dtype=torch.float32, device=device)
        
        # 確保是 2D tensor (N, 4)
        if orientations.dim() == 1:
            orientations = orientations.unsqueeze(0)
    
    return coords, orientations


##
# 使用範例 (修復版)
##

class GPUOptimizedUsageExampleFixed:
    """GPU 優化使用範例 (修復版)"""
    
    @staticmethod
    def example_set_world_poses():
        """正確的 set_world_poses 使用方式 (修復版)"""
        # ✅ 正確：使用修復版的 CUDA tensor 轉換
        coords, orient = convert_positions_to_cuda_fixed(
            coords=[[-8.0, 32.0, -2.3]], 
            orientations=[[0.0, 0.0, 0.0, 1.0]],
            device_id=0
        )
        # self.cube_xform1.set_world_poses(coords, orient)  # 這是正確的！
        
    @staticmethod
    def example_device_check():
        """檢查設備設定範例"""
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"當前 CUDA 設備: {torch.cuda.current_device()}")
            print(f"CUDA 設備名稱: {torch.cuda.get_device_name()}")
        else:
            print("使用 CPU 模式")
