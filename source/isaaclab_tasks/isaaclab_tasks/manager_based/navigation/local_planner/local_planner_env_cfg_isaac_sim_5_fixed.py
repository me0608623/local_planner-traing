# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Nova Carter 本地規劃器 Isaac Sim 5.0 完全兼容配置
解決 Isaac Sim 5.0 模組重構問題：omni.isaac.* -> isaacsim.*
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .local_planner_env_cfg import LocalPlannerEnvCfg, LocalPlannerEnvCfg_SIMPLE

##
# Isaac Sim 5.0 完全兼容環境配置
##

@configclass 
class LocalPlannerEnvCfg_ISAAC_SIM_5_FIXED(LocalPlannerEnvCfg):
    """Nova Carter 本地規劃器 - Isaac Sim 5.0 完全兼容版本
    
    解決 Isaac Sim 5.0 模組重構問題：
    - omni.isaac.core -> isaacsim.core.api
    - omni.isaac.core.utils.torch -> isaacsim.core.api.utils.torch
    
    採用多重兼容策略：
    1. 優先使用新版 isaacsim.* 模組
    2. 回退到舊版 omni.isaac.* 模組  
    3. 最終使用純 PyTorch 方法（最可靠）
    """
    
    def __post_init__(self):
        """後處理：Isaac Sim 5.0 兼容GPU優化設定"""
        
        # 🔧 第一步：Isaac Sim 5.0 兼容設備設定
        device_set = False
        
        print("🔧 [Isaac Sim 5.0 兼容] 開始設定CUDA設備...")
        
        # 方法1: 新版 Isaac Sim 5.0 模組（isaacsim.*）
        new_modules = [
            'isaacsim.core.api.utils.torch',
            'isaacsim.core.utils.torch',
            'isaacsim.core.api.torch',
            'isaacsim.utils.torch'
        ]
        
        for module_name in new_modules:
            try:
                # 動態導入模組
                module = __import__(module_name, fromlist=['set_cuda_device'])
                set_cuda_device = getattr(module, 'set_cuda_device', None)
                if set_cuda_device and callable(set_cuda_device):
                    set_cuda_device(0)
                    print(f"✅ [Isaac Sim 5.0] 使用新版模組 {module_name}")
                    device_set = True
                    break
            except (ImportError, AttributeError) as e:
                continue
        
        # 方法2: 舊版兼容（omni.isaac.*）
        if not device_set:
            try:
                from omni.isaac.core.utils.torch import set_cuda_device
                set_cuda_device(0)
                print("✅ [兼容模式] 使用舊版 omni.isaac.core.utils.torch")
                device_set = True
            except ImportError:
                pass
        
        # 方法3: 純 PyTorch 方法（最可靠的修復版核心）
        if not device_set:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                # Isaac Sim 5.0 建議的預設設備設定
                if hasattr(torch, 'set_default_device'):
                    torch.set_default_device('cuda:0')
                print("✅ [修復版核心] 使用 PyTorch 直接設定 CUDA 設備")
                device_set = True
            else:
                print("⚠️ [警告] CUDA 不可用，將使用 CPU 模式")
        
        # 調用父類設定
        super().__post_init__()
        
        # 🔧 第二步：強化 GPU 設定（Isaac Sim 5.0 優化）
        if torch.cuda.is_available() and device_set:
            self.sim.device = "cuda:0"
            
            # 確保 PhysX 完全使用 GPU
            self.sim.physx.use_gpu = True
            if hasattr(self.sim.physx, 'gpu_device_id'):
                self.sim.physx.gpu_device_id = 0
            
            # 🔧 第三步：Isaac Sim 5.0 優化的 GPU 緩衝區設定
            # 針對 Isaac Sim 5.0 的 PhysX 優化
            self.sim.physx.gpu_max_rigid_contact_count = 2048 * 1024    # 2M contacts
            self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024     # 1M patches
            self.sim.physx.gpu_found_lost_pairs_capacity = 2048 * 1024 # 2M pairs
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 512 * 1024  # 512K
            self.sim.physx.gpu_total_aggregate_pairs_capacity = 2048 * 1024      # 2M
            
            # 🔧 第四步：Isaac Sim 5.0 進階優化
            self.sim.physx.enable_ccd = True  # 連續碰撞檢測
            self.sim.physx.enable_stabilization = True  # 物理穩定化
            if hasattr(self.sim.physx, 'use_gpu_pipeline'):
                self.sim.physx.use_gpu_pipeline = True
            
            # 🔧 第五步：環境數量優化
            original_num_envs = self.scene.num_envs
            self.scene.num_envs = min(original_num_envs, 512)  # Isaac Sim 5.0 推薦最大值
            
            print("🎉 [Isaac Sim 5.0 兼容] GPU優化配置完成")
            print(f"   - Isaac Sim 版本: 5.0 (模組重構兼容)")
            print(f"   - 設備模式: {self.sim.device}")
            print(f"   - PhysX GPU: {self.sim.physx.use_gpu}")
            print(f"   - 環境數量: {self.scene.num_envs} (原始: {original_num_envs})")
            print(f"   - GPU 緩衝區: {self.sim.physx.gpu_max_rigid_contact_count // 1024}K contacts")
        else:
            # CPU 模式回退
            self.sim.device = "cpu"
            self.sim.physx.use_gpu = False
            self.scene.num_envs = min(self.scene.num_envs, 64)  # CPU模式限制
            print("🔧 [Isaac Sim 5.0 兼容] 使用 CPU 模式")


@configclass
class LocalPlannerEnvCfg_ISAAC_SIM_5_SIMPLE(LocalPlannerEnvCfg_SIMPLE):
    """Isaac Sim 5.0 兼容簡化版本 - 用於快速測試"""
    
    def __post_init__(self):
        """後處理：Isaac Sim 5.0 兼容簡化設定"""
        
        # 使用相同的兼容設備設定邏輯
        device_set = False
        
        # 嘗試新版模組
        for module_name in ['isaacsim.core.api.utils.torch', 'isaacsim.core.utils.torch']:
            try:
                module = __import__(module_name, fromlist=['set_cuda_device'])
                set_cuda_device = getattr(module, 'set_cuda_device', None)
                if set_cuda_device:
                    set_cuda_device(0)
                    device_set = True
                    break
            except (ImportError, AttributeError):
                continue
        
        # 回退方法
        if not device_set and torch.cuda.is_available():
            torch.cuda.set_device(0)
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device('cuda:0')
            device_set = True
        
        # 調用父類設定
        super().__post_init__()
        
        # 簡化的GPU優化
        if torch.cuda.is_available() and device_set:
            self.sim.device = "cuda:0"
            self.sim.physx.use_gpu = True
            
            # 適中的緩衝區設定（測試用）
            self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024    
            self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024     
            self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 
            
            # 測試友好的設定
            self.scene.num_envs = 32   # 適中數量便於測試
            self.episode_length_s = 20.0  # 較短回合便於快速測試
            
            print("🎉 [Isaac Sim 5.0 簡化版] 配置完成")
            print(f"   - 設備模式: {self.sim.device}")
            print(f"   - 環境數量: {self.scene.num_envs}")
            print(f"   - 回合長度: {self.episode_length_s}s")
        else:
            self.sim.device = "cpu"
            self.sim.physx.use_gpu = False
            self.scene.num_envs = 16
            print("🔧 [Isaac Sim 5.0 簡化版] 使用 CPU 模式")


##
# Isaac Sim 5.0 兼容張量工具函數
##

def ensure_cuda_tensor_isaac_sim_5(data, device_id: int = 0):
    """確保數據是 CUDA tensor (Isaac Sim 5.0 完全兼容版本)
    
    採用多重兼容策略解決 Isaac Sim 5.0 模組重構：
    1. 嘗試新版 isaacsim.* 模組
    2. 回退到舊版 omni.isaac.* 模組
    3. 最終使用純 PyTorch 方法
    
    Args:
        data: 輸入數據 (numpy array, CPU tensor, 或已經是 CUDA tensor)
        device_id: CUDA 設備 ID
    
    Returns:
        CUDA tensor (如果 CUDA 可用) 或 CPU tensor
    """
    
    # 方法1: Isaac Sim 5.0 新版模組
    for module_name in ['isaacsim.core.api.utils.torch', 'isaacsim.core.utils.torch']:
        try:
            module = __import__(module_name, fromlist=['tensor_from_numpy_array_to_device'])
            tensor_func = getattr(module, 'tensor_from_numpy_array_to_device', None)
            if tensor_func:
                if isinstance(data, torch.Tensor):
                    return data.cuda(device_id) if torch.cuda.is_available() else data.cpu()
                else:
                    device_name = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
                    return tensor_func(data, device=device_name)
        except (ImportError, AttributeError):
            continue
    
    # 方法2: 舊版兼容
    try:
        from omni.isaac.core.utils.torch import tensor_from_numpy_array_to_device
        if isinstance(data, torch.Tensor):
            return data.cuda(device_id) if torch.cuda.is_available() else data.cpu()
        else:
            device_name = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            return tensor_from_numpy_array_to_device(data, device=device_name)
    except ImportError:
        pass
    
    # 方法3: 純 PyTorch 方法（最可靠，修復版核心）
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=torch.float32)
    else:
        return torch.tensor(data, dtype=torch.float32, device=device)


def convert_positions_to_cuda_isaac_sim_5(coords, orientations=None, device_id: int = 0):
    """將位置和方向數據轉換為 CUDA tensor (Isaac Sim 5.0 完全兼容版本)
    
    專門用於 set_world_poses() 等 PhysX API 調用。
    完全兼容 Isaac Sim 5.0 模組重構。
    
    Args:
        coords: 位置座標 (N, 3) 或 (3,)
        orientations: 四元數方向 (N, 4) 或 (4,)，如果為 None 則使用預設方向
        device_id: CUDA 設備 ID
    
    Returns:
        tuple: (coords_tensor, orientations_tensor)
    """
    # 使用 Isaac Sim 5.0 兼容的張量轉換
    coords_tensor = ensure_cuda_tensor_isaac_sim_5(coords, device_id)
    
    # 確保是 2D tensor (N, 3)
    if coords_tensor.dim() == 1:
        coords_tensor = coords_tensor.unsqueeze(0)
    
    # 轉換方向
    if orientations is None:
        # 預設方向 (無旋轉): [0, 0, 0, 1]
        batch_size = coords_tensor.shape[0]
        device = coords_tensor.device
        orientations_tensor = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], 
            device=device, 
            dtype=torch.float32
        ).repeat(batch_size, 1)
    else:
        orientations_tensor = ensure_cuda_tensor_isaac_sim_5(orientations, device_id)
        if orientations_tensor.dim() == 1:
            orientations_tensor = orientations_tensor.unsqueeze(0)
    
    return coords_tensor, orientations_tensor


##
# Isaac Sim 5.0 使用範例
##

class IsaacSim5UsageExample:
    """Isaac Sim 5.0 兼容使用範例"""
    
    @staticmethod
    def example_device_check():
        """檢查 Isaac Sim 5.0 設備設定"""
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"當前 CUDA 設備: {torch.cuda.current_device()}")
            print(f"CUDA 設備名稱: {torch.cuda.get_device_name()}")
        
        # 測試模組可用性
        modules_to_test = [
            'isaacsim.core.api.utils.torch',
            'isaacsim.core.utils.torch', 
            'omni.isaac.core.utils.torch'
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"✅ {module_name} 可用")
            except ImportError:
                print(f"❌ {module_name} 不可用")
    
    @staticmethod
    def example_tensor_conversion():
        """Isaac Sim 5.0 兼容張量轉換範例"""
        import numpy as np
        
        # 測試數據
        coords = np.array([[-8.0, 32.0, -2.3]])
        
        # 使用 Isaac Sim 5.0 兼容函數
        coords_tensor, orient_tensor = convert_positions_to_cuda_isaac_sim_5(
            coords=coords,
            orientations=None,
            device_id=0
        )
        
        print(f"座標張量: {coords_tensor.device} -> {coords_tensor.shape}")
        print(f"方向張量: {orient_tensor.device} -> {orient_tensor.shape}")
        
        return coords_tensor, orient_tensor
