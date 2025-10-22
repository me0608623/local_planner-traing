# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Nova Carter 本地規劃器 GPU 深度優化配置
實施路線A：全程GPU（最優雅的解決方案）
"""

import torch
# Isaac Sim 5.0 兼容性導入
try:
    from isaacsim.core.api.utils.torch import set_cuda_device
    print("✅ [Isaac Sim 5.0] 使用新版模組 isaacsim.core.api.utils.torch")
except ImportError:
    try:
        from omni.isaac.core.utils.torch import set_cuda_device
        print("⚠️ [Isaac Sim 5.0] 回退到舊版模組 omni.isaac.core.utils.torch")
    except ImportError:
        print("❌ [Isaac Sim 5.0] 無法導入 Isaac Sim 相關模組，使用純 PyTorch 替代方案")
        def set_cuda_device(device: int):
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(device)
            else:
                print("CUDA 不可用，無法設定設備。")
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .local_planner_env_cfg import LocalPlannerEnvCfg, LocalPlannerEnvCfg_SIMPLE

##
# GPU 全程優化環境配置
##

@configclass 
class LocalPlannerEnvCfg_GPU_OPTIMIZED(LocalPlannerEnvCfg):
    """Nova Carter 本地規劃器 - GPU 深度優化版本
    
    路線A：全程GPU（建議）
    核心策略：確保所有與PhysX tensors API交互的數據都是CUDA tensor
    
    解決根本問題：
    - 不迴避問題，而是確保設備一致性
    - 所有傳給PhysX/Isaac API的數據都是CUDA tensor
    - 預先設定Isaac/Torch的預設裝置
    - 確保沒有CPU Tensor意外混入
    """
    
    def __post_init__(self):
        """後處理：GPU 深度優化設定"""
        # 🔧 第一步：預先設定設備（在任何其他操作之前）
        device_id = 0  # 使用 GPU 0
        
        # 設定 Isaac Core 的預設裝置
        set_cuda_device(device_id)
        print(f"🔧 [GPU優化] Isaac Core 設備設定: cuda:{device_id}")
        
        # 設定 PyTorch 的預設裝置
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print(f"🔧 [GPU優化] PyTorch 預設設備: {device}")
        
        # 調用父類設定
        super().__post_init__()
        
        # 🔧 第二步：強化 GPU 設定（覆蓋任何可能的 CPU 回退）
        self.sim.device = f"cuda:{device_id}"
        
        # 確保 PhysX 完全使用 GPU
        self.sim.physx.use_gpu = True
        self.sim.physx.gpu_device_id = device_id
        
        # 🔧 第三步：大幅增加 GPU 緩衝區容量（防止記憶體不足導致回退到CPU）
        # 比之前的設定更大，確保不會因記憶體不足而自動切換到CPU
        self.sim.physx.gpu_max_rigid_contact_count = 2048 * 1024    # 2M contacts
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024     # 1M patches
        self.sim.physx.gpu_found_lost_pairs_capacity = 2048 * 1024 # 2M pairs
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 512 * 1024  # 512K
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2048 * 1024      # 2M
        
        # 🔧 第四步：啟用進階 GPU 優化
        self.sim.physx.enable_ccd = True  # 連續碰撞檢測
        self.sim.physx.enable_stabilization = True  # 物理穩定化
        self.sim.physx.use_gpu_pipeline = True  # 強制使用 GPU pipeline
        
        # 🔧 第五步：優化環境數量以充分利用 GPU
        # 使用更多環境來充分利用 GPU 並行能力
        original_num_envs = self.scene.num_envs
        self.scene.num_envs = min(original_num_envs, 512)  # 最多512個環境，平衡性能和記憶體
        
        # 優化時間步長設定
        self.decimation = 4   # 保持原始設定以確保穩定性
        self.sim.dt = 0.01    # 保持高精度時間步長
        
        print("🔧 [GPU深度優化] 配置完成")
        print(f"   - 設備模式: {self.sim.device}")
        print(f"   - PhysX GPU: {self.sim.physx.use_gpu}")
        print(f"   - GPU 設備ID: {self.sim.physx.gpu_device_id}")
        print(f"   - 環境數量: {self.scene.num_envs} (原始: {original_num_envs})")
        print(f"   - GPU 緩衝區: {self.sim.physx.gpu_max_rigid_contact_count // 1024}K contacts")
        print(f"   - PyTorch 預設設備: {torch.get_default_device()}")


@configclass
class LocalPlannerEnvCfg_GPU_OPTIMIZED_SIMPLE(LocalPlannerEnvCfg_SIMPLE):
    """GPU 深度優化簡化版本 - 用於測試和驗證"""
    
    def __post_init__(self):
        """後處理：GPU 優化 + 簡化設定"""
        # 設定設備
        device_id = 0
        set_cuda_device(device_id)
        torch.cuda.set_device(device_id)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        
        # 調用父類設定
        super().__post_init__()
        
        # GPU 優化設定
        self.sim.device = f"cuda:{device_id}"
        self.sim.physx.use_gpu = True
        self.sim.physx.gpu_device_id = device_id
        
        # 適中的緩衝區設定（測試用）
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024    
        self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024     
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 
        
        # 測試友好的環境數量
        self.scene.num_envs = 32   # 適中數量便於測試
        self.episode_length_s = 20.0  # 較短回合便於快速測試
        
        print("🔧 [GPU深度優化-簡化] 配置完成")
        print(f"   - 設備模式: {self.sim.device}")
        print(f"   - 環境數量: {self.scene.num_envs}")
        print(f"   - 回合長度: {self.episode_length_s}s")


##
# 張量設備工具函數
##

def ensure_cuda_tensor(data, device_id: int = 0):
    """確保數據是 CUDA tensor
    
    用於與 PhysX/Isaac API 交互的所有數據
    
    Args:
        data: 輸入數據 (numpy array, CPU tensor, 或已經是 CUDA tensor)
        device_id: CUDA 設備 ID
    
    Returns:
        CUDA tensor
    """
    device = torch.device(f"cuda:{device_id}")
    
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=torch.float32)
    else:
        # numpy array 或其他格式
        return torch.tensor(data, dtype=torch.float32, device=device)


def convert_positions_to_cuda(coords, orientations=None, device_id: int = 0):
    """將位置和方向數據轉換為 CUDA tensor
    
    專門用於 set_world_poses() 等 PhysX API 調用
    
    Args:
        coords: 位置座標 (N, 3) 或 (3,)
        orientations: 四元數方向 (N, 4) 或 (4,)，如果為 None 則使用預設方向
        device_id: CUDA 設備 ID
    
    Returns:
        tuple: (cuda_coords, cuda_orientations)
    """
    device = torch.device(f"cuda:{device_id}")
    
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
# 使用範例和最佳實踐
##

class GPUOptimizedUsageExample:
    """GPU 優化使用範例"""
    
    @staticmethod
    def example_set_world_poses():
        """正確的 set_world_poses 使用方式"""
        # ❌ 錯誤：使用 numpy array（會導致張量設備不匹配）
        # coords_np = np.array([[-8.0, 32.0, -2.3]])
        # orient_np = np.array([[0.0, 0.0, 0.0, 1.0]]) 
        # self.cube_xform1.set_world_poses(coords_np, orient_np)  # 這會出錯！
        
        # ✅ 正確：使用 CUDA tensor
        coords, orient = convert_positions_to_cuda(
            coords=[[-8.0, 32.0, -2.3]], 
            orientations=[[0.0, 0.0, 0.0, 1.0]],
            device_id=0
        )
        # self.cube_xform1.set_world_poses(coords, orient)  # 這是正確的！
        
    @staticmethod
    def example_data_conversion():
        """數據轉換範例"""
        # 從 PhysX API 讀取的 CUDA tensor
        # cuda_positions = articulation_view.get_world_poses()
        
        # 如果需要與 numpy/ROS 交互，轉到 CPU
        # cpu_positions = cuda_positions.cpu().numpy()
        
        # ROS 訊息處理（CPU 沒關係）
        # ros_msg.data = cpu_positions.flatten().tolist()
        
        # 再次與 PhysX 交互時，轉回 CUDA
        # cuda_new_positions = ensure_cuda_tensor(cpu_positions, device_id=0)
        pass
