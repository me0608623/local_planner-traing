# 🏗️ Isaac Lab Nova Carter 訓練架構總結

## 🎯 目前訓練架構

### 核心環境配置

| 環境名稱 | 用途 | 特點 |
|---------|------|------|
| `Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0` | 正式訓練 | Isaac Sim 5.0 完全兼容 |
| `Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-Simple-v0` | 快速測試 | Isaac Sim 5.0 兼容簡化版 |
| `Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0` | 通用訓練 | 純PyTorch方法 |
| `Isaac-Navigation-LocalPlanner-Carter-CPU-v0` | 安全備用 | CPU模式 |

### 訓練命令範例

```bash
# 推薦：Isaac Sim 5.0 兼容版本
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0 \
    --num_envs 128 --max_iterations 1000

# 快速測試
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-Simple-v0 \
    --num_envs 32 --max_iterations 10
```

## 📁 代碼功能架構

### 主要配置文件

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/
├── local_planner_env_cfg.py                    # 基礎環境配置
├── local_planner_env_cfg_cpu.py               # CPU版本配置
├── local_planner_env_cfg_gpu_optimized.py     # GPU優化配置（原版）
├── local_planner_env_cfg_gpu_optimized_fixed.py # GPU修復版配置
├── local_planner_env_cfg_isaac_sim_5_fixed.py   # Isaac Sim 5.0兼容配置 ⭐
├── __init__.py                                 # 環境註冊
└── agents/
    ├── rsl_rl_ppo_cfg.py                      # PPO算法配置
    └── rsl_rl_ppo_cfg_cpu.py                 # CPU版PPO配置
```

### 核心功能

#### 1. **環境配置 (Environment Configuration)**
- **場景設定**：地形、機器人、LiDAR、障礙物
- **MDP設定**：觀察空間、動作空間、獎勵函數
- **模擬參數**：PhysX、設備管理、GPU緩衝區

#### 2. **設備管理 (Device Management)**
```python
# Isaac Sim 5.0 兼容策略
def __post_init__(self):
    # 1. 嘗試新版模組 isaacsim.*
    # 2. 回退舊版模組 omni.isaac.*
    # 3. 使用純 PyTorch 方法
```

#### 3. **張量工具 (Tensor Utilities)**
- `ensure_cuda_tensor_isaac_sim_5()` - 張量設備一致性
- `convert_positions_to_cuda_isaac_sim_5()` - 位置數據轉換

#### 4. **強化學習配置**
- **算法**：PPO (Proximal Policy Optimization)
- **網路**：Actor-Critic 架構
- **訓練參數**：學習率、批次大小、迭代次數

## 🔧 我做了什麼修復

### ✅ 1. 檔案清理與組織
- 移動所有 `.md` 文件到 `md/` 資料夾
- 刪除不必要的狀態文件和重複文檔
- 整理項目結構

### ✅ 2. Python 依賴版本修復
```bash
# 解決 TypeError: Type parameter ~_T1 without a default
TensorDict: 降版到 0.9.0
typing_extensions: 調整到 4.10.0  
numpy: 降版到 1.26.4
```

### ✅ 3. PhysX 張量設備匹配修復
```python
# 解決 [Error] [omni.physx.tensors.plugin] Incompatible device
self.sim.device = "cuda:0"
self.sim.physx.gpu_max_rigid_contact_count = 2048 * 1024
# 增加GPU緩衝區容量，防止CPU回退
```

### ✅ 4. Isaac Sim 依賴問題修復
創建了不依賴 `omni.isaac.core` 的修復版：
```python
# 直接使用 PyTorch 設備管理
import torch
torch.cuda.set_device(0)
if hasattr(torch, 'set_default_device'):
    torch.set_default_device('cuda:0')
```

### ✅ 5. Isaac Sim 5.0 模組重構兼容
解決 `ModuleNotFoundError: No module named 'omni.isaac.core'`：
```python
# 多重兼容策略
# 1. 新版：isaacsim.core.api.utils.torch
# 2. 舊版：omni.isaac.core.utils.torch  
# 3. 純PyTorch：完全獨立
```

## 🎯 技術核心

### GPU 優化策略
1. **路線A實施**：全程GPU，避免CPU/GPU數據拷貝
2. **緩衝區擴大**：防止GPU記憶體不足導致回退
3. **設備一致性**：確保所有張量在同一設備

### 兼容性設計
- **向前兼容**：支援 Isaac Sim 5.0 新模組
- **向後兼容**：仍支援 Isaac Sim 4.x
- **完全獨立**：最終回退純 PyTorch 方法

### 多層保護
1. **Isaac Sim 5.0 專用版** - 針對模組重構優化
2. **通用修復版** - 純 PyTorch 方法
3. **CPU 安全版** - 保證可用
4. **完整文檔** - 故障排除指南

## 📊 成功指標

### 啟動時應看到：
```
🔧 [Isaac Sim 5.0 兼容] 開始設定CUDA設備...
✅ [Isaac Sim 5.0] 使用新版模組 isaacsim.core.api.utils.torch
🎉 [Isaac Sim 5.0 兼容] GPU優化配置完成
   - Isaac Sim 版本: 5.0 (模組重構兼容)
   - 設備模式: cuda:0
   - PhysX GPU: True
   - 環境數量: 128
   - GPU 緩衝區: 2048K contacts
```

### 不再出現的錯誤：
- ❌ `ModuleNotFoundError: No module named 'omni.isaac.core'`
- ❌ `[Error] [omni.physx.tensors.plugin] Incompatible device`
- ❌ `TypeError: Type parameter ~_T1 without a default`

## 🚀 使用建議

### 推薦順序：
1. **首選**：`Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-Simple-v0`（快速測試）
2. **正式**：`Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0`（完整訓練）
3. **備用**：`Isaac-Navigation-LocalPlanner-Carter-CPU-v0`（如有問題）

### 訓練參數建議：
- **測試**：`--num_envs 32 --max_iterations 10`
- **訓練**：`--num_envs 128 --max_iterations 1000`
- **高性能**：`--num_envs 256 --max_iterations 3000`（RTX 5090）

---

**總結**：我創建了一個完整的、多層保護的訓練架構，解決了所有依賴問題、版本衝突和模組重構問題，讓您可以在 Isaac Sim 5.0 環境中順利進行 Nova Carter 本地規劃器的強化學習訓練。
