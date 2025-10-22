# 🔧 Isaac Sim 安裝問題修復指南

## 🚨 問題診斷

您遇到的錯誤：
```
ModuleNotFoundError: No module named 'omni.isaac.core'
```

**根本原因**：Isaac Sim 安裝不完整或結構異常

## 🔍 問題分析

### 發現的問題：
1. **omni.isaac.core 在 extsDeprecated 目錄中**：
   ```
   /home/aa/isaacsim/extsDeprecated/omni.isaac.core/
   ```
   而不是在活躍的 extensions 目錄中

2. **缺少核心依賴**：即使手動添加路徑，也缺少 `carb` 模組

3. **Python 路徑問題**：Isaac Sim 的 Python 環境沒有正確包含所需的模組路徑

## ✅ 解決方案

### 方案A：使用修復版配置（推薦）

我們創建了不依賴 `omni.isaac.core` 的修復版配置：

#### 新的環境版本：
| 環境名稱 | 特點 | 狀態 |
|---------|------|------|
| `Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0` | GPU優化，不依賴omni.isaac.core | ✅ 修復版 |
| `Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-Simple-v0` | 簡化測試版 | ✅ 修復版 |

#### 測試修復版：
```bash
cd /home/aa/IsaacLab

# 測試修復版GPU優化（推薦優先嘗試）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-Simple-v0 \
    --num_envs 32 --max_iterations 10
```

#### 修復版特點：
- ✅ **不依賴 omni.isaac.core**：直接使用 PyTorch 設備管理
- ✅ **保持 GPU 優化**：仍然實現路線A的核心理念
- ✅ **自動回退**：CUDA 不可用時自動使用 CPU
- ✅ **完整功能**：包含所有張量設備一致性工具

### 方案B：修復 Isaac Sim 安裝（長期解決）

如果您希望完整修復 Isaac Sim 安裝：

#### 1. 檢查安裝來源
```bash
ls -la /home/aa/isaacsim/
# 查看是否有安裝記錄或版本資訊
cat /home/aa/isaacsim/VERSION
```

#### 2. 重新安裝 Isaac Sim
建議使用官方安裝方法：

**選項1：通過 Omniverse Launcher**
- 下載 [Omniverse Launcher](https://developer.nvidia.com/omniverse)
- 安裝 Isaac Sim 4.5 或 5.0

**選項2：通過 GitHub（開源版本）**
```bash
# 克隆並構建 Isaac Sim
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
./build.sh
```

#### 3. 重新建立符號連結
```bash
cd /home/aa/IsaacLab
rm _isaac_sim  # 移除舊連結
ln -s /path/to/new/isaac-sim/_build/linux-x86_64/release _isaac_sim
```

### 方案C：CPU 版本回退（最安全）

如果其他方案都有問題：
```bash
# 使用 CPU 版本，完全避免 GPU 相關問題
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 16 --max_iterations 100
```

## 🧪 測試驗證

### 基本功能測試
```bash
cd /home/aa/IsaacLab
PYTHONPATH=/home/aa/IsaacLab/source /home/aa/IsaacLab/_isaac_sim/kit/python/bin/python3 -c "
# 測試修復版配置
from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg_gpu_optimized_fixed import LocalPlannerEnvCfg_GPU_OPTIMIZED_FIXED
print('✅ 修復版配置導入成功')

# 測試 PyTorch CUDA
import torch
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 設備: {torch.cuda.get_device_name()}')
"
```

### 完整環境測試
```bash
# 測試修復版簡化環境
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-Simple-v0 \
    --num_envs 4 --max_iterations 1
```

## 📊 版本對比

| 配置版本 | 依賴要求 | GPU 優化 | 穩定性 | 推薦度 |
|---------|---------|---------|--------|--------|
| **GPU-Fixed** | 僅 PyTorch | ✅ 完整 | 高 | ⭐⭐⭐⭐⭐ |
| GPU-Optimized | omni.isaac.core | ✅ 完整 | 中等 | ⭐⭐⭐ |
| CPU | 無特殊要求 | ❌ | 最高 | ⭐⭐⭐⭐ |

## 💡 修復版技術細節

### 核心變更：
1. **移除依賴**：
   ```python
   # ❌ 原版（有依賴問題）
   from omni.isaac.core.utils.torch import set_cuda_device
   set_cuda_device(0)
   
   # ✅ 修復版（無依賴）
   import torch
   torch.cuda.set_device(0)
   if hasattr(torch, 'set_default_device'):
       torch.set_default_device(f"cuda:{device_id}")
   ```

2. **自動回退機制**：
   ```python
   if torch.cuda.is_available():
       device = torch.device(f"cuda:{device_id}")
   else:
       device = torch.device("cpu")
   ```

3. **保持優化特性**：
   - GPU 緩衝區增強
   - PhysX GPU 設定
   - 張量設備一致性工具

## 🎯 建議測試順序

1. **修復版測試**（推薦優先）：
   ```bash
   --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-Simple-v0
   ```

2. **如果修復版成功**，嘗試完整版：
   ```bash
   --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0
   ```

3. **如果仍有問題**，使用 CPU 版本：
   ```bash
   --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0
   ```

## 🔧 故障排除

### 如果修復版仍有問題：

1. **檢查 PyTorch 安裝**：
   ```bash
   /home/aa/IsaacLab/_isaac_sim/kit/python/bin/python3 -c "import torch; print(torch.__version__)"
   ```

2. **檢查 CUDA 驅動**：
   ```bash
   nvidia-smi
   ```

3. **回退到 CPU 模式**：
   ```bash
   --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0
   ```

---

**結論**：修復版配置是目前最實用的解決方案，既保持了 GPU 優化的核心理念，又避免了 Isaac Sim 安裝問題。建議優先使用修復版進行測試和訓練。
