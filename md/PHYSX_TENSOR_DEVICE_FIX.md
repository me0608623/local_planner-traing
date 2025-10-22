# 🔧 PhysX 張量設備不匹配錯誤修復指南

## 🚨 錯誤描述

```
[Error] [omni.physx.tensors.plugin] Incompatible device of velocity tensor in function getVelocities: 
expected device 0, received device -1
```

**問題分析**：
- 系統期望張量在GPU（cuda:0，device index 0）
- 實際收到CPU張量（device index -1）
- 這是GPU模擬pipeline中的設備不一致問題

## ✅ 解決方案

### 方案一：使用CPU模式（推薦用於調試）

使用專門的CPU配置來避免設備不匹配：

```python
# 訓練指令
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 16 --headless
```

**特點**：
- ✅ 完全避免GPU/CPU張量不匹配
- ✅ 穩定可靠，適合調試
- ⚠️ 訓練速度較慢
- ⚠️ 環境數量受限（建議16個）

### 方案二：GPU模式修復版本（實驗性）

使用增強的GPU配置：

```python
# 訓練指令
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0 \
    --num_envs 64 --headless
```

**特點**：
- ✅ 較快的訓練速度
- ⚠️ 需要足夠的GPU記憶體
- ⚠️ 可能仍有張量不匹配風險

### 方案三：標準GPU模式（已增強）

```python
# 訓練指令
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 256 --headless
```

## 🔧 已實施的修復

### 1. 環境配置修復

在 `LocalPlannerEnvCfg.__post_init__()` 中：

```python
# 🔧 明確設定GPU設備
self.sim.device = "cuda:0"

# 🔧 增加GPU緩衝區容量
self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024
self.sim.physx.gpu_max_rigid_patch_count = 512 * 1024
self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
# ... 其他緩衝區設定
```

### 2. CPU模式配置

創建了專門的CPU環境和訓練配置：
- `LocalPlannerEnvCfg_CPU`：CPU環境配置
- `LocalPlannerPPORunnerCfg_CPU`：CPU訓練配置

### 3. 註冊多種環境版本

```python
# 標準GPU版本
"Isaac-Navigation-LocalPlanner-Carter-v0"

# CPU版本（修復張量不匹配）
"Isaac-Navigation-LocalPlanner-Carter-CPU-v0"

# 簡化CPU版本
"Isaac-Navigation-LocalPlanner-Carter-CPU-Simple-v0"

# GPU修復實驗版本
"Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0"
```

## 🧪 測試建議

### 步驟1：測試CPU版本

```bash
# 快速測試
./isaaclab.sh -p scripts/test_local_planner_fixed.py

# 完整訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-Simple-v0 \
    --num_envs 4 --max_iterations 100
```

### 步驟2：如果CPU版本成功，嘗試GPU修復版本

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0 \
    --num_envs 64 --max_iterations 100
```

### 步驟3：監控錯誤

如果仍出現張量設備錯誤，回退到CPU版本：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 16
```

## 🔍 根本原因分析

### 1. 設備初始化問題
- Isaac Lab默認使用`cuda:0`
- 但某些組件可能初始化在CPU上
- 導致張量設備不一致

### 2. GPU記憶體不足
- 當GPU記憶體不足時，PhysX自動回退到CPU
- 但其他組件仍在GPU上
- 造成設備不匹配

### 3. Nova Carter模型相容性
- 某些USD模型可能不完全支援GPU模擬
- 需要確保所有剛體和關節都啟用GPU模式

## ⚠️ 注意事項

1. **環境數量**：
   - CPU模式：建議≤16個環境
   - GPU修復版本：建議≤64個環境
   - 標準GPU版本：可用更多環境

2. **訓練速度**：
   - CPU版本較慢但穩定
   - GPU版本快但需確保記憶體充足

3. **調試建議**：
   - 初次使用建議從CPU版本開始
   - 確認環境正常後再試GPU版本

## ✅ 成功標準

環境啟動時應看到：

```
🔧 [修復] GPU/CPU 張量不匹配問題
   - 設備模式: cuda:0 (或 cpu)
   - PhysX GPU: True (或 False)
   - 環境數量: 64 (或適當數量)
```

沒有 `[Error] [omni.physx.tensors.plugin] Incompatible device` 錯誤訊息。

---

**更新時間**: {current_date}  
**狀態**: ✅ 完整修復方案  
**測試狀態**: 待用戶驗證
