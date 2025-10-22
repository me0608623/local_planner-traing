# Nova Carter Local Planner - 強化學習環境

基於 Isaac Lab 和 Isaac Sim 5.0 的 Nova Carter 機器人本地路徑規劃強化學習環境。

## 🚀 專案簡介

本專案實現了 Nova Carter 移動機器人的本地路徑規劃強化學習環境，支援障礙物迴避、目標導航和 LiDAR 感知。環境完全兼容 Isaac Sim 5.0，並修復了所有已知的 PhysX tensor device 匹配問題。

### ✨ 主要特性

- 🤖 **Nova Carter 機器人模擬**：完整的差動驅動機器人模型
- 🎯 **目標導航任務**：智能體需學習從起點導航到目標點
- 👁️ **LiDAR 感知**：360度雷射雷達感測器進行環境感知
- 🚧 **動態障礙物**：隨機生成的障礙物環境
- ⚡ **多模式支援**：CPU/GPU 訓練模式和不同複雜度配置
- 🔧 **Isaac Sim 5.0 兼容**：完全支援最新版本的 Isaac Sim

## 📋 環境要求

- **Isaac Sim 5.0+**
- **Isaac Lab 2.2+**
- **Python 3.11**
- **CUDA 12.x** (GPU 模式)
- **RSL-RL** 強化學習庫

## 🛠️ 安裝和設置

### 1. 環境準備

```bash
# 確保在正確的 conda 環境中
conda activate env_isaaclab

# 安裝必要的 Python 依賴
pip install packaging

# 確認 Isaac Sim 路徑
ls -la _isaac_sim  # 應該指向您的 Isaac Sim 安裝目錄
```

### 2. 環境註冊

環境會在導入時自動註冊，或可手動註冊：

```bash
python register_local_planner.py
```

## 🎮 使用方法

### 基本訓練

```bash
# GPU 模式訓練（推薦）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 \
    --headless

# CPU 模式訓練
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 2 \
    --headless
```

### 可用環境

| 環境名稱 | 描述 | 設備 | 適用模式 | 複雜度 |
|---------|------|------|----------|-------|
| `Isaac-Navigation-LocalPlanner-Carter-v0` | 標準配置 | CUDA | Headless | 中等 |
| `Isaac-Navigation-LocalPlanner-Carter-CPU-v0` | CPU 優化版本 | CPU | 兩者皆可 | 中等 |
| `Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0` | GPU 優化版本 | CUDA | Headless | 高 |
| `Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0` | **GUI 模式專用** ⭐ | CUDA | **GUI Only** | 中等 |
| `Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0` | Isaac Sim 5.0 專用 | CUDA | Headless | 中等 |

### 環境參數

```bash
# 調整環境數量
--num_envs 8        # 並行環境數（根據GPU記憶體調整）

# 訓練步數
--max_iterations 1000

# 無頭模式（服務器訓練，推薦）
--headless

# 啟用視覺化（本機訓練，需要特殊配置）
# 移除 --headless 參數，使用 GUI-Fixed 環境
```

## 🎮 GUI vs Headless 模式重要說明

### 🚨 **關鍵發現**

**PhysX tensor device 錯誤只在 GUI 模式出現，Headless 模式完全正常！**

### 模式對比

| 模式 | 狀態 | 原因 | 建議 |
|------|------|------|------|
| **Headless** | ✅ 完全正常 | 統一CPU處理或正確GPU管線 | **生產首選** |
| **GUI** | ❌ 出現錯誤 | 自動啟用GPU物理管線衝突 | 使用專用修復配置 |

### 最佳實踐

```bash
# 1. 開發和訓練：使用 Headless 模式（推薦）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 4 --headless

# 2. GUI 視覺化需求：使用專用配置
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0 \
    --num_envs 2
    # 注意：不使用 --headless

# 3. 診斷問題：
python scripts/diagnose_tensor_device.py
```

## 📊 環境詳細說明

### 觀測空間

- **LiDAR 數據**: 360度掃描，範圍10米
- **目標相對位置**: 機器人到目標的相對距離和角度
- **機器人速度**: 當前線速度和角速度
- **歷史動作**: 前一時步的控制指令

### 動作空間

- **線速度指令**: [-2.0, 2.0] m/s
- **角速度指令**: [-3.14, 3.14] rad/s

### 獎勵函數

- **到達目標**: +1000（終端獎勵）
- **接近目標**: 基於距離減少的連續獎勵
- **避開障礙物**: 基於 LiDAR 距離的懲罰
- **超時懲罰**: -100（超過最大步數）

## 🔧 配置選項

### 環境配置文件

- `local_planner_env_cfg.py`: 基本 GPU 配置
- `local_planner_env_cfg_cpu.py`: CPU 優化配置
- `local_planner_env_cfg_gpu_optimized.py`: GPU 高性能配置
- `local_planner_env_cfg_isaac_sim_5_fixed.py`: Isaac Sim 5.0 兼容配置

### 訓練配置

RSL-RL PPO 算法配置位於：
- `agents/rsl_rl_ppo_cfg.py`: GPU 訓練配置
- `agents/rsl_rl_ppo_cfg_cpu.py`: CPU 訓練配置

## 🐛 故障排除

### 常見問題

1. **PhysX tensor device mismatch** ⭐ **官方已知問題**
   ```
   錯誤：[Error] [omni.physx.tensors.plugin] Incompatible device of velocity tensor 
         in function getVelocities: expected device 0, received device -1
   
   原因：NVIDIA官方確認的API問題（非用戶環境錯誤）
         - NVIDIA Developer Forums 已記錄
         - Isaac Lab GitHub Issues 官方bug報告
   
   解決方案：使用我們的修復配置
   --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0
   
   診斷工具：
   python scripts/diagnose_tensor_device.py --full
   ```

2. **模組導入錯誤 (omni.isaac.core)**
   ```
   解決方案：使用 Isaac Sim 5.0 兼容配置
   --task Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0
   ```

3. **記憶體不足**
   ```
   解決方案：減少並行環境數量
   --num_envs 2  # 或更少
   ```

4. **訓練不穩定**
   ```
   解決方案：使用 CPU 模式或調整超參數
   --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0
   ```

### 詳細故障排除

更多詳細的故障排除指南請參考：
- [🔍 NVIDIA官方問題分析](md/NVIDIA_OFFICIAL_PHYSX_ISSUE_ANALYSIS.md) ⭐ **必讀**
- [PhysX 修復指南](md/PHYSX_TENSOR_DEVICE_FIX.md)
- [Isaac Sim 5.0 兼容性](md/ISAAC_SIM_5_MODULE_RESTRUCTURE_FIX.md)
- [完整問題解決方案](md/ALL_ISSUES_FIXED_SUMMARY.md)

## 📖 技術文檔

- [項目架構總覽](md/PROJECT_ARCHITECTURE_SUMMARY.md)
- [🎮 GUI vs Headless 深度分析](md/GUI_VS_HEADLESS_PHYSX_ANALYSIS.md) ⭐ **重要發現**
- [🔍 NVIDIA官方問題分析](md/NVIDIA_OFFICIAL_PHYSX_ISSUE_ANALYSIS.md) ⭐ **官方確認**
- [強化學習策略](md/RL_STRATEGY_ARCHITECTURE.md)
- [最終解決方案](md/FINAL_ISAAC_SIM_5_SOLUTION.md)
- [用戶指南](md/FINAL_USER_GUIDE.md)

## 🏗️ 項目結構

```
├── source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/
│   ├── __init__.py                 # 環境註冊
│   ├── local_planner_env_cfg.py    # 基本環境配置
│   ├── local_planner_env_cfg_*.py  # 各種配置變體
│   ├── agents/                     # 訓練算法配置
│   └── mdp/                        # MDP 組件
│       ├── actions.py              # 動作定義
│       ├── observations.py         # 觀測定義
│       ├── rewards.py              # 獎勵函數
│       └── terminations.py         # 終止條件
├── scripts/reinforcement_learning/rsl_rl/
│   └── train.py                    # 訓練腳本
├── register_local_planner.py       # 手動環境註冊
└── md/                             # 技術文檔
```

## 🎯 性能建議

### Headless 模式訓練 (強烈推薦) ⭐

```bash
# 高性能 Headless 訓練 - 最穩定的選擇
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 8 \
    --headless
```

### GUI 模式訓練 (特殊需求)

```bash
# GUI 模式專用配置 - 用於視覺化需求
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0 \
    --num_envs 2
    # 注意：環境數量較少以避免GUI渲染開銷
```

### CPU 訓練 (兼容性最佳)

```bash
# 適用於沒有 GPU 或最大兼容性需求
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-CPU-v0 \
    --num_envs 2 \
    --headless
```

### 工作流程建議

1. **開發階段**: 使用 Headless 模式快速迭代
2. **除錯階段**: 偶爾使用 GUI 模式觀察行為  
3. **生產訓練**: 始終使用 Headless 模式
4. **結果展示**: 訓練完成後使用 play 腳本

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案遵循 Isaac Lab 的授權條款。

## 🙏 致謝

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 團隊
- NVIDIA Isaac Sim 開發團隊
- RSL-RL 強化學習庫

---

**開始您的 Nova Carter 強化學習之旅！** 🚀